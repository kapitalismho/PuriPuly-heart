from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from selfpeer_probes import BASELINE, PHASES, cache_evidence
from summarize import distribution
from summarize_selfpeer import grouped, pair_stats, resources

TARGET = 8589934592
CEILING = 8697308774
MEMORY_FIELDS = ('process_gpu_dedicated_bytes', 'device_gpu_dedicated_bytes', 'WorkingSetSize', 'PrivateUsage', 'process_gpu_shared_bytes')
TIMING_FIELDS = ('prefill_ms', 'prefill_tps', 'decode_ms', 'decode_tps')


def signature(request):
    return tuple(request.get(key) for key in ('phase', 'index', 'round', 'channel', 'slot', 'case_index', 'prompt_sha256', 'system_sha256', 'temperature', 'seed', 'max_output_tokens'))


def memory_class(peak):
    if peak is None:
        return 'unavailable'
    return 'target' if peak <= TARGET else 'tolerance_band' if peak <= CEILING else 'over_ceiling'


def unit(value):
    return {'bytes': value, 'GiB': value / 2**30, 'GB': value / 1e9} if value is not None else None


def load_runs(public):
    records = []
    for path in public.glob('tuning8gib-*.json'):
        if any(part in path.name for part in ('.slots.', '.headroom.')):
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if 'requests' in record:
            records.append(record)
    return sorted(records, key=lambda record: (record['started_utc'], record['profile']))


def analyze(public):
    records = load_runs(public)
    controls = [record for record in records if (record['n_cpu_moe'], record['cache_mib']) == (24, 2048) and not record.get('contaminated_by_runtime_conflict')]
    reference = controls[0] if controls else None
    reference_requests = {signature(request): request for request in reference['requests']} if reference else {}
    runs, unique, token_differences = [], {}, []
    for record in records:
        name = record['profile']
        samples = [json.loads(line) for line in (public / (name + '.samples.jsonl')).read_text(encoding='utf-8').splitlines()]
        slots = json.loads((public / (name + '.slots.json')).read_text(encoding='utf-8'))
        requests = record['requests']
        parsed_cache = cache_evidence((public / (name + '.native.log')).read_text(encoding='utf-8').splitlines(), record['cache_mib'])
        if parsed_cache != record['cache_execution']:
            correction = {'run': name, 'original_cache_execution': record['cache_execution'], 'recovered_cache_execution': parsed_cache, 'reason': 'First control used inherited placement filter, which omitted the separate native group/slot line. Original full sanitized native log preserved; recovery is parsing, not a new measurement.'}
            (public / (name + '.cache-correction.json')).write_text(json.dumps(correction, indent=2), encoding='utf-8')
            record['cache_execution'] = parsed_cache
        signatures_match = bool(reference) and [signature(request) for request in requests] == [signature(request) for request in reference['requests']]
        pc_match = bool(reference) and len(requests) == 44
        for request in requests:
            previous = reference_requests.get(signature(request))
            fields = ('prompt_n', 'cache_n', 'predicted_n')
            counts = lambda row: {key: (row.get('timings') or {}).get(key) for key in fields}
            if previous is None or any(counts(previous)[key] != counts(request)[key] for key in fields[:2]):
                pc_match = False
            if previous and counts(previous) != counts(request):
                token_differences.append({'reference': reference['profile'], 'run': name, 'phase': request['phase'], 'channel': request['channel'], 'round': request['round'], 'reference_counts': counts(previous), 'counts': counts(request)})
            key = (request['input'], request['output'])
            item = unique.setdefault(key, {'id': hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16], 'input': request['input'], 'output': request['output'], 'automated_checks': request['quality_checks'], 'occurrences': []})
            item['occurrences'].append({'run': name, 'phase': request['phase'], 'channel': request['channel'], 'round': request['round'], 'stop_type': request['stop_type']})
        lifecycle = resources(samples)
        peak = lifecycle.get('process_gpu_dedicated_bytes', {}).get('max')
        overlap = sum(pair['phase'] == 'simultaneous_pairs' and pair['server_active_overlap_seconds'] > 0 for pair in slots['pairs'])
        cache = record['cache_execution']
        cache_groups = cache.get('groups', [])
        allocations = cache.get('allocations', [])
        expected_slots = min(record['n_cpu_moe'] * 128, record['cache_mib'] * 2**20 // 3345408 - 1)
        checks = {
            'exclusive_runtime_observed': record.get('runtime_conflict_observation_count', 0) > 0 and not record.get('contaminated_by_runtime_conflict'),
            'complete_44': record['status'] == 'completed' and len(requests) == 44,
            'phase_counts': Counter(request['phase'] for request in requests) == {'first_prefix': 2, 'serial_pairs': 12, 'simultaneous_pairs': 30},
            'eos_and_automated_bounded_quality': len(requests) == 44 and all(request['stop_type'] == 'eos' and all(request['quality_checks'][key] for key in ('nonempty', 'no_reasoning', 'not_truncated')) for request in requests),
            'fresh_graceful_exit': record.get('shutdown') == 'graceful_console_ctrl_c' and record.get('process_exit_code') == 0 and not record.get('safety_cutoff'),
            'workload_identity': signatures_match,
            'prompt_evaluated_cached_counts': pc_match,
            'self0_peer1_slot2_unused': sorted({event['slot'] for event in slots['events']}) == [0, 1] and all(request['slot'] == (0 if request['channel'] == 'Self' else 1) for request in requests),
            'native_overlap_15': overlap == 15 and not slots['unreleased'] and slots['slot_counts'] == {'0': 22, '1': 22},
            'effective_backend_controls_absent': all(value is None for value in record['allowlisted_backend_environment_effective'].values()),
            'pid_exact_adapter_dedicated_observed': peak is not None and peak > 0 and bool(record.get('counter_instances', {}).get('process_gpu_dedicated_bytes')) and all(instance.startswith(f"pid_{record['server_pid']}_") and record['gpu_adapter_luid'] in instance.lower() for instance in record.get('counter_instances', {}).get('process_gpu_dedicated_bytes', [])),
            'cache_used': cache['state'] == 'allocated_and_used' and any(bucket['hits'] + bucket['misses'] > 0 and bucket['uploaded_mib_rounded_native'] > 0 for bucket in cache['destructor_buckets']),
            'cache_slots_and_effective_size': len(cache_groups) == 1 and len(allocations) == 1 and cache_groups[0]['layers'] == record['n_cpu_moe'] and cache_groups[0]['slots'] == expected_slots and abs(allocations[0]['effective_mib_rounded_native'] - (expected_slots + 1) * 3345408 / 2**20) <= 0.0051,
            'dedicated_observed_during_load_and_each_phase': all(any(sample['phase'] == phase and sample.get('process_gpu_dedicated_bytes', 0) for sample in samples) for phase in ('load', *PHASES)),
            'vulkan_host': any('Vulkan_Host' in line and 'model buffer size' in line for line in record['placement_cache_log']),
        }
        item = {key: record.get(key) for key in ('profile', 'started_utc', 'status', 'error', 'n_cpu_moe', 'cache_mib', 'measurement_group', 'cold_readiness_seconds', 'elapsed_seconds', 'cpu_seconds', 'cpu_one_core_percent', 'process_exit_code', 'shutdown', 'safety_cutoff', 'gpu_adapter_luid', 'counter_errors', 'counter_instances', 'cache_execution', 'preflight')}
        item.update({key: record.get(key) for key in ('cohort', 'runtime_conflict_observations', 'runtime_conflict_observation_count', 'runtime_conflict_policy', 'contaminated_by_runtime_conflict')})
        item['sample_count'] = len(samples)
        item.update(completed=len(requests), checks=checks, complete_measurement=all(checks.values()), memory_class=memory_class(peak), lifecycle_resources=lifecycle, memory_units={key: unit(lifecycle[key]['max']) for key in MEMORY_FIELDS if key in lifecycle}, phase_resources={phase: resources([sample for sample in samples if sample['phase'] == phase]) for phase in ('load', *PHASES, 'shutdown')}, sample_cadence_seconds=distribution([right['time'] - left['time'] for left, right in zip(samples, samples[1:])]), phases=grouped(requests), pairs=pair_stats(record.get('pairs', [])), first_completion_seconds=min((request['ended_seconds'] for request in requests), default=None), native_overlap_rounds=overlap, native_overlap={phase: distribution([pair['server_active_overlap_seconds'] for pair in slots['pairs'] if pair['phase'] == phase]) for phase in PHASES}, host_model_buffers=[line for line in record['placement_cache_log'] if 'model buffer size' in line])
        runs.append(item)
    configs = {}
    for record in records:
        if record.get('contaminated_by_runtime_conflict'):
            continue
        key = f"n{record['n_cpu_moe']}-c{record['cache_mib']}"
        configs.setdefault(key, []).append(record)
    pooled = {}
    for key, selected in configs.items():
        selected_runs = [run for run in runs if run['profile'] in {record['profile'] for record in selected}]
        complete = [record for record, run in zip(selected, selected_runs) if run['complete_measurement']]
        requests = [request for record in complete for request in record['requests']]
        peaks = [run['lifecycle_resources']['process_gpu_dedicated_bytes']['max'] for run in selected_runs if 'process_gpu_dedicated_bytes' in run['lifecycle_resources']]
        peak = max(peaks) if peaks else None
        variation = max(peaks) - min(peaks) if peaks else None
        paired = pair_stats([pair for record in complete for pair in record.get('pairs', [])])
        item = {'n_cpu_moe': selected[0]['n_cpu_moe'], 'cache_mib': selected[0]['cache_mib'], 'run_ids': [record['profile'] for record in selected], 'run_count': len(selected), 'complete_measurement_run_count': len(complete), 'memory_class': memory_class(peak), 'dedicated_peak': unit(peak), 'observed_peak_variation_bytes': variation, 'margins': {name: {'bytes': boundary - peak, 'MiB': (boundary - peak) / 2**20, 'marginal': boundary >= peak and boundary - peak < variation} for name, boundary in (('target', TARGET), ('ceiling', CEILING))} if peak is not None else None, 'phases': grouped(requests), 'pairs': paired, 'run_simultaneous_medians_seconds': [run['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']['median'] for run in selected_runs if run['complete_measurement']], 'run_serial_medians_seconds': [run['pairs']['serial_pairs']['release_to_both_complete_seconds']['median'] for run in selected_runs if run['complete_measurement']], 'memory_peaks': {field: unit(max(run['lifecycle_resources'][field]['max'] for run in selected_runs if field in run['lifecycle_resources'])) for field in MEMORY_FIELDS if any(field in run['lifecycle_resources'] for run in selected_runs)}, 'pooling': 'Only complete observed runs pooled from original requests/pairs. Classification uses maximum lifecycle sampled peak across ALL runs, including failures.'}
        cache_runs = [run['cache_execution'] for run in selected_runs if run['complete_measurement']]
        counters = {}
        for bucket_name in ('<=8', '>8'):
            buckets = [bucket for cache in cache_runs for bucket in cache['destructor_buckets'] if bucket['ubatch'] == bucket_name]
            hits, misses = sum(bucket['hits'] for bucket in buckets), sum(bucket['misses'] for bucket in buckets)
            counters[bucket_name] = {'emitted_run_buckets': len(buckets), 'hits': hits, 'misses': misses, 'hit_rate': hits / (hits + misses) if hits + misses else None, 'uploaded_mib_rounded_native_sum': sum(bucket['uploaded_mib_rounded_native'] for bucket in buckets) if buckets else None}
        large = [bucket for cache in cache_runs for bucket in cache['large_batches']]
        item['cache_execution_pooled_complete_runs'] = {'requested_mib': selected[0]['cache_mib'], 'native_effective_mib_values': sorted({allocation['effective_mib_rounded_native'] for cache in cache_runs for allocation in cache['allocations']}), 'usable_slots_values': sorted({group['slots'] for cache in cache_runs for group in cache['groups']}), 'source_formula_effective_bytes_values': sorted({group['effective_bytes_source_formula'] for cache in cache_runs for group in cache['groups']}), 'buckets': counters, 'large_batch_emitted_run_count': len(large), 'large_batch_expert_tensor_copies_sum': sum(bucket['experts_copied'] for bucket in large) if large else None, 'large_batch_copied_mib_rounded_native_sum': sum(bucket['copied_mib_rounded_native'] for bucket in large) if large else None, 'limit': 'Complete runs only; counts include native warmup. Missing source-suppressed stats lines are not fabricated measured zero; raw per-run evidence retains failures separately. Logical MiB is not physical PCIe traffic.'}
        pooled[key] = item
    control = pooled.get('n24-c2048')
    if control:
        for item in pooled.values():
            delta = item['dedicated_peak']['bytes'] - control['dedicated_peak']['bytes'] if item['dedicated_peak'] and control['dedicated_peak'] else None
            efficiency = {'dedicated_delta_bytes': delta, 'dedicated_delta_GiB': delta / 2**30 if delta is not None else None, 'definition': 'Signed metric change versus pooled current-session control divided by signed GiB added (negative means saved). Near-zero denominators are not a useful efficiency estimate.', 'pairs': {}, 'phases': {}, 'host_memory': {}}
            if delta:
                divisor = delta / 2**30
                for stat in ('median', 'p95'):
                    current = item['pairs']['simultaneous_pairs']['release_to_both_complete_seconds'][stat]
                    previous = control['pairs']['simultaneous_pairs']['release_to_both_complete_seconds'][stat]
                    efficiency['pairs'][stat + '_ms_per_GiB'] = (current - previous) * 1000 / divisor if current is not None and previous is not None else None
                for phase in PHASES:
                    efficiency['phases'][phase] = {}
                    for channel in ('Self', 'Peer', 'pooled'):
                        efficiency['phases'][phase][channel] = {field + '_per_GiB': (item['phases'][phase][channel][field]['median'] - control['phases'][phase][channel][field]['median']) / divisor if item['phases'][phase][channel][field]['median'] is not None and control['phases'][phase][channel][field]['median'] is not None else None for field in TIMING_FIELDS}
                for field in ('WorkingSetSize', 'PrivateUsage'):
                    if field in item['memory_peaks'] and field in control['memory_peaks']:
                        efficiency['host_memory'][field + '_bytes_per_GiB'] = (item['memory_peaks'][field]['bytes'] - control['memory_peaks'][field]['bytes']) / divisor
            item['versus_control_per_GiB'] = efficiency
            candidate_medians = item['run_simultaneous_medians_seconds']
            control_medians = control['run_simultaneous_medians_seconds']
            item['repeatably_beats_control'] = len(candidate_medians) >= 2 and len(control_medians) >= 2 and max(candidate_medians) < min(control_medians)
    summary = {'baseline': BASELINE, 'target_bytes': TARGET, 'ceiling_bytes': CEILING, 'runs': runs, 'configurations': pooled, 'completed_requests': sum(run['completed'] for run in runs), 'unique_pair_count': len(unique), 'token_count_differences': token_differences, 'limits': '250-ms nominal whole-lifecycle PID exact-adapter samples are lower bounds; native API timings are per request, not pure GPU kernels. Independent manual quality review and Director judgment remain separate. No app/GUI/audio/physical8GiB/coexistence claims.'}
    comparisons = {}
    for key, candidate in pooled.items():
        comparisons[key] = {}
        medians = candidate['run_simultaneous_medians_seconds']
        for reference_key, previous in pooled.items():
            previous_medians = previous['run_simultaneous_medians_seconds']
            if not medians or not previous_medians:
                continue
            pair = candidate['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']
            previous_pair = previous['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']
            serial = candidate['pairs']['serial_pairs']['release_to_both_complete_seconds']
            previous_serial = previous['pairs']['serial_pairs']['release_to_both_complete_seconds']
            memory_delta = candidate['dedicated_peak']['bytes'] - previous['dedicated_peak']['bytes']
            gain = (previous_pair['median'] - pair['median']) * 1000
            comparisons[key][reference_key] = {'repeat_standard_met': len(medians) >= 2 and len(previous_medians) >= 2 and max(medians) < min(previous_medians), 'candidate_run_median_range_seconds': [min(medians), max(medians)], 'reference_run_median_range_seconds': [min(previous_medians), max(previous_medians)], 'pair_median_gain_ms': gain, 'pair_p95_gain_ms': (previous_pair['p95'] - pair['p95']) * 1000, 'serial_median_gain_ms': (previous_serial['median'] - serial['median']) * 1000, 'dedicated_delta_bytes': memory_delta, 'pair_median_gain_ms_per_100MiB_extra': gain / (memory_delta / (100 * 2**20)) if memory_delta > 0 else None}
    summary['pairwise_comparisons'] = comparisons
    curve_steps = []
    for placement in range(23, 17, -1):
        key, previous_key = f'n{placement}-c2048', f'n{placement + 1}-c2048'
        if key not in pooled or previous_key not in pooled:
            continue
        current, previous = pooled[key], pooled[previous_key]
        phases = {phase: {channel: {field: current['phases'][phase][channel][field]['median'] - previous['phases'][phase][channel][field]['median'] for field in TIMING_FIELDS if current['phases'][phase][channel][field]['median'] is not None and previous['phases'][phase][channel][field]['median'] is not None} for channel in ('Self', 'Peer', 'pooled')} for phase in PHASES}
        medians, previous_medians = current['run_simultaneous_medians_seconds'], previous['run_simultaneous_medians_seconds']
        change = current['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']['median'] - previous['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']['median']
        spread = max(max(medians) - min(medians), max(previous_medians) - min(previous_medians)) if medians and previous_medians else None
        curve_steps.append({'from': previous_key, 'to': key, 'additional_gpu_expert_layers': 1, 'dedicated_delta': unit(current['dedicated_peak']['bytes'] - previous['dedicated_peak']['bytes']), 'host_working_set_delta_bytes': current['memory_peaks']['WorkingSetSize']['bytes'] - previous['memory_peaks']['WorkingSetSize']['bytes'], 'private_commit_delta_bytes': current['memory_peaks']['PrivateUsage']['bytes'] - previous['memory_peaks']['PrivateUsage']['bytes'], 'phases_change': phases, 'pair_median_change_ms': change * 1000, 'pair_p95_change_ms': (current['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']['p95'] - previous['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']['p95']) * 1000, 'observed_run_median_spread_ms': spread * 1000 if spread is not None else None, 'pair_change_smaller_than_observed_spread': abs(change) < spread if spread is not None else None, 'limit': 'Most descriptive curve points have one run. Zero measured spread at n1 is not zero uncertainty; small steps do not establish isolated layer effects.'})
    summary['placement_curve_steps'] = curve_steps
    (public / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (public / 'quality-unique.json').write_text(json.dumps(sorted(unique.values(), key=lambda row: row['id']), ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def markdown(summary, directory):
    lines = ['# Issue 254 — every native timing observation group', '', 'Per-request native API timings; TPS is not aggregate throughput. P/C/O are uncached prompt evaluations / prior KV hits / API output tokens. HTTP wall and TTFT include apply-template. Nearest-rank p95; full n/min/max and actual counts in summary.json. No fabricated concurrent native timing joins.', '', '| Run or pooled configuration / phase / channel | n | Prefill ms med/p95 | Prefill TPS med/p95 | Decode ms med/p95 | Decode TPS med/p95 | P/C/O med | HTTP s med/p95 | TTFT s med/p95 |', '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    groups = [(run['profile'], run['phases']) for run in summary['runs']] + [(key + ' pooled', item['phases']) for key, item in summary['configurations'].items()]
    for name, phases in groups:
        for phase, channels in phases.items():
            for channel, group in channels.items():
                cell = lambda field: f"{group[field]['median']:.3f}/{group[field]['p95']:.3f}" if group[field]['n'] else 'N/A'
                counts = '/'.join(str(group[field]['median']) for field in ('prompt_evaluated_tokens', 'prompt_cached_tokens', 'predicted_output_tokens'))
                lines.append(f"| {name} / {phase} / {channel} | {group['prefill_ms']['n']} | {cell('prefill_ms')} | {cell('prefill_tps')} | {cell('decode_ms')} | {cell('decode_tps')} | {counts} | {cell('wall_seconds_including_apply_template')} | {cell('streaming_TTFT_seconds_including_apply_template')} |")
    (directory / 'TUNING8GIB_TIMINGS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/tuning8gib/clean')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    summary = analyze(args.results)
    markdown(summary, Path(__file__).resolve().parent)
    if args.check:
        assert summary['runs'], 'No intended runs collected'
        assert all(run['complete_measurement'] for run in summary['runs'] if run['status'] == 'completed' and not run.get('contaminated_by_runtime_conflict')), 'Completed run has missing evidence; failures and contaminated runs remain ledgered'
        assert all(any(item['n_cpu_moe'] == n and item['cache_mib'] == 2048 for item in summary['configurations'].values()) for n in range(18, 25)), 'Placement curve incomplete'
        check = {'status': 'passed', 'run_count': len(summary['runs']), 'completed_requests': summary['completed_requests'], 'assertions': ['All n24–n18 curve configurations collected', 'Every completed run:44 requests;2/12/30 phases;EOS/bounded automated checks;graceful0;matching identities/P/C;Self0/Peer1/slot2unused;15 native overlaps;effective controls absent;exact PID/adapter memory observed;cache executed;Vulkan_Host buffers'], 'failed_runs_retained': [run['profile'] for run in summary['runs'] if run['status'] != 'completed'], 'manual_quality_review': 'Separate required record, not replaced by automated checks'}
        (args.results / 'focused-check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps({'runs': len(summary['runs']), 'requests': summary['completed_requests'], 'unique_pairs': summary['unique_pair_count'], 'missing_checks': {run['profile']: [key for key, value in run['checks'].items() if not value] for run in summary['runs'] if not run['complete_measurement']}, 'configurations': {key: {'class': item['memory_class'], 'peak': item['dedicated_peak'], 'runs': item['run_count'], 'medians': item['run_simultaneous_medians_seconds']} for key, item in summary['configurations'].items()}}))


if __name__ == '__main__':
    main()
