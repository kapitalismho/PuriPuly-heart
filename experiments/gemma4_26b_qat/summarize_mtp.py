from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from selfpeer_probes import PHASES
from summarize import distribution
from summarize_selfpeer import grouped, pair_stats, resources
from summarize_tuning8gib import CEILING, MEMORY_FIELDS, TARGET, memory_class, signature, unit


BASELINE = 'c451f1b04a940f8d3e297ca58d13eefc338baee8'


def configuration(record):
    return f"c{record['cache_mib']}-" + (f"mtp{record['draft_max']}" if record['draft_max'] else 'off')


def acceptance(requests):
    observations = []
    for request in requests:
        timings = request.get('timings') or {}
        generated = timings.get('draft_n')
        accepted = timings.get('draft_n_accepted')
        observations.append({'phase': request['phase'], 'channel': request['channel'], 'round': request['round'], 'draft_n': generated, 'draft_n_accepted': accepted, 'acceptance_rate': accepted / generated if generated and accepted is not None else None, 'native_predicted_n': timings.get('predicted_n'), 'native_predicted_ms': timings.get('predicted_ms'), 'native_predicted_tps': timings.get('predicted_per_second'), 'whole_request_output_tokens_per_http_second': request['tokens_predicted'] / request['wall_seconds'] if request.get('tokens_predicted') is not None and request['wall_seconds'] > 0 else None})
    emitted = [row for row in observations if row['draft_n'] is not None]
    drafted = sum(row['draft_n'] for row in emitted)
    accepted = sum(row['draft_n_accepted'] for row in emitted if row['draft_n_accepted'] is not None)
    return {'api_rows_with_draft_counters': len(emitted), 'draft_n_sum': drafted if emitted else None, 'draft_n_accepted_sum': accepted if emitted else None, 'weighted_acceptance_rate': accepted / drafted if drafted else None, 'observations': observations, 'semantics': 'API draft_n counts proposals; draft_n_accepted excludes bonus target tokens and may include accepted suffix past EOS, so it is not delivered-output count. API predicted_n counts generated results including first token/EOS; native TPS=(predicted_n-1)*1000/predicted_ms. Elapsed includes drafting/verification/synchronization/scheduling. These server-slot stats are not libllama destructor evalTPS. Missing API draft fields are unavailable/unemitted, not invented zero.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/mtp')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    records = []
    for path in args.results.glob('mtp-*.json'):
        if any(part in path.name for part in ('.slots.', '.headroom.')):
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if 'requests' in record:
            records.append(record)
    records.sort(key=lambda record: (record['started_utc'], record['profile']))
    reference = next((record for record in records if record['cache_mib'] == 2890 and record['draft_max'] == 0 and not record.get('contaminated_by_runtime_conflict')), None)
    reference_requests = {signature(request): request for request in reference['requests']} if reference else {}
    runs, unique, differences = [], {}, []
    for record in records:
        name = record['profile']
        samples = [json.loads(line) for line in (args.results / (name + '.samples.jsonl')).read_text(encoding='utf-8').splitlines()]
        slots = json.loads((args.results / (name + '.slots.json')).read_text(encoding='utf-8'))
        requests = record['requests']
        for request in requests:
            previous = reference_requests.get(signature(request))
            counts = lambda row: {key: (row.get('timings') or {}).get(key) for key in ('prompt_n', 'cache_n', 'predicted_n')}
            if previous and counts(previous) != counts(request):
                differences.append({'reference': reference['profile'], 'run': name, 'phase': request['phase'], 'channel': request['channel'], 'round': request['round'], 'reference_counts': counts(previous), 'counts': counts(request)})
            key = (request['input'], request['output'])
            row = unique.setdefault(key, {'id': hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16], 'input': request['input'], 'output': request['output'], 'automated_checks': request['quality_checks'], 'occurrences': []})
            row['occurrences'].append({'run': name, 'phase': request['phase'], 'channel': request['channel'], 'round': request['round'], 'stop_type': request['stop_type']})
        life = resources(samples)
        peak = life.get('process_gpu_dedicated_bytes', {}).get('max')
        draft = acceptance(requests)
        native = (args.results / (name + '.native.log')).read_text(encoding='utf-8')
        cumulative = re.findall(r'#gen tokens =\s*(\d+), #acc tokens =\s*(\d+)', native)
        draft['native_last_cumulative_counters'] = {'draft_n': int(cumulative[-1][0]), 'draft_n_accepted': int(cumulative[-1][1])} if cumulative else None
        draft_region = native.split('loading draft model', 1)[1] if 'loading draft model' in native else ''
        draft_gpu = re.findall(r'Vulkan0 model buffer size =\s*([\d.]+) MiB', draft_region)
        draft['native_drafter_vulkan_model_mib'] = [float(value) for value in draft_gpu]
        draft['shared_target_kv_logged'] = 'sharing with layer 29' in draft_region and 'sharing with layer 28' in draft_region
        overlap = sum(pair['phase'] == 'simultaneous_pairs' and pair['server_active_overlap_seconds'] > 0 for pair in slots['pairs'])
        checks = {
            'complete44': record['status'] == 'completed' and len(requests) == 44,
            'phase_counts': Counter(request['phase'] for request in requests) == {'first_prefix': 2, 'serial_pairs': 12, 'simultaneous_pairs': 30},
            'identical_inputs_and_sampling': bool(reference) and [signature(request) for request in requests] == [signature(request) for request in reference['requests']],
            'eos_and_bounded_automated_checks': bool(requests) and all(request['stop_type'] == 'eos' and all(request['quality_checks'][key] for key in ('nonempty', 'no_reasoning', 'not_truncated')) for request in requests),
            'fresh_graceful_exit0': record.get('shutdown') == 'graceful_console_ctrl_c' and record.get('process_exit_code') == 0 and not record.get('safety_cutoff'),
            'exclusive_runtime_observed': record.get('runtime_conflict_observation_count', 0) > 0 and not record.get('contaminated_by_runtime_conflict'),
            'native_overlap15': overlap == 15 and slots['slot_counts'] == {'0': 22, '1': 22} and not slots['unreleased'],
            'self0_peer1_slot2_unused': sorted({event['slot'] for event in slots['events']}) == [0, 1] and all(request['slot'] == (0 if request['channel'] == 'Self' else 1) for request in requests),
            'pid_exact_adapter_observed': peak is not None and peak > 0 and bool(record.get('counter_instances', {}).get('process_gpu_dedicated_bytes')) and all(instance.startswith(f"pid_{record['server_pid']}_") and record['gpu_adapter_luid'] in instance.lower() for instance in record.get('counter_instances', {}).get('process_gpu_dedicated_bytes', [])),
            'pid_dedicated_load_and_all_workload_phases': all(any(sample['phase'] == phase and sample.get('process_gpu_dedicated_bytes', 0) for sample in samples) for phase in ('load', *PHASES)),
            'backend_environment_controls_absent': all(value is None for value in record['allowlisted_backend_environment_effective'].values()),
            'cache_used': record['cache_execution']['state'] == 'allocated_and_used',
            'mtp_executed_when_enabled': (draft['draft_n_sum'] is not None and draft['draft_n_sum'] > 0 and draft['draft_n_accepted_sum'] > 0) if record['draft_max'] else draft['api_rows_with_draft_counters'] == 0,
            'mtp_native_counters_match_api': bool(cumulative) and tuple(map(int, cumulative[-1])) == (draft['draft_n_sum'], draft['draft_n_accepted_sum']) if record['draft_max'] else not cumulative,
            'mtp_drafter_vulkan_and_shared_kv': bool(draft_gpu) and any(float(value) > 0 for value in draft_gpu) and draft['shared_target_kv_logged'] if record['draft_max'] else not draft_gpu,
            'native_api_decode_rate_formula': all((request.get('timings') or {}).get('predicted_ms', 0) > 0 and abs(request['timings']['predicted_per_second'] - (request['timings']['predicted_n'] - 1) * 1000 / request['timings']['predicted_ms']) < 0.000001 for request in requests) if requests else False,
            'requested_draft_max_effective': f"n_max={record['draft_max']}," in native if record['draft_max'] else all(not slot.get('speculative') for slot in record.get('slots', [])),
        }
        row = {key: record.get(key) for key in ('profile', 'started_utc', 'status', 'error', 'cohort', 'cache_mib', 'draft_max', 'measurement_group', 'cold_readiness_seconds', 'elapsed_seconds', 'cpu_seconds', 'cpu_one_core_percent', 'shutdown', 'process_exit_code', 'safety_cutoff', 'gpu_adapter_luid', 'counter_errors', 'counter_instances', 'runtime_conflict_observations', 'runtime_conflict_observation_count', 'contaminated_by_runtime_conflict', 'preflight', 'cache_execution', 'speculative_log_evidence', 'slots')}
        row.update(configuration=configuration(record), completed=len(requests), checks=checks, complete_measurement=all(checks.values()), memory_class=memory_class(peak), memory_units={field: unit(life[field]['max']) for field in MEMORY_FIELDS if field in life}, lifecycle_resources=life, phase_resources={phase: resources([sample for sample in samples if sample['phase'] == phase]) for phase in ('load', *PHASES, 'shutdown')}, sample_count=len(samples), sample_cadence_seconds=distribution([right['time']-left['time'] for left, right in zip(samples, samples[1:])]), phases=grouped(requests), pairs=pair_stats(record.get('pairs', [])), first_completion_seconds=min((request['ended_seconds'] for request in requests), default=None), native_overlap_rounds=overlap, acceptance=draft, acceptance_by_phase_channel={phase: {channel: acceptance([request for request in requests if request['phase'] == phase and (channel == 'pooled' or request['channel'] == channel)]) for channel in ('Self', 'Peer', 'pooled')} for phase in PHASES})
        runs.append(row)
    configs = {}
    for key in sorted({configuration(record) for record in records}):
        selected = [record for record in records if configuration(record) == key and not record.get('contaminated_by_runtime_conflict')]
        if not selected:
            continue
        selected_runs = [run for run in runs if run['profile'] in {record['profile'] for record in selected}]
        complete = [record for record, run in zip(selected, selected_runs) if run['complete_measurement']]
        requests = [request for record in complete for request in record['requests']]
        peaks = [run['memory_units']['process_gpu_dedicated_bytes']['bytes'] for run in selected_runs if 'process_gpu_dedicated_bytes' in run['memory_units']]
        peak, variation = (max(peaks), max(peaks)-min(peaks)) if peaks else (None, None)
        configs[key] = {'cache_mib': selected[0]['cache_mib'], 'draft_max': selected[0]['draft_max'], 'run_ids': [record['profile'] for record in selected], 'run_count': len(selected), 'complete_run_count': len(complete), 'memory_class': memory_class(peak), 'dedicated_peak': unit(peak), 'observed_peak_variation_bytes': variation, 'margins': {name: {'bytes': boundary-peak, 'MiB': (boundary-peak)/2**20, 'marginal': peak <= boundary and boundary-peak < variation} for name, boundary in (('target', TARGET), ('ceiling', CEILING))} if peak is not None else None, 'memory_peaks': {field: unit(max(run['lifecycle_resources'][field]['max'] for run in selected_runs if field in run['lifecycle_resources'])) for field in MEMORY_FIELDS if any(field in run['lifecycle_resources'] for run in selected_runs)}, 'phases': grouped(requests), 'pairs': pair_stats([pair for record in complete for pair in record.get('pairs', [])]), 'run_pair_medians': [run['pairs']['simultaneous_pairs']['release_to_both_complete_seconds']['median'] for run in selected_runs if run['complete_measurement']], 'acceptance': acceptance(requests), 'acceptance_by_phase_channel': {phase: {channel: acceptance([request for request in requests if request['phase'] == phase and (channel == 'pooled' or request['channel'] == channel)]) for channel in ('Self', 'Peer', 'pooled')} for phase in PHASES}}
    comparisons = {}
    for key, candidate in configs.items():
        comparisons[key] = {}
        for previous_key, previous in configs.items():
            medians, previous_medians = candidate['run_pair_medians'], previous['run_pair_medians']
            if not medians or not previous_medians:
                continue
            current_pair, previous_pair = (config['pairs']['simultaneous_pairs']['release_to_both_complete_seconds'] for config in (candidate, previous))
            current_serial, previous_serial = (config['pairs']['serial_pairs']['release_to_both_complete_seconds'] for config in (candidate, previous))
            comparisons[key][previous_key] = {'all_repeat_standard': len(medians) >= 2 and len(previous_medians) >= 2 and max(medians) < min(previous_medians), 'pair_median_gain_ms': (previous_pair['median']-current_pair['median'])*1000, 'pair_p95_gain_ms': (previous_pair['p95']-current_pair['p95'])*1000, 'serial_median_gain_ms': (previous_serial['median']-current_serial['median'])*1000, 'serial_p95_gain_ms': (previous_serial['p95']-current_serial['p95'])*1000, 'dedicated_delta_bytes': candidate['dedicated_peak']['bytes']-previous['dedicated_peak']['bytes']}
    result = {'baseline': BASELINE, 'cohort': 'owner_authorized_mtp_followup', 'runs': runs, 'configurations': configs, 'comparisons': comparisons, 'completed_requests': sum(run['completed'] for run in runs), 'unique_pair_count': len(unique), 'prompt_and_output_token_differences': differences, 'token_semantics': 'Native completion timings use server-slot statistics, not libllama destructor counters. predictedTPS=(predicted_n-1)*1000/predicted_ms; accepted/emitted output includes bonus target tokens, draft_n_accepted excludes bonus. P/C differences are disclosed, never forced. top-level tokens_evaluated is full prompt; tokens_cached is retained KV length, not prompt hits.', 'limits': 'Whole-lifecycle250ms nominal PID/exact-adapter sampled peaks are lower bounds. Native-only; no app/GUI/audio/physical8GiB/coexistence/adoption claim. Historical c451 evidence unchanged and not pooled.'}
    result['token_semantics'] += ' Accepted speculative suffix may extend past EOS; never derive delivered-output TPS from draft_n_accepted. Native generated-token counts include EOS under the pinned server convention.'
    result['actual_token_totals_by_configuration'] = {key: {field: sum((request.get('timings') or {}).get(native_key, 0) for record in records if configuration(record) == key and not record.get('contaminated_by_runtime_conflict') for request in record['requests']) for field, native_key in (('P', 'prompt_n'), ('C', 'cache_n'), ('O', 'predicted_n'))} for key in configs}
    (args.results / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.results / 'quality-unique.json').write_text(json.dumps(sorted(unique.values(), key=lambda row: row['id']), ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# MTP follow-up — all runs/configurations/phases/channels', '', 'Native API rates are per-request server-slot statistics, not libllama destructor TPS. API decode rate=(O-1)/elapsed, includes drafting/verification/scheduling. P=uncached prompt,C=priorKVhits,O=native generated results includingEOS. Draft accepted prefixes may include suffix pastEOS: not delivered-output counts. HTTP/TTFT include apply-template. All n/median/p95/min/max/counts/acceptance retained in summary.json.', '', '| Run or config / phase / channel | n | Prefill ms med/p95 ; TPS med/p95 | Decode ms med/p95 ; TPS med/p95 | P/C/O med | HTTP median/p95 ms | TTFT median/p95 ms | Proposed / accepted / rate |', '| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |']
    for name, item in [(run['profile'], run) for run in runs] + [(key+' pooled', item) for key, item in configs.items()]:
        for phase, channels in item['phases'].items():
            for channel, timing in channels.items():
                number = lambda field: 'N/A' if timing[field]['median'] is None else f"{timing[field]['median']:.3f}"
                cell = lambda field: 'N/A' if timing[field]['median'] is None else f"{timing[field]['median']*1000:.3f}/{timing[field]['p95']*1000:.3f}"
                native_cell = lambda field: 'N/A' if timing[field]['median'] is None else f"{timing[field]['median']:.3f}/{timing[field]['p95']:.3f}"
                a = item['acceptance_by_phase_channel'][phase][channel]
                rate = 'N/A' if a['weighted_acceptance_rate'] is None else f"{a['weighted_acceptance_rate']*100:.2f}%"
                lines.append(f"|{name}/{phase}/{channel}|{timing['prefill_ms']['n']}|{native_cell('prefill_ms')} ;{native_cell('prefill_tps')}|{native_cell('decode_ms')} ;{native_cell('decode_tps')}|{number('prompt_evaluated_tokens')}/{number('prompt_cached_tokens')}/{number('predicted_output_tokens')}|{cell('wall_seconds_including_apply_template')}|{cell('streaming_TTFT_seconds_including_apply_template')}|{a['draft_n_sum'] if a['draft_n_sum'] is not None else 'unemitted'} /{a['draft_n_accepted_sum'] if a['draft_n_accepted_sum'] is not None else 'unemitted'} /{rate}|")
    (Path(__file__).resolve().parent / 'MTP_TIMINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if args.check:
        assert records and reference, 'No fresh2890/off control collected'
        assert all(run['complete_measurement'] for run in runs if run['status'] == 'completed' and not run.get('contaminated_by_runtime_conflict')), 'Completed run lacks required evidence; retain failure'
        check = {'status': 'passed', 'runs': len(runs), 'completed': result['completed_requests'], 'unique_pairs': len(unique), 'assertions': ['Unchanged44-request workload and sampling identities','EOS/bounded automated surface checks','Fresh graceful0/no watchdog and no observed named-model conflict','Exact PID/adapter load+phase observations','15native overlaps;Self0/Peer1/unused2','Native cache execution','MTP-enabled runs have actual API proposed+accepted draft tokens;off runs omit draft counters'], 'quality': 'Independent manual unique-pair and final ledger audit required separately'}
        (args.results / 'focused-check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps({'runs': len(runs), 'completed': result['completed_requests'], 'unique_pairs': len(unique), 'missing_checks': {run['profile']: [key for key, value in run['checks'].items() if not value] for run in runs if not run['complete_measurement']}, 'configs': {key: {'class': cfg['memory_class'], 'peak': cfg['dedicated_peak'], 'medians': cfg['run_pair_medians'], 'proposed': cfg['acceptance']['draft_n_sum'], 'accepted': cfg['acceptance']['draft_n_accepted_sum']} for key, cfg in configs.items()}}))


if __name__ == '__main__':
    main()
