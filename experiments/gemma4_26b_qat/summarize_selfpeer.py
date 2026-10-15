from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from followup_probes import headroom
from selfpeer_probes import BASELINE, PHASES, slot_evidence
from summarize import distribution
from summarize_timings import stats


def request_stats(requests):
    rows = []
    for request in requests:
        t = request['timings'] or {}
        rows.append({'prefill_ms': t.get('prompt_ms'), 'prefill_tps': t.get('prompt_per_second'), 'decode_ms': t.get('predicted_ms'), 'decode_tps': t.get('predicted_per_second'), 'prompt_evaluated_tokens': t.get('prompt_n'), 'prompt_cached_tokens': t.get('cache_n'), 'prompt_total_tokens': request.get('tokens_evaluated'), 'predicted_output_tokens': t.get('predicted_n')})
    result = stats(rows)
    result['wall_seconds_including_apply_template'] = distribution([r['wall_seconds'] for r in requests])
    result['streaming_TTFT_seconds_including_apply_template'] = distribution([r['ttft_seconds'] for r in requests if r.get('ttft_seconds') is not None])
    return result


def grouped(requests):
    return {phase: {channel: request_stats([r for r in requests if r['phase'] == phase and (channel == 'pooled' or r['channel'] == channel)]) for channel in ('Self', 'Peer', 'pooled')} for phase in PHASES}


def resources(samples):
    return {key: distribution([s[key] for s in samples if s.get(key) is not None]) for key in sorted({k for s in samples for k in s}) if key not in ('phase', 'time')}


def pair_stats(pairs):
    return {phase: {key: distribution([r[key] for r in pairs if r['phase'] == phase]) for key in ('release_to_both_complete_seconds', 'client_start_skew_seconds', 'client_overlap_seconds')} for phase in ('serial_pairs', 'simultaneous_pairs')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    public = Path(__file__).resolve().parent / 'results/selfpeer'
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(public.glob('selfpeer-*.json')) if not any(s in path.name for s in ('.slots.', '.headroom.'))]
    runs, unique = [], {}
    for record in records:
        samples = [json.loads(line) for line in (public / (record['profile'] + '.samples.jsonl')).read_text().splitlines()]
        slots = slot_evidence(args.root / 'selfpeer/runs' / record['profile'] / 'server.log')
        item = {key: record.get(key) for key in ('profile', 'status', 'cache_mib', 'cold_readiness_seconds', 'process_exit_code', 'shutdown', 'safety_cutoff', 'cpu_seconds', 'cpu_one_core_percent', 'gpu_adapter_luid', 'counter_errors', 'counter_instances', 'preflight')}
        item.update(completed=len(record['requests']), phases=grouped(record['requests']), pairs=pair_stats(record['pairs']), lifecycle_resources=resources(samples), phase_resources={phase: resources([s for s in samples if s['phase'] == phase]) for phase in ('load', *PHASES, 'shutdown')}, active_resources=resources([s for s in samples if s['phase'] in ('serial_pairs', 'simultaneous_pairs')]), host_model_buffers=[line for line in record['placement_cache_log'] if 'model buffer size' in line], cache_counters=[line for line in record['placement_cache_log'] if 'llama_moe_cache:' in line], server_overlap={phase: distribution([p['server_active_overlap_seconds'] for p in slots['pairs'] if p['phase'] == phase]) for phase in PHASES}, server_overlap_rounds=sum(p['phase'] == 'simultaneous_pairs' and p['server_active_overlap_seconds'] > 0 for p in slots['pairs']), first_completion_launch_seconds=record['requests'][0]['ended_seconds'] if record['requests'] else None)
        item['server_pair_observations'] = slots['pairs']
        item['native_slots_used'] = sorted({event['slot'] for event in slots['events']})
        item['unreleased_native_tasks'] = slots['unreleased']
        runs.append(item)
        for request in record['requests']:
            key = (request['input'], request['output'])
            row = unique.setdefault(key, {'input': request['input'], 'output': request['output'], 'occurrences': [], 'automated_checks': request['quality_checks']})
            row['occurrences'].append({'run': record['profile'], 'phase': request['phase'], 'channel': request['channel'], 'round': request['round']})
    pooled = {}
    for cache in (2048, 3072):
        selected = [r for r in records if r['cache_mib'] == cache]
        requests = [q for r in selected for q in r['requests']]
        pooled[str(cache)] = {'run_count': len(selected), 'completed': len(requests), 'phases': grouped(requests), 'pairs': pair_stats([p for r in selected for p in r['pairs']]), 'pooling': 'All original observations pooled before distributions; no percentile-of-percentile.'}
        native_pairs = [p for run in runs if run['cache_mib'] == cache for p in run['server_pair_observations']]
        pooled[str(cache)]['server_overlap'] = {phase: distribution([p['server_active_overlap_seconds'] for p in native_pairs if p['phase'] == phase]) for phase in PHASES}
    signature = lambda r: [(q['phase'], q['index'], q['round'], q['channel'], q['slot'], q['case_index'], q['prompt_sha256'], q['temperature'], q['seed'], q['max_output_tokens']) for q in r['requests']]
    token_differences = []
    if records:
        reference = records[0]
        for record in records[1:]:
            for before, after in zip(reference['requests'], record['requests']):
                values = lambda q: {k: q['timings'].get(k) for k in ('prompt_n', 'cache_n', 'predicted_n')}
                if values(before) != values(after):
                    token_differences.append({'reference': reference['profile'], 'run': record['profile'], 'phase': after['phase'], 'channel': after['channel'], 'round': after['round'], 'reference_counts': values(before), 'counts': values(after)})
    summary = {'experiment_baseline': BASELINE, 'runs': runs, 'pooled_by_cache': pooled, 'completed': sum(r['completed'] for r in runs), 'unique_outputs': len(unique), 'prompt_workload_matches': bool(records) and all(signature(r) == signature(records[0]) for r in records), 'token_count_differences': token_differences, 'limits': 'API timings associated with original requests only; no concurrent native timing joins. Global paging not process-attributable;250ms sampled peaks lower bounds. Native-only, not app/audio/VRChat. Evidence only; judgment and selection reserved to Director.'}
    (public / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (public / 'quality-unique.json').write_text(json.dumps(list(unique.values()), ensure_ascii=False, indent=2), encoding='utf-8')
    if args.check:
        assert len(records) == 4 and summary['completed'] == 176
        assert summary['prompt_workload_matches']
        for record, run in zip(records, runs):
            assert len(record['requests']) == 44 and record['status'] == 'completed'
            assert record['shutdown'] == 'graceful_console_ctrl_c' and record['process_exit_code'] == 0 and not record.get('safety_cutoff')
            assert Counter(q['phase'] for q in record['requests']) == {'first_prefix': 2, 'serial_pairs': 12, 'simultaneous_pairs': 30}
            assert all(q['stop_type'] == 'eos' and all(q['quality_checks'][k] for k in ('nonempty', 'no_reasoning', 'not_truncated')) for q in record['requests'])
            assert all(q['slot'] == (0 if q['channel'] == 'Self' else 1) for q in record['requests'])
            assert all(value is None for value in record['allowlisted_backend_environment_effective'].values())
            assert run['server_overlap_rounds'] == 15
            assert run['native_slots_used'] == [0, 1] and not run['unreleased_native_tasks']
            assert all('Vulkan_Host' in line for line in run['host_model_buffers'] if 'Vulkan0' not in line)
        check = {'status': 'passed', 'completed': 176, 'run_count': 4, 'assertions': ['44requests/run;2first-prefix,12serial,30simultaneous', 'Exact prompts/workload match across four runs', 'All176EOS/nonempty/no-reasoning/not-truncated', 'Pinned Self0/Peer1,slot2unused', 'All15simultaneous rounds/run have positive native-server active overlap', 'All effective GGML allowlist absent', 'Actual Vulkan_Host model buffers', 'Four graceful0 exits,no safety cutoff'], 'token_differences_disclosed': len(token_differences)}
        (public / 'focused-check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    release = headroom()
    release['owned_exits'] = [{'run': r['profile'], 'exit': r['process_exit_code'], 'shutdown': r['shutdown']} for r in runs]
    (public / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
    lines = ['# Self/Peer native timings — every run and pooled original observations', '', 'Native API timings belong to original requests; wall/TTFT include apply-template and local streaming completion. Not app/audio E2E. n/median/p95/min/max and P/cache/output counts retained in summary.json. All pooled distributions use original observations.', '', '| Group /phase /channel | n | Prefill ms med/p95 | Prefill TPS med/p95 | Decode ms med/p95 | Decode TPS med/p95 | P /C /O med | HTTP wall s med/p95 | TTFT s med/p95 |', '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for name, groups in [(r['profile'], r['phases']) for r in runs] + [(f'pooled-cache{k}', v['phases']) for k, v in pooled.items()]:
        for phase, channels in groups.items():
            for channel, group in channels.items():
                cell = lambda key: 'N/A' if not group[key]['n'] else f"{group[key]['median']:.3f}/{group[key]['p95']:.3f}"
                count = lambda key: str(group[key]['median'])
                lines.append(f"| {name} /{phase} /{channel} | {group['prefill_ms']['n']} | {cell('prefill_ms')} | {cell('prefill_tps')} | {cell('decode_ms')} | {cell('decode_tps')} | {count('prompt_evaluated_tokens')} /{count('prompt_cached_tokens')} /{count('predicted_output_tokens')} | {cell('wall_seconds_including_apply_template')} | {cell('streaming_TTFT_seconds_including_apply_template')} |")
    (public.parent.parent / 'SELFPEER_TIMINGS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'runs': len(runs), 'completed': summary['completed'], 'unique_outputs': len(unique), 'token_differences': len(token_differences), 'remaining_model_processes': release['model_runtime_processes'], 'focused_check': args.check}))
    for run in runs:
        print(json.dumps({'run': run['profile'], 'pairs': run['pairs'], 'server_overlap_rounds': run['server_overlap_rounds'], 'prefill_decode_simultaneous_pooled': run['phases']['simultaneous_pairs']['pooled'], 'peaks': {key: value['max'] for key, value in run['lifecycle_resources'].items() if key in ('process_gpu_dedicated_bytes', 'process_gpu_shared_bytes', 'device_gpu_dedicated_bytes', 'WorkingSetSize', 'PrivateUsage')}}))


if __name__ == '__main__':
    main()
