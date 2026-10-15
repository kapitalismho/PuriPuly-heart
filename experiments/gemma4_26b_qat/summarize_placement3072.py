from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from followup_probes import headroom
from summarize import distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.results.glob('followup-*.json')) if not any(s in p.name for s in ('.analysis.', '.headroom.'))]
    analyses = {r['profile']: json.loads((args.results / (r['profile'] + '.analysis.json')).read_text(encoding='utf-8')) for r in records}
    unique = {}
    for record in records:
        for request in record['requests']:
            key = (request['pair'], request['input'], request['output'])
            item = unique.setdefault(key, {'pair': key[0], 'input': key[1], 'output': key[2], 'occurrences': [], 'automated_checks': request['quality_checks']})
            item['occurrences'].append({'run': record['profile'], 'phase': request['phase'], 'index': request['index']})
    (args.results / 'quality-unique.json').write_text(json.dumps(list(unique.values()), ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for record in records:
        analysis = analyses[record['profile']]
        resources = analysis['resource_distributions']
        rows.append({'run_id': record['profile'], 'status': record['status'], 'completed': len(record['requests']), 'cold_readiness_seconds': record.get('cold_readiness_seconds'), 'launch_to_first_completion_seconds': record['requests'][0]['ended_seconds'] if record['requests'] else None, 'first_http_seconds': record['requests'][0]['wall_seconds'] if record['requests'] else None, 'shutdown': record.get('shutdown'), 'exit_code': record.get('process_exit_code'), 'cpu_seconds': record.get('cpu_seconds'), 'cpu_one_core_percent': record.get('cpu_one_core_percent'), 'adapter': record.get('gpu_adapter_luid'), 'counter_errors': record.get('counter_errors'), 'preflight': analysis['preflight_headroom'], 'resources': resources, 'phase_resources': {phase: {key: distribution([s[key] for s in [json.loads(line) for line in (args.results / (record['profile'] + '.samples.jsonl')).read_text().splitlines()] if s['phase'] == phase and s.get(key) is not None]) for key in resources} for phase in ('load', 'first_prefix', 'hot_same_pair', 'pair_alternation', 'three_slots', 'shutdown')}, 'phases': analysis['phases'], 'host_buffers': analysis['native_host_model_buffers'], 'cache_counters': analysis['lifetime_cache_counters'], 'safety_cutoff': record.get('safety_cutoff')})
    (args.results / 'summary.json').write_text(json.dumps({'experiment_baseline': 'ee4620b5ce96b56202c17a06bf979852cb6d6cbd', 'runs': rows, 'completed': sum(r['completed'] for r in rows), 'unique_outputs': len(unique)}, ensure_ascii=False, indent=2), encoding='utf-8')
    release = headroom()
    release['recorded_utc'] = datetime.now(timezone.utc).isoformat()
    release['owned_process_exits'] = [{'run': r['run_id'], 'exit': r['exit_code'], 'shutdown': r['shutdown']} for r in rows]
    (args.results / 'release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
    print(json.dumps({'runs': len(rows), 'completed': sum(r['completed'] for r in rows), 'unique_outputs': len(unique), 'remaining_model_processes': release['model_runtime_processes']}))
    for row in rows:
        print(json.dumps({k: row[k] for k in ('run_id', 'completed', 'cold_readiness_seconds', 'launch_to_first_completion_seconds', 'first_http_seconds', 'cpu_seconds', 'cpu_one_core_percent')}))
        print(json.dumps({k: row['resources'][k] for k in ('process_gpu_dedicated_bytes', 'process_gpu_shared_bytes', 'device_gpu_dedicated_bytes', 'WorkingSetSize', 'PrivateUsage', 'system_available_physical_bytes', 'system_commit_headroom_bytes', 'system_pages_input_per_second')}))
        print(json.dumps({p: {k: v['median'] for k, v in g.items() if isinstance(v, dict) and 'median' in v} for p, g in row['phases'].items()}))


if __name__ == '__main__':
    main()
