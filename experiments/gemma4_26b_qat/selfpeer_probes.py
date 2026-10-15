from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import sys
from types import SimpleNamespace

import run_matrix as matrix
from followup_probes import CONTROLS, MODEL_RUNTIMES, headroom

BASELINE = 'ccab84af268ba58ce47d501132b69e7a583d861b'
WORKLOAD_SOURCE = 'c3da5bb5f97ede9b2cfb31ce1d0f1f590e7e2ec6'
PHASES = ('first_prefix', 'serial_pairs', 'simultaneous_pairs')
CHANNELS = (('Self', 0, (1, 5, 9)), ('Peer', 1, (0, 4, 8)))


def workload(base, result, phase, started):
    result['workload'] = 'selfpeer-two-slots-v1'
    result['pairs'] = []

    def request(channel, rep, temperature):
        name, slot, cases = CHANNELS[channel]
        row = matrix.request_translation(base, matrix.CASES[cases[rep % 3]], slot, phase[0], rep * 2 + channel, temperature)
        row.update(channel=name, round=rep, case_index=cases[rep % 3])
        return row

    phase[0] = 'first_prefix'
    for channel in range(2):
        result['requests'].append(request(channel, 0, 0.0))
    for label, rounds in (('serial_pairs', 6), ('simultaneous_pairs', 15)):
        phase[0] = label
        for rep in range(rounds):
            release = {}
            if label == 'simultaneous_pairs':
                barrier = threading.Barrier(3, action=lambda: release.update(time=time.perf_counter()))

                def released_request(channel):
                    barrier.wait(timeout=30)
                    return request(channel, rep, 0.6)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(released_request, channel) for channel in range(2)]
                    barrier.wait(timeout=30)
                    for future in futures:
                        result['requests'].append(future.result())
            else:
                release['time'] = time.perf_counter()
                for channel in ((0, 1) if rep % 2 == 0 else (1, 0)):
                    result['requests'].append(request(channel, rep, 0.6))
            rows = [r for r in result['requests'] if r['phase'] == label and r['round'] == rep]
            result['pairs'].append({
                'phase': label, 'round': rep, 'release_seconds': release['time'] - started,
                'release_to_both_complete_seconds': max(r['ended_perf'] for r in rows) - release['time'],
                'client_start_skew_seconds': abs(rows[0]['started_perf'] - rows[1]['started_perf']),
                'client_overlap_seconds': max(0, min(r['ended_perf'] for r in rows) - max(r['started_perf'] for r in rows)),
            })


def slot_evidence(log_path):
    events, active, intervals = [], {}, []
    for line_number, line in enumerate(log_path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        match = re.match(r'(\d+)\.(\d+)\.(\d+)\.(\d+) I slot .*?id\s+(\d+) \| task (\d+) \| (processing task|stop processing)', line)
        if not match:
            continue
        minute, second, millis, micros, slot, task = map(int, match.groups()[:6])
        stamp = minute * 60 + second + millis / 1000 + micros / 1000000
        event = {'native_seconds': stamp, 'slot': slot, 'task': task, 'event': match[7], 'line': line_number}
        events.append(event)
        key = (slot, task)
        if match[7] == 'processing task':
            active[key] = event
        elif key in active:
            launch = active.pop(key)
            intervals.append({'slot': slot, 'task': task, 'start': launch['native_seconds'], 'end': stamp, 'start_line': launch['line'], 'end_line': line_number})
    by_slot = {slot: sorted([r for r in intervals if r['slot'] == slot], key=lambda r: r['start']) for slot in (0, 1)}
    pairs = []
    if all(len(by_slot[slot]) == 22 for slot in (0, 1)):
        for index in range(22):
            left, right = (by_slot[slot][index] for slot in (0, 1))
            pairs.append({'phase': 'first_prefix' if index == 0 else 'serial_pairs' if index <= 6 else 'simultaneous_pairs', 'round': 0 if index == 0 else index - 1 if index <= 6 else index - 7, 'self_task': left['task'], 'peer_task': right['task'], 'server_active_overlap_seconds': max(0, min(left['end'], right['end']) - max(left['start'], right['start'])), 'self_interval': left, 'peer_interval': right})
    return {'events': events, 'intervals': intervals, 'unreleased': list(active.values()), 'slot_counts': {str(slot): len(rows) for slot, rows in by_slot.items()}, 'pairs': pairs, 'association': 'Per-slot ordinal pairing is valid only with22 complete intervals in each pinned slot and one request/channel/round; native timings are not joined to API request timings.'}


def cache_evidence(lines, requested_mib):
    allocation = []
    buckets = []
    groups = []
    large = []
    warnings = []
    for line in lines:
        match = re.search(r'MoE cache size =\s*([\d.]+) MiB for ([\d.]+) MiB of host experts', line)
        if match:
            allocation.append({'effective_mib_rounded_native': float(match[1]), 'host_experts_mib_rounded_native': float(match[2]), 'line': line})
        match = re.search(r'(\d+) layers, ([a-z0-9_]+):\s*(\d+) slots \(([\d.]+)%\)', line)
        if match:
            layers, slots = int(match[1]), int(match[3])
            groups.append({'layers': layers, 'tensor_type': match[2], 'slots': slots, 'host_expert_count': layers * 128, 'slot_coverage_fraction': slots / (layers * 128), 'effective_bytes_source_formula': (slots + 1) * 3345408, 'slot_bytes_source_formula': 3345408, 'line': line})
        match = re.search(r'llama_moe_cache: ubatch\s*(<=|>)\s*(\d+): hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB', line)
        if match:
            hits, misses = int(match[3]), int(match[4])
            buckets.append({'ubatch': match[1] + match[2], 'hits': hits, 'misses': misses, 'hit_rate': hits / (hits + misses) if hits + misses else None, 'uploaded_mib_rounded_native': float(match[6]), 'line': line})
        match = re.search(r'large batches: (\d+) experts copied from the cache, ([\d.]+) MiB', line)
        if match:
            large.append({'experts_copied': int(match[1]), 'copied_mib_rounded_native': float(match[2]), 'line': line})
        if 'cache' in line.lower() and any(word in line.lower() for word in ('disable', 'warning', 'too small', 'not enough', 'slots')):
            warnings.append(line)
    return {'requested_mib': requested_mib, 'state': 'allocated_and_used' if allocation and buckets and groups and any(b['hits'] + b['misses'] > 0 and b['uploaded_mib_rounded_native'] > 0 for b in buckets) else 'cache_off' if not allocation else 'execution_evidence_incomplete', 'allocations': allocation, 'groups': groups, 'destructor_buckets': buckets, 'large_batches': large, 'warnings': warnings, 'slot_formula_source': 'llama.cpp8e2d31e0e src/llama-moe-cache.cpp:264-328,444-456; fused Q4_0 gate/up2230272+down1115136 bytes/slot and one mandatory padding plane; exact source-derived bytes, native effective MiB rounded2dp.', 'scope': 'Whole lifecycle including native warmup; logical upload/copy MiB, not measured PCIe traffic.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--n-cpu-moe', type=int, choices=range(18, 25), required=True)
    parser.add_argument('--cache-mib', type=int, required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--group', choices=('control', 'curve', 'target', 'bracket', 'band', 'repeat'), required=True)
    parser.add_argument('--results', type=Path, default=matrix.REPO / 'experiments/gemma4_26b_qat/results/tuning8gib/clean')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-pressure', action='store_true')
    args = parser.parse_args()
    if args.cache_mib < 0:
        parser.error('Cache MiB must be nonnegative')
    if not re.fullmatch(r'tuning8gib-[a-z0-9-]+', args.run_id):
        parser.error('Use a new filename-safe tuning8gib identity')
    if not re.fullmatch(r'luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+', args.adapter):
        parser.error('Use the verified exact adapter LUID instance')
    public = args.results
    raw = args.root / 'tuning8gib/runs' / args.run_id
    if raw.exists() or list(public.glob(args.run_id + '.*')):
        raise FileExistsError('Preserve existing run identities')
    public.mkdir(parents=True, exist_ok=True)
    before = headroom(args.adapter)
    before.update(pressure_qualified_by_explicit_override=args.allow_pressure, pressure_boundary='User-approved existing background RAM; no unrelated processes changed; not actual VRChat coexistence.')
    (public / (args.run_id + '.headroom.json')).write_text(json.dumps(before, indent=2), encoding='utf-8')
    if before['model_runtime_processes'] or (before['resources']['system_available_physical_bytes'] < 16000000000 and not args.allow_pressure):
        raise RuntimeError('Contested or unapproved pressure preflight; no model launched')
    matrix.PROFILES[args.run_id] = ('b11537', ['--n-gpu-layers', '99', '--n-cpu-moe', str(args.n_cpu_moe), '--moe-cache-mib', str(args.cache_mib), '--swa-full', '--cache-ram', '0'])
    record = matrix.run(SimpleNamespace(root=args.root, e4b=None, results=public, repetitions=5, adapter=args.adapter, runtime_names=MODEL_RUNTIMES), args.run_id, load_mode='none', output_dir=raw, environment_overrides={k: None for k in CONTROLS}, safety_policy={'minimum_available_physical_bytes': 268435456, 'minimum_commit_headroom_bytes': 536870912, 'sustained_seconds': 8}, workload=workload)
    record.update(baseline_commit=BASELINE, fixture_origin_product_commit='7a8151be6114a950f062e2430bd36f899d57e4de', experiment_baseline=BASELINE, workload_source=WORKLOAD_SOURCE, n_cpu_moe=args.n_cpu_moe, cache_mib=args.cache_mib, measurement_group=args.group, executable_sha256=hashlib.sha256((args.root / 'runtimes/b11537/llama-server.exe').read_bytes()).hexdigest(), preflight=before)
    record['cohort'] = 'contaminated_runtime_overlap' if record.get('contaminated_by_runtime_conflict') else 'clean_restart_after_owner_reported_overlap'
    record['runtime_conflict_policy'] = {'observed_names': sorted(MODEL_RUNTIMES), 'nominal_interval_seconds': 0.25, 'fields': ['pid', 'name'], 'limit': 'Known native executable names only; short processes between samples and differently named/python-hosted models may be missed. No unrelated command lines captured or processes stopped.'}
    record['probe_command'] = ['<APP_PYTHON>', 'experiments/gemma4_26b_qat/selfpeer_probes.py', *[matrix.sanitize(value, args.root) for value in sys.argv[1:]]]
    evidence = slot_evidence(raw / 'server.log')
    log_text = matrix.sanitize((raw / 'server.log').read_text(encoding='utf-8', errors='replace'), args.root)
    record['cache_execution'] = cache_evidence(log_text.splitlines(), args.cache_mib)
    (public / (args.run_id + '.native.log')).write_text(log_text, encoding='utf-8')
    (public / (args.run_id + '.slots.json')).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    (public / (args.run_id + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    samples = json.loads((raw / 'samples.json').read_text(encoding='utf-8'))
    (public / (args.run_id + '.samples.jsonl')).write_text('\n'.join(json.dumps(s, separators=(',', ':')) for s in samples) + '\n', encoding='utf-8')
    print(json.dumps({'run': args.run_id, 'completed': len(record['requests']), 'status': record['status'], 'exit': record.get('process_exit_code'), 'cache': record['cache_execution'], 'server_simultaneous_rounds': sum(p['phase'] == 'simultaneous_pairs' and p['server_active_overlap_seconds'] > 0 for p in evidence['pairs'])}))
    if record['status'] != 'completed' or len(record['requests']) != 44 or record.get('shutdown') != 'graceful_console_ctrl_c' or record.get('process_exit_code') != 0 or record.get('contaminated_by_runtime_conflict'):
        raise RuntimeError('Retained incomplete run requires diagnosis')


if __name__ == '__main__':
    main()
