from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

import run_matrix as matrix
from followup_probes import CONTROLS, headroom

BASELINE = 'f052922ae80e26b7365cc66f63f8081e4941fd87'
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache-mib', type=int, choices=(2048, 3072), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-pressure', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'selfpeer-[a-z0-9-]+', args.run_id):
        parser.error('Use a new filename-safe selfpeer identity')
    public = matrix.REPO / 'experiments/gemma4_26b_qat/results/selfpeer'
    raw = args.root / 'selfpeer/runs' / args.run_id
    if raw.exists() or list(public.glob(args.run_id + '.*')):
        raise FileExistsError('Preserve existing run identities')
    public.mkdir(parents=True, exist_ok=True)
    before = headroom()
    before.update(pressure_qualified_by_explicit_override=args.allow_pressure, pressure_boundary='User-approved existing background RAM; no unrelated processes changed; not actual VRChat coexistence.')
    (public / (args.run_id + '.headroom.json')).write_text(json.dumps(before, indent=2), encoding='utf-8')
    if before['model_runtime_processes'] or (before['resources']['system_available_physical_bytes'] < 16000000000 and not args.allow_pressure):
        raise RuntimeError('Contested or unapproved pressure preflight; no model launched')
    matrix.PROFILES[args.run_id] = matrix.PROFILES[f'new-ncpu24-cache{args.cache_mib}-swa-nocram']
    record = matrix.run(SimpleNamespace(root=args.root, e4b=None, results=public, repetitions=5), args.run_id, load_mode='none', output_dir=raw, environment_overrides={k: None for k in CONTROLS}, safety_policy={'minimum_available_physical_bytes': 268435456, 'minimum_commit_headroom_bytes': 536870912, 'sustained_seconds': 8}, workload=workload)
    record.update(experiment_baseline=BASELINE, cache_mib=args.cache_mib, executable_sha256=hashlib.sha256((args.root / 'runtimes/b11537/llama-server.exe').read_bytes()).hexdigest(), preflight=before)
    evidence = slot_evidence(raw / 'server.log')
    (public / (args.run_id + '.slots.json')).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    (public / (args.run_id + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    samples = json.loads((raw / 'samples.json').read_text(encoding='utf-8'))
    (public / (args.run_id + '.samples.jsonl')).write_text('\n'.join(json.dumps(s, separators=(',', ':')) for s in samples) + '\n', encoding='utf-8')
    print(json.dumps({'run': args.run_id, 'completed': len(record['requests']), 'status': record['status'], 'exit': record.get('process_exit_code'), 'server_simultaneous_rounds': sum(p['phase'] == 'simultaneous_pairs' and p['server_active_overlap_seconds'] > 0 for p in evidence['pairs'])}))
    if record['status'] != 'completed' or len(record['requests']) != 44 or record.get('shutdown') != 'graceful_console_ctrl_c' or record.get('process_exit_code') != 0:
        raise RuntimeError('Retained incomplete run requires diagnosis')


if __name__ == '__main__':
    main()
