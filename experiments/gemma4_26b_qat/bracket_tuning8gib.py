from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from summarize_tuning8gib import CEILING, TARGET, load_runs

SLOT_BYTES = 3345408
MIB = 1048576


def requested_mib(slots):
    return math.ceil((slots + 1) * SLOT_BYTES / MIB)


def observations(public, placement):
    found = {}
    for record in load_runs(public):
        if record['n_cpu_moe'] != placement or record.get('contaminated_by_runtime_conflict'):
            continue
        if record['status'] != 'completed' or len(record['requests']) != 44 or record.get('process_exit_code') != 0 or record.get('safety_cutoff'):
            continue
        groups = record.get('cache_execution', {}).get('groups', [])
        if len(groups) != 1 or groups[0]['layers'] != placement:
            continue
        slots = groups[0]['slots']
        peak = record.get('memory_peaks', {}).get('process_gpu_dedicated_bytes')
        if peak is None:
            continue
        item = found.setdefault(slots, {'peak_bytes': peak, 'runs': [], 'requested_mib': []})
        item['peak_bytes'] = max(item['peak_bytes'], peak)
        item['runs'].append(record['profile'])
        item['requested_mib'].append(record['cache_mib'])
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--stage', choices=('target', 'band'), required=True)
    parser.add_argument('--placements', nargs='+', type=int, choices=range(18, 25), required=True)
    parser.add_argument('--search-id', required=True)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/tuning8gib/clean')
    args = parser.parse_args()
    boundary = TARGET if args.stage == 'target' else CEILING
    ledger_path = args.results / ('search-' + args.search_id + '.json')
    if ledger_path.exists():
        raise FileExistsError('Use a fresh search identity; all prior measurements remain reusable')
    ledger = {'stage': args.stage, 'boundary_bytes': boundary, 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'method': 'Exact source slot arithmetic predicts a starting point only. Each requested cache executes the unchanged44-request full lifecycle. Measured maximum lifecycle PID dedicated peaks decide. Finish only at an eligible slot and its observed next-slot over-boundary neighbor; integer requests sharing an allocation are not distinct effective caches.', 'placements': {}}
    persist = lambda: ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    for placement in args.placements:
        found = observations(args.results, placement)
        curve = next(record for record in load_runs(args.results) if record['n_cpu_moe'] == placement and record['cache_mib'] == 2048 and not record.get('contaminated_by_runtime_conflict'))
        curve_group = curve['cache_execution']['groups'][0]
        base = curve['memory_peaks']['process_gpu_dedicated_bytes'] - (curve_group['slots'] + 1) * SLOT_BYTES
        predicted = max(8, min(math.floor((boundary - base) / SLOT_BYTES) - 1, math.floor(3072 * MIB / SLOT_BYTES) - 1))
        item = {'curve_reference': curve['profile'], 'predicted_noncache_base_bytes': base, 'predicted_start_slots': predicted, 'new_runs': [], 'status': 'searching'}
        ledger['placements'][str(placement)] = item
        persist()
        slots = predicted
        while True:
            if slots not in found:
                cache = requested_mib(slots)
                identity = f'tuning8gib-clean-n{placement}-c{cache}-{args.search_id}-s{slots}'
                command = [sys.executable, str(Path(__file__).resolve().parent / 'selfpeer_probes.py'), '--root', str(args.root), '--n-cpu-moe', str(placement), '--cache-mib', str(cache), '--adapter', args.adapter, '--group', args.stage, '--run-id', identity, '--results', str(args.results), '--allow-pressure']
                item['new_runs'].append({'run': identity, 'slots': slots, 'requested_mib': cache})
                persist()
                completed = subprocess.run(command)
                if completed.returncode:
                    item['status'] = 'measurement_failed_or_contaminated_requires_diagnosis'
                    ledger['status'] = item['status']
                    persist()
                    raise RuntimeError(f'Retained failed bracket measurement: {identity}')
                found = observations(args.results, placement)
                if slots not in found:
                    raise RuntimeError('Requested slot allocation missing from actual native evidence')
            eligible = [count for count, row in found.items() if row['peak_bytes'] <= boundary]
            best = max(eligible, default=None)
            if best is not None and best + 1 in found and found[best + 1]['peak_bytes'] > boundary:
                item.update(status='measured_neighbor_bracket', eligible_slots=best, eligible_requested_mib_range=[requested_mib(best), requested_mib(best + 1) - 1], eligible=found[best], next_slots=best + 1, next=found[best + 1], margin_bytes=boundary - found[best]['peak_bytes'], slot_increment_bytes=SLOT_BYTES)
                persist()
                print(json.dumps({'placement': placement, 'stage': args.stage, 'slots': best, 'request_mib': found[best]['requested_mib'], 'peak_bytes': found[best]['peak_bytes'], 'next_peak_bytes': found[best + 1]['peak_bytes']}), flush=True)
                break
            over = [count for count, row in found.items() if row['peak_bytes'] > boundary and (best is None or count > best)]
            ceiling_slot = min(over, default=None)
            if best is not None:
                margin_slots = math.floor((boundary - found[best]['peak_bytes']) / SLOT_BYTES)
                slots = best + max(1, margin_slots)
                if ceiling_slot is not None:
                    slots = min(slots, ceiling_slot - 1)
            else:
                excess_slots = math.ceil((found[slots]['peak_bytes'] - boundary) / SLOT_BYTES)
                slots -= max(1, excess_slots)
            slots = max(8, slots)
            if slots in found:
                if slots == 8 and found[slots]['peak_bytes'] > boundary:
                    item.update(status='no_enabled_target_cache', minimum_enabled_slots=8, minimum_enabled=found[8])
                    persist()
                    break
                raise RuntimeError('Nonmonotonic measured bracket requires explicit analysis; no contender silently excluded')
            if requested_mib(slots) > 3072:
                raise RuntimeError('Known excluded Vulkan bank limit reached; do not reproduce it')
    ledger['status'] = 'completed'
    ledger['finished_utc'] = datetime.now(timezone.utc).isoformat()
    persist()


if __name__ == '__main__':
    main()
