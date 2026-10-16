from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import run_matrix as matrix
from followup_probes import CONTROLS, MODEL_RUNTIMES, headroom
from mtp_acquire import DRAFTER_NAME, DRAFTER_BYTES, DRAFTER_SHA256, SERVER_SHA256, identity
from selfpeer_probes import cache_evidence, slot_evidence, workload

BASELINE = 'c451f1b04a940f8d3e297ca58d13eefc338baee8'


def speculative_log_evidence(text):
    return [{'line': number, 'text': line} for number, line in enumerate(text.splitlines(), 1) if any(word in line.lower() for word in ('specul', 'draft', 'mtp', 'assistant', 'accepted'))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--cache-mib', type=int, required=True)
    parser.add_argument('--draft-max', type=int, choices=(0, 2, 4), required=True)
    parser.add_argument('--group', choices=('control', 'unchanged', 'reduced', 'bracket', 'repeat', 'max2'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--results', type=Path, default=matrix.REPO / 'experiments/gemma4_26b_qat/results/mtp')
    parser.add_argument('--allow-pressure', action='store_true')
    args = parser.parse_args()
    if not 29 <= args.cache_mib <= 3072:
        parser.error('Use an enabled cache below the established Vulkan bank cap')
    if not re.fullmatch(r'mtp-[a-z0-9-]+', args.run_id):
        parser.error('Use a fresh filename-safe mtp identity')
    if not re.fullmatch(r'luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+', args.adapter):
        parser.error('Use the verified exact adapter instance')
    raw = args.root / 'mtp/runs' / args.run_id
    if raw.exists() or list(args.results.glob(args.run_id + '.*')):
        raise FileExistsError('Preserve all previous run identities')
    args.results.mkdir(parents=True, exist_ok=True)
    before = headroom(args.adapter)
    before.update(pressure_qualified_by_explicit_override=args.allow_pressure, pressure_boundary='Existing constrained RAM accepted by owner; no unrelated process stops; not actual coexistence.')
    (args.results / (args.run_id + '.headroom.json')).write_text(json.dumps(before, indent=2), encoding='utf-8')
    if before['model_runtime_processes'] or (before['resources']['system_available_physical_bytes'] < 16000000000 and not args.allow_pressure):
        raise RuntimeError('Contested or unapproved preflight; no model launch')
    executable = args.root / 'runtimes/b11537/llama-server.exe'
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    if executable_sha256 != SERVER_SHA256:
        raise RuntimeError('Pinned server executable changed; no launch')
    draft_path = args.root / 'models' / DRAFTER_NAME
    draft_identity = identity(draft_path)
    if (draft_identity['bytes'], draft_identity['sha256']) != (DRAFTER_BYTES, DRAFTER_SHA256):
        raise RuntimeError('Pinned Q4_0 drafter changed; no launch')
    placement = ['--n-gpu-layers', '99', '--n-cpu-moe', '24', '--moe-cache-mib', str(args.cache_mib), '--swa-full', '--cache-ram', '0']
    if args.draft_max:
        placement.extend(['--model-draft', str(draft_path), '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(args.draft_max)])
    matrix.PROFILES[args.run_id] = ('b11537', placement)
    matrix.request_translation = partial(matrix.request_translation, capture_completion_metadata=True)
    record = matrix.run(SimpleNamespace(root=args.root, e4b=None, results=args.results, repetitions=5, adapter=args.adapter, runtime_names=MODEL_RUNTIMES), args.run_id, load_mode='none', output_dir=raw, environment_overrides={key: None for key in CONTROLS}, safety_policy={'minimum_available_physical_bytes': 268435456, 'minimum_commit_headroom_bytes': 536870912, 'sustained_seconds': 8}, workload=workload)
    record.update(baseline_commit=BASELINE, experiment_baseline=BASELINE, cohort='owner_authorized_mtp_followup', historical_evidence='c451f1b0 evidence is separate, unchanged, not a matched-session control', n_cpu_moe=24, cache_mib=args.cache_mib, draft_max=args.draft_max, speculative_type='draft-mtp' if args.draft_max else 'none', measurement_group=args.group, executable_sha256=executable_sha256, drafter_identity=draft_identity if args.draft_max else None, preflight=before)
    record['probe_command'] = ['<APP_PYTHON>', 'experiments/gemma4_26b_qat/mtp_probes.py', *[matrix.sanitize(value, args.root) for value in sys.argv[1:]]]
    record['runtime_conflict_policy'] = {'observed_names': sorted(MODEL_RUNTIMES), 'nominal_interval_seconds': 0.25, 'fields': ['pid', 'name'], 'limit': 'Known native executable names only; differently named/Python-hosted or entirely between-sample models may be missed. No unrelated command lines captured or processes stopped.'}
    if record.get('contaminated_by_runtime_conflict'):
        record['cohort'] = 'contaminated_runtime_overlap'
    text = matrix.sanitize((raw / 'server.log').read_text(encoding='utf-8', errors='replace'), args.root)
    slots = slot_evidence(raw / 'server.log')
    record['cache_execution'] = cache_evidence(text.splitlines(), args.cache_mib)
    record['speculative_log_evidence'] = speculative_log_evidence(text)
    record = json.loads(matrix.sanitize(json.dumps(record, ensure_ascii=False), args.root))
    (args.results / (args.run_id + '.native.log')).write_text(text, encoding='utf-8')
    (args.results / (args.run_id + '.slots.json')).write_text(json.dumps(slots, indent=2), encoding='utf-8')
    (args.results / (args.run_id + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    samples = json.loads((raw / 'samples.json').read_text(encoding='utf-8'))
    (args.results / (args.run_id + '.samples.jsonl')).write_text('\n'.join(json.dumps(sample, separators=(',', ':')) for sample in samples) + '\n', encoding='utf-8')
    print(json.dumps({'run': args.run_id, 'status': record['status'], 'completed': len(record['requests']), 'exit': record.get('process_exit_code'), 'shutdown': record.get('shutdown'), 'cache_mib': args.cache_mib, 'draft_max': args.draft_max, 'dedicated_peak': record.get('memory_peaks', {}).get('process_gpu_dedicated_bytes'), 'native_overlaps': sum(pair['phase'] == 'simultaneous_pairs' and pair['server_active_overlap_seconds'] > 0 for pair in slots['pairs'])}))
    if record['status'] != 'completed' or len(record['requests']) != 44 or record.get('shutdown') != 'graceful_console_ctrl_c' or record.get('process_exit_code') != 0 or record.get('contaminated_by_runtime_conflict'):
        raise RuntimeError('Preserved incomplete/contaminated MTP lifecycle requires diagnosis')


if __name__ == '__main__':
    main()
