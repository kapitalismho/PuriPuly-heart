from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import psutil
import run_matrix as matrix
from summarize import distribution
from summarize_timings import PHASES, native_records, stats
from windows_metrics import Sampler

# Only controls whose semantics were read in the pinned Vulkan source/research.
# None removes a variable from this owned child; '=0' does not disable presence flags.
CONTROLS = (
    'GGML_VK_FORCE_MAX_BUFFER_SIZE', 'GGML_VK_FORCE_MAX_ALLOCATION_SIZE',
    'GGML_VK_SUBALLOCATION_BLOCK_SIZE', 'GGML_VK_PREFER_HOST_MEMORY',
    'GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM', 'GGML_VK_ALLOW_SYSMEM_FALLBACK',
    'GGML_VK_DISABLE_ASYNC', 'GGML_VK_ASYNC_USE_TRANSFER_QUEUE',
    'GGML_VK_ALLOW_GRAPHICS_QUEUE', 'GGML_OP_OFFLOAD_MIN_BATCH',
    'GGML_VK_PERF_LOGGER', 'GGML_VK_PERF_LOGGER_CONCURRENT',
    'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS',
    'GGML_VK_DISABLE_FUSION', 'GGML_VK_DISABLE_GRAPH_OPTIMIZE',
    'GGML_VK_DISABLE_DESCRIPTOR_REUSE', 'GGML_VK_SERIALIZE_SUBMISSIONS',
    'GGML_VK_DISABLE_MMVQ', 'GGML_VK_FORCE_MMVQ',
    'GGML_VK_DISABLE_COOPMAT', 'GGML_VK_DISABLE_COOPMAT2',
    'GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR', 'GGML_VK_DISABLE_F16',
)
PLACEMENTS = {
    'allhost3072': 'new-cpu-moe-cache3072-swa-nocram',
    'n24cache2048': 'new-ncpu24-cache2048-swa-nocram',
    'n18cache3072': 'new-ncpu18-cache3072-swa-nocram',
    'n24cache3072': 'new-ncpu24-cache3072-swa-nocram',
}
MODEL_RUNTIMES = {'llama-server.exe', 'llama-cli.exe', 'llama-bench.exe', 'strata.exe', 'strata-poly.exe'}


def headroom(adapter="luid_0x00000000_0x0000f60b_phys_0"):
    sampler = Sampler(os.getpid(), adapter=adapter)
    try:
        sampler.sample()
        time.sleep(0.3)
        sample = sampler.sample()
        return {
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'model_runtime_processes': [
                {'pid': p.info['pid'], 'name': p.info['name']}
                for p in psutil.process_iter(['pid', 'name'])
                if (p.info.get('name') or '').lower() in MODEL_RUNTIMES
            ],
            'resources': {k: v for k, v in sample.items() if k.startswith(('device_gpu_', 'system_'))},
            'counter_errors': sampler.errors,
        }
    finally:
        sampler.close()


def project(record, samples, log_path):
    native = native_records(log_path)
    aligned = len(native) == len(record['requests']) and bool(native)
    phases, offset = {}, 0
    for phase in PHASES:
        requests = [r for r in record['requests'] if r['phase'] == phase]
        rows = []
        for request in requests:
            t = request.get('timings') or {}
            rows.append({
                'prefill_ms': t.get('prompt_ms'), 'prefill_tps': t.get('prompt_per_second'),
                'decode_ms': t.get('predicted_ms'), 'decode_tps': t.get('predicted_per_second'),
                'prompt_evaluated_tokens': t.get('prompt_n'), 'prompt_cached_tokens': t.get('cache_n'),
                'prompt_total_tokens': request.get('tokens_evaluated'),
                'predicted_output_tokens': t.get('predicted_n'),
            })
        group = stats(rows)
        group['native_ordered_phase_records'] = native[offset:offset+len(requests)] if aligned else []
        group['decode_eval_tokens'] = distribution([r['decode_eval_tokens'] for r in group['native_ordered_phase_records']])
        group['wall_seconds'] = distribution([r['wall_seconds'] for r in requests])
        group['streaming_TTFT_including_apply_template_seconds'] = distribution([r['ttft_seconds'] for r in requests if r.get('ttft_seconds') is not None])
        phases[phase] = group
        offset += len(requests)
    host_log = record.get('host_allocation_log', [])
    return {
        'run_id': record['profile'], 'status': record['status'], 'load_mode': record['load_mode'],
        'completed_requests': len(record['requests']), 'phases': phases,
        'native_log_count': len(native), 'native_log_phase_alignment': aligned,
        'association_limit': 'API timings belong to original requests; concurrent native logs are phase distributions, not per-request joins. Printed decode count is not a traced pure forward-pass count; native TPS is retained.',
        'resource_distributions': {key: distribution([s[key] for s in samples if s.get(key) is not None]) for key in sorted({k for s in samples for k in s}) if key not in {'phase'}},
        'native_host_model_buffers': [line for line in record.get('placement_cache_log', []) if 'model buffer size' in line],
        'Vulkan_Host_model_buffer_logged': any('Vulkan_Host' in line and 'model buffer size' in line for line in host_log),
        'CPU_Mapped_model_buffer_logged': any('CPU_Mapped' in line and 'model buffer size' in line for line in host_log),
        'pinned_allocation_failure_warnings': [line for line in host_log if 'Failed to allocate pinned' in line],
        'per_miss_async_route_measured': False,
        'path_evidence_limit': 'Actual buffer names and warnings are observed; recognized-pinned transfer behavior is source-backed conditional reasoning, not a per-upload trace. Router-ID readback/synchronization remains even on cache hits/read mode.',
        'lifetime_cache_counters': [line for line in record.get('placement_cache_log', []) if 'llama_moe_cache:' in line],
        'counter_limit': 'Includes native warmup and the full43-request lifecycle, not per-phase transfer bytes. Logical uploaded bytes are not measured physical PCIe bytes or bandwidth.',
        'quality_automated_nonempty_no_reasoning_not_truncated': all(all(r['quality_checks'][k] for k in ('nonempty', 'no_reasoning', 'not_truncated')) for r in record['requests']) if record['requests'] else None,
        'quality_limit': 'Automated digits are not a semantic gate; complete unique outputs require the documented bounded semantic review.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--placement', choices=PLACEMENTS, required=True)
    parser.add_argument('--load-mode', choices=('mmap', 'none'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--minimum-free-bytes', type=int, default=16000000000)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--experiment-baseline', default='eff651d517c7fd15ef2f96a019b348cccf662d06')
    parser.add_argument('--allow-pressure', action='store_true', help='Requires explicit Director acceptance; records pressure qualification, never an efficient-offload claim.')
    args = parser.parse_args()
    if not re.fullmatch(r'followup-[a-z0-9-]+', args.run_id):
        parser.error('Use a new followup-prefixed filename-safe run identity')
    public = args.results or matrix.REPO / 'experiments/gemma4_26b_qat/results/followup/firstwave'
    public.mkdir(parents=True, exist_ok=True)
    raw = args.root / 'followup/runs' / args.run_id
    paths = [public / f'{args.run_id}{suffix}' for suffix in ('.headroom.json', '.json', '.analysis.json', '.samples.jsonl')]
    if raw.exists() or any(path.exists() for path in paths):
        raise FileExistsError('Preserve every existing followup record; choose a new run identity')
    before = headroom()
    before['minimum_free_bytes_for_unqualified_run'] = args.minimum_free_bytes
    before['minimum_is_safety_policy_not_exact_model_requirement'] = True
    before['pressure_qualified_by_explicit_override'] = args.allow_pressure
    before['pressure_boundary_authorization'] = 'User requested existing constrained RAM as pragmatic concurrent-load condition; not actual VRChat co-rendering or clean transfer-only causality.'
    paths[0].write_text(json.dumps(before, indent=2), encoding='utf-8')
    available = before['resources']['system_available_physical_bytes']
    if before['model_runtime_processes'] or (available < args.minimum_free_bytes and not args.allow_pressure):
        raise RuntimeError('Preflight unsafe/contested; no model launched. Preserve headroom record and obtain Director decision.')
    profile = PLACEMENTS[args.placement]
    matrix.PROFILES[args.run_id] = matrix.PROFILES[profile]
    namespace = SimpleNamespace(root=args.root, e4b=None, results=public, repetitions=5)
    record = matrix.run(namespace, args.run_id, load_mode=args.load_mode, output_dir=raw, environment_overrides={k: None for k in CONTROLS}, safety_policy={'minimum_available_physical_bytes': 268435456, 'minimum_commit_headroom_bytes': 536870912, 'sustained_seconds': 8})
    samples = json.loads((raw / 'samples.json').read_text(encoding='utf-8'))
    analysis = project(record, samples, raw / 'server.log')
    analysis.update(
        matched_original_profile=profile,
        experiment_baseline=args.experiment_baseline,
        executable_sha256=hashlib.sha256((args.root / 'runtimes/b11537/llama-server.exe').read_bytes()).hexdigest(),
        mode_boundary='none is the supported b11537 read/non-mmap mode, not a pinning guarantee.',
        preflight_headroom=before,
    )
    paths[2].write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
    paths[3].write_text('\n'.join(json.dumps(s, separators=(',', ':')) for s in samples)+'\n', encoding='utf-8')
    print(json.dumps({'run_id':args.run_id,'status':record['status'],'completed':len(record['requests']),'shutdown':record.get('shutdown'),'exit_code':record.get('process_exit_code'),'host_buffers':analysis['native_host_model_buffers'],'pinning_failures':analysis['pinned_allocation_failure_warnings'],'min_free_bytes':analysis['resource_distributions'].get('system_available_physical_bytes'),'phases':{p:{k:analysis['phases'][p][k]['median'] for k in ('prefill_ms','prefill_tps','decode_ms','decode_tps')} for p in PHASES}}), flush=True)
    if record['status'] != 'completed' or len(record['requests']) != 43 or record.get('shutdown') != 'graceful_console_ctrl_c' or record.get('process_exit_code') != 0:
        raise RuntimeError('Followup run incomplete or release not graceful; retained evidence requires diagnosis before continuing')


if __name__ == '__main__':
    main()
