from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfpeer_probes import PHASES

FIELDS = ('process_gpu_dedicated_bytes', 'process_gpu_shared_bytes', 'process_gpu_committed_bytes', 'device_gpu_dedicated_bytes', 'WorkingSetSize', 'PrivateUsage')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/mtp')
    args = parser.parse_args()
    summary = json.loads((args.results / 'summary.json').read_text(encoding='utf-8'))
    runs = []
    for run in summary['runs']:
        native = (args.results / (run['profile'] + '.native.log')).read_text(encoding='utf-8')
        allocation_lines = [{'line': index, 'text': line} for index, line in enumerate(native.splitlines(), 1) if any(word in line for word in ('model buffer size', 'compute buffer size', 'output buffer size', 'KV buffer size', 'MoE cache size', 'sharing with layer', 'loading draft model', 'memory breakdown'))]
        item = {'run': run['profile'], 'configuration': run['configuration'], 'server_pid': next((int(label.split('_')[1]) for label in run['counter_instances'].get('process_gpu_dedicated_bytes', [])), None), 'adapter': run['gpu_adapter_luid'], 'counter_errors': run['counter_errors'], 'counter_instances': {field: run['counter_instances'].get(field) for field in FIELDS if field.startswith('process_gpu')}, 'sample_count': run['sample_count'], 'sample_cadence_seconds': run['sample_cadence_seconds'], 'lifecycle': {field: run['lifecycle_resources'].get(field) for field in FIELDS}, 'phases': {phase: {field: run['phase_resources'][phase].get(field) for field in FIELDS} for phase in ('load', *PHASES, 'shutdown')}, 'native_allocation_and_alias_lines': allocation_lines}
        runs.append(item)
    comparisons = {}
    keys = sorted({run['configuration'] for run in summary['runs']})
    for key in keys:
        configuration = summary['configurations'][key]
        if not configuration['draft_max']:
            continue
        control_key = f"c{configuration['cache_mib']}-off"
        selected = [run for run in runs if run['configuration'] == key]
        controls = [run for run in runs if run['configuration'] == control_key]
        if not selected or not controls:
            continue
        phases = {}
        for phase in ('lifecycle', 'load', *PHASES, 'shutdown'):
            view = lambda run: run['lifecycle'] if phase == 'lifecycle' else run['phases'][phase]
            fields = {}
            for field in FIELDS:
                current = [view(run)[field]['max'] for run in selected if view(run).get(field) and view(run)[field]['n']]
                previous = [view(run)[field]['max'] for run in controls if view(run).get(field) and view(run)[field]['n']]
                if not current or not previous:
                    fields[field] = {'available': False}
                    continue
                fields[field] = {'mtp_run_peaks_bytes': current, 'off_run_peaks_bytes': previous, 'mtp_max_minus_off_max_bytes': max(current)-max(previous), 'delta_MiB': (max(current)-max(previous))/2**20, 'comparison': 'Independent same-phase maxima across all runs, not a time-coincident sum or per-allocation residency attribution.'}
            phases[phase] = fields
        comparisons[key] = {'control': control_key, 'phases': phases}
    result = {'scope': 'Owned same-cache MTP/off lifecycle and phase accounting audit; all original samples/logs remain preserved.', 'runs': runs, 'comparisons': comparisons, 'observed_native_allocations': 'Q4_0 draft Vulkan0 model225.21MiB, Vulkan_Host model144.00MiB, draft Vulkan0 compute62.12MiB; target model/cache/KV/compute controls unchanged. Native layer-sharing lines prove aliases of targetKV; logical18+108MiB KV size reports are not extra allocations.', 'source_interpretation': 'Pinned Vulkan allocator tries DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT then DEVICE_LOCAL. HOST_VISIBLE does not mean systemRAM. Drafter embedding may be duplicated for CPU input/GPU output; sharedKV avoids a second targetKV allocation but does not eliminate real drafter model/compute/output costs. See runtime-research.json citations.', 'conclusion': 'The file-size-as-added-PID-dedicated proxy is contradicted by measured counter deltas. Exact PID/adapter counters and phase observations distinguish that result from a selected steady snapshot or simple baseline-lifecycle/active-phase mismatch. Dedicated, shared, committed, working set and logical Vulkan buffer sizes do not reconcile per allocation from these observations. This is not proof of zero memory cost, CPU fallback, physical residency, duplicate systemRAM, or uncontrolled WDDM spill.', 'limits': '250ms nominal sampled lower bounds; independent maxima can occur at different times. No driver allocation tracing or physical8GiB/card-coexistence test. Never add the overlapping accounting views.'}
    (args.results / 'memory-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'runs': len(runs), 'comparisons': {key: {phase: {field: row.get('delta_MiB') for field, row in fields.items()} for phase, fields in comparison['phases'].items() if phase in ('lifecycle', 'simultaneous_pairs')} for key, comparison in comparisons.items()}}))


if __name__ == '__main__':
    main()
