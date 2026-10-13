from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def normalized(text):
    return re.sub(r'[^\w\s]', '', text.lower(), flags=re.UNICODE).split()


def analyze(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    result = {'trace_path': path.as_posix(), 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': report['status'], 'candidate': report['candidate'], 'route': report['route'], 'gemma_effective_backend': report.get('gemma', {}).get('readiness', {}).get('effective_backend'), 'cases': []}
    for case in report['cases']:
        started = case['playback_started_at']
        sources = case['sources']
        text = ' '.join(s['source']['raw_text'] for s in sources)
        mapping = case['assembly_scoring_only']
        reference = ' '.join(report['inputs'][piece['input']]['source_text'] for piece in mapping if 'input' in piece)
        speech_ends = [e['at'] for e in case['trace'] if e['event'] == 'vad' and e['value']['event'].get('reason') is not None]
        end = min(speech_ends) if speech_ends else None
        last_real = max((piece['scenario_sample_range'][1] for piece in mapping if 'scenario_sample_range' in piece), default=case['input_samples'] - 20000)
        scored = []
        for item in sources:
            source = item['source']
            identity = source['source_id']
            events = [e for e in case['trace'] if e.get('source_id') == identity]
            times = {e['event']: e['at'] for e in events}
            submission = next((e['submission'] for e in case['trace'] if e['event'] == 'ordered_publication_wait' and e['submission']['source_child']['source_id'] == identity), None)
            logical_turn = None if submission is None else 'peer:' + submission['child_utterance_id']
            snapshots = [s for s in case.get('renderer_snapshots', []) if any(b.get('logical_turn_key') == logical_turn for b in s['snapshot']['blocks'])] if logical_turn else []
            rendered = next((s.get('software_commit_observed_at') for s in snapshots if s.get('software_commit_observed_at')), None)
            matching_readbacks = [r for r in case.get('renderer_readbacks', []) if any(b.get('logical_turn_key') == logical_turn for b in r['requested_snapshot']['blocks'])]
            translations = [e['result']['output']['translation']['translated_text'] for e in events if e['event'] == 'translation_end' and e['result'].get('output') and e['result']['output'].get('translation')]
            details = {'logical_turn_key': logical_turn, 'translations': translations, 'readbacks': matching_readbacks, 'boundary_label': 'ACK is real Flet dispatch completion, not verified pixel onset; readback upper bounds need image observation'}
            scored.append({'source_id': identity, 'text': source['raw_text'], 'speaker': source['speaker'], 'order': source['order'], 'source_audio_ranges': [r for o in source['origins'] for r in o['source_ranges']], 'context_audio_ranges': [r for o in source['origins'] for r in o['context_ranges']], 'display_group_id': source['display_group_id'], 'admitted_after_playback_s': item['admitted_at'] - started, 'admission_lead_to_first_enclosing_end_s': None if end is None else end - item['admitted_at'], 'translation_start_after_playback_s': times.get('translation_start', started) - started, 'translation_elapsed_s': times.get('translation_end', started) - times.get('translation_start', started), 'renderer_dispatch_ack_after_playback_s': None if rendered is None else rendered - started, 'renderer_dispatch_ack_lead_to_first_enclosing_end_s': None if rendered is None or end is None else end - rendered, 'renderer_dispatch_ack_minus_last_reference_audio_s': None if rendered is None else rendered - started - last_real / 16000, 'prior_context': item['prior_context_texts'], 'effective_text_policy': 'raw text unchanged; trimming OFF'})
            scored[-1].update(details)
        known = [s for s in sources if s['source']['speaker'] and s['source']['speaker']['speaker_id'] is not None]
        states = Counter((s['source']['speaker'] or {}).get('attribution_state', 'unavailable') for s in sources)
        submits = [e for e in case['trace'] if e['event'] == 'native_submit']
        native = [e for e in case['trace'] if e['event'] == 'native_result']
        invalid = [payload for e in submits for payload in e['payloads'] if payload['matches_declared_source_pcm'] is not True]
        process_peaks = {}
        for sample in case['resources']:
            for proc in sample['processes']:
                old = process_peaks.setdefault(proc['name'], {'rss': 0, 'private': 0, 'pids': []})
                old['rss'] = max(old['rss'], proc['rss'])
                old['private'] = max(old['private'], proc['private'] or 0)
                if proc['pid'] not in old['pids']:
                    old['pids'].append(proc['pid'])
        buffers = {key: max((sample['speaker'][key] for sample in case['resources'] if sample['speaker']), default=0) for key in ('pending_samples', 'pending_items', 'retained_map_pieces', 'evidence_items', 'evidence_evictions', 'overloads', 'copied_samples', 'max_copy_samples')}
        actual = normalized(text)
        expected = normalized(reference)
        cer_ref = ''.join(expected)
        cer_actual = ''.join(actual)
        result['cases'].append({'case': case['case'], 'arm': 'baseline' if case['mode'] == 'baseline' else case['profile'], 'repetition': case['repetition'], 'status': case['status'], 'error': case.get('error'), 'errors': case['errors'], 'reference': reference, 'recognized': text, 'WER': None if not expected else distance(expected, actual) / len(expected), 'CER': None if not cer_ref else distance(cer_ref, cer_actual) / len(cer_ref), 'metric_limits': 'Case-level normalized edit distance; crop references retain original public sentence; synthetic overlap has no single-turn WER reference. Not diarization accuracy or pass threshold.', 'useful_color_source_char_coverage': sum(len(s['source']['raw_text']) for s in known) / max(1, sum(len(s['source']['raw_text']) for s in sources)), 'attribution_states': dict(states), 'sources': scored, 'ASR_calls': len(submits), 'ASR_seal_reasons': dict(Counter(e['seal_reason'] for e in submits)), 'native_results': len(native), 'translation_calls': sum(e['event'] == 'translation_start' for e in case['trace']), 'translation_terminals': [e['outcome'] for e in case['trace'] if e['event'] == 'child_terminal'], 'capture_received_once': case['capture_received_once'], 'payload_mismatches': len(invalid), 'frame_delivery_lag_peak_s': max(case['frame_delivery_lags'], default=0), 'frame_delivery_lag_p95_s': sorted(case['frame_delivery_lags'])[int(0.95 * (len(case['frame_delivery_lags']) - 1))] if case['frame_delivery_lags'] else None, 'process_memory_peaks_by_name': process_peaks, 'speaker_observed_buffer_peaks': buffers, 'speaker_final': case['speaker_final'], 'actual_software_commits': sum(e['record']['event_type'] == 'render_commit_acknowledgement' for e in case['renderer_events']), 'snapshot_capture': case['snapshot_capture']})
    counters = path.parent / 'gpu_counters.jsonl'
    if counters.exists():
        samples = [json.loads(line) for line in counters.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        peak = {}
        invalid = 0
        for sample in samples:
            for counter in sample['counters']:
                if counter['status'] != 0:
                    invalid += 1
                    continue
                key = counter['path']
                peak[key] = max(peak.get(key, 0), counter['value'])
        result['gpu_counters'] = {'samples': len(samples), 'invalid_counter_values': invalid, 'peak_by_instance': peak, 'interpretation': 'Windows1s counters; adapter includes background; PID instances correlate to process sample PIDs. Occupancy/utilization, not per-inference GPU timestamp.'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = {'schema': 'issue218-trace-summary-1', 'command': __import__('sys').argv, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': [analyze(path) for path in args.trace]}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
