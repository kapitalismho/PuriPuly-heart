from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import zipfile


def tokens(text):
    return re.findall(r'\w+', text.lower(), re.UNICODE)


def required_sequence(expected, actual):
    lengths = [[0] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    for i in range(len(expected) - 1, -1, -1):
        for j in range(len(actual) - 1, -1, -1):
            lengths[i][j] = 1 + lengths[i + 1][j + 1] if expected[i] == actual[j] else max(lengths[i + 1][j], lengths[i][j + 1])
    missing = []
    i = j = 0
    while i < len(expected):
        if j < len(actual) and expected[i] == actual[j]:
            i += 1
            j += 1
        elif j < len(actual) and lengths[i][j + 1] > lengths[i + 1][j]:
            j += 1
        else:
            missing.append(expected[i])
            i += 1
    return not missing, missing


def source_details(item, case):
    source = item['source']
    events = [e for e in case['trace'] if e.get('source_id') == source['source_id']]
    submission = next((e['submission'] for e in case['trace'] if e['event'] == 'ordered_publication_wait' and e['submission']['source_child']['source_id'] == source['source_id']), None)
    key = 'peer:' + submission['child_utterance_id'] if submission else None
    snapshots = [s for s in case['renderer_snapshots'] if any(b.get('logical_turn_key') == key for b in s['snapshot']['blocks'])] if key else []
    readbacks = [r for r in case['renderer_readbacks'] if any(b.get('logical_turn_key') == key for b in r['requested_snapshot']['blocks'])] if key else []
    times = {e['event']: e['at'] for e in events}
    translations = [e['result']['output']['translation']['translated_text'] for e in events if e['event'] == 'translation_end' and e['result'].get('output') and e['result']['output'].get('translation')]
    seals = [e['at'] for e in case['trace'] if e['event'] == 'vad' and e['value']['event'].get('reason')]
    first_seal = min(seals) if seals else None
    ack = min((s['software_commit_observed_at'] for s in snapshots), default=None)
    stable = [r for r in readbacks if r.get('readback_valid', True) and r.get('capture', {}).get('sha256') and sum(r['capture'].get('exact_color_pixel_counts', {}).values()) > 0 and r['revision'] == r['current_revision_at_readback_start'] == r['current_revision_at_readback_end']]
    readback = min((r['readback_completed_at'] for r in stable), default=None)
    return {'source_id': source['source_id'], 'text': source['raw_text'], 'speaker': source['speaker'], 'order': source['order'], 'origins': source['origins'], 'display_group_id': source['display_group_id'], 'admitted_at': item['admitted_at'], 'prior_context_texts': item['prior_context_texts'], 'prepared': item['prepared'], 'history_after': item['history_after'], 'translations': translations, 'logical_turn_key': key, 'translation_started_at': times.get('translation_start'), 'translation_completed_at': times.get('translation_end'), 'ordered_publication_completed_at': times.get('ordered_publication_complete'), 'actual_render_styles': [b.get('speaker_style') for s in snapshots for b in s['snapshot']['blocks'] if b.get('logical_turn_key') == key], 'renderer_snapshots': snapshots, 'readbacks': readbacks, 'dispatch_ack_at': ack, 'stable_software_readback_upper_bound_at': readback, 'admission_lead_to_first_enclosing_seal_s': None if first_seal is None else first_seal - item['admitted_at'], 'dispatch_ack_lead_to_first_enclosing_seal_s': None if first_seal is None or ack is None else first_seal - ack, 'stable_readback_lead_to_first_enclosing_seal_s': None if first_seal is None or readback is None else first_seal - readback, 'measurement_limit': 'ACK is dispatch completion; owned PNG proves software pixels only after image observation. Readback completion is an upper bound, not onset or photons.'}

def conservation(case):
    contributions = {}
    for event in case['trace']:
        if event['event'] != 'provider_result':
            continue
        value = event['value']
        identity = value.get('identity', {}).get('provider_turn_id')
        if identity is None:
            continue
        if value.get('outcome') in ('final', 'degraded') and value.get('text') and not value.get('included_contributions'):
            end = len(value['text'])
            contributions[f'{identity}|terminal:0:{end}'] = {'start': 0, 'end': end, 'text': value['text']}
        for part in value.get('included_contributions', ()):
            contributions[identity + '|' + part['contribution_id']] = {'start': part['text_start'], 'end': part['text_end'], 'text': value['text'][part['text_start']:part['text_end']]}
        part = value.get('contribution')
        if part and value.get('assembly') == 'replace':
            contributions[identity + '|' + part['contribution_id']] = {'start': part['text_start'], 'end': part['text_end'], 'text': value['text'][part['text_start']:part['text_end']]}
    coverage = {key: [] for key in contributions}
    mismatches = []
    for item in case['sources']:
        source = item['source']
        for origin in source['origins']:
            key = origin['request_id'] + '|' + origin['contribution_id']
            start, end = origin['contribution_range']
            actual = source['raw_text'][origin['text_start']:origin['text_end']]
            part = contributions.get(key)
            expected = part['text'][start - part['start']:end - part['start']] if part else None
            if actual != expected:
                mismatches.append({'source_id': source['source_id'], 'contribution_id': key, 'range': [start, end], 'actual': actual, 'expected': expected})
            coverage.setdefault(key, []).append([start, end])
    gaps = []
    for key, part in contributions.items():
        cursor = part['start']
        for start, end in sorted(coverage[key]):
            if start != cursor:
                gaps.append({'contribution_id': key, 'expected_cursor': cursor, 'actual_start': start})
            cursor = end
        if cursor != part['end']:
            gaps.append({'contribution_id': key, 'expected_end': part['end'], 'covered_end': cursor})
    return {'exact_text_conserved_once': not mismatches and not gaps, 'authoritative_contributions': contributions, 'admitted_ranges': coverage, 'text_mismatches': mismatches, 'gaps_or_duplicates': gaps}



def analyze(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    result = {'trace_path': path.as_posix(), 'trace_sha256': hashlib.sha256(raw).hexdigest(), 'candidate': report['candidate'], 'route': report['route'], 'status': report['status'], 'driver_sha256': report['driver_sha256'], 'source_hashes': report['sources_sha256'], 'gemma_effective_backend': report.get('gemma', {}).get('readiness', {}).get('effective_backend'), 'replay': report.get('replay'), 'cases': []}
    for case in report['cases']:
        sources = [source_details(s, case) for s in case['sources']]
        text = ''.join(s['text'] for s in sources)
        reference = ' '.join(report['inputs'][m['input']]['source_text'] for m in case['assembly_scoring_only'])
        complete, missing = required_sequence(tokens(reference), tokens(text))
        submitted = [e for e in case['trace'] if e['event'] == 'native_submit']
        native = [e for e in case['trace'] if e['event'] == 'native_result']
        native_finals = [e for e in native if e['value'].get('outcome') is not None]
        snapshots = [e for e in case['trace'] if e['event'] == 'own_native_final_attribution']
        polls = [e['at'] for e in case['trace'] if e['event'] == 'evidence_poll']
        gaps = [b - a for a, b in zip(polls, polls[1:])]
        identified = [s for s in sources if (s['speaker'] or {}).get('speaker_id') is not None]
        keys = [(s['speaker']['session_scope'], s['speaker']['speaker_id']) for s in identified]
        a_keys = [(s['speaker']['session_scope'], s['speaker']['speaker_id']) for s in identified if 'washed' in s['text'].lower()]
        b_keys = [(s['speaker']['session_scope'], s['speaker']['speaker_id']) for s in identified if 'gentlemen' in s['text'].lower()]
        no_sources = [s for s in sources if 'no' in tokens(s['text'])]
        process_peaks = {}
        for sample in case['resources']:
            for proc in sample['processes']:
                peak = process_peaks.setdefault(proc['name'], {'rss': 0, 'private': 0, 'pids': []})
                peak['rss'] = max(peak['rss'], proc['rss'])
                peak['private'] = max(peak['private'], proc['private'] or 0)
                if proc['pid'] not in peak['pids']:
                    peak['pids'].append(proc['pid'])
        buffers = {key: max((s['speaker'].get(key, 0) for s in case['resources'] if s['speaker']), default=0) for key in ('pending_samples', 'pending_items', 'retained_map_pieces', 'evidence_items', 'evidence_evictions', 'overloads', 'copied_samples', 'max_copy_samples')}
        matching_pixels = [s['stable_software_readback_upper_bound_at'] for s in sources if s['stable_software_readback_upper_bound_at'] is not None]
        row = {'case': case['case'], 'arm': 'baseline' if case['mode'] == 'baseline' else case['profile'], 'repetition': case['repetition'], 'execution_status': case['status'], 'errors': case['errors'], 'error': case.get('error'), 'reference': reference, 'recognized_exact_concatenation': text, 'required_word_order_complete': complete and bool(text), 'missing_required_words': missing, 'content_status': 'complete_required_words' if complete and text else 'failed_content', 'complete_turn_software_readback_after_playback_s': max(matching_pixels) - case['playback_started_at'] if complete and len(matching_pixels) == len(sources) and sources else None, 'attribution_states': dict(Counter((s['speaker'] or {}).get('attribution_state', 'unavailable') for s in sources)), 'identified_source_count': len(identified), 'distinct_identity_keys': list(dict.fromkeys(keys)), 'clear_AB_distinct_observed': bool(a_keys and b_keys and set(a_keys).isdisjoint(b_keys)), 'returning_A_identity_stable_observed': len(a_keys) >= 2 and len(set(a_keys)) == 1, 'short_No_source_units': [{'text': s['text'], 'speaker': s['speaker'], 'translations': s['translations']} for s in no_sources], 'translation_calls': sum(e['event'] == 'translation_start' for e in case['trace']), 'sources': sources, 'capture_received_once': case['capture_received_once'], 'native_submits': submitted, 'native_finals': native_finals, 'provider_results': [e for e in case['trace'] if e['event'] == 'provider_result'], 'own_native_final_attribution_snapshots': snapshots, 'own_final_snapshot_count_matches_native_finals': len(snapshots) == len(native_finals) if case['mode'] == 'new' and report['route'] == 'gpu' else None, 'native_pcm_mismatch_count': sum(p.get('matches_declared_native_input') is not True for e in submitted for p in e['payloads']), 'recorded_native': case.get('recorded_native'), 'evidence_poll_interval_s': {'minimum': min(gaps) if gaps else None, 'median': statistics.median(gaps) if gaps else None, 'maximum': max(gaps) if gaps else None}, 'speaker_buffers': buffers, 'process_peaks': process_peaks, 'gpu_activation': case.get('gpu_activation'), 'speaker_initial': case.get('speaker_initial'), 'speaker_final': case.get('speaker_final'), 'frame_delivery_lag_max_s': max(case['frame_delivery_lags'], default=None), 'frame_delivery_lag_median_s': statistics.median(case['frame_delivery_lags']) if case['frame_delivery_lags'] else None}
        row['source_conservation'] = conservation(case)
        row['history_registered_once_per_source'] = all(sum(h['source_id'] == item['source']['source_id'] for h in item['history_after']) == 1 for item in case['sources'])
        orders = [s['order'] for s in sources]
        publications = [s['ordered_publication_completed_at'] for s in sources if s['ordered_publication_completed_at'] is not None]
        row['authoritative_source_order_preserved'] = orders == sorted(orders) and len(set(tuple(o) for o in orders)) == len(orders)
        row['ordered_publication_time_preserved'] = publications == sorted(publications) and len(publications) == len(sources)
        row['original_capture_availability_verified'] = all(e['submitted_end_sample'] <= e['available_capture_samples'] for e in case['trace'] if e['event'] == 'original_capture_availability') if report['route'] == 'deepgram-native-replay' else None
        row['captured_readback_count'] = sum(bool(r.get('capture', {}).get('sha256')) for r in case['renderer_readbacks'])
        row['nonempty_caption_color_pixel_readbacks'] = sum(sum(r.get('capture', {}).get('exact_color_pixel_counts', {}).values()) > 0 for r in case['renderer_readbacks'])
        row['blank_owned_readback_paths'] = [r['capture']['path'] for r in case['renderer_readbacks'] if r.get('capture', {}).get('sha256') and not sum(r['capture'].get('exact_color_pixel_counts', {}).values())]
        result['cases'].append(row)
    result['totals'] = {'cases': len(result['cases']), 'failed_content_cases': sum(c['content_status'] == 'failed_content' for c in result['cases']), 'execution_failures': sum(c['execution_status'] != 'executed' for c in result['cases']), 'translation_calls': sum(c['translation_calls'] for c in result['cases']), 'capture_mismatches': sum(not c['capture_received_once'] for c in result['cases']), 'native_pcm_mismatches': sum(c['native_pcm_mismatch_count'] for c in result['cases'])}
    return result


def archive_run(path, destination):
    members = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(path.rglob('*')):
            if file.is_file():
                name = file.relative_to(path).as_posix()
                raw = file.read_bytes()
                archive.writestr(name, raw)
                members[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'LF_only': b'\r\n' not in raw if file.suffix in ('.json', '.jsonl') else None}
    return {'path': destination.as_posix(), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'members': members}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    result = {'schema': 'issue218-practical-policy-summary-1', 'command': sys.argv, 'summary_driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': [analyze(p) for p in args.trace], 'archives': [archive_run(p.parent, p.parent.with_suffix('.zip')) for p in args.trace] if args.archive else [], 'limits': ['Strict required-word order detects empty/omission; does not substitute first-token success.', 'Native Deepgram events/timings are recorded; current source composition, real Nemotron, Gemma translation and Flet pixels are new.', 'No profile default, no VR/HMD or photon timing acceptance. All-gray is not identity acceptance.']}
    with args.output.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'runs': [{'route': r['route'], 'candidate': r['candidate'], 'totals': r['totals']} for r in result['runs']]}), flush=True)
