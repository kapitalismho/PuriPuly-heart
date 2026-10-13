from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import issue218_summarize as local


def clean_b_word_spans(case):
    from dataclasses import asdict
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from puripuly_heart.core.audio.format import AudioCaptureSpan
    from puripuly_heart.core.speaker.contracts import ProbabilityFrame, SourceMapPiece, SpeakerEvidence
    from puripuly_heart.core.stt.streaming_speaker import _Timeline
    maps, cursor = [], 0
    for event in case['trace']:
        if event['event'] != 'native_submit':
            continue
        for payload in event['payloads']:
            left = cursor
            for span in payload['source_ranges']:
                length = span['normalized_end_sample'] - span['normalized_start_sample']
                maps.append((left, left + length, AudioCaptureSpan(**span), payload['context_only']))
                left += length
            cursor += payload['samples']
    original = next(e['value'] for e in case['trace'] if e['event'] == 'native_result' and e['value'].get('text'))
    provenance = original['provenance']
    clocks = {clock['clock_id']: clock for clock in provenance['audio_clocks']}
    facts = []
    for timing in provenance['text_timing']:
        if timing['semantics'] != 'word_interval':
            continue
        clock = clocks[timing['clock_id']]
        scale = clock['sample_rate_hz'] / (1000 if timing['unit'] == 'ms' else 1)
        start = clock['origin_sent_sample'] + math.floor(timing['raw_start'] * scale)
        end = clock['origin_sent_sample'] + math.ceil(timing['raw_end'] * scale)
        position, ranges = start, []
        for left, right, span, context_only in maps:
            a, b = max(position, left), min(end, right)
            if a >= b:
                continue
            if a != position:
                break
            ranges.append(span.slice_normalized(span.normalized_start_sample + a - left, span.normalized_start_sample + b - left))
            position = b
        admitted = next(s for s in case['sources'] if any(o['contribution_id'] == timing['contribution_id'] and o['contribution_range'][0] <= timing['text_start'] and timing['text_end'] <= o['contribution_range'][1] for o in s['source']['origins']))
        timeline, used = _Timeline(), 0
        for event in case['trace']:
            if event['at'] > admitted['admitted_at']:
                break
            if event['event'] != 'evidence_delivery':
                continue
            batch = []
            for item in case['evidence'][used:used + event['count']]:
                fields = dict(item)
                fields['source_map'] = tuple(SourceMapPiece(**{**piece, 'capture': AudioCaptureSpan(**piece['capture']) if piece['capture'] else None}) for piece in item['source_map'])
                fields['frame'] = ProbabilityFrame(**{**item['frame'], 'probabilities': tuple(item['frame']['probabilities']), 'active_slots': tuple(item['frame']['active_slots'])}) if item['frame'] else None
                batch.append(SpeakerEvidence(**fields))
            used += event['count']
            timeline.observe(batch)
        intersections = {entry['input']: sum(max(0, min(span.normalized_end_sample, entry['scenario_sample_range'][1]) - max(span.normalized_start_sample, entry['scenario_sample_range'][0])) for span in ranges) for entry in case['assembly_scoring_only']}
        boundaries = [asdict(boundary) for boundary in timeline.boundaries() if any(span.capture_epoch == boundary.capture_epoch and span.normalized_start_sample < boundary.transition_end_sample and boundary.transition_start_sample < span.normalized_end_sample for span in ranges)]
        facts.append({'word': original['text'][timing['text_start']:timing['text_end']], 'native_estimated_word_interval': timing, 'clock': clock, 'mapped_capture_ranges': [asdict(span) for span in ranges], 'map_complete': position == end, 'reference_intersection_samples_scoring_only': intersections, 'outside_reference_samples': sum(span.normalized_sample_count for span in ranges) - sum(intersections.values()), 'actual_admitted_text': admitted['source']['raw_text'], 'actual_admitted_speaker': admitted['source']['speaker'], 'full_span_timeline_at_admission': asdict(timeline.attribute(tuple(ranges))), 'overlapping_recorded_model_transition_ranges': boundaries, 'limit': 'Offline explanation of recorded actual output using current full-span policy; not a new E2E run or exact phonetic alignment. Reference assembly never participates in decisions. All four lexical words are in the known B reference sentence; estimated word-span overlap with A or a model transition is retained as ambiguity, not assigned by majority.'})
    return facts


def analyze(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    regular = bool(report['cases']) and all('playback_started_at' in c for c in report['cases'])
    summary = local.analyze(path) if regular else {'trace_path': path.as_posix(), 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'route': report['route'], 'status': report['status'], 'cases': []}
    summary['candidate'] = report['candidate']
    summary['credential_selection'] = report['credential_selection']
    summary['error'] = report.get('error')
    summary['budget_before'] = report['budget_before']
    summary['budget_after'] = report['budget_after']
    summary['evidence_poll_seconds'] = 0.05
    summary['billing'] = 'Monetary charge unknown unless native response reports it; reserved attempted audio is a conservative spend bound, written audio means successful SDK write, not provider billed duration.'
    if not regular:
        summary['case_attempts'] = [{k: c.get(k) for k in ('case', 'mode', 'profile', 'repetition', 'status', 'error', 'actual_audio_samples_written', 'capture_received_once', 'errors')} for c in report['cases']]
        return summary
    for item, case in zip(summary['cases'], report['cases']):
        trace = case['trace']
        exact_source = ''.join(s['source']['raw_text'] for s in case['sources'])
        expected_words = re.findall(r'\w+', item['reference'].lower())
        exact_words = re.findall(r'\w+', exact_source.lower())
        item['exact_source_concatenation'] = exact_source
        item['exact_source_WER_punctuation_tokenized'] = local.distance(expected_words, exact_words) / len(expected_words) if expected_words else None
        item['scoring_limit'] = 'The inherited recognized/WER fields insert spaces between source units. Exact-source WER here tokenizes punctuation and inserts no boundary text; raw source/native text remains authoritative. No score is an acceptance threshold.'
        raw = [e for e in trace if e['event'] == 'raw_wire_result']
        item['ASR_input_seals'] = item.pop('ASR_calls')
        item['wire_connection_attempts'] = [e for e in trace if e['event'] == 'wire_connection_attempt']
        item['actual_audio_seconds_successful_write'] = case['actual_audio_samples_written'] / 16000
        item['controls'] = [e for e in trace if e['event'] == 'wire_write_complete' and e['control']]
        item['raw_result_types'] = dict(Counter(str(e['value'].get('type', 'google_live_server_message')) for e in raw))
        item['reported_usage_events'] = [e for e in raw if e['value'].get('usage_metadata') is not None or e['value'].get('duration') is not None and str(e['value'].get('type')) == 'Metadata']
        item['returned_model_events'] = [e for e in raw if e['value'].get('metadata') or e['value'].get('model_version')]
        item['native_independent_units'] = [e for e in trace if e['event'] == 'provider_result' and isinstance(e['value'], dict) and e['value'].get('unit', {}).get('identity', {}).get('unit_id')]
        item['continuous_stream_payload_mismatches'] = sum(e['event'] == 'native_stream_input' and not e['matches_declared_source_pcm'] for e in trace)
        item['input_and_source_ownership_limit'] = 'Gemini independent native final is not assigned to a local current request, speaker or display group from receipt proximity; no simulated source clock/text mapping.' if report['route'] == 'gemini_transcribe' else 'Deepgram qualified finalized lexical time/source join from actual submitted maps, no optional physical speaker cut; gray crossings retained.'
        item['complete_required_words_availability'] = {'status': 'passed' if expected_words == exact_words else 'failed', 'definition': 'Exact ordered reference words available in admitted source; punctuation is excluded from this bounded check. Omitted or substituted required content cannot receive a shorter complete-turn latency.', 'renderer_dispatch_ack_after_playback_s': max((s['renderer_dispatch_ack_after_playback_s'] for s in item['sources'] if s['renderer_dispatch_ack_after_playback_s'] is not None), default=None) if expected_words == exact_words else None}
        item['native_observation_boundary'] = 'SDK-decoded native response immediately before the product adapter; raw websocket JSON bytes were not captured.'
        if report['route'] == 'deepgram' and case['case'] == 'clean_A_B' and case['mode'] == 'new':
            item['clean_B_sentence_word_span_attribution_explanation'] = clean_b_word_spans(case)
        if report['route'] == 'gemini_transcribe':
            item['native_sequence'] = [{'trace_index': index, **event} for index, event in enumerate(trace) if event['event'] in {'raw_wire_result', 'vad', 'native_submit', 'native_stream_end'} or event['event'] == 'wire_write_complete' and event['control']]
            native_finals = [e for e in raw if (e['value'].get('server_content') or {}).get('input_transcription') is not None]
            mappings = []
            for sequence, event in enumerate(native_finals, 1):
                native = event['value']['server_content']['input_transcription']
                terminals = [e for e in item['native_independent_units'] if e['value']['unit']['identity']['receipt_sequence'] == sequence]
                terminal_ids = {e['value']['unit']['identity']['unit_id'] for e in terminals}
                admitted = [s for s in case['sources'] if any(o['identity']['unit_id'] in terminal_ids for o in s['source']['origins'])]
                mappings.append({'receipt_sequence': sequence, 'native_received_at': event['at'], 'native_final_field': native, 'terminal_units': terminals, 'admitted_sources': admitted, 'exact_fragment_preserved': len(terminals) == len(admitted) == 1 and native.get('text', '') == terminals[0]['value']['unit']['text'] == admitted[0]['source']['raw_text'], 'association': 'Native adapter receipt sequence and recognition-unit identity only; no source audio/speaker association inferred.'})
            item['raw_final_to_admitted_mapping'] = mappings
            item['all_received_authoritative_final_fragments_preserved'] = bool(mappings) and all(m['exact_fragment_preserved'] for m in mappings)
            item['finality_disposition'] = 'Recorded authoritative input_transcription fragments are independently admitted; earlier words appear only in speculative interim_input_transcription. No discarded authoritative earlier final is demonstrated here. End-to-end normal-final text coverage fails where required words are absent; service-versus-SDK/native-contract cause remains unresolved without raw-wire/contract qualification. Do not promote interim or infer speaker/source correspondence.'
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'schema': 'issue218-cloud-summary-1', 'command': sys.argv, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': [analyze(path) for path in args.trace], 'scope': 'Fresh authorized Deepgram and exact Gemini Transcribe routes only; no whole218 completion/default selection. Software dispatch ACK and timed owned readbacks remain separate from pixel onset/HMD photons.'}
    with args.output.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
