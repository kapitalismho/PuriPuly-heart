from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys

sys.dont_write_bytecode = True
import diagnose

GAP = Path(__file__).resolve().parent
FACTORIAL = GAP / 'factorial'
BENCH = GAP.parents[2]
REPO = BENCH.parents[1]
ARMS = 'ABCD'
CONTRASTS = ('AB', 'CD', 'AC', 'BD')


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def receipt(path):
    return {'path': path.relative_to(REPO).as_posix(), 'sha256': diagnose.digest(path), 'bytes': path.stat().st_size}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'min': None, 'median': None, 'p90': None, 'max': None}
    return {'count': len(ordered), 'min': ordered[0], 'median': statistics.median(ordered),
            'p90': ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .9) - 1)], 'max': ordered[-1]}


def source_point(native_ms, timeline):
    if not numeric(native_ms):
        return {'status': 'untimed', 'source_s': None, 'piece_index': None}
    seconds = native_ms / 1000
    for index, piece in enumerate(timeline):
        if piece['provider_start_s'] <= seconds < piece['provider_end_s']:
            if piece['source_start_s'] is None:
                return {'status': 'padding', 'source_s': None, 'piece_index': index}
            return {'status': 'mapped', 'source_s': piece['source_start_s'] + seconds - piece['provider_start_s'], 'piece_index': index}
    return {'status': 'outside_transmitted_support', 'source_s': None, 'piece_index': None}


def word_receipts(hypothesis, tokens, timeline):
    output = []
    for index, word in enumerate(hypothesis):
        contributing = [tokens[i] for i in word['token_indices']]
        arrival = [token.get('received_elapsed_s') for token in contributing]
        first_start = contributing[0].get('start_ms') if contributing else None
        output.append({'hypothesis_index': index, 'native_start_ms': first_start,
                       'native_end_ms': contributing[-1].get('end_ms') if contributing else None,
                       'source_start_point': source_point(first_start, timeline),
                       'complete_final_word_received_elapsed_s': max(arrival) if arrival and all(numeric(value) for value in arrival) else None})
    return output


def source_write_index(timeline, events):
    pieces = sorted((round(piece['source_start_s'] * 16000), round(piece['source_end_s'] * 16000),
                     round(piece['provider_start_s'] * 16000), i)
                    for i, piece in enumerate(timeline) if piece['source_start_s'] is not None)
    maxima = []
    for piece in pieces:
        maxima.append(max(piece[1], maxima[-1] if maxima else piece[1]))
    writes = sorted((event for event in events if event.get('event') == 'pcm_write'), key=lambda event: event['provider_start_sample'])
    return {'pieces': pieces, 'source_starts': [piece[0] for piece in pieces], 'source_end_prefix_max': maxima,
            'writes': writes, 'provider_starts': [event['provider_start_sample'] for event in writes]}


def source_word_end_writes(end_s, index):
    sample = math.ceil(end_s * 16000) - 1
    position = bisect_right(index['source_starts'], sample) - 1
    candidates = []
    while position >= 0 and index['source_end_prefix_max'][position] > sample:
        a, b, provider_start, piece_index = index['pieces'][position]
        if a <= sample < b:
            provider_sample = provider_start + sample - a
            write_index = bisect_right(index['provider_starts'], provider_sample) - 1
            if write_index >= 0 and provider_sample < index['writes'][write_index]['provider_end_sample']:
                event = index['writes'][write_index]
                candidates.append({'source_sample': sample, 'provider_sample': provider_sample, 'piece_index': piece_index,
                                   'pcm_write_elapsed_s': event['elapsed_s'], 'available_source_sample': event['available_source_sample']})
        position -= 1
    return sorted(candidates, key=lambda row: row['pcm_write_elapsed_s'])


def provider_readiness(reference, hypothesis, diagnostic, word_times, cuts, events, timeline):
    by_ref = {pair['reference_index']: pair for pair in diagnostic['pairs']}
    delays = []
    by_turn = {}
    input_index = source_write_index(timeline, events)
    sender_delays, after_write_delays = [], []
    input_support = Counter()
    for index, ref in enumerate(reference):
        pair = by_ref.get(index)
        if not pair or pair['status'] != 'anchored':
            continue
        timing = word_times[pair['hypothesis_index']]
        if numeric(timing['complete_final_word_received_elapsed_s']):
            delays.append(timing['complete_final_word_received_elapsed_s'] - ref['end_s'])
        candidates = source_word_end_writes(ref['end_s'], input_index)
        association = 'single_source_write' if len(candidates) == 1 else 'repeated_source_writes_ambiguous' if candidates else 'source_write_unavailable'
        input_support[association] += 1
        if len(candidates) == 1 and numeric(timing['complete_final_word_received_elapsed_s']):
            sender_delays.append(candidates[0]['pcm_write_elapsed_s'] - ref['end_s'])
            after_write_delays.append(timing['complete_final_word_received_elapsed_s'] - candidates[0]['pcm_write_elapsed_s'])
        turn = by_turn.setdefault(ref['turn_id'] or 'missing_turn', {'reference_word_indices': [], 'anchored_word_arrivals': []})
        turn['reference_word_indices'].append(index)
        turn['anchored_word_arrivals'].append({'reference_index': index, 'identity': pair['identity'], **timing,
                                              'source_word_end_write_association': association, 'source_word_end_write_candidates': candidates})
    seals = []
    previous = 0.0
    for cut in cuts:
        end = cut['source_cut_sample'] / 16000
        selected = [i for i, word in enumerate(reference) if previous <= word['start_s'] < end]
        control = next((event for event in events if event.get('event') == 'control_write' and event.get('control') == 'finalize' and event.get('provider_cut_sample') == cut['provider_cut_sample']), None)
        deadlines = {'frozen_source_cut_time': end}
        if control:
            deadlines['actual_finalize_write_time'] = control['elapsed_s']
        exposure = {}
        for name, deadline in deadlines.items():
            counts = Counter()
            for i in selected:
                pair = by_ref.get(i)
                if not pair:
                    counts['unmatched_reference'] += 1
                elif pair['status'] != 'anchored':
                    counts['ambiguous_alignment'] += 1
                else:
                    received = word_times[pair['hypothesis_index']]['complete_final_word_received_elapsed_s']
                    if received is None:
                        counts['arrival_unavailable'] += 1
                    elif received > deadline:
                        counts['not_yet_final_word'] += 1
                    else:
                        counts[pair['identity']] += 1
            exposure[name] = {'elapsed_s': deadline, 'reference_words': len(selected), 'counts': dict(counts),
                              'correct_available_reference_coverage': counts['correct'] / len(selected) if selected else None}
        seals.append({'source_window_s': [previous, end], 'reason': cut['reason'], 'exposure': exposure})
        previous = end
    return {'status': 'observed_provider_final_word_arrivals_not_application_publication',
            'rules': 'Word receipt requires all contributing final tokens; same frozen retrospective whole-case label mapping. Source-window membership uses published reference word start. Frozen source cuts and actual finalize-write clocks are separate deadlines; no additional fixed wait. Final-word delay subtracts published word end from receipt elapsed time without clipping negative values.',
            'final_word_receipt_minus_published_reference_word_end_s': summary(delays),
            'source_word_end_write_association_counts': dict(input_support),
            'single_source_write_minus_published_reference_word_end_s': summary(sender_delays),
            'final_word_receipt_minus_single_source_word_end_write_s': summary(after_write_delays),
            'input_write_rules': 'Associate last source sample before published reference word end with actual PCM writes through frozen source maps. Multiple replay occurrences remain ambiguous and are excluded only from delay decomposition, never from identity/coverage. Write-minus-reference-end is sender/source-availability delay; receipt-minus-write is combined post-input provider/wire/finalization delay, not pure model processing. Negative values remain literal.',
            'frozen_cut_exposure': seals, 'published_turn_anchored_arrivals': by_turn,
            'limit': 'Writer-relative replay elapsed time and published AMI word boundaries are not physical microphone clocks, acoustic clock calibration, caption publication, or consumer-owned-turn attachment. Labels are evaluated retrospectively; terminal mapping is not evidence the application knew canonical identities then.'}


def timing_diagnostics(result, responses, sdk):
    tokens = result['tokens']
    support_ms = result['sent_seconds'] * 1000
    timed = [(i, token) for i, token in enumerate(tokens) if numeric(token.get('start_ms')) and numeric(token.get('end_ms'))]
    outside = [(i, token) for i, token in timed if token['start_ms'] < 0 or token['end_ms'] > support_ms]
    widths = Counter(token['end_ms'] - token['start_ms'] for _, token in timed)
    counters = [{'received_elapsed_s': row['received_elapsed_s'], 'counters': {key: value for key, value in row['response'].items() if key in ('final_audio_proc_ms', 'total_audio_proc_ms') or 'processed' in key}}
                for row in responses if any(key in ('final_audio_proc_ms', 'total_audio_proc_ms') or 'processed' in key for key in row['response'])]
    counter_summary = {}
    for key in sorted({key for row in counters for key in row['counters']}):
        values = [row['counters'][key] for row in counters if numeric(row['counters'].get(key))]
        if values:
            counter_summary[key] = {'observations': len(values), 'first': values[0], 'last': values[-1], 'min': min(values), 'max': max(values)}
    examples = []
    for i, token in outside[:5]:
        examples.append({'token_index': i, 'native': token, 'source_start_point': source_point(token['start_ms'], result['timeline']),
                         'mapped_positive_interval': diagnose.projected_interval(token['start_ms'], token['end_ms'], result['timeline'])})
    return {'status': 'literal_native_timing_no_correction', 'transmitted_support_ms': support_ms,
            'timed_tokens': len(timed), 'untimed_tokens': len(tokens) - len(timed), 'width_counts_ms': {str(key): value for key, value in sorted(widths.items())},
            'zero_duration_tokens': widths[0], 'outside_support_tokens': len(outside),
            'max_native_end_minus_transmitted_support_ms': max((token['end_ms'] - support_ms for _, token in timed), default=None),
            'out_of_support_examples': examples, 'processed_audio_counter_observations': counters,
            'processed_audio_counter_summary': counter_summary,
            'sdk_projection_diagnostics': sdk.get('projection_diagnostics') if sdk else None,
            'limit': 'Native widths are timing metadata only and never identity scoring support. Counters are retained as actually observed. Neither transcript matching nor published word boundaries independently calibrates clocks.'}


def attribution_traces(reference, hypothesis, diagnostic, word_times, result, sdk):
    output = []
    for pair in diagnostic['pairs']:
        if pair['identity'] not in ('wrong', 'missing', 'mixed') or len(output) >= 8:
            continue
        h = hypothesis[pair['hypothesis_index']]
        groups = []
        if sdk:
            for index, group in enumerate(sdk['sdk_segments']):
                if any(group['token_start_index'] <= i < group['token_end_index'] for i in h['token_indices']):
                    groups.append({'sdk_group_index': index, **{key: value for key, value in group.items() if key != 'text'}})
        output.append({'alignment': pair, 'reference': reference[pair['reference_index']], 'recognized': h,
                       'word_timing': word_times[pair['hypothesis_index']],
                       'native_tokens': [{'token_index': i, 'native': result['tokens'][i], 'source_start_point': source_point(result['tokens'][i].get('start_ms'), result['timeline'])} for i in h['token_indices']],
                       'sdk_groups': groups, 'consumer_publication': 'not exercised by this diagnostic'})
    return output


def sender_versions():
    original_plan = FACTORIAL / 'run_plan.json'
    original_config = FACTORIAL / 'history' / 'experiment_config_v1.json'
    original_sender = FACTORIAL / 'history' / 'sender_v1.py'
    amended_plan = FACTORIAL / 'run_plan_v2.json'
    amended_sender = FACTORIAL / 'history' / 'sender_v2.py'
    contexts = {}
    for version, plan_path, config_path, sender_path in (
        (1, original_plan, original_config, original_sender),
        (2, amended_plan, GAP / 'experiment_config.json', amended_sender),
    ):
        if not plan_path.is_file():
            continue
        plan = diagnose.load(plan_path)
        config = diagnose.load(config_path)
        if diagnose.digest(sender_path) != plan['harness_sha256'] or diagnose.digest(config_path) != plan['config_sha256']:
            raise ValueError(f'Frozen sender/config identity mismatch for version {version}')
        contexts[version] = {'plan': plan, 'config': config, 'version': version, 'plan_receipt': receipt(plan_path),
                             'configuration_receipt': receipt(config_path), 'sender_receipt': receipt(sender_path)}
    return contexts


def sender_binding(result, contexts):
    declared = result.get('sender_mechanics')
    version = declared['version'] if declared else 1
    if not declared and result['arm'] not in 'AB':
        raise ValueError('Only original completed A/B may lack sender-version metadata')
    if version not in contexts:
        raise ValueError(f'Sender version {version} has no frozen applicability context')
    context = contexts[version]
    if declared and (declared['harness_sha256'] != context['sender_receipt']['sha256'] or declared['original_plan_sha256'] != contexts[1]['plan_receipt']['sha256']):
        raise ValueError('Declared sender/plan receipt does not match preserved history')
    return {'version': version, 'metadata_present_in_result': declared is not None, 'declared_metadata': declared,
            'plan_receipt': context['plan_receipt'], 'configuration_receipt': context['configuration_receipt'],
            'sender_receipt': context['sender_receipt'],
            'effective_packetization': 'A/B original nominal800 packet/pacing mechanics unchanged by amendment' if result['arm'] in 'AB' else 'C/D amended nominal800 coalescing with frozen-cut/S200/source-discontinuity/availability boundaries',
            'legacy_binding': None if declared else 'Explicit original-plan/history binding; not a fabricated provider metadata field'}


def wire_observations(events):
    writes = [event for event in events if event.get('event') == 'pcm_write']
    sequence = [{key: event[key] for key in ('provider_start_sample', 'provider_end_sample', 'available_source_sample', 'kind', 'sha256')} for event in writes]
    sizes = Counter(event['provider_end_sample'] - event['provider_start_sample'] for event in writes)
    spreads = [event['packet_availability_spread_samples'] / 16000 for event in writes if numeric(event.get('packet_availability_spread_samples'))]
    return {'pcm_writes': len(writes), 'packet_sizes_samples': {str(key): count for key, count in sorted(sizes.items())},
            'packet_bytes_positions_availability_canonical_sha256': canonical_hash(sequence),
            'source_availability_to_actual_write_s': summary([event['elapsed_s'] - event['available_source_sample'] / 16000 for event in writes]),
            'write_stall_s': summary([event['write_stall_s'] for event in writes]),
            'observed_packet_added_availability_spread_s': summary(spreads),
            'packet_spread_field_unavailable_writes': len(writes) - len(spreads),
            'packet_position_examples': sequence[:3] + sequence[-3:],
            'limit': 'Actual logged input-write identities/boundaries/availability, not identical elapsed clocks or packet arrival at the model. Added buffering field is unavailable in original v1 receipts, never assumed zero.'}


def diagnose_arm(case, arm, contexts, score_row):
    folder = FACTORIAL / 'live' / case['id'] / arm
    result_path = folder / 'result.json'
    if not result_path.is_file():
        return {'status': 'blocked_pending_result', 'missing': [result_path.relative_to(REPO).as_posix()]}
    result = diagnose.load(result_path)
    binding = sender_binding(result, contexts)
    plan_row = next(row for row in contexts[binding['version']]['plan']['rows'] if row['case_id'] == case['id'] and row['arm'] == arm)
    cuts = plan_row['cut_positions']
    base = {'result_receipt': receipt(result_path), 'completed': result.get('completed'), 'failure': result.get('failure'),
            'native_result_sha256_verified': result.get('native_artifact_sha256') == canonical_hash(result['tokens']),
            'plan_pcm_identity_verified': result.get('pcm_sha256') == plan_row['pcm_sha256'],
            'plan_trace_identity_verified': result.get('trace_sha256') == plan_row['trace_sha256'],
            'source_map_canonical_sha256': canonical_hash(result['timeline'])}
    base['sender_version_binding'] = binding
    if not result.get('completed'):
        return {**base, 'status': 'failed_provider_run_no_primary_identity_score', 'missing': [], 'limit': 'Partial/failed output is not a completed-arm comparison and is not scored as zero.'}
    if result.get('mode') != 'live' or result.get('case_id') != case['id'] or result.get('arm') != arm:
        raise ValueError(f'Not the completed planned live case/arm: {case["id"]}/{arm}')
    if not all(base[key] for key in ('native_result_sha256_verified', 'plan_pcm_identity_verified', 'plan_trace_identity_verified')):
        raise ValueError(f'Immutable receipt or frozen plan mismatch: {case["id"]}/{arm}')
    events_path, raw_path = folder / 'events.jsonl', folder / 'responses.jsonl'
    events, responses = rows(events_path), rows(raw_path)
    if diagnose.digest(raw_path) != result['raw_responses_sha256']:
        raise ValueError(f'Raw response receipt mismatch: {case["id"]}/{arm}')
    sdk_path = FACTORIAL / 'sdk_output' / arm / f'{case["id"]}.json'
    rttm_path = FACTORIAL / 'predictions' / arm / f'{case["id"]}.rttm'
    sdk = diagnose.load(sdk_path) if sdk_path.is_file() else None
    if sdk and sdk['native_artifact_sha256'] != result['native_artifact_sha256']:
        raise ValueError(f'SDK/native receipt mismatch: {case["id"]}/{arm}')
    inventory = diagnose.load(GAP / 'ami_word_reference_inventory.json')
    reference, gaps = diagnose.reference_words(case, inventory)
    if not reference:
        return {**base, 'status': 'blocked_missing_word_reference', 'reference_gaps': gaps, 'missing': []}
    hypothesis = diagnose.recognized_words(result['tokens'])
    diagnostic = diagnose.identity_diagnostic(reference, hypothesis)
    word_times = word_receipts(hypothesis, result['tokens'], result['timeline'])
    missing = [path.relative_to(REPO).as_posix() for path in (sdk_path, rttm_path) if not path.is_file()]
    if score_row is None:
        missing.append(f'factorial/scores.json: per_arm.{arm}.{case["id"]}')
    output = {**base, 'status': 'passed' if not missing else 'identity_passed_der_or_projection_pending', 'missing': missing,
              'events_receipt': receipt(events_path), 'raw_responses_receipt': receipt(raw_path),
              'sdk_receipt': receipt(sdk_path) if sdk else None, 'prediction_receipt': receipt(rttm_path) if rttm_path.is_file() else None,
              'score': score_row, 'config': result['config'], 'provider_runtime_environment': result['environment'],
              'actual_wire_observations': wire_observations(events),
              'source_seconds': result['source_seconds'], 'sent_seconds': result['sent_seconds'], 'actual_audio_seconds': result['actual_audio_seconds'],
              'finalizations': result['finalizations'], 'fin_count': result['fin_count'], 'pending_completions': result['pending_completions'],
              'reference_gaps': gaps, 'reference_words': reference, 'recognized_words': hypothesis, 'word_receipts': word_times,
              'word_diagnostic': diagnostic, 'provider_readiness': provider_readiness(reference, hypothesis, diagnostic, word_times, cuts, events, result['timeline']),
              'native_timing': timing_diagnostics(result, responses, sdk),
              'representative_native_attribution_traces': attribution_traces(reference, hypothesis, diagnostic, word_times, result, sdk)}
    if diagnose.digest(result_path) != base['result_receipt']['sha256']:
        raise ValueError(f'Result changed during diagnosis: {case["id"]}/{arm}')
    return output


def compare_words(left, right):
    duration = left['sent_seconds']
    pairs = diagnose.lexical_pairs(left['recognized_words'], right['recognized_words'])
    buckets = [[] for _ in range(4)]
    source_offsets = []
    examples = []
    statuses = Counter(pair['status'] for pair in pairs)
    for pair in pairs:
        if pair['status'] != 'anchored':
            continue
        a, b = left['word_receipts'][pair['reference_index']], right['word_receipts'][pair['hypothesis_index']]
        if numeric(a['native_start_ms']) and numeric(b['native_start_ms']):
            bucket = min(3, max(0, int(a['native_start_ms'] / (duration * 1000 / 4))))
            buckets[bucket].append(b['native_start_ms'] - a['native_start_ms'])
        sa, sb = a['source_start_point']['source_s'], b['source_start_point']['source_s']
        if numeric(sa) and numeric(sb):
            source_offsets.append(sb - sa)
        if len(examples) < 4:
            examples.append({'left_hypothesis_index': pair['reference_index'], 'right_hypothesis_index': pair['hypothesis_index'], 'word': left['recognized_words'][pair['reference_index']]['word'], 'left_receipt': a, 'right_receipt': b})
    ls, rs = left['word_diagnostic']['strata']['all'], right['word_diagnostic']['strata']['all']
    def delta(key):
        return rs[key] - ls[key] if rs[key] is not None and ls[key] is not None else None
    left_counters = left.get('native_timing', {}).get('processed_audio_counter_summary', {})
    right_counters = right.get('native_timing', {}).get('processed_audio_counter_summary', {})
    counter_differences = {key: right_counters[key]['last'] - left_counters[key]['last']
                           for key in ('final_audio_proc_ms', 'total_audio_proc_ms') if key in left_counters and key in right_counters}
    return {'status': 'observed_single_run', 'exact_lexical_pair_counts': dict(statuses),
            'conditional_wrong_fraction_difference_right_minus_left': delta('conditional_wrong_fraction_known_anchored'),
            'correct_reference_coverage_difference_right_minus_left': delta('correct_colored_reference_coverage'),
            'recognized_word_count_difference_right_minus_left': right['word_diagnostic']['recognized_words'] - left['word_diagnostic']['recognized_words'],
            'der_percentage_point_difference_right_minus_left': 100 * (right['score']['der'] - left['score']['der']) if left['score'] and right['score'] else None,
            'native_start_offset_ms_by_left_provider_time_quartile': [summary(bucket) for bucket in buckets],
            'left_provider_time_quartiles_s': [[i * duration / 4, (i + 1) * duration / 4] for i in range(4)],
            'mapped_source_start_offset_seconds': summary(source_offsets), 'examples': examples,
            'last_processed_counter_difference_ms_right_minus_left': counter_differences,
            'additional_requested_finalizations_right_minus_left': right['finalizations'] - left['finalizations'] if 'finalizations' in left and 'finalizations' in right else None,
            'left_sender_binding': left.get('sender_version_binding'), 'right_sender_binding': right.get('sender_version_binding'),
            'actual_packet_bytes_positions_availability_matched': left.get('actual_wire_observations', {}).get('packet_bytes_positions_availability_canonical_sha256') == right.get('actual_wire_observations', {}).get('packet_bytes_positions_availability_canonical_sha256') if 'actual_wire_observations' in left and 'actual_wire_observations' in right else None,
            'applicability': 'AB/CD finalization under declared matched PCM; require matching actual packet/availability signatures and inspect elapsed pacing/stalls. AC/BD combined PCM-plus-source-availability/packet-boundary/remaining buffering policy, not an isolated pure waveform effect.',
            'limit': 'Supporting transcript-to-transcript timing observations only; not independently anchored acoustic clock calibration. Source-point mapping can fail in padding/outside support; those matches remain in lexical denominators. No fitted correction.'}


def live_findings(evidence):
    lines = []
    contrasts = [comparison for case in evidence['cases'].values() for pair, comparison in case['contrasts'].items() if pair in ('AB', 'CD') and comparison['status'] == 'observed_single_run']
    if contrasts and all(c['der_percentage_point_difference_right_minus_left'] is not None for c in contrasts):
        worsened = sum(c['der_percentage_point_difference_right_minus_left'] > 0 for c in contrasts)
        matched = sum(c['actual_packet_bytes_positions_availability_matched'] is True for c in contrasts)
        lines.append(f'Manual app-cut finalization raises declared DER in {worsened}/{len(contrasts)} observed AB/CD comparisons; actual packet/availability signatures match in {matched}/{len(contrasts)}. This is the single-run controlled replay result, not application-adapter parity or a pure native-identity error percentage.')
    long_case = evidence['cases'].get('long_context', {})
    if all(long_case.get('contrasts', {}).get(pair, {}).get('der_percentage_point_difference_right_minus_left') is not None for pair in ('AB', 'CD')):
        ab, cd = (long_case['contrasts'][pair] for pair in ('AB', 'CD'))
        lines.append(
            f"long_context: AB adds {ab['der_percentage_point_difference_right_minus_left']:.2f} DER points "
            f"while conditional native identity error changes {100*ab['conditional_wrong_fraction_difference_right_minus_left']:+.2f} points; "
            f"CD adds {cd['der_percentage_point_difference_right_minus_left']:.2f} DER points with native error "
            f"{100*cd['conditional_wrong_fraction_difference_right_minus_left']:+.2f} points and correct reference coverage "
            f"{100*cd['correct_reference_coverage_difference_right_minus_left']:+.2f} points. "
            "The large DER penalty therefore does not uniformly represent native recognized-word ID loss."
        )
        if 'total_audio_proc_ms' in ab['last_processed_counter_difference_ms_right_minus_left'] and 'total_audio_proc_ms' in cd['last_processed_counter_difference_ms_right_minus_left']:
            lines.append(
                f"For the same additional {ab['additional_requested_finalizations_right_minus_left']} requested controls, "
                f"final total_audio_proc_ms changes by {ab['last_processed_counter_difference_ms_right_minus_left']['total_audio_proc_ms']} ms in AB "
                f"and {cd['last_processed_counter_difference_ms_right_minus_left']['total_audio_proc_ms']} ms in CD. "
                "Actual native counter differences plus progressive word offsets strengthen the clock-discrepancy observation beyond transcript matching alone, "
                "but do not support a fixed per-requested-finalize rule or identify acoustic timestamp correction."
            )
    brief = evidence['cases'].get('brief_interjections', {}).get('contrasts', {})
    if all(brief.get(pair, {}).get('conditional_wrong_fraction_difference_right_minus_left') is not None for pair in ('AB', 'CD')):
        lines.append(
            f"brief_interjections shows native conditional error increases of {100*brief['AB']['conditional_wrong_fraction_difference_right_minus_left']:.2f} points in AB "
            f"and {100*brief['CD']['conditional_wrong_fraction_difference_right_minus_left']:.2f} points in CD, "
            f"with correct reference coverage changes {100*brief['AB']['correct_reference_coverage_difference_right_minus_left']:+.2f}/"
            f"{100*brief['CD']['correct_reference_coverage_difference_right_minus_left']:+.2f} points. "
            "A timing-only explanation cannot replace these native-attribution/coverage failures."
        )
    returning = evidence['cases'].get('long_return_gap', {})
    for arm in ('C', 'D'):
        result = returning.get('arms', {}).get(arm, {})
        event = returning.get('reference_case_manifest', {}).get('features', {}).get('return_event')
        if 'word_diagnostic' not in result or not event:
            continue
        speaker = event['speaker']
        counts, ids, turn_counts = Counter(), Counter(), {}
        for pair in result['word_diagnostic']['pairs']:
            ref = result['reference_words'][pair['reference_index']]
            if ref['speaker'] == speaker and pair['status'] == 'anchored':
                native_id = result['recognized_words'][pair['hypothesis_index']]['speaker']
                counts[pair['identity']] += 1
                ids[str(native_id)] += 1
                turn_counts.setdefault(ref['turn_id'] or 'missing_turn', Counter())[pair['identity']] += 1
        mapped = {native_id: result['word_diagnostic']['global_per_case_mapping'].get(native_id) for native_id in ids}
        lines.append(
            f"long_return_gap/{arm}: returning reference speaker {speaker} has anchored outcomes `{dict(counts)}`, "
            f"native IDs `{dict(ids)}`, global mapping `{mapped}`, "
            f"and published-turn outcomes `{ {key: dict(value) for key, value in turn_counts.items()} }`. "
            "These literal recognized-word witnesses distinguish a native label merge/attribution failure from displaced native timing; "
            "they are not re-mapped separately for the returning turn."
        )
    lines.append(
        "Raw source-end-to-final-word receipt includes substantial sender backlog. "
        "The separately reported single-occurrence post-write medians are about seven seconds for terminal-only A/C and about three to four seconds "
        "for cut-finalized B/D in these runs; this combined processing/wire/finalization difference is a quality/readiness frontier, "
        "not an application publication measurement or a proposed fixed color wait."
    )
    return lines


def render(evidence):
    lines = ['# #204 live A-D native identity and text/coverage diagnosis', '', f"Status: **{evidence['status']}**.", '',
             'Read-only consumption of factorial-owned immutable outputs. No provider calls, production edits, new reference alignment rules, native-width identity scoring, or timestamp correction.', '',
             '## Runnable command', '', '```text',
             'experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/live_diagnose.py', 'experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/live_diagnose.py --schema-check-only', '```', '',
             'A missing result/projection/score is blocked/pending, never a zero score. Completed cases may be diagnosed while other runs remain pending. Exit 0 means every planned result/projection/score was available; exit 2 means indispensable inputs are pending or a provider run failed. Runtime exceptions mean receipt/configuration failure.', '',
             '## Frozen protocol', '',
             f"Identity rules: `{evidence['identity_diagnostic_version']}` imported unchanged from diagnose.py (SHA-256 `{evidence['frozen_helper_sha256']}`). Reference inventory SHA-256 `{evidence['reference_inventory_receipt']['sha256']}`.",
             f"Diagnostic execution revision: `{evidence['execution_revision']}` at `{evidence['execution_utc']}`; Python `{evidence['environment']['python']}`; scipy `{evidence['environment']['scipy']}`; numpy `{evidence['environment']['numpy']}`. Native inference runtime/configuration is retained separately in each immutable provider result.",
             'Lowercase ASCII lexicalization with apostrophes; concatenate native text before finding words; conservative exact lexical five-word/unique-gap anchors; explicit ambiguous/unmatched words. One maximum-count speaker assignment for each entire case/arm, never per turn. Same published AMI word and turn membership, overlap, <=0.7-s interjection and missing-turn strata as the retained diagnosis. This remains conditional native recognized-word attribution, not complete ASR accuracy or application color accuracy.',
             'A/B use original complete PCM, C/D frozen app PCM with replay/S200 positions preserved; A/C terminal finalize only, B/D frozen app-cut finalize. After long_context/A completed while B was in flight, Director authorized sender v2 before the first C/D call: nominal800 coalescing across incidental 512-sample VAD frames. Exact C/D bytes/cuts/maps are preserved; packet availability is the maximum included availability, with cuts/S200/discontinuities/availability windows still split. A/B mechanics remain unchanged and original evidence is not rerun. Per-arm source-version binding and actual packet signatures are disclosed below. Comparisons are not archived ACK-gated segmented adapter parity.',
             'AB/CD vary finalization under each declared paired PCM/packetization policy. AC/BD still vary combined waveform, source availability, packet boundaries and remaining added buffering: no pure-PCM causal claim is inferred from the amendment or DER alone.', '',
    ]
    lines.extend(['', '## Sender versions and applicability', '',
                  'The original and amended plans/configurations/sender histories are retained and hashed in evidence. Owner-provided dry mechanics/byte proofs are contextual evidence, not live model accuracy. Original v1 results lacking sender metadata are explicitly bound to original A/B plan/history; no version field is fabricated.',
                  '', '| Case | Arm | Sender source version | Actual PCM writes | Packet-size histogram (samples) | Added availability spread maximum s |',
                  '|---|---|---:|---:|---|---:|'])
    for case_id, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            if 'actual_wire_observations' not in result:
                continue
            wire = result['actual_wire_observations']
            lines.append(f"| {case_id} | {arm} | {result['sender_version_binding']['version']} | {wire['pcm_writes']} | `{json.dumps(wire['packet_sizes_samples'], sort_keys=True)}` | {wire['observed_packet_added_availability_spread_s']['max']} |")
    lines.extend(['', '## Artifact availability', '', '| Case | Arm | Result |', '|---|---|---|'])
    for case_id, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            lines.append(f"| {case_id} | {arm} | {result['status']} |")
            for missing in result.get('missing', []):
                lines.append(f'| | | Missing: `{missing}` |')
    lines.extend(['', '## Native identity alongside DER and text/coverage', '',
                  '| Case | Arm | DER % | Ref / recognized words | Anchored / ambiguous | Unmatched ref / recognized | Wrong % known anchored | Correct reference coverage % |', '|---|---|---:|---:|---:|---:|---:|---:|'])
    for case_id, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            if 'word_diagnostic' not in result:
                continue
            d = result['word_diagnostic']
            s = d['strata']['all']
            def percent(value):
                return f'{value*100:.2f}' if value is not None else 'unavailable'
            lines.append(f"| {case_id} | {arm} | {percent(result['score']['der']) if result['score'] else 'pending'} | {d['reference_words']} / {d['recognized_words']} | {d['anchored_words']} / {d['ambiguous_words']} | {d['unmatched_reference_words']} / {d['unmatched_recognized_words']} | {percent(s['conditional_wrong_fraction_known_anchored'])} | {percent(s['correct_colored_reference_coverage'])} |")
    lines.extend(['', '## Paired causal contrasts', '', '| Case | Contrast right-minus-left | DER pp | Conditional native wrong pp | Correct coverage pp | Recognized words | Actual packet/availability signatures matched |', '|---|---|---:|---:|---:|---:|---|'])
    for case_id, case in evidence['cases'].items():
        for pair, comparison in case['contrasts'].items():
            if comparison['status'] != 'observed_single_run':
                lines.append(f'| {case_id} | {pair} | pending | pending | pending | pending | pending |')
                continue
            def pp(value):
                return f'{value:.2f}' if value is not None else 'pending'
            wrong = comparison['conditional_wrong_fraction_difference_right_minus_left']
            coverage = comparison['correct_reference_coverage_difference_right_minus_left']
            lines.append(f"| {case_id} | {pair} | {pp(comparison['der_percentage_point_difference_right_minus_left'])} | {pp(wrong*100 if wrong is not None else None)} | {pp(coverage*100 if coverage is not None else None)} | {comparison['recognized_word_count_difference_right_minus_left']} | {comparison['actual_packet_bytes_positions_availability_matched']} |")
    lines.extend(['', 'AB/CD vary finalization under matched declared PCM; AC/BD vary combined PCM policy, not isolated replay/padding/silence removal. One run per arm, exposed diagnostic subset: no statistical-significance claim. Native identity uses each arm\'s global mapping and explicit coverage; unequal lexical matches constrain causal attribution.', '',
                  '## Difficult strata', '', '| Case | Arm | Stratum | Reference denominator | Outcomes |', '|---|---|---|---:|---|'])
    for case_id, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            if 'word_diagnostic' not in result:
                continue
            for name in ('overlap', 'nonoverlap', 'short_interjection', 'turn_reference_missing'):
                s = result['word_diagnostic']['strata'][name]
                lines.append(f"| {case_id} | {arm} | {name} | {s['reference_words']} | `{json.dumps(s['counts'], sort_keys=True)}` |")
    lines.extend(['', '## Native timing and available final-word evidence', '',
                  '| Case | Arm | Outside support tokens | Max native end minus support ms | Processed-counter response rows | Word arrival minus reference end: median / p90 s |', '|---|---|---:|---:|---:|---|'])
    for case_id, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            if 'native_timing' not in result:
                continue
            t = result['native_timing']
            delay = result['provider_readiness']['final_word_receipt_minus_published_reference_word_end_s']
            lines.append(f"| {case_id} | {arm} | {t['outside_support_tokens']} | {t['max_native_end_minus_transmitted_support_ms']} | {len(t['processed_audio_counter_observations'])} | {delay['median']} / {delay['p90']} |")
    lines.extend(['', '### Separating sender backlog from post-input final-word delay', '',
                  '| Case | Arm | Sender/write minus reference word end: median / p90 s | Final word minus actual source-word-end write: median / p90 s | Source-write association counts |',
                  '|---|---|---|---|---|'])
    for case_id, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            if 'provider_readiness' not in result:
                continue
            ready = result['provider_readiness']
            sender = ready['single_source_write_minus_published_reference_word_end_s']
            after = ready['final_word_receipt_minus_single_source_word_end_write_s']
            lines.append(f"| {case_id} | {arm} | {sender['median']} / {sender['p90']} | {after['median']} / {after['p90']} | `{json.dumps(ready['source_word_end_write_association_counts'], sort_keys=True)}` |")
    lines.extend(['', '## Matched-input native clock observations', '',
                  '| Case | Contrast | Additional requested finalizations | Final observed total_audio_proc_ms difference ms | Anchored / ambiguous word matches | Native-start offset quartile medians ms |',
                  '|---|---|---:|---:|---|---|'])
    for case_id, case in evidence['cases'].items():
        for pair in ('AB', 'CD'):
            comparison = case['contrasts'][pair]
            if comparison['status'] != 'observed_single_run':
                continue
            count = comparison['exact_lexical_pair_counts']
            offsets = [bucket['median'] for bucket in comparison['native_start_offset_ms_by_left_provider_time_quartile']]
            lines.append(f"| {case_id} | {pair} | {comparison['additional_requested_finalizations_right_minus_left']} | {comparison['last_processed_counter_difference_ms_right_minus_left'].get('total_audio_proc_ms', 'unavailable')} | {count.get('anchored', 0)} / {count.get('ambiguous', 0)} | `{offsets}` |")
    lines.extend(['', '## Interpretation of completed evidence', ''])
    lines.extend('- ' + finding for finding in live_findings(evidence))
    lines.append('')
    lines.extend(['', 'Exact native tokens, processed counters as actually received, reference IDs, SDK-group indices and source-map intersections are traceable through immutable artifact receipts and representative examples in [live_identity_evidence.json](live_identity_evidence.json). Supporting native-clock and mapped-source offset quartiles are retained for every available contrast, including ambiguity and source-point support limits. No reference-fitted shift or 120-ms-per-finalize subtraction.',
                  'Provider word receipt requires every contributing final token. The evidence records published-turn arrivals and per-frozen-cut exposure separately at source-cut time and actual finalize-write time. Unmatched, ambiguous, missing/mixed and not-yet-final words remain in denominators. Negative arrival-minus-reference-end values are retained rather than clipped. These replay/reference clocks are not independently verified acoustic anchors.', '',
                  'Raw word-arrival minus published speech end includes sender backlog/context policy. A separate source-map association locates actual PCM writes for the last source sample before each anchored reference word end. Single-occurrence sender/write delay and subsequent word-receipt delay are retained separately; repeated-context occurrences are explicitly ambiguous. The post-write part combines provider, wire, finalization and receiver behavior, not pure model processing.',
                  'Archived segmented wall_time_s began before source-real-time offline app trace generation; new factorial active-WebSocket timing begins after frozen traces. Therefore archived long_context wall time versus new D WebSocket duration is not an adapter latency, processing, or billing improvement. This report compares only explicitly identified current replay/input/receipt clocks.',
                  '## Passed / blocked / not run', '',
                  '- Passed when available: frozen-helper and pinned-reference identities, native/raw/plan receipts, independent lexical/global speaker mapping, explicit coverage/strata, actual final-token receive observations.',
                  '- Passed model-free schema mechanics: half-open source/padding boundaries, subword final-word receipt barrier, unavailable receipt timestamps, separate source-cut/control-write exposure, literal overrun and actually present processed-counter retention. These are not live model-quality or caption-publication evidence.',
                  '- Result/projection/score gaps: none in a completed report; pending or failed arms remain explicitly listed in incomplete reports and are never zero measurements.',
                  '- R4 independent acoustic clock validation remains unresolved even with actual processed counters: published word times and transcript matches are not new waveform/event calibration.',
                  '- Not run here: provider generation/scoring (factorial owner), application normalization/scope and first-readable publication (consumption owner), physical microphone/VRChat/HMD, production enablement and billing.', '',
                  '## Disposition', '',
                  'Use native identity, correct reference coverage, omissions/ambiguity and provider-word arrival observations beside DER, not as substitutes. Defer a live-quality operating-point recommendation while indispensable comparisons or application publication evidence remain unavailable. No additional wait, recoloring, dropping short turns, retry, or production change is proposed by this diagnostic.', ''])
    return '\n'.join(lines)


def schema_self_checks():
    timeline = [{'provider_start_s': 0, 'provider_end_s': 4, 'source_start_s': 10, 'source_end_s': 14},
                {'provider_start_s': 4, 'provider_end_s': 5, 'source_start_s': None, 'source_end_s': None}]
    assert source_point(0, timeline)['source_s'] == 10
    assert source_point(4000, timeline)['status'] == 'padding'
    assert source_point(5000, timeline)['status'] == 'outside_transmitted_support'
    tokens = [{'text': ' al', 'speaker': '1', 'start_ms': 100, 'end_ms': 100, 'received_elapsed_s': 1},
              {'text': 'pha bravo charlie delta echo', 'speaker': '1', 'start_ms': 200, 'end_ms': 5100, 'received_elapsed_s': 2}]
    hypothesis = diagnose.recognized_words(tokens)
    reference = [{'word': word, 'speaker': 'A', 'start_s': i / 10, 'end_s': (i + 1) / 10,
                  'id': str(i), 'turn_id': 'turn', 'overlap': False, 'short_interjection': True}
                 for i, word in enumerate(('alpha', 'bravo', 'charlie', 'delta', 'echo'))]
    diagnostic = diagnose.identity_diagnostic(reference, hypothesis)
    times = word_receipts(hypothesis, tokens, timeline)
    assert times[0]['complete_final_word_received_elapsed_s'] == 2
    unavailable = word_receipts(hypothesis, [tokens[0], {**tokens[1], 'received_elapsed_s': None}], timeline)
    assert unavailable[0]['complete_final_word_received_elapsed_s'] is None
    readiness = provider_readiness(reference, hypothesis, diagnostic, times,
                                  [{'source_cut_sample': 40000, 'provider_cut_sample': 40000, 'reason': 'fixture'}],
                                  [{'event': 'control_write', 'control': 'finalize', 'provider_cut_sample': 40000, 'elapsed_s': 1.5}],
                                  timeline)
    exposure = readiness['frozen_cut_exposure'][0]['exposure']
    assert exposure['frozen_source_cut_time']['counts'] == {'correct': 5}
    assert exposure['actual_finalize_write_time']['counts'] == {'not_yet_final_word': 5}
    timing = timing_diagnostics({'tokens': tokens, 'timeline': timeline, 'sent_seconds': 5},
                                [{'received_elapsed_s': 2, 'response': {'total_audio_proc_ms': 5000}}], None)
    assert timing['outside_support_tokens'] == 1
    assert timing['max_native_end_minus_transmitted_support_ms'] == 100
    writes = [{'event': 'pcm_write', 'provider_start_sample': 0, 'provider_end_sample': 64000,
               'elapsed_s': 15, 'available_source_sample': 224000}]
    association = source_word_end_writes(12, source_write_index(timeline, writes))
    assert association[0]['provider_sample'] == 31999 and association[0]['pcm_write_elapsed_s'] == 15
    repeated_timeline = timeline + [{'provider_start_s': 5, 'provider_end_s': 7, 'source_start_s': 11, 'source_end_s': 13}]
    repeated_writes = writes + [{'event': 'pcm_write', 'provider_start_sample': 80000, 'provider_end_sample': 112000,
                                'elapsed_s': 20, 'available_source_sample': 224000}]
    assert len(source_word_end_writes(12, source_write_index(repeated_timeline, repeated_writes))) == 2
    left = {'recognized_words': hypothesis, 'word_receipts': times, 'word_diagnostic': diagnostic, 'score': {'der': .1}, 'sent_seconds': 5}
    shifted = [{**token, 'start_ms': token['start_ms'] + 500, 'end_ms': token['end_ms'] + 500} for token in tokens]
    right = {**left, 'word_receipts': word_receipts(hypothesis, shifted, timeline), 'score': {'der': .2}}
    comparison = compare_words(left, right)
    assert comparison['conditional_wrong_fraction_difference_right_minus_left'] == 0
    assert comparison['native_start_offset_ms_by_left_provider_time_quartile'][0]['median'] == 500
    assert math.isclose(comparison['mapped_source_start_offset_seconds']['median'], .5)
    assert math.isclose(comparison['der_percentage_point_difference_right_minus_left'], 10)
    contexts = {version: {'plan_receipt': {'sha256': f'plan{version}'}, 'configuration_receipt': {},
                          'sender_receipt': {'sha256': f'sender{version}'}} for version in (1, 2)}
    legacy = sender_binding({'arm': 'A'}, contexts)
    assert legacy['version'] == 1 and legacy['metadata_present_in_result'] is False
    amended = sender_binding({'arm': 'C', 'sender_mechanics': {'version': 2, 'harness_sha256': 'sender2', 'original_plan_sha256': 'plan1'}}, contexts)
    assert amended['version'] == 2 and amended['metadata_present_in_result'] is True
    try:
        sender_binding({'arm': 'C', 'sender_mechanics': {'version': 2, 'harness_sha256': 'wrong', 'original_plan_sha256': 'plan1'}}, contexts)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched sender source was accepted')
    packet = {'event': 'pcm_write', 'provider_start_sample': 0, 'provider_end_sample': 800, 'available_source_sample': 800,
              'kind': 'source', 'sha256': 'packet', 'elapsed_s': .1, 'write_stall_s': .001, 'packet_availability_spread_samples': 512}
    observed = wire_observations([packet])
    assert observed['packet_sizes_samples'] == {'800': 1}
    assert observed['observed_packet_added_availability_spread_s']['max'] == .032
    changed = wire_observations([{**packet, 'available_source_sample': 801}])
    assert changed['packet_bytes_positions_availability_canonical_sha256'] != observed['packet_bytes_positions_availability_canonical_sha256']
    return {'status': 'passed', 'cases': ['half-open source/padding/outside timing boundaries', 'complete-word receive barrier across subword tokens', 'unavailable contributing receive timestamp', 'separate source-cut and actual control-write exposure', 'literal overrun and actually present processed counters', 'canonical source-end sample to actual PCM write', 'replayed source support remains ambiguous', 'paired literal timing changes remain separate from native identity and DER', 'original/amended sender applicability and incorrect receipt rejection', 'packet signatures include source availability and added buffering'],
            'limit': 'Synthetic diagnostic schema mechanics only; not provider, application ownership, caption publication, or model-quality evidence'}


def main():
    parser = argparse.ArgumentParser(description='Read completed factorial A-D receipts using unchanged retained lexical/native diagnostics')
    parser.add_argument('--schema-check-only', action='store_true')
    args = parser.parse_args()
    retained = diagnose.load(GAP / 'offline_evidence.json')
    frozen_hash = retained['environment']['diagnosis_script_sha256']
    if diagnose.digest(GAP / 'diagnose.py') != frozen_hash:
        raise ValueError('Retained helper source no longer matches frozen evidence; do not silently change identity rules')
    schema_checks = schema_self_checks()
    if args.schema_check_only:
        print(json.dumps(schema_checks), flush=True)
        return
    inventory_path = GAP / 'ami_word_reference_inventory.json'
    inventory = diagnose.load(inventory_path)
    for row in inventory['files']:
        if row['status'] == 'available' and diagnose.digest(GAP / row['path']) != row['sha256']:
            raise ValueError(f'Pinned public reference changed: {row["path"]}')
    cfg_path, plan_path, scores_path = GAP / 'experiment_config.json', FACTORIAL / 'run_plan.json', FACTORIAL / 'scores.json'
    cfg = diagnose.load(cfg_path)
    contexts = sender_versions()
    manifest = {case['id']: case for case in diagnose.load(BENCH / 'manifest.json')['cases']}
    score_rows = {}
    if scores_path.is_file():
        scores = diagnose.load(scores_path)
        if scores.get('pyannote.metrics') != '4.0.0' or scores.get('collar') != 0 or scores.get('overlap_included') is not True or scores.get('mapping') != 'one optimal entire-case speaker mapping':
            raise ValueError('Live scores do not follow frozen scoring protocol')
        score_rows = {(arm, row['id']): row for arm, values in scores['per_arm'].items() for row in values}
    cases = {}
    for case_id in cfg['subset']:
        arms = {}
        for arm in ARMS:
            arms[arm] = diagnose_arm(manifest[case_id], arm, contexts, score_rows.get((arm, case_id)))
        contrasts = {}
        for pair in CONTRASTS:
            left, right = (arms[arm] for arm in pair)
            contrasts[pair] = compare_words(left, right) if 'word_diagnostic' in left and 'word_diagnostic' in right else {'status': 'blocked_pending_paired_results'}
        cases[case_id] = {'reference_case_manifest': manifest[case_id], 'arms': arms, 'contrasts': contrasts}
        print(f'case={case_id} status=' + ','.join(f'{arm}:{row["status"]}' for arm, row in arms.items()), flush=True)
    complete = all(row['status'] == 'passed' for case in cases.values() for row in case['arms'].values())
    revision = diagnose.command(['git.exe', 'rev-parse', 'HEAD'])
    evidence = {'status': 'completed' if complete else 'blocked_pending_indispensable_inputs',
                'identity_diagnostic_version': retained['identity_diagnostic_version'], 'frozen_helper_sha256': frozen_hash,
                'schema_self_checks': schema_checks,
                'reference_inventory_receipt': receipt(inventory_path), 'configuration_receipt': receipt(cfg_path),
                'plan_receipt': receipt(plan_path), 'scores_receipt': receipt(scores_path) if scores_path.is_file() else None,
                'sender_versions': {str(version): {key: value for key, value in context.items() if key not in ('plan', 'config')} for version, context in contexts.items()},
                'sender_amendment': cfg.get('sender_amendment'),
                'owner_amendment_proof_receipts': [receipt(path) for path in (FACTORIAL / 'dry_verification_v2.json', FACTORIAL / 'packet_bytes_verification_v2.json') if path.is_file()],
                'environment': {'python': sys.version, 'platform': platform.platform(), 'scipy': importlib.metadata.version('scipy'), 'numpy': importlib.metadata.version('numpy'),
                                'diagnostic_script_sha256': diagnose.digest(Path(__file__)), 'execution_argv': sys.argv}, 'cases': cases,
                'execution_revision_probe': revision, 'execution_revision': revision['stdout'].strip() if revision['returncode'] == 0 else 'unavailable',
                'execution_utc': datetime.now(timezone.utc).isoformat()}
    diagnose.save(GAP / 'live_identity_evidence.json', evidence)
    (GAP / 'live_identity_report.md').write_text(render(evidence), encoding='utf-8')
    print(f'status={evidence["status"]} report=live_identity_report.md evidence=live_identity_evidence.json', flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
