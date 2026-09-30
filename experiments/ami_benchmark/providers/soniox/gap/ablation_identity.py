from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import live_diagnose as live

GAP = live.GAP
ROOT = live.FACTORIAL / 'ablation'
ARM = 'D_no_S200'
CASES = ('brief_interjections', 'long_return_gap')


def exposure_summary(arm):
    output = {}
    for deadline in ('frozen_source_cut_time', 'actual_finalize_write_time'):
        counts = Counter()
        words = windows = missing = 0
        for cut in arm['provider_readiness']['frozen_cut_exposure']:
            row = cut['exposure'].get(deadline)
            if row is None:
                missing += 1
                continue
            words += row['reference_words']
            counts.update(row['counts'])
            windows += 1
        output[deadline] = {'reference_words': words, 'windows': windows, 'missing_deadline_windows': missing,
                            'counts': dict(counts), 'correct_available_reference_coverage': counts['correct'] / words if words else None}
    return output


def return_witness(case, arm):
    event = case.get('features', {}).get('return_event')
    if not event:
        return None
    speaker = event['speaker']
    pairs = {pair['reference_index']: pair for pair in arm['word_diagnostic']['pairs']}
    counts, ids, turns = Counter(), Counter(), {}
    for index, ref in enumerate(arm['reference_words']):
        if ref['speaker'] != speaker:
            continue
        pair = pairs.get(index)
        outcome = pair['identity'] if pair else 'unmatched_reference'
        counts[outcome] += 1
        turn = turns.setdefault(ref['turn_id'] or 'missing_turn', {'counts': Counter(), 'native_ids': Counter(), 'anchored_word_witnesses': []})
        turn['counts'][outcome] += 1
        if pair and pair['status'] == 'anchored':
            hyp = arm['recognized_words'][pair['hypothesis_index']]
            native = str(hyp['speaker'])
            ids[native] += 1
            turn['native_ids'][native] += 1
            turn['anchored_word_witnesses'].append({'reference_index': index, 'hypothesis_index': pair['hypothesis_index'],
                                                   'reference_id': ref['id'], 'word': ref['word'], 'identity': outcome,
                                                   'native_speaker': hyp['speaker'], 'native_token_indices': hyp['token_indices'],
                                                   'word_receipt': arm['word_receipts'][pair['hypothesis_index']]})
    for turn in turns.values():
        turn['counts'], turn['native_ids'] = dict(turn['counts']), dict(turn['native_ids'])
    return {'reference_speaker': speaker, 'manifest_event': event, 'all_reference_word_outcomes': dict(counts),
            'anchored_native_id_counts': dict(ids), 'whole_case_native_mapping': arm['word_diagnostic']['global_per_case_mapping'],
            'published_turns': turns, 'limit': 'One unchanged whole-case mapping, never per-turn relabeling; retrospective evaluation is not application-known canonical identity.'}


def compare(left, right):
    result = live.compare_words(left, right)
    result['applicability'] = 'Only S200 packets removed, paired to immutable original D. Retained source/replay packets, cuts and configuration require independent byte/availability verification. Removal also changes provider positions and permits earlier controls/input; actual elapsed pacing is not matched. Single runs, no ACK-gated production parity.'
    result['native_offset_limit'] = 'Different transmitted PCM duration and provider positions mechanically affect native transcript offsets; these are not clock calibration or timestamp-correction estimates.'
    result['sent_seconds_difference_right_minus_left'] = right['sent_seconds'] - left['sent_seconds']
    result['reference_strata'] = {}
    for name, left_row in left['word_diagnostic']['strata'].items():
        right_row = right['word_diagnostic']['strata'][name]
        result['reference_strata'][name] = {'D': left_row, ARM: right_row}
    result['arrival_exposure'] = {'D': exposure_summary(left), ARM: exposure_summary(right)}
    result['post_input_final_word_delay_s'] = {name: arm['provider_readiness']['final_word_receipt_minus_single_source_word_end_write_s'] for name, arm in (('D', left), (ARM, right))}
    result['sender_source_availability_delay_s'] = {name: arm['provider_readiness']['single_source_write_minus_published_reference_word_end_s'] for name, arm in (('D', left), (ARM, right))}
    return result


def percentage(value):
    return 'unavailable' if value is None else f'{100 * value:.2f}%'


def number(value):
    return 'unavailable' if value is None else f'{value:.3f}'


def render(evidence):
    lines = ['# #204 S200-only bounded native/text/readiness diagnosis', '',
             f"Status: **{evidence['status']}**. Diagnostic version: **S200-only v1**.", '',
             '## Scope and unchanged evidence', '',
             'Two user-approved public-AMI calls only: brief_interjections and long_return_gap, D-minus-S200 paired against immutable original D. No provider calls are made by this diagnostic. Original A-D live_identity_evidence.json and live_identity_report.md, retained diagnose.py, public references and baseline artifacts are not rewritten.', '',
             'The exact frozen lexicalization, conservative timing-independent anchors, one whole-case optimal native-label mapping, reference coverage/strata and provider-arrival exposure helpers are reused. Native token widths never provide identity support. Missing, mixed, ambiguous, unmatched and not-yet-final words stay explicit; a lower conditional error alone is not a text/quality win.', '',
             '## Bounded outcomes', '',
             '| Case | Arm | DER | Known anchored native error | Correct reference-word coverage | Recognized words | Sent PCM seconds | Post-input final-word median s | Sender/source-availability median s |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case_id, case in evidence['cases'].items():
        for name, arm in case['arms'].items():
            if 'word_diagnostic' not in arm:
                lines.append(f'| {case_id} | {name} | {arm["status"]} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |')
                continue
            all_words = arm['word_diagnostic']['strata']['all']
            ready = arm['provider_readiness']
            der = arm.get('score', {}).get('der') if arm.get('score') else None
            lines.append(f'| {case_id} | {name} | {percentage(der)} | {percentage(all_words["conditional_wrong_fraction_known_anchored"])} | {percentage(all_words["correct_colored_reference_coverage"])} | {arm["word_diagnostic"]["recognized_words"]} | {arm["sent_seconds"]:.3f} | {number(ready["final_word_receipt_minus_single_source_word_end_write_s"]["median"])} | {number(ready["single_source_write_minus_published_reference_word_end_s"]["median"])} |')
    lines.extend(['', '## Quality, overlap and short-interjection strata', '',
                  'Coverage is correct reference words divided by all published reference words in the stratum, not just recognized matches. Conditional native error uses only known anchored words. Every residual category is retained below.'])
    for case_id, case in evidence['cases'].items():
        contrast = case.get('contrast')
        if not contrast:
            continue
        lines.extend(['', f'### {case_id}', '',
                      f"D-minus-S200 minus D: DER {contrast['der_percentage_point_difference_right_minus_left']:+.2f} points; native conditional error {100 * contrast['conditional_wrong_fraction_difference_right_minus_left']:+.2f} points; correct reference coverage {100 * contrast['correct_reference_coverage_difference_right_minus_left']:+.2f} points; recognized words {contrast['recognized_word_count_difference_right_minus_left']:+d}; transmitted PCM {contrast['sent_seconds_difference_right_minus_left']:+.3f}s.", '',
                      '| Stratum | Arm | Reference words | Native error | Correct coverage | Explicit outcomes |', '|---|---|---:|---:|---:|---|'])
        for stratum, rows in contrast['reference_strata'].items():
            for name, row in rows.items():
                lines.append(f'| {stratum} | {name} | {row["reference_words"]} | {percentage(row["conditional_wrong_fraction_known_anchored"])} | {percentage(row["correct_colored_reference_coverage"])} | `{row["counts"]}` |')
        lines.extend(['', '### Complete final-word arrivals and cut exposure', '',
                      'A word is received only when every contributing final token is received. Separate source-cut and actual-finalize-write deadlines use the unchanged retrospective whole-case mapping; there is no added fixed color wait. Post-input delay subtracts the actual write containing the last published word sample; replay ambiguities are excluded only from that decomposition. This combined provider/wire/finalization observation is not pure model processing or application publication.', '',
                      '| Deadline | Arm | Reference words | Correct available coverage | Explicit exposure outcomes |', '|---|---|---:|---:|---|'])
        for name, exposures in contrast['arrival_exposure'].items():
            for deadline, row in exposures.items():
                lines.append(f'| {deadline} | {name} | {row["reference_words"]} | {percentage(row["correct_available_reference_coverage"])} | `{row["counts"]}` |')
        lines.extend(['', 'Actual-wire S200-only verification:', '', f"```json\n{live.json.dumps(case['s200_only_actual_wire_verification'], indent=2)}\n```", '',
                      f"Literal final processed-counter differences (minus-S200 minus D): `{contrast['last_processed_counter_difference_ms_right_minus_left']}`. Requested-finalize difference: `{contrast['additional_requested_finalizations_right_minus_left']}`. Different PCM provider positions/duration mechanically affect transcript native offsets; no guessed correction or clock calibration is derived."])
    returning = evidence['cases'].get('long_return_gap', {})
    if returning.get('return_witness'):
        lines.extend(['', '## Returning-speaker witness', '', 'Original published turns and a single whole-case native-label mapping are retained. Full word/token/receipt witnesses are in the JSON.'])
        for name, witness in returning['return_witness'].items():
            lines.extend(['', f"**{name}**, reference speaker `{witness['reference_speaker']}`: all-reference outcomes `{witness['all_reference_word_outcomes']}`, anchored native IDs `{witness['anchored_native_id_counts']}`, global mapping `{witness['whole_case_native_mapping']}`."])
            for turn_id, turn in witness['published_turns'].items():
                lines.append(f"- `{turn_id}`: `{turn['counts']}`; native IDs `{turn['native_ids']}`.")
    lines.extend(['', '## Supported recommendation and limits', '', *evidence['recommendation'], '',
                  '## Receipts and reproducibility', '', f"Execution UTC: `{evidence['execution_utc']}`.", '',
                  f"```json\n{live.json.dumps(evidence['receipts'], indent=2)}\n```", '',
                  f"Frozen helper SHA-256: `{evidence['frozen_helper_sha256']}`. Frozen live-helper SHA-256: `{evidence['frozen_live_helper_sha256']}`.", '',
                  'Input result/events/responses/SDK/RTTM receipts, all per-word alignment outcomes, published-turn arrivals, every cut exposure, actual counters, literal native times and source-write associations are in ablation_identity_evidence.json. Public AMI annotations are independently published references, not newly listener-calibrated acoustic anchors. These two exposed clips are not holdout validation or independent participant sampling.', ''])
    return '\n'.join(lines)


def verify_actual_wire(original, result, old_events, events, planned):
    old_pcm = [row for row in old_events if row['event'] == 'pcm_write']
    new_pcm = [row for row in events if row['event'] == 'pcm_write']
    retained = [row for row in old_pcm if row['kind'] != 'S200']
    fields = ('sha256', 'source_start_sample', 'source_end_sample', 'available_source_sample', 'kind', 'bytes', 'packet_availability_spread_samples')
    source_signature = lambda values: [{key: row.get(key) for key in fields} for row in values]
    expected = planned['ablation']['source_packet_remap']
    actual = [{'old_provider_start_sample': old['provider_start_sample'], 'old_provider_end_sample': old['provider_end_sample'],
               'new_provider_start_sample': new['provider_start_sample'], 'new_provider_end_sample': new['provider_end_sample'],
               'source_start_sample': new['source_start_sample'], 'source_end_sample': new['source_end_sample'],
               'available_source_sample': new['available_source_sample'], 'sha256': new['sha256']}
              for old, new in zip(retained, new_pcm)]
    controls = [row for row in events if row['event'] == 'control_write' and row['control'] == 'finalize']
    old_controls = [row for row in old_events if row['event'] == 'control_write' and row['control'] == 'finalize']
    control_remap = [{'old_provider_cut_sample': old['provider_cut_sample'], 'new_provider_cut_sample': new['provider_cut_sample'],
                      'available_source_sample': new['available_source_sample'], 'terminal': new['terminal']}
                     for old, new in zip(old_controls, controls)]
    source_pieces = lambda timeline: [(row['source_start_s'], row['source_end_s']) for row in timeline if row['source_start_s'] is not None]
    source_cuts = lambda cuts: [(row['source_cut_sample'], row['reason']) for row in cuts]
    checks = {
        'no_actual_S200_packets': all(row['kind'] != 'S200' for row in new_pcm),
        'same_source_replay_packets_order_bytes_boundaries_availability': source_signature(retained) == source_signature(new_pcm),
        'actual_provider_positions_equal_frozen_packet_remap': len(retained) == len(new_pcm) and actual == expected,
        'same_finalize_source_availability_terminal_sequence': [(row['available_source_sample'], row['terminal']) for row in old_controls] == [(row['available_source_sample'], row['terminal']) for row in controls],
        'actual_control_positions_equal_frozen_remap': len(old_controls) == len(controls) and control_remap == planned['ablation']['control_remap'],
        'same_source_timeline_and_replay': source_pieces(original['timeline']) == source_pieces(result['timeline']),
        'same_source_cuts_and_reasons': source_cuts(original['cut_positions']) == source_cuts(result['cut_positions']),
        'same_provider_configuration': original['config'] == result['config'],
        'same_requested_and_completed_finalization_counts': original['finalizations'] == result['finalizations'] == len(controls) and original['fin_count'] == result['fin_count'],
        'actual_audio_equals_shortened_transmitted_support': result['actual_audio_seconds'] == result['sent_seconds'],
        'exact_planned_S200_sample_removal': round((original['sent_seconds'] - result['sent_seconds']) * 16000) == planned['ablation']['removed_samples'],
    }
    if not all(checks.values()):
        raise ValueError(f'S200-only actual-wire verification failed: {checks}')
    return {'status': 'passed', 'checks': checks, 'removed_samples': planned['ablation']['removed_samples'],
            'removed_seconds': planned['ablation']['removed_seconds'], 'original_pcm_packets': len(old_pcm),
            'retained_source_packets': len(retained), 'actual_ablation_pcm_packets': len(new_pcm),
            'retained_source_packet_signature_sha256': live.canonical_hash(source_signature(retained)),
            'actual_source_packet_signature_sha256': live.canonical_hash(source_signature(new_pcm)),
            'limit': 'S200-only byte/availability/cut isolation, not identical elapsed pacing or provider model arrival times. Removing paced padding can advance controls and successors. Raw sender does not gate intermediate ACKs.'}


def diagnose_ablation(case, baseline, planned, plan, score):
    result_path = ROOT / 'live' / case['id'] / ARM / 'result.json'
    if not result_path.is_file():
        return {'status': 'blocked_pending_result', 'missing': [str(result_path.relative_to(live.REPO))]}, None
    before = live.receipt(result_path)
    result = live.diagnose.load(result_path)
    base = {'result_receipt': before, 'completed': result['completed'], 'failure': result['failure']}
    if not result['completed']:
        return {**base, 'status': 'failed_provider_run_no_primary_identity_score', 'limit': 'Partial output is not scored as zero.'}, None
    if result['mode'] != 'live' or result['case_id'] != case['id'] or result['arm'] != ARM:
        raise ValueError('Not the planned completed live S200-only case')
    if result['native_artifact_sha256'] != live.canonical_hash(result['tokens']) or result['pcm_sha256'] != planned['pcm_sha256'] or result['trace_sha256'] != planned['trace_sha256'] or result['timeline'] != planned['timeline'] or result['ablation'] != planned['ablation']:
        raise ValueError('Native/PCM/trace/map/ablation identity mismatch')
    if result['sender_mechanics']['harness_sha256'] != plan['base_harness_sha256']:
        raise ValueError('Executed sender hash does not match frozen ablation plan')
    old_result_path = live.REPO / baseline['result_receipt']['path']
    old_events_path = live.REPO / baseline['events_receipt']['path']
    for path, digest in ((old_result_path, baseline['result_receipt']['sha256']), (old_events_path, baseline['events_receipt']['sha256'])):
        if live.diagnose.digest(path) != digest:
            raise ValueError(f'Original paired D input changed: {path}')
    if planned['ablation']['original_D_result_sha256'] != baseline['result_receipt']['sha256'] or planned['ablation']['original_D_events_sha256'] != baseline['events_receipt']['sha256']:
        raise ValueError('Ablation targets a different original D result')
    original = live.diagnose.load(old_result_path)
    events_path, raw_path = result_path.parent / 'events.jsonl', result_path.parent / 'responses.jsonl'
    events, responses = live.rows(events_path), live.rows(raw_path)
    if live.diagnose.digest(raw_path) != result['raw_responses_sha256']:
        raise ValueError('Immutable raw response hash mismatch')
    verification = verify_actual_wire(original, result, live.rows(old_events_path), events, planned)
    sdk_path = ROOT / 'sdk_output' / ARM / f'{case["id"]}.json'
    prediction_path = ROOT / 'predictions' / ARM / f'{case["id"]}.rttm'
    sdk = live.diagnose.load(sdk_path) if sdk_path.is_file() else None
    if sdk and sdk['native_artifact_sha256'] != result['native_artifact_sha256']:
        raise ValueError('SDK native input receipt mismatch')
    reference, gaps = live.diagnose.reference_words(case, live.diagnose.load(GAP / 'ami_word_reference_inventory.json'))
    if reference != baseline['reference_words']:
        raise ValueError('Independent reference word/turn rules differ from frozen paired D')
    hypothesis = live.diagnose.recognized_words(result['tokens'])
    diagnostic = live.diagnose.identity_diagnostic(reference, hypothesis)
    word_times = live.word_receipts(hypothesis, result['tokens'], result['timeline'])
    missing = [str(path.relative_to(live.REPO)) for path in (sdk_path, prediction_path) if not path.is_file()]
    if score is None:
        missing.append(f'ablation/scores.json: {case["id"]}')
    output = {**base, 'status': 'passed' if not missing else 'identity_passed_projection_or_der_pending', 'missing': missing,
              'events_receipt': live.receipt(events_path), 'raw_responses_receipt': live.receipt(raw_path),
              'sdk_receipt': live.receipt(sdk_path) if sdk else None, 'prediction_receipt': live.receipt(prediction_path) if prediction_path.is_file() else None,
              'score': score, 'config': result['config'], 'provider_runtime_environment': result['environment'],
              'sender_version_binding': {'declared': result['sender_mechanics'], 'base_harness_sha256': plan['base_harness_sha256'],
                                         'ablation_runner_sha256': plan['ablation_runner_sha256'], 'plan_receipt': live.receipt(ROOT / 'run_plan.json')},
              'actual_wire_observations': live.wire_observations(events),
              'source_seconds': result['source_seconds'], 'sent_seconds': result['sent_seconds'], 'actual_audio_seconds': result['actual_audio_seconds'],
              'finalizations': result['finalizations'], 'fin_count': result['fin_count'], 'pending_completions': result['pending_completions'],
              'reference_gaps': gaps, 'reference_words': reference, 'recognized_words': hypothesis, 'word_receipts': word_times,
              'word_diagnostic': diagnostic,
              'provider_readiness': live.provider_readiness(reference, hypothesis, diagnostic, word_times, planned['cut_positions'], events, result['timeline']),
              'native_timing': live.timing_diagnostics(result, responses, sdk),
              'representative_native_attribution_traces': live.attribution_traces(reference, hypothesis, diagnostic, word_times, result, sdk)}
    if live.diagnose.digest(result_path) != before['sha256']:
        raise ValueError('Immutable result changed during diagnosis')
    return output, verification


def recommendation(cases, completed):
    if not completed:
        return ['- Indispensable ablation inputs remain unavailable or incomplete; no quality/readiness or production recommendation is manufactured from partial data.']
    lines = []
    for case_id, case in cases.items():
        contrast = case['contrast']
        coverage = contrast['correct_reference_coverage_difference_right_minus_left']
        error = contrast['conditional_wrong_fraction_difference_right_minus_left']
        post = contrast['post_input_final_word_delay_s']
        left, right = post['D']['median'], post[ARM]['median']
        short = contrast['reference_strata']['short_interjection']
        short_delta = None if short['D']['correct_colored_reference_coverage'] is None or short[ARM]['correct_colored_reference_coverage'] is None else short[ARM]['correct_colored_reference_coverage'] - short['D']['correct_colored_reference_coverage']
        quality = 'improves both conditional native attribution and all-reference correct coverage' if coverage > 0 and error < 0 else 'has a mixed or non-improving native-attribution/reference-coverage result'
        lines.append(f'- **{case_id}**: S200 removal {quality}. Correct all-reference coverage changes {100 * coverage:+.2f} points; short-interjection correct coverage changes ' + ('unavailable' if short_delta is None else f'{100 * short_delta:+.2f} points') + f'. Single-source-occurrence post-input complete-final-word median is {number(left)}s → {number(right)}s. These are bounded final-text/provider-readiness observations, not caption readiness.')
        exposures = contrast['arrival_exposure']
        before = exposures['D']['actual_finalize_write_time']
        after = exposures[ARM]['actual_finalize_write_time']
        lines.append(f'- **{case_id}, readiness tradeoff:** correctly attributed complete-final words already available at actual finalize writes cover {percentage(before["correct_available_reference_coverage"])} → {percentage(after["correct_available_reference_coverage"])} of the same reference windows. Controls can advance after S200 removal, so a lower post-input median does not imply better availability at those seals. These remain provider-arrival exposures, not consumer publication.')
        if short_delta is not None and short_delta < 0:
            before_short = short['D']['counts'].get('correct', 0)
            after_short = short[ARM]['counts'].get('correct', 0)
            unmatched_delta = case['arms'][ARM]['word_diagnostic']['unmatched_reference_words'] - case['arms']['D']['word_diagnostic']['unmatched_reference_words']
            lines.append(f'- **{case_id}, short-word/text caution:** correct short-interjection words fall {before_short} → {after_short} of {short["D"]["reference_words"]}; conditional native error {percentage(short["D"]["conditional_wrong_fraction_known_anchored"])} → {percentage(short[ARM]["conditional_wrong_fraction_known_anchored"])} must not be sold as a short-interjection win. Recognized words change {contrast["recognized_word_count_difference_right_minus_left"]:+d}; all-reference unmatched words change {unmatched_delta:+d}. Correct attribution coverage is not a complete ASR/WER score.')
    witnesses = cases.get('long_return_gap', {}).get('return_witness', {})
    if witnesses:
        old, new = witnesses['D'], witnesses[ARM]
        lines.append(f'- **Returning-speaker outcome:** wrong anchored-word counts are {old["all_reference_word_outcomes"].get("wrong", 0)} → {new["all_reference_word_outcomes"].get("wrong", 0)}. Published initial/return turn outcomes and native-ID/global-mapping witnesses remain explicit above; S200 removal does not erase the observed native returning-speaker failure merely by changing timestamp support.')
    improved = all(case['contrast']['correct_reference_coverage_difference_right_minus_left'] > 0 and case['contrast']['conditional_wrong_fraction_difference_right_minus_left'] < 0 for case in cases.values())
    lines.append('- **Decision supported by these two runs:** ' + ('S200-off is favored over the original D policy on the measured conditional native-error and all-reference correct-coverage outcomes for these two clips; the strata, recognized-word counts and returning-turn witness above bound that recommendation.' if improved else 'do not treat S200-off as a blanket quality/text/readiness improvement; the case-specific and stratum outcomes above are the supported result.'))
    lines.extend(['- S200-only isolation is verified against actual original/new packet hashes, boundaries, source availability, replay, source cuts, provider configuration and controls. Removing paced S200 necessarily also permits earlier controls/input; this is the intended bounded intervention, not a wall-clock-matched pure acoustic estimate.',
                  '- Single new runs against earlier immutable D results have no repeat/significance or independent-holdout claim. Actual processed counters and native timing remain literal; neither changing PCM duration nor transcript matching calibrates the provider clock. No fitted timestamp correction is justified.',
                  '- **No production cutover/readiness certification:** this raw sender independently receives but ignores intermediate ACK gates. It does not exercise production socket reuse, pending-finalization ownership, canonical colors known at publication, acknowledged consumer attachment, or the first-readable caption. Neither disabling S200 in production nor adding a fixed color wait follows from these recordings alone. No further provider calls were made by this diagnostic.'])
    return lines


def main():
    parser = argparse.ArgumentParser(description='Apply unchanged frozen identity/coverage/readiness rules to two immutable S200-only ablation results')
    parser.add_argument('--schema-check-only', action='store_true')
    args = parser.parse_args()
    baseline_path = GAP / 'live_identity_evidence.json'
    baseline = live.diagnose.load(baseline_path)
    frozen = baseline['frozen_helper_sha256']
    frozen_live = baseline['environment']['diagnostic_script_sha256']
    if live.diagnose.digest(GAP / 'diagnose.py') != frozen or live.diagnose.digest(GAP / 'live_diagnose.py') != frozen_live:
        raise ValueError('Frozen identity/readiness helper source changed; do not silently change rules')
    checks = live.schema_self_checks()
    if args.schema_check_only:
        print(live.json.dumps(checks), flush=True)
        return
    preserved = [GAP / name for name in ('live_identity_evidence.json', 'live_identity_report.md', 'offline_evidence.json', 'offline_report.md', 'diagnose.py', 'live_diagnose.py', 'ami_word_reference_inventory.json')]
    hashes = {str(path.relative_to(live.REPO)): live.diagnose.digest(path) for path in preserved}
    inventory = live.diagnose.load(GAP / 'ami_word_reference_inventory.json')
    for row in inventory['files']:
        if row['status'] == 'available' and live.diagnose.digest(GAP / row['path']) != row['sha256']:
            raise ValueError(f'Public reference changed: {row["path"]}')
    plan_path, config_path, scores_path = ROOT / 'run_plan.json', ROOT / 'config.json', ROOT / 'scores.json'
    plan, config = live.diagnose.load(plan_path), live.diagnose.load(config_path)
    if tuple(row['case_id'] for row in plan['rows']) != CASES or config['arm'] != ARM:
        raise ValueError('Ablation scope differs from approved two cases')
    for path, key in ((config_path, 'ablation_config_sha256'), (GAP / 'experiment_config.json', 'base_config_sha256'),
                      (ROOT / 'frozen' / 'ablation_sender.py', 'ablation_runner_sha256'), (ROOT / 'frozen' / 'shared_sender.py', 'base_harness_sha256')):
        if live.diagnose.digest(path) != plan[key]:
            raise ValueError(f'Ablation source/config hash mismatch: {path}')
    score_rows = {}
    if scores_path.is_file():
        scores = live.diagnose.load(scores_path)
        if scores['pyannote.metrics'] != '4.0.0' or scores['collar'] != 0 or scores['overlap_included'] is not True or scores['mapping'] != 'one entire-case optimal speaker mapping':
            raise ValueError('Ablation scores violate declared frozen DER protocol')
        score_rows = {row['id']: row for row in scores['per_arm'][ARM]}
    cases = {}
    for planned in plan['rows']:
        case_id = planned['case_id']
        old_case = baseline['cases'][case_id]
        old = old_case['arms']['D']
        new, verification = diagnose_ablation(old_case['reference_case_manifest'], old, planned, plan, score_rows.get(case_id))
        row = {'reference_case_manifest': old_case['reference_case_manifest'], 'arms': {'D': old, ARM: new},
               's200_only_actual_wire_verification': verification}
        if new['status'] == 'passed':
            row['contrast'] = compare(old, new)
            if case_id == 'long_return_gap':
                row['return_witness'] = {name: return_witness(old_case['reference_case_manifest'], arm) for name, arm in row['arms'].items()}
        cases[case_id] = row
        print(f'case={case_id} arm={ARM} status={new["status"]}', flush=True)
    completed = all(case['arms'][ARM]['status'] == 'passed' for case in cases.values())
    after = {str(path.relative_to(live.REPO)): live.diagnose.digest(path) for path in preserved}
    if after != hashes:
        raise ValueError('Frozen A-D or retained evidence changed during ablation diagnosis')
    evidence = {'status': 'completed' if completed else 'blocked_pending_indispensable_inputs', 'ablation_identity_version': 'S200-only v1',
                'identity_diagnostic_version': baseline['identity_diagnostic_version'], 'frozen_helper_sha256': frozen,
                'frozen_live_helper_sha256': frozen_live, 'schema_self_checks': checks, 'cases': cases,
                'recommendation': recommendation(cases, completed), 'execution_utc': datetime.now(timezone.utc).isoformat(),
                'environment': {'python': sys.version, 'diagnostic_script_sha256': live.diagnose.digest(Path(__file__)), 'execution_argv': sys.argv},
                'preservation': {'before': hashes, 'after': after, 'unchanged': after == hashes},
                'receipts': {'original_live_evidence': live.receipt(baseline_path), 'ablation_plan': live.receipt(plan_path),
                             'ablation_configuration': live.receipt(config_path),
                             'ablation_runner': live.receipt(ROOT / 'frozen' / 'ablation_sender.py'),
                             'shared_sender': live.receipt(ROOT / 'frozen' / 'shared_sender.py'),
                             'scores': live.receipt(scores_path) if scores_path.is_file() else None,
                             'ablation_quota': live.receipt(ROOT / 'quota_ledger.json') if (ROOT / 'quota_ledger.json').is_file() else None}}
    live.diagnose.save(GAP / 'ablation_identity_evidence.json', evidence)
    (GAP / 'ablation_identity_report.md').write_text(render(evidence), encoding='utf-8')
    print(f'status={evidence["status"]} report=ablation_identity_report.md evidence=ablation_identity_evidence.json', flush=True)
    if not completed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
