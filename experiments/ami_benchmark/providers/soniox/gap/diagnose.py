from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import wave
from xml.etree import ElementTree as ET

sys.dont_write_bytecode = True
GAP = Path(__file__).resolve().parent
SONIOX = GAP.parent
BENCH = SONIOX.parents[1]
REPO = BENCH.parents[1]
ARMS = ('continuous', 'forced', 'segmented', 'async_full_file')
REVISION = 'a249db53de253a3a0a864c93a51d613159dfec7c'
BASE = f'https://huggingface.co/datasets/ggfox00000/dia-AMICorpus-all/resolve/{REVISION}'
WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
NITE = '{http://nite.sourceforge.net/}'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def words(text):
    return WORD.findall(text.lower().replace('’', "'"))


def command(args, cwd=REPO):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return {'argv': [str(x) for x in args], 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def comparison():
    original = {name: digest(BENCH / name) for name in ('provider_comparison.json', 'provider_comparison.md')}
    with tempfile.TemporaryDirectory(prefix='soniox-retained-') as temporary:
        target = Path(temporary) / 'experiments' / 'ami_benchmark'
        shutil.copytree(BENCH, target, ignore=shutil.ignore_patterns('audio', 'source_audio', 'gap', 'sdk', '__pycache__', '.venv', 'node_modules'))
        (target / '.venv').symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
        invocation = command(['experiments/ami_benchmark/.venv/bin/python', 'experiments/ami_benchmark/compare_providers.py'], Path(temporary))
        generated = load(target / 'provider_comparison.json') if invocation['returncode'] == 0 else None
        same = generated == load(BENCH / 'provider_comparison.json') if generated else False
    after = {name: digest(BENCH / name) for name in original}
    return {'status': 'passed' if invocation['returncode'] == 0 and same and original == after else 'failed',
            'command': invocation, 'sandbox': 'Temporary copy of retained benchmark; exact retained command; no original generator writes',
            'generated_json_semantically_identical': same, 'baseline_artifact_hashes_before': original,
            'baseline_artifact_hashes_after': after, 'comparison': generated}


def fetch_references(cases):
    inventory_path = GAP / 'ami_word_reference_inventory.json'
    existing = load(inventory_path) if inventory_path.exists() else None
    if existing:
        for row in existing['files']:
            if row.get('status') == 'available' and digest(GAP / row['path']) != row['sha256']:
                raise ValueError(f"Reference digest changed: {row['path']}")
        return existing
    meetings = {case['meeting'] for case in cases}
    tasks = []
    for kind in ('words', 'segments'):
        api = f'https://huggingface.co/api/datasets/ggfox00000/dia-AMICorpus-all/tree/{REVISION}/ami_public_manual_1.6.2/{kind}?limit=1000'
        with urlopen(api, timeout=90) as response:
            listing = json.load(response)
        for row in listing:
            name = Path(row['path']).name
            if name.split('.')[0] in meetings:
                tasks.append((row['path'], f'ami_words/{name}'))
    tasks.extend([(name, f'ami_words/{Path(name).name}') for name in ('CCBY4.0.txt', 'ami_public_manual_1.6.2/LICENCE.txt', 'ami_public_manual_1.6.2/00README_MANUAL.txt')])
    (GAP / 'ami_words').mkdir(exist_ok=True)
    def retrieve(task):
        remote, local = task
        url = BASE + '/' + remote
        try:
            with urlopen(url, timeout=90) as response:
                data = response.read()
            (GAP / local).write_bytes(data)
            return {'path': local, 'url': url, 'sha256': sha256(data).hexdigest(), 'bytes': len(data), 'status': 'available'}
        except Exception as error:
            return {'path': local, 'url': url, 'status': 'missing', 'error': f'{type(error).__name__}: {error}'}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(retrieve, tasks))
    inventory = {'repository_revision': REVISION, 'license': 'CC BY 4.0',
                 'official_download_page': 'https://groups.inf.ed.ac.uk/ami/download/',
                 'official_release': 'AMI manual annotations v1.6.2; unchanged annotations since 2014, CC BY 4.0 since 2017',
                 'attribution': 'AMI Consortium; Carletta et al. (2005), The AMI Meeting Corpus: A Pre-Announcement',
                 'independence': 'Published AMI lexical/speaker annotations anchored in original meeting coordinates, not fitted to Soniox native times or token widths. Not newly listener-calibrated waveform/event anchors.',
                 'files': rows}
    save(inventory_path, inventory)
    return inventory


def reference_words(case, inventory):
    available = {row['path'] for row in inventory['files'] if row['status'] == 'available'}
    files = [GAP / path for path in sorted(available) if Path(path).name.startswith(case['meeting'] + '.') and path.endswith('.words.xml')]
    result = []
    gaps = [f"Unavailable public reference: {row['path']} ({row.get('error', 'missing')})" for row in inventory['files'] if row['status'] != 'available' and Path(row['path']).name.startswith(case['meeting'] + '.')]
    for path in files:
        speaker = '.'.join(path.name.split('.')[:2])
        segment_path = path.with_name(path.name.replace('.words.xml', '.segments.xml'))
        if segment_path.relative_to(GAP).as_posix() not in available:
            gaps.append(f'Missing turn reference: {segment_path.name}')
            segments = []
        else:
            segments = []
            for node in ET.parse(segment_path).getroot():
                child = node.find(NITE + 'child')
                ids = re.findall(r'id\(([^)]+)\)', child.get('href', '')) if child is not None else []
                a, b = float(node.get('transcriber_start')), float(node.get('transcriber_end'))
                segments.append((ids, a, b, node.get(NITE + 'id')))
        for node in ET.parse(path).getroot():
            if node.tag != 'w' or not words(node.text or ''):
                continue
            if node.get('starttime') is None or node.get('endtime') is None:
                gaps.append(f"Untimed reference word {node.get(NITE + 'id')}")
                continue
            a, b = float(node.get('starttime')), float(node.get('endtime'))
            offset = case['original_offset_s']
            if not offset <= a < offset + case['duration_s']:
                continue
            wid = node.get(NITE + 'id')
            turn = None
            for ids, left, right, sid in segments:
                if ids and ids[0].rsplit('words', 1)[0] == wid.rsplit('words', 1)[0]:
                    index = int(wid.rsplit('words', 1)[1])
                    if int(ids[0].rsplit('words', 1)[1]) <= index <= int(ids[-1].rsplit('words', 1)[1]):
                        turn = (sid, left - offset, right - offset)
                        break
            for normalized in words(node.text or ''):
                result.append({'word': normalized, 'id': wid, 'speaker': speaker, 'start_s': a - offset, 'end_s': b - offset,
                               'turn_id': turn[0] if turn else None, 'short_interjection': turn[2] - turn[1] <= .7 if turn else None})
    result.sort(key=lambda row: (row['start_s'], row['speaker'], row['id']))
    for row in result:
        row['overlap'] = any(other['speaker'] != row['speaker'] and max(row['start_s'], other['start_s']) < min(row['end_s'], other['end_s']) for other in result)
    if not files:
        gaps.append('No published word references available')
    return result, gaps


def recognized_words(tokens):
    text, spans = '', []
    for index, token in enumerate(tokens):
        start = len(text)
        text += token.get('text', '')
        spans.append((start, len(text), index))
    normalized = text.lower().replace('’', "'")
    result = []
    for match in WORD.finditer(normalized):
        indices = [index for start, end, index in spans if max(start, match.start()) < min(end, match.end())]
        speakers = {str(tokens[index]['speaker']) if tokens[index].get('speaker') not in (None, '') else None for index in indices}
        result.append({'word': match.group(), 'token_indices': indices, 'speaker': next(iter(speakers)) if len(speakers) == 1 else None,
                       'metadata': 'mixed' if len(speakers) > 1 else 'missing' if None in speakers else 'known'})
    return result


def lexical_pairs(reference, hypothesis):
    a, b = [row['word'] for row in reference], [row['word'] for row in hypothesis]
    blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    counts_a = Counter(tuple(a[i:i + 5]) for i in range(max(0, len(a) - 4)))
    counts_b = Counter(tuple(b[i:i + 5]) for i in range(max(0, len(b) - 4)))
    anchors = set()
    matches = []
    for block in blocks:
        for shift in range(max(0, block.size - 4)):
            phrase = tuple(a[block.a + shift:block.a + shift + 5])
            if counts_a[phrase] == counts_b[phrase] == 1:
                anchors.update((block.a + shift + n, block.b + shift + n) for n in range(5))
        matches.extend((block.a + n, block.b + n) for n in range(block.size))
    accepted = set(anchors)
    sorted_anchors = sorted(anchors)
    boundaries = [(-1, -1)] + sorted_anchors + [(len(a), len(b))]
    for (left_a, left_b), (right_a, right_b) in zip(boundaries, boundaries[1:]):
        ca, cb = Counter(a[left_a + 1:right_a]), Counter(b[left_b + 1:right_b])
        for i, j in matches:
            if left_a < i < right_a and left_b < j < right_b and ca[a[i]] == cb[b[j]] == 1:
                accepted.add((i, j))
    return [{'reference_index': i, 'hypothesis_index': j, 'status': 'anchored' if (i, j) in accepted else 'ambiguous'} for i, j in matches]


def identity_diagnostic(reference, hypothesis):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    pairs = lexical_pairs(reference, hypothesis)
    refs = sorted({row['speaker'] for row in reference})
    hyps = sorted({row['speaker'] for row in hypothesis if row['speaker'] is not None})
    counts = np.zeros((len(hyps), len(refs)), dtype=int)
    for pair in pairs:
        if pair['status'] == 'anchored':
            r, h = reference[pair['reference_index']], hypothesis[pair['hypothesis_index']]
            if h['speaker'] is not None:
                counts[hyps.index(h['speaker']), refs.index(r['speaker'])] += 1
    mapping = {}
    if counts.size:
        rows, cols = linear_sum_assignment(-counts)
        mapping = {hyps[i]: refs[j] for i, j in zip(rows, cols) if counts[i, j] > 0}
    for pair in pairs:
        r, h = reference[pair['reference_index']], hypothesis[pair['hypothesis_index']]
        pair['identity'] = 'ambiguous_alignment' if pair['status'] != 'anchored' else h['metadata'] if h['speaker'] is None else 'correct' if mapping.get(h['speaker']) == r['speaker'] else 'wrong'
    by_ref = {pair['reference_index']: pair for pair in pairs}
    strata = {}
    for name, selected in [('all', list(range(len(reference)))), ('overlap', [i for i, row in enumerate(reference) if row['overlap']]), ('nonoverlap', [i for i, row in enumerate(reference) if not row['overlap']]), ('short_interjection', [i for i, row in enumerate(reference) if row['short_interjection'] is True]), ('turn_reference_missing', [i for i, row in enumerate(reference) if row['short_interjection'] is None])]:
        count = Counter(by_ref[i]['identity'] if i in by_ref else 'unmatched_reference' for i in selected)
        known = count['correct'] + count['wrong']
        strata[name] = {'reference_words': len(selected), 'counts': dict(count), 'conditional_wrong_fraction_known_anchored': count['wrong'] / known if known else None,
                        'correct_colored_reference_coverage': count['correct'] / len(selected) if selected else None}
    turns = defaultdict(Counter)
    for i, row in enumerate(reference):
        turns[row['turn_id'] or 'missing_turn'][by_ref[i]['identity'] if i in by_ref else 'unmatched_reference'] += 1
    return {'global_per_case_mapping': mapping, 'mapping_word_contingency': counts.tolist(), 'mapping_reference_labels': refs, 'mapping_native_labels': hyps,
            'reference_words': len(reference), 'recognized_words': len(hypothesis), 'lexically_matched_words': len(pairs),
            'anchored_words': sum(pair['status'] == 'anchored' for pair in pairs), 'ambiguous_words': sum(pair['status'] == 'ambiguous' for pair in pairs),
            'unmatched_reference_words': len(reference) - len(pairs), 'unmatched_recognized_words': len(hypothesis) - len(pairs),
            'strata': strata, 'turns': {key: dict(value) for key, value in turns.items()}, 'pairs': pairs}


def metric_self_checks():
    tokens = [{'text': 'hel', 'speaker': '1', 'start_ms': 1, 'end_ms': 1},
              {'text': 'lo', 'speaker': '1', 'start_ms': 10, 'end_ms': 70},
              {'text': ' again', 'speaker': None, 'start_ms': 100, 'end_ms': 100},
              {'text': ' mix', 'speaker': '1', 'start_ms': 110, 'end_ms': 170},
              {'text': 'ed', 'speaker': '2', 'start_ms': 180, 'end_ms': 240}]
    native = recognized_words(tokens)
    assert [row['word'] for row in native] == ['hello', 'again', 'mixed']
    assert [row['metadata'] for row in native] == ['known', 'missing', 'mixed']
    repeated = [{'word': 'yes'}, {'word': 'yes'}]
    assert all(pair['status'] == 'ambiguous' for pair in lexical_pairs(repeated, repeated))
    text = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf']
    reference = [{'word': word, 'speaker': 'A' if i != 3 else 'B', 'overlap': False, 'short_interjection': False, 'turn_id': str(i)} for i, word in enumerate(text)]
    hypothesis = [{'word': word, 'speaker': '1' if i < 3 else '2' if i == 3 else '3', 'metadata': 'known'} for i, word in enumerate(text)]
    diagnostic = identity_diagnostic(reference, hypothesis)
    assert diagnostic['strata']['all']['counts'] == {'correct': 4, 'wrong': 3}
    assert projected_interval(1000, 1060, [{'provider_start_s': 0, 'provider_end_s': 1.0000000000000002, 'source_start_s': 0}]) == []
    return {'status': 'passed', 'cases': ['split-token lexical conservation', 'zero-duration recognized content', 'missing/mixed metadata', 'repeated-text ambiguity', 'global mapping exposes A-to-B-to-new-ID return split', 'floating source-boundary slivers match retained numerical tolerance'],
            'limit': 'Synthetic metric mechanics only; not provider accuracy, ownership, or realtime application evidence'}


def projected_interval(start_ms, end_ms, timeline):
    output = []
    for index, piece in enumerate(timeline):
        a, b = max(start_ms / 1000, piece['provider_start_s']), min(end_ms / 1000, piece['provider_end_s'])
        if b - a >= 1e-6:
            output.append({'piece_index': index, 'provider_s': [a, b], 'source_s': None if piece['source_start_s'] is None else [piece['source_start_s'] + a - piece['provider_start_s'], piece['source_start_s'] + b - piece['provider_start_s']]})
    return output


def case_evidence(case, reference, gaps):
    with wave.open(str(BENCH / case['audio']), 'rb') as source:
        format_info = {'sample_rate': source.getframerate(), 'channels': source.getnchannels(), 'sample_width': source.getsampwidth(), 'frames': source.getnframes()}
        pcm = source.readframes(source.getnframes())
    identity = {'case': case, 'observed_wav_sha256': digest(BENCH / case['audio']), 'observed_pcm_sha256': sha256(pcm).hexdigest(),
                'observed_source_audio_sha256': digest(BENCH / case['source_audio']), 'format': format_info}
    identity['manifest_identities_verified'] = identity['observed_wav_sha256'] == case['clip_wav_sha256'] and identity['observed_pcm_sha256'] == case['clip_pcm_sha256'] and identity['observed_source_audio_sha256'] == case['source_audio_sha256']
    arms = {}
    for arm in ARMS:
        path = SONIOX / 'results' / case['id'] / f'{arm}.json'
        result = load(path)
        receipt_path = SONIOX / 'predictions' / arm / 'provenance.json'
        receipt = next(row for row in load(receipt_path)['cases'] if row['id'] == case['id'])
        sdk_path = SONIOX / 'sdk_output' / arm / f"{case['id']}.json"
        sdk = load(sdk_path)
        regenerated = subprocess.run(['node', str(SONIOX / 'sdk' / 'segments.mjs')], input=json.dumps(result['tokens']), text=True, capture_output=True, check=True)
        grouped = json.loads(regenerated.stdout)
        stripped = []
        at = 0
        intervals = []
        for group in grouped:
            end = at + len(group['tokens'])
            stripped.append({**{key: value for key, value in group.items() if key != 'tokens'}, 'token_start_index': at, 'token_end_index': end})
            at = end
            if group.get('speaker') not in (None, '') and group.get('start_ms') is not None and group.get('end_ms') is not None:
                for mapped in projected_interval(group['start_ms'], group['end_ms'], result['timeline']):
                    if mapped['source_s']:
                        intervals.append([round(mapped['source_s'][0], 6), round(mapped['source_s'][1], 6), str(group['speaker'])])
        projection_verified = stripped == sdk['sdk_segments'] and sorted(intervals) == sdk['projected_intervals']
        hypothesis = recognized_words(result['tokens'])
        diagnostic = identity_diagnostic(reference, hypothesis) if reference else {'status': 'blocked', 'gaps': gaps}
        sent = bytearray()
        ranges = []
        maps = []
        previous_source_end = None
        for index, piece in enumerate(result['timeline']):
            frames = round((piece['provider_end_s'] - piece['provider_start_s']) * 16000)
            if piece['source_start_s'] is None:
                data = bytes(frames * 2)
                source_samples = None
            else:
                a, b = round(piece['source_start_s'] * 16000), round(piece['source_end_s'] * 16000)
                data = pcm[a * 2:b * 2]
                source_samples = [a, b]
                ranges.append((a, b))
            sent.extend(data)
            maps.append({**piece, 'provider_samples': [round(piece['provider_start_s'] * 16000), round(piece['provider_end_s'] * 16000)],
                         'source_samples': source_samples, 'pcm_sha256': sha256(data).hexdigest(), 'source_discontinuity': source_samples is not None and previous_source_end is not None and source_samples[0] != previous_source_end})
            if source_samples:
                previous_source_end = source_samples[1]
        union = 0
        end = 0
        for a, b in sorted(ranges):
            union += max(0, b - max(a, end))
            end = max(end, b)
        special_tokens = {}
        for name, selected in (
            ('missing_metadata', [i for i, token in enumerate(result['tokens']) if token.get('speaker') in (None, '')]),
            ('zero_duration', [i for i, token in enumerate(result['tokens']) if token.get('start_ms') is not None and token.get('start_ms') == token.get('end_ms')]),
            ('outside_transmitted_support', [i for i, token in enumerate(result['tokens']) if token.get('end_ms', 0) is not None and token.get('end_ms', 0) > result['sent_seconds'] * 1000]),
        ):
            special_tokens[name] = [{'token_index': i, 'native': result['tokens'][i],
                                    'mapped_interval': projected_interval(result['tokens'][i]['start_ms'], result['tokens'][i]['end_ms'], result['timeline'])}
                                   for i in selected[:5] if result['tokens'][i].get('start_ms') is not None and result['tokens'][i].get('end_ms') is not None]
        widest_gap = None
        for group_index, group in enumerate(sdk['sdk_segments']):
            for i in range(group['token_start_index'] + 1, group['token_end_index']):
                left, right = result['tokens'][i - 1], result['tokens'][i]
                if left.get('end_ms') is None or right.get('start_ms') is None:
                    continue
                gap = right['start_ms'] - left['end_ms']
                if gap > 0 and (widest_gap is None or gap > widest_gap['gap_ms']):
                    widest_gap = {'gap_ms': gap, 'sdk_group_index': group_index,
                                  'native_tokens': [{'index': i - 1, 'native': left}, {'index': i, 'native': right}],
                                  'mapped_bridged_interval': projected_interval(left['end_ms'], right['start_ms'], result['timeline']),
                                  'classification': 'SDK same-speaker envelope bridges token gap; not continuous speech evidence'}
        traces = []
        for pair in diagnostic.get('pairs', []):
            if pair['identity'] != 'wrong' or len(traces) >= 5:
                continue
            r, h = reference[pair['reference_index']], hypothesis[pair['hypothesis_index']]
            token_details = [{'index': i, 'native': result['tokens'][i], 'mapped_native_interval': projected_interval(result['tokens'][i].get('start_ms', 0), result['tokens'][i].get('end_ms', 0), result['timeline'])} for i in h['token_indices']]
            group_indices = [i for i, group in enumerate(sdk['sdk_segments']) if any(group['token_start_index'] <= token < group['token_end_index'] for token in h['token_indices'])]
            traces.append({'reference': r, 'recognized': h, 'identity': pair['identity'], 'native_tokens': token_details,
                           'sdk_group_indices': group_indices, 'sdk_groups': [sdk['sdk_segments'][i] for i in group_indices],
                           'available_first_readable': 'not retained; final native list cannot establish arrival or caption attribution'})
        message_keys = sorted({key for row in result.get('messages', []) for key in row})
        arms[arm] = {'native_path': path.relative_to(REPO).as_posix(), 'native_result_sha256': digest(path), 'native_receipt': receipt,
                     'receipt_path': receipt_path.relative_to(REPO).as_posix(), 'receipt_sha256': digest(receipt_path),
                     'sdk_path': sdk_path.relative_to(REPO).as_posix(), 'sdk_sha256': digest(sdk_path), 'sdk_reproduction_verified': projection_verified,
                     'receipt_native_hash_verified': receipt['native_result_sha256'] == digest(path),
                     'config': result['config'], 'model': result['model'], 'lifecycle': result.get('lifecycle'),
                     'segmentation_metadata': {key: value for key, value in result.items() if 'segment' in key and key != 'tokens'},
                     'source_to_provider_sample_maps': maps, 'reconstructed_transmitted_pcm_sha256': sha256(sent).hexdigest(),
                     'reconstructed_transmitted_seconds': len(sent) / 32000, 'recorded_sent_seconds': result['sent_seconds'],
                     'reconstructed_pcm_is_observed_wire_receipt': False,
                     'replayed_source_seconds': (sum(b-a for a,b in ranges) - union) / 16000,
                     'omitted_source_seconds': case['duration_s'] - union / 16000,
                     'padding_seconds': sum(row['provider_end_s'] - row['provider_start_s'] for row in result['timeline'] if row['source_start_s'] is None),
                     'native_token_counts': dict(Counter('missing_speaker' if token.get('speaker') in (None, '') else 'known_speaker' for token in result['tokens'])),
                     'historical_native_timing_diagnostics': result.get('conversion'), 'sdk_projection_diagnostics': sdk['projection_diagnostics'],
                     'retained_message_fields': message_keys, 'processed_audio_counters_retained': [row for row in result.get('messages', []) if any('processed' in key for key in row)],
                     'word_diagnostic': diagnostic, 'recognized_words': hypothesis, 'representative_wrong_native_attribution': traces,
                     'representative_native_metadata_timing_events': special_tokens, 'representative_projection_bridge': widest_gap,
                     'reconstructed_sent_duration_verified': abs(len(sent) / 32000 - result['sent_seconds']) < 1e-5}
    drift = {}
    continuous = arms['continuous']['recognized_words']
    forced = arms['forced']['recognized_words']
    native_a = load(SONIOX / 'results' / case['id'] / 'continuous.json')['tokens']
    native_b = load(SONIOX / 'results' / case['id'] / 'forced.json')['tokens']
    import statistics
    drift_pairs = lexical_pairs(continuous, forced)
    for index in range(4):
        offsets = []
        examples = []
        for pair in drift_pairs:
            if pair['status'] != 'anchored':
                continue
            a = continuous[pair['reference_index']]
            b = forced[pair['hypothesis_index']]
            left = native_a[a['token_indices'][0]].get('start_ms')
            right = native_b[b['token_indices'][0]].get('start_ms')
            if left is not None and right is not None and index * case['duration_s'] / 4 <= left / 1000 < (index + 1) * case['duration_s'] / 4:
                offsets.append(right - left)
                if len(examples) < 2:
                    examples.append({'word': a['word'], 'continuous_token_indices': a['token_indices'], 'forced_token_indices': b['token_indices'], 'continuous_start_ms': left, 'forced_start_ms': right})
        drift[str(index)] = {'continuous_native_time_bucket_s': [index * case['duration_s'] / 4, (index + 1) * case['duration_s'] / 4], 'matched_words': len(offsets), 'median_forced_minus_continuous_ms': statistics.median(offsets) if offsets else None, 'examples': examples}
    return {'input_identity': identity, 'reference_words': reference, 'reference_gaps': gaps, 'arms': arms,
            'retained_clock_comparison': {'status': 'observed_unresolved', 'method': 'Conservative exact lexical anchors between independent retained transcripts, first contributing token start, four source-duration quartiles by continuous native time; not waveform clock calibration', 'buckets': drift}}


def report(evidence):
    lines = ['# #204 retained offline diagnosis', '', '## Disposition', '',
             '**Defer causal/candidate adoption.** Retained corrected DER and independent lexical identity diagnostics are reproducible offline. They do not isolate finalization from PCM/ACK policy, calibrate native clocks, or establish first-readable usefulness. Live factorial and consumption evidence belong to separate owners. No production edits, provider calls, guessed timestamp correction, reference-informed clipping, or native 60-ms identity support were used.', '',
             '## Execution and preservation', '', f"Execution revision: `{evidence['environment']['revision']}`. Original baseline: `4dc46fdac8c771d3bb58e46bd44304b86fde9c47`.",
             f"Runtime: `{evidence['environment']['python']}`; `{evidence['environment']['platform']}`; Node `{evidence['environment']['node_version']}`; pyannote.metrics `{evidence['environment']['scorer_version']}`.",
             'Archived inference: stt-rt-v5 at `wss://stt-rt.soniox.com/transcribe-websocket`, en hint, diarization enabled, language identification and endpoint detection disabled, pcm_s16le / 16 kHz / mono, one WS per case; async stt-async-v5 at `https://api.soniox.com/v1`. Those archived settings are not asserted to match the current application.',
             'Archived sender: 800-sample (50-ms) PCM chunks with post-write sleep; forced/segmented wait for each requested fin acknowledgement before successor PCM. Padding is a separately sent all-zero S200 write without equivalent sleep. Archived segmented policy uses Silero-v6.2.1, 500-ms ring/pre-roll/hangover, onset 0.3 / derived exit 0.2, SmartTurn threshold 0.75, delivery profile on; source-language ownership snapshot zh-CN / expected zh-CN,ja differs from provider en hint. All frozen segments/source pieces are retained; no labels enter segmentation.',
             'The exact retained comparison command ran inside a temporary copy with the pinned scoring venv symlinked; the generator never wrote original baseline artifacts. Generated comparison JSON was checked semantically against retained JSON, and original comparison JSON/Markdown hashes were unchanged.', '',
             '```text', 'experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/diagnose.py --fetch-word-references', 'experiments/ami_benchmark/.venv/bin/python experiments/ami_benchmark/providers/soniox/gap/diagnose.py', '```', '',
             'All exact subprocess argv/stdout/stderr, package versions, manifest/SDK/source identities, native receipt identities, source-to-provider sample maps, and alignment rows are in [offline_evidence.json](offline_evidence.json). Public annotation acquisition URLs/hashes and licensing are in [ami_word_reference_inventory.json](ami_word_reference_inventory.json).', '',
             '## Checks', '', '| Check | Result |', '|---|---|']
    lines.extend(f"| {key} | {value} |" for key, value in evidence['checks'].items())
    lines.extend(['', 'Failed during local harness development (repaired, not provider failures): the acquisition run completed all case diagnostics but report rendering hit a distribution-name KeyError; the next run exposed diagnostic-only floating-boundary slivers in SDK verification. The final harness uses importlib.metadata.version and the existing projection numerical tolerance (<1 microsecond), without changing any baseline output or improving provider scores.', ''])
    lines.extend(['', '## Corrected declared-output DER', '', 'Full-clip UEM; zero collar; overlap included; pyannote.metrics 4.0.0; one optimal global mapping per case. SDK 2.3.0 speaker grouping happens before source mapping. Existing silence bridges and false alarms are retained. These are not word/person error percentages.', '', '| Arm | Six-primary DER % | Miss s | False alarm s | Confusion s | Far-field DER % |', '|---|---:|---:|---:|---:|---:|'])
    for run in evidence['comparison']['comparison']['runs']:
        if run['provider'] != 'Soniox':
            continue
        s, f = run['score']['independent_six'], run['score']['paired_far_field']
        lines.append(f"| {run['arm']} | {s['der']*100:.2f} | {s['missed detection']:.3f} | {s['false alarm']:.3f} | {s['confusion']:.3f} | {f['der']*100:.2f} |")
    lines.extend(['', '| Case | Arm | DER % | Miss s | False alarm s | Confusion s | Predicted / reference speakers |', '|---|---|---:|---:|---:|---:|---:|'])
    for run in evidence['comparison']['comparison']['runs']:
        if run['provider'] == 'Soniox':
            for row in run['score']['per_case']:
                lines.append(f"| {row['id']} | {run['arm']} | {row['der']*100:.2f} | {row['missed detection']:.3f} | {row['false alarm']:.3f} | {row['confusion']:.3f} | {row['predicted_speakers']} / {row['reference_speakers']} |")
    lines.extend(['', 'Far-field pair is dependent on overlap_heavy (same meeting/window, different microphone) and is never pooled into six-primary results. Six primary windows are exposed annotation-selected diagnostics, not an independent participant sample or holdout.', '', '## Independent reference-aligned native identity diagnostic', '',
                  'Published AMI v1.6.2 word XML provides lexical content, speaker identity, and original-meeting time; segment XML provides independent turn membership. Inputs are pinned to the existing audio mirror revision and CC BY 4.0. These annotations are independent of Soniox outputs, but not new manually audited acoustic/event anchors. References never enter inference.',
                  'Rules: lowercase ASCII alphanumeric/apostrophe lexical words, punctuation removed; native token text concatenated before lexicalization, so subword tokens are conserved. A word with conflicting contributing native speakers is mixed; absent metadata is missing. Reference words are sorted by published start, speaker, and ID; overlap linearization can misalign concurrent words and is a limitation, not fixed with native timing.',
                  'SequenceMatcher(auto junk off) yields monotone exact lexical blocks. A match is anchored only by a unique shared contiguous five-word phrase, or by a word occurring exactly once in each gap between those anchors (including clip-edge gaps). All other exact matches are ambiguous; nonmatches remain explicit. No token widths or timing shifts determine identity. One maximum-count one-to-one label assignment for the complete case/arm is fitted on anchored known-speaker words; it is not re-fitted per turn. This conditional diagnostic is not full ASR accuracy or realtime color accuracy.',
                  'Overlap means any positive reference-word intersection with a different speaker; short interjection means published full turn duration <=0.7 s, not native token duration. Separate denominators and per-turn counts are retained, including unknown-turn membership.', '',
                  '| Case | Arm | Ref / recognized words | Anchored / ambiguous | Unmatched ref / recognized | Wrong % known anchored | Correct reference coverage % |', '|---|---|---:|---:|---:|---:|---:|'])
    for cid, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            d = result['word_diagnostic']
            if 'strata' not in d:
                lines.append(f"| {cid} | {arm} | blocked: word references unavailable | - | - | - | - |")
                continue
            s = d['strata']['all']
            wrong = s['conditional_wrong_fraction_known_anchored']
            coverage = s['correct_colored_reference_coverage']
            wrong_text = f'{wrong*100:.2f}' if wrong is not None else 'unavailable'
            coverage_text = f'{coverage*100:.2f}' if coverage is not None else 'unavailable'
            lines.append(f"| {cid} | {arm} | {d['reference_words']} / {d['recognized_words']} | {d['anchored_words']} / {d['ambiguous_words']} | {d['unmatched_reference_words']} / {d['unmatched_recognized_words']} | {wrong_text} | {coverage_text} |")
    lines.extend(['', '### Diagnostic interpretation', '',
                  'The long_context native conditional wrong-speaker fraction is only 1.83% continuous, 1.43% forced, and 1.01% segmented among conservatively anchored known-speaker words, while corrected DER is 42.58%, 77.28%, and 74.18%. Its correct reference-word coverage remains 68.90%, 67.91%, and 67.18%. Thus the large DER gap does not establish a large native recognized-word identity loss in this case; timing/support/projection and uncovered words remain distinct factors, not a proven unique cause.',
                  'Overlap_heavy segmented has lower conditional identity error (19.93%) than continuous (41.10%) despite worse DER (77.23% vs 70.90%); independently linearized overlap and uncovered/ambiguous words limit interpretation. Brief_interjections segmented has worse conditional identity (41.50% vs 32.62%) and lower correct reference coverage (32.53% vs 41.32%). These exposed-set observations rule out treating the pooled DER change as a uniform voice-identity gain/loss.',
                  'Representative metadata, zero-duration, overrun and largest envelope-bridge events are retained beside each arm, including exact native token indices and source-piece mappings. Synthetic metric self-checks are separate from provider evidence.',
                  '', '### Explicit difficult strata', '', '| Case | Arm | Stratum | Reference denominator | Outcome counts |', '|---|---|---|---:|---|'])
    for cid, case in evidence['cases'].items():
        for arm, result in case['arms'].items():
            if 'strata' not in result['word_diagnostic']:
                continue
            for name in ('overlap', 'short_interjection', 'turn_reference_missing'):
                s = result['word_diagnostic']['strata'][name]
                lines.append(f"| {cid} | {arm} | {name} | {s['reference_words']} | `{json.dumps(s['counts'], sort_keys=True)}` |")
    lines.extend(['', '## Trace and classification findings', '',
                  '- Native attribution: representative lexically anchored wrong-speaker words retain public reference IDs, native token indices/content/speaker/time, contributing SDK groups, and mapped source-piece intersections in `cases.CASE.arms.ARM.representative_wrong_native_attribution`. The native disagreement is visible independently of 60-ms support; mappings are global. Missing/ambiguous words are not silently scored correct.',
                  '- Native content/metadata: all reference and recognized word counts, unmatched words, mixed/missing contributing metadata, and ambiguous alignments are explicit. Unmatched lexical words are not definitive ASR omissions: transcription conventions, overlap order, contractions, and segmentation can also break exact matching. Native missing speaker counts are separate from timing omissions.',
                  '- Projection: SDK groups and source-projected intervals are regenerated using the pinned actual SDK and compared to retained artifacts. Large same-speaker envelopes bridge silence; false alarm cost is preserved. Native zero-duration tokens remain recognized lexical evidence where text exists, without positive diarization support. Historical embedded conversion metadata describes old pulse projection, not the corrected score.',
                  '- PCM/source policy: the artifact retains sample-coordinate maps and reconstructs PCM identities from source pieces plus all-zero padding. Replay, skipped support, discontinuities and padding are counted separately. These are reconstructed from retained sender maps, **not observed per-write network receipts**. Original app-cut segment metadata is retained; the offline script does not regenerate app segmentation or claim current integrated parity.',
                  '- Normalization/deadline: final native receipts do not retain first-readable publication, native-token arrival timestamps, consumer scope or text conservation. No caption attribution or live-ready claim follows from this diagnosis. The independent application-consumption owner handles those fixtures/traces.', '', '## R4 retained timestamp observations — unresolved, no correction', '',
                  'Exact text anchors between continuous and forced final transcripts compare first contributing native token starts, grouped by continuous-native-time quartile. This supporting text comparison is separate from identity scoring, with no fitted subtraction. Native processed-audio counters were not retained in the simplified message envelopes. Actual PCM/control/receive write timestamps are absent. A native clock, sender behavior, matching ambiguity and source/output conversion cannot be uniquely distinguished from these receipts.', '',
                  '| Case | Forced finalizations | Max native overrun ms (historical diagnostic) | Quartile matched-word median offsets ms |', '|---|---:|---:|---|'])
    for cid, case in evidence['cases'].items():
        forced = case['arms']['forced']
        offsets = [bucket['median_forced_minus_continuous_ms'] for bucket in case['retained_clock_comparison']['buckets'].values()]
        lines.append(f"| {cid} | {forced['lifecycle']['finalize_requested']} | {forced['historical_native_timing_diagnostics']['max_overrun_ms']} | `{offsets}` |")
    lines.extend(['', 'These progressive offsets and overrun values reproduce supporting retained observations; they do not prove 120 ms of synthetic PCM per finalize. No per-finalize subtraction, reference-fitted calibration, silence clipping, or correction was applied. Published AMI lexical times were not promoted to independently verified waveform clock anchors. A small licensed/consented event-anchor fixture, exact PCM/control receipts and actual processed-audio counters (where returned) remain necessary for causal calibration and separately frozen correction validation.', '',
                  '## Limits and documentation proposals for Director', '',
                  '- Recommend documenting corrected DER and the conditional lexical identity diagnostic together, with unmatched/ambiguous/overlap/interjection denominators rather than calling DER voice-ID accuracy.',
                  '- Record the timestamp finding as bounded unresolved: retained messages omit counters/receive times; no correction is supported. Preserve literal native timing.',
                  '- Record this execution as replacement WSL offline scoring/SDK environment, not the historical Windows provider runtime. Retained Python/provider SDK and model server versions cannot be retroactively recovered; current model names/config and pinned projection/scorer versions are retained.',
                  '- Current production endpoint/config parity is not established by archived endpoint-off inference. No historical endpoint behavior was restored. This task did not exercise physical microphone, VRChat/HMD, Korean/Japanese or billing.',
                  '- No new Soniox requests or private audio/credentials were consumed by this script. Public input acquisition was only pinned AMI XML/license metadata. Factorial live-run quota and candidate readiness evidence are outside this retained-only assignment.',
                  '- Maintainer decisions still concern causal live results and supported quality/latency/cost tradeoffs; this offline evidence does not authorize production adoption. No supported negative model-quality conclusion can replace an indispensable missing controlled comparison.', ''])
    lines.extend(['', '## Public reference availability', ''])
    missing = [(cid, gap) for cid, case in evidence['cases'].items() for gap in case['reference_gaps']]
    if missing:
        lines.extend(f'- {cid}: {gap}' for cid, gap in missing)
    else:
        lines.append('No missing published word/turn reference inputs or untimed selected lexical reference words were observed. This does not certify the acoustic calibration or correctness of the published reference timings.')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Reproduce and diagnose retained Soniox outputs without provider calls or baseline writes')
    parser.add_argument('--fetch-word-references', action='store_true')
    args = parser.parse_args()
    reproduced = comparison()
    if reproduced['status'] != 'passed':
        save(GAP / 'offline_evidence.json', {'comparison': reproduced, 'status': 'failed'})
        raise RuntimeError('Corrected baseline comparison failed; see offline_evidence.json')
    manifest = load(BENCH / 'manifest.json')
    inventory = fetch_references(manifest['cases']) if args.fetch_word_references else load(GAP / 'ami_word_reference_inventory.json')
    if not args.fetch_word_references:
        for row in inventory['files']:
            if row['status'] == 'available' and digest(GAP / row['path']) != row['sha256']:
                raise ValueError(f"Pinned reference digest changed: {row['path']}")
    revision_probe = command(['git', 'rev-parse', 'HEAD'])
    revision_probes = [revision_probe]
    if revision_probe['returncode'] != 0 and shutil.which('git.exe'):
        revision_probe = command(['git.exe', 'rev-parse', 'HEAD'])
        revision_probes.append(revision_probe)
    if revision_probe['returncode'] != 0:
        raise RuntimeError(f"Execution revision unavailable: {revision_probe['stderr']}")
    metric_checks = metric_self_checks()
    environment = {'revision': revision_probe['stdout'].strip(), 'revision_commands': revision_probes, 'python': sys.version,
                   'executable': str(Path(sys.executable).relative_to(REPO)), 'platform': platform.platform(),
                   'node_version': command(['node', '--version'])['stdout'].strip(),
                   'packages': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()},
                   'scorer_version': importlib.metadata.version('pyannote.metrics'),
                   'execution_argv': [str(Path(sys.executable).relative_to(REPO)), *sys.argv],
                   'diagnosis_script_sha256': digest(Path(__file__)),
                   'sdk_package_lock_sha256': digest(SONIOX / 'sdk' / 'package-lock.json'), 'sdk_bridge_sha256': digest(SONIOX / 'sdk' / 'segments.mjs'),
                   'sdk_package': load(SONIOX / 'sdk' / 'node_modules' / '@soniox' / 'node' / 'package.json'),
                   'manifest_sha256': digest(BENCH / 'manifest.json'), 'scorer_sha256': digest(BENCH / 'score.py'),
                   'archived_endpoint': 'wss://stt-rt.soniox.com/transcribe-websocket', 'async_endpoint': 'https://api.soniox.com/v1',
                   'archived_sender_sha256': digest(REPO / 'experiments' / 'soniox_diarization' / 'probe.py'),
                   'archived_chunk_samples': 800,
                   'archived_pacing_receive_policy': 'Post-write 50-ms/source-sized sleep; forced/segmented block successor PCM until corresponding fin acknowledgement; S200 padding written without pacing sleep; one connection per case',
                   'archived_segmentation_configuration': {'vad_model': 'silero-v6.2.1', 'ring_buffer_ms': 500, 'onset_threshold': 0.3, 'exit_threshold': 0.2, 'hangover_ms': 500, 'pre_roll_ms': 500, 'smart_turn_complete_threshold': 0.75, 'delivery_profile': 'on', 'source_language_snapshot': 'zh-CN', 'expected_languages_snapshot': ['zh-CN', 'ja'], 'padding_policy': 'S200 for delivery_deadline or delivery_pause with 4000 <= owned content_ms < 7000'},
                   'archived_segmentation_sha256': digest(REPO / 'experiments' / 'soniox_diarization' / 'app_segments.py'),
                   'original_provider_runtime_versions': 'not retained; current offline environment is not evidence of historical inference package/server versions'}
    cases = {}
    for case in manifest['cases']:
        reference, gaps = reference_words(case, inventory)
        cases[case['id']] = case_evidence(case, reference, gaps)
        print(f"case={case['id']} reference_words={len(reference)} arms=4", flush=True)
    all_arms = [arm for case in cases.values() for arm in case['arms'].values()]
    checks = {'Corrected baseline offline command': reproduced['status'],
              'Original generator output preservation': 'passed',
              'Seven clip WAV/PCM/full-source identities': 'passed' if all(case['input_identity']['manifest_identities_verified'] for case in cases.values()) else 'failed',
              '28 native receipt identities': 'passed' if all(arm['receipt_native_hash_verified'] for arm in all_arms) else 'failed',
              '28 actual SDK group/source projection reproductions': 'passed' if all(arm['sdk_reproduction_verified'] for arm in all_arms) else 'failed',
              'Reconstructed PCM duration matches retained support': 'passed' if all(arm['reconstructed_sent_duration_verified'] for arm in all_arms) else 'failed',
              'Synthetic diagnostic mechanics only': metric_checks['status'],
              'Published independent lexical/turn reference diagnostic': 'passed with declared alignment limits' if all(case['reference_words'] for case in cases.values()) else 'blocked: missing word references',
              'Independently verified waveform clock calibration/correction': 'blocked: no retained event calibration/counters; no guessed correction',
              'Live controlled factorial comparison': 'not run by retained-only owner; separate owner',
              'Application normalization/first-readable consumption': 'not run by retained-only owner; separate owner',
              'Current integrated app endpoint parity': 'not run; no archived-parity claim',
              'Production enablement/VRChat/HMD and Korean/Japanese': 'not run'}
    evidence = {'environment': environment, 'comparison': reproduced, 'word_reference_inventory_sha256': digest(GAP / 'ami_word_reference_inventory.json'),
                'identity_diagnostic_version': 'independent-lexical-anchor-v1', 'checks': checks, 'metric_self_checks': metric_checks, 'cases': cases}
    save(GAP / 'offline_evidence.json', evidence)
    (GAP / 'offline_report.md').write_text(report(evidence), encoding='utf-8')
    print(json.dumps({'checks': checks, 'evidence': 'offline_evidence.json', 'report': 'offline_report.md'}), flush=True)
    if any(value == 'failed' for value in checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
