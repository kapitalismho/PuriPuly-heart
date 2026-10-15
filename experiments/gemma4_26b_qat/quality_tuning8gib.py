from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/tuning8gib/clean')
    parser.add_argument('--review', type=Path, nargs='+', default=[Path(__file__).resolve().parent / 'results/tuning8gib/quality-review-initial.json'])
    args = parser.parse_args()
    pairs = json.loads((args.results / 'quality-unique.json').read_text(encoding='utf-8'))
    judgments = {}
    authors = []
    for source in args.review:
        review = json.loads(source.read_text(encoding='utf-8'))
        author = review.get('review_author', 'Independent Director-delegated quality researcher')
        authors.append(author)
        for item in review['per_id']:
            judgments[(item['input'], item['output'])] = {'source_review': source.name, 'author': author, 'original_judgment': item}
    covered, missing = [], []
    for pair in pairs:
        entry = judgments.get((pair['input'], pair['output']))
        if entry is None:
            missing.append(pair)
            continue
        covered.append({'id': pair['id'], 'input': pair['input'], 'output': pair['output'], 'review_author': entry['author'], 'source_review': entry['source_review'], 'manual_judgment_with_original_limits': entry['original_judgment'], 'clean_occurrences': len(pair['occurrences']), 'clean_occurrences_all_eos': all(occurrence['stop_type'] == 'eos' for occurrence in pair['occurrences']), 'clean_occurrence_ledger': pair['occurrences']})
    result = {'scope': 'Bounded independent manual review transferred ONLY on exact source AND output string equality. Clean completion coverage below is recomputed from clean records, never inherited from contaminated occurrence counts.', 'authors': sorted(set(authors)), 'unique_pairs': len(pairs), 'reviewed_unique_pairs': len(covered), 'unreviewed_unique_pairs': missing, 'completed_output_occurrences': sum(len(pair['occurrences']) for pair in pairs), 'reviewed_output_occurrences': sum(pair['clean_occurrences'] for pair in covered), 'per_id': covered, 'finding': 'Use the full per-ID judgments, not an unconditional fidelity pass. Inherited within-five-minutes upper-bound wording and added only-two emphasis remain material disclosed nuances. Exact matching does not establish unseen-output or broad model quality.', 'performance_boundary': 'Manual judgments say nothing about timing, memory, repeats or eligibility. All pre-restart measurements remain excluded.', 'surface_quality': 'Independent reviewer observed no visible truncation, pathological repetition, reasoning trace, translation commentary or instruction leakage in reviewed strings; fresh EOS/automated checks are separate measured evidence.'}
    (args.results / 'quality-review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('unique_pairs', 'reviewed_unique_pairs', 'completed_output_occurrences', 'reviewed_output_occurrences')}))
    if missing or not all(pair['clean_occurrences_all_eos'] for pair in covered):
        raise RuntimeError('Manual review coverage or fresh EOS evidence incomplete; preserve record and obtain missing review')


if __name__ == '__main__':
    main()
