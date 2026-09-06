from __future__ import annotations

import unittest

from experiments.psem_state_corrected_adaptation_gate.lifecycle import (
    AnchorEpisode,
    build_source_authority,
)
from experiments.psem_state_corrected_adaptation_gate.multiplicity import (
    WARMUP_FRAMES,
    build_multiplicity,
)
from experiments.psem_state_corrected_adaptation_gate.partition import (
    SourceExposure,
    assign_train_calib,
    validate_partition_support,
)


def _exposures(per_corpus: int = 20) -> list[SourceExposure]:
    sources: list[SourceExposure] = []
    for i in range(per_corpus):
        sources.append(
            SourceExposure(f"ami-{i:02d}", "AMI", 100.0, positive_frames=5, negative_frames=90)
        )
    for i in range(per_corpus):
        sources.append(
            SourceExposure(f"ali-{i:02d}", "AliMeeting", 100.0, positive_frames=5, negative_frames=90)
        )
    return sources

class PartitionTest(unittest.TestCase):
    def test_disjoint_deterministic_and_bounded(self):
        first = assign_train_calib(_exposures())
        second = assign_train_calib(_exposures())
        self.assertEqual(first, second)
        self.assertFalse(set(first["fit"]) & set(first["calib"]))
        self.assertEqual(len(first["fit"]) + len(first["calib"]), 40)
        for corpus, prefix in (("AMI", "ami-"), ("AliMeeting", "ali-")):
            calib = [s for s in first["calib"] if s.startswith(prefix)]
            self.assertTrue(2 <= len(calib) <= 3, f"{corpus}: {calib}")

    def test_support_and_corpora(self):
        assignment = assign_train_calib(_exposures())
        by_source = {s.source_id: s for s in _exposures()}
        support = validate_partition_support(assignment, by_source)
        self.assertTrue(all(support.values()))
        self.assertTrue(any(s.startswith("ami-") for s in assignment["calib"]))
        self.assertTrue(any(s.startswith("ali-") for s in assignment["calib"]))

    def test_component_disjointness(self):
        components = {f"ami-{i:02d}": f"ami-comp-{i // 2}" for i in range(20)}
        components.update({f"ali-{i:02d}": f"ali-comp-{i // 2}" for i in range(20)})
        assignment = assign_train_calib(_exposures(), components=components)
        for comp_sources in (["ami-00", "ami-01"], ["ali-00", "ali-01"]):
            placed = [
                "calib" if s in assignment["calib"] else "fit" for s in comp_sources
            ]
            self.assertEqual(len(set(placed)), 1)
    def test_lumpy_components_stay_bounded(self):
        sources = [
            SourceExposure("ami-big", "AMI", 790.0, positive_frames=50, negative_frames=700),
            SourceExposure("ami-s0", "AMI", 60.0, positive_frames=5, negative_frames=50),
            SourceExposure("ami-s1", "AMI", 50.0, positive_frames=5, negative_frames=45),
            SourceExposure("ami-s2", "AMI", 100.0, positive_frames=8, negative_frames=90),
            SourceExposure("ali-big", "AliMeeting", 790.0, positive_frames=50, negative_frames=700),
            SourceExposure("ali-s0", "AliMeeting", 60.0, positive_frames=5, negative_frames=50),
            SourceExposure("ali-s1", "AliMeeting", 50.0, positive_frames=5, negative_frames=45),
            SourceExposure("ali-s2", "AliMeeting", 100.0, positive_frames=8, negative_frames=90),
        ]
        first = assign_train_calib(sources)
        self.assertEqual(first, assign_train_calib(sources))
        totals = {"AMI": 1000.0, "AliMeeting": 1000.0}
        by_source = {s.source_id: s for s in sources}
        for corpus in ("AMI", "AliMeeting"):
            calib = [s for s in first["calib"] if by_source[s].corpus == corpus]
            frac = sum(by_source[s].exposure for s in calib) / totals[corpus]
            self.assertTrue(0.10 <= frac <= 0.15, f"{corpus}: {frac}")
        self.assertFalse(set(first["fit"]) & set(first["calib"]))
        self.assertTrue(all(validate_partition_support(first, by_source).values()))

    def test_oversized_single_component_fails_loudly(self):
        sources = [
            SourceExposure("ami-huge", "AMI", 900.0, positive_frames=50, negative_frames=800),
            SourceExposure("ami-tiny", "AMI", 10.0, positive_frames=1, negative_frames=9),
            SourceExposure("ali-huge", "AliMeeting", 900.0, positive_frames=50, negative_frames=800),
            SourceExposure("ali-tiny", "AliMeeting", 10.0, positive_frames=1, negative_frames=9),
        ]
        with self.assertRaises(Exception):
            assign_train_calib(sources)


class LifecycleMultiplicityTest(unittest.TestCase):
    def test_full_source_authority_and_overlap_multiplicity(self):
        num_frames = 750
        active = [
            ("spkA",) if 100 <= f < 400 else (("spkB",) if 400 <= f < 600 else ())
            for f in range(num_frames)
        ]
        authority = build_source_authority(
            "src", num_frames, (AnchorEpisode("ep1", "spkA", 100, 600),), active
        )
        self.assertEqual(authority.y_anchor[150], 1.0)
        self.assertEqual(authority.y_replace[150], 0.0)
        self.assertEqual(authority.y_replace[450], 1.0)
        self.assertTrue(len(authority.ledger["opportunities"]) >= 1)
        crops = [(0.0, 30.0), (15.0, 45.0)]
        mask = build_multiplicity(num_frames, crops, authority.valid)
        self.assertTrue(all(m == 0 for m in mask[:WARMUP_FRAMES]))
        overlap_frame = int(20.0 / 0.08)
        self.assertEqual(mask[overlap_frame], 2)
        warmup_only_frame = 10
        self.assertEqual(mask[warmup_only_frame], 0)


if __name__ == "__main__":
    unittest.main()
