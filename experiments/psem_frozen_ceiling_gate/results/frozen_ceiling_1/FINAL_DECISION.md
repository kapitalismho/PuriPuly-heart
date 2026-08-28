# FROZEN-CEILING-1 final decision

This report is generated only after the scored artifacts exist. It is development-known path-selection evidence, not production readiness or a fresh holdout.

## Ordered answers

1. Yes as the bounded evaluator floor: G reproduces the shared GT Simple Anchor event ledger across the predeclared confirmation frontier without neural evidence. This does not declare one scalar utility or production readiness.

2. At the predeclared 500 ms / 0.5 diagnostic cell, the action-policy plus neural gap is recorded as G→S-current below; the full frontier remains authoritative.

3. S-probe versus S-current is quantified below at the fixed diagnostic cell; no architecture or threshold was selected on the held-out families.

4. P-C versus S-probe has Pareto counts {'left_dominates': 0, 'right_dominates': 1, 'tradeoff': 17}; the fixed-cell delta is recorded below.

5. P-NC versus P-C has Pareto counts {'left_dominates': 0, 'right_dominates': 0, 'tradeoff': 18}; future context remains diagnostic and its frontier delay includes the future evidence availability.

6. The fixed-cell residual diagnostics are {"anchor_absent_live": {"mean_hazard": 0.5494415374860946, "support_seconds": 1918.21}, "anchor_only": {"mean_hazard": 0.13322748833479311, "support_seconds": 18580.731}, "anchor_overlap": {"mean_hazard": 0.2536533551837219, "support_seconds": 1554.42}} and the topology metrics below retain direct handoff, silence-gap handoff, pause/resume, overlap return, overlap takeover, and short-backchannel slices.

7. No; the posterior result does not yet authorize native S2.

8. opened; causal and bounded non-causal hidden ceilings both failed despite passing train-fit sanity: {'train_fit_sanity': {'H-C': {'status': 'passed', 'reference_probe_class': 'tiny_mlp', 'fold_count': 2, 'minimum_average_precision': 0.8116016984776114, 'minimum_accuracy': 0.9457091999728517}, 'H-NC': {'status': 'passed', 'reference_probe_class': 'tiny_mlp', 'fold_count': 2, 'minimum_average_precision': 0.8047884575857124, 'minimum_accuracy': 0.9253816447157331}}, 'hidden_causal_over_posterior_causal': {'status': 'failed', 'frontier_comparison': {'left_dominates': 4, 'right_dominates': 2, 'tradeoff': 12}, 'reference_cell_improvements': {'exclusive_other_contamination_seconds_per_active_speech_hour': True, 'false_cut_count': True, 'missed_replacement_count': False}, 'source_family_improved_metric_counts': {'alimeeting_far_ch0': 3, 'ami_mix_headset': 1}}, 'hidden_noncausal_over_hidden_causal': {'status': 'failed', 'frontier_comparison': {'left_dominates': 0, 'right_dominates': 8, 'tradeoff': 10}, 'reference_cell_improvements': {'exclusive_other_contamination_seconds_per_active_speech_hour': False, 'false_cut_count': False, 'missed_replacement_count': False}, 'source_family_improved_metric_counts': {'alimeeting_far_ch0': 0, 'ami_mix_headset': 2}}, 'neural_acoustic_failure_concentration': {'per_source_family': {'alimeeting_far_ch0': {'overlap_masking': True, 'competitor_separation': False}, 'ami_mix_headset': {'overlap_masking': False, 'competitor_separation': False}}, 'source_family_domain': {'status': 'passed', 'minimum_improved_metrics': 2, 'improved_metric_counts': {'alimeeting_far_ch0': 3, 'ami_mix_headset': 1}, 'passing_families': ['alimeeting_far_ch0'], 'failing_families': ['ami_mix_headset']}, 'status': 'passed'}}

9. Selected path: D. Sortformer task adaptation / full FT -> KD.

10. Further decoder proliferation and direct scalar-posterior distillation are rejected; task adaptation is now justified.

11. The corrected committed-live-only replay changed contamination seconds per active speech hour from 1885.250 to 2149.283; this is integration hygiene, not teacher selection.

12. Actual persistent-state VAD gating remains deferred until a viable oracle-binding path and native S2 pass. The recorded duty-cycle split supports a later A/B but cannot validate the gated model trajectory.

13. Single next issue: Sortformer task adaptation.

## Nested gap attribution at 500 ms / 0.5

```json
{
  "G_to_H_C_residual": {
    "contamination_seconds_per_active_speech_hour": 2018.0518153975786,
    "false_cut_count": 715,
    "missed_replacement_count": 2236,
    "overlap_return_preservation_rate": 0.23747276688453167,
    "overlap_takeover_success_rate": -0.7535070140280561,
    "replacement_emit_delay_p90_ms": 902.5
  },
  "G_to_P_C_residual": {
    "contamination_seconds_per_active_speech_hour": 2041.9013628131074,
    "false_cut_count": 2568,
    "missed_replacement_count": 2234,
    "overlap_return_preservation_rate": 0.027233115468409674,
    "overlap_takeover_success_rate": -0.7595190380761523,
    "replacement_emit_delay_p90_ms": 902.5
  },
  "G_to_S_current": {
    "contamination_seconds_per_active_speech_hour": 1532.1146010379603,
    "false_cut_count": 418,
    "missed_replacement_count": 1692,
    "overlap_return_preservation_rate": 0.2734204793028323,
    "overlap_takeover_success_rate": -0.7474949899799599,
    "replacement_emit_delay_p90_ms": 910.5
  },
  "H_C_to_H_NC": {
    "contamination_seconds_per_active_speech_hour": 110.18318570729343,
    "false_cut_count": 460,
    "missed_replacement_count": 74,
    "overlap_return_preservation_rate": -0.0816993464052288,
    "overlap_takeover_success_rate": -0.0020040080160320644,
    "replacement_emit_delay_p90_ms": 35.90000000000009
  },
  "P_C_to_H_C": {
    "contamination_seconds_per_active_speech_hour": -23.849547415528832,
    "false_cut_count": -1853,
    "missed_replacement_count": 2,
    "overlap_return_preservation_rate": 0.210239651416122,
    "overlap_takeover_success_rate": 0.006012024048096192,
    "replacement_emit_delay_p90_ms": 0.0
  },
  "P_C_to_P_NC": {
    "contamination_seconds_per_active_speech_hour": -119.40416216629274,
    "false_cut_count": 103,
    "missed_replacement_count": -135,
    "overlap_return_preservation_rate": -0.009803921568627527,
    "overlap_takeover_success_rate": -0.002004008016032064,
    "replacement_emit_delay_p90_ms": 30.0
  },
  "S_current_to_S_probe": {
    "contamination_seconds_per_active_speech_hour": 148.69438828298757,
    "false_cut_count": 793,
    "missed_replacement_count": 195,
    "overlap_return_preservation_rate": -0.02178649237472763,
    "overlap_takeover_success_rate": -0.004008016032064127,
    "replacement_emit_delay_p90_ms": 0.0
  },
  "S_probe_to_P_C": {
    "contamination_seconds_per_active_speech_hour": 361.0923734921596,
    "false_cut_count": 1357,
    "missed_replacement_count": 347,
    "overlap_return_preservation_rate": -0.224400871459695,
    "overlap_takeover_success_rate": -0.008016032064128258,
    "replacement_emit_delay_p90_ms": -8.0
  },
  "hidden_decision_diagnostics": {
    "hidden_causal_over_posterior_causal": {
      "frontier_comparison": {
        "left_dominates": 4,
        "right_dominates": 2,
        "tradeoff": 12
      },
      "reference_cell_improvements": {
        "exclusive_other_contamination_seconds_per_active_speech_hour": true,
        "false_cut_count": true,
        "missed_replacement_count": false
      },
      "source_family_improved_metric_counts": {
        "alimeeting_far_ch0": 3,
        "ami_mix_headset": 1
      },
      "status": "failed"
    },
    "hidden_noncausal_over_hidden_causal": {
      "frontier_comparison": {
        "left_dominates": 0,
        "right_dominates": 8,
        "tradeoff": 10
      },
      "reference_cell_improvements": {
        "exclusive_other_contamination_seconds_per_active_speech_hour": false,
        "false_cut_count": false,
        "missed_replacement_count": false
      },
      "source_family_improved_metric_counts": {
        "alimeeting_far_ch0": 0,
        "ami_mix_headset": 2
      },
      "status": "failed"
    },
    "neural_acoustic_failure_concentration": {
      "per_source_family": {
        "alimeeting_far_ch0": {
          "competitor_separation": false,
          "overlap_masking": true
        },
        "ami_mix_headset": {
          "competitor_separation": false,
          "overlap_masking": false
        }
      },
      "source_family_domain": {
        "failing_families": [
          "ami_mix_headset"
        ],
        "improved_metric_counts": {
          "alimeeting_far_ch0": 3,
          "ami_mix_headset": 1
        },
        "minimum_improved_metrics": 2,
        "passing_families": [
          "alimeeting_far_ch0"
        ],
        "status": "passed"
      },
      "status": "passed"
    },
    "train_fit_sanity": {
      "H-C": {
        "fold_count": 2,
        "minimum_accuracy": 0.9457091999728517,
        "minimum_average_precision": 0.8116016984776114,
        "reference_probe_class": "tiny_mlp",
        "status": "passed"
      },
      "H-NC": {
        "fold_count": 2,
        "minimum_accuracy": 0.9253816447157331,
        "minimum_average_precision": 0.8047884575857124,
        "reference_probe_class": "tiny_mlp",
        "status": "passed"
      }
    }
  },
  "pareto_counts": {
    "P_C_vs_S_probe": {
      "left_dominates": 0,
      "right_dominates": 1,
      "tradeoff": 17
    },
    "P_NC_vs_P_C": {
      "left_dominates": 0,
      "right_dominates": 0,
      "tradeoff": 18
    }
  }
}
```

## Reference product cells

```json
{
  "G": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": 0.0,
      "p90": 0.0
    },
    "diagnostic_slices": {},
    "exclusive_other_contamination_seconds_per_active_speech_hour": 208.0215770137066,
    "false_cut_count": 0,
    "matched_replacement_count": 2798,
    "missed_replacement_count": 0,
    "overlap_return_preservation_rate": 0.6437908496732025,
    "overlap_takeover_success_rate": 0.7635270541082164,
    "predicted_cut_count": 2798,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 500.0,
      "p90": 500.0
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 416.0431540274132,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 128,
        "episodes_with_predicted_cut": 128,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 327,
        "episodes_with_predicted_cut": 327,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.6437908496732025,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 381,
        "episodes_with_predicted_cut": 381,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.7635270541082164
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 946,
        "episodes_with_predicted_cut": 946,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 336,
        "episodes_with_predicted_cut": 336,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 1097,
        "episodes_with_predicted_cut": 1097,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "H-C": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -30.0,
      "p90": 0.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.5494415374860946,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.13322748833479311,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.2536533551837219,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 2226.073392411285,
    "false_cut_count": 715,
    "matched_replacement_count": 562,
    "missed_replacement_count": 2236,
    "overlap_return_preservation_rate": 0.8812636165577342,
    "overlap_takeover_success_rate": 0.01002004008016032,
    "predicted_cut_count": 1277,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1172.5,
      "p90": 1402.5
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 189.88102490815106,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 1,
        "episodes_with_predicted_cut": 35,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 109,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.8812636165577342,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 5,
        "episodes_with_predicted_cut": 158,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.01002004008016032
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 36,
        "episodes_with_predicted_cut": 259,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 21,
        "episodes_with_predicted_cut": 208,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 53,
        "episodes_with_predicted_cut": 386,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "H-NC": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -40.0,
      "p90": 0.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.592816964714313,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.28960190329857916,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.32329508567914594,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 2336.2565781185785,
    "false_cut_count": 1175,
    "matched_replacement_count": 488,
    "missed_replacement_count": 2310,
    "overlap_return_preservation_rate": 0.7995642701525054,
    "overlap_takeover_success_rate": 0.008016032064128256,
    "predicted_cut_count": 1663,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1242.5,
      "p90": 1438.4
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 247.27654222572843,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 54,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 5,
        "episodes_with_predicted_cut": 184,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.7995642701525054,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 4,
        "episodes_with_predicted_cut": 213,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.008016032064128256
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 26,
        "episodes_with_predicted_cut": 519,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 14,
        "episodes_with_predicted_cut": 251,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 39,
        "episodes_with_predicted_cut": 468,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "P-C": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -40.0,
      "p90": 0.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.6959882498187386,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.5673660037974635,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.6421341719006693,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 2249.922939826814,
    "false_cut_count": 2568,
    "matched_replacement_count": 564,
    "missed_replacement_count": 2234,
    "overlap_return_preservation_rate": 0.6710239651416122,
    "overlap_takeover_success_rate": 0.004008016032064128,
    "predicted_cut_count": 3132,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1142.5,
      "p90": 1402.5
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 465.7066327426227,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 2,
        "episodes_with_predicted_cut": 76,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 2,
        "episodes_with_predicted_cut": 302,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.6710239651416122,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 2,
        "episodes_with_predicted_cut": 281,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.004008016032064128
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 39,
        "episodes_with_predicted_cut": 1009,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 19,
        "episodes_with_predicted_cut": 334,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 52,
        "episodes_with_predicted_cut": 646,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "P-NC": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -40.0,
      "p90": -1.7999999999999545
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.9008696285804515,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.57601560665708,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.8507102975002999,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 2130.518777660521,
    "false_cut_count": 2671,
    "matched_replacement_count": 699,
    "missed_replacement_count": 2099,
    "overlap_return_preservation_rate": 0.6612200435729847,
    "overlap_takeover_success_rate": 0.002004008016032064,
    "predicted_cut_count": 3370,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1232.5,
      "p90": 1432.5
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 501.0955786534605,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 0,
        "episodes_with_predicted_cut": 95,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 311,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.6612200435729847,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 1,
        "episodes_with_predicted_cut": 316,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.002004008016032064
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 37,
        "episodes_with_predicted_cut": 1083,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 16,
        "episodes_with_predicted_cut": 366,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 56,
        "episodes_with_predicted_cut": 724,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "S-current": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -30.0,
      "p90": 0.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.88736634360337,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.037065064952851216,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.1298966824169302,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 1740.1361780516668,
    "false_cut_count": 418,
    "matched_replacement_count": 1106,
    "missed_replacement_count": 1692,
    "overlap_return_preservation_rate": 0.9172113289760349,
    "overlap_takeover_success_rate": 0.01603206412825651,
    "predicted_cut_count": 1524,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1186.5,
      "p90": 1410.5
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 226.60820826939877,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 4,
        "episodes_with_predicted_cut": 45,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 7,
        "episodes_with_predicted_cut": 76,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.9172113289760349,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 8,
        "episodes_with_predicted_cut": 151,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.01603206412825651
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 53,
        "episodes_with_predicted_cut": 164,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 24,
        "episodes_with_predicted_cut": 207,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 72,
        "episodes_with_predicted_cut": 598,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "S-probe": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -30.0,
      "p90": 0.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.5664786033758554,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.48632351836142385,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.4987778033760422,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 1888.8305663346543,
    "false_cut_count": 1211,
    "matched_replacement_count": 911,
    "missed_replacement_count": 1887,
    "overlap_return_preservation_rate": 0.8954248366013072,
    "overlap_takeover_success_rate": 0.012024048096192385,
    "predicted_cut_count": 2122,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1182.5,
      "p90": 1410.5
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 315.52665219663004,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 45,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 5,
        "episodes_with_predicted_cut": 96,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.8954248366013072,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 6,
        "episodes_with_predicted_cut": 155,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.012024048096192385
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 47,
        "episodes_with_predicted_cut": 685,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 19,
        "episodes_with_predicted_cut": 296,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 61,
        "episodes_with_predicted_cut": 565,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  }
}
```
