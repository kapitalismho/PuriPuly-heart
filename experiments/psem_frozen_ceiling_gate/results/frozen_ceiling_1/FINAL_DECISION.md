# FROZEN-CEILING-1 final decision

This report is generated only after the scored artifacts exist. It is development-known path-selection evidence, not production readiness or a fresh holdout.

## Ordered answers

1. Yes as the bounded evaluator floor: every predeclared G arm is simulated independently from exact-source-time GT activity, while the 500 ms arm exactly reproduces the sealed issue-98 Simple Anchor reference. This does not declare one scalar utility or production readiness.

2. The G action-policy tradeoff is explicit: moving from the sealed 500 ms arm to the 100 ms arm changes contamination by +242.689 seconds/hour, false cuts by +2448, misses by +385, and p90 delay by -400.0 ms. At the same sealed policy point, S-current remains worse than G by +1532.115 seconds/hour, +418 false cuts, and +1692 misses. The frontier therefore identifies a latency/cut tradeoff, while the much larger fixed-policy residual is neural-evidence loss rather than the 500 ms action policy alone.

3. The flexible scalar S-probe changes contamination by -6.028 seconds/hour, false cuts by +110, misses by -13, and p90 delay by +0.0 ms at the fixed cell. Its training-only sanity gate did not pass (minimum AP 0.231, minimum accuracy 0.774), so these held-out tradeoffs are descriptive and do not establish the scalar projection ceiling; no held-out architecture or threshold was selected.

4. Full-slot P-C also does not close additional gap over S-probe at the fixed cell: contamination changes by +647.297 seconds/hour, false cuts by -202, misses by +672, with frontier counts {'left_dominates': 0, 'right_dominates': 0, 'tradeoff': 18}. The causal full-slot evidence is therefore not a viable frozen-posterior ceiling under this probe contract.

5. Bounded P-NC is a tradeoff rather than a material overall win over P-C: contamination changes by -206.188 seconds/hour and misses by -201, but false cuts change by +4 and p90 delay by +20.0 ms; frontier counts are {'left_dominates': 0, 'right_dominates': 1, 'tradeoff': 17}. Future context remains diagnostic and does not establish a streaming/context-only path.

6. Direct H-C-to-G residuals are source-family concentrated in ['ami_mix_headset'], with worst-metric counts {'alimeeting_far_ch0': 1, 'ami_mix_headset': 3} across the four frozen residual fields. The acoustic checks additionally find overlap masking is detected in ['alimeeting_far_ch0'] and competitor separation in ['none']. The rule measures hidden-to-G product residuals directly, so the evidence localizes failure to overlap/source-family conditions rather than hidden-over-posterior improvement; device/noise is represented only through the frozen corpus-device families.

7. No; the posterior result does not yet authorize native S2.

8. opened; causal and bounded non-causal hidden ceilings both failed despite passing train-fit sanity: {'train_fit_sanity': {'H-C': {'status': 'passed', 'reference_probe_class': 'tiny_mlp', 'fold_count': 2, 'minimum_average_precision': 0.8116016984776114, 'minimum_accuracy': 0.9457091999728517}, 'H-NC': {'status': 'passed', 'reference_probe_class': 'tiny_mlp', 'fold_count': 2, 'minimum_average_precision': 0.8047884575857124, 'minimum_accuracy': 0.9253816447157331}}, 'hidden_causal_over_posterior_causal': {'status': 'failed', 'frontier_comparison': {'left_dominates': 0, 'right_dominates': 2, 'tradeoff': 16}, 'reference_cell_improvements': {'exclusive_other_contamination_seconds_per_active_speech_hour': True, 'false_cut_count': False, 'missed_replacement_count': True}, 'source_family_improved_metric_counts': {'alimeeting_far_ch0': 3, 'ami_mix_headset': 2}}, 'hidden_noncausal_over_hidden_causal': {'status': 'failed', 'frontier_comparison': {'left_dominates': 0, 'right_dominates': 8, 'tradeoff': 10}, 'reference_cell_improvements': {'exclusive_other_contamination_seconds_per_active_speech_hour': False, 'false_cut_count': False, 'missed_replacement_count': False}, 'source_family_improved_metric_counts': {'alimeeting_far_ch0': 0, 'ami_mix_headset': 2}}, 'neural_acoustic_failure_concentration': {'per_source_family': {'alimeeting_far_ch0': {'overlap_masking': True, 'competitor_separation': False, 'overlap_takeover_deficit': True}, 'ami_mix_headset': {'overlap_masking': False, 'competitor_separation': False, 'overlap_takeover_deficit': True}}, 'source_family_domain': {'status': 'passed', 'minimum_worst_residual_metrics': 3, 'direct_hidden_to_gt_residuals': {'alimeeting_far_ch0': {'contamination_seconds_per_active_speech_hour_gap': 1678.4448786702362, 'false_cut_count_per_active_speech_hour_gap': 91.41696292534282, 'missed_replacement_rate_gap': 0.6488919667590027, 'overlap_takeover_success_rate_gap': 0.7608695652173914}, 'ami_mix_headset': {'contamination_seconds_per_active_speech_hour_gap': 2331.3196202908566, 'false_cut_count_per_active_speech_hour_gap': 120.0586000309675, 'missed_replacement_rate_gap': 0.9593796159527327, 'overlap_takeover_success_rate_gap': 0.7443946188340808}}, 'worst_families_by_metric': {'contamination_seconds_per_active_speech_hour_gap': ['ami_mix_headset'], 'false_cut_count_per_active_speech_hour_gap': ['ami_mix_headset'], 'missed_replacement_rate_gap': ['ami_mix_headset'], 'overlap_takeover_success_rate_gap': ['alimeeting_far_ch0']}, 'worst_metric_counts': {'alimeeting_far_ch0': 1, 'ami_mix_headset': 3}, 'concentrated_families': ['ami_mix_headset']}, 'status': 'passed'}}

9. Selected path: D. Sortformer task adaptation / full FT -> KD.

10. Further decoder proliferation and direct scalar-posterior distillation are rejected; task adaptation is now justified.

11. No: excluding pre-roll from confirmation did not reverse or explain away the issue-98 VAD sensitivity; contamination increased from 1885.250 to 2149.283 seconds per active speech hour. The corrected replay changes the numeric estimate but preserves the qualitative interpretation that a later persistent-state gating A/B is required; it remains integration hygiene, not teacher selection.

12. Actual persistent-state VAD gating remains deferred until a viable oracle-binding path and native S2 pass. The recorded duty-cycle split supports a later A/B but cannot validate the gated model trajectory.

13. Single next issue: Sortformer task adaptation.

## Nested gap attribution at 500 ms / 0.5

```json
{
  "G_action_frontier_fast_to_reference": {
    "contamination_seconds_per_active_speech_hour": 242.68891555267348,
    "false_cut_count": 2448,
    "missed_replacement_count": 385,
    "overlap_return_preservation_rate": -0.05991285403050095,
    "overlap_takeover_success_rate": -0.018036072144288595,
    "replacement_emit_delay_p90_ms": -400.0
  },
  "G_to_H_C_residual": {
    "contamination_seconds_per_active_speech_hour": 2018.0518153975786,
    "false_cut_count": 715,
    "missed_replacement_count": 2236,
    "overlap_return_preservation_rate": 0.23747276688453167,
    "overlap_takeover_success_rate": -0.7535070140280561,
    "replacement_emit_delay_p90_ms": 902.5
  },
  "G_to_P_C_residual": {
    "contamination_seconds_per_active_speech_hour": 2173.384010127649,
    "false_cut_count": 326,
    "missed_replacement_count": 2351,
    "overlap_return_preservation_rate": 0.2919389978213508,
    "overlap_takeover_success_rate": -0.7575150300601202,
    "replacement_emit_delay_p90_ms": 912.5
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
    "contamination_seconds_per_active_speech_hour": -155.33219473007057,
    "false_cut_count": 389,
    "missed_replacement_count": -115,
    "overlap_return_preservation_rate": -0.05446623093681913,
    "overlap_takeover_success_rate": 0.004008016032064128,
    "replacement_emit_delay_p90_ms": -10.0
  },
  "P_C_to_P_NC": {
    "contamination_seconds_per_active_speech_hour": -206.18834039969533,
    "false_cut_count": 4,
    "missed_replacement_count": -201,
    "overlap_return_preservation_rate": 0.004357298474945592,
    "overlap_takeover_success_rate": 0.0,
    "replacement_emit_delay_p90_ms": 20.0
  },
  "S_current_to_S_probe": {
    "contamination_seconds_per_active_speech_hour": -6.027867555796092,
    "false_cut_count": 110,
    "missed_replacement_count": -13,
    "overlap_return_preservation_rate": -0.013071895424836555,
    "overlap_takeover_success_rate": 0.002004008016032066,
    "replacement_emit_delay_p90_ms": 0.0
  },
  "S_probe_to_P_C": {
    "contamination_seconds_per_active_speech_hour": 647.297276645485,
    "false_cut_count": -202,
    "missed_replacement_count": 672,
    "overlap_return_preservation_rate": 0.031590413943355045,
    "overlap_takeover_success_rate": -0.012024048096192386,
    "replacement_emit_delay_p90_ms": 2.0
  },
  "hidden_decision_diagnostics": {
    "hidden_causal_over_posterior_causal": {
      "frontier_comparison": {
        "left_dominates": 0,
        "right_dominates": 2,
        "tradeoff": 16
      },
      "reference_cell_improvements": {
        "exclusive_other_contamination_seconds_per_active_speech_hour": true,
        "false_cut_count": false,
        "missed_replacement_count": true
      },
      "source_family_improved_metric_counts": {
        "alimeeting_far_ch0": 3,
        "ami_mix_headset": 2
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
          "overlap_masking": true,
          "overlap_takeover_deficit": true
        },
        "ami_mix_headset": {
          "competitor_separation": false,
          "overlap_masking": false,
          "overlap_takeover_deficit": true
        }
      },
      "source_family_domain": {
        "concentrated_families": [
          "ami_mix_headset"
        ],
        "direct_hidden_to_gt_residuals": {
          "alimeeting_far_ch0": {
            "contamination_seconds_per_active_speech_hour_gap": 1678.4448786702362,
            "false_cut_count_per_active_speech_hour_gap": 91.41696292534282,
            "missed_replacement_rate_gap": 0.6488919667590027,
            "overlap_takeover_success_rate_gap": 0.7608695652173914
          },
          "ami_mix_headset": {
            "contamination_seconds_per_active_speech_hour_gap": 2331.3196202908566,
            "false_cut_count_per_active_speech_hour_gap": 120.0586000309675,
            "missed_replacement_rate_gap": 0.9593796159527327,
            "overlap_takeover_success_rate_gap": 0.7443946188340808
          }
        },
        "minimum_worst_residual_metrics": 3,
        "status": "passed",
        "worst_families_by_metric": {
          "contamination_seconds_per_active_speech_hour_gap": [
            "ami_mix_headset"
          ],
          "false_cut_count_per_active_speech_hour_gap": [
            "ami_mix_headset"
          ],
          "missed_replacement_rate_gap": [
            "ami_mix_headset"
          ],
          "overlap_takeover_success_rate_gap": [
            "alimeeting_far_ch0"
          ]
        },
        "worst_metric_counts": {
          "alimeeting_far_ch0": 1,
          "ami_mix_headset": 3
        }
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
      "right_dominates": 0,
      "tradeoff": 18
    },
    "P_NC_vs_P_C": {
      "left_dominates": 0,
      "right_dominates": 1,
      "tradeoff": 17
    }
  },
  "posterior_train_fit_sanity": {
    "P-C": {
      "fold_count": 2,
      "minimum_accuracy": 0.9288374823477415,
      "minimum_average_precision": 0.8018623735379253,
      "reference_probe_class": "tiny_mlp",
      "status": "passed"
    },
    "P-NC": {
      "fold_count": 2,
      "minimum_accuracy": 0.9235661464368115,
      "minimum_average_precision": 0.8000988623852348,
      "reference_probe_class": "tiny_mlp",
      "status": "passed"
    },
    "S-probe": {
      "fold_count": 2,
      "minimum_accuracy": 0.7744752483940064,
      "minimum_average_precision": 0.23131581137284507,
      "reference_probe_class": "tiny_mlp",
      "status": "failed"
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
        "mean_hazard": 0.42914921207269896,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.02239213765421582,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.21612055621132045,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 2381.4055871413557,
    "false_cut_count": 326,
    "matched_replacement_count": 447,
    "missed_replacement_count": 2351,
    "overlap_return_preservation_rate": 0.9357298474945533,
    "overlap_takeover_success_rate": 0.006012024048096192,
    "predicted_cut_count": 773,
    "reference_replacement_count": 2798,
    "replacement_emit_delay_ms": {
      "p50": 1152.5,
      "p90": 1412.5
    },
    "source_count": 19,
    "source_families": [
      "alimeeting_far_ch0",
      "ami_mix_headset"
    ],
    "speaker_induced_cut_count_per_active_speech_hour": 114.9397276852003,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 18,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 59,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.9357298474945533,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 161,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.006012024048096192
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 36,
        "episodes_with_predicted_cut": 70,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 18,
        "episodes_with_predicted_cut": 151,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 43,
        "episodes_with_predicted_cut": 225,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  },
  "P-NC": {
    "active_speech_hours": 6.725263888888889,
    "backdated_boundary_error_ms": {
      "p50": -30.0,
      "p90": 0.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.4473115852892377,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.035294694368692286,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.23142513100493653,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 2175.2172467416603,
    "false_cut_count": 330,
    "matched_replacement_count": 648,
    "missed_replacement_count": 2150,
    "overlap_return_preservation_rate": 0.9400871459694989,
    "overlap_takeover_success_rate": 0.006012024048096192,
    "predicted_cut_count": 978,
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
    "speaker_induced_cut_count_per_active_speech_hour": 145.42180294453544,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 2,
        "episodes_with_predicted_cut": 33,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 2,
        "episodes_with_predicted_cut": 55,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.9400871459694989,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 172,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.006012024048096192
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 42,
        "episodes_with_predicted_cut": 89,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 24,
        "episodes_with_predicted_cut": 177,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 60,
        "episodes_with_predicted_cut": 349,
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
      "p90": -2.0
    },
    "diagnostic_slices": {
      "anchor_absent_live": {
        "mean_hazard": 0.7569178287112994,
        "support_seconds": 1918.21
      },
      "anchor_only": {
        "mean_hazard": 0.1227114410055057,
        "support_seconds": 18580.731
      },
      "anchor_overlap": {
        "mean_hazard": 0.21892699089575055,
        "support_seconds": 1554.42
      }
    },
    "exclusive_other_contamination_seconds_per_active_speech_hour": 1734.1083104958707,
    "false_cut_count": 528,
    "matched_replacement_count": 1119,
    "missed_replacement_count": 1679,
    "overlap_return_preservation_rate": 0.9041394335511983,
    "overlap_takeover_success_rate": 0.018036072144288578,
    "predicted_cut_count": 1647,
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
    "speaker_induced_cut_count_per_active_speech_hour": 244.89745342499984,
    "topology": {
      "clean_direct_different_speaker_handoff": {
        "eligible_episode_count": 197,
        "episodes_with_aligned_cut": 3,
        "episodes_with_predicted_cut": 50,
        "episodes_with_reference_replacement": 128,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "overlap_return": {
        "eligible_episode_count": 918,
        "episodes_with_aligned_cut": 6,
        "episodes_with_predicted_cut": 88,
        "episodes_with_reference_replacement": 327,
        "overlap_return_preservation_rate": 0.9041394335511983,
        "overlap_takeover_success_rate": null
      },
      "overlap_takeover": {
        "eligible_episode_count": 499,
        "episodes_with_aligned_cut": 9,
        "episodes_with_predicted_cut": 175,
        "episodes_with_reference_replacement": 381,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": 0.018036072144288578
      },
      "same_speaker_silence_gap_resume": {
        "eligible_episode_count": 5864,
        "episodes_with_aligned_cut": 48,
        "episodes_with_predicted_cut": 217,
        "episodes_with_reference_replacement": 946,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "short_backchannel_return": {
        "eligible_episode_count": 791,
        "episodes_with_aligned_cut": 24,
        "episodes_with_predicted_cut": 238,
        "episodes_with_reference_replacement": 336,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      },
      "silence_gap_different_speaker_handoff": {
        "eligible_episode_count": 1472,
        "episodes_with_aligned_cut": 69,
        "episodes_with_predicted_cut": 596,
        "episodes_with_reference_replacement": 1097,
        "overlap_return_preservation_rate": null,
        "overlap_takeover_success_rate": null
      }
    }
  }
}
```
