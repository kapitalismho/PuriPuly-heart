use super::*;
use serde_json::json;

fn photo(attempt: u64) -> Photo {
    serde_json::from_value(json!({
        "photo_id":"test-a", "path":"test.png", "file_size":1, "shutter_unix_ms":1000,
        "attempt_id":attempt, "phase":if attempt == 0 {"offer"} else {"requesting"},
        "offer_open":attempt == 0, "visible":attempt > 0, "image":null,
        "preparations":[], "blocks":[], "failure_code":null, "first_block_drawn":false
    }))
    .unwrap()
}

fn input(grip: bool, distance: f32) -> LegacyVrInput {
    let mut head = IDENTITY;
    head[2][3] = 0.5;
    let mut left = IDENTITY;
    left[0][3] = distance;
    LegacyVrInput {
        head,
        left,
        right: IDENTITY,
        left_handgrip: IDENTITY,
        right_handgrip: IDENTITY,
        reference_epoch: 1,
        left_index: 1,
        right_index: 2,
        grip,
    }
}

fn angled_input(grip: bool, distance: f32, forward_deg: f32, facing_deg: f32) -> LegacyVrInput {
    let mut sample = input(grip, distance);
    for (pose, degrees) in [
        (&mut sample.head, forward_deg),
        (&mut sample.right, facing_deg),
    ] {
        let angle = degrees.to_radians();
        pose[0][0] = angle.cos();
        pose[0][2] = angle.sin();
        pose[2][0] = -angle.sin();
        pose[2][2] = angle.cos();
        assert!(valid_transform(*pose));
    }
    sample
}

struct Fixture {
    interaction: PhotoInteraction,
    photo: Photo,
    start: Instant,
}
impl Fixture {
    fn new(attempt: u64) -> Self {
        let start = Instant::now();
        let config = TemporaryInteractionConfig {
            wrist_transform: Some(IDENTITY),
            ..Default::default()
        };
        let mut interaction = PhotoInteraction::new(config);
        let photo = photo(attempt);
        interaction.snapshot(Some(&photo), start, 1000);
        Self {
            interaction,
            photo,
            start,
        }
    }
    fn step(&mut self, ms: u64, input: Option<LegacyVrInput>) -> InteractionFrame {
        self.interaction.step(
            Some(&self.photo),
            input,
            self.start + Duration::from_millis(ms),
        )
    }
    fn look(&mut self) {
        assert!(self.step(0, Some(input(false, 0.0))).icon.is_none());
        assert!(self.step(49, Some(input(false, 0.0))).icon.is_none());
        let first = self.step(50, Some(input(false, 0.0))).icon.unwrap();
        assert!((first.alpha - 0.01).abs() < 0.0001);
        let half = self.step(100, Some(input(false, 0.0))).icon.unwrap();
        assert!((half.alpha - 0.51).abs() < 0.0001);
        assert_eq!(
            self.step(150, Some(input(false, 0.0))).icon.unwrap().alpha,
            1.0
        );
        let icon = self.step(350, Some(input(false, 0.0))).icon.unwrap();
        assert!((icon.alpha - 1.0).abs() < 0.0001);
        assert_eq!(icon.device, 2);
    }
}

#[test]
fn independent_show_cones_include_boundaries_and_the_former_rejected_region() {
    for sign in [-1.0, 1.0] {
        for (forward, facing, visible) in [
            (35.0, 44.0, true),
            (40.0, 0.0, true),
            (0.0, 45.0, true),
            (40.01, 0.0, false),
            (0.0, 45.01, false),
            (40.01, 45.01, false),
        ] {
            let mut f = Fixture::new(0);
            let sample = angled_input(false, 0.0, sign * forward, sign * facing);
            assert!(f.step(0, Some(sample)).icon.is_none());
            assert_eq!(f.step(50, Some(sample)).icon.is_some(), visible);
            assert_eq!(f.step(150, Some(sample)).icon.is_some(), visible);
        }
    }
}

#[test]
fn either_hide_cone_requires_300ms_then_fades_over_100ms() {
    for (forward, facing) in [(45.01, 0.0), (0.0, 55.01), (45.01, 55.01)] {
        let mut f = Fixture::new(0);
        f.look();
        let sample = angled_input(false, 0.0, forward, facing);
        assert_eq!(f.step(360, Some(sample)).icon.unwrap().alpha, 1.0);
        assert_eq!(f.step(659, Some(sample)).icon.unwrap().alpha, 1.0);
        let first = f.step(660, Some(sample)).icon.unwrap();
        assert!((first.alpha - 0.99).abs() < 0.0001);
        let half = f.step(710, Some(sample)).icon.unwrap();
        assert!((half.alpha - 0.49).abs() < 0.0001);
        assert!(f.step(760, Some(sample)).icon.is_none());
    }
}

#[test]
fn retain_boundaries_and_reentry_clear_pending_hide_without_restarting_show() {
    let mut f = Fixture::new(0);
    f.look();
    let edge = angled_input(false, 0.0, 45.0, 55.0);
    assert_eq!(f.step(360, Some(edge)).icon.unwrap().alpha, 1.0);
    assert_eq!(f.step(760, Some(edge)).icon.unwrap().alpha, 1.0);
    let away = angled_input(false, 0.0, 60.0, 0.0);
    assert_eq!(f.step(770, Some(away)).icon.unwrap().alpha, 1.0);
    assert_eq!(f.step(1069, Some(away)).icon.unwrap().alpha, 1.0);
    assert_eq!(f.step(1070, Some(edge)).icon.unwrap().alpha, 1.0);
    assert!(f.step(1071, Some(angled_input(true, 0.0, 45.0, 55.0))).grab);
    f.step(1072, Some(edge));
    f.step(1297, Some(edge));
    assert_eq!(f.step(1300, Some(away)).icon.unwrap().alpha, 1.0);
    assert_eq!(f.step(1599, Some(away)).icon.unwrap().alpha, 1.0);
    assert!((f.step(1600, Some(away)).icon.unwrap().alpha - 0.99).abs() < 0.0001);
}

#[test]
fn interrupted_show_dwell_resets_and_hidden_reentry_uses_show_cones_again() {
    for (forward, facing) in [(40.01, 0.0), (0.0, 45.01)] {
        let mut f = Fixture::new(0);
        assert!(f.step(0, Some(input(false, 0.0))).icon.is_none());
        let away = angled_input(false, 0.0, forward, facing);
        assert!(f.step(49, Some(away)).icon.is_none());
        assert!(f.step(50, Some(input(false, 0.0))).icon.is_none());
        assert!(f.step(99, Some(input(false, 0.0))).icon.is_none());
        assert!((f.step(100, Some(input(false, 0.0))).icon.unwrap().alpha - 0.01).abs() < 0.0001);
        let far = angled_input(false, 0.0, 60.0, 0.0);
        f.step(200, Some(far));
        f.step(499, Some(far));
        f.step(500, Some(far));
        assert!(f.step(600, Some(far)).icon.is_none());
        assert!(f.step(700, Some(away)).icon.is_none());
        assert!(f.step(1000, Some(away)).icon.is_none());
        assert!(f.step(1010, Some(input(false, 0.0))).icon.is_none());
        assert!(f.step(1059, Some(input(false, 0.0))).icon.is_none());
        assert!(f.step(1060, Some(input(false, 0.0))).icon.is_some());
    }
}

#[test]
fn input_requires_full_opacity_and_200ms_since_shown_with_a_fresh_grip_edge() {
    for rejected_at in [100, 249] {
        let mut f = Fixture::new(0);
        f.step(0, Some(input(false, 0.0)));
        f.step(50, Some(input(false, 0.0)));
        let rejected = f.step(rejected_at, Some(input(true, 0.0)));
        assert!(rejected.icon.is_some());
        assert!(!rejected.grab);
        assert!(!rejected.glow);
        assert!(!f.step(250, Some(input(true, 0.0))).grab);
        assert!(!f.step(350, Some(input(true, 0.12))).pull);
        f.step(351, Some(input(false, 0.0)));
        assert!(f.step(352, Some(input(true, 0.0))).grab);
    }
    let mut f = Fixture::new(0);
    f.step(0, Some(input(false, 0.0)));
    f.step(50, Some(input(false, 0.0)));
    let ready = f.step(250, Some(input(true, 0.0)));
    assert_eq!(ready.icon.unwrap().alpha, 1.0);
    assert!(ready.grab);
}

#[test]
fn temporary_show_dwell_and_angle_overrides_keep_fixed_hide_fade_and_input_policy() {
    let mut f = Fixture::new(0);
    f.interaction.config.dwell_ms = 120;
    f.interaction.config.forward_show_deg = 20.0;
    f.interaction.config.forward_hide_deg = 30.0;
    let outside = angled_input(false, 0.0, 25.0, 0.0);
    assert!(f.step(0, Some(outside)).icon.is_none());
    assert!(f.step(500, Some(outside)).icon.is_none());
    f.step(510, Some(input(false, 0.0)));
    assert!(f.step(629, Some(input(false, 0.0))).icon.is_none());
    assert!(f.step(630, Some(input(false, 0.0))).icon.is_some());
    assert_eq!(f.step(730, Some(outside)).icon.unwrap().alpha, 1.0);
    assert!(!f.step(829, Some(input(true, 0.0))).grab);
    assert!(!f.step(830, Some(input(true, 0.0))).grab);
    f.step(831, Some(input(false, 0.0)));
    assert!(f.step(832, Some(input(true, 0.0))).grab);
}

#[test]
fn active_grab_and_return_remain_visible_outside_both_cones() {
    let mut f = Fixture::new(0);
    f.look();
    assert!(f.step(360, Some(input(true, 0.0))).grab);
    let away = angled_input(true, 0.1, 90.0, 90.0);
    assert_eq!(f.step(380, Some(away)).icon.unwrap().alpha, 1.0);
    let held = f.step(680, Some(away));
    assert_eq!(held.icon.unwrap().device, 1);
    assert_eq!(held.icon.unwrap().alpha, 1.0);
    assert!(!held.pull);
    let released = angled_input(false, 0.1, 90.0, 90.0);
    assert!(!f.step(700, Some(released)).pull);
    assert_eq!(f.step(925, Some(released)).icon.unwrap().alpha, 1.0);
    assert_eq!(f.step(926, Some(released)).icon.unwrap().alpha, 1.0);
    assert_eq!(f.step(1225, Some(released)).icon.unwrap().alpha, 1.0);
    assert!((f.step(1226, Some(released)).icon.unwrap().alpha - 0.99).abs() < 0.0001);
}

#[test]
fn grip_must_begin_in_range_not_merely_enter_while_held() {
    let mut f = Fixture::new(0);
    f.look();
    assert!(!f.step(360, Some(input(true, 0.081))).grab);
    assert!(!f.step(370, Some(input(true, 0.0))).grab);
    f.step(380, Some(input(false, 0.0)));
    assert!(f.step(390, Some(input(true, 0.08))).grab);
    assert_eq!(f.interaction.frame.icon.unwrap().device, 1);
}

#[test]
fn relative_pull_rejects_common_arm_motion_and_triggers_exact_threshold_once() {
    let mut f = Fixture::new(0);
    f.look();
    assert!(f.step(360, Some(input(true, 0.0))).grab);
    let mut together = input(true, 0.0);
    together.left[0][3] = 0.5;
    together.right[0][3] = 0.5;
    assert!(!f.step(380, Some(together)).pull);
    let world_icon = compose(together.left, f.interaction.frame.icon.unwrap().transform);
    assert!((world_icon[0][3] - 0.5).abs() < 0.0001);
    assert!(!f.step(400, Some(input(true, 0.1199))).pull);
    let trigger = f.step(420, Some(input(true, 0.12)));
    assert!(trigger.pull);
    assert!(trigger.photo.is_none());
    assert_eq!(trigger.icon.unwrap().device, 2);
    assert!(!f.step(440, Some(input(true, 0.20))).pull);
    f.step(650, Some(input(false, 0.0)));
    assert!(!f.step(660, Some(input(true, 0.0))).grab);
    f.photo.attempt_id = 1;
    f.photo.phase = PhotoPhase::Preparing;
    f.photo.offer_open = false;
    f.photo.visible = true;
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_millis(670), 1670);
    assert!(f.step(680, Some(input(true, 0.0))).photo.is_none());
}

#[test]
fn early_release_sends_nothing_and_returns_with_wrist_relative_ease_out() {
    let mut f = Fixture::new(0);
    f.look();
    f.step(360, Some(input(true, 0.0)));
    f.step(380, Some(input(true, 0.10)));
    let release = f.step(400, Some(input(false, 0.10)));
    assert!(!release.pull);
    assert!(!release.hidden);
    assert!((release.icon.unwrap().transform[0][3] - 0.10).abs() < 0.0001);
    let mut moved = input(false, 0.0);
    moved.right[0][3] = 0.05;
    let half = f.step(512, Some(moved)).icon.unwrap();
    assert!((half.transform[0][3] - 0.0252).abs() < 0.001);
    assert_eq!(half.device, 2);
    let end = f.step(625, Some(input(false, 0.0))).icon.unwrap();
    assert_eq!(end.transform, IDENTITY);
    assert!(f.step(640, Some(input(true, 0.0))).grab);
}

#[test]
fn retained_pull_appears_and_reverses_alpha_and_width_without_snapshot_reopen() {
    let mut f = Fixture::new(1);
    f.interaction.config.photo_width_m = 0.64;
    f.look();
    f.step(360, Some(input(true, 0.0)));
    assert!(f.step(380, Some(input(true, 0.12))).pull);
    let half = f.step(455, Some(input(true, 0.12))).photo.unwrap();
    assert!((half.alpha - 0.5).abs() < 0.0001);
    assert!((half.width - 0.335).abs() < 0.0001);
    let full = f.step(530, Some(input(true, 0.12))).photo.unwrap();
    assert_eq!(full.width, 0.64);
    assert_eq!(full.alpha, 1.0);
    let release = f.step(550, Some(input(false, 0.12)));
    assert!(release.hidden);
    assert_eq!(release.photo.unwrap().alpha, 1.0);
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_millis(560), 1560);
    let half = f.step(625, Some(input(false, 0.12))).photo.unwrap();
    assert!((half.alpha - 0.5).abs() < 0.0001);
    assert!((half.width - 0.335).abs() < 0.0001);
    assert!(f.step(700, Some(input(false, 0.12))).photo.is_none());
    assert!(f.step(900, Some(input(false, 0.12))).photo.is_none());
    f.step(920, Some(input(true, 0.0)));
    assert!(f.step(940, Some(input(true, 0.12))).pull);
}

#[test]
fn tracking_input_loss_and_fresh_roles_cancel_without_resuming_held_grip() {
    for holding in [false, true] {
        let mut f = Fixture::new(1);
        f.look();
        f.step(360, Some(input(true, 0.0)));
        if holding {
            f.step(380, Some(input(true, 0.12)));
            f.step(530, Some(input(true, 0.12)));
        }
        let loss = f.step(540, None);
        assert!(loss.icon.is_none());
        assert!(loss.photo.is_none());
        assert_eq!(loss.hidden, holding);
        assert!(!f.step(550, Some(input(true, 0.0))).grab);
        f.step(600, Some(input(false, 0.0)));
        f.step(800, Some(input(false, 0.0)));
        assert!(f.step(820, Some(input(true, 0.0))).grab);
        let mut replaced = input(true, 0.2);
        replaced.left_index = 7;
        let changed = f.step(840, Some(replaced));
        assert!(!changed.pull);
        assert!(changed.icon.is_none());
        assert!(changed.photo.is_none());
        assert!(!f.step(860, Some(replaced)).grab);
    }
}

#[test]
fn local_deadlines_cannot_extend_stale_offers_or_retention_while_policy_is_delayed() {
    let mut f = Fixture::new(0);
    f.look();
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_secs(9), 1000);
    assert!(f.step(9999, Some(input(false, 0.0))).icon.is_some());
    assert!(f.step(10000, Some(input(true, 0.0))).icon.is_none());
    f.photo.attempt_id = 1;
    f.photo.phase = PhotoPhase::Requesting;
    f.photo.offer_open = false;
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_secs(11), 12000);
    assert!(f
        .interaction
        .active(&f.photo, f.start + Duration::from_secs(599)));
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_secs(599), 1000);
    assert!(!f
        .interaction
        .active(&f.photo, f.start + Duration::from_secs(600)));
}

#[test]
fn replacement_cancels_grab_retry_keeps_hold_and_failed_icon_never_requests() {
    let mut f = Fixture::new(1);
    f.look();
    f.step(360, Some(input(true, 0.0)));
    f.step(380, Some(input(true, 0.12)));
    f.step(530, Some(input(true, 0.12)));
    f.photo.attempt_id = 2;
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_millis(540), 1540);
    assert_eq!(
        f.step(550, Some(input(true, 0.12))).photo.unwrap().alpha,
        1.0
    );
    f.photo.photo_id = "replacement".into();
    f.interaction
        .snapshot(Some(&f.photo), f.start + Duration::from_millis(560), 1560);
    let replaced = f.step(570, Some(input(true, 0.12)));
    assert!(replaced.photo.is_none());
    assert!(!replaced.pull);
    assert!(!replaced.grab);
    f.photo.phase = PhotoPhase::Failed;
    f.step(580, Some(input(false, 0.0)));
    f.step(800, Some(input(false, 0.0)));
    f.step(950, Some(input(false, 0.0)));
    let failed = f.step(960, Some(input(true, 0.0)));
    assert!(failed.icon.is_some());
    assert!(!failed.grab);
    assert!(!failed.pull);
}

#[test]
fn temporary_config_rejects_nonrigid_and_nonfinite_values() {
    assert_eq!(TemporaryInteractionConfig::default().validate(), Ok(()));
    let mut config = TemporaryInteractionConfig::default();
    config.grab_m = f32::NAN;
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
    config = Default::default();
    let mut invalid = IDENTITY;
    invalid[0][0] = 2.0;
    config.wrist_transform = Some(invalid);
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
}

#[test]
fn loss_role_change_replacement_removal_and_expiry_cancel_every_active_placement() {
    for placement in ["grab", "return", "held"] {
        for interruption in [
            "loss",
            "roles",
            "replacement",
            "removed",
            "component",
            "expired",
            "deadline",
        ] {
            let mut f = Fixture::new(if placement == "held" { 1 } else { 0 });
            f.look();
            assert!(f.step(360, Some(input(true, 0.0))).grab);
            match placement {
                "return" => {
                    f.step(380, Some(input(false, 0.1)));
                }
                "held" => {
                    assert!(f.step(380, Some(input(true, 0.12))).pull);
                    assert_eq!(
                        f.step(530, Some(input(true, 0.12))).photo.unwrap().alpha,
                        1.0
                    );
                }
                _ => {}
            }
            let mut sample = input(true, 0.2);
            let mut ms = 540;
            match interruption {
                "roles" => sample.left_index = 7,
                "component" => sample.reference_epoch += 1,
                "replacement" => {
                    f.photo.photo_id = "new-photo".into();
                    f.interaction.snapshot(
                        Some(&f.photo),
                        f.start + Duration::from_millis(ms),
                        1540,
                    );
                }
                "removed" => {
                    f.interaction
                        .snapshot(None, f.start + Duration::from_millis(ms), 1540)
                }
                "expired" => f.photo.phase = PhotoPhase::Expired,
                "deadline" => ms = if placement == "held" { 600_000 } else { 10_000 },
                _ => {}
            }
            let frame = f.step(
                ms,
                if interruption == "loss" {
                    None
                } else {
                    Some(sample)
                },
            );
            assert!(frame.icon.is_none(), "{placement}/{interruption}");
            assert!(frame.photo.is_none(), "{placement}/{interruption}");
            assert!(!frame.grab, "{placement}/{interruption}");
            assert!(!frame.pull, "{placement}/{interruption}");
            if matches!(interruption, "loss" | "roles" | "component") {
                assert_eq!(frame.hidden, placement == "held");
                assert!(!f.step(550, Some(sample)).grab);
                f.step(600, Some(sample));
                let ready_but_held = f.step(800, Some(sample));
                assert!(ready_but_held.icon.is_some());
                assert!(!ready_but_held.grab);
                assert!(!ready_but_held.pull);
                sample.left[0][3] = 0.0;
                sample.grip = false;
                f.step(810, Some(sample));
                sample.grip = true;
                assert!(f.step(820, Some(sample)).grab);
            }
        }
    }
}

#[test]
fn tracking_loss_during_hide_dwell_resets_visibility_and_show_input_timers() {
    let mut f = Fixture::new(0);
    f.look();
    let away = angled_input(false, 0.0, 60.0, 60.0);
    assert_eq!(f.step(360, Some(away)).icon.unwrap().alpha, 1.0);
    assert!(f.step(500, None).icon.is_none());
    assert!(f.step(510, Some(input(false, 0.0))).icon.is_none());
    assert!(f.step(559, Some(input(false, 0.0))).icon.is_none());
    assert!(f.step(560, Some(input(false, 0.0))).icon.is_some());
    assert_eq!(
        f.step(660, Some(input(false, 0.0))).icon.unwrap().alpha,
        1.0
    );
    assert!(!f.step(759, Some(input(true, 0.0))).grab);
    assert!(!f.step(760, Some(input(true, 0.0))).grab);
    f.step(761, Some(input(false, 0.0)));
    assert!(f.step(762, Some(input(true, 0.0))).grab);
}

#[test]
fn default_held_photo_keeps_375mm_width_and_150ms_reversible_transition() {
    let mut f = Fixture::new(1);
    f.look();
    f.step(360, Some(input(true, 0.0)));
    let pulled = f.step(380, Some(input(true, 0.12)));
    assert!(pulled.pull);
    assert!(pulled.photo.is_none());
    let half = f.step(455, Some(input(true, 0.12))).photo.unwrap();
    assert!((half.alpha - 0.5).abs() < 0.0001);
    assert!((half.width - 0.2025).abs() < 0.0001);
    let full = f.step(530, Some(input(true, 0.12))).photo.unwrap();
    assert_eq!(full.alpha, 1.0);
    assert_eq!(full.width, 0.375);
    assert!(f.step(540, Some(input(false, 0.12))).hidden);
    let half = f.step(615, Some(input(false, 0.12))).photo.unwrap();
    assert!((half.alpha - 0.5).abs() < 0.0001);
    assert!((half.width - 0.2025).abs() < 0.0001);
    assert!(f.step(690, Some(input(false, 0.12))).photo.is_none());
}

fn referenced_input(grip: bool, separation: f32) -> LegacyVrInput {
    let center = [0.347, -0.314, -0.44];
    LegacyVrInput {
        head: [
            [1.0, 0.0, 0.0, center[0]],
            [0.0, 1.0, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2] + 0.5],
        ],
        left: [
            [0.0, 0.0, 1.0, center[0] + separation - 0.03],
            [0.0, 1.0, 0.0, center[1] + 0.07],
            [-1.0, 0.0, 0.0, center[2] + 0.14],
        ],
        right: [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, -1.0, -0.2],
            [0.0, 1.0, 0.0, -0.4],
        ],
        left_handgrip: [
            [1.0, 0.0, 0.0, 0.14],
            [0.0, 0.0, -1.0, -0.07],
            [0.0, 1.0, 0.0, 0.03],
        ],
        right_handgrip: [
            [0.0, -1.0, 0.0, 0.11],
            [1.0, 0.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.09],
        ],
        left_index: 3,
        right_index: 8,
        reference_epoch: 7,
        grip,
    }
}

fn assert_transform_close(actual: Transform, expected: Transform) {
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!(
            (actual - expected).abs() < 0.00001,
            "{actual} != {expected}"
        );
    }
}

fn referenced_fixture(attempt: u64) -> Fixture {
    let mut f = Fixture::new(attempt);
    f.interaction.config.wrist_transform = None;
    for ms in [0, 50, 150, 350] {
        f.step(ms, Some(referenced_input(false, 0.04)));
    }
    f
}

#[test]
fn rotated_handgrips_share_the_visible_near_and_relative_pull_anchor() {
    let mut f = referenced_fixture(0);
    let sample = referenced_input(false, 0.04);
    let world_wrist = [
        [1.0, 0.0, 0.0, 0.347],
        [0.0, 1.0, 0.0, -0.314],
        [0.0, 0.0, 1.0, -0.44],
    ];
    let icon = f.interaction.frame.icon.unwrap();
    assert_eq!(icon.device, sample.right_index);
    assert_transform_close(compose(sample.right, icon.transform), world_wrist);
    assert!(distance(sample.left, world_wrist) > f.interaction.config.grab_m);
    assert!(f.step(351, Some(sample)).glow);
    assert!(!f.step(352, Some(referenced_input(true, 0.081))).grab);
    f.step(353, Some(sample));
    let grabbed = f.step(360, Some(referenced_input(true, 0.04)));
    assert!(grabbed.grab);
    assert_transform_close(
        compose(sample.left, grabbed.icon.unwrap().transform),
        world_wrist,
    );
    assert!(!f.step(370, Some(referenced_input(true, 0.159))).pull);
    assert!(f.step(380, Some(referenced_input(true, 0.161))).pull);
    assert!(!f.step(390, Some(referenced_input(true, 0.30))).pull);
}

#[test]
fn referenced_grab_follows_raw_attachment_and_returns_to_the_moving_right_anchor() {
    let mut f = referenced_fixture(0);
    let sample = referenced_input(true, 0.04);
    let grabbed = f.step(360, Some(sample)).icon.unwrap();
    let mut moved = referenced_input(true, 0.10);
    moved.left[1][3] += 0.01;
    let followed = f.step(380, Some(moved));
    assert!(!followed.pull);
    assert_eq!(followed.icon.unwrap().transform, grabbed.transform);
    let followed_world = compose(moved.left, grabbed.transform);
    moved.grip = false;
    let returned = f.step(400, Some(moved));
    assert!(!returned.pull);
    assert_eq!(returned.icon.unwrap().device, moved.right_index);
    assert_transform_close(
        compose(moved.right, returned.icon.unwrap().transform),
        followed_world,
    );
    moved.right[0][3] += 0.2;
    let half = f.step(512, Some(moved)).icon.unwrap();
    let end = f.step(625, Some(moved)).icon.unwrap();
    let endpoint = compose(moved.right, end.transform);
    assert!((endpoint[0][3] - 0.547).abs() < 0.00001);
    assert!((endpoint[1][3] + 0.314).abs() < 0.00001);
    assert!((endpoint[2][3] + 0.44).abs() < 0.00001);
    assert!(
        distance(half.transform, end.transform)
            < distance(returned.icon.unwrap().transform, end.transform)
    );
}

#[test]
fn common_rigid_motion_of_rotated_references_cannot_authorize_a_request() {
    let mut f = referenced_fixture(0);
    assert!(f.step(360, Some(referenced_input(true, 0.04))).grab);
    let motion = [
        [0.0, 0.0, 1.0, 0.8],
        [0.0, 1.0, 0.0, -0.3],
        [-1.0, 0.0, 0.0, 0.2],
    ];
    let mut sample = referenced_input(true, 0.04);
    sample.head = compose(motion, sample.head);
    sample.left = compose(motion, sample.left);
    sample.right = compose(motion, sample.right);
    let moved = f.step(380, Some(sample));
    assert!(!moved.pull);
    assert_eq!(moved.icon.unwrap().device, sample.left_index);
    sample.grip = false;
    assert!(!f.step(400, Some(sample)).pull);
}

#[test]
fn default_wrist_moves_with_component_position_but_explicit_matrix_remains_raw() {
    for explicit in [false, true] {
        let mut f = Fixture::new(0);
        let raw_wrist = [
            [1.0, 0.0, 0.0, 0.047],
            [0.0, 0.0, 1.0, -0.04],
            [0.0, -1.0, 0.0, 0.114],
        ];
        f.interaction.config = serde_json::from_value(if explicit {
            json!({"wrist_transform": raw_wrist})
        } else {
            json!({})
        })
        .unwrap();
        let mut sample = referenced_input(false, 0.04);
        for ms in [0, 50, 150, 350] {
            f.step(ms, Some(sample));
        }
        assert_transform_close(f.interaction.frame.icon.unwrap().transform, raw_wrist);
        sample.right_handgrip[0][3] += 0.025;
        sample.reference_epoch += 1;
        assert!(f.step(360, Some(sample)).icon.is_none());
        for ms in [410, 510, 610] {
            f.step(ms, Some(sample));
        }
        let mut expected = raw_wrist;
        if !explicit {
            expected[0][3] += 0.025;
        }
        assert_transform_close(f.interaction.frame.icon.unwrap().transform, expected);
    }
}

#[test]
fn referenced_retained_photo_stays_raw_left_relative_with_unchanged_animation() {
    let mut f = referenced_fixture(1);
    f.step(360, Some(referenced_input(true, 0.04)));
    assert!(f.step(380, Some(referenced_input(true, 0.161))).pull);
    let mut sample = referenced_input(true, 0.161);
    sample.left[0][3] += 0.2;
    let half = f.step(455, Some(sample)).photo.unwrap();
    assert_eq!(half.device, sample.left_index);
    assert_eq!(half.transform, f.interaction.config.photo_transform);
    assert!((half.alpha - 0.5).abs() < 0.00001);
    assert!((half.width - 0.2025).abs() < 0.00001);
    let full = f.step(530, Some(sample)).photo.unwrap();
    assert_eq!(full.width, 0.375);
    assert_eq!(full.alpha, 1.0);
    assert_eq!(full.transform, f.interaction.config.photo_transform);
    sample.grip = false;
    assert!(f.step(540, Some(sample)).hidden);
    let half = f.step(615, Some(sample)).photo.unwrap();
    assert!((half.alpha - 0.5).abs() < 0.00001);
    assert!((half.width - 0.2025).abs() < 0.00001);
    assert!(f.step(690, Some(sample)).photo.is_none());
}
