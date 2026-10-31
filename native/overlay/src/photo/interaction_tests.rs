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
        left_index: 1,
        right_index: 2,
        grip,
    }
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
            wrist_transform: IDENTITY,
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
        assert!(self.step(199, Some(input(false, 0.0))).icon.is_none());
        assert!(self.step(200, Some(input(false, 0.0))).icon.is_none());
        let icon = self.step(350, Some(input(false, 0.0))).icon.unwrap();
        assert!((icon.alpha - 1.0).abs() < 0.0001);
        assert_eq!(icon.device, 2);
    }
}

#[test]
fn wrist_dwell_hysteresis_and_linear_fade_only_show_during_wrist_look() {
    let mut f = Fixture::new(0);
    f.look();
    let mut away = input(false, 0.0);
    let angle = 30f32.to_radians();
    away.head = [
        [angle.cos(), 0.0, angle.sin(), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-angle.sin(), 0.0, angle.cos(), 0.5],
    ];
    assert_eq!(f.step(370, Some(away)).icon.unwrap().alpha, 1.0);
    let angle = 60f32.to_radians();
    away.head = [
        [angle.cos(), 0.0, angle.sin(), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-angle.sin(), 0.0, angle.cos(), 0.5],
    ];
    assert_eq!(f.step(380, Some(away)).icon.unwrap().alpha, 1.0);
    assert!((f.step(455, Some(away)).icon.unwrap().alpha - 0.5).abs() < 0.0001);
    assert!(f.step(530, Some(away)).icon.is_none());
    let mut turned_wrist = input(false, 0.0);
    turned_wrist.right = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ];
    assert!(f.step(800, Some(turned_wrist)).icon.is_none());
}

#[test]
fn grip_must_begin_in_range_not_merely_enter_while_held() {
    let mut f = Fixture::new(0);
    f.look();
    assert!(!f.step(360, Some(input(true, 0.071))).grab);
    assert!(!f.step(370, Some(input(true, 0.0))).grab);
    f.step(380, Some(input(false, 0.0)));
    assert!(f.step(390, Some(input(true, 0.07))).grab);
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
        f.step(560, Some(input(false, 0.0)));
        f.step(760, Some(input(false, 0.0)));
        assert!(f.step(780, Some(input(true, 0.0))).grab);
        let mut replaced = input(true, 0.2);
        replaced.left_index = 7;
        let changed = f.step(800, Some(replaced));
        assert!(!changed.pull);
        assert!(changed.photo.is_none());
        assert!(!f.step(820, Some(replaced)).grab);
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
    config.wrist_transform[0][0] = 2.0;
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
}
