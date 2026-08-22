use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::path::Path;
use std::process::Command;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex, OnceLock,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tokio_tungstenite::{accept_async, tungstenite::Message};

use puripuly_heart_overlay::logging::OverlayLogger;
use puripuly_heart_overlay::runtime::SnapshotApplyOutcome;
use puripuly_heart_overlay::{
    load_manifest, resolve_quiet_tail_profile, run_with_manifest, submit_texture,
    validate_manifest, AdapterIdentity, BridgeClient, CaptionBlock, CaptionChannel,
    CaptionRenderer, FakeOpenVr, NativePresentationOwner, OpenVrError, OpenVrRuntimeEvent,
    OverlayBridgeEvent, OverlayFrameSubmitter, OverlayLoggingMode, OverlayManifest,
    OverlayPresentationBlock, OverlayPresentationBlockVariant, OverlayPresentationCalibration,
    OverlayPresentationSnapshot, OverlayRuntime, PresentationBackend, PresentationCause,
    PresentationCauseChannel, PresentationCauseKind, PresentationOutcome, PresentationStage,
    PresentationStrategy, QuietTailProfile, ReadinessOutcome, RenderedFrame, RuntimeFailure,
    SpatialReanchorOutcome, StartupError, EXPECTED_CONTRACT_VERSION, NATIVE_FRESH_RETRY_CADENCE,
    NATIVE_FRESH_RETRY_DEADLINE, NATIVE_FRESH_RETRY_MAX_COMPLETED,
    NATIVE_READINESS_TIMEOUT_RETRY_MAX,
};

#[test]
fn native_fresh_retry_production_policy_matches_dd_002() {
    assert_eq!(NATIVE_FRESH_RETRY_CADENCE, Duration::from_millis(100));
    assert_eq!(NATIVE_FRESH_RETRY_DEADLINE, Duration::from_millis(500));
    assert_eq!(NATIVE_FRESH_RETRY_MAX_COMPLETED, 5);
    assert_eq!(NATIVE_READINESS_TIMEOUT_RETRY_MAX, 5);
    assert_ne!(u64::MAX, 0);
}

#[test]
fn quiet_tail_profiles_have_exact_walls_and_opportunity_maxima() {
    for (profile, milliseconds, maximum) in [
        (QuietTailProfile::P05, 500, 5),
        (QuietTailProfile::P10, 1000, 10),
        (QuietTailProfile::P15, 1500, 15),
        (QuietTailProfile::P20, 2000, 20),
        (QuietTailProfile::NoRetry, 0, 0),
        (QuietTailProfile::OneRetry, 2000, 1),
    ] {
        assert_eq!(
            profile.scheduling_wall(),
            Duration::from_millis(milliseconds)
        );
        assert_eq!(profile.max_final_opportunities(), maximum);
    }
}

#[test]
fn old_manifest_json_without_quiet_tail_profile_uses_product_default_p05() {
    let path = unique_temp_file("legacy-profile", "json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "contract_version": EXPECTED_CONTRACT_VERSION,
            "app_version": "2.3.0",
            "overlay_instance_id": "legacy",
            "bridge_url": "ws://127.0.0.1:1",
            "session_token": "token",
            "parent_pid": 1,
            "startup_deadline_ms": 3000,
            "log_dir": std::env::temp_dir(),
            "log_level": "INFO",
            "locale": "en",
            "logging_mode": "basic"
        }))
        .unwrap(),
    )
    .unwrap();
    load_manifest(&path).unwrap();
    assert_eq!(
        resolve_quiet_tail_profile(None).unwrap(),
        QuietTailProfile::P05
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn quiet_tail_environment_profiles_resolve_strictly() {
    for (value, profile) in [
        ("p05", QuietTailProfile::P05),
        ("p10", QuietTailProfile::P10),
        ("p15", QuietTailProfile::P15),
        ("p20", QuietTailProfile::P20),
        ("no_retry", QuietTailProfile::NoRetry),
        ("one_retry", QuietTailProfile::OneRetry),
    ] {
        assert_eq!(
            resolve_quiet_tail_profile(Some(std::ffi::OsStr::new(value))).unwrap(),
            profile
        );
    }
    let error = resolve_quiet_tail_profile(Some(std::ffi::OsStr::new("P20"))).unwrap_err();
    assert!(matches!(error, StartupError::Manifest(_)));
    assert!(!error.to_string().contains("P20"));
}

#[cfg(windows)]
#[test]
fn non_unicode_quiet_tail_environment_value_fails_without_value_disclosure() {
    use std::os::windows::ffi::OsStringExt;

    let value = std::ffi::OsString::from_wide(&[0xd800]);
    let error = resolve_quiet_tail_profile(Some(value.as_os_str())).unwrap_err();
    assert!(matches!(error, StartupError::Manifest(_)));
    assert_eq!(error.failure_reason(), "manifest_invalid");
    assert_eq!(
        error.to_string(),
        "manifest invalid: quiet tail profile environment value is invalid"
    );
}

#[test]
fn cli_reports_invalid_quiet_tail_profile_as_structured_manifest_error() {
    let path = unique_temp_file("invalid-profile-cli", "json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "contract_version": EXPECTED_CONTRACT_VERSION,
            "app_version": "2.3.0",
            "overlay_instance_id": "invalid-profile",
            "bridge_url": "ws://127.0.0.1:1",
            "session_token": "token",
            "parent_pid": 1,
            "startup_deadline_ms": 3000,
            "log_dir": std::env::temp_dir(),
            "log_level": "INFO",
            "locale": "en",
            "logging_mode": "basic"
        }))
        .unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_PuriPulyHeartOverlay"))
        .args(["--config", path.to_str().unwrap()])
        .env(
            "PURIPULY_OVERLAY_QUIET_TAIL_PROFILE",
            "invalid-secret-value",
        )
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(
        stderr.contains(r#"EVENT {"failure_reason":"manifest_invalid","type":"startup_error"}"#)
    );
    assert!(!stderr.contains("invalid-secret-value"));
}

fn test_manifest() -> OverlayManifest {
    OverlayManifest {
        contract_version: EXPECTED_CONTRACT_VERSION,
        app_version: env!("CARGO_PKG_VERSION").into(),
        overlay_instance_id: "overlay-test".into(),
        bridge_url: "ws://127.0.0.1:1".into(),
        session_token: "expected-token".into(),
        parent_pid: 1,
        startup_deadline_ms: 3000,
        log_dir: std::env::temp_dir()
            .join("puripuly-heart-overlay-tests")
            .display()
            .to_string(),
        log_level: "INFO".into(),
        locale: "en".into(),
        logging_mode: OverlayLoggingMode::Basic,
    }
}

fn unique_log_dir(name: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos();
    std::env::temp_dir()
        .join(format!("puripuly-heart-overlay-tests-{name}-{nonce}"))
        .display()
        .to_string()
}

fn unique_temp_file(name: &str, extension: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos();
    std::env::temp_dir().join(format!("puripuly-heart-overlay-{name}-{nonce}.{extension}"))
}

fn overlay_binary() -> &'static str {
    env!("CARGO_BIN_EXE_PuriPulyHeartOverlay")
}

fn parse_event_payloads(stderr: &[u8]) -> Vec<serde_json::Value> {
    String::from_utf8_lossy(stderr)
        .lines()
        .filter_map(|line| line.strip_prefix("EVENT "))
        .filter_map(|payload| serde_json::from_str(payload).ok())
        .collect()
}

fn block(
    id: &str,
    channel: &str,
    primary_text: &str,
    secondary_text: &str,
    secondary_enabled: bool,
) -> OverlayPresentationBlock {
    OverlayPresentationBlock {
        id: id.to_string(),
        occupant_key: id.to_string(),
        appearance_seq: 1,
        channel: channel.to_string(),
        block_variant: OverlayPresentationBlockVariant::Finalized,
        primary_text: primary_text.to_string(),
        secondary_text: secondary_text.to_string(),
        secondary_enabled,
        primary_language: None,
        secondary_language: None,
        update_id: None,
        origin_wall_clock_ms: None,
        session_scope: None,
    }
}

fn spatial_calibration() -> OverlayPresentationCalibration {
    OverlayPresentationCalibration {
        anchor: "spatial_locked".to_string(),
        ..OverlayPresentationCalibration::default()
    }
}

fn presentation_snapshot(
    revision: u64,
    calibration: OverlayPresentationCalibration,
    blocks: Vec<OverlayPresentationBlock>,
) -> OverlayPresentationSnapshot {
    OverlayPresentationSnapshot {
        revision,
        calibration,
        blocks,
        native_fresh_render_generations: None,
    }
}

fn slot_block(
    id: &str,
    occupant_key: &str,
    appearance_seq: u64,
    channel: &str,
    primary_text: &str,
    secondary_text: &str,
    secondary_enabled: bool,
) -> OverlayPresentationBlock {
    OverlayPresentationBlock {
        id: id.to_string(),
        occupant_key: occupant_key.to_string(),
        appearance_seq,
        channel: channel.to_string(),
        block_variant: OverlayPresentationBlockVariant::Finalized,
        primary_text: primary_text.to_string(),
        secondary_text: secondary_text.to_string(),
        secondary_enabled,
        primary_language: None,
        secondary_language: None,
        update_id: None,
        origin_wall_clock_ms: None,
        session_scope: None,
    }
}

fn active_self_block(id: &str, primary_text: &str) -> OverlayPresentationBlock {
    OverlayPresentationBlock {
        id: id.to_string(),
        occupant_key: id.to_string(),
        appearance_seq: 1,
        channel: "self".to_string(),
        block_variant: OverlayPresentationBlockVariant::ActiveSelf,
        primary_text: primary_text.to_string(),
        secondary_text: String::new(),
        secondary_enabled: true,
        primary_language: None,
        secondary_language: None,
        update_id: None,
        origin_wall_clock_ms: None,
        session_scope: None,
    }
}

static SCRIPTED_BRIDGE_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

enum BridgeAction {
    WaitMs(u64),
    SendSnapshot(serde_json::Value),
    SendShutdown,
}

async fn run_overlay_binary_with_scripted_bridge(
    name: &str,
    initial_snapshot: serde_json::Value,
    actions: Vec<BridgeAction>,
) -> std::process::Output {
    let _guard = SCRIPTED_BRIDGE_TEST_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": initial_snapshot,
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        while let Some(message) = ws.next().await {
            let Ok(Message::Text(text)) = message else {
                continue;
            };
            let payload: serde_json::Value = serde_json::from_str(&text).unwrap();
            if payload["type"] == "overlay_ready" {
                break;
            }
        }

        for action in actions {
            match action {
                BridgeAction::WaitMs(delay_ms) => {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                BridgeAction::SendSnapshot(snapshot) => {
                    if ws
                        .send(Message::Text(
                            json!({
                                "type": "snapshot",
                                "payload": snapshot,
                            })
                            .to_string()
                            .into(),
                        ))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                BridgeAction::SendShutdown => {
                    if ws
                        .send(Message::Text(
                            json!({"type": "shutdown"}).to_string().into(),
                        ))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
    });

    let manifest_path = unique_temp_file(name, "json");
    let manifest = OverlayManifest {
        bridge_url: format!("ws://{}", address),
        logging_mode: OverlayLoggingMode::Detailed,
        ..test_manifest()
    };
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let manifest_path_for_process = manifest_path.clone();
    let output = tokio::time::timeout(Duration::from_secs(10), async move {
        tokio::task::spawn_blocking(move || {
            Command::new(overlay_binary())
                .arg("--config")
                .arg(&manifest_path_for_process)
                .output()
                .unwrap()
        })
        .await
        .unwrap()
    })
    .await
    .unwrap();

    server.await.unwrap();
    let _ = std::fs::remove_file(manifest_path);
    output
}

#[derive(Default)]
struct RecordingSubmitter {
    calls: usize,
    spatial_reanchor_calls: usize,
    spatial_reanchor_outcome: Option<SpatialReanchorOutcome>,
    calibration_anchors: Vec<String>,
    fail: bool,
    operations: Vec<&'static str>,
    visibility_changes: Vec<bool>,
    last_visible: Option<bool>,
    fail_show: bool,
    fail_hide: bool,
}

impl RecordingSubmitter {
    fn failing() -> Self {
        Self {
            fail: true,
            ..Self::default()
        }
    }
}

impl OverlayFrameSubmitter for RecordingSubmitter {
    fn apply_calibration(
        &mut self,
        calibration: &OverlayPresentationCalibration,
    ) -> Result<(), OpenVrError> {
        self.calibration_anchors.push(calibration.anchor.clone());
        Ok(())
    }

    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.calls += 1;
        let operation = if frame.layout().visible_blocks.is_empty() {
            "submit:empty"
        } else {
            "submit:text"
        };
        self.operations.push(operation);
        if self.fail {
            return Err(OpenVrError::Submit("submit failed".into()));
        }
        assert_eq!(frame.width(), 4096);
        assert_eq!(frame.height(), 1056);
        Ok(())
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        self.spatial_reanchor_calls += 1;
        self.operations.push("reanchor");
        Ok(self
            .spatial_reanchor_outcome
            .unwrap_or(SpatialReanchorOutcome::Applied))
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.operations.push(if visible { "show" } else { "hide" });
        if (visible && self.fail_show) || (!visible && self.fail_hide) {
            return Err(OpenVrError::Submit("visibility failed".into()));
        }
        self.last_visible = Some(visible);
        self.visibility_changes.push(visible);
        Ok(())
    }
}

#[derive(Default)]
struct OwnedSubmitterState {
    operations: Mutex<Vec<&'static str>>,
    drops: AtomicUsize,
}

struct OwnedSubmitterProbe {
    state: Arc<OwnedSubmitterState>,
    fail_submit: bool,
    fail_on_submission: Option<usize>,
    submit_delay: Duration,
}

impl OverlayFrameSubmitter for OwnedSubmitterProbe {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        std::thread::sleep(self.submit_delay);
        let mut operations = self.state.operations.lock().unwrap();
        let submission = operations
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count()
            + 1;
        operations.push(if frame.layout().visible_blocks.is_empty() {
            "submit:empty"
        } else {
            "submit:text"
        });
        if self.fail_submit || self.fail_on_submission == Some(submission) {
            return Err(OpenVrError::Submit("owned submit failed".into()));
        }
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if visible { "show" } else { "hide" });
        Ok(())
    }
}

#[derive(Default)]
struct DivergingVisibilitySubmitter {
    operations: Vec<&'static str>,
    observed: Option<bool>,
}

impl OverlayFrameSubmitter for DivergingVisibilitySubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.operations
            .push(if frame.layout().visible_blocks.is_empty() {
                "submit:empty"
            } else {
                "submit:text"
            });
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.operations.push(if visible { "show" } else { "hide" });
        self.observed = Some(visible);
        Ok(())
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        self.observed
    }
}

#[derive(Default)]
struct EventFloodState {
    operations: Mutex<Vec<&'static str>>,
    poll_calls: AtomicUsize,
    max_events_in_one_poll: AtomicUsize,
}

struct EventFloodSubmitter {
    state: Arc<EventFloodState>,
}

impl OverlayFrameSubmitter for EventFloodSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if frame.layout().visible_blocks.is_empty() {
                "submit:empty"
            } else {
                "submit:text"
            });
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if visible { "show" } else { "hide" });
        Ok(())
    }

    fn poll_runtime_events(&mut self, max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        self.state.poll_calls.fetch_add(1, Ordering::SeqCst);
        self.state
            .max_events_in_one_poll
            .fetch_max(max_events, Ordering::SeqCst);
        vec![OpenVrRuntimeEvent::Ignored(1); max_events]
    }
}

struct OverlayHiddenSubmitter {
    state: Arc<OwnedSubmitterState>,
    shown: bool,
    hidden_emitted: bool,
}

impl OverlayFrameSubmitter for OverlayHiddenSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if frame.layout().visible_blocks.is_empty() {
                "submit:empty"
            } else {
                "submit:text"
            });
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if visible { "show" } else { "hide" });
        if visible {
            self.shown = true;
        }
        Ok(())
    }

    fn poll_runtime_events(&mut self, _max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        if self.shown && !self.hidden_emitted {
            self.hidden_emitted = true;
            vec![OpenVrRuntimeEvent::OverlayHidden]
        } else {
            Vec::new()
        }
    }
}

struct ObservedVisibilitySubmitter {
    state: Arc<OwnedSubmitterState>,
    observed: Option<bool>,
}

impl OverlayFrameSubmitter for ObservedVisibilitySubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if frame.layout().visible_blocks.is_empty() {
                "submit:empty"
            } else {
                "submit:text"
            });
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if visible { "show" } else { "hide" });
        self.observed = Some(visible);
        Ok(())
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        self.observed
    }
}

impl Drop for OwnedSubmitterProbe {
    fn drop(&mut self) {
        self.state.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct PresenterTraceSubmitterState {
    operations: Mutex<Vec<String>>,
    submissions: Semaphore,
}

impl Default for PresenterTraceSubmitterState {
    fn default() -> Self {
        Self {
            operations: Mutex::new(Vec::new()),
            submissions: Semaphore::new(0),
        }
    }
}

struct PresenterTraceSubmitter {
    state: Arc<PresenterTraceSubmitterState>,
}

impl OverlayFrameSubmitter for PresenterTraceSubmitter {
    fn apply_calibration(
        &mut self,
        calibration: &OverlayPresentationCalibration,
    ) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(format!("calibration:{}", calibration.anchor));
        Ok(())
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push("reanchor".to_string());
        Ok(SpatialReanchorOutcome::Applied)
    }

    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.state.operations.lock().unwrap().push(
            if frame.layout().visible_blocks.is_empty() {
                "submit:empty"
            } else {
                "submit:text"
            }
            .to_string(),
        );
        self.state.submissions.add_permits(1);
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if visible { "show" } else { "hide" }.to_string());
        Ok(())
    }
}

struct DelayedSecondSubmitter {
    state: Arc<OwnedSubmitterState>,
    submissions: usize,
}

impl OverlayFrameSubmitter for DelayedSecondSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.submissions += 1;
        if self.submissions == 2 {
            std::thread::sleep(Duration::from_millis(150));
        }
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if frame.layout().visible_blocks.is_empty() {
                "submit:empty"
            } else {
                "submit:text"
            });
        Ok(())
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.state
            .operations
            .lock()
            .unwrap()
            .push(if visible { "show" } else { "hide" });
        Ok(())
    }
}

async fn connect_test_bridge() -> (
    BridgeClient,
    tokio::task::JoinHandle<Vec<serde_json::Value>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let auth = ws.next().await.unwrap().unwrap();
        let Message::Text(auth_text) = auth else {
            panic!("expected auth text frame");
        };
        let auth_payload: serde_json::Value = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(auth_payload["type"], "auth");
        assert_eq!(auth_payload["session_token"], "expected-token");

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 0,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [],
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        let mut messages = Vec::new();
        while let Some(message) = ws.next().await {
            let Ok(Message::Text(text)) = message else {
                break;
            };
            messages.push(serde_json::from_str(&text).unwrap());
        }

        messages
    });

    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (client, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    assert!(snapshot.blocks.is_empty());
    (client, server)
}

async fn connect_test_bridge_with_followups(
    followups: Vec<serde_json::Value>,
) -> (
    BridgeClient,
    OverlayPresentationSnapshot,
    tokio::task::JoinHandle<Vec<serde_json::Value>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 0,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": []
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        for followup in followups {
            ws.send(Message::Text(followup.to_string().into()))
                .await
                .unwrap();
        }
        let mut messages = Vec::new();
        while let Ok(Some(message)) = tokio::time::timeout(Duration::from_secs(5), ws.next()).await
        {
            let Ok(Message::Text(text)) = message else {
                continue;
            };
            messages.push(serde_json::from_str(&text).unwrap());
        }
        messages
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    (bridge, snapshot, server)
}

async fn test_logger(name: &str) -> OverlayLogger {
    OverlayLogger::open(unique_log_dir(name), OverlayLoggingMode::Detailed)
        .await
        .unwrap()
}

fn production_presenter_refresh_trace_contract() -> &'static serde_json::Value {
    static CONTRACT: OnceLock<serde_json::Value> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let output = Command::new("uv")
            .current_dir(&repository_root)
            .env("PYTHONPATH", &repository_root)
            .args([
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "tests.helpers.overlay_refresh_trace",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    })
}

#[tokio::test]
async fn spatial_locked_reanchors_only_for_unseen_drawable_turn_ids() {
    let (mut bridge, server) = connect_test_bridge().await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("spatial-semantic-turns").await;
    let mut runtime = OverlayRuntime::new(presentation_snapshot(
        1,
        spatial_calibration(),
        vec![block("self:A", "self", "A", "", true)],
    ));
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    let mut same_a = block("self:A", "self", "A streaming", "translated", true);
    same_a.block_variant = OverlayPresentationBlockVariant::ActiveSelf;
    runtime.apply_snapshot(presentation_snapshot(
        2,
        spatial_calibration(),
        vec![same_a],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let mut refreshed_a = block("self:A", "self", "A final", "translated final", true);
    refreshed_a.session_scope = Some("refresh-nonce".to_string());
    refreshed_a.update_id = Some("refresh-update".to_string());
    runtime.apply_snapshot(presentation_snapshot(
        3,
        spatial_calibration(),
        vec![refreshed_a.clone()],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert!(runtime.request_native_presentation_retry());
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    runtime.apply_snapshot(presentation_snapshot(4, spatial_calibration(), vec![]));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime.apply_snapshot(presentation_snapshot(
        5,
        spatial_calibration(),
        vec![refreshed_a.clone()],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    runtime.apply_snapshot(presentation_snapshot(
        6,
        spatial_calibration(),
        vec![refreshed_a, block("peer:B", "peer", "B", "", true)],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 2);

    runtime.apply_snapshot(presentation_snapshot(
        7,
        spatial_calibration(),
        vec![block("peer:B", "peer", "B", "", true)],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 2);

    runtime.apply_snapshot(presentation_snapshot(
        8,
        spatial_calibration(),
        vec![
            block("peer:B", "peer", "B", "", true),
            block("self:C", "self", "C", "", true),
        ],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 3);

    runtime.apply_snapshot(presentation_snapshot(
        9,
        spatial_calibration(),
        vec![
            block("self:D", "self", "D", "", true),
            block("peer:E", "peer", "E", "", true),
        ],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 4);

    assert!(matches!(
        runtime.apply_snapshot(presentation_snapshot(
            8,
            spatial_calibration(),
            vec![block("self:F", "self", "F", "", true)],
        )),
        SnapshotApplyOutcome::Ignored { .. }
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 4);
    for (index, operation) in submitter.operations.iter().enumerate() {
        if *operation == "reanchor" {
            assert_eq!(submitter.operations.get(index + 1), Some(&"submit:text"));
        }
    }

    drop(bridge);
    server.await.unwrap();
}

#[tokio::test]
async fn spatial_mode_and_placement_calibration_transitions_request_once() {
    let (mut bridge, server) = connect_test_bridge().await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("spatial-mode-calibration").await;
    let visible_a = vec![block("self:A", "self", "A", "", true)];
    let mut runtime = OverlayRuntime::new(presentation_snapshot(
        1,
        OverlayPresentationCalibration::default(),
        visible_a.clone(),
    ));
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 0);

    runtime.apply_snapshot(presentation_snapshot(
        2,
        spatial_calibration(),
        visible_a.clone(),
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    let mut presentation_only = spatial_calibration();
    presentation_only.background_alpha = 0.2;
    presentation_only.text_scale = 1.25;
    runtime.apply_snapshot(presentation_snapshot(
        3,
        presentation_only.clone(),
        vec![block("self:A", "self", "A updated", "", true)],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    presentation_only.offset_x = 0.25;
    runtime.apply_snapshot(presentation_snapshot(
        4,
        presentation_only,
        visible_a.clone(),
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 2);

    let mut offset_y_changed = spatial_calibration();
    offset_y_changed.offset_x = 0.25;
    offset_y_changed.offset_y = -0.5;
    runtime.apply_snapshot(presentation_snapshot(
        5,
        offset_y_changed.clone(),
        visible_a.clone(),
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 3);

    offset_y_changed.distance = 1.75;
    runtime.apply_snapshot(presentation_snapshot(
        6,
        offset_y_changed,
        visible_a.clone(),
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 4);

    runtime.apply_snapshot(presentation_snapshot(
        7,
        OverlayPresentationCalibration::default(),
        visible_a.clone(),
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 4);
    assert_eq!(submitter.calibration_anchors.last().unwrap(), "head_locked");

    runtime.apply_snapshot(presentation_snapshot(8, spatial_calibration(), vec![]));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 4);
    runtime.apply_snapshot(presentation_snapshot(9, spatial_calibration(), visible_a));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 5);

    drop(bridge);
    server.await.unwrap();
}

#[tokio::test]
async fn unavailable_spatial_pose_is_consumed_without_blocking_texture_submit() {
    let (mut bridge, server) = connect_test_bridge().await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("spatial-pose-unavailable").await;
    let mut runtime = OverlayRuntime::new(presentation_snapshot(
        1,
        spatial_calibration(),
        vec![block("self:A", "self", "A", "", true)],
    ));
    let mut submitter = RecordingSubmitter {
        spatial_reanchor_outcome: Some(SpatialReanchorOutcome::PoseUnavailable),
        ..RecordingSubmitter::default()
    };

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.operations[0..2], ["reanchor", "submit:text"]);
    assert_eq!(submitter.calls, 1);
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    assert!(runtime.request_native_presentation_retry());
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime.apply_snapshot(presentation_snapshot(
        2,
        spatial_calibration(),
        vec![block("self:A", "self", "A refresh", "", true)],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.calls, 3);
    assert_eq!(submitter.spatial_reanchor_calls, 1);

    submitter.spatial_reanchor_outcome = Some(SpatialReanchorOutcome::Applied);
    runtime.apply_snapshot(presentation_snapshot(
        3,
        spatial_calibration(),
        vec![
            block("self:A", "self", "A refresh", "", true),
            block("peer:B", "peer", "B", "", true),
        ],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.calls, 4);
    assert_eq!(submitter.spatial_reanchor_calls, 2);

    drop(bridge);
    server.await.unwrap();
}

#[tokio::test]
async fn pending_spatial_reanchor_skips_empty_frame_and_applies_on_drawable_reappearance() {
    let (mut bridge, server) = connect_test_bridge().await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("spatial-empty-before-ready").await;
    let visible_a = vec![block("self:A", "self", "A", "", true)];
    let mut runtime = OverlayRuntime::new(presentation_snapshot(
        1,
        spatial_calibration(),
        visible_a.clone(),
    ));
    let mut submitter = RecordingSubmitter::default();

    runtime.apply_snapshot(presentation_snapshot(2, spatial_calibration(), vec![]));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 0);
    assert_eq!(submitter.operations, vec!["submit:empty"]);

    runtime.apply_snapshot(presentation_snapshot(3, spatial_calibration(), visible_a));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.spatial_reanchor_calls, 1);
    assert_eq!(
        submitter.operations,
        vec!["submit:empty", "reanchor", "submit:text", "show"]
    );

    drop(bridge);
    server.await.unwrap();
}

#[tokio::test]
async fn spatial_reanchor_is_deferred_until_latest_gpu_ready_frame_after_preemption() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let readiness_started = Arc::new(tokio::sync::Notify::new());
    let server_readiness_started = readiness_started.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":presentation_snapshot(
                1,
                spatial_calibration(),
                vec![block("self:A","self","A","",true)]
            )})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message.to_text().unwrap().contains("overlay_ready") {
                break;
            }
        }
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":presentation_snapshot(
                2,
                spatial_calibration(),
                vec![
                    block("self:A","self","A","",true),
                    block("peer:B","peer","B","",true)
                ]
            )})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        server_readiness_started.notified().await;
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":presentation_snapshot(
                3,
                spatial_calibration(),
                vec![
                    block("peer:B","peer","B","",true),
                    block("self:C","self","C","",true)
                ]
            )})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields_on_call(2, usize::MAX);
    renderer.set_test_readiness_started_notify_on_call(2, readiness_started);
    let logger = test_logger("spatial-preemption").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    tokio::time::timeout(
        Duration::from_secs(3),
        runtime.run_event_loop(&mut bridge, &renderer, &mut submitter, &logger),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(submitter.spatial_reanchor_calls, 2);
    assert_eq!(submitter.calls, 2);
    assert_eq!(
        submitter.operations,
        vec!["reanchor", "submit:text", "show", "reanchor", "submit:text"]
    );
    assert_eq!(runtime.state().snapshot().revision, 3);
    server.await.unwrap();
}

#[tokio::test]
async fn event_loop_cancels_stale_readiness_submits_latest_then_handles_shutdown() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        for revision in [0, 1, 2] {
            ws.send(Message::Text(
                json!({
                    "type": "snapshot",
                    "payload": {
                        "revision": revision,
                        "calibration": OverlayPresentationCalibration::default(),
                        "blocks": if revision == 0 { vec![] } else { vec![json!({
                            "id": "self:cancel",
                            "occupant_key": "self:cancel",
                            "appearance_seq": 1,
                            "channel": "self",
                            "block_variant": "finalized",
                            "primary_text": format!("synthetic-{revision}"),
                            "secondary_text": "",
                            "secondary_enabled": true
                        })] }
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            if revision == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, initial_snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(1_000_000);
    let logger = test_logger("readiness-shutdown-preemption").await;
    let mut runtime = OverlayRuntime::new(initial_snapshot);
    let mut submitter = RecordingSubmitter::default();

    tokio::time::timeout(
        Duration::from_secs(1),
        runtime.run_event_loop(&mut bridge, &renderer, &mut submitter, &logger),
    )
    .await
    .unwrap()
    .unwrap();

    server.await.unwrap();
    assert_eq!(submitter.calls, 1);
    assert_eq!(submitter.operations.first(), Some(&"submit:text"));
    assert!(runtime.is_stopped());
    assert!(runtime.presentation_diagnostics().records().is_empty());
}

#[tokio::test]
async fn initial_readiness_processes_new_snapshot_before_submit_and_ready() {
    let followup = json!({
        "type": "snapshot",
        "payload": {
            "revision": 1,
            "calibration": OverlayPresentationCalibration::default(),
            "blocks": [json!({
                "id": "self:initial-latest",
                "occupant_key": "self:initial-latest",
                "appearance_seq": 1,
                "channel": "self",
                "block_variant": "finalized",
                "primary_text": "latest",
                "secondary_text": "",
                "secondary_enabled": true
            })]
        }
    });
    let (mut bridge, snapshot, server) = connect_test_bridge_with_followups(vec![followup]).await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(10_000);
    let logger = test_logger("initial-readiness-snapshot-preemption").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    assert_eq!(submitter.calls, 1);
    assert_eq!(runtime.state().snapshot().revision, 1);
    assert!(runtime.ready_sent());
    drop(bridge);
    let messages = server.await.unwrap();
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["type"] == "overlay_ready")
            .count(),
        1
    );
}

#[tokio::test]
async fn initial_shutdown_preempts_readiness_without_submit_or_ready() {
    let (mut bridge, snapshot, server) =
        connect_test_bridge_with_followups(vec![json!({"type": "shutdown"})]).await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(10_000);
    let logger = test_logger("initial-readiness-shutdown-tie").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    assert!(runtime.is_stopped());
    assert_eq!(submitter.calls, 0);
    assert!(!runtime.ready_sent());
    drop(bridge);
    assert!(server
        .await
        .unwrap()
        .iter()
        .all(|message| message["type"] != "overlay_ready"));
}

#[tokio::test]
async fn heartbeat_and_noop_control_do_not_cancel_initial_readiness() {
    let followups = vec![
        json!({"type": "heartbeat"}),
        json!({"type": "runtime_control", "payload": {"logging_mode": "detailed"}}),
    ];
    let (mut bridge, snapshot, server) = connect_test_bridge_with_followups(followups).await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(100);
    let logger = test_logger("readiness-heartbeat-noop-control").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    assert_eq!(submitter.calls, 1);
    assert!(runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .all(|record| record.outcome != PresentationOutcome::Cancelled));
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn ignored_message_flood_polls_readiness_and_reaches_ready() {
    let mut followups = Vec::new();
    for index in 0..256 {
        followups.push(if index % 2 == 0 {
            json!({"type": "heartbeat"})
        } else {
            json!({"type": "runtime_control", "payload": {"logging_mode": "detailed"}})
        });
    }
    let (mut bridge, snapshot, server) = connect_test_bridge_with_followups(followups).await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(3);
    let logger = test_logger("readiness-ignored-flood-ready").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    tokio::time::timeout(
        Duration::from_millis(200),
        runtime.submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger),
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(submitter.calls, 1);
    assert!(runtime.ready_sent());
    assert!(runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .any(|record| record.outcome == PresentationOutcome::Ready));
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn continuous_ignored_messages_hit_owner_timeout_without_submission() {
    let followups = (0..2_000)
        .map(|index| {
            if index % 2 == 0 {
                json!({"type": "heartbeat"})
            } else {
                json!({"type": "runtime_control", "payload": {"logging_mode": "detailed"}})
            }
        })
        .collect();
    let (mut bridge, snapshot, server) = connect_test_bridge_with_followups(followups).await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(1_000_000);
    let logger = test_logger("readiness-ignored-flood-timeout").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();
    let started = std::time::Instant::now();

    let failure = runtime
        .submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap_err();

    assert_eq!(failure, RuntimeFailure::ReadinessTimedOut);
    assert!(started.elapsed() < Duration::from_millis(200));
    assert_eq!(submitter.calls, 0);
    assert_eq!(
        runtime
            .presentation_diagnostics()
            .records()
            .back()
            .unwrap()
            .outcome,
        PresentationOutcome::TimedOut
    );
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn shutdown_after_ignored_flood_preempts_before_submission() {
    let mut followups = (0..64)
        .map(|_| json!({"type": "heartbeat"}))
        .collect::<Vec<_>>();
    followups.push(json!({"type": "shutdown"}));
    let (mut bridge, snapshot, server) = connect_test_bridge_with_followups(followups).await;
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(1_000_000);
    let logger = test_logger("readiness-ignored-flood-shutdown").await;
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_initial_frame_message_aware(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    assert!(runtime.is_stopped());
    assert_eq!(submitter.calls, 0);
    assert!(!runtime.ready_sent());
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn forced_readiness_terminal_failures_never_submit_and_are_typed() {
    for (outcome, expected_failure, expected_diagnostic) in [
        (
            ReadinessOutcome::TimedOut,
            RuntimeFailure::ReadinessTimedOut,
            PresentationOutcome::TimedOut,
        ),
        (
            ReadinessOutcome::Failed,
            RuntimeFailure::ReadinessFailed,
            PresentationOutcome::Failure,
        ),
    ] {
        let renderer = CaptionRenderer::new_for_test().unwrap();
        renderer.set_test_readiness_terminal_outcome(outcome);
        let logger = test_logger("forced-readiness-terminal").await;
        let (mut bridge, server) = connect_test_bridge().await;
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        let mut submitter = RecordingSubmitter::default();

        let failure = runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap_err();

        assert_eq!(failure, expected_failure);
        assert_eq!(submitter.calls, 0);
        let records = runtime.presentation_diagnostics().records();
        assert_eq!(
            records.back().unwrap().stage,
            PresentationStage::ReadinessObserved
        );
        assert_eq!(records.back().unwrap().outcome, expected_diagnostic);
        assert!(!records
            .iter()
            .any(|record| record.stage == PresentationStage::SubmissionAttempted));
        drop(bridge);
        let _ = server.await.unwrap();
    }
}

#[test]
fn runtime_accepts_app_version_mismatch_when_contract_version_matches() {
    let manifest = OverlayManifest {
        contract_version: EXPECTED_CONTRACT_VERSION,
        app_version: "0.0.1-test".into(),
        ..test_manifest()
    };

    let result = validate_manifest(&manifest);

    assert!(result.is_ok());
}

#[test]
fn readiness_failures_preserve_parent_failure_reason_compatibility() {
    for failure in [
        RuntimeFailure::ReadinessTimedOut,
        RuntimeFailure::ReadinessCancelled,
        RuntimeFailure::ReadinessFailed,
    ] {
        assert_eq!(failure.failure_reason(), "renderer_init_failed");
    }
}

#[test]
fn runtime_expected_contract_version_includes_language_metadata_boundary() {
    assert_eq!(EXPECTED_CONTRACT_VERSION, 6);
}

#[test]
fn runtime_returns_standardized_startup_failure_codes_before_ready() {
    assert_eq!(StartupError::ContractMismatch("bad".into()).exit_code(), 10);
    assert_eq!(StartupError::BridgeAuth("bad token".into()).exit_code(), 12);
    assert_eq!(StartupError::SteamVrNotInstalled.exit_code(), 20);
    assert_eq!(StartupError::SteamVrNotRunning.exit_code(), 20);
    assert_eq!(StartupError::HmdNotFound.exit_code(), 20);
    assert_eq!(
        StartupError::OpenVrInit("steamvr missing".into()).exit_code(),
        20
    );
    assert_eq!(
        StartupError::RendererInit("d3d init failed".into()).exit_code(),
        21
    );
}

#[test]
fn runtime_exposes_specific_preflight_failure_reasons() {
    assert_eq!(
        StartupError::SteamVrNotInstalled.failure_reason(),
        "steamvr_not_installed"
    );
    assert_eq!(
        StartupError::SteamVrNotRunning.failure_reason(),
        "steamvr_not_running"
    );
    assert_eq!(StartupError::HmdNotFound.failure_reason(), "hmd_not_found");
}

#[tokio::test]
async fn runtime_stops_cleanly_on_shutdown_event() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());

    runtime
        .handle_event(OverlayBridgeEvent::Shutdown)
        .await
        .unwrap();

    assert!(runtime.is_stopped());
    assert!(runtime.presentation_diagnostics().records().is_empty());
}

#[tokio::test]
async fn runtime_rejects_submission_after_shutdown_without_new_work() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("post-shutdown-submit").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    let mut submitter = RecordingSubmitter::default();
    runtime
        .handle_event(OverlayBridgeEvent::Shutdown)
        .await
        .unwrap();

    let error = runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap_err();

    assert_eq!(error, RuntimeFailure::Stopped);
    assert_eq!(submitter.calls, 0);
    assert!(runtime.presentation_diagnostics().records().is_empty());
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn runtime_rejects_submission_after_bridge_loss_without_new_work() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("post-bridge-loss-submit").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    let mut submitter = RecordingSubmitter::default();
    runtime.handle_bridge_loss_for_test().await.unwrap();

    let error = runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap_err();

    assert_eq!(error, RuntimeFailure::Stopped);
    assert_eq!(submitter.calls, 0);
    assert!(runtime.presentation_diagnostics().records().is_empty());
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn runtime_reports_bridge_loss_as_runtime_disconnect_after_ready() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    runtime.mark_ready_for_test();

    let err = runtime.handle_bridge_loss_for_test().await.unwrap_err();

    assert_eq!(err.failure_reason(), "runtime_disconnected");
}

#[tokio::test]
async fn runtime_applies_new_snapshot_calibration_to_state() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration {
            anchor: "head_locked".into(),
            offset_x: 0.15,
            offset_y: -0.2,
            distance: 1.2,
            text_scale: 1.1,
            background_alpha: 0.4,
        },
        native_fresh_render_generations: None,
        blocks: vec![],
    });

    assert_eq!(runtime.state().calibration().distance, 1.2);
    assert_eq!(runtime.state().calibration().background_alpha, 0.4);
}

#[tokio::test]
async fn runtime_emits_overlay_ready_only_after_first_texture_submit() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("ready-gating-success").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    let mut submitter = RecordingSubmitter::default();

    assert!(!runtime.ready_sent());

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    drop(bridge);
    let messages = server.await.unwrap();

    assert_eq!(submitter.calls, 1);
    assert!(runtime.ready_sent());
    assert!(messages
        .iter()
        .any(|message| message["type"] == "overlay_ready"));
}

#[tokio::test]
async fn runtime_correlates_allowlisted_presentation_stages_without_payload_data() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("safe-presentation-correlation").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut unsafe_block = block(
        "raw-user-identifier",
        "peer",
        "private caption transcript",
        "private translation payload",
        true,
    );
    unsafe_block.update_id = Some("provider-payload-identifier".into());
    unsafe_block.session_scope = Some("credential-like-session".into());
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 934,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![unsafe_block],
    });
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let records = runtime.presentation_diagnostics().records();
    assert_eq!(records.len(), 6);
    assert_eq!(
        records
            .iter()
            .map(|record| record.stage)
            .collect::<Vec<_>>(),
        vec![
            PresentationStage::LogicalRevisionAccepted,
            PresentationStage::RenderReturned,
            PresentationStage::ReadinessObserved,
            PresentationStage::SubmissionAttempted,
            PresentationStage::SubmissionReturned,
            PresentationStage::VisibilityObserved,
        ]
    );
    assert_eq!(records[2].outcome, PresentationOutcome::Ready);
    assert_eq!(records[4].outcome, PresentationOutcome::Success);
    assert_eq!(records[1].scene_generation, 934);
    assert_eq!(records[1].logical_causes.len(), 1);
    assert_eq!(
        records[1].logical_causes[0].kind,
        puripuly_heart_overlay::PresentationCauseKind::Startup
    );
    assert!(records[1].cpu_prepare_us.is_some());
    assert!(records[1].cpu_render_us.is_some());
    assert!(records[2].readiness_us.is_some());
    assert!(records[4].submission_return_us.is_some());
    assert_eq!(records[4].retry_profile, "p05");
    assert!(!records[4].candidate_build_identity.is_empty());
    assert!(records[4].candidate_build_identity.len() <= 64);
    assert_eq!(records[4].environment_identity, "not_recorded");
    assert_eq!(records[4].manual_hmd_observation, "not_recorded");
    assert!(records
        .iter()
        .all(|record| record.strategy == PresentationStrategy::BoundedGpuCompletion));
    assert!(records
        .iter()
        .all(|record| record.backend != PresentationBackend::D3d11Warp));
    assert!(records
        .iter()
        .all(|record| record.renderer_adapter_identity != AdapterIdentity::NotObservedStageOne));
    assert_eq!(records[5].desired_visible, Some(true));
    assert_eq!(records[5].observed_runtime_visible, Some(true));
    assert!(records
        .iter()
        .skip(1)
        .all(|record| record.logical_revision == records[0].logical_revision));
    assert!(records
        .iter()
        .skip(1)
        .all(|record| record.render_generation == Some(1)));
    let serialized = serde_json::to_string(records).unwrap();
    for prohibited in [
        "raw-user-identifier",
        "private caption transcript",
        "private translation payload",
        "provider-payload-identifier",
        "credential-like-session",
    ] {
        assert!(!serialized.contains(prohibited));
    }
    assert!(serialized.contains("physical_hmd_visibility\":\"not_observable"));

    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn bridge_client_close_sends_close_frame() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let auth = ws.next().await.unwrap().unwrap();
        let Message::Text(auth_text) = auth else {
            panic!("expected auth text frame");
        };
        let auth_payload: serde_json::Value = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(auth_payload["type"], "auth");

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 0,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [],
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        while let Some(message) = ws.next().await {
            match message.unwrap() {
                Message::Close(_) => return true,
                Message::Text(_) | Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => {}
                Message::Frame(_) => {}
            }
        }

        false
    });

    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (mut client, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    assert!(snapshot.blocks.is_empty());

    client.close().await.unwrap();

    assert!(server.await.unwrap());
}

#[tokio::test]
async fn runtime_caption_blocks_keep_channel_metadata_for_color_only_rendering() {
    let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 3,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            block("self:1", "self", "hello", "안녕", true),
            block("peer:2", "peer", "세상", "world", false),
        ],
    });

    let blocks = runtime.caption_blocks();
    let channels = blocks
        .iter()
        .map(|block| {
            (
                block.id.as_str(),
                (
                    block.channel,
                    block.primary_text.as_str(),
                    block.secondary_enabled,
                ),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(
        channels.get("self:1"),
        Some(&(Some(CaptionChannel::SelfChannel), "hello", true))
    );
    assert_eq!(
        channels.get("peer:2"),
        Some(&(Some(CaptionChannel::PeerChannel), "세상", false))
    );
}

#[tokio::test]
async fn runtime_caption_blocks_carry_primary_and_secondary_languages_from_slots() {
    let mut localized = block("peer:localized", "peer", "日本語", "번역", true);
    localized.primary_language = Some("ja".into());
    localized.secondary_language = Some("ko".into());
    let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 4,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![localized],
    });

    let blocks = runtime.caption_blocks();

    assert_eq!(blocks[0].primary_language.as_deref(), Some("ja"));
    assert_eq!(blocks[0].secondary_language.as_deref(), Some("ko"));
}

#[test]
fn runtime_language_only_snapshot_redraws_without_slot_identity_reset() {
    let mut initial = slot_block(
        "self:language",
        "self:language",
        7,
        "self",
        "こんにちは",
        "",
        true,
    );
    initial.primary_language = Some("ko".into());
    initial.update_id = Some("update-stays".into());
    let mut updated = initial.clone();
    updated.primary_language = Some("ja".into());
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![initial],
    });
    runtime.clear_redraw_flag();
    let original_slot = runtime.state().scene().slots()[0]
        .as_ref()
        .expect("initial slot should exist")
        .clone();

    let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![updated],
    });

    assert!(matches!(
        outcome,
        SnapshotApplyOutcome::Applied {
            visual_changed: true,
            redraw_requested: true,
            ..
        }
    ));
    let slot = runtime.state().scene().slots()[0]
        .as_ref()
        .expect("updated slot should stay assigned");
    assert_eq!(slot.slot_index, original_slot.slot_index);
    assert_eq!(slot.slot_entry_order, original_slot.slot_entry_order);
    assert_eq!(slot.occupant_key, original_slot.occupant_key);
    assert_eq!(slot.appearance_seq, original_slot.appearance_seq);
    assert_eq!(slot.update_id, original_slot.update_id);
    assert_eq!(slot.primary_language.as_deref(), Some("ja"));
    assert_eq!(
        runtime.caption_blocks()[0].primary_language.as_deref(),
        Some("ja")
    );
}

#[test]
fn runtime_seeds_initial_snapshot_with_static_block_visual_state() {
    let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![slot_block("self:1", "self:1", 1, "self", "hello", "", true)],
    });

    let blocks = runtime.caption_blocks();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].offset_y_px, 0.0);
    assert_eq!(blocks[0].height_scale, 1.0);
}

#[test]
fn runtime_new_snapshot_keeps_blocks_static_after_seeded_start() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![slot_block("self:1", "self:1", 1, "self", "hello", "", true)],
    });

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "hello", "", true),
            slot_block("peer:2", "peer:2", 2, "peer", "two", "", true),
        ],
    });

    let peer_block = runtime
        .caption_blocks()
        .into_iter()
        .find(|block| block.id == "peer:2")
        .expect("new peer block should render");
    assert_eq!(peer_block.offset_y_px, 0.0);
    assert_eq!(peer_block.height_scale, 1.0);
}

#[test]
fn runtime_keeps_slot_two_top_fixed_when_slot_one_secondary_changes() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "one", "", true),
            slot_block("peer:2", "peer:2", 2, "peer", "two", "", true),
        ],
    });
    let first = runtime.caption_blocks();

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "one", "번역", true),
            slot_block("peer:2", "peer:2", 2, "peer", "two", "", true),
        ],
    });

    let second = runtime.caption_blocks();
    assert_eq!(first[1].slot_top_px, second[1].slot_top_px);
}

#[test]
fn runtime_clears_missing_snapshot_blocks_immediately() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![slot_block("self:1", "self:1", 1, "self", "hello", "", true)],
    });

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![],
    });

    assert!(runtime.caption_blocks().is_empty());
}

#[test]
fn runtime_keeps_active_self_and_finalized_rows_visible_within_two_slot_cap() {
    let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("peer:2", "peer:2", 2, "peer", "second", "", true),
            OverlayPresentationBlock {
                id: "self:active".into(),
                occupant_key: "self:merge-1".into(),
                appearance_seq: 3,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::ActiveSelf,
                primary_text: "speaking".into(),
                secondary_text: String::new(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: None,
                origin_wall_clock_ms: None,
                session_scope: None,
            },
        ],
    });

    assert_eq!(
        runtime
            .caption_blocks()
            .iter()
            .map(|block| block.id.as_str())
            .collect::<Vec<_>>(),
        vec!["peer:2", "self:active"]
    );
}

#[test]
fn runtime_keeps_fixed_slot_visual_state_when_secondary_slot_changes() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "hello", "", false),
            slot_block("peer:2", "peer:2", 2, "peer", "second", "", false),
        ],
    });

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "hello", "translated", true),
            slot_block("peer:2", "peer:2", 2, "peer", "second", "", false),
        ],
    });

    let second = runtime
        .caption_blocks()
        .into_iter()
        .find(|block| block.id == "peer:2")
        .expect("peer block should remain visible");
    let first = runtime
        .caption_blocks()
        .into_iter()
        .find(|block| block.id == "self:1")
        .expect("self block should remain visible");

    assert_eq!(second.offset_y_px, 0.0);
    assert_eq!(first.height_scale, 1.0);
}

#[test]
fn runtime_renderer_uses_fixed_slot_bounds_when_secondary_slot_changes() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "hello", "", false),
            slot_block("peer:2", "peer:2", 2, "peer", "second", "", false),
        ],
    });
    let initial = renderer.render_blocks(runtime.caption_blocks()).unwrap();

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "hello", "translated", true),
            slot_block("peer:2", "peer:2", 2, "peer", "second", "", false),
        ],
    });

    let updated = renderer.render_blocks(runtime.caption_blocks()).unwrap();
    let initial_peer = initial
        .layout()
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:2")
        .expect("initial peer block should render");
    let updated_self = updated
        .layout()
        .visible_blocks
        .iter()
        .find(|block| block.id == "self:1")
        .expect("updated self block should render");
    let updated_peer = updated
        .layout()
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:2")
        .expect("updated peer block should render");

    assert_eq!(
        updated_self.bounds.top_px,
        initial.layout().visible_blocks[0].bounds.top_px
    );
    assert_eq!(updated_peer.bounds.top_px, initial_peer.bounds.top_px);
}

#[cfg(windows)]
#[test]
fn runtime_active_self_frames_do_not_hit_finalized_block_cache() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![active_self_block("self:active", "speaking now")],
    });

    renderer.render_blocks(runtime.caption_blocks()).unwrap();
    let second = renderer.render_blocks(runtime.caption_blocks()).unwrap();

    assert_eq!(second.diagnostics().block_cache_hits, 0);
    assert!(second.diagnostics().line_cache_hits >= 1);
}

#[test]
fn runtime_does_not_render_duplicate_row_when_same_id_reappears_during_exit() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:1", "self", "hello", "", true)],
    });

    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![],
    });
    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 3,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![block("self:1", "self", "hello again", "", true)],
    });

    assert_eq!(
        runtime
            .caption_blocks()
            .iter()
            .map(|block| block.id.as_str())
            .collect::<Vec<_>>(),
        vec!["self:1"]
    );
}

#[tokio::test]
async fn runtime_does_not_emit_overlay_ready_when_first_texture_submit_fails() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("ready-gating-failure").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    let mut submitter = RecordingSubmitter::failing();

    let err = runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap_err();

    drop(bridge);
    let messages = server.await.unwrap();

    assert_eq!(submitter.calls, 1);
    assert!(matches!(err, RuntimeFailure::OpenVr(_)));
    assert!(!runtime.ready_sent());
    let records = runtime.presentation_diagnostics().records();
    assert_eq!(
        records.back().unwrap().stage,
        PresentationStage::SubmissionReturned
    );
    assert_eq!(
        records.back().unwrap().outcome,
        PresentationOutcome::Failure
    );
    assert!(!messages
        .iter()
        .any(|message| message["type"] == "overlay_ready"));
}

#[tokio::test]
async fn runtime_records_failed_show_visibility_without_false_observation() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("failed-show-visibility").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:show", "self", "visible", "", true)],
    });
    let mut submitter = RecordingSubmitter {
        fail_show: true,
        ..RecordingSubmitter::default()
    };

    let error = runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap_err();

    assert!(matches!(error, RuntimeFailure::OpenVr(_)));
    let visibility = runtime.presentation_diagnostics().records().back().unwrap();
    assert_eq!(visibility.stage, PresentationStage::VisibilityObserved);
    assert_eq!(visibility.outcome, PresentationOutcome::Failure);
    assert_eq!(visibility.desired_visible, Some(true));
    assert_eq!(visibility.observed_runtime_visible, Some(false));
    assert_eq!(submitter.operations, vec!["submit:text", "show"]);
    assert!(!runtime.ready_sent());
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn runtime_records_failed_hide_visibility_without_false_observation() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("failed-hide-visibility").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:hide", "self", "visible", "", true)],
    });
    let mut submitter = RecordingSubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime.apply_snapshot(OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![],
    });
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    let grace_visibility = runtime.presentation_diagnostics().records().back().unwrap();
    assert_eq!(
        grace_visibility.stage,
        PresentationStage::VisibilityObserved
    );
    assert_eq!(grace_visibility.outcome, PresentationOutcome::Success);
    assert_eq!(grace_visibility.desired_visible, Some(true));
    assert_eq!(grace_visibility.observed_runtime_visible, Some(true));
    submitter.fail_hide = true;
    tokio::time::sleep(Duration::from_millis(550)).await;

    let error = runtime
        .run_event_loop(&mut bridge, &renderer, &mut submitter, &logger)
        .await
        .unwrap_err();

    assert!(matches!(error, RuntimeFailure::OpenVr(_)));
    let visibility = runtime.presentation_diagnostics().records().back().unwrap();
    assert_eq!(visibility.stage, PresentationStage::VisibilityObserved);
    assert_eq!(visibility.outcome, PresentationOutcome::Failure);
    assert_eq!(visibility.desired_visible, Some(false));
    assert_eq!(visibility.observed_runtime_visible, Some(true));
    assert_eq!(submitter.operations.last(), Some(&"hide"));
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn runtime_reasserts_show_when_cached_visible_but_actual_hidden() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("cached-visible-actual-hidden").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:visible", "self", "visible", "", true)],
    });
    let mut submitter = DivergingVisibilitySubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert_eq!(submitter.operations, vec!["submit:text", "show"]);

    submitter.observed = Some(false);
    runtime.apply_snapshot(OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:later", "self", "later", "", true)],
    });
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    assert_eq!(
        submitter.operations,
        vec!["submit:text", "show", "submit:text", "show"]
    );
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn runtime_submits_same_peer_refresh_target_when_session_scope_nonce_changes() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("peer-refresh-nonce-submit").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let peer_refresh_1 = OverlayPresentationBlock {
        id: "peer:turn-1".into(),
        occupant_key: "peer:turn-1".into(),
        appearance_seq: 1,
        channel: "peer".into(),
        block_variant: OverlayPresentationBlockVariant::Finalized,
        primary_text: "translated peer line".into(),
        secondary_text: "source peer line".into(),
        secondary_enabled: true,
        primary_language: None,
        secondary_language: None,
        update_id: Some("peer-update-1".into()),
        origin_wall_clock_ms: None,
        session_scope: Some("peer_presentation_refresh=1".into()),
    };
    let peer_refresh_2 = OverlayPresentationBlock {
        session_scope: Some("peer_presentation_refresh=2".into()),
        ..peer_refresh_1.clone()
    };

    let first_outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![peer_refresh_1],
    });
    assert!(matches!(
        first_outcome,
        puripuly_heart_overlay::runtime::SnapshotApplyOutcome::Applied {
            visual_changed: true,
            redraw_requested: true,
            ..
        }
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let second_outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![peer_refresh_2],
    });
    assert!(matches!(
        second_outcome,
        puripuly_heart_overlay::runtime::SnapshotApplyOutcome::Applied {
            visual_changed: true,
            redraw_requested: true,
            ..
        }
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    drop(bridge);
    let _messages = server.await.unwrap();

    assert_eq!(submitter.calls, 3);
    assert_eq!(
        submitter.calibration_anchors,
        vec!["head_locked", "head_locked", "head_locked"]
    );
    assert_eq!(
        submitter.operations,
        vec!["submit:empty", "submit:text", "show", "submit:text"]
    );
    let submissions = runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| record.stage == PresentationStage::SubmissionReturned)
        .map(|record| (record.logical_revision, record.render_generation))
        .collect::<Vec<_>>();
    assert_eq!(submissions, vec![(1, Some(1)), (2, Some(2)), (2, Some(3))]);
    assert_eq!(
        runtime
            .presentation_diagnostics()
            .records()
            .iter()
            .filter(|record| record.stage == PresentationStage::LogicalRevisionAccepted)
            .count(),
        2
    );
}

#[tokio::test]
async fn runtime_self_refresh_keeps_logical_identity_and_fresh_render_cadence() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("self-refresh-logical-identity").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    let mut submitter = RecordingSubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    let self_refresh_1 = OverlayPresentationBlock {
        id: "self:finalized".into(),
        occupant_key: "self:finalized".into(),
        appearance_seq: 1,
        channel: "self".into(),
        block_variant: OverlayPresentationBlockVariant::Finalized,
        primary_text: "stable self caption".into(),
        secondary_text: "stable translation".into(),
        secondary_enabled: true,
        primary_language: None,
        secondary_language: None,
        update_id: Some("self-update".into()),
        origin_wall_clock_ms: None,
        session_scope: Some("self_presentation_refresh=1".into()),
    };
    let self_refresh_2 = OverlayPresentationBlock {
        session_scope: Some("self_presentation_refresh=2".into()),
        ..self_refresh_1.clone()
    };

    for (revision, block) in [(1, self_refresh_1), (2, self_refresh_2)] {
        let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block],
        });
        assert!(matches!(
            outcome,
            SnapshotApplyOutcome::Applied {
                visual_changed: true,
                redraw_requested: true,
                ..
            }
        ));
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap();
    }

    assert_eq!(submitter.calls, 3);
    assert_eq!(
        submitter.calibration_anchors,
        vec!["head_locked", "head_locked", "head_locked"]
    );
    assert_eq!(
        submitter.operations,
        vec!["submit:empty", "submit:text", "show", "submit:text"]
    );
    let submissions = runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| record.stage == PresentationStage::SubmissionReturned)
        .map(|record| (record.logical_revision, record.render_generation))
        .collect::<Vec<_>>();
    assert_eq!(submissions, vec![(1, Some(1)), (2, Some(2)), (2, Some(3))]);
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn spatial_peer_and_self_refresh_keep_fresh_submit_cadence_without_reanchoring() {
    for (channel, id, scope_prefix) in [
        ("peer", "peer:refresh", "peer_presentation_refresh"),
        ("self", "self:refresh", "self_presentation_refresh"),
    ] {
        let renderer = CaptionRenderer::new_for_test().unwrap();
        let logger = test_logger(&format!("spatial-{channel}-refresh")).await;
        let (mut bridge, server) = connect_test_bridge().await;
        let mut runtime =
            OverlayRuntime::new(presentation_snapshot(0, spatial_calibration(), vec![]));
        let mut submitter = RecordingSubmitter::default();
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap();

        for revision in 1..=2 {
            let mut refreshed = block(id, channel, "stable caption", "stable source", true);
            refreshed.session_scope = Some(format!("{scope_prefix}={revision}"));
            runtime.apply_snapshot(presentation_snapshot(
                revision,
                spatial_calibration(),
                vec![refreshed],
            ));
            runtime
                .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
                .await
                .unwrap();
        }

        assert!(runtime.request_native_presentation_retry());
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap();

        assert_eq!(submitter.calls, 4);
        assert_eq!(submitter.spatial_reanchor_calls, 1);
        assert_eq!(
            submitter.calibration_anchors,
            vec![
                "spatial_locked",
                "spatial_locked",
                "spatial_locked",
                "spatial_locked"
            ]
        );
        assert_eq!(
            runtime
                .presentation_diagnostics()
                .records()
                .iter()
                .filter(|record| record.stage == PresentationStage::SubmissionReturned)
                .map(|record| record.render_generation)
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3), Some(4)]
        );

        drop(bridge);
        let _ = server.await.unwrap();
    }
}

async fn run_refresh_parity_sequence(
    anchor: &str,
    channel: &str,
) -> (
    usize,
    usize,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<&'static str>,
) {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger(&format!("{anchor}-{channel}-refresh-parity")).await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut calibration = OverlayPresentationCalibration::default();
    calibration.anchor = anchor.to_string();
    let mut runtime =
        OverlayRuntime::new(presentation_snapshot(0, calibration.clone(), Vec::new()));
    let mut submitter = RecordingSubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let id = format!("{channel}:refresh-parity");
    for revision in 1..=4 {
        let mut refreshed = block(&id, channel, "stable caption", "stable source", true);
        if revision < 4 {
            refreshed.session_scope = Some(format!("{channel}_presentation_refresh={revision}"));
        }
        runtime.apply_snapshot(presentation_snapshot(
            revision,
            calibration.clone(),
            vec![refreshed],
        ));
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap();
    }

    let render_generations = runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| {
            record.stage == PresentationStage::RenderReturned
                && record.outcome == PresentationOutcome::Success
        })
        .filter_map(|record| record.render_generation)
        .collect::<Vec<_>>();
    let readiness_generations = runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| {
            record.stage == PresentationStage::ReadinessObserved
                && record.outcome == PresentationOutcome::Ready
        })
        .filter_map(|record| record.render_generation)
        .collect::<Vec<_>>();
    let submission_generations = runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| {
            record.stage == PresentationStage::SubmissionReturned
                && record.outcome == PresentationOutcome::Success
        })
        .filter_map(|record| record.render_generation)
        .collect::<Vec<_>>();
    let submission_operations = submitter
        .operations
        .iter()
        .copied()
        .filter(|operation| operation.starts_with("submit"))
        .collect::<Vec<_>>();
    let result = (
        submitter.calls,
        submitter.spatial_reanchor_calls,
        render_generations,
        readiness_generations,
        submission_generations,
        submission_operations,
    );
    drop(bridge);
    let _ = server.await.unwrap();
    result
}

fn successful_stage_generations(runtime: &OverlayRuntime, stage: PresentationStage) -> Vec<u64> {
    runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| {
            record.stage == stage
                && match stage {
                    PresentationStage::RenderReturned | PresentationStage::SubmissionReturned => {
                        record.outcome == PresentationOutcome::Success
                    }
                    PresentationStage::ReadinessObserved => {
                        record.outcome == PresentationOutcome::Ready
                    }
                    _ => false,
                }
        })
        .filter_map(|record| record.render_generation)
        .collect()
}

struct OwnerTraceResult {
    snapshot_count: usize,
    operations: Vec<String>,
    completed_fresh_retries: usize,
}

async fn consume_submission_permit(
    state: &PresenterTraceSubmitterState,
    trace_name: &str,
    snapshot_index: usize,
) {
    let permit = tokio::time::timeout(Duration::from_secs(5), state.submissions.acquire())
        .await
        .unwrap_or_else(|_| {
            panic!("submission timeout trace={trace_name} snapshot_index={snapshot_index}")
        })
        .unwrap();
    permit.forget();
}

async fn run_production_presenter_trace_through_native_owner(trace_name: &str) -> OwnerTraceResult {
    let snapshots = production_presenter_refresh_trace_contract()["traces"][trace_name]
        ["snapshots"]
        .as_array()
        .unwrap()
        .clone();
    let parsed = snapshots
        .iter()
        .cloned()
        .map(serde_json::from_value::<OverlayPresentationSnapshot>)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut previous_generations = parsed[0].native_fresh_render_generations.clone();
    let mut previous_calibration = parsed[0].calibration.clone();
    let mut previous_blocks = parsed[0].blocks.clone();
    let mut expected_submissions = vec![true];
    for snapshot in parsed.iter().skip(1) {
        let visual_changed =
            snapshot.calibration != previous_calibration || snapshot.blocks != previous_blocks;
        let generation_started = snapshot.native_fresh_render_generations.is_some()
            && snapshot.native_fresh_render_generations != previous_generations;
        expected_submissions.push(visual_changed || generation_started);
        previous_generations = snapshot.native_fresh_render_generations.clone();
        previous_calibration = snapshot.calibration.clone();
        previous_blocks = snapshot.blocks.clone();
    }
    let state = Arc::new(PresenterTraceSubmitterState::default());
    let server_state = state.clone();
    let server_trace_name = trace_name.to_string();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type": "snapshot", "payload": snapshots[0]})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message
                .to_text()
                .is_ok_and(|text| text.contains("overlay_ready"))
            {
                break;
            }
        }
        consume_submission_permit(&server_state, &server_trace_name, 0).await;
        for (snapshot_index, (snapshot, expects_submission)) in snapshots
            .iter()
            .skip(1)
            .zip(expected_submissions.iter().skip(1))
            .enumerate()
        {
            ws.send(Message::Text(
                json!({"type": "snapshot", "payload": snapshot})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
            if *expects_submission {
                consume_submission_permit(&server_state, &server_trace_name, snapshot_index + 1)
                    .await;
            } else {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, initial_snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    assert_eq!(initial_snapshot, parsed[0]);
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        initial_snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        PresenterTraceSubmitter {
            state: state.clone(),
        },
        Duration::from_millis(1),
        Duration::from_millis(50),
        1,
    );
    owner
        .run(
            &mut bridge,
            &test_logger(&format!("presenter-trace-{trace_name}")).await,
        )
        .await
        .unwrap();
    server.await.unwrap();
    assert!(owner.resources_released());
    let result = OwnerTraceResult {
        snapshot_count: parsed.len(),
        operations: state.operations.lock().unwrap().clone(),
        completed_fresh_retries: owner
            .fresh_retry_audit_for_test()
            .iter()
            .filter(|fact| fact.2 == "completed")
            .count(),
    };
    result
}

#[tokio::test]
async fn head_and_spatial_refresh_bursts_keep_baseline_equivalent_render_submit_counts() {
    for channel in ["peer", "self"] {
        let head = run_refresh_parity_sequence("head_locked", channel).await;
        let spatial = run_refresh_parity_sequence("spatial_locked", channel).await;

        assert_eq!(head.0, 5);
        assert_eq!(spatial.0, head.0);
        assert_eq!(head.1, 0);
        assert_eq!(spatial.1, 1);
        assert_eq!(head.2, vec![1, 2, 3, 4, 5]);
        assert_eq!(spatial.2, head.2);
        assert_eq!(head.3, head.2);
        assert_eq!(head.4, head.2);
        assert_eq!(spatial.3, head.3);
        assert_eq!(spatial.4, head.4);
        assert_eq!(spatial.5, head.5);
        assert_eq!(
            spatial.5,
            vec![
                "submit:empty",
                "submit:text",
                "submit:text",
                "submit:text",
                "submit:text"
            ]
        );
    }
}

#[tokio::test]
async fn spatial_new_turn_during_refresh_reanchors_once_without_skipping_ticks() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("spatial-new-turn-during-refresh").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let calibration = spatial_calibration();
    let mut runtime =
        OverlayRuntime::new(presentation_snapshot(0, calibration.clone(), Vec::new()));
    let mut submitter = RecordingSubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    for revision in 1..=6 {
        let mut a = block("peer:A", "peer", "A", "source A", true);
        let mut blocks = vec![a.clone()];
        match revision {
            1..=3 => {
                a.session_scope = Some(format!("peer_presentation_refresh={revision}"));
                blocks = vec![a];
            }
            4..=5 => {
                a.session_scope = Some(format!("peer_presentation_refresh={revision}"));
                let mut b = block("peer:B", "peer", "B", "source B", true);
                b.session_scope = Some(format!("peer_presentation_refresh={revision}"));
                blocks = vec![a, b];
            }
            6 => {
                blocks.push(block("peer:B", "peer", "B", "source B", true));
            }
            _ => unreachable!(),
        }
        runtime.apply_snapshot(presentation_snapshot(revision, calibration.clone(), blocks));
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap();
    }

    assert_eq!(submitter.calls, 7);
    assert_eq!(submitter.spatial_reanchor_calls, 2);
    assert_eq!(
        submitter.operations,
        vec![
            "submit:empty",
            "reanchor",
            "submit:text",
            "show",
            "submit:text",
            "submit:text",
            "reanchor",
            "submit:text",
            "submit:text",
            "submit:text"
        ]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::RenderReturned),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::ReadinessObserved),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::SubmissionReturned),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn mode_and_calibration_changes_during_refresh_preserve_every_submission() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("mode-calibration-during-refresh").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let head = OverlayPresentationCalibration::default();
    let mut spatial = spatial_calibration();
    let mut initial = block("self:A", "self", "A", "", true);
    initial.session_scope = Some("self_presentation_refresh=1".to_string());
    let mut runtime = OverlayRuntime::new(presentation_snapshot(1, head.clone(), vec![initial]));
    let mut submitter = RecordingSubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    for revision in 2..=8 {
        let calibration = match revision {
            2 => head.clone(),
            3..=4 => spatial.clone(),
            5 => {
                spatial.offset_x = 0.35;
                spatial.clone()
            }
            6 => spatial.clone(),
            7..=8 => head.clone(),
            _ => unreachable!(),
        };
        let mut refreshed = block("self:A", "self", "A", "", true);
        if revision != 8 {
            refreshed.session_scope = Some(format!("self_presentation_refresh={revision}"));
        }
        runtime.apply_snapshot(presentation_snapshot(
            revision,
            calibration,
            vec![refreshed],
        ));
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await
            .unwrap();
    }

    assert_eq!(submitter.calls, 8);
    assert_eq!(submitter.spatial_reanchor_calls, 2);
    assert_eq!(
        submitter.calibration_anchors,
        vec![
            "head_locked",
            "head_locked",
            "spatial_locked",
            "spatial_locked",
            "spatial_locked",
            "spatial_locked",
            "head_locked",
            "head_locked"
        ]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::RenderReturned),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::ReadinessObserved),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::SubmissionReturned),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn refresh_cleanup_cancellation_and_target_replacement_follow_stable_identity() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("spatial-refresh-target-lifecycle").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let calibration = spatial_calibration();
    let mut a = block("peer:A", "peer", "A", "source A", true);
    a.session_scope = Some("peer_presentation_refresh=1".to_string());
    let mut runtime = OverlayRuntime::new(presentation_snapshot(
        1,
        calibration.clone(),
        vec![a.clone()],
    ));
    let mut submitter = RecordingSubmitter::default();
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    a.session_scope = Some("peer_presentation_refresh=2".to_string());
    runtime.apply_snapshot(presentation_snapshot(
        2,
        calibration.clone(),
        vec![a.clone()],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime.apply_snapshot(presentation_snapshot(3, calibration.clone(), Vec::new()));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let mut b = block("peer:B", "peer", "B", "source B", true);
    b.session_scope = Some("peer_presentation_refresh=1".to_string());
    runtime.apply_snapshot(presentation_snapshot(
        4,
        calibration.clone(),
        vec![b.clone()],
    ));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    b.session_scope = None;
    runtime.apply_snapshot(presentation_snapshot(5, calibration.clone(), vec![b]));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    a.session_scope = None;
    runtime.apply_snapshot(presentation_snapshot(6, calibration, vec![a]));
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    assert_eq!(submitter.calls, 6);
    assert_eq!(submitter.spatial_reanchor_calls, 2);
    assert_eq!(
        submitter
            .operations
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        6
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::RenderReturned),
        vec![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::ReadinessObserved),
        vec![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        successful_stage_generations(&runtime, PresentationStage::SubmissionReturned),
        vec![1, 2, 3, 4, 5, 6]
    );
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn production_presenter_traces_preserve_native_owner_burst_and_ownership_semantics() {
    let peer_head = run_production_presenter_trace_through_native_owner("peer_head_natural").await;
    let peer_spatial =
        run_production_presenter_trace_through_native_owner("peer_spatial_natural").await;
    let self_head = run_production_presenter_trace_through_native_owner("self_head_natural").await;
    let self_spatial =
        run_production_presenter_trace_through_native_owner("self_spatial_natural").await;

    for (head, spatial) in [(&peer_head, &peer_spatial), (&self_head, &self_spatial)] {
        let head_submissions = head
            .operations
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count();
        let spatial_submissions = spatial
            .operations
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count();
        assert_eq!(head_submissions, head.snapshot_count);
        assert_eq!(spatial_submissions, spatial.snapshot_count);
        assert_eq!(spatial_submissions, head_submissions);
        assert_eq!(
            head.operations
                .iter()
                .filter(|operation| operation.as_str() == "reanchor")
                .count(),
            0
        );
        assert_eq!(
            spatial
                .operations
                .iter()
                .filter(|operation| operation.as_str() == "reanchor")
                .count(),
            1
        );
    }

    let lifecycle = run_production_presenter_trace_through_native_owner("spatial_lifecycle").await;
    assert_eq!(
        lifecycle
            .operations
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        lifecycle.snapshot_count
    );
    assert_eq!(
        lifecycle
            .operations
            .iter()
            .filter(|operation| operation.as_str() == "reanchor")
            .count(),
        3
    );

    let ownership = run_production_presenter_trace_through_native_owner("spatial_ownership").await;
    let ownership_submission_count = ownership
        .operations
        .iter()
        .filter(|operation| operation.starts_with("submit"))
        .count();
    assert_eq!(ownership_submission_count, ownership.snapshot_count - 1);
    assert_eq!(ownership.completed_fresh_retries, 1);
    assert_eq!(
        ownership
            .operations
            .iter()
            .filter(|operation| operation.as_str() == "reanchor")
            .count(),
        1
    );
}

#[tokio::test]
async fn native_owner_retries_unchanged_caption_with_new_generation() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let logger = test_logger("native-owner-retry-generation").await;
    let (mut bridge, server) = connect_test_bridge().await;
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:retry", "self", "stable", "", true)],
    });
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    assert!(runtime.request_native_presentation_retry());
    assert!(runtime.request_native_presentation_retry());
    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();

    let submissions = runtime
        .presentation_diagnostics()
        .records()
        .iter()
        .filter(|record| record.stage == PresentationStage::SubmissionReturned)
        .map(|record| (record.logical_revision, record.render_generation))
        .collect::<Vec<_>>();
    assert_eq!(submissions, vec![(1, Some(1)), (1, Some(2))]);
    assert_eq!(submitter.calls, 2);
    assert_eq!(
        submitter.calibration_anchors,
        vec!["head_locked", "head_locked"]
    );

    runtime
        .handle_event(OverlayBridgeEvent::Shutdown)
        .await
        .unwrap();
    assert!(!runtime.request_native_presentation_retry());
    assert!(!runtime.redraw_requested());
    drop(bridge);
    let _ = server.await.unwrap();
}

#[test]
fn native_owner_coalesces_to_latest_snapshot_and_rejects_stale_overwrite() {
    let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
    for revision in 1..=32 {
        let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block(
                "self:rapid",
                "self",
                &format!("synthetic-{revision}"),
                "",
                true,
            )],
        });
        assert!(matches!(outcome, SnapshotApplyOutcome::Applied { .. }));
    }
    let stale = runtime.apply_snapshot(OverlayPresentationSnapshot {
        revision: 12,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        blocks: vec![block("self:rapid", "self", "stale", "", true)],
    });

    assert!(matches!(stale, SnapshotApplyOutcome::Ignored { .. }));
    assert_eq!(runtime.state().snapshot().revision, 32);
    assert_eq!(
        runtime.state().snapshot().blocks[0].primary_text,
        "synthetic-32"
    );
}

#[tokio::test]
async fn production_owner_coalesces_retry_and_releases_resources_on_shutdown() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 1,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:owner", "self", "stable", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let submitter = OwnedSubmitterProbe {
        state: state.clone(),
        fail_submit: false,
        fail_on_submission: None,
        submit_delay: Duration::ZERO,
    };
    let mut owner = NativePresentationOwner::new(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        submitter,
    );
    let retry = owner.retry_handle();
    assert!(retry.request());
    assert!(retry.request());

    owner
        .run(&mut bridge, &test_logger("production-owner-shutdown").await)
        .await
        .unwrap();

    assert!(owner.resources_released());
    assert!(!retry.request());
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        2
    );
    assert_eq!(state.operations.lock().unwrap().last(), Some(&"hide"));
    server.await.unwrap();
}

#[tokio::test]
async fn diagnostic_profiles_execute_exact_delayed_physical_and_logical_attempts() {
    for (profile, expected_retry_attempts) in [
        (QuietTailProfile::OneRetry, 1usize),
        (QuietTailProfile::NoRetry, 0usize),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();
            let _auth = ws.next().await.unwrap().unwrap();
            for (revision, text) in [(1, "one"), (2, "two")] {
                if revision == 2 {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                ws.send(Message::Text(
                    json!({"type":"snapshot","payload":{
                        "revision":revision,
                        "native_fresh_render_generations":{"self":1},
                        "native_fresh_render_targets":{"self":"self:diagnostic"},
                        "native_quiet_tail_episodes":{"self":{"phase":"final","generation":1}},
                        "blocks":[block("self:diagnostic","self",text,"",true)]
                    }})
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            }
            tokio::time::sleep(Duration::from_millis(450)).await;
            ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        });
        let mut manifest = test_manifest();
        manifest.bridge_url = format!("ws://{address}");
        let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
        let state = Arc::new(OwnedSubmitterState::default());
        let mut owner = NativePresentationOwner::new_with_profile(
            snapshot,
            CaptionRenderer::new_for_test().unwrap(),
            DelayedSecondSubmitter {
                state: state.clone(),
                submissions: 0,
            },
            profile,
        );
        owner
            .run(
                &mut bridge,
                &test_logger("diagnostic-profile-attempts").await,
            )
            .await
            .unwrap();
        let retry_attempts = owner
            .successful_attempt_audit_for_test()
            .iter()
            .filter(|attempt| {
                attempt
                    .logical_causes
                    .to_vec()
                    .iter()
                    .any(|cause| cause.kind == PresentationCauseKind::NativeFreshRetry)
            })
            .count();
        let logical_completions = owner
            .fresh_retry_audit_for_test()
            .iter()
            .filter(|fact| fact.2 == "completed")
            .count();
        assert_eq!(retry_attempts, expected_retry_attempts);
        assert_eq!(logical_completions, expected_retry_attempts);
        assert!(
            state
                .operations
                .lock()
                .unwrap()
                .iter()
                .filter(|operation| operation.starts_with("submit"))
                .count()
                >= 1 + expected_retry_attempts
        );
        server.await.unwrap();
    }
}

#[tokio::test]
async fn production_owner_runs_independent_self_and_peer_fresh_schedules_to_exact_max() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 1,
                    "native_fresh_render_generations": {"self": u64::MAX, "peer": 0},
                    "blocks": [
                        block("self:auto", "self", "self", "", true),
                        block("peer:auto", "peer", "peer", "", true)
                    ]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let submitter = OwnedSubmitterProbe {
        state: state.clone(),
        fail_submit: false,
        fail_on_submission: None,
        submit_delay: Duration::ZERO,
    };
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        submitter,
        Duration::from_millis(1),
        Duration::from_millis(50),
        2,
    );

    owner
        .run(&mut bridge, &test_logger("automatic-channel-retries").await)
        .await
        .unwrap();

    let submits = state
        .operations
        .lock()
        .unwrap()
        .iter()
        .filter(|operation| operation.starts_with("submit"))
        .count();
    assert_eq!(submits, 3);
    let audit = owner.fresh_retry_audit_for_test();
    let scheduled_self = audit
        .iter()
        .find(|fact| fact.0 == "self" && fact.2 == "scheduled")
        .unwrap();
    let first_self_completion = audit
        .iter()
        .find(|fact| fact.0 == "self" && fact.2 == "completed")
        .unwrap();
    assert!(first_self_completion.4 >= scheduled_self.4 + Duration::from_millis(1));
    assert_eq!(
        audit
            .iter()
            .filter(|fact| fact.2 == "completed")
            .map(|fact| (fact.0, fact.1, fact.3))
            .collect::<Vec<_>>(),
        vec![
            ("self", u64::MAX, 1),
            ("peer", 0, 1),
            ("self", u64::MAX, 2),
            ("peer", 0, 2),
        ]
    );
    let coalesced = owner
        .successful_attempt_audit_for_test()
        .into_iter()
        .filter(|attempt| {
            attempt.logical_causes.contains(PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::SelfChannel),
                trigger_generation: Some(u64::MAX),
            }) && attempt.logical_causes.contains(PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::Peer),
                trigger_generation: Some(0),
            })
        })
        .count();
    assert_eq!(coalesced, 2);
    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_staggered_due_channels_keep_non_due_intent_pending() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let server_state = state.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,"native_fresh_render_generations":{"self":1},
                "blocks":[block("self:stagger","self","self","",true),block("peer:stagger","peer","peer","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
        ws.send(Message::Text(json!({"type":"snapshot","payload":{
            "revision":2,"native_fresh_render_generations":{"self":1,"peer":2},
            "blocks":[block("self:stagger","self","self","",true),block("peer:stagger","peer","peer","",true)]
        }}).to_string().into())).await.unwrap();
        while server_state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count()
            < 2
        {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        while server_state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count()
            < 3
        {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(20),
        Duration::from_millis(100),
        1,
    );
    owner
        .run(&mut bridge, &test_logger("staggered-due-owner").await)
        .await
        .unwrap();
    let attempts = owner.successful_attempt_audit_for_test();
    let self_attempt = attempts
        .iter()
        .find(|attempt| {
            attempt.logical_causes.contains(PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::SelfChannel),
                trigger_generation: Some(1),
            })
        })
        .unwrap();
    assert!(!self_attempt.logical_causes.contains(PresentationCause {
        kind: PresentationCauseKind::NativeFreshRetry,
        channel: Some(PresentationCauseChannel::Peer),
        trigger_generation: Some(2),
    }));
    assert!(attempts.iter().any(
        |attempt| attempt.logical_causes.contains(PresentationCause {
            kind: PresentationCauseKind::NativeFreshRetry,
            channel: Some(PresentationCauseChannel::Peer),
            trigger_generation: Some(2),
        })
    ));
    let audit = owner.fresh_retry_audit_for_test();
    let peer_scheduled = audit
        .iter()
        .position(|fact| fact.0 == "peer" && fact.2 == "scheduled")
        .unwrap();
    let self_completed = audit
        .iter()
        .position(|fact| fact.0 == "self" && fact.2 == "completed")
        .unwrap();
    let peer_completed = audit
        .iter()
        .position(|fact| fact.0 == "peer" && fact.2 == "completed")
        .unwrap();
    assert!(peer_scheduled < self_completed);
    assert!(self_completed < peer_completed);
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count(),
        3
    );
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_stale_scene_cannot_satisfy_newer_schedule() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let server_state = state.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,"native_fresh_render_generations":{"self":1},
                "blocks":[block("self:stale","self","one","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":2,"native_fresh_render_generations":{"self":2},
                "blocks":[block("self:stale","self","one","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            server_state
                .operations
                .lock()
                .unwrap()
                .iter()
                .filter(|op| op.starts_with("submit"))
                .count(),
            1
        );
        while server_state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count()
            < 2
        {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(50),
        Duration::from_millis(100),
        1,
    );
    owner
        .run(&mut bridge, &test_logger("stale-scene-owner").await)
        .await
        .unwrap();
    let audit = owner.fresh_retry_audit_for_test();
    assert!(!audit
        .iter()
        .any(|fact| fact.1 == 1 && fact.2 == "completed"));
    let generation_two_scheduled = audit
        .iter()
        .position(|fact| fact.1 == 2 && fact.2 == "scheduled")
        .unwrap();
    let generation_two_completed = audit
        .iter()
        .position(|fact| fact.1 == 2 && fact.2 == "completed")
        .unwrap();
    assert!(generation_two_scheduled < generation_two_completed);
    assert!(!owner
        .successful_attempt_audit_for_test()
        .iter()
        .any(|attempt| attempt.scene_generation == 1
            && attempt.logical_causes.contains(PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::SelfChannel),
                trigger_generation: Some(2),
            })));
    assert_eq!(
        audit
            .iter()
            .filter(|fact| fact.1 == 2 && fact.2 == "completed")
            .count(),
        1
    );
    assert!(owner
        .successful_attempt_audit_for_test()
        .iter()
        .any(|attempt| {
            attempt.scene_generation == 2
                && attempt.logical_causes.contains(PresentationCause {
                    kind: PresentationCauseKind::NativeFreshRetry,
                    channel: Some(PresentationCauseChannel::SelfChannel),
                    trigger_generation: Some(2),
                })
        }));
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count(),
        2
    );
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_self_cancellation_leaves_peer_schedule_completing() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(json!({"type":"snapshot","payload":{"revision":1,
            "native_fresh_render_generations":{"self":1,"peer":2},"blocks":[
                block("self:cancel-one","self","self","",true),block("peer:continue","peer","peer","",true)]}}).to_string().into())).await.unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
        ws.send(Message::Text(json!({"type":"snapshot","payload":{"revision":2,
            "native_fresh_render_generations":{"self":1,"peer":2},"blocks":[block("peer:continue","peer","peer","",true)]}}).to_string().into())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        1,
    );
    owner
        .run(&mut bridge, &test_logger("independent-cancel-owner").await)
        .await
        .unwrap();
    let audit = owner.fresh_retry_audit_for_test();
    assert_eq!(
        audit
            .iter()
            .filter(|fact| fact.0 == "self" && fact.2 == "cancelled")
            .count(),
        1
    );
    assert!(!audit
        .iter()
        .any(|fact| fact.0 == "self" && fact.2 == "completed"));
    assert_eq!(
        audit
            .iter()
            .filter(|fact| fact.0 == "peer" && fact.2 == "completed")
            .count(),
        1
    );
    assert!(owner
        .successful_attempt_audit_for_test()
        .iter()
        .any(
            |attempt| attempt.logical_causes.contains(PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::Peer),
                trigger_generation: Some(2),
            })
        ));
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count(),
        3
    );
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_coalesced_two_channel_submission_failure_bounds_both() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(json!({"type":"snapshot","payload":{"revision":1,
            "native_fresh_render_generations":{"self":1,"peer":2},"blocks":[
                block("self:fail-both","self","self","",true),block("peer:fail-both","peer","peer","",true)]}}).to_string().into())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: Some(2),
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(5),
        Duration::from_millis(100),
        1,
    );
    assert!(matches!(
        owner
            .run(&mut bridge, &test_logger("coalesced-failure-owner").await)
            .await
            .unwrap_err(),
        RuntimeFailure::OpenVr(_)
    ));
    let audit = owner.fresh_retry_audit_for_test();
    assert_eq!(audit.iter().filter(|fact| fact.2 == "failed").count(), 2);
    assert_eq!(audit.iter().filter(|fact| fact.2 == "completed").count(), 0);
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count(),
        2
    );
    assert!(owner.resources_released());
    assert!(owner
        .runtime()
        .pending_presentation_causes_for_test()
        .is_empty());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_coalesced_two_channel_shutdown_tears_down_both() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(json!({"type":"snapshot","payload":{"revision":1,
            "native_fresh_render_generations":{"self":1,"peer":2},"blocks":[
                block("self:stop-both","self","self","",true),block("peer:stop-both","peer","peer","",true)]}}).to_string().into())).await.unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(100),
        Duration::from_secs(1),
        2,
    );
    owner
        .run(&mut bridge, &test_logger("coalesced-shutdown-owner").await)
        .await
        .unwrap();
    let audit = owner.fresh_retry_audit_for_test();
    assert_eq!(audit.iter().filter(|fact| fact.2 == "teardown").count(), 2);
    assert_eq!(audit.iter().filter(|fact| fact.2 == "completed").count(), 0);
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|op| op.starts_with("submit"))
            .count(),
        1
    );
    assert!(owner.resources_released());
    assert!(owner
        .runtime()
        .pending_presentation_causes_for_test()
        .is_empty());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_replaces_channel_token_and_empty_snapshot_cancels_schedule() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"self":u64::MAX},
                "blocks":[block("self:replace","self","one","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":2,
                "native_fresh_render_generations":{"self":u64::MAX},
                "blocks":[block("self:new-target","self","two","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":3,
                "native_fresh_render_generations":{"self":u64::MAX},
                "blocks":[]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let retry_submitter = OwnedSubmitterProbe {
        state: state.clone(),
        fail_submit: false,
        fail_on_submission: None,
        submit_delay: Duration::ZERO,
    };
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        retry_submitter,
        Duration::from_millis(50),
        Duration::from_secs(1),
        20,
    );
    let retry = owner.retry_handle();

    owner
        .run(&mut bridge, &test_logger("replacement-empty-cancel").await)
        .await
        .unwrap();

    assert_eq!(owner.runtime().state().snapshot().revision, 3);
    assert!(owner.runtime().state().blocks().is_empty());
    assert!(owner.resources_released());
    assert!(!retry.request());
    let submit_count = state
        .operations
        .lock()
        .unwrap()
        .iter()
        .filter(|operation| operation.starts_with("submit"))
        .count();
    let audit = owner.fresh_retry_audit_for_test();
    let scheduled = audit
        .iter()
        .enumerate()
        .filter(|(_, fact)| fact.0 == "self" && fact.1 == u64::MAX && fact.2 == "scheduled")
        .collect::<Vec<_>>();
    assert_eq!(scheduled.len(), 1);
    assert!(!audit
        .iter()
        .any(|fact| fact.0 == "self" && fact.1 == u64::MAX && fact.2 == "replaced"));
    assert!(audit
        .iter()
        .any(|fact| fact.0 == "self" && fact.1 == u64::MAX && fact.2 == "cancelled"));
    assert_eq!(submit_count, 3);
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_preemption_preserves_due_and_completes_on_pending_snapshot() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let readiness_started = Arc::new(tokio::sync::Notify::new());
    let server_readiness_started = readiness_started.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"self":7},
                "native_fresh_render_targets":{"self":"self:preempt"},
                "native_quiet_tail_episodes":{"self":{"phase":"final","generation":1}},
                "blocks":[block("self:preempt","self","one","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message.to_text().unwrap().contains("overlay_ready") {
                break;
            }
        }
        server_readiness_started.notified().await;
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":2,
                "native_fresh_render_generations":{"self":8},
                "native_fresh_render_targets":{"self":"self:preempt"},
                "native_quiet_tail_episodes":{"self":{"phase":"final","generation":1}},
                "blocks":[block("self:preempt","self","two","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields_on_call(2, usize::MAX);
    renderer.set_test_readiness_started_notify_on_call(2, readiness_started);
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        renderer,
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::ZERO,
        Duration::from_millis(200),
        1,
    );

    owner
        .run(
            &mut bridge,
            &test_logger("generation-transfer-preemption").await,
        )
        .await
        .unwrap();

    let audit = owner.fresh_retry_audit_for_test();
    let preempted = audit
        .iter()
        .find(|fact| fact.0 == "self" && fact.1 == 7 && fact.2 == "preempted")
        .unwrap();
    assert_eq!(preempted.3, 0);
    let completed = audit
        .iter()
        .find(|fact| fact.0 == "self" && fact.1 == 8 && fact.2 == "completed")
        .unwrap();
    assert_eq!(completed.3, 1);
    let normal_completion = owner
        .successful_attempt_audit_for_test()
        .into_iter()
        .find(|attempt| {
            attempt.scene_generation == 2
                && attempt.logical_causes.contains(PresentationCause {
                    kind: PresentationCauseKind::NativeFreshRetry,
                    channel: Some(PresentationCauseChannel::SelfChannel),
                    trigger_generation: Some(7),
                })
        })
        .unwrap();
    assert_eq!(normal_completion.scene_generation, 2);
    assert!(completed.4 >= preempted.4);
    assert!(owner.runtime().state().snapshot().revision >= 2);
    assert_eq!(owner.runtime().state().blocks()[0].primary_text, "two");
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        2
    );
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_active_schedule_submission_failure_is_terminal() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"self":9},
                "blocks":[block("self:failure","self","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(800)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: Some(2),
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );
    let retry = owner.retry_handle();

    let failure = owner
        .run(
            &mut bridge,
            &test_logger("active-schedule-submit-failure").await,
        )
        .await
        .unwrap_err();

    assert!(matches!(failure, RuntimeFailure::OpenVr(_)));
    assert_eq!(
        owner
            .fresh_retry_audit_for_test()
            .iter()
            .filter(|fact| fact.0 == "self" && fact.1 == 9 && fact.2 == "failed")
            .count(),
        1
    );
    assert!(!owner
        .fresh_retry_audit_for_test()
        .iter()
        .any(|fact| fact.0 == "self" && fact.1 == 9 && fact.2 == "teardown"));
    assert!(owner.resources_released());
    assert!(!retry.request());
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        2
    );
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_active_schedule_readiness_failure_is_terminal() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"peer":4},
                "blocks":[block("peer:failure","peer","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(800)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_terminal_outcome_on_call(2, ReadinessOutcome::Failed);
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        renderer,
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );
    let retry = owner.retry_handle();

    let failure = owner
        .run(
            &mut bridge,
            &test_logger("active-schedule-readiness-failure").await,
        )
        .await
        .unwrap_err();

    assert_eq!(failure, RuntimeFailure::ReadinessFailed);
    assert_eq!(
        owner
            .fresh_retry_audit_for_test()
            .iter()
            .filter(|fact| fact.0 == "peer" && fact.1 == 4 && fact.2 == "failed")
            .count(),
        1
    );
    assert!(owner.resources_released());
    assert!(!retry.request());
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        1
    );
    assert!(owner
        .runtime()
        .pending_presentation_causes_for_test()
        .is_empty());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_single_readiness_timeout_retries_without_submit_or_exit() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"peer":4},
                "blocks":[block("peer:timeout","peer","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message.to_text().unwrap().contains("overlay_ready") {
                break;
            }
        }
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_terminal_outcome_on_call(1, ReadinessOutcome::TimedOut);
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        renderer,
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );

    owner
        .run(
            &mut bridge,
            &test_logger("single-readiness-timeout-retry").await,
        )
        .await
        .unwrap();

    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        1
    );
    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_openvr_event_flood_does_not_starve_snapshot_submit() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = Arc::new(EventFloodState::default());
    let server_state = state.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "blocks":[block("self:flood-1","self","first","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message.to_text().unwrap().contains("overlay_ready") {
                break;
            }
        }
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":2,
                "blocks":[block("self:flood-2","self","second","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        let waited = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let submits = server_state
                    .operations
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|operation| operation.starts_with("submit"))
                    .count();
                if submits >= 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            waited.is_ok(),
            "second snapshot submit starved by OpenVR event flood"
        );
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        EventFloodSubmitter {
            state: state.clone(),
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );

    owner
        .run(
            &mut bridge,
            &test_logger("openvr-event-flood-fairness").await,
        )
        .await
        .unwrap();

    assert!(state.poll_calls.load(Ordering::SeqCst) >= 1);
    assert!(state.max_events_in_one_poll.load(Ordering::SeqCst) <= 8);
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        2
    );
    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_overlay_hidden_reasserts_show_when_desired_visible() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let server_state = state.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "blocks":[block("self:hidden","self","visible","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message.to_text().unwrap().contains("overlay_ready") {
                break;
            }
        }
        let waited = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let shows = server_state
                    .operations
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|operation| **operation == "show")
                    .count();
                if shows >= 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(waited.is_ok(), "OverlayHidden did not reassert ShowOverlay");
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OverlayHiddenSubmitter {
            state: state.clone(),
            shown: false,
            hidden_emitted: false,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );

    owner
        .run(
            &mut bridge,
            &test_logger("overlay-hidden-reassert-show").await,
        )
        .await
        .unwrap();

    let operations = state.operations.lock().unwrap().clone();
    assert!(
        operations.starts_with(&["submit:text", "show", "show"]),
        "unexpected visibility operations: {operations:?}"
    );
    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_event_pump_preserves_idle_hide_tail() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let server_state = state.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "blocks":[block("self:tail","self","visible","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            let message = ws.next().await.unwrap().unwrap();
            if message.to_text().unwrap().contains("overlay_ready") {
                break;
            }
        }
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":2,
                "blocks":[]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if server_state
                    .operations
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|operation| *operation == "submit:empty")
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("empty snapshot was not submitted");
        tokio::time::sleep(Duration::from_millis(150)).await;
        let mid_tail = server_state.operations.lock().unwrap().clone();
        assert!(
            !mid_tail.iter().any(|operation| *operation == "hide"),
            "event pump hid overlay during idle-hide tail: {mid_tail:?}"
        );
        tokio::time::sleep(Duration::from_millis(450)).await;
        let after_tail = server_state.operations.lock().unwrap().clone();
        assert!(
            after_tail.iter().any(|operation| *operation == "hide"),
            "overlay was not hidden after idle-hide tail: {after_tail:?}"
        );
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        ObservedVisibilitySubmitter {
            state: state.clone(),
            observed: None,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );

    owner
        .run(
            &mut bridge,
            &test_logger("event-pump-preserves-idle-hide-tail").await,
        )
        .await
        .unwrap();

    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_consecutive_readiness_timeouts_escalate_without_submit() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"peer":4},
                "blocks":[block("peer:timeouts","peer","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(800)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    renderer.set_test_readiness_pending_yields(1_000_000);
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        renderer,
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(10),
        Duration::from_millis(100),
        2,
    );
    owner.set_max_consecutive_readiness_timeouts_for_test(2);

    let failure = owner
        .run(
            &mut bridge,
            &test_logger("consecutive-readiness-timeouts").await,
        )
        .await
        .unwrap_err();

    assert_eq!(failure, RuntimeFailure::ReadinessTimedOut);
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        0
    );
    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_shutdown_records_active_schedule_teardown() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"self":31},
                "blocks":[block("self:shutdown","self","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: Arc::new(OwnedSubmitterState::default()),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(100),
        Duration::from_secs(1),
        2,
    );

    owner
        .run(&mut bridge, &test_logger("shutdown-active-schedule").await)
        .await
        .unwrap();

    assert_eq!(
        owner
            .fresh_retry_audit_for_test()
            .iter()
            .filter(|fact| fact.0 == "self" && fact.1 == 31 && fact.2 == "teardown")
            .count(),
        1
    );
    assert!(owner
        .runtime()
        .pending_presentation_causes_for_test()
        .is_empty());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_non_retry_disconnect_records_active_schedule_teardown() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"peer":32},
                "blocks":[block("peer:disconnect","peer","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        loop {
            if ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .to_text()
                .unwrap()
                .contains("overlay_ready")
            {
                break;
            }
        }
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: Arc::new(OwnedSubmitterState::default()),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::ZERO,
        },
        Duration::from_millis(100),
        Duration::from_secs(1),
        2,
    );

    let failure = owner
        .run(
            &mut bridge,
            &test_logger("disconnect-active-schedule").await,
        )
        .await
        .unwrap_err();

    assert!(matches!(
        failure,
        RuntimeFailure::RuntimeDisconnected | RuntimeFailure::Bridge(_)
    ));
    assert_eq!(
        owner
            .fresh_retry_audit_for_test()
            .iter()
            .filter(|fact| fact.0 == "peer" && fact.1 == 32 && fact.2 == "teardown")
            .count(),
        1
    );
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_slow_submission_has_no_catch_up_and_expires_cleanly() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":{
                "revision":1,
                "native_fresh_render_generations":{"self":12},
                "blocks":[block("self:slow","self","text","",true)]
            }})
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(800)).await;
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let state = Arc::new(OwnedSubmitterState::default());
    let mut owner = NativePresentationOwner::new_with_retry_policy_for_test(
        snapshot,
        CaptionRenderer::new_for_test().unwrap(),
        OwnedSubmitterProbe {
            state: state.clone(),
            fail_submit: false,
            fail_on_submission: None,
            submit_delay: Duration::from_millis(20),
        },
        Duration::from_millis(5),
        Duration::from_millis(200),
        2,
    );

    owner
        .run(&mut bridge, &test_logger("slow-no-catch-up").await)
        .await
        .unwrap();

    let audit = owner.fresh_retry_audit_for_test();
    let completions = audit
        .iter()
        .filter(|fact| fact.2 == "completed")
        .collect::<Vec<_>>();
    assert_eq!(completions.len(), 2);
    for pair in completions.windows(2) {
        assert!(pair[1].4 >= pair[0].4 + Duration::from_millis(25));
    }
    assert_eq!(
        state
            .operations
            .lock()
            .unwrap()
            .iter()
            .filter(|operation| operation.starts_with("submit"))
            .count(),
        3
    );
    assert!(!owner.runtime().redraw_requested());
    assert!(owner.resources_released());
    server.await.unwrap();
}

#[tokio::test]
async fn production_owner_releases_resources_on_terminal_submission_failure() {
    let (mut bridge, server) = connect_test_bridge().await;
    let state = Arc::new(OwnedSubmitterState::default());
    let submitter = OwnedSubmitterProbe {
        state: state.clone(),
        fail_submit: true,
        fail_on_submission: None,
        submit_delay: Duration::ZERO,
    };
    let mut owner = NativePresentationOwner::new(
        OverlayPresentationSnapshot::default(),
        CaptionRenderer::new_for_test().unwrap(),
        submitter,
    );
    let retry = owner.retry_handle();

    let failure = owner
        .run(&mut bridge, &test_logger("production-owner-failure").await)
        .await
        .unwrap_err();

    assert!(matches!(failure, RuntimeFailure::OpenVr(_)));
    assert!(owner.resources_released());
    assert!(!retry.request());
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    assert_eq!(state.operations.lock().unwrap().last(), Some(&"hide"));
    drop(bridge);
    let _ = server.await.unwrap();
}

#[tokio::test]
async fn runtime_hides_overlay_after_empty_state_stays_idle_past_delay() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let _ = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 1,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:1", "self", "hello", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        let _ = ws.next().await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 2,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": []
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(650)).await;

        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
    });

    let logger = test_logger("idle-hide").await;
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime
        .run_event_loop(&mut bridge, &renderer, &mut submitter, &logger)
        .await
        .unwrap();

    server.await.unwrap();

    assert!(submitter.visibility_changes.contains(&false));
}

#[tokio::test]
async fn runtime_cancels_pending_idle_hide_when_new_text_arrives() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let _ = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 1,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:1", "self", "hello", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        let _ = ws.next().await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 2,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": []
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(250)).await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 3,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:2", "self", "back again", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(650)).await;

        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
    });

    let logger = test_logger("idle-hide-cancel").await;
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime
        .run_event_loop(&mut bridge, &renderer, &mut submitter, &logger)
        .await
        .unwrap();

    server.await.unwrap();

    assert!(!submitter.visibility_changes.contains(&false));
}

#[tokio::test]
async fn runtime_shows_overlay_again_when_text_returns_after_idle_hide() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let _ = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 1,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:1", "self", "hello", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        let _ = ws.next().await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 2,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": []
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(650)).await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 3,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:2", "self", "visible again", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
    });

    let logger = test_logger("idle-hide-restore").await;
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime
        .run_event_loop(&mut bridge, &renderer, &mut submitter, &logger)
        .await
        .unwrap();

    server.await.unwrap();

    assert!(submitter
        .visibility_changes
        .windows(2)
        .any(|pair| pair == [false, true]));
}

#[tokio::test]
async fn runtime_submits_text_frame_before_revealing_overlay_after_idle_hide() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let _ = ws.next().await.unwrap().unwrap();
        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 1,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:1", "self", "hello", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        let _ = ws.next().await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 2,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": []
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(650)).await;

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 3,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [block("self:2", "self", "visible again", "", true)]
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        ws.send(Message::Text(
            json!({"type": "shutdown"}).to_string().into(),
        ))
        .await
        .unwrap();
    });

    let logger = test_logger("reveal-order-after-hide").await;
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let mut runtime = OverlayRuntime::new(snapshot);
    let mut submitter = RecordingSubmitter::default();

    runtime
        .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
        .await
        .unwrap();
    runtime
        .run_event_loop(&mut bridge, &renderer, &mut submitter, &logger)
        .await
        .unwrap();

    server.await.unwrap();

    let hide_index = submitter
        .operations
        .iter()
        .rposition(|operation| *operation == "hide")
        .expect("expected idle hide before reveal");
    assert_eq!(
        &submitter.operations[hide_index + 1..hide_index + 3],
        &["submit:text", "show"]
    );
}

#[tokio::test]
async fn bridge_client_authenticates_and_receives_initial_snapshot() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let auth = ws.next().await.unwrap().unwrap();
        let Message::Text(auth_text) = auth else {
            panic!("expected auth text frame");
        };
        let auth_payload: serde_json::Value = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(auth_payload["type"], "auth");
        assert_eq!(auth_payload["session_token"], "expected-token");

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 0,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [],
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    });

    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (_client, snapshot) = BridgeClient::connect(&manifest).await.unwrap();

    server.await.unwrap();
    assert!(snapshot.blocks.is_empty());
}

#[tokio::test]
async fn bridge_client_receives_runtime_logging_mode_updates() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        let auth = ws.next().await.unwrap().unwrap();
        let Message::Text(auth_text) = auth else {
            panic!("expected auth text frame");
        };
        let auth_payload: serde_json::Value = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(auth_payload["type"], "auth");

        ws.send(Message::Text(
            json!({
                "type": "snapshot",
                "payload": {
                    "revision": 0,
                    "calibration": OverlayPresentationCalibration::default(),
                    "blocks": [],
                }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        ws.send(Message::Text(
            json!({
                "type": "runtime_control",
                "payload": {"logging_mode": "detailed"},
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    });

    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{}", address);

    let (mut client, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
    assert!(snapshot.blocks.is_empty());

    let message = client.next_message().await.unwrap();

    assert!(matches!(
        message,
        puripuly_heart_overlay::BridgeIncoming::Control(control)
            if control.logging_mode == OverlayLoggingMode::Detailed
    ));
    server.await.unwrap();
}

#[tokio::test]
#[ignore = "child-process timing race under parallel cargo; covered by src/runtime.rs unit tests"]
async fn runtime_emits_snapshot_slot_correlation_and_overlay_visible_update_rendered_logs() {
    let output = run_overlay_binary_with_scripted_bridge(
        "slot-correlation-visible-update-rendered",
        json!({
            "revision": 1,
            "calibration": OverlayPresentationCalibration::default(),
            "blocks": [
                {
                    "id": "self:1",
                    "occupant_key": "self:1",
                    "appearance_seq": 1,
                    "channel": "self",
                    "block_variant": "finalized",
                    "primary_text": "hello",
                    "secondary_text": "",
                    "secondary_enabled": true,
                    "update_id": "upd-self-1",
                    "origin_wall_clock_ms": 1712345678901u64,
                    "session_scope": "session:self"
                }
            ]
        }),
        vec![
            BridgeAction::SendSnapshot(json!({
                "revision": 2,
                "calibration": OverlayPresentationCalibration::default(),
                "blocks": [
                    {
                        "id": "self:1",
                        "occupant_key": "self:1",
                        "appearance_seq": 1,
                        "channel": "self",
                        "block_variant": "finalized",
                        "primary_text": "hello again",
                        "secondary_text": "translated",
                        "secondary_enabled": true,
                        "update_id": "upd-self-2",
                        "origin_wall_clock_ms": 1712345678955u64,
                        "session_scope": "session:self"
                    }
                ]
            })),
            BridgeAction::WaitMs(200),
            BridgeAction::SendShutdown,
        ],
    )
    .await;

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("snapshot_slot_correlation"));
    assert!(stdout.contains("update_ids=[upd-self-2]"));
    assert!(stdout.contains("session_scope=session:self"));
    assert!(stdout.contains("presenter_order=0"));
    assert!(stdout.contains("slot_index=0"));
    assert!(stdout.contains("overlay_visible_update_applied"));
    assert!(stdout.contains("overlay_visible_update_rendered"));
}

#[tokio::test]
#[ignore = "child-process timing race under parallel cargo; covered by src/runtime.rs unit tests"]
async fn runtime_emits_two_row_window_closed_log_when_visible_window_collapses() {
    let output = run_overlay_binary_with_scripted_bridge(
        "two-row-window-closed",
        json!({
            "revision": 1,
            "calibration": OverlayPresentationCalibration::default(),
            "blocks": [
                {
                    "id": "self:1",
                    "occupant_key": "self:1",
                    "appearance_seq": 1,
                    "channel": "self",
                    "block_variant": "finalized",
                    "primary_text": "one",
                    "secondary_text": "",
                    "secondary_enabled": true,
                    "update_id": "upd-self-1",
                    "origin_wall_clock_ms": 1712345678901u64,
                    "session_scope": "session:self"
                },
                {
                    "id": "peer:2",
                    "occupant_key": "peer:2",
                    "appearance_seq": 2,
                    "channel": "peer",
                    "block_variant": "finalized",
                    "primary_text": "two",
                    "secondary_text": "",
                    "secondary_enabled": true,
                    "update_id": "upd-peer-2",
                    "origin_wall_clock_ms": 1712345678910u64,
                    "session_scope": "session:peer"
                }
            ]
        }),
        vec![
            BridgeAction::WaitMs(120),
            BridgeAction::SendSnapshot(json!({
                "revision": 2,
                "calibration": OverlayPresentationCalibration::default(),
                "blocks": [
                    {
                        "id": "self:1",
                        "occupant_key": "self:1",
                        "appearance_seq": 1,
                        "channel": "self",
                        "block_variant": "finalized",
                        "primary_text": "one",
                        "secondary_text": "",
                        "secondary_enabled": true,
                        "update_id": "upd-self-1",
                        "origin_wall_clock_ms": 1712345678901u64,
                        "session_scope": "session:self"
                    }
                ]
            })),
            BridgeAction::WaitMs(200),
            BridgeAction::SendShutdown,
        ],
    )
    .await;

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("two_row_window_closed"));
    assert!(stdout.contains("threshold_ms=500"));
    assert!(stdout.contains("too_brief_to_be_perceptibly_stable=true"));
}

#[test]
fn runtime_disconnect_failure_reason_is_stable() {
    assert_eq!(
        RuntimeFailure::RuntimeDisconnected.failure_reason(),
        "runtime_disconnected"
    );
}

#[test]
fn openvr_submission_uses_set_overlay_texture_for_rendered_frames() {
    let openvr = FakeOpenVr::default();
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let frame = renderer
        .render_blocks(vec![CaptionBlock::new("peer-1", "hello")])
        .unwrap();

    submit_texture(&openvr, &frame).unwrap();

    assert_eq!(openvr.last_call().as_deref(), Some("SetOverlayTexture"));
}

#[test]
fn check_startup_contract_reports_current_contract_version() {
    let output = Command::new(overlay_binary())
        .arg("--check-startup-contract")
        .output()
        .unwrap();

    assert!(output.status.success());
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["contract_version"], EXPECTED_CONTRACT_VERSION);
}

#[test]
fn validate_manifest_rejects_contract_version_mismatch() {
    let manifest = OverlayManifest {
        contract_version: EXPECTED_CONTRACT_VERSION + 1,
        ..test_manifest()
    };

    let error = validate_manifest(&manifest).unwrap_err();

    assert!(matches!(error, StartupError::ContractMismatch(_)));
}

#[tokio::test]
async fn run_with_manifest_reports_bridge_auth_failures_as_startup_errors() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _ = ws.next().await;
        ws.send(Message::Text(
            json!({"type": "auth_error"}).to_string().into(),
        ))
        .await
        .unwrap();
    });

    let log_dir = unique_log_dir("bridge-auth-failure");
    let exit_code = run_with_manifest(OverlayManifest {
        bridge_url: format!("ws://{}", address),
        log_dir,
        ..test_manifest()
    })
    .await;

    server.await.unwrap();
    assert_eq!(exit_code, StartupError::BridgeAuth("x".into()).exit_code());
}

#[test]
fn cli_requires_config_argument_or_supported_flags() {
    let output = Command::new(overlay_binary()).output().unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage:"));
}

#[test]
fn cli_emits_startup_failure_event_when_manifest_is_missing() {
    let missing_path = unique_temp_file("missing-manifest", "json");
    let output = Command::new(overlay_binary())
        .arg("--config")
        .arg(&missing_path)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stderr_events = parse_event_payloads(&output.stderr);
    assert!(stderr_events
        .iter()
        .any(|event| event["type"] == "startup_error"));
}
