use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::time::Duration;

use futures_util::FutureExt;
use serde_json::json;
use thiserror::Error;
use tokio::io::{self, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::time::{sleep_until, Instant};

use crate::bridge::{BridgeClient, BridgeError, BridgeIncoming, OverlayBridgeEvent};
use crate::logging::{OverlayLogger, OverlayLoggingMode};
use crate::manifest::{
    load_manifest, resolve_quiet_tail_profile_from_env, validate_manifest, OverlayManifest,
    QuietTailProfile, EXPECTED_CONTRACT_VERSION,
};
#[cfg(test)]
use crate::openvr::OpenVrError;
use crate::openvr::{
    format_openvr_visibility_api_call_log, perform_startup_preflight, FrameTimingSample,
    OpenVrEventClass, OpenVrOverlay, OpenVrRuntimeEvent, OpenVrStartupPreflightError,
    OverlayFrameSubmitter, SpatialReanchorOutcome,
};
use crate::presentation::{
    PresentationBackend, PresentationCause, PresentationCauseChannel, PresentationCauseKind,
    PresentationCauses, PresentationCorrelation, PresentationDiagnostics, ReadinessCancellation,
    ReadinessOutcome,
};
use crate::renderer::{
    CaptionBlock, CaptionBlockVariant, CaptionChannel, CaptionDebugOverlay, CaptionLayoutResult,
    CaptionPresentation, CaptionRenderer,
};
#[cfg(test)]
use crate::renderer::{RenderDiagnostics, StyleBucketSourceCount};
use crate::state::{
    NativeQuietTailEpisode, NativeQuietTailPhase, OverlayPresentationBlock,
    OverlayPresentationBlockVariant, OverlayPresentationSnapshot, OverlaySlot, OverlayState,
};

const EMPTY_OVERLAY_HIDE_DELAY: Duration = Duration::from_millis(500);
const TWO_ROW_WINDOW_STABILITY_THRESHOLD_MS: u64 = 500;
const PRESENTATION_DIAGNOSTIC_WRITE_TIMEOUT: Duration = Duration::from_millis(25);
const GPU_READINESS_OWNER_TIMEOUT: Duration = Duration::from_millis(50);
const MAX_IGNORED_MESSAGES_BEFORE_READINESS_POLL: usize = 8;
const MAX_OPENVR_EVENTS_PER_TURN: usize = 8;
const OPENVR_EVENT_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StartupError {
    #[error("manifest invalid: {0}")]
    Manifest(String),
    #[error("contract mismatch: {0}")]
    ContractMismatch(String),
    #[error("bridge auth failed: {0}")]
    BridgeAuth(String),
    #[error("SteamVR/OpenVR runtime is not installed")]
    SteamVrNotInstalled,
    #[error("SteamVR is not running")]
    SteamVrNotRunning,
    #[error("VR headset not found")]
    HmdNotFound,
    #[error("openvr init failed: {0}")]
    OpenVrInit(String),
    #[error("renderer init failed: {0}")]
    RendererInit(String),
    #[error("startup failed: {0}")]
    Other(String),
}

impl StartupError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::ContractMismatch(_) => 10,
            Self::BridgeAuth(_) => 12,
            Self::SteamVrNotInstalled | Self::SteamVrNotRunning | Self::HmdNotFound => 20,
            Self::OpenVrInit(_) => 20,
            Self::RendererInit(_) => 21,
            Self::Manifest(_) | Self::Other(_) => 1,
        }
    }

    pub fn failure_reason(&self) -> &'static str {
        match self {
            Self::Manifest(_) => "manifest_invalid",
            Self::ContractMismatch(_) => "contract_mismatch",
            Self::BridgeAuth(_) => "bridge_auth_failed",
            Self::SteamVrNotInstalled => "steamvr_not_installed",
            Self::SteamVrNotRunning => "steamvr_not_running",
            Self::HmdNotFound => "hmd_not_found",
            Self::OpenVrInit(_) => "openvr_init_failed",
            Self::RendererInit(_) => "renderer_init_failed",
            Self::Other(_) => "unknown",
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RuntimeFailure {
    #[error("runtime disconnected")]
    RuntimeDisconnected,
    #[error("runtime stopped")]
    Stopped,
    #[error("runtime bridge error: {0}")]
    Bridge(String),
    #[error("renderer draw failed: {0}")]
    Render(String),
    #[error("openvr submit failed: {0}")]
    OpenVr(String),
    #[error("GPU readiness timed out")]
    ReadinessTimedOut,
    #[error("GPU readiness cancelled")]
    ReadinessCancelled,
    #[error("GPU readiness failed")]
    ReadinessFailed,
}

impl RuntimeFailure {
    pub fn failure_reason(&self) -> &'static str {
        match self {
            Self::RuntimeDisconnected => "runtime_disconnected",
            Self::Stopped => "stopped",
            Self::ReadinessTimedOut | Self::ReadinessCancelled | Self::ReadinessFailed => {
                "renderer_init_failed"
            }
            Self::Bridge(_) | Self::Render(_) | Self::OpenVr(_) => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationRuntime {
    ready: bool,
    first_texture_submitted: bool,
    overlay_visible: bool,
    last_submitted_had_self: bool,
    stopped: bool,
    state: OverlayState,
    redraw_requested: bool,
    hide_deadline: Option<Instant>,
    pending_peer_first_emit_logs: Vec<String>,
    pending_peer_first_render_ids: HashSet<String>,
    pending_visible_update_rows: Vec<DiagnosticRow>,
    pending_visible_update_render_slot_orders: HashSet<u64>,
    seen_peer_overlay_ids: HashSet<String>,
    last_snapshot_slot_correlation_signature: Option<String>,
    last_submitted_visible_rows: HashMap<u64, String>,
    two_row_window: Option<TwoRowWindowState>,
    last_frame_timing_sampled_at: Option<Instant>,
    presentation_diagnostics: PresentationDiagnostics,
    pending_logical_revision_acceptance: bool,
    last_logical_caption_identity: LogicalCaptionIdentity,
    last_presentation_correlation: Option<PresentationCorrelation>,
    last_presentation_backend: Option<PresentationBackend>,
    pending_presentation_causes: PresentationCauses,
    spatial_lock: SpatialLockState,
    pending_spatial_diagnostics: Vec<SpatialDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotApplyOutcome {
    Applied {
        incoming_revision: u64,
        current_revision: u64,
        visual_changed: bool,
        redraw_requested: bool,
    },
    Ignored {
        incoming_revision: u64,
        current_revision: u64,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct DiagnosticRow {
    id: String,
    occupant_key: String,
    channel: String,
    block_variant: OverlayPresentationBlockVariant,
    update_id: Option<String>,
    origin_wall_clock_ms: Option<u64>,
    session_scope: Option<String>,
    presenter_order: usize,
    slot_order: u64,
    slot_index: usize,
    slot_anchor_top_px: f32,
    primary_text: String,
    secondary_text: String,
    secondary_enabled: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct RenderedDiagnosticRow {
    row: DiagnosticRow,
    bounds: crate::renderer::BlockBounds,
    visual_bounds: crate::renderer::VisualBounds,
    secondary_present: bool,
    truncated_secondary: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TwoRowWindowState {
    started_at: Instant,
    slot_signature: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct LogicalCaptionIdentity(Vec<LogicalCaptionBlockIdentity>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct LogicalCaptionBlockIdentity {
    slot_index: usize,
    channel: String,
    block_variant: OverlayPresentationBlockVariant,
    primary_text: String,
    secondary_text: String,
    secondary_enabled: bool,
    primary_language: Option<String>,
    secondary_language: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct FrameStageDurations {
    receive_to_apply_us: Option<u128>,
    render_duration_us: Option<u128>,
    receive_to_submit_us: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialReanchorReason {
    InitialVisible,
    NewTurn,
    ModeEntered,
    PlacementCalibrationChanged,
}

impl SpatialReanchorReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::InitialVisible => "initial_visible",
            Self::NewTurn => "new_turn",
            Self::ModeEntered => "mode_entered",
            Self::PlacementCalibrationChanged => "placement_calibration_changed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingSpatialReanchor {
    reason: SpatialReanchorReason,
    requested_revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SpatialDiagnostic {
    Info(String),
    Warning(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct SpatialLockState {
    active: bool,
    seen_turn_ids: HashSet<String>,
    pending_reanchor: Option<PendingSpatialReanchor>,
}

impl SpatialLockState {
    fn from_initial_snapshot(
        snapshot: &OverlayPresentationSnapshot,
    ) -> (Self, Vec<SpatialDiagnostic>) {
        if snapshot.calibration.anchor != "spatial_locked" {
            return (Self::default(), Vec::new());
        }
        let seen_turn_ids = drawable_turn_ids(snapshot);
        let mut state = Self {
            active: true,
            seen_turn_ids,
            pending_reanchor: None,
        };
        let mut diagnostics = vec![SpatialDiagnostic::Info(format!(
            "spatial_lock_mode_entered revision={}",
            snapshot.revision
        ))];
        if !state.seen_turn_ids.is_empty() {
            state.request_reanchor(
                SpatialReanchorReason::InitialVisible,
                snapshot.revision,
                &mut diagnostics,
            );
        }
        (state, diagnostics)
    }

    fn apply_snapshot_transition(
        &mut self,
        previous_calibration: &crate::state::OverlayPresentationCalibration,
        snapshot: &OverlayPresentationSnapshot,
    ) -> Vec<SpatialDiagnostic> {
        let mut diagnostics = Vec::new();
        let current_spatial = snapshot.calibration.anchor == "spatial_locked";
        match (self.active, current_spatial) {
            (false, false) => {}
            (false, true) => {
                self.active = true;
                self.seen_turn_ids = drawable_turn_ids(snapshot);
                diagnostics.push(SpatialDiagnostic::Info(format!(
                    "spatial_lock_mode_entered revision={}",
                    snapshot.revision
                )));
                if !self.seen_turn_ids.is_empty() {
                    self.request_reanchor(
                        SpatialReanchorReason::ModeEntered,
                        snapshot.revision,
                        &mut diagnostics,
                    );
                }
            }
            (true, false) => {
                self.active = false;
                self.seen_turn_ids.clear();
                self.pending_reanchor = None;
                diagnostics.push(SpatialDiagnostic::Info(format!(
                    "spatial_lock_mode_exited revision={}",
                    snapshot.revision
                )));
            }
            (true, true) => {
                let visible_ids = drawable_turn_ids(snapshot);
                let first_drawable = self.seen_turn_ids.is_empty() && !visible_ids.is_empty();
                let has_new_turn = visible_ids
                    .iter()
                    .any(|block_id| !self.seen_turn_ids.contains(block_id));
                self.seen_turn_ids.extend(visible_ids);
                let placement_changed = previous_calibration.offset_x
                    != snapshot.calibration.offset_x
                    || previous_calibration.offset_y != snapshot.calibration.offset_y
                    || previous_calibration.distance != snapshot.calibration.distance;
                let reason = if placement_changed {
                    Some(SpatialReanchorReason::PlacementCalibrationChanged)
                } else if first_drawable {
                    Some(SpatialReanchorReason::InitialVisible)
                } else if has_new_turn {
                    Some(SpatialReanchorReason::NewTurn)
                } else {
                    None
                };
                if let Some(reason) = reason {
                    self.request_reanchor(reason, snapshot.revision, &mut diagnostics);
                }
            }
        }
        diagnostics
    }

    fn request_reanchor(
        &mut self,
        reason: SpatialReanchorReason,
        revision: u64,
        diagnostics: &mut Vec<SpatialDiagnostic>,
    ) {
        if self.pending_reanchor.is_some() {
            return;
        }
        self.pending_reanchor = Some(PendingSpatialReanchor {
            reason,
            requested_revision: revision,
        });
        diagnostics.push(SpatialDiagnostic::Info(format!(
            "spatial_reanchor_requested reason={} revision={revision}",
            reason.as_str()
        )));
    }

    fn pending(&self) -> Option<PendingSpatialReanchor> {
        self.pending_reanchor
    }

    fn take_pending(&mut self) -> Option<PendingSpatialReanchor> {
        self.pending_reanchor.take()
    }
}

fn drawable_turn_ids(snapshot: &OverlayPresentationSnapshot) -> HashSet<String> {
    snapshot
        .blocks
        .iter()
        .filter(|block| {
            !block.primary_text.trim().is_empty()
                || (block.secondary_enabled && !block.secondary_text.trim().is_empty())
        })
        .map(|block| block.id.clone())
        .collect()
}

pub type OverlayRuntime = PresentationRuntime;

impl PresentationRuntime {
    fn configure_retry_profile(&mut self, retry_profile: &'static str) {
        self.presentation_diagnostics
            .configure_retry_profile(retry_profile);
    }
    pub fn new(snapshot: OverlayPresentationSnapshot) -> Self {
        let seeded_peer_ids = peer_overlay_first_emit_block_ids_from_snapshot(&snapshot);
        let seen_peer_overlay_ids = seeded_peer_ids.iter().cloned().collect::<HashSet<_>>();
        let (spatial_lock, pending_spatial_diagnostics) =
            SpatialLockState::from_initial_snapshot(&snapshot);
        let mut runtime = Self {
            ready: false,
            first_texture_submitted: false,
            overlay_visible: false,
            last_submitted_had_self: false,
            stopped: false,
            state: OverlayState::default(),
            redraw_requested: false,
            hide_deadline: None,
            pending_peer_first_emit_logs: seeded_peer_ids.clone(),
            pending_peer_first_render_ids: seeded_peer_ids.into_iter().collect(),
            pending_visible_update_rows: Vec::new(),
            pending_visible_update_render_slot_orders: HashSet::new(),
            seen_peer_overlay_ids,
            last_snapshot_slot_correlation_signature: None,
            last_submitted_visible_rows: HashMap::new(),
            two_row_window: None,
            last_frame_timing_sampled_at: None,
            presentation_diagnostics: PresentationDiagnostics::new(),
            pending_logical_revision_acceptance: true,
            last_logical_caption_identity: LogicalCaptionIdentity::default(),
            last_presentation_correlation: None,
            last_presentation_backend: None,
            pending_presentation_causes: {
                let mut causes = PresentationCauses::default();
                causes.insert(PresentationCause {
                    kind: PresentationCauseKind::Startup,
                    channel: None,
                    trigger_generation: None,
                });
                causes
            },
            spatial_lock,
            pending_spatial_diagnostics,
        };
        if runtime.state.seed_snapshot(&snapshot) {
            runtime.redraw_requested = true;
        }
        runtime.last_logical_caption_identity = logical_caption_identity(runtime.state());
        runtime
    }

    pub fn state(&self) -> &OverlayState {
        &self.state
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    pub fn mark_ready_for_test(&mut self) {
        self.ready = true;
    }

    pub fn ready_sent(&self) -> bool {
        self.ready
    }

    pub async fn submit_first_texture_for_test(&mut self) -> Result<(), RuntimeFailure> {
        self.first_texture_submitted = true;
        self.ready = true;
        Ok(())
    }

    pub fn apply_snapshot(
        &mut self,
        snapshot: OverlayPresentationSnapshot,
    ) -> SnapshotApplyOutcome {
        let current_revision = self.state.snapshot().revision;
        if snapshot.revision <= current_revision {
            return SnapshotApplyOutcome::Ignored {
                incoming_revision: snapshot.revision,
                current_revision,
            };
        }

        for block_id in peer_overlay_first_emit_block_ids_from_snapshot(&snapshot) {
            if self.seen_peer_overlay_ids.insert(block_id.clone()) {
                self.pending_peer_first_emit_logs.push(block_id.clone());
                self.pending_peer_first_render_ids.insert(block_id);
            }
        }

        let previous_calibration = self.state.calibration().clone();
        let visual_changed = self.state.apply_snapshot(&snapshot);
        self.pending_spatial_diagnostics.extend(
            self.spatial_lock
                .apply_snapshot_transition(&previous_calibration, self.state.snapshot()),
        );
        let logical_caption_identity = logical_caption_identity(self.state());
        if logical_caption_identity != self.last_logical_caption_identity {
            self.pending_logical_revision_acceptance = true;
            self.last_logical_caption_identity = logical_caption_identity;
        }
        if visual_changed {
            self.redraw_requested = true;
            self.pending_presentation_causes.insert(PresentationCause {
                kind: PresentationCauseKind::SceneUpdate,
                channel: None,
                trigger_generation: Some(snapshot.revision),
            });
        }
        let previous_visible_rows = self.last_submitted_visible_rows.clone();
        let diagnostic_rows = collect_diagnostic_rows(self.state());
        let visible_update_rows = diagnostic_rows
            .into_iter()
            .filter(|row| {
                previous_visible_rows
                    .get(&row.slot_order)
                    .is_some_and(|previous| previous != &diagnostic_row_signature(row))
            })
            .collect::<Vec<_>>();
        self.pending_visible_update_render_slot_orders = visible_update_rows
            .iter()
            .map(|row| row.slot_order)
            .collect();
        self.pending_visible_update_rows = visible_update_rows;
        SnapshotApplyOutcome::Applied {
            incoming_revision: snapshot.revision,
            current_revision: self.state.snapshot().revision,
            visual_changed,
            redraw_requested: self.redraw_requested,
        }
    }

    pub fn redraw_requested(&self) -> bool {
        self.redraw_requested
    }

    pub fn clear_redraw_flag(&mut self) {
        self.redraw_requested = false;
    }

    pub fn request_native_presentation_retry(&mut self) -> bool {
        if self.stopped {
            return false;
        }
        self.redraw_requested = true;
        self.pending_presentation_causes.insert(PresentationCause {
            kind: PresentationCauseKind::ExternalRetry,
            channel: None,
            trigger_generation: None,
        });
        true
    }

    fn request_fresh_presentation_retry(
        &mut self,
        channel: FreshRetryChannel,
        trigger_generation: u64,
    ) -> bool {
        if self.stopped {
            return false;
        }
        self.redraw_requested = true;
        self.pending_presentation_causes.insert(PresentationCause {
            kind: PresentationCauseKind::NativeFreshRetry,
            channel: Some(match channel {
                FreshRetryChannel::SelfChannel => PresentationCauseChannel::SelfChannel,
                FreshRetryChannel::Peer => PresentationCauseChannel::Peer,
            }),
            trigger_generation: Some(trigger_generation),
        });
        true
    }

    fn retain_failed_presentation_causes(&mut self, correlation: PresentationCorrelation) {
        self.pending_presentation_causes
            .merge(correlation.logical_causes);
    }

    fn apply_runtime_logging_mode(
        &mut self,
        logger: &OverlayLogger,
        mode: OverlayLoggingMode,
    ) -> bool {
        let was_detailed = logger.is_detailed();
        logger.set_mode(mode);
        let is_detailed = logger.is_detailed();
        let changed = was_detailed != is_detailed;
        if changed {
            self.redraw_requested = true;
            self.pending_presentation_causes.insert(PresentationCause {
                kind: PresentationCauseKind::RuntimeControl,
                channel: None,
                trigger_generation: None,
            });
        }
        changed
    }

    fn runtime_logging_mode_would_change(
        &self,
        logger: &OverlayLogger,
        mode: OverlayLoggingMode,
    ) -> bool {
        logger.is_detailed() != matches!(mode, OverlayLoggingMode::Detailed)
    }

    async fn emit_snapshot_slot_correlation_if_changed(
        &mut self,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let rows = collect_diagnostic_rows(self.state());
        let signature = snapshot_slot_correlation_signature(self.state(), &rows);
        let should_log = match &self.last_snapshot_slot_correlation_signature {
            Some(previous) => previous != &signature,
            None => !rows.is_empty(),
        };
        self.last_snapshot_slot_correlation_signature = Some(signature);
        if should_log {
            log_runtime_info(
                logger,
                format_snapshot_slot_correlation_log(self.state(), &rows),
            )
            .await?;
        }
        Ok(())
    }

    async fn emit_pending_visible_update_applied_diagnostics(
        &mut self,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let rows = std::mem::take(&mut self.pending_visible_update_rows);
        for row in rows {
            log_runtime_info(
                logger,
                format_overlay_visible_update_applied_log(self.state.snapshot().revision, &row),
            )
            .await?;
        }
        Ok(())
    }

    async fn emit_visible_update_rendered_diagnostics(
        &mut self,
        logger: &OverlayLogger,
        rendered_rows: &[RenderedDiagnosticRow],
    ) -> Result<(), RuntimeFailure> {
        let mut rendered_slot_orders = Vec::new();
        for rendered in rendered_rows {
            if !self
                .pending_visible_update_render_slot_orders
                .contains(&rendered.row.slot_order)
            {
                continue;
            }
            rendered_slot_orders.push(rendered.row.slot_order);
            log_runtime_info(
                logger,
                format_overlay_visible_update_rendered_log(
                    self.state.snapshot().revision,
                    rendered,
                ),
            )
            .await?;
        }
        for slot_order in rendered_slot_orders {
            self.pending_visible_update_render_slot_orders
                .remove(&slot_order);
        }
        Ok(())
    }

    async fn note_submitted_visible_rows(
        &mut self,
        logger: &OverlayLogger,
        rendered_rows: &[RenderedDiagnosticRow],
        submitted_at: Instant,
    ) -> Result<(), RuntimeFailure> {
        self.update_two_row_window(logger, rendered_rows, submitted_at)
            .await?;
        self.last_submitted_visible_rows = rendered_rows
            .iter()
            .map(|rendered| {
                (
                    rendered.row.slot_order,
                    diagnostic_row_signature(&rendered.row),
                )
            })
            .collect();
        Ok(())
    }

    async fn update_two_row_window(
        &mut self,
        logger: &OverlayLogger,
        rendered_rows: &[RenderedDiagnosticRow],
        submitted_at: Instant,
    ) -> Result<(), RuntimeFailure> {
        let next_window = if rendered_rows.len() == 2 {
            Some(TwoRowWindowState {
                started_at: submitted_at,
                slot_signature: two_row_window_slot_signature(rendered_rows),
            })
        } else {
            None
        };

        match (&mut self.two_row_window, next_window) {
            (Some(previous), Some(next)) if previous.slot_signature == next.slot_signature => {
                let _ = next;
            }
            (Some(previous), Some(next)) => {
                log_runtime_info(
                    logger,
                    format_two_row_window_closed_log(
                        self.state.snapshot().revision,
                        previous,
                        submitted_at,
                    ),
                )
                .await?;
                self.two_row_window = Some(next);
            }
            (Some(previous), None) => {
                log_runtime_info(
                    logger,
                    format_two_row_window_closed_log(
                        self.state.snapshot().revision,
                        previous,
                        submitted_at,
                    ),
                )
                .await?;
                self.two_row_window = None;
            }
            (None, Some(next)) => {
                self.two_row_window = Some(next);
            }
            (None, None) => {}
        }

        Ok(())
    }

    pub async fn handle_event(&mut self, event: OverlayBridgeEvent) -> Result<(), RuntimeFailure> {
        match event {
            OverlayBridgeEvent::Shutdown => {
                self.shutdown_presentation();
                Ok(())
            }
        }
    }

    pub async fn handle_bridge_loss_for_test(&mut self) -> Result<(), RuntimeFailure> {
        let was_ready = self.ready;
        self.shutdown_presentation();
        if was_ready {
            Err(RuntimeFailure::RuntimeDisconnected)
        } else {
            Ok(())
        }
    }

    fn shutdown_presentation(&mut self) {
        self.stopped = true;
        self.redraw_requested = false;
        self.hide_deadline = None;
        self.overlay_visible = false;
        self.first_texture_submitted = false;
        self.presentation_diagnostics.shutdown();
        self.last_presentation_correlation = None;
        self.last_presentation_backend = None;
        self.pending_presentation_causes = PresentationCauses::default();
    }

    pub async fn emit_ready(
        &mut self,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let ready_event = json!({
            "type": "overlay_ready",
            "capabilities": {
                "native_presentation_retry": {
                    "version": 1,
                    "ownership": "exclusive"
                }
            }
        });
        bridge
            .send_json(ready_event.clone())
            .await
            .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
        logger
            .emit_stdout_event(&ready_event)
            .await
            .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
        logger
            .info("overlay_ready_sent")
            .await
            .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
        self.ready = true;
        Ok(())
    }

    pub async fn submit_frame_if_needed<S: OverlayFrameSubmitter>(
        &mut self,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        self.submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, None, None, false)
            .await
            .map(|_| ())
    }

    async fn submit_frame_if_needed_with_timing<S: OverlayFrameSubmitter>(
        &mut self,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
        snapshot_received_at: Option<Instant>,
        receive_to_apply_us: Option<u128>,
        preemptible: bool,
    ) -> Result<FrameCycleOutcome, RuntimeFailure> {
        if self.stopped {
            return Err(RuntimeFailure::Stopped);
        }
        if self.first_texture_submitted && !self.redraw_requested {
            return Ok(FrameCycleOutcome::NoWork);
        }

        let presentation_backend = renderer.presentation_backend();
        let openvr_adapter_identity = renderer.openvr_adapter_identity();
        self.presentation_diagnostics.configure_adapter_handoff(
            openvr_adapter_identity,
            renderer.adapter_identity(),
            renderer.adapter_match(openvr_adapter_identity),
        );
        let scene_generation = self.state.snapshot().revision;
        let presentation_causes = std::mem::take(&mut self.pending_presentation_causes);
        if self.pending_logical_revision_acceptance {
            self.presentation_diagnostics.accept_logical_revision(
                presentation_backend,
                scene_generation,
                presentation_causes.clone(),
            );
            self.pending_logical_revision_acceptance = false;
        }
        let presentation_correlation = self
            .presentation_diagnostics
            .begin_presentation(scene_generation, presentation_causes)
            .expect("active presentation diagnostics owner");
        let prepare_started = Instant::now();
        renderer.set_presentation(CaptionPresentation {
            background_alpha: self.state.calibration().background_alpha,
            text_scale: self.state.calibration().text_scale,
        });
        openvr
            .apply_calibration(self.state.calibration())
            .map_err(|error| RuntimeFailure::OpenVr(error.to_string()))?;
        let detailed_logging = logger.is_detailed();
        let visual_debug_overlays = false;
        let blocks = self.caption_blocks_for_render(visual_debug_overlays);
        let mut cpu_prepare_us = duration_us(prepare_started.elapsed());
        self.emit_pending_peer_overlay_first_emit_hooks(logger)
            .await?;
        let prepare_resumed = Instant::now();
        let has_drawable_text = blocks.iter().any(CaptionBlock::has_drawable_text);
        let debug_overlay = debug_overlay_for_frame(
            visual_debug_overlays,
            self.state.snapshot().revision,
            &blocks,
        );
        let peer_overlay_first_render_ids = peer_overlay_first_render_block_ids_from_caption_blocks(
            &blocks,
            &self.pending_peer_first_render_ids,
        );
        cpu_prepare_us = cpu_prepare_us.saturating_add(duration_us(prepare_resumed.elapsed()));
        if has_drawable_text {
            if let Some(actual_visible) = openvr.observed_overlay_visible() {
                self.overlay_visible = actual_visible;
            }
        }
        let overlay_visible_before = self.overlay_visible;
        let should_show_after_submit = has_drawable_text && !self.overlay_visible;
        let hide_deadline_was_active = self.hide_deadline.is_some();
        let last_submitted_visible_row_count = self.last_submitted_visible_rows.len();
        if has_drawable_text {
            self.hide_deadline = None;
        } else if self.first_texture_submitted
            && self.overlay_visible
            && self.hide_deadline.is_none()
        {
            self.hide_deadline = Some(Instant::now() + EMPTY_OVERLAY_HIDE_DELAY);
        }
        let render_started = Instant::now();
        let render_result = if blocks.is_empty() {
            renderer.render_empty_frame()
        } else {
            renderer.render_blocks_with_debug_overlay(blocks, debug_overlay)
        };
        let cpu_render_us = duration_us(render_started.elapsed());
        self.presentation_diagnostics.record_render_return(
            presentation_correlation,
            presentation_backend,
            render_result.is_ok(),
            cpu_prepare_us,
            cpu_render_us,
        );
        let frame = match render_result {
            Ok(frame) => frame,
            Err(error) => {
                self.retain_failed_presentation_causes(presentation_correlation);
                self.emit_pending_presentation_diagnostics(logger).await?;
                return Err(RuntimeFailure::Render(error.to_string()));
            }
        };
        let render_duration_us = detailed_logging.then_some(u128::from(cpu_render_us));
        let self_block_count = visible_self_block_count(frame.layout());
        let fully_transparent = frame.is_fully_transparent();
        let rendered_diagnostic_rows =
            collect_rendered_diagnostic_rows(self.state(), frame.layout());
        if !peer_overlay_first_render_ids.is_empty() {
            log_runtime_info(
                logger,
                format_peer_first_render_visibility_checkpoint_log(
                    self.state.snapshot().revision,
                    &peer_overlay_first_render_ids,
                    has_drawable_text,
                    overlay_visible_before,
                    should_show_after_submit,
                    hide_deadline_was_active,
                    self.first_texture_submitted,
                    self.redraw_requested,
                    frame.layout().visible_blocks.len(),
                    self_block_count,
                    fully_transparent,
                ),
            )
            .await?;
            if has_drawable_text
                && overlay_visible_before
                && !should_show_after_submit
                && !hide_deadline_was_active
                && last_submitted_visible_row_count == 0
            {
                log_runtime_warn(
                    logger,
                    format_peer_first_render_visibility_desync_suspected_log(
                        self.state.snapshot().revision,
                        &peer_overlay_first_render_ids,
                        overlay_visible_before,
                        should_show_after_submit,
                        hide_deadline_was_active,
                        self.first_texture_submitted,
                        self.redraw_requested,
                        last_submitted_visible_row_count,
                    ),
                )
                .await?;
                log_runtime_info(
                    logger,
                    format_openvr_visibility_api_call_log(
                        true,
                        overlay_visible_before,
                        "SkippedByRuntimeCachedVisibleState",
                        self.overlay_visible,
                    ),
                )
                .await?;
            }
        }
        self.emit_visible_update_rendered_diagnostics(logger, &rendered_diagnostic_rows)
            .await?;
        let submit_started = if detailed_logging {
            Some(Instant::now())
        } else {
            None
        };
        let readiness_cancellation = ReadinessCancellation::default();
        let readiness_started = Instant::now();
        if self.stopped {
            readiness_cancellation.cancel();
        }
        let mut readiness =
            Box::pin(renderer.prepare_frame_for_submission(&readiness_cancellation));
        let mut pending_message = None;
        let readiness_deadline = Instant::now() + GPU_READINESS_OWNER_TIMEOUT;
        let mut ignored_message_count = 0usize;
        let readiness_outcome = if preemptible {
            loop {
                tokio::select! {
                    biased;
                    message = bridge.next_message() => {
                        let ignored = matches!(message, Ok(BridgeIncoming::Heartbeat)) || matches!(
                            &message,
                            Ok(BridgeIncoming::Control(control))
                                if !self.runtime_logging_mode_would_change(
                                    logger,
                                    control.logging_mode,
                                )
                        );
                        if ignored {
                            if Instant::now() >= readiness_deadline {
                                break ReadinessOutcome::TimedOut;
                            }
                            ignored_message_count += 1;
                            if ignored_message_count >= MAX_IGNORED_MESSAGES_BEFORE_READINESS_POLL {
                                ignored_message_count = 0;
                                if let Some(outcome) = readiness.as_mut().now_or_never() {
                                    break outcome;
                                }
                            }
                            continue;
                        }
                        readiness_cancellation.cancel();
                        pending_message = Some(message);
                        break readiness.await;
                    }
                    _ = sleep_until(readiness_deadline) => break ReadinessOutcome::TimedOut,
                    outcome = &mut readiness => break outcome,
                }
            }
        } else {
            readiness.await
        };
        self.presentation_diagnostics.record_readiness(
            presentation_correlation,
            presentation_backend,
            readiness_outcome,
            duration_us(readiness_started.elapsed()),
        );
        if readiness_outcome != ReadinessOutcome::Ready {
            self.retain_failed_presentation_causes(presentation_correlation);
            if readiness_outcome == ReadinessOutcome::Cancelled && pending_message.is_some() {
                if let Some(pending) = self.spatial_lock.pending() {
                    self.pending_spatial_diagnostics
                        .push(SpatialDiagnostic::Info(format!(
                            "spatial_reanchor_deferred_by_preemption reason={} revision={}",
                            pending.reason.as_str(),
                            pending.requested_revision
                        )));
                }
                return Ok(FrameCycleOutcome::Preempted(
                    pending_message.expect("cancelled readiness has pending message"),
                ));
            }
            self.emit_pending_presentation_diagnostics(logger).await?;
            let failure = match readiness_outcome {
                ReadinessOutcome::TimedOut => RuntimeFailure::ReadinessTimedOut,
                ReadinessOutcome::Cancelled => RuntimeFailure::ReadinessCancelled,
                ReadinessOutcome::Failed => RuntimeFailure::ReadinessFailed,
                ReadinessOutcome::Ready => unreachable!(),
            };
            return Err(failure);
        }
        if has_drawable_text {
            if let Some(pending) = self.spatial_lock.take_pending() {
                let reanchor_result = openvr.reanchor_spatial_locked();
                match reanchor_result {
                    Ok(SpatialReanchorOutcome::Applied) => {
                        self.pending_spatial_diagnostics
                            .push(SpatialDiagnostic::Info(format!(
                                "spatial_reanchor_applied reason={} revision={scene_generation}",
                                pending.reason.as_str()
                            )));
                    }
                    Ok(SpatialReanchorOutcome::PoseUnavailable) => {
                        self.pending_spatial_diagnostics
                            .push(SpatialDiagnostic::Warning(format!(
                                "spatial_reanchor_pose_unavailable reason={} revision={scene_generation}",
                                pending.reason.as_str()
                            )));
                    }
                    Err(error) => {
                        self.retain_failed_presentation_causes(presentation_correlation);
                        self.emit_pending_presentation_diagnostics(logger).await?;
                        return Err(RuntimeFailure::OpenVr(error.to_string()));
                    }
                }
            }
        }
        self.presentation_diagnostics
            .record_submission_attempt(presentation_correlation, presentation_backend);
        let submission_started = Instant::now();
        let submission_result = openvr.submit_frame(&frame);
        self.presentation_diagnostics.record_submission_return(
            presentation_correlation,
            presentation_backend,
            submission_result.is_ok(),
            duration_us(submission_started.elapsed()),
        );
        if let Err(error) = submission_result {
            self.emit_pending_spatial_diagnostics(logger).await;
            self.retain_failed_presentation_causes(presentation_correlation);
            self.emit_pending_presentation_diagnostics(logger).await?;
            return Err(RuntimeFailure::OpenVr(error.to_string()));
        }
        let submit_duration_us = submit_started.map(|start| start.elapsed().as_micros());
        if should_show_after_submit {
            let visibility_result = openvr.set_overlay_visible(true);
            let visibility_succeeded = visibility_result.is_ok();
            if visibility_succeeded {
                self.overlay_visible = true;
            }
            self.presentation_diagnostics.record_visibility(
                presentation_correlation,
                presentation_backend,
                true,
                self.overlay_visible,
                visibility_succeeded,
            );
            if let Err(error) = visibility_result {
                self.emit_pending_presentation_diagnostics(logger).await?;
                return Err(RuntimeFailure::OpenVr(error.to_string()));
            }
            if let Some(message) = openvr.take_visibility_api_call_log() {
                log_runtime_info(logger, message).await?;
            }
            log_runtime_info(
                logger,
                "overlay_visibility_changed visible=true reason=frame_submit_text_visible"
                    .to_string(),
            )
            .await?;
        }
        if !should_show_after_submit {
            let desired_runtime_visible = has_drawable_text || self.hide_deadline.is_some();
            self.presentation_diagnostics.record_visibility(
                presentation_correlation,
                presentation_backend,
                desired_runtime_visible,
                self.overlay_visible,
                desired_runtime_visible == self.overlay_visible,
            );
        }
        self.emit_pending_spatial_diagnostics(logger).await;
        self.last_presentation_correlation = Some(presentation_correlation);
        self.last_presentation_backend = Some(presentation_backend);
        if detailed_logging {
            self.sample_and_log_frame_timing(
                openvr,
                logger,
                self.state.snapshot().revision,
                submit_duration_us,
                presentation_backend,
            )
            .await?;
        }
        self.emit_pending_presentation_diagnostics(logger).await?;
        self.note_submitted_visible_rows(logger, &rendered_diagnostic_rows, Instant::now())
            .await?;
        self.emit_peer_overlay_first_render_hooks(logger, peer_overlay_first_render_ids)
            .await?;
        if detailed_logging {
            let stage_durations = FrameStageDurations {
                receive_to_apply_us,
                render_duration_us,
                receive_to_submit_us: snapshot_received_at.map(|start| start.elapsed().as_micros()),
            };
            log_runtime_info(
                logger,
                format_frame_submitted_log(
                    frame.layout(),
                    self.state.snapshot().revision,
                    fully_transparent,
                    overlay_visible_before,
                    self.overlay_visible,
                    should_show_after_submit,
                    submit_duration_us,
                    &rendered_diagnostic_rows,
                    stage_durations,
                ),
            )
            .await?;
        }
        self.last_submitted_had_self = self_block_count > 0;
        self.redraw_requested = false;

        if !self.first_texture_submitted {
            logger
                .info("first_texture_submitted")
                .await
                .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
            self.first_texture_submitted = true;
            self.emit_ready(bridge, logger).await?;
        }

        Ok(FrameCycleOutcome::Submitted)
    }

    pub async fn run_event_loop<S: OverlayFrameSubmitter>(
        &mut self,
        bridge: &mut BridgeClient,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let mut pending_message = None;
        loop {
            let hide_deadline = self.hide_deadline;
            let message = if let Some(message) = pending_message.take() {
                Some(message)
            } else {
                tokio::select! {
                    _ = sleep_until(hide_deadline.unwrap_or_else(Instant::now)), if hide_deadline.is_some() => {
                        self.handle_hide_deadline(openvr, logger).await?;
                        None
                    }
                    message = bridge.next_message() => Some(message)
                }
            };
            if let Some(message) = message {
                let (continue_running, preempted_message) = self
                    .handle_bridge_message(message, renderer, openvr, bridge, logger)
                    .await?;
                if !continue_running {
                    return Ok(());
                }
                pending_message = preempted_message;
            }
        }
    }

    pub async fn submit_initial_frame_message_aware<S: OverlayFrameSubmitter>(
        &mut self,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let mut pending_message = match self
            .submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, None, None, true)
            .await?
        {
            FrameCycleOutcome::Preempted(message) => Some(message),
            FrameCycleOutcome::Submitted | FrameCycleOutcome::NoWork => None,
        };
        while let Some(message) = pending_message.take() {
            let (continue_running, next_message) = self
                .handle_bridge_message(message, renderer, openvr, bridge, logger)
                .await?;
            if !continue_running {
                return Ok(());
            }
            pending_message = next_message;
        }
        Ok(())
    }

    async fn handle_bridge_message<S: OverlayFrameSubmitter>(
        &mut self,
        message: Result<BridgeIncoming, BridgeError>,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(bool, Option<Result<BridgeIncoming, BridgeError>>), RuntimeFailure> {
        match message {
            Ok(BridgeIncoming::Heartbeat) => Ok((true, None)),
            Ok(BridgeIncoming::Control(control)) => {
                if self.apply_runtime_logging_mode(logger, control.logging_mode) {
                    let pending = self
                        .submit_frame_if_needed_with_timing(
                            renderer, openvr, bridge, logger, None, None, true,
                        )
                        .await?;
                    return Ok((true, pending.pending_message()));
                }
                Ok((true, None))
            }
            Ok(BridgeIncoming::Snapshot(snapshot)) => {
                let snapshot_received_at = Instant::now();
                log_runtime_info(logger, format_snapshot_received_log(&snapshot)).await?;
                let outcome = self.apply_snapshot(snapshot);
                let receive_to_apply_us = snapshot_received_at.elapsed().as_micros();
                let _ = outcome;
                self.emit_pending_visible_update_applied_diagnostics(logger)
                    .await?;
                let pending = self
                    .submit_frame_if_needed_with_timing(
                        renderer,
                        openvr,
                        bridge,
                        logger,
                        Some(snapshot_received_at),
                        Some(receive_to_apply_us),
                        true,
                    )
                    .await?;
                Ok((true, pending.pending_message()))
            }
            Ok(BridgeIncoming::Event(event)) => {
                self.handle_event(event).await?;
                if self.stopped {
                    return Ok((false, None));
                }
                let pending = self
                    .submit_frame_if_needed_with_timing(
                        renderer, openvr, bridge, logger, None, None, true,
                    )
                    .await?;
                Ok((true, pending.pending_message()))
            }
            Err(BridgeError::Disconnected) => {
                logger
                    .error("runtime_disconnected")
                    .await
                    .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
                self.handle_bridge_loss_for_test().await?;
                logger
                    .emit_stdout_event(&json!({
                        "type": "runtime_error",
                        "failure_reason": "runtime_disconnected"
                    }))
                    .await
                    .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
                Err(RuntimeFailure::RuntimeDisconnected)
            }
            Err(error) => Err(RuntimeFailure::Bridge(error.to_string())),
        }
    }

    async fn handle_hide_deadline<S: OverlayFrameSubmitter>(
        &mut self,
        openvr: &mut S,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        self.hide_deadline = None;
        if !self.first_texture_submitted || !self.overlay_visible || self.has_drawable_text() {
            return Ok(());
        }
        let visibility_result = openvr.set_overlay_visible(false);
        let visibility_succeeded = visibility_result.is_ok();
        if visibility_succeeded {
            self.overlay_visible = false;
        }
        if let (Some(correlation), Some(backend)) = (
            self.last_presentation_correlation,
            self.last_presentation_backend,
        ) {
            self.presentation_diagnostics.record_visibility(
                correlation,
                backend,
                false,
                self.overlay_visible,
                visibility_succeeded,
            );
            self.emit_pending_presentation_diagnostics(logger).await?;
        }
        if let Err(error) = visibility_result {
            return Err(RuntimeFailure::OpenVr(error.to_string()));
        }
        if let Some(message) = openvr.take_visibility_api_call_log() {
            log_runtime_info(logger, message).await?;
        }
        log_runtime_info(
            logger,
            "overlay_visibility_changed visible=false reason=idle_hide_deadline".to_string(),
        )
        .await?;
        Ok(())
    }

    fn has_drawable_text(&self) -> bool {
        self.caption_blocks()
            .iter()
            .any(CaptionBlock::has_drawable_text)
    }

    fn desires_overlay_visible(&self) -> bool {
        self.first_texture_submitted && (self.has_drawable_text() || self.hide_deadline.is_some())
    }

    fn note_observed_runtime_visible(&mut self, visible: bool) {
        self.overlay_visible = visible;
    }

    async fn emit_pending_spatial_diagnostics(&mut self, logger: &OverlayLogger) {
        let diagnostics = std::mem::take(&mut self.pending_spatial_diagnostics);
        for diagnostic in diagnostics {
            match diagnostic {
                SpatialDiagnostic::Info(message) => {
                    let _ = logger.info(message).await;
                }
                SpatialDiagnostic::Warning(message) => {
                    let _ = logger.warn(message).await;
                }
            }
        }
    }

    async fn emit_pending_peer_overlay_first_emit_hooks(
        &mut self,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let block_ids = std::mem::take(&mut self.pending_peer_first_emit_logs);
        for block_id in block_ids {
            log_runtime_info(
                logger,
                format_peer_overlay_stage_log("peer_overlay_first_emit", &block_id),
            )
            .await?;
        }
        Ok(())
    }

    async fn emit_peer_overlay_first_render_hooks(
        &mut self,
        logger: &OverlayLogger,
        rendered_ids: Vec<String>,
    ) -> Result<(), RuntimeFailure> {
        for block_id in rendered_ids {
            self.pending_peer_first_render_ids.remove(&block_id);
            log_runtime_info(
                logger,
                format_peer_overlay_stage_log("peer_overlay_first_render", &block_id),
            )
            .await?;
        }
        Ok(())
    }

    async fn sample_and_log_frame_timing<S: OverlayFrameSubmitter>(
        &mut self,
        openvr: &S,
        logger: &OverlayLogger,
        revision: u64,
        submit_duration_us: Option<u128>,
        backend: PresentationBackend,
    ) -> Result<(), RuntimeFailure> {
        const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
        let now = Instant::now();
        if let Some(last) = self.last_frame_timing_sampled_at {
            if now.duration_since(last) < SAMPLE_INTERVAL {
                return Ok(());
            }
        }
        self.last_frame_timing_sampled_at = Some(now);
        let Some(t) = openvr.sample_frame_timing() else {
            return Ok(());
        };
        self.presentation_diagnostics.record_compositor_observation(
            backend,
            t.frame_index,
            t.num_dropped_frames,
            t.num_mis_presented,
            milliseconds_to_microseconds(t.compositor_render_cpu_ms),
            milliseconds_to_microseconds(t.total_render_gpu_ms),
            milliseconds_to_microseconds(t.post_submit_gpu_ms),
        );
        log_runtime_info(
            logger,
            format_frame_timing_log(revision, &t, submit_duration_us),
        )
        .await?;
        Ok(())
    }

    async fn emit_pending_presentation_diagnostics(
        &mut self,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let pending = self.presentation_diagnostics.pending_batch();
        if !pending.records.is_empty() {
            let message = format!("presentation_diagnostics [{}]", pending.records.join(","));
            let write_result = tokio::time::timeout(
                PRESENTATION_DIAGNOSTIC_WRITE_TIMEOUT,
                logger.detailed_info(message),
            )
            .await;
            if matches!(write_result, Ok(Ok(true))) {
                if let Some(sequence) = pending.through_sequence {
                    self.presentation_diagnostics.acknowledge_through(sequence);
                }
            }
        }
        Ok(())
    }

    pub fn presentation_diagnostics(&self) -> &PresentationDiagnostics {
        &self.presentation_diagnostics
    }

    #[doc(hidden)]
    pub fn pending_presentation_causes_for_test(&self) -> Vec<PresentationCause> {
        self.pending_presentation_causes.to_vec()
    }
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn milliseconds_to_microseconds(milliseconds: f32) -> Option<u64> {
    if milliseconds.is_finite() && milliseconds >= 0.0 {
        Some((milliseconds * 1_000.0).round() as u64)
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub struct NativePresentationRetryHandle {
    sender: mpsc::Sender<()>,
}

#[derive(Debug)]
enum FrameCycleOutcome {
    Submitted,
    Preempted(Result<BridgeIncoming, BridgeError>),
    NoWork,
}

impl FrameCycleOutcome {
    fn pending_message(self) -> Option<Result<BridgeIncoming, BridgeError>> {
        match self {
            Self::Preempted(message) => Some(message),
            Self::Submitted | Self::NoWork => None,
        }
    }
}

pub const NATIVE_FRESH_RETRY_CADENCE: Duration = Duration::from_millis(100);
pub const NATIVE_FRESH_RETRY_DEADLINE: Duration = Duration::from_millis(500);
pub const NATIVE_FRESH_RETRY_MAX_COMPLETED: u32 = 5;
pub const NATIVE_STREAM_RETRY_MAX_COMPLETED: u32 = 4;
pub const NATIVE_READINESS_TIMEOUT_RETRY_MAX: u32 = NATIVE_FRESH_RETRY_MAX_COMPLETED;
const NATIVE_FRESH_AUDIT_CAPACITY: usize = 128;

#[derive(Debug, Clone, Copy)]
struct NativeFreshRetryPolicy {
    cadence: Duration,
    deadline: Duration,
    max_completed: u32,
}

impl Default for NativeFreshRetryPolicy {
    fn default() -> Self {
        Self {
            cadence: NATIVE_FRESH_RETRY_CADENCE,
            deadline: NATIVE_FRESH_RETRY_DEADLINE,
            max_completed: NATIVE_FRESH_RETRY_MAX_COMPLETED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FreshRetryChannel {
    SelfChannel,
    Peer,
}

impl FreshRetryChannel {
    fn name(self) -> &'static str {
        match self {
            Self::SelfChannel => "self",
            Self::Peer => "peer",
        }
    }
}

#[derive(Debug, Clone)]
struct NativeFreshSchedule {
    channel: FreshRetryChannel,
    trigger_generation: u64,
    required_scene_generation: u64,
    target_identity: String,
    completed: u32,
    max_completed: u32,
    phase: NativeQuietTailPhase,
    episode_generation: u64,
    deadline: Instant,
    next_due: Instant,
}

#[derive(Debug, Clone)]
struct NativeEpisodeAccounting {
    target_identity: String,
    phase: NativeQuietTailPhase,
    episode_generation: u64,
    completed: u32,
    max_completed: u32,
    deadline: Instant,
}

impl NativeFreshSchedule {
    fn same_intent(&self, other: &Self) -> bool {
        self.channel == other.channel
            && self.trigger_generation == other.trigger_generation
            && self.required_scene_generation == other.required_scene_generation
            && self.target_identity == other.target_identity
            && self.episode_generation == other.episode_generation
            && self.phase == other.phase
    }

    fn expired_at(&self, now: Instant) -> bool {
        now > self.deadline
    }

    fn accepts_transferred_due_from(&self, other: &Self) -> bool {
        self.channel == other.channel
            && self.trigger_generation > other.trigger_generation
            && self.required_scene_generation > other.required_scene_generation
            && self.target_identity == other.target_identity
            && self.completed == other.completed
            && self.max_completed == other.max_completed
            && self.phase == other.phase
            && self.episode_generation == other.episode_generation
            && self.deadline == other.deadline
            && self.next_due == other.next_due
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeFreshAuditFact {
    channel: FreshRetryChannel,
    trigger_generation: u64,
    outcome: &'static str,
    completed: u32,
    at: Duration,
}

impl NativePresentationRetryHandle {
    pub fn request(&self) -> bool {
        match self.sender.try_send(()) {
            Ok(()) | Err(mpsc::error::TrySendError::Full(())) => true,
            Err(mpsc::error::TrySendError::Closed(())) => false,
        }
    }
}

pub struct NativePresentationOwner<S: OverlayFrameSubmitter> {
    runtime: PresentationRuntime,
    renderer: Option<CaptionRenderer>,
    openvr: Option<S>,
    retry_sender: Option<mpsc::Sender<()>>,
    retry_receiver: mpsc::Receiver<()>,
    retry_policy: NativeFreshRetryPolicy,
    observed_self_generation: Option<u64>,
    observed_peer_generation: Option<u64>,
    self_schedule: Option<NativeFreshSchedule>,
    peer_schedule: Option<NativeFreshSchedule>,
    self_accounting: Option<NativeEpisodeAccounting>,
    peer_accounting: Option<NativeEpisodeAccounting>,
    self_ended_episode: Option<(String, NativeQuietTailPhase, u64)>,
    peer_ended_episode: Option<(String, NativeQuietTailPhase, u64)>,
    retry_profile: &'static str,
    audit_started_at: Instant,
    fresh_retry_audit: VecDeque<NativeFreshAuditFact>,
    fresh_retry_audit_dropped: u64,
    successful_attempt_audit: VecDeque<PresentationCorrelation>,
    consecutive_readiness_timeouts: u32,
    max_consecutive_readiness_timeouts: u32,
    readiness_retry_due: Option<Instant>,
}

impl<S: OverlayFrameSubmitter> NativePresentationOwner<S> {
    pub fn new(
        snapshot: OverlayPresentationSnapshot,
        renderer: CaptionRenderer,
        openvr: S,
    ) -> Self {
        let (retry_sender, retry_receiver) = mpsc::channel(1);
        Self {
            runtime: PresentationRuntime::new(snapshot),
            renderer: Some(renderer),
            openvr: Some(openvr),
            retry_sender: Some(retry_sender),
            retry_receiver,
            retry_policy: NativeFreshRetryPolicy::default(),
            observed_self_generation: None,
            observed_peer_generation: None,
            self_schedule: None,
            peer_schedule: None,
            self_accounting: None,
            peer_accounting: None,
            self_ended_episode: None,
            peer_ended_episode: None,
            retry_profile: "p05",
            audit_started_at: Instant::now(),
            fresh_retry_audit: VecDeque::with_capacity(NATIVE_FRESH_AUDIT_CAPACITY),
            fresh_retry_audit_dropped: 0,
            successful_attempt_audit: VecDeque::with_capacity(NATIVE_FRESH_AUDIT_CAPACITY),
            consecutive_readiness_timeouts: 0,
            max_consecutive_readiness_timeouts: NATIVE_READINESS_TIMEOUT_RETRY_MAX,
            readiness_retry_due: None,
        }
    }

    pub fn new_with_profile(
        snapshot: OverlayPresentationSnapshot,
        renderer: CaptionRenderer,
        openvr: S,
        profile: crate::manifest::QuietTailProfile,
    ) -> Self {
        let mut owner = Self::new(snapshot, renderer, openvr);
        owner.retry_policy.max_completed = profile.max_final_opportunities();
        owner.retry_policy.deadline = profile.scheduling_wall();
        owner.retry_profile = profile.id();
        owner.runtime.configure_retry_profile(profile.id());
        owner
    }

    #[doc(hidden)]
    pub fn new_with_retry_policy_for_test(
        snapshot: OverlayPresentationSnapshot,
        renderer: CaptionRenderer,
        openvr: S,
        cadence: Duration,
        deadline: Duration,
        max_completed: u32,
    ) -> Self {
        let mut owner = Self::new(snapshot, renderer, openvr);
        owner.retry_policy = NativeFreshRetryPolicy {
            cadence,
            deadline,
            max_completed,
        };
        owner
    }

    #[doc(hidden)]
    pub fn set_max_consecutive_readiness_timeouts_for_test(&mut self, max_timeouts: u32) {
        self.max_consecutive_readiness_timeouts = max_timeouts;
    }

    pub fn runtime(&self) -> &PresentationRuntime {
        &self.runtime
    }

    pub fn retry_handle(&self) -> NativePresentationRetryHandle {
        NativePresentationRetryHandle {
            sender: self
                .retry_sender
                .as_ref()
                .expect("active native presentation owner")
                .clone(),
        }
    }

    pub fn resources_released(&self) -> bool {
        self.renderer.is_none() && self.openvr.is_none() && self.retry_sender.is_none()
    }

    #[doc(hidden)]
    pub fn fresh_retry_audit_for_test(
        &self,
    ) -> Vec<(&'static str, u64, &'static str, u32, Duration)> {
        self.fresh_retry_audit
            .iter()
            .map(|fact| {
                (
                    fact.channel.name(),
                    fact.trigger_generation,
                    fact.outcome,
                    fact.completed,
                    fact.at,
                )
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn fresh_retry_audit_dropped_for_test(&self) -> u64 {
        self.fresh_retry_audit_dropped
    }

    #[doc(hidden)]
    pub fn successful_attempt_audit_for_test(&self) -> Vec<PresentationCorrelation> {
        self.successful_attempt_audit.iter().copied().collect()
    }

    fn capture_successful_attempt(&mut self) {
        self.consecutive_readiness_timeouts = 0;
        self.readiness_retry_due = None;
        let Some(correlation) = self.runtime.last_presentation_correlation else {
            return;
        };
        if self
            .successful_attempt_audit
            .back()
            .is_some_and(|previous| previous.submission_attempt == correlation.submission_attempt)
        {
            return;
        }
        if self.successful_attempt_audit.len() == NATIVE_FRESH_AUDIT_CAPACITY {
            self.successful_attempt_audit.pop_front();
        }
        self.successful_attempt_audit.push_back(correlation);
    }

    async fn record_fresh_retry(
        &mut self,
        logger: &OverlayLogger,
        schedule: NativeFreshSchedule,
        outcome: &'static str,
    ) -> Result<(), RuntimeFailure> {
        self.push_fresh_retry_audit(schedule.clone(), outcome);
        log_fresh_retry(
            logger,
            schedule,
            outcome,
            self.retry_policy,
            self.retry_profile,
        )
        .await
    }

    fn push_fresh_retry_audit(&mut self, schedule: NativeFreshSchedule, outcome: &'static str) {
        if self.fresh_retry_audit.len() == NATIVE_FRESH_AUDIT_CAPACITY {
            self.fresh_retry_audit.pop_front();
            self.fresh_retry_audit_dropped += 1;
        }
        self.fresh_retry_audit.push_back(NativeFreshAuditFact {
            channel: schedule.channel,
            trigger_generation: schedule.trigger_generation,
            outcome,
            completed: schedule.completed,
            at: self.audit_started_at.elapsed(),
        });
    }

    fn finish_initial_reconcile(
        &mut self,
        result: Result<(), RuntimeFailure>,
    ) -> Result<(), RuntimeFailure> {
        if result.is_err() {
            self.teardown();
        }
        result
    }

    fn channel_generation(&self, channel: FreshRetryChannel) -> Option<u64> {
        let generations = self
            .runtime
            .state()
            .snapshot()
            .native_fresh_render_generations
            .as_ref();
        match channel {
            FreshRetryChannel::SelfChannel => generations.and_then(|value| value.self_generation),
            FreshRetryChannel::Peer => generations.and_then(|value| value.peer),
        }
    }

    fn channel_target_identity(&self, channel: FreshRetryChannel) -> Option<String> {
        let generations = self.runtime.state().native_fresh_render_generations()?;
        let selected = match channel {
            FreshRetryChannel::SelfChannel => generations.self_target.as_ref(),
            FreshRetryChannel::Peer => generations.peer_target.as_ref(),
        };
        let fallback;
        let selected = if let Some(selected) = selected {
            selected
        } else {
            let candidates = self
                .runtime
                .state()
                .blocks()
                .iter()
                .filter(|block| {
                    block.channel == channel.name()
                        && block.block_variant == OverlayPresentationBlockVariant::Finalized
                        && !block.primary_text.trim().is_empty()
                })
                .map(|block| block.id.as_str())
                .collect::<Vec<_>>();
            if candidates.len() != 1 {
                return None;
            }
            fallback = candidates[0].to_string();
            &fallback
        };
        let episode = self.channel_episode(channel)?;
        self.runtime
            .state()
            .blocks()
            .iter()
            .any(|block| {
                block.channel == channel.name()
                    && block.id == *selected
                    && (episode.phase == NativeQuietTailPhase::Stream
                        || block.block_variant == OverlayPresentationBlockVariant::Finalized)
                    && !block.primary_text.trim().is_empty()
            })
            .then(|| selected.clone())
    }

    fn channel_episode(&self, channel: FreshRetryChannel) -> Option<NativeQuietTailEpisode> {
        let generations = self.runtime.state().native_fresh_render_generations()?;
        if let Some(episodes) = generations.quiet_tail_episodes.as_ref() {
            return match channel {
                FreshRetryChannel::SelfChannel => episodes.self_episode.clone(),
                FreshRetryChannel::Peer => episodes.peer.clone(),
            };
        }
        self.channel_generation(channel)
            .map(|generation| NativeQuietTailEpisode {
                phase: NativeQuietTailPhase::Final,
                generation,
            })
    }

    async fn reconcile_fresh_schedules(
        &mut self,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        for channel in [FreshRetryChannel::SelfChannel, FreshRetryChannel::Peer] {
            let generation = self.channel_generation(channel);
            let episode = self.channel_episode(channel);
            let target_identity = self.channel_target_identity(channel);
            let current_schedule = match channel {
                FreshRetryChannel::SelfChannel => self.self_schedule.clone(),
                FreshRetryChannel::Peer => self.peer_schedule.clone(),
            };
            let current_accounting = match channel {
                FreshRetryChannel::SelfChannel => self.self_accounting.clone(),
                FreshRetryChannel::Peer => self.peer_accounting.clone(),
            };
            let observed = match channel {
                FreshRetryChannel::SelfChannel => &mut self.observed_self_generation,
                FreshRetryChannel::Peer => &mut self.observed_peer_generation,
            };
            let generation_changed = generation.is_some() && generation != *observed;
            let episode_changed = current_schedule.as_ref().is_some_and(|schedule| {
                target_identity.as_ref() != Some(&schedule.target_identity)
                    || episode.as_ref().is_none_or(|episode| {
                        episode.generation != schedule.episode_generation
                            || episode.phase != schedule.phase
                    })
            });
            *observed = generation;
            let schedule = match channel {
                FreshRetryChannel::SelfChannel => &mut self.self_schedule,
                FreshRetryChannel::Peer => &mut self.peer_schedule,
            };
            if target_identity.is_none() || generation.is_none() || episode.is_none() {
                match channel {
                    FreshRetryChannel::SelfChannel => {
                        if let Some(value) = self.self_accounting.take() {
                            self.self_ended_episode = Some((
                                value.target_identity,
                                value.phase,
                                value.episode_generation,
                            ));
                        }
                    }
                    FreshRetryChannel::Peer => {
                        if let Some(value) = self.peer_accounting.take() {
                            self.peer_ended_episode = Some((
                                value.target_identity,
                                value.phase,
                                value.episode_generation,
                            ));
                        }
                    }
                }
                if let Some(cancelled) = schedule.take() {
                    self.record_fresh_retry(logger, cancelled, "cancelled")
                        .await?;
                }
                continue;
            }
            if episode_changed && !generation_changed {
                if let Some(cancelled) = schedule.take() {
                    self.record_fresh_retry(logger, cancelled, "cancelled")
                        .await?;
                }
                continue;
            }
            if generation_changed {
                let now = Instant::now();
                let episode = episode.expect("checked episode");
                let target_identity = target_identity.expect("checked target identity");
                let ended_episode = match channel {
                    FreshRetryChannel::SelfChannel => self.self_ended_episode.as_ref(),
                    FreshRetryChannel::Peer => self.peer_ended_episode.as_ref(),
                };
                if ended_episode.is_some_and(|ended| {
                    ended.0 == target_identity
                        && ended.1 == episode.phase
                        && ended.2 == episode.generation
                }) {
                    continue;
                }
                match channel {
                    FreshRetryChannel::SelfChannel => self.self_ended_episode = None,
                    FreshRetryChannel::Peer => self.peer_ended_episode = None,
                }
                let same_episode = current_accounting.as_ref().is_some_and(|accounting| {
                    accounting.target_identity == target_identity
                        && accounting.episode_generation == episode.generation
                        && accounting.phase == episode.phase
                });
                let completed = if same_episode {
                    current_accounting
                        .as_ref()
                        .map_or(0, |value| value.completed)
                } else {
                    0
                };
                let deadline = if same_episode {
                    current_accounting
                        .as_ref()
                        .map_or(now, |value| value.deadline)
                } else {
                    now + self.retry_policy.deadline
                };
                let max_completed = if same_episode {
                    current_accounting
                        .as_ref()
                        .map_or(self.retry_policy.max_completed, |value| value.max_completed)
                } else {
                    match episode.phase {
                        NativeQuietTailPhase::Stream => {
                            NATIVE_STREAM_RETRY_MAX_COMPLETED.min(self.retry_policy.max_completed)
                        }
                        NativeQuietTailPhase::Final => self.retry_policy.max_completed,
                    }
                };
                let next_due = if same_episode {
                    current_schedule
                        .as_ref()
                        .map_or(now + self.retry_policy.cadence, |value| value.next_due)
                } else {
                    now + self.retry_policy.cadence
                };
                let next = NativeFreshSchedule {
                    channel,
                    trigger_generation: generation.expect("checked generation"),
                    required_scene_generation: self.runtime.state().snapshot().revision,
                    target_identity,
                    completed,
                    max_completed,
                    phase: episode.phase,
                    episode_generation: episode.generation,
                    deadline,
                    next_due,
                };
                let accounting = NativeEpisodeAccounting {
                    target_identity: next.target_identity.clone(),
                    phase: next.phase,
                    episode_generation: next.episode_generation,
                    completed,
                    max_completed,
                    deadline,
                };
                match channel {
                    FreshRetryChannel::SelfChannel => self.self_accounting = Some(accounting),
                    FreshRetryChannel::Peer => self.peer_accounting = Some(accounting),
                }
                let disabled = max_completed == 0 || completed >= max_completed || now > deadline;
                let replaced = if disabled {
                    schedule.take()
                } else {
                    schedule.replace(next.clone())
                };
                if let Some(replaced) = replaced {
                    self.record_fresh_retry(logger, replaced, "replaced")
                        .await?;
                }
                self.record_fresh_retry(logger, next, "scheduled").await?;
            }
        }
        Ok(())
    }

    fn next_fresh_due(&self) -> Option<Instant> {
        [self.self_schedule.clone(), self.peer_schedule.clone()]
            .into_iter()
            .flatten()
            .map(|schedule| schedule.next_due)
            .min()
    }

    fn next_retry_wake(&self) -> Option<Instant> {
        [self.next_fresh_due(), self.readiness_retry_due]
            .into_iter()
            .flatten()
            .min()
    }

    async fn note_readiness_timeout(
        &mut self,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        self.consecutive_readiness_timeouts = self.consecutive_readiness_timeouts.saturating_add(1);
        self.runtime.request_native_presentation_retry();
        let due = Instant::now() + self.retry_policy.cadence;
        let mut deferred_schedule = false;
        if let Some(schedule) = self.self_schedule.as_mut() {
            schedule.next_due = due;
            deferred_schedule = true;
        }
        if let Some(schedule) = self.peer_schedule.as_mut() {
            schedule.next_due = due;
            deferred_schedule = true;
        }
        if !deferred_schedule {
            self.readiness_retry_due = Some(due);
        }
        log_runtime_info(
            logger,
            format!(
                "readiness_timeout_retry consecutive={} max={} physical_hmd_visibility=not_observable",
                self.consecutive_readiness_timeouts, self.max_consecutive_readiness_timeouts,
            ),
        )
        .await?;
        if self.consecutive_readiness_timeouts >= self.max_consecutive_readiness_timeouts {
            return Err(RuntimeFailure::ReadinessTimedOut);
        }
        Ok(())
    }

    async fn complete_frame_cycle(
        &mut self,
        result: Result<FrameCycleOutcome, RuntimeFailure>,
        logger: &OverlayLogger,
    ) -> Result<Option<FrameCycleOutcome>, RuntimeFailure> {
        match result {
            Ok(outcome) => Ok(Some(outcome)),
            Err(RuntimeFailure::ReadinessTimedOut) => {
                self.note_readiness_timeout(logger).await?;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn due_fresh_channels(&self, now: Instant) -> Vec<FreshRetryChannel> {
        [self.self_schedule.clone(), self.peer_schedule.clone()]
            .into_iter()
            .flatten()
            .filter(|schedule| schedule.next_due <= now)
            .map(|schedule| schedule.channel)
            .collect()
    }

    fn active_fresh_schedules(&self, now: Instant) -> Vec<NativeFreshSchedule> {
        [self.self_schedule.clone(), self.peer_schedule.clone()]
            .into_iter()
            .flatten()
            .filter(|schedule| schedule.next_due <= now && now <= schedule.deadline)
            .collect()
    }

    fn remove_temporary_intent_causes(&mut self, schedules: &[NativeFreshSchedule]) {
        for schedule in schedules {
            self.runtime
                .pending_presentation_causes
                .remove(Self::intent_cause(
                    schedule,
                    PresentationCauseKind::ActiveRetryIntent,
                ));
        }
    }

    fn intent_cause(
        schedule: &NativeFreshSchedule,
        kind: PresentationCauseKind,
    ) -> PresentationCause {
        PresentationCause {
            kind,
            channel: Some(match schedule.channel {
                FreshRetryChannel::SelfChannel => PresentationCauseChannel::SelfChannel,
                FreshRetryChannel::Peer => PresentationCauseChannel::Peer,
            }),
            trigger_generation: Some(schedule.trigger_generation),
        }
    }

    fn submission_covers_schedule(
        correlation: PresentationCorrelation,
        active: &NativeFreshSchedule,
        captured: &NativeFreshSchedule,
        cause_kind: PresentationCauseKind,
        current_generation: Option<u64>,
        current_target_identity: Option<&str>,
    ) -> bool {
        (active.same_intent(captured) || active.accepts_transferred_due_from(captured))
            && correlation
                .logical_causes
                .contains(Self::intent_cause(captured, cause_kind))
            && current_target_identity == Some(active.target_identity.as_str())
            && current_generation == Some(active.trigger_generation)
            && correlation.scene_generation >= active.required_scene_generation
    }

    async fn run_due_fresh_attempt(
        &mut self,
        channels: Vec<FreshRetryChannel>,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<FrameCycleOutcome, RuntimeFailure> {
        let now = Instant::now();
        let mut due = Vec::new();
        for channel in channels {
            let schedule = match channel {
                FreshRetryChannel::SelfChannel => self.self_schedule.clone(),
                FreshRetryChannel::Peer => self.peer_schedule.clone(),
            };
            let Some(schedule) = schedule else {
                continue;
            };
            if schedule.expired_at(now) {
                match channel {
                    FreshRetryChannel::SelfChannel => self.self_schedule = None,
                    FreshRetryChannel::Peer => self.peer_schedule = None,
                }
                self.record_fresh_retry(logger, schedule, "expired").await?;
            } else {
                due.push(schedule);
            }
        }
        if due.is_empty() {
            return Ok(FrameCycleOutcome::NoWork);
        }
        self.readiness_retry_due = None;
        for schedule in &due {
            self.runtime
                .request_fresh_presentation_retry(schedule.channel, schedule.trigger_generation);
        }
        let attempt = {
            let renderer = self.renderer.as_ref().expect("active renderer");
            let openvr = self.openvr.as_mut().expect("active OpenVR session");
            self.runtime
                .submit_frame_if_needed_with_timing(
                    renderer, openvr, bridge, logger, None, None, true,
                )
                .await
        };
        let outcome = match attempt {
            Ok(outcome) => outcome,
            Err(RuntimeFailure::ReadinessTimedOut) => {
                self.note_readiness_timeout(logger).await?;
                return Ok(FrameCycleOutcome::NoWork);
            }
            Err(primary_failure) => {
                for schedule in due {
                    let active = match schedule.channel {
                        FreshRetryChannel::SelfChannel => {
                            self.self_accounting = None;
                            self.self_schedule.take()
                        }
                        FreshRetryChannel::Peer => {
                            self.peer_accounting = None;
                            self.peer_schedule.take()
                        }
                    };
                    if let Some(active) = active.filter(|active| active.same_intent(&schedule)) {
                        self.push_fresh_retry_audit(active.clone(), "failed");
                        let _ = log_fresh_retry(
                            logger,
                            active,
                            "failed",
                            self.retry_policy,
                            self.retry_profile,
                        )
                        .await;
                    }
                }
                return Err(primary_failure);
            }
        };
        match &outcome {
            FrameCycleOutcome::Submitted => {
                self.capture_successful_attempt();
                self.satisfy_schedules_from_last_submission(
                    logger,
                    &due,
                    PresentationCauseKind::NativeFreshRetry,
                )
                .await?;
            }
            FrameCycleOutcome::Preempted(_) => {
                for schedule in due {
                    let slot = match schedule.channel {
                        FreshRetryChannel::SelfChannel => &mut self.self_schedule,
                        FreshRetryChannel::Peer => &mut self.peer_schedule,
                    };
                    if let Some(active) =
                        slot.as_ref().filter(|active| active.same_intent(&schedule))
                    {
                        let fact = active.clone();
                        self.record_fresh_retry(logger, fact, "preempted").await?;
                    }
                }
            }
            FrameCycleOutcome::NoWork => {}
        }
        Ok(outcome)
    }

    async fn satisfy_schedules_from_last_submission(
        &mut self,
        logger: &OverlayLogger,
        captured_schedules: &[NativeFreshSchedule],
        cause_kind: PresentationCauseKind,
    ) -> Result<(), RuntimeFailure> {
        let Some(correlation) = self.runtime.last_presentation_correlation else {
            return Ok(());
        };
        for captured in captured_schedules {
            let channel = captured.channel;
            let current_generation = self.channel_generation(channel);
            let current_target_identity = self.channel_target_identity(channel);
            let slot = match channel {
                FreshRetryChannel::SelfChannel => &mut self.self_schedule,
                FreshRetryChannel::Peer => &mut self.peer_schedule,
            };
            let Some(active) = slot.as_mut() else {
                continue;
            };
            if !Self::submission_covers_schedule(
                correlation,
                active,
                captured,
                cause_kind,
                current_generation,
                current_target_identity.as_deref(),
            ) {
                continue;
            }
            active.completed += 1;
            active.next_due = Instant::now() + self.retry_policy.cadence;
            let completed = active.clone();
            if active.completed >= active.max_completed || Instant::now() > active.deadline {
                *slot = None;
            }
            let accounting = match channel {
                FreshRetryChannel::SelfChannel => &mut self.self_accounting,
                FreshRetryChannel::Peer => &mut self.peer_accounting,
            };
            if let Some(accounting) = accounting.as_mut().filter(|accounting| {
                accounting.target_identity == completed.target_identity
                    && accounting.phase == completed.phase
                    && accounting.episode_generation == completed.episode_generation
            }) {
                accounting.completed = completed.completed;
            }
            self.record_fresh_retry(logger, completed, "completed")
                .await?;
        }
        Ok(())
    }

    pub async fn run(
        &mut self,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let initial_result = {
            let renderer = self.renderer.as_ref().expect("active renderer");
            let openvr = self.openvr.as_mut().expect("active OpenVR session");
            self.runtime
                .submit_initial_frame_message_aware(renderer, openvr, bridge, logger)
                .await
        };
        let initial_timed_out = matches!(&initial_result, Err(RuntimeFailure::ReadinessTimedOut));
        if let Err(error) = initial_result {
            if !initial_timed_out {
                self.teardown();
                return Err(error);
            }
            if let Err(error) = self.note_readiness_timeout(logger).await {
                self.teardown();
                return Err(error);
            }
        }
        if self.runtime.is_stopped() {
            self.teardown();
            return Ok(());
        }
        if !initial_timed_out {
            self.capture_successful_attempt();
        }
        let reconcile_result = self.reconcile_fresh_schedules(logger).await;
        self.finish_initial_reconcile(reconcile_result)?;

        let result = self.run_owned_event_loop(bridge, logger).await;
        self.teardown();
        result
    }

    async fn pump_openvr_events(&mut self, logger: &OverlayLogger) -> Result<(), RuntimeFailure> {
        let events = {
            let openvr = self.openvr.as_mut().expect("active OpenVR session");
            openvr.poll_runtime_events(MAX_OPENVR_EVENTS_PER_TURN)
        };
        let mut saw_overlay_hidden = false;
        for event in events {
            match event.classify() {
                OpenVrEventClass::Ignore => {}
                OpenVrEventClass::Fatal => {
                    return Err(RuntimeFailure::OpenVr(format!("event={}", event.as_str())));
                }
                OpenVrEventClass::Reconfigure => {
                    log_runtime_info(
                        logger,
                        format!(
                            "openvr_event_classified type={} class=reconfigure physical_hmd_visibility=not_observable",
                            event.as_str()
                        ),
                    )
                    .await?;
                    match event {
                        OpenVrRuntimeEvent::OverlayShown => {
                            self.runtime.note_observed_runtime_visible(true);
                        }
                        OpenVrRuntimeEvent::OverlayHidden => {
                            saw_overlay_hidden = true;
                            self.runtime.note_observed_runtime_visible(false);
                        }
                        _ => {}
                    }
                }
            }
        }
        let observed = {
            let openvr = self.openvr.as_mut().expect("active OpenVR session");
            openvr.observed_overlay_visible()
        };
        if let Some(visible) = observed {
            self.runtime.note_observed_runtime_visible(visible);
        }
        let desired_visible = self.runtime.desires_overlay_visible();
        let needs_reassert = match observed {
            Some(visible) => visible != desired_visible,
            None => saw_overlay_hidden && desired_visible,
        };
        if needs_reassert {
            let message = {
                let openvr = self.openvr.as_mut().expect("active OpenVR session");
                openvr
                    .set_overlay_visible(desired_visible)
                    .map_err(|error| RuntimeFailure::OpenVr(error.to_string()))?;
                openvr.take_visibility_api_call_log()
            };
            self.runtime.note_observed_runtime_visible(desired_visible);
            if let Some(message) = message {
                log_runtime_info(logger, message).await?;
            }
        }
        Ok(())
    }

    async fn run_owned_event_loop(
        &mut self,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        let mut pending_message = None;
        loop {
            self.pump_openvr_events(logger).await?;
            let hide_deadline = self.runtime.hide_deadline;
            let message = if let Some(message) = pending_message.take() {
                Some(message)
            } else {
                tokio::select! {
                    biased;
                    _ = sleep_until(self.next_retry_wake().unwrap_or_else(Instant::now)), if self.next_retry_wake().is_some() => {
                        let now = Instant::now();
                        let channels = self.due_fresh_channels(now);
                        if !channels.is_empty() {
                            let outcome = self.run_due_fresh_attempt(channels, bridge, logger).await?;
                            pending_message = outcome.pending_message();
                        } else if self.readiness_retry_due.is_some_and(|due| due <= now) {
                            self.readiness_retry_due = None;
                            let result = {
                                let renderer = self.renderer.as_ref().expect("active renderer");
                                let openvr = self.openvr.as_mut().expect("active OpenVR session");
                                self.runtime
                                    .submit_frame_if_needed_with_timing(
                                        renderer, openvr, bridge, logger, None, None, true,
                                    )
                                    .await
                            };
                            if let Some(outcome) = self.complete_frame_cycle(result, logger).await? {
                                if matches!(&outcome, FrameCycleOutcome::Submitted) {
                                    self.capture_successful_attempt();
                                }
                                pending_message = outcome.pending_message();
                            }
                        }
                        None
                    }
                    retry = self.retry_receiver.recv() => {
                        if retry.is_none() {
                            return Ok(());
                        }
                        let captured_schedules = self.active_fresh_schedules(Instant::now());
                        for schedule in &captured_schedules {
                            self.runtime.pending_presentation_causes.insert(Self::intent_cause(
                                schedule,
                                PresentationCauseKind::ActiveRetryIntent,
                            ));
                        }
                        self.runtime.request_native_presentation_retry();
                        let renderer = self.renderer.as_ref().expect("active renderer");
                        let openvr = self.openvr.as_mut().expect("active OpenVR session");
                        let result = self.runtime
                            .submit_frame_if_needed_with_timing(
                                renderer, openvr, bridge, logger, None, None, true,
                            )
                            .await;
                        match self.complete_frame_cycle(result, logger).await? {
                            Some(outcome) => {
                                if matches!(&outcome, FrameCycleOutcome::Submitted) {
                                    self.capture_successful_attempt();
                                    self.satisfy_schedules_from_last_submission(
                                        logger,
                                        &captured_schedules,
                                        PresentationCauseKind::ActiveRetryIntent,
                                    ).await?;
                                } else {
                                    for schedule in captured_schedules {
                                        self.runtime.pending_presentation_causes.remove(Self::intent_cause(
                                            &schedule,
                                            PresentationCauseKind::ActiveRetryIntent,
                                        ));
                                    }
                                }
                                pending_message = outcome.pending_message();
                            }
                            None => {
                                self.remove_temporary_intent_causes(&captured_schedules);
                                pending_message = None;
                            }
                        }
                        None
                    }
                    _ = sleep_until(hide_deadline.unwrap_or_else(Instant::now)), if hide_deadline.is_some() => {
                        let openvr = self.openvr.as_mut().expect("active OpenVR session");
                        self.runtime.handle_hide_deadline(openvr, logger).await?;
                        None
                    }
                    message = bridge.next_message() => Some(message),
                    _ = sleep_until(Instant::now() + OPENVR_EVENT_POLL_INTERVAL) => None,
                }
            };
            if let Some(message) = message {
                let previous_submission = self.runtime.last_presentation_correlation;
                let captured_schedules = self.active_fresh_schedules(Instant::now());
                for schedule in &captured_schedules {
                    self.runtime
                        .pending_presentation_causes
                        .insert(Self::intent_cause(
                            schedule,
                            PresentationCauseKind::ActiveRetryIntent,
                        ));
                }
                let renderer = self.renderer.as_ref().expect("active renderer");
                let openvr = self.openvr.as_mut().expect("active OpenVR session");
                let handled = self
                    .runtime
                    .handle_bridge_message(message, renderer, openvr, bridge, logger)
                    .await;
                let (continue_running, preempted_message) = match handled {
                    Ok(handled) => handled,
                    Err(RuntimeFailure::ReadinessTimedOut) => {
                        self.note_readiness_timeout(logger).await?;
                        self.remove_temporary_intent_causes(&captured_schedules);
                        continue;
                    }
                    Err(error) => {
                        self.remove_temporary_intent_causes(&captured_schedules);
                        return Err(error);
                    }
                };
                if !continue_running {
                    self.remove_temporary_intent_causes(&captured_schedules);
                    return Ok(());
                }
                pending_message = preempted_message;
                self.reconcile_fresh_schedules(logger).await?;
                if self.runtime.last_presentation_correlation != previous_submission
                    && self.runtime.last_presentation_correlation.is_some()
                {
                    self.capture_successful_attempt();
                    self.satisfy_schedules_from_last_submission(
                        logger,
                        &captured_schedules,
                        PresentationCauseKind::ActiveRetryIntent,
                    )
                    .await?;
                } else {
                    self.remove_temporary_intent_causes(&captured_schedules);
                }
            }
        }
    }

    fn teardown(&mut self) {
        self.retry_sender = None;
        self.retry_receiver.close();
        if let Some(schedule) = self.self_schedule.take() {
            self.push_fresh_retry_audit(schedule, "teardown");
        }
        if let Some(schedule) = self.peer_schedule.take() {
            self.push_fresh_retry_audit(schedule, "teardown");
        }
        self.self_accounting = None;
        self.peer_accounting = None;
        self.self_ended_episode = None;
        self.peer_ended_episode = None;
        self.runtime.shutdown_presentation();
        if let Some(openvr) = self.openvr.as_mut() {
            let _ = openvr.set_overlay_visible(false);
        }
        self.openvr = None;
        self.renderer = None;
    }
}

async fn log_fresh_retry(
    logger: &OverlayLogger,
    schedule: NativeFreshSchedule,
    outcome: &str,
    policy: NativeFreshRetryPolicy,
    retry_profile: &'static str,
) -> Result<(), RuntimeFailure> {
    log_runtime_info(
        logger,
        format!(
            "native_fresh_retry channel={} phase={} profile={} trigger_generation={} outcome={} completed={} max={} cadence_ms={} deadline_ms={} physical_hmd_visibility=not_observable",
            schedule.channel.name(),
            match schedule.phase { NativeQuietTailPhase::Stream => "stream", NativeQuietTailPhase::Final => "final" },
            retry_profile,
            schedule.trigger_generation,
            outcome,
            schedule.completed,
            schedule.max_completed,
            policy.cadence.as_millis(),
            policy.deadline.as_millis(),
        ),
    )
    .await
}

fn peer_overlay_first_emit_block_ids_from_snapshot(
    snapshot: &OverlayPresentationSnapshot,
) -> Vec<String> {
    snapshot
        .blocks
        .iter()
        .filter(|block| is_peer_overlay_first_emit_candidate(block))
        .map(|block| block.id.clone())
        .collect()
}

fn is_peer_overlay_first_emit_candidate(block: &OverlayPresentationBlock) -> bool {
    block.channel == "peer"
        && matches!(
            block.block_variant,
            OverlayPresentationBlockVariant::ActivePeer
                | OverlayPresentationBlockVariant::Finalized
        )
        && (!block.primary_text.trim().is_empty()
            || (block.secondary_enabled && !block.secondary_text.trim().is_empty()))
}

fn peer_overlay_first_render_block_ids_from_caption_blocks(
    blocks: &[CaptionBlock],
    pending: &HashSet<String>,
) -> Vec<String> {
    blocks
        .iter()
        .filter(|block| {
            pending.contains(&block.id) && is_peer_overlay_first_render_candidate(block)
        })
        .map(|block| block.id.clone())
        .collect()
}

fn is_peer_overlay_first_render_candidate(block: &CaptionBlock) -> bool {
    block.channel == Some(CaptionChannel::PeerChannel)
        && matches!(
            block.block_variant,
            CaptionBlockVariant::ActivePeer | CaptionBlockVariant::Finalized
        )
        && block.has_drawable_text()
}

fn format_peer_overlay_stage_log(stage: &str, block_id: &str) -> String {
    let _ = block_id;
    format!("latency_trace stage={stage} identity=redacted")
}

#[cfg(test)]
fn log_runtime_secondary_state(enabled: bool, text: &str) -> String {
    format!(
        "{}/{}",
        if enabled { "enabled" } else { "disabled" },
        text.len()
    )
}

fn overlay_variant_name(variant: OverlayPresentationBlockVariant) -> &'static str {
    match variant {
        OverlayPresentationBlockVariant::ActiveSelf => "active_self",
        OverlayPresentationBlockVariant::ActivePeer => "active_peer",
        OverlayPresentationBlockVariant::Finalized => "finalized",
    }
}

#[cfg(test)]
fn caption_variant_name(variant: CaptionBlockVariant) -> &'static str {
    match variant {
        CaptionBlockVariant::ActiveSelf => "active_self",
        CaptionBlockVariant::ActivePeer => "active_peer",
        CaptionBlockVariant::Finalized => "finalized",
    }
}

fn format_snapshot_received_log(snapshot: &OverlayPresentationSnapshot) -> String {
    format!(
        "bridge_snapshot_received revision={} block_count={}",
        snapshot.revision,
        snapshot.blocks.len(),
    )
}

fn logical_caption_identity(state: &OverlayState) -> LogicalCaptionIdentity {
    LogicalCaptionIdentity(
        state
            .scene()
            .slots()
            .iter()
            .flatten()
            .map(|slot| LogicalCaptionBlockIdentity {
                slot_index: slot.slot_index,
                channel: slot.channel.clone(),
                block_variant: slot.block_variant,
                primary_text: slot.primary_text.clone(),
                secondary_text: slot.secondary_text.clone(),
                secondary_enabled: slot.secondary_enabled,
                primary_language: slot.primary_language.clone(),
                secondary_language: slot.secondary_language.clone(),
            })
            .collect(),
    )
}

fn format_state_snapshot_log(
    outcome: &SnapshotApplyOutcome,
    state: &OverlayState,
    redraw_requested: bool,
) -> String {
    match outcome {
        SnapshotApplyOutcome::Applied {
            incoming_revision,
            current_revision,
            visual_changed,
            redraw_requested: outcome_redraw_requested,
        } => format!(
            "state_snapshot_applied incoming_revision={} current_revision={} visual_changed={} redraw_requested={} block_count={} occupied_slot_count={}",
            incoming_revision,
            current_revision,
            visual_changed,
            outcome_redraw_requested,
            state.snapshot().blocks.len(),
            state.scene().slots().iter().flatten().count(),
        ),
        SnapshotApplyOutcome::Ignored {
            incoming_revision,
            current_revision,
        } => format!(
            "state_snapshot_ignored incoming_revision={} current_revision={} redraw_requested={} block_count={} occupied_slot_count={}",
            incoming_revision,
            current_revision,
            redraw_requested,
            state.snapshot().blocks.len(),
            state.scene().slots().iter().flatten().count(),
        ),
    }
}

fn collect_diagnostic_rows(state: &OverlayState) -> Vec<DiagnosticRow> {
    let slots_by_occupant_key = state
        .scene()
        .slots()
        .iter()
        .flatten()
        .map(|slot| (slot.occupant_key.as_str(), slot))
        .collect::<HashMap<_, _>>();

    state
        .snapshot()
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(presenter_order, block)| {
            let slot = slots_by_occupant_key.get(block.occupant_key.as_str())?;
            Some(DiagnosticRow {
                id: block.id.clone(),
                occupant_key: block.occupant_key.clone(),
                channel: block.channel.clone(),
                block_variant: block.block_variant,
                update_id: block.update_id.clone(),
                origin_wall_clock_ms: block.origin_wall_clock_ms,
                session_scope: block.session_scope.clone(),
                presenter_order,
                slot_order: slot.slot_entry_order,
                slot_index: slot.slot_index,
                slot_anchor_top_px: slot.anchor_top_px,
                primary_text: block.primary_text.clone(),
                secondary_text: block.secondary_text.clone(),
                secondary_enabled: block.secondary_enabled,
            })
        })
        .collect()
}

fn diagnostic_row_signature(row: &DiagnosticRow) -> String {
    format!(
        "id={} occupant_key={} channel={} variant={} presenter_order={} slot_order={} slot_index={} slot_anchor_top_px={:.3} update_id={:?} origin_wall_clock_ms={:?} session_scope={:?} primary_text={:?} secondary_text={:?} secondary_enabled={}",
        row.id,
        row.occupant_key,
        row.channel,
        overlay_variant_name(row.block_variant),
        row.presenter_order,
        row.slot_order,
        row.slot_index,
        row.slot_anchor_top_px,
        row.update_id,
        row.origin_wall_clock_ms,
        row.session_scope,
        row.primary_text,
        row.secondary_text,
        row.secondary_enabled,
    )
}

fn snapshot_slot_correlation_signature(state: &OverlayState, rows: &[DiagnosticRow]) -> String {
    format!(
        "anchor={} offset_x={:.3} offset_y={:.3} distance={:.3} text_scale={:.3} background_alpha={:.3} rows=[{}]",
        state.calibration().anchor,
        state.calibration().offset_x,
        state.calibration().offset_y,
        state.calibration().distance,
        state.calibration().text_scale,
        state.calibration().background_alpha,
        rows.iter()
            .map(diagnostic_row_signature)
            .collect::<Vec<_>>()
            .join("; ")
    )
}

fn format_snapshot_slot_correlation_log(state: &OverlayState, rows: &[DiagnosticRow]) -> String {
    format!(
        "snapshot_slot_correlation revision={} anchor={} offset_x={:.3} offset_y={:.3} distance={:.3} text_scale={:.3} background_alpha={:.3} row_count={} occupied_slot_count={}",
        state.snapshot().revision,
        state.calibration().anchor,
        state.calibration().offset_x,
        state.calibration().offset_y,
        state.calibration().distance,
        state.calibration().text_scale,
        state.calibration().background_alpha,
        rows.len(),
        state.scene().slots().iter().flatten().count(),
    )
}

fn collect_rendered_diagnostic_rows(
    state: &OverlayState,
    layout: &CaptionLayoutResult,
) -> Vec<RenderedDiagnosticRow> {
    let rows_by_id = collect_diagnostic_rows(state)
        .into_iter()
        .map(|row| (row.id.clone(), row))
        .collect::<HashMap<_, _>>();

    layout
        .visible_blocks
        .iter()
        .filter_map(|block| {
            let row = rows_by_id.get(block.id.as_str())?;
            Some(RenderedDiagnosticRow {
                row: row.clone(),
                bounds: block.bounds,
                visual_bounds: block.visual_bounds,
                secondary_present: block.secondary_line.is_some(),
                truncated_secondary: block.truncated_secondary,
            })
        })
        .collect()
}

fn format_overlay_visible_update_applied_log(revision: u64, row: &DiagnosticRow) -> String {
    format!(
        "overlay_visible_update_applied revision={} slot_index={} variant={} primary_len={} secondary_len={}",
        revision,
        row.slot_index,
        overlay_variant_name(row.block_variant),
        row.primary_text.len(),
        if row.secondary_enabled { row.secondary_text.len() } else { 0 },
    )
}

fn format_overlay_visible_update_rendered_log(
    revision: u64,
    rendered: &RenderedDiagnosticRow,
) -> String {
    format!(
        "overlay_visible_update_rendered revision={} slot_index={} variant={} primary_len={} secondary_len={} bounds={:.1},{:.1},{:.1},{:.1} visual_bounds={:.1},{:.1},{:.1},{:.1} secondary_present={} truncated_secondary={}",
        revision,
        rendered.row.slot_index,
        overlay_variant_name(rendered.row.block_variant),
        rendered.row.primary_text.len(),
        if rendered.row.secondary_enabled { rendered.row.secondary_text.len() } else { 0 },
        rendered.bounds.left_px,
        rendered.bounds.top_px,
        rendered.bounds.right_px,
        rendered.bounds.bottom_px,
        rendered.visual_bounds.left_px,
        rendered.visual_bounds.top_px,
        rendered.visual_bounds.right_px,
        rendered.visual_bounds.bottom_px,
        rendered.secondary_present,
        rendered.truncated_secondary,
    )
}

fn two_row_window_slot_signature(rows: &[RenderedDiagnosticRow]) -> Vec<u64> {
    let mut signature = rows
        .iter()
        .map(|row| row.row.slot_order)
        .collect::<Vec<_>>();
    signature.sort_unstable();
    signature
}

fn format_two_row_window_closed_log(
    revision: u64,
    window: &TwoRowWindowState,
    closed_at: Instant,
) -> String {
    let dwell_ms = closed_at.duration_since(window.started_at).as_millis() as u64;
    format!(
        "two_row_window_closed revision={} dwell_ms={} threshold_ms={} too_brief_to_be_perceptibly_stable={} row_count=2",
        revision,
        dwell_ms,
        TWO_ROW_WINDOW_STABILITY_THRESHOLD_MS,
        dwell_ms < TWO_ROW_WINDOW_STABILITY_THRESHOLD_MS,
    )
}

#[cfg(test)]
fn format_caption_block_summary(block: &CaptionBlock) -> String {
    format!(
        "id={} variant={} sec={}",
        block.id,
        caption_variant_name(block.block_variant),
        log_runtime_secondary_state(block.secondary_enabled, &block.secondary_text)
    )
}

#[cfg(test)]
fn format_caption_blocks_built_log(blocks: &[CaptionBlock]) -> String {
    format!(
        "caption_blocks_built block_count={} blocks=[{}]",
        blocks.len(),
        blocks
            .iter()
            .map(format_caption_block_summary)
            .collect::<Vec<_>>()
            .join("; ")
    )
}

#[cfg(test)]
fn short_tail(value: &str) -> String {
    let trimmed = value.trim();
    let without_prefix = trimmed
        .strip_prefix("peer:")
        .or_else(|| trimmed.strip_prefix("self:"))
        .unwrap_or(trimmed);
    let chars = without_prefix.chars().collect::<Vec<_>>();
    let start = chars.len().saturating_sub(8);
    chars[start..].iter().collect()
}

#[cfg(test)]
fn stable_short_hash(value: &str) -> u32 {
    let mut hash = 0x811c9dc5u32;
    for byte in value.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

#[cfg(test)]
fn debug_watermark_label_for_frame(revision: u64, blocks: &[CaptionBlock]) -> Option<String> {
    if !blocks.iter().any(CaptionBlock::has_drawable_text) {
        return None;
    }

    let active_peer = blocks.iter().find(|block| {
        block.channel == Some(CaptionChannel::PeerChannel)
            && block.block_variant == CaptionBlockVariant::ActivePeer
            && block.has_drawable_text()
    });

    let active_peer_tail = active_peer
        .map(|block| short_tail(&block.id))
        .unwrap_or_else(|| "none".to_string());

    let hash_input = active_peer
        .map(|block| format!("{}\n{}", block.primary_text, block.secondary_text))
        .unwrap_or_default();
    let hash = stable_short_hash(&hash_input) & 0xffff;

    let block_ids = blocks
        .iter()
        .filter(|block| block.has_drawable_text())
        .take(3)
        .map(|block| {
            let prefix = if block.channel == Some(CaptionChannel::PeerChannel) {
                "peer"
            } else {
                "self"
            };
            format!("{}:{}", prefix, short_tail(&block.id))
        })
        .collect::<Vec<_>>()
        .join(",");

    Some(format!(
        "DBG r{} ap={} h={:04x} b={}",
        revision, active_peer_tail, hash, block_ids
    ))
}

fn debug_overlay_for_frame(
    visual_debug_overlays: bool,
    revision: u64,
    blocks: &[CaptionBlock],
) -> Option<CaptionDebugOverlay> {
    let _ = (visual_debug_overlays, revision, blocks);
    None
}

fn append_optional_duration(line: &mut String, name: &str, duration_us: Option<u128>) {
    if let Some(duration_us) = duration_us {
        line.push_str(&format!(" {name}={duration_us}"));
    }
}

#[cfg(test)]
fn format_frame_rendered_log(
    layout: &CaptionLayoutResult,
    fully_transparent: bool,
    rendered_rows: &[RenderedDiagnosticRow],
    render_duration_us: Option<u128>,
) -> String {
    let mut line = format!(
        "frame_rendered visible_block_count={} fully_transparent={} secondary_present_count={} truncated_secondary_count={}",
        layout.visible_blocks.len(),
        fully_transparent,
        rendered_rows.iter().filter(|row| row.secondary_present).count(),
        rendered_rows.iter().filter(|row| row.truncated_secondary).count(),
    );
    append_optional_duration(&mut line, "render_duration_us", render_duration_us);
    line
}

fn format_frame_submitted_log(
    layout: &CaptionLayoutResult,
    revision: u64,
    fully_transparent: bool,
    overlay_visible_before: bool,
    overlay_visible_after: bool,
    should_show_after_submit: bool,
    submit_duration_us: Option<u128>,
    _rendered_rows: &[RenderedDiagnosticRow],
    stage_durations: FrameStageDurations,
) -> String {
    let mut line = format!(
        "frame_submitted revision={} visible_block_count={} self_block_count={} fully_transparent={} overlay_visible_before={} overlay_visible_after={} should_show_after_submit={}",
        revision,
        layout.visible_blocks.len(),
        visible_self_block_count(layout),
        fully_transparent,
        overlay_visible_before,
        overlay_visible_after,
        should_show_after_submit,
    );
    append_optional_duration(&mut line, "submit_duration_us", submit_duration_us);
    append_optional_duration(
        &mut line,
        "receive_to_submit_us",
        stage_durations.receive_to_submit_us,
    );
    line
}

fn visible_self_block_count(layout: &CaptionLayoutResult) -> usize {
    layout
        .visible_blocks
        .iter()
        .filter(|block| block.channel == Some(CaptionChannel::SelfChannel))
        .count()
}

fn format_frame_timing_log(
    revision: u64,
    timing: &FrameTimingSample,
    submit_duration_us: Option<u128>,
) -> String {
    let submit_duration = submit_duration_us
        .map(|duration| duration.to_string())
        .unwrap_or_else(|| "none".to_string());
    format!(
        "frame_timing revision={} dropped_frames={} post_submit_gpu_ms={:.2} total_render_gpu_ms={:.2} submit_duration_us={}",
        revision,
        timing.num_dropped_frames,
        timing.post_submit_gpu_ms,
        timing.total_render_gpu_ms,
        submit_duration,
    )
}

#[cfg(test)]
fn format_cache_stats_log(diagnostics: &RenderDiagnostics) -> String {
    format!(
        "cache_stats text_format_size={} layout_size={} line_size={} block_size={} text_format_hits={} text_format_misses={} font_warmup_attempts={} font_warmup_failures={} directwrite_layout_successes={} heuristic_layout_fallbacks={} layout_hits={} layout_misses={} line_hits={} line_misses={} block_hits={} block_misses={} style_bucket_source_counts=[{}]",
        diagnostics.text_format_cache_size,
        diagnostics.layout_cache_size,
        diagnostics.line_cache_size,
        diagnostics.block_cache_size,
        diagnostics.text_format_cache_hits,
        diagnostics.text_format_cache_misses,
        diagnostics.font_warmup_attempts,
        diagnostics.font_warmup_failures,
        diagnostics.directwrite_layout_success_count,
        diagnostics.heuristic_layout_fallback_count,
        diagnostics.layout_cache_hits,
        diagnostics.layout_cache_misses,
        diagnostics.line_cache_hits,
        diagnostics.line_cache_misses,
        diagnostics.block_cache_hits,
        diagnostics.block_cache_misses,
        format_style_bucket_source_counts(&diagnostics.style_bucket_source_counts),
    )
}

#[cfg(test)]
fn format_style_bucket_source_counts(counts: &[StyleBucketSourceCount]) -> String {
    counts
        .iter()
        .map(|count| format!("{:?}/{:?}:{}", count.bucket, count.source, count.count))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_peer_first_render_visibility_checkpoint_log(
    revision: u64,
    peer_ids: &[String],
    has_drawable_text: bool,
    overlay_visible_before: bool,
    should_show_after_submit: bool,
    hide_deadline_active: bool,
    first_texture_submitted: bool,
    redraw_requested: bool,
    visible_block_count: usize,
    self_block_count: usize,
    fully_transparent: bool,
) -> String {
    format!(
        "peer_first_render_visibility_checkpoint revision={} peer_count={} has_drawable_text={} overlay_visible_before={} should_show_after_submit={} hide_deadline_active={} first_texture_submitted={} redraw_requested={} visible_block_count={} self_block_count={} fully_transparent={}",
        revision,
        peer_ids.len(),
        has_drawable_text,
        overlay_visible_before,
        should_show_after_submit,
        hide_deadline_active,
        first_texture_submitted,
        redraw_requested,
        visible_block_count,
        self_block_count,
        fully_transparent,
    )
}

fn format_peer_first_render_visibility_desync_suspected_log(
    revision: u64,
    peer_ids: &[String],
    overlay_visible_before: bool,
    should_show_after_submit: bool,
    hide_deadline_active: bool,
    first_texture_submitted: bool,
    redraw_requested: bool,
    last_submitted_visible_row_count: usize,
) -> String {
    format!(
        "peer_first_render_visibility_desync_suspected revision={} peer_count={} overlay_visible_before={} should_show_after_submit={} hide_deadline_active={} first_texture_submitted={} redraw_requested={} last_submitted_visible_row_count={}",
        revision,
        peer_ids.len(),
        overlay_visible_before,
        should_show_after_submit,
        hide_deadline_active,
        first_texture_submitted,
        redraw_requested,
        last_submitted_visible_row_count,
    )
}

async fn log_runtime_info(logger: &OverlayLogger, message: String) -> Result<(), RuntimeFailure> {
    logger
        .info(message)
        .await
        .map_err(|error| RuntimeFailure::Bridge(error.to_string()))
}

async fn log_runtime_warn(logger: &OverlayLogger, message: String) -> Result<(), RuntimeFailure> {
    logger
        .warn(message)
        .await
        .map_err(|error| RuntimeFailure::Bridge(error.to_string()))
}

pub fn startup_error_from_bridge_error(error: BridgeError) -> StartupError {
    match error {
        BridgeError::Auth(message) => StartupError::BridgeAuth(message),
        BridgeError::Connect(message) | BridgeError::Protocol(message) => {
            StartupError::Other(format!("bridge startup failed: {message}"))
        }
        BridgeError::Disconnected => {
            StartupError::Other("bridge disconnected during startup".into())
        }
    }
}

fn startup_error_from_preflight(error: OpenVrStartupPreflightError) -> StartupError {
    match error {
        OpenVrStartupPreflightError::SteamVrNotInstalled => StartupError::SteamVrNotInstalled,
        OpenVrStartupPreflightError::SteamVrNotRunning => StartupError::SteamVrNotRunning,
        OpenVrStartupPreflightError::HmdNotFound => StartupError::HmdNotFound,
        OpenVrStartupPreflightError::Init(message) => StartupError::OpenVrInit(message),
    }
}

pub async fn run_with_manifest(manifest: OverlayManifest) -> i32 {
    run_with_manifest_and_profile(manifest, QuietTailProfile::P05).await
}

async fn run_with_manifest_and_profile(
    manifest: OverlayManifest,
    quiet_tail_profile: QuietTailProfile,
) -> i32 {
    let logger = match OverlayLogger::open(&manifest.log_dir, manifest.logging_mode).await {
        Ok(logger) => logger,
        Err(error) => {
            eprintln!("[overlay][ERROR] failed to initialize logging: {error}");
            return 1;
        }
    };

    let _ = logger.info("manifest_loaded").await;
    if let Err(error) = validate_manifest(&manifest) {
        emit_startup_failure(&logger, &error).await;
        return error.exit_code();
    }

    if manifest.app_version != env!("CARGO_PKG_VERSION") {
        let _ = logger
            .warn(&format!(
                "app_version mismatch accepted: manifest={} runtime={}",
                manifest.app_version,
                env!("CARGO_PKG_VERSION")
            ))
            .await;
    }

    let (mut bridge, snapshot) = match BridgeClient::connect(&manifest).await {
        Ok(result) => result,
        Err(error) => {
            let startup_error = startup_error_from_bridge_error(error);
            emit_startup_failure(&logger, &startup_error).await;
            return startup_error.exit_code();
        }
    };
    let _ = logger.info("bridge_connected").await;
    let _ = logger.info("bridge_authenticated").await;
    let _ = logger.info(format_snapshot_received_log(&snapshot)).await;

    if let Err(error) = perform_startup_preflight() {
        let startup_error = startup_error_from_preflight(error);
        let _ = bridge.close().await;
        emit_startup_failure(&logger, &startup_error).await;
        return startup_error.exit_code();
    }

    let (renderer, openvr) = match initialize_runtime_resources(&manifest, &logger).await {
        Ok(resources) => resources,
        Err(error) => {
            let _ = bridge.close().await;
            emit_startup_failure(&logger, &error).await;
            return error.exit_code();
        }
    };

    let _ = logger
        .info(format!("quiet_tail_profile={}", quiet_tail_profile.id()))
        .await;
    let mut owner =
        NativePresentationOwner::new_with_profile(snapshot, renderer, openvr, quiet_tail_profile);
    let initial_outcome = SnapshotApplyOutcome::Applied {
        incoming_revision: owner.runtime().state().snapshot().revision,
        current_revision: owner.runtime().state().snapshot().revision,
        visual_changed: owner.runtime().redraw_requested(),
        redraw_requested: owner.runtime().redraw_requested(),
    };
    let _ = logger
        .info(format_state_snapshot_log(
            &initial_outcome,
            owner.runtime().state(),
            owner.runtime().redraw_requested(),
        ))
        .await;
    let _ = owner
        .runtime
        .emit_snapshot_slot_correlation_if_changed(&logger)
        .await;
    let runtime_result = owner.run(&mut bridge, &logger).await;
    let reached_ready = owner.runtime().ready_sent();
    let _ = bridge.close().await;

    if let Err(error) = runtime_result.as_ref() {
        if !reached_ready {
            let startup_error = startup_error_from_runtime_failure(error.clone());
            emit_startup_failure(&logger, &startup_error).await;
            return startup_error.exit_code();
        }
    }

    match runtime_result {
        Ok(()) => 0,
        Err(RuntimeFailure::RuntimeDisconnected) => 1,
        Err(error) => {
            let _ = logger
                .error(format!("runtime_failure reason={}", error.failure_reason()))
                .await;
            let _ = logger
                .emit_stdout_event(&json!({
                    "type": "runtime_error",
                    "failure_reason": error.failure_reason(),
                }))
                .await;
            1
        }
    }
}

pub async fn run_cli(args: &[String]) -> i32 {
    if args.len() == 2 && args[1] == "--version" {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return 0;
    }

    if args.len() == 2 && args[1] == "--check-startup-contract" {
        println!(
            "{}",
            json!({
                "contract_version": EXPECTED_CONTRACT_VERSION,
                "app_version": env!("CARGO_PKG_VERSION"),
            })
        );
        return 0;
    }

    if args.len() != 3 || args[1] != "--config" {
        eprintln!(
            "usage: PuriPulyHeartOverlay --config <manifest.json> | --check-startup-contract | --version"
        );
        return 2;
    }

    let manifest = match load_manifest(Path::new(&args[2])) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!(
                "[overlay][ERROR] startup_failure reason={}",
                error.failure_reason()
            );
            emit_startup_failure_to_stderr(&error).await;
            return error.exit_code();
        }
    };

    let quiet_tail_profile = match resolve_quiet_tail_profile_from_env() {
        Ok(profile) => profile,
        Err(error) => {
            eprintln!(
                "[overlay][ERROR] startup_failure reason={}",
                error.failure_reason()
            );
            emit_startup_failure_to_stderr(&error).await;
            return error.exit_code();
        }
    };
    run_with_manifest_and_profile(manifest, quiet_tail_profile).await
}

fn startup_error_from_runtime_failure(error: RuntimeFailure) -> StartupError {
    match error {
        RuntimeFailure::Render(message) => StartupError::RendererInit(message),
        RuntimeFailure::OpenVr(message) => StartupError::OpenVrInit(message),
        RuntimeFailure::ReadinessTimedOut
        | RuntimeFailure::ReadinessCancelled
        | RuntimeFailure::ReadinessFailed => StartupError::RendererInit(error.to_string()),
        RuntimeFailure::Bridge(message) => StartupError::Other(message),
        RuntimeFailure::RuntimeDisconnected => {
            StartupError::Other("runtime disconnected before ready".into())
        }
        RuntimeFailure::Stopped => StartupError::Other("runtime stopped before ready".into()),
    }
}

fn startup_error_from_openvr(error: crate::openvr::OpenVrError) -> StartupError {
    StartupError::OpenVrInit(error.to_string())
}

fn startup_error_from_renderer(error: crate::renderer::CaptionRenderError) -> StartupError {
    StartupError::RendererInit(error.to_string())
}

#[cfg(test)]
fn prepare_openvr_runtime<T, P, F>(
    overlay_instance_id: &str,
    preflight: P,
    overlay_factory: F,
) -> Result<T, StartupError>
where
    P: FnOnce() -> Result<(), OpenVrStartupPreflightError>,
    F: FnOnce(&str) -> Result<T, OpenVrError>,
{
    preflight().map_err(startup_error_from_preflight)?;
    overlay_factory(overlay_instance_id).map_err(startup_error_from_openvr)
}

async fn initialize_runtime_resources(
    manifest: &OverlayManifest,
    logger: &OverlayLogger,
) -> Result<(CaptionRenderer, OpenVrOverlay), StartupError> {
    let openvr =
        OpenVrOverlay::new(&manifest.overlay_instance_id).map_err(startup_error_from_openvr)?;
    logger
        .info("openvr_ready")
        .await
        .map_err(|error| StartupError::Other(error.to_string()))?;
    let renderer = create_runtime_renderer(&openvr).map_err(startup_error_from_renderer)?;
    logger
        .info("renderer_resources_ready")
        .await
        .map_err(|error| StartupError::Other(error.to_string()))?;
    Ok((renderer, openvr))
}

fn create_runtime_renderer(
    openvr: &OpenVrOverlay,
) -> Result<CaptionRenderer, crate::renderer::CaptionRenderError> {
    #[cfg(windows)]
    {
        CaptionRenderer::new_for_openvr(&openvr.output_adapter())
    }

    #[cfg(not(windows))]
    {
        let _ = openvr;
        CaptionRenderer::new_for_test()
    }
}

impl PresentationRuntime {
    pub fn caption_blocks(&self) -> Vec<CaptionBlock> {
        self.caption_blocks_for_render(false)
    }

    pub fn caption_blocks_for_render(&self, visual_debug_prefixes: bool) -> Vec<CaptionBlock> {
        self.state
            .scene()
            .slots()
            .iter()
            .flatten()
            .map(|strip| caption_block_for_strip(strip, visual_debug_prefixes))
            .collect()
    }
}

fn caption_block_for_strip(strip: &OverlaySlot, visual_debug_prefixes: bool) -> CaptionBlock {
    let channel = if strip.channel == "peer" {
        CaptionChannel::PeerChannel
    } else {
        CaptionChannel::SelfChannel
    };
    let variant = match strip.block_variant {
        crate::state::OverlayPresentationBlockVariant::ActiveSelf => {
            CaptionBlockVariant::ActiveSelf
        }
        crate::state::OverlayPresentationBlockVariant::ActivePeer => {
            CaptionBlockVariant::ActivePeer
        }
        crate::state::OverlayPresentationBlockVariant::Finalized => CaptionBlockVariant::Finalized,
    };
    let prefix = if visual_debug_prefixes {
        peer_visual_debug_prefix_for_strip(strip)
    } else {
        None
    };
    let primary_text = apply_visual_debug_prefix(&strip.primary_text, prefix.as_deref());
    let secondary_text = apply_visual_debug_prefix(&strip.secondary_text, prefix.as_deref());

    CaptionBlock::new(strip.id.clone(), primary_text)
        .with_channel(channel)
        .with_variant(variant)
        .with_secondary_text(secondary_text, strip.secondary_enabled)
        .with_language_metadata(
            strip.primary_language.clone(),
            strip.secondary_language.clone(),
        )
        .with_visual_state(1.0, 0.0, 1.0)
        .with_slot(strip.slot_index, strip.anchor_top_px)
}

fn peer_visual_debug_prefix_for_strip(strip: &OverlaySlot) -> Option<String> {
    if strip.channel != "peer" {
        return None;
    }
    let turn_token = short_visual_debug_token(&strip.id);
    let stage_token = strip
        .update_id
        .as_deref()
        .map(short_visual_debug_token)
        .unwrap_or_else(|| "src".to_string());
    Some(format!("[P {}/{}]", turn_token, stage_token))
}

fn short_visual_debug_token(value: &str) -> String {
    let trimmed = value.trim();
    let without_prefix = trimmed
        .strip_prefix("peer:")
        .or_else(|| trimmed.strip_prefix("self:"))
        .unwrap_or(trimmed);
    let token = without_prefix
        .chars()
        .filter(|char| char.is_ascii_alphanumeric())
        .take(4)
        .collect::<String>()
        .to_ascii_lowercase();
    if token.is_empty() {
        "none".to_string()
    } else {
        token
    }
}

fn apply_visual_debug_prefix(text: &str, prefix: Option<&str>) -> String {
    let Some(prefix) = prefix else {
        return text.to_string();
    };
    if text.trim().is_empty() {
        return text.to_string();
    }
    format!("{} {}", prefix, text)
}

#[cfg(test)]
mod tests {
    use super::{
        collect_diagnostic_rows, collect_rendered_diagnostic_rows, debug_overlay_for_frame,
        debug_watermark_label_for_frame, diagnostic_row_signature, format_cache_stats_log,
        format_caption_blocks_built_log, format_frame_rendered_log, format_frame_submitted_log,
        format_frame_timing_log, format_overlay_visible_update_rendered_log,
        format_peer_first_render_visibility_checkpoint_log,
        format_peer_first_render_visibility_desync_suspected_log, format_snapshot_received_log,
        format_snapshot_slot_correlation_log, format_state_snapshot_log,
        format_two_row_window_closed_log, milliseconds_to_microseconds,
        peer_overlay_first_emit_block_ids_from_snapshot,
        peer_overlay_first_render_block_ids_from_caption_blocks, prepare_openvr_runtime,
        DiagnosticRow, FrameCycleOutcome, FrameStageDurations, FreshRetryChannel,
        NativeFreshSchedule, NativePresentationOwner, OverlayRuntime, RenderedDiagnosticRow,
        RuntimeFailure, SnapshotApplyOutcome, StartupError, TwoRowWindowState,
        NATIVE_FRESH_AUDIT_CAPACITY, NATIVE_FRESH_RETRY_MAX_COMPLETED,
    };
    use crate::bridge::{BridgeClient, BridgeIncoming};
    use crate::logging::{OverlayLogger, OverlayLoggingMode};
    use crate::manifest::{OverlayManifest, EXPECTED_CONTRACT_VERSION};
    use crate::openvr::{
        FakeOpenVr, FrameTimingSample, OpenVrError, OpenVrStartupPreflightError,
        OverlayFrameSubmitter, SpatialReanchorOutcome,
    };
    use crate::presentation::{
        PresentationBackend, PresentationCause, PresentationCauseChannel, PresentationCauseKind,
        PresentationCauses, PresentationCorrelation, PresentationOutcome, PresentationStage,
    };
    use crate::renderer::{
        CaptionBlock, CaptionBlockVariant, CaptionChannel, CaptionLayoutPolicy,
        CaptionPresentation, CaptionRenderer, FontLanguageBucket, FontSource, RenderDiagnostics,
        RenderedFrame, StyleBucketSourceCount,
    };
    use crate::state::{
        OverlayPresentationBlock, OverlayPresentationBlockVariant, OverlayPresentationCalibration,
        OverlayPresentationSnapshot,
    };
    use futures_util::{SinkExt, StreamExt};
    use serde_json::json;
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::io;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    #[test]
    fn native_fresh_audit_capacity_covers_simultaneous_production_journey() {
        let maximum_journey = 2 * (NATIVE_FRESH_RETRY_MAX_COMPLETED as usize + 2);
        assert!(NATIVE_FRESH_AUDIT_CAPACITY >= maximum_journey);
    }

    #[test]
    fn direct_native_owner_uses_and_reports_p05_defaults() {
        let mut owner = NativePresentationOwner::new(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );
        assert_eq!(owner.retry_profile, "p05");
        assert_eq!(owner.retry_policy.deadline, Duration::from_millis(500));
        assert_eq!(owner.retry_policy.max_completed, 5);
        owner
            .runtime
            .presentation_diagnostics
            .accept_logical_revision(PresentationBackend::Test, 1, PresentationCauses::default());
        assert_eq!(
            owner
                .runtime
                .presentation_diagnostics
                .records()
                .back()
                .unwrap()
                .retry_profile,
            "p05"
        );
    }

    #[test]
    fn native_fresh_audit_drops_oldest_with_bounded_count() {
        let mut owner = NativePresentationOwner::new(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );
        let now = tokio::time::Instant::now();
        for generation in 0..=NATIVE_FRESH_AUDIT_CAPACITY as u64 {
            owner.push_fresh_retry_audit(
                NativeFreshSchedule {
                    channel: FreshRetryChannel::SelfChannel,
                    trigger_generation: generation,
                    required_scene_generation: generation,
                    target_identity: "[\"target\"]".into(),
                    completed: 0,
                    max_completed: 20,
                    phase: crate::state::NativeQuietTailPhase::Final,
                    episode_generation: generation,
                    deadline: now,
                    next_due: now,
                },
                "scheduled",
            );
        }

        assert_eq!(owner.fresh_retry_audit.len(), NATIVE_FRESH_AUDIT_CAPACITY);
        assert_eq!(owner.fresh_retry_audit_dropped, 1);
        assert_eq!(
            owner.fresh_retry_audit.front().unwrap().trigger_generation,
            1
        );
    }

    fn schedule(
        channel: FreshRetryChannel,
        trigger_generation: u64,
        required_scene_generation: u64,
        now: Instant,
    ) -> NativeFreshSchedule {
        NativeFreshSchedule {
            channel,
            trigger_generation,
            required_scene_generation,
            target_identity: "[\"target\"]".into(),
            completed: 0,
            max_completed: 20,
            phase: crate::state::NativeQuietTailPhase::Final,
            episode_generation: trigger_generation,
            deadline: now + Duration::from_secs(2),
            next_due: now + Duration::from_millis(100),
        }
    }

    #[test]
    fn quiet_tail_deadline_is_inclusive_at_exact_due_boundary() {
        let now = Instant::now();
        let mut value = schedule(FreshRetryChannel::SelfChannel, 1, 1, now);
        value.next_due = now + Duration::from_millis(100);
        value.deadline = value.next_due;
        assert!(!value.expired_at(value.deadline));
        assert!(value.expired_at(value.deadline + Duration::from_nanos(1)));
    }

    #[test]
    fn normal_and_coalesced_capture_only_due_schedules_at_100ms() {
        let mut owner = NativePresentationOwner::new(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );
        let now = Instant::now();
        let self_schedule = schedule(FreshRetryChannel::SelfChannel, 1, 1, now);
        let peer_schedule = schedule(FreshRetryChannel::Peer, 2, 1, now);
        owner.self_schedule = Some(self_schedule);
        owner.peer_schedule = Some(peer_schedule);
        assert!(owner
            .active_fresh_schedules(now + Duration::from_millis(99))
            .is_empty());
        let due = owner.active_fresh_schedules(now + Duration::from_millis(100));
        assert_eq!(due.len(), 2);
        assert!(due
            .iter()
            .any(|value| value.channel == FreshRetryChannel::SelfChannel));
        assert!(due
            .iter()
            .any(|value| value.channel == FreshRetryChannel::Peer));
    }

    #[tokio::test]
    async fn retry_log_records_safe_phase_profile_and_schedule_maximum() {
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(OverlayLoggingMode::Detailed, stdout.clone());
        let mut owner = NativePresentationOwner::new_with_profile(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
            crate::manifest::QuietTailProfile::P05,
        );
        let now = Instant::now();
        let mut value = schedule(FreshRetryChannel::Peer, 3, 1, now);
        value.target_identity = "peer:must-not-log".into();
        value.phase = crate::state::NativeQuietTailPhase::Stream;
        value.episode_generation = 77;
        value.max_completed = 4;
        owner
            .record_fresh_retry(&logger, value, "scheduled")
            .await
            .unwrap();
        let log = String::from_utf8(stdout.contents()).unwrap();
        assert!(log.contains("phase=stream"));
        assert!(log.contains("profile=p05"));
        assert!(log.contains("max=4"));
        assert!(!log.contains("must-not-log"));
        assert!(!log.contains("77"));
    }

    #[tokio::test]
    async fn stream_generation_replacement_preserves_due_and_budget_while_final_resets() {
        fn snapshot(
            revision: u64,
            generation: u64,
            phase: &str,
            episode: u64,
        ) -> OverlayPresentationSnapshot {
            serde_json::from_value(serde_json::json!({
                "revision": revision,
                "native_fresh_render_generations": {"peer": generation},
                "native_fresh_render_targets": {"peer": "peer:stable"},
                "native_quiet_tail_episodes": {"peer": {"phase": phase, "generation": episode}},
                "blocks": [{
                    "id": "peer:stable", "occupant_key": "peer:stable", "appearance_seq": 1,
                    "channel": "peer", "block_variant": if phase == "final" { "finalized" } else { "active_peer" },
                    "primary_text": "visible", "secondary_text": "", "secondary_enabled": false
                }]
            })).unwrap()
        }

        let logger = OverlayLogger::open(std::env::temp_dir(), OverlayLoggingMode::Detailed)
            .await
            .unwrap();
        let mut owner = NativePresentationOwner::new(
            snapshot(1, 1, "stream", 7),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        let first_schedule = owner.peer_schedule.as_ref().unwrap().clone();
        owner.peer_schedule.as_mut().unwrap().completed = 2;
        owner.peer_accounting.as_mut().unwrap().completed = 2;
        owner.runtime.apply_snapshot(snapshot(2, 2, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        let replaced_generation = owner.peer_schedule.as_ref().unwrap();
        assert_eq!(replaced_generation.trigger_generation, 2);
        assert_eq!(replaced_generation.required_scene_generation, 2);
        assert_eq!(replaced_generation.target_identity, "peer:stable");
        assert_eq!(replaced_generation.completed, 2);
        assert_eq!(replaced_generation.max_completed, 4);
        assert_eq!(replaced_generation.next_due, first_schedule.next_due);
        assert_eq!(replaced_generation.deadline, first_schedule.deadline);

        owner.peer_accounting.as_mut().unwrap().completed = 4;
        owner.peer_schedule = None;
        owner.runtime.apply_snapshot(snapshot(3, 3, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner.peer_schedule.is_none());
        assert_eq!(owner.peer_accounting.as_ref().unwrap().completed, 4);

        owner.peer_accounting.as_mut().unwrap().completed = 0;
        owner.peer_accounting.as_mut().unwrap().deadline =
            Instant::now() - Duration::from_millis(1);
        owner.runtime.apply_snapshot(snapshot(4, 4, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner.peer_schedule.is_none());

        owner.runtime.apply_snapshot(OverlayPresentationSnapshot {
            revision: 5,
            ..OverlayPresentationSnapshot::default()
        });
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        owner.runtime.apply_snapshot(OverlayPresentationSnapshot {
            revision: 6,
            ..OverlayPresentationSnapshot::default()
        });
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        owner.runtime.apply_snapshot(snapshot(7, 5, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner.peer_schedule.is_none());

        owner.runtime.apply_snapshot(snapshot(8, 6, "stream", 8));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner.peer_schedule.is_some());

        owner.runtime.apply_snapshot(snapshot(9, 7, "final", 9));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        let final_schedule = owner.peer_schedule.as_ref().unwrap();
        assert_eq!(final_schedule.completed, 0);
        assert_eq!(final_schedule.max_completed, 5);
    }

    #[tokio::test]
    async fn diagnostic_profiles_schedule_zero_or_exactly_one_delayed_due_completion() {
        fn snapshot(revision: u64, generation: u64) -> OverlayPresentationSnapshot {
            serde_json::from_value(serde_json::json!({
                "revision": revision,
                "native_fresh_render_generations": {"self": generation},
                "native_fresh_render_targets": {"self": "self:stable"},
                "native_quiet_tail_episodes": {"self": {"phase": "final", "generation": 1}},
                "blocks": [{
                    "id": "self:stable", "occupant_key": "self:stable", "appearance_seq": 1,
                    "channel": "self", "block_variant": "finalized", "primary_text": "visible",
                    "secondary_text": "", "secondary_enabled": false
                }]
            }))
            .unwrap()
        }

        let logger = OverlayLogger::open(std::env::temp_dir(), OverlayLoggingMode::Detailed)
            .await
            .unwrap();
        let mut none = NativePresentationOwner::new_with_profile(
            snapshot(1, 1),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
            crate::manifest::QuietTailProfile::NoRetry,
        );
        none.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(none.self_schedule.is_none());

        let mut one = NativePresentationOwner::new_with_profile(
            snapshot(1, 1),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
            crate::manifest::QuietTailProfile::OneRetry,
        );
        one.reconcile_fresh_schedules(&logger).await.unwrap();
        let scheduled = one.self_schedule.as_ref().unwrap().clone();
        assert_eq!(scheduled.max_completed, 1);
        assert!(!scheduled.expired_at(scheduled.next_due + Duration::from_millis(50)));
        assert_eq!(
            one.active_fresh_schedules(scheduled.next_due + Duration::from_millis(50))
                .len(),
            1
        );
        one.self_accounting.as_mut().unwrap().completed = 1;
        one.self_schedule = None;
        one.runtime.apply_snapshot(snapshot(2, 2));
        one.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(one.self_schedule.is_none());
        assert_eq!(one.self_accounting.as_ref().unwrap().completed, 1);
    }

    fn correlation_for(
        schedule: &NativeFreshSchedule,
        scene_generation: u64,
        kind: PresentationCauseKind,
    ) -> PresentationCorrelation {
        let mut logical_causes = PresentationCauses::default();
        logical_causes.insert(PresentationCause {
            kind,
            channel: Some(match schedule.channel {
                FreshRetryChannel::SelfChannel => PresentationCauseChannel::SelfChannel,
                FreshRetryChannel::Peer => PresentationCauseChannel::Peer,
            }),
            trigger_generation: Some(schedule.trigger_generation),
        });
        PresentationCorrelation {
            logical_revision: 1,
            render_generation: 1,
            submission_attempt: 1,
            scene_generation,
            logical_causes,
        }
    }

    #[test]
    fn normal_submission_requires_exact_captured_intent_and_causal_scene() {
        let now = Instant::now();
        let mut captured = schedule(FreshRetryChannel::SelfChannel, 7, 11, now);
        captured.completed = 2;
        let correlation = correlation_for(&captured, 11, PresentationCauseKind::ActiveRetryIntent);
        assert!(
            NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation,
                &captured,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(7),
                Some("[\"target\"]"),
            )
        );

        let replacement = schedule(FreshRetryChannel::SelfChannel, 8, 12, now);
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation,
                &replacement,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(8),
                Some("[\"target\"]"),
            )
        );

        let mut transferred = captured.clone();
        transferred.trigger_generation = 8;
        transferred.required_scene_generation = 12;
        assert!(
            NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation_for(&captured, 12, PresentationCauseKind::ActiveRetryIntent),
                &transferred,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(8),
                Some("[\"target\"]"),
            )
        );
        transferred.target_identity = "[\"different-target\"]".into();
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation_for(&captured, 12, PresentationCauseKind::ActiveRetryIntent),
                &transferred,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(8),
                Some("[\"different-target\"]"),
            )
        );
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation_for(&captured, 10, PresentationCauseKind::ActiveRetryIntent),
                &captured,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(7),
                Some("[\"target\"]"),
            )
        );
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation,
                &captured,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(7),
                None,
            )
        );
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation,
                &captured,
                &captured,
                PresentationCauseKind::ActiveRetryIntent,
                Some(7),
                Some("[\"different-target\"]"),
            )
        );
    }

    #[test]
    fn retry_submission_satisfies_only_included_schedule_identity() {
        let now = Instant::now();
        let due_self = schedule(FreshRetryChannel::SelfChannel, 3, 20, now);
        let staggered_peer = schedule(FreshRetryChannel::Peer, 4, 20, now);
        let correlation = correlation_for(&due_self, 20, PresentationCauseKind::NativeFreshRetry);
        assert!(
            NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation,
                &due_self,
                &due_self,
                PresentationCauseKind::NativeFreshRetry,
                Some(3),
                Some("[\"target\"]"),
            )
        );
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                correlation,
                &staggered_peer,
                &staggered_peer,
                PresentationCauseKind::NativeFreshRetry,
                Some(4),
                Some("[\"target\"]"),
            )
        );
    }

    #[test]
    fn initial_reconcile_failure_path_releases_owner_resources_and_handle() {
        let mut owner = NativePresentationOwner::new(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );
        let retry = owner.retry_handle();

        let result = owner.finish_initial_reconcile(Err(RuntimeFailure::Bridge(
            "injected reconcile logging failure".into(),
        )));

        assert!(matches!(result, Err(RuntimeFailure::Bridge(_))));
        assert!(owner.resources_released());
        assert!(!retry.request());
    }
    use std::time::Duration;
    use tokio::io::AsyncWrite;
    use tokio::time::Instant;

    #[derive(Clone, Copy)]
    enum ControlledSinkMode {
        Success,
        Error,
        Pending,
    }

    #[derive(Clone)]
    struct ControlledSink {
        mode: ControlledSinkMode,
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl ControlledSink {
        fn new(mode: ControlledSinkMode) -> Self {
            Self {
                mode,
                bytes: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn contents(&self) -> Vec<u8> {
            self.bytes.lock().unwrap().clone()
        }
    }

    impl AsyncWrite for ControlledSink {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            bytes: &[u8],
        ) -> Poll<Result<usize, io::Error>> {
            match self.mode {
                ControlledSinkMode::Success => {
                    self.bytes.lock().unwrap().extend_from_slice(bytes);
                    Poll::Ready(Ok(bytes.len()))
                }
                ControlledSinkMode::Error => Poll::Ready(Err(io::Error::other("sink failed"))),
                ControlledSinkMode::Pending => Poll::Pending,
            }
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
            match self.mode {
                ControlledSinkMode::Success => Poll::Ready(Ok(())),
                ControlledSinkMode::Error => Poll::Ready(Err(io::Error::other("sink failed"))),
                ControlledSinkMode::Pending => Poll::Pending,
            }
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), io::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    fn controlled_logger(mode: OverlayLoggingMode, stdout: ControlledSink) -> OverlayLogger {
        OverlayLogger::from_streams(
            Box::pin(stdout),
            Box::pin(ControlledSink::new(ControlledSinkMode::Success)),
            mode,
        )
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

    fn slot_block(
        id: &str,
        occupant_key: &str,
        appearance_seq: u64,
        channel: &str,
        primary_text: &str,
    ) -> OverlayPresentationBlock {
        OverlayPresentationBlock {
            id: id.to_string(),
            occupant_key: occupant_key.to_string(),
            appearance_seq,
            channel: channel.to_string(),
            block_variant: OverlayPresentationBlockVariant::Finalized,
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

    struct SpatialSubmitProbe {
        outcome: SpatialReanchorOutcome,
        operations: Vec<&'static str>,
    }

    impl OverlayFrameSubmitter for SpatialSubmitProbe {
        fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
            self.operations.push("reanchor");
            Ok(self.outcome)
        }

        fn submit_frame(&mut self, _frame: &RenderedFrame) -> Result<(), OpenVrError> {
            self.operations.push("submit");
            Ok(())
        }

        fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
            self.operations.push(if visible { "show" } else { "hide" });
            Ok(())
        }
    }

    async fn controlled_test_bridge(
        followup: Option<(Arc<tokio::sync::Notify>, OverlayPresentationSnapshot)>,
    ) -> (BridgeClient, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();
            let _auth = ws.next().await.unwrap().unwrap();
            ws.send(Message::Text(
                json!({
                    "type": "snapshot",
                    "payload": OverlayPresentationSnapshot::default()
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            if let Some((readiness_started, snapshot)) = followup {
                readiness_started.notified().await;
                ws.send(Message::Text(
                    json!({"type": "snapshot", "payload": snapshot})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
            }
            while ws.next().await.is_some() {}
        });
        let manifest = OverlayManifest {
            contract_version: EXPECTED_CONTRACT_VERSION,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            overlay_instance_id: "spatial-runtime-unit".to_string(),
            bridge_url: format!("ws://{address}"),
            session_token: "unit-token".to_string(),
            parent_pid: 1,
            startup_deadline_ms: 3000,
            log_dir: std::env::temp_dir().display().to_string(),
            log_level: "INFO".to_string(),
            locale: "en".to_string(),
            logging_mode: OverlayLoggingMode::Detailed,
        };
        let (bridge, _) = BridgeClient::connect(&manifest).await.unwrap();
        (bridge, server)
    }

    #[tokio::test]
    async fn spatial_diagnostic_write_failure_cannot_block_pose_unavailable_texture_submit() {
        let (mut bridge, server) = controlled_test_bridge(None).await;
        let renderer = CaptionRenderer::new_for_test().unwrap();
        let logger = controlled_logger(
            OverlayLoggingMode::Detailed,
            ControlledSink::new(ControlledSinkMode::Error),
        );
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 1,
            calibration: OverlayPresentationCalibration {
                anchor: "spatial_locked".to_string(),
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![block("self:A", "self", "A", "", true)],
            native_fresh_render_generations: None,
        });
        runtime.first_texture_submitted = true;
        runtime.overlay_visible = true;
        let mut submitter = SpatialSubmitProbe {
            outcome: SpatialReanchorOutcome::PoseUnavailable,
            operations: Vec::new(),
        };

        let result = runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await;

        assert!(matches!(result, Err(RuntimeFailure::Bridge(_))));
        assert_eq!(submitter.operations, vec!["reanchor", "submit"]);
        drop(bridge);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn pending_spatial_diagnostic_cannot_block_first_visible_pose_unavailable_reveal() {
        let (mut bridge, server) = controlled_test_bridge(None).await;
        let renderer = CaptionRenderer::new_for_test().unwrap();
        let logger = controlled_logger(
            OverlayLoggingMode::Basic,
            ControlledSink::new(ControlledSinkMode::Pending),
        );
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 1,
            calibration: OverlayPresentationCalibration {
                anchor: "spatial_locked".to_string(),
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![block("self:A", "self", "A", "", true)],
            native_fresh_render_generations: None,
        });
        runtime.first_texture_submitted = true;
        let mut submitter = SpatialSubmitProbe {
            outcome: SpatialReanchorOutcome::PoseUnavailable,
            operations: Vec::new(),
        };

        let result = tokio::time::timeout(
            Duration::from_millis(50),
            runtime.submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger),
        )
        .await;

        assert!(result.is_err());
        assert_eq!(submitter.operations, vec!["reanchor", "submit", "show"]);
        assert!(runtime.overlay_visible);
        drop(bridge);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn spatial_diagnostic_write_failure_cannot_drop_preempted_latest_frame() {
        let readiness_started = Arc::new(tokio::sync::Notify::new());
        let latest_snapshot = OverlayPresentationSnapshot {
            revision: 3,
            calibration: OverlayPresentationCalibration {
                anchor: "spatial_locked".to_string(),
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![
                block("self:B", "self", "B", "", true),
                block("self:C", "self", "C", "", true),
            ],
            native_fresh_render_generations: None,
        };
        let (mut bridge, server) =
            controlled_test_bridge(Some((readiness_started.clone(), latest_snapshot))).await;
        let renderer = CaptionRenderer::new_for_test().unwrap();
        renderer.set_test_readiness_pending_yields_on_call(1, usize::MAX);
        renderer.set_test_readiness_started_notify_on_call(1, readiness_started);
        let failing_logger = controlled_logger(
            OverlayLoggingMode::Detailed,
            ControlledSink::new(ControlledSinkMode::Error),
        );
        let healthy_logger = controlled_logger(
            OverlayLoggingMode::Detailed,
            ControlledSink::new(ControlledSinkMode::Success),
        );
        let calibration = OverlayPresentationCalibration {
            anchor: "spatial_locked".to_string(),
            ..OverlayPresentationCalibration::default()
        };
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 1,
            calibration: calibration.clone(),
            blocks: vec![block("self:A", "self", "A", "", true)],
            native_fresh_render_generations: None,
        });
        runtime.first_texture_submitted = true;
        runtime.overlay_visible = true;
        runtime.redraw_requested = false;
        runtime.spatial_lock.take_pending();
        runtime.pending_spatial_diagnostics.clear();
        runtime.apply_snapshot(OverlayPresentationSnapshot {
            revision: 2,
            calibration,
            blocks: vec![
                block("self:A", "self", "A", "", true),
                block("self:B", "self", "B", "", true),
            ],
            native_fresh_render_generations: None,
        });
        let mut submitter = SpatialSubmitProbe {
            outcome: SpatialReanchorOutcome::Applied,
            operations: Vec::new(),
        };

        let preempted = runtime
            .submit_frame_if_needed_with_timing(
                &renderer,
                &mut submitter,
                &mut bridge,
                &failing_logger,
                None,
                None,
                true,
            )
            .await
            .unwrap();

        let FrameCycleOutcome::Preempted(Ok(BridgeIncoming::Snapshot(snapshot))) = preempted else {
            panic!("expected latest snapshot preemption");
        };
        assert!(submitter.operations.is_empty());
        runtime.apply_snapshot(snapshot);
        runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &healthy_logger)
            .await
            .unwrap();
        assert_eq!(submitter.operations, vec!["reanchor", "submit"]);
        assert_eq!(runtime.state().snapshot().revision, 3);
        drop(bridge);
        server.await.unwrap();
    }

    #[test]
    fn runtime_accumulates_normal_external_self_and_peer_attempt_causes() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        runtime.apply_snapshot(OverlayPresentationSnapshot {
            revision: 1,
            calibration: OverlayPresentationCalibration::default(),
            native_fresh_render_generations: None,
            blocks: vec![block("synthetic", "self", "synthetic", "", true)],
        });
        assert!(runtime.request_native_presentation_retry());
        assert!(runtime.request_fresh_presentation_retry(FreshRetryChannel::SelfChannel, 4));
        assert!(runtime.request_fresh_presentation_retry(FreshRetryChannel::Peer, 9));

        let causes = runtime.pending_presentation_causes.to_vec();
        assert!(causes
            .iter()
            .any(|cause| cause.kind == PresentationCauseKind::Startup));
        assert!(causes
            .iter()
            .any(|cause| cause.kind == PresentationCauseKind::SceneUpdate
                && cause.trigger_generation == Some(1)));
        assert!(causes
            .iter()
            .any(|cause| cause.kind == PresentationCauseKind::ExternalRetry));
        assert!(causes.iter().any(|cause| cause.channel
            == Some(PresentationCauseChannel::SelfChannel)
            && cause.trigger_generation == Some(4)));
        assert!(causes.iter().any(
            |cause| cause.channel == Some(PresentationCauseChannel::Peer)
                && cause.trigger_generation == Some(9)
        ));
    }

    #[test]
    fn failed_attempt_causes_are_retained_with_newer_pending_activity() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        let attempt_causes = std::mem::take(&mut runtime.pending_presentation_causes);
        runtime.presentation_diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            attempt_causes,
        );
        let correlation = runtime
            .presentation_diagnostics
            .begin_presentation(0, attempt_causes)
            .unwrap();
        runtime.request_native_presentation_retry();

        runtime.retain_failed_presentation_causes(correlation);

        let causes = runtime.pending_presentation_causes.to_vec();
        assert!(causes
            .iter()
            .any(|cause| cause.kind == PresentationCauseKind::Startup));
        assert!(causes
            .iter()
            .any(|cause| cause.kind == PresentationCauseKind::ExternalRetry));
    }

    #[test]
    fn compositor_metric_conversion_preserves_unavailable_values() {
        assert_eq!(milliseconds_to_microseconds(1.25), Some(1_250));
        assert_eq!(milliseconds_to_microseconds(f32::NAN), None);
        assert_eq!(milliseconds_to_microseconds(f32::INFINITY), None);
        assert_eq!(milliseconds_to_microseconds(-1.0), None);
    }

    #[test]
    fn caption_blocks_follow_snapshot_order_exactly() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 3,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                block("peer:1", "peer", "peer one", "원문", true),
                block("self:2", "self", "self two", "translated", true),
            ],
        });

        let blocks = runtime.caption_blocks();

        assert_eq!(
            blocks
                .iter()
                .map(|block| (block.id.as_str(), block.primary_text.as_str()))
                .collect::<Vec<_>>(),
            vec![("peer:1", "peer one"), ("self:2", "self two"),]
        );
    }

    #[test]
    fn caption_blocks_for_render_prefixes_peer_lines_when_visual_debug_is_enabled() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 3,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                OverlayPresentationBlock {
                    id: "peer:41c6ffff-1111-2222-3333-444455556666".to_string(),
                    occupant_key: "peer-active".to_string(),
                    appearance_seq: 1,
                    channel: "peer".to_string(),
                    block_variant: OverlayPresentationBlockVariant::ActivePeer,
                    primary_text: String::new(),
                    secondary_text: "peer source".to_string(),
                    secondary_enabled: true,
                    primary_language: None,
                    secondary_language: None,
                    update_id: None,
                    origin_wall_clock_ms: None,
                    session_scope: None,
                },
                OverlayPresentationBlock {
                    id: "peer:9c27ffff-1111-2222-3333-444455556666".to_string(),
                    occupant_key: "peer-final".to_string(),
                    appearance_seq: 2,
                    channel: "peer".to_string(),
                    block_variant: OverlayPresentationBlockVariant::Finalized,
                    primary_text: "peer translation".to_string(),
                    secondary_text: "peer original".to_string(),
                    secondary_enabled: true,
                    primary_language: None,
                    secondary_language: None,
                    update_id: Some("3bd7ffff-1111-2222-3333-444455556666".to_string()),
                    origin_wall_clock_ms: None,
                    session_scope: None,
                },
            ],
        });

        let normal_blocks = runtime.caption_blocks_for_render(false);
        let debug_blocks = runtime.caption_blocks_for_render(true);

        assert_eq!(normal_blocks[0].secondary_text, "peer source");
        assert_eq!(normal_blocks[1].primary_text, "peer translation");
        assert_eq!(debug_blocks[0].primary_text, "");
        assert_eq!(debug_blocks[0].secondary_text, "[P 41c6/src] peer source");
        assert_eq!(
            debug_blocks[1].primary_text,
            "[P 9c27/3bd7] peer translation"
        );
        assert_eq!(
            debug_blocks[1].secondary_text,
            "[P 9c27/3bd7] peer original"
        );
    }

    #[test]
    fn apply_snapshot_replaces_snapshot_blocks_and_calibration_without_retaining_removed_rows() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 1,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("self:1", "self", "self one", "", true)],
        });

        runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 2,
            calibration: OverlayPresentationCalibration {
                distance: 1.5,
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![block("peer:2", "peer", "peer two", "", true)],
        });

        let blocks = runtime.caption_blocks();

        assert_eq!(
            runtime
                .state()
                .snapshot()
                .blocks
                .iter()
                .map(|block| (block.id.as_str(), block.primary_text.as_str()))
                .collect::<Vec<_>>(),
            vec![("peer:2", "peer two")]
        );
        assert_eq!(
            blocks
                .iter()
                .map(|block| (block.id.as_str(), block.primary_text.as_str()))
                .collect::<Vec<_>>(),
            vec![("peer:2", "peer two")]
        );
        assert_eq!(runtime.state().snapshot().revision, 2);
        assert_eq!(runtime.state().snapshot().calibration.distance, 1.5);
    }

    #[test]
    fn runtime_orders_snapshot_blocks_by_appearance_seq() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 4,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                slot_block("peer:newer", "peer:newer", 2, "peer", "newer"),
                slot_block("self:older", "self:older", 1, "self", "older"),
            ],
        });

        let blocks = runtime.caption_blocks();

        assert_eq!(
            blocks
                .iter()
                .map(|block| (block.id.as_str(), block.primary_text.as_str()))
                .collect::<Vec<_>>(),
            vec![("self:older", "older"), ("peer:newer", "newer"),]
        );
    }

    #[test]
    fn runtime_converts_active_peer_snapshot_to_active_peer_caption_block() {
        let mut active_peer = slot_block("peer:active", "peer:turn-1", 1, "peer", "");
        active_peer.block_variant = OverlayPresentationBlockVariant::ActivePeer;
        active_peer.secondary_text = "Can you hear me?".into();
        active_peer.secondary_enabled = true;
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 5,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![active_peer],
        });

        let blocks = runtime.caption_blocks();

        assert_eq!(blocks[0].id, "peer:active");
        assert_eq!(blocks[0].block_variant, CaptionBlockVariant::ActivePeer);
        assert_eq!(blocks[0].channel, Some(CaptionChannel::PeerChannel));
        assert_eq!(blocks[0].primary_text, "");
        assert_eq!(blocks[0].secondary_text, "Can you hear me?");
        assert!(blocks[0].secondary_enabled);
    }

    #[test]
    fn runtime_detects_peer_overlay_first_emit_blocks_from_snapshot() {
        let snapshot = OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 4,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                slot_block("self:older", "self:older", 1, "self", "older"),
                slot_block("peer:newer", "peer:newer", 2, "peer", "newer"),
            ],
        };

        assert_eq!(
            peer_overlay_first_emit_block_ids_from_snapshot(&snapshot),
            vec!["peer:newer".to_string()]
        );
    }

    #[test]
    fn runtime_detects_active_peer_first_emit_blocks_from_snapshot() {
        let mut active_peer = slot_block("peer:active", "peer:turn-1", 1, "peer", "");
        active_peer.block_variant = OverlayPresentationBlockVariant::ActivePeer;
        active_peer.secondary_text = "source".into();
        active_peer.secondary_enabled = true;
        let snapshot = OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 6,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![active_peer],
        };

        assert_eq!(
            peer_overlay_first_emit_block_ids_from_snapshot(&snapshot),
            vec!["peer:active".to_string()]
        );
    }

    #[test]
    fn runtime_only_detects_peer_first_render_for_canonical_pending_peer_block_ids() {
        let pending = HashSet::from([
            String::from("peer:11111111-1111-1111-1111-111111111111"),
            String::from("peer:22222222-2222-2222-2222-222222222222"),
            String::from("peer:missing"),
        ]);
        let blocks = vec![
            CaptionBlock::new("self:older", "older")
                .with_channel(CaptionChannel::SelfChannel)
                .with_variant(CaptionBlockVariant::Finalized),
            CaptionBlock::new("peer:not-pending", "not pending")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::Finalized),
            CaptionBlock::new("peer:active", "active")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::ActiveSelf),
            CaptionBlock::new("peer:blank", "")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::Finalized),
            CaptionBlock::new("peer:11111111-1111-1111-1111-111111111111", "translated")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::Finalized),
            CaptionBlock::new("peer:22222222-2222-2222-2222-222222222222", "newer")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::Finalized),
            CaptionBlock::new(
                "peer:33333333-3333-3333-3333-333333333333/render-primary",
                "synthetic suffix form",
            )
            .with_channel(CaptionChannel::PeerChannel)
            .with_variant(CaptionBlockVariant::Finalized),
        ];

        assert_eq!(
            peer_overlay_first_render_block_ids_from_caption_blocks(&blocks, &pending),
            vec![
                "peer:11111111-1111-1111-1111-111111111111".to_string(),
                "peer:22222222-2222-2222-2222-222222222222".to_string(),
            ]
        );
    }

    #[test]
    fn runtime_detects_active_peer_first_render_for_pending_peer_block_ids() {
        let pending = HashSet::from([String::from("peer:active")]);
        let blocks = vec![
            CaptionBlock::new("peer:active", "source")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::ActivePeer),
            CaptionBlock::new("peer:not-pending", "source")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::ActivePeer),
        ];

        assert_eq!(
            peer_overlay_first_render_block_ids_from_caption_blocks(&blocks, &pending),
            vec!["peer:active".to_string()]
        );
    }

    #[test]
    fn runtime_starts_empty_when_snapshot_has_no_blocks() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 0,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![],
        });

        assert!(runtime.caption_blocks().is_empty());
        assert_eq!(runtime.state().snapshot().revision, 0);
        assert_eq!(
            runtime.state().snapshot().calibration,
            OverlayPresentationCalibration::default()
        );
    }

    #[test]
    fn debug_watermark_label_reports_revision_active_peer_and_hash() {
        let blocks = vec![
            CaptionBlock::new("peer:11111111-2222-3333-4444-555555555555", "")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::ActivePeer)
                .with_secondary_text("Can you hear me?", true),
            CaptionBlock::new("self:active", "hello")
                .with_channel(CaptionChannel::SelfChannel)
                .with_variant(CaptionBlockVariant::ActiveSelf),
        ];

        let label = debug_watermark_label_for_frame(73, &blocks).unwrap();

        assert!(label.starts_with("DBG r73 "));
        assert!(label.contains("ap=55555555"));
        assert!(label.contains("h="));
        assert!(label.contains("b=peer:55555555,self:active"));
    }

    #[test]
    fn debug_watermark_label_is_absent_without_drawable_blocks() {
        assert_eq!(debug_watermark_label_for_frame(73, &[]), None);
    }

    #[test]
    fn debug_watermark_label_is_absent_when_only_disabled_secondary_has_text() {
        let blocks = vec![CaptionBlock::new("peer:hidden", "")
            .with_channel(CaptionChannel::PeerChannel)
            .with_variant(CaptionBlockVariant::ActivePeer)
            .with_secondary_text("hidden source", false)];

        assert_eq!(debug_watermark_label_for_frame(73, &blocks), None);
    }

    #[test]
    fn debug_overlay_for_frame_is_absent_in_basic_mode() {
        let blocks = vec![CaptionBlock::new("self:active", "hello")
            .with_channel(CaptionChannel::SelfChannel)
            .with_variant(CaptionBlockVariant::ActiveSelf)];

        assert!(debug_overlay_for_frame(false, 73, &blocks).is_none());
    }

    #[test]
    fn debug_overlay_for_frame_is_absent_in_detailed_mode_with_drawable_text() {
        let blocks = vec![CaptionBlock::new("self:active", "hello")
            .with_channel(CaptionChannel::SelfChannel)
            .with_variant(CaptionBlockVariant::ActiveSelf)];

        assert!(debug_overlay_for_frame(true, 73, &blocks).is_none());
    }

    #[tokio::test]
    async fn runtime_logging_mode_change_requests_redraw_for_watermark_clear() {
        let logger = OverlayLogger::open(std::env::temp_dir(), OverlayLoggingMode::Detailed)
            .await
            .unwrap();
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        runtime.clear_redraw_flag();

        assert!(runtime.apply_runtime_logging_mode(&logger, OverlayLoggingMode::Basic));
        assert!(runtime.redraw_requested());

        runtime.clear_redraw_flag();

        assert!(!runtime.apply_runtime_logging_mode(&logger, OverlayLoggingMode::Basic));
        assert!(!runtime.redraw_requested());
    }

    #[tokio::test]
    async fn presentation_diagnostics_retain_basic_records_until_detailed_write_succeeds() {
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(OverlayLoggingMode::Basic, stdout.clone());
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        runtime.presentation_diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            PresentationCauses::default(),
        );

        runtime
            .emit_pending_presentation_diagnostics(&logger)
            .await
            .unwrap();
        assert_eq!(runtime.presentation_diagnostics.pending_json().len(), 1);
        assert!(stdout.contents().is_empty());

        logger.set_mode(OverlayLoggingMode::Detailed);
        runtime
            .emit_pending_presentation_diagnostics(&logger)
            .await
            .unwrap();

        assert!(runtime.presentation_diagnostics.pending_json().is_empty());
        assert!(String::from_utf8(stdout.contents())
            .unwrap()
            .contains("presentation_diagnostics"));
    }

    #[tokio::test]
    async fn presentation_diagnostics_retain_records_after_write_error() {
        let logger = controlled_logger(
            OverlayLoggingMode::Detailed,
            ControlledSink::new(ControlledSinkMode::Error),
        );
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        runtime.presentation_diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            PresentationCauses::default(),
        );

        runtime
            .emit_pending_presentation_diagnostics(&logger)
            .await
            .unwrap();

        assert_eq!(runtime.presentation_diagnostics.pending_json().len(), 1);
    }

    #[tokio::test]
    async fn presentation_diagnostics_retain_records_after_write_timeout() {
        let logger = controlled_logger(
            OverlayLoggingMode::Detailed,
            ControlledSink::new(ControlledSinkMode::Pending),
        );
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        runtime.presentation_diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            PresentationCauses::default(),
        );

        runtime
            .emit_pending_presentation_diagnostics(&logger)
            .await
            .unwrap();

        assert_eq!(runtime.presentation_diagnostics.pending_json().len(), 1);
    }

    #[tokio::test]
    async fn successful_hide_records_reconciled_lifecycle_visibility() {
        let logger = controlled_logger(
            OverlayLoggingMode::Detailed,
            ControlledSink::new(ControlledSinkMode::Success),
        );
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        runtime.first_texture_submitted = true;
        runtime.overlay_visible = true;
        runtime.presentation_diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            PresentationCauses::default(),
        );
        let correlation = runtime
            .presentation_diagnostics
            .begin_presentation(0, PresentationCauses::default())
            .unwrap();
        runtime.last_presentation_correlation = Some(correlation);
        runtime.last_presentation_backend = Some(PresentationBackend::Test);
        let mut openvr = FakeOpenVr::default();
        openvr.set_overlay_visible(true).unwrap();

        runtime
            .handle_hide_deadline(&mut openvr, &logger)
            .await
            .unwrap();

        let visibility = runtime.presentation_diagnostics.records().back().unwrap();
        assert_eq!(visibility.stage, PresentationStage::VisibilityObserved);
        assert_eq!(visibility.outcome, PresentationOutcome::Success);
        assert_eq!(visibility.desired_visible, Some(false));
        assert_eq!(visibility.observed_runtime_visible, Some(false));
    }

    #[test]
    fn prepare_openvr_runtime_stops_before_overlay_factory_when_preflight_fails() {
        let overlay_factory_calls = Cell::new(0);

        let result = prepare_openvr_runtime(
            "overlay-test",
            || Err(OpenVrStartupPreflightError::SteamVrNotRunning),
            |_| {
                overlay_factory_calls.set(overlay_factory_calls.get() + 1);
                Ok(())
            },
        );

        assert_eq!(result, Err(StartupError::SteamVrNotRunning));
        assert_eq!(overlay_factory_calls.get(), 0);
    }

    #[test]
    fn prepare_openvr_runtime_initializes_overlay_after_successful_preflight() {
        let overlay_factory_calls = Cell::new(0);

        let result = prepare_openvr_runtime(
            "overlay-test",
            || Ok(()),
            |_| {
                overlay_factory_calls.set(overlay_factory_calls.get() + 1);
                Ok::<_, OpenVrError>("overlay-ready")
            },
        );

        assert_eq!(result, Ok("overlay-ready"));
        assert_eq!(overlay_factory_calls.get(), 1);
    }

    #[test]
    fn snapshot_summary_omits_block_details_for_log_noise_reduction() {
        let summary = format_snapshot_received_log(&OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 7,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                OverlayPresentationBlock {
                    id: "self:1".into(),
                    occupant_key: "self:1".into(),
                    appearance_seq: 1,
                    channel: "self".into(),
                    block_variant: OverlayPresentationBlockVariant::Finalized,
                    primary_text: "hello".into(),
                    secondary_text: String::new(),
                    secondary_enabled: true,
                    primary_language: None,
                    secondary_language: None,
                    update_id: Some("upd-self-1".into()),
                    origin_wall_clock_ms: Some(1712345678901),
                    session_scope: Some("session:self".into()),
                },
                OverlayPresentationBlock {
                    id: "self:active".into(),
                    occupant_key: "self:merge-1".into(),
                    appearance_seq: 2,
                    channel: "self".into(),
                    block_variant: OverlayPresentationBlockVariant::ActiveSelf,
                    primary_text: "speaking".into(),
                    secondary_text: "hidden".into(),
                    secondary_enabled: false,
                    primary_language: None,
                    secondary_language: None,
                    update_id: None,
                    origin_wall_clock_ms: None,
                    session_scope: None,
                },
            ],
        });

        assert!(summary.contains("bridge_snapshot_received revision=7 block_count=2"));
        assert!(!summary.contains("upd-self-1"));
        assert!(!summary.contains("blocks="));
        assert!(!summary.contains("id=self:1 variant=finalized sec=enabled/0"));
        assert!(!summary.contains("session_scope=session:self"));
        assert!(!summary.contains("origin_wall_clock_ms=1712345678901"));
    }

    #[test]
    fn state_snapshot_summary_excludes_raw_slot_identifiers() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 7,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![OverlayPresentationBlock {
                id: "self:1".into(),
                occupant_key: "self:1".into(),
                appearance_seq: 1,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: "hello".into(),
                secondary_text: "translated".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: Some("upd-self-1".into()),
                origin_wall_clock_ms: Some(1712345678901),
                session_scope: Some("session:self".into()),
            }],
        });
        let outcome = SnapshotApplyOutcome::Applied {
            incoming_revision: 7,
            current_revision: 7,
            visual_changed: true,
            redraw_requested: true,
        };

        let summary = format_state_snapshot_log(&outcome, runtime.state(), true);

        assert!(summary.contains("state_snapshot_applied incoming_revision=7 current_revision=7"));
        assert!(summary.contains("block_count=1 occupied_slot_count=1"));
        assert!(!summary.contains("self:1"));
        assert!(!summary.contains("upd-self-1"));
        assert!(!summary.contains("session:self"));
    }

    #[test]
    fn snapshot_slot_correlation_summary_reports_safe_bounded_counts() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 7,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                OverlayPresentationBlock {
                    id: "peer:2".into(),
                    occupant_key: "peer:2".into(),
                    appearance_seq: 2,
                    channel: "peer".into(),
                    block_variant: OverlayPresentationBlockVariant::Finalized,
                    primary_text: "peer line".into(),
                    secondary_text: String::new(),
                    secondary_enabled: true,
                    primary_language: None,
                    secondary_language: None,
                    update_id: Some("upd-peer-2".into()),
                    origin_wall_clock_ms: Some(1712345678902),
                    session_scope: Some("session:peer".into()),
                },
                OverlayPresentationBlock {
                    id: "self:1".into(),
                    occupant_key: "self:1".into(),
                    appearance_seq: 1,
                    channel: "self".into(),
                    block_variant: OverlayPresentationBlockVariant::Finalized,
                    primary_text: "self line".into(),
                    secondary_text: String::new(),
                    secondary_enabled: true,
                    primary_language: None,
                    secondary_language: None,
                    update_id: Some("upd-self-1".into()),
                    origin_wall_clock_ms: Some(1712345678901),
                    session_scope: Some("session:self".into()),
                },
            ],
        });

        let rows = collect_diagnostic_rows(runtime.state());
        let summary = format_snapshot_slot_correlation_log(runtime.state(), &rows);

        assert!(summary.contains("snapshot_slot_correlation revision=7"));
        assert!(summary.contains("row_count=2 occupied_slot_count=2"));
        assert!(!summary.contains("upd-peer-2"));
        assert!(!summary.contains("session:peer"));
    }

    #[test]
    fn apply_snapshot_marks_visible_updates_for_existing_slot_order() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 1,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![OverlayPresentationBlock {
                id: "self:1".into(),
                occupant_key: "self:1".into(),
                appearance_seq: 1,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: "hello".into(),
                secondary_text: String::new(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: Some("upd-self-1".into()),
                origin_wall_clock_ms: Some(1712345678901),
                session_scope: Some("session:self".into()),
            }],
        });
        let rows = collect_diagnostic_rows(runtime.state());
        let slot_order = rows[0].slot_order;
        runtime
            .last_submitted_visible_rows
            .insert(slot_order, diagnostic_row_signature(&rows[0]));

        let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 2,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![OverlayPresentationBlock {
                id: "self:1".into(),
                occupant_key: "self:1".into(),
                appearance_seq: 1,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: "hello again".into(),
                secondary_text: "translated".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: Some("upd-self-2".into()),
                origin_wall_clock_ms: Some(1712345678955),
                session_scope: Some("session:self".into()),
            }],
        });

        assert!(matches!(outcome, SnapshotApplyOutcome::Applied { .. }));
        assert_eq!(runtime.pending_visible_update_rows.len(), 1);
        assert_eq!(
            runtime.pending_visible_update_rows[0].slot_order,
            slot_order
        );
        assert!(runtime
            .pending_visible_update_render_slot_orders
            .contains(&slot_order));
    }

    #[test]
    fn overlay_visible_update_rendered_summary_reports_bounds_and_slot_mapping() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 8,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![OverlayPresentationBlock {
                id: "self:1".into(),
                occupant_key: "self:1".into(),
                appearance_seq: 1,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: "hello".into(),
                secondary_text: "translated".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: Some("upd-self-2".into()),
                origin_wall_clock_ms: Some(1712345678955),
                session_scope: Some("session:self".into()),
            }],
        });
        let layout = CaptionLayoutPolicy::default().layout_blocks_for_presentation(
            runtime.caption_blocks(),
            640,
            600,
            &CaptionPresentation::default(),
        );
        let rendered = collect_rendered_diagnostic_rows(runtime.state(), &layout);
        let summary = format_overlay_visible_update_rendered_log(8, &rendered[0]);

        assert!(summary.contains("overlay_visible_update_rendered revision=8"));
        assert!(summary.contains("slot_index=0"));
        assert!(summary.contains("primary_len=5 secondary_len=10"));
        assert!(!summary.contains("upd-self-2"));
        assert!(!summary.contains("session:self"));
        assert!(summary.contains("bounds="));
        assert!(summary.contains("visual_bounds="));
    }

    #[test]
    fn two_row_window_closed_summary_reports_exact_dwell_and_threshold() {
        let _rows = vec![
            RenderedDiagnosticRow {
                row: DiagnosticRow {
                    id: "self:1".into(),
                    occupant_key: "self:1".into(),
                    channel: "self".into(),
                    block_variant: OverlayPresentationBlockVariant::Finalized,
                    update_id: Some("upd-self-1".into()),
                    origin_wall_clock_ms: Some(1712345678901),
                    session_scope: Some("session:self".into()),
                    presenter_order: 0,
                    slot_order: 0,
                    slot_index: 0,
                    slot_anchor_top_px: 40.0,
                    primary_text: "one".into(),
                    secondary_text: String::new(),
                    secondary_enabled: true,
                },
                bounds: crate::renderer::BlockBounds::new(0.0, 40.0, 320.0, 220.0),
                visual_bounds: crate::renderer::VisualBounds::new(0.0, 40.0, 320.0, 220.0),
                secondary_present: false,
                truncated_secondary: false,
            },
            RenderedDiagnosticRow {
                row: DiagnosticRow {
                    id: "peer:2".into(),
                    occupant_key: "peer:2".into(),
                    channel: "peer".into(),
                    block_variant: OverlayPresentationBlockVariant::Finalized,
                    update_id: Some("upd-peer-2".into()),
                    origin_wall_clock_ms: Some(1712345678902),
                    session_scope: Some("session:peer".into()),
                    presenter_order: 1,
                    slot_order: 1,
                    slot_index: 1,
                    slot_anchor_top_px: 256.0,
                    primary_text: "two".into(),
                    secondary_text: String::new(),
                    secondary_enabled: true,
                },
                bounds: crate::renderer::BlockBounds::new(0.0, 256.0, 320.0, 436.0),
                visual_bounds: crate::renderer::VisualBounds::new(0.0, 256.0, 320.0, 436.0),
                secondary_present: false,
                truncated_secondary: false,
            },
        ];
        let started_at = Instant::now();
        let window = TwoRowWindowState {
            started_at,
            slot_signature: vec![0, 1],
        };
        let summary =
            format_two_row_window_closed_log(9, &window, started_at + Duration::from_millis(420));

        assert!(summary.contains("two_row_window_closed revision=9"));
        assert!(summary.contains("dwell_ms=420"));
        assert!(summary.contains("threshold_ms=500"));
        assert!(summary.contains("too_brief_to_be_perceptibly_stable=true"));
        assert!(summary.contains("row_count=2"));
        assert!(!summary.contains("upd-self-1"));
        assert!(!summary.contains("upd-peer-2"));
    }

    #[test]
    fn caption_block_summary_includes_hidden_secondary_and_active_variant() {
        let summary = format_caption_blocks_built_log(&[
            CaptionBlock::new("self:1", "hello").with_secondary_text("", true),
            CaptionBlock::new("self:active", "speaking")
                .with_variant(CaptionBlockVariant::ActiveSelf)
                .with_secondary_text("hidden", false),
        ]);

        assert!(summary.contains("caption_blocks_built block_count=2"));
        assert!(summary.contains("id=self:1 variant=finalized sec=enabled/0"));
        assert!(summary.contains("id=self:active variant=active_self sec=disabled/6"));
    }

    #[test]
    fn frame_rendered_summary_reports_secondary_presence_and_truncation() {
        let layout = CaptionLayoutPolicy::default().layout_blocks_for_presentation(
            vec![CaptionBlock::new("self:1", "primary").with_secondary_text(
                "this secondary line should be truncated in a narrow layout",
                true,
            )],
            320,
            600,
            &CaptionPresentation::default(),
        );

        let rendered_rows = vec![RenderedDiagnosticRow {
            row: DiagnosticRow {
                id: "self:1".into(),
                occupant_key: "self:1".into(),
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                update_id: Some("upd-self-1".into()),
                origin_wall_clock_ms: Some(1712345678901),
                session_scope: Some("session:self".into()),
                presenter_order: 0,
                slot_order: 0,
                slot_index: 0,
                slot_anchor_top_px: 40.0,
                primary_text: "primary".into(),
                secondary_text: "this secondary line should be truncated in a narrow layout".into(),
                secondary_enabled: true,
            },
            bounds: crate::renderer::BlockBounds::new(0.0, 40.0, 320.0, 220.0),
            visual_bounds: crate::renderer::VisualBounds::new(0.0, 40.0, 320.0, 220.0),
            secondary_present: true,
            truncated_secondary: true,
        }];

        let summary = format_frame_rendered_log(&layout, false, &rendered_rows, Some(1234));

        assert!(summary.contains("frame_rendered visible_block_count=1 fully_transparent=false"));
        assert!(!summary.contains("upd-self-1"));
        assert!(!summary.contains("self:1"));
        assert!(summary.contains("render_duration_us=1234"));
        assert!(!summary.contains("session:self"));
        assert!(summary.contains("secondary_present_count=1"));
        assert!(summary.contains("truncated_secondary_count=1"));
    }

    #[test]
    fn frame_submitted_summary_reports_revision_and_visibility_fields() {
        let layout = CaptionLayoutPolicy::default().layout_blocks_for_presentation(
            vec![
                CaptionBlock::new("self:1", "primary")
                    .with_channel(CaptionChannel::SelfChannel)
                    .with_secondary_text("translated", true),
                CaptionBlock::new("peer:1", "peer")
                    .with_channel(CaptionChannel::PeerChannel)
                    .with_secondary_text("", true),
            ],
            640,
            600,
            &CaptionPresentation::default(),
        );

        let rendered_rows = vec![RenderedDiagnosticRow {
            row: DiagnosticRow {
                id: "self:1".into(),
                occupant_key: "self:1".into(),
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                update_id: Some("upd-self-1".into()),
                origin_wall_clock_ms: Some(1712345678901),
                session_scope: Some("session:self".into()),
                presenter_order: 0,
                slot_order: 0,
                slot_index: 0,
                slot_anchor_top_px: 40.0,
                primary_text: "primary".into(),
                secondary_text: "translated".into(),
                secondary_enabled: true,
            },
            bounds: crate::renderer::BlockBounds::new(0.0, 40.0, 320.0, 220.0),
            visual_bounds: crate::renderer::VisualBounds::new(0.0, 40.0, 320.0, 220.0),
            secondary_present: true,
            truncated_secondary: false,
        }];

        let summary = format_frame_submitted_log(
            &layout,
            7,
            false,
            false,
            true,
            true,
            None,
            &rendered_rows,
            FrameStageDurations::default(),
        );

        assert!(summary.contains("frame_submitted revision=7"));
        assert!(!summary.contains("upd-self-1"));
        assert!(!summary.contains("block_ids="));
        assert!(!summary.contains("rows="));
        assert!(!summary.contains("session_scope=session:self"));
        assert!(!summary.contains("origin_wall_clock_ms=1712345678901"));
        assert!(summary.contains("visible_block_count=2"));
        assert!(summary.contains("self_block_count=1"));
        assert!(summary.contains("fully_transparent=false"));
        assert!(summary.contains("overlay_visible_before=false"));
        assert!(summary.contains("overlay_visible_after=true"));
        assert!(summary.contains("should_show_after_submit=true"));
        assert!(!summary.contains("submit_duration_us="));

        let summary_with_duration = format_frame_submitted_log(
            &layout,
            7,
            false,
            false,
            true,
            true,
            Some(421),
            &rendered_rows,
            FrameStageDurations {
                receive_to_apply_us: Some(11),
                render_duration_us: Some(1234),
                receive_to_submit_us: Some(3456),
            },
        );
        assert!(summary_with_duration.contains("submit_duration_us=421"));
        assert!(!summary_with_duration.contains("receive_to_apply_us=11"));
        assert!(!summary_with_duration.contains("render_duration_us=1234"));
        assert!(summary_with_duration.contains("receive_to_submit_us=3456"));
    }

    #[test]
    fn frame_timing_summary_reports_revision_gpu_and_submit_duration_fields() {
        let sample = FrameTimingSample {
            frame_index: 4,
            num_frame_presents: 2,
            num_mis_presented: 0,
            num_dropped_frames: 1,
            system_time_seconds: 12.5,
            client_frame_interval_ms: 11.1,
            present_call_cpu_ms: 0.2,
            wait_for_present_cpu_ms: 0.3,
            compositor_render_cpu_ms: 0.4,
            total_render_gpu_ms: 0.56,
            post_submit_gpu_ms: 0.23,
        };

        let summary = format_frame_timing_log(9, &sample, Some(421));

        assert_eq!(
            summary,
            "frame_timing revision=9 dropped_frames=1 post_submit_gpu_ms=0.23 total_render_gpu_ms=0.56 submit_duration_us=421"
        );

        let summary_without_duration = format_frame_timing_log(9, &sample, None);
        assert!(summary_without_duration.contains("submit_duration_us=none"));
    }

    #[test]
    fn cache_stats_summary_reports_cache_sizes_and_hit_miss_counts() {
        let diagnostics = RenderDiagnostics {
            text_format_cache_size: 3,
            layout_cache_size: 4,
            line_cache_size: 5,
            block_cache_size: 6,
            text_format_cache_hits: 7,
            text_format_cache_misses: 8,
            font_warmup_attempts: 9,
            font_warmup_failures: 1,
            directwrite_layout_success_count: 10,
            heuristic_layout_fallback_count: 2,
            layout_cache_hits: 11,
            layout_cache_misses: 12,
            line_cache_hits: 13,
            line_cache_misses: 14,
            block_cache_hits: 15,
            block_cache_misses: 16,
            style_bucket_source_counts: vec![
                StyleBucketSourceCount {
                    bucket: FontLanguageBucket::CjkJa,
                    source: FontSource::SystemFont,
                    count: 2,
                },
                StyleBucketSourceCount {
                    bucket: FontLanguageBucket::CjkZhHant,
                    source: FontSource::BundledNotoCjkMedium,
                    count: 1,
                },
            ],
            ..RenderDiagnostics::default()
        };

        assert_eq!(
            format_cache_stats_log(&diagnostics),
            "cache_stats text_format_size=3 layout_size=4 line_size=5 block_size=6 text_format_hits=7 text_format_misses=8 font_warmup_attempts=9 font_warmup_failures=1 directwrite_layout_successes=10 heuristic_layout_fallbacks=2 layout_hits=11 layout_misses=12 line_hits=13 line_misses=14 block_hits=15 block_misses=16 style_bucket_source_counts=[CjkJa/SystemFont:2,CjkZhHant/BundledNotoCjkMedium:1]"
        );
    }

    #[test]
    fn peer_first_render_visibility_checkpoint_summary_reports_visibility_gate_fields() {
        let summary = format_peer_first_render_visibility_checkpoint_log(
            11,
            &["peer:utterance-3".to_string()],
            true,
            true,
            false,
            true,
            true,
            true,
            1,
            0,
            false,
        );

        assert!(summary.contains("peer_first_render_visibility_checkpoint revision=11"));
        assert!(summary.contains("peer_count=1"));
        assert!(!summary.contains("peer:utterance-3"));
        assert!(summary.contains("overlay_visible_before=true"));
        assert!(summary.contains("should_show_after_submit=false"));
        assert!(summary.contains("hide_deadline_active=true"));
        assert!(summary.contains("visible_block_count=1"));
        assert!(summary.contains("self_block_count=0"));
        assert!(summary.contains("fully_transparent=false"));
    }

    #[test]
    fn peer_first_render_visibility_desync_warning_summary_reports_suspect_state() {
        let summary = format_peer_first_render_visibility_desync_suspected_log(
            12,
            &["peer:utterance-4".to_string()],
            true,
            false,
            true,
            true,
            true,
            0,
        );

        assert!(summary.contains("peer_first_render_visibility_desync_suspected revision=12"));
        assert!(summary.contains("peer_count=1"));
        assert!(!summary.contains("peer:utterance-4"));
        assert!(summary.contains("overlay_visible_before=true"));
        assert!(summary.contains("should_show_after_submit=false"));
        assert!(summary.contains("hide_deadline_active=true"));
        assert!(summary.contains("first_texture_submitted=true"));
        assert!(summary.contains("redraw_requested=true"));
        assert!(summary.contains("last_submitted_visible_row_count=0"));
    }

    #[test]
    fn runtime_apply_snapshot_reports_ignored_revisions_without_redraw() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 3,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("self:1", "self", "hello", "", true)],
        });
        runtime.clear_redraw_flag();

        let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 2,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("peer:2", "peer", "ignored", "", true)],
        });

        assert_eq!(
            outcome,
            SnapshotApplyOutcome::Ignored {
                incoming_revision: 2,
                current_revision: 3,
            }
        );
        assert!(!runtime.redraw_requested());
    }
}

async fn emit_startup_failure(logger: &OverlayLogger, error: &StartupError) {
    let _ = logger
        .error(format!("startup_failure reason={}", error.failure_reason()))
        .await;
    let _ = logger
        .emit_stderr_event(&json!({
            "type": "startup_error",
            "failure_reason": error.failure_reason(),
        }))
        .await;
}

async fn emit_startup_failure_to_stderr(error: &StartupError) {
    let mut stderr = io::stderr();
    let line = format!(
        "EVENT {}\n",
        json!({
            "type": "startup_error",
            "failure_reason": error.failure_reason(),
        })
    );
    let _ = stderr.write_all(line.as_bytes()).await;
    let _ = stderr.flush().await;
}
