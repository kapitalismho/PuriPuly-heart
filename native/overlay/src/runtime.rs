use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use serde_json::json;
use thiserror::Error;
use tokio::io::{self, AsyncWriteExt};
use tokio::time::{sleep_until, Instant};

use crate::bridge::{BridgeClient, BridgeError, BridgeIncoming, OverlayBridgeEvent};
use crate::logging::{OverlayLogger, OverlayLoggingMode};
use crate::manifest::{
    load_manifest, validate_manifest, OverlayManifest, EXPECTED_CONTRACT_VERSION,
};
#[cfg(test)]
use crate::openvr::OpenVrError;
use crate::openvr::{
    format_openvr_visibility_api_call_log, perform_startup_preflight, FrameTimingSample,
    OpenVrOverlay, OpenVrStartupPreflightError, OverlayFrameSubmitter,
};
use crate::renderer::{
    CaptionBlock, CaptionBlockVariant, CaptionChannel, CaptionDebugOverlay, CaptionLayoutResult,
    CaptionPresentation, CaptionRenderer,
};
#[cfg(test)]
use crate::renderer::{RenderDiagnostics, StyleBucketSourceCount, VisibleCaptionBlock};
use crate::state::{
    OverlayPresentationBlock, OverlayPresentationBlockVariant, OverlayPresentationSnapshot,
    OverlayScene, OverlaySlot, OverlayState,
};

const EMPTY_OVERLAY_HIDE_DELAY: Duration = Duration::from_millis(500);
const TWO_ROW_WINDOW_STABILITY_THRESHOLD_MS: u64 = 500;

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
}

impl RuntimeFailure {
    pub fn failure_reason(&self) -> &'static str {
        match self {
            Self::RuntimeDisconnected => "runtime_disconnected",
            Self::Stopped => "stopped",
            Self::Bridge(_) | Self::Render(_) | Self::OpenVr(_) => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayRuntime {
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
    rows_summary: String,
    update_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct FrameStageDurations {
    receive_to_apply_us: Option<u128>,
    render_duration_us: Option<u128>,
    receive_to_submit_us: Option<u128>,
}

impl OverlayRuntime {
    pub fn new(snapshot: OverlayPresentationSnapshot) -> Self {
        let seeded_peer_ids = peer_overlay_first_emit_block_ids_from_snapshot(&snapshot);
        let seen_peer_overlay_ids = seeded_peer_ids.iter().cloned().collect::<HashSet<_>>();
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
        };
        if runtime.state.seed_snapshot(&snapshot) {
            runtime.redraw_requested = true;
        }
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

        let visual_changed = self.state.apply_snapshot(&snapshot);
        if visual_changed {
            self.redraw_requested = true;
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
        }
        changed
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
                rows_summary: format_two_row_window_rows(rendered_rows),
                update_ids: rendered_rows
                    .iter()
                    .filter_map(|row| row.row.update_id.clone())
                    .collect(),
            })
        } else {
            None
        };

        match (&mut self.two_row_window, next_window) {
            (Some(previous), Some(next)) if previous.slot_signature == next.slot_signature => {
                previous.rows_summary = next.rows_summary;
                previous.update_ids = next.update_ids;
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
                self.stopped = true;
                Ok(())
            }
        }
    }

    pub async fn handle_bridge_loss_for_test(&mut self) -> Result<(), RuntimeFailure> {
        self.stopped = true;
        if self.ready {
            Err(RuntimeFailure::RuntimeDisconnected)
        } else {
            Ok(())
        }
    }

    pub async fn emit_ready(
        &mut self,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        bridge
            .send_json(json!({"type": "overlay_ready"}))
            .await
            .map_err(|error| RuntimeFailure::Bridge(error.to_string()))?;
        logger
            .emit_stdout_event(&json!({"type": "overlay_ready"}))
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
        self.submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, None, None)
            .await
    }

    async fn submit_frame_if_needed_with_timing<S: OverlayFrameSubmitter>(
        &mut self,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
        snapshot_received_at: Option<Instant>,
        receive_to_apply_us: Option<u128>,
    ) -> Result<(), RuntimeFailure> {
        if self.first_texture_submitted && !self.redraw_requested {
            return Ok(());
        }

        renderer.set_presentation(CaptionPresentation {
            background_alpha: self.state.calibration().background_alpha,
            text_scale: self.state.calibration().text_scale,
        });
        openvr
            .apply_calibration(self.state.calibration())
            .map_err(|error| RuntimeFailure::OpenVr(error.to_string()))?;
        let detailed_logging = logger.is_detailed();
        // Detailed logging should remain file/stream diagnostics only.  Visual
        // debug prefixes and the native debug watermark obscure captions in
        // live overlays, so never couple runtime logging mode to rendered text.
        let visual_debug_overlays = false;
        let blocks = self.caption_blocks_for_render(visual_debug_overlays);
        self.emit_pending_peer_overlay_first_emit_hooks(logger)
            .await?;
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
        let render_started = if detailed_logging {
            Some(Instant::now())
        } else {
            None
        };
        let frame = if blocks.is_empty() {
            renderer
                .render_empty_frame()
                .map_err(|error| RuntimeFailure::Render(error.to_string()))?
        } else {
            renderer
                .render_blocks_with_debug_overlay(blocks, debug_overlay)
                .map_err(|error| RuntimeFailure::Render(error.to_string()))?
        };
        let render_duration_us = render_started.map(|start| start.elapsed().as_micros());
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
        openvr
            .submit_frame(&frame)
            .map_err(|error| RuntimeFailure::OpenVr(error.to_string()))?;
        let submit_duration_us = submit_started.map(|start| start.elapsed().as_micros());
        if should_show_after_submit {
            openvr
                .set_overlay_visible(true)
                .map_err(|error| RuntimeFailure::OpenVr(error.to_string()))?;
            self.overlay_visible = true;
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
        if detailed_logging {
            self.sample_and_log_frame_timing(
                openvr,
                logger,
                self.state.snapshot().revision,
                submit_duration_us,
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

        Ok(())
    }

    pub async fn run_event_loop<S: OverlayFrameSubmitter>(
        &mut self,
        bridge: &mut BridgeClient,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        loop {
            let hide_deadline = self.hide_deadline;

            tokio::select! {
                _ = sleep_until(hide_deadline.unwrap_or_else(Instant::now)), if hide_deadline.is_some() => {
                    self.handle_hide_deadline(openvr, logger).await?;
                }
                message = bridge.next_message() => {
                    if !self
                        .handle_bridge_message(message, renderer, openvr, bridge, logger)
                        .await?
                    {
                        return Ok(());
                    }
                }
            }
        }
    }

    async fn handle_bridge_message<S: OverlayFrameSubmitter>(
        &mut self,
        message: Result<BridgeIncoming, BridgeError>,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<bool, RuntimeFailure> {
        match message {
            Ok(BridgeIncoming::Heartbeat) => Ok(true),
            Ok(BridgeIncoming::Control(control)) => {
                if self.apply_runtime_logging_mode(logger, control.logging_mode) {
                    self.submit_frame_if_needed(renderer, openvr, bridge, logger)
                        .await?;
                }
                Ok(true)
            }
            Ok(BridgeIncoming::Snapshot(snapshot)) => {
                let snapshot_received_at = Instant::now();
                log_runtime_info(logger, format_snapshot_received_log(&snapshot)).await?;
                let outcome = self.apply_snapshot(snapshot);
                let receive_to_apply_us = snapshot_received_at.elapsed().as_micros();
                let _ = outcome;
                self.emit_pending_visible_update_applied_diagnostics(logger)
                    .await?;
                self.submit_frame_if_needed_with_timing(
                    renderer,
                    openvr,
                    bridge,
                    logger,
                    Some(snapshot_received_at),
                    Some(receive_to_apply_us),
                )
                .await?;
                Ok(true)
            }
            Ok(BridgeIncoming::Event(event)) => {
                self.handle_event(event).await?;
                if self.stopped {
                    return Ok(false);
                }
                self.submit_frame_if_needed(renderer, openvr, bridge, logger)
                    .await?;
                Ok(true)
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
        openvr
            .set_overlay_visible(false)
            .map_err(|error| RuntimeFailure::OpenVr(error.to_string()))?;
        self.overlay_visible = false;
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
        log_runtime_info(
            logger,
            format_frame_timing_log(revision, &t, submit_duration_us),
        )
        .await?;
        Ok(())
    }
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
    let utterance_id = block_id.strip_prefix("peer:").unwrap_or(block_id);
    format!(
        "latency_trace stage={} utterance_id={} block_id={}",
        stage, utterance_id, block_id
    )
}

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

fn update_ids_from_snapshot(snapshot: &OverlayPresentationSnapshot) -> Vec<String> {
    snapshot
        .blocks
        .iter()
        .filter_map(|block| block.update_id.clone())
        .filter(|update_id| !update_id.is_empty())
        .collect()
}

fn format_snapshot_received_log(snapshot: &OverlayPresentationSnapshot) -> String {
    format!(
        "bridge_snapshot_received revision={} block_count={} update_ids=[{}]",
        snapshot.revision,
        snapshot.blocks.len(),
        update_ids_from_snapshot(snapshot).join(","),
    )
}

fn format_scene_slots(scene: &OverlayScene) -> String {
    scene
        .slots()
        .iter()
        .enumerate()
        .map(|(slot_index, slot)| match slot {
            Some(slot) => format!(
                "slot{}=id={} variant={} channel={} update_id={} session_scope={} origin_wall_clock_ms={} sec={}",
                slot_index,
                slot.id,
                overlay_variant_name(slot.block_variant),
                slot.channel,
                format_optional_str(slot.update_id.as_deref()),
                format_optional_str(slot.session_scope.as_deref()),
                format_optional_u64(slot.origin_wall_clock_ms),
                log_runtime_secondary_state(slot.secondary_enabled, &slot.secondary_text)
            ),
            None => format!("slot{}=empty", slot_index),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn update_ids_from_scene(scene: &OverlayScene) -> Vec<String> {
    scene
        .slots()
        .iter()
        .flatten()
        .filter_map(|slot| slot.update_id.clone())
        .filter(|update_id| !update_id.is_empty())
        .collect()
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
            "state_snapshot_applied incoming_revision={} current_revision={} visual_changed={} redraw_requested={} update_ids=[{}] slots=[{}]",
            incoming_revision,
            current_revision,
            visual_changed,
            outcome_redraw_requested,
            update_ids_from_scene(state.scene()).join(","),
            format_scene_slots(state.scene())
        ),
        SnapshotApplyOutcome::Ignored {
            incoming_revision,
            current_revision,
        } => format!(
            "state_snapshot_ignored incoming_revision={} current_revision={} redraw_requested={} update_ids=[{}] slots=[{}]",
            incoming_revision,
            current_revision,
            redraw_requested,
            update_ids_from_scene(state.scene()).join(","),
            format_scene_slots(state.scene())
        ),
    }
}

fn format_optional_str(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
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

fn format_diagnostic_row_summary(row: &DiagnosticRow) -> String {
    format!(
        "id={} channel={} variant={} presenter_order={} slot_order={} slot_index={} slot_anchor_top_px={:.1} update_id={} session_scope={} origin_wall_clock_ms={} primary_len={} secondary_len={}",
        row.id,
        row.channel,
        overlay_variant_name(row.block_variant),
        row.presenter_order,
        row.slot_order,
        row.slot_index,
        row.slot_anchor_top_px,
        format_optional_str(row.update_id.as_deref()),
        format_optional_str(row.session_scope.as_deref()),
        format_optional_u64(row.origin_wall_clock_ms),
        row.primary_text.len(),
        if row.secondary_enabled {
            row.secondary_text.len()
        } else {
            0
        },
    )
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

fn update_ids_from_rows(rows: &[DiagnosticRow]) -> Vec<String> {
    rows.iter()
        .filter_map(|row| row.update_id.clone())
        .filter(|update_id| !update_id.is_empty())
        .collect()
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
        "snapshot_slot_correlation revision={} anchor={} offset_x={:.3} offset_y={:.3} distance={:.3} text_scale={:.3} background_alpha={:.3} update_ids=[{}] rows=[{}]",
        state.snapshot().revision,
        state.calibration().anchor,
        state.calibration().offset_x,
        state.calibration().offset_y,
        state.calibration().distance,
        state.calibration().text_scale,
        state.calibration().background_alpha,
        update_ids_from_rows(rows).join(","),
        rows.iter()
            .map(format_diagnostic_row_summary)
            .collect::<Vec<_>>()
            .join("; ")
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
        "overlay_visible_update_applied revision={} {}",
        revision,
        format_diagnostic_row_summary(row)
    )
}

fn format_overlay_visible_update_rendered_log(
    revision: u64,
    rendered: &RenderedDiagnosticRow,
) -> String {
    format!(
        "overlay_visible_update_rendered revision={} {} bounds={:.1},{:.1},{:.1},{:.1} visual_bounds={:.1},{:.1},{:.1},{:.1} secondary_present={} truncated_secondary={}",
        revision,
        format_diagnostic_row_summary(&rendered.row),
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

fn format_two_row_window_rows(rows: &[RenderedDiagnosticRow]) -> String {
    rows.iter()
        .map(|row| format_diagnostic_row_summary(&row.row))
        .collect::<Vec<_>>()
        .join("; ")
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
        "two_row_window_closed revision={} dwell_ms={} threshold_ms={} too_brief_to_be_perceptibly_stable={} update_ids=[{}] rows=[{}]",
        revision,
        dwell_ms,
        TWO_ROW_WINDOW_STABILITY_THRESHOLD_MS,
        dwell_ms < TWO_ROW_WINDOW_STABILITY_THRESHOLD_MS,
        window.update_ids.join(","),
        window.rows_summary,
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

fn stable_short_hash(value: &str) -> u32 {
    let mut hash = 0x811c9dc5u32;
    for byte in value.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

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
    detailed_logging: bool,
    revision: u64,
    blocks: &[CaptionBlock],
) -> Option<CaptionDebugOverlay> {
    if !detailed_logging {
        return None;
    }
    debug_watermark_label_for_frame(revision, blocks).and_then(CaptionDebugOverlay::new)
}

#[cfg(test)]
fn format_visible_block_summary(block: &VisibleCaptionBlock) -> String {
    format!(
        "id={} variant={} secondary_present={} secondary_reserved={} truncated_secondary={}",
        block.id,
        caption_variant_name(block.block_variant),
        block.secondary_line.is_some(),
        block.secondary_reserved,
        block.truncated_secondary
    )
}

fn update_ids_from_rendered_rows(rows: &[RenderedDiagnosticRow]) -> Vec<String> {
    rows.iter()
        .filter_map(|row| row.row.update_id.clone())
        .filter(|update_id| !update_id.is_empty())
        .collect()
}

#[cfg(test)]
fn block_ids_from_layout(layout: &CaptionLayoutResult) -> Vec<String> {
    layout
        .visible_blocks
        .iter()
        .map(|block| block.id.clone())
        .collect()
}

#[cfg(test)]
fn format_rendered_diagnostic_row_summary(row: &RenderedDiagnosticRow) -> String {
    format!(
        "{} bounds={:.1},{:.1},{:.1},{:.1} visual_bounds={:.1},{:.1},{:.1},{:.1} secondary_present={} truncated_secondary={}",
        format_diagnostic_row_summary(&row.row),
        row.bounds.left_px,
        row.bounds.top_px,
        row.bounds.right_px,
        row.bounds.bottom_px,
        row.visual_bounds.left_px,
        row.visual_bounds.top_px,
        row.visual_bounds.right_px,
        row.visual_bounds.bottom_px,
        row.secondary_present,
        row.truncated_secondary,
    )
}

#[cfg(test)]
fn format_rendered_diagnostic_rows(rows: &[RenderedDiagnosticRow]) -> String {
    rows.iter()
        .map(format_rendered_diagnostic_row_summary)
        .collect::<Vec<_>>()
        .join("; ")
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
        "frame_rendered visible_block_count={} fully_transparent={} update_ids=[{}] block_ids=[{}] rows=[{}] blocks=[{}]",
        layout.visible_blocks.len(),
        fully_transparent,
        update_ids_from_rendered_rows(rendered_rows).join(","),
        block_ids_from_layout(layout).join(","),
        format_rendered_diagnostic_rows(rendered_rows),
        layout
            .visible_blocks
            .iter()
            .map(format_visible_block_summary)
            .collect::<Vec<_>>()
            .join("; ")
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
    rendered_rows: &[RenderedDiagnosticRow],
    stage_durations: FrameStageDurations,
) -> String {
    let mut line = format!(
        "frame_submitted revision={} visible_block_count={} self_block_count={} fully_transparent={} overlay_visible_before={} overlay_visible_after={} should_show_after_submit={} update_ids=[{}]",
        revision,
        layout.visible_blocks.len(),
        visible_self_block_count(layout),
        fully_transparent,
        overlay_visible_before,
        overlay_visible_after,
        should_show_after_submit,
        update_ids_from_rendered_rows(rendered_rows).join(","),
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
        "peer_first_render_visibility_checkpoint revision={} peer_ids=[{}] has_drawable_text={} overlay_visible_before={} should_show_after_submit={} hide_deadline_active={} first_texture_submitted={} redraw_requested={} visible_block_count={} self_block_count={} fully_transparent={}",
        revision,
        peer_ids.join(","),
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
        "peer_first_render_visibility_desync_suspected revision={} peer_ids=[{}] overlay_visible_before={} should_show_after_submit={} hide_deadline_active={} first_texture_submitted={} redraw_requested={} last_submitted_visible_row_count={}",
        revision,
        peer_ids.join(","),
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

    let (renderer, mut openvr) = match initialize_runtime_resources(&manifest, &logger).await {
        Ok(resources) => resources,
        Err(error) => {
            let _ = bridge.close().await;
            emit_startup_failure(&logger, &error).await;
            return error.exit_code();
        }
    };

    let mut runtime = OverlayRuntime::new(snapshot);
    let initial_outcome = SnapshotApplyOutcome::Applied {
        incoming_revision: runtime.state().snapshot().revision,
        current_revision: runtime.state().snapshot().revision,
        visual_changed: runtime.redraw_requested(),
        redraw_requested: runtime.redraw_requested(),
    };
    let _ = logger
        .info(format_state_snapshot_log(
            &initial_outcome,
            runtime.state(),
            runtime.redraw_requested(),
        ))
        .await;
    let _ = runtime
        .emit_snapshot_slot_correlation_if_changed(&logger)
        .await;
    if let Err(error) = runtime
        .submit_frame_if_needed(&renderer, &mut openvr, &mut bridge, &logger)
        .await
    {
        let startup_error = startup_error_from_runtime_failure(error);
        let _ = bridge.close().await;
        emit_startup_failure(&logger, &startup_error).await;
        return startup_error.exit_code();
    }

    let runtime_result = runtime
        .run_event_loop(&mut bridge, &renderer, &mut openvr, &logger)
        .await;
    let _ = bridge.close().await;

    match runtime_result {
        Ok(()) => 0,
        Err(RuntimeFailure::RuntimeDisconnected) => 1,
        Err(error) => {
            let _ = logger.error(&error.to_string()).await;
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
            eprintln!("[overlay][ERROR] {error}");
            emit_startup_failure_to_stderr(&error).await;
            return error.exit_code();
        }
    };

    run_with_manifest(manifest).await
}

fn startup_error_from_runtime_failure(error: RuntimeFailure) -> StartupError {
    match error {
        RuntimeFailure::Render(message) => StartupError::RendererInit(message),
        RuntimeFailure::OpenVr(message) => StartupError::OpenVrInit(message),
        RuntimeFailure::Bridge(message) => StartupError::Other(message),
        RuntimeFailure::RuntimeDisconnected => {
            StartupError::Other("runtime disconnected before ready".into())
        }
        RuntimeFailure::Stopped => StartupError::Other("runtime stopped before ready".into()),
    }
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
    overlay_factory(overlay_instance_id)
        .map_err(|error| StartupError::OpenVrInit(error.to_string()))
}

async fn initialize_runtime_resources(
    manifest: &OverlayManifest,
    logger: &OverlayLogger,
) -> Result<(CaptionRenderer, OpenVrOverlay), StartupError> {
    let openvr = OpenVrOverlay::new(&manifest.overlay_instance_id)
        .map_err(|error| StartupError::OpenVrInit(error.to_string()))?;
    logger
        .info("openvr_ready")
        .await
        .map_err(|error| StartupError::Other(error.to_string()))?;
    let renderer =
        create_runtime_renderer().map_err(|error| StartupError::RendererInit(error.to_string()))?;
    logger
        .info("renderer_resources_ready")
        .await
        .map_err(|error| StartupError::Other(error.to_string()))?;
    Ok((renderer, openvr))
}

fn create_runtime_renderer() -> Result<CaptionRenderer, crate::renderer::CaptionRenderError> {
    #[cfg(windows)]
    {
        CaptionRenderer::new()
    }

    #[cfg(not(windows))]
    {
        CaptionRenderer::new_for_test()
    }
}

impl OverlayRuntime {
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
        format_two_row_window_closed_log, peer_overlay_first_emit_block_ids_from_snapshot,
        peer_overlay_first_render_block_ids_from_caption_blocks, prepare_openvr_runtime,
        DiagnosticRow, FrameStageDurations, OverlayRuntime, RenderedDiagnosticRow,
        SnapshotApplyOutcome, StartupError, TwoRowWindowState,
    };
    use crate::logging::{OverlayLogger, OverlayLoggingMode};
    use crate::openvr::{FrameTimingSample, OpenVrError, OpenVrStartupPreflightError};
    use crate::renderer::{
        CaptionBlock, CaptionBlockVariant, CaptionChannel, CaptionLayoutPolicy,
        CaptionPresentation, FontLanguageBucket, FontSource, RenderDiagnostics,
        StyleBucketSourceCount,
    };
    use crate::state::{
        OverlayPresentationBlock, OverlayPresentationBlockVariant, OverlayPresentationCalibration,
        OverlayPresentationSnapshot,
    };
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::time::Duration;
    use tokio::time::Instant;

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

    #[test]
    fn caption_blocks_follow_snapshot_order_exactly() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
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
            revision: 1,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("self:1", "self", "self one", "", true)],
        });

        runtime.apply_snapshot(OverlayPresentationSnapshot {
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
    fn debug_overlay_for_frame_is_present_in_detailed_mode_with_drawable_text() {
        let blocks = vec![CaptionBlock::new("self:active", "hello")
            .with_channel(CaptionChannel::SelfChannel)
            .with_variant(CaptionBlockVariant::ActiveSelf)];

        let overlay = debug_overlay_for_frame(true, 73, &blocks).unwrap();

        assert!(overlay.label().starts_with("DBG r73 "));
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
    fn snapshot_summary_includes_variants_and_secondary_lengths() {
        let summary = format_snapshot_received_log(&OverlayPresentationSnapshot {
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
        assert!(summary.contains("update_ids=[upd-self-1]"));
        assert!(!summary.contains("blocks="));
        assert!(!summary.contains("id=self:1 variant=finalized sec=enabled/0"));
        assert!(!summary.contains("session_scope=session:self"));
    }

    #[test]
    fn state_snapshot_summary_includes_slot_update_ids() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
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
        assert!(summary.contains("update_ids=[upd-self-1]"));
        assert!(summary.contains("slot0=id=self:1"));
        assert!(summary.contains("update_id=upd-self-1"));
        assert!(summary.contains("session_scope=session:self"));
        assert!(summary.contains("origin_wall_clock_ms=1712345678901"));
    }

    #[test]
    fn snapshot_slot_correlation_summary_reports_update_ids_and_slot_mapping() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
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
        assert!(summary.contains("update_ids=[upd-peer-2,upd-self-1]"));
        assert!(summary.contains("session_scope=session:peer"));
        assert!(summary.contains("presenter_order=0"));
        assert!(summary.contains("slot_order=1"));
        assert!(summary.contains("slot_index=1"));
        assert!(summary.contains("slot_anchor_top_px="));
    }

    #[test]
    fn apply_snapshot_marks_visible_updates_for_existing_slot_order() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
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
        assert!(summary.contains("update_id=upd-self-2"));
        assert!(summary.contains("session_scope=session:self"));
        assert!(summary.contains("slot_order=0"));
        assert!(summary.contains("slot_index=0"));
        assert!(summary.contains("bounds="));
        assert!(summary.contains("visual_bounds="));
    }

    #[test]
    fn two_row_window_closed_summary_reports_exact_dwell_and_threshold() {
        let rows = vec![
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
            rows_summary: super::format_two_row_window_rows(&rows),
            update_ids: vec!["upd-self-1".into(), "upd-peer-2".into()],
        };
        let summary =
            format_two_row_window_closed_log(9, &window, started_at + Duration::from_millis(420));

        assert!(summary.contains("two_row_window_closed revision=9"));
        assert!(summary.contains("dwell_ms=420"));
        assert!(summary.contains("threshold_ms=500"));
        assert!(summary.contains("too_brief_to_be_perceptibly_stable=true"));
        assert!(summary.contains("update_ids=[upd-self-1,upd-peer-2]"));
        assert!(summary.contains("slot_order=0"));
        assert!(summary.contains("slot_order=1"));
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
        assert!(summary.contains("update_ids=[upd-self-1]"));
        assert!(summary.contains("block_ids=[self:1]"));
        assert!(summary.contains("render_duration_us=1234"));
        assert!(summary.contains("session_scope=session:self"));
        assert!(summary.contains("id=self:1 variant=finalized secondary_present=true"));
        assert!(summary.contains("truncated_secondary=true"));
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
        assert!(summary.contains("update_ids=[upd-self-1]"));
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
        assert!(summary.contains("peer_ids=[peer:utterance-3]"));
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
        assert!(summary.contains("peer_ids=[peer:utterance-4]"));
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
            revision: 3,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("self:1", "self", "hello", "", true)],
        });
        runtime.clear_redraw_flag();

        let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
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
    let _ = logger.error(&error.to_string()).await;
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
