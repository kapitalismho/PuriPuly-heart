use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures_util::FutureExt;
use serde_json::json;
use thiserror::Error;
use tokio::io::{self, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio::time::{sleep_until, Instant};

use crate::bridge::{BridgeClient, BridgeError, BridgeIncoming, OverlayBridgeEvent};
use crate::frame_cycle::{FrameCycleResult as FrameCycleOutcome, FrameProgress};
use crate::logging::OverlayLogger;
use crate::manifest::{
    load_manifest, resolve_handoff_experiment_from_env, resolve_quiet_tail_profile_from_env,
    validate_manifest, HandoffExperiment, OverlayManifest, QuietTailProfile,
    EXPECTED_CONTRACT_VERSION,
};
#[cfg(test)]
use crate::openvr::OpenVrError;
use crate::openvr::{
    perform_startup_preflight, OpenVrEventClass, OpenVrOverlay, OpenVrRuntimeEvent,
    OpenVrStartupPreflightError, OverlayFrameSubmitter, SpatialReanchorOutcome,
};
use crate::presentation::{
    HandoffMode, PresentationBackend, PresentationCause, PresentationCauseChannel,
    PresentationCauseKind, PresentationCauses, PresentationCorrelation, PresentationDiagnostics,
    ReadinessCancellation, ReadinessOutcome,
};
use crate::renderer::{
    CaptionBlock, CaptionBlockVariant, CaptionChannel, CaptionLayoutResult, CaptionPresentation,
    CaptionRenderer, FontSource, RenderDiagnostics, RenderedFrame,
};
use crate::retry_episode::{
    FreshRetryChannel, FreshRetryPolicy as NativeFreshRetryPolicy,
    FreshSchedule as NativeFreshSchedule, RetryEpisodes, RetryIntent,
    RETRY_AUDIT_CAPACITY as NATIVE_FRESH_AUDIT_CAPACITY,
};
use crate::runtime_diagnostics::{project_owner_status, OwnerStatusInput};
use crate::spatial_policy::{SpatialDiagnostic, SpatialReanchorPolicy};
use crate::state::{
    NativeQuietTailEpisode, NativeQuietTailPhase, OverlayPresentationBlockVariant,
    OverlayPresentationSnapshot, OverlaySlot, OverlayState,
};

const EMPTY_OVERLAY_HIDE_DELAY: Duration = Duration::from_millis(500);
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
    #[error("GPU readiness timed out")]
    ReadinessTimedOut,
    #[error("GPU readiness was cancelled")]
    ReadinessCancelled,
    #[error("GPU readiness query failed")]
    ReadinessFailed,
    #[error("runtime bridge failed: {0}")]
    RuntimeBridge(String),
    #[error("runtime render failed: {0}")]
    RuntimeRender(String),
    #[error("runtime OpenVR failed: {0}")]
    RuntimeOpenVr(String),
    #[error("runtime disconnected before ready")]
    RuntimeDisconnected,
    #[error("runtime stopped before ready")]
    RuntimeStopped,
    #[error("GPU readiness made no progress")]
    ReadinessStalled,
    #[error("startup failed: {0}")]
    Other(String),
}

impl StartupError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::ContractMismatch(_) => 10,
            Self::BridgeAuth(_) => 12,
            Self::SteamVrNotInstalled
            | Self::SteamVrNotRunning
            | Self::HmdNotFound
            | Self::OpenVrInit(_) => 20,
            Self::RendererInit(_)
            | Self::ReadinessTimedOut
            | Self::ReadinessCancelled
            | Self::ReadinessFailed
            | Self::ReadinessStalled
            | Self::RuntimeRender(_)
            | Self::RuntimeOpenVr(_) => 21,
            Self::Manifest(_)
            | Self::Other(_)
            | Self::RuntimeBridge(_)
            | Self::RuntimeDisconnected
            | Self::RuntimeStopped => 1,
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
            Self::ReadinessTimedOut => "gpu_readiness_late",
            Self::ReadinessCancelled => "gpu_readiness_cancelled",
            Self::ReadinessFailed => "gpu_query_failed",
            Self::ReadinessStalled => "gpu_stalled",
            Self::RuntimeBridge(_) => "bridge_failed",
            Self::RuntimeRender(_) => "render_failed",
            Self::RuntimeOpenVr(_) => "openvr_failed",
            Self::RuntimeDisconnected => "runtime_disconnected",
            Self::RuntimeStopped => "stopped",
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
    #[error("GPU readiness made no progress")]
    ReadinessStalled,
}

impl RuntimeFailure {
    pub fn failure_reason(&self) -> &'static str {
        match self {
            Self::RuntimeDisconnected => "runtime_disconnected",
            Self::Stopped => "stopped",
            Self::ReadinessTimedOut => "gpu_readiness_late",
            Self::ReadinessCancelled => "gpu_readiness_cancelled",
            Self::ReadinessFailed => "gpu_query_failed",
            Self::ReadinessStalled => "gpu_stalled",
            Self::Bridge(_) => "bridge_failed",
            Self::Render(_) => "render_failed",
            Self::OpenVr(_) => "openvr_failed",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ReadinessStatusContext {
    due_started_at: Option<Instant>,
    recovering: bool,
}

impl ReadinessStatusContext {
    fn due_elapsed_ms(self) -> u64 {
        self.due_started_at
            .map(|started| {
                Instant::now()
                    .saturating_duration_since(started)
                    .as_millis() as u64
            })
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RendererDegradationSummary {
    style_resolution_fallback_lines: u32,
    heuristic_layout_fallbacks: u32,
}

impl RendererDegradationSummary {
    fn from_diagnostics(diagnostics: &RenderDiagnostics) -> Option<Self> {
        let style_resolution_fallback_lines = diagnostics
            .style_bucket_source_counts
            .iter()
            .filter(|count| count.source == FontSource::SystemFallbackSentinel)
            .map(|count| count.count)
            .sum();
        let summary = Self {
            style_resolution_fallback_lines,
            heuristic_layout_fallbacks: diagnostics.heuristic_layout_fallback_count,
        };
        (summary.style_resolution_fallback_lines > 0 || summary.heuristic_layout_fallbacks > 0)
            .then_some(summary)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationRuntime {
    ready: bool,
    first_texture_submitted: bool,
    overlay_visible: bool,
    runtime_visibility_observed: Option<bool>,
    visibility_request_pending: Option<bool>,
    last_submitted_had_self: bool,
    stopped: bool,
    state: OverlayState,
    redraw_requested: bool,
    hide_deadline: Option<Instant>,
    presentation_diagnostics: PresentationDiagnostics,
    pending_logical_revision_acceptance: bool,
    last_logical_caption_identity: LogicalCaptionIdentity,
    last_presentation_correlation: Option<PresentationCorrelation>,
    last_presentation_backend: Option<PresentationBackend>,
    pending_presentation_causes: PresentationCauses,
    spatial_lock: SpatialReanchorPolicy,
    pending_spatial_diagnostics: Vec<SpatialDiagnostic>,
    handoff_experiment: HandoffExperiment,
    retained_frame: Option<RetainedFrame>,
    spatial_pose_unavailable: bool,
    readiness_status_context: ReadinessStatusContext,
    last_renderer_degradation: Option<RendererDegradationSummary>,
}

#[derive(Debug, Clone)]
struct RetainedFrame {
    frame: Arc<RenderedFrame>,
    scene_generation: u64,
    render_generation: u64,
    blocks: Vec<CaptionBlock>,
    presentation: CaptionPresentation,
    backend: PresentationBackend,
    openvr_adapter_identity: crate::presentation::AdapterIdentity,
    renderer_adapter_identity: crate::presentation::AdapterIdentity,
    content_identity: u64,
}

impl PartialEq for RetainedFrame {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.frame, &other.frame)
            && self.scene_generation == other.scene_generation
            && self.render_generation == other.render_generation
            && self.blocks == other.blocks
            && self.presentation == other.presentation
            && self.backend == other.backend
            && self.openvr_adapter_identity == other.openvr_adapter_identity
            && self.renderer_adapter_identity == other.renderer_adapter_identity
            && self.content_identity == other.content_identity
    }
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

fn retain_semantically_current_blocks(snapshot: &mut OverlayPresentationSnapshot) {
    let frontiers = snapshot
        .semantic_retirement_frontiers
        .iter()
        .map(|frontier| {
            (
                (frontier.scope.as_str(), frontier.generation),
                frontier.order,
            )
        })
        .collect::<HashMap<_, _>>();
    snapshot.blocks.retain(|block| {
        let (Some(scope), Some(generation), Some(order)) = (
            block.publication_scope.as_deref(),
            block.publication_generation,
            block.publication_order,
        ) else {
            return true;
        };
        frontiers
            .get(&(scope, generation))
            .is_none_or(|frontier| order > *frontier)
    });
}

pub type OverlayRuntime = PresentationRuntime;

impl PresentationRuntime {
    fn configure_retry_profile(&mut self, retry_profile: &'static str) {
        self.presentation_diagnostics
            .configure_retry_profile(retry_profile);
    }

    fn configure_handoff_experiment(&mut self, experiment: HandoffExperiment) {
        self.handoff_experiment = experiment;
    }
    pub fn new(mut snapshot: OverlayPresentationSnapshot) -> Self {
        retain_semantically_current_blocks(&mut snapshot);
        let (spatial_lock, spatial_result) =
            SpatialReanchorPolicy::from_initial_snapshot(&snapshot);
        let mut runtime = Self {
            ready: false,
            first_texture_submitted: false,
            overlay_visible: false,
            runtime_visibility_observed: None,
            visibility_request_pending: None,
            last_submitted_had_self: false,
            stopped: false,
            state: OverlayState::default(),
            redraw_requested: false,
            hide_deadline: None,
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
            pending_spatial_diagnostics: spatial_result.diagnostics,
            spatial_pose_unavailable: false,
            handoff_experiment: HandoffExperiment::Off,
            retained_frame: None,
            readiness_status_context: ReadinessStatusContext::default(),
            last_renderer_degradation: None,
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
        mut snapshot: OverlayPresentationSnapshot,
    ) -> SnapshotApplyOutcome {
        let current_revision = self.state.snapshot().revision;
        if snapshot.revision <= current_revision {
            return SnapshotApplyOutcome::Ignored {
                incoming_revision: snapshot.revision,
                current_revision,
            };
        }
        retain_semantically_current_blocks(&mut snapshot);
        self.retained_frame = None;

        let previous_calibration = self.state.calibration().clone();
        let visual_changed = self.state.apply_snapshot(&snapshot);
        self.pending_spatial_diagnostics.extend(
            self.spatial_lock
                .apply_snapshot_transition(&previous_calibration, self.state.snapshot())
                .diagnostics,
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

        SnapshotApplyOutcome::Applied {
            incoming_revision: snapshot.revision,
            current_revision: self.state.snapshot().revision,
            visual_changed,
            redraw_requested: self.redraw_requested,
        }
    }

    async fn emit_owner_status(
        &self,
        bridge: &mut BridgeClient,
        health_challenge_id: Option<u64>,
        due_elapsed_ms: u64,
        recovering: bool,
        terminal_failed: bool,
        due_active: bool,
        primary_failure_reason: Option<&'static str>,
        cleanup_failure_reason: Option<&'static str>,
    ) -> Result<(), RuntimeFailure> {
        let latest_handoff_revision = self
            .last_presentation_correlation
            .map(|correlation| correlation.scene_generation);
        let projection = project_owner_status(OwnerStatusInput {
            overlay_instance_id: bridge.overlay_instance_id(),
            runtime_generation: bridge.runtime_generation(),
            health_challenge_id,
            latest_applied_revision: self.state.snapshot().revision,
            latest_handoff_revision,
            desired_visible: self.desires_overlay_visible(),
            observed_runtime_visible: self.runtime_visibility_observed,
            has_drawable_text: self.has_drawable_text(),
            first_texture_submitted: self.first_texture_submitted,
            spatial_pose_unavailable: self.spatial_pose_unavailable,
            due_elapsed_ms,
            recovering,
            terminal_failed,
            due_active,
            in_flight_stage: self
                .presentation_diagnostics
                .records()
                .back()
                .map(|record| record.stage),
            primary_failure_reason,
            cleanup_failure_reason,
        });
        bridge
            .send_json(projection.payload)
            .await
            .map_err(|error| RuntimeFailure::Bridge(error.to_string()))
    }

    async fn emit_readiness_health_status(
        &self,
        bridge: &mut BridgeClient,
        health_challenge_id: u64,
    ) -> Result<(), RuntimeFailure> {
        let status = self.readiness_status_context;
        self.emit_owner_status(
            bridge,
            Some(health_challenge_id),
            status.due_elapsed_ms(),
            status.recovering,
            false,
            status.due_started_at.is_some(),
            None,
            None,
        )
        .await
    }

    fn has_accepted_due_work(&self) -> bool {
        !self.spatial_pose_retry_pending() && self.redraw_requested
    }

    fn spatial_pose_retry_pending(&self) -> bool {
        self.spatial_pose_unavailable && self.spatial_lock.pending().is_some()
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

    fn retained_frame_matches(
        &self,
        blocks: &[CaptionBlock],
        presentation: &CaptionPresentation,
        backend: PresentationBackend,
        openvr_adapter_identity: crate::presentation::AdapterIdentity,
        renderer_adapter_identity: crate::presentation::AdapterIdentity,
        presentation_causes: PresentationCauses,
    ) -> bool {
        self.handoff_experiment == HandoffExperiment::CachedFrameRehandoff
            && presentation_causes.is_native_fresh_retry_only()
            && self.first_texture_submitted
            && !self.spatial_pose_retry_pending()
            && self.retained_frame.as_ref().is_some_and(|retained| {
                retained.scene_generation == self.state.snapshot().revision
                    && retained.blocks == blocks
                    && retained.presentation == *presentation
                    && retained.backend == backend
                    && retained.openvr_adapter_identity == openvr_adapter_identity
                    && retained.renderer_adapter_identity == renderer_adapter_identity
            })
    }

    async fn emit_renderer_degradation_if_changed(
        &mut self,
        logger: &OverlayLogger,
        diagnostics: &RenderDiagnostics,
    ) -> Result<(), RuntimeFailure> {
        let current = RendererDegradationSummary::from_diagnostics(diagnostics);
        if current == self.last_renderer_degradation {
            return Ok(());
        }
        self.last_renderer_degradation = current;
        let Some(summary) = current else {
            return Ok(());
        };
        logger
            .warn(format!(
                "renderer_degradation style_resolution_fallback_lines={} heuristic_layout_fallbacks={} scope=changed physical_hmd_visibility=not_observable",
                summary.style_resolution_fallback_lines,
                summary.heuristic_layout_fallbacks,
            ))
            .await
            .map_err(|error| RuntimeFailure::Bridge(error.to_string()))
    }

    fn record_visibility_request(
        &mut self,
        reason: &'static str,
        desired_visible: bool,
        succeeded: bool,
    ) {
        let (Some(correlation), Some(backend)) = (
            self.last_presentation_correlation,
            self.last_presentation_backend,
        ) else {
            return;
        };
        self.presentation_diagnostics.configure_event_metadata(
            reason,
            HandoffMode::Off,
            self.retained_frame
                .as_ref()
                .map(|retained| retained.content_identity),
        );
        self.presentation_diagnostics.record_visibility_request(
            correlation,
            backend,
            desired_visible,
            succeeded,
        );
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
        self.retained_frame = None;
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
            "overlay_instance_id": bridge.overlay_instance_id(),
            "runtime_generation": bridge.runtime_generation(),
            "capabilities": {
                "execution_contract": {
                    "version": 1,
                    "revision": "r2"
                },
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
        self.submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, false)
            .await
            .map(|_| ())
    }

    async fn submit_frame_if_needed_with_timing<S: OverlayFrameSubmitter>(
        &mut self,
        renderer: &CaptionRenderer,
        openvr: &mut S,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
        preemptible: bool,
    ) -> Result<FrameCycleOutcome, RuntimeFailure> {
        if renderer.has_incomplete_producer() {
            let retained_observer = ReadinessCancellation::default();
            match renderer
                .prepare_frame_for_submission(&retained_observer)
                .await
            {
                ReadinessOutcome::Ready => {}
                ReadinessOutcome::TimedOut => return Err(RuntimeFailure::ReadinessTimedOut),
                ReadinessOutcome::Cancelled => return Err(RuntimeFailure::ReadinessCancelled),
                ReadinessOutcome::Failed => return Err(RuntimeFailure::ReadinessFailed),
            }
        }
        if self.stopped {
            return Err(RuntimeFailure::Stopped);
        }
        if self.first_texture_submitted && !self.redraw_requested {
            return Ok(FrameCycleOutcome::NoWork);
        }

        let prepare_started = Instant::now();
        let presentation = CaptionPresentation {
            background_alpha: self.state.calibration().background_alpha,
            text_scale: self.state.calibration().text_scale,
        };
        renderer.set_presentation(presentation.clone());
        openvr
            .apply_calibration(self.state.calibration())
            .map_err(|error| RuntimeFailure::OpenVr(format!("stage=calibration {error}")))?;

        let presentation_backend = renderer.presentation_backend();
        let openvr_adapter_identity = renderer.openvr_adapter_identity();
        let renderer_adapter_identity = renderer.adapter_identity();
        self.presentation_diagnostics.configure_adapter_handoff(
            openvr_adapter_identity,
            renderer_adapter_identity,
            renderer.adapter_match(openvr_adapter_identity),
        );
        let scene_generation = self.state.snapshot().revision;
        let presentation_causes = std::mem::take(&mut self.pending_presentation_causes);
        let blocks = self.caption_blocks();
        let cached_rehandoff = !renderer.has_incomplete_producer()
            && self.retained_frame_matches(
                &blocks,
                &presentation,
                presentation_backend,
                openvr_adapter_identity,
                renderer_adapter_identity,
                presentation_causes,
            );
        let content_identity = self
            .retained_frame
            .as_ref()
            .filter(|_| cached_rehandoff)
            .map(|retained| retained.content_identity)
            .or_else(|| Some(frame_content_identity(&blocks, &presentation)));
        let handoff_mode = if cached_rehandoff {
            HandoffMode::CachedFrameRehandoff
        } else {
            HandoffMode::Off
        };
        self.presentation_diagnostics.configure_event_metadata(
            if cached_rehandoff {
                "cadence_identical_completed_frame"
            } else {
                "fresh_render_required"
            },
            handoff_mode,
            content_identity,
        );
        if self.pending_logical_revision_acceptance {
            self.presentation_diagnostics.accept_logical_revision(
                presentation_backend,
                scene_generation,
                presentation_causes.clone(),
            );
            self.pending_logical_revision_acceptance = false;
        }
        let presentation_correlation = if cached_rehandoff {
            let retained_render_generation = self
                .retained_frame
                .as_ref()
                .expect("matched retained frame")
                .render_generation;
            self.presentation_diagnostics.begin_rehandoff(
                scene_generation,
                presentation_causes,
                retained_render_generation,
            )
        } else {
            self.presentation_diagnostics
                .begin_presentation(scene_generation, presentation_causes)
        }
        .expect("active presentation diagnostics owner");
        let cpu_prepare_us = duration_us(prepare_started.elapsed());
        let has_drawable_text = blocks.iter().any(CaptionBlock::has_drawable_text);
        if has_drawable_text {
            if let Some(actual_visible) = openvr.observed_overlay_visible() {
                self.note_observed_runtime_visible(actual_visible);
                self.presentation_diagnostics.configure_event_metadata(
                    "runtime_visibility_query",
                    handoff_mode,
                    content_identity,
                );
                self.presentation_diagnostics.record_visibility(
                    presentation_correlation,
                    presentation_backend,
                    true,
                    actual_visible,
                    actual_visible,
                );
            }
        }
        let should_show_after_submit = has_drawable_text
            && !self.overlay_visible
            && self.visibility_request_pending != Some(true);
        if has_drawable_text {
            self.hide_deadline = None;
        } else if self.first_texture_submitted
            && (self.overlay_visible || self.visibility_request_pending == Some(true))
            && self.hide_deadline.is_none()
        {
            self.hide_deadline = Some(Instant::now() + EMPTY_OVERLAY_HIDE_DELAY);
        }
        let retained_before_cycle = self.retained_frame.take();
        let render_started = Instant::now();
        let (frame, fresh_render) = if cached_rehandoff {
            (
                retained_before_cycle
                    .expect("matching retained frame remains available")
                    .frame,
                false,
            )
        } else {
            drop(retained_before_cycle);
            let render_result = if blocks.is_empty() {
                renderer.render_empty_frame()
            } else {
                renderer.render_blocks_with_debug_overlay(blocks.clone(), None)
            };
            let cpu_render_us = duration_us(render_started.elapsed());
            self.presentation_diagnostics.record_render_return(
                presentation_correlation,
                presentation_backend,
                render_result.is_ok(),
                cpu_prepare_us,
                cpu_render_us,
            );
            match render_result {
                Ok(frame) => (Arc::new(frame), true),
                Err(error) => {
                    self.retain_failed_presentation_causes(presentation_correlation);
                    return Err(RuntimeFailure::Render(format!(
                        "stage=render_frame {error}"
                    )));
                }
            }
        };
        if fresh_render {
            self.emit_renderer_degradation_if_changed(logger, frame.diagnostics())
                .await?;
        }
        let self_block_count = visible_self_block_count(frame.layout());
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
                        let health_challenge = match &message {
                            Ok(BridgeIncoming::HealthChallenge(challenge)) => {
                                self.emit_readiness_health_status(
                                    bridge,
                                    challenge.challenge_id,
                                )
                                .await?;
                                true
                            }
                            _ => false,
                        };
                        let ignored =
                            health_challenge || matches!(message, Ok(BridgeIncoming::Heartbeat));
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
                return Ok(FrameCycleOutcome::Preempted(
                    pending_message.expect("cancelled readiness has pending message"),
                ));
            }
            let failure = match readiness_outcome {
                ReadinessOutcome::TimedOut => RuntimeFailure::ReadinessTimedOut,
                ReadinessOutcome::Cancelled => RuntimeFailure::ReadinessCancelled,
                ReadinessOutcome::Failed => RuntimeFailure::ReadinessFailed,
                ReadinessOutcome::Ready => unreachable!(),
            };
            return Err(failure);
        }
        if has_drawable_text {
            if let Some(pending) = self.spatial_lock.pending() {
                let reanchor_result = openvr.reanchor_spatial_locked();
                match reanchor_result {
                    Ok(SpatialReanchorOutcome::Applied) => {
                        self.spatial_lock.complete_pending();
                        self.spatial_pose_unavailable = false;
                    }
                    Ok(SpatialReanchorOutcome::PoseUnavailable) => {
                        self.spatial_pose_unavailable = true;
                        self.redraw_requested = true;
                        self.retain_failed_presentation_causes(presentation_correlation);
                        self.pending_spatial_diagnostics.push(SpatialDiagnostic(format!(
                            "spatial_reanchor_pose_unavailable reason={} revision={scene_generation}",
                            pending.reason.as_str()
                        )));
                        self.emit_pending_spatial_diagnostics(logger).await;
                        return Ok(FrameCycleOutcome::NoWork);
                    }
                    Err(error) => {
                        self.retain_failed_presentation_causes(presentation_correlation);
                        return Err(RuntimeFailure::OpenVr(format!("stage=reanchor {error}")));
                    }
                }
            }
        }
        self.presentation_diagnostics.configure_event_metadata(
            if cached_rehandoff {
                "cached_completed_frame_rehandoff"
            } else {
                "fresh_render_submission"
            },
            handoff_mode,
            content_identity,
        );
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
            return Err(RuntimeFailure::OpenVr(format!(
                "stage=submit_frame {error}"
            )));
        }
        if should_show_after_submit {
            self.presentation_diagnostics.configure_event_metadata(
                "frame_submit_text_visible",
                handoff_mode,
                content_identity,
            );
            let visibility_result = openvr.set_overlay_visible(true);
            let visibility_succeeded = visibility_result.is_ok();
            self.presentation_diagnostics.record_visibility_request(
                presentation_correlation,
                presentation_backend,
                true,
                visibility_succeeded,
            );
            if visibility_succeeded {
                self.visibility_request_pending = Some(true);
                if let Some(observed) = openvr.observed_overlay_visible() {
                    self.note_observed_runtime_visible(observed);
                    self.presentation_diagnostics.record_visibility(
                        presentation_correlation,
                        presentation_backend,
                        true,
                        observed,
                        observed,
                    );
                }
            }
            if let Err(error) = visibility_result {
                return Err(RuntimeFailure::OpenVr(format!(
                    "stage=show_overlay {error}"
                )));
            }
        }
        if !has_drawable_text && self.first_texture_submitted {
            self.hide_deadline = Some(Instant::now() + EMPTY_OVERLAY_HIDE_DELAY);
        }
        self.emit_pending_spatial_diagnostics(logger).await;
        self.last_presentation_correlation = Some(presentation_correlation);

        self.last_presentation_backend = Some(presentation_backend);
        self.last_submitted_had_self = self_block_count > 0;
        self.redraw_requested = false;

        if !self.first_texture_submitted {
            self.first_texture_submitted = true;
            self.emit_ready(bridge, logger).await?;
        }

        self.retained_frame = has_drawable_text.then(|| RetainedFrame {
            frame,
            scene_generation,
            render_generation: presentation_correlation.render_generation,
            blocks,
            presentation,
            backend: presentation_backend,
            openvr_adapter_identity,
            renderer_adapter_identity,
            content_identity: content_identity.expect("frame identity is always computed"),
        });

        Ok(if cached_rehandoff {
            FrameCycleOutcome::CachedFrameRehandoff
        } else {
            FrameCycleOutcome::Submitted
        })
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
                        self.handle_hide_deadline(openvr).await?;
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
            .submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, true)
            .await?
        {
            FrameCycleOutcome::Preempted(message) => Some(message),
            FrameCycleOutcome::Submitted
            | FrameCycleOutcome::CachedFrameRehandoff
            | FrameCycleOutcome::NoWork => None,
        };
        while let Some(message) = pending_message.take() {
            let (continue_running, next_message) = self
                .handle_bridge_message(message, renderer, openvr, bridge, logger)
                .await?;
            if !continue_running {
                return Ok(());
            }
            pending_message = if next_message.is_some() || !self.redraw_requested {
                next_message
            } else {
                match self
                    .submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, true)
                    .await?
                {
                    FrameCycleOutcome::Preempted(message) => Some(message),
                    FrameCycleOutcome::Submitted
                    | FrameCycleOutcome::CachedFrameRehandoff
                    | FrameCycleOutcome::NoWork => None,
                }
            };
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
            Ok(BridgeIncoming::HealthChallenge(challenge)) => {
                self.emit_owner_status(
                    bridge,
                    Some(challenge.challenge_id),
                    0,
                    false,
                    false,
                    false,
                    None,
                    None,
                )
                .await?;
                Ok((true, None))
            }
            Ok(BridgeIncoming::Snapshot(snapshot)) => {
                self.apply_snapshot(snapshot);
                let pending = self
                    .submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, true)
                    .await?;
                Ok((true, pending.pending_message()))
            }
            Ok(BridgeIncoming::Event(event)) => {
                self.handle_event(event).await?;
                if self.stopped {
                    return Ok((false, None));
                }
                let pending = self
                    .submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, true)
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
    ) -> Result<(), RuntimeFailure> {
        self.hide_deadline = None;
        if !self.first_texture_submitted
            || (!self.overlay_visible && self.visibility_request_pending != Some(true))
            || self.has_drawable_text()
        {
            return Ok(());
        }
        self.presentation_diagnostics.configure_event_metadata(
            "empty_scene_hide_deadline",
            HandoffMode::Off,
            self.retained_frame
                .as_ref()
                .map(|retained| retained.content_identity),
        );
        let visibility_result = openvr.set_overlay_visible(false);
        let visibility_succeeded = visibility_result.is_ok();
        if visibility_succeeded {
            self.visibility_request_pending = Some(false);
        }
        if let (Some(correlation), Some(backend)) = (
            self.last_presentation_correlation,
            self.last_presentation_backend,
        ) {
            self.presentation_diagnostics.record_visibility_request(
                correlation,
                backend,
                false,
                visibility_succeeded,
            );
            if visibility_succeeded {
                if let Some(observed) = openvr.observed_overlay_visible() {
                    self.note_observed_runtime_visible(observed);
                    self.presentation_diagnostics.record_visibility(
                        correlation,
                        backend,
                        false,
                        observed,
                        !observed,
                    );
                }
            }
        }
        if let Err(error) = visibility_result {
            return Err(RuntimeFailure::OpenVr(format!(
                "stage=hide_overlay {error}"
            )));
        }
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
        self.runtime_visibility_observed = Some(visible);
        self.visibility_request_pending = None;
    }

    async fn emit_pending_spatial_diagnostics(&mut self, logger: &OverlayLogger) {
        let diagnostics = std::mem::take(&mut self.pending_spatial_diagnostics);
        for diagnostic in diagnostics {
            let SpatialDiagnostic(message) = diagnostic;
            let _ = logger.warn(message).await;
        }
    }

    pub fn presentation_diagnostics(&self) -> &PresentationDiagnostics {
        &self.presentation_diagnostics
    }

    #[doc(hidden)]
    pub fn pending_presentation_causes_for_test(&self) -> Vec<PresentationCause> {
        self.pending_presentation_causes.to_vec()
    }

    fn frame_progress(&self) -> FrameProgress {
        FrameProgress {
            correlation: self.last_presentation_correlation,
        }
    }
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn frame_content_identity(blocks: &[CaptionBlock], presentation: &CaptionPresentation) -> u64 {
    let mut hasher = DefaultHasher::new();
    blocks.len().hash(&mut hasher);
    for block in blocks {
        block.id.hash(&mut hasher);
        block.primary_text.hash(&mut hasher);
        block.secondary_text.hash(&mut hasher);
        block.secondary_enabled.hash(&mut hasher);
        block.primary_language.hash(&mut hasher);
        block.secondary_language.hash(&mut hasher);
        block.block_variant.hash(&mut hasher);
        block.channel.hash(&mut hasher);
        block.opacity.to_bits().hash(&mut hasher);
        block.offset_y_px.to_bits().hash(&mut hasher);
        block.height_scale.to_bits().hash(&mut hasher);
        block.slot_index.hash(&mut hasher);
        block.slot_top_px.to_bits().hash(&mut hasher);
        block.slot_assigned.hash(&mut hasher);
    }
    presentation.background_alpha.to_bits().hash(&mut hasher);
    presentation.text_scale.to_bits().hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug, Clone)]
pub struct NativePresentationRetryHandle {
    sender: mpsc::Sender<()>,
}

pub const NATIVE_FRESH_RETRY_CADENCE: Duration = Duration::from_millis(100);
pub const NATIVE_FRESH_RETRY_DEADLINE: Duration = Duration::from_millis(500);
pub const NATIVE_FRESH_RETRY_MAX_COMPLETED: u32 = 5;
pub const NATIVE_STREAM_RETRY_MAX_COMPLETED: u32 = 4;
pub const NATIVE_READINESS_NO_PROGRESS_TIMEOUT: Duration = Duration::from_secs(2);

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
    retry_episodes: RetryEpisodes,
    retry_profile: &'static str,
    successful_attempt_audit: VecDeque<PresentationCorrelation>,
    readiness_timeouts_since_success: u32,
    readiness_no_progress_timeout: Duration,
    readiness_no_progress_deadline: Option<Instant>,
    readiness_retry_due: Option<Instant>,
    pose_wait_suspended: bool,
    next_status_due: Instant,
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
            retry_policy: NativeFreshRetryPolicy::new(
                NATIVE_FRESH_RETRY_CADENCE,
                NATIVE_FRESH_RETRY_DEADLINE,
                NATIVE_FRESH_RETRY_MAX_COMPLETED,
            ),
            retry_episodes: RetryEpisodes::new(),
            retry_profile: "p05",
            successful_attempt_audit: VecDeque::with_capacity(NATIVE_FRESH_AUDIT_CAPACITY),
            readiness_timeouts_since_success: 0,
            readiness_no_progress_timeout: NATIVE_READINESS_NO_PROGRESS_TIMEOUT,
            readiness_no_progress_deadline: None,
            readiness_retry_due: None,
            pose_wait_suspended: false,
            next_status_due: Instant::now(),
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

    pub fn new_with_profile_and_experiment(
        snapshot: OverlayPresentationSnapshot,
        renderer: CaptionRenderer,
        openvr: S,
        profile: crate::manifest::QuietTailProfile,
        experiment: HandoffExperiment,
    ) -> Self {
        let mut owner = Self::new_with_profile(snapshot, renderer, openvr, profile);
        owner.runtime.configure_handoff_experiment(experiment);
        owner
    }

    #[doc(hidden)]
    pub fn set_handoff_experiment_for_test(&mut self, experiment: HandoffExperiment) {
        self.runtime.configure_handoff_experiment(experiment);
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
    pub fn set_readiness_no_progress_timeout_for_test(&mut self, timeout: Duration) {
        self.readiness_no_progress_timeout = timeout;
        self.readiness_no_progress_deadline = None;
    }

    #[doc(hidden)]
    pub fn readiness_timeout_count_for_test(&self) -> u32 {
        self.readiness_timeouts_since_success
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
        self.retry_episodes.audit()
    }

    #[doc(hidden)]
    pub fn fresh_retry_audit_dropped_for_test(&self) -> u64 {
        self.retry_episodes.audit_dropped()
    }

    #[doc(hidden)]
    pub fn successful_attempt_audit_for_test(&self) -> Vec<PresentationCorrelation> {
        self.successful_attempt_audit.iter().copied().collect()
    }

    fn complete_due_progress(&mut self) {
        self.readiness_timeouts_since_success = 0;
        self.readiness_no_progress_deadline = None;
        self.readiness_retry_due = None;
    }

    fn capture_successful_attempt(&mut self) {
        self.complete_due_progress();
        let Some(correlation) = self.runtime.frame_progress().correlation else {
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

    fn record_fresh_retry(&mut self, schedule: NativeFreshSchedule, outcome: &'static str) {
        self.push_fresh_retry_audit(schedule, outcome);
    }

    fn push_fresh_retry_audit(&mut self, schedule: NativeFreshSchedule, outcome: &'static str) {
        self.retry_episodes.record(&schedule, outcome);
    }

    #[cfg(test)]
    fn finish_initial_reconcile(
        &mut self,
        result: Result<(), RuntimeFailure>,
    ) -> Result<(), RuntimeFailure> {
        if result.is_err() {
            let _ = self.teardown();
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
        _logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        for channel in [FreshRetryChannel::SelfChannel, FreshRetryChannel::Peer] {
            let current_cause = self
                .retry_episodes
                .schedule(channel)
                .is_some_and(|schedule| {
                    self.runtime
                        .pending_presentation_causes
                        .contains(Self::intent_cause(
                            schedule,
                            PresentationCauseKind::NativeFreshRetry,
                        ))
                });
            let result = self.retry_episodes.reconcile(
                channel,
                RetryIntent {
                    generation: self.channel_generation(channel),
                    episode: self.channel_episode(channel),
                    target_identity: self.channel_target_identity(channel),
                    required_scene_generation: self.runtime.state().snapshot().revision,
                },
                Instant::now(),
                self.retry_policy,
                NATIVE_STREAM_RETRY_MAX_COMPLETED,
                current_cause,
            );
            if let Some((_, next)) = result.cause_transfer {
                self.runtime
                    .pending_presentation_causes
                    .insert(Self::intent_cause(
                        &next,
                        PresentationCauseKind::NativeFreshRetry,
                    ));
            }
            for transition in result.transitions {
                self.record_fresh_retry(transition.schedule, transition.outcome);
            }
        }
        Ok(())
    }

    fn next_fresh_due(&self) -> Option<Instant> {
        if self.runtime.spatial_pose_retry_pending() {
            return None;
        }
        self.retry_episodes
            .schedules()
            .map(|schedule| schedule.next_due)
            .min()
    }

    fn next_retry_wake(&self) -> Option<Instant> {
        [
            self.next_fresh_due(),
            self.readiness_retry_due,
            self.readiness_no_progress_deadline,
            Some(self.next_status_due),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    fn arm_due_deadline(&mut self) {
        if self.readiness_no_progress_deadline.is_none() {
            self.readiness_no_progress_deadline =
                Some(Instant::now() + self.readiness_no_progress_timeout);
        }
    }
    fn sync_runtime_readiness_status_context(&mut self) {
        let due_started_at = self.readiness_no_progress_deadline.map(|deadline| {
            deadline
                .checked_sub(self.readiness_no_progress_timeout)
                .unwrap_or(deadline)
        });
        self.runtime.readiness_status_context = ReadinessStatusContext {
            due_started_at,
            recovering: self.readiness_retry_due.is_some()
                || self.readiness_timeouts_since_success > 0,
        };
    }

    async fn emit_current_status(
        &mut self,
        bridge: &mut BridgeClient,
        health_challenge_id: Option<u64>,
    ) -> Result<(), RuntimeFailure> {
        let now = Instant::now();
        let due_elapsed_ms = self
            .readiness_no_progress_deadline
            .map(|deadline| {
                let started = deadline
                    .checked_sub(self.readiness_no_progress_timeout)
                    .unwrap_or(deadline);
                now.saturating_duration_since(started).as_millis() as u64
            })
            .unwrap_or(0);
        self.runtime
            .emit_owner_status(
                bridge,
                health_challenge_id,
                due_elapsed_ms,
                self.readiness_retry_due.is_some() || self.readiness_timeouts_since_success > 0,
                false,
                self.readiness_no_progress_deadline.is_some(),
                None,
                None,
            )
            .await
    }

    async fn emit_terminal_status(
        &mut self,
        bridge: &mut BridgeClient,
        primary_failure_reason: Option<&'static str>,
        cleanup_failure_reason: Option<&'static str>,
    ) {
        let now = Instant::now();
        let due_elapsed_ms = self
            .readiness_no_progress_deadline
            .map(|deadline| {
                let started = deadline
                    .checked_sub(self.readiness_no_progress_timeout)
                    .unwrap_or(deadline);
                now.saturating_duration_since(started).as_millis() as u64
            })
            .unwrap_or(0);
        let _ = self
            .runtime
            .emit_owner_status(
                bridge,
                None,
                due_elapsed_ms,
                false,
                true,
                self.readiness_no_progress_deadline.is_some(),
                primary_failure_reason,
                cleanup_failure_reason,
            )
            .await;
    }

    async fn finish_run(
        &mut self,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
        result: Result<(), RuntimeFailure>,
    ) -> Result<(), RuntimeFailure> {
        let primary_failure_reason = result.as_ref().err().map(RuntimeFailure::failure_reason);
        let cleanup_result = self.teardown();
        let cleanup_failure_reason = cleanup_result
            .as_ref()
            .err()
            .map(RuntimeFailure::failure_reason);
        if primary_failure_reason.is_some() || cleanup_failure_reason.is_some() {
            self.emit_terminal_status(bridge, primary_failure_reason, cleanup_failure_reason)
                .await;
        }
        let outcome = match result {
            Err(primary) => Err(primary),
            Ok(()) => match cleanup_result {
                Err(cleanup) => Err(cleanup),
                Ok(()) => logger
                    .emit_stdout_event(&json!({
                        "type": "shutdown_complete",
                        "overlay_instance_id": bridge.overlay_instance_id(),
                        "runtime_generation": bridge.runtime_generation()
                    }))
                    .await
                    .map_err(|error| RuntimeFailure::Bridge(error.to_string())),
            },
        };
        outcome
    }

    fn note_readiness_timeout(&mut self) -> Result<(), RuntimeFailure> {
        let now = Instant::now();
        self.readiness_timeouts_since_success =
            self.readiness_timeouts_since_success.saturating_add(1);
        let deadline = *self
            .readiness_no_progress_deadline
            .get_or_insert(now + self.readiness_no_progress_timeout);
        if now >= deadline {
            return Err(RuntimeFailure::ReadinessStalled);
        }
        self.runtime.request_native_presentation_retry();
        let due = now + self.retry_policy.cadence;
        self.readiness_retry_due = Some(due);
        self.retry_episodes.set_all_next_due(due);
        Ok(())
    }

    async fn complete_frame_cycle(
        &mut self,
        result: Result<FrameCycleOutcome, RuntimeFailure>,
        _logger: &OverlayLogger,
    ) -> Result<Option<FrameCycleOutcome>, RuntimeFailure> {
        if self.pose_wait_suspended && !self.runtime.spatial_pose_retry_pending() {
            self.pose_wait_suspended = false;
        }
        match result {
            Ok(FrameCycleOutcome::NoWork) if self.runtime.spatial_pose_retry_pending() => {
                self.pose_wait_suspended = true;
                self.complete_due_progress();
                self.readiness_retry_due = Some(Instant::now() + self.retry_policy.cadence);
                Ok(Some(FrameCycleOutcome::NoWork))
            }
            Ok(outcome) => Ok(Some(outcome)),
            Err(RuntimeFailure::ReadinessTimedOut) => {
                self.note_readiness_timeout()?;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn due_fresh_channels(&self, now: Instant) -> Vec<FreshRetryChannel> {
        self.retry_episodes
            .schedules()
            .filter(|schedule| schedule.next_due <= now)
            .map(|schedule| schedule.channel)
            .collect()
    }

    fn active_fresh_schedules(&self, now: Instant) -> Vec<NativeFreshSchedule> {
        self.retry_episodes
            .schedules()
            .filter(|schedule| schedule.next_due <= now && now <= schedule.deadline)
            .cloned()
            .collect()
    }

    fn submission_eligible_fresh_schedules(
        &self,
        now: Instant,
        current_handoff_due: bool,
    ) -> Vec<NativeFreshSchedule> {
        let scene_revision = self.runtime.state().snapshot().revision;
        self.retry_episodes
            .schedules()
            .filter(|schedule| {
                now <= schedule.deadline
                    && (schedule.next_due <= now
                        || (schedule.completed == 0
                            && schedule.required_scene_generation == scene_revision
                            && (self.runtime.redraw_requested() || current_handoff_due)))
            })
            .cloned()
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
            let Some(schedule) = self.retry_episodes.schedule(channel).cloned() else {
                continue;
            };
            if schedule.expired_at(now) {
                self.retry_episodes.take_schedule(channel);
                let disposition = if self.runtime.handoff_experiment
                    == HandoffExperiment::CachedFrameRehandoff
                    && schedule.completed == 0
                {
                    "experiment_expired_unsatisfied"
                } else {
                    "expired"
                };
                self.record_fresh_retry(schedule, disposition);
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
                .submit_frame_if_needed_with_timing(renderer, openvr, bridge, logger, true)
                .await
        };
        let outcome = match attempt {
            Ok(outcome) => outcome,
            Err(RuntimeFailure::ReadinessTimedOut) => {
                self.note_readiness_timeout()?;
                return Ok(FrameCycleOutcome::NoWork);
            }
            Err(primary_failure) => {
                for schedule in due {
                    if let Some(active) = self.retry_episodes.fail_matching(&schedule) {
                        self.push_fresh_retry_audit(active, "failed");
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
            FrameCycleOutcome::CachedFrameRehandoff => {
                let now = Instant::now();
                for schedule in due {
                    if let Some(fact) = self.retry_episodes.defer_cached_rehandoff(
                        &schedule,
                        now,
                        self.retry_policy.cadence,
                    ) {
                        self.record_fresh_retry(fact, "experiment_cached_frame_rehandoff");
                    }
                }
            }
            FrameCycleOutcome::Preempted(_) => {
                for schedule in due {
                    if let Some(fact) = self.retry_episodes.matching_schedule(&schedule) {
                        self.record_fresh_retry(fact, "preempted");
                    }
                }
            }
            FrameCycleOutcome::NoWork => {}
        }
        Ok(outcome)
    }

    async fn satisfy_schedules_from_last_submission(
        &mut self,
        _logger: &OverlayLogger,
        captured_schedules: &[NativeFreshSchedule],
        cause_kind: PresentationCauseKind,
    ) -> Result<(), RuntimeFailure> {
        let Some(correlation) = self.runtime.frame_progress().correlation else {
            return Ok(());
        };
        for captured in captured_schedules {
            let channel = captured.channel;
            let current_generation = self.channel_generation(channel);
            let current_target_identity = self.channel_target_identity(channel);
            let Some(active) = self.retry_episodes.schedule(channel) else {
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
            let Some(completed) = self.retry_episodes.complete_matching(
                captured,
                Instant::now(),
                self.retry_policy.cadence,
            ) else {
                continue;
            };
            self.record_fresh_retry(completed, "completed");
        }
        Ok(())
    }

    pub async fn run(
        &mut self,
        bridge: &mut BridgeClient,
        logger: &OverlayLogger,
    ) -> Result<(), RuntimeFailure> {
        self.emit_current_status(bridge, None).await?;
        self.next_status_due = Instant::now() + Duration::from_millis(250);
        if self.runtime.has_accepted_due_work() {
            self.arm_due_deadline();
        }
        self.sync_runtime_readiness_status_context();
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
                return self.finish_run(bridge, logger, Err(error)).await;
            }
            if let Err(error) = self.note_readiness_timeout() {
                return self.finish_run(bridge, logger, Err(error)).await;
            }
        }
        if self.runtime.spatial_pose_retry_pending() {
            self.pose_wait_suspended = true;
            self.complete_due_progress();
            self.readiness_retry_due = Some(Instant::now() + self.retry_policy.cadence);
        }
        if self.runtime.is_stopped() {
            return self.finish_run(bridge, logger, Ok(())).await;
        }
        if !initial_timed_out && self.runtime.frame_progress().correlation.is_some() {
            self.capture_successful_attempt();
        }
        if let Err(error) = self.reconcile_fresh_schedules(logger).await {
            return self.finish_run(bridge, logger, Err(error)).await;
        }
        let result = self.run_owned_event_loop(bridge, logger).await;
        self.finish_run(bridge, logger, result).await
    }

    async fn pump_openvr_events(&mut self) -> Result<(), RuntimeFailure> {
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
                OpenVrEventClass::Reconfigure => match event {
                    OpenVrRuntimeEvent::OverlayShown => {
                        self.runtime.note_observed_runtime_visible(true);
                    }
                    OpenVrRuntimeEvent::OverlayHidden => {
                        saw_overlay_hidden = true;
                        self.runtime.note_observed_runtime_visible(false);
                    }
                    _ => {}
                },
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
        if observed.is_some_and(|visible| visible == desired_visible)
            && self.runtime.visibility_request_pending.is_none()
            && !self.runtime.spatial_pose_retry_pending()
            && !self.runtime.has_accepted_due_work()
        {
            self.complete_due_progress();
        }
        let needs_reassert = match observed {
            Some(visible) => visible != desired_visible,
            None => saw_overlay_hidden && desired_visible,
        };
        if needs_reassert && self.runtime.visibility_request_pending != Some(desired_visible) {
            self.arm_due_deadline();
            let visibility_result = {
                let openvr = self.openvr.as_mut().expect("active OpenVR session");
                openvr.set_overlay_visible(desired_visible)
            };
            self.runtime.record_visibility_request(
                "runtime_visibility_reconcile",
                desired_visible,
                visibility_result.is_ok(),
            );
            visibility_result.map_err(|error| {
                RuntimeFailure::OpenVr(format!("stage=reconcile_visibility {error}"))
            })?;
            self.runtime.visibility_request_pending = Some(desired_visible);
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
            self.pump_openvr_events().await?;
            if self
                .readiness_no_progress_deadline
                .is_some_and(|deadline| deadline <= Instant::now())
            {
                return Err(RuntimeFailure::ReadinessStalled);
            }
            self.sync_runtime_readiness_status_context();
            let hide_deadline = self.runtime.hide_deadline;
            let message = if let Some(message) = pending_message.take() {
                Some(message)
            } else {
                tokio::select! {
                    _ = sleep_until(self.next_retry_wake().unwrap_or_else(Instant::now)), if self.next_retry_wake().is_some() => {
                        let now = Instant::now();
                        if self
                            .readiness_no_progress_deadline
                            .is_some_and(|deadline| deadline <= now)
                        {
                            return Err(RuntimeFailure::ReadinessStalled);
                        }
                        if now >= self.next_status_due {
                            self.emit_current_status(bridge, None).await?;
                            self.next_status_due = now
                                + if self.readiness_no_progress_deadline.is_some() {
                                    Duration::from_millis(250)
                                } else {
                                    Duration::from_secs(1)
                                };
                        }
                        let channels = self.due_fresh_channels(now);
                        if !channels.is_empty() {
                            self.arm_due_deadline();
                            self.sync_runtime_readiness_status_context();
                            let outcome = self.run_due_fresh_attempt(channels, bridge, logger).await?;
                            pending_message = outcome.pending_message();
                        } else if self.readiness_retry_due.is_some_and(|due| due <= now) {
                            self.readiness_retry_due = None;
                            if self.runtime.has_accepted_due_work() {
                                self.arm_due_deadline();
                            }
                            self.sync_runtime_readiness_status_context();
                            let result = {
                                let renderer = self.renderer.as_ref().expect("active renderer");
                                let openvr = self.openvr.as_mut().expect("active OpenVR session");
                                self.runtime
                                    .submit_frame_if_needed_with_timing(
                                        renderer, openvr, bridge, logger, true,
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
                        if self.runtime.has_accepted_due_work() {
                            self.arm_due_deadline();
                        }
                        self.sync_runtime_readiness_status_context();
                        self.runtime.request_native_presentation_retry();
                        let renderer = self.renderer.as_ref().expect("active renderer");
                        let openvr = self.openvr.as_mut().expect("active OpenVR session");
                        let result = self
                            .runtime
                            .submit_frame_if_needed_with_timing(
                                renderer, openvr, bridge, logger, true,
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
                        self.runtime.handle_hide_deadline(openvr).await?;
                        None
                    }
                    message = bridge.next_message() => Some(message),
                    _ = sleep_until(Instant::now() + OPENVR_EVENT_POLL_INTERVAL) => None,
                }
            };
            if let Some(message) = message {
                if let Ok(BridgeIncoming::HealthChallenge(challenge)) = &message {
                    self.emit_current_status(bridge, Some(challenge.challenge_id))
                        .await?;
                    continue;
                }
                let current_handoff_due = matches!(
                    &message,
                    Ok(BridgeIncoming::Snapshot(snapshot))
                        if snapshot.revision > self.runtime.state().snapshot().revision
                );
                let accepted_due_message = match &message {
                    Ok(BridgeIncoming::Snapshot(snapshot)) => {
                        snapshot.revision > self.runtime.state().snapshot().revision
                    }
                    _ => false,
                };
                if accepted_due_message {
                    self.arm_due_deadline();
                }
                self.sync_runtime_readiness_status_context();
                let previous_submission = self.runtime.frame_progress().correlation;
                let captured_schedules =
                    self.submission_eligible_fresh_schedules(Instant::now(), current_handoff_due);
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
                        self.note_readiness_timeout()?;
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
                if self.runtime.has_accepted_due_work() {
                    self.arm_due_deadline();
                }
                self.reconcile_fresh_schedules(logger).await?;
                let current_submission = self.runtime.frame_progress().correlation;
                if current_submission != previous_submission && current_submission.is_some() {
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

    fn teardown(&mut self) -> Result<(), RuntimeFailure> {
        self.retry_sender = None;
        self.retry_receiver.close();
        for schedule in self.retry_episodes.clear() {
            self.push_fresh_retry_audit(schedule, "teardown");
        }
        self.runtime.shutdown_presentation();
        let cleanup_result = self
            .openvr
            .as_mut()
            .map(|openvr| openvr.set_overlay_visible(false))
            .transpose()
            .map(|_| ())
            .map_err(|error| RuntimeFailure::OpenVr(format!("cleanup hide failed: {error}")));
        self.openvr = None;
        self.renderer = None;
        cleanup_result
    }
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

fn visible_self_block_count(layout: &CaptionLayoutResult) -> usize {
    layout
        .visible_blocks
        .iter()
        .filter(|block| block.channel == Some(CaptionChannel::SelfChannel))
        .count()
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
    run_with_manifest_and_profile(manifest, QuietTailProfile::P05, HandoffExperiment::Off).await
}

async fn run_with_manifest_and_profile(
    manifest: OverlayManifest,
    quiet_tail_profile: QuietTailProfile,
    handoff_experiment: HandoffExperiment,
) -> i32 {
    let logger = match OverlayLogger::open(&manifest.log_dir).await {
        Ok(logger) => logger,
        Err(error) => {
            eprintln!("[overlay][ERROR] failed to initialize logging: {error}");
            return 1;
        }
    };

    let exit_code = 'runtime: {
        if let Err(error) = validate_manifest(&manifest) {
            emit_startup_failure(&logger, &error).await;
            break 'runtime error.exit_code();
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
                break 'runtime startup_error.exit_code();
            }
        };

        if let Err(error) = perform_startup_preflight() {
            let startup_error = startup_error_from_preflight(error);
            let _ = bridge.close().await;
            emit_startup_failure(&logger, &startup_error).await;
            break 'runtime startup_error.exit_code();
        }

        let (renderer, openvr) = match initialize_runtime_resources(&manifest, &logger).await {
            Ok(resources) => resources,
            Err(error) => {
                let _ = bridge.close().await;
                emit_startup_failure(&logger, &error).await;
                break 'runtime error.exit_code();
            }
        };

        let mut owner = NativePresentationOwner::new_with_profile_and_experiment(
            snapshot,
            renderer,
            openvr,
            quiet_tail_profile,
            handoff_experiment,
        );
        let runtime_result = owner.run(&mut bridge, &logger).await;
        let reached_ready = owner.runtime().ready_sent();
        let _ = bridge.close().await;

        if let Err(error) = runtime_result.as_ref() {
            if !reached_ready {
                let startup_error = startup_error_from_runtime_failure(error.clone());
                emit_startup_failure(&logger, &startup_error).await;
                break 'runtime startup_error.exit_code();
            }
        }

        break 'runtime match runtime_result {
            Ok(()) => 0,
            Err(RuntimeFailure::RuntimeDisconnected) => 1,
            Err(error) => {
                let detail: String = match &error {
                    RuntimeFailure::Render(message) | RuntimeFailure::OpenVr(message) => {
                        message.chars().take(512).collect()
                    }
                    _ => String::new(),
                };
                let _ = logger
                    .error(format!(
                        "runtime_failure reason={} revision={} detail={detail:?}",
                        error.failure_reason(),
                        owner.runtime().state().snapshot().revision,
                    ))
                    .await;
                let _ = logger
                    .emit_stdout_event(&json!({
                        "type": "runtime_error",
                        "failure_reason": error.failure_reason(),
                    }))
                    .await;
                1
            }
        };
    };
    if logger.shutdown().is_err() && exit_code == 0 {
        1
    } else {
        exit_code
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
                "execution_contract": {"version": 1, "revision": "r2"},
                "native_presentation_retry": {"version": 1, "ownership": "exclusive"}
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
    let handoff_experiment = match resolve_handoff_experiment_from_env() {
        Ok(experiment) => experiment,
        Err(error) => {
            eprintln!(
                "[overlay][ERROR] startup_failure reason={}",
                error.failure_reason()
            );
            emit_startup_failure_to_stderr(&error).await;
            return error.exit_code();
        }
    };
    run_with_manifest_and_profile(manifest, quiet_tail_profile, handoff_experiment).await
}

fn startup_error_from_runtime_failure(error: RuntimeFailure) -> StartupError {
    match error {
        RuntimeFailure::Render(message) => StartupError::RuntimeRender(message),
        RuntimeFailure::OpenVr(message) => StartupError::RuntimeOpenVr(message),
        RuntimeFailure::ReadinessTimedOut => StartupError::ReadinessTimedOut,
        RuntimeFailure::ReadinessCancelled => StartupError::ReadinessCancelled,
        RuntimeFailure::ReadinessFailed => StartupError::ReadinessFailed,
        RuntimeFailure::ReadinessStalled => StartupError::ReadinessStalled,
        RuntimeFailure::Bridge(message) => StartupError::RuntimeBridge(message),
        RuntimeFailure::RuntimeDisconnected => StartupError::RuntimeDisconnected,
        RuntimeFailure::Stopped => StartupError::RuntimeStopped,
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
    let renderer = create_runtime_renderer(&openvr).map_err(startup_error_from_renderer)?;
    if let Some(warning) = renderer.font_initialization_warning() {
        logger
            .warn(warning)
            .await
            .map_err(|error| StartupError::Other(error.to_string()))?;
    }
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
        self.state
            .scene()
            .slots()
            .iter()
            .flatten()
            .map(caption_block_for_strip)
            .collect()
    }
}

fn caption_block_for_strip(strip: &OverlaySlot) -> CaptionBlock {
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

    CaptionBlock::new(strip.id.clone(), strip.primary_text.clone())
        .with_channel(channel)
        .with_variant(variant)
        .with_secondary_text(strip.secondary_text.clone(), strip.secondary_enabled)
        .with_language_metadata(
            strip.primary_language.clone(),
            strip.secondary_language.clone(),
        )
        .with_visual_state(1.0, 0.0, 1.0)
        .with_slot(strip.slot_index, strip.anchor_top_px)
}

#[cfg(test)]
mod tests {
    use super::{
        prepare_openvr_runtime, startup_error_from_runtime_failure, FrameCycleOutcome,
        FreshRetryChannel, NativeFreshSchedule, NativePresentationOwner, OverlayRuntime,
        RetainedFrame, RuntimeFailure, SnapshotApplyOutcome, StartupError,
        NATIVE_FRESH_AUDIT_CAPACITY, NATIVE_FRESH_RETRY_MAX_COMPLETED,
    };
    use crate::bridge::{BridgeClient, BridgeIncoming};
    use crate::logging::OverlayLogger;
    use crate::manifest::{HandoffExperiment, OverlayManifest, EXPECTED_CONTRACT_VERSION};

    #[test]
    fn runtime_and_startup_failure_reasons_preserve_first_distinct_cause() {
        assert_eq!(
            RuntimeFailure::Bridge("send".into()).failure_reason(),
            "bridge_failed"
        );
        assert_eq!(
            RuntimeFailure::Render("draw".into()).failure_reason(),
            "render_failed"
        );
        assert_eq!(
            RuntimeFailure::OpenVr("device".into()).failure_reason(),
            "openvr_failed"
        );
        assert_eq!(
            startup_error_from_runtime_failure(RuntimeFailure::ReadinessTimedOut).failure_reason(),
            "gpu_readiness_late"
        );
        assert_eq!(
            startup_error_from_runtime_failure(RuntimeFailure::Bridge("auth".into()))
                .failure_reason(),
            "bridge_failed"
        );
    }
    use crate::openvr::{
        FakeOpenVr, OpenVrError, OpenVrRuntimeEvent, OpenVrStartupPreflightError,
        OverlayFrameSubmitter, SpatialReanchorOutcome,
    };
    use crate::presentation::{
        AdapterIdentity, PresentationBackend, PresentationCause, PresentationCauseChannel,
        PresentationCauseKind, PresentationCauses, PresentationCorrelation,
    };
    use crate::renderer::{
        CaptionBlock, CaptionBlockVariant, CaptionChannel, CaptionPresentation, CaptionRenderer,
        FontLanguageBucket, FontSource, RenderDiagnostics, RenderedFrame, StyleBucketSourceCount,
    };
    use crate::state::{
        OverlayPresentationBlock, OverlayPresentationBlockVariant, OverlayPresentationCalibration,
        OverlayPresentationSnapshot,
    };
    use futures_util::{SinkExt, StreamExt};
    use serde_json::json;
    use std::cell::Cell;
    use std::io;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    #[test]
    fn cached_rehandoff_requires_full_current_raster_and_control_identity() {
        let renderer = CaptionRenderer::new_for_test().unwrap();
        let blocks = vec![CaptionBlock::new("self:stable", "stable")
            .with_channel(CaptionChannel::SelfChannel)
            .with_variant(CaptionBlockVariant::Finalized)];
        let presentation = CaptionPresentation::default();
        let frame = Arc::new(renderer.render_blocks(blocks.clone()).unwrap());
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 7,
            blocks: vec![block("self:stable", "self", "stable", "", true)],
            ..Default::default()
        });
        runtime.first_texture_submitted = true;
        runtime.configure_handoff_experiment(HandoffExperiment::CachedFrameRehandoff);
        let mut retry = PresentationCauses::default();
        retry.insert(PresentationCause {
            kind: PresentationCauseKind::NativeFreshRetry,
            channel: Some(PresentationCauseChannel::SelfChannel),
            trigger_generation: Some(1),
        });
        runtime.retained_frame = Some(RetainedFrame {
            frame,
            scene_generation: 7,
            render_generation: 3,
            blocks: blocks.clone(),
            presentation: presentation.clone(),
            backend: PresentationBackend::Test,
            openvr_adapter_identity: AdapterIdentity::Test,
            renderer_adapter_identity: AdapterIdentity::Test,
            content_identity: 42,
        });

        assert!(runtime.retained_frame_matches(
            &blocks,
            &presentation,
            PresentationBackend::Test,
            AdapterIdentity::Test,
            AdapterIdentity::Test,
            retry,
        ));
        let mut changed_blocks = blocks.clone();
        changed_blocks[0].primary_text = "changed".into();
        assert!(!runtime.retained_frame_matches(
            &changed_blocks,
            &presentation,
            PresentationBackend::Test,
            AdapterIdentity::Test,
            AdapterIdentity::Test,
            retry,
        ));
        let changed_presentation = CaptionPresentation {
            text_scale: 1.25,
            ..presentation.clone()
        };
        assert!(!runtime.retained_frame_matches(
            &blocks,
            &changed_presentation,
            PresentationBackend::Test,
            AdapterIdentity::Test,
            AdapterIdentity::Test,
            retry,
        ));
        let mut control = retry;
        control.insert(PresentationCause {
            kind: PresentationCauseKind::RuntimeControl,
            channel: None,
            trigger_generation: None,
        });
        assert!(!runtime.retained_frame_matches(
            &blocks,
            &presentation,
            PresentationBackend::Test,
            AdapterIdentity::Test,
            AdapterIdentity::Test,
            control,
        ));
    }
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

        let audit = owner.fresh_retry_audit_for_test();
        assert_eq!(audit.len(), NATIVE_FRESH_AUDIT_CAPACITY);
        assert_eq!(owner.fresh_retry_audit_dropped_for_test(), 1);
        assert_eq!(audit.first().unwrap().1, 1);
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
        owner
            .retry_episodes
            .replace_schedule(FreshRetryChannel::SelfChannel, self_schedule);
        owner
            .retry_episodes
            .replace_schedule(FreshRetryChannel::Peer, peer_schedule);
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

        let logger = OverlayLogger::open(std::env::temp_dir()).await.unwrap();
        let mut owner = NativePresentationOwner::new(
            snapshot(1, 1, "stream", 7),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        let first_schedule = owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .unwrap()
            .clone();
        owner
            .retry_episodes
            .schedule_mut(FreshRetryChannel::Peer)
            .unwrap()
            .completed = 2;
        owner
            .retry_episodes
            .accounting_mut(FreshRetryChannel::Peer)
            .unwrap()
            .completed = 2;
        owner.runtime.apply_snapshot(snapshot(2, 2, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        let replaced_generation = owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .unwrap();
        assert_eq!(replaced_generation.trigger_generation, 2);
        assert_eq!(replaced_generation.required_scene_generation, 2);
        assert_eq!(replaced_generation.target_identity, "peer:stable");
        assert_eq!(replaced_generation.completed, 2);
        assert_eq!(replaced_generation.max_completed, 4);
        assert_eq!(replaced_generation.next_due, first_schedule.next_due);
        assert_eq!(replaced_generation.deadline, first_schedule.deadline);

        owner
            .retry_episodes
            .accounting_mut(FreshRetryChannel::Peer)
            .unwrap()
            .completed = 4;
        owner.retry_episodes.take_schedule(FreshRetryChannel::Peer);
        owner.runtime.apply_snapshot(snapshot(3, 3, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .is_none());
        assert_eq!(
            owner
                .retry_episodes
                .accounting(FreshRetryChannel::Peer)
                .unwrap()
                .completed,
            4
        );

        let accounting = owner
            .retry_episodes
            .accounting_mut(FreshRetryChannel::Peer)
            .unwrap();
        accounting.completed = 0;
        accounting.deadline = Instant::now() - Duration::from_millis(1);
        owner.runtime.apply_snapshot(snapshot(4, 4, "stream", 7));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .is_none());

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
        assert!(owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .is_none());

        owner.runtime.apply_snapshot(snapshot(8, 6, "stream", 8));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .is_some());

        owner.runtime.apply_snapshot(snapshot(9, 7, "final", 9));
        owner.reconcile_fresh_schedules(&logger).await.unwrap();
        let final_schedule = owner
            .retry_episodes
            .schedule(FreshRetryChannel::Peer)
            .unwrap();
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

        let logger = OverlayLogger::open(std::env::temp_dir()).await.unwrap();
        let mut none = NativePresentationOwner::new_with_profile(
            snapshot(1, 1),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
            crate::manifest::QuietTailProfile::NoRetry,
        );
        none.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(none
            .retry_episodes
            .schedule(FreshRetryChannel::SelfChannel)
            .is_none());

        let mut one = NativePresentationOwner::new_with_profile(
            snapshot(1, 1),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
            crate::manifest::QuietTailProfile::OneRetry,
        );
        one.reconcile_fresh_schedules(&logger).await.unwrap();
        let scheduled = one
            .retry_episodes
            .schedule(FreshRetryChannel::SelfChannel)
            .unwrap()
            .clone();
        assert_eq!(scheduled.max_completed, 1);
        assert!(!scheduled.expired_at(scheduled.next_due + Duration::from_millis(50)));
        assert_eq!(
            one.active_fresh_schedules(scheduled.next_due + Duration::from_millis(50))
                .len(),
            1
        );
        one.retry_episodes
            .accounting_mut(FreshRetryChannel::SelfChannel)
            .unwrap()
            .completed = 1;
        one.retry_episodes
            .take_schedule(FreshRetryChannel::SelfChannel);
        one.runtime.apply_snapshot(snapshot(2, 2));
        one.reconcile_fresh_schedules(&logger).await.unwrap();
        assert!(one
            .retry_episodes
            .schedule(FreshRetryChannel::SelfChannel)
            .is_none());
        assert_eq!(
            one.retry_episodes
                .accounting(FreshRetryChannel::SelfChannel)
                .unwrap()
                .completed,
            1
        );
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
        let due_self = schedule(FreshRetryChannel::SelfChannel, 3, 20, now);
        let staggered_peer = schedule(FreshRetryChannel::Peer, 4, 20, now);
        let retry_correlation =
            correlation_for(&due_self, 20, PresentationCauseKind::NativeFreshRetry);
        assert!(
            NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                retry_correlation,
                &due_self,
                &due_self,
                PresentationCauseKind::NativeFreshRetry,
                Some(3),
                Some("[\"target\"]"),
            )
        );
        assert!(
            !NativePresentationOwner::<FakeOpenVr>::submission_covers_schedule(
                retry_correlation,
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
    use std::io::Write;
    use std::time::Duration;
    use tokio::time::Instant;

    #[derive(Clone, Copy)]
    enum ControlledSinkMode {
        Success,
        Error,
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

        async fn wait_for_text(&self, needle: &str) {
            tokio::time::timeout(Duration::from_millis(100), async {
                while !String::from_utf8_lossy(&self.contents()).contains(needle) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
    }

    impl Write for ControlledSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            match self.mode {
                ControlledSinkMode::Success => {
                    self.bytes.lock().unwrap().extend_from_slice(bytes);
                    Ok(bytes.len())
                }
                ControlledSinkMode::Error => Err(io::Error::other("sink failed")),
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            match self.mode {
                ControlledSinkMode::Success => Ok(()),
                ControlledSinkMode::Error => Err(io::Error::other("sink failed")),
            }
        }
    }

    fn controlled_logger(stdout: ControlledSink) -> OverlayLogger {
        OverlayLogger::from_streams(
            Box::new(stdout),
            Box::new(ControlledSink::new(ControlledSinkMode::Success)),
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
            ..Default::default()
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
            ..Default::default()
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
            self.state.operations.lock().unwrap().push(
                if frame.layout().visible_blocks.is_empty() {
                    "submit:empty"
                } else {
                    "submit:text"
                },
            );
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

    struct HideFailureProbe;

    impl OverlayFrameSubmitter for HideFailureProbe {
        fn submit_frame(&mut self, _frame: &RenderedFrame) -> Result<(), OpenVrError> {
            Ok(())
        }

        fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
            if visible {
                Ok(())
            } else {
                Err(OpenVrError::Submit("cleanup hide failed".into()))
            }
        }
    }

    fn controlled_manifest(address: std::net::SocketAddr) -> OverlayManifest {
        OverlayManifest {
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
        }
    }

    async fn wait_for_owner_ready(
        ws: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    ) {
        ws.send(Message::Text(
            json!({
                "type": "health_challenge",
                "challenge_id": 1,
                "overlay_instance_id": "spatial-runtime-unit",
                "runtime_generation": 1
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        let mut ready = false;
        let mut healthy = false;
        while !ready || !healthy {
            let message = ws.next().await.unwrap().unwrap();
            let payload: serde_json::Value =
                serde_json::from_str(message.to_text().unwrap()).unwrap();
            ready |= payload["type"] == "overlay_ready";
            healthy |= payload["type"] == "owner_status" && payload["health_challenge_id"] == 1;
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
        let manifest = controlled_manifest(address);
        let (bridge, _) = BridgeClient::connect(&manifest).await.unwrap();
        (bridge, server)
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
            let first = json!({
                "revision": 1,
                "blocks": [block("self:flood-1", "self", "first", "", true)]
            });
            ws.send(Message::Text(
                json!({"type":"snapshot","payload":first})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
            wait_for_owner_ready(&mut ws).await;
            let second = json!({
                "revision": 2,
                "blocks": [block("self:flood-2", "self", "second", "", true)]
            });
            ws.send(Message::Text(
                json!({"type":"snapshot","payload":second})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
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
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("second snapshot submit starved by OpenVR event flood");
            ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let manifest = controlled_manifest(address);
        let (mut bridge, snapshot) = BridgeClient::connect(&manifest).await.unwrap();
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(stdout.clone());
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

        owner.run(&mut bridge, &logger).await.unwrap();

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
        assert!(String::from_utf8(stdout.contents())
            .unwrap()
            .contains("\"type\":\"shutdown_complete\""));
        logger.shutdown().unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn orderly_owner_teardown_emits_shutdown_complete_after_resource_release() {
        let (mut bridge, server) = controlled_test_bridge(None).await;
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(stdout.clone());
        let mut owner = NativePresentationOwner::new(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            FakeOpenVr::default(),
        );

        owner
            .finish_run(&mut bridge, &logger, Ok(()))
            .await
            .unwrap();
        assert!(owner.resources_released());
        stdout.wait_for_text("\"type\":\"shutdown_complete\"").await;
        let emitted = String::from_utf8(stdout.contents()).unwrap();
        assert!(emitted.contains("\"overlay_instance_id\":\"spatial-runtime-unit\""));
        assert!(emitted.contains("\"runtime_generation\":1"));
        logger.shutdown().unwrap();
        drop(bridge);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn failed_owner_teardown_does_not_emit_shutdown_complete() {
        let (mut bridge, server) = controlled_test_bridge(None).await;
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(stdout.clone());
        let mut owner = NativePresentationOwner::new(
            OverlayPresentationSnapshot::default(),
            CaptionRenderer::new_for_test().unwrap(),
            HideFailureProbe,
        );

        assert!(matches!(
            owner.finish_run(&mut bridge, &logger, Ok(())).await,
            Err(RuntimeFailure::OpenVr(_))
        ));
        assert!(owner.resources_released());
        tokio::task::yield_now().await;
        assert!(!String::from_utf8(stdout.contents())
            .unwrap()
            .contains("\"type\":\"shutdown_complete\""));
        logger.shutdown().unwrap();
        drop(bridge);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn spatial_diagnostic_write_failure_cannot_block_pose_unavailable_texture_submit() {
        let (mut bridge, server) = controlled_test_bridge(None).await;
        let renderer = CaptionRenderer::new_for_test().unwrap();
        let logger = controlled_logger(ControlledSink::new(ControlledSinkMode::Error));
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 1,
            calibration: OverlayPresentationCalibration {
                anchor: "spatial_locked".to_string(),
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![block("self:A", "self", "A", "", true)],
            native_fresh_render_generations: None,
            ..Default::default()
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

        assert!(result.is_ok());
        assert_eq!(submitter.operations, vec!["reanchor"]);
        assert!(runtime.redraw_requested);
        drop(bridge);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn spatial_diagnostic_write_error_does_not_block_first_visible_pose_unavailable_reveal() {
        let (mut bridge, server) = controlled_test_bridge(None).await;
        let renderer = CaptionRenderer::new_for_test().unwrap();
        let logger = controlled_logger(ControlledSink::new(ControlledSinkMode::Error));
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 1,
            calibration: OverlayPresentationCalibration {
                anchor: "spatial_locked".to_string(),
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![block("self:A", "self", "A", "", true)],
            native_fresh_render_generations: None,
            ..Default::default()
        });
        runtime.first_texture_submitted = true;
        let mut submitter = SpatialSubmitProbe {
            outcome: SpatialReanchorOutcome::PoseUnavailable,
            operations: Vec::new(),
        };

        let result = runtime
            .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
            .await;

        assert!(result.is_ok());
        assert_eq!(submitter.operations, vec!["reanchor"]);
        assert!(runtime.redraw_requested);
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
            ..Default::default()
        };
        let (mut bridge, server) =
            controlled_test_bridge(Some((readiness_started.clone(), latest_snapshot))).await;
        let renderer = CaptionRenderer::new_for_test().unwrap();
        renderer.set_test_readiness_pending_yields_on_call(1, usize::MAX);
        renderer.set_test_readiness_started_notify_on_call(1, readiness_started);
        let failing_logger = controlled_logger(ControlledSink::new(ControlledSinkMode::Error));
        let healthy_logger = controlled_logger(ControlledSink::new(ControlledSinkMode::Success));
        let calibration = OverlayPresentationCalibration {
            anchor: "spatial_locked".to_string(),
            ..OverlayPresentationCalibration::default()
        };
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            revision: 1,
            calibration: calibration.clone(),
            blocks: vec![block("self:A", "self", "A", "", true)],
            native_fresh_render_generations: None,
            ..Default::default()
        });
        runtime.first_texture_submitted = true;
        runtime.overlay_visible = true;
        runtime.redraw_requested = false;
        runtime.spatial_lock.complete_pending();
        runtime.pending_spatial_diagnostics.clear();
        runtime.apply_snapshot(OverlayPresentationSnapshot {
            revision: 2,
            calibration,
            blocks: vec![
                block("self:A", "self", "A", "", true),
                block("self:B", "self", "B", "", true),
            ],
            native_fresh_render_generations: None,
            ..Default::default()
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
            ..Default::default()
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
    fn caption_blocks_follow_snapshot_order_exactly() {
        let runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 3,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![
                block("peer:1", "peer", "peer one", "원문", true),
                block("self:2", "self", "self two", "translated", true),
            ],
            ..Default::default()
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
    fn apply_snapshot_replaces_snapshot_blocks_and_calibration_without_retaining_removed_rows() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 1,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("self:1", "self", "self one", "", true)],
            ..Default::default()
        });

        runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 2,
            calibration: OverlayPresentationCalibration {
                distance: 1.5,
                ..OverlayPresentationCalibration::default()
            },
            blocks: vec![block("peer:2", "peer", "peer two", "", true)],
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
        });

        let blocks = runtime.caption_blocks();

        assert_eq!(blocks[0].id, "peer:active");
        assert_eq!(blocks[0].block_variant, CaptionBlockVariant::ActivePeer);
        assert_eq!(blocks[0].channel, Some(CaptionChannel::PeerChannel));
        assert_eq!(blocks[0].primary_text, "");
        assert_eq!(blocks[0].secondary_text, "Can you hear me?");
        assert!(blocks[0].secondary_enabled);
    }

    #[tokio::test]
    async fn renderer_degradation_warning_is_bounded_to_changed_failure_episode() {
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(stdout.clone());
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        let degraded = RenderDiagnostics {
            heuristic_layout_fallback_count: 1,
            style_bucket_source_counts: vec![StyleBucketSourceCount {
                bucket: FontLanguageBucket::General,
                source: FontSource::SystemFallbackSentinel,
                count: 1,
            }],
            ..RenderDiagnostics::default()
        };

        for diagnostics in [
            &degraded,
            &degraded,
            &RenderDiagnostics::default(),
            &degraded,
        ] {
            runtime
                .emit_renderer_degradation_if_changed(&logger, diagnostics)
                .await
                .unwrap();
        }
        logger.warn("drain_marker").await.unwrap();
        stdout.wait_for_text("drain_marker").await;
        logger.shutdown().unwrap();
        let output = String::from_utf8(stdout.contents()).unwrap();
        assert_eq!(output.matches("renderer_degradation").count(), 2);
    }

    #[tokio::test]
    async fn font_fallback_line_changes_do_not_repeat_startup_warnings() {
        let stdout = ControlledSink::new(ControlledSinkMode::Success);
        let logger = controlled_logger(stdout.clone());
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot::default());
        for count in [1, 2, 0, 3] {
            let diagnostics = RenderDiagnostics {
                font_warmup_failures: 1,
                style_bucket_source_counts: vec![StyleBucketSourceCount {
                    bucket: FontLanguageBucket::CjkKo,
                    source: FontSource::SystemFont,
                    count,
                }],
                ..RenderDiagnostics::default()
            };
            runtime
                .emit_renderer_degradation_if_changed(&logger, &diagnostics)
                .await
                .unwrap();
        }
        logger.warn("drain_marker").await.unwrap();
        stdout.wait_for_text("drain_marker").await;
        logger.shutdown().unwrap();
        assert_eq!(
            String::from_utf8(stdout.contents()).unwrap(),
            "[overlay][WARN] drain_marker\n",
        );
    }

    #[test]
    fn prepare_openvr_runtime_orders_preflight_before_overlay_factory() {
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
    fn runtime_apply_snapshot_reports_ignored_revisions_without_redraw() {
        let mut runtime = OverlayRuntime::new(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 3,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("self:1", "self", "hello", "", true)],
            ..Default::default()
        });
        runtime.clear_redraw_flag();

        let outcome = runtime.apply_snapshot(OverlayPresentationSnapshot {
            native_fresh_render_generations: None,
            revision: 2,
            calibration: OverlayPresentationCalibration::default(),
            blocks: vec![block("peer:2", "peer", "ignored", "", true)],
            ..Default::default()
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
