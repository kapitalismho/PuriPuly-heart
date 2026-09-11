use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use serde::Serialize;

const MAX_PRESENTATION_DIAGNOSTIC_RECORDS: usize = 128;
const MAX_PENDING_PRESENTATION_DIAGNOSTIC_RECORDS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationStage {
    LogicalRevisionAccepted,
    LeaseAdmission,
    LeasePending,
    LeaseExpired,
    RenderReturned,
    ReadinessObserved,
    SubmissionAttempted,
    SubmissionReturned,
    VisibilityRequested,
    VisibilityObserved,
    CompositorObserved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationOutcome {
    Accepted,
    Success,
    Failure,
    LegacyNotObserved,
    Attempted,
    Ready,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationStrategy {
    LegacyDirectTextureSubmit,
    BoundedGpuCompletion,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffMode {
    #[default]
    Off,
    CachedFrameRehandoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationBackend {
    D3d11Hardware,
    D3d11Warp,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterIdentity {
    NotObservedStageOne,
    Unavailable,
    DxgiLuid { high: i32, low: u32 },
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterMatch {
    Match,
    Mismatch,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessOutcome {
    Ready,
    TimedOut,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct ReadinessCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ReadinessCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalHmdVisibility {
    NotObservable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationCauseKind {
    Startup,
    SceneUpdate,
    RuntimeControl,
    ExternalRetry,
    ActiveRetryIntent,
    NativeFreshRetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationCauseChannel {
    SelfChannel,
    Peer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PresentationCause {
    pub kind: PresentationCauseKind,
    pub channel: Option<PresentationCauseChannel>,
    pub trigger_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PresentationCauses {
    startup: bool,
    scene_update_generation: Option<u64>,
    runtime_control: bool,
    external_retry: bool,
    active_self_retry_generation: Option<u64>,
    active_peer_retry_generation: Option<u64>,
    self_retry_generation: Option<u64>,
    peer_retry_generation: Option<u64>,
}

impl PresentationCauses {
    pub fn insert(&mut self, cause: PresentationCause) {
        match (cause.kind, cause.channel) {
            (PresentationCauseKind::Startup, _) => self.startup = true,
            (PresentationCauseKind::SceneUpdate, _) => {
                self.scene_update_generation = cause.trigger_generation
            }
            (PresentationCauseKind::RuntimeControl, _) => self.runtime_control = true,
            (PresentationCauseKind::ExternalRetry, _) => self.external_retry = true,
            (
                PresentationCauseKind::ActiveRetryIntent,
                Some(PresentationCauseChannel::SelfChannel),
            ) => self.active_self_retry_generation = cause.trigger_generation,
            (PresentationCauseKind::ActiveRetryIntent, Some(PresentationCauseChannel::Peer)) => {
                self.active_peer_retry_generation = cause.trigger_generation
            }
            (PresentationCauseKind::ActiveRetryIntent, None) => {}
            (
                PresentationCauseKind::NativeFreshRetry,
                Some(PresentationCauseChannel::SelfChannel),
            ) => self.self_retry_generation = cause.trigger_generation,
            (PresentationCauseKind::NativeFreshRetry, Some(PresentationCauseChannel::Peer)) => {
                self.peer_retry_generation = cause.trigger_generation
            }
            (PresentationCauseKind::NativeFreshRetry, None) => {}
        }
    }

    pub fn merge(&mut self, other: Self) {
        for cause in other.to_vec() {
            self.insert(cause);
        }
    }

    pub fn remove(&mut self, cause: PresentationCause) {
        match (cause.kind, cause.channel) {
            (
                PresentationCauseKind::ActiveRetryIntent,
                Some(PresentationCauseChannel::SelfChannel),
            ) if self.active_self_retry_generation == cause.trigger_generation => {
                self.active_self_retry_generation = None
            }
            (PresentationCauseKind::ActiveRetryIntent, Some(PresentationCauseChannel::Peer))
                if self.active_peer_retry_generation == cause.trigger_generation =>
            {
                self.active_peer_retry_generation = None
            }
            _ => {}
        }
    }

    pub fn contains(&self, cause: PresentationCause) -> bool {
        match (cause.kind, cause.channel) {
            (
                PresentationCauseKind::ActiveRetryIntent,
                Some(PresentationCauseChannel::SelfChannel),
            ) => self.active_self_retry_generation == cause.trigger_generation,
            (PresentationCauseKind::ActiveRetryIntent, Some(PresentationCauseChannel::Peer)) => {
                self.active_peer_retry_generation == cause.trigger_generation
            }
            (
                PresentationCauseKind::NativeFreshRetry,
                Some(PresentationCauseChannel::SelfChannel),
            ) => self.self_retry_generation == cause.trigger_generation,
            (PresentationCauseKind::NativeFreshRetry, Some(PresentationCauseChannel::Peer)) => {
                self.peer_retry_generation == cause.trigger_generation
            }
            _ => false,
        }
    }

    pub fn to_vec(self) -> Vec<PresentationCause> {
        let mut causes = Vec::with_capacity(8);
        for (present, kind) in [
            (self.startup, PresentationCauseKind::Startup),
            (self.runtime_control, PresentationCauseKind::RuntimeControl),
            (self.external_retry, PresentationCauseKind::ExternalRetry),
        ] {
            if present {
                causes.push(PresentationCause {
                    kind,
                    channel: None,
                    trigger_generation: None,
                });
            }
        }
        if let Some(trigger_generation) = self.scene_update_generation {
            causes.push(PresentationCause {
                kind: PresentationCauseKind::SceneUpdate,
                channel: None,
                trigger_generation: Some(trigger_generation),
            });
        }
        for (channel, trigger_generation) in [
            (
                PresentationCauseChannel::SelfChannel,
                self.active_self_retry_generation,
            ),
            (
                PresentationCauseChannel::Peer,
                self.active_peer_retry_generation,
            ),
        ] {
            if let Some(trigger_generation) = trigger_generation {
                causes.push(PresentationCause {
                    kind: PresentationCauseKind::ActiveRetryIntent,
                    channel: Some(channel),
                    trigger_generation: Some(trigger_generation),
                });
            }
        }
        for (channel, trigger_generation) in [
            (
                PresentationCauseChannel::SelfChannel,
                self.self_retry_generation,
            ),
            (PresentationCauseChannel::Peer, self.peer_retry_generation),
        ] {
            if let Some(trigger_generation) = trigger_generation {
                causes.push(PresentationCause {
                    kind: PresentationCauseKind::NativeFreshRetry,
                    channel: Some(channel),
                    trigger_generation: Some(trigger_generation),
                });
            }
        }
        causes
    }

    pub fn is_native_fresh_retry_only(self) -> bool {
        let causes = self.to_vec();
        !causes.is_empty()
            && causes
                .iter()
                .all(|cause| cause.kind == PresentationCauseKind::NativeFreshRetry)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositorAttribution {
    NotOverlayAttributable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PresentationDiagnosticRecord {
    pub schema_version: u8,
    pub sequence: u64,
    pub logical_revision: u64,
    pub scene_generation: u64,
    pub logical_causes: Vec<PresentationCause>,
    pub render_generation: Option<u64>,
    pub submission_attempt: Option<u64>,
    pub stage: PresentationStage,
    pub outcome: PresentationOutcome,
    pub strategy: PresentationStrategy,
    pub backend: PresentationBackend,
    pub adapter_identity: AdapterIdentity,
    pub openvr_adapter_identity: AdapterIdentity,
    pub renderer_adapter_identity: AdapterIdentity,
    pub adapter_match: AdapterMatch,
    pub desired_visible: Option<bool>,
    pub observed_runtime_visible: Option<bool>,
    pub physical_hmd_visibility: PhysicalHmdVisibility,
    pub cpu_prepare_us: Option<u64>,
    pub cpu_render_us: Option<u64>,
    pub readiness_us: Option<u64>,
    pub submission_return_us: Option<u64>,
    pub compositor_frame_index: Option<u32>,
    pub compositor_dropped_frames: Option<u32>,
    pub compositor_mis_presented: Option<u32>,
    pub compositor_render_cpu_us: Option<u64>,
    pub compositor_total_render_gpu_us: Option<u64>,
    pub compositor_post_submit_gpu_us: Option<u64>,
    pub compositor_attribution: Option<CompositorAttribution>,
    pub build_version: &'static str,
    pub build_profile: &'static str,
    pub target_os: &'static str,
    pub retry_profile: &'static str,
    pub candidate_build_identity: &'static str,
    pub environment_identity: &'static str,
    pub baseline_checkpoint_identity: &'static str,
    pub manual_hmd_observation: &'static str,
    pub dropped_unacknowledged_records: u64,
    pub observed_at_ms: u64,
    pub reason: &'static str,
    pub lease_disposition: &'static str,
    pub handoff_mode: HandoffMode,
    pub content_identity: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationCorrelation {
    pub logical_revision: u64,
    pub render_generation: u64,
    pub submission_attempt: u64,
    pub scene_generation: u64,
    pub logical_causes: PresentationCauses,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingPresentationDiagnostics {
    pub records: Vec<String>,
    pub through_sequence: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationDiagnostics {
    records: VecDeque<PresentationDiagnosticRecord>,
    acknowledged_through_sequence: u64,
    next_sequence: u64,
    next_logical_revision: u64,
    next_render_generation: u64,
    next_submission_attempt: u64,
    active_logical_revision: u64,
    active_scene_generation: u64,
    active_logical_causes: PresentationCauses,
    stopped: bool,
    dropped_unacknowledged_records: u64,
    strategy: PresentationStrategy,
    openvr_adapter_identity: AdapterIdentity,
    renderer_adapter_identity: AdapterIdentity,
    adapter_match: AdapterMatch,
    retry_profile: &'static str,
    observed_origin: Instant,
    reason: &'static str,
    lease_disposition: &'static str,
    handoff_mode: HandoffMode,
    content_identity: Option<u64>,
}

impl PresentationDiagnostics {
    pub fn new() -> Self {
        Self {
            records: VecDeque::with_capacity(MAX_PRESENTATION_DIAGNOSTIC_RECORDS),
            acknowledged_through_sequence: 0,
            next_sequence: 1,
            next_logical_revision: 1,
            next_render_generation: 1,
            next_submission_attempt: 1,
            active_logical_revision: 0,
            active_scene_generation: 0,
            active_logical_causes: PresentationCauses::default(),
            stopped: false,
            dropped_unacknowledged_records: 0,
            strategy: PresentationStrategy::LegacyDirectTextureSubmit,
            openvr_adapter_identity: AdapterIdentity::NotObservedStageOne,
            renderer_adapter_identity: AdapterIdentity::NotObservedStageOne,
            adapter_match: AdapterMatch::Unavailable,
            retry_profile: "p05",
            observed_origin: Instant::now(),
            reason: "unspecified",
            lease_disposition: "not_applicable",
            handoff_mode: HandoffMode::Off,
            content_identity: None,
        }
    }

    pub fn configure_retry_profile(&mut self, retry_profile: &'static str) {
        self.retry_profile = retry_profile;
    }

    pub fn configure_adapter_handoff(
        &mut self,
        openvr_adapter_identity: AdapterIdentity,
        renderer_adapter_identity: AdapterIdentity,
        adapter_match: AdapterMatch,
    ) {
        self.strategy = PresentationStrategy::BoundedGpuCompletion;
        self.openvr_adapter_identity = openvr_adapter_identity;
        self.renderer_adapter_identity = renderer_adapter_identity;
        self.adapter_match = adapter_match;
    }

    pub fn configure_event_metadata(
        &mut self,
        reason: &'static str,
        lease_disposition: &'static str,
        handoff_mode: HandoffMode,
        content_identity: Option<u64>,
    ) {
        self.reason = reason;
        self.lease_disposition = lease_disposition;
        self.handoff_mode = handoff_mode;
        self.content_identity = content_identity;
    }

    pub fn record_lease_event(
        &mut self,
        stage: PresentationStage,
        outcome: PresentationOutcome,
        backend: PresentationBackend,
        scene_generation: u64,
    ) {
        self.push(
            stage,
            outcome,
            backend,
            None,
            None,
            None,
            None,
            self.active_logical_revision,
            scene_generation,
            PresentationCauses::default(),
        );
    }

    pub fn accept_logical_revision(
        &mut self,
        backend: PresentationBackend,
        scene_generation: u64,
        logical_causes: PresentationCauses,
    ) {
        if self.stopped {
            return;
        }
        self.active_logical_revision = self.next_logical_revision;
        self.next_logical_revision = self.next_logical_revision.saturating_add(1);
        self.active_scene_generation = scene_generation;
        self.active_logical_causes = logical_causes;
        self.push(
            PresentationStage::LogicalRevisionAccepted,
            PresentationOutcome::Accepted,
            backend,
            None,
            None,
            None,
            None,
            self.active_logical_revision,
            scene_generation,
            logical_causes,
        );
    }

    pub fn begin_presentation(
        &mut self,
        scene_generation: u64,
        logical_causes: PresentationCauses,
    ) -> Option<PresentationCorrelation> {
        if self.stopped {
            return None;
        }
        self.active_scene_generation = scene_generation;
        self.active_logical_causes = logical_causes;
        let correlation = PresentationCorrelation {
            logical_revision: self.active_logical_revision,
            render_generation: self.next_render_generation,
            submission_attempt: self.next_submission_attempt,
            scene_generation,
            logical_causes,
        };
        self.next_render_generation = self.next_render_generation.saturating_add(1);
        self.next_submission_attempt = self.next_submission_attempt.saturating_add(1);
        Some(correlation)
    }

    pub fn begin_rehandoff(
        &mut self,
        scene_generation: u64,
        logical_causes: PresentationCauses,
        retained_render_generation: u64,
    ) -> Option<PresentationCorrelation> {
        if self.stopped {
            return None;
        }
        self.active_scene_generation = scene_generation;
        self.active_logical_causes = logical_causes;
        let correlation = PresentationCorrelation {
            logical_revision: self.active_logical_revision,
            render_generation: retained_render_generation,
            submission_attempt: self.next_submission_attempt,
            scene_generation,
            logical_causes,
        };
        self.next_submission_attempt = self.next_submission_attempt.saturating_add(1);
        Some(correlation)
    }

    pub fn record_render_return(
        &mut self,
        correlation: PresentationCorrelation,
        backend: PresentationBackend,
        succeeded: bool,
        cpu_prepare_us: u64,
        cpu_render_us: u64,
    ) {
        self.push_for_correlation(
            correlation,
            PresentationStage::RenderReturned,
            if succeeded {
                PresentationOutcome::Success
            } else {
                PresentationOutcome::Failure
            },
            backend,
            None,
            None,
        );
        if let Some(record) = self.records.back_mut() {
            record.cpu_prepare_us = Some(cpu_prepare_us);
            record.cpu_render_us = Some(cpu_render_us);
        }
    }

    pub fn record_readiness(
        &mut self,
        correlation: PresentationCorrelation,
        backend: PresentationBackend,
        outcome: ReadinessOutcome,
        readiness_us: u64,
    ) {
        self.push_for_correlation(
            correlation,
            PresentationStage::ReadinessObserved,
            match outcome {
                ReadinessOutcome::Ready => PresentationOutcome::Ready,
                ReadinessOutcome::TimedOut => PresentationOutcome::TimedOut,
                ReadinessOutcome::Cancelled => PresentationOutcome::Cancelled,
                ReadinessOutcome::Failed => PresentationOutcome::Failure,
            },
            backend,
            None,
            None,
        );
        if let Some(record) = self.records.back_mut() {
            record.readiness_us = Some(readiness_us);
        }
    }

    pub fn record_submission_attempt(
        &mut self,
        correlation: PresentationCorrelation,
        backend: PresentationBackend,
    ) {
        self.push_for_correlation(
            correlation,
            PresentationStage::SubmissionAttempted,
            PresentationOutcome::Attempted,
            backend,
            None,
            None,
        );
    }

    pub fn record_submission_return(
        &mut self,
        correlation: PresentationCorrelation,
        backend: PresentationBackend,
        succeeded: bool,
        submission_return_us: u64,
    ) {
        self.push_for_correlation(
            correlation,
            PresentationStage::SubmissionReturned,
            if succeeded {
                PresentationOutcome::Success
            } else {
                PresentationOutcome::Failure
            },
            backend,
            None,
            None,
        );
        if let Some(record) = self.records.back_mut() {
            record.submission_return_us = Some(submission_return_us);
        }
    }

    pub fn record_visibility_request(
        &mut self,
        correlation: PresentationCorrelation,
        backend: PresentationBackend,
        desired_visible: bool,
        succeeded: bool,
    ) {
        self.push_for_correlation(
            correlation,
            PresentationStage::VisibilityRequested,
            if succeeded {
                PresentationOutcome::Success
            } else {
                PresentationOutcome::Failure
            },
            backend,
            Some(desired_visible),
            None,
        );
    }

    pub fn record_compositor_observation(
        &mut self,
        backend: PresentationBackend,
        frame_index: u32,
        dropped_frames: u32,
        mis_presented: u32,
        render_cpu_us: Option<u64>,
        total_render_gpu_us: Option<u64>,
        post_submit_gpu_us: Option<u64>,
    ) {
        self.push(
            PresentationStage::CompositorObserved,
            PresentationOutcome::Success,
            backend,
            None,
            None,
            None,
            None,
            self.active_logical_revision,
            self.active_scene_generation,
            PresentationCauses::default(),
        );
        if let Some(record) = self.records.back_mut() {
            record.compositor_frame_index = Some(frame_index);
            record.compositor_dropped_frames = Some(dropped_frames);
            record.compositor_mis_presented = Some(mis_presented);
            record.compositor_render_cpu_us = render_cpu_us;
            record.compositor_total_render_gpu_us = total_render_gpu_us;
            record.compositor_post_submit_gpu_us = post_submit_gpu_us;
            record.compositor_attribution = Some(CompositorAttribution::NotOverlayAttributable);
        }
    }

    pub fn record_visibility(
        &mut self,
        correlation: PresentationCorrelation,
        backend: PresentationBackend,
        desired_visible: bool,
        observed_runtime_visible: bool,
        succeeded: bool,
    ) {
        self.push_for_correlation(
            correlation,
            PresentationStage::VisibilityObserved,
            if succeeded {
                PresentationOutcome::Success
            } else {
                PresentationOutcome::Failure
            },
            backend,
            Some(desired_visible),
            Some(observed_runtime_visible),
        );
    }

    pub fn pending_json(&self) -> Vec<String> {
        self.pending_batch().records
    }

    pub fn pending_batch(&self) -> PendingPresentationDiagnostics {
        if self.stopped {
            return PendingPresentationDiagnostics {
                records: Vec::new(),
                through_sequence: None,
            };
        }
        let pending_records = self
            .records
            .iter()
            .filter(|record| record.sequence > self.acknowledged_through_sequence)
            .take(MAX_PENDING_PRESENTATION_DIAGNOSTIC_RECORDS)
            .collect::<Vec<_>>();
        let through_sequence = pending_records.last().map(|record| record.sequence);
        let records = pending_records
            .into_iter()
            .filter_map(|record| serde_json::to_string(record).ok())
            .collect();
        PendingPresentationDiagnostics {
            records,
            through_sequence,
        }
    }

    pub fn acknowledge_through(&mut self, sequence: u64) {
        self.acknowledged_through_sequence = self
            .acknowledged_through_sequence
            .max(sequence.min(self.next_sequence.saturating_sub(1)));
    }

    pub fn records(&self) -> &VecDeque<PresentationDiagnosticRecord> {
        &self.records
    }

    pub fn shutdown(&mut self) {
        self.records.clear();
        self.stopped = true;
    }

    fn push_for_correlation(
        &mut self,
        correlation: PresentationCorrelation,
        stage: PresentationStage,
        outcome: PresentationOutcome,
        backend: PresentationBackend,
        desired_visible: Option<bool>,
        observed_runtime_visible: Option<bool>,
    ) {
        self.push(
            stage,
            outcome,
            backend,
            Some(correlation.render_generation),
            Some(correlation.submission_attempt),
            desired_visible,
            observed_runtime_visible,
            correlation.logical_revision,
            correlation.scene_generation,
            correlation.logical_causes,
        );
    }

    fn push(
        &mut self,
        stage: PresentationStage,
        outcome: PresentationOutcome,
        backend: PresentationBackend,
        render_generation: Option<u64>,
        submission_attempt: Option<u64>,
        desired_visible: Option<bool>,
        observed_runtime_visible: Option<bool>,
        logical_revision: u64,
        scene_generation: u64,
        logical_causes: PresentationCauses,
    ) {
        if self.stopped {
            return;
        }
        if self.records.len() == MAX_PRESENTATION_DIAGNOSTIC_RECORDS {
            if self
                .records
                .front()
                .is_some_and(|record| record.sequence > self.acknowledged_through_sequence)
            {
                self.dropped_unacknowledged_records =
                    self.dropped_unacknowledged_records.saturating_add(1);
            }
            self.records.pop_front();
        }
        self.records.push_back(PresentationDiagnosticRecord {
            schema_version: 1,
            sequence: self.next_sequence,
            logical_revision,
            scene_generation,
            logical_causes: logical_causes.to_vec(),
            render_generation,
            submission_attempt,
            stage,
            outcome,
            strategy: self.strategy,
            backend,
            adapter_identity: self.renderer_adapter_identity,
            openvr_adapter_identity: self.openvr_adapter_identity,
            renderer_adapter_identity: self.renderer_adapter_identity,
            adapter_match: self.adapter_match,
            desired_visible,
            observed_runtime_visible,
            physical_hmd_visibility: PhysicalHmdVisibility::NotObservable,
            cpu_prepare_us: None,
            cpu_render_us: None,
            readiness_us: None,
            submission_return_us: None,
            compositor_frame_index: None,
            compositor_dropped_frames: None,
            compositor_mis_presented: None,
            compositor_render_cpu_us: None,
            compositor_total_render_gpu_us: None,
            compositor_post_submit_gpu_us: None,
            compositor_attribution: None,
            build_version: env!("CARGO_PKG_VERSION"),
            build_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            target_os: std::env::consts::OS,
            retry_profile: self.retry_profile,
            candidate_build_identity: allowlisted_build_identity(option_env!(
                "PURIPULY_OVERLAY_BUILD_ID"
            )),
            environment_identity: "not_recorded",
            baseline_checkpoint_identity: "92eff4229021189b7e9a82288cfc4eb6d260e838",
            manual_hmd_observation: "not_recorded",
            dropped_unacknowledged_records: self.dropped_unacknowledged_records,
            observed_at_ms: u64::try_from(self.observed_origin.elapsed().as_millis())
                .unwrap_or(u64::MAX),
            reason: self.reason,
            lease_disposition: self.lease_disposition,
            handoff_mode: self.handoff_mode,
            content_identity: self.content_identity,
        });
        self.next_sequence = self.next_sequence.saturating_add(1);
    }
}

fn allowlisted_build_identity(identity: Option<&'static str>) -> &'static str {
    identity
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        })
        .unwrap_or("not_recorded")
}

impl Default for PresentationDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_bounded_and_shutdown_removes_owned_work() {
        let mut diagnostics = PresentationDiagnostics::new();
        diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            1,
            PresentationCauses::default(),
        );
        assert!(diagnostics
            .records()
            .iter()
            .all(|record| record.retry_profile == "p05"));
        for _ in 0..200 {
            diagnostics.accept_logical_revision(
                PresentationBackend::Test,
                0,
                PresentationCauses::default(),
            );
        }

        assert_eq!(
            diagnostics.records().len(),
            MAX_PRESENTATION_DIAGNOSTIC_RECORDS
        );
        assert_eq!(
            diagnostics.pending_json().len(),
            MAX_PENDING_PRESENTATION_DIAGNOSTIC_RECORDS
        );

        diagnostics.shutdown();
        diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            7,
            PresentationCauses::default(),
        );

        assert!(diagnostics.records().is_empty());
        assert!(diagnostics.pending_json().is_empty());
    }

    #[test]
    fn serialized_records_are_structurally_allowlisted() {
        let mut diagnostics = PresentationDiagnostics::new();
        diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            PresentationCauses::default(),
        );
        let correlation = diagnostics
            .begin_presentation(7, PresentationCauses::default())
            .unwrap();
        diagnostics.record_render_return(correlation, PresentationBackend::Test, true, 1, 2);
        diagnostics.record_readiness(
            correlation,
            PresentationBackend::Test,
            ReadinessOutcome::Ready,
            3,
        );
        diagnostics.record_submission_attempt(correlation, PresentationBackend::Test);
        diagnostics.record_submission_return(correlation, PresentationBackend::Test, true, 4);
        diagnostics.record_visibility(correlation, PresentationBackend::Test, true, true, true);

        let allowed = [
            "schema_version",
            "sequence",
            "logical_revision",
            "scene_generation",
            "logical_causes",
            "render_generation",
            "submission_attempt",
            "stage",
            "outcome",
            "strategy",
            "backend",
            "adapter_identity",
            "openvr_adapter_identity",
            "renderer_adapter_identity",
            "adapter_match",
            "desired_visible",
            "observed_runtime_visible",
            "physical_hmd_visibility",
            "cpu_prepare_us",
            "cpu_render_us",
            "readiness_us",
            "submission_return_us",
            "compositor_frame_index",
            "compositor_dropped_frames",
            "compositor_mis_presented",
            "compositor_render_cpu_us",
            "compositor_total_render_gpu_us",
            "compositor_post_submit_gpu_us",
            "compositor_attribution",
            "build_version",
            "build_profile",
            "target_os",
            "retry_profile",
            "candidate_build_identity",
            "environment_identity",
            "baseline_checkpoint_identity",
            "manual_hmd_observation",
            "dropped_unacknowledged_records",
            "observed_at_ms",
            "reason",
            "lease_disposition",
            "handoff_mode",
            "content_identity",
        ];
        for line in diagnostics.pending_json() {
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            let object = value.as_object().unwrap();
            assert!(object.keys().all(|key| allowed.contains(&key.as_str())));
            assert_eq!(object["physical_hmd_visibility"], "not_observable");
            assert!(!line.contains("caption"));
            assert!(!line.contains("exception"));
            assert!(!line.contains("stack"));
        }
    }

    #[test]
    fn pending_records_require_explicit_acknowledgement() {
        let mut diagnostics = PresentationDiagnostics::new();
        diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            0,
            PresentationCauses::default(),
        );

        let first = diagnostics.pending_json();
        let repeated = diagnostics.pending_json();
        assert_eq!(first, repeated);

        diagnostics.acknowledge_through(1);

        assert!(diagnostics.pending_json().is_empty());
    }

    #[test]
    fn bounded_overflow_accounts_for_unacknowledged_records() {
        let mut diagnostics = PresentationDiagnostics::new();
        for _ in 0..=MAX_PRESENTATION_DIAGNOSTIC_RECORDS {
            diagnostics.accept_logical_revision(
                PresentationBackend::Test,
                0,
                PresentationCauses::default(),
            );
        }

        assert_eq!(
            diagnostics
                .records()
                .back()
                .unwrap()
                .dropped_unacknowledged_records,
            1
        );
    }

    #[test]
    fn adapter_handoff_and_readiness_failures_are_explicit_without_submission_success() {
        let outcomes = [
            (ReadinessOutcome::TimedOut, PresentationOutcome::TimedOut),
            (ReadinessOutcome::Cancelled, PresentationOutcome::Cancelled),
            (ReadinessOutcome::Failed, PresentationOutcome::Failure),
        ];
        for (readiness, expected) in outcomes {
            let mut diagnostics = PresentationDiagnostics::new();
            let openvr_adapter = AdapterIdentity::DxgiLuid { high: 7, low: 11 };
            let renderer_adapter = AdapterIdentity::DxgiLuid { high: 7, low: 12 };
            diagnostics.configure_adapter_handoff(
                openvr_adapter,
                renderer_adapter,
                AdapterMatch::Mismatch,
            );
            diagnostics.accept_logical_revision(
                PresentationBackend::D3d11Hardware,
                9,
                PresentationCauses::default(),
            );
            let correlation = diagnostics
                .begin_presentation(9, PresentationCauses::default())
                .unwrap();
            diagnostics.record_render_return(
                correlation,
                PresentationBackend::D3d11Hardware,
                true,
                1,
                2,
            );
            diagnostics.record_readiness(
                correlation,
                PresentationBackend::D3d11Hardware,
                readiness,
                3,
            );

            let record = diagnostics.records().back().unwrap();
            assert_eq!(record.stage, PresentationStage::ReadinessObserved);
            assert_eq!(record.outcome, expected);
            assert_eq!(record.strategy, PresentationStrategy::BoundedGpuCompletion);
            assert_eq!(record.openvr_adapter_identity, openvr_adapter);
            assert_eq!(record.renderer_adapter_identity, renderer_adapter);
            assert_eq!(record.adapter_match, AdapterMatch::Mismatch);
            assert!(!diagnostics
                .records()
                .iter()
                .any(|record| record.stage == PresentationStage::SubmissionReturned));
        }
    }

    #[test]
    fn causes_accumulate_deduplicate_and_keep_latest_channel_generation() {
        let mut causes = PresentationCauses::default();
        for cause in [
            PresentationCause {
                kind: PresentationCauseKind::ExternalRetry,
                channel: None,
                trigger_generation: None,
            },
            PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::SelfChannel),
                trigger_generation: Some(3),
            },
            PresentationCause {
                kind: PresentationCauseKind::ExternalRetry,
                channel: None,
                trigger_generation: None,
            },
            PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::SelfChannel),
                trigger_generation: Some(4),
            },
            PresentationCause {
                kind: PresentationCauseKind::NativeFreshRetry,
                channel: Some(PresentationCauseChannel::Peer),
                trigger_generation: Some(8),
            },
        ] {
            causes.insert(cause);
        }

        let facts = causes.to_vec();
        assert_eq!(facts.len(), 3);
        assert!(facts
            .iter()
            .any(|cause| cause.kind == PresentationCauseKind::ExternalRetry));
        assert!(facts.iter().any(|cause| cause.channel
            == Some(PresentationCauseChannel::SelfChannel)
            && cause.trigger_generation == Some(4)));
        assert!(facts.iter().any(
            |cause| cause.channel == Some(PresentationCauseChannel::Peer)
                && cause.trigger_generation == Some(8)
        ));
    }

    #[test]
    fn correlated_records_keep_immutable_attempt_identity_after_newer_activity() {
        let mut diagnostics = PresentationDiagnostics::new();
        let mut original = PresentationCauses::default();
        original.insert(PresentationCause {
            kind: PresentationCauseKind::ExternalRetry,
            channel: None,
            trigger_generation: None,
        });
        diagnostics.accept_logical_revision(PresentationBackend::Test, 10, original);
        let correlation = diagnostics.begin_presentation(10, original).unwrap();
        let mut newer = PresentationCauses::default();
        newer.insert(PresentationCause {
            kind: PresentationCauseKind::SceneUpdate,
            channel: None,
            trigger_generation: Some(11),
        });
        diagnostics.accept_logical_revision(PresentationBackend::Test, 11, newer);

        diagnostics.record_submission_return(correlation, PresentationBackend::Test, true, 5);

        let record = diagnostics.records().back().unwrap();

        assert_eq!(record.scene_generation, 10);
        assert_eq!(record.logical_revision, correlation.logical_revision);
        assert_eq!(record.logical_causes, original.to_vec());
    }
    #[test]
    fn cached_rehandoff_reuses_render_generation_but_advances_submission_attempt() {
        let mut diagnostics = PresentationDiagnostics::new();
        diagnostics.accept_logical_revision(
            PresentationBackend::Test,
            7,
            PresentationCauses::default(),
        );
        let rendered = diagnostics
            .begin_presentation(7, PresentationCauses::default())
            .unwrap();
        diagnostics.record_render_return(rendered, PresentationBackend::Test, true, 1, 2);
        let rehandoff = diagnostics
            .begin_rehandoff(7, PresentationCauses::default(), rendered.render_generation)
            .unwrap();
        diagnostics.configure_event_metadata(
            "cached_completed_frame_rehandoff",
            "current_lease_valid",
            HandoffMode::CachedFrameRehandoff,
            Some(42),
        );
        diagnostics.record_submission_attempt(rehandoff, PresentationBackend::Test);
        diagnostics.record_submission_return(rehandoff, PresentationBackend::Test, true, 3);

        assert_eq!(rehandoff.render_generation, rendered.render_generation);
        assert!(rehandoff.submission_attempt > rendered.submission_attempt);
        assert_eq!(
            diagnostics
                .records()
                .iter()
                .filter(|record| record.stage == PresentationStage::RenderReturned)
                .count(),
            1
        );
        let submission = diagnostics.records().back().unwrap();
        assert_eq!(submission.handoff_mode, HandoffMode::CachedFrameRehandoff);
        assert_eq!(submission.content_identity, Some(42));
    }

    #[test]
    fn compositor_observation_is_unattributed_and_preserves_unavailable_metrics() {
        let mut diagnostics = PresentationDiagnostics::new();
        diagnostics.record_compositor_observation(
            PresentationBackend::Test,
            0,
            2,
            3,
            None,
            Some(1_250),
            None,
        );

        let record = diagnostics.records().back().unwrap();
        assert_eq!(record.compositor_frame_index, Some(0));
        assert_eq!(record.render_generation, None);
        assert_eq!(record.submission_attempt, None);
        assert_eq!(record.compositor_render_cpu_us, None);
        assert_eq!(record.compositor_total_render_gpu_us, Some(1_250));
        assert_eq!(record.compositor_post_submit_gpu_us, None);
        assert_eq!(
            record.compositor_attribution,
            Some(CompositorAttribution::NotOverlayAttributable)
        );
        assert_eq!(
            record.physical_hmd_visibility,
            PhysicalHmdVisibility::NotObservable
        );
    }

    #[test]
    fn candidate_build_identity_has_safe_explicit_fallback() {
        assert_eq!(allowlisted_build_identity(None), "not_recorded");
        assert_eq!(
            allowlisted_build_identity(Some("candidate-42.ab")),
            "candidate-42.ab"
        );
        assert_eq!(
            allowlisted_build_identity(Some("unsafe identity")),
            "not_recorded"
        );
        assert_eq!(
            allowlisted_build_identity(Some("secret=payload")),
            "not_recorded"
        );
    }
}
