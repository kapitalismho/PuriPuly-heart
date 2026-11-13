use std::fmt;

use serde::Serialize;
use serde_json::{json, Value};

use crate::presentation::PresentationStage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NativeFailure {
    pub native_stage: &'static str,
    pub native_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secondary: Option<NativeCause>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NativeCause {
    pub native_stage: &'static str,
    pub native_code: Option<i32>,
}

impl NativeFailure {
    pub fn new(stage: &'static str, code: Option<i32>) -> Self {
        Self { native_stage: stage, native_code: code, secondary: None }
    }

    pub fn at(mut self, stage: &'static str) -> Self {
        self.native_stage = stage;
        self
    }

    pub fn with_secondary(mut self, other: Self) -> Self {
        self.secondary = Some(NativeCause {
            native_stage: other.native_stage,
            native_code: other.native_code,
        });
        self
    }
}

impl fmt::Display for NativeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} code={:?}", self.native_stage, self.native_code)
    }
}


#[derive(Debug, Clone, Copy)]
pub(crate) struct OwnerStatusInput<'a> {
    pub(crate) overlay_instance_id: &'a str,
    pub(crate) runtime_generation: u64,
    pub(crate) health_challenge_id: Option<u64>,
    pub(crate) latest_applied_revision: u64,
    pub(crate) latest_handoff_revision: Option<u64>,
    pub(crate) desired_visible: bool,
    pub(crate) observed_runtime_visible: Option<bool>,
    pub(crate) has_drawable_text: bool,
    pub(crate) first_texture_submitted: bool,
    pub(crate) spatial_pose_unavailable: bool,
    pub(crate) due_elapsed_ms: u64,
    pub(crate) recovering: bool,
    pub(crate) terminal_failed: bool,
    pub(crate) due_active: bool,
    pub(crate) in_flight_stage: Option<PresentationStage>,
    pub(crate) primary_failure_reason: Option<&'static str>,
    pub(crate) cleanup_failure_reason: Option<&'static str>,
    pub(crate) diagnostic_dropped: u64,
    pub(crate) reliable_delivery_failures: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnerStatusProjection {
    pub(crate) payload: Value,
}

pub(crate) fn project_owner_status(input: OwnerStatusInput<'_>) -> OwnerStatusProjection {
    let current_covered_handoff =
        input.latest_handoff_revision == Some(input.latest_applied_revision);
    let confirmed_hide = !input.desired_visible && input.observed_runtime_visible == Some(false);
    let classification = if input.terminal_failed {
        "terminal_failed"
    } else if input.spatial_pose_unavailable {
        "pose_unavailable"
    } else if input.recovering {
        "recovering"
    } else if input.due_active {
        "due"
    } else if !input.has_drawable_text {
        if input.first_texture_submitted && confirmed_hide {
            "intentional_hidden"
        } else {
            "no_drawable_content"
        }
    } else {
        "healthy_idle"
    };
    OwnerStatusProjection {
        payload: json!({
            "type": "owner_status",
            "overlay_instance_id": input.overlay_instance_id,
            "runtime_generation": input.runtime_generation,
            "health_challenge_id": input.health_challenge_id,
            "latest_applied_revision": input.latest_applied_revision,
            "latest_handoff_revision": input.latest_handoff_revision,
            "current_covered_handoff": current_covered_handoff,
            "confirmed_hide": confirmed_hide,
            "desired_visible": input.desired_visible,
            "observed_runtime_visible": input.observed_runtime_visible,
            "due_elapsed_ms": input.due_elapsed_ms,
            "classification": classification,
            "in_flight_stage": input.in_flight_stage,
            "primary_failure_reason": input.primary_failure_reason,
            "cleanup_failure_reason": input.cleanup_failure_reason,
            "diagnostic_dropped": input.diagnostic_dropped,
            "reliable_delivery_failures": input.reliable_delivery_failures,
        }),
    }
}
