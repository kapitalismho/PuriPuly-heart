use crate::bridge::{BridgeError, BridgeIncoming};
use crate::presentation::PresentationCorrelation;

#[derive(Debug)]
pub(crate) enum FrameCycleResult {
    Submitted,
    CachedFrameRehandoff,
    Preempted(Result<BridgeIncoming, BridgeError>),
    NoWork,
}

impl FrameCycleResult {
    pub(crate) fn pending_message(self) -> Option<Result<BridgeIncoming, BridgeError>> {
        match self {
            Self::Preempted(message) => Some(message),
            Self::Submitted | Self::CachedFrameRehandoff | Self::NoWork => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FrameProgress {
    pub(crate) correlation: Option<PresentationCorrelation>,
}
