use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::logging::OverlayLoggingMode;
use crate::runtime::StartupError;

pub const QUIET_TAIL_PROFILE_ENV: &str = "PURIPULY_OVERLAY_QUIET_TAIL_PROFILE";

pub const HANDOFF_EXPERIMENT_ENV: &str = "PURIPULY_OVERLAY_HANDOFF_EXPERIMENT";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffExperiment {
    #[default]
    Off,
    CachedFrameRehandoff,
}

impl HandoffExperiment {
    pub fn id(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::CachedFrameRehandoff => "cached_frame_rehandoff",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuietTailProfile {
    P05,
    P10,
    P15,
    P20,
    NoRetry,
    OneRetry,
}

impl Default for QuietTailProfile {
    fn default() -> Self {
        Self::P05
    }
}

impl QuietTailProfile {
    pub fn max_final_opportunities(self) -> u32 {
        match self {
            Self::P05 => 5,
            Self::P10 => 10,
            Self::P15 => 15,
            Self::P20 => 20,
            Self::NoRetry => 0,
            Self::OneRetry => 1,
        }
    }

    pub fn scheduling_wall(self) -> std::time::Duration {
        std::time::Duration::from_millis(match self {
            Self::P05 => 500,
            Self::P10 => 1000,
            Self::P15 => 1500,
            Self::P20 => 2000,
            Self::NoRetry => 0,
            Self::OneRetry => 2000,
        })
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::P05 => "p05",
            Self::P10 => "p10",
            Self::P15 => "p15",
            Self::P20 => "p20",
            Self::NoRetry => "no_retry",
            Self::OneRetry => "one_retry",
        }
    }
}

pub const EXPECTED_CONTRACT_VERSION: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OverlayManifest {
    pub contract_version: u32,
    pub app_version: String,
    pub overlay_instance_id: String,
    pub bridge_url: String,
    pub session_token: String,
    pub parent_pid: u32,
    pub startup_deadline_ms: u32,
    pub log_dir: String,
    pub log_level: String,
    pub locale: String,
    pub logging_mode: OverlayLoggingMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct OverlayManifestSerde {
    contract_version: u32,
    app_version: String,
    overlay_instance_id: String,
    bridge_url: String,
    session_token: String,
    parent_pid: u32,
    startup_deadline_ms: u32,
    log_dir: String,
    log_level: String,
    locale: String,
    #[serde(default)]
    logging_mode: Option<OverlayLoggingMode>,
    #[serde(default)]
    diagnostics_enabled: Option<bool>,
}

impl TryFrom<OverlayManifestSerde> for OverlayManifest {
    type Error = StartupError;

    fn try_from(raw: OverlayManifestSerde) -> Result<Self, Self::Error> {
        let logging_mode = match (raw.logging_mode, raw.diagnostics_enabled) {
            (Some(mode), _) => mode,
            (None, Some(true)) => OverlayLoggingMode::Detailed,
            (None, Some(false)) => OverlayLoggingMode::Basic,
            (None, None) => {
                return Err(StartupError::Manifest(
                    "missing field `logging_mode`".to_string(),
                ))
            }
        };

        Ok(Self {
            contract_version: raw.contract_version,
            app_version: raw.app_version,
            overlay_instance_id: raw.overlay_instance_id,
            bridge_url: raw.bridge_url,
            session_token: raw.session_token,
            parent_pid: raw.parent_pid,
            startup_deadline_ms: raw.startup_deadline_ms,
            log_dir: raw.log_dir,
            log_level: raw.log_level,
            locale: raw.locale,
            logging_mode,
        })
    }
}

pub fn resolve_quiet_tail_profile_from_env() -> Result<QuietTailProfile, StartupError> {
    resolve_quiet_tail_profile(std::env::var_os(QUIET_TAIL_PROFILE_ENV).as_deref())
}

pub fn resolve_quiet_tail_profile(
    value: Option<&std::ffi::OsStr>,
) -> Result<QuietTailProfile, StartupError> {
    let Some(value) = value else {
        return Ok(QuietTailProfile::P05);
    };
    let Some(value) = value.to_str() else {
        return Err(StartupError::Manifest(
            "quiet tail profile environment value is invalid".to_string(),
        ));
    };
    match value {
        "p05" => Ok(QuietTailProfile::P05),
        "p10" => Ok(QuietTailProfile::P10),
        "p15" => Ok(QuietTailProfile::P15),
        "p20" => Ok(QuietTailProfile::P20),
        "no_retry" => Ok(QuietTailProfile::NoRetry),
        "one_retry" => Ok(QuietTailProfile::OneRetry),
        _ => Err(StartupError::Manifest(
            "quiet tail profile environment value is unsupported".to_string(),
        )),
    }
}

pub fn resolve_handoff_experiment_from_env() -> Result<HandoffExperiment, StartupError> {
    resolve_handoff_experiment(std::env::var_os(HANDOFF_EXPERIMENT_ENV).as_deref())
}

pub fn resolve_handoff_experiment(
    value: Option<&std::ffi::OsStr>,
) -> Result<HandoffExperiment, StartupError> {
    let Some(value) = value else {
        return Ok(HandoffExperiment::Off);
    };
    let Some(value) = value.to_str() else {
        return Err(StartupError::Manifest(
            "handoff experiment environment value is invalid".to_string(),
        ));
    };
    match value {
        "off" => Ok(HandoffExperiment::Off),
        "cached_frame_rehandoff" => Ok(HandoffExperiment::CachedFrameRehandoff),
        _ => Err(StartupError::Manifest(
            "handoff experiment environment value is unsupported".to_string(),
        )),
    }
}

pub fn load_manifest(path: impl AsRef<Path>) -> Result<OverlayManifest, StartupError> {
    let content =
        std::fs::read_to_string(path).map_err(|error| StartupError::Manifest(error.to_string()))?;
    let manifest: OverlayManifestSerde = serde_json::from_str(&content)
        .map_err(|error| StartupError::Manifest(error.to_string()))?;
    manifest.try_into()
}

pub fn validate_manifest(manifest: &OverlayManifest) -> Result<(), StartupError> {
    if manifest.contract_version != EXPECTED_CONTRACT_VERSION {
        return Err(StartupError::ContractMismatch(format!(
            "expected contract_version={} but received {}",
            EXPECTED_CONTRACT_VERSION, manifest.contract_version
        )));
    }
    Ok(())
}
