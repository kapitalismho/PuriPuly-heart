use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use std::path::Path;

pub const CONTRACT_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    Discovery,
    Persistent,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaunchManifest {
    pub contract_version: u32,
    pub session_id: String,
    pub auth_token: String,
    pub connect_host: String,
    pub connect_port: u16,
    pub heartbeat_interval_ms: u64,
    pub mode: WorkerMode,
}

impl LaunchManifest {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|_| "manifest_read_failed".to_string())?;
        let manifest: Self =
            serde_json::from_slice(&bytes).map_err(|_| "manifest_invalid".to_string())?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != CONTRACT_VERSION {
            return Err("contract_mismatch".to_string());
        }
        if self.session_id.len() < 16 || self.session_id.len() > 128 {
            return Err("session_id_invalid".to_string());
        }
        if self.auth_token.len() < 32 || self.auth_token.len() > 256 {
            return Err("auth_token_invalid".to_string());
        }
        if self.connect_host.parse::<Ipv4Addr>() != Ok(Ipv4Addr::LOCALHOST) {
            return Err("connect_host_not_loopback".to_string());
        }
        if self.connect_port == 0 {
            return Err("connect_port_invalid".to_string());
        }
        if !(100..=60_000).contains(&self.heartbeat_interval_ms) {
            return Err("heartbeat_interval_invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Discover {
        contract_version: u32,
        session_id: String,
        request_id: String,
    },
    Activate {
        contract_version: u32,
        session_id: String,
        request_id: String,
        model_path: String,
        device_id: String,
    },
    Transcribe {
        contract_version: u32,
        session_id: String,
        request_id: String,
        channel: String,
        audio_path: String,
        language_hint: Option<String>,
    },
    Cancel {
        contract_version: u32,
        session_id: String,
        request_id: String,
        target_request_id: String,
    },
    Shutdown {
        contract_version: u32,
        session_id: String,
        request_id: String,
    },
}

impl Request {
    pub fn validate_session(&self, expected_session_id: &str) -> Result<(), String> {
        let (contract_version, session_id, request_id) = match self {
            Self::Discover {
                contract_version,
                session_id,
                request_id,
            }
            | Self::Activate {
                contract_version,
                session_id,
                request_id,
                ..
            }
            | Self::Transcribe {
                contract_version,
                session_id,
                request_id,
                ..
            }
            | Self::Cancel {
                contract_version,
                session_id,
                request_id,
                ..
            }
            | Self::Shutdown {
                contract_version,
                session_id,
                request_id,
            } => (contract_version, session_id, request_id),
        };
        if *contract_version != CONTRACT_VERSION {
            return Err("contract_mismatch".to_string());
        }
        if session_id != expected_session_id {
            return Err("session_mismatch".to_string());
        }
        if request_id.is_empty() || request_id.len() > 128 {
            return Err("request_id_invalid".to_string());
        }
        Ok(())
    }

    pub fn request_id(&self) -> &str {
        match self {
            Self::Discover { request_id, .. }
            | Self::Activate { request_id, .. }
            | Self::Transcribe { request_id, .. }
            | Self::Cancel { request_id, .. }
            | Self::Shutdown { request_id, .. } => request_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LaunchManifest, Request, WorkerMode, CONTRACT_VERSION};

    #[test]
    fn manifest_accepts_only_ipv4_loopback_and_bounded_auth_fields() {
        let valid = LaunchManifest {
            contract_version: CONTRACT_VERSION,
            session_id: "session-123456789".to_string(),
            auth_token: "a".repeat(64),
            connect_host: "127.0.0.1".to_string(),
            connect_port: 12345,
            heartbeat_interval_ms: 500,
            mode: WorkerMode::Persistent,
        };
        assert_eq!(valid.validate(), Ok(()));

        for host in ["0.0.0.0", "localhost", "::1", "127.0.0.2", "192.0.2.1"] {
            let mut invalid = valid.clone();
            invalid.connect_host = host.to_string();
            assert_eq!(
                invalid.validate(),
                Err("connect_host_not_loopback".to_string()),
                "{host}"
            );
        }
        for length in [31, 257] {
            let mut invalid = valid.clone();
            invalid.auth_token = "a".repeat(length);
            assert_eq!(invalid.validate(), Err("auth_token_invalid".to_string()));
        }
        for length in [32, 256] {
            let mut boundary = valid.clone();
            boundary.auth_token = "a".repeat(length);
            assert_eq!(boundary.validate(), Ok(()), "auth token length {length}");
        }
    }

    #[test]
    fn request_rejects_contract_session_and_request_identity_mismatch() {
        for (version, session, request_id, expected) in [
            (
                CONTRACT_VERSION,
                "expected-session",
                "discover-1".to_string(),
                Ok(()),
            ),
            (
                CONTRACT_VERSION + 1,
                "expected-session",
                "discover-1".to_string(),
                Err("contract_mismatch".to_string()),
            ),
            (
                CONTRACT_VERSION,
                "different-session",
                "discover-1".to_string(),
                Err("session_mismatch".to_string()),
            ),
            (
                CONTRACT_VERSION,
                "expected-session",
                String::new(),
                Err("request_id_invalid".to_string()),
            ),
            (
                CONTRACT_VERSION,
                "expected-session",
                "r".repeat(129),
                Err("request_id_invalid".to_string()),
            ),
            (CONTRACT_VERSION, "expected-session", "r".repeat(128), Ok(())),
        ] {
            let request = Request::Discover {
                contract_version: version,
                session_id: session.to_string(),
                request_id,
            };
            assert_eq!(request.validate_session("expected-session"), expected);
        }
    }

    #[test]
    fn transcribe_request_preserves_optional_language_hint() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "type": "transcribe",
            "contract_version": CONTRACT_VERSION,
            "session_id": "expected-session",
            "request_id": "transcribe-1",
            "channel": "peer",
            "audio_path": "sample.wav",
            "language_hint": "ja"
        }))
        .expect("transcribe request");

        match request {
            Request::Transcribe { language_hint, .. } => {
                assert_eq!(language_hint.as_deref(), Some("ja"));
            }
            _ => panic!("expected transcribe request"),
        }
    }
}
