use serde::{Deserialize, Serialize};
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

pub const PHOTO_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhotoBackend {
    Openvr,
    Probe,
    Offscreen,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhotoManifest {
    pub protocol_version: u32,
    pub role: String,
    pub bridge_url: String,
    pub session_token: String,
    pub instance_id: String,
    pub generation: u64,
    pub backend: PhotoBackend,
    #[serde(default)]
    pub image: PhotoImageConfig,
    #[serde(default)]
    pub interaction: super::interaction::TemporaryInteractionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhotoImageConfig {
    pub model_path: Option<PathBuf>,
    pub runtime_path: Option<PathBuf>,
    pub max_size: [u32; 2],
    pub export_dir: Option<PathBuf>,
    pub debug_controls: bool,
}

impl Default for PhotoImageConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            runtime_path: None,
            max_size: [1920, 1080],
            export_dir: None,
            debug_controls: false,
        }
    }
}

impl PhotoImageConfig {
    pub fn model_path(&self) -> Result<PathBuf, PhotoError> {
        self.model_path.clone().map(Ok).unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|path| {
                    path.parent()
                        .map(|path| path.join("assets/photo/mbnv4_text_seg.onnx"))
                })
                .ok_or(PhotoError::Manifest)
        })
    }

    pub fn validate(&self) -> Result<(), PhotoError> {
        if self.max_size.iter().any(|size| *size == 0 || *size > 16384) {
            return Err(PhotoError::Manifest);
        }
        Ok(())
    }
}
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PhotoError {
    #[error("photo manifest invalid")]
    Manifest,
    #[error("photo bridge failed")]
    Bridge,
    #[error("photo authentication rejected")]
    Authentication,
    #[error("photo protocol invalid")]
    Protocol,
    #[error("photo OpenVR unavailable")]
    OpenVr,
    #[error("photo graphics failed")]
    Graphics,
    #[error("photo image processing failed")]
    Image,
}

impl PhotoError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Manifest => "manifest_invalid",
            Self::Bridge => "bridge_failed",
            Self::Authentication => "authentication_rejected",
            Self::Protocol => "protocol_invalid",
            Self::OpenVr => "openvr_unavailable",
            Self::Graphics => "graphics_failed",
            Self::Image => "image_processing_failed",
        }
    }
}

impl PhotoManifest {
    pub fn load(path: &Path) -> Result<Self, PhotoError> {
        let bytes = std::fs::read(path).map_err(|_| PhotoError::Manifest)?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|_| PhotoError::Manifest)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), PhotoError> {
        let address = self
            .bridge_url
            .strip_prefix("ws://127.0.0.1:")
            .ok_or(PhotoError::Manifest)?;
        let port = address.strip_suffix('/').unwrap_or(address);
        if self.protocol_version != PHOTO_PROTOCOL_VERSION
            || self.role != "photo"
            || port.parse::<u16>().ok().filter(|port| *port != 0).is_none()
            || self.session_token.is_empty()
            || self.session_token.len() > 4096
            || !valid_instance_id(&self.instance_id)
            || self.generation == 0
        {
            return Err(PhotoError::Manifest);
        }
        self.image.validate()?;
        self.interaction.validate()?;
        Ok(())
    }
}

fn valid_instance_id(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}
