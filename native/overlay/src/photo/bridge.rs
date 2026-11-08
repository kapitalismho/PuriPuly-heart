use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{protocol::WebSocketConfig, Message},
    MaybeTlsStream, WebSocketStream,
};

use super::manifest::{PhotoError, PhotoManifest, PHOTO_PROTOCOL_VERSION};
use super::prepare::PrepareRequest;
use super::state::Photo;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum PhotoIncoming {
    #[serde(rename = "photo_snapshot")]
    Snapshot { revision: u64, photo: Option<Photo> },
    #[serde(rename = "photo_prepare")]
    Prepare(PrepareRequest),
    #[serde(rename = "photo_probe_pull")]
    ProbePull { photo_id: String, event_id: String },
    #[serde(rename = "photo_probe_hide")]
    ProbeHide { photo_id: String },
    #[serde(rename = "photo_debug_original")]
    DebugOriginal { photo_id: String, enabled: bool },
    #[serde(rename = "photo_debug_export")]
    DebugExport { photo_id: String, label: String },
    #[serde(rename = "photo_debug_quiesce")]
    DebugQuiesce { request_id: String },
    #[serde(rename = "photo_debug_preview_source")]
    DebugPreviewSource { photo_id: String },
    #[serde(rename = "photo_haptic")]
    Haptic {
        photo_id: String,
        attempt_id: u64,
        block_id: u64,
    },
    #[serde(rename = "shutdown")]
    Shutdown,
}

pub struct PhotoBridge {
    stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
}

impl PhotoBridge {
    pub async fn connect(manifest: &PhotoManifest) -> Result<(Self, PhotoIncoming), PhotoError> {
        let mut config = WebSocketConfig::default();
        config.max_message_size = Some(4 * 1024 * 1024);
        config.max_frame_size = Some(4 * 1024 * 1024);
        let (stream, _) = connect_async_with_config(&manifest.bridge_url, Some(config), false)
            .await
            .map_err(|_| PhotoError::Bridge)?;
        let mut bridge = Self { stream };
        bridge
            .send(json!({
                "type": "photo_auth",
                "protocol_version": PHOTO_PROTOCOL_VERSION,
                "role": "photo",
                "session_token": manifest.session_token,
                "instance_id": manifest.instance_id,
                "generation": manifest.generation,
            }))
            .await?;
        let first = bridge.next().await?;
        if !matches!(first, PhotoIncoming::Snapshot { .. }) {
            return Err(PhotoError::Protocol);
        }
        Ok((bridge, first))
    }

    pub async fn send(&mut self, payload: Value) -> Result<(), PhotoError> {
        self.stream
            .send(Message::Text(payload.to_string().into()))
            .await
            .map_err(|_| PhotoError::Bridge)
    }

    pub async fn send_timing(
        &mut self,
        photo_id: &str,
        attempt_id: u64,
        block_id: u64,
        stage: &str,
        duration_ms: f64,
    ) -> Result<(), PhotoError> {
        self.send(json!({
            "type":"photo_image_timing","photo_id":photo_id,"attempt_id":attempt_id,
            "block_id":block_id,"stage":stage,"duration_ms":duration_ms,
        }))
        .await
    }

    pub async fn next(&mut self) -> Result<PhotoIncoming, PhotoError> {
        loop {
            let message = self
                .stream
                .next()
                .await
                .ok_or(PhotoError::Bridge)?
                .map_err(|_| PhotoError::Bridge)?;
            match message {
                Message::Text(text) => {
                    let value: Value =
                        serde_json::from_str(&text).map_err(|_| PhotoError::Protocol)?;
                    if matches!(
                        value.get("type").and_then(Value::as_str),
                        Some("auth_error" | "photo_auth_error")
                    ) {
                        return Err(PhotoError::Authentication);
                    }
                    return serde_json::from_value(value).map_err(|_| PhotoError::Protocol);
                }
                Message::Close(_) => return Err(PhotoError::Bridge),
                Message::Binary(_) => return Err(PhotoError::Protocol),
                Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
            }
        }
    }

    pub async fn close(&mut self) -> Result<(), PhotoError> {
        self.stream
            .close(None)
            .await
            .map_err(|_| PhotoError::Bridge)
    }
}
