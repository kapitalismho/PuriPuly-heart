mod engine;
pub mod protocol;

use engine::{ASREngine, EngineError, TranscriptionFailure};
use protocol::{LaunchManifest, Request, WorkerMode, CONTRACT_VERSION};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use transcribe_cpp::{disable_logging, CancelToken};

const WORKER_NAME: &str = "PuriPulyHeartASRWorker";

struct ActiveOperation {
    request_id: String,
    token: CancelToken,
    handle: JoinHandle<OperationResult>,
}

struct OperationResult {
    request_id: String,
    payload: Result<Value, OperationFailure>,
}

struct OperationFailure {
    error: EngineError,
    fields: Value,
    attempt_started: bool,
}

impl From<EngineError> for OperationFailure {
    fn from(error: EngineError) -> Self {
        Self {
            error,
            fields: json!({}),
            attempt_started: false,
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
                "app_version": env!("CARGO_PKG_VERSION"),
                "contract_version": CONTRACT_VERSION,
                "native_version": transcribe_cpp::version(),
                "binding_version": transcribe_cpp::compiled_version(),
                "native_commit": transcribe_cpp::version_commit(),
            })
        );
        return 0;
    }
    if args.len() != 3 || args[1] != "--config" {
        return 2;
    }
    let manifest = match LaunchManifest::load(Path::new(&args[2])) {
        Ok(value) => value,
        Err(_) => return 10,
    };
    match run_worker(manifest).await {
        Ok(()) => 0,
        Err(code) => code,
    }
}

async fn run_worker(manifest: LaunchManifest) -> Result<(), i32> {
    disable_logging();
    let stream = TcpStream::connect((manifest.connect_host.as_str(), manifest.connect_port))
        .await
        .map_err(|_| 11)?;
    let (read_half, mut write_half) = stream.into_split();
    write_frame(
        &mut write_half,
        &json!({
            "type": "authenticate",
            "contract_version": CONTRACT_VERSION,
            "session_id": manifest.session_id,
            "auth_token": manifest.auth_token,
            "worker": WORKER_NAME,
            "pid": std::process::id(),
            "mode": manifest.mode,
        }),
    )
    .await
    .map_err(|_| 11)?;
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<Value>();
    let heartbeat_interval = Duration::from_millis(manifest.heartbeat_interval_ms);
    let session_id_for_writer = manifest.session_id.clone();
    let writer = tokio::spawn(async move {
        let mut interval = tokio::time::interval(heartbeat_interval);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let heartbeat = json!({
                        "type": "heartbeat",
                        "contract_version": CONTRACT_VERSION,
                        "session_id": session_id_for_writer,
                    });
                    if write_frame(&mut write_half, &heartbeat).await.is_err() {
                        return;
                    }
                }
                payload = outgoing_rx.recv() => {
                    let Some(payload) = payload else { return; };
                    if write_frame(&mut write_half, &payload).await.is_err() {
                        return;
                    }
                }
            }
        }
    });
    outgoing_tx
        .send(event(&manifest.session_id, "startup", None, json!({})))
        .map_err(|_| 11)?;

    let engine = Arc::new(Mutex::new(ASREngine::default()));
    let mut reader = BufReader::new(read_half).lines();
    let mut active: Option<ActiveOperation> = None;
    let mut shutdown_request_id: Option<String> = None;

    loop {
        tokio::select! {
            line = reader.next_line() => {
                let line = match line {
                    Ok(Some(value)) => value,
                    _ => {
                        if let Some(operation) = &active {
                            operation.token.cancel();
                        }
                        break;
                    }
                };
                let request = match serde_json::from_str::<Request>(&line) {
                    Ok(value) => value,
                    Err(_) => {
                        let _ = outgoing_tx.send(failure(
                            &manifest.session_id,
                            None,
                            "protocol_invalid",
                            false,
                            json!({}),
                        ));
                        continue;
                    }
                };
                if let Err(code) = request.validate_session(&manifest.session_id) {
                    let _ = outgoing_tx.send(failure(
                        &manifest.session_id,
                        Some(request.request_id()),
                        &code,
                        false,
                        json!({}),
                    ));
                    continue;
                }
                match request {
                    Request::Cancel { request_id, target_request_id, .. } => {
                        let cancelled = active.as_ref().is_some_and(|operation| {
                            if operation.request_id == target_request_id {
                                operation.token.cancel();
                                true
                            } else {
                                false
                            }
                        });
                        let _ = outgoing_tx.send(event(
                            &manifest.session_id,
                            "cancellation_requested",
                            Some(&request_id),
                            json!({"target_request_id": target_request_id, "active": cancelled}),
                        ));
                    }
                    Request::Shutdown { request_id, .. } => {
                        shutdown_request_id = Some(request_id);
                        if let Some(operation) = &active {
                            operation.token.cancel();
                        } else {
                            break;
                        }
                    }
                    other if active.is_some() => {
                        let _ = outgoing_tx.send(failure(
                            &manifest.session_id,
                            Some(other.request_id()),
                            "worker_busy",
                            false,
                            json!({}),
                        ));
                    }
                    Request::Discover { request_id, .. } => {
                        active = Some(spawn_discovery(request_id));
                    }
                    Request::Activate { request_id, model_path, device_id, backend, channel, n_threads, .. } => {
                        if manifest.mode != WorkerMode::Persistent {
                            let _ = outgoing_tx.send(failure(
                                &manifest.session_id,
                                Some(&request_id),
                                "mode_rejected",
                                false,
                                json!({}),
                            ));
                            continue;
                        }
                        active = Some(spawn_activation(
                            request_id,
                            PathBuf::from(model_path),
                            device_id,
                            backend,
                            channel,
                            n_threads,
                            Arc::clone(&engine),
                            outgoing_tx.clone(),
                            manifest.session_id.clone(),
                        ));
                    }
                    Request::Transcribe { request_id, channel, audio_path, language_hint, .. } => {
                        if manifest.mode != WorkerMode::Persistent || !matches!(channel.as_str(), "self" | "peer") {
                            let _ = outgoing_tx.send(failure(
                                &manifest.session_id,
                                Some(&request_id),
                                "request_rejected",
                                false,
                                json!({}),
                            ));
                            continue;
                        }
                        active = Some(spawn_transcription(
                            request_id,
                            channel,
                            PathBuf::from(audio_path),
                            language_hint,
                            Arc::clone(&engine),
                            outgoing_tx.clone(),
                            manifest.session_id.clone(),
                        ));
                    }
                }
            }
            result = async { (&mut active.as_mut().expect("active operation").handle).await }, if active.is_some() => {
                let operation_result = match result {
                    Ok(value) => value,
                    Err(_) => OperationResult {
                        request_id: active.as_ref().expect("active operation").request_id.clone(),
                        payload: Err(EngineError::BackendFailure.into()),
                    },
                };
                match operation_result.payload {
                    Ok(payload) => {
                        let _ = outgoing_tx.send(response(&manifest.session_id, &operation_result.request_id, payload));
                    }
                    Err(failure_result) => {
                        let _ = outgoing_tx.send(failure(
                            &manifest.session_id,
                            Some(&operation_result.request_id),
                            failure_result.error.code(),
                            failure_result.attempt_started,
                            failure_result.fields,
                        ));
                    }
                }
                active = None;
                if shutdown_request_id.is_some() {
                    break;
                }
            }
        }
    }

    if let Some(operation) = active {
        operation.token.cancel();
        let _ = operation.handle.await;
    }
    let engine_for_unload = Arc::clone(&engine);
    let _ = tokio::task::spawn_blocking(move || {
        engine_for_unload
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .unload();
    })
    .await;
    let _ = outgoing_tx.send(event(
        &manifest.session_id,
        "shutdown",
        shutdown_request_id.as_deref(),
        json!({"outcome": "completed"}),
    ));
    tokio::time::sleep(Duration::from_millis(20)).await;
    drop(outgoing_tx);
    writer.abort();
    let _ = writer.await;
    Ok(())
}

fn spawn_discovery(request_id: String) -> ActiveOperation {
    let token = CancelToken::new();
    let task_request_id = request_id.clone();
    let handle = tokio::task::spawn_blocking(move || {
        OperationResult {
            request_id: task_request_id,
            payload: ASREngine::discover()
                .map(|devices| json!({"devices": devices}))
                .map_err(Into::into),
        }
    });
    ActiveOperation {
        request_id,
        token,
        handle,
    }
}

fn spawn_activation(
    request_id: String,
    model_path: PathBuf,
    device_id: String,
    backend: String,
    channel: Option<String>,
    n_threads: i32,
    engine: Arc<Mutex<ASREngine>>,
    outgoing: mpsc::UnboundedSender<Value>,
    session_id: String,
) -> ActiveOperation {
    let token = CancelToken::new();
    let task_token = token.clone();
    let task_request_id = request_id.clone();
    let handle = tokio::task::spawn_blocking(move || {
        let payload = engine
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .activate(&model_path, &device_id, &backend, channel.as_deref(), n_threads, &task_token, |phase, progress| {
                let _ = outgoing.send(event(
                    &session_id,
                    "activation_progress",
                    Some(&task_request_id),
                    json!({"phase": phase, "progress": progress}),
                ));
            })
            .map(|result| json!({"activation": result}))
            .map_err(Into::into);
        OperationResult {
            request_id: task_request_id,
            payload,
        }
    });
    ActiveOperation {
        request_id,
        token,
        handle,
    }
}

fn spawn_transcription(
    request_id: String,
    channel: String,
    audio_path: PathBuf,
    language_hint: Option<String>,
    engine: Arc<Mutex<ASREngine>>,
    outgoing: mpsc::UnboundedSender<Value>,
    session_id: String,
) -> ActiveOperation {
    let token = CancelToken::new();
    let task_token = token.clone();
    let task_request_id = request_id.clone();
    let handle = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap_or_else(|error| error.into_inner());
        let backend = engine.backend();
        let payload = engine
            .transcribe(&channel, &audio_path, language_hint, &task_token, |audio_seconds| {
                let _ = outgoing.send(event(
                    &session_id,
                    "transcribe_started",
                    Some(&task_request_id),
                    json!({
                        "channel": channel,
                        "backend": backend,
                        "audio_seconds": audio_seconds,
                    }),
                ));
            })
            .map(|result| json!({"channel": channel, "backend": backend, "transcription": result}))
            .map_err(|failure| transcription_operation_failure(&channel, backend, failure));
        OperationResult {
            request_id: task_request_id,
            payload,
        }
    });
    ActiveOperation {
        request_id,
        token,
        handle,
    }
}

fn transcription_operation_failure(
    channel: &str,
    backend: &str,
    failure: TranscriptionFailure,
) -> OperationFailure {
    let started_timing = failure.started_timing();
    OperationFailure {
        error: failure.error,
        attempt_started: started_timing.is_some(),
        fields: match started_timing {
            Some((audio_seconds, decode_seconds, rtf)) => json!({
                "channel": channel,
                "backend": backend,
                "audio_seconds": audio_seconds,
                "decode_seconds": decode_seconds,
                "rtf": rtf,
            }),
            None => json!({
                "channel": channel,
                "backend": backend,
            }),
        },
    }
}

fn response(session_id: &str, request_id: &str, payload: Value) -> Value {
    json!({
        "type": "response",
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "request_id": request_id,
        "status": "ok",
        "payload": payload,
    })
}

fn failure(
    session_id: &str,
    request_id: Option<&str>,
    code: &str,
    attempt_started: bool,
    fields: Value,
) -> Value {
    json!({
        "type": "response",
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "request_id": request_id,
        "status": "failed",
        "error_code": code,
        "attempt_started": attempt_started,
        "payload": fields,
    })
}

fn event(session_id: &str, name: &str, request_id: Option<&str>, fields: Value) -> Value {
    json!({
        "type": "event",
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "event": name,
        "request_id": request_id,
        "fields": fields,
    })
}

async fn write_frame<W>(writer: &mut W, payload: &Value) -> Result<(), std::io::Error>
where
    W: AsyncWrite + Unpin,
{
    let mut bytes = serde_json::to_vec(payload)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await
}

#[cfg(test)]
mod tests {
    use super::{failure, transcription_operation_failure};
    use crate::engine::{EngineError, TranscriptionFailure};
    use serde_json::json;

    #[test]
    fn failure_frame_distinguishes_prestart_from_started_attempts() {
        let prestart = failure("session", Some("request"), "worker_busy", false, json!({}));
        let started = failure(
            "session",
            Some("request"),
            "decode_failure",
            true,
            json!({"decode_seconds": 0.25}),
        );

        assert_eq!(prestart["attempt_started"], false);
        assert_eq!(started["attempt_started"], true);
        assert_eq!(started["payload"]["decode_seconds"], 0.25);
    }

    #[test]
    fn transcription_failure_protocol_uses_only_valid_started_timing() {
        let prestart = transcription_operation_failure(
            "peer",
            "Vulkan",
            TranscriptionFailure {
                error: EngineError::AudioInvalid,
                audio_seconds: None,
                decode_seconds: 0.0,
                attempt_started: false,
            },
        );
        let started = transcription_operation_failure(
            "self",
            "Vulkan",
            TranscriptionFailure {
                error: EngineError::Cancelled,
                audio_seconds: Some(2.0),
                decode_seconds: 0.5,
                attempt_started: true,
            },
        );

        assert!(!prestart.attempt_started);
        assert!(prestart.fields.get("audio_seconds").is_none());
        assert!(prestart.fields.get("decode_seconds").is_none());
        assert!(prestart.fields.get("rtf").is_none());
        assert!(started.attempt_started);
        assert_eq!(started.fields["audio_seconds"], 2.0);
        assert_eq!(started.fields["decode_seconds"], 0.5);
        assert_eq!(started.fields["rtf"], 0.25);
        assert_eq!(started.fields["channel"], "self");
        assert_eq!(started.fields["backend"], "Vulkan");

        for (audio_seconds, decode_seconds) in [
            (None, 0.5),
            (Some(0.0), 0.5),
            (Some(-1.0), 0.5),
            (Some(f64::NAN), 0.5),
            (Some(f64::INFINITY), 0.5),
            (Some(2.0), -0.5),
            (Some(2.0), f64::NAN),
            (Some(2.0), f64::INFINITY),
        ] {
            let invalid = transcription_operation_failure(
                "peer",
                "Vulkan",
                TranscriptionFailure {
                    error: EngineError::DecodeFailure,
                    audio_seconds,
                    decode_seconds,
                    attempt_started: true,
                },
            );
            assert!(!invalid.attempt_started, "{audio_seconds:?} {decode_seconds}");
            assert_eq!(
                invalid.fields,
                json!({"channel": "peer", "backend": "Vulkan"}),
                "{audio_seconds:?} {decode_seconds}"
            );
        }
    }
}
