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
            fields: native_failure_fields(&error),
            error,
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
    let (outgoing_tx, outgoing_rx) = mpsc::unbounded_channel::<Value>();
    let heartbeat_interval = Duration::from_millis(manifest.heartbeat_interval_ms);
    let mut writer = spawn_writer(
        write_half,
        manifest.session_id.clone(),
        heartbeat_interval,
        outgoing_rx,
    );
    outgoing_tx
        .send(event(&manifest.session_id, "startup", None, json!({})))
        .map_err(|_| 11)?;

    let engine = Arc::new(Mutex::new(ASREngine::default()));
    let mut reader = BufReader::new(read_half).lines();
    let mut active: Option<ActiveOperation> = None;
    let mut shutdown_request_id: Option<String> = None;
    let mut terminal_code = None;
    let mut writer_finished = false;

    loop {
        tokio::select! {
            result = &mut writer => {
                writer_finished = true;
                terminal_code = Some(14);
                let _ = result;
                if let Some(operation) = &active {
                    operation.token.cancel();
                }
                break;
            }
            line = reader.next_line() => {
                let line = match line {
                    Ok(Some(value)) => value,
                    Ok(None) => {
                        terminal_code = Some(13);
                        if let Some(operation) = &active {
                            operation.token.cancel();
                        }
                        break;
                    }
                    Err(_) => {
                        terminal_code = Some(15);
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
                    Err(error) => OperationResult {
                        request_id: active.as_ref().expect("active operation").request_id.clone(),
                        payload: Err(join_failure(&error).into()),
                    },
                };
                active = None;
                if let Err(code) = send_operation_result(&outgoing_tx, &manifest.session_id, operation_result) {
                    terminal_code = terminal_code.or(Some(code));
                    break;
                }
                if shutdown_request_id.is_some() {
                    break;
                }
            }
        }
    }

    if let Some(operation) = active {
        settle_active_operation(
            operation,
            &outgoing_tx,
            &manifest.session_id,
            &mut terminal_code,
        ).await;
    }
    let engine_for_unload = Arc::clone(&engine);
    if let Err(error) = tokio::task::spawn_blocking(move || {
        engine_for_unload
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .unload();
    })
    .await
    {
        terminal_code = terminal_code.or(Some(12));
        let _ = outgoing_tx.send(failure(
            &manifest.session_id,
            shutdown_request_id.as_deref(),
            if error.is_panic() { "worker_unload_panicked" } else { "worker_unload_cancelled" },
            false,
            json!({}),
        ));
    }
    if !writer_finished {
        let _ = outgoing_tx.send(event(
            &manifest.session_id,
            "shutdown",
            shutdown_request_id.as_deref(),
            json!({"outcome": if terminal_code.is_some() { "failed" } else { "completed" }}),
        ));
    }
    drop(outgoing_tx);
    if !writer_finished {
        finish_writer(writer, &mut terminal_code).await;
    }
    terminal_code.map_or(Ok(()), Err)
}

fn spawn_writer<W>(
    mut writer: W,
    session_id: String,
    heartbeat_interval: Duration,
    mut outgoing: mpsc::UnboundedReceiver<Value>,
) -> JoinHandle<Result<(), std::io::Error>>
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(heartbeat_interval);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    write_frame(&mut writer, &json!({
                        "type": "heartbeat",
                        "contract_version": CONTRACT_VERSION,
                        "session_id": session_id,
                    })).await?;
                }
                payload = outgoing.recv() => {
                    let Some(payload) = payload else { return Ok(()); };
                    write_frame(&mut writer, &payload).await?;
                }
            }
        }
    })
}

async fn settle_active_operation(
    operation: ActiveOperation,
    outgoing: &mpsc::UnboundedSender<Value>,
    session_id: &str,
    terminal_code: &mut Option<i32>,
) {
    operation.token.cancel();
    let result = match operation.handle.await {
        Ok(result) => result,
        Err(error) => OperationResult {
            request_id: operation.request_id,
            payload: Err(join_failure(&error).into()),
        },
    };
    if let Err(code) = send_operation_result(outgoing, session_id, result) {
        *terminal_code = terminal_code.or(Some(code));
    }
}

async fn finish_writer(
    writer: JoinHandle<Result<(), std::io::Error>>,
    terminal_code: &mut Option<i32>,
) {
    if !matches!(writer.await, Ok(Ok(()))) {
        *terminal_code = terminal_code.or(Some(14));
    }
}

fn join_failure(error: &tokio::task::JoinError) -> EngineError {
    if error.is_panic() {
        EngineError::OperationPanicked
    } else {
        EngineError::OperationCancelled
    }
}

fn send_operation_result(
    outgoing: &mpsc::UnboundedSender<Value>,
    session_id: &str,
    result: OperationResult,
) -> Result<(), i32> {
    let payload = match result.payload {
        Ok(payload) => response(session_id, &result.request_id, payload),
        Err(result_failure) => failure(
            session_id,
            Some(&result.request_id),
            result_failure.error.code(),
            result_failure.attempt_started,
            result_failure.fields,
        ),
    };
    outgoing.send(payload).map_err(|_| 14)
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
    let mut fields = native_failure_fields(&failure.error);
    if let Some(object) = fields.as_object_mut() {
        object.insert("channel".into(), json!(channel));
        object.insert("backend".into(), json!(backend));
        if let Some((audio_seconds, decode_seconds, rtf)) = started_timing {
            object.insert("audio_seconds".into(), json!(audio_seconds));
            object.insert("decode_seconds".into(), json!(decode_seconds));
            object.insert("rtf".into(), json!(rtf));
        }
    }
    OperationFailure {
        error: failure.error,
        attempt_started: started_timing.is_some(),
        fields,
    }
}

fn native_failure_fields(error: &EngineError) -> Value {
    error.native_cause().map_or_else(
        || json!({}),
        |cause| json!({
            "native_reason": cause.native_reason,
            "native_status": cause.native_status,
        }),
    )
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
    fn native_failure_frame_retains_only_safe_cause_and_timing() {
        let error = EngineError::Native {
            classification: "decode_failure",
            cause: crate::engine::NativeCause {
                native_reason: "output_repetition",
                native_status: 15,
            },
        };
        let result = transcription_operation_failure(
            "self",
            "cpu",
            TranscriptionFailure {
                error,
                audio_seconds: Some(1.0),
                decode_seconds: 0.2,
                attempt_started: true,
            },
        );
        let frame = failure(
            "session",
            Some("request"),
            result.error.code(),
            result.attempt_started,
            result.fields,
        );
        assert_eq!(frame["error_code"], "decode_failure");
        assert_eq!(frame["payload"]["native_reason"], "output_repetition");
        assert_eq!(frame["payload"]["native_status"], 15);
        assert_eq!(frame["payload"]["decode_seconds"], 0.2);
        assert!(frame["payload"].get("message").is_none());
        assert!(frame["payload"].get("partial").is_none());
    }

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

    #[tokio::test]
    async fn worker_shutdown_drains_writer_and_disconnect_is_not_success() {
        use super::{run_worker, write_frame, LaunchManifest, WorkerMode, CONTRACT_VERSION};
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::TcpListener;

        for mode in ["shutdown", "disconnect", "invalid_utf8"] {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let manifest = LaunchManifest {
                contract_version: CONTRACT_VERSION,
                session_id: "synthetic-session".into(),
                auth_token: "synthetic-auth-token-not-a-secret".into(),
                connect_host: "127.0.0.1".into(),
                connect_port: listener.local_addr().unwrap().port(),
                heartbeat_interval_ms: 100,
                mode: WorkerMode::Persistent,
            };
            let worker = tokio::spawn(run_worker(manifest));
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = stream.into_split();
            let mut lines = BufReader::new(reader).lines();
            assert!(lines.next_line().await.unwrap().is_some());
            if mode == "shutdown" {
                write_frame(
                    &mut writer,
                    &json!({
                        "type": "shutdown",
                        "contract_version": CONTRACT_VERSION,
                        "session_id": "synthetic-session",
                        "request_id": "synthetic-shutdown",
                    }),
                ).await.unwrap();
                let mut completed = false;
                while let Some(line) = lines.next_line().await.unwrap() {
                    let frame: serde_json::Value = serde_json::from_str(&line).unwrap();
                    if frame["event"] == "shutdown" {
                        assert_eq!(frame["fields"]["outcome"], "completed");
                        completed = true;
                    }
                }
                assert!(completed);
                assert_eq!(worker.await.unwrap(), Ok(()));
            } else if mode == "invalid_utf8" {
                writer.write_all(&[0xff, b'\n']).await.unwrap();
                let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), worker)
                    .await.unwrap().unwrap();
                assert_eq!(outcome, Err(15));
            } else {
                drop(lines);
                drop(writer);
                let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), worker)
                    .await.unwrap().unwrap();
                assert!(matches!(outcome, Err(13 | 14)));
            }
        }
    }

    #[tokio::test]
    async fn worker_join_failures_keep_panic_and_cancellation_distinct() {
        let panicked = tokio::spawn(async { panic!("PRIVATE_PANIC_DETAIL") })
            .await.unwrap_err();
        assert_eq!(super::join_failure(&panicked).code(), "worker_operation_panicked");
        let cancelled = tokio::spawn(std::future::pending::<()>());
        cancelled.abort();
        let error = cancelled.await.unwrap_err();
        assert_eq!(super::join_failure(&error).code(), "worker_operation_cancelled");
    }

    #[tokio::test]
    async fn failed_writer_does_not_replace_primary_while_active_work_settles() {
        use super::{finish_writer, settle_active_operation, spawn_writer, ActiveOperation, OperationResult};
        use std::time::Duration;
        use tokio::sync::{mpsc, oneshot};
        use transcribe_cpp::CancelToken;

        for primary in [None, Some(13), Some(15)] {
            let (transport, peer) = tokio::io::duplex(64);
            let (outgoing, incoming) = mpsc::unbounded_channel();
            let writer = spawn_writer(
                transport, "synthetic-session".into(), Duration::from_millis(100), incoming
            );
            let token = CancelToken::new();
            let active_token = token.clone();
            let (started, running) = oneshot::channel();
            let operation = ActiveOperation {
                request_id: "synthetic-active".into(),
                token,
                handle: tokio::spawn(async move {
                    started.send(()).unwrap();
                    while !active_token.is_cancelled() {
                        tokio::task::yield_now().await;
                    }
                    OperationResult {
                        request_id: "synthetic-active".into(),
                        payload: Err(EngineError::Cancelled.into()),
                    }
                }),
            };
            running.await.unwrap();
            drop(peer);
            let mut terminal = primary;
            finish_writer(writer, &mut terminal).await;
            assert_eq!(terminal, primary.or(Some(14)));
            settle_active_operation(operation, &outgoing, "synthetic-session", &mut terminal).await;
            assert_eq!(terminal, primary.or(Some(14)));
            assert!(outgoing.is_closed());
        }
    }
}
