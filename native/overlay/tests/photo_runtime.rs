use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use puripuly_heart_overlay::photo::image::{
    PhotoImageBoundary, PhotoImageOutput, PhotoImageResult, PreparationStage,
};
use puripuly_heart_overlay::photo::manifest::{PhotoBackend, PhotoError, PhotoManifest};
use puripuly_heart_overlay::photo::runtime::surface_identity;
use puripuly_heart_overlay::photo::state::{
    Photo, PhotoBlock, PhotoImage, PhotoPhase, PhotoPreparation, PhotoState, SnapshotAdmission,
};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};

fn manifest() -> PhotoManifest {
    PhotoManifest {
        protocol_version: 1,
        role: "photo".into(),
        bridge_url: "ws://127.0.0.1:12345".into(),
        session_token: "isolated-synthetic-token".into(),
        instance_id: "11111111-1111-4111-8111-111111111111".into(),
        generation: 1,
        backend: PhotoBackend::Probe,
    }
}

fn photo() -> Photo {
    Photo {
        photo_id: "synthetic-photo-a".into(),
        path: "synthetic-metadata.png".into(),
        file_size: 123,
        shutter_unix_ms: 1,
        attempt_id: 0,
        phase: PhotoPhase::Offer,
        offer_open: true,
        visible: false,
        image: None,
        preparations: Vec::new(),
        blocks: Vec::new(),
        failure_code: None,
        first_block_drawn: false,
    }
}

fn requested() -> Photo {
    Photo {
        attempt_id: 1,
        phase: PhotoPhase::Requesting,
        offer_open: false,
        visible: true,
        image: Some(PhotoImage {
            path: "synthetic-owned-snapshot.png".into(),
            width: 1920,
            height: 1080,
        }),
        ..photo()
    }
}

fn block(id: u64) -> PhotoBlock {
    PhotoBlock {
        block_id: id,
        bbox: [1.25, 2.5, 300.0, 100.0],
        source: "A\nB".into(),
        translation: "가\n나".into(),
    }
}

#[test]
fn photo_manifest_and_startup_contract_are_role_specific() {
    let mut config = manifest();
    assert_eq!(config.validate(), Ok(()));
    for address in [
        "ws://example.com:123",
        "wss://127.0.0.1:123",
        "ws://127.0.0.1:0",
        "ws://127.0.0.1:123/private",
    ] {
        config.bridge_url = address.into();
        assert_eq!(config.validate(), Err(PhotoError::Manifest));
    }
    config = manifest();
    config.role = "caption".into();
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
    let photo = surface_identity(&manifest(), "photo");
    let icon = surface_identity(&manifest(), "icon");
    assert_ne!(photo, icon);
    assert!(photo.contains(&format!(".p{}.", std::process::id())));
    config = manifest();
    config.generation = 2;
    assert_ne!(photo, surface_identity(&config, "photo"));
}

#[test]
fn cumulative_state_preserves_independent_blocks_and_fences_replacements() {
    let mut state = PhotoState::default();
    let mut current = requested();
    current.blocks.push(block(0));
    assert_eq!(
        state.apply(1, Some(current.clone())),
        Ok(SnapshotAdmission::Applied)
    );
    current.blocks.push(block(1));
    assert_eq!(
        state.apply(2, Some(current.clone())),
        Ok(SnapshotAdmission::Applied)
    );
    assert_eq!(state.photo().unwrap().blocks.len(), 2);
    assert_eq!(state.apply(1, Some(photo())), Ok(SnapshotAdmission::Stale));
    let mut dropping = current.clone();
    dropping.blocks.remove(0);
    assert_eq!(state.apply(3, Some(dropping)), Err(PhotoError::Protocol));
    let mut altered = current.clone();
    altered.blocks[0].translation = "changed".into();
    assert_eq!(state.apply(3, Some(altered)), Err(PhotoError::Protocol));
    current.attempt_id = 2;
    current.blocks.clear();
    assert_eq!(
        state.apply(3, Some(current.clone())),
        Ok(SnapshotAdmission::Applied)
    );
    assert!(state.photo().unwrap().blocks.is_empty());
    assert_eq!(
        state.apply(4, Some(requested())),
        Ok(SnapshotAdmission::Stale)
    );
    let mut replacement = photo();
    replacement.photo_id = "synthetic-photo-b".into();
    state.apply(4, Some(replacement)).unwrap();
    assert_eq!(state.apply(5, Some(current)), Ok(SnapshotAdmission::Stale));
    state.apply(5, None).unwrap();
    assert_eq!(state.apply(6, Some(photo())), Ok(SnapshotAdmission::Stale));
    assert!(state.photo().is_none());
}

#[test]
fn early_geometry_never_invents_source_lines_or_changes_completed_content() {
    let mut state = PhotoState::default();
    let mut current = requested();
    current.preparations.push(PhotoPreparation {
        block_id: 0,
        bbox: block(0).bbox,
        source: None,
        source_line_count: None,
    });
    state.apply(1, Some(current.clone())).unwrap();
    current.preparations[0].source = Some("A\nB".into());
    assert_eq!(
        state.apply(2, Some(current.clone())),
        Err(PhotoError::Protocol)
    );
    current.preparations[0].source_line_count = Some(1);
    assert_eq!(
        state.apply(2, Some(current.clone())),
        Err(PhotoError::Protocol)
    );
    current.preparations[0].source_line_count = Some(2);
    state.apply(2, Some(current.clone())).unwrap();
    current.blocks.push(block(0));
    state.apply(3, Some(current.clone())).unwrap();
    current.preparations[0].source_line_count = None;
    current.preparations[0].source = None;
    assert_eq!(state.apply(4, Some(current)), Err(PhotoError::Protocol));
}

#[tokio::test]
async fn asynchronous_image_boundary_coalesces_full_inputs_and_rejects_stale_results() {
    let (mut boundary, mut worker) = PhotoImageBoundary::channel();
    let mut state = PhotoState::default();
    let mut current = requested();
    current.preparations.push(PhotoPreparation {
        block_id: 0,
        bbox: block(0).bbox,
        source: None,
        source_line_count: None,
    });
    state.apply(1, Some(current.clone())).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let bbox_input = worker.next_input().await.unwrap();
    assert!(bbox_input.photo.as_ref().unwrap().preparations[0]
        .source_line_count
        .is_none());
    current.preparations[0].source = Some("A\nB".into());
    current.preparations[0].source_line_count = Some(2);
    current.blocks.push(block(0));
    state.apply(2, Some(current.clone())).unwrap();
    boundary.publish(state.snapshot().unwrap());
    current.blocks.push(block(1));
    state.apply(3, Some(current.clone())).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let input = worker.next_input().await.unwrap();
    assert_eq!(input.photo.as_ref().unwrap().blocks.len(), 2);
    assert_eq!(
        input.photo.as_ref().unwrap().preparations[0].source_line_count,
        Some(2)
    );
    worker
        .complete(PhotoImageResult {
            input: input.clone(),
            output: PhotoImageOutput::Prepared {
                block_id: 0,
                stage: PreparationStage::Source,
                elapsed_ms: 1.25,
            },
        })
        .await
        .ok()
        .unwrap();
    let result = boundary.next_result().await.unwrap();
    assert!(boundary.accept(&result, state.snapshot().unwrap()));
    assert!(!boundary.accept(&result, state.snapshot().unwrap()));
    current.attempt_id = 2;
    current.blocks.clear();
    current.preparations.clear();
    state.apply(4, Some(current)).unwrap();
    boundary.publish(state.snapshot().unwrap());
    assert!(worker.is_cancelled(&input));
    assert!(!boundary.accept(&result, state.snapshot().unwrap()));
    assert!(!boundary.accept(
        &PhotoImageResult {
            input: bbox_input,
            output: PhotoImageOutput::Prepared {
                block_id: 0,
                stage: PreparationStage::Bbox,
                elapsed_ms: 2.0
            }
        },
        state.snapshot().unwrap()
    ));
    boundary.close();
    assert!(worker.is_cancelled(&input));
    assert!(worker
        .complete(PhotoImageResult {
            input,
            output: PhotoImageOutput::Prepared {
                block_id: 0,
                stage: PreparationStage::Source,
                elapsed_ms: 2.0
            }
        })
        .await
        .is_err());
}

async fn receive(socket: &mut WebSocketStream<TcpStream>) -> Value {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(10), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Text(text) = message {
            return serde_json::from_str(&text).unwrap();
        }
    }
}

async fn send(socket: &mut WebSocketStream<TcpStream>, value: Value) {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .unwrap();
}

struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn start_child(
    config: &PhotoManifest,
    name: &str,
) -> (ChildGuard, WebSocketStream<TcpStream>, std::path::PathBuf) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let directory =
        std::env::temp_dir().join(format!("puri339-native-{}-{name}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("launch.json");
    let document = json!({
        "protocol_version": config.protocol_version, "role": config.role,
        "bridge_url": format!("ws://{}", listener.local_addr().unwrap()), "session_token": config.session_token,
        "instance_id": config.instance_id, "generation": config.generation, "backend": config.backend,
    });
    std::fs::write(&path, document.to_string()).unwrap();
    let child = ChildGuard(
        std::process::Command::new(env!("CARGO_BIN_EXE_PuriPulyHeartOverlay"))
            .args(["--role", "photo", "--config"])
            .arg(&path)
            .env(
                "PURIPULY_OVERLAY_QUIET_TAIL_PROFILE",
                "invalid-caption-only-setting",
            )
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let (stream, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
        .await
        .unwrap()
        .unwrap();
    let socket = accept_async(stream).await.unwrap();
    (child, socket, directory)
}

#[tokio::test]
async fn actual_photo_process_authenticates_accumulates_pulls_replays_and_exits() {
    let mut config = manifest();
    let mut first_pid = 0;
    let mut retained = requested();
    retained.blocks = vec![block(0), block(1)];
    retained.phase = PhotoPhase::Completed;
    retained.first_block_drawn = true;
    for run in 0..2 {
        config.generation += run;
        if run == 1 {
            config.instance_id = "22222222-2222-4222-8222-222222222222".into();
        }
        let (mut child, mut socket, directory) =
            start_child(&config, &format!("smoke-{run}")).await;
        let pid = child.0.id();
        if run == 0 {
            first_pid = pid;
        } else {
            assert_ne!(pid, first_pid);
        }
        let auth = receive(&mut socket).await;
        assert_eq!(
            auth,
            json!({"type":"photo_auth","protocol_version":1,"role":"photo","session_token":config.session_token,"instance_id":config.instance_id,"generation":config.generation})
        );
        let initial = if run == 0 { photo() } else { retained.clone() };
        send(
            &mut socket,
            json!({"type":"photo_snapshot","revision":1,"photo":initial}),
        )
        .await;
        assert_eq!(
            receive(&mut socket).await,
            json!({"type":"photo_state_received","photo_id":"synthetic-photo-a","attempt_id":initial.attempt_id,"revision":1,"block_count":initial.blocks.len()})
        );
        assert_eq!(receive(&mut socket).await, json!({"type":"photo_ready"}));
        send(
            &mut socket,
            json!({"type":"photo_probe_pull","photo_id":"wrong-photo","event_id":"wrong"}),
        )
        .await;
        send(
            &mut socket,
            json!({"type":"photo_probe_pull","photo_id":"synthetic-photo-a","event_id":"pull-1"}),
        )
        .await;
        assert_eq!(
            receive(&mut socket).await,
            json!({"type":"photo_pull","photo_id":"synthetic-photo-a","event_id":"pull-1"})
        );
        send(
            &mut socket,
            json!({"type":"photo_probe_pull","photo_id":"synthetic-photo-a","event_id":"pull-1"}),
        )
        .await;
        send(
            &mut socket,
            json!({"type":"photo_snapshot","revision":2,"photo":retained}),
        )
        .await;
        assert_eq!(receive(&mut socket).await["block_count"], 2);
        send(
            &mut socket,
            json!({"type":"photo_snapshot","revision":1,"photo":null}),
        )
        .await;
        send(
            &mut socket,
            json!({"type":"photo_probe_hide","photo_id":"synthetic-photo-a"}),
        )
        .await;
        assert_eq!(
            receive(&mut socket).await,
            json!({"type":"photo_hidden","photo_id":"synthetic-photo-a"})
        );
        send(&mut socket, json!({"type":"shutdown"})).await;
        assert_eq!(
            receive(&mut socket).await,
            json!({"type":"photo_shutdown_complete"})
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if child.0.try_wait().unwrap().is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(child.0.wait().unwrap().success());
        let mut stdout = String::new();
        std::io::Read::read_to_string(&mut child.0.stdout.take().unwrap(), &mut stdout).unwrap();
        let records: Vec<Value> = stdout
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["pid"], pid);
        assert_eq!(records[0]["backend"], "probe");
        assert_eq!(records[0]["instance_id"], config.instance_id);
        assert_eq!(records[1]["type"], "photo_shutdown_complete");
        println!("actual photo child pid={pid} instance={} generation={} backend=probe exit=0 received_blocks=2", config.instance_id, config.generation);
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[tokio::test]
async fn actual_photo_process_rejects_authentication_and_unknown_messages() {
    for (name, response) in [
        ("auth-rejected", json!({"type":"auth_error"})),
        ("unknown-message", json!({"type":"snapshot","payload":{}})),
    ] {
        let (mut child, mut socket, directory) = start_child(&manifest(), name).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_auth");
        send(&mut socket, response).await;
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if child.0.try_wait().unwrap().is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(!child.0.wait().unwrap().success());
        let mut stderr = String::new();
        std::io::Read::read_to_string(&mut child.0.stderr.take().unwrap(), &mut stderr).unwrap();
        assert!(!stderr.contains("isolated-synthetic-token"));
        assert!(!stderr.contains("synthetic-metadata"));
        assert!(stderr.contains(if name == "auth-rejected" {
            "authentication_rejected"
        } else {
            "protocol_invalid"
        }));
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn actual_startup_contracts_remain_separate() {
    for (arguments, key, value) in [
        (
            vec!["--check-startup-contract"],
            "contract_version",
            json!(16),
        ),
        (
            vec!["--role", "photo", "--check-startup-contract"],
            "protocol_version",
            json!(1),
        ),
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_PuriPulyHeartOverlay"))
            .args(arguments)
            .output()
            .unwrap();
        assert!(output.status.success());
        let contract: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(contract[key], value);
        if key == "protocol_version" {
            assert_eq!(contract["role"], "photo");
            assert!(contract.get("native_presentation_retry").is_none());
        } else {
            assert_eq!(contract["native_presentation_retry"]["version"], 2);
        }
    }
}

#[tokio::test]
async fn decoded_source_is_retained_without_copy_for_retry_and_fenced_on_replacement() {
    let (mut boundary, mut worker) = PhotoImageBoundary::channel();
    let mut state = PhotoState::default();
    let mut current = requested();
    state.apply(1, Some(current.clone())).unwrap();
    let pixels = std::sync::Arc::new(image::RgbImage::new(1920, 1080));
    boundary.retain_original(
        current.photo_id.clone(),
        1,
        current.image.as_ref().unwrap().path.clone(),
        pixels.clone(),
    );
    boundary.publish(state.snapshot().unwrap());
    let input = worker.next_input().await.unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &worker.decoded_original(&input).unwrap().pixels,
        &pixels
    ));
    current.attempt_id = 2;
    state.apply(2, Some(current)).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let retry_input = worker.next_input().await.unwrap();
    assert!(worker.is_cancelled(&input));
    assert!(std::sync::Arc::ptr_eq(
        &worker.decoded_original(&retry_input).unwrap().pixels,
        &pixels
    ));
    state.apply(3, None).unwrap();
    boundary.publish(state.snapshot().unwrap());
    assert!(worker.decoded_original(&retry_input).is_none());
}

#[tokio::test]
async fn actual_photo_process_settles_retired_and_shutdown_preparations_before_exit() {
    for replace in [true, false] {
        let (mut child, mut socket, directory) = start_child(
            &manifest(),
            if replace {
                "prepare-retired"
            } else {
                "prepare-shutdown"
            },
        )
        .await;
        assert_eq!(receive(&mut socket).await["type"], "photo_auth");
        let source = directory.join("incomplete.png");
        std::fs::write(&source, b"incomplete").unwrap();
        let output = directory.join("output");
        std::fs::create_dir(&output).unwrap();
        let identity = puripuly_heart_overlay::photo::prepare::file_identity(
            &std::fs::File::open(&source).unwrap(),
        )
        .unwrap();
        let mut current = requested();
        current.path = source.to_str().unwrap().into();
        current.phase = PhotoPhase::Preparing;
        current.image = None;
        send(
            &mut socket,
            json!({"type":"photo_snapshot","revision":1,"photo":current}),
        )
        .await;
        assert_eq!(receive(&mut socket).await["type"], "photo_state_received");
        assert_eq!(receive(&mut socket).await["type"], "photo_ready");
        send(
            &mut socket,
            json!({"type":"photo_prepare","photo_id":current.photo_id,"attempt_id":1,
            "source_path":source,"output_dir":output,"expected_file_id":identity}),
        )
        .await;
        if replace {
            let mut replacement = photo();
            replacement.photo_id = "replacement".into();
            send(
                &mut socket,
                json!({"type":"photo_snapshot","revision":2,"photo":replacement}),
            )
            .await;
            assert_eq!(receive(&mut socket).await["revision"], 2);
        } else {
            send(&mut socket, json!({"type":"shutdown"})).await;
        }
        assert_eq!(
            receive(&mut socket).await,
            json!({"type":"photo_prepare_failed",
            "photo_id":current.photo_id,"attempt_id":1,"code":"preparation_cancelled"})
        );
        assert!(std::fs::read_dir(&output).unwrap().next().is_none());
        if replace {
            send(&mut socket, json!({"type":"shutdown"})).await;
        }
        assert_eq!(
            receive(&mut socket).await,
            json!({"type":"photo_shutdown_complete"})
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if child.0.try_wait().unwrap().is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(child.0.wait().unwrap().success());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
