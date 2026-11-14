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
        protocol_version: puripuly_heart_overlay::photo::manifest::PHOTO_PROTOCOL_VERSION,
        role: "photo".into(),
        bridge_url: "ws://127.0.0.1:12345".into(),
        session_token: "isolated-synthetic-token".into(),
        instance_id: "11111111-1111-4111-8111-111111111111".into(),
        generation: 1,
        backend: PhotoBackend::Probe,
        image: Default::default(),
        interaction: Default::default(),
    }
}

fn photo() -> Photo {
    Photo {
        photo_id: "synthetic-photo-a".into(),
        path: "synthetic-metadata.png".into(),
        file_size: 123,
        expected_file_id: ["1".into(), "2".into()],
        shutter_unix_ms: 1,
        attempt_id: 0,
        phase: PhotoPhase::Offer,
        offer_open: true,
        visible: false,
        image: None,
        preparations: Vec::new(),
        blocks: Vec::new(),
        excluded_blocks: Vec::new(),
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
fn experiment_manifest_validates_points_labels_and_debug_authority() {
    let mut config = manifest();
    config.interaction.wrist_handgrip_point_m = Some([0.0, 0.04373328, 0.01863404]);
    assert_eq!(config.validate(), Ok(()));
    config.interaction.wrist_transform = Some(puripuly_heart_overlay::photo::interaction::IDENTITY);
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
    config.interaction.wrist_transform = None;
    config.interaction.wrist_handgrip_point_m = Some([0.0, f32::INFINITY, 0.0]);
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
    config.interaction.wrist_handgrip_point_m = None;
    config.interaction.debug_grab_volume = true;
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
    config.image.debug_controls = true;
    assert_eq!(config.validate(), Ok(()));
    for label in ["", "one\ntwo\nthree", "one\n", "\ntwo", "tab\t", "bad\r", "한글", "\u{7f}"] {
        config.interaction.debug_case_label = Some(label.into());
        assert_eq!(config.validate(), Err(PhotoError::Manifest), "{label:?}");
    }
    config.interaction.debug_case_label = Some("a".repeat(129));
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
    for label in ["a".repeat(128), "Position A\nGrab 8 cm fixed".into()] {
        config.interaction.debug_case_label = Some(label);
        assert_eq!(config.validate(), Ok(()));
    }
    config.interaction.debug_grab_volume = false;
    config.image.debug_controls = false;
    assert_eq!(config.validate(), Err(PhotoError::Manifest));
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
            let value: Value = serde_json::from_str(&text).unwrap();
            if value["type"] == "photo_image_timing" {
                assert_eq!(value.as_object().unwrap().len(), 6);
                assert!(matches!(
                    value["stage"].as_str(),
                    Some("mask" | "erase" | "typeset" | "roi_upload" | "compose_gpu")
                ));
                let duration = value["duration_ms"].as_f64().unwrap();
                assert!(duration.is_finite() && duration >= 0.0);
                continue;
            }
            return value;
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
        "image": config.image,
        "interaction": config.interaction,
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
            json!({"type":"photo_auth","protocol_version":config.protocol_version,"role":"photo","session_token":config.session_token,"instance_id":config.instance_id,"generation":config.generation})
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

#[tokio::test]
async fn decoded_source_transfers_once_to_current_attempt_and_drops_on_replacement() {
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
        None,
    );
    boundary.publish(state.snapshot().unwrap());
    let input = worker.next_input().await.unwrap();
    let original = std::sync::Arc::downgrade(&pixels);
    drop(pixels);
    current.attempt_id = 2;
    state.apply(2, Some(current)).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let retry_input = worker.next_input().await.unwrap();
    assert!(worker.is_cancelled(&input));
    assert!(worker.take_original(&input).is_none());
    let retained = worker.take_original(&retry_input).unwrap();
    let puripuly_heart_overlay::photo::image::SourcePixels::Original(pixels) = retained.pixels else {
        panic!("Unresized source must transfer to the worker");
    };
    assert!(std::sync::Arc::ptr_eq(&pixels, &original.upgrade().unwrap()));
    assert!(worker.take_original(&retry_input).is_none());
    boundary.retain_original(
        retained.photo_id, retained.attempt_id, retained.path, pixels, None,
    );
    state.apply(3, None).unwrap();
    boundary.publish(state.snapshot().unwrap());
    assert!(worker.take_original(&retry_input).is_none());
    assert!(original.upgrade().is_none());
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
        current.expected_file_id = identity.clone();
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

#[cfg(windows)]
async fn wait_export(directory: &std::path::Path, label: &str) -> (image::RgbaImage, Value) {
    let image = directory.join(format!("{label}.png"));
    let metadata = directory.join(format!("{label}.json"));
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Ok(bytes) = std::fs::read(&metadata) {
                if let Ok(value) = serde_json::from_slice(&bytes) {
                    return (image::open(&image).unwrap().into_rgba8(), value);
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

#[cfg(windows)]
async fn image_stage(
    receiver: &mut tokio::sync::mpsc::UnboundedReceiver<Value>,
    stage: &str,
    blocks: Option<u64>,
) -> Value {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let value = receiver
                .recv()
                .await
                .expect("native stdout ended before stage");
            if value["stage"] == stage && blocks.is_none_or(|blocks| value["block_count"] == blocks)
            {
                return value;
            }
        }
    })
    .await
    .unwrap()
}

#[cfg(windows)]
#[tokio::test]
#[ignore = "requires the developer-supplied selected mask and ONNX Runtime"]
async fn actual_offscreen_native_pipeline_reuses_decode_hides_erase_accumulates_and_clears_retry() {
    actual_native_pipeline().await;
}

#[cfg(windows)]
#[tokio::test]
#[ignore = "requires SteamVR; submits hidden textures without controller input"]
async fn actual_openvr_hidden_texture_submission_survives_surface_and_child_session_shutdown() {
    use puripuly_heart_overlay::openvr::OverlayTextureSubmitter;
    use puripuly_heart_overlay::photo::{
        compositor::PhotoCompositor, processing::WorkingPhoto, runtime::PhotoOpenVrSurfaces,
    };
    use windows::core::Interface;
    let mut config = manifest();
    config.backend = PhotoBackend::Openvr;
    let mut surfaces = PhotoOpenVrSurfaces::new(&config).unwrap();
    let adapter = surfaces.photo().unwrap().output_adapter();
    let mut compositor = PhotoCompositor::new(Some(&adapter)).unwrap();
    compositor
        .set_original(
            std::sync::Arc::new(
                WorkingPhoto::new(std::sync::Arc::new(image::RgbImage::new(8, 8)), [8, 8]).unwrap(),
            ),
            "hidden-session",
        )
        .unwrap();
    assert!(compositor.render(std::time::Instant::now(), false).unwrap());
    compositor.wait_for_completion().await.unwrap();
    let texture = compositor.completed_texture().unwrap();
    surfaces.close_icon();
    surfaces
        .photo()
        .unwrap()
        .set_overlay_texture(texture.texture().as_raw())
        .unwrap();
    let (mut child, mut socket, directory) = start_child(&config, "actual-openvr-session").await;
    assert_eq!(receive(&mut socket).await["type"], "photo_auth");
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":1,"photo":null}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 1);
    assert_eq!(receive(&mut socket).await["type"], "photo_ready");
    send(&mut socket, json!({"type":"shutdown"})).await;
    assert_eq!(
        receive(&mut socket).await["type"],
        "photo_shutdown_complete"
    );
    tokio::time::timeout(Duration::from_secs(10), async {
        while child.0.try_wait().unwrap().is_none() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(child.0.wait().unwrap().success());
    surfaces
        .photo()
        .unwrap()
        .set_overlay_texture(texture.texture().as_raw())
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(windows)]
async fn actual_native_pipeline() {
    use std::io::BufRead;
    let mut config = manifest();
    config.backend = PhotoBackend::Offscreen;
    config.image.model_path = Some(
        std::env::var_os("PURIPULY_PHOTO_MASK_MODEL")
            .expect("explicit external selected model required")
            .into(),
    );
    config.image.runtime_path = std::env::var_os("PURIPULY_PHOTO_ORT_RUNTIME").map(Into::into);
    config.image.max_size = [320, 180];
    config.image.debug_controls = true;
    let name = "actual-image-pipeline";
    let root = std::env::temp_dir().join(format!("puri339-native-{}-{name}", std::process::id()));
    let exports = root.join("exports");
    config.image.export_dir = Some(exports.clone());
    let (mut child, mut socket, directory) = start_child(&config, name).await;
    let (stages, mut stage_receiver) = tokio::sync::mpsc::unbounded_channel();
    let stdout = child.0.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
            if stages.send(value).is_err() {
                break;
            }
        }
    });
    assert_eq!(receive(&mut socket).await["type"], "photo_auth");
    let mut current = photo();
    current.shutter_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let source = directory.join("capture.png");
    let mut pixels = image::RgbImage::from_pixel(320, 180, image::Rgb([230, 230, 220]));
    for y in 45..65 {
        for x in 30..180 {
            pixels.put_pixel(x, y, image::Rgb([15, 15, 20]));
        }
    }
    pixels.save(&source).unwrap();
    current.path = source.to_string_lossy().into_owned();
    current.expected_file_id = puripuly_heart_overlay::photo::prepare::file_identity(
        &std::fs::File::open(&source).unwrap(),
    ).unwrap();
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":1,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 1);
    assert_eq!(receive(&mut socket).await["type"], "photo_ready");
    assert!(!exports.exists());
    send(
        &mut socket,
        json!({"type":"photo_probe_pull","photo_id":current.photo_id,"event_id":"authorized-pull"}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["type"], "photo_pull");
    current.attempt_id = 1;
    current.phase = PhotoPhase::Preparing;
    current.visible = false;
    current.offer_open = false;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":2,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 2);
    let output = directory.join("prepared");
    std::fs::create_dir(&output).unwrap();
    let identity = puripuly_heart_overlay::photo::prepare::file_identity(
        &std::fs::File::open(&source).unwrap(),
    )
    .unwrap();
    send(&mut socket,json!({"type":"photo_prepare","photo_id":current.photo_id,"attempt_id":1,"source_path":source,"output_dir":output,"expected_file_id":identity})).await;
    let prepared = receive(&mut socket).await;
    assert_eq!(prepared["type"], "photo_prepared");
    let retained_source = std::path::PathBuf::from(prepared["source_path"].as_str().unwrap());
    std::fs::remove_file(&source).unwrap();
    current.image = Some(PhotoImage {
        path: retained_source.to_string_lossy().into_owned(),
        width: 320,
        height: 180,
    });
    current.phase = PhotoPhase::Requesting;
    current.preparations.push(PhotoPreparation {
        block_id: 0,
        bbox: [25.0, 35.0, 200.0, 80.0],
        source: None,
        source_line_count: None,
    });
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":3,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 3);
    let display = image_stage(&mut stage_receiver, "display_source_ready", None).await;
    assert_eq!(display["recovered"], false);
    image_stage(&mut stage_receiver, "bbox_candidate", None).await;
    send(
        &mut socket,
        json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"bbox"}),
    )
    .await;
    let (original, bbox_metadata) = wait_export(&exports, "bbox").await;
    assert_eq!(bbox_metadata["blocks"].as_array().unwrap().len(), 0);
    current.preparations[0].source = Some("SOURCE".into());
    current.preparations[0].source_line_count = Some(1);
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":4,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 4);
    image_stage(&mut stage_receiver, "Source", None).await;
    send(
        &mut socket,
        json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"hidden-erase"}),
    )
    .await;
    let (hidden, hidden_metadata) = wait_export(&exports, "hidden-erase").await;
    assert_eq!(hidden, original);
    assert_eq!(hidden_metadata["blocks"].as_array().unwrap().len(), 0);
    current.blocks.push(PhotoBlock {
        block_id: 0,
        bbox: current.preparations[0].bbox,
        source: "SOURCE".into(),
        translation: "Translated".into(),
    });
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":5,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 5);
    for id in 1..3 {
        current.blocks.push(PhotoBlock {
            block_id: id,
            bbox: [
                210.0,
                25.0 + id as f64 * 40.0,
                310.0,
                55.0 + id as f64 * 40.0,
            ],
            source: format!("SOURCE {id}"),
            translation: format!("Result {id}"),
        });
        send(
            &mut socket,
            json!({"type":"photo_snapshot","revision":5+id,"photo":current}),
        )
        .await;
        assert_eq!(receive(&mut socket).await["revision"], 5 + id);
    }
    let submitted = image_stage(&mut stage_receiver, "composed", Some(3)).await;
    assert_eq!(submitted["submitted"], false);
    tokio::time::sleep(Duration::from_millis(130)).await;
    send(
        &mut socket,
        json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"complete"}),
    )
    .await;
    let (complete, metadata) = wait_export(&exports, "complete").await;
    assert_ne!(complete, original);
    assert_eq!(
        metadata["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|block| block["block_id"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(metadata["original_uploads"], 1);
    assert!(metadata["composition_gpu_ms"].is_number());
    assert!(metadata["font"]["family"]
        .as_str()
        .unwrap()
        .contains("Noto Sans CJK"));
    assert_eq!(metadata["font"]["source"], "BundledNotoCjkMedium");
    assert!(metadata["font"]["hangul_ink_ratio"].as_f64().unwrap() > 0.0);
    current.phase = PhotoPhase::Completed;
    current.visible = true;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":8,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 8);
    image_stage(&mut stage_receiver, "processing_released", None).await;
    current.visible = false;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":9,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 9);
    current.visible = true;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":10,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 10);
    current.attempt_id = 2;
    current.phase = PhotoPhase::Requesting;
    current.preparations.clear();
    current.blocks.clear();
    current.first_block_drawn = false;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":11,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 11);
    let recovered = image_stage(&mut stage_receiver, "display_source_ready", None).await;
    assert_eq!(recovered["recovered"], true);
    send(
        &mut socket,
        json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"retry"}),
    )
    .await;
    let (retry, retry_metadata) = wait_export(&exports, "retry").await;
    assert_eq!(retry, original);
    assert_eq!(retry_metadata["blocks"].as_array().unwrap().len(), 0);
    assert_eq!(retry_metadata["original_uploads"], 1);
    current.visible = true;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":12,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 12);
    send(
        &mut socket,
        json!({"type":"photo_probe_hide","photo_id":current.photo_id}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["type"], "photo_hidden");
    current.visible = false;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":13,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 13);
    current.visible = true;
    send(
        &mut socket,
        json!({"type":"photo_snapshot","revision":14,"photo":current}),
    )
    .await;
    assert_eq!(receive(&mut socket).await["revision"], 14);
    send(
        &mut socket,
        json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"reopen"}),
    )
    .await;
    assert_eq!(
        wait_export(&exports, "reopen").await.1["original_uploads"],
        1
    );
    send(&mut socket, json!({"type":"shutdown"})).await;
    assert_eq!(
        receive(&mut socket).await["type"],
        "photo_shutdown_complete"
    );
    tokio::time::timeout(Duration::from_secs(10), async {
        while child.0.try_wait().unwrap().is_none() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(child.0.wait().unwrap().success());
    reader.join().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn actual_photo_process_quiesce_latches_until_an_accepted_new_photo() {
    let mut config = manifest();
    config.image.debug_controls = true;
    config.interaction.debug_case_label = Some("Position B\nGrab 8 cm fixed".into());
    let (mut child, mut socket, directory) = start_child(&config, "quiesce").await;
    assert_eq!(receive(&mut socket).await["type"], "photo_auth");
    let current = requested();
    send(&mut socket, json!({"type":"photo_snapshot","revision":1,"photo":current})).await;
    assert_eq!(receive(&mut socket).await["revision"], 1);
    assert_eq!(receive(&mut socket).await["type"], "photo_ready");
    send(&mut socket, json!({"type":"photo_debug_quiesce","request_id":"first"})).await;
    assert_eq!(receive(&mut socket).await, json!({"type":"photo_debug_quiescent","request_id":"first"}));
    send(&mut socket, json!({"type":"photo_probe_pull","photo_id":current.photo_id,"event_id":"blocked"})).await;
    send(&mut socket, json!({"type":"photo_debug_quiesce","request_id":"first"})).await;
    send(&mut socket, json!({"type":"photo_snapshot","revision":2,"photo":current})).await;
    assert_eq!(receive(&mut socket).await["revision"], 2);
    let mut replacement = current.clone();
    replacement.photo_id = "synthetic-photo-b".into();
    send(&mut socket, json!({"type":"photo_snapshot","revision":1,"photo":replacement})).await;
    send(&mut socket, json!({"type":"photo_probe_pull","photo_id":current.photo_id,"event_id":"still-blocked"})).await;
    send(&mut socket, json!({"type":"photo_probe_hide","photo_id":current.photo_id})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_hidden");
    send(&mut socket, json!({"type":"photo_snapshot","revision":3,"photo":replacement})).await;
    assert_eq!(receive(&mut socket).await["photo_id"], replacement.photo_id);
    send(&mut socket, json!({"type":"photo_debug_quiesce","request_id":"first"})).await;
    send(&mut socket, json!({"type":"photo_probe_pull","photo_id":replacement.photo_id,"event_id":"resumed"})).await;
    assert_eq!(receive(&mut socket).await["event_id"], "resumed");
    send(&mut socket, json!({"type":"photo_debug_quiesce","request_id":"second"})).await;
    assert_eq!(receive(&mut socket).await, json!({"type":"photo_debug_quiescent","request_id":"second"}));
    send(&mut socket, json!({"type":"photo_snapshot","revision":4,"photo":current})).await;
    send(&mut socket, json!({"type":"photo_probe_pull","photo_id":replacement.photo_id,"event_id":"retired-does-not-resume"})).await;
    send(&mut socket, json!({"type":"photo_probe_hide","photo_id":replacement.photo_id})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_hidden");
    send(&mut socket, json!({"type":"shutdown"})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_shutdown_complete");
    let status = child.0.wait().unwrap();
    assert!(status.success());
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn actual_photo_process_rejects_unauthorized_or_invalid_quiesce() {
    for (index, enabled, label, request) in [
        (0, false, None, "id".to_owned()),
        (1, true, None, "id".to_owned()),
        (2, true, Some("Case"), String::new()),
        (3, true, Some("Case"), "a".repeat(129)),
        (4, true, Some("Case"), "한글".to_owned()),
    ] {
        let mut config = manifest();
        config.image.debug_controls = enabled;
        config.interaction.debug_case_label = label.map(str::to_owned);
        let (mut child, mut socket, directory) = start_child(&config, &format!("quiesce-invalid-{index}")).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_auth");
        send(&mut socket, json!({"type":"photo_snapshot","revision":1,"photo":null})).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_state_received");
        assert_eq!(receive(&mut socket).await["type"], "photo_ready");
        send(&mut socket, json!({"type":"photo_debug_quiesce","request_id":request})).await;
        use std::io::Read;
        assert!(!child.0.wait().unwrap().success());
        let mut stderr = String::new();
        child.0.stderr.as_mut().unwrap().read_to_string(&mut stderr).unwrap();
        assert!(stderr.contains("protocol_invalid"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[test]
fn photo_protocol_requires_current_schema_and_immutable_capture_identity() {
    let mut old_manifest = manifest();
    old_manifest.protocol_version = 2;
    assert_eq!(old_manifest.validate(), Err(PhotoError::Manifest));
    let mut document = serde_json::to_value(photo()).unwrap();
    document.as_object_mut().unwrap().remove("expected_file_id");
    assert!(serde_json::from_value::<Photo>(document).is_err());
    let mut state = PhotoState::default();
    state.apply(1, Some(photo())).unwrap();
    let mut changed = photo();
    changed.expected_file_id[1] = "3".into();
    assert_eq!(state.apply(2, Some(changed)), Err(PhotoError::Protocol));
    let mut invalid = photo();
    invalid.expected_file_id[0] = "not-a-decimal-id".into();
    assert_eq!(PhotoState::default().apply(1, Some(invalid)), Err(PhotoError::Protocol));
}

#[cfg(windows)]
#[tokio::test]
async fn actual_offscreen_grab_preview_precedes_authorization_and_reuses_its_pixels() {
    actual_offscreen_preview_handoff(true).await;
}

#[cfg(windows)]
#[tokio::test]
async fn actual_offscreen_standalone_snapshot_reuses_preview_through_hide_and_reopen() {
    actual_offscreen_preview_handoff(false).await;
}

#[cfg(windows)]
async fn actual_offscreen_preview_handoff(prepare_private: bool) {
    use std::io::BufRead;
    let name = if prepare_private { "prepared" } else { "standalone" };
    let mut config = manifest();
    config.backend = PhotoBackend::Offscreen;
    config.image.debug_controls = true;
    config.image.max_size = if prepare_private { [320, 180] } else { [1920, 1080] };
    config.image.model_path = Some("must-not-load-for-original-preview.onnx".into());
    config.interaction.debug_case_label = Some("Watch candidate\nRadius fixed: 10 cm".into());
    let external = std::env::var_os("PURIPULY_PHOTO341_EVIDENCE").map(std::path::PathBuf::from);
    let exports = external.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("puri341-local-preview-{}-{name}", std::process::id()))
    }).join(format!("offscreen-child-{name}"));
    config.image.export_dir = Some(exports.clone());
    let (mut child, mut socket, directory) = start_child(&config, &format!("grab-local-preview-{name}")).await;
    let (stages, mut stage_receiver) = tokio::sync::mpsc::unbounded_channel();
    let stdout = child.0.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut log = Vec::new();
        for line in std::io::BufReader::new(stdout).lines() {
            let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
            log.push(value.clone());
            let _ = stages.send(value);
        }
        log
    });
    receive(&mut socket).await;
    std::fs::create_dir_all(&exports).unwrap();
    let source = exports.join("capture.png");
    let mut pixels = image::RgbImage::new(320, 180);
    for (x, y, pixel) in pixels.enumerate_pixels_mut() {
        *pixel = image::Rgb([if x < 160 { 210 } else { 35 }, if y < 90 { 55 } else { 195 }, 85]);
    }
    pixels.save(&source).unwrap();
    let mut current = photo();
    current.path = source.to_string_lossy().into_owned();
    current.expected_file_id = puripuly_heart_overlay::photo::prepare::file_identity(
        &std::fs::File::open(&source).unwrap(),
    ).unwrap();
    current.file_size = std::fs::metadata(&source).unwrap().len();
    current.shutter_unix_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
    std::fs::write(exports.join("capture.json"), serde_json::to_vec_pretty(&json!({
        "path":current.path,"expected_file_id":current.expected_file_id,"width":320,"height":180
    })).unwrap()).unwrap();
    send(&mut socket, json!({"type":"photo_snapshot","revision":1,"photo":current})).await;
    assert_eq!(receive(&mut socket).await["revision"], 1);
    assert_eq!(receive(&mut socket).await["type"], "photo_ready");
    let output = directory.join("prepared");
    std::fs::create_dir(&output).unwrap();
    send(&mut socket, json!({"type":"photo_debug_preview_source","photo_id":"wrong-photo"})).await;
    send(&mut socket, json!({"type":"photo_debug_preview_source","photo_id":current.photo_id})).await;
    let ready = image_stage(&mut stage_receiver, "display_source_ready", None).await;
    assert_eq!(ready["local_preview"], true);
    assert_eq!(ready["before_authorization"], true);
    assert!(tokio::time::timeout(Duration::from_millis(20), socket.next()).await.is_err());
    assert!(std::fs::read_dir(&output).unwrap().next().is_none());
    send(&mut socket, json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"before-pull"})).await;
    let (preview, metadata) = wait_export(&exports, "before-pull").await;
    assert_eq!(preview.dimensions(), (config.image.max_size[0], config.image.max_size[1]));
    let expected = if prepare_private {
        pixels
    } else {
        image::imageops::resize(&pixels, 1920, 1080, image::imageops::FilterType::Lanczos3)
    };
    assert!(preview.pixels().zip(expected.pixels()).all(|(actual, expected)| actual.0[..3] == expected.0 && actual[3] == 255));
    assert_eq!(metadata["original_uploads"], 1);
    assert_eq!(metadata["blocks"], json!([]));
    send(&mut socket, json!({"type":"photo_prepare","photo_id":current.photo_id,"attempt_id":1,
        "source_path":current.path,"output_dir":output,"expected_file_id":current.expected_file_id})).await;
    assert_eq!(receive(&mut socket).await["code"], "preparation_not_current");
    assert!(std::fs::read_dir(&output).unwrap().next().is_none());
    send(&mut socket, json!({"type":"photo_probe_pull","photo_id":current.photo_id,"event_id":"software-authorized-pull"})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_pull");
    current.attempt_id = 1;
    current.phase = if prepare_private { PhotoPhase::Preparing } else { PhotoPhase::Completed };
    current.offer_open = false;
    current.visible = true;
    if !prepare_private {
        current.image = Some(PhotoImage { path:current.path.clone(), width:320, height:180 });
    }
    send(&mut socket, json!({"type":"photo_snapshot","revision":2,"photo":current})).await;
    assert_eq!(receive(&mut socket).await["revision"], 2);
    let mut revision = 2;
    if prepare_private {
        send(&mut socket, json!({"type":"photo_prepare","photo_id":current.photo_id,"attempt_id":1,
            "source_path":current.path,"output_dir":output,"expected_file_id":current.expected_file_id})).await;
        let prepared = receive(&mut socket).await;
        assert_eq!(prepared["type"], "photo_prepared");
        assert_eq!(prepared["decode_ms"], 0.0);
        assert_eq!(std::fs::read(prepared["source_path"].as_str().unwrap()).unwrap(), std::fs::read(&source).unwrap());
        current.image = Some(PhotoImage {
            path: prepared["source_path"].as_str().unwrap().into(), width:320, height:180,
        });
        current.phase = PhotoPhase::Requesting;
        revision += 1;
        send(&mut socket, json!({"type":"photo_snapshot","revision":revision,"photo":current})).await;
        assert_eq!(receive(&mut socket).await["revision"], revision);
    }
    let authorized_ready = image_stage(&mut stage_receiver, "display_source_ready", None).await;
    assert_eq!(authorized_ready["recovered"], false);
    let after_label = if prepare_private { "after-prepare" } else { "after-pull" };
    send(&mut socket, json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":after_label})).await;
    let (after, metadata) = wait_export(&exports, after_label).await;
    assert_eq!(after, preview);
    assert_eq!(metadata["original_uploads"], 1);
    if !prepare_private {
        send(&mut socket, json!({"type":"photo_probe_hide","photo_id":current.photo_id})).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_hidden");
        current.visible = false;
        revision += 1;
        send(&mut socket, json!({"type":"photo_snapshot","revision":revision,"photo":current})).await;
        assert_eq!(receive(&mut socket).await["revision"], revision);
        send(&mut socket, json!({"type":"photo_probe_pull","photo_id":current.photo_id,"event_id":"software-reopen"})).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_pull");
        current.visible = true;
        revision += 1;
        send(&mut socket, json!({"type":"photo_snapshot","revision":revision,"photo":current})).await;
        assert_eq!(receive(&mut socket).await["revision"], revision);
        send(&mut socket, json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":"reopened"})).await;
        let (reopened, metadata) = wait_export(&exports, "reopened").await;
        assert_eq!(reopened, preview);
        assert_eq!(metadata["original_uploads"], 1);
        assert!(std::fs::read_dir(&output).unwrap().next().is_none());
    }
    send(&mut socket, json!({"type":"photo_debug_quiesce","request_id":"preview-complete"})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_debug_quiescent");
    send(&mut socket, json!({"type":"shutdown"})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_shutdown_complete");
    assert!(child.0.wait().unwrap().success());
    let log = reader.join().unwrap();
    std::fs::write(exports.join("native-log.json"), serde_json::to_vec_pretty(&log).unwrap()).unwrap();
    assert_eq!(log.iter().filter(|line| line["type"] == "photo_image_prepared").count(), usize::from(prepare_private));
    assert!(!log.iter().any(|line| line["stage"] == "display_source_ready" && line["recovered"] == true));
    std::fs::remove_dir_all(directory).unwrap();
    if external.is_none() { std::fs::remove_dir_all(exports.parent().unwrap()).unwrap(); }
}

#[cfg(windows)]
#[tokio::test]
async fn preview_source_control_rejects_non_offscreen_or_non_debug_children() {
    for (index, backend, debug) in [
        (0, PhotoBackend::Probe, false), (1, PhotoBackend::Probe, true),
        (2, PhotoBackend::Offscreen, false),
    ] {
        let mut config = manifest();
        config.backend = backend;
        config.image.debug_controls = debug;
        let (mut child, mut socket, directory) = start_child(&config, &format!("preview-gate-{index}")).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_auth");
        send(&mut socket, json!({"type":"photo_snapshot","revision":1,"photo":null})).await;
        assert_eq!(receive(&mut socket).await["type"], "photo_state_received");
        assert_eq!(receive(&mut socket).await["type"], "photo_ready");
        send(&mut socket, json!({"type":"photo_debug_preview_source","photo_id":"synthetic"})).await;
        assert!(!child.0.wait().unwrap().success());
        use std::io::Read;
        let mut stderr = String::new();
        child.0.stderr.as_mut().unwrap().read_to_string(&mut stderr).unwrap();
        assert!(stderr.contains("protocol_invalid"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[tokio::test]
async fn local_original_survives_offer_preparing_and_matching_raw_image_only() {
    use puripuly_heart_overlay::photo::processing::WorkingPhoto;
    use std::sync::Arc;
    let (mut boundary, mut worker) = PhotoImageBoundary::channel();
    let mut state = PhotoState::default();
    let mut current = photo();
    state.apply(1, Some(current.clone())).unwrap();
    let pixels = Arc::new(image::RgbImage::new(640, 360));
    let working = Arc::new(WorkingPhoto::new(pixels.clone(), [320, 180]).unwrap());
    boundary.retain_original(current.photo_id.clone(), 0, current.path.clone(), pixels.clone(), Some(working.clone()));
    let original = Arc::downgrade(&pixels);
    drop(pixels);
    assert!(original.upgrade().is_none());
    boundary.publish(state.snapshot().unwrap());
    let offer = worker.next_input().await.unwrap();
    assert!(worker.take_original(&offer).is_none());
    current.attempt_id = 1;
    current.phase = PhotoPhase::Preparing;
    current.offer_open = false;
    state.apply(2, Some(current.clone())).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let preparing = worker.next_input().await.unwrap();
    assert!(worker.take_original(&preparing).is_none());
    current.phase = PhotoPhase::Completed;
    current.image = Some(PhotoImage { path:current.path.clone(), width:640, height:360 });
    state.apply(3, Some(current.clone())).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let completed = worker.next_input().await.unwrap();
    let retained = worker.take_original(&completed).unwrap();
    let puripuly_heart_overlay::photo::image::SourcePixels::Working(transferred) = retained.pixels else {
        panic!("Only display pixels must cross the image boundary");
    };
    assert!(Arc::ptr_eq(&transferred, &working));
    assert!(worker.take_original(&completed).is_none());
    current.attempt_id = 2;
    current.image.as_mut().unwrap().path = "different-source.png".into();
    state.apply(4, Some(current)).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let different = worker.next_input().await.unwrap();
    assert!(worker.take_original(&different).is_none());
    state.apply(5, None).unwrap();
    boundary.publish(state.snapshot().unwrap());
    assert!(worker.take_original(&completed).is_none());
}

#[tokio::test]
async fn excluded_preparations_cannot_resurrect_or_publish_late_results() {
    let (mut boundary, mut worker) = PhotoImageBoundary::channel();
    let mut state = PhotoState::default();
    let mut current = requested();
    for id in [0, 1] {
        current.preparations.push(PhotoPreparation {
            block_id: id,
            bbox: block(id).bbox,
            source: Some("A\nB".into()),
            source_line_count: Some(2),
        });
    }
    state.apply(1, Some(current.clone())).unwrap();
    boundary.publish(state.snapshot().unwrap());
    let old_input = worker.next_input().await.unwrap();
    current.preparations.remove(0);
    assert_eq!(state.apply(2, Some(current.clone())), Err(PhotoError::Protocol));
    current.excluded_blocks.push(0);
    current.blocks.push(block(1));
    current.phase = PhotoPhase::Partial;
    state.apply(2, Some(current.clone())).unwrap();
    assert!(state.photo().unwrap().can_pull());
    boundary.publish(state.snapshot().unwrap());
    let late = PhotoImageResult {
        input: old_input,
        output: PhotoImageOutput::Prepared {
            block_id: 0,
            stage: PreparationStage::Source,
            elapsed_ms: 1.0,
        },
    };
    assert!(!boundary.accept(&late, state.snapshot().unwrap()));
    let mut resurrected = current.clone();
    resurrected.excluded_blocks.clear();
    resurrected.blocks.push(block(0));
    assert_eq!(state.apply(3, Some(resurrected)), Err(PhotoError::Protocol));
    let mut discarded_completed = current.clone();
    discarded_completed.excluded_blocks.push(1);
    assert_eq!(state.apply(3, Some(discarded_completed)), Err(PhotoError::Protocol));
    current.attempt_id += 1;
    current.excluded_blocks.clear();
    current.preparations.clear();
    current.blocks = vec![block(0)];
    current.phase = PhotoPhase::CompletedCorrected;
    state.apply(3, Some(current)).unwrap();
    assert!(state.photo().unwrap().can_pull());
    assert_eq!(state.photo().unwrap().blocks[0].block_id, 0);
}

#[test]
fn no_text_photo_remains_reopenable_without_a_translation_layer() {
    let mut state = PhotoState::default();
    let mut current = requested();
    current.phase = PhotoPhase::NoText;
    state.apply(1, Some(current)).unwrap();
    assert!(state.photo().unwrap().can_pull());
    assert!(state.photo().unwrap().blocks.is_empty());
}

#[cfg(windows)]
#[tokio::test]
async fn actual_same_photo_retry_with_a_distinct_equal_size_source_replaces_gpu_pixels() {
    use std::io::BufRead;
    let mut config = manifest();
    config.backend = PhotoBackend::Offscreen;
    config.image.debug_controls = true;
    config.image.max_size = [64, 48];
    config.image.model_path = Some("not-needed-for-original-only-retry.onnx".into());
    let name = "equal-size-source-retry";
    let root = std::env::temp_dir().join(format!("puri339-native-{}-{name}", std::process::id()));
    let exports = root.join("exports");
    config.image.export_dir = Some(exports.clone());
    let (mut child, mut socket, directory) = start_child(&config, name).await;
    let (stages, mut stage_receiver) = tokio::sync::mpsc::unbounded_channel();
    let stdout = child.0.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
            let _ = stages.send(value);
        }
    });
    assert_eq!(receive(&mut socket).await["type"], "photo_auth");
    let first = directory.join("first.png");
    let second = directory.join("second.png");
    image::RgbImage::from_pixel(64, 48, image::Rgb([210, 30, 40])).save(&first).unwrap();
    image::RgbImage::from_pixel(64, 48, image::Rgb([20, 70, 220])).save(&second).unwrap();
    let mut current = photo();
    current.path = first.to_string_lossy().into_owned();
    current.file_size = std::fs::metadata(&first).unwrap().len();
    current.expected_file_id = puripuly_heart_overlay::photo::prepare::file_identity(
        &std::fs::File::open(&first).unwrap(),
    ).unwrap();
    current.shutter_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
    current.phase = PhotoPhase::Completed;
    current.offer_open = false;
    current.visible = true;
    for (attempt, source, expected, label) in [
        (1, &first, [210, 30, 40, 255], "first"),
        (2, &second, [20, 70, 220, 255], "second"),
    ] {
        current.attempt_id = attempt;
        current.image = Some(PhotoImage {
            path: source.to_string_lossy().into_owned(), width: 64, height: 48,
        });
        send(&mut socket, json!({"type":"photo_snapshot","revision":attempt,"photo":current})).await;
        assert_eq!(receive(&mut socket).await["revision"], attempt);
        if attempt == 1 {
            assert_eq!(receive(&mut socket).await["type"], "photo_ready");
        }
        image_stage(&mut stage_receiver, "display_source_ready", None).await;
        send(&mut socket, json!({"type":"photo_debug_export","photo_id":current.photo_id,"label":label})).await;
        let (pixels, metadata) = wait_export(&exports, label).await;
        assert!(pixels.pixels().all(|pixel| pixel.0 == expected));
        assert_eq!(metadata["original_uploads"], attempt);
        assert_eq!(metadata["blocks"], json!([]));
    }
    send(&mut socket, json!({"type":"shutdown"})).await;
    assert_eq!(receive(&mut socket).await["type"], "photo_shutdown_complete");
    assert!(child.0.wait().unwrap().success());
    reader.join().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn limited_offer_is_never_pullable_and_cannot_claim_an_open_or_visible_request() {
    let mut limited = photo();
    limited.phase = PhotoPhase::Limited;
    limited.offer_open = false;
    assert!(!limited.can_pull());
    PhotoState::default().apply(1, Some(limited.clone())).unwrap();
    for invalid in [
        Photo { offer_open: true, ..limited.clone() },
        Photo { visible: true, ..limited.clone() },
        Photo { attempt_id: 1, ..limited },
    ] {
        assert_eq!(PhotoState::default().apply(1, Some(invalid)), Err(PhotoError::Protocol));
    }
}
