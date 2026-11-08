use super::*;

#[test]
fn feedback_requires_live_draw_and_is_once_per_attempt_even_after_snapshot_ack() {
    let manifest = PhotoManifest {
        protocol_version: PHOTO_PROTOCOL_VERSION,
        role: "photo".into(),
        bridge_url: "ws://127.0.0.1:12345".into(),
        session_token: "synthetic".into(),
        instance_id: "11111111-1111-4111-8111-111111111111".into(),
        generation: 1,
        backend: PhotoBackend::Probe,
        image: Default::default(),
        interaction: Default::default(),
    };
    let (mut runtime, _worker) = PhotoRuntime::new(&manifest).unwrap();
    let photo = serde_json::from_value(json!({
        "photo_id":"synthetic", "path":"capture.png", "file_size":1, "shutter_unix_ms":1,
        "expected_file_id":["1","2"],
        "attempt_id":1,"phase":"completed","offer_open":false,"visible":true,
        "image":{"path":"snapshot.png","width":1280,"height":720},"preparations":[],
        "blocks":[{"block_id":0,"bbox":[1,2,100,50],"source":"A","translation":"B"}],
        "failure_code":null,"first_block_drawn":true,
    }))
    .unwrap();
    runtime.state.apply(1, Some(photo)).unwrap();
    runtime.drawn_reported = true;
    runtime.grant_haptic("synthetic", 1, 0);
    assert!(!runtime.take_haptic_grant());
    runtime.live_drawn_block = Some(0);
    runtime.grant_haptic("old-photo", 1, 0);
    runtime.grant_haptic("synthetic", 2, 0);
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(!runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 1, 0);
    assert!(runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 1, 0);
    assert!(!runtime.take_haptic_grant());
}

#[test]
fn feedback_tracks_first_drawable_block_after_kept_prefix() {
    let manifest: PhotoManifest = serde_json::from_value(json!({
        "protocol_version":2,"role":"photo","bridge_url":"ws://127.0.0.1:12345",
        "session_token":"synthetic","instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1,"backend":"probe"
    }))
    .unwrap();
    let (mut runtime, _worker) = PhotoRuntime::new(&manifest).unwrap();
    let photo = serde_json::from_value(json!({
        "photo_id":"synthetic","path":"capture.png","file_size":1,"shutter_unix_ms":1,
        "expected_file_id":["1","2"],
        "attempt_id":1,"phase":"completed","offer_open":false,"visible":true,
        "image":{"path":"snapshot.png","width":1280,"height":720},"preparations":[],
        "blocks":[
            {"block_id":0,"bbox":[1,2,100,50],"source":"Brand","translation":"Brand"},
            {"block_id":1,"bbox":[200,2,300,50],"source":"Source","translation":"Translation"}
        ],"failure_code":null,"first_block_drawn":false
    }))
    .unwrap();
    runtime.state.apply(1, Some(photo)).unwrap();
    runtime.live_drawn_block = Some(1);
    runtime.grant_haptic("synthetic", 1, 0);
    assert!(!runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(!runtime.take_haptic_grant());
}

#[cfg(windows)]
#[tokio::test]
async fn windows_graphics_photo341_icons_and_held_preview_readback() {
    use super::super::typeset::PhotoTypesetter;
    let manifest: PhotoManifest = serde_json::from_value(json!({
        "protocol_version":2,"role":"photo","bridge_url":"ws://127.0.0.1:12345",
        "session_token":"synthetic","instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1,"backend":"offscreen",
        "image":{"debug_controls":true},
        "interaction":{"grab_m":0.12,"debug_grab_volume":true,"debug_case_label":"Radius 12 cm\nPosition B fixed"}
    }))
    .unwrap();
    let mut default_manifest = manifest.clone();
    default_manifest.interaction = Default::default();
    default_manifest.image.debug_controls = false;
    let (default_runtime, _) = PhotoRuntime::new(&default_manifest).unwrap();
    assert!(default_runtime.grab_ring.is_none());
    assert!(default_runtime.case_labels.is_none());
    drop(default_runtime);
    let (mut runtime, _worker) = PhotoRuntime::new(&manifest).unwrap();
    let supplied = std::env::var_os("PURIPULY_PHOTO341_EVIDENCE");
    let directory = supplied
        .clone()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("puri341-gpu-{}", std::process::id()))
        });
    std::fs::create_dir_all(&directory).unwrap();
    let compositor = runtime.compositor.as_mut().unwrap();
    compositor.ready().await.unwrap();
    let mut icons = Vec::new();
    for (index, name) in ["idle", "grab", "error-x"].into_iter().enumerate() {
        let path = directory.join(format!("icon-{name}.png"));
        compositor
            .save_texture(&runtime.icons.as_ref().unwrap()[index], 128, 128, &path)
            .unwrap();
        let image = image::open(path).unwrap().into_rgba8();
        assert!(image.pixels().any(|pixel| pixel[3] == 0));
        assert!(image.pixels().any(|pixel| pixel[3] == 255));
        icons.push(image);
    }
    assert_ne!(icons[0], icons[1]);
    assert_ne!(icons[0], icons[2]);
    let ring_path = directory.join("grab-ring.png");
    compositor.save_texture(runtime.grab_ring.as_ref().unwrap(), 256, 256, &ring_path).unwrap();
    let ring = image::open(ring_path).unwrap().into_rgba8();
    assert_eq!(ring.get_pixel(128, 128)[3], 0);
    assert_eq!(ring.get_pixel(0, 0)[3], 0);
    let mut weighted_radius = 0.0_f64;
    let mut weight = 0.0_f64;
    let mut ink = 0;
    for (x, y, pixel) in ring.enumerate_pixels() {
        assert!(pixel.0[..3].iter().all(|channel| *channel <= pixel[3]));
        let radius = ((x as f64 + 0.5 - 128.0).powi(2) + (y as f64 + 0.5 - 128.0).powi(2)).sqrt();
        if pixel[3] > 0 {
            assert!((radius - 112.0).abs() < 1.5);
            assert!(pixel[3] <= 150);
            ink += 1;
            weighted_radius += radius * f64::from(pixel[3]);
            weight += f64::from(pixel[3]);
        }
    }
    assert!((1200..2400).contains(&ink));
    assert!((weighted_radius / weight - 112.0).abs() < 0.02);
    let labels = runtime.case_labels.as_ref().unwrap();
    let mut readbacks = Vec::new();
    for (index, status) in ["case", "draining"].into_iter().enumerate() {
        let path = directory.join(format!("label-{status}.png"));
        compositor.save_texture(&labels[index], 1024, 256, &path).unwrap();
        let label = image::open(path).unwrap().into_rgba8();
        assert_eq!(label.get_pixel(0, 0)[3], 0);
        assert!(label.enumerate_pixels().any(|(_, y, pixel)| y < 150 && pixel[3] > 200));
        assert!(label.pixels().all(|pixel| pixel.0[..3].iter().all(|channel| *channel <= pixel[3])));
        assert_eq!(label.enumerate_pixels().any(|(_, y, pixel)| y > 160 && pixel[3] > 200), index == 1);
        readbacks.push(label);
    }
    assert_ne!(readbacks[0], readbacks[1]);
    assert_eq!(
        image::imageops::crop_imm(&readbacks[1], 0, 0, 1024, 150).to_image(),
        image::imageops::crop_imm(&readbacks[0], 0, 0, 1024, 150).to_image(),
        "Draining must retain the complete case and fixed-factor pixels",
    );
    let (working, region) = synthetic_drawable();
    compositor.set_original(working.clone()).unwrap();
    let now = Instant::now();
    compositor.render(now, false).unwrap();
    compositor.ready().await.unwrap();
    compositor
        .save(&directory.join("held-original.png"))
        .unwrap();
    compositor.add_block(0, None, None, 0.0, now).unwrap();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter
        .typeset(&working, &region, "Temporary photo")
        .unwrap();
    assert!(!text.runs.is_empty());
    compositor
        .add_block(1, Some(region), Some(text), 0.0, now)
        .unwrap();
    assert!(compositor.first_visible_block(now).is_none());
    compositor
        .render(now + Duration::from_millis(50), false)
        .unwrap();
    compositor.ready().await.unwrap();
    compositor
        .save(&directory.join("held-block-half.png"))
        .unwrap();
    compositor
        .render(now + Duration::from_millis(100), false)
        .unwrap();
    compositor.ready().await.unwrap();
    compositor
        .save(&directory.join("held-translated.png"))
        .unwrap();
    assert!(!compositor.needs_frame(now + Duration::from_millis(150)));
    let visible = compositor.first_visible_block(now + Duration::from_millis(150));
    assert_eq!(visible, Some(1));
    let original = image::open(directory.join("held-original.png")).unwrap();
    let translated = image::open(directory.join("held-translated.png")).unwrap();
    assert_ne!(original, translated);
    let _late_label = compositor.case_label_texture("RADIUS 2/3: 12 cm\nPosition fixed: B", true).unwrap();
    compositor.render(now + Duration::from_millis(150), false).unwrap();
    compositor.ready().await.unwrap();
    compositor.save(&directory.join("held-after-label.png")).unwrap();
    assert_eq!(image::open(directory.join("held-after-label.png")).unwrap(), translated);
    if supplied.is_none() {
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[cfg(windows)]
#[tokio::test]
#[ignore = "requires an already-running SteamVR runtime; creates hidden surfaces only"]
async fn windows_graphics_photo341_quiet_texture_handoff_reaches_steamvr() {
    use std::ffi::{CStr, CString};

    let session = OpenVrSession::new().unwrap();
    let identity = format!("photo341.quiet-handoff.p{}", std::process::id());
    let surface = OpenVrOverlay::in_session(&session, &identity).unwrap();
    let adapter = surface.output_adapter();
    let mut first = PhotoCompositor::new(Some(&adapter)).unwrap();
    let mut second = PhotoCompositor::new(Some(&adapter)).unwrap();
    let idle = icon::pixels(IconState::Idle);
    let grab = icon::pixels(IconState::Grab);
    let failed = icon::pixels(IconState::Failed);
    let textures = [
        first.icon_texture(&idle, icon::ICON_SIZE).unwrap(),
        first.icon_texture(&grab, icon::ICON_SIZE).unwrap(),
        second.icon_texture(&failed, icon::ICON_SIZE).unwrap(),
    ];
    first.ready().await.unwrap();
    second.ready().await.unwrap();
    first.clear_photo();
    first.invalidate();
    assert!(!first.needs_frame(Instant::now()));
    assert!(first.render(Instant::now(), false).unwrap().is_none());

    let version = CStr::from_bytes_with_nul(openvr_sys::IVROverlay_Version).unwrap();
    let version = CString::new(format!("FnTable:{}", version.to_str().unwrap())).unwrap();
    let mut error = 0;
    let pointer = unsafe { openvr_sys::VR_GetGenericInterface(version.as_ptr(), &mut error) };
    assert_eq!(error, 0);
    assert_ne!(pointer, 0);
    let api = unsafe { &*(pointer as *const openvr_sys::VR_IVROverlay_FnTable) };
    let key = CString::new(format!("com.puripuly.heart.overlay.{identity}")).unwrap();
    let mut handle = 0;
    assert_eq!(
        unsafe { api.FindOverlay.unwrap()(key.as_ptr().cast_mut(), &mut handle) },
        0
    );

    for (stage, compositor, texture, expected) in [
        ("initial-idle", &first, &textures[0], &idle),
        ("changed-grab", &first, &textures[1], &grab),
        ("different-device", &second, &textures[2], &failed),
        ("restored-idle", &first, &textures[0], &idle),
    ] {
        let expected_bgra: Vec<u8> = expected
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
            .collect();
        compositor.submit_texture(&surface, texture).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut pixels = vec![0; expected.len()];
        let mut width = 0;
        let mut height = 0;
        loop {
            let error = unsafe {
                api.GetOverlayImageData.unwrap()(
                    handle,
                    pixels.as_mut_ptr().cast(),
                    pixels.len() as u32,
                    &mut width,
                    &mut height,
                )
            };
            if error == 0 && pixels == expected_bgra {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "{stage}: SteamVR texture differs: error={error}, dimensions={width}x{height}, nonzero_alpha={}, first_mismatch={:?}",
                pixels.chunks_exact(4).filter(|pixel| pixel[3] != 0).count(),
                pixels.chunks_exact(4).zip(expected_bgra.chunks_exact(4)).find(|(actual, expected)| actual != expected)
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!((width, height), (icon::ICON_SIZE, icon::ICON_SIZE));
        assert_eq!(surface.observed_overlay_visible(), Some(false));
    }
}

#[cfg(windows)]
fn synthetic_drawable() -> (
    std::sync::Arc<super::super::processing::WorkingPhoto>,
    super::super::processing::PreparedRegion,
) {
    use super::super::processing::{ErasePatch, PixelRect, PreparedRegion, Rect, WorkingPhoto};
    use std::sync::Arc;
    let working = Arc::new(
        WorkingPhoto::new(
            Arc::new(image::RgbImage::from_pixel(
                320,
                180,
                image::Rgb([35, 45, 60]),
            )),
            [320, 180],
        )
        .unwrap(),
    );
    let region = PreparedRegion {
        bbox: Rect {
            left: 30.0,
            top: 45.0,
            right: 290.0,
            bottom: 125.0,
        },
        roi: PixelRect {
            x: 30,
            y: 45,
            width: 260,
            height: 80,
        },
        core: image::GrayImage::from_pixel(260, 80, image::Luma([255])),
        glyph_size: 40.0,
        fill: [245, 245, 240],
        plate: false,
        erase: Some(ErasePatch {
            rect: PixelRect {
                x: 30,
                y: 45,
                width: 260,
                height: 80,
            },
            pixels: image::RgbaImage::from_pixel(260, 80, image::Rgba([70, 90, 110, 255])),
        }),
        mask_ms: 0.0,
        erase_ms: 0.0,
    };
    (working, region)
}

#[cfg(windows)]
impl PhotoRuntime {
    pub(crate) async fn assert_controlled_held_render_frame(
        icon: OpenVrOverlay,
        photo_surface: OpenVrOverlay,
        session: OpenVrSession,
        photo_width_m: Option<f32>,
        mut assert_held_width: impl FnMut(),
        mut fail_photo_show: impl FnMut(bool),
        mut set_input: impl FnMut(bool, f32),
        mut assert_pulses: impl FnMut(&[(u32, u32, u16)]),
    ) {
        use super::super::interaction::IDENTITY;
        use super::super::state::{Photo, PhotoPhase};
        use super::super::typeset::PhotoTypesetter;
        use futures_util::{SinkExt, StreamExt};
        use tokio::net::{TcpListener, TcpStream};
        use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};

        async fn message(socket: &mut WebSocketStream<TcpStream>) -> serde_json::Value {
            let incoming = tokio::time::timeout(Duration::from_secs(2), socket.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            serde_json::from_str(incoming.to_text().unwrap()).unwrap()
        }

        async fn no_report(socket: &mut WebSocketStream<TcpStream>) {
            assert!(
                tokio::time::timeout(Duration::from_millis(20), socket.next())
                    .await
                    .is_err()
            );
        }

        async fn publish(
            runtime: &mut PhotoRuntime,
            bridge: &mut PhotoBridge,
            socket: &mut WebSocketStream<TcpStream>,
            revision: u64,
            photo: &Photo,
        ) {
            socket
                .send(Message::Text(
                    json!({
                        "type":"photo_snapshot", "revision":revision, "photo":photo
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            let incoming = bridge.next().await.unwrap();
            runtime.receive(incoming, bridge).await.unwrap();
            let receipt = message(socket).await;
            assert_eq!(receipt["type"], "photo_state_received");
            assert_eq!(receipt["revision"], revision);
        }

        async fn grant(
            runtime: &mut PhotoRuntime,
            bridge: &mut PhotoBridge,
            socket: &mut WebSocketStream<TcpStream>,
            photo_id: &str,
            attempt_id: u64,
            block_id: u64,
        ) {
            socket
                .send(Message::Text(
                    json!({
                        "type":"photo_haptic", "photo_id":photo_id,
                        "attempt_id":attempt_id, "block_id":block_id
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            let incoming = bridge.next().await.unwrap();
            runtime.receive(incoming, bridge).await.unwrap();
        }

        fn compose_blocks(runtime: &mut PhotoRuntime) {
            let (working, region) = synthetic_drawable();
            let text = PhotoTypesetter::new()
                .unwrap()
                .typeset(&working, &region, "Temporary photo")
                .unwrap();
            assert!(!text.runs.is_empty());
            let compositor = runtime.compositor.as_mut().unwrap();
            compositor.set_original(working).unwrap();
            let faded = Instant::now() - Duration::from_millis(150);
            compositor.add_block(0, None, None, 0.0, faded).unwrap();
            compositor
                .add_block(1, Some(region), Some(text), 0.0, faded)
                .unwrap();
        }

        fn hold(runtime: &mut PhotoRuntime, clock: Instant, separation: f32) {
            let photo = runtime.state.photo().unwrap();
            for (ms, grip, distance) in [
                (0, false, 0.0),
                (50, false, 0.0),
                (350, false, 0.0),
                (360, true, 0.0),
                (380, true, separation),
                (455, true, separation),
            ] {
                let mut head = IDENTITY;
                head[2][3] = 0.5;
                let mut left = IDENTITY;
                left[0][3] = distance;
                runtime.interaction.step(
                    Some(photo),
                    Some(crate::openvr::LegacyVrInput {
                        head,
                        left,
                        right: IDENTITY,
                        left_handgrip: IDENTITY,
                        right_handgrip: IDENTITY,
                        reference_epoch: 1,
                        left_index: 1,
                        right_index: 2,
                        grip,
                    }),
                    clock + Duration::from_millis(ms),
                );
            }
            assert!(runtime.interaction.frame.photo.is_some());
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut manifest = json!({
            "protocol_version":2, "role":"photo",
            "bridge_url":format!("ws://{}", listener.local_addr().unwrap()),
            "session_token":"controlled-runtime", "instance_id":"11111111-1111-4111-8111-111111111111",
            "generation":1, "backend":"offscreen", "interaction":{"wrist_transform":IDENTITY},
        });
        if let Some(width) = photo_width_m {
            manifest["interaction"]["photo_width_m"] = json!(width);
        }
        let manifest: PhotoManifest = serde_json::from_value(manifest).unwrap();
        let server = async {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let auth = message(&mut socket).await;
            assert_eq!(auth["type"], "photo_auth");
            assert_eq!(auth["session_token"], manifest.session_token);
            socket
                .send(Message::Text(
                    json!({
                        "type":"photo_snapshot", "revision":0, "photo":null
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            socket
        };
        let (connected, mut socket) = tokio::join!(PhotoBridge::connect(&manifest), server);
        let (mut bridge, first) = connected.unwrap();
        let (mut runtime, _worker) = Self::new(&manifest).unwrap();
        runtime.compositor.as_mut().unwrap().ready().await.unwrap();
        runtime.backend = Backend::OpenVr(PhotoOpenVrSurfaces {
            icon: Some(icon),
            photo: Some(photo_surface),
            grab_volume: [None, None, None],
            case_label: None,
            _session: session,
        });
        runtime.receive(first, &mut bridge).await.unwrap();
        assert_eq!(message(&mut socket).await["type"], "photo_state_received");
        let mut photo: Photo = serde_json::from_value(json!({
            "photo_id":"controlled-photo", "path":"synthetic.png", "file_size":1,
            "expected_file_id":["1","2"],
            "shutter_unix_ms":unix_ms(), "attempt_id":1, "phase":"requesting",
            "offer_open":false, "visible":false,
            "image":{"path":"synthetic.png", "width":320, "height":180}, "preparations":[],
            "blocks":[
                {"block_id":0, "bbox":[1,2,100,50], "source":"Brand", "translation":"Brand"},
                {"block_id":1, "bbox":[30,45,290,125], "source":"Source", "translation":"Temporary photo"}
            ], "failure_code":null, "first_block_drawn":false
        })).unwrap();
        let blocks = std::mem::take(&mut photo.blocks);
        publish(&mut runtime, &mut bridge, &mut socket, 1, &photo).await;
        photo.blocks = blocks;
        publish(&mut runtime, &mut bridge, &mut socket, 2, &photo).await;
        compose_blocks(&mut runtime);
        runtime.render_frame(&mut bridge).await.unwrap();
        assert!(!runtime
            .compositor
            .as_ref()
            .unwrap()
            .needs_frame(Instant::now()));
        no_report(&mut socket).await;
        set_input(false, 0.0);
        runtime.interaction_tick(&mut bridge).await.unwrap();
        tokio::time::sleep(Duration::from_millis(350)).await;
        runtime.interaction_tick(&mut bridge).await.unwrap();
        tokio::time::sleep(Duration::from_millis(210)).await;
        set_input(true, 0.0);
        runtime.interaction_tick(&mut bridge).await.unwrap();
        assert!(runtime.interaction.frame.grab);
        assert!(runtime.interaction.frame.glow);
        assert_pulses(&[]);
        set_input(true, 0.119);
        runtime.interaction_tick(&mut bridge).await.unwrap();
        assert!(!runtime.interaction.frame.pull);
        assert_pulses(&[]);
        let partial = runtime.interaction.frame.photo.unwrap();
        assert!(partial.alpha > 0.0 && partial.alpha < 1.0);
        assert!((partial.width - photo_width_m.unwrap_or(0.375) * partial.alpha).abs() < 0.00001);
        assert!(!runtime.state.photo().unwrap().visible);
        assert_eq!(runtime.pull_sequence, 0);
        runtime.compositor.as_mut().unwrap().view_original(true);
        runtime.render_frame(&mut bridge).await.unwrap();
        no_report(&mut socket).await;
        assert!(!runtime.drawn_reported);
        let supplied = std::env::var_os("PURIPULY_PHOTO341_EVIDENCE");
        let directory = supplied.clone().map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::env::temp_dir().join(format!("puri341-first-draw-{}", std::process::id()))
        });
        std::fs::create_dir_all(&directory).unwrap();
        let case = if photo_width_m.is_some() { "override" } else { "default" };
        let original_path = directory.join(format!("hidden-partial-{case}-original.png"));
        runtime.compositor.as_ref().unwrap().save(&original_path).unwrap();
        let original = image::open(&original_path).unwrap().into_rgba8();
        assert!(original.pixels().all(|pixel| pixel.0 == [35, 45, 60, 255]));
        runtime.compositor.as_mut().unwrap().view_original(false);
        if let Backend::OpenVr(surfaces) = &mut runtime.backend {
            surfaces.photo.as_mut().unwrap().set_overlay_visible(false).unwrap();
        }
        fail_photo_show(true);
        assert_eq!(
            runtime.render_frame(&mut bridge).await,
            Err(PhotoError::OpenVr)
        );
        assert!(!runtime.drawn_reported);
        no_report(&mut socket).await;
        fail_photo_show(false);
        runtime.render_frame(&mut bridge).await.unwrap();
        let visible = match &runtime.backend {
            Backend::OpenVr(surfaces) => surfaces.photo.as_ref().unwrap().observed_overlay_visible(),
            _ => panic!("Controlled OpenVR presentation required"),
        };
        assert_eq!(visible, Some(true));
        let translated_path = directory.join(format!("hidden-partial-{case}-translated.png"));
        let compositor = runtime.compositor.as_ref().unwrap();
        compositor.save(&translated_path).unwrap();
        let translated = image::open(&translated_path).unwrap().into_rgba8();
        let glyph_pixels = translated.pixels().filter(|pixel| pixel[0] > 200 && pixel[1] > 200 && pixel[2] > 190).count();
        let erased_pixels = translated.pixels().filter(|pixel| pixel.0 == [70, 90, 110, 255]).count();
        assert!(glyph_pixels > 100);
        assert!(erased_pixels > 100);
        for (x, y) in [(0, 0), (319, 0), (0, 179), (319, 179)] {
            assert_eq!(translated.get_pixel(x, y), original.get_pixel(x, y));
        }
        std::fs::write(directory.join(format!("hidden-partial-{case}-presentation.json")), serde_json::to_vec_pretty(&json!({
            "photo":runtime.state.photo(), "placement":partial, "observed_overlay_visible":visible,
            "pull_sequence":runtime.pull_sequence, "preparation_started":runtime.preparation_started,
            "glyph_pixels":glyph_pixels, "erased_pixels":erased_pixels, "compositor":compositor.metadata(),
            "input_boundary":"Controlled OpenVR system/surface function tables; production interaction/render_frame and real D3D pixels; no HMD",
        })).unwrap()).unwrap();
        let first = message(&mut socket).await;
        assert_eq!(
            first,
            json!({
                "type":"photo_first_block_drawn", "photo_id":photo.photo_id,
                "attempt_id":1, "block_id":1,
            })
        );
        assert_eq!(runtime.state.photo().unwrap().phase, PhotoPhase::Requesting);
        assert_eq!(runtime.pull_sequence, 0);
        no_report(&mut socket).await;
        grant(&mut runtime, &mut bridge, &mut socket, &photo.photo_id, 1, 0).await;
        assert_pulses(&[]);
        grant(&mut runtime, &mut bridge, &mut socket, &photo.photo_id, 1, 1).await;
        grant(&mut runtime, &mut bridge, &mut socket, &photo.photo_id, 1, 1).await;
        assert_pulses(&[(2, 0, 900)]);
        set_input(true, 0.12);
        runtime.interaction_tick(&mut bridge).await.unwrap();
        assert!(runtime.interaction.frame.pull);
        assert_pulses(&[(2, 0, 900), (1, 0, 900)]);
        assert_eq!(message(&mut socket).await["type"], "photo_pull");
        tokio::time::sleep(Duration::from_millis(160)).await;
        runtime.interaction_tick(&mut bridge).await.unwrap();
        assert!(!runtime.interaction.frame.pull);
        runtime.render_frame(&mut bridge).await.unwrap();
        assert_held_width();
        no_report(&mut socket).await;
        assert_pulses(&[(2, 0, 900), (1, 0, 900)]);
        let clock = Instant::now();
        if supplied.is_none() {
            std::fs::remove_dir_all(&directory).unwrap();
        }
        for ms in [550, 700] {
            let mut head = IDENTITY;
            head[2][3] = 0.5;
            runtime.interaction.step(
                Some(&photo),
                Some(crate::openvr::LegacyVrInput {
                    head,
                    left: IDENTITY,
                    right: IDENTITY,
                    left_handgrip: IDENTITY,
                    right_handgrip: IDENTITY,
                    reference_epoch: 1,
                    left_index: 1,
                    right_index: 2,
                    grip: false,
                }),
                clock + Duration::from_millis(ms),
            );
        }
        runtime.render_frame(&mut bridge).await.unwrap();
        publish(&mut runtime, &mut bridge, &mut socket, 4, &photo).await;
        assert!(runtime.interaction.frame.photo.is_none());
        hold(&mut runtime, clock + Duration::from_millis(800), 0.12);
        runtime.render_frame(&mut bridge).await.unwrap();
        no_report(&mut socket).await;

        let backend = std::mem::replace(&mut runtime.backend, Backend::Offscreen);
        let (recovered, _worker) = Self::new(&manifest).unwrap();
        runtime = recovered;
        runtime.compositor.as_mut().unwrap().ready().await.unwrap();
        runtime.backend = backend;
        publish(&mut runtime, &mut bridge, &mut socket, 5, &photo).await;
        compose_blocks(&mut runtime);
        hold(&mut runtime, Instant::now(), 0.12);
        runtime.render_frame(&mut bridge).await.unwrap();
        no_report(&mut socket).await;
        grant(
            &mut runtime,
            &mut bridge,
            &mut socket,
            &photo.photo_id,
            1,
            1,
        )
        .await;
        assert!(runtime.live_drawn_block.is_none());
        photo.attempt_id = 2;
        publish(&mut runtime, &mut bridge, &mut socket, 6, &photo).await;
        compose_blocks(&mut runtime);
        runtime.render_frame(&mut bridge).await.unwrap();
        assert_eq!(
            message(&mut socket).await,
            json!({
                "type":"photo_first_block_drawn", "photo_id":photo.photo_id,
                "attempt_id":2, "block_id":1,
            })
        );
        grant(
            &mut runtime,
            &mut bridge,
            &mut socket,
            &photo.photo_id,
            1,
            1,
        )
        .await;
        grant(
            &mut runtime,
            &mut bridge,
            &mut socket,
            &photo.photo_id,
            2,
            1,
        )
        .await;
        grant(
            &mut runtime,
            &mut bridge,
            &mut socket,
            &photo.photo_id,
            2,
            1,
        )
        .await;
        let old_id = photo.photo_id.clone();
        photo.photo_id = "replacement-photo".into();
        photo.shutter_unix_ms = unix_ms();
        photo.attempt_id = 0;
        photo.phase = PhotoPhase::Offer;
        photo.offer_open = true;
        photo.visible = false;
        photo.image = None;
        photo.blocks.clear();
        publish(&mut runtime, &mut bridge, &mut socket, 7, &photo).await;
        grant(&mut runtime, &mut bridge, &mut socket, &old_id, 2, 1).await;
        assert!(runtime.interaction.frame.photo.is_none());
        no_report(&mut socket).await;
        runtime.compositor.as_mut().unwrap().set_original(synthetic_drawable().0).unwrap();
        let pull_sequence = runtime.pull_sequence;
        hold(&mut runtime, Instant::now(), 0.119);
        runtime.render_frame(&mut bridge).await.unwrap();
        assert!(runtime.interaction.frame.photo.unwrap().alpha > 0.0);
        assert!(!runtime.interaction.frame.pull);
        assert_eq!(runtime.pull_sequence, pull_sequence);
        assert!(!runtime.preparation_started);
        assert!(!runtime.drawn_reported);
        assert!(runtime.live_drawn_block.is_none());
        no_report(&mut socket).await;
        runtime.interaction.step(Some(&photo), None, Instant::now() + Duration::from_secs(1));
        runtime.render_frame(&mut bridge).await.unwrap();
        bridge.close().await.unwrap();
    }
}

#[tokio::test]
async fn quiescent_bridge_ack_waits_for_interaction_and_release_or_loss() {
    use super::super::interaction::IDENTITY;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};
    for (phase, loss) in [(0, false), (1, false), (2, false), (3, false), (2, true)] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let manifest: PhotoManifest = serde_json::from_value(json!({
            "protocol_version":2, "role":"photo", "bridge_url":format!("ws://{}",listener.local_addr().unwrap()),
            "session_token":"synthetic", "instance_id":"11111111-1111-4111-8111-111111111111",
            "generation":1, "backend":"probe", "image":{"debug_controls":true},
            "interaction":{"wrist_transform":IDENTITY,"debug_case_label":"Position B\nGrab 8 cm fixed"}
        })).unwrap();
        let current = json!({
            "photo_id":"synthetic", "path":"synthetic.png", "file_size":1, "shutter_unix_ms":unix_ms(),
            "expected_file_id":["1","2"],
            "attempt_id":1, "phase":"requesting", "offer_open":false, "visible":true, "image":null,
            "preparations":[], "blocks":[], "failure_code":null, "first_block_drawn":false
        });
        let server = async {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let auth = socket.next().await.unwrap().unwrap();
            assert_eq!(serde_json::from_str::<serde_json::Value>(auth.to_text().unwrap()).unwrap()["type"], "photo_auth");
            socket.send(Message::Text(json!({"type":"photo_snapshot","revision":1,"photo":current}).to_string().into())).await.unwrap();
            socket
        };
        let (connection, mut socket) = tokio::join!(PhotoBridge::connect(&manifest), server);
        let (mut bridge, first) = connection.unwrap();
        let (mut runtime, _worker) = PhotoRuntime::new(&manifest).unwrap();
        runtime.receive(first, &mut bridge).await.unwrap();
        let receipt = socket.next().await.unwrap().unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(receipt.to_text().unwrap()).unwrap()["type"], "photo_state_received");
        let clock = Instant::now();
        let step = |runtime: &mut PhotoRuntime, ms, grip, separation| {
            let mut head = IDENTITY;
            head[2][3] = 0.5;
            let mut left = IDENTITY;
            left[0][3] = separation;
            runtime.interaction.step(runtime.state.photo(), Some(crate::openvr::LegacyVrInput {
                head, left, right: IDENTITY, left_handgrip: IDENTITY, right_handgrip: IDENTITY,
                reference_epoch:1, left_index:1, right_index:2, grip,
            }), clock + Duration::from_millis(ms))
        };
        for ms in [0, 50, 150, 350] { step(&mut runtime, ms, false, 0.0); }
        assert!(step(&mut runtime, 360, true, 0.0).grab);
        if phase >= 1 { assert!(step(&mut runtime, 380, true, 0.12).pull); }
        if phase >= 2 { assert!(step(&mut runtime, 530, true, 0.12).photo.is_some()); }
        if phase >= 3 { assert!(step(&mut runtime, 550, false, 0.12).photo.is_some()); }
        socket.send(Message::Text(json!({"type":"photo_debug_quiesce","request_id":"bounded-request"}).to_string().into())).await.unwrap();
        let incoming = bridge.next().await.unwrap();
        runtime.receive(incoming, &mut bridge).await.unwrap();
        runtime.ack_quiescent(&mut bridge).await.unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(20), socket.next()).await.is_err());
        if loss {
            runtime.interaction.step(runtime.state.photo(), None, clock + Duration::from_millis(600));
        } else {
            step(&mut runtime, 600, false, 0.12);
            runtime.ack_quiescent(&mut bridge).await.unwrap();
            assert!(tokio::time::timeout(Duration::from_millis(20), socket.next()).await.is_err());
            step(&mut runtime, 825, false, 0.0);
        }
        runtime.ack_quiescent(&mut bridge).await.unwrap();
        let ack = socket.next().await.unwrap().unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(ack.to_text().unwrap()).unwrap(),
            json!({"type":"photo_debug_quiescent","request_id":"bounded-request"}));
        assert!(!step(&mut runtime, 840, true, 0.0).grab);
        runtime.ack_quiescent(&mut bridge).await.unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(20), socket.next()).await.is_err());
    }
}

#[cfg(windows)]
#[tokio::test]
async fn first_grab_local_pixels_and_late_source_results_preserve_hold_fences() {
    use super::super::interaction::compose;
    use super::super::prepare::file_identity;
    use super::super::state::{Photo, PhotoPhase};
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};
    let directory = std::env::temp_dir().join(format!("puri341-grab-preview-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("capture.png");
    let pixels = image::RgbImage::from_pixel(320, 180, image::Rgb([45, 125, 205]));
    pixels.save(&source).unwrap();
    let mut photo: Photo = serde_json::from_value(json!({
        "photo_id":"local-grab-a","path":source,"file_size":std::fs::metadata(&source).unwrap().len(),
        "expected_file_id":file_identity(&std::fs::File::open(&source).unwrap()).unwrap(),
        "shutter_unix_ms":unix_ms(),"attempt_id":0,"phase":"offer","offer_open":true,
        "visible":false,"image":null,"preparations":[],"blocks":[],"failure_code":null,"first_block_drawn":false,
    })).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let manifest: PhotoManifest = serde_json::from_value(json!({
        "protocol_version":2,"role":"photo","bridge_url":format!("ws://{}",listener.local_addr().unwrap()),
        "session_token":"synthetic","instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1,"backend":"offscreen","image":{"max_size":[320,180],"debug_controls":true}
    })).unwrap();
    let server = async {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        let auth = socket.next().await.unwrap().unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(auth.to_text().unwrap()).unwrap()["type"], "photo_auth");
        socket.send(Message::Text(json!({"type":"photo_snapshot","revision":1,"photo":photo}).to_string().into())).await.unwrap();
        socket
    };
    let (connection, mut socket) = tokio::join!(PhotoBridge::connect(&manifest), server);
    let (mut bridge, first) = connection.unwrap();
    let (mut runtime, _worker) = PhotoRuntime::new(&manifest).unwrap();
    runtime.receive(first, &mut bridge).await.unwrap();
    socket.next().await.unwrap().unwrap();
    let clock = Instant::now();
    let sample = |grip: bool, separation: f32| {
        let right = [[1.0,0.0,0.0,0.3],[0.0,0.0,-1.0,-0.2],[0.0,1.0,0.0,-0.4]];
        let right_handgrip = [[0.0,-1.0,0.0,0.11],[1.0,0.0,0.0,-0.04],[0.0,0.0,1.0,0.09]];
        let left_handgrip = [[1.0,0.0,0.0,0.14],[0.0,0.0,-1.0,-0.07],[0.0,1.0,0.0,0.03]];
        let wrist = compose(right, super::super::interaction::TemporaryInteractionConfig::default().effective_wrist_transform(right_handgrip));
        let mut left = [[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0],[-1.0,0.0,0.0,0.0]];
        let pivot = compose(left, left_handgrip);
        let mut head = [[0.9800666,-0.19866933,0.0,0.0],[0.19866933,0.9800666,0.0,0.0],[0.0,0.0,1.0,0.0]];
        for i in 0..3 {
            left[i][3] = wrist[i][3] - pivot[i][3] + if i == 0 { separation } else { 0.0 };
            head[i][3] = wrist[i][3] + if i == 2 { 0.5 } else { 0.0 };
        }
        crate::openvr::LegacyVrInput {
            head,left,right,left_handgrip,right_handgrip,reference_epoch:7,left_index:3,right_index:8,grip,
        }
    };
    for ms in [0,50,150,350] {
        runtime.interaction_input(Some(sample(false,0.0)),clock+Duration::from_millis(ms),&mut bridge).await.unwrap();
    }
    assert!(runtime.source_request.is_none());
    runtime.interaction_input(Some(sample(true,0.0)),clock+Duration::from_millis(360),&mut bridge).await.unwrap();
    assert!(runtime.interaction.frame.grab);
    assert!(runtime.source_request.is_some());
    runtime.interaction_input(Some(sample(true,0.06)),clock+Duration::from_millis(380),&mut bridge).await.unwrap();
    let PreparationCompletion::Source(completion) = runtime.preparation.next_result().await.unwrap() else { panic!("Real local decode required") };
    runtime.preview_source(completion,&mut bridge).await.unwrap();
    assert!((runtime.interaction.frame.photo.unwrap().alpha-0.5).abs()<0.00001);
    assert_eq!(runtime.pull_sequence,0);
    assert!(!runtime.preparation_started);
    let first_directory = std::env::var_os("PURIPULY_PHOTO341_EVIDENCE").map(std::path::PathBuf::from).unwrap_or_else(||directory.clone());
    std::fs::create_dir_all(&first_directory).unwrap();
    let first_path = first_directory.join("compiled-first-grab-before-pull.png");
    let compositor = runtime.compositor.as_mut().unwrap();
    compositor.ready().await.unwrap();
    compositor.save(&first_path).unwrap();
    assert_eq!(image::open(first_path).unwrap().into_rgb8(),pixels);
    runtime.preparation.start_source(runtime.source_request.clone().unwrap()).unwrap();
    let PreparationCompletion::Source(late_release) = runtime.preparation.next_result().await.unwrap() else { panic!("Actual late release result required") };
    runtime.interaction_input(Some(sample(false,0.06)),clock+Duration::from_millis(400),&mut bridge).await.unwrap();
    runtime.interaction_input(Some(sample(false,0.0)),clock+Duration::from_millis(550),&mut bridge).await.unwrap();
    assert!(runtime.interaction.frame.photo.is_none());
    runtime.preview_source(late_release,&mut bridge).await.unwrap();
    assert!(runtime.texture.is_some());
    assert!(runtime.interaction.frame.photo.is_none());
    assert_eq!(runtime.pull_sequence,0);
    assert!(!runtime.preparation_started);
    for (ms,grip,separation) in [(625,false,0.0),(650,false,0.0),(660,true,0.0),(680,true,0.06)] {
        runtime.interaction_input(Some(sample(grip,separation)),clock+Duration::from_millis(ms),&mut bridge).await.unwrap();
    }
    let partial = runtime.interaction.frame;
    assert!((partial.photo.unwrap().alpha-0.5).abs()<0.00001);
    assert!((partial.photo.unwrap().width-0.1875).abs()<0.00001);
    assert_eq!(runtime.pull_sequence,0);
    assert!(tokio::time::timeout(Duration::from_millis(20),socket.next()).await.is_err());
    assert!(!runtime.drawn_reported);
    assert!(runtime.live_drawn_block.is_none());
    assert!(!runtime.take_haptic_grant());
    let evidence = std::env::var_os("PURIPULY_PHOTO341_EVIDENCE").map(std::path::PathBuf::from);
    let image_path = evidence.clone().unwrap_or_else(||directory.clone()).join("compiled-grab-before-pull.png");
    std::fs::create_dir_all(image_path.parent().unwrap()).unwrap();
    let compositor = runtime.compositor.as_mut().unwrap();
    compositor.ready().await.unwrap();
    compositor.save(&image_path).unwrap();
    let rendered = image::open(&image_path).unwrap().into_rgb8();
    assert_eq!(rendered,pixels);
    assert_eq!(compositor.original_uploads,1);
    if let Some(evidence) = &evidence {
        std::fs::write(evidence.join("compiled-grab-before-pull.json"),serde_json::to_vec_pretty(&json!({
            "frame":partial,"attempt_id":runtime.state.photo().unwrap().attempt_id,
            "authorized_pulls":runtime.pull_sequence,"preparation_started":runtime.preparation_started,
            "input_boundary":"Synthetic nonidentity native poses through the production interaction_input function; not OpenVR/HMD input",
        })).unwrap()).unwrap();
    }
    runtime.interaction_input(Some(sample(true,0.121)),clock+Duration::from_millis(700),&mut bridge).await.unwrap();
    let pull = socket.next().await.unwrap().unwrap();
    assert_eq!(serde_json::from_str::<serde_json::Value>(pull.to_text().unwrap()).unwrap()["type"],"photo_pull");
    assert_eq!(runtime.interaction.frame.photo.unwrap().alpha,1.0);
    assert!(tokio::time::timeout(Duration::from_millis(20),socket.next()).await.is_err());
    photo.attempt_id=1;
    photo.phase=PhotoPhase::Preparing;
    photo.offer_open=false;
    photo.visible=true;
    runtime.receive(PhotoIncoming::Snapshot { revision:2,photo:Some(photo.clone()) },&mut bridge).await.unwrap();
    socket.next().await.unwrap().unwrap();
    runtime.interaction_input(Some(sample(true,0.0)),clock+Duration::from_millis(720),&mut bridge).await.unwrap();
    assert_eq!(runtime.interaction.frame.photo.unwrap().width,0.375);
    assert_eq!(runtime.pull_sequence,1);
    assert_eq!(runtime.compositor.as_ref().unwrap().original_uploads,1);
    runtime.interaction_input(None,clock+Duration::from_millis(740),&mut bridge).await.unwrap();
    socket.next().await.unwrap().unwrap();
    assert!(runtime.interaction.frame.photo.is_none());
    runtime.preparation.start_source(runtime.source_request.clone().unwrap()).unwrap();
    let PreparationCompletion::Source(late) = runtime.preparation.next_result().await.unwrap() else { panic!("Cached local result required") };
    runtime.preview_source(late,&mut bridge).await.unwrap();
    assert!(runtime.interaction.frame.photo.is_none());
    assert_eq!(runtime.pull_sequence,1);
    runtime.preparation.start_source(runtime.source_request.clone().unwrap()).unwrap();
    let PreparationCompletion::Source(stale_a) = runtime.preparation.next_result().await.unwrap() else { panic!("Actual stale A result required") };
    let second = directory.join("second.png");
    image::RgbImage::from_pixel(320,180,image::Rgb([205,45,125])).save(&second).unwrap();
    let mut replacement = photo.clone();
    replacement.photo_id="local-grab-b".into();
    replacement.path=second.to_string_lossy().into_owned();
    replacement.expected_file_id=file_identity(&std::fs::File::open(&second).unwrap()).unwrap();
    replacement.attempt_id=0;
    replacement.phase=PhotoPhase::Offer;
    replacement.offer_open=true;
    replacement.visible=false;
    runtime.receive(PhotoIncoming::Snapshot { revision:3,photo:Some(replacement.clone()) },&mut bridge).await.unwrap();
    socket.next().await.unwrap().unwrap();
    runtime.preview_source(stale_a,&mut bridge).await.unwrap();
    assert!(runtime.texture.is_none());
    assert_eq!(runtime.compositor.as_ref().unwrap().metadata()["width"],serde_json::Value::Null);
    runtime.enqueue_preview_source();
    let PreparationCompletion::Source(stale_b) = runtime.preparation.next_result().await.unwrap() else { panic!("Actual B decode required") };
    replacement.phase=PhotoPhase::Expired;
    replacement.offer_open=false;
    runtime.receive(PhotoIncoming::Snapshot { revision:4,photo:Some(replacement) },&mut bridge).await.unwrap();
    socket.next().await.unwrap().unwrap();
    runtime.preview_source(stale_b,&mut bridge).await.unwrap();
    assert!(runtime.texture.is_none());
    assert!(runtime.interaction.frame.photo.is_none());
    std::fs::write(&source,b"incomplete").unwrap();
    photo.photo_id="local-grab-c".into();
    photo.attempt_id=0;
    photo.phase=PhotoPhase::Offer;
    photo.offer_open=true;
    photo.visible=false;
    photo.shutter_unix_ms=unix_ms();
    runtime.receive(PhotoIncoming::Snapshot { revision:5,photo:Some(photo) },&mut bridge).await.unwrap();
    socket.next().await.unwrap().unwrap();
    runtime.enqueue_preview_source();
    let PreparationCompletion::Source(failed) = runtime.preparation.next_result().await.unwrap() else { panic!("Actual incomplete-source failure required") };
    assert_eq!(failed.result.as_ref().err().copied(),Some("source_incomplete"));
    runtime.preview_source(failed,&mut bridge).await.unwrap();
    assert!(runtime.source_request.is_none());
    assert!(runtime.texture.is_none());
    assert!(!runtime.preparation_started);
    assert_eq!(runtime.pull_sequence,1);
    assert!(tokio::time::timeout(Duration::from_millis(20),socket.next()).await.is_err());
    pixels.save(&source).unwrap();
    runtime.enqueue_preview_source();
    let PreparationCompletion::Source(recovered) = runtime.preparation.next_result().await.unwrap() else { panic!("Explicit source recovery required") };
    runtime.preview_source(recovered,&mut bridge).await.unwrap();
    assert!(runtime.texture.is_some());
    assert!(runtime.interaction.frame.photo.is_none());
    assert_eq!(runtime.pull_sequence,1);
    assert!(!runtime.preparation_started);
    runtime.preparation.shutdown().await;
    drop(runtime);
    std::fs::remove_dir_all(directory).unwrap();
}
