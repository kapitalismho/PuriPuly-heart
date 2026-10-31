use super::*;

#[test]
fn feedback_requires_live_draw_and_is_once_per_attempt_even_after_snapshot_ack() {
    let manifest = PhotoManifest {
        protocol_version: 1,
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
        "protocol_version":1,"role":"photo","bridge_url":"ws://127.0.0.1:12345",
        "session_token":"synthetic","instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1,"backend":"probe"
    }))
    .unwrap();
    let (mut runtime, _worker) = PhotoRuntime::new(&manifest).unwrap();
    let photo = serde_json::from_value(json!({
        "photo_id":"synthetic","path":"capture.png","file_size":1,"shutter_unix_ms":1,
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
async fn windows_graphics_photo341_icons_quiet_first_block_and_held_preview_readback() {
    use super::super::processing::{ErasePatch, PixelRect, PreparedRegion, Rect, WorkingPhoto};
    use super::super::typeset::PhotoTypesetter;
    use std::sync::Arc;
    let manifest: PhotoManifest = serde_json::from_value(json!({
        "protocol_version":1,"role":"photo","bridge_url":"ws://127.0.0.1:12345",
        "session_token":"synthetic","instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1,"backend":"offscreen"
    }))
    .unwrap();
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
    compositor.set_original(working.clone()).unwrap();
    let now = Instant::now();
    compositor.render(now, false).unwrap();
    compositor.ready().await.unwrap();
    compositor
        .save(&directory.join("held-original.png"))
        .unwrap();
    compositor.add_block(0, None, None, 0.0, now).unwrap();
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
    let photo: super::super::state::Photo = serde_json::from_value(json!({
        "photo_id":"synthetic","path":"capture.png","file_size":1,"shutter_unix_ms":1,
        "attempt_id":1,"phase":"requesting","offer_open":false,"visible":true,
        "image":{"path":"snapshot.png","width":320,"height":180},"preparations":[],
        "blocks":[
            {"block_id":0,"bbox":[1,2,100,50],"source":"Brand","translation":"Brand"},
            {"block_id":1,"bbox":[30,45,290,125],"source":"Source","translation":"Temporary photo"}
        ],"failure_code":null,"first_block_drawn":false
    }))
    .unwrap();
    runtime.state.apply(1, Some(photo.clone())).unwrap();
    assert!(runtime.take_draw_report(false, visible).is_none());
    assert!(runtime.take_draw_report(true, None).is_none());
    assert_eq!(
        runtime.take_draw_report(true, visible),
        Some(("synthetic".into(), 1, 1))
    );
    runtime.grant_haptic("synthetic", 1, 0);
    assert!(!runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(!runtime.take_haptic_grant());
    assert!(runtime.take_draw_report(false, visible).is_none());
    assert!(runtime.take_draw_report(true, visible).is_none());
    runtime.reset_feedback(true);
    assert!(runtime.take_draw_report(true, visible).is_none());
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(!runtime.take_haptic_grant());
    let mut recovered = photo.clone();
    recovered.first_block_drawn = true;
    runtime.state.apply(2, Some(recovered)).unwrap();
    runtime.reset_feedback(true);
    assert!(runtime.take_draw_report(true, visible).is_none());
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(!runtime.take_haptic_grant());
    let mut retry = photo;
    retry.attempt_id = 2;
    runtime.state.apply(3, Some(retry)).unwrap();
    runtime.reset_feedback(false);
    assert_eq!(
        runtime.take_draw_report(true, visible),
        Some(("synthetic".into(), 2, 1))
    );
    runtime.grant_haptic("synthetic", 1, 1);
    assert!(!runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 2, 1);
    assert!(runtime.take_haptic_grant());
    runtime.grant_haptic("synthetic", 2, 1);
    assert!(!runtime.take_haptic_grant());
    if supplied.is_none() {
        std::fs::remove_dir_all(directory).unwrap();
    }
}
