use super::super::processing::{ErasePatch, PreparedRegion};
use super::super::typeset::PhotoTypesetter;
use super::*;

fn synthetic() -> Arc<WorkingPhoto> {
    Arc::new(
        WorkingPhoto::new(
            Arc::new(image::RgbImage::from_pixel(
                128,
                72,
                image::Rgb([255, 0, 0]),
            )),
            [128, 72],
        )
        .unwrap(),
    )
}

fn region() -> PreparedRegion {
    PreparedRegion {
        bbox: Rect {
            left: 24.0,
            top: 16.0,
            right: 116.0,
            bottom: 60.0,
        },
        roi: PixelRect {
            x: 24,
            y: 16,
            width: 92,
            height: 44,
        },
        recovered: image::GrayImage::from_pixel(92, 44, image::Luma([255])),
        core: image::GrayImage::from_pixel(92, 44, image::Luma([255])),
        glyph_size: 44.0,
        source_lines: 1,
        fill: [0, 0, 255],
        erase: Some(ErasePatch {
            rect: PixelRect {
                x: 2,
                y: 2,
                width: 4,
                height: 4,
            },
            pixels: image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 255, 0, 255])),
        }),
        mask_ms: 0.0,
        erase_ms: 0.0,
    }
}

async fn capture(compositor: &mut PhotoCompositor, time: Instant, file: &Path) -> image::RgbaImage {
    compositor.render(time, false).unwrap();
    compositor.wait_for_completion().await.unwrap();
    compositor.save(file).unwrap();
    image::open(file).unwrap().into_rgba8()
}

#[tokio::test]
async fn real_gpu_joint_fade_original_toggle_and_retry_preserve_original_upload() {
    let directory = std::env::temp_dir().join(format!("puri340-layer-test-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone(), "camera.png").unwrap();
    let now = Instant::now();
    let original = capture(&mut compositor, now, &directory.join("original.png")).await;
    compositor.add_block(0, None, None, 0.0, now).unwrap();
    assert!(compositor.first_visible_block_in(now + BLOCK_FADE, [0.0, 0.0, 1.0, 1.0]).is_none());
    let region = region();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter.typeset(&working, &region, "Translated").unwrap();
    assert!(!text.runs.is_empty());
    compositor
        .add_block(1, Some(region), Some(text), 0.0, now)
        .unwrap();
    assert!(compositor.first_visible_block_in(now, [0.0, 0.0, 1.0, 1.0]).is_none());
    let zero = capture(&mut compositor, now, &directory.join("zero.png")).await;
    assert_eq!(zero, original);
    let half = capture(
        &mut compositor,
        now + Duration::from_millis(50),
        &directory.join("half.png"),
    )
    .await;
    assert!(compositor.needs_frame(now + Duration::from_millis(150)));
    let full = capture(
        &mut compositor,
        now + BLOCK_FADE,
        &directory.join("full.png"),
    )
    .await;
    assert!(!compositor.needs_frame(now + Duration::from_millis(150)));
    assert_eq!(compositor.completed_prefix, 2);
    assert!(compositor.completed_text.is_some());
    assert_eq!(compositor.first_visible_block_in(now + BLOCK_FADE, [0.0, 0.0, 1.0, 1.0]), Some(1));
    assert_eq!(full.get_pixel(3, 3).0, [0, 255, 0, 255]);
    let middle = half.get_pixel(3, 3).0;
    assert!((i16::from(middle[0]) - 127).abs() <= 1);
    assert!((i16::from(middle[1]) - 128).abs() <= 1);
    let (x, y, _) = full
        .enumerate_pixels()
        .find(|(x, y, pixel)| *x > 20 && *y > 10 && pixel[2] > 200)
        .expect("DirectWrite text must reach the GPU target");
    let full_text = full.get_pixel(x, y).0;
    let half_text = half.get_pixel(x, y).0;
    for channel in 0..3 {
        let expected =
            (u16::from(original.get_pixel(x, y)[channel]) + u16::from(full_text[channel])) / 2;
        assert!((i32::from(half_text[channel]) - i32::from(expected)).abs() <= 2);
    }
    compositor.clear_attempt();
    assert_eq!(compositor.completed_prefix, 0);
    assert!(compositor.completed_text.is_none());
    compositor.set_original(working, "camera.png").unwrap();
    assert_eq!(compositor.original_uploads, 1);
    let retry = capture(
        &mut compositor,
        now + BLOCK_FADE,
        &directory.join("retry.png"),
    )
    .await;
    assert_eq!(retry, original);
    assert_eq!(compositor.block_count(), 0);
    assert!(compositor.frame_gpu_ms.is_some());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn joint_fade_has_exact_endpoints_and_linear_midpoint() {
    let start = Instant::now();
    assert_eq!(fade_opacity(start, start), 0.0);
    assert_eq!(fade_opacity(start, start + Duration::from_millis(50)), 0.5);
    assert_eq!(fade_opacity(start, start + Duration::from_millis(100)), 1.0);
    assert_eq!(fade_opacity(start, start + Duration::from_millis(500)), 1.0);
}

#[tokio::test]
async fn real_gpu_outline_preserves_logical_exterior_ink_radius() {
    let directory =
        std::env::temp_dir().join(format!("puri340-outline-radius-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let working = Arc::new(
        WorkingPhoto::new(
            Arc::new(image::RgbImage::from_pixel(
                512,
                256,
                image::Rgb([200, 200, 200]),
            )),
            [512, 256],
        )
        .unwrap(),
    );
    let prepared = PreparedRegion {
        bbox: Rect {
            left: 64.0,
            top: 64.0,
            right: 448.0,
            bottom: 160.0,
        },
        roi: PixelRect {
            x: 64,
            y: 64,
            width: 384,
            height: 96,
        },
        recovered: image::GrayImage::from_pixel(384, 96, image::Luma([255])),
        core: image::GrayImage::from_pixel(384, 96, image::Luma([255])),
        glyph_size: 96.0,
        source_lines: 1,
        fill: [0, 255, 0],
        erase: None,
        mask_ms: 0.0,
        erase_ms: 0.0,
    };
    let font = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/puripuly_heart/data/fonts/NotoSansCJK-Medium.ttc");
    let mut typesetter = PhotoTypesetter::with_font_path(&font).unwrap();
    let text = typesetter.typeset(&working, &prepared, "한글H").unwrap();
    let logical = text.style.outline.as_ref().unwrap().width;
    assert!(logical >= 6.0);
    let mut compositor = PhotoCompositor::new(None).unwrap();
    compositor.set_original(working, "camera.png").unwrap();
    let mut reference = text.clone();
    reference.style.outline.as_mut().unwrap().width = 0.0001;
    let start = Instant::now();
    compositor
        .add_block(0, None, Some(reference), 0.0, start)
        .unwrap();
    let fill = capture(
        &mut compositor,
        start + BLOCK_FADE,
        &directory.join("fill.png"),
    )
    .await;
    compositor.clear_attempt();
    compositor
        .add_block(0, None, Some(text), 0.0, start)
        .unwrap();
    let outlined = capture(
        &mut compositor,
        start + BLOCK_FADE,
        &directory.join("outlined.png"),
    )
    .await;
    let bounds = |image: &image::RgbaImage| {
        let mut bounds = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
        for (x, y, pixel) in image.enumerate_pixels() {
            if pixel[0] < 100 && pixel[2] < 100 {
                bounds[0] = bounds[0].min(x as i32);
                bounds[1] = bounds[1].min(y as i32);
                bounds[2] = bounds[2].max(x as i32 + 1);
                bounds[3] = bounds[3].max(y as i32 + 1);
            }
        }
        assert!(bounds[0] < bounds[2] && bounds[1] < bounds[3]);
        bounds
    };
    let fill_bounds = bounds(&fill);
    let outline_bounds = bounds(&outlined);
    let outsets = [
        fill_bounds[0] - outline_bounds[0],
        fill_bounds[1] - outline_bounds[1],
        outline_bounds[2] - fill_bounds[2],
        outline_bounds[3] - fill_bounds[3],
    ];
    for (side, outset) in outsets.into_iter().enumerate() {
        assert!(
            (outset as f32 - logical).abs() <= 1.0,
            "side {side}: exterior {outset}px, logical outline {logical}px"
        );
    }
    assert_ne!(fill, outlined);
    assert_eq!(compositor.original_uploads, 1);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn clipped_out_translated_block_cannot_report_first_draw() {
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone(), "camera.png").unwrap();
    let now = Instant::now();
    let region = region();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter.typeset(&working, &region, "Readable").unwrap();
    compositor.add_block(1, Some(region), Some(text), 0.0, now).unwrap();
    compositor.render(now + BLOCK_FADE, false).unwrap();
    compositor.wait_for_completion().await.unwrap();
    assert!(compositor.first_visible_block_in(now + BLOCK_FADE, [0.0, 0.0, 0.1, 1.0]).is_none());
    assert_eq!(compositor.first_visible_block_in(now + BLOCK_FADE, [0.0, 0.0, 1.0, 1.0]), Some(1));
}

#[test]
fn owned_patch_premultiplication_is_byte_exact_without_reallocation() {
    let mut pixels = Vec::new();
    let mut expected = Vec::new();
    for alpha in 0..=255u16 {
        for channel in 0..=255u16 {
            let rgba = [channel as u8, (255 - channel) as u8, (channel / 2) as u8, alpha as u8];
            pixels.extend_from_slice(&rgba);
            expected.extend_from_slice(&[
                ((u16::from(rgba[2]) * alpha + 127) / 255) as u8,
                ((u16::from(rgba[1]) * alpha + 127) / 255) as u8,
                ((u16::from(rgba[0]) * alpha + 127) / 255) as u8,
                rgba[3],
            ]);
        }
    }
    let allocation = pixels.as_ptr();
    premultiply_bgra(&mut pixels);
    assert_eq!(pixels.as_ptr(), allocation);
    assert_eq!(pixels, expected);
}

#[test]
fn original_upload_releases_cpu_photo_and_reuses_new_retry_arc() {
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    let weak = Arc::downgrade(&working);
    let reference_scale = working.reference_scale;
    compositor.set_original(working, "camera.png").unwrap();
    assert!(weak.upgrade().is_none());
    assert_eq!(compositor.dimensions(), Some([128, 72]));
    assert_eq!(compositor.metadata()["reference_scale"], json!(reference_scale));
    compositor.clear_attempt();
    compositor.set_original(synthetic(), "camera.png").unwrap();
    assert_eq!(compositor.original_uploads, 1);
    compositor.accept_prepared_source("unrelated-camera.png", "private-source.png");
    compositor.set_original(synthetic(), "camera.png").unwrap();
    assert_eq!(compositor.original_uploads, 1);
    compositor.accept_prepared_source("camera.png", "private-source.png");
    compositor.clear_attempt();
    compositor.set_original(synthetic(), "private-source.png").unwrap();
    compositor.clear_attempt();
    compositor.set_original(synthetic(), "private-source.png").unwrap();
    assert_eq!(compositor.original_uploads, 1);
    compositor.clear_photo();
    assert_eq!(compositor.dimensions(), None);
    assert!(compositor.metadata()["reference_scale"].is_null());
}

#[tokio::test]
async fn pending_fence_preserves_new_blocks_and_rendered_draw_eligibility() {
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone(), "camera.png").unwrap();
    let now = Instant::now();
    assert!(compositor.render(now, false).unwrap());
    compositor.hold_completion = true;
    let submitted = compositor.pending.as_ref().unwrap().started;
    assert!(!compositor.poll_completion(submitted).unwrap());
    assert!(compositor.completed_texture().is_none());
    assert_eq!(compositor.render(now, false), Err(PhotoError::Graphics));
    assert_eq!(compositor.begin_completion(), Err(PhotoError::Graphics));
    assert_eq!(compositor.pending.as_ref().unwrap().started, submitted);
    let region = region();
    let text = PhotoTypesetter::new().unwrap().typeset(&working, &region, "Ready later").unwrap();
    compositor.add_block(1, Some(region), Some(text), 0.0, now - BLOCK_FADE).unwrap();
    assert!(compositor.needs_frame(now));
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    compositor.hold_completion = false;
    compositor.wait_for_completion().await.unwrap();
    assert!(compositor.completed_texture().is_some());
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    assert!(compositor.blocks[0].roi_timer.is_some());
    assert!(compositor.needs_frame(now));
    compositor.render(now + BLOCK_FADE, false).unwrap();
    compositor.wait_for_completion().await.unwrap();
    assert_eq!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]), Some(1));
    assert!(compositor.blocks[0].roi_timer.is_none());
    assert!(!compositor.needs_frame(now + BLOCK_FADE));
}

#[test]
fn held_gpu_completion_has_a_fixed_fifty_millisecond_deadline() {
    let mut compositor = PhotoCompositor::new(None).unwrap();
    compositor.begin_completion().unwrap();
    compositor.hold_completion = true;
    let started = compositor.pending.as_ref().unwrap().started;
    assert!(!compositor.poll_completion(started + GPU_TIMEOUT - Duration::from_millis(1)).unwrap());
    assert_eq!(compositor.poll_completion(started + GPU_TIMEOUT), Err(PhotoError::Graphics));
    assert_eq!(compositor.pending.as_ref().unwrap().started, started);
}

#[tokio::test]
async fn pending_render_uses_a_spare_target_and_keeps_completed_draw_authority() {
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone(), "camera.png").unwrap();
    let now = Instant::now();
    compositor.render(now, false).unwrap();
    compositor.wait_for_completion().await.unwrap();
    let original = compositor.completed_texture().unwrap();
    assert!(compositor.spare_target.is_none());
    compositor.mark_presented(&original);
    let region = region();
    let text = PhotoTypesetter::new().unwrap().typeset(&working, &region, "Not drawn yet").unwrap();
    compositor.add_block(1, Some(region), Some(text), 0.0, now - BLOCK_FADE).unwrap();
    compositor.render(now, false).unwrap();
    compositor.hold_completion = true;
    let pending = PhotoTexture::new(compositor.texture.as_ref().unwrap().clone()).pointer();
    assert_ne!(pending, original.pointer());
    assert_eq!(PhotoTexture::new(compositor.spare_target.as_ref().unwrap().1.clone()).pointer(), original.pointer());
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    assert!(compositor.completed_texture().is_none());
    compositor.hold_completion = false;
    compositor.wait_for_completion().await.unwrap();
    compositor.render(now + Duration::from_millis(1), false).unwrap();
    assert_eq!(PhotoTexture::new(compositor.texture.as_ref().unwrap().clone()).pointer(), pending);
    assert_eq!(PhotoTexture::new(compositor.spare_target.as_ref().unwrap().1.clone()).pointer(), original.pointer());
    compositor.wait_for_completion().await.unwrap();
    compositor.mark_presented(&compositor.completed_texture().unwrap());
    assert_eq!(compositor.completed_texture().unwrap().pointer(), pending);
    assert_eq!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]), Some(1));
    compositor.render(now + BLOCK_FADE, false).unwrap();
    assert_eq!(PhotoTexture::new(compositor.texture.as_ref().unwrap().clone()).pointer(), original.pointer());
    compositor.wait_for_completion().await.unwrap();
}

#[tokio::test]
async fn same_size_distinct_source_keys_replace_original_pixels() {
    let directory = std::env::temp_dir().join(format!("puri-photo-source-key-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    compositor.set_original(synthetic(), "red-source.png").unwrap();
    let now = Instant::now();
    let red = capture(&mut compositor, now, &directory.join("red.png")).await;
    let blue = || Arc::new(WorkingPhoto::new(
        Arc::new(image::RgbImage::from_pixel(128, 72, image::Rgb([0, 0, 255]))),
        [128, 72],
    ).unwrap());
    compositor.clear_attempt();
    compositor.set_original(blue(), "blue-source.png").unwrap();
    let replacement = capture(&mut compositor, now, &directory.join("blue.png")).await;
    assert_eq!(compositor.original_uploads, 2);
    assert_eq!(red.dimensions(), replacement.dimensions());
    assert!(red.pixels().all(|pixel| pixel.0 == [255, 0, 0, 255]));
    assert!(replacement.pixels().all(|pixel| pixel.0 == [0, 0, 255, 255]));
    compositor.clear_attempt();
    compositor.set_original(blue(), "blue-source.png").unwrap();
    assert_eq!(compositor.original_uploads, 2);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn completed_text_cache_appends_only_the_fade_completed_prefix() {
    let directory = std::env::temp_dir().join(format!("puri-photo-cache-rollover-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone(), "camera.png").unwrap();
    let now = Instant::now();
    let prepared = region();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let first = typesetter.typeset(&working, &prepared, "First").unwrap();
    let mut second = first.clone();
    second.style.fill = [255, 255, 0];
    for run in &mut second.runs {
        run.origin[1] += 24.0;
    }
    let mut third = first.clone();
    third.style.fill = [0, 255, 255];
    for run in &mut third.runs {
        run.origin[1] -= 24.0;
    }
    compositor.add_block(1, None, Some(first), 0.0, now).unwrap();
    compositor.add_block(2, None, Some(second), 0.0, now + BLOCK_FADE / 2).unwrap();
    compositor.add_block(3, None, Some(third), 0.0, now - BLOCK_FADE / 2).unwrap();
    capture(&mut compositor, now + BLOCK_FADE / 2, &directory.join("active.png")).await;
    assert_eq!(compositor.completed_prefix, 0);
    assert!(compositor.completed_text.is_none());
    let boundary = capture(&mut compositor, now + BLOCK_FADE, &directory.join("boundary.png")).await;
    assert_eq!(compositor.completed_prefix, 1);
    let cache = compositor.completed_text.as_ref().unwrap().as_raw();
    assert_eq!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]), Some(1));
    let geometry = std::mem::take(&mut compositor.blocks[0].runs);
    let reused = capture(&mut compositor, now + BLOCK_FADE, &directory.join("reused.png")).await;
    assert_eq!(reused, boundary);
    compositor.blocks[0].runs = geometry;
    let complete = capture(
        &mut compositor,
        now + BLOCK_FADE + BLOCK_FADE / 2,
        &directory.join("complete.png"),
    ).await;
    assert_eq!(compositor.completed_prefix, 3);
    assert_eq!(compositor.completed_text.as_ref().unwrap().as_raw(), cache);
    let repeated = capture(
        &mut compositor,
        now + BLOCK_FADE * 2,
        &directory.join("repeated.png"),
    ).await;
    assert_eq!(repeated, complete);
    assert_eq!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]), Some(1));
    assert!(!compositor.needs_frame(now + BLOCK_FADE * 2));
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn overlapping_erases_stay_below_cached_and_active_text_through_retry() {
    let directory = std::env::temp_dir().join(format!("puri-photo-cache-order-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone(), "camera.png").unwrap();
    let now = Instant::now();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter.typeset(&working, &region(), "Text").unwrap();
    let mut active = text.clone();
    active.style.fill = [255, 255, 0];
    compositor.add_block(1, None, Some(text), 0.0, now).unwrap();
    let before = capture(&mut compositor, now + BLOCK_FADE, &directory.join("before.png")).await;
    let (x, y, _) = before.enumerate_pixels()
        .find(|(_, _, pixel)| pixel.0 == [0, 0, 255, 255])
        .expect("outlined text must include opaque fill");
    let mut erase = region();
    erase.erase = Some(ErasePatch {
        rect: PixelRect { x: 0, y: 0, width: 128, height: 72 },
        pixels: image::RgbaImage::from_pixel(128, 72, image::Rgba([0, 255, 0, 255])),
    });
    compositor.add_block(2, Some(erase), None, 0.0, now + BLOCK_FADE).unwrap();
    compositor.add_block(3, None, Some(active), 0.0, now + BLOCK_FADE).unwrap();
    let half = capture(
        &mut compositor,
        now + BLOCK_FADE + BLOCK_FADE / 2,
        &directory.join("half.png"),
    ).await;
    assert_eq!(compositor.completed_prefix, 1);
    for (channel, expected) in [128i16, 128, 127].into_iter().enumerate() {
        assert!((i16::from(half.get_pixel(x, y)[channel]) - expected).abs() <= 1);
    }
    for (channel, expected) in [127i16, 128, 0].into_iter().enumerate() {
        assert!((i16::from(half.get_pixel(0, 0)[channel]) - expected).abs() <= 1);
    }
    let complete = capture(
        &mut compositor,
        now + BLOCK_FADE * 2,
        &directory.join("complete.png"),
    ).await;
    assert_eq!(compositor.completed_prefix, 3);
    assert_eq!(complete.get_pixel(x, y).0, [255, 255, 0, 255]);
    let cache = compositor.completed_text.as_ref().unwrap().as_raw();
    compositor.render(now + BLOCK_FADE * 2, true).unwrap();
    compositor.wait_for_completion().await.unwrap();
    compositor.save(&directory.join("erased.png")).unwrap();
    let erased = image::open(directory.join("erased.png")).unwrap().into_rgba8();
    assert!(erased.pixels().all(|pixel| pixel.0 == [0, 255, 0, 255]));
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    assert_eq!(compositor.completed_text.as_ref().unwrap().as_raw(), cache);
    let restored = capture(
        &mut compositor,
        now + BLOCK_FADE * 2,
        &directory.join("restored.png"),
    ).await;
    assert_eq!(restored, complete);
    assert_eq!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]), Some(1));
    compositor.clear_attempt();
    assert!(compositor.completed_text.is_none());
    assert_eq!(compositor.completed_prefix, 0);
    compositor.set_original(working, "camera.png").unwrap();
    let retry = capture(&mut compositor, now + BLOCK_FADE * 2, &directory.join("retry.png")).await;
    assert!(retry.pixels().all(|pixel| pixel.0 == [255, 0, 0, 255]));
    assert_eq!(compositor.original_uploads, 1);
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    let working = synthetic();
    let text = PhotoTypesetter::new().unwrap().typeset(&working, &region(), "Retry").unwrap();
    compositor.add_block(4, None, Some(text), 0.0, now).unwrap();
    capture(&mut compositor, now + BLOCK_FADE * 2, &directory.join("retry-text.png")).await;
    assert_eq!(compositor.completed_prefix, 1);
    compositor.set_original(working, "new-source.png").unwrap();
    assert!(compositor.completed_text.is_none());
    assert_eq!(compositor.completed_prefix, 0);
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    let replaced = capture(&mut compositor, now + BLOCK_FADE * 2, &directory.join("replaced.png")).await;
    assert_eq!(replaced, retry);
    assert_eq!(compositor.original_uploads, 2);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn warm_up_restores_clean_photo_state_and_preserves_queued_card_pixels() {
    let directory = std::env::temp_dir().join(format!("puri-photo-warmup-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let initial = compositor.metadata();
    assert!(!compositor.warmed_up);
    let card = compositor.icon_texture(&[205, 205, 200, 255], 1).unwrap();
    compositor.warm_up().await.unwrap();
    assert!(compositor.warmed_up);
    assert!(!compositor.warming_up);
    assert_eq!(compositor.metadata(), initial);
    assert_eq!(compositor.dimensions(), None);
    assert_eq!(compositor.upload_serial, 0);
    assert_eq!(compositor.block_count(), 0);
    assert_eq!(compositor.completed_prefix, 0);
    assert!(compositor.completed_text.is_none());
    assert!(compositor.original.is_none());
    assert!(compositor.target.is_none());
    assert!(compositor.texture.is_none());
    assert!(compositor.spare_target.is_none());
    assert!(compositor.presented_texture.is_none());
    assert!(!compositor.has_pending_completion());
    assert!(compositor.completed_texture().is_none());
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    assert!(!compositor.needs_frame(Instant::now()));
    assert!(!compositor.render(Instant::now(), false).unwrap());
    compositor.save_texture(&card, 1, 1, &directory.join("card.png")).unwrap();
    let card = image::open(directory.join("card.png")).unwrap().into_rgba8();
    assert_eq!(card.get_pixel(0, 0).0, [205, 205, 200, 255]);
    compositor.warm_up().await.unwrap();
    assert_eq!(compositor.metadata(), initial);
    compositor.set_original(synthetic(), "real-photo.png").unwrap();
    let photo = capture(&mut compositor, Instant::now(), &directory.join("photo.png")).await;
    assert!(photo.pixels().all(|pixel| pixel.0 == [255, 0, 0, 255]));
    assert_eq!(compositor.original_uploads, 1);
    assert!(compositor.first_drawn_block_in([0.0, 0.0, 1.0, 1.0]).is_none());
    std::fs::remove_dir_all(directory).unwrap();
}
