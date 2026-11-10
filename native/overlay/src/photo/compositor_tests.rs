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
        core: image::GrayImage::from_pixel(92, 44, image::Luma([255])),
        glyph_size: 44.0,
        fill: [0, 0, 255],
        plate: false,
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
    compositor.ready().await.unwrap();
    compositor.save(file).unwrap();
    image::open(file).unwrap().into_rgba8()
}

#[tokio::test]
async fn real_gpu_joint_fade_original_toggle_and_retry_preserve_original_upload() {
    let directory = std::env::temp_dir().join(format!("puri340-layer-test-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone()).unwrap();
    let now = Instant::now();
    let original = capture(&mut compositor, now, &directory.join("original.png")).await;
    compositor.add_block(0, None, None, 0.0, now).unwrap();
    assert!(compositor.first_visible_block_in_extent(now + BLOCK_FADE, 1.0).is_none());
    let region = region();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter.typeset(&working, &region, "Translated").unwrap();
    assert!(!text.runs.is_empty());
    compositor
        .add_block(1, Some(region), Some(text), 0.0, now)
        .unwrap();
    assert!(compositor.first_visible_block_in_extent(now, 1.0).is_none());
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
    assert_eq!(compositor.first_visible_block_in_extent(now + BLOCK_FADE, 1.0), Some(1));
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
    compositor.view_original(true);
    let peek = capture(
        &mut compositor,
        now + BLOCK_FADE,
        &directory.join("peek.png"),
    )
    .await;
    assert_eq!(peek, original);
    compositor.view_original(false);
    let restored = capture(
        &mut compositor,
        now + BLOCK_FADE,
        &directory.join("restored.png"),
    )
    .await;
    assert_eq!(restored, full);
    compositor.clear_attempt();
    compositor.set_original(working).unwrap();
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
async fn real_gpu_plate_alpha_is_applied_once_at_half_and_full_fade() {
    let directory =
        std::env::temp_dir().join(format!("puri340-plate-alpha-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut compositor = PhotoCompositor::new(None).unwrap();
    let working = synthetic();
    compositor.set_original(working.clone()).unwrap();
    let mut prepared = region();
    prepared.plate = true;
    prepared.erase = None;
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter
        .typeset(&working, &prepared, "Translated")
        .unwrap();
    let start = Instant::now();
    compositor
        .add_block(0, Some(prepared), Some(text), 0.0, start)
        .unwrap();
    let half = capture(
        &mut compositor,
        start + Duration::from_millis(50),
        &directory.join("half.png"),
    )
    .await;
    let full = capture(
        &mut compositor,
        start + BLOCK_FADE,
        &directory.join("full.png"),
    )
    .await;
    for (pixels, fade) in [(&half, 0.5f32), (&full, 1.0f32)] {
        let actual = pixels.get_pixel(24, 16).0;
        let alpha = 238.0 / 255.0 * fade;
        for (channel, (source, plate)) in [255.0f32, 0.0, 0.0]
            .into_iter()
            .zip([16.0f32, 20.0, 28.0])
            .enumerate()
        {
            let expected = (source * (1.0 - alpha) + plate * alpha).round() as i32;
            assert!((i32::from(actual[channel]) - expected).abs() <= 1);
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
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
        core: image::GrayImage::from_pixel(384, 96, image::Luma([255])),
        glyph_size: 96.0,
        fill: [0, 255, 0],
        plate: false,
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
    compositor.set_original(working).unwrap();
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
    compositor.set_original(working.clone()).unwrap();
    let now = Instant::now();
    let region = region();
    let mut typesetter = PhotoTypesetter::new().unwrap();
    let text = typesetter.typeset(&working, &region, "Readable").unwrap();
    compositor.add_block(1, Some(region), Some(text), 0.0, now).unwrap();
    compositor.render(now + BLOCK_FADE, false).unwrap();
    compositor.ready().await.unwrap();
    assert!(compositor.first_visible_block_in_extent(now + BLOCK_FADE, 0.1).is_none());
    assert_eq!(compositor.first_visible_block_in_extent(now + BLOCK_FADE, 1.0), Some(1));
}
