use std::sync::Arc;

use image::{GrayImage, Luma, Rgb, RgbImage};

use super::*;
use crate::photo::processing::PixelRect;

fn typesetter() -> PhotoTypesetter {
    PhotoTypesetter::with_font_path(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/puripuly_heart/data/fonts/NotoSansCJK-Medium.ttc"),
    )
    .unwrap()
}

fn photo(height: u32) -> WorkingPhoto {
    WorkingPhoto {
        pixels: Arc::new(RgbImage::from_pixel(
            height * 4 / 3,
            height,
            Rgb([60, 70, 80]),
        )),
        scale_x: 1.0,
        scale_y: 1.0,
        reference_scale: height as f32 / 1440.0,
    }
}

fn region(scale: f32) -> PreparedRegion {
    let mut core = GrayImage::new((400.0 * scale) as u32, (100.0 * scale) as u32);
    for (x, y, pixel) in core.enumerate_pixels_mut() {
        let x = x as f32 / scale;
        let y = y as f32 / scale;
        if (20.0..380.0).contains(&x) && ((8.0..36.0).contains(&y) || (52.0..80.0).contains(&y)) {
            *pixel = Luma([255]);
        }
    }
    let x = (80.0 * scale) as u32;
    let y = (100.0 * scale) as u32;
    PreparedRegion {
        bbox: rect(
            x as f32 + 20.0 * scale,
            y as f32 + 8.0 * scale,
            x as f32 + 380.0 * scale,
            y as f32 + 80.0 * scale,
        ),
        roi: PixelRect {
            x,
            y,
            width: core.width(),
            height: core.height(),
        },
        recovered: core.clone(),
        core,
        glyph_size: 28.0 * scale,
        source_lines: 2,
        fill: [220, 160, 40],
        erase: None,
        mask_ms: 0.0,
        erase_ms: 0.0,
    }
}

#[test]
fn windows_graphics_real_bundled_metrics_fit_and_preserve_position() {
    let mut typesetter = typesetter();
    assert_eq!(
        typesetter.font_metrics().source,
        FontSource::BundledNotoCjkMedium
    );
    assert_eq!(typesetter.font_metrics().family, "Noto Sans CJK KR");
    assert!((0.8..1.05).contains(&typesetter.font_metrics().hangul_ink_ratio));
    let photo = photo(1440);
    let region = region(1.0);
    let first = typesetter
        .typeset(&photo, &region, "한글가나다라\n한글가나다라")
        .unwrap();
    assert_eq!(
        first.kind,
        LayoutKind::Lines,
        "size={} cap={} metrics={:?}",
        first.selected_size,
        first.source_cap,
        typesetter.font_metrics()
    );
    assert!(!first.collision);
    assert!(!first.region_overflow);
    assert!(first.selected_size <= first.source_cap);
    assert!(first.selected_size >= first.floor);
    assert!(first.source_cap - first.selected_size < 0.1);
    let saved: Vec<_> = first
        .runs
        .iter()
        .map(|run| (run.origin, run.size))
        .collect();
    let saved_boxes = first.collision_boxes.clone();
    let second = typesetter
        .typeset(&photo, &region, "또다른 문장\n두번째 문장")
        .unwrap();
    assert!(second.collision || second.selected_size < first.selected_size);
    assert_eq!(
        saved,
        first
            .runs
            .iter()
            .map(|run| (run.origin, run.size))
            .collect::<Vec<_>>()
    );
    assert_eq!(typesetter.placed_boxes()[0].left, saved_boxes[0].left);
    for run in &first.runs {
        let mut actual_size = 0.0;
        unsafe {
            run.layout.GetFontSize(0, &mut actual_size, None).unwrap();
        }
        assert_eq!(actual_size, run.size);
        let mut metrics = DWRITE_TEXT_METRICS::default();
        unsafe {
            run.layout.GetMetrics(&mut metrics).unwrap();
        }
        assert!(metrics.width > 0.0);
        assert!(contains(region.bbox, run.ink));
    }
    assert!(!typesetter.metrics.cache.is_empty());
    assert!(!typesetter.metrics.formats.is_empty());
    assert!(!typesetter.metrics.layouts.is_empty());
    let factory = typesetter.metrics.factory.as_raw();
    let normalized_format = typesetter.metrics.normalized_format.as_raw();
    typesetter.reset();
    assert!(typesetter.placed_boxes().is_empty());
    assert!(typesetter.metrics.cache.is_empty());
    assert!(typesetter.metrics.formats.is_empty());
    assert!(typesetter.metrics.layouts.is_empty());
    assert_eq!(typesetter.metrics.factory.as_raw(), factory);
    assert_eq!(
        typesetter.metrics.normalized_format.as_raw(),
        normalized_format
    );
    for run in &first.runs {
        let mut metrics = DWRITE_TEXT_METRICS::default();
        unsafe { run.layout.GetMetrics(&mut metrics).unwrap() };
        assert!(metrics.width > 0.0);
    }
    let replay = typesetter
        .typeset(&photo, &region, "한글가나다라\n한글가나다라")
        .unwrap();
    assert_eq!(
        saved,
        replay
            .runs
            .iter()
            .map(|run| (run.origin, run.size))
            .collect::<Vec<_>>()
    );
}

#[test]
fn windows_graphics_dynamic_scale_uses_actual_height_once() {
    let mut typesetter = typesetter();
    let mut reference: Option<f32> = None;
    for height in [720, 1080, 1440, 2160] {
        typesetter.reset();
        let scale = height as f32 / 1440.0;
        let result = typesetter
            .typeset(&photo(height), &region(scale), "번역 문장\n다음 문장")
            .unwrap();
        let long = (height * 4 / 3) as f32;
        assert!((result.floor - FLOOR_RATIO * long).abs() < 0.001);
        assert!(!result.runs.is_empty());
        for run in &result.runs {
            let mut actual_size = 0.0;
            unsafe {
                run.layout.GetFontSize(0, &mut actual_size, None).unwrap();
            }
            assert_eq!(actual_size, run.size);
            assert!(actual_size >= result.floor);
        }
        let normalized = result.source_cap / scale;
        if let Some(expected) = reference {
            assert!((normalized - expected).abs() < 0.05);
        } else {
            reference = Some(normalized);
        }
        assert!(result.selected_size <= result.source_cap);
        assert!(!result.region_overflow);
        assert_eq!(result.style.outline.as_ref().unwrap().color, [0, 0, 0]);
    }
}

#[test]
fn windows_graphics_text_too_long_for_a_tiny_box_spills_past_it_at_the_floor_without_a_plate() {
    let mut typesetter = typesetter();
    for height in [720, 1080, 1440, 2160] {
        typesetter.reset();
        let scale = height as f32 / 1440.0;
        let mut region = region(scale);
        region.core = GrayImage::new(8, 8);
        region.roi.width = 8;
        region.roi.height = 8;
        region.bbox = rect(80.0, 100.0, 88.0, 108.0);
        region.glyph_size = 4.0 * scale;
        let block = typesetter
            .typeset(&photo(height), &region, "작은 칸에 들어가지 않는 긴 문장")
            .unwrap();
        let expected_floor = FLOOR_RATIO * (height * 4 / 3) as f32;
        assert!((block.floor - expected_floor).abs() < 0.001);
        assert_eq!(block.selected_size, block.floor);
        assert!(block.source_cap < block.floor);
        assert!(!block.region_overflow);
        assert!(!block.collision);
        assert!(spills(&block, &region));
        assert_spilled_text_keeps_source_style(&block, &region);
        assert!(!block.runs.is_empty());
        for run in &block.runs {
            let mut actual_size = 0.0;
            unsafe {
                run.layout.GetFontSize(0, &mut actual_size, None).unwrap();
            }
            assert_eq!(actual_size, run.size);
            assert!((actual_size - expected_floor).abs() < 0.001);
        }
    }
}

#[test]
fn windows_graphics_floor_rescue_wraps_complete_paragraph_without_moving_prior_ink() {
    let mut typesetter = typesetter();
    let photo = photo(1080);
    let first = typesetter
        .typeset(&photo, &region(0.75), "먼저 배치한 문장\n그대로 남는 문장")
        .unwrap();
    let prior = typesetter.placed_boxes().to_vec();
    let mut paragraph = region(0.75);
    paragraph.roi = PixelRect {
        x: 91,
        y: 461,
        width: 278,
        height: 48,
    };
    paragraph.bbox = rect(100.0, 470.0, 360.0, 500.0);
    paragraph.core = GrayImage::from_fn(278, 48, |x, y| {
        Luma([if (9..269).contains(&x) && (12..23).contains(&y) {
            255
        } else {
            0
        }])
    });
    let text = "좁은 영역에서도 읽을 수 있어야 하는 아주 길고 자세한 번역 문장이 여러 줄에 걸쳐 계속 이어집니다.";
    let block = typesetter.typeset(&photo, &paragraph, text).unwrap();
    assert_eq!(block.kind, LayoutKind::Horizontal);
    assert_eq!(block.selected_size, block.floor);
    assert!(block.source_cap < block.floor);
    assert!(block.runs.len() >= 2);
    assert!(spills(&block, &paragraph));
    assert_spilled_text_keeps_source_style(&block, &paragraph);
    let compact = |text: &str| {
        text.chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
    };
    assert_eq!(
        compact(
            &block
                .runs
                .iter()
                .map(|run| run.text.as_str())
                .collect::<String>()
        ),
        compact(text)
    );
    let bounds = rect(0.0, 0.0, photo.pixels.width() as f32, 1080.0);
    assert!(block
        .collision_boxes
        .iter()
        .all(|&ink| contains(bounds, ink)));
    assert_eq!(&typesetter.placed_boxes()[..prior.len()], prior.as_slice());
    assert_eq!(first.collision_boxes, prior);
    assert!(!hits(&block.collision_boxes, &prior));
    for run in &block.runs {
        let mut size = 0.0;
        unsafe {
            run.layout.GetFontSize(0, &mut size, None).unwrap();
        }
        assert_eq!(size, block.floor);
    }
}

#[test]
fn windows_graphics_floor_rescue_margin_is_bounded_and_clamped_to_photo() {
    let mut typesetter = typesetter();
    let extent = rect(1.0, 10.0, 70.0, 20.0);
    let model_box = rect(0.0, 5.0, 70.0, 25.0);
    let info = SourceMeasurement {
        angle_degrees: 0.0,
        direction: Direction::Horizontal,
        extent,
        slots: vec![extent],
        glyph_size: 10.0,
        center: [38.0, 15.0],
    };
    let text = "가나다라마바사아자차카타파하";
    let photo_bounds = rect(0.0, 0.0, 85.0, 30.0);
    let block = layout::choose(
        &mut typesetter.metrics,
        text,
        &info,
        model_box,
        20.0,
        20.0,
        10.0,
        &[],
        photo_bounds,
    )
    .unwrap();
    assert_eq!(block.kind, LayoutKind::Horizontal);
    assert_eq!(block.size, 20.0);
    assert!(block.overflow);
    assert!(block
        .runs
        .iter()
        .all(|run| run.ink.left >= 0.0 && run.ink.right <= 80.0));
    assert_eq!(block.runs.len(), 4);
    let ink = block.runs.iter().map(|run| run.ink).reduce(union).unwrap();
    assert!(((ink.left + ink.right) * 0.5 - 40.0).abs() < 0.001);
    assert!(((ink.top + ink.bottom) * 0.5 - 15.0).abs() < 0.001);
    assert_eq!(info.extent, extent);
    assert_eq!(info.slots, vec![extent]);
    assert_eq!(info.center, [38.0, 15.0]);
}

fn spills(block: &TypesetBlock, region: &PreparedRegion) -> bool {
    block.runs.iter().any(|run| !contains(region.bbox, run.ink))
}

fn assert_spilled_text_keeps_source_style(block: &TypesetBlock, region: &PreparedRegion) {
    assert_eq!(block.style.fill, region.fill);
    assert!(block.style.outline.is_some());
}

#[test]
fn windows_graphics_spill_grows_only_as_needed_and_keeps_prior_blocks_clear() {
    let mut typesetter = typesetter();
    let photo = photo(1080);
    let mut neighbor = region(0.75);
    neighbor.roi = PixelRect {
        x: 400,
        y: 280,
        width: 300,
        height: 75,
    };
    neighbor.bbox = rect(415.0, 287.0, 685.0, 347.0);
    neighbor.core = GrayImage::from_fn(300, 75, |x, y| {
        Luma([if (15..285).contains(&x) && ((7..22).contains(&y) || (45..60).contains(&y)) {
            255
        } else {
            0
        }])
    });
    let fits = typesetter
        .typeset(&photo, &neighbor, "그대로 들어가는 문장
두 번째 줄")
        .unwrap();
    assert!(!spills(&fits, &neighbor));
    let prior = typesetter.placed_boxes().to_vec();
    let mut tight = region(0.75);
    tight.roi = PixelRect {
        x: 470,
        y: 360,
        width: 120,
        height: 40,
    };
    tight.bbox = rect(480.0, 368.0, 580.0, 392.0);
    tight.core = GrayImage::from_fn(120, 40, |x, y| {
        Luma([if (10..110).contains(&x) && (10..30).contains(&y) { 255 } else { 0 }])
    });
    let block = typesetter
        .typeset(&photo, &tight, "짧은 원문에 비해 훨씬 긴 번역 문장입니다")
        .unwrap();
    assert_eq!(block.selected_size, block.floor);
    assert!(spills(&block, &tight));
    assert!(!block.region_overflow);
    assert!(!block.collision);
    assert!(!hits(&block.collision_boxes, &prior));
    assert_spilled_text_keeps_source_style(&block, &tight);
    let ink = block.collision_boxes.iter().copied().reduce(union).unwrap();
    let reach = 4.0 * block.floor;
    assert!(ink.left >= tight.bbox.left - reach && ink.right <= tight.bbox.right + reach);
    assert!(ink.top >= tight.bbox.top - reach && ink.bottom <= tight.bbox.bottom + reach);
}

#[test]
fn windows_graphics_floor_fallback_retains_slots_when_not_every_candidate_overflows() {
    let mut typesetter = typesetter();
    let extent = rect(100.0, 100.0, 300.0, 130.0);
    let mut info = SourceMeasurement {
        angle_degrees: 0.0,
        direction: Direction::Horizontal,
        extent,
        slots: vec![extent],
        glyph_size: 20.0,
        center: [200.0, 115.0],
    };
    let photo_bounds = rect(0.0, 0.0, 1920.0, 1080.0);
    let placed = [photo_bounds];
    let blocked = layout::choose(
        &mut typesetter.metrics,
        "한글",
        &info,
        extent,
        20.0,
        20.0,
        20.0,
        &placed,
        photo_bounds,
    )
    .unwrap();
    assert_eq!(blocked.kind, LayoutKind::Lines);
    assert!(!blocked.overflow);
    assert!(hits(&blocked.boxes(&info), &placed));
    info.angle_degrees = 12.0;
    let rotated = layout::choose(
        &mut typesetter.metrics,
        "회전된 긴 문장은 기존 줄 배치를 유지합니다",
        &info,
        extent,
        20.0,
        20.0,
        10.0,
        &[],
        photo_bounds,
    )
    .unwrap();
    assert_eq!(rotated.kind, LayoutKind::Lines);
    assert!(rotated.overflow);
    info.angle_degrees = 0.0;
    info.direction = Direction::Vertical;
    info.extent = rect(100.0, 100.0, 112.0, 150.0);
    info.slots = vec![info.extent];
    let vertical = layout::choose(
        &mut typesetter.metrics,
        "한글 세로 문장",
        &info,
        info.extent,
        20.0,
        20.0,
        10.0,
        &[],
        info.extent,
    )
    .unwrap();
    assert_eq!(vertical.kind, LayoutKind::Lines);
    assert!(vertical.overflow);
    assert_eq!(vertical.size, 20.0);
}

#[test]
fn windows_graphics_native_vertical_forms_tate_chu_yoko_and_fallback_smoke() {
    let mut typesetter = typesetter();
    let mut region = region(1.0);
    region.core = GrayImage::new(120, 420);
    for (x, y, pixel) in region.core.enumerate_pixels_mut() {
        if ((20..50).contains(&x) || (80..110).contains(&x)) && (10..410).contains(&y) {
            *pixel = Luma([255]);
        }
    }
    region.roi.width = 120;
    region.roi.height = 420;
    region.bbox = rect(100.0, 110.0, 190.0, 510.0);
    let block = typesetter
        .typeset(&photo(1440), &region, "「한글」12!?\n다음 문장")
        .unwrap();
    assert!(block.source_vertical);
    assert!(matches!(
        block.kind,
        LayoutKind::Lines | LayoutKind::Vertical
    ));
    assert!(block.runs.iter().any(|run| run.text == "﹁"));
    for pair in ["12", "!?"] {
        let run = block.runs.iter().find(|run| run.text == pair).unwrap();
        assert!((run.size - block.selected_size * 0.75).abs() < 0.001);
    }
    for run in &block.runs {
        assert!(contains(region.bbox, run.ink));
    }
    let fallback = typesetter
        .metrics
        .create_layout("한글 العربية 😀", 24.0)
        .unwrap();
    let mut metrics = DWRITE_TEXT_METRICS::default();
    unsafe {
        fallback.GetMetrics(&mut metrics).unwrap();
    }
    assert!(metrics.width > 80.0);
    let observed = Arc::new(std::sync::Mutex::new(ObservedRuns::default()));
    let renderer: windows::Win32::Graphics::DirectWrite::IDWriteTextRenderer = GlyphObserver {
        observed: observed.clone(),
    }
    .into();
    unsafe {
        fallback.Draw(None, &renderer, 0.0, 0.0).unwrap();
    }
    let observed = observed.lock().unwrap();
    assert!(observed.glyphs > 0);
    assert_eq!(observed.missing, 0);
    assert!(observed.faces.len() >= 2);
}

#[test]
fn source_measurement_deskews_and_returns_two_refined_slots() {
    let source = region(1.0);
    let measured = measure_source(&source.core, [80.0, 100.0], 1.0, 2).unwrap();
    assert_eq!(measured.angle_degrees, 0.0);
    assert_eq!(measured.direction, Direction::Horizontal);
    assert_eq!(measured.slots.len(), 2);
    assert_eq!(measured.glyph_size, 28.0);
    let mut tilted = GrayImage::new(400, 180);
    let (s, c) = 12.0f32.to_radians().sin_cos();
    for (x, y, pixel) in tilted.enumerate_pixels_mut() {
        let dx = x as f32 - 200.0;
        let dy = y as f32 - 90.0;
        let sx = c * dx + s * dy + 200.0;
        let sy = -s * dx + c * dy + 50.0;
        if (20.0..380.0).contains(&sx) && ((10.0..30.0).contains(&sy) || (60.0..80.0).contains(&sy))
        {
            *pixel = Luma([255]);
        }
    }
    let rotated = measure_source(&tilted, [10.0, 20.0], 1.0, 2).unwrap();
    assert!(
        (rotated.angle_degrees - 12.0).abs() <= 1.0,
        "{}",
        rotated.angle_degrees
    );
    assert_eq!(rotated.slots.len(), 2);
    assert!((rotated.glyph_size - 20.0).abs() <= 2.0);
    assert_eq!(rotated.center, [210.0, 110.0]);
}

fn glyph_grid(lines: u32, per_line: u32, vertical: bool, line_gap: u32, tilt: f32) -> GrayImage {
    let (glyph, glyph_gap, margin) = (30.0f32, 3.0f32, 40.0f32);
    let along = per_line as f32 * (glyph + glyph_gap) - glyph_gap;
    let across = lines as f32 * (glyph + line_gap as f32) - line_gap as f32;
    let (width, height) = if vertical {
        (across + 2.0 * margin, along + 2.0 * margin)
    } else {
        (along + 2.0 * margin, across + 2.0 * margin)
    };
    let mut core = GrayImage::new(width as u32, height as u32);
    let center = [width / 2.0, height / 2.0];
    let (s, c) = tilt.to_radians().sin_cos();
    for (x, y, pixel) in core.enumerate_pixels_mut() {
        let dx = x as f32 - center[0];
        let dy = y as f32 - center[1];
        let sx = c * dx + s * dy + center[0] - margin;
        let sy = -s * dx + c * dy + center[1] - margin;
        let (a, b) = if vertical {
            (sy, across - sx)
        } else {
            (sx, sy)
        };
        if a < 0.0 || b < 0.0 || a >= along || b >= across {
            continue;
        }
        let line = (b / (glyph + line_gap as f32)) as u32;
        let slot = (a / (glyph + glyph_gap)) as u32;
        let u = a - slot as f32 * (glyph + glyph_gap);
        let v = b - line as f32 * (glyph + line_gap as f32);
        let stroke = |t: f32| t < 4.0 || (13.0..17.0).contains(&t) || (26.0..30.0).contains(&t);
        if u < glyph && v < glyph && (stroke(u) || stroke(v)) && slot < per_line - line % 3 {
            *pixel = Luma([255]);
        }
    }
    core
}

#[test]
fn tall_horizontal_source_lines_override_aspect_biased_vertical_projection() {
    for lines in [7, 14, 20] {
        let core = glyph_grid(lines, 7, false, 10, 0.0);
        assert_eq!(
            measure_source(&core, [0.0, 0.0], 1.0, 1).unwrap().direction,
            Direction::Vertical
        );
        let measured = measure_source(&core, [5.0, 7.0], 1.0, lines).unwrap();
        assert_eq!(measured.direction, Direction::Horizontal, "lines={lines}");
        assert_eq!(measured.angle_degrees, 0.0);
        assert_eq!(measured.slots.len(), lines as usize);
        assert!((measured.glyph_size - 30.0).abs() <= 1.0);
        assert_eq!(measured.slots[0].top, 47.0);
    }
    let tilted = measure_source(&glyph_grid(14, 7, false, 10, 6.0), [0.0, 0.0], 1.0, 14).unwrap();
    assert_eq!(tilted.direction, Direction::Horizontal);
    assert!(
        (tilted.angle_degrees - 6.0).abs() <= 1.0,
        "{}",
        tilted.angle_degrees
    );
}

#[test]
fn vertical_sources_and_unconfirmed_line_counts_keep_projection_direction() {
    for (columns, per_column, gap) in [(1, 8, 10), (2, 8, 10), (3, 8, 10), (6, 8, 10), (6, 8, 4)] {
        let core = glyph_grid(columns, per_column, true, gap, 0.0);
        let measured = measure_source(&core, [0.0, 0.0], 1.0, columns).unwrap();
        assert_eq!(
            measured.direction,
            Direction::Vertical,
            "columns={columns} gap={gap}"
        );
    }
    let touching = glyph_grid(14, 7, false, 0, 0.0);
    let mismatched = glyph_grid(14, 7, false, 10, 0.0);
    for (core, lines) in [(&touching, 14), (&mismatched, 13), (&mismatched, 1)] {
        assert_eq!(
            measure_source(core, [0.0, 0.0], 1.0, lines)
                .unwrap()
                .direction,
            Direction::Vertical,
            "lines={lines}"
        );
    }
}

#[test]
fn source_projection_uses_all_foreground_without_periodic_stride_aliasing() {
    let mut core = GrayImage::new(680, 180);
    for (x, y, pixel) in core.enumerate_pixels_mut() {
        let mut hash = x.wrapping_mul(1_103_515_245).wrapping_add(y * 12_345);
        hash ^= hash >> 16;
        if (x % 27 < 18 && (5 * y + x) % 360 < 198) || hash % 1000 < 10 {
            *pixel = Luma([255]);
        }
    }
    assert_eq!(core.as_raw().iter().filter(|&&v| v != 0).count(), 45_550);
    let measured = measure_source(&core, [0.0, 0.0], 1.0, 1).unwrap();
    assert_eq!(measured.direction, Direction::Horizontal);
    assert_eq!(measured.angle_degrees, -11.0);
}

#[test]
#[ignore = "requires original angle masks and angles.json via PURI_PHOTO340_REFERENCE"]
fn original_angle_masks_match_full_foreground_reference() {
    let root = std::path::PathBuf::from(
        std::env::var_os("PURI_PHOTO340_REFERENCE").expect("reference fixture directory"),
    );
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("angles.json")).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let core = image::open(root.join(case["mask"].as_str().unwrap()))
            .unwrap()
            .into_luma8();
        let start = std::time::Instant::now();
        let measured = measure_source(&core, [0.0, 0.0], 1.0, 1).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(
            measured.angle_degrees,
            case["angle"].as_f64().unwrap() as f32
        );
        assert_eq!(
            measured.direction == Direction::Vertical,
            case["vertical"].as_bool().unwrap()
        );
        println!(
            "angle_probe case={} pixels={} angle={} vertical={} measure_ms={:.3}",
            case["case"].as_str().unwrap(),
            core.as_raw().iter().filter(|&&v| v != 0).count(),
            measured.angle_degrees,
            measured.direction == Direction::Vertical,
            elapsed_ms
        );
    }
}

#[test]
fn recovered_vertical_forms_mapping_is_complete_and_preserves_pair_cells() {
    let input = "「」『』（）()…‥ー—-〈〉《》、。, .，:;!?[]{}'‘’“”~～";
    let expected = "﹁﹂﹃﹄︵︶︵︶︙︰│││︿﹀︽︾︑︒︐ ︒︐︓︔!?﹇﹈︷︸﹁﹁﹂﹃﹄︴︴";
    assert_eq!(layout::vertical_cells(input).concat(), expected);
    assert_eq!(
        layout::vertical_cells("12!!34!?56?!78??"),
        ["12", "!!", "34", "!?", "56", "?!", "78", "??"]
    );
}

#[test]
fn windows_graphics_later_block_shrinks_around_immutable_previous_ink() {
    let mut typesetter = typesetter();
    let mut first_region = region(1.0);
    first_region.core = GrayImage::new(60, 60);
    first_region.roi = PixelRect {
        x: 100,
        y: 100,
        width: 60,
        height: 60,
    };
    first_region.bbox = rect(100.0, 100.0, 160.0, 160.0);
    first_region.glyph_size = 40.0;
    let first = typesetter
        .typeset(&photo(1440), &first_region, "가")
        .unwrap();
    let mut next_region = region(1.0);
    next_region.core = GrayImage::new(160, 60);
    next_region.roi = PixelRect {
        x: 100,
        y: 100,
        width: 160,
        height: 60,
    };
    next_region.bbox = rect(100.0, 100.0, 260.0, 160.0);
    next_region.glyph_size = 40.0;
    let second = typesetter
        .typeset(&photo(1440), &next_region, "가나다")
        .unwrap();
    assert!(second.selected_size < second.source_cap - 1.0);
    assert!(second.selected_size >= second.floor);
    assert!(!second.collision);
    assert!(!hits(&second.collision_boxes, &first.collision_boxes));
    assert_eq!(
        typesetter.placed_boxes()[0].left,
        first.collision_boxes[0].left
    );
}

#[test]
fn source_slots_scale_fixed_two_pixel_merge_and_discard_thresholds() {
    let mut core = GrayImage::new(160, 40);
    for (x, y, pixel) in core.enumerate_pixels_mut() {
        if (5..155).contains(&x)
            && ((4..6).contains(&y) || (7..9).contains(&y) || (15..17).contains(&y))
        {
            *pixel = Luma([255]);
        }
    }
    let full = measure_source(&core, [0.0, 0.0], 1.0, 1).unwrap();
    let half = measure_source(&core, [0.0, 0.0], 0.5, 1).unwrap();
    assert_eq!(full.slots.len(), 2);
    assert_eq!(half.slots.len(), 3);
    assert_eq!(full.slots[0].height(), 5.0);
    assert_eq!(half.slots[0].height(), 2.0);
}

#[derive(Default)]
struct ObservedRuns {
    faces: Vec<usize>,
    glyphs: usize,
    missing: usize,
}

#[windows::core::implement(windows::Win32::Graphics::DirectWrite::IDWriteTextRenderer)]
struct GlyphObserver {
    observed: Arc<std::sync::Mutex<ObservedRuns>>,
}

impl windows::Win32::Graphics::DirectWrite::IDWritePixelSnapping_Impl for GlyphObserver_Impl {
    fn IsPixelSnappingDisabled(
        &self,
        _: *const core::ffi::c_void,
    ) -> windows::core::Result<windows_core::BOOL> {
        Ok(true.into())
    }
    fn GetCurrentTransform(
        &self,
        _: *const core::ffi::c_void,
        transform: *mut windows::Win32::Graphics::DirectWrite::DWRITE_MATRIX,
    ) -> windows::core::Result<()> {
        unsafe {
            transform.write(windows::Win32::Graphics::DirectWrite::DWRITE_MATRIX {
                m11: 1.0,
                m12: 0.0,
                m21: 0.0,
                m22: 1.0,
                dx: 0.0,
                dy: 0.0,
            });
        }
        Ok(())
    }
    fn GetPixelsPerDip(&self, _: *const core::ffi::c_void) -> windows::core::Result<f32> {
        Ok(1.0)
    }
}

impl windows::Win32::Graphics::DirectWrite::IDWriteTextRenderer_Impl for GlyphObserver_Impl {
    fn DrawGlyphRun(
        &self,
        _: *const core::ffi::c_void,
        _: f32,
        _: f32,
        _: windows::Win32::Graphics::DirectWrite::DWRITE_MEASURING_MODE,
        glyphrun: *const windows::Win32::Graphics::DirectWrite::DWRITE_GLYPH_RUN,
        _: *const windows::Win32::Graphics::DirectWrite::DWRITE_GLYPH_RUN_DESCRIPTION,
        _: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        let run = unsafe { &*glyphrun };
        let mut observed = self.observed.lock().unwrap();
        let face = run.fontFace.as_ref().unwrap().as_raw() as usize;
        if !observed.faces.contains(&face) {
            observed.faces.push(face);
        }
        let indices =
            unsafe { std::slice::from_raw_parts(run.glyphIndices, run.glyphCount as usize) };
        observed.glyphs += indices.len();
        observed.missing += indices.iter().filter(|&&index| index == 0).count();
        Ok(())
    }
    fn DrawUnderline(
        &self,
        _: *const core::ffi::c_void,
        _: f32,
        _: f32,
        _: *const windows::Win32::Graphics::DirectWrite::DWRITE_UNDERLINE,
        _: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
    fn DrawStrikethrough(
        &self,
        _: *const core::ffi::c_void,
        _: f32,
        _: f32,
        _: *const windows::Win32::Graphics::DirectWrite::DWRITE_STRIKETHROUGH,
        _: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
    fn DrawInlineObject(
        &self,
        _: *const core::ffi::c_void,
        _: f32,
        _: f32,
        _: windows::core::Ref<windows::Win32::Graphics::DirectWrite::IDWriteInlineObject>,
        _: windows_core::BOOL,
        _: windows_core::BOOL,
        _: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

#[test]
fn windows_graphics_crowded_text_shrinks_toward_the_hard_floor_instead_of_overlapping() {
    let mut typesetter = typesetter();
    let extent = rect(100.0, 100.0, 300.0, 130.0);
    let info = SourceMeasurement {
        angle_degrees: 0.0,
        direction: Direction::Horizontal,
        extent,
        slots: vec![extent],
        glyph_size: 20.0,
        center: [200.0, 115.0],
    };
    let photo_bounds = rect(0.0, 0.0, 1920.0, 1080.0);
    let placed = [
        rect(0.0, 0.0, 1920.0, 99.0),
        rect(0.0, 131.0, 1920.0, 1080.0),
        rect(0.0, 99.0, 99.0, 131.0),
        rect(301.0, 99.0, 1920.0, 131.0),
    ];
    let block = layout::choose(
        &mut typesetter.metrics,
        "한글한글한글한글한글한글",
        &info,
        extent,
        20.0,
        12.0,
        20.0,
        &placed,
        photo_bounds,
    )
    .unwrap();
    assert!((12.0..20.0).contains(&block.size), "size={}", block.size);
    assert!(!block.overflow);
    assert!(!hits(&block.boxes(&info), &placed));
}

