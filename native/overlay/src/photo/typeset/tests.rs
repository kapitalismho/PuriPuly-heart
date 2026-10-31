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
        if (20.0..380.0).contains(&x) && ((10.0..30.0).contains(&y) || (60.0..80.0).contains(&y)) {
            *pixel = Luma([255]);
        }
    }
    let x = (80.0 * scale) as u32;
    let y = (100.0 * scale) as u32;
    PreparedRegion {
        bbox: rect(
            x as f32 + 20.0 * scale,
            y as f32 + 10.0 * scale,
            x as f32 + 380.0 * scale,
            y as f32 + 80.0 * scale,
        ),
        roi: PixelRect {
            x,
            y,
            width: core.width(),
            height: core.height(),
        },
        core,
        glyph_size: 20.0 * scale,
        fill: [220, 160, 40],
        plate: false,
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
    typesetter.reset();
    assert!(typesetter.placed_boxes().is_empty());
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
        assert!((result.floor - 0.011 * height as f32).abs() < 0.001);
        let normalized = result.source_cap / scale;
        if let Some(expected) = reference {
            assert!((normalized - expected).abs() < 0.05);
        } else {
            reference = Some(normalized);
        }
        assert!(result.selected_size <= result.source_cap);
        assert!(!result.region_overflow);
        assert_eq!(result.style.outline.as_ref().unwrap().color, [0, 0, 0]);
        let mut plate = region(scale);
        plate.plate = true;
        typesetter.reset();
        let plate_result = typesetter
            .typeset(&photo(height), &plate, "번역 문장\n다음 문장")
            .unwrap();
        assert_eq!(plate_result.style.fill, [245, 245, 240]);
        assert!(plate_result.style.outline.is_none());
        let style = plate_result.style.plate.unwrap();
        assert!((style.rect.left - (plate.bbox.left - 8.0 * scale)).abs() < 0.001);
        assert_eq!(style.color, [16, 20, 28]);
        assert!((style.alpha - 238.0 / 255.0).abs() < 0.00001);
    }
}

#[test]
fn windows_graphics_floor_fallback_never_renders_below_readable_floor() {
    let mut typesetter = typesetter();
    let mut region = region(1.0);
    region.core = GrayImage::new(8, 8);
    region.roi.width = 8;
    region.roi.height = 8;
    region.bbox = rect(80.0, 100.0, 88.0, 108.0);
    region.glyph_size = 4.0;
    let block = typesetter
        .typeset(&photo(1440), &region, "작은 칸에 들어가지 않는 긴 문장")
        .unwrap();
    assert_eq!(block.selected_size, block.floor);
    assert!(block.source_cap < block.floor);
    assert!(block.region_overflow);
    assert!(block.collision);
    assert!(!block.runs.is_empty());
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
    let measured = measure_source(&source.core, [80.0, 100.0], 1.0).unwrap();
    assert_eq!(measured.angle_degrees, 0.0);
    assert_eq!(measured.direction, Direction::Horizontal);
    assert_eq!(measured.slots.len(), 2);
    assert_eq!(measured.glyph_size, 20.0);
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
    let rotated = measure_source(&tilted, [10.0, 20.0], 1.0).unwrap();
    assert!(
        (rotated.angle_degrees - 12.0).abs() <= 1.0,
        "{}",
        rotated.angle_degrees
    );
    assert_eq!(rotated.slots.len(), 2);
    assert!((rotated.glyph_size - 20.0).abs() <= 2.0);
    assert_eq!(rotated.center, [210.0, 110.0]);
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
    let measured = measure_source(&core, [0.0, 0.0], 1.0).unwrap();
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
        let measured = measure_source(&core, [0.0, 0.0], 1.0).unwrap();
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
    let full = measure_source(&core, [0.0, 0.0], 1.0).unwrap();
    let half = measure_source(&core, [0.0, 0.0], 0.5).unwrap();
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
