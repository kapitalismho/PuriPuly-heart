use puripuly_heart_overlay::photo::processing::{MaskProcessor, Rect, WorkingPhoto};
use std::path::PathBuf;
use std::sync::Arc;

#[test]
#[ignore = "requires chosen external ONNX, ORT DLL and controlled original-case reference"]
fn chosen_model_infers_original_photo_against_offline_reference() {
    let root = PathBuf::from(
        std::env::var_os("PURI_PHOTO340_REFERENCE").expect("reference fixture directory"),
    );
    let model = PathBuf::from(std::env::var_os("PURI_PHOTO340_MODEL").expect("chosen ONNX model"));
    let runtime =
        PathBuf::from(std::env::var_os("PURI_PHOTO340_ORT").expect("ONNX Runtime dynamic library"));
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("mask.json")).unwrap()).unwrap();
    let source = Arc::new(
        image::open(fixture["source"].as_str().unwrap())
            .unwrap()
            .into_rgb8(),
    );
    let original_dimensions = source.dimensions();
    let work = WorkingPhoto::new(
        source.clone(),
        [original_dimensions.0, original_dimensions.1],
    )
    .unwrap();
    assert!(Arc::ptr_eq(&source, &work.pixels));
    let bbox = fixture["bbox"].as_array().unwrap();
    let bbox = Rect {
        left: bbox[0].as_f64().unwrap() as f32,
        top: bbox[1].as_f64().unwrap() as f32,
        right: bbox[2].as_f64().unwrap() as f32,
        bottom: bbox[3].as_f64().unwrap() as f32,
    };
    let mut processor = MaskProcessor::new(&model, Some(&runtime)).unwrap();
    let prepared = processor
        .prepare(
            &work,
            bbox,
            fixture["lines"].as_u64().unwrap() as u32,
            &work.pixels,
        )
        .unwrap();
    let expected = image::open(root.join("mask_expected_core.png"))
        .unwrap()
        .into_luma8();
    assert_eq!(prepared.core.dimensions(), expected.dimensions());
    let intersection = prepared
        .core
        .as_raw()
        .iter()
        .zip(expected.as_raw())
        .filter(|(a, b)| **a != 0 && **b != 0)
        .count();
    let union = prepared
        .core
        .as_raw()
        .iter()
        .zip(expected.as_raw())
        .filter(|(a, b)| **a != 0 || **b != 0)
        .count();
    assert!(union > 100, "real source text must be detected");
    let iou = intersection as f64 / union as f64;
    eprintln!(
        "Chosen model: core_iou={iou:.6}, plate={}, mask_ms={:.3}, erase_ms={:.3}, ROI={:?}",
        prepared.plate, prepared.mask_ms, prepared.erase_ms, prepared.roi
    );
    assert!(
        iou >= 0.98,
        "selected export mask diverges from original offline preprocessing: IoU {iou}"
    );
    assert!(!prepared.plate, "fixture is a positive erasing case");
    assert!(prepared
        .erase
        .as_ref()
        .unwrap()
        .pixels
        .pixels()
        .any(|p| p[3] != 0));
}
