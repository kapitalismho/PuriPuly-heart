use super::*;
use image::Rgb;
use std::path::PathBuf;

fn photo(width: u32, height: u32) -> WorkingPhoto {
    WorkingPhoto::new(Arc::new(RgbImage::new(width, height)), [width, height]).unwrap()
}

#[test]
fn decoded_source_is_shared_and_custom_aspects_use_actual_content_height() {
    let original = Arc::new(RgbImage::new(1920, 1080));
    let work = WorkingPhoto::new(original.clone(), DEFAULT_MAX_DIMENSIONS).unwrap();
    assert!(Arc::ptr_eq(&original, &work.pixels));
    assert_eq!(work.reference_scale, 0.75);
    let wide =
        WorkingPhoto::new(Arc::new(RgbImage::new(3840, 1080)), DEFAULT_MAX_DIMENSIONS).unwrap();
    assert_eq!(wide.pixels.dimensions(), (1920, 540));
    assert_eq!(wide.reference_scale, 0.375);
    assert_eq!(
        wide.map_bbox([100.5, 20.5, 200.5, 50.5]),
        Rect {
            left: 50.25,
            top: 10.25,
            right: 100.25,
            bottom: 25.25
        }
    );
}

#[test]
fn supported_resolutions_scale_absolute_constants_once_not_derived_geometry() {
    for (width, height) in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)] {
        let work = photo(width, height);
        let s = height as f32 / 1440.0;
        let bbox = Rect {
            left: 100.0 * s,
            top: 100.0 * s,
            right: 500.0 * s,
            bottom: 220.0 * s,
        };
        let (_, roi, glyph) = geometry(&work, bbox, 3).unwrap();
        assert_eq!(glyph, 40.0 * s);
        let expected = (0.3 * glyph)
            .round_ties_even()
            .clamp(6.0 * s, 40.0 * s)
            .round_ties_even()
            .max(1.0);
        assert_eq!(roi.x, (bbox.left - expected).floor() as u32);
        let tiny = Rect {
            bottom: bbox.top + 1.0,
            ..bbox
        };
        assert_eq!(geometry(&work, tiny, 3).unwrap().2, 8.0 * s);
    }
    let work =
        WorkingPhoto::new(Arc::new(RgbImage::new(3840, 2160)), DEFAULT_MAX_DIMENSIONS).unwrap();
    assert_eq!(work.reference_scale, 0.75);
    assert_eq!(
        geometry(&work, work.map_bbox([200.0, 200.0, 1000.0, 440.0]), 3)
            .unwrap()
            .2,
        40.0
    );
}

#[test]
fn source_geometry_requires_complete_physical_line_count_and_vertical_extent() {
    let work = photo(2560, 1440);
    let bbox = Rect {
        left: 10.25,
        top: 20.5,
        right: 50.25,
        bottom: 220.5,
    };
    assert_eq!(
        geometry(&work, bbox, 0).unwrap_err(),
        "photo_source_lines_required"
    );
    assert_eq!(geometry(&work, bbox, 2).unwrap().2, 20.0);
    assert!(candidate_region(&work, bbox).unwrap().height >= 200);
    assert!(geometry(
        &work,
        Rect {
            left: f32::NAN,
            ..bbox
        },
        1
    )
    .is_err());
    assert!(geometry(
        &work,
        Rect {
            right: -1.0,
            ..bbox
        },
        1
    )
    .is_err());
}

#[test]
fn keep_rule_matches_nfkc_unicode_alphanumeric_lowercase() {
    assert!(keep_original(" ＡＢＣ＿１２３！\n", "abc123"));
    assert!(keep_original("Straße — Ⅳ", "straße IV"));
    assert!(keep_original("ガ", "ガ"));
    assert!(keep_original("Ος", "ΟΣ"));
    assert!(keep_original("عَلَم", "علم"));
    assert!(keep_original("\u{0345}", ""));
    assert!(!keep_original("Tea 5", "Tea 6"));
    assert!(!keep_original("한글", "中文"));
}

#[test]
fn long_side_tiles_cover_the_edge_and_keep_dimensionless_overlap() {
    assert_eq!(tiles(100, 50), vec![(0, 100)]);
    assert_eq!(tiles(251, 50), vec![(0, 100), (85, 185), (151, 251)]);
    assert_eq!(tiles(5, 1), vec![(0, 2), (1, 3), (2, 4), (3, 5)]);
}

#[test]
fn ellipse_dilation_and_sampling_ring_preserve_reference_shape() {
    let mut core = GrayImage::new(9, 9);
    core.put_pixel(4, 4, Luma([255]));
    assert_eq!(
        dilate(&core, 1, true)
            .as_raw()
            .iter()
            .filter(|&&p| p != 0)
            .count(),
        5
    );
    assert_eq!(
        dilate(&core, 3, false)
            .as_raw()
            .iter()
            .filter(|&&p| p != 0)
            .count(),
        49
    );
}

#[test]
fn fill_sampling_and_leftover_detect_unmasked_same_color_text() {
    let mut original = RgbImage::from_pixel(100, 100, Rgb([240, 240, 240]));
    let mut core = GrayImage::new(100, 100);
    for y in 35..65 {
        for x in 35..45 {
            core.put_pixel(x, y, Luma([255]));
            original.put_pixel(x, y, Rgb([10, 20, 30]));
        }
        for x in 55..65 {
            original.put_pixel(x, y, Rgb([10, 20, 30]));
        }
    }
    let work = WorkingPhoto {
        pixels: Arc::new(original),
        scale_x: 1.0,
        scale_y: 1.0,
        reference_scale: 1.0,
    };
    let bbox = Rect {
        left: 20.0,
        top: 20.0,
        right: 80.0,
        bottom: 80.0,
    };
    let roi = PixelRect {
        x: 0,
        y: 0,
        width: 100,
        height: 100,
    };
    let mask = dilate(&core, 1, true);
    let (fill, leftover, count) = colors_and_leftover(&work, bbox, roi, &core, &mask, 30.0);
    assert_eq!(fill, [10, 20, 30]);
    assert_eq!(count, 300);
    assert_eq!(leftover, 1.0);
}

#[test]
fn wider_erase_preserves_core_fill_and_coverage_but_can_reduce_leftover_plate() {
    let mut pixels = RgbImage::from_pixel(100, 100, Rgb([240, 240, 240]));
    let mut core = GrayImage::new(100, 100);
    for y in 35..65 {
        for x in 40..50 {
            core.put_pixel(x, y, Luma([255]));
            pixels.put_pixel(x, y, Rgb([10, 20, 30]));
        }
        for x in 51..54 {
            pixels.put_pixel(x, y, Rgb([10, 20, 30]));
        }
    }
    let work = WorkingPhoto {
        pixels: Arc::new(pixels),
        scale_x: 1.0,
        scale_y: 1.0,
        reference_scale: 0.75,
    };
    let bbox = Rect {
        left: 20.0,
        top: 20.0,
        right: 80.0,
        bottom: 80.0,
    };
    let roi = PixelRect {
        x: 0,
        y: 0,
        width: 100,
        height: 100,
    };
    let before = colors_and_leftover(&work, bbox, roi, &core, &dilate(&core, 1, true), 20.0);
    let after = colors_and_leftover(
        &work,
        bbox,
        roi,
        &core,
        &dilate(&core, erase_radius(20.0, work.reference_scale), true),
        20.0,
    );
    assert_eq!(before.0, [10, 20, 30]);
    assert_eq!(after.0, before.0);
    assert_eq!(before.2, 300);
    assert_eq!(after.2, before.2);
    assert_eq!(before.1, 0.3);
    assert_eq!(after.1, 0.2);
    let plate = |count: usize, leftover: f32| {
        count as f32 / (bbox.width() * bbox.height()) < 0.03 || leftover >= 0.3
    };
    assert!(plate(before.2, before.1));
    assert!(!plate(after.2, after.1));
    for mask in [&dilate(&core, 1, true), &dilate(&core, 10, true)] {
        let sparse = GrayImage::from_fn(100, 100, |x, y| {
            Luma([if x == 40 && (35..65).contains(&y) {
                255
            } else {
                0
            }])
        });
        let (_, leftover, count) = colors_and_leftover(&work, bbox, roi, &sparse, mask, 20.0);
        assert_eq!(count, 30);
        assert!(plate(count, leftover));
    }
}

#[test]
fn refreshed_erase_uses_actual_height_radius_and_preserves_prepared_measurements() {
    for (height, extra) in [(719, 1), (1068, 1), (1080, 1), (1440, 1), (2160, 2)] {
        let work = WorkingPhoto::new(
            Arc::new(RgbImage::from_fn(height, height, |x, y| {
                Rgb([
                    (x * 17 % 251) as u8,
                    (y * 23 % 251) as u8,
                    (x * y % 251) as u8,
                ])
            })),
            [height, height],
        )
        .unwrap();
        let roi = PixelRect {
            x: 20,
            y: 20,
            width: 60,
            height: 60,
        };
        let core = GrayImage::from_fn(60, 60, |x, y| {
            Luma([if (20..40).contains(&x) && (20..40).contains(&y) {
                255
            } else {
                0
            }])
        });
        for (glyph, baseline) in [(20.0, 1), (50.0, 2), (70.0, 4)] {
            let mut region = PreparedRegion {
                bbox: Rect {
                    left: 40.0,
                    top: 40.0,
                    right: 60.0,
                    bottom: 60.0,
                },
                roi,
                core: core.clone(),
                glyph_size: glyph,
                fill: [7, 8, 9],
                plate: false,
                erase: None,
                mask_ms: 17.0,
                erase_ms: 19.0,
            };
            let expected = erase_patch(
                &work,
                roi,
                &dilate(&core, baseline + extra, true),
                &work.pixels,
            )
            .unwrap();
            let old =
                erase_patch(&work, roi, &dilate(&core, baseline, true), &work.pixels).unwrap();
            assert_ne!(expected.pixels, old.pixels);
            refresh_erase(&work, &mut region, &work.pixels).unwrap();
            let actual = region.erase.as_ref().unwrap();
            assert_eq!(actual.rect, roi);
            assert_eq!(actual.pixels, expected.pixels);
            assert_eq!(region.core, core);
            assert_eq!(region.fill, [7, 8, 9]);
            assert_eq!(region.glyph_size, glyph);
            assert!(!region.plate);
            assert_eq!(region.mask_ms, 17.0);
            assert!(region.erase_ms > 19.0);
        }
    }
}

#[test]
#[ignore = "requires selected external ONNX, ORT and controlled mask.json source fixture"]
fn selected_model_prepare_and_refresh_share_actual_scale_erase_support() {
    let root =
        PathBuf::from(std::env::var_os("PURI_PHOTO340_REFERENCE").expect("reference fixtures"));
    let model = PathBuf::from(std::env::var_os("PURI_PHOTO340_MODEL").expect("chosen ONNX model"));
    let runtime = PathBuf::from(std::env::var_os("PURI_PHOTO340_ORT").expect("ONNX Runtime"));
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("mask.json")).unwrap()).unwrap();
    let original = Arc::new(
        image::open(fixture["source"].as_str().unwrap())
            .unwrap()
            .into_rgb8(),
    );
    let bbox: [f64; 4] = std::array::from_fn(|i| fixture["bbox"][i].as_f64().unwrap());
    let mut processor = MaskProcessor::new(&model, Some(&runtime)).unwrap();
    for (height, extra) in [(719, 1), (1068, 1), (1080, 1), (1440, 1), (2160, 2)] {
        let width = (f64::from(original.width()) * f64::from(height) / f64::from(original.height()))
            .ceil() as u32;
        let work = WorkingPhoto::new(original.clone(), [width, height]).unwrap();
        assert_eq!(work.pixels.height(), height);
        let mut region = processor
            .prepare(
                &work,
                work.map_bbox(bbox),
                fixture["lines"].as_u64().unwrap() as u32,
                &work.pixels,
            )
            .unwrap();
        assert!(
            !region.plate,
            "fixture must remain a positive erase case at {height}"
        );
        let core = region.core.clone();
        assert!(core.pixels().filter(|p| p[0] != 0).count() > 100);
        let radius = (0.05 * region.glyph_size).round_ties_even().max(1.0) as u32 + extra;
        let expected = erase_patch(
            &work,
            region.roi,
            &dilate(&core, radius, true),
            &work.pixels,
        )
        .unwrap();
        assert_eq!(region.erase.as_ref().unwrap().pixels, expected.pixels);
        let measurements = (
            region.fill,
            region.plate,
            region.mask_ms,
            region.bbox,
            region.roi,
        );
        refresh_erase(&work, &mut region, &work.pixels).unwrap();
        assert_eq!(region.erase.as_ref().unwrap().pixels, expected.pixels);
        assert_eq!(region.core, core);
        assert_eq!(
            (
                region.fill,
                region.plate,
                region.mask_ms,
                region.bbox,
                region.roi
            ),
            measurements
        );
    }
}

#[test]
fn erase_patch_changes_only_masked_pixels_and_keeps_source_immutable() {
    let original = Arc::new(RgbImage::from_fn(100, 100, |x, y| {
        if (40..60).contains(&x) && (40..60).contains(&y) {
            Rgb([0, 0, 0])
        } else {
            Rgb([200, 180, 160])
        }
    }));
    let work = WorkingPhoto {
        pixels: original.clone(),
        scale_x: 1.0,
        scale_y: 1.0,
        reference_scale: 1.0,
    };
    let roi = PixelRect {
        x: 30,
        y: 30,
        width: 40,
        height: 40,
    };
    let mask = GrayImage::from_fn(40, 40, |x, y| {
        Luma([if (10..30).contains(&x) && (10..30).contains(&y) {
            255
        } else {
            0
        }])
    });
    let patch = erase_patch(&work, roi, &mask, &original).unwrap();
    let mut changed = 0;
    for (x, y, pixel) in patch.pixels.enumerate_pixels() {
        if mask.get_pixel(x, y)[0] == 0 {
            assert_eq!(pixel[3], 0);
        }
        if pixel[3] != 0 {
            changed += 1;
            assert_ne!(&pixel.0[..3], &original.get_pixel(x + 30, y + 30).0);
        }
    }
    assert_eq!(changed, 400);
    assert_eq!(original.get_pixel(50, 50).0, [0, 0, 0]);
}

#[test]
fn telea_rejects_mismatch_and_leaves_unmasked_image_unchanged() {
    let image = RgbImage::from_fn(17, 15, |x, y| Rgb([x as u8 * 10, y as u8 * 10, 70]));
    assert_eq!(
        telea::inpaint(&image, &GrayImage::new(17, 15), 5).unwrap(),
        image
    );
    assert!(telea::inpaint(&image, &GrayImage::new(16, 15), 5).is_err());
}
#[test]
#[ignore = "requires controlled offline OpenCV reference fixtures via PURI_PHOTO340_REFERENCE"]
fn rust_telea_matches_controlled_opencv_reference() {
    let root = PathBuf::from(
        std::env::var_os("PURI_PHOTO340_REFERENCE").expect("reference fixture directory"),
    );
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("telea.json")).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let source = image::open(root.join(format!("{name}_input.png")))
            .unwrap()
            .into_rgb8();
        let mask = image::open(root.join(format!("{name}_mask.png")))
            .unwrap()
            .into_luma8();
        let expected = image::open(root.join(format!("{name}_expected.png")))
            .unwrap()
            .into_rgb8();
        let start = std::time::Instant::now();
        let actual =
            telea::inpaint(&source, &mask, case["radius"].as_u64().unwrap() as u32).unwrap();
        let elapsed = start.elapsed();
        let differences = actual
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .filter(|(a, b)| a != b)
            .count();
        let max_difference = actual
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        eprintln!(
            "Telea {name}: {differences} channel differences, max={max_difference}, rust_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0
        );
        assert_eq!(
            differences, 0,
            "OpenCV reference case {name}, maximum channel difference {max_difference}"
        );
        for (index, &value) in mask.as_raw().iter().enumerate() {
            if value == 0 {
                assert_eq!(
                    &actual.as_raw()[index * 3..index * 3 + 3],
                    &source.as_raw()[index * 3..index * 3 + 3]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires controlled offline OpenCV byte resampling references"]
fn mask_preprocessing_matches_opencv_linear_bytes() {
    let root = PathBuf::from(
        std::env::var_os("PURI_PHOTO340_REFERENCE").expect("reference fixture directory"),
    );
    let cases: Vec<String> =
        serde_json::from_slice(&std::fs::read(root.join("resize.json")).unwrap()).unwrap();
    for name in cases {
        let input = image::open(root.join(format!("resize_{name}_input.png")))
            .unwrap()
            .into_rgb8();
        let expected = image::open(root.join(format!("resize_{name}_expected.png")))
            .unwrap()
            .into_rgb8();
        let mut actual = vec![0; MODEL_SIZE * MODEL_SIZE * 3];
        resize_tile(
            &input,
            PixelRect {
                x: 0,
                y: 0,
                width: input.width(),
                height: input.height(),
            },
            &mut actual,
        );
        let mut differences = 0;
        let mut maximum = 0;
        for (x, y, pixel) in expected.enumerate_pixels() {
            for channel in 0..3 {
                let value = actual
                    [channel * MODEL_SIZE * MODEL_SIZE + y as usize * MODEL_SIZE + x as usize];
                differences += usize::from(value != pixel[channel]);
                maximum = maximum.max(value.abs_diff(pixel[channel]));
            }
        }
        eprintln!("Mask input resize {name}: {differences} differing bytes, max={maximum}");
        assert_eq!(
            differences, 0,
            "OpenCV byte resize case {name}, max={maximum}"
        );
    }
}

#[test]
fn fill_uses_the_farthest_forty_percent_not_the_core_average() {
    let mut pixels = RgbImage::from_pixel(40, 30, Rgb([240, 240, 240]));
    let mut core = GrayImage::new(40, 30);
    for x in 10..20 {
        core.put_pixel(x, 15, Luma([255]));
        pixels.put_pixel(
            x,
            15,
            Rgb(if x < 16 {
                [220, 220, 220]
            } else {
                [100, 10, 30]
            }),
        );
    }
    let work = WorkingPhoto {
        pixels: Arc::new(pixels),
        scale_x: 1.0,
        scale_y: 1.0,
        reference_scale: 1.0,
    };
    let roi = PixelRect {
        x: 0,
        y: 0,
        width: 40,
        height: 30,
    };
    let bbox = Rect {
        left: 5.0,
        top: 5.0,
        right: 35.0,
        bottom: 25.0,
    };
    assert_eq!(
        colors_and_leftover(&work, bbox, roi, &core, &core, 10.0).0,
        [100, 10, 30]
    );
}

fn apply_erase(background: &mut RgbImage, patch: &ErasePatch) {
    for (x, y, pixel) in patch.pixels.enumerate_pixels() {
        if pixel[3] != 0 {
            background.put_pixel(
                patch.rect.x + x,
                patch.rect.y + y,
                Rgb([pixel[0], pixel[1], pixel[2]]),
            );
        }
    }
}

#[test]
fn erase_dependency_is_the_exact_scaled_expanded_input() {
    let work = photo(1920, 1080);
    let roi = PixelRect {
        x: 5,
        y: 4,
        width: 20,
        height: 15,
    };
    assert_eq!(
        erase_input_rect(&work, roi),
        PixelRect {
            x: 0,
            y: 0,
            width: 37,
            height: 31
        }
    );
    let roi = PixelRect {
        x: 1900,
        y: 1070,
        width: 20,
        height: 10,
    };
    assert_eq!(
        erase_input_rect(&work, roi),
        PixelRect {
            x: 1888,
            y: 1058,
            width: 32,
            height: 22
        }
    );
}

#[test]
fn refresh_rejects_wrong_background_without_changing_region_and_skips_plates() {
    let work = photo(60, 40);
    let mut region = PreparedRegion {
        bbox: Rect {
            left: 10.0,
            top: 10.0,
            right: 30.0,
            bottom: 30.0,
        },
        roi: PixelRect {
            x: 5,
            y: 5,
            width: 30,
            height: 30,
        },
        core: GrayImage::new(30, 30),
        glyph_size: 20.0,
        fill: [1, 2, 3],
        plate: false,
        erase: None,
        mask_ms: 17.0,
        erase_ms: 19.0,
    };
    assert_eq!(
        refresh_erase(&work, &mut region, &RgbImage::new(59, 40)).unwrap_err(),
        "photo_erase_source_dimensions"
    );
    assert_eq!(region.mask_ms, 17.0);
    assert_eq!(region.erase_ms, 19.0);
    assert!(region.erase.is_none());
    region.plate = true;
    refresh_erase(&work, &mut region, &RgbImage::new(59, 40)).unwrap();
    assert!(region.erase.is_none());
    assert_eq!(region.erase_ms, 19.0);
}

#[test]
#[ignore = "requires controlled overlapping sequential OpenCV references"]
fn sequential_telea_reference_and_wider_refresh_preserve_committed_context() {
    let root = PathBuf::from(
        std::env::var_os("PURI_PHOTO340_REFERENCE").expect("reference fixture directory"),
    );
    let original = Arc::new(image::open(root.join("seq_input.png")).unwrap().into_rgb8());
    let work = WorkingPhoto::new(original.clone(), [original.width(), original.height()]).unwrap();
    assert_eq!(work.reference_scale, 1.0);
    let steps: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("sequential.json")).unwrap()).unwrap();
    let roi = |index: usize| {
        let a = steps[index]["roi"].as_array().unwrap();
        PixelRect {
            x: a[0].as_u64().unwrap() as u32,
            y: a[1].as_u64().unwrap() as u32,
            width: a[2].as_u64().unwrap() as u32,
            height: a[3].as_u64().unwrap() as u32,
        }
    };
    let first_core = image::open(root.join("seq0_core.png"))
        .unwrap()
        .into_luma8();
    let second_core = image::open(root.join("seq1_core.png"))
        .unwrap()
        .into_luma8();
    let first_patch = erase_patch(&work, roi(0), &dilate(&first_core, 1, true), &original).unwrap();
    let hidden_patch =
        erase_patch(&work, roi(1), &dilate(&second_core, 1, true), &original).unwrap();
    let mut committed = (*original).clone();
    apply_erase(&mut committed, &first_patch);
    let first_expected = image::open(root.join("seq0_expected.png"))
        .unwrap()
        .into_rgb8();
    assert_eq!(committed.as_raw(), first_expected.as_raw());
    let second_expected = image::open(root.join("seq1_expected.png"))
        .unwrap()
        .into_rgb8();
    let mut stale_result = committed.clone();
    apply_erase(&mut stale_result, &hidden_patch);
    assert_ne!(
        stale_result.as_raw(),
        second_expected.as_raw(),
        "fixture must distinguish original from committed Telea context"
    );
    let reference_patch =
        erase_patch(&work, roi(1), &dilate(&second_core, 1, true), &committed).unwrap();
    let mut reference_result = committed.clone();
    apply_erase(&mut reference_result, &reference_patch);
    assert_eq!(reference_result.as_raw(), second_expected.as_raw());
    let wider_mask = dilate(&second_core, 2, true);
    let expected_patch = erase_patch(&work, roi(1), &wider_mask, &committed).unwrap();
    let stale_patch = erase_patch(&work, roi(1), &wider_mask, &original).unwrap();
    let mut expected_result = committed.clone();
    apply_erase(&mut expected_result, &expected_patch);
    let mut stale_wider_result = committed.clone();
    apply_erase(&mut stale_wider_result, &stale_patch);
    assert_ne!(stale_wider_result.as_raw(), expected_result.as_raw());
    let mut second = PreparedRegion {
        bbox: Rect {
            left: 45.0,
            top: 35.0,
            right: 60.0,
            bottom: 50.0,
        },
        roi: roi(1),
        core: second_core.clone(),
        glyph_size: 20.0,
        fill: [7, 8, 9],
        plate: false,
        erase: Some(hidden_patch),
        mask_ms: 17.0,
        erase_ms: 19.0,
    };
    refresh_erase(&work, &mut second, &committed).unwrap();
    assert_eq!(second.core, second_core);
    assert_eq!(second.fill, [7, 8, 9]);
    assert_eq!(second.mask_ms, 17.0);
    assert!(second.erase_ms > 19.0);
    apply_erase(&mut committed, second.erase.as_ref().unwrap());
    let differences = committed
        .as_raw()
        .iter()
        .zip(expected_result.as_raw())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("Sequential overlapping erase: {differences} differing channels");
    assert_eq!(differences, 0);
    assert_eq!(original.as_raw(), work.pixels.as_raw());
}

#[test]
fn portrait_photos_fit_the_default_bounds_turned_upright_with_short_side_scale() {
    for (source, expected) in [([3840, 2160], [1920, 1080]), ([2160, 3840], [1080, 1920]), ([1440, 2048], [1080, 1536])] {
        let work = WorkingPhoto::new(Arc::new(RgbImage::new(source[0], source[1])), [1920, 1080]).unwrap();
        assert_eq!(work.pixels.dimensions(), (expected[0], expected[1]));
        assert!((work.reference_scale - expected[0].min(expected[1]) as f32 / 1440.0).abs() < 0.00001);
    }
}
