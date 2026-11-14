use super::*;
use serde_json::json;

fn input(
    revision: u64,
    attempt: u64,
    blocks: serde_json::Value,
    preparations: serde_json::Value,
) -> PhotoSnapshot {
    PhotoSnapshot {revision,photo:Some(Arc::new(serde_json::from_value(json!({
        "photo_id":"synthetic","path":"no-decoder-must-open-this.png","file_size":1,"shutter_unix_ms":1,
        "expected_file_id":["1","2"],
        "attempt_id":attempt,"phase":"requesting","offer_open":false,"visible":false,
        "image":{"path":"retained-only.png","width":128,"height":72},"blocks":blocks,"preparations":preparations,
        "failure_code":null,"first_block_drawn":false,
        "excluded_blocks":[],
    })).unwrap()))}
}

fn kept(id: u64) -> serde_json::Value {
    json!({"block_id":id,"bbox":[1.25,2.5,100.5,50.25],"source":"Brand 123","translation":"Brand 123"})
}

#[test]
fn coalesced_cumulative_blocks_preserve_arrival_order_without_redecoding_or_model_initialization() {
    let pixels = Arc::new(image::RgbImage::new(128, 72));
    let mut pipeline = ImagePipeline::new(PhotoImageConfig {
        max_size: [128, 72],
        model_path: Some("missing-selected-model.onnx".into()),
        ..Default::default()
    });
    let snapshot = input(2, 1, json!([kept(8), kept(3)]), json!([]));
    let mut completed = Vec::new();
    let mut original = None;
    pipeline
        .process(
            &snapshot,
            Some(pixels.clone()),
            || true,
            |output| {
                match output {
                    PhotoImageOutput::Original { working, recovered } => {
                        assert!(!recovered);
                        assert!(Arc::ptr_eq(&working.pixels, &pixels));
                        original = Some(working);
                    }
                    PhotoImageOutput::Block {
                        block_id, region, ..
                    } => {
                        assert!(region.is_none());
                        completed.push(block_id);
                    }
                    _ => panic!("kept blocks must not initialize masking"),
                }
                true
            },
        )
        .unwrap();
    assert_eq!(completed, vec![8, 3]);
    assert!(pipeline.processor.is_none());
    pipeline
        .process(
            &snapshot,
            Some(pixels.clone()),
            || true,
            |_| panic!("completed blocks must not repeat"),
        )
        .unwrap();
    let retry = input(3, 2, json!([kept(5)]), json!([]));
    pipeline
        .process(
            &retry,
            Some(pixels),
            || true,
            |output| {
                match output {
                    PhotoImageOutput::Original { working, .. } => {
                        assert!(Arc::ptr_eq(&working, original.as_ref().unwrap()))
                    }
                    PhotoImageOutput::Block { block_id, .. } => assert_eq!(block_id, 5),
                    _ => panic!("retry must reuse the source and clear only upper-layer work"),
                }
                true
            },
        )
        .unwrap();
}

#[test]
fn bbox_candidate_does_not_invent_source_geometry_and_missing_model_is_explicit_at_source_ready() {
    let pixels = Arc::new(image::RgbImage::new(128, 72));
    let mut pipeline = ImagePipeline::new(PhotoImageConfig {
        max_size: [128, 72],
        model_path: Some("missing-selected-model.onnx".into()),
        ..Default::default()
    });
    let bbox = json!([{"block_id":0,"bbox":[10,10,100,50],"source":null,"source_line_count":null}]);
    let snapshot = input(1, 1, json!([]), bbox);
    let mut candidate = false;
    pipeline
        .process(
            &snapshot,
            Some(pixels.clone()),
            || true,
            |output| {
                match output {
                    PhotoImageOutput::Original { .. } => {}
                    PhotoImageOutput::Candidate { rect, .. } => {
                        assert!(rect.width >= 90);
                        assert!(rect.height >= 40);
                        candidate = true;
                    }
                    _ => panic!("bbox alone must never mask or reveal"),
                }
                true
            },
        )
        .unwrap();
    assert!(candidate);
    assert!(pipeline.processor.is_none());
    assert!(pipeline.prepared.is_empty());
    let source = input(
        2,
        1,
        json!([]),
        json!([{"block_id":0,"bbox":[10,10,100,50],"source":"A\nB","source_line_count":2}]),
    );
    let result = pipeline.process(
        &source,
        Some(pixels),
        || true,
        |_| panic!("missing model must not pretend erase is ready"),
    );
    assert_eq!(result, Err("mask_model_missing"));
}

fn prepared_patch() -> PreparedRegion {
    PreparedRegion {
        bbox: Rect {
            left: 10.0,
            top: 10.0,
            right: 70.0,
            bottom: 40.0,
        },
        roi: PixelRect {
            x: 10,
            y: 10,
            width: 60,
            height: 30,
        },
        recovered: image::GrayImage::from_pixel(60, 30, image::Luma([255])),
        core: image::GrayImage::from_pixel(60, 30, image::Luma([255])),
        glyph_size: 16.0,
        source_lines: 1,
        fill: [10, 20, 30],
        erase: Some(super::super::processing::ErasePatch {
            rect: PixelRect {
                x: 9,
                y: 12,
                width: 10,
                height: 1,
            },
            pixels: image::RgbaImage::from_fn(10, 1, |x, _| {
                image::Rgba([200, 20, 30, if x == 0 || x == 9 { 255 } else { 0 }])
            }),
        }),
        mask_ms: 17.0,
        erase_ms: 2.0,
    }
}

fn pipeline_with_pending_region() -> (ImagePipeline, Arc<image::RgbImage>) {
    let pixels = Arc::new(image::RgbImage::new(128, 72));
    let mut pipeline = ImagePipeline::new(PhotoImageConfig {
        max_size: [128, 72],
        model_path: Some("missing-selected-model.onnx".into()),
        ..Default::default()
    });
    pipeline
        .process(
            &input(1, 1, json!([]), json!([])),
            Some(pixels.clone()),
            || true,
            |_| true,
        )
        .unwrap();
    pipeline.candidates.insert(99);
    pipeline.prepared.insert(
        99,
        PendingRegion {
            region: prepared_patch(),
            context_revision: 0,
        },
    );
    pipeline.commit_erase(&prepared_patch());
    (pipeline, pixels)
}

fn source_preparation(id: u64) -> serde_json::Value {
    json!({"block_id":id,"bbox":[10,10,70,40],"source":"SOURCE","source_line_count":1})
}

#[test]
fn committed_context_tracks_actual_changed_runs_and_retry_discards_only_attempt_pixels() {
    let original = image::RgbImage::new(128, 72);
    let mut pipeline = ImagePipeline::new(PhotoImageConfig::default());
    pipeline.erase_source = Some(original.clone());
    let region = prepared_patch();
    pipeline.prepared.insert(
        99,
        PendingRegion {
            region: prepared_patch(),
            context_revision: 0,
        },
    );
    pipeline.commit_erase(&region);
    assert_eq!(pipeline.commit_revision, 1);
    assert_eq!(
        pipeline.erase_source.as_ref().unwrap().get_pixel(9, 12).0,
        [200, 20, 30]
    );
    assert_eq!(
        pipeline.erase_source.as_ref().unwrap().get_pixel(10, 12).0,
        [0, 0, 0]
    );
    assert_eq!(pipeline.changes[0].1.len(), 2);
    let unchanged_hole = PixelRect {
        x: 11,
        y: 12,
        width: 3,
        height: 1,
    };
    assert!(!pipeline.changes[0]
        .1
        .iter()
        .any(|run| intersects(*run, unchanged_hole)));
    pipeline.commit_erase(&region);
    assert_eq!(pipeline.commit_revision, 1);
    pipeline.clear_attempt(2);
    assert!(pipeline.erase_source.is_none());
    assert!(pipeline.changes.is_empty());
    assert_eq!(pipeline.commit_revision, 0);
    assert_eq!(original.get_pixel(9, 12).0, [0, 0, 0]);
}

#[test]
fn keep_completion_discards_hidden_erase_without_contaminating_next_region() {
    let pixels = Arc::new(image::RgbImage::new(128, 72));
    let mut pipeline = ImagePipeline::new(PhotoImageConfig {
        max_size: [128, 72],
        ..Default::default()
    });
    pipeline
        .process(
            &input(1, 1, json!([]), json!([])),
            Some(pixels.clone()),
            || true,
            |_| true,
        )
        .unwrap();
    pipeline.erase_source = Some((*pixels).clone());
    pipeline.prepared.insert(
        8,
        PendingRegion {
            region: prepared_patch(),
            context_revision: 0,
        },
    );
    let snapshot = input(2, 1, json!([kept(8)]), json!([]));
    pipeline
        .process(
            &snapshot,
            Some(pixels.clone()),
            || true,
            |output| {
                assert!(matches!(
                    output,
                    PhotoImageOutput::Block { region: None, .. }
                ));
                true
            },
        )
        .unwrap();
    assert!(pipeline.prepared.is_empty());
    assert_eq!(pipeline.erase_source.as_ref().unwrap(), &*pixels);
    assert_eq!(pipeline.commit_revision, 0);
}

#[cfg(windows)]
#[test]
fn hidden_erase_refresh_reuses_mask_and_only_current_successful_text_commits() {
    use std::cell::Cell;
    let mut source = image::RgbImage::new(128, 72);
    for y in 18..25 {
        for x in 18..55 {
            source.put_pixel(x, y, image::Rgb([255, 255, 255]));
        }
    }
    let pixels = Arc::new(source);
    let mut pipeline = ImagePipeline::new(PhotoImageConfig {
        max_size: [128, 72],
        ..Default::default()
    });
    pipeline
        .process(
            &input(1, 1, json!([]), json!([])),
            Some(pixels.clone()),
            || true,
            |_| true,
        )
        .unwrap();
    pipeline.erase_source = Some((*pixels).clone());
    let mut pending = prepared_patch();
    pending.core = image::GrayImage::new(60, 30);
    for y in 8..15 {
        for x in 8..45 {
            pending.core.put_pixel(x, y, image::Luma([255]));
        }
    }
    let core = pending.core.clone();
    pipeline.prepared.insert(
        1,
        PendingRegion {
            region: pending,
            context_revision: 0,
        },
    );
    pipeline.commit_erase(&prepared_patch());
    let snapshot = input(
        2,
        1,
        json!([{"block_id":1,"bbox":[10,10,70,40],"source":"SOURCE","translation":"Translated"}]),
        json!([]),
    );
    let calls = Cell::new(0);
    let before = pipeline.erase_source.as_ref().unwrap().clone();
    pipeline
        .process(
            &snapshot,
            Some(pixels.clone()),
            || {
                let call = calls.get() + 1;
                calls.set(call);
                call < 3
            },
            |output| {
                assert!(matches!(
                    output,
                    PhotoImageOutput::Timing { stage: "erase", .. }
                ));
                true
            },
        )
        .unwrap();
    assert_eq!(pipeline.erase_source.as_ref().unwrap(), &before);
    assert_eq!(pipeline.commit_revision, 1);
    let mut pending = prepared_patch();
    pending.core = core.clone();
    pipeline.prepared.insert(
        1,
        PendingRegion {
            region: pending,
            context_revision: 0,
        },
    );
    let mut completed = false;
    pipeline
        .process(
            &snapshot,
            Some(pixels),
            || true,
            |output| {
                if let PhotoImageOutput::Timing { elapsed_ms, .. } = output {
                    assert!(elapsed_ms >= 0.0);
                    return true;
                }
                if let PhotoImageOutput::FontReady { font } = output {
                    assert!(!font.family.is_empty());
                    assert!(font.hangul_ink_ratio > 0.0);
                    return true;
                }
                let PhotoImageOutput::Block {
                    region: Some(region),
                    ..
                } = output
                else {
                    panic!("current translation must complete")
                };
                assert_eq!(region.core, core);
                assert_eq!(region.fill, [10, 20, 30]);
                assert_eq!(region.mask_ms, 17.0);
                assert!(region.erase_ms > 2.0);
                completed = true;
                true
            },
        )
        .unwrap();
    assert!(completed);
    assert!(pipeline.commit_revision >= 2);
}

#[test]
fn quarantined_pending_erase_is_discarded_without_modifying_the_original() {
    let pixels = Arc::new(image::RgbImage::new(128, 72));
    let mut pipeline = ImagePipeline::new(PhotoImageConfig {
        max_size: [128, 72],
        ..Default::default()
    });
    pipeline.process(
        &input(1, 1, json!([]), json!([])),
        Some(pixels.clone()),
        || true,
        |_| true,
    ).unwrap();
    pipeline.candidates.insert(0);
    pipeline.prepared.insert(0, PendingRegion {
        region: prepared_patch(),
        context_revision: 0,
    });
    let mut next = input(2, 1, json!([kept(1)]), json!([]));
    Arc::get_mut(next.photo.as_mut().unwrap()).unwrap().excluded_blocks = vec![0];
    let mut completed = Vec::new();
    pipeline.process(
        &next,
        Some(pixels.clone()),
        || true,
        |output| {
            if let PhotoImageOutput::Block { block_id, region, .. } = output {
                assert!(region.is_none());
                completed.push(block_id);
            }
            true
        },
    ).unwrap();
    assert_eq!(completed, vec![1]);
    assert!(pipeline.prepared.is_empty());
    assert!(!pipeline.candidates.contains(&0));
    assert!(pipeline.erase_source.is_none());
    assert_eq!(pipeline.working.as_ref().unwrap().pixels.as_ref(), pixels.as_ref());
}

#[test]
fn terminal_phases_drain_blocks_and_retire_unused_work_without_reprocessing_on_reopen() {
    for phase in [
        PhotoPhase::Completed,
        PhotoPhase::CompletedCorrected,
        PhotoPhase::Partial,
        PhotoPhase::NoText,
        PhotoPhase::Failed,
    ] {
        let (mut pipeline, pixels) = pipeline_with_pending_region();
        let working = Arc::downgrade(pipeline.working.as_ref().unwrap());
        let expected = if phase == PhotoPhase::NoText {
            vec![]
        } else {
            vec![8, 3]
        };
        let blocks = if expected.is_empty() {
            json!([])
        } else {
            json!([kept(8), kept(3)])
        };
        let mut snapshot = input(2, 1, blocks, json!([source_preparation(99)]));
        Arc::get_mut(snapshot.photo.as_mut().unwrap()).unwrap().phase = phase;
        let mut completed = Vec::new();
        pipeline
            .process(&snapshot, Some(pixels), || true, |output| {
                let PhotoImageOutput::Block { block_id, region, .. } = output else {
                    panic!("terminal source-only blocks must not be prepared")
                };
                assert!(region.is_none());
                completed.push(block_id);
                true
            })
            .unwrap();
        assert_eq!(completed, expected);
        assert_eq!(pipeline.completed.len(), expected.len());
        assert!(expected.iter().all(|id| pipeline.completed.contains(id)));
        assert!(pipeline.processor.is_none());
        assert!(pipeline.erase_source.is_none());
        assert!(pipeline.prepared.is_empty());
        assert!(pipeline.candidates.is_empty());
        assert!(pipeline.changes.is_empty());
        assert_eq!(pipeline.commit_revision, 0);
        assert!(working.upgrade().is_none());
        snapshot.revision += 1;
        Arc::make_mut(snapshot.photo.as_mut().unwrap()).visible = true;
        pipeline
            .process(&snapshot, None, || true, |_| {
                panic!("retained terminal results must reopen without repeated work")
            })
            .unwrap();
    }
}

#[test]
fn hidden_streaming_and_interrupted_terminal_outputs_keep_needed_processing_state() {
    use std::cell::Cell;

    let (mut pipeline, pixels) = pipeline_with_pending_region();
    for phase in [PhotoPhase::Preparing, PhotoPhase::Requesting] {
        let mut snapshot = input(2, 1, json!([]), json!([source_preparation(99)]));
        Arc::get_mut(snapshot.photo.as_mut().unwrap()).unwrap().phase = phase;
        pipeline
            .process(&snapshot, Some(pixels.clone()), || true, |_| {
                panic!("already prepared hidden work must not repeat")
            })
            .unwrap();
        assert!(pipeline.prepared.contains_key(&99));
        assert!(pipeline.erase_source.is_some());
        assert!(!pipeline.changes.is_empty());
        assert_eq!(pipeline.commit_revision, 1);
    }
    let mut terminal = input(
        3,
        1,
        json!([kept(8), kept(3)]),
        json!([source_preparation(99)]),
    );
    Arc::get_mut(terminal.photo.as_mut().unwrap()).unwrap().phase = PhotoPhase::Completed;
    pipeline
        .process(&terminal, Some(pixels.clone()), || false, |_| {
            panic!("cancelled terminal work must not emit")
        })
        .unwrap();
    assert!(pipeline.completed.is_empty());
    assert!(pipeline.prepared.contains_key(&99));
    assert!(pipeline.erase_source.is_some());
    pipeline
        .process(&terminal, Some(pixels.clone()), || true, |output| {
            let PhotoImageOutput::Block { block_id, .. } = output else {
                panic!("only completed blocks may be emitted")
            };
            block_id == 8
        })
        .unwrap();
    assert!(pipeline.completed.contains(&8));
    assert!(!pipeline.completed.contains(&3));
    assert!(pipeline.prepared.contains_key(&99));
    assert!(pipeline.erase_source.is_some());
    assert!(!pipeline.changes.is_empty());
    let current = Cell::new(true);
    pipeline
        .process(&terminal, Some(pixels), || current.get(), |output| {
            assert!(matches!(output, PhotoImageOutput::Block { block_id: 3, .. }));
            current.set(false);
            true
        })
        .unwrap();
    assert!(pipeline.completed.contains(&3));
    assert!(pipeline.prepared.contains_key(&99));
    assert!(pipeline.erase_source.is_some());
    assert!(!pipeline.changes.is_empty());
    pipeline
        .process(&terminal, None, || true, |_| {
            panic!("successfully emitted terminal blocks must remain deduplicated")
        })
        .unwrap();
    assert!(pipeline.prepared.is_empty());
    assert!(pipeline.erase_source.is_none());
    assert!(pipeline.changes.is_empty());
}

#[test]
fn retired_results_allow_retry_rebuild_and_failed_attempts_do_not_leak_into_new_photos() {
    for has_unprocessed_block in [false, true] {
        let (mut pipeline, pixels) = pipeline_with_pending_region();
        let working = pipeline.working.as_ref().unwrap().clone();
        let mut terminal = input(2, 1, json!([kept(8)]), json!([source_preparation(99)]));
        Arc::get_mut(terminal.photo.as_mut().unwrap()).unwrap().phase = PhotoPhase::Completed;
        pipeline
            .process(&terminal, Some(pixels.clone()), || true, |_| true)
            .unwrap();
        let retry = input(3, 2, json!([kept(8)]), json!([]));
        let mut retried = Vec::new();
        pipeline
            .process(&retry, Some(pixels.clone()), || true, |output| {
                match output {
                    PhotoImageOutput::Original { working: original, recovered } => {
                        assert!(!recovered);
                        assert!(Arc::ptr_eq(&original.pixels, &pixels));
                    }
                    PhotoImageOutput::Block { block_id, .. } => retried.push(block_id),
                    _ => panic!("retry must rebuild only its own blocks"),
                }
                true
            })
            .unwrap();
        assert_eq!(retried, vec![8]);
        let source = input(4, 2, json!([kept(8)]), json!([source_preparation(99)]));
        assert_eq!(
            pipeline.process(&source, Some(pixels.clone()), || true, |output| {
                assert!(matches!(output, PhotoImageOutput::Candidate { block_id: 99, .. }));
                true
            }),
            Err("mask_model_missing")
        );
        assert!(pipeline.failed);
        assert!(pipeline.prepared.is_empty());
        let blocks = if has_unprocessed_block {
            json!([kept(8), kept(3)])
        } else {
            json!([kept(8)])
        };
        let mut failed = input(5, 2, blocks, json!([source_preparation(99)]));
        Arc::get_mut(failed.photo.as_mut().unwrap()).unwrap().phase = PhotoPhase::Failed;
        pipeline
            .process(&failed, Some(pixels), || true, |_| {
                panic!("a failed mask attempt must not claim further completion")
            })
            .unwrap();
        assert_eq!(pipeline.candidates.contains(&99), has_unprocessed_block);
        assert!(pipeline.completed.contains(&8));
        assert!(!pipeline.completed.contains(&3));
        let replacement_pixels = Arc::new(image::RgbImage::from_pixel(
            128,
            72,
            image::Rgb([1, 2, 3]),
        ));
        let mut replacement = input(6, 1, json!([kept(8)]), json!([]));
        Arc::get_mut(replacement.photo.as_mut().unwrap()).unwrap().photo_id = "replacement".into();
        let mut completed = Vec::new();
        pipeline
            .process(&replacement, Some(replacement_pixels.clone()), || true, |output| {
                match output {
                    PhotoImageOutput::Original { working: original, .. } => {
                        assert!(!Arc::ptr_eq(&original, &working));
                        assert!(Arc::ptr_eq(&original.pixels, &replacement_pixels));
                    }
                    PhotoImageOutput::Block { block_id, .. } => completed.push(block_id),
                    _ => panic!("replacement must not inherit retired work"),
                }
                true
            })
            .unwrap();
        assert_eq!(completed, vec![8]);
        assert!(!pipeline.failed);
        assert!(pipeline.candidates.is_empty());
        assert!(pipeline.prepared.is_empty());
        assert!(pipeline.erase_source.is_none());
    }
}

#[cfg(windows)]
#[test]
fn terminal_translated_blocks_keep_renderable_layouts_after_processing_is_retired() {
    let (mut pipeline, pixels) = pipeline_with_pending_region();
    pipeline.prepared.get_mut(&99).unwrap().context_revision = pipeline.commit_revision;
    let mut second = prepared_patch();
    second.erase = None;
    pipeline.prepared.insert(
        3,
        PendingRegion {
            region: second,
            context_revision: pipeline.commit_revision,
        },
    );
    let mut snapshot = input(
        2,
        1,
        json!([
            {"block_id":99,"bbox":[10,10,70,40],"source":"SOURCE","translation":"한글"},
            {"block_id":3,"bbox":[10,10,70,40],"source":"SOURCE","translation":"번역"}
        ]),
        json!([source_preparation(100)]),
    );
    Arc::get_mut(snapshot.photo.as_mut().unwrap()).unwrap().phase = PhotoPhase::Completed;
    let mut rendered = Vec::new();
    pipeline
        .process(&snapshot, Some(pixels), || true, |output| {
            match output {
                PhotoImageOutput::Block { block_id, region, typeset, .. } => {
                    assert_eq!(region.unwrap().mask_ms, 17.0);
                    rendered.push((block_id, typeset.unwrap()));
                }
                PhotoImageOutput::Timing { .. } | PhotoImageOutput::FontReady { .. } => {}
                _ => panic!("terminal source-only blocks must not initialize masking"),
            }
            true
        })
        .unwrap();
    assert_eq!(rendered.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![99, 3]);
    assert!(pipeline.processor.is_none());
    assert!(pipeline.prepared.is_empty());
    assert!(pipeline.erase_source.is_none());
    assert!(pipeline.changes.is_empty());
    assert!(pipeline.typesetter.as_ref().unwrap().placed_boxes().is_empty());
    snapshot.revision += 1;
    Arc::make_mut(snapshot.photo.as_mut().unwrap()).visible = true;
    pipeline
        .process(&snapshot, None, || true, |_| {
            panic!("retained translated blocks must not be inferred or placed again")
        })
        .unwrap();
    for (_, block) in &rendered {
        assert!(!block.runs.is_empty());
        for run in &block.runs {
            let mut metrics = windows::Win32::Graphics::DirectWrite::DWRITE_TEXT_METRICS::default();
            unsafe { run.layout.GetMetrics(&mut metrics).unwrap() };
            assert!(metrics.width > 0.0);
            let mut size = 0.0;
            unsafe { run.layout.GetFontSize(0, &mut size, None).unwrap() };
            assert_eq!(size, run.size);
        }
    }
}
