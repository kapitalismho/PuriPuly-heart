use super::*;

#[test]
fn windows_graphics_measurement_formats_preserve_configuration_and_reconstruct_after_eviction() {
    let engine = DirectWriteLayoutEngine::new_for_test().unwrap();
    let policy = CaptionLayoutPolicy::default();
    let style = engine.resolve_text_style(&policy, Some("en"), "configuration");
    let first = engine
        .create_text_format_for_resolved_style(&style, 132.0, DWRITE_WORD_WRAPPING_NO_WRAP, None)
        .unwrap();
    let hit = engine
        .create_text_format_for_resolved_style(&style, 132.0, DWRITE_WORD_WRAPPING_NO_WRAP, None)
        .unwrap();
    assert_eq!(first.as_raw(), hit.as_raw());
    let fractional = engine
        .create_text_format_for_resolved_style(&style, 132.001, DWRITE_WORD_WRAPPING_NO_WRAP, None)
        .unwrap();
    assert_ne!(first.as_raw(), fractional.as_raw());
    let wrapped = engine
        .create_text_format_for_resolved_style(
            &style,
            132.0,
            windows::Win32::Graphics::DirectWrite::DWRITE_WORD_WRAPPING_WRAP,
            None,
        )
        .unwrap();
    assert_ne!(first.as_raw(), wrapped.as_raw());
    let japanese = engine.resolve_text_style(&policy, Some("ja"), "configuration");
    let other_locale = engine
        .create_text_format_for_resolved_style(&japanese, 132.0, DWRITE_WORD_WRAPPING_NO_WRAP, None)
        .unwrap();
    assert_ne!(first.as_raw(), other_locale.as_raw());
    let sign = unsafe { engine.factory.CreateEllipsisTrimmingSign(&first).unwrap() };
    let count = engine.measurement_formats.borrow().len();
    let trimmed = engine
        .create_text_format_for_resolved_style(
            &style,
            132.0,
            DWRITE_WORD_WRAPPING_NO_WRAP,
            Some(&sign),
        )
        .unwrap();
    assert_ne!(first.as_raw(), trimmed.as_raw());
    assert_eq!(engine.measurement_formats.borrow().len(), count);
    for size in 40..80 {
        engine
            .create_text_format_for_resolved_style(
                &style,
                size as f32,
                DWRITE_WORD_WRAPPING_NO_WRAP,
                None,
            )
            .unwrap();
        assert!(engine.measurement_formats.borrow().len() <= TEXT_FORMAT_CACHE_CAP);
    }
    let rebuilt = engine
        .create_text_format_for_resolved_style(&style, 132.0, DWRITE_WORD_WRAPPING_NO_WRAP, None)
        .unwrap();
    assert_ne!(first.as_raw(), rebuilt.as_raw());
    assert_eq!(unsafe { rebuilt.GetFontSize() }, 132.0);
    let before = engine
        .measure_centered_line_for_resolved_style(&policy, &style, "ffi AV e\u{301}", 500.0, 132.0)
        .unwrap();
    *engine.measurement_formats.borrow_mut() =
        BoundedLruCache::with_capacity(TEXT_FORMAT_CACHE_CAP);
    let after = engine
        .measure_centered_line_for_resolved_style(&policy, &style, "ffi AV e\u{301}", 500.0, 132.0)
        .unwrap();
    assert_eq!(before.width_px, after.width_px);
    assert_eq!(before.visual_bounds, after.visual_bounds);
    assert_eq!(before.style_key, after.style_key);
}

#[test]
fn windows_graphics_exact_wrapping_reuse_matches_sequential_boundaries_and_stays_bounded() {
    let engine = DirectWriteLayoutEngine::new_for_test().unwrap();
    let oracle = DirectWriteLayoutEngine::new_for_test().unwrap();
    let policy = CaptionLayoutPolicy::default();
    for (language, base) in [
        ("ko", "한글 음절과 한글 경계 확인 "),
        ("ja", "日本語 mixed Latin 中文 の字幕 "),
        (
            "en",
            "ffi AV Cafe\u{301} a\u{308}\u{301} العربية हिन्दी ไทย ",
        ),
        (
            "en",
            "👩\u{200d}💻 ❤️ ☕️ 🇰🇷 👨\u{200d}👩\u{200d}👧\u{200d}👦 ",
        ),
        ("en", "first\n\nthird\nfourth "),
    ] {
        for size in [81.84, 105.6, 118.8, 132.0] {
            for width in [1.0, 132.0, 399.5, 900.0] {
                for revision in 0..4 {
                    let text = match revision {
                        0 => base.repeat(5),
                        1 => format!("{} suffix", base.repeat(5)),
                        2 => format!("CORRECTION {}", base.repeat(4)),
                        _ => base.repeat(2),
                    };
                    let style = engine.resolve_text_style(&policy, Some(language), &text);
                    let oracle_style = oracle.resolve_text_style(&policy, Some(language), &text);
                    oracle.wrapping_boundaries.borrow_mut().clear();
                    let expected = oracle
                        .wrap_primary_text(&policy, &oracle_style, &text, width, size, 3)
                        .unwrap();
                    let actual = engine
                        .wrap_primary_text(&policy, &style, &text, width, size, 3)
                        .unwrap();
                    assert_eq!(actual, expected, "{language} {size} {width} {revision}");
                    let entries = engine.wrapping_boundaries.borrow();
                    assert!(entries.len() <= WRAPPING_BOUNDARY_CAP);
                    assert!(entries
                        .iter()
                        .all(|entry| entry.tested_prefix.len() <= WRAPPING_BOUNDARY_BYTES));
                    assert!(
                        entries
                            .iter()
                            .map(|entry| entry.tested_prefix.capacity())
                            .sum::<usize>()
                            <= WRAPPING_BOUNDARY_CAP * WRAPPING_BOUNDARY_BYTES
                    );
                }
            }
        }
    }
    let style = engine.resolve_text_style(&policy, Some("en"), "a");
    let width = engine
        .measure_centered_line_for_resolved_style(&policy, &style, "a", 100.0, 132.0)
        .unwrap()
        .width_px;
    for threshold in [width - 0.01, width, width + 0.01] {
        oracle.wrapping_boundaries.borrow_mut().clear();
        assert_eq!(
            engine
                .longest_fitting_prefix(&policy, &style, "aaaa", threshold, 132.0)
                .unwrap(),
            oracle
                .longest_fitting_prefix(&policy, &style, "aaaa", threshold, 132.0)
                .unwrap()
        );
    }
}

#[test]
fn windows_graphics_layout_cache_counts_membership_before_mutation_even_on_failure() {
    use crate::renderer::CaptionChannel;

    let engine = DirectWriteLayoutEngine::new_for_test().unwrap();
    let policy = CaptionLayoutPolicy::default();
    let presentation = CaptionPresentation::default();
    let mut cache = LayoutCache::with_capacity(1);
    let neighbor = CaptionBlock::new("self:one", "Stable e\u{301} 🐋")
        .with_channel(CaptionChannel::SelfChannel)
        .with_secondary_text("같은 문장", true)
        .with_slot(1, 544.0);
    let resolve = |blocks: &[CaptionBlock], cache: &mut LayoutCache| {
        let mut diagnostics = RenderDiagnostics::default();
        let result = policy.resolve_blocks_for_presentation_windows_cached(
            blocks,
            4096,
            1056,
            &presentation,
            &engine,
            Some((cache, &mut diagnostics)),
        );
        (
            result,
            (diagnostics.layout_cache_hits, diagnostics.layout_cache_misses),
        )
    };

    let duplicate = CaptionBlock {
        id: "same-content".into(),
        ..neighbor.clone()
    }
    .with_slot(0, 40.0);
    let (result, counts) = resolve(&[duplicate, neighbor.clone()], &mut cache);
    let first = result.unwrap();
    assert_eq!(counts, (0, 2));
    assert_eq!(first.visible_blocks[0].id, "same-content");
    assert_eq!(first.visible_blocks[1].id, "self:one");
    assert_eq!(cache.len(), 1);

    let mut changing = CaptionBlock::new("peer:one", "Changed 日本語 한국어 Latin")
        .with_channel(CaptionChannel::PeerChannel)
        .with_slot(0, 40.0);
    let (result, counts) = resolve(&[changing.clone(), neighbor.clone()], &mut cache);
    assert_eq!(counts, (1, 1));
    assert_eq!(result.unwrap().visible_blocks[1], first.visible_blocks[1]);
    assert_eq!(cache.len(), 1);

    changing.fit_scales = Some(CaptionFitScales {
        primary: -1.0,
        secondary: 1.0,
    });
    let (result, counts) = resolve(&[changing, neighbor.clone()], &mut cache);
    assert!(result.is_err());
    assert_eq!(counts, (1, 1));
    assert_eq!(cache.len(), 1);

    let (result, counts) = resolve(&[neighbor], &mut cache);
    assert_eq!(counts, (1, 0));
    assert_eq!(result.unwrap().visible_blocks[0], first.visible_blocks[1]);
}
