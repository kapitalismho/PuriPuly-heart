use std::path::{Path, PathBuf};

use puripuly_heart_overlay::renderer::LineRole;
#[cfg(windows)]
use puripuly_heart_overlay::WindowsBundledFontCollection;
use puripuly_heart_overlay::{
    bundled_font_path_from_exe_dir, AdapterIdentity, BlockBounds, BundledFaceId, CaptionBlock,
    CaptionBlockVariant, CaptionChannel, CaptionDebugOverlay, CaptionLayoutPolicy,
    CaptionPresentation, CaptionRenderer, DamageBand, FontFallbackReason, FontLanguageBucket,
    FontResolver, FontSource, FontWeight, OverlayPlacementPolicy, OverlayPresentationBlock,
    OverlayPresentationBlockVariant, OverlayPresentationCalibration, OverlayPresentationSnapshot,
    OverlayState, PresentationBackend, ReadinessCancellation, ReadinessOutcome,
};
fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 0.01,
        "expected {expected}, got {actual}"
    );
}

fn test_block(text: &str) -> CaptionBlock {
    CaptionBlock::new("block-1", text)
}

fn bilingual_block(
    id: &str,
    primary_text: &str,
    secondary_text: &str,
    secondary_enabled: bool,
) -> CaptionBlock {
    CaptionBlock::new(id, primary_text).with_secondary_text(secondary_text, secondary_enabled)
}

fn localized_bilingual_block(
    id: &str,
    primary_text: &str,
    primary_language: &str,
    secondary_text: &str,
    secondary_language: &str,
) -> CaptionBlock {
    CaptionBlock::new(id, primary_text)
        .with_primary_language(primary_language)
        .with_secondary_text(secondary_text, true)
        .with_secondary_language(secondary_language)
}

fn long_block(id: &str) -> CaptionBlock {
    let text = "streaming translation captions should keep the newest utterance readable while \
                older blocks are dropped when the fixed overlay surface overflows "
        .repeat(24);
    CaptionBlock::new(id, text)
}

#[test]
fn renderer_default_caption_weight_order_excludes_bold() {
    let weights = CaptionLayoutPolicy::default().preferred_weights();
    assert_eq!(
        weights.first(),
        Some(&"Semibold"),
        "DWrite resolves the first matching weight, so Semibold must keep precedence"
    );
    assert_eq!(weights.len(), 3);
    assert!(
        weights.contains(&"Medium"),
        "Medium must stay resolvable instead of degrading to normal"
    );
    assert!(
        !weights.contains(&"Bold"),
        "Bold would silently degrade to normal through the catch-all weight arm"
    );
}

#[test]
fn renderer_preferred_face_resolution_uses_latin_and_cjk_order_before_system_fallback() {
    let policy = CaptionLayoutPolicy::default();

    assert_eq!(policy.latin_face_chain()[0], "Noto Sans");
    assert_eq!(
        policy.latin_face_chain().last(),
        Some(&"DirectWrite system fallback")
    );
    assert_eq!(policy.cjk_face_chain()[0], "Malgun Gothic");
    assert!(policy.cjk_face_chain().contains(&"Segoe UI"));
    for installed_noto_cjk in [
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
    ] {
        assert!(
            !policy.cjk_face_chain().contains(&installed_noto_cjk),
            "installed {installed_noto_cjk} must not be searched as an intermediate system candidate"
        );
    }
}

#[test]
fn renderer_font_language_bucket_normalizes_cjk_and_general_languages() {
    assert_eq!(
        FontLanguageBucket::for_text(Some("ko"), "hello"),
        FontLanguageBucket::CjkKo
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("ko-KR"), "hello"),
        FontLanguageBucket::CjkKo
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("ja-JP"), "hello"),
        FontLanguageBucket::CjkJa
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("zh-CN"), "hello"),
        FontLanguageBucket::CjkZhHans
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("zh-Hans"), "hello"),
        FontLanguageBucket::CjkZhHans
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("zh-TW"), "hello"),
        FontLanguageBucket::CjkZhHant
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("zh-Hant"), "hello"),
        FontLanguageBucket::CjkZhHant
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("zh-HK"), "hello"),
        FontLanguageBucket::CjkZhHant
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("fr-CA"), "bonjour"),
        FontLanguageBucket::General
    );
}

#[test]
fn renderer_font_language_bucket_uses_heuristic_for_missing_and_unknown_language() {
    assert_eq!(
        FontLanguageBucket::for_text(None, "日本語"),
        FontLanguageBucket::CjkKo,
        "missing-language CJK keeps the compatibility KR-first default"
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("x-madeup"), "中文"),
        FontLanguageBucket::CjkKo,
        "unknown explicit CJK language uses the unknown-CJK compatibility path"
    );
    assert_eq!(
        FontLanguageBucket::for_text(Some("x-madeup"), "hello"),
        FontLanguageBucket::General
    );
    assert_eq!(
        FontLanguageBucket::for_text(None, "hello"),
        FontLanguageBucket::General
    );
}

#[test]
fn renderer_font_language_bucket_prefers_ui_language_hint_for_unknown_cjk_text() {
    assert_eq!(
        FontLanguageBucket::for_text_with_ui_language(None, "日本語", Some("ja-JP")),
        FontLanguageBucket::CjkJa
    );
    assert_eq!(
        FontLanguageBucket::for_text_with_ui_language(Some("x-madeup"), "中文", Some("zh-Hans")),
        FontLanguageBucket::CjkZhHans
    );
    assert_eq!(
        FontLanguageBucket::for_text_with_ui_language(None, "繁體", Some("zh-HK")),
        FontLanguageBucket::CjkZhHant
    );
    assert_eq!(
        FontLanguageBucket::for_text_with_ui_language(None, "안녕", Some("ko-KR")),
        FontLanguageBucket::CjkKo
    );
    assert_eq!(
        FontLanguageBucket::for_text_with_ui_language(None, "中文", Some("fr-FR")),
        FontLanguageBucket::CjkKo,
        "unmapped UI languages keep the KR-first compatibility default"
    );
    assert_eq!(
        FontLanguageBucket::for_text_with_ui_language(None, "hello", Some("ja-JP")),
        FontLanguageBucket::General,
        "UI CJK hint must not force General text onto a CJK branch"
    );
}

#[test]
fn renderer_font_resolver_selects_bundled_faces_and_locales_by_bucket() {
    let resolver = FontResolver::with_bundle_available();
    let cases = [
        (
            "ko-KR",
            "안녕하세요",
            FontLanguageBucket::CjkKo,
            "ko-KR",
            BundledFaceId::NotoCjkKrMedium,
            "Noto Sans CJK KR",
            &["Malgun Gothic", "Segoe UI", "DirectWrite system fallback"] as &[&str],
        ),
        (
            "ja-JP",
            "日本語",
            FontLanguageBucket::CjkJa,
            "ja-JP",
            BundledFaceId::NotoCjkJpMedium,
            "Noto Sans CJK JP",
            &[
                "Yu Gothic UI",
                "Meiryo UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
        ),
        (
            "zh-Hans",
            "中文",
            FontLanguageBucket::CjkZhHans,
            "zh-CN",
            BundledFaceId::NotoCjkScMedium,
            "Noto Sans CJK SC",
            &[
                "Microsoft YaHei UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
        ),
        (
            "zh-HK",
            "繁體",
            FontLanguageBucket::CjkZhHant,
            "zh-TW",
            BundledFaceId::NotoCjkTcMedium,
            "Noto Sans CJK TC",
            &[
                "Microsoft JhengHei UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
        ),
    ];

    for (language, text, bucket, locale, face, family, system_fallbacks) in cases {
        let style = resolver.resolve(Some(language), text);

        assert_eq!(style.bucket, bucket);
        assert_eq!(style.source, FontSource::BundledNotoCjkMedium);
        assert_eq!(style.bundled_face, Some(face));
        assert_eq!(style.family_name, family);
        assert_eq!(style.weight, FontWeight::Medium);
        assert_eq!(style.locale, locale);
        assert_eq!(style.system_fallback_families(), system_fallbacks);
        assert_no_installed_noto_cjk_intermediates(style.system_fallback_families());
    }
}

#[test]
fn renderer_font_resolver_uses_ui_language_hint_for_missing_unknown_cjk_text() {
    let japanese_hint = FontResolver::with_bundle_available().with_ui_language_hint("ja-JP");
    let japanese = japanese_hint.resolve(None, "日本語");
    assert_eq!(japanese.bucket, FontLanguageBucket::CjkJa);
    assert_eq!(japanese.locale, "ja-JP");
    assert_eq!(japanese.bundled_face, Some(BundledFaceId::NotoCjkJpMedium));

    let hong_kong_hint = FontResolver::with_bundle_available().with_ui_language_hint("zh-HK");
    let traditional = hong_kong_hint.resolve(Some("x-madeup"), "繁體");
    assert_eq!(traditional.bucket, FontLanguageBucket::CjkZhHant);
    assert_eq!(traditional.locale, "zh-TW");
    assert_eq!(
        traditional.bundled_face,
        Some(BundledFaceId::NotoCjkTcMedium)
    );

    let unmapped_hint = FontResolver::with_bundle_available().with_ui_language_hint("fr-FR");
    assert_eq!(
        unmapped_hint.resolve(None, "中文").bucket,
        FontLanguageBucket::CjkKo
    );
}

#[test]
fn renderer_font_resolver_uses_system_branches_when_bundle_unavailable() {
    let resolver = FontResolver::with_bundle_unavailable("missing test bundle");
    let cases = [
        (
            "ko",
            "안녕",
            FontLanguageBucket::CjkKo,
            "ko-KR",
            "Malgun Gothic",
            &["Malgun Gothic", "Segoe UI", "DirectWrite system fallback"] as &[&str],
        ),
        (
            "ja",
            "日本語",
            FontLanguageBucket::CjkJa,
            "ja-JP",
            "Yu Gothic UI",
            &[
                "Yu Gothic UI",
                "Meiryo UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
        ),
        (
            "zh-CN",
            "中文",
            FontLanguageBucket::CjkZhHans,
            "zh-CN",
            "Microsoft YaHei UI",
            &[
                "Microsoft YaHei UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
        ),
        (
            "zh-Hant",
            "繁體",
            FontLanguageBucket::CjkZhHant,
            "zh-TW",
            "Microsoft JhengHei UI",
            &[
                "Microsoft JhengHei UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
        ),
    ];

    for (language, text, bucket, locale, family, system_fallbacks) in cases {
        let style = resolver.resolve(Some(language), text);

        assert_eq!(style.bucket, bucket);
        assert_eq!(style.source, FontSource::SystemFont);
        assert_eq!(style.bundled_face, None);
        assert_eq!(style.family_name, family);
        assert_eq!(style.locale, locale);
        assert_eq!(style.system_fallback_families(), system_fallbacks);
        assert_no_installed_noto_cjk_intermediates(style.system_fallback_families());
    }
}

#[test]
fn renderer_general_font_resolver_preserves_explicit_locale_and_defaults_en_us() {
    let resolver = FontResolver::with_bundle_available();
    let explicit = resolver.resolve(Some("fr-ca"), "bonjour");
    let german = resolver.resolve(Some("de-DE"), "guten tag");
    let missing = resolver.resolve(None, "hello");
    let private_use = resolver.resolve(Some("x-madeup"), "hello");
    let undefined = resolver.resolve(Some("und"), "hello");
    let invalid = resolver.resolve(Some("not a tag!"), "hello");

    assert_eq!(explicit.bucket, FontLanguageBucket::General);
    assert_eq!(explicit.source, FontSource::SystemFont);
    assert_eq!(explicit.bundled_face, None);
    assert_eq!(explicit.family_name, "Noto Sans");
    assert_eq!(explicit.locale, "fr-CA");
    assert_eq!(
        explicit.system_fallback_families(),
        &["Noto Sans", "Segoe UI", "DirectWrite system fallback"]
    );

    assert_eq!(german.bucket, FontLanguageBucket::General);
    assert_eq!(german.locale, "de-DE");
    assert_eq!(missing.bucket, FontLanguageBucket::General);
    assert_eq!(missing.locale, "en-US");
    assert_eq!(private_use.bucket, FontLanguageBucket::General);
    assert_eq!(private_use.locale, "en-US");
    assert_eq!(undefined.bucket, FontLanguageBucket::General);
    assert_eq!(undefined.locale, "en-US");
    assert_eq!(invalid.bucket, FontLanguageBucket::General);
    assert_eq!(invalid.locale, "en-US");
}

#[test]
fn renderer_runtime_bundled_font_path_uses_packaged_app_data_fonts_directory() {
    assert_eq!(
        bundled_font_path_from_exe_dir(Path::new("C:/PuriPulyHeart")),
        PathBuf::from("C:/PuriPulyHeart")
            .join("puripuly_heart")
            .join("data")
            .join("fonts")
            .join("NotoSansCJK-Medium.ttc")
    );
}

#[cfg(windows)]
#[test]
fn windows_graphics_loads_committed_ttc_as_bundled_collection() {
    let font_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("src")
        .join("puripuly_heart")
        .join("data")
        .join("fonts")
        .join("NotoSansCJK-Medium.ttc");

    assert!(font_path.exists(), "expected source TTC at {font_path:?}");
    let expected_path = font_path
        .canonicalize()
        .expect("source TTC path should canonicalize");
    let collection = WindowsBundledFontCollection::load_from_path(&font_path)
        .expect("source NotoSansCJK-Medium.ttc should load as a DirectWrite font collection");

    assert_eq!(collection.path(), expected_path.as_path());
}

#[test]
fn renderer_style_resolution_failure_fallback_is_distinct_from_layout_fallback() {
    let style = FontResolver::style_resolution_failure_fallback(FontLanguageBucket::CjkJa);

    assert_eq!(style.source, FontSource::SystemFallbackSentinel);
    assert_eq!(style.family_name, "Segoe UI");
    assert_eq!(style.weight, FontWeight::Regular);
    assert_eq!(style.locale, "ja-JP");
    assert_eq!(
        FontFallbackReason::DirectWriteStyleResolutionFailure.log_label(),
        "directwrite_style_resolution_failure"
    );
    assert!(!FontFallbackReason::DirectWriteStyleResolutionFailure.uses_heuristic_layout_fallback());
    assert!(
        FontFallbackReason::CatastrophicDirectWriteLayoutFailure.uses_heuristic_layout_fallback()
    );
}

#[test]
fn renderer_style_resolution_fallback_for_requested_style_stays_out_of_layout_fallback() {
    let requested = FontResolver::with_bundle_available().resolve(Some("zh-Hant"), "繁體");
    let fallback = FontResolver::style_resolution_failure_fallback_for_style(&requested);

    assert_eq!(fallback.bucket, FontLanguageBucket::CjkZhHant);
    assert_eq!(fallback.source, FontSource::SystemFallbackSentinel);
    assert_eq!(fallback.family_name, "Segoe UI");
    assert_eq!(fallback.weight, FontWeight::Regular);
    assert_eq!(fallback.locale, "zh-TW");
    assert_eq!(
        fallback.fallback_reason,
        Some(FontFallbackReason::DirectWriteStyleResolutionFailure)
    );
    assert!(!fallback
        .fallback_reason
        .expect("style fallback reason")
        .uses_heuristic_layout_fallback());
    assert_eq!(
        FontFallbackReason::CatastrophicDirectWriteLayoutFailure.log_label(),
        "catastrophic_directwrite_layout_failure"
    );
}

fn assert_no_installed_noto_cjk_intermediates(candidates: &[&str]) {
    for installed_noto_cjk in [
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
    ] {
        assert!(
            !candidates.contains(&installed_noto_cjk),
            "installed {installed_noto_cjk} must not be searched as an intermediate system candidate"
        );
    }
}

#[test]
fn renderer_uses_fixed_surface_defaults_for_mvp_caption_layout() {
    let policy = CaptionLayoutPolicy::default();
    assert_eq!(policy.default_surface_size(), (4096, 1056));
}

#[test]
fn renderer_default_typography_resolves_132px_primary_and_81_84px_secondary() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![bilingual_block(
            "peer:translated",
            "translated peer line",
            "source peer line",
            true,
        )],
        3840,
        1024,
    );

    let block = &result.visible_blocks[0];
    let primary = &block.primary_lines[0];
    let secondary = block
        .secondary_line
        .as_ref()
        .expect("secondary source line should be present");

    assert_close(primary.font_size_px, 132.0);
    assert_close(secondary.font_size_px, 81.84);
    assert!(secondary.font_size_px < primary.font_size_px);
}

#[test]
fn renderer_layout_preserves_channel_metadata_for_visible_blocks() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![CaptionBlock::new("peer", "hello").with_channel(CaptionChannel::PeerChannel)],
        3840,
        1024,
    );

    assert_eq!(
        result.visible_blocks[0].channel,
        Some(CaptionChannel::PeerChannel)
    );
}

#[test]
fn renderer_keeps_active_self_and_finalized_variants_distinct() {
    let policy = CaptionLayoutPolicy::default();
    let active = CaptionBlock::new("self:active", "hello")
        .with_variant(CaptionBlockVariant::ActiveSelf)
        .with_secondary_text("", true);
    let finalized = bilingual_block("self:1", "hello", "안녕", true);
    let result = policy.layout_blocks(vec![finalized, active], 1600, 1600);

    let variants = result
        .visible_blocks
        .iter()
        .map(|block| block.block_variant)
        .collect::<Vec<_>>();

    assert!(variants.contains(&CaptionBlockVariant::Finalized));
    assert!(variants.contains(&CaptionBlockVariant::ActiveSelf));
}

#[test]
fn openvr_overlay_policy_defaults_to_head_locked_mode() {
    let policy = OverlayPlacementPolicy::default();
    assert!(policy.is_head_locked());
}

#[test]
fn renderer_preserves_all_supplied_blocks_without_a_renderer_side_visibility_cap() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![
            CaptionBlock::new("old", "short one"),
            CaptionBlock::new("mid", "short two"),
            CaptionBlock::new("new", "short three"),
        ],
        3840,
        4096,
    );

    assert_eq!(
        result
            .visible_blocks
            .iter()
            .map(|block| block.id.as_str())
            .collect::<Vec<_>>(),
        vec!["old", "mid", "new"]
    );
    assert!(result.dropped_block_ids.is_empty());
}

#[test]
fn renderer_overflow_keeps_supplied_blocks_and_truncates_per_block_content_only() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(vec![long_block("old"), long_block("new")], 3840, 1024);

    assert!(result.visible_blocks.iter().any(|block| block.id == "old"));
    assert!(result.visible_blocks.iter().any(|block| block.id == "new"));
    assert!(result.dropped_block_ids.is_empty());
    assert!(result
        .visible_blocks
        .iter()
        .find(|block| block.id == "new")
        .is_some_and(|block| !block.primary_lines.is_empty()));
}

#[test]
fn renderer_wraps_by_measured_width_and_keeps_lines_inside_strip_width() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![CaptionBlock::new(
            "new",
            "measured width wrapping should stay inside the strip bounds for every rendered line",
        )],
        1600,
        900,
    );

    let block = &result.visible_blocks[0];
    assert!(block.primary_lines.len() > 1);
    assert!(block
        .primary_lines
        .iter()
        .all(|line| line.width_px <= block.content_width_px + f32::EPSILON));
}

#[test]
fn renderer_centers_each_line_within_strip_bounds() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(vec![CaptionBlock::new("new", "center me")], 1600, 900);

    let block = &result.visible_blocks[0];
    let line = &block.primary_lines[0];
    let line_center_x = line.origin_x + line.width_px * 0.5;

    assert!((line_center_x - block.bounds.center_x()).abs() < 1.0);
}

#[test]
fn renderer_secondary_origin_includes_30px_gap_after_primary_budget() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![bilingual_block(
            "peer:translated",
            "translated peer line",
            "source peer line",
            true,
        )],
        3840,
        1024,
    );

    let block = &result.visible_blocks[0];
    let secondary = block
        .secondary_line
        .as_ref()
        .expect("secondary source line should be present");

    assert_close(
        secondary.origin_y,
        block.bounds.top_px + 32.0 + 2.0 * 150.0 + 30.0,
    );
}

#[test]
fn renderer_secondary_origin_gap_scales_with_text_scale() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks_for_presentation(
        vec![bilingual_block(
            "peer:translated",
            "translated peer line",
            "source peer line",
            true,
        )],
        3840,
        1536,
        &CaptionPresentation {
            text_scale: 1.5,
            ..CaptionPresentation::default()
        },
    );

    let block = &result.visible_blocks[0];
    let secondary = block
        .secondary_line
        .as_ref()
        .expect("secondary source line should be present");

    assert_close(
        secondary.origin_y,
        block.bounds.top_px + (32.0 + 2.0 * 150.0 + 30.0) * 1.5,
    );
}

#[test]
fn renderer_damage_band_covers_old_and_new_strip_bounds() {
    let damage = DamageBand::from_bounds([
        BlockBounds::new(0.0, 100.0, 1000.0, 260.0),
        BlockBounds::new(0.0, 180.0, 1000.0, 340.0),
    ])
    .unwrap();

    assert_eq!(damage.top_px, 100.0);
    assert_eq!(damage.bottom_px, 340.0);
}

#[test]
fn renderer_uses_slot_top_px_instead_of_stacking_input_order() {
    let policy = CaptionLayoutPolicy::default();
    let layout = policy.layout_blocks(
        vec![
            CaptionBlock::new("peer:2", "two")
                .with_slot(1, 420.0)
                .with_channel(CaptionChannel::PeerChannel),
            CaptionBlock::new("self:1", "one")
                .with_slot(0, 40.0)
                .with_channel(CaptionChannel::SelfChannel),
        ],
        3840,
        1024,
    );

    assert_eq!(layout.visible_blocks[0].bounds.top_px, 420.0);
    assert_eq!(layout.visible_blocks[1].bounds.top_px, 40.0);
}

#[test]
fn renderer_active_peer_with_state_generated_slots_does_not_overlap_next_row() {
    let mut state = OverlayState::default();
    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![
            OverlayPresentationBlock {
                id: "peer:active".into(),
                occupant_key: "peer:turn-1".into(),
                appearance_seq: 1,
                channel: "peer".into(),
                block_variant: OverlayPresentationBlockVariant::ActivePeer,
                primary_text: String::new(),
                secondary_text: "Can you hear me?".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: None,
                origin_wall_clock_ms: None,
                session_scope: None,
                ..Default::default()
            },
            OverlayPresentationBlock {
                id: "self:final".into(),
                occupant_key: "self:final".into(),
                appearance_seq: 2,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: "hello".into(),
                secondary_text: "안녕".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: None,
                origin_wall_clock_ms: None,
                session_scope: None,
                ..Default::default()
            },
        ],
        ..Default::default()
    }));

    let caption_blocks = state
        .scene()
        .slots()
        .iter()
        .flatten()
        .map(|slot| {
            let channel = if slot.channel == "peer" {
                CaptionChannel::PeerChannel
            } else {
                CaptionChannel::SelfChannel
            };
            let variant = match slot.block_variant {
                OverlayPresentationBlockVariant::ActiveSelf => CaptionBlockVariant::ActiveSelf,
                OverlayPresentationBlockVariant::ActivePeer => CaptionBlockVariant::ActivePeer,
                OverlayPresentationBlockVariant::Finalized => CaptionBlockVariant::Finalized,
            };
            CaptionBlock::new(slot.id.clone(), slot.primary_text.clone())
                .with_variant(variant)
                .with_channel(channel)
                .with_secondary_text(slot.secondary_text.clone(), slot.secondary_enabled)
                .with_slot(slot.slot_index, slot.anchor_top_px)
        })
        .collect::<Vec<_>>();

    let result = CaptionLayoutPolicy::default().layout_blocks(caption_blocks, 3840, 1024);

    let peer = result
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:active")
        .unwrap();
    let self_block = result
        .visible_blocks
        .iter()
        .find(|block| block.id == "self:final")
        .unwrap();

    assert_eq!(peer.block_variant, CaptionBlockVariant::ActivePeer);
    assert_eq!(peer.channel, Some(CaptionChannel::PeerChannel));
    assert!(peer.primary_lines.iter().all(|line| line.text.is_empty()));
    assert_eq!(
        peer.secondary_line.as_ref().map(|line| line.text.as_str()),
        Some("Can you hear me?")
    );
    assert!(peer.secondary_reserved);
    assert!(
        peer.bounds.bottom_px <= self_block.bounds.top_px,
        "peer bounds {:?} should not overlap next row {:?}",
        peer.bounds,
        self_block.bounds
    );
}

#[test]
fn renderer_source_only_peer_finalized_with_state_generated_slots_does_not_overlap_next_row() {
    let mut state = OverlayState::default();
    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![
            OverlayPresentationBlock {
                id: "peer:source-only".into(),
                occupant_key: "peer:turn-2".into(),
                appearance_seq: 1,
                channel: "peer".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: String::new(),
                secondary_text: "translation unavailable, showing original source text".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: None,
                origin_wall_clock_ms: None,
                session_scope: None,
                ..Default::default()
            },
            OverlayPresentationBlock {
                id: "self:final".into(),
                occupant_key: "self:final".into(),
                appearance_seq: 2,
                channel: "self".into(),
                block_variant: OverlayPresentationBlockVariant::Finalized,
                primary_text: "hello".into(),
                secondary_text: "안녕".into(),
                secondary_enabled: true,
                primary_language: None,
                secondary_language: None,
                update_id: None,
                origin_wall_clock_ms: None,
                session_scope: None,
                ..Default::default()
            },
        ],
        ..Default::default()
    }));

    let caption_blocks = state
        .scene()
        .slots()
        .iter()
        .flatten()
        .map(|slot| {
            let channel = if slot.channel == "peer" {
                CaptionChannel::PeerChannel
            } else {
                CaptionChannel::SelfChannel
            };
            let variant = match slot.block_variant {
                OverlayPresentationBlockVariant::ActiveSelf => CaptionBlockVariant::ActiveSelf,
                OverlayPresentationBlockVariant::ActivePeer => CaptionBlockVariant::ActivePeer,
                OverlayPresentationBlockVariant::Finalized => CaptionBlockVariant::Finalized,
            };
            CaptionBlock::new(slot.id.clone(), slot.primary_text.clone())
                .with_variant(variant)
                .with_channel(channel)
                .with_secondary_text(slot.secondary_text.clone(), slot.secondary_enabled)
                .with_slot(slot.slot_index, slot.anchor_top_px)
        })
        .collect::<Vec<_>>();

    let result = CaptionLayoutPolicy::default().layout_blocks(caption_blocks, 3840, 1024);

    let peer = result
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:source-only")
        .unwrap();
    let self_block = result
        .visible_blocks
        .iter()
        .find(|block| block.id == "self:final")
        .unwrap();

    assert_eq!(peer.block_variant, CaptionBlockVariant::Finalized);
    assert_eq!(peer.channel, Some(CaptionChannel::PeerChannel));
    assert!(peer.primary_lines.iter().all(|line| line.text.is_empty()));
    assert_eq!(
        peer.secondary_line.as_ref().map(|line| line.text.as_str()),
        Some("translation unavailable, showing original source text")
    );
    assert!(peer.secondary_reserved);
    assert!(
        peer.bounds.bottom_px <= self_block.bounds.top_px,
        "peer source-only bounds {:?} should not overlap next row {:?}",
        peer.bounds,
        self_block.bounds
    );
}

#[test]
fn renderer_secondary_reserved_block_height_remains_468px_at_default_scale() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![bilingual_block(
            "peer:translated",
            "translated peer line",
            "source peer line",
            true,
        )],
        3840,
        1024,
    );

    let block = &result.visible_blocks[0];
    assert_close(block.bounds.bottom_px - block.bounds.top_px, 468.0);
}

#[test]
fn renderer_two_secondary_reserved_slots_fit_default_surface_after_typography_spacing_update() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![
            bilingual_block(
                "peer:translated",
                "translated peer text with enough words to represent a normal two-line HMD row",
                "source original 안녕하세요 hello こんにちは 你好",
                true,
            )
            .with_variant(CaptionBlockVariant::ActivePeer)
            .with_channel(CaptionChannel::PeerChannel)
            .with_slot(0, 40.0),
            bilingual_block(
                "self:translated",
                "self transcript text",
                "self translated secondary",
                true,
            )
            .with_variant(CaptionBlockVariant::Finalized)
            .with_channel(CaptionChannel::SelfChannel)
            .with_slot(1, 544.0),
        ],
        4096,
        1056,
    );

    assert_eq!(result.visible_blocks.len(), 2);
    for block in &result.visible_blocks {
        assert!(
            block.bounds.top_px >= 0.0 && block.bounds.bottom_px <= 1056.0,
            "block bounds {:?} should stay inside default surface",
            block.bounds
        );
        assert!(
            block.visual_bounds.top_px >= 0.0 && block.visual_bounds.bottom_px <= 1056.0,
            "visual bounds {:?} should stay inside default surface",
            block.visual_bounds
        );
    }
}

#[test]
fn renderer_source_only_peer_remains_secondary_only_with_readable_default_size() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.resolve_blocks_for_presentation(
        vec![CaptionBlock::new("peer:source-only", "")
            .with_variant(CaptionBlockVariant::Finalized)
            .with_channel(CaptionChannel::PeerChannel)
            .with_secondary_text("source-only peer fallback remains readable", true)
            .with_slot(0, 40.0)],
        3840,
        1024,
        &CaptionPresentation::default(),
    );

    let block = &result.visible_blocks[0];
    let secondary = block
        .secondary_line
        .as_ref()
        .expect("source-only peer should render as secondary text");

    assert!(block.primary_lines.iter().all(|line| line.text.is_empty()));
    assert_eq!(secondary.role, LineRole::Secondary);
    assert_eq!(secondary.text, "source-only peer fallback remains readable");
    assert_close(secondary.font_size_px, 81.84);
    assert!(secondary.origin_y > block.bounds.top_px + 32.0 + 2.0 * 150.0);
}

#[test]
fn renderer_layout_uses_primary_and_secondary_language_style_keys() {
    let policy = CaptionLayoutPolicy::default();
    let primary_text = "日本語";
    let secondary_text = "繁體";
    let result = policy.resolve_blocks_for_presentation(
        vec![localized_bilingual_block(
            "localized",
            primary_text,
            "ja",
            secondary_text,
            "zh-Hant",
        )],
        3840,
        1024,
        &CaptionPresentation::default(),
    );

    let block = &result.visible_blocks[0];
    let primary = block
        .primary_lines
        .first()
        .expect("primary line should be present");
    let secondary = block
        .secondary_line
        .as_ref()
        .expect("secondary line should be present");
    let resolver = FontResolver::default();
    let expected_primary = resolver.resolve(Some("ja"), primary_text).style_key();
    let expected_secondary = resolver
        .resolve(Some("zh-Hant"), secondary_text)
        .style_key();

    assert_eq!(block.layout_cache_key.primary_style_key, expected_primary);
    assert_eq!(primary.style_key, expected_primary);
    assert_eq!(
        block.layout_cache_key.secondary_style_key,
        expected_secondary
    );
    assert_eq!(secondary.style_key, expected_secondary);
}

#[test]
fn renderer_same_text_with_different_languages_produces_different_style_keys_without_flush() {
    let policy = CaptionLayoutPolicy::default();
    let korean = CaptionBlock::new("same-text", "漢字").with_primary_language("ko");
    let japanese = CaptionBlock::new("same-text", "漢字").with_primary_language("ja");

    let korean_layout = policy.resolve_blocks_for_presentation(
        vec![korean],
        3840,
        1024,
        &CaptionPresentation::default(),
    );
    let japanese_layout = policy.resolve_blocks_for_presentation(
        vec![japanese],
        3840,
        1024,
        &CaptionPresentation::default(),
    );
    let korean_block = &korean_layout.visible_blocks[0];
    let japanese_block = &japanese_layout.visible_blocks[0];

    assert_ne!(
        korean_block.layout_cache_key.primary_style_key,
        japanese_block.layout_cache_key.primary_style_key
    );
    assert_ne!(
        korean_block.block_cache_key(),
        japanese_block.block_cache_key()
    );
    assert_eq!(
        korean_block.layout_cache_key.primary_text, japanese_block.layout_cache_key.primary_text,
        "only style metadata should separate these cache entries"
    );
}

#[test]
fn renderer_render_path_expands_damage_band_to_rendered_bounds_with_safety_margin() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let first = renderer
        .render_blocks(vec![CaptionBlock::new("self", "hello")])
        .unwrap();
    let previous_visual_bounds = first.layout().visible_blocks[0].visual_bounds;

    let second = renderer.render_empty_frame().unwrap();
    let damage = second
        .layout()
        .damage_band
        .expect("damage band should be present");

    assert_eq!(
        damage.top_px,
        (previous_visual_bounds.top_px - 32.0).max(0.0)
    );
    assert_eq!(
        damage.bottom_px,
        (previous_visual_bounds.bottom_px + 32.0).min(first.layout().surface_height_px as f32)
    );
}

#[test]
fn renderer_damage_band_includes_same_peer_slot_when_primary_text_arrives() {
    let renderer = CaptionRenderer::new_for_test().unwrap();

    let _ = renderer
        .render_blocks(vec![CaptionBlock::new("peer:turn-1", "")
            .with_variant(CaptionBlockVariant::ActivePeer)
            .with_channel(CaptionChannel::PeerChannel)
            .with_secondary_text("peer source", true)
            .with_slot(0, 40.0)])
        .unwrap();

    let frame = renderer
        .render_blocks(vec![
            CaptionBlock::new("peer:turn-1", "translated peer body")
                .with_variant(CaptionBlockVariant::ActivePeer)
                .with_channel(CaptionChannel::PeerChannel)
                .with_secondary_text("peer source", true)
                .with_slot(0, 40.0),
            CaptionBlock::new("self:turn-2", "self transcript")
                .with_variant(CaptionBlockVariant::Finalized)
                .with_channel(CaptionChannel::SelfChannel)
                .with_secondary_text("self translation", true)
                .with_slot(1, 544.0),
        ])
        .unwrap();

    let damage_band = frame
        .layout()
        .damage_band
        .expect("second frame should have a damage band");
    let peer_block = frame
        .layout()
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:turn-1")
        .expect("peer block should remain visible");

    assert!(
        peer_block.visual_bounds.bottom_px >= damage_band.top_px
            && peer_block.visual_bounds.top_px <= damage_band.bottom_px,
        "damage band [{}, {}] should include changed peer slot [{}, {}] when primary text arrives in-place",
        damage_band.top_px,
        damage_band.bottom_px,
        peer_block.visual_bounds.top_px,
        peer_block.visual_bounds.bottom_px
    );
}

#[test]
fn renderer_damage_band_includes_same_slot_secondary_text_movement() {
    let renderer = CaptionRenderer::new_for_test().unwrap();

    let first_frame = renderer
        .render_blocks(vec![CaptionBlock::new(
            "peer:turn-1",
            "translated peer body",
        )
        .with_variant(CaptionBlockVariant::ActivePeer)
        .with_channel(CaptionChannel::PeerChannel)
        .with_secondary_text("short source", true)
        .with_slot(0, 40.0)])
        .unwrap();
    let first_block = first_frame
        .layout()
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:turn-1")
        .expect("peer block should be visible in first frame");
    let old_visual_bounds = first_block.visual_bounds;
    let old_secondary = first_block
        .secondary_line
        .as_ref()
        .expect("first frame should include secondary text")
        .clone();

    let frame = renderer
        .render_blocks(vec![CaptionBlock::new(
            "peer:turn-1",
            "translated peer body",
        )
        .with_variant(CaptionBlockVariant::ActivePeer)
        .with_channel(CaptionChannel::PeerChannel)
        .with_secondary_text(
            "longer source text moves the secondary visual bounds and must stay inside damage",
            true,
        )
        .with_slot(0, 40.0)])
        .unwrap();

    let damage_band = frame
        .layout()
        .damage_band
        .expect("second frame should have a damage band");
    let block = frame
        .layout()
        .visible_blocks
        .iter()
        .find(|block| block.id == "peer:turn-1")
        .expect("peer block should remain visible");
    let new_visual_bounds = block.visual_bounds;
    let new_secondary = block
        .secondary_line
        .as_ref()
        .expect("second frame should include secondary text")
        .clone();

    assert_ne!(old_secondary.text, new_secondary.text);

    let union_top_px = old_visual_bounds.top_px.min(new_visual_bounds.top_px);
    let union_bottom_px = old_visual_bounds.bottom_px.max(new_visual_bounds.bottom_px);

    assert!(
        damage_band.top_px <= union_top_px && damage_band.bottom_px >= union_bottom_px,
        "damage band [{}, {}] should contain old/new same-slot secondary visual union [{}, {}] from old [{}, {}] and new [{}, {}]",
        damage_band.top_px,
        damage_band.bottom_px,
        union_top_px,
        union_bottom_px,
        old_visual_bounds.top_px,
        old_visual_bounds.bottom_px,
        new_visual_bounds.top_px,
        new_visual_bounds.bottom_px
    );
}

#[test]
fn renderer_secondary_line_is_single_line_with_ellipsis() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![bilingual_block(
            "peer",
            "primary",
            "this secondary line should truncate before it wraps into another row",
            true,
        )],
        1100,
        900,
    );

    let block = &result.visible_blocks[0];
    let secondary = block.secondary_line.as_ref().unwrap();

    assert!(secondary.text.ends_with("..."));
    assert!(secondary.width_px <= block.content_width_px + f32::EPSILON);
}

#[test]
fn renderer_primary_lines_do_not_append_ellipsis_when_over_budget() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![bilingual_block(
            "peer",
            "this primary translation line is intentionally long enough to exceed the measured two-line budget and should clip without appending an ellipsis marker to the rendered text",
            "secondary",
            true,
        )],
        1100,
        900,
    );

    let block = &result.visible_blocks[0];
    let last_primary = block.primary_lines.last().unwrap();

    assert!(block.truncated_primary);
    assert!(!last_primary.text.ends_with("..."));
}

#[test]
fn renderer_long_translated_peer_remains_two_primary_lines_plus_single_secondary_line() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(
        vec![CaptionBlock::new(
            "peer:long-translated",
            "this translated peer utterance is deliberately long enough to exceed the fixed two-line primary budget but must not create extra renderer blocks or pages",
        )
        .with_variant(CaptionBlockVariant::ActivePeer)
        .with_channel(CaptionChannel::PeerChannel)
        .with_secondary_text(
            "source text also stays one ellipsized secondary row instead of becoming another block",
            true,
        )],
        1100,
        900,
    );

    assert_eq!(result.visible_blocks.len(), 1);
    assert!(result.dropped_block_ids.is_empty());
    let block = &result.visible_blocks[0];
    assert_eq!(block.primary_lines.len(), 2);
    assert!(block.truncated_primary);
    assert!(block.secondary_line.is_some());
    assert!(block.truncated_secondary);
}

#[test]
fn renderer_reserves_secondary_row_height_even_before_secondary_text_arrives() {
    let policy = CaptionLayoutPolicy::default();
    let empty_secondary = policy.layout_blocks(
        vec![bilingual_block("self", "hello there", "", true)],
        1600,
        900,
    );
    let filled_secondary = policy.layout_blocks(
        vec![bilingual_block(
            "self",
            "hello there",
            "secondary line",
            true,
        )],
        1600,
        900,
    );

    let empty_block = &empty_secondary.visible_blocks[0];
    let filled_block = &filled_secondary.visible_blocks[0];

    assert!(empty_block.secondary_reserved);
    assert!(filled_block.secondary_reserved);
    assert_eq!(
        empty_block.bounds.bottom_px - empty_block.bounds.top_px,
        filled_block.bounds.bottom_px - filled_block.bounds.top_px
    );
}

#[test]
fn renderer_turning_secondary_off_expands_primary_line_budget() {
    let policy = CaptionLayoutPolicy::default();
    let text = "primary text should gain an extra line when the secondary slot is disabled and the width budget stays the same";
    let with_secondary = policy.layout_blocks(
        vec![bilingual_block("self", text, "secondary", true)],
        1100,
        900,
    );
    let without_secondary =
        policy.layout_blocks(vec![bilingual_block("self", text, "", false)], 1100, 900);

    assert_eq!(with_secondary.visible_blocks[0].primary_lines.len(), 2);
    assert_eq!(without_secondary.visible_blocks[0].primary_lines.len(), 3);
}

#[test]
fn renderer_height_scale_changes_rendered_block_bounds_height() {
    let policy = CaptionLayoutPolicy::default();
    let full = policy.layout_blocks(vec![CaptionBlock::new("self", "hello")], 1600, 900);
    let exiting = policy.layout_blocks(
        vec![CaptionBlock::new("self", "hello").with_visual_state(1.0, 0.0, 0.5)],
        1600,
        900,
    );

    let full_height =
        full.visible_blocks[0].bounds.bottom_px - full.visible_blocks[0].bounds.top_px;
    let exiting_height =
        exiting.visible_blocks[0].bounds.bottom_px - exiting.visible_blocks[0].bounds.top_px;

    assert!(exiting_height < full_height);
}

#[test]
fn renderer_layout_cache_key_ignores_animation_only_visual_state() {
    let policy = CaptionLayoutPolicy::default();
    let presentation = CaptionPresentation::default();
    let stable = bilingual_block("self:1", "hello there", "secondary line", true)
        .with_channel(CaptionChannel::SelfChannel)
        .with_variant(CaptionBlockVariant::Finalized);
    let animated = stable.clone().with_visual_state(0.42, 64.0, 0.5);

    let stable_layout =
        policy.resolve_blocks_for_presentation(vec![stable], 1600, 900, &presentation);
    let animated_layout =
        policy.resolve_blocks_for_presentation(vec![animated], 1600, 900, &presentation);

    assert_eq!(
        stable_layout.visible_blocks[0].layout_cache_key,
        animated_layout.visible_blocks[0].layout_cache_key
    );
}

#[test]
fn renderer_layout_cache_key_separates_slotted_source_only_peer_geometry() {
    let policy = CaptionLayoutPolicy::default();
    let presentation = CaptionPresentation::default();
    let source_only_peer = CaptionBlock::new("peer:source-only", "showing original source only")
        .with_channel(CaptionChannel::PeerChannel)
        .with_variant(CaptionBlockVariant::Finalized)
        .with_secondary_text("", false);
    let slotted_source_only_peer = source_only_peer.clone().with_slot(0, 40.0);

    let non_slotted_layout =
        policy.resolve_blocks_for_presentation(vec![source_only_peer], 3840, 1024, &presentation);
    let slotted_layout = policy.resolve_blocks_for_presentation(
        vec![slotted_source_only_peer],
        3840,
        1024,
        &presentation,
    );

    let non_slotted = &non_slotted_layout.visible_blocks[0];
    let slotted = &slotted_layout.visible_blocks[0];

    assert!(!non_slotted.secondary_reserved);
    assert!(slotted.secondary_reserved);
    assert_ne!(
        non_slotted.layout_cache_key, slotted.layout_cache_key,
        "slotted and non-slotted peer source-only rows use different reserved geometry"
    );
}

#[test]
fn renderer_block_cache_key_ignores_animation_only_visual_state() {
    let policy = CaptionLayoutPolicy::default();
    let presentation = CaptionPresentation::default();
    let stable = bilingual_block("self:1", "hello there", "secondary line", true)
        .with_channel(CaptionChannel::SelfChannel)
        .with_variant(CaptionBlockVariant::Finalized);
    let animated = stable.clone().with_visual_state(0.35, -48.0, 1.35);

    let stable_layout =
        policy.resolve_blocks_for_presentation(vec![stable], 1600, 900, &presentation);
    let animated_layout =
        policy.resolve_blocks_for_presentation(vec![animated], 1600, 900, &presentation);

    assert_eq!(
        stable_layout.visible_blocks[0].block_cache_key(),
        animated_layout.visible_blocks[0].block_cache_key()
    );
}

#[test]
fn renderer_applies_offset_and_height_scale_to_transformed_bounds() {
    let policy = CaptionLayoutPolicy::default();
    let presentation = CaptionPresentation::default();
    let stable = policy.resolve_blocks_for_presentation(
        vec![CaptionBlock::new("self", "hello there").with_secondary_text("secondary", true)],
        1600,
        900,
        &presentation,
    );
    let animated = policy.resolve_blocks_for_presentation(
        vec![CaptionBlock::new("self", "hello there")
            .with_secondary_text("secondary", true)
            .with_visual_state(0.4, 48.0, 0.5)],
        1600,
        900,
        &presentation,
    );

    let stable_block = &stable.visible_blocks[0];
    let animated_block = &animated.visible_blocks[0];
    let stable_height = stable_block.bounds.bottom_px - stable_block.bounds.top_px;
    let animated_height = animated_block.bounds.bottom_px - animated_block.bounds.top_px;
    let stable_line = &stable_block.primary_lines[0];
    let animated_line = &animated_block.primary_lines[0];

    assert_eq!(animated_block.render_offset_y_px, 48.0);
    assert_eq!(animated_block.render_height_scale, 0.5);
    assert_close(
        animated_block.bounds.top_px,
        stable_block.bounds.top_px + animated_block.render_offset_y_px,
    );
    assert_close(
        animated_height,
        stable_height * animated_block.render_height_scale,
    );
    assert_close(
        animated_line.origin_y,
        animated_block.bounds.top_px
            + (stable_line.origin_y - stable_block.bounds.top_px)
                * animated_block.render_height_scale,
    );
    assert_close(
        animated_block.visual_bounds.top_px,
        animated_block.bounds.top_px
            + (stable_block.visual_bounds.top_px - stable_block.bounds.top_px)
                * animated_block.render_height_scale,
    );
    assert_close(
        animated_block.visual_bounds.bottom_px,
        animated_block.bounds.top_px
            + (stable_block.visual_bounds.bottom_px - stable_block.bounds.top_px)
                * animated_block.render_height_scale,
    );
}

#[test]
fn renderer_uses_near_full_safe_width_for_content() {
    let policy = CaptionLayoutPolicy::default();
    let result = policy.layout_blocks(vec![CaptionBlock::new("self", "hello")], 1600, 900);

    assert!(result.visible_blocks[0].content_width_px > 1600.0 * 0.88);
}

#[test]
fn renderer_first_usable_frame_is_fully_transparent_before_real_caption_content() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let frame = renderer.render_empty_frame().unwrap();

    assert!(frame.is_fully_transparent());
}

#[test]
fn renderer_debug_overlay_is_absent_by_default_and_reported_when_supplied() {
    let renderer = CaptionRenderer::new_for_test().unwrap();

    let normal = renderer.render_blocks(vec![test_block("hello")]).unwrap();
    assert_eq!(normal.debug_overlay_label(), None);

    let empty = renderer
        .render_blocks_with_debug_overlay(
            Vec::new(),
            Some(CaptionDebugOverlay::new("DBG should stay hidden").unwrap()),
        )
        .unwrap();
    assert!(empty.is_fully_transparent());
    assert_eq!(empty.debug_overlay_label(), None);

    let hidden_secondary = renderer
        .render_blocks_with_debug_overlay(
            vec![CaptionBlock::new("peer:hidden", "")
                .with_channel(CaptionChannel::PeerChannel)
                .with_variant(CaptionBlockVariant::ActivePeer)
                .with_secondary_text("hidden source", false)],
            Some(CaptionDebugOverlay::new("DBG should also stay hidden").unwrap()),
        )
        .unwrap();
    assert!(hidden_secondary.is_fully_transparent());
    assert_eq!(hidden_secondary.debug_overlay_label(), None);

    let debug = renderer
        .render_blocks_with_debug_overlay(
            vec![test_block("hello")],
            Some(CaptionDebugOverlay::new("DBG r7 ap=peer h=1a2b b=peer").unwrap()),
        )
        .unwrap();

    assert_eq!(
        debug.debug_overlay_label(),
        Some("DBG r7 ap=peer h=1a2b b=peer")
    );
}

#[test]
fn renderer_presentation_text_scale_changes_block_bounds_height() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let default_frame = renderer.render_blocks(vec![test_block("hello")]).unwrap();
    let default_height = default_frame.layout().visible_blocks[0].bounds.bottom_px
        - default_frame.layout().visible_blocks[0].bounds.top_px;

    renderer.set_presentation(CaptionPresentation {
        text_scale: 1.25,
        ..CaptionPresentation::default()
    });
    let scaled_frame = renderer.render_blocks(vec![test_block("hello")]).unwrap();
    let scaled_height = scaled_frame.layout().visible_blocks[0].bounds.bottom_px
        - scaled_frame.layout().visible_blocks[0].bounds.top_px;

    assert!(scaled_height > default_height);
}

#[cfg(windows)]
#[test]
fn windows_graphics_returns_a_renderable_d3d11_texture_result() {
    let renderer = CaptionRenderer::new().unwrap();
    let frame = renderer.render_blocks(vec![test_block("hello")]).unwrap();

    assert!(frame.texture_ptr().is_some());
    assert!(frame.d3d11_texture().is_some());
    assert_eq!(frame.width(), 4096);
    assert_eq!(frame.height(), 1056);
}

#[cfg(windows)]
#[test]
fn renderer_windows_public_layout_api_uses_fallback_measurement_for_mixed_script_text() {
    let policy = CaptionLayoutPolicy::default();
    let text = "fallback hello 안녕하세요 你好 mixed text";
    let result = policy.layout_blocks(vec![CaptionBlock::new("mix", text)], 1200, 900);
    let block = &result.visible_blocks[0];
    let combined = block
        .primary_lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(!block.primary_lines.is_empty());
    assert!(combined.contains("hello"));
    assert!(combined.contains("안녕하세요") || combined.contains("你好"));
    assert!(block
        .primary_lines
        .iter()
        .all(|line| line.width_px <= block.content_width_px + 1.0));
}

#[cfg(windows)]
#[test]
fn windows_graphics_startup_warmup_reports_attempts_without_populating_visual_caches() {
    let renderer = CaptionRenderer::new().unwrap();
    let frame = renderer.render_empty_frame().unwrap();
    let diagnostics = frame.diagnostics();

    assert!(frame.is_fully_transparent());
    assert_eq!(diagnostics.font_warmup_attempts, 8);
    assert!(diagnostics.font_warmup_failures <= diagnostics.font_warmup_attempts);
    assert_eq!(diagnostics.text_format_cache_size, 0);
    assert_eq!(diagnostics.line_cache_size, 0);
    assert_eq!(diagnostics.block_cache_size, 0);
}

#[cfg(windows)]
#[test]
fn windows_graphics_pipeline_reports_directwrite_layout_for_mixed_script_frame() {
    let renderer = CaptionRenderer::new().unwrap();
    let frame = renderer
        .render_blocks(vec![CaptionBlock::new(
            "mix",
            "fallback hello 안녕하세요 你好 mixed text",
        )])
        .unwrap();

    assert!(!frame.is_fully_transparent());
    assert!(frame.texture_ptr().is_some());
    assert!(frame.d3d11_texture().is_some());
    assert!(!frame.layout().visible_blocks[0].primary_lines.is_empty());
    assert_eq!(frame.diagnostics().directwrite_layout_success_count, 1);
    assert_eq!(frame.diagnostics().heuristic_layout_fallback_count, 0);
}

#[cfg(windows)]
#[test]
fn windows_graphics_first_active_self_frame_after_empty_frame_is_renderable() {
    let renderer = CaptionRenderer::new().unwrap();
    let empty = renderer.render_empty_frame().unwrap();
    let active = CaptionBlock::new("self:active", "live self preview")
        .with_variant(CaptionBlockVariant::ActiveSelf)
        .with_secondary_text("", true)
        .with_channel(CaptionChannel::SelfChannel);
    let frame = renderer.render_blocks(vec![active]).unwrap();

    assert!(empty.is_fully_transparent());
    assert!(!frame.is_fully_transparent());
    assert_eq!(
        frame.layout().visible_blocks[0].block_variant,
        CaptionBlockVariant::ActiveSelf
    );
    assert!(frame.texture_ptr().is_some());
    assert!(frame.d3d11_texture().is_some());
}

#[cfg(windows)]
#[test]
fn windows_graphics_first_finalized_bilingual_frame_after_empty_frame_is_renderable() {
    let renderer = CaptionRenderer::new().unwrap();
    let empty = renderer.render_empty_frame().unwrap();
    let finalized = bilingual_block("self:1", "hello there", "secondary line", true)
        .with_channel(CaptionChannel::SelfChannel);
    let frame = renderer.render_blocks(vec![finalized]).unwrap();

    assert!(empty.is_fully_transparent());
    assert!(!frame.is_fully_transparent());
    assert_eq!(
        frame.layout().visible_blocks[0].block_variant,
        CaptionBlockVariant::Finalized
    );
    assert!(frame.texture_ptr().is_some());
    assert!(frame.d3d11_texture().is_some());
}

#[cfg(windows)]
#[test]
fn windows_graphics_secondary_only_finalized_peer_frame_is_renderable() {
    let renderer = CaptionRenderer::new().unwrap();
    let source_only_peer = CaptionBlock::new("peer:source-only", "")
        .with_channel(CaptionChannel::PeerChannel)
        .with_variant(CaptionBlockVariant::Finalized)
        .with_secondary_text("Can you hear me?", true);

    let frame = renderer.render_blocks(vec![source_only_peer]).unwrap();
    let block = &frame.layout().visible_blocks[0];

    assert!(!frame.is_fully_transparent());
    assert_eq!(block.block_variant, CaptionBlockVariant::Finalized);
    assert!(block.primary_lines.iter().all(|line| line.text.is_empty()));
    assert_eq!(
        block.secondary_line.as_ref().map(|line| line.text.as_str()),
        Some("Can you hear me?")
    );
    assert!(frame.texture_ptr().is_some());
    assert!(frame.d3d11_texture().is_some());
}

#[cfg(windows)]
#[test]
fn windows_graphics_debug_overlay_frame_is_renderable() {
    let renderer = CaptionRenderer::new().unwrap();
    let frame = renderer
        .render_blocks_with_debug_overlay(
            vec![test_block("hello")],
            Some(CaptionDebugOverlay::new("DBG r7 ap=peer h=1a2b b=peer").unwrap()),
        )
        .unwrap();

    assert!(!frame.is_fully_transparent());
    assert_eq!(
        frame.debug_overlay_label(),
        Some("DBG r7 ap=peer h=1a2b b=peer")
    );
    assert!(frame.texture_ptr().is_some());
    assert!(frame.d3d11_texture().is_some());
    assert_eq!(frame.diagnostics().debug_overlay_draw_count, 1);
}

#[cfg(windows)]
#[test]
fn windows_graphics_clears_debug_overlay_band_when_overlay_is_removed() {
    let renderer = CaptionRenderer::new().unwrap();
    let block = test_block("hello");

    let first = renderer
        .render_blocks_with_debug_overlay(
            vec![block.clone()],
            Some(CaptionDebugOverlay::new("DBG r7 ap=peer h=1a2b b=peer").unwrap()),
        )
        .unwrap();
    assert_eq!(first.diagnostics().debug_overlay_draw_count, 1);

    let second = renderer.render_blocks(vec![block]).unwrap();

    assert_eq!(second.debug_overlay_label(), None);
    assert_eq!(second.diagnostics().debug_overlay_draw_count, 0);
    assert_eq!(second.diagnostics().debug_overlay_clear_count, 1);
}

#[cfg(windows)]
#[test]
fn windows_graphics_second_render_hits_layout_and_block_caches() {
    let renderer = CaptionRenderer::new().unwrap();
    let block = bilingual_block("self:1", "hello there", "secondary line", true)
        .with_channel(CaptionChannel::SelfChannel);

    let first = renderer.render_blocks(vec![block.clone()]).unwrap();
    let second = renderer.render_blocks(vec![block]).unwrap();

    assert!(first.diagnostics().text_format_cache_misses >= 1);
    assert!(first.diagnostics().layout_cache_misses >= 1);
    assert!(first.diagnostics().block_cache_misses >= 1);
    assert!(second.diagnostics().layout_cache_hits >= 1);
    assert!(second.diagnostics().block_cache_hits >= 1);
    assert!(second.diagnostics().text_format_cache_size <= 32);
    assert!(second.diagnostics().layout_cache_size <= 512);
    assert!(second.diagnostics().line_cache_size <= 2048);
    assert!(second.diagnostics().block_cache_size <= 1024);
}

#[cfg(windows)]
#[test]
fn windows_graphics_text_format_cache_reports_hits_for_same_bucket_new_line_visual() {
    let renderer = CaptionRenderer::new().unwrap();
    let first = CaptionBlock::new("self:active", "live preview one")
        .with_variant(CaptionBlockVariant::ActiveSelf)
        .with_channel(CaptionChannel::SelfChannel);
    let second = CaptionBlock::new("self:active", "live preview two")
        .with_variant(CaptionBlockVariant::ActiveSelf)
        .with_channel(CaptionChannel::SelfChannel);

    let first_frame = renderer.render_blocks(vec![first]).unwrap();
    let second_frame = renderer.render_blocks(vec![second]).unwrap();

    assert!(first_frame.diagnostics().text_format_cache_misses >= 1);
    assert!(second_frame.diagnostics().line_cache_misses >= 1);
    assert!(second_frame.diagnostics().text_format_cache_hits >= 1);
}

#[cfg(windows)]
#[test]
fn windows_graphics_draws_general_text_with_explicit_non_default_locale() {
    let renderer = CaptionRenderer::new().unwrap();
    let frame = renderer
        .render_blocks(vec![
            CaptionBlock::new("fr", "bonjour tout le monde").with_primary_language("fr-CA")
        ])
        .unwrap();

    assert!(!frame.is_fully_transparent());
    assert_eq!(frame.diagnostics().heuristic_layout_fallback_count, 0);
}

#[cfg(windows)]
#[test]
fn windows_graphics_reuses_finalized_block_cache_across_animation_states() {
    let renderer = CaptionRenderer::new().unwrap();
    let stable = bilingual_block("self:1", "hello there", "secondary line", true)
        .with_channel(CaptionChannel::SelfChannel);

    renderer.render_blocks(vec![stable.clone()]).unwrap();
    let entering = renderer
        .render_blocks(vec![stable.clone().with_visual_state(0.4, 64.0, 0.7)])
        .unwrap();
    let reflow = renderer
        .render_blocks(vec![stable.with_visual_state(1.0, -42.0, 1.2)])
        .unwrap();

    assert!(entering.diagnostics().block_cache_hits >= 1);
    assert!(reflow.diagnostics().block_cache_hits >= 1);
}

#[cfg(windows)]
#[test]
fn windows_graphics_secondary_translation_update_reuses_primary_line_cache_only() {
    let renderer = CaptionRenderer::new().unwrap();
    let primary_text =
        "this primary text should wrap into multiple lines so cached line visuals can be reused";
    let first = bilingual_block("self:1", primary_text, "secondary one", true)
        .with_channel(CaptionChannel::SelfChannel);
    let second = bilingual_block("self:1", primary_text, "secondary two", true)
        .with_channel(CaptionChannel::SelfChannel);

    renderer.render_blocks(vec![first]).unwrap();
    let frame = renderer.render_blocks(vec![second]).unwrap();

    assert!(frame.diagnostics().layout_cache_misses >= 1);
    assert!(frame.diagnostics().line_cache_hits >= 1);
    assert!(frame.diagnostics().block_cache_misses >= 1);
}

#[cfg(not(windows))]
#[test]
fn renderer_returns_a_renderable_texture_contract_off_windows() {
    let renderer = CaptionRenderer::new_for_test().unwrap();
    let frame = renderer.render_blocks(vec![test_block("hello")]).unwrap();

    assert!(frame.texture_ptr().is_some());
    assert_eq!(frame.width(), 4096);
    assert_eq!(frame.height(), 1056);
}

#[cfg(not(windows))]
#[test]
fn renderer_runtime_backend_is_rejected_outside_windows() {
    let result = CaptionRenderer::new();
    assert!(result.is_err());
    let error = result.err().unwrap();

    assert!(error
        .to_string()
        .contains("Direct3D11 caption renderer is only available on Windows"));
}

#[cfg(windows)]
#[tokio::test]
async fn windows_graphics_real_query_cancelled_after_enqueue_is_retained_until_late_completion() {
    let renderer = CaptionRenderer::new().unwrap();
    assert!(renderer.presentation_backend() != PresentationBackend::Test);

    let enqueued = std::sync::Arc::new(tokio::sync::Notify::new());
    let release = std::sync::Arc::new(tokio::sync::Notify::new());
    renderer.set_windows_readiness_enqueue_barrier_for_test(enqueued.clone(), release.clone());
    let cancellation = ReadinessCancellation::default();
    let cancel_after_enqueue = async {
        enqueued.notified().await;
        cancellation.cancel();
        release.notify_one();
    };
    let (cancelled_outcome, ()) = tokio::join!(
        renderer.prepare_frame_for_submission(&cancellation),
        cancel_after_enqueue
    );

    assert_eq!(cancelled_outcome, ReadinessOutcome::Cancelled);
    assert!(renderer.has_incomplete_producer());
    assert_eq!(
        renderer
            .prepare_frame_for_submission(&ReadinessCancellation::default())
            .await,
        ReadinessOutcome::Ready
    );
    assert!(!renderer.has_incomplete_producer());
}

#[tokio::test]
async fn renderer_test_backend_reports_test_identity_with_immediate_readiness() {
    let renderer = CaptionRenderer::new_for_test().unwrap();

    assert_eq!(renderer.presentation_backend(), PresentationBackend::Test);
    assert_eq!(renderer.adapter_identity(), AdapterIdentity::Test);

    let frame = renderer.render_blocks(vec![test_block("hello")]).unwrap();
    assert!(frame.texture_ptr().is_some());
    #[cfg(windows)]
    assert!(frame.d3d11_texture().is_none());

    let live_cancellation = ReadinessCancellation::default();
    assert_eq!(
        renderer
            .prepare_frame_for_submission(&live_cancellation)
            .await,
        ReadinessOutcome::Ready
    );

    let cancelled = ReadinessCancellation::default();
    cancelled.cancel();
    assert_eq!(
        renderer.prepare_frame_for_submission(&cancelled).await,
        ReadinessOutcome::Cancelled
    );
}
