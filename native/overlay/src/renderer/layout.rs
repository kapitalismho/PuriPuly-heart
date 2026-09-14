#[cfg(windows)]
use super::cache::LayoutCache;
use super::cache::{CachedBlockLayoutTemplate, CachedLineLayoutTemplate};
#[cfg(windows)]
use super::font_resolver::{
    FontLanguageBucket, FontSource, FontWeight, ResolvedFontStyle, WindowsBundledFontCollection,
};
use super::font_resolver::{FontResolver, TextStyleKey};
use super::types::{
    BlockBounds, CaptionBlock, CaptionBlockVariant, CaptionLayoutResult, CaptionPresentation,
    LayoutCacheKey, LineRole, ResolvedBlockLayout, ResolvedFrameLayout, ResolvedLineLayout,
    TextStyleDescriptor, VisualBounds, DEFAULT_AVERAGE_GLYPH_ADVANCE_PX, DEFAULT_BLOCK_SPACING_PX,
    DEFAULT_FONT_SIZE_PX, DEFAULT_HORIZONTAL_PADDING_PX, DEFAULT_PRIMARY_LINE_HEIGHT_PX,
    DEFAULT_SECONDARY_LINE_HEIGHT_PX, DEFAULT_STRIP_HORIZONTAL_PADDING_PX,
    DEFAULT_STRIP_VERTICAL_PADDING_PX, DEFAULT_SURFACE_HEIGHT_PX, DEFAULT_SURFACE_WIDTH_PX,
    DEFAULT_VERTICAL_PADDING_PX, PRIMARY_SECONDARY_GAP_PX, SECONDARY_FONT_SCALE,
    TEXT_OUTLINE_OVERHANG_PX,
};
#[cfg(windows)]
use windows::core::PCWSTR;
#[cfg(all(windows, test))]
use windows::Win32::Graphics::DirectWrite::{
    DWriteCreateFactory, IDWriteFactory2, DWRITE_FACTORY_TYPE_SHARED,
};
#[cfg(windows)]
use windows::Win32::Graphics::DirectWrite::{
    IDWriteFactory, IDWriteFontCollection, IDWriteFontFallback, IDWriteFontFamily,
    IDWriteInlineObject, IDWriteTextFormat, IDWriteTextFormat1, IDWriteTextLayout,
    IDWriteTextLayout2, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT,
    DWRITE_FONT_WEIGHT_MEDIUM, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_WEIGHT_SEMI_BOLD,
    DWRITE_TEXT_ALIGNMENT_CENTER, DWRITE_TEXT_METRICS, DWRITE_TRIMMING,
    DWRITE_TRIMMING_GRANULARITY_CHARACTER, DWRITE_WORD_WRAPPING_NO_WRAP,
};
#[cfg(windows)]
use windows_core::Interface;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionLayoutPolicy {
    preferred_weights: [&'static str; 3],
    latin_face_chain: [&'static str; 3],
    cjk_face_chain: [&'static str; 6],
    channel_uses_color_only: bool,
    show_speaker_labels_by_default: bool,
    visible_window_target_blocks: usize,
    horizontal_padding_px: u32,
    vertical_padding_px: u32,
    primary_line_height_px: u32,
    secondary_line_height_px: u32,
    block_spacing_px: u32,
    strip_horizontal_padding_px: u32,
    strip_vertical_padding_px: u32,
    average_glyph_advance_px: u32,
}

impl Default for CaptionLayoutPolicy {
    fn default() -> Self {
        Self {
            preferred_weights: ["Semibold", "Medium", "Regular"],
            latin_face_chain: ["Noto Sans", "Segoe UI", "DirectWrite system fallback"],
            cjk_face_chain: [
                "Malgun Gothic",
                "Yu Gothic UI",
                "Microsoft YaHei UI",
                "Microsoft JhengHei UI",
                "Segoe UI",
                "DirectWrite system fallback",
            ],
            channel_uses_color_only: true,
            show_speaker_labels_by_default: false,
            visible_window_target_blocks: 2,
            horizontal_padding_px: DEFAULT_HORIZONTAL_PADDING_PX,
            vertical_padding_px: DEFAULT_VERTICAL_PADDING_PX,
            primary_line_height_px: DEFAULT_PRIMARY_LINE_HEIGHT_PX,
            secondary_line_height_px: DEFAULT_SECONDARY_LINE_HEIGHT_PX,
            block_spacing_px: DEFAULT_BLOCK_SPACING_PX,
            strip_horizontal_padding_px: DEFAULT_STRIP_HORIZONTAL_PADDING_PX,
            strip_vertical_padding_px: DEFAULT_STRIP_VERTICAL_PADDING_PX,
            average_glyph_advance_px: DEFAULT_AVERAGE_GLYPH_ADVANCE_PX,
        }
    }
}

impl CaptionLayoutPolicy {
    pub fn preferred_weights(&self) -> Vec<&'static str> {
        self.preferred_weights.to_vec()
    }

    pub fn latin_face_chain(&self) -> &[&'static str] {
        &self.latin_face_chain
    }

    pub fn cjk_face_chain(&self) -> &[&'static str] {
        &self.cjk_face_chain
    }

    pub fn visible_window_target_blocks(&self) -> usize {
        self.visible_window_target_blocks
    }

    pub fn channel_uses_color_only(&self) -> bool {
        self.channel_uses_color_only
    }

    pub fn show_speaker_labels_by_default(&self) -> bool {
        self.show_speaker_labels_by_default
    }

    pub fn default_surface_size(&self) -> (u32, u32) {
        (DEFAULT_SURFACE_WIDTH_PX, DEFAULT_SURFACE_HEIGHT_PX)
    }

    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) fn strip_horizontal_padding_px(&self) -> u32 {
        self.strip_horizontal_padding_px
    }

    pub fn layout_blocks(
        &self,
        blocks: Vec<CaptionBlock>,
        surface_width_px: u32,
        surface_height_px: u32,
    ) -> CaptionLayoutResult {
        self.layout_blocks_for_presentation(
            blocks,
            surface_width_px,
            surface_height_px,
            &CaptionPresentation::default(),
        )
    }

    pub fn layout_blocks_for_presentation(
        &self,
        blocks: Vec<CaptionBlock>,
        surface_width_px: u32,
        surface_height_px: u32,
        presentation: &CaptionPresentation,
    ) -> CaptionLayoutResult {
        self.resolve_blocks_for_presentation(
            blocks,
            surface_width_px,
            surface_height_px,
            presentation,
        )
        .into()
    }

    #[allow(dead_code)]
    pub(crate) fn layout_cache_key_for_block(
        &self,
        block: &CaptionBlock,
        surface_width_px: u32,
        presentation: &CaptionPresentation,
    ) -> LayoutCacheKey {
        let resolver = FontResolver::default();
        let primary_style = style_descriptor_for_text(
            &resolver,
            block.primary_language.as_deref(),
            &block.primary_text,
        );
        let secondary_style = style_descriptor_for_text(
            &resolver,
            block.secondary_language.as_deref(),
            &block.secondary_text,
        );
        layout_cache_key_for_block(
            block,
            self.content_width_px(surface_width_px),
            presentation.text_scale.max(0.1),
            primary_style.style_key,
            secondary_style.style_key,
        )
    }

    #[cfg(windows)]
    pub(crate) fn layout_cache_key_for_block_windows(
        &self,
        block: &CaptionBlock,
        surface_width_px: u32,
        presentation: &CaptionPresentation,
        engine: &DirectWriteLayoutEngine,
    ) -> LayoutCacheKey {
        let primary_style_key =
            engine.line_style_key(self, block.primary_language.as_deref(), &block.primary_text);
        let secondary_style_key = engine.line_style_key(
            self,
            block.secondary_language.as_deref(),
            &block.secondary_text,
        );
        layout_cache_key_for_block(
            block,
            self.content_width_px(surface_width_px),
            presentation.text_scale.max(0.1),
            primary_style_key,
            secondary_style_key,
        )
    }

    pub fn resolve_blocks_for_presentation(
        &self,
        blocks: Vec<CaptionBlock>,
        surface_width_px: u32,
        surface_height_px: u32,
        presentation: &CaptionPresentation,
    ) -> ResolvedFrameLayout {
        self.resolve_blocks_for_presentation_fallback(
            blocks,
            surface_width_px,
            surface_height_px,
            presentation,
        )
    }

    fn resolve_blocks_for_presentation_fallback(
        &self,
        blocks: Vec<CaptionBlock>,
        surface_width_px: u32,
        surface_height_px: u32,
        presentation: &CaptionPresentation,
    ) -> ResolvedFrameLayout {
        let content_width_px = self.content_width_px(surface_width_px);
        let text_scale = presentation.text_scale.max(0.1);
        let strip_left_px = self.horizontal_padding_px as f32;
        let mut top_px = self.vertical_padding_px as f32;
        let mut resolved_blocks = Vec::with_capacity(blocks.len());
        let resolver = FontResolver::default();

        for block in blocks {
            let primary_style = style_descriptor_for_text(
                &resolver,
                block.primary_language.as_deref(),
                &block.primary_text,
            );
            let secondary_style = style_descriptor_for_text(
                &resolver,
                block.secondary_language.as_deref(),
                &block.secondary_text,
            );
            let layout_cache_key = layout_cache_key_for_block(
                &block,
                content_width_px,
                text_scale,
                primary_style.style_key,
                secondary_style.style_key,
            );
            let template = self.build_fallback_block_template(
                &block,
                content_width_px,
                text_scale,
                primary_style,
                secondary_style,
            );
            let stable_block_height_px = template.bounds.bottom_px - template.bounds.top_px;
            let block_top_px = if block.slot_assigned {
                block.slot_top_px
            } else {
                top_px
            };
            resolved_blocks.push(materialize_resolved_block_layout(
                &block,
                layout_cache_key,
                &template,
                strip_left_px,
                block_top_px,
            ));
            if !block.slot_assigned {
                top_px += stable_block_height_px + self.block_spacing_px as f32;
            }
        }

        ResolvedFrameLayout {
            visible_blocks: resolved_blocks,
            dropped_block_ids: Vec::new(),
            surface_width_px,
            surface_height_px,
            damage_band: None,
        }
    }

    pub fn measured_block_height_px(
        &self,
        secondary_enabled: bool,
        text_scale: f32,
        height_scale: f32,
    ) -> f32 {
        self.stable_block_height_px(secondary_enabled, text_scale) * height_scale
    }

    pub(crate) fn stable_block_height_px(&self, secondary_enabled: bool, text_scale: f32) -> f32 {
        let primary_lines = if secondary_enabled { 2 } else { 3 };
        let secondary_lines: u32 = if secondary_enabled { 1 } else { 0 };
        let base_height_px = self.strip_vertical_padding_px.saturating_mul(2)
            + primary_lines * self.primary_line_height_px
            + secondary_lines.saturating_mul(self.secondary_line_height_px);
        base_height_px as f32 * text_scale.max(0.1)
    }

    fn reserves_secondary_row(&self, block: &CaptionBlock) -> bool {
        block_reserves_secondary_row(block)
    }

    fn content_width_px(&self, surface_width_px: u32) -> f32 {
        surface_width_px
            .saturating_sub(self.horizontal_padding_px.saturating_mul(2))
            .saturating_sub(self.strip_horizontal_padding_px.saturating_mul(2))
            .max(self.average_glyph_advance_px) as f32
    }

    fn primary_line_budget(&self, block: &CaptionBlock) -> usize {
        if self.reserves_secondary_row(block) {
            2
        } else {
            3
        }
    }

    fn build_fallback_block_template(
        &self,
        block: &CaptionBlock,
        content_width_px: f32,
        text_scale: f32,
        primary_style: TextStyleDescriptor,
        secondary_style: TextStyleDescriptor,
    ) -> CachedBlockLayoutTemplate {
        let content_width_px = content_width_px.max(1.0);
        let secondary_row_reserved = self.reserves_secondary_row(block);
        let primary_budget = self.primary_line_budget(block);
        let primary_font_size_px = DEFAULT_FONT_SIZE_PX * text_scale;
        let secondary_font_size_px = primary_font_size_px * SECONDARY_FONT_SCALE;
        let primary_line_height_px = self.primary_line_height_px as f32 * text_scale;
        let vertical_padding_px = self.strip_vertical_padding_px as f32 * text_scale;
        let primary_secondary_gap_px = if secondary_row_reserved {
            PRIMARY_SECONDARY_GAP_PX * text_scale
        } else {
            0.0
        };
        let strip_width_px = content_width_px + self.strip_horizontal_padding_px as f32 * 2.0;
        let block_height_px = self.stable_block_height_px(secondary_row_reserved, text_scale);
        let local_bounds = BlockBounds::new(0.0, 0.0, strip_width_px, block_height_px);
        let primary_advance_px = self.average_glyph_advance_px as f32 * text_scale;
        let wrapped_primary = wrap_text(&block.primary_text, content_width_px, primary_advance_px);
        let truncated_primary = wrapped_primary.len() > primary_budget;
        let primary_lines = wrapped_primary
            .into_iter()
            .take(primary_budget)
            .enumerate()
            .map(|(index, text)| {
                let width_px = measure_text_width(&text, primary_advance_px);
                let origin_x = self.strip_horizontal_padding_px as f32
                    + ((content_width_px - width_px).max(0.0) * 0.5);
                let origin_y = vertical_padding_px + index as f32 * primary_line_height_px;
                CachedLineLayoutTemplate {
                    visual_bounds: line_visual_bounds(0.0, 0.0, width_px, primary_font_size_px)
                        .translate(origin_x, origin_y),
                    text,
                    role: LineRole::Primary,
                    style_key: primary_style.style_key,
                    style: primary_style.clone(),
                    width_px,
                    origin_x,
                    origin_y,
                    font_size_px: primary_font_size_px,
                }
            })
            .collect::<Vec<_>>();
        let (secondary_text, truncated_secondary) = if block.secondary_enabled {
            ellipsize_text(
                &block.secondary_text,
                content_width_px,
                primary_advance_px * SECONDARY_FONT_SCALE,
            )
        } else {
            (None, false)
        };
        let secondary_line = secondary_text.map(|text| {
            let width_px = measure_text_width(&text, primary_advance_px * SECONDARY_FONT_SCALE);
            let origin_x = self.strip_horizontal_padding_px as f32
                + ((content_width_px - width_px).max(0.0) * 0.5);
            let origin_y = vertical_padding_px
                + primary_budget as f32 * primary_line_height_px
                + primary_secondary_gap_px;
            CachedLineLayoutTemplate {
                visual_bounds: line_visual_bounds(0.0, 0.0, width_px, secondary_font_size_px)
                    .translate(origin_x, origin_y),
                text,
                role: LineRole::Secondary,
                style_key: secondary_style.style_key,
                style: secondary_style,
                width_px,
                origin_x,
                origin_y,
                font_size_px: secondary_font_size_px,
            }
        });
        let local_visual_bounds = block_visual_bounds_from_templates(
            local_bounds,
            &primary_lines,
            secondary_line.as_ref(),
        );

        CachedBlockLayoutTemplate {
            primary_lines,
            secondary_line,
            secondary_reserved: secondary_row_reserved,
            bounds: local_bounds,
            visual_bounds: local_visual_bounds,
            content_width_px,
            truncated_primary,
            truncated_secondary,
        }
    }

    #[cfg(windows)]
    fn build_windows_block_template(
        &self,
        engine: &DirectWriteLayoutEngine,
        block: &CaptionBlock,
        content_width_px: f32,
        text_scale: f32,
        primary_style: &DirectWriteResolvedTextStyle,
        secondary_style: &DirectWriteResolvedTextStyle,
    ) -> Result<CachedBlockLayoutTemplate, windows::core::Error> {
        let secondary_row_reserved = self.reserves_secondary_row(block);
        let primary_budget = self.primary_line_budget(block);
        let primary_font_size_px = DEFAULT_FONT_SIZE_PX * text_scale;
        let secondary_font_size_px = primary_font_size_px * SECONDARY_FONT_SCALE;
        let primary_line_height_px = self.primary_line_height_px as f32 * text_scale;
        let vertical_padding_px = self.strip_vertical_padding_px as f32 * text_scale;
        let primary_secondary_gap_px = if secondary_row_reserved {
            PRIMARY_SECONDARY_GAP_PX * text_scale
        } else {
            0.0
        };
        let block_height_px = self.stable_block_height_px(secondary_row_reserved, text_scale);
        let local_bounds = BlockBounds::new(
            0.0,
            0.0,
            content_width_px + self.strip_horizontal_padding_px as f32 * 2.0,
            block_height_px,
        );

        let (primary_lines_text, truncated_primary) = engine.wrap_primary_text(
            self,
            primary_style,
            &block.primary_text,
            content_width_px,
            primary_font_size_px,
            primary_budget,
        )?;
        let primary_lines = primary_lines_text
            .iter()
            .enumerate()
            .map(|(index, text)| {
                let measured = engine.measure_centered_line_for_resolved_style(
                    self,
                    primary_style,
                    text,
                    content_width_px,
                    primary_font_size_px,
                )?;
                let origin_x = self.strip_horizontal_padding_px as f32 + measured.origin_x_px;
                let origin_y = vertical_padding_px + index as f32 * primary_line_height_px;
                Ok::<CachedLineLayoutTemplate, windows::core::Error>(CachedLineLayoutTemplate {
                    text: text.clone(),
                    role: LineRole::Primary,
                    style_key: measured.style_key,
                    style: measured.style,
                    width_px: measured.width_px,
                    origin_x,
                    origin_y,
                    font_size_px: primary_font_size_px,
                    visual_bounds: measured.visual_bounds.translate(origin_x, origin_y),
                })
            })
            .collect::<Result<Vec<_>, windows::core::Error>>()?;

        let (secondary_line, truncated_secondary) = if block.secondary_enabled {
            let (text, truncated) = engine.ellipsize_secondary_text(
                self,
                secondary_style,
                &block.secondary_text,
                content_width_px,
                secondary_font_size_px,
            )?;
            let line = text
                .as_ref()
                .map(|text| {
                    let measured = engine.measure_centered_line_for_resolved_style(
                        self,
                        secondary_style,
                        text,
                        content_width_px,
                        secondary_font_size_px,
                    )?;
                    let origin_x = self.strip_horizontal_padding_px as f32 + measured.origin_x_px;
                    let origin_y = vertical_padding_px
                        + primary_budget as f32 * primary_line_height_px
                        + primary_secondary_gap_px;
                    Ok::<CachedLineLayoutTemplate, windows::core::Error>(CachedLineLayoutTemplate {
                        text: text.clone(),
                        role: LineRole::Secondary,
                        style_key: measured.style_key,
                        style: measured.style,
                        width_px: measured.width_px,
                        origin_x,
                        origin_y,
                        font_size_px: secondary_font_size_px,
                        visual_bounds: measured.visual_bounds.translate(origin_x, origin_y),
                    })
                })
                .transpose()?;
            (line, truncated)
        } else {
            (None, false)
        };
        let visual_bounds = block_visual_bounds_from_templates(
            local_bounds,
            &primary_lines,
            secondary_line.as_ref(),
        );

        Ok(CachedBlockLayoutTemplate {
            primary_lines,
            secondary_line,
            secondary_reserved: secondary_row_reserved,
            bounds: local_bounds,
            visual_bounds,
            content_width_px,
            truncated_primary,
            truncated_secondary,
        })
    }

    #[cfg(windows)]
    pub(crate) fn resolve_blocks_for_presentation_windows_cached(
        &self,
        blocks: Vec<CaptionBlock>,
        surface_width_px: u32,
        surface_height_px: u32,
        presentation: &CaptionPresentation,
        engine: &DirectWriteLayoutEngine,
        mut layout_cache: Option<&mut LayoutCache>,
    ) -> Result<ResolvedFrameLayout, windows::core::Error> {
        let content_width_px = self.content_width_px(surface_width_px);
        let text_scale = presentation.text_scale.max(0.1);
        let strip_left_px = self.horizontal_padding_px as f32;
        let mut top_px = self.vertical_padding_px as f32;
        let mut resolved_blocks = Vec::with_capacity(blocks.len());

        for block in blocks {
            let primary_style = engine.resolve_text_style(
                self,
                block.primary_language.as_deref(),
                &block.primary_text,
            );
            let secondary_style = engine.resolve_text_style(
                self,
                block.secondary_language.as_deref(),
                &block.secondary_text,
            );
            let primary_style_key = primary_style.style_key;
            let secondary_style_key = secondary_style.style_key;
            let layout_cache_key = layout_cache_key_for_block(
                &block,
                content_width_px,
                text_scale,
                primary_style_key,
                secondary_style_key,
            );
            let template = if let Some(cache) = layout_cache.as_deref_mut() {
                if let Some(cached) = cache.get(&layout_cache_key) {
                    cached.clone()
                } else {
                    let template = self.build_windows_block_template(
                        engine,
                        &block,
                        content_width_px,
                        text_scale,
                        &primary_style,
                        &secondary_style,
                    )?;
                    cache.insert(layout_cache_key.clone(), template.clone());
                    template
                }
            } else {
                self.build_windows_block_template(
                    engine,
                    &block,
                    content_width_px,
                    text_scale,
                    &primary_style,
                    &secondary_style,
                )?
            };
            let stable_block_height_px = template.bounds.bottom_px - template.bounds.top_px;
            let block_top_px = if block.slot_assigned {
                block.slot_top_px
            } else {
                top_px
            };
            resolved_blocks.push(materialize_resolved_block_layout(
                &block,
                layout_cache_key,
                &template,
                strip_left_px,
                block_top_px,
            ));
            if !block.slot_assigned {
                top_px += stable_block_height_px + self.block_spacing_px as f32;
            }
        }

        Ok(ResolvedFrameLayout {
            visible_blocks: resolved_blocks,
            dropped_block_ids: Vec::new(),
            surface_width_px,
            surface_height_px,
            damage_band: None,
        })
    }
}

#[cfg(windows)]
#[derive(Debug, Clone)]
struct MeasuredLine {
    style_key: TextStyleKey,
    style: TextStyleDescriptor,
    width_px: f32,
    origin_x_px: f32,
    visual_bounds: VisualBounds,
}

#[cfg(windows)]
#[derive(Debug, Clone)]
struct DirectWriteResolvedTextStyle {
    family_name: String,
    weight: DWRITE_FONT_WEIGHT,
    locale: String,
    source: FontSource,
    bucket: FontLanguageBucket,
    style_key: TextStyleKey,
    is_style_failure_fallback: bool,
}

#[cfg(windows)]
impl DirectWriteResolvedTextStyle {
    fn from_style(style: ResolvedFontStyle, weight: DWRITE_FONT_WEIGHT) -> Self {
        let style_key = style.style_key();
        Self {
            family_name: style.family_name.to_string(),
            weight,
            locale: style.locale,
            source: style.source,
            bucket: style.bucket,
            style_key,
            is_style_failure_fallback: style.fallback_reason
                == Some(
                    super::font_resolver::FontFallbackReason::DirectWriteStyleResolutionFailure,
                ),
        }
    }

    fn descriptor(&self) -> TextStyleDescriptor {
        TextStyleDescriptor::from_parts(
            self.family_name.clone(),
            font_weight_from_dwrite_weight(self.weight),
            self.locale.clone(),
            self.source,
            self.bucket,
            self.style_key,
        )
    }
}

#[cfg(windows)]
pub(crate) struct DirectWriteLayoutEngine {
    factory: IDWriteFactory,
    system_font_collection: IDWriteFontCollection,
    system_font_fallback: IDWriteFontFallback,
    font_resolver: FontResolver,
    bundled_font_collection: Option<WindowsBundledFontCollection>,
}

#[cfg(windows)]
impl DirectWriteLayoutEngine {
    pub(crate) fn from_shared_resources(
        factory: &IDWriteFactory,
        system_font_collection: &IDWriteFontCollection,
        system_font_fallback: &IDWriteFontFallback,
        font_resolver: FontResolver,
        bundled_font_collection: Option<WindowsBundledFontCollection>,
    ) -> Self {
        Self {
            factory: factory.clone(),
            system_font_collection: system_font_collection.clone(),
            system_font_fallback: system_font_fallback.clone(),
            font_resolver,
            bundled_font_collection,
        }
    }

    fn line_style_key(
        &self,
        policy: &CaptionLayoutPolicy,
        language: Option<&str>,
        text: &str,
    ) -> TextStyleKey {
        self.resolve_text_style(policy, language, text).style_key
    }

    #[cfg(test)]
    fn resolved_text_style_key_for_test(
        &self,
        policy: &CaptionLayoutPolicy,
        text: &str,
    ) -> TextStyleKey {
        self.resolve_text_style(policy, None, text).style_key
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Result<Self, windows::core::Error> {
        let factory: IDWriteFactory = unsafe { DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)? };
        let factory2: IDWriteFactory2 = factory.cast()?;
        let mut collection = None;
        unsafe {
            factory.GetSystemFontCollection(&mut collection, false)?;
        }
        Ok(Self {
            factory,
            system_font_collection: collection.expect("system font collection"),
            system_font_fallback: unsafe { factory2.GetSystemFontFallback()? },
            font_resolver: FontResolver::with_bundle_unavailable(
                "test DirectWrite layout engine uses system fallback resources",
            ),
            bundled_font_collection: None,
        })
    }

    fn wrap_primary_text(
        &self,
        policy: &CaptionLayoutPolicy,
        style: &DirectWriteResolvedTextStyle,
        text: &str,
        max_width_px: f32,
        font_size_px: f32,
        budget: usize,
    ) -> Result<(Vec<String>, bool), windows::core::Error> {
        let mut lines = Vec::new();
        let mut remaining = text.trim();
        if remaining.is_empty() {
            return Ok((vec![String::new()], false));
        }

        while !remaining.is_empty() && lines.len() < budget {
            let line =
                self.longest_fitting_prefix(policy, style, remaining, max_width_px, font_size_px)?;
            if line.is_empty() {
                break;
            }
            let trimmed_line = line.trim().to_string();
            lines.push(trimmed_line.clone());
            remaining = remaining[line.len()..].trim_start();
        }

        // Primary rows clip once the reserved line budget is exhausted.
        Ok((lines, !remaining.is_empty()))
    }

    fn ellipsize_secondary_text(
        &self,
        policy: &CaptionLayoutPolicy,
        style: &DirectWriteResolvedTextStyle,
        text: &str,
        max_width_px: f32,
        font_size_px: f32,
    ) -> Result<(Option<String>, bool), windows::core::Error> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok((None, false));
        }
        if self
            .measure_centered_line_for_resolved_style(
                policy,
                style,
                trimmed,
                max_width_px,
                font_size_px,
            )?
            .width_px
            <= max_width_px
        {
            return Ok((Some(trimmed.to_string()), false));
        }
        Ok((
            Some(self.ellipsize_text(policy, style, trimmed, max_width_px, font_size_px)?),
            true,
        ))
    }

    fn ellipsize_text(
        &self,
        policy: &CaptionLayoutPolicy,
        style: &DirectWriteResolvedTextStyle,
        text: &str,
        max_width_px: f32,
        font_size_px: f32,
    ) -> Result<String, windows::core::Error> {
        let trimmed = text.trim();
        let ellipsis = "...";
        let ellipsis_width = self
            .measure_centered_line_for_resolved_style(
                policy,
                style,
                ellipsis,
                max_width_px,
                font_size_px,
            )?
            .width_px;
        if ellipsis_width >= max_width_px {
            return Ok(ellipsis.to_string());
        }

        let mut best = String::new();
        let chars = trimmed.char_indices().collect::<Vec<_>>();
        let mut low = 0usize;
        let mut high = chars.len();
        while low <= high {
            let mid = (low + high) / 2;
            let candidate = match chars.get(mid) {
                Some((index, _)) => format!("{}{}", &trimmed[..*index], ellipsis),
                None => format!("{trimmed}{ellipsis}"),
            };
            let fits = self
                .measure_centered_line_for_resolved_style(
                    policy,
                    style,
                    &candidate,
                    max_width_px,
                    font_size_px,
                )?
                .width_px
                <= max_width_px;
            if fits {
                best = candidate;
                low = mid.saturating_add(1);
            } else if mid == 0 {
                break;
            } else {
                high = mid - 1;
            }
        }

        Ok(if best.is_empty() {
            ellipsis.to_string()
        } else {
            best
        })
    }

    fn longest_fitting_prefix<'a>(
        &self,
        policy: &CaptionLayoutPolicy,
        style: &DirectWriteResolvedTextStyle,
        text: &'a str,
        max_width_px: f32,
        font_size_px: f32,
    ) -> Result<&'a str, windows::core::Error> {
        let trimmed = text.trim_start();
        if trimmed.is_empty() {
            return Ok("");
        }

        if self
            .measure_centered_line_for_resolved_style(
                policy,
                style,
                trimmed,
                max_width_px,
                font_size_px,
            )?
            .width_px
            <= max_width_px
        {
            return Ok(trimmed);
        }

        let mut best_end = 0usize;
        for (index, ch) in trimmed.char_indices() {
            let end = index + ch.len_utf8();
            let candidate = trimmed[..end].trim_end();
            if candidate.is_empty() {
                continue;
            }
            if self
                .measure_centered_line_for_resolved_style(
                    policy,
                    style,
                    candidate,
                    max_width_px,
                    font_size_px,
                )?
                .width_px
                <= max_width_px
            {
                best_end = if ch.is_whitespace() { index } else { end };
                continue;
            }
            break;
        }

        if best_end == 0 {
            for (index, ch) in trimmed.char_indices() {
                let end = index + ch.len_utf8();
                let candidate = &trimmed[..end];
                if self
                    .measure_centered_line_for_resolved_style(
                        policy,
                        style,
                        candidate,
                        max_width_px,
                        font_size_px,
                    )?
                    .width_px
                    > max_width_px
                {
                    return Ok(trimmed[..index].trim_end());
                }
                if end == trimmed.len() {
                    return Ok(candidate);
                }
            }
        }

        let prefix = trimmed[..best_end].trim_end();
        if prefix.is_empty() {
            Ok(trimmed
                .chars()
                .next()
                .map(|ch| &trimmed[..ch.len_utf8()])
                .unwrap_or(""))
        } else {
            Ok(prefix)
        }
    }

    fn measure_centered_line_for_resolved_style(
        &self,
        policy: &CaptionLayoutPolicy,
        resolved_style: &DirectWriteResolvedTextStyle,
        text: &str,
        content_width_px: f32,
        font_size_px: f32,
    ) -> Result<MeasuredLine, windows::core::Error> {
        let (text_layout, style) = self.create_text_layout_for_resolved_style(
            policy,
            resolved_style,
            text,
            font_size_px,
            content_width_px,
            font_size_px * 1.5,
            DWRITE_WORD_WRAPPING_NO_WRAP,
            None,
        )?;
        let mut metrics = DWRITE_TEXT_METRICS::default();
        unsafe {
            text_layout.GetMetrics(&mut metrics)?;
        }
        let overhang = unsafe { text_layout.GetOverhangMetrics()? };
        Ok(MeasuredLine {
            style_key: style.style_key,
            style,
            width_px: metrics.width,
            origin_x_px: metrics.left,
            visual_bounds: VisualBounds::new(
                metrics.left - overhang.left,
                -overhang.top,
                metrics.left + metrics.width + overhang.right,
                metrics.height + overhang.bottom,
            ),
        })
    }

    fn create_text_layout_for_resolved_style(
        &self,
        _policy: &CaptionLayoutPolicy,
        resolved_style: &DirectWriteResolvedTextStyle,
        text: &str,
        font_size_px: f32,
        max_width_px: f32,
        max_height_px: f32,
        word_wrapping: windows::Win32::Graphics::DirectWrite::DWRITE_WORD_WRAPPING,
        trimming_sign: Option<&IDWriteInlineObject>,
    ) -> Result<(IDWriteTextLayout, TextStyleDescriptor), windows::core::Error> {
        let (text_format, style) = match self.create_text_format_for_resolved_style(
            resolved_style,
            font_size_px,
            word_wrapping,
            trimming_sign,
        ) {
            Ok(text_format) => (text_format, resolved_style.descriptor()),
            Err(_error) if !resolved_style.is_style_failure_fallback => {
                let fallback = FontResolver::style_resolution_failure_fallback_for_bucket_locale(
                    resolved_style.bucket,
                    resolved_style.locale.clone(),
                );
                let fallback_style =
                    DirectWriteResolvedTextStyle::from_style(fallback, DWRITE_FONT_WEIGHT_NORMAL);
                let text_format = self.create_text_format_for_resolved_style(
                    &fallback_style,
                    font_size_px,
                    word_wrapping,
                    trimming_sign,
                )?;
                (text_format, fallback_style.descriptor())
            }
            Err(error) => return Err(error),
        };
        let utf16: Vec<u16> = text.encode_utf16().collect();
        let text_layout = unsafe {
            self.factory
                .CreateTextLayout(&utf16, &text_format, max_width_px, max_height_px)?
        };
        if let Ok(text_layout_2) = text_layout.cast::<IDWriteTextLayout2>() {
            unsafe {
                text_layout_2.SetFontFallback(&self.system_font_fallback)?;
            }
        }
        Ok((text_layout, style))
    }

    fn create_text_format_for_resolved_style(
        &self,
        resolved_style: &DirectWriteResolvedTextStyle,
        font_size_px: f32,
        word_wrapping: windows::Win32::Graphics::DirectWrite::DWRITE_WORD_WRAPPING,
        trimming_sign: Option<&IDWriteInlineObject>,
    ) -> Result<IDWriteTextFormat, windows::core::Error> {
        if resolved_style.source == FontSource::BundledNotoCjkMedium
            && self.bundled_font_collection.is_none()
        {
            let fallback = FontResolver::style_resolution_failure_fallback_for_bucket_locale(
                resolved_style.bucket,
                resolved_style.locale.clone(),
            );
            let fallback_style =
                DirectWriteResolvedTextStyle::from_style(fallback, DWRITE_FONT_WEIGHT_NORMAL);
            return self.create_text_format_for_resolved_style(
                &fallback_style,
                font_size_px,
                word_wrapping,
                trimming_sign,
            );
        }
        let locale = utf16_null(&resolved_style.locale);
        let face_name = utf16_null(&resolved_style.family_name);
        let text_format = unsafe {
            self.factory.CreateTextFormat(
                PCWSTR::from_raw(face_name.as_ptr()),
                if resolved_style.source == FontSource::BundledNotoCjkMedium {
                    self.bundled_font_collection
                        .as_ref()
                        .map(|collection| collection.collection())
                } else {
                    None
                },
                resolved_style.weight,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                font_size_px,
                PCWSTR::from_raw(locale.as_ptr()),
            )?
        };
        unsafe {
            text_format.SetWordWrapping(word_wrapping)?;
            text_format.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER)?;
            if let Ok(text_format_1) = text_format.cast::<IDWriteTextFormat1>() {
                text_format_1.SetFontFallback(&self.system_font_fallback)?;
            }
            if let Some(trimming_sign) = trimming_sign {
                text_format.SetTrimming(
                    &DWRITE_TRIMMING {
                        granularity: DWRITE_TRIMMING_GRANULARITY_CHARACTER,
                        delimiter: 0,
                        delimiterCount: 0,
                    },
                    trimming_sign,
                )?;
            }
        }
        Ok(text_format)
    }

    fn resolve_text_style(
        &self,
        policy: &CaptionLayoutPolicy,
        language: Option<&str>,
        text: &str,
    ) -> DirectWriteResolvedTextStyle {
        let requested_style = self.font_resolver.resolve(language, text);
        if requested_style.source == FontSource::BundledNotoCjkMedium
            && self.bundled_font_collection.is_some()
        {
            return DirectWriteResolvedTextStyle::from_style(
                requested_style,
                DWRITE_FONT_WEIGHT_MEDIUM,
            );
        }
        for family_name in requested_style
            .system_fallback_families()
            .iter()
            .copied()
            .filter(|candidate| *candidate != "DirectWrite system fallback")
        {
            let family = match self.find_font_family(family_name) {
                Ok(Some(family)) => family,
                Ok(None) => continue,
                Err(_error) => {
                    break;
                }
            };
            let Some(weight) = (match resolve_family_weight(&family, policy) {
                Ok(weight) => weight,
                Err(_error) => {
                    break;
                }
            }) else {
                continue;
            };
            let style_key = TextStyleKey::from_parts(
                requested_style.bucket,
                FontSource::SystemFont,
                None,
                family_name,
                font_weight_from_dwrite_weight(weight),
                &requested_style.locale,
            );
            return DirectWriteResolvedTextStyle {
                family_name: family_name.to_string(),
                weight,
                locale: requested_style.locale,
                source: FontSource::SystemFont,
                bucket: requested_style.bucket,
                style_key,
                is_style_failure_fallback: false,
            };
        }
        let fallback = FontResolver::style_resolution_failure_fallback_for_style(&requested_style);
        DirectWriteResolvedTextStyle::from_style(fallback, DWRITE_FONT_WEIGHT_NORMAL)
    }

    fn find_font_family(
        &self,
        family_name: &str,
    ) -> Result<Option<IDWriteFontFamily>, windows::core::Error> {
        let family_name = utf16_null(family_name);
        let mut index = 0;
        let mut exists = false.into();
        unsafe {
            self.system_font_collection.FindFamilyName(
                PCWSTR::from_raw(family_name.as_ptr()),
                &mut index,
                &mut exists,
            )?;
            if !exists.as_bool() {
                return Ok(None);
            }
            self.system_font_collection.GetFontFamily(index).map(Some)
        }
    }
}

pub(crate) fn resolved_layout_has_drawable_text(layout: &ResolvedFrameLayout) -> bool {
    layout.visible_blocks.iter().any(|block| {
        block
            .primary_lines
            .iter()
            .any(|line| !line.text.trim().is_empty())
            || block
                .secondary_line
                .as_ref()
                .is_some_and(|line| !line.text.trim().is_empty())
    })
}

fn layout_cache_key_for_block(
    block: &CaptionBlock,
    content_width_px: f32,
    text_scale: f32,
    primary_style_key: TextStyleKey,
    secondary_style_key: TextStyleKey,
) -> LayoutCacheKey {
    LayoutCacheKey {
        primary_text: block.primary_text.clone(),
        secondary_text: block.secondary_text.clone(),
        primary_style_key,
        secondary_style_key,
        channel: block.channel,
        block_variant: block.block_variant,
        secondary_enabled: block.secondary_enabled,
        secondary_reserved: block_reserves_secondary_row(block),
        primary_font_size_key: scalar_key(DEFAULT_FONT_SIZE_PX * text_scale),
        secondary_font_size_key: scalar_key(
            DEFAULT_FONT_SIZE_PX * text_scale * SECONDARY_FONT_SCALE,
        ),
        content_width_key: content_width_px.round() as u32,
        text_scale_key: scalar_key(text_scale),
    }
}

fn style_descriptor_for_text(
    resolver: &FontResolver,
    language: Option<&str>,
    text: &str,
) -> TextStyleDescriptor {
    let style = resolver.resolve(language, text);
    let style_key = style.style_key();
    TextStyleDescriptor::from_parts(
        style.family_name,
        style.weight,
        style.locale,
        style.source,
        style.bucket,
        style_key,
    )
}

fn block_reserves_secondary_row(block: &CaptionBlock) -> bool {
    block.secondary_enabled
        || block.slot_assigned
        || matches!(block.block_variant, CaptionBlockVariant::ActivePeer)
}

fn materialize_resolved_block_layout(
    block: &CaptionBlock,
    layout_cache_key: LayoutCacheKey,
    template: &CachedBlockLayoutTemplate,
    strip_left_px: f32,
    stable_top_px: f32,
) -> ResolvedBlockLayout {
    let render_top_px = stable_top_px + block.offset_y_px;
    let bounds = template
        .bounds
        .translate(strip_left_px, render_top_px)
        .scale_y_from_top(block.height_scale);
    let primary_lines = template
        .primary_lines
        .iter()
        .map(|line| {
            materialize_resolved_line_layout(line, strip_left_px, render_top_px, block.height_scale)
        })
        .collect::<Vec<_>>();
    let secondary_line = template.secondary_line.as_ref().map(|line| {
        materialize_resolved_line_layout(line, strip_left_px, render_top_px, block.height_scale)
    });
    let visual_bounds = template
        .visual_bounds
        .translate(strip_left_px, render_top_px)
        .scale_y_from_top(render_top_px, block.height_scale);

    ResolvedBlockLayout {
        id: block.id.clone(),
        layout_cache_key,
        channel: block.channel,
        block_variant: block.block_variant,
        primary_lines,
        secondary_line,
        secondary_reserved: template.secondary_reserved,
        bounds,
        visual_bounds,
        content_width_px: template.content_width_px,
        opacity: block.opacity,
        render_offset_y_px: block.offset_y_px,
        render_height_scale: block.height_scale,
        truncated_primary: template.truncated_primary,
        truncated_secondary: template.truncated_secondary,
    }
}

fn materialize_resolved_line_layout(
    line: &CachedLineLayoutTemplate,
    strip_left_px: f32,
    render_top_px: f32,
    render_height_scale: f32,
) -> ResolvedLineLayout {
    ResolvedLineLayout {
        text: line.text.clone(),
        role: line.role,
        style_key: line.style_key,
        style: line.style.clone(),
        width_px: line.width_px,
        origin_x: strip_left_px + line.origin_x,
        origin_y: render_top_px + line.origin_y * render_height_scale,
        font_size_px: line.font_size_px,
        visual_bounds: line
            .visual_bounds
            .translate(strip_left_px, render_top_px)
            .scale_y_from_top(render_top_px, render_height_scale),
    }
}

fn scalar_key(value: f32) -> u32 {
    (value * 100.0).round() as u32
}

fn line_visual_bounds(
    origin_x: f32,
    origin_y: f32,
    width_px: f32,
    font_size_px: f32,
) -> VisualBounds {
    VisualBounds::new(
        origin_x - TEXT_OUTLINE_OVERHANG_PX,
        origin_y - TEXT_OUTLINE_OVERHANG_PX,
        origin_x + width_px + TEXT_OUTLINE_OVERHANG_PX,
        origin_y + font_size_px * 1.15 + TEXT_OUTLINE_OVERHANG_PX,
    )
}

fn block_visual_bounds_from_templates(
    bounds: BlockBounds,
    primary_lines: &[CachedLineLayoutTemplate],
    secondary_line: Option<&CachedLineLayoutTemplate>,
) -> VisualBounds {
    let mut left_px = bounds.left_px - TEXT_OUTLINE_OVERHANG_PX;
    let mut top_px = bounds.top_px - TEXT_OUTLINE_OVERHANG_PX;
    let mut right_px = bounds.right_px + TEXT_OUTLINE_OVERHANG_PX;
    let mut bottom_px = bounds.bottom_px + TEXT_OUTLINE_OVERHANG_PX;

    for line in primary_lines
        .iter()
        .chain(secondary_line.iter().copied())
        .filter(|line| !line.text.trim().is_empty())
    {
        left_px = left_px.min(line.visual_bounds.left_px);
        top_px = top_px.min(line.visual_bounds.top_px);
        right_px = right_px.max(line.visual_bounds.right_px);
        bottom_px = bottom_px.max(line.visual_bounds.bottom_px);
    }

    VisualBounds::new(left_px, top_px, right_px, bottom_px)
}

fn wrap_text(text: &str, max_width_px: f32, average_glyph_advance_px: f32) -> Vec<String> {
    let mut lines = Vec::new();

    for paragraph in text.lines() {
        let words: Vec<&str> = paragraph.split_whitespace().collect();
        if words.is_empty() {
            lines.push(String::new());
            continue;
        }

        let mut current = String::new();
        for word in words {
            if current.is_empty() {
                push_word_chunks(
                    &mut lines,
                    &mut current,
                    word,
                    max_width_px,
                    average_glyph_advance_px,
                );
                continue;
            }

            let candidate = format!("{current} {word}");
            if measure_text_width(&candidate, average_glyph_advance_px) <= max_width_px {
                current.push(' ');
                current.push_str(word);
                continue;
            }

            lines.push(std::mem::take(&mut current));
            push_word_chunks(
                &mut lines,
                &mut current,
                word,
                max_width_px,
                average_glyph_advance_px,
            );
        }

        if !current.is_empty() {
            lines.push(current);
        }
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

fn ellipsize_text(
    text: &str,
    max_width_px: f32,
    average_glyph_advance_px: f32,
) -> (Option<String>, bool) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (None, false);
    }
    if measure_text_width(trimmed, average_glyph_advance_px) <= max_width_px {
        return (Some(trimmed.to_string()), false);
    }

    let ellipsis = "...";
    let ellipsis_width = measure_text_width(ellipsis, average_glyph_advance_px);
    if ellipsis_width >= max_width_px {
        return (Some(ellipsis.to_string()), true);
    }

    let mut out = String::new();
    for ch in trimmed.chars() {
        let candidate = format!("{out}{ch}");
        if measure_text_width(&candidate, average_glyph_advance_px) + ellipsis_width > max_width_px
        {
            break;
        }
        out.push(ch);
    }

    if out.is_empty() {
        (Some(ellipsis.to_string()), true)
    } else {
        out.push_str(ellipsis);
        (Some(out), true)
    }
}

fn push_word_chunks(
    lines: &mut Vec<String>,
    current: &mut String,
    word: &str,
    max_width_px: f32,
    average_glyph_advance_px: f32,
) {
    if measure_text_width(word, average_glyph_advance_px) <= max_width_px {
        current.push_str(word);
        return;
    }

    let mut piece = String::new();
    for ch in word.chars() {
        let candidate = format!("{piece}{ch}");
        if !piece.is_empty()
            && measure_text_width(&candidate, average_glyph_advance_px) > max_width_px
        {
            if current.is_empty() {
                lines.push(std::mem::take(&mut piece));
            } else {
                lines.push(std::mem::take(current));
                lines.push(std::mem::take(&mut piece));
            }
        }
        piece.push(ch);
    }

    if !piece.is_empty() {
        if current.is_empty() {
            current.push_str(&piece);
        } else {
            lines.push(std::mem::take(current));
            current.push_str(&piece);
        }
    }
}

fn measure_text_width(text: &str, average_glyph_advance_px: f32) -> f32 {
    text.chars()
        .map(|ch| match ch {
            ' ' => average_glyph_advance_px * 0.45,
            '0'..='9' | 'A'..='Z' => average_glyph_advance_px * 0.68,
            'a'..='z' => average_glyph_advance_px * 0.62,
            '.' | ',' | ':' | ';' | '\'' | '"' | '!' | '?' | '(' | ')' | '[' | ']' | '{' | '}'
            | '-' | '_' | '/' => average_glyph_advance_px * 0.55,
            _ if super::types::contains_cjk(&ch.to_string()) => average_glyph_advance_px,
            _ => average_glyph_advance_px * 0.72,
        })
        .sum()
}

#[cfg(windows)]
fn resolve_family_weight(
    family: &IDWriteFontFamily,
    policy: &CaptionLayoutPolicy,
) -> Result<Option<windows::Win32::Graphics::DirectWrite::DWRITE_FONT_WEIGHT>, windows::core::Error>
{
    for weight in preferred_weight_chain(policy) {
        let font = unsafe {
            family.GetFirstMatchingFont(
                weight,
                DWRITE_FONT_STRETCH_NORMAL,
                DWRITE_FONT_STYLE_NORMAL,
            )?
        };
        if unsafe { font.GetWeight() } == weight {
            return Ok(Some(weight));
        }
    }
    Ok(None)
}

#[cfg(windows)]
fn preferred_weight_chain(
    policy: &CaptionLayoutPolicy,
) -> Vec<windows::Win32::Graphics::DirectWrite::DWRITE_FONT_WEIGHT> {
    policy
        .preferred_weights()
        .into_iter()
        .map(|weight| match weight {
            "Semibold" => DWRITE_FONT_WEIGHT_SEMI_BOLD,
            "Medium" => DWRITE_FONT_WEIGHT_MEDIUM,
            _ => DWRITE_FONT_WEIGHT_NORMAL,
        })
        .collect()
}

#[cfg(windows)]
fn font_weight_from_dwrite_weight(weight: DWRITE_FONT_WEIGHT) -> FontWeight {
    if weight == DWRITE_FONT_WEIGHT_SEMI_BOLD {
        FontWeight::SemiBold
    } else if weight == DWRITE_FONT_WEIGHT_MEDIUM {
        FontWeight::Medium
    } else {
        FontWeight::Regular
    }
}

#[cfg(windows)]
fn utf16_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::super::types::TEXT_OUTLINE_OVERHANG_PX;
    use super::{measure_text_width, wrap_text, CaptionLayoutPolicy};
    use crate::renderer::{
        effective_background_alpha, fill_color_for_channel, outline_offsets_px, text_script_bucket,
        CaptionBlock, CaptionChannel, CaptionPresentation, FontResolver, TextScriptBucket,
    };

    fn fallback_styles(
        block: &CaptionBlock,
    ) -> (super::TextStyleDescriptor, super::TextStyleDescriptor) {
        let resolver = FontResolver::default();
        (
            super::style_descriptor_for_text(
                &resolver,
                block.primary_language.as_deref(),
                &block.primary_text,
            ),
            super::style_descriptor_for_text(
                &resolver,
                block.secondary_language.as_deref(),
                &block.secondary_text,
            ),
        )
    }

    #[test]
    fn wrap_text_splits_long_words_into_measured_chunks() {
        let lines = wrap_text("abcdefgh", 160.0, 80.0);
        assert_eq!(lines, vec!["abc", "def", "gh"]);
    }

    #[test]
    fn measure_text_width_treats_cjk_as_wider_than_latin() {
        assert!(measure_text_width("안녕", 80.0) > measure_text_width("hi", 80.0));
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.01,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn fallback_layout_uses_scaled_average_glyph_advance_for_132px_primary() {
        let policy = CaptionLayoutPolicy::default();
        let block = CaptionBlock::new("fallback", "aaaaaaaaaaaaaaa");
        let (primary_style, secondary_style) = fallback_styles(&block);
        let template = policy.build_fallback_block_template(
            &block,
            699.0,
            1.0,
            primary_style,
            secondary_style,
        );

        let primary_texts = template
            .primary_lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(primary_texts, vec!["aaaaaaaaaaaaaaa"]);
        assert_close(template.primary_lines[0].font_size_px, 132.0);
    }

    #[test]
    fn fallback_layout_secondary_origin_includes_scaled_primary_secondary_gap() {
        let policy = CaptionLayoutPolicy::default();
        let block = CaptionBlock::new("peer:translated", "translated peer text")
            .with_secondary_text("source peer text", true);
        let (primary_style, secondary_style) = fallback_styles(&block);
        let template = policy.build_fallback_block_template(
            &block,
            3200.0,
            1.25,
            primary_style,
            secondary_style,
        );
        let secondary = template
            .secondary_line
            .as_ref()
            .expect("secondary line should be present");

        assert_close(secondary.origin_y, (32.0 + 2.0 * 150.0 + 30.0) * 1.25);
    }

    #[cfg(windows)]
    #[test]
    fn windows_graphics_directwrite_layout_secondary_origin_includes_scaled_primary_secondary_gap()
    {
        let policy = CaptionLayoutPolicy::default();
        let engine = super::DirectWriteLayoutEngine::new_for_test()
            .expect("DirectWrite layout should initialize on Windows");
        let layout = policy
            .resolve_blocks_for_presentation_windows_cached(
                vec![CaptionBlock::new("peer:translated", "translated peer text")
                    .with_secondary_text("source peer text", true)],
                3840,
                1024,
                &CaptionPresentation::default(),
                &engine,
                None,
            )
            .expect("DirectWrite layout should initialize on Windows");
        let block = &layout.visible_blocks[0];
        let secondary = block
            .secondary_line
            .as_ref()
            .expect("secondary line should be present");

        assert_close(
            secondary.origin_y,
            block.bounds.top_px + 32.0 + 2.0 * 150.0 + 30.0,
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_graphics_layout_key_and_line_key_use_same_resolved_style_as_measurement() {
        let policy = CaptionLayoutPolicy::default();
        let engine = super::DirectWriteLayoutEngine::new_for_test()
            .expect("DirectWrite layout should initialize on Windows");
        let layout = policy
            .resolve_blocks_for_presentation_windows_cached(
                vec![CaptionBlock::new("latin", "hello style identity")],
                3840,
                1024,
                &CaptionPresentation::default(),
                &engine,
                None,
            )
            .expect("DirectWrite layout should initialize on Windows");

        let block = &layout.visible_blocks[0];
        let line = block
            .primary_lines
            .first()
            .expect("primary line should be present");
        let resolved_style_key = engine.resolved_text_style_key_for_test(&policy, &line.text);

        assert_eq!(block.layout_cache_key.primary_style_key, resolved_style_key);
        assert_eq!(line.style_key, resolved_style_key);
    }

    #[cfg(windows)]
    #[test]
    fn windows_graphics_wrapped_mixed_script_layout_cache_key_is_reachable_on_second_resolve() {
        let policy = CaptionLayoutPolicy::default();
        let engine = super::DirectWriteLayoutEngine::new_for_test()
            .expect("DirectWrite layout should initialize on Windows");
        let presentation = CaptionPresentation::default();
        let block = CaptionBlock::new(
            "mixed",
            format!(
                "{} 日本語",
                "streaming translation captions should keep the newest utterance readable "
                    .repeat(4)
            ),
        );
        let mut cache = super::LayoutCache::with_capacity(4);

        let first = policy
            .resolve_blocks_for_presentation_windows_cached(
                vec![block.clone()],
                1100,
                900,
                &presentation,
                &engine,
                Some(&mut cache),
            )
            .expect("DirectWrite layout should initialize on Windows");
        assert!(
            first.visible_blocks[0].primary_lines.len() > 1,
            "test text must wrap so lookup and measured-line style can diverge"
        );

        let lookup_key =
            policy.layout_cache_key_for_block_windows(&block, 1100, &presentation, &engine);
        assert!(
            cache.get(&lookup_key).is_some(),
            "first resolve should insert using the same key the second resolve will look up"
        );
        let cache_len_after_first = cache.len();

        let _ = policy
            .resolve_blocks_for_presentation_windows_cached(
                vec![block],
                1100,
                900,
                &presentation,
                &engine,
                Some(&mut cache),
            )
            .expect("DirectWrite layout should initialize on Windows");

        assert_eq!(cache.len(), cache_len_after_first);
        assert!(cache.get(&lookup_key).is_some());
    }

    #[test]
    fn text_script_bucket_prefers_latin_for_non_cjk_text() {
        assert_eq!(text_script_bucket("hello world"), TextScriptBucket::Latin);
    }

    #[test]
    fn text_script_bucket_uses_cjk_bucket_for_korean_text() {
        assert_eq!(text_script_bucket("안녕하세요"), TextScriptBucket::Cjk);
    }

    #[test]
    fn fill_color_for_channel_uses_fixed_text_only_palette() {
        let this = fill_color_for_channel(CaptionChannel::SelfChannel);
        let peer = fill_color_for_channel(CaptionChannel::PeerChannel);
        assert_eq!(this, (1.0, 1.0, 1.0, 1.0));
        assert_eq!(peer, (1.0, 215.0 / 255.0, 0.0, 1.0));
        assert_ne!(this, peer);
        assert_eq!(this.3, 1.0);
        assert_eq!(peer.3, 1.0);
    }

    #[test]
    fn outline_offsets_px_match_the_vr_outline_profile() {
        let offsets = outline_offsets_px();
        assert_eq!(offsets.len(), 4);
        assert_eq!(offsets[0].1, 0.0);
        assert_eq!(offsets[1].1, 0.0);
        assert_eq!(offsets[2].0, 0.0);
        assert_eq!(offsets[3].0, 0.0);
        assert_eq!(offsets[0].0, -offsets[1].0);
        assert_eq!(offsets[2].1, -offsets[3].1);
        assert!(offsets[0].0.abs() > 0.0);
        assert!(offsets[2].1.abs() > 0.0);
        assert_eq!(
            offsets[0].0.abs().max(offsets[2].1.abs()),
            TEXT_OUTLINE_OVERHANG_PX
        );
    }

    #[test]
    fn effective_background_alpha_is_zero_for_text_only_overlay() {
        let presentation = CaptionPresentation {
            background_alpha: 0.82,
            text_scale: 1.0,
        };

        assert_eq!(effective_background_alpha(true, &presentation), 0.0);
        assert_eq!(effective_background_alpha(false, &presentation), 0.0);
    }
}
