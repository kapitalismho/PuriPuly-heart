use thiserror::Error;

#[cfg(windows)]
use windows::Win32::Graphics::DirectWrite::DWRITE_FONT_WEIGHT;

use super::font_resolver::{FontLanguageBucket, FontSource, FontWeight, TextStyleKey};

pub(crate) const DEFAULT_SURFACE_WIDTH_PX: u32 = 4096;
pub(crate) const DEFAULT_SURFACE_HEIGHT_PX: u32 = 1056;
pub(crate) const DEFAULT_HORIZONTAL_PADDING_PX: u32 = 48;
pub(crate) const DEFAULT_VERTICAL_PADDING_PX: u32 = 40;
pub(crate) const DEFAULT_PRIMARY_LINE_HEIGHT_PX: u32 = 150;
pub(crate) const DEFAULT_SECONDARY_LINE_HEIGHT_PX: u32 = 104;
pub(crate) const DEFAULT_BLOCK_SPACING_PX: u32 = 36;
pub(crate) const DEFAULT_STRIP_HORIZONTAL_PADDING_PX: u32 = 24;
pub(crate) const DEFAULT_STRIP_VERTICAL_PADDING_PX: u32 = 32;
pub(crate) const DEFAULT_AVERAGE_GLYPH_ADVANCE_RATIO: f32 = 80.0 / 140.0;
pub(crate) const DEFAULT_FONT_SIZE_PX: f32 = 132.0;
// 132.0 * (80.0 / 140.0) = 75.43; truncating keeps fallback metrics close to the baseline ratio.
pub(crate) const DEFAULT_AVERAGE_GLYPH_ADVANCE_PX: u32 =
    (DEFAULT_FONT_SIZE_PX * DEFAULT_AVERAGE_GLYPH_ADVANCE_RATIO) as u32;
pub(crate) const SECONDARY_FONT_SCALE: f32 = 0.62;
#[allow(dead_code)]
pub(crate) const PRIMARY_SECONDARY_GAP_PX: f32 = 30.0;
pub(crate) const TEXT_OUTLINE_OVERHANG_PX: f32 = 5.0;
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) const SELF_TEXT_FILL_COLOR: (f32, f32, f32, f32) = (1.0, 1.0, 1.0, 1.0);
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) const PEER_TEXT_FILL_COLOR: (f32, f32, f32, f32) = (1.0, 215.0 / 255.0, 0.0, 1.0);
#[cfg(windows)]
pub(crate) const TEXT_OUTLINE_COLOR: (f32, f32, f32, f32) = (0.0, 0.0, 0.0, 1.0);
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) const TEXT_OUTLINE_OFFSETS_PX: [(f32, f32); 4] =
    [(-5.0, 0.0), (5.0, 0.0), (0.0, -5.0), (0.0, 5.0)];

#[derive(Debug, Error)]
pub enum CaptionRenderError {
    #[error("renderer init failed: {0}")]
    Init(String),
    #[error("hardware device failed: {0}")]
    HardwareDevice(String),
    #[error("renderer draw failed: {0}")]
    Draw(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptionBlock {
    pub id: String,
    pub primary_text: String,
    pub secondary_text: String,
    pub secondary_enabled: bool,
    pub primary_language: Option<String>,
    pub secondary_language: Option<String>,
    pub block_variant: CaptionBlockVariant,
    pub channel: Option<CaptionChannel>,
    pub opacity: f32,
    pub offset_y_px: f32,
    pub height_scale: f32,
    pub slot_index: usize,
    pub slot_top_px: f32,
    pub slot_assigned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptionBlockVariant {
    ActiveSelf,
    // Reserved compatibility/fallback variant; normal peer product rendering is
    // driven by translated finalized rows rather than source-only active_peer.
    ActivePeer,
    Finalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptionChannel {
    SelfChannel,
    PeerChannel,
}

impl CaptionBlock {
    pub fn new(id: impl Into<String>, primary_text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            primary_text: primary_text.into(),
            secondary_text: String::new(),
            secondary_enabled: true,
            primary_language: None,
            secondary_language: None,
            block_variant: CaptionBlockVariant::Finalized,
            channel: None,
            opacity: 1.0,
            offset_y_px: 0.0,
            height_scale: 1.0,
            slot_index: 0,
            slot_top_px: 0.0,
            slot_assigned: false,
        }
    }

    pub fn with_secondary_text(
        mut self,
        secondary_text: impl Into<String>,
        secondary_enabled: bool,
    ) -> Self {
        self.secondary_text = secondary_text.into();
        self.secondary_enabled = secondary_enabled;
        self
    }

    pub fn with_primary_language(mut self, language: impl Into<String>) -> Self {
        let language = language.into();
        self.primary_language = clean_language_option(language);
        self
    }

    pub fn with_secondary_language(mut self, language: impl Into<String>) -> Self {
        let language = language.into();
        self.secondary_language = clean_language_option(language);
        self
    }

    pub fn with_language_metadata(
        mut self,
        primary_language: Option<String>,
        secondary_language: Option<String>,
    ) -> Self {
        self.primary_language = primary_language.and_then(clean_language_option);
        self.secondary_language = secondary_language.and_then(clean_language_option);
        self
    }

    pub fn with_variant(mut self, block_variant: CaptionBlockVariant) -> Self {
        self.block_variant = block_variant;
        self
    }

    pub fn with_channel(mut self, channel: CaptionChannel) -> Self {
        self.channel = Some(channel);
        self
    }

    pub fn with_visual_state(mut self, opacity: f32, offset_y_px: f32, height_scale: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self.offset_y_px = offset_y_px;
        self.height_scale = height_scale.clamp(0.35, 4.0);
        self
    }

    pub fn with_slot(mut self, slot_index: usize, slot_top_px: f32) -> Self {
        self.slot_index = slot_index;
        self.slot_top_px = slot_top_px;
        self.slot_assigned = true;
        self
    }

    pub fn has_drawable_text(&self) -> bool {
        !self.primary_text.trim().is_empty()
            || (self.secondary_enabled && !self.secondary_text.trim().is_empty())
    }
}

fn clean_language_option(language: String) -> Option<String> {
    let trimmed = language.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockBounds {
    pub left_px: f32,
    pub top_px: f32,
    pub right_px: f32,
    pub bottom_px: f32,
}

impl BlockBounds {
    pub fn new(left_px: f32, top_px: f32, right_px: f32, bottom_px: f32) -> Self {
        Self {
            left_px,
            top_px,
            right_px,
            bottom_px,
        }
    }

    pub fn center_x(&self) -> f32 {
        (self.left_px + self.right_px) * 0.5
    }

    pub fn translate(self, offset_x: f32, offset_y: f32) -> Self {
        Self::new(
            self.left_px + offset_x,
            self.top_px + offset_y,
            self.right_px + offset_x,
            self.bottom_px + offset_y,
        )
    }

    pub fn scale_y_from_top(self, scale_y: f32) -> Self {
        Self::new(
            self.left_px,
            self.top_px,
            self.right_px,
            self.top_px + (self.bottom_px - self.top_px) * scale_y,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualBounds {
    pub left_px: f32,
    pub top_px: f32,
    pub right_px: f32,
    pub bottom_px: f32,
}

impl VisualBounds {
    pub fn new(left_px: f32, top_px: f32, right_px: f32, bottom_px: f32) -> Self {
        Self {
            left_px,
            top_px,
            right_px,
            bottom_px,
        }
    }

    pub fn as_block_bounds(self) -> BlockBounds {
        BlockBounds::new(self.left_px, self.top_px, self.right_px, self.bottom_px)
    }

    pub fn translate(self, offset_x: f32, offset_y: f32) -> Self {
        Self::new(
            self.left_px + offset_x,
            self.top_px + offset_y,
            self.right_px + offset_x,
            self.bottom_px + offset_y,
        )
    }

    pub fn scale_y_from_top(self, top_px: f32, scale_y: f32) -> Self {
        Self::new(
            self.left_px,
            top_px + (self.top_px - top_px) * scale_y,
            self.right_px,
            top_px + (self.bottom_px - top_px) * scale_y,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptionLineLayout {
    pub text: String,
    pub width_px: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub font_size_px: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DamageBand {
    pub top_px: f32,
    pub bottom_px: f32,
}

impl DamageBand {
    pub fn from_bounds<I>(bounds: I) -> Option<Self>
    where
        I: IntoIterator<Item = BlockBounds>,
    {
        let mut iter = bounds.into_iter();
        let first = iter.next()?;
        let mut top_px = first.top_px;
        let mut bottom_px = first.bottom_px;

        for bounds in iter {
            top_px = top_px.min(bounds.top_px);
            bottom_px = bottom_px.max(bounds.bottom_px);
        }

        Some(Self { top_px, bottom_px })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisibleCaptionBlock {
    pub id: String,
    pub channel: Option<CaptionChannel>,
    pub block_variant: CaptionBlockVariant,
    pub primary_lines: Vec<CaptionLineLayout>,
    pub secondary_line: Option<CaptionLineLayout>,
    pub secondary_reserved: bool,
    pub bounds: BlockBounds,
    pub visual_bounds: VisualBounds,
    pub content_width_px: f32,
    pub opacity: f32,
    pub truncated_primary: bool,
    pub truncated_secondary: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptionLayoutResult {
    pub visible_blocks: Vec<VisibleCaptionBlock>,
    pub dropped_block_ids: Vec<String>,
    pub surface_width_px: u32,
    pub surface_height_px: u32,
    pub damage_band: Option<DamageBand>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptionPresentation {
    pub background_alpha: f32,
    pub text_scale: f32,
}

impl Default for CaptionPresentation {
    fn default() -> Self {
        Self {
            background_alpha: 0.24,
            text_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionDebugOverlay {
    label: String,
}

impl CaptionDebugOverlay {
    pub fn new(label: impl Into<String>) -> Option<Self> {
        let label = label.into();
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(Self {
            label: trimmed.chars().take(96).collect(),
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineRole {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLineLayout {
    pub text: String,
    pub role: LineRole,
    pub style_key: TextStyleKey,
    pub style: TextStyleDescriptor,
    pub width_px: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub font_size_px: f32,
    pub visual_bounds: VisualBounds,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedBlockLayout {
    pub id: String,
    pub layout_cache_key: LayoutCacheKey,
    pub channel: Option<CaptionChannel>,
    pub block_variant: CaptionBlockVariant,
    pub primary_lines: Vec<ResolvedLineLayout>,
    pub secondary_line: Option<ResolvedLineLayout>,
    pub secondary_reserved: bool,
    pub bounds: BlockBounds,
    pub visual_bounds: VisualBounds,
    pub content_width_px: f32,
    pub opacity: f32,
    pub render_offset_y_px: f32,
    pub render_height_scale: f32,
    pub truncated_primary: bool,
    pub truncated_secondary: bool,
}

impl ResolvedBlockLayout {
    pub fn block_cache_key(&self) -> BlockCacheKey {
        BlockCacheKey {
            id: self.id.clone(),
            layout: self.layout_cache_key.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFrameLayout {
    pub visible_blocks: Vec<ResolvedBlockLayout>,
    pub dropped_block_ids: Vec<String>,
    pub surface_width_px: u32,
    pub surface_height_px: u32,
    pub damage_band: Option<DamageBand>,
}

impl From<ResolvedLineLayout> for CaptionLineLayout {
    fn from(value: ResolvedLineLayout) -> Self {
        Self {
            text: value.text,
            width_px: value.width_px,
            origin_x: value.origin_x,
            origin_y: value.origin_y,
            font_size_px: value.font_size_px,
        }
    }
}

impl From<ResolvedBlockLayout> for VisibleCaptionBlock {
    fn from(value: ResolvedBlockLayout) -> Self {
        Self {
            id: value.id,
            channel: value.channel,
            block_variant: value.block_variant,
            primary_lines: value.primary_lines.into_iter().map(Into::into).collect(),
            secondary_line: value.secondary_line.map(Into::into),
            secondary_reserved: value.secondary_reserved,
            bounds: value.bounds,
            visual_bounds: value.visual_bounds,
            content_width_px: value.content_width_px,
            opacity: value.opacity,
            truncated_primary: value.truncated_primary,
            truncated_secondary: value.truncated_secondary,
        }
    }
}

impl From<ResolvedFrameLayout> for CaptionLayoutResult {
    fn from(value: ResolvedFrameLayout) -> Self {
        Self {
            visible_blocks: value.visible_blocks.into_iter().map(Into::into).collect(),
            dropped_block_ids: value.dropped_block_ids,
            surface_width_px: value.surface_width_px,
            surface_height_px: value.surface_height_px,
            damage_band: value.damage_band,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayoutCacheKey {
    pub primary_text: String,
    pub secondary_text: String,
    pub primary_style_key: TextStyleKey,
    pub secondary_style_key: TextStyleKey,
    pub channel: Option<CaptionChannel>,
    pub block_variant: CaptionBlockVariant,
    pub secondary_enabled: bool,
    pub secondary_reserved: bool,
    pub primary_font_size_key: u32,
    pub secondary_font_size_key: u32,
    pub content_width_key: u32,
    pub text_scale_key: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LineCacheKey {
    pub text: String,
    pub role: LineRole,
    pub style_key: TextStyleKey,
    pub channel: Option<CaptionChannel>,
    pub block_variant: CaptionBlockVariant,
    pub font_size_key: u32,
    pub content_width_key: u32,
    pub text_scale_key: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockCacheKey {
    pub id: String,
    pub layout: LayoutCacheKey,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderDiagnostics {
    pub text_format_cache_size: usize,
    pub layout_cache_size: usize,
    pub line_cache_size: usize,
    pub block_cache_size: usize,
    pub text_format_cache_hits: u32,
    pub text_format_cache_misses: u32,
    pub font_warmup_attempts: u32,
    pub font_warmup_failures: u32,
    pub directwrite_layout_success_count: u32,
    pub heuristic_layout_fallback_count: u32,
    pub layout_cache_hits: u32,
    pub layout_cache_misses: u32,
    pub line_cache_hits: u32,
    pub line_cache_misses: u32,
    pub block_cache_hits: u32,
    pub block_cache_misses: u32,
    pub debug_overlay_draw_count: u32,
    pub debug_overlay_clear_count: u32,
    pub style_bucket_source_counts: Vec<StyleBucketSourceCount>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StyleBucketSourceCount {
    pub bucket: FontLanguageBucket,
    pub source: FontSource,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextStyleDescriptor {
    pub family_name: String,
    pub weight: FontWeight,
    pub locale: String,
    pub source: FontSource,
    pub bucket: FontLanguageBucket,
    pub style_key: TextStyleKey,
}

impl TextStyleDescriptor {
    pub fn from_parts(
        family_name: impl Into<String>,
        weight: FontWeight,
        locale: impl Into<String>,
        source: FontSource,
        bucket: FontLanguageBucket,
        style_key: TextStyleKey,
    ) -> Self {
        Self {
            family_name: family_name.into(),
            weight,
            locale: locale.into(),
            source,
            bucket,
            style_key,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TextScriptBucket {
    Latin,
    Cjk,
}

#[cfg(windows)]
#[derive(Debug, Clone)]
pub(crate) struct ResolvedTextStyle {
    pub family_name: String,
    pub weight: DWRITE_FONT_WEIGHT,
    pub locale: String,
    pub source: FontSource,
    pub bucket: FontLanguageBucket,
    pub style_key: TextStyleKey,
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn fill_color_for_channel(channel: CaptionChannel) -> (f32, f32, f32, f32) {
    match channel {
        CaptionChannel::SelfChannel => SELF_TEXT_FILL_COLOR,
        CaptionChannel::PeerChannel => PEER_TEXT_FILL_COLOR,
    }
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn outline_offsets_px() -> [(f32, f32); 4] {
    TEXT_OUTLINE_OFFSETS_PX
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn effective_background_alpha(
    _has_drawable_text: bool,
    _presentation: &CaptionPresentation,
) -> f32 {
    0.0
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn contains_cjk(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(
            ch as u32,
            0x3040..=0x30ff
                | 0x3400..=0x4dbf
                | 0x4e00..=0x9fff
                | 0xac00..=0xd7af
                | 0xf900..=0xfaff
        )
    })
}

#[allow(dead_code)]
pub(crate) fn text_script_bucket(text: &str) -> TextScriptBucket {
    if contains_cjk(text) {
        TextScriptBucket::Cjk
    } else {
        TextScriptBucket::Latin
    }
}
