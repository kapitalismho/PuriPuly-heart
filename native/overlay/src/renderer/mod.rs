mod backend;
mod cache;
mod font_resolver;
mod glyph_run;
mod layout;
mod types;

pub use backend::{CaptionRenderer, RenderedFrame};
#[cfg(windows)]
pub use font_resolver::WindowsBundledFontCollection;
pub use font_resolver::{
    bundled_font_path_from_exe_dir, runtime_bundled_font_path, BundledFaceId, FontFallbackReason,
    FontLanguageBucket, FontResolver, FontSource, FontWeight, ResolvedFontStyle, TextFamilyKey,
    TextLocaleKey, TextStyleKey, BUNDLED_NOTO_CJK_FILE_NAME,
};
pub use layout::CaptionLayoutPolicy;
pub use types::{
    BlockBounds, BlockCacheKey, CaptionBlock, CaptionBlockVariant, CaptionChannel,
    CaptionDebugOverlay, CaptionLayoutResult, CaptionLineLayout, CaptionPresentation,
    CaptionRenderError, DamageBand, LayoutCacheKey, LineCacheKey, LineRole, RenderDiagnostics,
    ResolvedBlockLayout, ResolvedFrameLayout, ResolvedLineLayout, SpeakerDividerBand, SpeakerStyle,
    StyleBucketSourceCount, TextStyleDescriptor, VisibleCaptionBlock, VisualBounds,
};
