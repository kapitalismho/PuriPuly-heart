use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

#[cfg(windows)]
use windows::Win32::Graphics::Direct2D::ID2D1CommandList;
#[cfg(windows)]
use windows::Win32::Graphics::DirectWrite::IDWriteTextFormat;

#[cfg(windows)]
use super::font_resolver::TextStyleKey;
use super::types::{BlockBounds, LayoutCacheKey, LineRole, TextStyleDescriptor, VisualBounds};
#[cfg(windows)]
use super::types::{BlockCacheKey, LineCacheKey};

pub(crate) const TEXT_FORMAT_CACHE_CAP: usize = 16;
pub(crate) const LAYOUT_CACHE_CAP: usize = 64;
pub(crate) const LINE_CACHE_CAP: usize = 256;
pub(crate) const BLOCK_CACHE_CAP: usize = 128;

#[derive(Debug)]
pub(crate) struct BoundedLruCache<K, V> {
    capacity: usize,
    entries: HashMap<K, V>,
    recency: VecDeque<K>,
}

impl<K, V> BoundedLruCache<K, V>
where
    K: Clone + Eq + Hash,
{
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            recency: VecDeque::with_capacity(capacity),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    pub(crate) fn get(&mut self, key: &K) -> Option<&V> {
        if !self.entries.contains_key(key) {
            return None;
        }
        self.touch(key);
        self.entries.get(key)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), value);
            self.touch(&key);
            return;
        }
        while self.entries.len() >= self.capacity {
            let Some(oldest_key) = self.recency.pop_front() else {
                break;
            };
            self.entries.remove(&oldest_key);
        }
        self.recency.push_back(key.clone());
        self.entries.insert(key, value);
    }

    fn touch(&mut self, key: &K) {
        self.recency.retain(|candidate| candidate != key);
        self.recency.push_back(key.clone());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CachedLineLayoutTemplate {
    pub text: String,
    pub role: LineRole,
    pub style_key: super::font_resolver::TextStyleKey,
    pub style: TextStyleDescriptor,
    pub width_px: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub font_size_px: f32,
    pub visual_bounds: VisualBounds,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CachedBlockLayoutTemplate {
    pub primary_lines: Vec<CachedLineLayoutTemplate>,
    pub secondary_line: Option<CachedLineLayoutTemplate>,
    pub secondary_reserved: bool,
    pub bounds: BlockBounds,
    pub visual_bounds: VisualBounds,
    pub content_width_px: f32,
    pub truncated_primary: bool,
    pub truncated_secondary: bool,
}

#[derive(Debug)]
pub(crate) struct LayoutCache {
    entries: BoundedLruCache<LayoutCacheKey, CachedBlockLayoutTemplate>,
}

impl Default for LayoutCache {
    fn default() -> Self {
        Self::with_capacity(LAYOUT_CACHE_CAP)
    }
}

impl LayoutCache {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: BoundedLruCache::with_capacity(capacity),
        }
    }

    pub(crate) fn get(&mut self, key: &LayoutCacheKey) -> Option<&CachedBlockLayoutTemplate> {
        self.entries.get(key)
    }

    pub(crate) fn insert(&mut self, key: LayoutCacheKey, value: CachedBlockLayoutTemplate) {
        self.entries.insert(key, value);
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn contains_key(&self, key: &LayoutCacheKey) -> bool {
        self.entries.contains_key(key)
    }
}

#[cfg(windows)]
#[derive(Debug)]
pub(crate) struct WindowsRendererCaches {
    pub text_format_cache: BoundedLruCache<(TextStyleKey, u32), IDWriteTextFormat>,
    pub layout_cache: LayoutCache,
    pub line_cache: BoundedLruCache<LineCacheKey, ID2D1CommandList>,
    pub block_cache: BoundedLruCache<BlockCacheKey, ID2D1CommandList>,
}

#[cfg(windows)]
impl Default for WindowsRendererCaches {
    fn default() -> Self {
        Self {
            text_format_cache: BoundedLruCache::with_capacity(TEXT_FORMAT_CACHE_CAP),
            layout_cache: LayoutCache::default(),
            line_cache: BoundedLruCache::with_capacity(LINE_CACHE_CAP),
            block_cache: BoundedLruCache::with_capacity(BLOCK_CACHE_CAP),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn selected_lru_caps_hold_at_capacity_and_capacity_plus_one() {
        assert_eq!(
            [
                super::TEXT_FORMAT_CACHE_CAP,
                super::LAYOUT_CACHE_CAP,
                super::LINE_CACHE_CAP,
                super::BLOCK_CACHE_CAP
            ],
            [16, 64, 256, 128],
        );
        for capacity in [16, 64, 256, 128] {
            let mut cache = super::BoundedLruCache::with_capacity(capacity);
            for key in 0..capacity {
                cache.insert(key, key);
            }
            assert_eq!(cache.len(), capacity);
            assert_eq!(cache.get(&0), Some(&0));
            cache.insert(capacity, capacity);
            assert_eq!(cache.len(), capacity);
            assert_eq!(cache.get(&1), None);
            assert_eq!(cache.get(&0), Some(&0));
            assert_eq!(cache.get(&capacity), Some(&capacity));
        }
    }
    use super::{
        CachedBlockLayoutTemplate, CachedLineLayoutTemplate, LayoutCache,
        LAYOUT_CACHE_CAP,
    };
    use crate::renderer::{
        BlockBounds, CaptionBlockVariant, FontResolver, LayoutCacheKey, LineRole,
        TextStyleDescriptor, TextStyleKey, VisualBounds,
    };

    fn style_key(language: &str) -> TextStyleKey {
        FontResolver::with_bundle_available()
            .resolve(Some(language), "漢字")
            .style_key()
    }

    fn style_descriptor(language: &str) -> TextStyleDescriptor {
        let style = FontResolver::with_bundle_available().resolve(Some(language), "漢字");
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

    fn layout_key(seed: usize) -> LayoutCacheKey {
        LayoutCacheKey {
            primary_text: format!("primary {seed}"),
            secondary_text: format!("secondary {seed}"),
            primary_style_key: style_key("ko"),
            secondary_style_key: style_key("ja"),
            channel: None,
            speaker_style: crate::renderer::SpeakerStyle::Gray,
            block_variant: CaptionBlockVariant::Finalized,
            secondary_enabled: true,
            secondary_reserved: true,
            primary_font_size_key: 12_800,
            secondary_font_size_key: 7_936,
            content_width_key: 3_200,
            text_scale_key: 100,
        }
    }

    fn layout_template(seed: usize) -> CachedBlockLayoutTemplate {
        CachedBlockLayoutTemplate {
            primary_lines: vec![CachedLineLayoutTemplate {
                text: format!("primary {seed}"),
                role: LineRole::Primary,
                style_key: style_key("ko"),
                style: style_descriptor("ko"),
                width_px: 100.0,
                origin_x: 10.0,
                origin_y: 20.0,
                font_size_px: 128.0,
                visual_bounds: VisualBounds::new(0.0, 0.0, 100.0, 140.0),
            }],
            secondary_line: None,
            secondary_reserved: true,
            bounds: BlockBounds::new(0.0, 0.0, 200.0, 240.0),
            visual_bounds: VisualBounds::new(0.0, 0.0, 200.0, 240.0),
            content_width_px: 180.0,
            truncated_primary: false,
            truncated_secondary: false,
        }
    }

    #[test]
    fn layout_cache_uses_default_cap_and_evicts_the_oldest_unused_entry() {
        let mut cache = LayoutCache::default();

        for seed in 0..=LAYOUT_CACHE_CAP {
            cache.insert(layout_key(seed), layout_template(seed));
        }

        assert_eq!(cache.len(), LAYOUT_CACHE_CAP);
        assert!(cache.get(&layout_key(0)).is_none());
        assert!(cache.get(&layout_key(1)).is_some());
        assert!(cache.get(&layout_key(LAYOUT_CACHE_CAP)).is_some());
    }
}
