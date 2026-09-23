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
    #[cfg_attr(not(test), allow(dead_code))]
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
#[derive(Debug, Clone)]
pub(crate) struct CachedLineVisual {
    pub command_list: ID2D1CommandList,
    pub visual_bounds: VisualBounds,
}

#[cfg(windows)]
#[derive(Debug, Clone)]
pub(crate) struct CachedBlockVisual {
    pub command_list: ID2D1CommandList,
    #[allow(dead_code)]
    pub visual_bounds: VisualBounds,
}

#[cfg(windows)]
#[derive(Debug)]
pub(crate) struct WindowsRendererCaches {
    pub text_format_cache: BoundedLruCache<(TextStyleKey, u32), IDWriteTextFormat>,
    #[allow(dead_code)]
    pub layout_cache: LayoutCache,
    pub line_cache: BoundedLruCache<LineCacheKey, CachedLineVisual>,
    pub block_cache: BoundedLruCache<BlockCacheKey, CachedBlockVisual>,
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
        BoundedLruCache, CachedBlockLayoutTemplate, CachedLineLayoutTemplate, LayoutCache,
        LAYOUT_CACHE_CAP, LINE_CACHE_CAP,
    };
    use crate::renderer::{
        BlockBounds, BlockCacheKey, BundledFaceId, CaptionBlockVariant, FontLanguageBucket,
        FontResolver, FontSource, LayoutCacheKey, LineCacheKey, LineRole, TextStyleDescriptor,
        TextStyleKey, VisualBounds,
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
            block_variant: CaptionBlockVariant::Finalized,
            secondary_enabled: true,
            secondary_reserved: true,
            primary_font_size_key: 12_800,
            secondary_font_size_key: 7_936,
            content_width_key: 3_200,
            text_scale_key: 100,
        }
    }

    fn line_key(style_key: TextStyleKey) -> LineCacheKey {
        LineCacheKey {
            text: "漢字".into(),
            role: LineRole::Primary,
            style_key,
            channel: None,
            block_variant: CaptionBlockVariant::Finalized,
            font_size_key: 13_200,
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
    fn bounded_lru_cache_evicts_the_oldest_unused_entry_after_capacity_is_exceeded() {
        let mut cache = BoundedLruCache::with_capacity(2);

        cache.insert("old", 1);
        cache.insert("middle", 2);
        cache.insert("new", 3);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"old"), None);
        assert_eq!(cache.get(&"middle"), Some(&2));
        assert_eq!(cache.get(&"new"), Some(&3));
    }

    #[test]
    fn bounded_lru_cache_hit_updates_recency_before_eviction() {
        let mut cache = BoundedLruCache::with_capacity(2);

        cache.insert("old-but-used", 1);
        cache.insert("middle", 2);
        assert_eq!(cache.get(&"old-but-used"), Some(&1));
        cache.insert("new", 3);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"old-but-used"), Some(&1));
        assert_eq!(cache.get(&"middle"), None);
        assert_eq!(cache.get(&"new"), Some(&3));
    }

    #[test]
    fn text_format_cache_keys_keep_same_text_and_size_separate_per_cjk_style() {
        let font_size_key = 13_200;
        let cases = [
            ("ko", FontLanguageBucket::CjkKo),
            ("ja", FontLanguageBucket::CjkJa),
            ("zh-Hans", FontLanguageBucket::CjkZhHans),
            ("zh-Hant", FontLanguageBucket::CjkZhHant),
        ];
        let mut cache = BoundedLruCache::with_capacity(cases.len());

        for (index, (language, bucket)) in cases.into_iter().enumerate() {
            let style_key = style_key(language);
            assert_eq!(style_key.bucket, bucket);
            cache.insert((style_key, font_size_key), index);
        }

        assert_eq!(cache.len(), cases.len());
        for (index, (language, _)) in cases.into_iter().enumerate() {
            assert_eq!(
                cache.get(&(style_key(language), font_size_key)),
                Some(&index)
            );
        }
    }

    #[test]
    fn bundled_kr_jp_sc_tc_style_keys_have_distinct_face_identities() {
        let cases = [
            ("ko", BundledFaceId::NotoCjkKrMedium),
            ("ja", BundledFaceId::NotoCjkJpMedium),
            ("zh-Hans", BundledFaceId::NotoCjkScMedium),
            ("zh-Hant", BundledFaceId::NotoCjkTcMedium),
        ];
        let mut keys = std::collections::HashSet::new();

        for (language, face) in cases {
            let style_key = style_key(language);
            assert_eq!(style_key.source, FontSource::BundledNotoCjkMedium);
            assert_eq!(style_key.bundled_face, Some(face));
            keys.insert(style_key);
        }

        assert_eq!(keys.len(), cases.len());
    }

    #[test]
    fn layout_and_block_cache_keys_include_primary_and_secondary_style_identity() {
        let base = layout_key(7);
        let changed_primary = LayoutCacheKey {
            primary_style_key: style_key("zh-Hans"),
            ..base.clone()
        };
        let changed_secondary = LayoutCacheKey {
            secondary_style_key: style_key("zh-Hant"),
            ..base.clone()
        };

        assert_ne!(base, changed_primary);
        assert_ne!(base, changed_secondary);
        assert_ne!(
            (BlockCacheKey {
                id: "same-block".into(),
                layout: base.clone(),
            }),
            (BlockCacheKey {
                id: "same-block".into(),
                layout: changed_secondary,
            })
        );
    }

    #[test]
    fn line_cache_key_includes_line_style_identity() {
        let ko = line_key(style_key("ko"));
        let ja = LineCacheKey {
            style_key: style_key("ja"),
            ..ko.clone()
        };

        assert_ne!(ko, ja);
    }

    #[test]
    fn bounded_lru_cache_stress_keeps_size_at_or_below_capacity() {
        let mut cache = BoundedLruCache::with_capacity(LINE_CACHE_CAP);

        for seed in 0..(LINE_CACHE_CAP * 3) {
            cache.insert(seed, seed * 2);
            if seed % 3 == 0 {
                let _ = cache.get(&seed.saturating_sub(1));
            }
            assert!(cache.len() <= LINE_CACHE_CAP);
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
