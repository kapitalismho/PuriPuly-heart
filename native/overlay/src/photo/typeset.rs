use std::collections::HashMap;
use std::path::Path;

use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::DirectWrite::{
    DWriteCreateFactory, IDWriteFactory, IDWriteFactory2, IDWriteFontFallback, IDWriteTextFormat,
    IDWriteTextFormat1, IDWriteTextLayout, IDWriteTextLayout2, DWRITE_FACTORY_TYPE_SHARED,
    DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT_MEDIUM,
    DWRITE_TEXT_METRICS, DWRITE_WORD_WRAPPING_NO_WRAP,
};

use super::processing::{PreparedRegion, Rect, WorkingPhoto};
use crate::renderer::{
    runtime_bundled_font_path, FontResolver, FontSource, ResolvedFontStyle,
    WindowsBundledFontCollection,
};

mod layout;
mod measurement;

pub use measurement::{measure_source, SourceMeasurement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutKind {
    Lines,
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug)]
pub struct PositionedTextRun {
    pub layout: IDWriteTextLayout,
    pub text: String,
    pub size: f32,
    pub origin: [f32; 2],
    pub ink: Rect,
}

#[derive(Clone, Debug)]
pub struct OutlineStyle {
    pub color: [u8; 3],
    pub width: f32,
}

#[derive(Clone, Debug)]
pub struct PlateStyle {
    pub rect: Rect,
    pub color: [u8; 3],
    pub alpha: f32,
}

#[derive(Clone, Debug)]
pub struct TextStyle {
    pub fill: [u8; 3],
    pub outline: Option<OutlineStyle>,
    pub plate: Option<PlateStyle>,
}

#[derive(Clone, Debug)]
pub struct TypesetBlock {
    pub runs: Vec<PositionedTextRun>,
    pub collision_boxes: Vec<Rect>,
    pub angle_degrees: f32,
    pub center: [f32; 2],
    pub style: TextStyle,
    pub selected_size: f32,
    pub source_cap: f32,
    pub floor: f32,
    pub collision: bool,
    pub overlap_area: f32,
    pub region_overflow: bool,
    pub kind: LayoutKind,
    pub source_vertical: bool,
}

#[derive(Clone, Debug)]
pub struct FontMetrics {
    pub family: String,
    pub source: FontSource,
    pub hangul_ink_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TextMetrics {
    pub advance: f32,
    pub ink: Rect,
}

pub struct PhotoTypesetter {
    metrics: DirectWriteMetrics,
    placed: Vec<Rect>,
}

impl PhotoTypesetter {
    pub fn new() -> Result<Self, String> {
        let path = runtime_bundled_font_path().ok();
        Self::create(path.as_deref(), false)
    }

    pub fn with_font_path(path: &Path) -> Result<Self, String> {
        Self::create(Some(path), true)
    }

    fn create(path: Option<&Path>, required: bool) -> Result<Self, String> {
        Ok(Self {
            metrics: DirectWriteMetrics::new(path, required)?,
            placed: Vec::new(),
        })
    }

    pub fn reset(&mut self) {
        self.placed.clear();
    }

    pub fn font_metrics(&self) -> &FontMetrics {
        &self.metrics.font_metrics
    }

    pub fn measure_text(&mut self, text: &str, size: f32) -> Result<TextMetrics, String> {
        self.metrics.measure(text, size)
    }

    pub fn placed_boxes(&self) -> &[Rect] {
        &self.placed
    }

    pub fn typeset(
        &mut self,
        photo: &WorkingPhoto,
        region: &PreparedRegion,
        translation: &str,
    ) -> Result<TypesetBlock, String> {
        self.metrics.cache.clear();
        self.metrics.formats.clear();
        self.metrics.layouts.clear();
        let height = photo.pixels.width().min(photo.pixels.height()) as f32;
        let scale = height / 1440.0;
        let floor = (8.0 * scale).max(0.015 * height);
        let measured = measure_source(
            &region.core,
            [region.roi.x as f32, region.roi.y as f32],
            scale,
        );
        let info = measured.unwrap_or_else(|| SourceMeasurement {
            angle_degrees: 0.0,
            direction: if region.bbox.height() > 1.6 * region.bbox.width() {
                Direction::Vertical
            } else {
                Direction::Horizontal
            },
            extent: region.bbox,
            slots: Vec::new(),
            glyph_size: region.glyph_size,
            center: [
                region.roi.x as f32 + region.roi.width as f32 / 2.0,
                region.roi.y as f32 + region.roi.height as f32 / 2.0,
            ],
        });
        let cap = info.glyph_size / self.metrics.font_metrics.hangul_ink_ratio;
        let text = translation.trim();
        if text.is_empty() {
            return Ok(TypesetBlock {
                runs: Vec::new(),
                collision_boxes: Vec::new(),
                angle_degrees: info.angle_degrees,
                center: info.center,
                style: style(region, floor, scale),
                selected_size: 0.0,
                source_cap: cap,
                floor,
                collision: false,
                overlap_area: 0.0,
                region_overflow: false,
                kind: LayoutKind::Horizontal,
                source_vertical: info.direction == Direction::Vertical,
            });
        }
        let best = layout::choose(
            &mut self.metrics,
            text,
            &info,
            region.bbox,
            floor,
            cap,
            &self.placed,
            rect(0.0, 0.0, photo.pixels.width() as f32, height),
        )?;
        let collision_boxes = best.boxes(&info);
        let overlap_area = overlap_sum(&collision_boxes, &self.placed);
        let collision = hits(&collision_boxes, &self.placed) || best.overflow;
        let mut runs = Vec::with_capacity(best.runs.len());
        for run in &best.runs {
            runs.push(PositionedTextRun {
                layout: self.metrics.create_layout(&run.text, run.size)?,
                text: run.text.clone(),
                size: run.size,
                origin: run.origin,
                ink: run.ink,
            });
        }
        self.placed.extend_from_slice(&collision_boxes);
        Ok(TypesetBlock {
            runs,
            collision_boxes,
            angle_degrees: info.angle_degrees,
            center: info.center,
            style: style(region, best.size, scale),
            selected_size: best.size,
            source_cap: cap,
            floor,
            collision,
            overlap_area,
            region_overflow: best.overflow,
            kind: best.kind,
            source_vertical: info.direction == Direction::Vertical,
        })
    }
}

struct DirectWriteMetrics {
    factory: IDWriteFactory,
    fallback: IDWriteFontFallback,
    collection: Option<WindowsBundledFontCollection>,
    resolved: ResolvedFontStyle,
    family: String,
    normalized_format: IDWriteTextFormat,
    cache: HashMap<String, TextMetrics>,
    formats: HashMap<u32, IDWriteTextFormat>,
    layouts: HashMap<u32, HashMap<String, IDWriteTextLayout>>,
    font_metrics: FontMetrics,
}

impl DirectWriteMetrics {
    fn new(path: Option<&Path>, required: bool) -> Result<Self, String> {
        let factory: IDWriteFactory =
            unsafe { DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED) }.map_err(dwrite_error)?;
        let collection = path
            .and_then(|path| WindowsBundledFontCollection::load_with_factory(&factory, path).ok());
        if required && collection.is_none() {
            return Err("photo bundled font loading failed".into());
        }
        let resolver = if collection.is_some() {
            FontResolver::with_bundle_available()
        } else {
            FontResolver::with_bundle_unavailable("photo bundled font unavailable")
        };
        let resolved = resolver.resolve(Some("ko"), "한글");
        let mut family = resolved.family_name.to_string();
        if collection.is_none() {
            let mut system = None;
            unsafe { factory.GetSystemFontCollection(&mut system, false) }.map_err(dwrite_error)?;
            let system = system.ok_or("photo system font collection unavailable")?;
            for candidate in resolved
                .system_fallback_families()
                .iter()
                .copied()
                .filter(|family| *family != "DirectWrite system fallback")
            {
                let wide = utf16(candidate);
                let mut index = 0;
                let mut exists = false.into();
                unsafe {
                    system.FindFamilyName(PCWSTR::from_raw(wide.as_ptr()), &mut index, &mut exists)
                }
                .map_err(dwrite_error)?;
                if exists.as_bool() {
                    family = candidate.to_owned();
                    break;
                }
            }
        }
        let factory2: IDWriteFactory2 = factory.cast().map_err(dwrite_error)?;
        let fallback = unsafe { factory2.GetSystemFontFallback() }.map_err(dwrite_error)?;
        let normalized_format = create_format(
            &factory,
            &fallback,
            collection.as_ref(),
            &family,
            &resolved.locale,
            200.0,
        )?;
        let mut result = Self {
            factory,
            fallback,
            collection,
            resolved,
            family: family.clone(),
            normalized_format,
            cache: HashMap::new(),
            formats: HashMap::new(),
            layouts: HashMap::new(),
            font_metrics: FontMetrics {
                family,
                source: FontSource::SystemFont,
                hangul_ink_ratio: 1.0,
            },
        };
        result.font_metrics.source = result.resolved.source;
        result.font_metrics.hangul_ink_ratio = result.measure("한글가나다라", 1.0)?.ink.height();
        if result.font_metrics.hangul_ink_ratio <= 0.0 {
            return Err("photo Hangul ink measurement is empty".into());
        }
        result.cache.clear();
        Ok(result)
    }

    fn measure(&mut self, text: &str, size: f32) -> Result<TextMetrics, String> {
        let normalized = if let Some(metrics) = self.cache.get(text) {
            *metrics
        } else {
            let layout = unsafe {
                self.factory.CreateTextLayout(
                    &text.encode_utf16().collect::<Vec<_>>(),
                    &self.normalized_format,
                    32768.0,
                    32768.0,
                )
            }
            .map_err(dwrite_error)?;
            let layout2: IDWriteTextLayout2 = layout.cast().map_err(dwrite_error)?;
            unsafe { layout2.SetFontFallback(&self.fallback) }.map_err(dwrite_error)?;
            let mut metrics = DWRITE_TEXT_METRICS::default();
            unsafe { layout.GetMetrics(&mut metrics) }.map_err(dwrite_error)?;
            let overhang = unsafe { layout.GetOverhangMetrics() }.map_err(dwrite_error)?;
            let metrics = TextMetrics {
                advance: metrics.widthIncludingTrailingWhitespace / 200.0,
                ink: rect(
                    -overhang.left / 200.0,
                    -overhang.top / 200.0,
                    (32768.0 + overhang.right) / 200.0,
                    (32768.0 + overhang.bottom) / 200.0,
                ),
            };
            self.cache.insert(text.to_owned(), metrics);
            metrics
        };
        Ok(TextMetrics {
            advance: normalized.advance * size,
            ink: rect(
                normalized.ink.left * size,
                normalized.ink.top * size,
                normalized.ink.right * size,
                normalized.ink.bottom * size,
            ),
        })
    }

    fn create_layout(&mut self, text: &str, size: f32) -> Result<IDWriteTextLayout, String> {
        if let Some(layout) = self
            .layouts
            .get(&size.to_bits())
            .and_then(|layouts| layouts.get(text))
        {
            return Ok(layout.clone());
        }
        let format = if let Some(format) = self.formats.get(&size.to_bits()) {
            format.clone()
        } else {
            let format = create_format(
                &self.factory,
                &self.fallback,
                self.collection.as_ref(),
                &self.family,
                &self.resolved.locale,
                size,
            )?;
            self.formats.insert(size.to_bits(), format.clone());
            format
        };
        let layout = unsafe {
            self.factory.CreateTextLayout(
                &text.encode_utf16().collect::<Vec<_>>(),
                &format,
                32768.0,
                32768.0,
            )
        }
        .map_err(dwrite_error)?;
        let layout2: IDWriteTextLayout2 = layout.cast().map_err(dwrite_error)?;
        unsafe { layout2.SetFontFallback(&self.fallback) }.map_err(dwrite_error)?;
        self.layouts
            .entry(size.to_bits())
            .or_default()
            .insert(text.to_owned(), layout.clone());
        Ok(layout)
    }
}

fn create_format(
    factory: &IDWriteFactory,
    fallback: &IDWriteFontFallback,
    collection: Option<&WindowsBundledFontCollection>,
    family: &str,
    locale: &str,
    size: f32,
) -> Result<IDWriteTextFormat, String> {
    let family = utf16(family);
    let locale = utf16(locale);
    let format = unsafe {
        factory.CreateTextFormat(
            PCWSTR::from_raw(family.as_ptr()),
            collection.map(WindowsBundledFontCollection::collection),
            DWRITE_FONT_WEIGHT_MEDIUM,
            DWRITE_FONT_STYLE_NORMAL,
            DWRITE_FONT_STRETCH_NORMAL,
            size,
            PCWSTR::from_raw(locale.as_ptr()),
        )
    }
    .map_err(dwrite_error)?;
    unsafe { format.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP) }.map_err(dwrite_error)?;
    let format1: IDWriteTextFormat1 = format.cast().map_err(dwrite_error)?;
    unsafe { format1.SetFontFallback(fallback) }.map_err(dwrite_error)?;
    Ok(format)
}

fn utf16(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}
fn dwrite_error(error: windows::core::Error) -> String {
    format!("photo DirectWrite failed ({:?})", error.code())
}
fn rect(left: f32, top: f32, right: f32, bottom: f32) -> Rect {
    Rect {
        left,
        top,
        right,
        bottom,
    }
}
fn translated(bounds: Rect, x: f32, y: f32) -> Rect {
    rect(
        bounds.left + x,
        bounds.top + y,
        bounds.right + x,
        bounds.bottom + y,
    )
}
fn union(a: Rect, b: Rect) -> Rect {
    rect(
        a.left.min(b.left),
        a.top.min(b.top),
        a.right.max(b.right),
        a.bottom.max(b.bottom),
    )
}
fn contains(outer: Rect, inner: Rect) -> bool {
    inner.left >= outer.left
        && inner.top >= outer.top
        && inner.right <= outer.right
        && inner.bottom <= outer.bottom
}
fn overlap(a: Rect, b: Rect) -> f32 {
    (a.right.min(b.right) - a.left.max(b.left)).max(0.0)
        * (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0)
}
fn overlap_sum(boxes: &[Rect], placed: &[Rect]) -> f32 {
    boxes
        .iter()
        .flat_map(|&b| placed.iter().map(move |&p| overlap(b, p)))
        .sum()
}
fn hits(boxes: &[Rect], placed: &[Rect]) -> bool {
    boxes.iter().any(|&b| {
        placed
            .iter()
            .any(|&p| overlap(b, p) > 0.02 * (b.width() * b.height()).max(1.0))
    })
}

fn image_bounds(bounds: Rect, info: &SourceMeasurement) -> Rect {
    if info.angle_degrees == 0.0 {
        return bounds;
    }
    let (s, c) = info.angle_degrees.to_radians().sin_cos();
    let mut result = rect(
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for [x, y] in [
        [bounds.left, bounds.top],
        [bounds.right, bounds.top],
        [bounds.right, bounds.bottom],
        [bounds.left, bounds.bottom],
    ] {
        let dx = x - info.center[0];
        let dy = y - info.center[1];
        let px = c * dx - s * dy + info.center[0];
        let py = s * dx + c * dy + info.center[1];
        result.left = result.left.min(px);
        result.right = result.right.max(px);
        result.top = result.top.min(py);
        result.bottom = result.bottom.max(py);
    }
    result
}

fn style(region: &PreparedRegion, size: f32, scale: f32) -> TextStyle {
    if region.plate {
        let padding = 8.0 * scale;
        return TextStyle {
            fill: [245, 245, 240],
            outline: None,
            plate: Some(PlateStyle {
                rect: rect(
                    region.bbox.left - padding,
                    region.bbox.top - padding,
                    region.bbox.right + padding,
                    region.bbox.bottom + padding,
                ),
                color: [16, 20, 28],
                alpha: 238.0 / 255.0,
            }),
        };
    }
    let luminance = region
        .fill
        .iter()
        .zip([0.2126, 0.7152, 0.0722])
        .map(|(&v, weight)| {
            let c = v as f32 / 255.0;
            weight
                * if c <= 0.04045 {
                    c / 12.92
                } else {
                    ((c + 0.055) / 1.055).powf(2.4)
                }
        })
        .sum::<f32>();
    let color = if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) {
        [0, 0, 0]
    } else {
        [255, 255, 255]
    };
    TextStyle {
        fill: region.fill,
        outline: Some(OutlineStyle {
            color,
            width: (0.08 * size).round().max(1.0),
        }),
        plate: None,
    }
}

#[cfg(test)]
mod tests;
