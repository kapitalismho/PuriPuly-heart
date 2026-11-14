use image::{imageops, GrayImage, Luma, RgbImage, Rgba, RgbaImage};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

use super::telea;
pub use super::telea::Workspace;

pub const DEFAULT_MAX_DIMENSIONS: [u32; 2] = [1920, 1080];
pub const MASK_MODEL_SHA256: &str =
    "bfe95d72db1eea72101074293c8d27b77eb87663fa48a23e8258e59249510bcf";
const MODEL_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

impl Rect {
    pub fn width(self) -> f32 {
        self.right - self.left
    }
    pub fn height(self) -> f32 {
        self.bottom - self.top
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct WorkingPhoto {
    pub pixels: Arc<RgbImage>,
    pub scale_x: f32,
    pub scale_y: f32,
    pub reference_scale: f32,
}

impl WorkingPhoto {
    pub fn new(original: Arc<RgbImage>, max_dimensions: [u32; 2]) -> Result<Self, String> {
        let [max_width, max_height] = if (original.height() > original.width()) == (max_dimensions[0] > max_dimensions[1]) {
            [max_dimensions[1], max_dimensions[0]]
        } else {
            max_dimensions
        };
        if original.width() == 0 || original.height() == 0 || max_width == 0 || max_height == 0 {
            return Err("photo_invalid_dimensions".into());
        }
        let scale = (f64::from(max_width) / f64::from(original.width()))
            .min(f64::from(max_height) / f64::from(original.height()));
        let width = (f64::from(original.width()) * scale)
            .round_ties_even()
            .clamp(1.0, f64::from(max_width)) as u32;
        let height = (f64::from(original.height()) * scale)
            .round_ties_even()
            .clamp(1.0, f64::from(max_height)) as u32;
        let scale_x = width as f32 / original.width() as f32;
        let scale_y = height as f32 / original.height() as f32;
        let pixels = if original.dimensions() == (width, height) {
            original
        } else {
            Arc::new(imageops::resize(
                &*original,
                width,
                height,
                imageops::FilterType::Lanczos3,
            ))
        };
        Ok(Self {
            pixels,
            scale_x,
            scale_y,
            reference_scale: width.min(height) as f32 / 1440.0,
        })
    }

    pub fn map_bbox(&self, bbox: [f64; 4]) -> Rect {
        Rect {
            left: (bbox[0] * f64::from(self.scale_x)) as f32,
            top: (bbox[1] * f64::from(self.scale_y)) as f32,
            right: (bbox[2] * f64::from(self.scale_x)) as f32,
            bottom: (bbox[3] * f64::from(self.scale_y)) as f32,
        }
    }
}

pub struct ErasePatch {
    pub rect: PixelRect,
    pub pixels: RgbaImage,
}

pub struct PreparedRegion {
    pub bbox: Rect,
    pub roi: PixelRect,
    pub core: GrayImage,
    pub recovered: GrayImage,
    pub glyph_size: f32,
    pub source_lines: u32,
    pub fill: [u8; 3],
    pub erase: Option<ErasePatch>,
    pub mask_ms: f64,
    pub erase_ms: f64,
}

pub fn keep_original(source: &str, translation: &str) -> bool {
    fn normalize(text: &str) -> String {
        text.nfkc()
            .filter(|ch| {
                ch.is_alphanumeric() && !unicode_normalization::char::is_combining_mark(*ch)
            })
            .collect::<String>()
            .to_lowercase()
    }
    normalize(source) == normalize(translation)
}

pub fn candidate_region(photo: &WorkingPhoto, bbox: Rect) -> Result<PixelRect, String> {
    let bbox = clipped_bbox(photo, bbox)?;
    Ok(pixel_bounds(
        photo.pixels.dimensions(),
        bbox,
        80.0 * photo.reference_scale,
    ))
}

fn clipped_bbox(photo: &WorkingPhoto, bbox: Rect) -> Result<Rect, String> {
    if ![bbox.left, bbox.top, bbox.right, bbox.bottom]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err("photo_invalid_bbox".into());
    }
    let bbox = Rect {
        left: bbox.left.max(0.0),
        top: bbox.top.max(0.0),
        right: bbox.right.min(photo.pixels.width() as f32),
        bottom: bbox.bottom.min(photo.pixels.height() as f32),
    };
    if bbox.width() <= 0.0 || bbox.height() <= 0.0 {
        return Err("photo_invalid_bbox".into());
    }
    Ok(bbox)
}

fn pixel_bounds(dimensions: (u32, u32), bbox: Rect, padding: f32) -> PixelRect {
    let x = (bbox.left - padding)
        .floor()
        .clamp(0.0, dimensions.0 as f32) as u32;
    let y = (bbox.top - padding).floor().clamp(0.0, dimensions.1 as f32) as u32;
    let right = (bbox.right + padding)
        .ceil()
        .clamp(0.0, dimensions.0 as f32) as u32;
    let bottom = (bbox.bottom + padding)
        .ceil()
        .clamp(0.0, dimensions.1 as f32) as u32;
    PixelRect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    }
}

fn geometry(
    photo: &WorkingPhoto,
    bbox: Rect,
    source_line_count: u32,
) -> Result<(Rect, PixelRect, f32), String> {
    if source_line_count == 0 {
        return Err("photo_source_lines_required".into());
    }
    let bbox = clipped_bbox(photo, bbox)?;
    let extent = if bbox.height() > 1.6 * bbox.width() {
        bbox.width()
    } else {
        bbox.height()
    };
    let glyph = (extent / source_line_count as f32).max(8.0 * photo.reference_scale);
    let padding = (RECOVERY_REACH * glyph)
        .round_ties_even()
        .clamp(6.0 * photo.reference_scale, 80.0 * photo.reference_scale)
        .round_ties_even()
        .max(1.0);
    Ok((
        bbox,
        pixel_bounds(photo.pixels.dimensions(), bbox, padding),
        glyph,
    ))
}

fn tiles(long: u32, short: u32) -> Vec<(u32, u32)> {
    if long <= short * 2 {
        return vec![(0, long)];
    }
    let size = short * 2;
    let step = (size as f64 * 0.85) as u32;
    let mut starts: Vec<_> = (0..=long - size).step_by(step.max(1) as usize).collect();
    if starts.last().copied().unwrap_or(0) + size < long {
        starts.push(long - size);
    }
    starts
        .into_iter()
        .map(|start| (start, start + size))
        .collect()
}

fn linear_coordinate(destination: u32, source_size: u32, target_size: u32) -> (u32, u32, f32) {
    let position = ((f64::from(destination) + 0.5) * f64::from(source_size)
        / f64::from(target_size)
        - 0.5) as f32;
    let lower = position.floor() as i64;
    (
        lower.clamp(0, i64::from(source_size) - 1) as u32,
        (lower + 1).clamp(0, i64::from(source_size) - 1) as u32,
        position - position.floor(),
    )
}

fn resize_tile(source: &RgbImage, rect: PixelRect, input: &mut [u8]) {
    let xs: [(u32, u32, f32); MODEL_SIZE] =
        std::array::from_fn(|x| linear_coordinate(x as u32, rect.width, MODEL_SIZE as u32));
    for y in 0..MODEL_SIZE as u32 {
        let (y0, y1, fy) = linear_coordinate(y, rect.height, MODEL_SIZE as u32);
        let by = (fy * 2048.0).round_ties_even() as i32;
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            let bx = (fx * 2048.0).round_ties_even() as i32;
            let p00 = source.get_pixel(rect.x + x0, rect.y + y0).0;
            let p01 = source.get_pixel(rect.x + x1, rect.y + y0).0;
            let p10 = source.get_pixel(rect.x + x0, rect.y + y1).0;
            let p11 = source.get_pixel(rect.x + x1, rect.y + y1).0;
            for channel in 0..3 {
                let top = i32::from(p00[channel]) * (2048 - bx) + i32::from(p01[channel]) * bx;
                let bottom = i32::from(p10[channel]) * (2048 - bx) + i32::from(p11[channel]) * bx;
                let value =
                    ((((2048 - by) * (top >> 4)) >> 16) + ((by * (bottom >> 4)) >> 16) + 2) >> 2;
                input[channel * MODEL_SIZE * MODEL_SIZE + y as usize * MODEL_SIZE + x] =
                    value as u8;
            }
        }
    }
}

fn merge_tile_mask(
    core: &mut GrayImage,
    roi: PixelRect,
    tile: PixelRect,
    bbox: Rect,
    values: &[f32],
) {
    let xs: Vec<_> = (0..tile.width)
        .map(|x| linear_coordinate(x, MODEL_SIZE as u32, tile.width))
        .collect();
    for y in 0..tile.height {
        let local_y = tile.y - roi.y + y;
        let photo_y = local_y as f32 + roi.y as f32 + 0.5;
        if photo_y < bbox.top || photo_y >= bbox.bottom {
            continue;
        }
        let (y0, y1, fy) = linear_coordinate(y, MODEL_SIZE as u32, tile.height);
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            let local_x = tile.x - roi.x + x as u32;
            let photo_x = local_x as f32 + roi.x as f32 + 0.5;
            if photo_x < bbox.left || photo_x >= bbox.right {
                continue;
            }
            let pixel = &mut core.as_mut()[(local_y * roi.width + local_x) as usize];
            if *pixel != 0 {
                continue;
            }
            let a = values[y0 as usize * MODEL_SIZE + x0 as usize] * (1.0 - fx)
                + values[y0 as usize * MODEL_SIZE + x1 as usize] * fx;
            let b = values[y1 as usize * MODEL_SIZE + x0 as usize] * (1.0 - fx)
                + values[y1 as usize * MODEL_SIZE + x1 as usize] * fx;
            if a * (1.0 - fy) + b * fy > 0.3 {
                *pixel = 255;
            }
        }
    }
}

pub struct MaskProcessor {
    session: Session,
    input: Vec<u8>,
}

impl MaskProcessor {
    pub fn new(model_path: &Path, runtime_path: Option<&Path>) -> Result<Self, String> {
        if !model_path.is_file() {
            return Err("mask_model_unavailable".into());
        }
        if let Some(path) = runtime_path {
            if !path.is_file() {
                return Err("mask_runtime_unavailable".into());
            }
            ort::init_from(path.to_string_lossy())
                .with_name("photo-mask")
                .commit()
                .map_err(|e| e.to_string())?;
        }
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_intra_threads(2)
            .map_err(|e| e.to_string())?
            .with_intra_op_spinning(false)
            .map_err(|e| e.to_string())?
            .with_inter_threads(1)
            .map_err(|e| e.to_string())?
            .with_parallel_execution(false)
            .map_err(|e| e.to_string())?
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;
        let valid_input = session.inputs.len() == 1
            && session.inputs[0].name == "image"
            && session.inputs[0].input_type.tensor_type() == Some(TensorElementType::Uint8)
            && session.inputs[0]
                .input_type
                .tensor_shape()
                .is_some_and(|s| s.as_ref() == [1, 3, 256, 256]);
        let valid_output = session.outputs.len() == 1
            && session.outputs[0].name == "prob"
            && session.outputs[0].output_type.tensor_type() == Some(TensorElementType::Float32)
            && session.outputs[0]
                .output_type
                .tensor_shape()
                .is_some_and(|s| s.as_ref() == [1, 1, 256, 256]);
        if !valid_input || !valid_output {
            return Err("mask_selected_export_required".into());
        }
        Ok(Self {
            session,
            input: vec![0; MODEL_SIZE * MODEL_SIZE * 3],
        })
    }

    pub fn prepare(
        &mut self,
        photo: &WorkingPhoto,
        bbox: Rect,
        source_line_count: u32,
        erase_source: &RgbImage,
        workspace: &mut telea::Workspace,
    ) -> Result<PreparedRegion, String> {
        let mask_start = Instant::now();
        if erase_source.dimensions() != photo.pixels.dimensions() {
            return Err("photo_erase_source_dimensions".into());
        }
        let (bbox, roi, glyph_size) = geometry(photo, bbox, source_line_count)?;
        let horizontal = roi.width >= roi.height;
        let mut foreground = GrayImage::new(roi.width, roi.height);
        let whole = Rect {
            left: roi.x as f32,
            top: roi.y as f32,
            right: (roi.x + roi.width) as f32,
            bottom: (roi.y + roi.height) as f32,
        };
        for (start, end) in tiles(roi.width.max(roi.height), roi.width.min(roi.height)) {
            let tile = if horizontal {
                PixelRect {
                    x: roi.x + start,
                    y: roi.y,
                    width: end - start,
                    height: roi.height,
                }
            } else {
                PixelRect {
                    x: roi.x,
                    y: roi.y + start,
                    width: roi.width,
                    height: end - start,
                }
            };
            resize_tile(&photo.pixels, tile, &mut self.input);
            let input = TensorRef::from_array_view((
                [1usize, 3, MODEL_SIZE, MODEL_SIZE],
                self.input.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let outputs = self
                .session
                .run(ort::inputs!["image" => input])
                .map_err(|e| e.to_string())?;
            let (_, values) = outputs["prob"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            if values.len() != MODEL_SIZE * MODEL_SIZE
                || !values
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            {
                return Err("mask_invalid_probability".into());
            }
            merge_tile_mask(&mut foreground, roi, tile, whole, values);
        }
        let core = GrayImage::from_fn(roi.width, roi.height, |x, y| {
            Luma([if inside(roi, bbox, x, y) {
                foreground.get_pixel(x, y)[0]
            } else {
                0
            }])
        });
        let vertical = bbox.height() > 1.6 * bbox.width();
        let owned = owned_lines(&foreground, roi, bbox, glyph_size, vertical);
        let recovered = GrayImage::from_fn(roi.width, roi.height, |x, y| {
            Luma([if owned.get_pixel(x, y)[0] != 0 || core.get_pixel(x, y)[0] != 0 {
                255
            } else {
                0
            }])
        });
        let mask = dilate(
            &recovered,
            erase_radius(glyph_size, photo.reference_scale),
            true,
        );
        let (fill, _, _) = colors_and_leftover(photo, bbox, roi, &core, &mask, glyph_size);
        let mask_ms = mask_start.elapsed().as_secs_f64() * 1000.0;
        let erase_start = Instant::now();
        let erase = Some(erase_patch(photo, roi, &mask, erase_source, workspace)?);
        let erase_ms = erase_start.elapsed().as_secs_f64() * 1000.0;
        Ok(PreparedRegion {
            bbox,
            roi,
            core,
            recovered,
            glyph_size,
            source_lines: source_line_count,
            fill,
            erase,
            mask_ms,
            erase_ms,
        })
    }
}

const RECOVERY_REACH: f32 = 0.8;

fn inside(roi: PixelRect, bbox: Rect, x: u32, y: u32) -> bool {
    let px = x as f32 + roi.x as f32 + 0.5;
    let py = y as f32 + roi.y as f32 + 0.5;
    px >= bbox.left && px < bbox.right && py >= bbox.top && py < bbox.bottom
}

struct Part {
    bounds: [i64; 4],
    inside: u32,
    total: u32,
}

fn owned_lines(
    foreground: &GrayImage,
    roi: PixelRect,
    bbox: Rect,
    glyph: f32,
    vertical: bool,
) -> GrayImage {
    let (width, height) = foreground.dimensions();
    let mut labels = vec![u32::MAX; (width * height) as usize];
    let mut parts: Vec<Part> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..width * height {
        if labels[start as usize] != u32::MAX
            || foreground.as_raw()[start as usize] == 0
        {
            continue;
        }
        let label = parts.len() as u32;
        let mut part = Part {
            bounds: [i64::MAX, i64::MAX, i64::MIN, i64::MIN],
            inside: 0,
            total: 0,
        };
        labels[start as usize] = label;
        stack.push(start);
        while let Some(index) = stack.pop() {
            let (x, y) = (index % width, index / width);
            part.bounds[0] = part.bounds[0].min(i64::from(x));
            part.bounds[1] = part.bounds[1].min(i64::from(y));
            part.bounds[2] = part.bounds[2].max(i64::from(x) + 1);
            part.bounds[3] = part.bounds[3].max(i64::from(y) + 1);
            part.total += 1;
            part.inside += u32::from(inside(roi, bbox, x, y));
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (i64::from(x) + dx, i64::from(y) + dy);
                    if nx < 0 || ny < 0 || nx >= i64::from(width) || ny >= i64::from(height) {
                        continue;
                    }
                    let next = (ny as u32 * width + nx as u32) as usize;
                    if labels[next] == u32::MAX && foreground.as_raw()[next] != 0 {
                        labels[next] = label;
                        stack.push(next as u32);
                    }
                }
            }
        }
        parts.push(part);
    }
    let mut parent: Vec<usize> = (0..parts.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let (along, across) = if vertical { (1, 0) } else { (0, 1) };
    let reach = (0.8 * glyph) as i64;
    for i in 0..parts.len() {
        for j in i + 1..parts.len() {
            let (a, b) = (&parts[i].bounds, &parts[j].bounds);
            let gap = a[along].max(b[along]) - a[along + 2].min(b[along + 2]);
            let shared = a[across + 2].min(b[across + 2]) - a[across].max(b[across]);
            let thin = (a[across + 2] - a[across]).min(b[across + 2] - b[across]);
            if gap <= reach && shared * 5 >= thin * 2 {
                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                parent[ri] = rj;
            }
        }
    }
    let mut lines = vec![(0u32, 0u32); parts.len()];
    for i in 0..parts.len() {
        let r = root(&mut parent, i);
        lines[r].0 += parts[i].inside;
        lines[r].1 += parts[i].total;
    }
    let owned: Vec<bool> = (0..parts.len())
        .map(|i| {
            let (inside, total) = lines[root(&mut parent, i)];
            inside > 0 && inside * 2 >= total
        })
        .collect();
    GrayImage::from_fn(width, height, |x, y| {
        let label = labels[(y * width + x) as usize];
        Luma([if label != u32::MAX && owned[label as usize] {
            255
        } else {
            0
        }])
    })
}

#[inline]
fn erase_radius(glyph_size: f32, reference_scale: f32) -> u32 {
    (0.05 * glyph_size).round_ties_even().max(1.0) as u32 + reference_scale.ceil().max(1.0) as u32
}

fn dilate(core: &GrayImage, radius: u32, ellipse: bool) -> GrayImage {
    let mut output = GrayImage::new(core.width(), core.height());
    let radius = radius as i64;
    for dy in -radius..=radius {
        let reach = if ellipse {
            (radius as f64 * ((radius * radius - dy * dy) as f64 / (radius * radius) as f64).sqrt())
                .round_ties_even() as i64
        } else {
            radius
        };
        for y in 0..core.height() {
            let target_y = i64::from(y) + dy;
            if target_y < 0 || target_y >= i64::from(core.height()) {
                continue;
            }
            for x in 0..core.width() {
                if core.get_pixel(x, y)[0] == 0 {
                    continue;
                }
                let first = (i64::from(x) - reach).max(0) as u32;
                let last = (i64::from(x) + reach).min(i64::from(core.width()) - 1) as u32;
                for target_x in first..=last {
                    output.put_pixel(target_x, target_y as u32, Luma([255]));
                }
            }
        }
    }
    output
}

fn lab(rgb: [u8; 3]) -> [u8; 3] {
    static LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
        std::array::from_fn(|i| {
            let v = i as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
    });
    let [r, g, b] = rgb.map(|v| LINEAR[v as usize]);
    let f = |v: f32| {
        if v > 216.0 / 24389.0 {
            v.cbrt()
        } else {
            (24389.0 / 27.0 * v + 16.0) / 116.0
        }
    };
    let x = f((0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456);
    let y = f(0.212671 * r + 0.715160 * g + 0.072169 * b);
    let z = f((0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754);
    [
        ((116.0 * y - 16.0) * 255.0 / 100.0)
            .round_ties_even()
            .clamp(0.0, 255.0) as u8,
        (500.0 * (x - y) + 128.0)
            .round_ties_even()
            .clamp(0.0, 255.0) as u8,
        (200.0 * (y - z) + 128.0)
            .round_ties_even()
            .clamp(0.0, 255.0) as u8,
    ]
}

struct ColorHistogram {
    channels: [[usize; 256]; 3],
    count: usize,
}

impl ColorHistogram {
    fn new() -> Self {
        Self {
            channels: [[0; 256]; 3],
            count: 0,
        }
    }

    fn push(&mut self, color: [u8; 3]) {
        for (channel, value) in self.channels.iter_mut().zip(color) {
            channel[value as usize] += 1;
        }
        self.count += 1;
    }

    fn median(&self) -> [f32; 3] {
        if self.count == 0 {
            return [0.0; 3];
        }
        let low = (self.count - 1) / 2;
        let high = self.count / 2;
        std::array::from_fn(|channel| {
            let mut total = 0;
            let mut lower = 0;
            for (value, &count) in self.channels[channel].iter().enumerate() {
                if total <= low && total + count > low {
                    lower = value;
                }
                total += count;
                if total > high {
                    return (lower + value) as f32 * 0.5;
                }
            }
            unreachable!("color histogram count mismatch")
        })
    }
}

fn delta(a: [f32; 3], b: [f32; 3], normalized: bool) -> f32 {
    let l = (a[0] - b[0]) * if normalized { 100.0 / 255.0 } else { 1.0 };
    (l * l + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn colors_and_leftover(
    photo: &WorkingPhoto,
    bbox: Rect,
    roi: PixelRect,
    core: &GrayImage,
    mask: &GrayImage,
    glyph: f32,
) -> ([u8; 3], f32, usize) {
    let ring = dilate(
        core,
        (3.0 * photo.reference_scale).round_ties_even().max(1.0) as u32,
        false,
    );
    let border_width = (0.08 * glyph)
        .round_ties_even()
        .max(3.0 * photo.reference_scale)
        .round_ties_even()
        .max(1.0) as u32;
    let mut labs = Vec::with_capacity(core.as_raw().len());
    let mut text = ColorHistogram::new();
    let mut background = ColorHistogram::new();
    let mut border = ColorHistogram::new();
    for y in 0..roi.height {
        for x in 0..roi.width {
            let value = lab(photo.pixels.get_pixel(roi.x + x, roi.y + y).0);
            labs.push(value);
            if core.get_pixel(x, y)[0] != 0 {
                text.push(value);
            } else if ring.get_pixel(x, y)[0] != 0 {
                background.push(value);
            }
            if x < border_width
                || y < border_width
                || x.saturating_add(border_width) >= roi.width
                || y.saturating_add(border_width) >= roi.height
            {
                border.push(value);
            }
        }
    }
    if text.count == 0 {
        return ([245, 245, 240], 0.0, 0);
    }
    let bg = if background.count == 0 {
        border.median()
    } else {
        background.median()
    };
    let text_color = text.median();
    let border_color = border.median();
    let mut distances = Vec::with_capacity(text.count);
    for (&value, &on) in labs.iter().zip(core.as_raw()) {
        if on != 0 {
            distances.push(delta(value.map(f32::from), bg, false));
        }
    }
    let position = (distances.len() - 1) as f64 * 0.6;
    let low = position.floor() as usize;
    let fraction = (position - position.floor()) as f32;
    let (_, lower, right) = distances.select_nth_unstable_by(low, f32::total_cmp);
    let upper = right
        .iter()
        .copied()
        .min_by(f32::total_cmp)
        .unwrap_or(*lower);
    let threshold = *lower + (upper - *lower) * fraction;
    let mut fill_pixels = ColorHistogram::new();
    let mut leftover = 0;
    for y in 0..roi.height {
        for x in 0..roi.width {
            let index = (y * roi.width + x) as usize;
            let value = labs[index].map(f32::from);
            if core.as_raw()[index] != 0 && delta(value, bg, false) >= threshold {
                fill_pixels.push(
                    photo
                        .pixels
                        .get_pixel(roi.x + x, roi.y + y)
                        .0,
                );
            }
            let inside = x as f32 + roi.x as f32 + 0.5 >= bbox.left
                && x as f32 + roi.x as f32 + 0.5 < bbox.right
                && y as f32 + roi.y as f32 + 0.5 >= bbox.top
                && y as f32 + roi.y as f32 + 0.5 < bbox.bottom;
            if inside
                && mask.as_raw()[index] == 0
                && delta(value, text_color, true) < 12.0
                && delta(value, border_color, true) > 20.0
            {
                leftover += 1;
            }
        }
    }
    (
        fill_pixels.median().map(|v| v as u8),
        leftover as f32 / text.count as f32,
        text.count,
    )
}

pub fn refresh_erase(
    photo: &WorkingPhoto,
    region: &mut PreparedRegion,
    erase_source: &RgbImage,
    workspace: &mut Workspace,
) -> Result<(), String> {
    let start = Instant::now();
    let mask = dilate(
        &region.recovered,
        erase_radius(region.glyph_size, photo.reference_scale),
        true,
    );
    region.erase = Some(erase_patch(photo, region.roi, &mask, erase_source, workspace)?);
    region.erase_ms += start.elapsed().as_secs_f64() * 1000.0;
    Ok(())
}

pub fn erase_input_rect(photo: &WorkingPhoto, roi: PixelRect) -> PixelRect {
    let extent = (16.0 * photo.reference_scale).round_ties_even().max(1.0) as u32;
    let x = roi.x.saturating_sub(extent).min(photo.pixels.width());
    let y = roi.y.saturating_sub(extent).min(photo.pixels.height());
    let right = roi
        .x
        .saturating_add(roi.width)
        .saturating_add(extent)
        .min(photo.pixels.width());
    let bottom = roi
        .y
        .saturating_add(roi.height)
        .saturating_add(extent)
        .min(photo.pixels.height());
    PixelRect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    }
}

fn erase_patch(
    photo: &WorkingPhoto,
    roi: PixelRect,
    mask: &GrayImage,
    erase_source: &RgbImage,
    workspace: &mut telea::Workspace,
) -> Result<ErasePatch, String> {
    if erase_source.dimensions() != photo.pixels.dimensions() {
        return Err("photo_erase_source_dimensions".into());
    }
    let input = erase_input_rect(photo, roi);
    let x = input.x;
    let y = input.y;
    let crop = imageops::crop_imm(erase_source, x, y, input.width, input.height).to_image();
    let mut expanded_mask = GrayImage::new(crop.width(), crop.height());
    imageops::replace(
        &mut expanded_mask,
        mask,
        i64::from(roi.x - x),
        i64::from(roi.y - y),
    );
    let erased = telea::inpaint(
        crop,
        &expanded_mask,
        (5.0 * photo.reference_scale).round_ties_even().max(1.0) as u32,
        workspace,
    )
    .map_err(str::to_string)?;
    let pixels = RgbaImage::from_fn(roi.width, roi.height, |px, py| {
        let ex = roi.x - x + px;
        let ey = roi.y - y + py;
        let rgb = erased.get_pixel(ex, ey).0;
        let alpha = if mask.get_pixel(px, py)[0] != 0
            && rgb != erase_source.get_pixel(roi.x + px, roi.y + py).0
        {
            255
        } else {
            0
        };
        Rgba([rgb[0], rgb[1], rgb[2], alpha])
    });
    Ok(ErasePatch { rect: roi, pixels })
}

#[cfg(test)]
mod tests;
