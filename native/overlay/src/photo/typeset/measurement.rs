use image::GrayImage;

use super::{rect, Direction, Rect};

#[derive(Clone, Debug)]
pub struct SourceMeasurement {
    pub angle_degrees: f32,
    pub direction: Direction,
    pub extent: Rect,
    pub slots: Vec<Rect>,
    pub glyph_size: f32,
    pub center: [f32; 2],
}

pub fn measure_source(core: &GrayImage, offset: [f32; 2], scale: f32) -> Option<SourceMeasurement> {
    let count = core.as_raw().iter().filter(|&&v| v != 0).count();
    if count < 20 {
        return None;
    }
    let (angle_degrees, direction) = projection_angle(core, count);
    let width = core.width();
    let height = core.height();
    let center = [width as f32 / 2.0, height as f32 / 2.0];
    let mut rotated = None;
    if angle_degrees != 0.0 {
        let mut work = GrayImage::new(width, height);
        let (s, c) = angle_degrees.to_radians().sin_cos();
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let sx = (c * dx - s * dy + center[0]).round() as i32;
                let sy = (s * dx + c * dy + center[1]).round() as i32;
                if sx >= 0 && sy >= 0 && sx < width as i32 && sy < height as i32 {
                    work.put_pixel(x, y, *core.get_pixel(sx as u32, sy as u32));
                }
            }
        }
        rotated = Some(work);
    }
    let deskewed = rotated.as_ref().unwrap_or(core);
    let mut bounds = [width, height, 0, 0];
    for (x, y, value) in deskewed.enumerate_pixels() {
        if value[0] != 0 {
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x + 1);
            bounds[3] = bounds[3].max(y + 1);
        }
    }
    if bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
        return None;
    }
    let vertical = direction == Direction::Vertical;
    let mut glyph_size = if vertical {
        bounds[2] - bounds[0]
    } else {
        bounds[3] - bounds[1]
    } as f32
        / 3.0;
    let mut slots = Vec::new();
    for _ in 0..2 {
        slots = measure_slots(&deskewed, bounds, vertical, glyph_size, scale);
        if slots.is_empty() {
            break;
        }
        let mut thickness: Vec<f32> = slots
            .iter()
            .map(|slot| {
                if vertical {
                    slot.width()
                } else {
                    slot.height()
                }
            })
            .collect();
        thickness.sort_by(f32::total_cmp);
        let mid = thickness.len() / 2;
        glyph_size = if thickness.len() % 2 == 0 {
            (thickness[mid - 1] + thickness[mid]) / 2.0
        } else {
            thickness[mid]
        };
    }
    for slot in &mut slots {
        slot.left += offset[0];
        slot.right += offset[0];
        slot.top += offset[1];
        slot.bottom += offset[1];
    }
    Some(SourceMeasurement {
        angle_degrees,
        direction,
        extent: rect(
            bounds[0] as f32 + offset[0],
            bounds[1] as f32 + offset[1],
            bounds[2] as f32 + offset[0],
            bounds[3] as f32 + offset[1],
        ),
        slots,
        glyph_size,
        center: [center[0] + offset[0], center[1] + offset[1]],
    })
}

fn projection_angle(core: &GrayImage, count: usize) -> (f32, Direction) {
    let mut points = Vec::with_capacity(count);
    for (x, y, pixel) in core.enumerate_pixels() {
        if pixel[0] != 0 {
            points.push([x as f64 + 0.5, y as f64 + 0.5]);
        }
    }
    let bins = ((core.width() as f32).hypot(core.height() as f32).ceil() as usize) + 3;
    let mut histogram = vec![0u32; bins];
    let mut projections = vec![0.0f64; count];
    let mut best_h = (-1.0f64, 0.0f32);
    let mut best_v = best_h;
    let mut zero = [0.0; 2];
    for step in -90..=90 {
        let angle = step as f32 * 0.5;
        let (s, c) = (angle as f64).to_radians().sin_cos();
        let score = |vertical: bool, histogram: &mut [u32], projections: &mut [f64]| {
            histogram.fill(0);
            let mut min = f64::INFINITY;
            for (point, projected) in points.iter().zip(projections.iter_mut()) {
                *projected = if vertical {
                    point[0] * c + point[1] * s
                } else {
                    point[1] * c - point[0] * s
                };
                min = min.min(*projected);
            }
            for &projected in projections.iter() {
                histogram[(projected - min) as usize] += 1;
            }
            histogram
                .iter()
                .map(|&n| (n as f64) * (n as f64))
                .sum::<f64>()
        };
        let hs = score(false, &mut histogram, &mut projections);
        let vs = score(true, &mut histogram, &mut projections);
        if step == 0 {
            zero = [hs, vs];
        }
        if hs > best_h.0 {
            best_h = (hs, angle);
        }
        if vs > best_v.0 {
            best_v = (vs, angle);
        }
    }
    let vertical = if (best_h.0 - best_v.0).abs() <= best_h.0.max(best_v.0) * 0.02 {
        core.height() > core.width()
    } else {
        best_v.0 > best_h.0
    };
    let best = if vertical { best_v } else { best_h };
    let angle = if best.1.abs() < 3.0 || best.0 < 1.1 * zero[usize::from(vertical)] {
        0.0
    } else {
        best.1
    };
    (
        angle,
        if vertical {
            Direction::Vertical
        } else {
            Direction::Horizontal
        },
    )
}

fn measure_slots(
    core: &GrayImage,
    bounds: [u32; 4],
    vertical: bool,
    glyph: f32,
    scale: f32,
) -> Vec<Rect> {
    let (start, end) = if vertical {
        (bounds[0], bounds[2])
    } else {
        (bounds[1], bounds[3])
    };
    let mut bands: Vec<[u32; 2]> = Vec::new();
    let mut open = None;
    for i in start..=end {
        let on = i < end
            && if vertical {
                (bounds[1]..bounds[3]).any(|y| core.get_pixel(i, y)[0] != 0)
            } else {
                (bounds[0]..bounds[2]).any(|x| core.get_pixel(x, i)[0] != 0)
            };
        if on && open.is_none() {
            open = Some(i);
        } else if !on {
            if let Some(a) = open.take() {
                if let Some(last) = bands
                    .last_mut()
                    .filter(|last| ((a - last[1]) as f32) < (2.0 * scale).max(0.15 * glyph))
                {
                    last[1] = i;
                } else {
                    bands.push([a, i]);
                }
            }
        }
    }
    let mut slots = Vec::new();
    for [a, b] in bands {
        if ((b - a) as f32) < (2.0 * scale).max(0.25 * glyph) {
            continue;
        }
        let mut lo = u32::MAX;
        let mut hi = 0;
        for i in a..b {
            let (cross_start, cross_end) = if vertical {
                (bounds[1], bounds[3])
            } else {
                (bounds[0], bounds[2])
            };
            for cross in cross_start..cross_end {
                let (x, y) = if vertical { (i, cross) } else { (cross, i) };
                if core.get_pixel(x, y)[0] != 0 {
                    lo = lo.min(cross);
                    hi = hi.max(cross + 1);
                }
            }
        }
        slots.push(if vertical {
            rect(a as f32, lo as f32, b as f32, hi as f32)
        } else {
            rect(lo as f32, a as f32, hi as f32, b as f32)
        });
    }
    if vertical {
        slots.reverse();
    }
    slots
}
