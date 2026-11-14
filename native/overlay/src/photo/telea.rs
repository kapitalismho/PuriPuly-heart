use image::{GrayImage, RgbImage};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const INSIDE: u8 = 2;
const BAND: u8 = 1;
const CHANGE: u8 = 3;

#[derive(Clone, Copy)]
struct Entry {
    distance: f32,
    order: usize,
    index: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.order == other.order
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.order.cmp(&self.order))
    }
}

fn solve(a: usize, b: usize, state: &[u8], time: &[f32]) -> f32 {
    let x = f64::from(time[a]);
    let y = f64::from(time[b]);
    match (state[a] != INSIDE, state[b] != INSIDE) {
        (true, true) if (x - y).abs() < 1.0 => {
            ((x + y + (2.0 - (x - y) * (x - y)).sqrt()) * 0.5) as f32
        }
        (true, true) | (false, false) => (1.0 + x.min(y)) as f32,
        (true, false) => (1.0 + x) as f32,
        (false, true) => (1.0 + y) as f32,
    }
}

fn distance(p: usize, stride: usize, state: &[u8], time: &[f32]) -> f32 {
    solve(p - stride, p - 1, state, time)
        .min(solve(p + stride, p - 1, state, time))
        .min(solve(p - stride, p + 1, state, time))
        .min(solve(p + stride, p + 1, state, time))
}

fn gradient(p: usize, offset: usize, state: &[u8], time: &[f32]) -> f32 {
    match (state[p + offset] != INSIDE, state[p - offset] != INSIDE) {
        (true, true) => (time[p + offset] - time[p - offset]) * 0.5,
        (true, false) => time[p + offset] - time[p],
        (false, true) => time[p] - time[p - offset],
        (false, false) => 0.0,
    }
}

fn square_dilate(
    mask: &[u8],
    width: usize,
    height: usize,
    radius: usize,
    horizontal: &mut [u8],
    result: &mut [u8],
) {
    for y in 0..height {
        let mut count = 0;
        for x in 0..width + radius {
            if x < width && mask[y * width + x] != 0 {
                count += 1;
            }
            if x > radius * 2 && mask[y * width + x - radius * 2 - 1] != 0 {
                count -= 1;
            }
            if x >= radius && x - radius < width {
                horizontal[y * width + x - radius] = u8::from(count != 0);
            }
        }
    }
    for x in 0..width {
        let mut count = 0;
        for y in 0..height + radius {
            if y < height && horizontal[y * width + x] != 0 {
                count += 1;
            }
            if y > radius * 2 && horizontal[(y - radius * 2 - 1) * width + x] != 0 {
                count -= 1;
            }
            if y >= radius && y - radius < height {
                result[(y - radius) * width + x] = u8::from(count != 0);
            }
        }
    }
}

#[derive(Default)]
pub struct Workspace {
    state: Vec<u8>,
    time: Vec<f32>,
    heap: BinaryHeap<Entry>,
    band: Vec<usize>,
    outside: Vec<u8>,
    horizontal: Vec<u8>,
    distance_weights: Vec<f32>,
}

pub fn inpaint(
    mut output: RgbImage,
    mask: &GrayImage,
    radius: u32,
    workspace: &mut Workspace,
) -> Result<RgbImage, &'static str> {
    if output.dimensions() != mask.dimensions() || output.width() == 0 || output.height() == 0 {
        return Err("telea_invalid_dimensions");
    }
    let width = output.width() as usize;
    let height = output.height() as usize;
    let stride = width + 2;
    let rows = height + 2;
    let radius = radius.clamp(1, 100) as usize;
    if !mask.as_raw().iter().any(|&value| value != 0) {
        return Ok(output);
    }
    workspace.distance_weights.extend(
        (workspace.distance_weights.len()..=radius * radius).map(|squared| {
            let squared = squared as f64;
            (1.0 / (squared * squared.sqrt())) as f32
        }),
    );
    workspace.state.clear();
    workspace.state.resize(stride * rows, 0);
    workspace.time.clear();
    workspace.time.resize(workspace.state.len(), 1.0e6_f32);
    workspace.heap.clear();
    workspace.band.clear();
    workspace.outside.resize(workspace.state.len(), 0);
    workspace.horizontal.resize(workspace.state.len(), 0);
    let Workspace {
        state,
        time,
        heap,
        band,
        outside,
        horizontal,
        distance_weights,
    } = workspace;
    for y in 0..height {
        for x in 0..width {
            state[(y + 1) * stride + x + 1] = if mask.as_raw()[y * width + x] != 0 {
                INSIDE
            } else {
                0
            };
        }
    }
    let mut order = 0;
    for y in 1..=height {
        for x in 1..=width {
            let p = y * stride + x;
            if state[p] == 0
                && [p - stride, p - 1, p + stride, p + 1]
                    .iter()
                    .any(|&q| state[q] == INSIDE)
            {
                band.push(p);
                time[p] = 0.0;
                heap.push(Entry {
                    distance: 0.0,
                    order,
                    index: p,
                });
                order += 1;
            }
        }
    }
    square_dilate(state, stride, rows, radius, horizontal, outside);
    for p in 0..outside.len() {
        outside[p] = if outside[p] != 0 && state[p] == 0 {
            INSIDE
        } else {
            0
        };
    }
    for &p in band.iter() {
        outside[p] = 0;
    }
    for x in 0..stride {
        outside[x] = 0;
        outside[(rows - 1) * stride + x] = 0;
    }
    for y in 0..rows {
        outside[y * stride] = 0;
        outside[y * stride + stride - 1] = 0;
    }
    while let Some(entry) = heap.pop() {
        let p = entry.index;
        outside[p] = CHANGE;
        for q in [p - stride, p - 1, p + stride, p + 1] {
            if outside[q] == INSIDE {
                let value = distance(q, stride, outside, time);
                time[q] = value;
                outside[q] = BAND;
                heap.push(Entry {
                    distance: value,
                    order,
                    index: q,
                });
                order += 1;
            }
        }
    }
    for p in 0..outside.len() {
        if outside[p] == CHANGE {
            time[p] = -time[p];
        }
    }
    for &p in band.iter() {
        heap.push(Entry {
            distance: 0.0,
            order,
            index: p,
        });
        order += 1;
    }
    while let Some(entry) = heap.pop() {
        let p = entry.index;
        state[p] = 0;
        for q in [p - stride, p - 1, p + stride, p + 1] {
            if state[q] != INSIDE {
                continue;
            }
            let value = distance(q, stride, state, time);
            time[q] = value;
            let gx = gradient(q, 1, state, time);
            let gy = gradient(q, stride, state, time);
            let y = q / stride;
            let x = q % stride;
            let mut intensity = [0.0_f32; 3];
            let mut jx = [0.0_f32; 3];
            let mut jy = [0.0_f32; 3];
            let mut sum = 1.0e-20_f32;
            for k in y.saturating_sub(radius).max(1)..=(y + radius).min(height) {
                let km = k as isize - 1 + isize::from(k == 1);
                let kp = k as isize - 1 - isize::from(k == height);
                for l in x.saturating_sub(radius).max(1)..=(x + radius).min(width) {
                    let n = k * stride + l;
                    let dy = y as i32 - k as i32;
                    let dx = x as i32 - l as i32;
                    let squared = dx * dx + dy * dy;
                    if state[n] == INSIDE || squared as usize > radius * radius {
                        continue;
                    }
                    let rx = dx as f32;
                    let ry = dy as f32;
                    let dst = distance_weights[squared as usize];
                    let lev = (1.0 / (1.0 + f64::from((time[n] - time[q]).abs()))) as f32;
                    let direction = rx * gx + ry * gy;
                    let direction = if direction.abs() <= 0.01 {
                        0.000001
                    } else {
                        direction
                    };
                    let weight = (dst * lev * direction).abs();
                    let lm = l as isize - 1 + isize::from(l == 1);
                    let lp = l as isize - 1 - isize::from(l == width);
                    let pixel = |row: isize, col: isize, channel: usize| {
                        let index = (row * width as isize + col)
                            .clamp(0, (width * height - 1) as isize)
                            as usize;
                        f32::from(output.as_raw()[index * 3 + channel])
                    };
                    for channel in 0..3 {
                        let ix = match (state[n + 1] != INSIDE, state[n - 1] != INSIDE) {
                            (true, true) => {
                                (pixel(km, lp + 1, channel) - pixel(km, lm - 1, channel)) * 2.0
                            }
                            (true, false) => pixel(km, lp + 1, channel) - pixel(km, lm, channel),
                            (false, true) => pixel(km, lp, channel) - pixel(km, lm - 1, channel),
                            (false, false) => 0.0,
                        };
                        let iy = match (state[n + stride] != INSIDE, state[n - stride] != INSIDE) {
                            (true, true) => {
                                (pixel(kp + 1, lm, channel) - pixel(km - 1, lm, channel)) * 2.0
                            }
                            (true, false) => pixel(kp + 1, lm, channel) - pixel(km, lm, channel),
                            (false, true) => pixel(kp, lm, channel) - pixel(km - 1, lm, channel),
                            (false, false) => 0.0,
                        };
                        intensity[channel] +=
                            weight * pixel(k as isize - 1, l as isize - 1, channel);
                        jx[channel] -= weight * (ix * rx);
                        jy[channel] -= weight * (iy * ry);
                    }
                    sum += weight;
                }
            }
            for channel in 0..3 {
                let norm = f64::from(jx[channel] * jx[channel] + jy[channel] * jy[channel]).sqrt()
                    + 1.0e-20_f64;
                let result = (f64::from(intensity[channel] / sum)
                    + f64::from(jx[channel] + jy[channel]) / norm)
                    as f32;
                output.as_mut()[((y - 1) * width + x - 1) * 3 + channel] = (f64::from(result) + 0.5)
                    .round_ties_even()
                    .clamp(0.0, 255.0)
                    as u8;
            }
            state[q] = BAND;
            heap.push(Entry {
                distance: value,
                order,
                index: q,
            });
            order += 1;
        }
    }
    Ok(output)
}
