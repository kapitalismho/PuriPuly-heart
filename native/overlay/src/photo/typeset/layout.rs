use super::{
    contains, hits, image_bounds, overlap, rect, translated, union, DirectWriteMetrics, Direction,
    LayoutKind, Rect, SourceMeasurement,
};

const SPILL_STEPS: [f32; 5] = [0.5, 1.0, 2.0, 3.0, 4.0];

#[derive(Clone)]
pub(super) struct Run {
    pub text: String,
    pub size: f32,
    pub origin: [f32; 2],
    pub ink: Rect,
}

#[derive(Clone)]
pub(super) struct Candidate {
    pub runs: Vec<Run>,
    pub collision: Vec<Rect>,
    pub size: f32,
    pub kind: LayoutKind,
    pub overflow: bool,
}

impl Candidate {
    pub fn boxes(&self, info: &SourceMeasurement) -> Vec<Rect> {
        self.collision
            .iter()
            .map(|&b| image_bounds(b, info))
            .collect()
    }
}

pub(super) fn choose(
    metrics: &mut DirectWriteMetrics,
    text: &str,
    info: &SourceMeasurement,
    model_box: Rect,
    floor: f32,
    hard_floor: f32,
    cap: f32,
    placed: &[Rect],
    photo_bounds: Rect,
) -> Result<Candidate, String> {
    let mut regions = vec![info.extent];
    if info.angle_degrees == 0.0 {
        let expanded = union(info.extent, model_box);
        if expanded.left != info.extent.left
            || expanded.top != info.extent.top
            || expanded.right != info.extent.right
            || expanded.bottom != info.extent.bottom
        {
            regions.push(expanded);
        }
    }
    let mut free = Vec::new();
    let mut fallback = Vec::new();
    let mut tried = Vec::new();
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    for region in regions {
        let mut kinds = vec![LayoutKind::Horizontal];
        if lines.len() == info.slots.len() && !lines.is_empty() {
            kinds.push(LayoutKind::Lines);
        }
        if info.direction == Direction::Vertical {
            kinds.push(LayoutKind::Vertical);
        }
        for kind in kinds {
            tried.push((region, kind));
            let make = |metrics: &mut DirectWriteMetrics, size: f32, strict: bool| {
                layout(metrics, text, &lines, info, region, kind, size, strict)
            };
            let forced = make(metrics, hard_floor, false)?;
            if let Some(candidate) = forced {
                fallback.push(candidate);
            }
            if cap < floor {
                continue;
            }
            let Some(candidate) = bisect(metrics, floor, cap, |metrics, size| {
                make(metrics, size, true)
            })?
            else {
                continue;
            };
            if !hits(&candidate.boxes(info), placed) {
                free.push(candidate);
                continue;
            }
            if let Some(candidate) = bisect(metrics, floor, candidate.size, |metrics, size| {
                Ok(make(metrics, size, true)?
                    .filter(|candidate| !hits(&candidate.boxes(info), placed)))
            })? {
                free.push(candidate);
            }
        }
    }
    let score = |candidate: &Candidate| score(candidate, info);
    if let Some(best) = free.into_iter().max_by(|a, b| {
        score(a)
            .total_cmp(&score(b))
            .then_with(|| (a.kind == LayoutKind::Lines).cmp(&(b.kind == LayoutKind::Lines)))
    }) {
        return Ok(best);
    }
    if let Some(best) = spill(metrics, text, &lines, info, model_box, floor, placed, photo_bounds)? {
        return Ok(best);
    }
    let mut shrunk = Vec::new();
    for (region, kind) in tried {
        if let Some(candidate) = bisect(metrics, hard_floor, floor, |metrics, size| {
            Ok(layout(metrics, text, &lines, info, region, kind, size, true)?
                .filter(|candidate| !hits(&candidate.boxes(info), placed)))
        })? {
            shrunk.push(candidate);
        }
    }
    if let Some(best) = shrunk.into_iter().max_by(|a, b| score(a).total_cmp(&score(b))) {
        return Ok(best);
    }
    if hard_floor < floor {
        if let Some(best) =
            spill(metrics, text, &lines, info, model_box, hard_floor, placed, photo_bounds)?
        {
            return Ok(best);
        }
    }
    let rescue = info.angle_degrees == 0.0
        && info.direction == Direction::Horizontal
        && fallback.iter().all(|candidate| candidate.overflow);
    if rescue {
        let base = union(info.extent, model_box);
        let margin = hard_floor * 0.5;
        let expanded = rect(
            (base.left - margin).max(photo_bounds.left),
            (base.top - margin).max(photo_bounds.top),
            (base.right + margin).min(photo_bounds.right),
            (base.bottom + margin).min(photo_bounds.bottom),
        );
        if let Some(candidate) = layout(
            metrics,
            text,
            &lines,
            info,
            expanded,
            LayoutKind::Horizontal,
            hard_floor,
            false,
        )? {
            fallback.push(candidate);
        }
    }
    let span = |candidate: &Candidate| {
        let (top, bottom) = candidate
            .runs
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(top, bottom), run| {
                (top.min(run.ink.top), bottom.max(run.ink.bottom))
            });
        bottom - top
    };
    fallback
        .into_iter()
        .map(|candidate| {
            let overlap = candidate
                .collision
                .iter()
                .map(|&bounds| image_bounds(bounds, info))
                .flat_map(|bounds| {
                    placed
                        .iter()
                        .map(move |&previous| overlap(bounds, previous))
                })
                .sum::<f32>();
            (candidate, overlap)
        })
        .min_by(|(a, a_overlap), (b, b_overlap)| {
            a_overlap
                .total_cmp(b_overlap)
                .then_with(|| a.overflow.cmp(&b.overflow))
                .then_with(|| score(b).total_cmp(&score(a)))
                .then_with(|| {
                    if rescue {
                        (b.kind == LayoutKind::Horizontal)
                            .cmp(&(a.kind == LayoutKind::Horizontal))
                            .then_with(|| span(a).total_cmp(&span(b)))
                    } else {
                        (b.kind == LayoutKind::Lines).cmp(&(a.kind == LayoutKind::Lines))
                    }
                })
        })
        .map(|(candidate, _)| candidate)
        .ok_or_else(|| "photo translation has no drawable text".into())
}

fn score(candidate: &Candidate, info: &SourceMeasurement) -> f32 {
    let vertical = candidate.kind == LayoutKind::Vertical
        || (candidate.kind == LayoutKind::Lines && info.direction == Direction::Vertical);
    candidate.size
        * if vertical == (info.direction == Direction::Vertical) {
            1.0
        } else {
            0.85
        }
}

fn spill(
    metrics: &mut DirectWriteMetrics,
    text: &str,
    lines: &[&str],
    info: &SourceMeasurement,
    model_box: Rect,
    size: f32,
    placed: &[Rect],
    photo_bounds: Rect,
) -> Result<Option<Candidate>, String> {
    if info.angle_degrees != 0.0 {
        return Ok(None);
    }
    let base = union(info.extent, model_box);
    let mut kinds = vec![LayoutKind::Horizontal];
    if info.direction == Direction::Vertical {
        kinds.push(LayoutKind::Vertical);
    }
    for step in SPILL_STEPS {
        let margin = size * step;
        let region = rect(
            (base.left - margin).max(photo_bounds.left),
            (base.top - margin).max(photo_bounds.top),
            (base.right + margin).min(photo_bounds.right),
            (base.bottom + margin).min(photo_bounds.bottom),
        );
        let mut spread = Vec::new();
        for &kind in &kinds {
            if let Some(candidate) = layout(metrics, text, lines, info, region, kind, size, true)? {
                if !hits(&candidate.boxes(info), placed) {
                    spread.push(candidate);
                }
            }
        }
        if let Some(best) = spread
            .into_iter()
            .max_by(|a, b| score(a, info).total_cmp(&score(b, info)))
        {
            return Ok(Some(best));
        }
    }
    Ok(None)
}

fn bisect(
    metrics: &mut DirectWriteMetrics,
    mut lo: f32,
    mut hi: f32,
    mut make: impl FnMut(&mut DirectWriteMetrics, f32) -> Result<Option<Candidate>, String>,
) -> Result<Option<Candidate>, String> {
    let Some(mut best) = make(metrics, lo)? else {
        return Ok(None);
    };
    if let Some(candidate) = make(metrics, hi)? {
        return Ok(Some(candidate));
    }
    for _ in 0..12 {
        let mid = (lo + hi) / 2.0;
        if let Some(candidate) = make(metrics, mid)? {
            lo = mid;
            best = candidate;
        } else {
            hi = mid;
        }
        if hi - lo < 0.05 {
            break;
        }
    }
    Ok(Some(best))
}

fn layout(
    metrics: &mut DirectWriteMetrics,
    text: &str,
    lines: &[&str],
    info: &SourceMeasurement,
    region: Rect,
    kind: LayoutKind,
    size: f32,
    strict: bool,
) -> Result<Option<Candidate>, String> {
    let mut output = match kind {
        LayoutKind::Horizontal => horizontal(metrics, text, region, size)?,
        LayoutKind::Vertical => vertical(metrics, text, region, size)?,
        LayoutKind::Lines => source_lines(metrics, lines, info, size)?,
    };
    let Some(candidate) = &mut output else {
        return Ok(None);
    };
    let mut overflow = candidate.overflow
        || candidate.runs.iter().any(|run| !contains(region, run.ink))
        || candidate
            .collision
            .iter()
            .any(|&bounds| !contains(region, bounds));
    if kind == LayoutKind::Lines {
        overflow |= candidate
            .collision
            .iter()
            .zip(&info.slots)
            .any(|(&bounds, &slot)| !contains(slot, bounds));
    }
    candidate.overflow = overflow;
    if strict && overflow {
        return Ok(None);
    }
    Ok(output)
}

fn horizontal(
    metrics: &mut DirectWriteMetrics,
    text: &str,
    region: Rect,
    size: f32,
) -> Result<Option<Candidate>, String> {
    let lines = wrap(metrics, text, region.width(), size)?;
    if lines.is_empty() {
        return Ok(None);
    }
    let mut runs = Vec::with_capacity(lines.len());
    let mut top = f32::INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for (i, line) in lines.into_iter().enumerate() {
        let measured = metrics.measure(&line, size)?;
        let x = (region.left + region.right - measured.ink.left - measured.ink.right) / 2.0;
        let y = i as f32 * size * 1.25;
        let ink = translated(measured.ink, x, y);
        top = top.min(ink.top);
        bottom = bottom.max(ink.bottom);
        runs.push(Run {
            text: line,
            size,
            origin: [x, y],
            ink,
        });
    }
    let dy = (region.top + region.bottom - top - bottom) / 2.0;
    let mut collision = Vec::with_capacity(runs.len());
    for run in &mut runs {
        run.origin[1] += dy;
        run.ink = translated(run.ink, 0.0, dy);
        collision.push(run.ink);
    }
    Ok(Some(Candidate {
        runs,
        collision,
        size,
        kind: LayoutKind::Horizontal,
        overflow: false,
    }))
}

fn wrap(
    metrics: &mut DirectWriteMetrics,
    text: &str,
    width: f32,
    size: f32,
) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let mut current = String::new();
        for word in paragraph.split_whitespace() {
            let proposed = if current.is_empty() {
                word.to_owned()
            } else {
                format!("{current} {word}")
            };
            if !current.is_empty() && metrics.measure(&proposed, size)?.advance > width {
                lines.push(std::mem::take(&mut current));
                current.push_str(word);
            } else {
                current = proposed;
            }
            while metrics.measure(&current, size)?.advance > width && current.chars().count() > 1 {
                let ends: Vec<usize> = current
                    .char_indices()
                    .skip(1)
                    .map(|(i, _)| i)
                    .chain(std::iter::once(current.len()))
                    .collect();
                let mut cut = ends[0];
                for &end in &ends[1..] {
                    if metrics.measure(&current[..end], size)?.advance <= width {
                        cut = end;
                    } else {
                        break;
                    }
                }
                let remainder = current.split_off(cut);
                lines.push(current);
                current = remainder;
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    Ok(lines)
}

fn source_lines(
    metrics: &mut DirectWriteMetrics,
    lines: &[&str],
    info: &SourceMeasurement,
    size: f32,
) -> Result<Option<Candidate>, String> {
    let mut runs = Vec::new();
    let mut collision = Vec::new();
    for (&line, &slot) in lines.iter().zip(&info.slots) {
        if info.direction == Direction::Vertical {
            let cells = vertical_cells(line);
            if let Some(bounds) = column(
                metrics,
                &cells,
                (slot.left + slot.right) / 2.0,
                slot.top,
                size,
                &mut runs,
            )? {
                collision.push(bounds);
            }
        } else {
            let measured = metrics.measure(line, size)?;
            let dx = (slot.width() - measured.ink.width()) / 2.0;
            let dy = (slot.height() - measured.ink.height()) / 2.0;
            let ink = rect(
                slot.left + dx,
                slot.top + dy,
                slot.right - dx,
                slot.bottom - dy,
            );
            let origin = [ink.left - measured.ink.left, ink.top - measured.ink.top];
            collision.push(ink);
            runs.push(Run {
                text: line.to_owned(),
                size,
                origin,
                ink,
            });
        }
    }
    if runs.is_empty() {
        return Ok(None);
    }
    Ok(Some(Candidate {
        runs,
        collision,
        size,
        kind: LayoutKind::Lines,
        overflow: false,
    }))
}

fn vertical(
    metrics: &mut DirectWriteMetrics,
    text: &str,
    region: Rect,
    size: f32,
) -> Result<Option<Candidate>, String> {
    let per_column = (region.height() / (size * 1.05)).floor().max(1.0) as usize;
    let columns = vertical_columns(text, per_column);
    if columns.is_empty() {
        return Ok(None);
    }
    let width = columns.len() as f32 * size * 1.15 - size * 0.15;
    let mut xc = (region.left + region.right + width - size) / 2.0;
    let mut runs = Vec::new();
    let mut collision = Vec::with_capacity(columns.len());
    for cells in columns {
        if let Some(bounds) = column(metrics, &cells, xc, region.top, size, &mut runs)? {
            collision.push(bounds);
        }
        xc -= size * 1.15;
    }
    let overflow = width > region.width();
    Ok(Some(Candidate {
        runs,
        collision,
        size,
        kind: LayoutKind::Vertical,
        overflow,
    }))
}

fn column(
    metrics: &mut DirectWriteMetrics,
    cells: &[String],
    xc: f32,
    top: f32,
    size: f32,
    runs: &mut Vec<Run>,
) -> Result<Option<Rect>, String> {
    let mut y = top;
    let mut bounds: Option<Rect> = None;
    for cell in cells {
        if cell == " " {
            y += size * 0.4;
            continue;
        }
        let cell_size = size * if cell.chars().count() == 2 { 0.75 } else { 1.0 };
        let measured = metrics.measure(cell, cell_size)?;
        let origin = [
            xc - (measured.ink.left + measured.ink.right) / 2.0,
            y - measured.ink.top,
        ];
        let ink = translated(measured.ink, origin[0], origin[1]);
        bounds = Some(bounds.map_or(ink, |bounds| union(bounds, ink)));
        runs.push(Run {
            text: cell.clone(),
            size: cell_size,
            origin,
            ink,
        });
        y += size * 1.05;
    }
    Ok(bounds.map(|bounds| rect(bounds.left, top, bounds.right, bounds.bottom.max(y))))
}

pub(super) fn vertical_cells(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut cells = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if i + 1 < chars.len()
            && ((matches!(ch, '!' | '?') && matches!(chars[i + 1], '!' | '?'))
                || (ch.is_numeric()
                    && chars[i + 1].is_numeric()
                    && (i + 2 == chars.len() || !chars[i + 2].is_numeric())))
        {
            cells.push([ch, chars[i + 1]].iter().collect());
            i += 2;
            continue;
        }
        let form = match ch {
            '「' | '\'' | '‘' => '﹁',
            '」' | '’' => '﹂',
            '『' | '“' => '﹃',
            '』' | '”' => '﹄',
            '（' | '(' => '︵',
            '）' | ')' => '︶',
            '…' => '︙',
            '‥' => '︰',
            'ー' | '—' | '-' => '│',
            '〈' => '︿',
            '〉' => '﹀',
            '《' => '︽',
            '》' => '︾',
            '、' => '︑',
            '。' | '.' => '︒',
            ',' | '，' => '︐',
            ':' => '︓',
            ';' => '︔',
            '!' => '︕',
            '?' => '︖',
            '[' => '﹇',
            ']' => '﹈',
            '{' => '︷',
            '}' => '︸',
            '~' | '～' => '︴',
            _ => ch,
        };
        cells.push(form.to_string());
        i += 1;
    }
    cells
}

fn vertical_columns(text: &str, per_column: usize) -> Vec<Vec<String>> {
    let mut columns = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let cells = vertical_cells(word);
        if cells.is_empty() {
            continue;
        }
        let need = cells.len() + usize::from(!current.is_empty());
        if !current.is_empty() && current.len() + need > per_column {
            columns.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(" ".into());
        }
        for cell in cells {
            if current.len() == per_column {
                columns.push(std::mem::take(&mut current));
            }
            current.push(cell);
        }
    }
    if !current.is_empty() {
        columns.push(current);
    }
    columns
}
