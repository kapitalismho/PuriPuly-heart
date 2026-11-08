use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::image::{PhotoImageOutput, PhotoImageResult, PhotoImageWorker, PreparationStage};
use super::manifest::{PhotoError, PhotoImageConfig};
use super::prepare::recover_original;
use super::processing::{
    erase_input_rect, keep_original, refresh_erase, MaskProcessor, PixelRect, PreparedRegion, Rect,
    WorkingPhoto,
};
use super::state::{PhotoPhase, PhotoSnapshot};

pub struct RunningImageWorker {
    thread: Option<std::thread::JoinHandle<()>>,
}

impl RunningImageWorker {
    pub fn start(
        mut boundary: PhotoImageWorker,
        config: PhotoImageConfig,
    ) -> Result<Self, PhotoError> {
        let runtime = tokio::runtime::Handle::current();
        let thread = std::thread::Builder::new()
            .name("photo-compose".into())
            .spawn(move || {
                let mut pipeline = ImagePipeline::new(config);
                while let Some(input) = runtime.block_on(boundary.next_input()) {
                    if input.photo.as_ref().is_some_and(|photo| {
                        photo.attempt_id > 0
                            && photo.image.is_some()
                            && photo.phase != PhotoPhase::Expired
                    }) && boundary.is_cancelled(&input)
                    {
                        continue;
                    }
                    let retained = boundary.decoded_original(&input);
                    if let (Some(source), Some(photo)) = (&retained, &input.photo) {
                        if let Some(working) = &source.working {
                            let key = (photo.photo_id.clone(), source.path.clone());
                            if pipeline.source_key.as_ref() != Some(&key) {
                                pipeline.source_key = Some(key);
                                pipeline.working = Some(working.clone());
                                pipeline.clear_attempt(0);
                            }
                        }
                    }
                    let outcome = pipeline.process(
                        &input,
                        retained.map(|source| source.pixels.clone()),
                        || !boundary.is_cancelled(&input),
                        |output| {
                            if boundary.is_cancelled(&input) {
                                return false;
                            }
                            boundary.complete_blocking(PhotoImageResult {
                                input: input.clone(),
                                output,
                            })
                        },
                    );
                    if let Err(code) = outcome {
                        if !boundary.is_cancelled(&input) {
                            boundary.complete_blocking(PhotoImageResult {
                                input: input.clone(),
                                output: PhotoImageOutput::Failed { code },
                            });
                        }
                    }
                }
            })
            .map_err(|_| PhotoError::Image)?;
        Ok(Self {
            thread: Some(thread),
        })
    }

    pub async fn shutdown(mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
    }
}

struct PendingRegion {
    region: PreparedRegion,
    context_revision: u64,
}

pub(crate) struct ImagePipeline {
    config: PhotoImageConfig,
    source_key: Option<(String, String)>,
    attempt: u64,
    working: Option<Arc<WorkingPhoto>>,
    processor: Option<MaskProcessor>,
    candidates: HashSet<u64>,
    prepared: HashMap<u64, PendingRegion>,
    completed: HashSet<u64>,
    #[cfg(windows)]
    typesetter: Option<super::typeset::PhotoTypesetter>,
    failed: bool,
    erase_source: Option<image::RgbImage>,
    commit_revision: u64,
    changes: Vec<(u64, Vec<PixelRect>)>,
    font_reported: bool,
}

impl ImagePipeline {
    pub fn new(config: PhotoImageConfig) -> Self {
        Self {
            config,
            source_key: None,
            attempt: 0,
            working: None,
            processor: None,
            candidates: HashSet::new(),
            prepared: HashMap::new(),
            completed: HashSet::new(),
            #[cfg(windows)]
            typesetter: None,
            failed: false,
            erase_source: None,
            commit_revision: 0,
            changes: Vec::new(),
            font_reported: false,
        }
    }

    pub fn process(
        &mut self,
        input: &PhotoSnapshot,
        retained: Option<Arc<image::RgbImage>>,
        is_current: impl Fn() -> bool,
        mut emit: impl FnMut(PhotoImageOutput) -> bool,
    ) -> Result<(), &'static str> {
        let Some(photo) = input
            .photo
            .as_deref()
            .filter(|photo| photo.attempt_id > 0 && photo.phase != PhotoPhase::Expired)
        else {
            self.source_key = None;
            self.working = None;
            self.processor = None;
            self.clear_attempt(0);
            return Ok(());
        };
        let Some(image) = photo.image.as_ref() else {
            return Ok(());
        };
        let key = (photo.photo_id.clone(), image.path.clone());
        if self.source_key.as_ref() != Some(&key) {
            let recovered = retained.is_none();
            let original = match retained {
                Some(pixels) => pixels,
                None => recover_original(Path::new(&image.path))?,
            };
            if original.dimensions() != (image.width, image.height) {
                return Err("source_dimensions_mismatch");
            }
            let working = Arc::new(
                WorkingPhoto::new(original, self.config.max_size)
                    .map_err(|_| "source_resize_failed")?,
            );
            self.source_key = Some(key);
            self.working = Some(working.clone());
            self.clear_attempt(photo.attempt_id);
            if !emit(PhotoImageOutput::Original { working, recovered }) {
                return Ok(());
            }
        } else if self.attempt != photo.attempt_id {
            self.clear_attempt(photo.attempt_id);
            if !emit(PhotoImageOutput::Original {
                working: self.working.as_ref().unwrap().clone(),
                recovered: false,
            }) {
                return Ok(());
            }
        }
        if self.failed {
            return Ok(());
        }
        let working = self.working.as_ref().unwrap().clone();
        for block in &photo.blocks {
            if self.completed.contains(&block.block_id) {
                continue;
            }
            if !is_current() {
                return Ok(());
            }
            if keep_original(&block.source, &block.translation) {
                self.prepared.remove(&block.block_id);
                if !emit(PhotoImageOutput::Block {
                    block_id: block.block_id,
                    region: None,
                    #[cfg(windows)]
                    typeset: None,
                    typeset_ms: 0.0,
                }) {
                    return Ok(());
                }
            } else {
                let pending = match self.prepared.remove(&block.block_id) {
                    Some(pending) => pending,
                    None => {
                        let region = self.prepare_region(
                            &working,
                            block.bbox,
                            block.source.split('\n').count() as u32,
                        )?;
                        if !emit_preparation_costs(block.block_id, &region, &mut emit) {
                            return Ok(());
                        }
                        PendingRegion {
                            region,
                            context_revision: self.commit_revision,
                        }
                    }
                };
                let mut region = pending.region;
                if !region.plate {
                    let dependency = erase_input_rect(&working, region.roi);
                    if self.changes.iter().any(|(revision, runs)| {
                        *revision > pending.context_revision
                            && runs.iter().any(|run| intersects(*run, dependency))
                    }) {
                        let before = region.erase_ms;
                        refresh_erase(&working, &mut region, self.erase_source.as_ref().unwrap())
                            .map_err(|_| "erase_refresh_failed")?;
                        if !emit(PhotoImageOutput::Timing {
                            block_id: block.block_id,
                            stage: "erase",
                            elapsed_ms: region.erase_ms - before,
                        }) {
                            return Ok(());
                        }
                        println!(
                            "{}",
                            serde_json::json!({
                                "type":"photo_image_stage","stage":"erase_refreshed","block_id":block.block_id,
                                "elapsed_ms":region.erase_ms-before,"mask_reused":true,
                            })
                        );
                    }
                }
                if !is_current() {
                    return Ok(());
                }
                let start = Instant::now();
                #[cfg(windows)]
                {
                    if self.typesetter.is_none() {
                        self.typesetter = Some(
                            super::typeset::PhotoTypesetter::new()
                                .map_err(|_| "typesetter_unavailable")?,
                        );
                    }
                    let typeset = self
                        .typesetter
                        .as_mut()
                        .unwrap()
                        .typeset(&working, &region, &block.translation)
                        .map_err(|_| "typeset_failed")?;
                    let typeset_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if !is_current() {
                        return Ok(());
                    }
                    if !self.font_reported {
                        if let Some(run) = typeset.runs.first() {
                            let mut length = 0;
                            unsafe { run.layout.GetFontFamilyNameLength(0, &mut length, None) }
                                .map_err(|_| "font_metrics_unavailable")?;
                            let mut name = vec![0; length as usize + 1];
                            unsafe { run.layout.GetFontFamilyName(0, &mut name, None) }
                                .map_err(|_| "font_metrics_unavailable")?;
                            let metrics = self.typesetter.as_ref().unwrap().font_metrics();
                            let font = super::image::PhotoFont {
                                family: String::from_utf16(&name[..length as usize])
                                    .map_err(|_| "font_metrics_unavailable")?,
                                source: format!("{:?}", metrics.source),
                                hangul_ink_ratio: metrics.hangul_ink_ratio,
                            };
                            if !emit(PhotoImageOutput::FontReady { font }) {
                                return Ok(());
                            }
                            self.font_reported = true;
                        }
                    }
                    if !emit(PhotoImageOutput::Timing {
                        block_id: block.block_id,
                        stage: "typeset",
                        elapsed_ms: typeset_ms,
                    }) {
                        return Ok(());
                    }
                    if !is_current() {
                        return Ok(());
                    }
                    self.commit_erase(&region);
                    if !emit(PhotoImageOutput::Block {
                        block_id: block.block_id,
                        region: Some(region),
                        typeset: Some(typeset),
                        typeset_ms,
                    }) {
                        return Ok(());
                    }
                }
                #[cfg(not(windows))]
                return Err("graphics_platform_unavailable");
            }
            self.completed.insert(block.block_id);
        }
        for preparation in &photo.preparations {
            if self.completed.contains(&preparation.block_id) {
                continue;
            }
            if !is_current() {
                return Ok(());
            }
            if self.candidates.insert(preparation.block_id) {
                let start = Instant::now();
                let rect = super::processing::candidate_region(
                    &working,
                    mapped_rect(preparation.bbox, &working),
                )
                .map_err(|_| "candidate_geometry_failed")?;
                if !emit(PhotoImageOutput::Candidate {
                    block_id: preparation.block_id,
                    rect,
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                }) {
                    return Ok(());
                }
            }
            if let (Some(_), Some(lines)) = (&preparation.source, preparation.source_line_count) {
                if !self.prepared.contains_key(&preparation.block_id) {
                    let region = self.prepare_region(&working, preparation.bbox, lines)?;
                    if !emit_preparation_costs(preparation.block_id, &region, &mut emit) {
                        return Ok(());
                    }
                    let elapsed_ms = region.mask_ms + region.erase_ms;
                    if !emit(PhotoImageOutput::Prepared {
                        block_id: preparation.block_id,
                        stage: PreparationStage::Source,
                        elapsed_ms,
                    }) {
                        return Ok(());
                    }
                    self.prepared.insert(
                        preparation.block_id,
                        PendingRegion {
                            region,
                            context_revision: self.commit_revision,
                        },
                    );
                }
            }
        }
        let oldest = self
            .prepared
            .values()
            .map(|pending| pending.context_revision)
            .min()
            .unwrap_or(self.commit_revision);
        self.changes.retain(|(revision, _)| *revision > oldest);
        Ok(())
    }

    fn clear_attempt(&mut self, attempt: u64) {
        self.attempt = attempt;
        self.candidates.clear();
        self.prepared.clear();
        self.completed.clear();
        self.failed = false;
        self.erase_source = None;
        self.commit_revision = 0;
        self.changes.clear();
        self.font_reported = false;
        #[cfg(windows)]
        if let Some(typesetter) = &mut self.typesetter {
            typesetter.reset();
        }
    }

    fn prepare_region(
        &mut self,
        working: &WorkingPhoto,
        bbox: [f64; 4],
        lines: u32,
    ) -> Result<PreparedRegion, &'static str> {
        if self.processor.is_none() {
            let model = self
                .config
                .model_path()
                .map_err(|_| "mask_model_unavailable")?;
            if !model.is_file() {
                self.failed = true;
                return Err("mask_model_missing");
            }
            self.processor = Some(
                MaskProcessor::new(&model, self.config.runtime_path.as_deref())
                    .map_err(|_| "mask_runtime_unavailable")?,
            );
        }
        self.processor
            .as_mut()
            .unwrap()
            .prepare(
                working,
                mapped_rect(bbox, working),
                lines,
                self.erase_source
                    .as_ref()
                    .unwrap_or(working.pixels.as_ref()),
            )
            .map_err(|_| "mask_processing_failed")
    }

    fn commit_erase(&mut self, region: &PreparedRegion) {
        if region.plate {
            return;
        }
        let Some(patch) = &region.erase else {
            return;
        };
        if self.erase_source.is_none() {
            if !patch.pixels.pixels().any(|pixel| pixel[3] > 0) {
                return;
            }
            self.erase_source = Some((*self.working.as_ref().unwrap().pixels).clone());
        }
        let source = self.erase_source.as_mut().unwrap();
        let mut changed = Vec::new();
        let track_changes = !self.prepared.is_empty();
        let mut changed_any = false;
        for y in 0..patch.rect.height {
            let mut start = None;
            for x in 0..patch.rect.width {
                let pixel = patch.pixels.get_pixel(x, y);
                let target = source.get_pixel_mut(patch.rect.x + x, patch.rect.y + y);
                let rgb = image::Rgb([pixel[0], pixel[1], pixel[2]]);
                let differs = pixel[3] > 0 && *target != rgb;
                if differs {
                    *target = rgb;
                    changed_any = true;
                    if track_changes {
                        start.get_or_insert(x);
                    }
                } else if let Some(left) = start.take() {
                    changed.push(PixelRect {
                        x: patch.rect.x + left,
                        y: patch.rect.y + y,
                        width: x - left,
                        height: 1,
                    });
                }
            }
            if let Some(left) = start {
                changed.push(PixelRect {
                    x: patch.rect.x + left,
                    y: patch.rect.y + y,
                    width: patch.rect.width - left,
                    height: 1,
                });
            }
        }
        if changed_any {
            self.commit_revision += 1;
            if track_changes {
                self.changes.push((self.commit_revision, changed));
            }
        }
    }
}

fn emit_preparation_costs(
    id: u64,
    region: &PreparedRegion,
    emit: &mut impl FnMut(PhotoImageOutput) -> bool,
) -> bool {
    emit(PhotoImageOutput::Timing {
        block_id: id,
        stage: "mask",
        elapsed_ms: region.mask_ms,
    }) && emit(PhotoImageOutput::Timing {
        block_id: id,
        stage: "erase",
        elapsed_ms: region.erase_ms,
    })
}

fn intersects(a: PixelRect, b: PixelRect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

fn mapped_rect(bbox: [f64; 4], working: &WorkingPhoto) -> Rect {
    Rect {
        left: bbox[0] as f32 * working.scale_x,
        top: bbox[1] as f32 * working.scale_y,
        right: bbox[2] as f32 * working.scale_x,
        bottom: bbox[3] as f32 * working.scale_y,
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
