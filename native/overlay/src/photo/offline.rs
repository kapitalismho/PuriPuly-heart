use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;

use super::compositor::{PhotoCompositor, BLOCK_FADE};
use super::image::PhotoImageOutput;
use super::manifest::{PhotoError, PhotoImageConfig};
use super::prepare::recover_original;
use super::state::{
    Photo, PhotoBlock, PhotoImage, PhotoPhase, PhotoPreparation, PhotoSnapshot, PhotoState,
};
use super::worker::ImagePipeline;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OfflineJob {
    source_path: PathBuf,
    output_dir: PathBuf,
    model_path: PathBuf,
    runtime_path: Option<PathBuf>,
    #[serde(default = "default_size")]
    max_size: [u32; 2],
    blocks: Vec<PhotoBlock>,
}

fn default_size() -> [u32; 2] {
    [1920, 1080]
}

pub async fn run(path: &Path) -> Result<(), PhotoError> {
    let bytes = std::fs::read(path).map_err(|_| PhotoError::Manifest)?;
    let job: OfflineJob = serde_json::from_slice(&bytes).map_err(|_| PhotoError::Manifest)?;
    let config = PhotoImageConfig {
        model_path: Some(job.model_path),
        runtime_path: job.runtime_path,
        max_size: job.max_size,
        ..Default::default()
    };
    config.validate()?;
    let start = Instant::now();
    std::fs::create_dir_all(&job.output_dir).map_err(|_| PhotoError::Image)?;
    let output_dir = job.output_dir.clone();
    let (sender, mut results) = tokio::sync::mpsc::channel(8);
    let worker = tokio::task::spawn_blocking(move || -> Result<(), PhotoError> {
        let original = recover_original(&job.source_path).map_err(|_| PhotoError::Image)?;
        let source = job.source_path.to_string_lossy().into_owned();
        let mut photo = Photo {
            photo_id: "offline".into(),
            path: source.clone(),
            file_size: 1,
            expected_file_id: super::prepare::file_identity(
                &std::fs::File::open(&job.source_path).map_err(|_| PhotoError::Image)?
            ).map_err(|_| PhotoError::Image)?,
            shutter_unix_ms: 1,
            attempt_id: 1,
            phase: PhotoPhase::Requesting,
            offer_open: false,
            visible: true,
            image: Some(PhotoImage {
                path: source,
                width: original.width(),
                height: original.height(),
            }),
            preparations: Vec::new(),
            blocks: Vec::new(),
            excluded_blocks: Vec::new(),
            failure_code: None,
            first_block_drawn: false,
        };
        let mut pipeline = ImagePipeline::new(config);
        let mut state = PhotoState::default();
        let mut revision = 0;
        let mut process = |photo: &Photo| -> Result<(), PhotoError> {
            revision += 1;
            state.apply(revision, Some(photo.clone()))?;
            let input: &PhotoSnapshot = state.snapshot().ok_or(PhotoError::Protocol)?;
            let mut artifact_failed = false;
            let result = pipeline.process(
                input,
                Some(original.clone()),
                || true,
                |output| {
                    if let PhotoImageOutput::Block {
                        block_id,
                        region: Some(region),
                        ..
                    } = &output
                    {
                        let mask = job.output_dir.join(format!("mask-{block_id}.png"));
                        if region.core.save(mask).is_err() {
                            artifact_failed = true;
                            return false;
                        }
                        if let Some(patch) = &region.erase {
                            if patch
                                .pixels
                                .save(job.output_dir.join(format!("erase-{block_id}.png")))
                                .is_err()
                            {
                                artifact_failed = true;
                                return false;
                            }
                        }
                    }
                    sender.blocking_send(output).is_ok()
                },
            );
            if artifact_failed {
                return Err(PhotoError::Image);
            }
            result.map_err(|code| {
                eprintln!("{}", json!({"type": "photo_image_failed", "code": code}));
                PhotoError::Image
            })
        };
        process(&photo)?;
        for block in job.blocks {
            photo.preparations.push(PhotoPreparation {
                block_id: block.block_id,
                bbox: block.bbox,
                source: None,
                source_line_count: None,
            });
            process(&photo)?;
            let preparation = photo.preparations.last_mut().unwrap();
            preparation.source_line_count = Some(block.source.split('\n').count() as u32);
            preparation.source = Some(block.source.clone());
            process(&photo)?;
            photo.blocks.push(block);
            process(&photo)?;
        }
        Ok(())
    });
    let mut compositor = PhotoCompositor::new(None)?;
    while let Some(output) = results.recv().await {
        match output {
            PhotoImageOutput::Timing {
                block_id,
                stage,
                elapsed_ms,
            } => {
                compositor.record_timing(block_id, stage, elapsed_ms);
            }
            PhotoImageOutput::FontReady { font } => {
                println!(
                    "{}",
                    json!({
                        "type":"photo_image_stage","stage":"typesetter_ready",
                        "family":font.family,"source":font.source,"hangul_ink_ratio":font.hangul_ink_ratio,
                    })
                );
                compositor.set_font(font);
            }
            PhotoImageOutput::Original { working, .. } => {
                compositor.set_original(working)?;
                compositor.render(Instant::now(), false)?;
                compositor.ready().await?;
                compositor.save(&output_dir.join("original.png"))?;
            }
            PhotoImageOutput::Block {
                block_id,
                region,
                typeset,
                typeset_ms,
            } => {
                compositor.add_block(block_id, region, typeset, typeset_ms, Instant::now())?;
            }
            PhotoImageOutput::Prepared { .. } | PhotoImageOutput::Candidate { .. } => {}
            PhotoImageOutput::Failed { .. } => return Err(PhotoError::Image),
        }
    }
    worker.await.map_err(|_| PhotoError::Image)??;
    let final_time = Instant::now() + BLOCK_FADE;
    compositor.render(final_time, true)?;
    compositor.ready().await?;
    let erased_gpu_ms = compositor.frame_gpu_ms;
    compositor.save(&output_dir.join("erased.png"))?;
    compositor.render(final_time, false)?;
    compositor.ready().await?;
    let composite_gpu_ms = compositor.frame_gpu_ms;
    compositor.save(&output_dir.join("composite.png"))?;
    compositor.view_original(true);
    compositor.render(final_time, false)?;
    compositor.ready().await?;
    compositor.save(&output_dir.join("original_view.png"))?;
    let mut metadata = compositor.metadata();
    metadata["erased_gpu_ms"] = json!(erased_gpu_ms);
    metadata["composition_gpu_ms"] = json!(composite_gpu_ms);
    metadata["gpu_timing_valid"] = json!(composite_gpu_ms.is_some());
    metadata["original_view_gpu_ms"] = json!(compositor.frame_gpu_ms);
    metadata["offline_artifact_wall_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
    std::fs::write(
        output_dir.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|_| PhotoError::Image)?,
    )
    .map_err(|_| PhotoError::Image)?;
    println!(
        "{}",
        json!({"type": "photo_offline_complete", "block_count": compositor.block_count(), "original_uploads": compositor.original_uploads})
    );
    Ok(())
}
