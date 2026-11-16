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
    #[cfg(feature = "photo-experiment")]
    quad_path: Option<PathBuf>,
    #[cfg(feature = "photo-experiment")]
    #[serde(default)]
    source_line_counts: std::collections::HashMap<u64, u32>,
}

fn default_size() -> [u32; 2] {
    [1920, 1080]
}

pub async fn run(path: &Path) -> Result<(), PhotoError> {
    let bytes = std::fs::read(path).map_err(|_| PhotoError::Manifest)?;
    let job: OfflineJob = serde_json::from_slice(&bytes).map_err(|_| PhotoError::Manifest)?;
    #[cfg(feature = "photo-experiment")]
    if job.source_line_counts.iter().any(|(id, count)| *count == 0 || !job.blocks.iter().any(|block| block.block_id == *id)) {
        return Err(PhotoError::Manifest);
    }
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
    let source_key = job.source_path.to_string_lossy().into_owned();
    let (sender, mut results) = tokio::sync::mpsc::channel(8);
    let worker = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, PhotoError> {
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
        #[cfg(feature = "photo-experiment")]
        let quads = if let Some(path) = &job.quad_path {
            let ids = job.blocks.iter().map(|block| block.block_id).collect();
            Some(std::sync::Arc::new(super::experiment::load_quads(
                path, [original.width(), original.height()], &original, &ids, &job.output_dir,
            ).map_err(|error| {
                eprintln!("{}", json!({"type":"photo_experiment_error","detail":error.to_string()}));
                PhotoError::Manifest
            })?))
        } else {
            None
        };
        #[cfg(feature = "photo-experiment")]
        {
            pipeline.quads = quads.clone();
        }
        #[cfg(feature = "photo-experiment")]
        let mut experiment = json!({
            "quad_path":job.quad_path,"quad_coordinate_space":"original_photo",
            "quad_order":["TL","TR","BR","BL"],"perspective_warping":"unevaluated",
            "roi_and_glyph_sizing":"unchanged_bbox","blocks":{},
        });
        #[cfg(not(feature = "photo-experiment"))]
        let experiment = serde_json::Value::Null;
        let mut state = PhotoState::default();
        let mut revision = 0;
        #[cfg(feature = "photo-experiment")]
        let mut region_scale = [1.0, 1.0];
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
                    #[cfg(feature = "photo-experiment")]
                    if let PhotoImageOutput::Original { working, .. } = &output {
                        region_scale = [working.scale_x, working.scale_y];
                    }
                    if let PhotoImageOutput::Block {
                        block_id,
                        region: Some(region),
                        typeset,
                        ..
                    } = &output
                    {
                        #[cfg(not(feature = "photo-experiment"))]
                        let _ = typeset;
                        let mask = job.output_dir.join(format!("mask-{block_id}.png"));
                        if region.core.save(mask).is_err() {
                            artifact_failed = true;
                            return false;
                        }
                        #[cfg(feature = "photo-experiment")]
                        {
                            if region.recovered.save(job.output_dir.join(format!("recovered-{block_id}.png"))).is_err() {
                                artifact_failed = true;
                                return false;
                            }
                            let quad = quads.as_ref().and_then(|q| q.get(block_id)).map(|q| {
                                super::experiment::Quad { corners: q.corners.map(|p| [
                                    p[0]*f64::from(region_scale[0]),p[1]*f64::from(region_scale[1]),
                                ]) }
                            });
                            experiment["blocks"][block_id.to_string()] = super::experiment::block_evidence(region, quad, typeset.as_ref());
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
            #[cfg(feature = "photo-experiment")]
            if let Some(count) = job.source_line_counts.get(&block.block_id) {
                preparation.source_line_count = Some(*count);
            }
            preparation.source = Some(block.source.clone());
            process(&photo)?;
            photo.blocks.push(block);
            process(&photo)?;
        }
        Ok(experiment)
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
                compositor.set_original(working, &source_key)?;
                compositor.render(Instant::now(), false)?;
                compositor.wait_for_completion().await?;
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
    let experiment = worker.await.map_err(|_| PhotoError::Image)??;
    let final_time = Instant::now() + BLOCK_FADE;
    compositor.render(final_time, true)?;
    compositor.wait_for_completion().await?;
    let erased_gpu_ms = compositor.frame_gpu_ms;
    compositor.save(&output_dir.join("erased.png"))?;
    compositor.render(final_time, false)?;
    compositor.wait_for_completion().await?;
    let composite_gpu_ms = compositor.frame_gpu_ms;
    compositor.save(&output_dir.join("composite.png"))?;
    let mut metadata = compositor.metadata();
    if !experiment.is_null() {
        metadata["experiment"] = experiment;
    }
    metadata["erased_gpu_ms"] = json!(erased_gpu_ms);
    metadata["composition_gpu_ms"] = json!(composite_gpu_ms);
    metadata["gpu_timing_valid"] = json!(composite_gpu_ms.is_some());
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

#[cfg(all(test, feature = "photo-experiment"))]
mod tests {
    use super::*;

    #[test]
    fn experiment_job_preserves_original_blocks_order_and_physical_line_counts() {
        let job: OfflineJob = serde_json::from_value(json!({
            "source_path":"original.png","output_dir":"output","model_path":"mask.onnx",
            "quad_path":"quads.json","source_line_counts":{"7":3},
            "blocks":[
                {"block_id":7,"bbox":[10.0,20.0,30.0,40.0],"source":"first\nsecond","translation":"번역"},
                {"block_id":2,"bbox":[50.0,20.0,80.0,40.0],"source":"last","translation":"마지막"}
            ]
        })).unwrap();
        assert_eq!(job.max_size,[1920,1080]);
        assert_eq!(job.blocks.iter().map(|b| b.block_id).collect::<Vec<_>>(),[7,2]);
        assert_eq!(job.blocks[0].bbox,[10.0,20.0,30.0,40.0]);
        assert_eq!(job.blocks[0].source,"first\nsecond");
        assert_eq!(job.source_line_counts[&7],3);
        assert_eq!(job.quad_path,Some(PathBuf::from("quads.json")));
    }

    #[test]
    fn baseline_job_needs_no_quad_or_line_count_extension() {
        let job: OfflineJob = serde_json::from_value(json!({
            "source_path":"original.png","output_dir":"output","model_path":"mask.onnx","blocks":[]
        })).unwrap();
        assert!(job.quad_path.is_none());
        assert!(job.source_line_counts.is_empty());
        assert_eq!(job.max_size,default_size());
    }
}
