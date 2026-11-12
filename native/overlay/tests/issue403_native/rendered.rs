use super::*;
use std::io::Write;
use std::path::{Path, PathBuf};

fn write_bitmap(path: &Path, frame: &RenderedFrame, pixels: &[u8]) {
    let width = frame.width();
    let height = frame.height();
    let row_size = (width * 3 + 3) & !3;
    let image_size = row_size * height;
    let mut output = std::fs::File::create(path).unwrap();
    output.write_all(b"BM").unwrap();
    output.write_all(&(54 + image_size).to_le_bytes()).unwrap();
    output.write_all(&[0; 4]).unwrap();
    output.write_all(&54u32.to_le_bytes()).unwrap();
    output.write_all(&40u32.to_le_bytes()).unwrap();
    output.write_all(&(width as i32).to_le_bytes()).unwrap();
    output.write_all(&(-(height as i32)).to_le_bytes()).unwrap();
    output.write_all(&1u16.to_le_bytes()).unwrap();
    output.write_all(&24u16.to_le_bytes()).unwrap();
    output.write_all(&0u32.to_le_bytes()).unwrap();
    output.write_all(&image_size.to_le_bytes()).unwrap();
    output.write_all(&[0; 16]).unwrap();
    let mut row = vec![0; row_size as usize];
    for source in pixels.chunks_exact(width as usize * 4) {
        for (input, target) in source.chunks_exact(4).zip(row.chunks_exact_mut(3)) {
            target.copy_from_slice(&input[..3]);
        }
        output.write_all(&row).unwrap();
    }
}

struct ImageSubmitter {
    path: PathBuf,
    layout: Option<Value>,
}

impl OverlayFrameSubmitter for ImageSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        let pixels = super::super::gpu_updates::read_pixels(frame);
        assert!(pixels
            .chunks_exact(4)
            .any(|pixel| pixel[3] != 0 && pixel[..3] != [0, 0, 0]));
        assert!(frame.layout().dropped_block_ids.is_empty());
        let blocks = frame.layout().visible_blocks.iter().map(|block| {
            assert!(!block.truncated_primary, "{}", block.id);
            assert!(block.primary_lines.len() <= 2);
            assert!(block.visual_bounds.top_px >= 0.0);
            assert!(block.visual_bounds.bottom_px <= frame.height() as f32);
            for line in &block.primary_lines {
                assert!(line.origin_x >= 0.0);
                assert!(line.origin_x + line.width_px <= frame.width() as f32);
            }
            json!({
                "id": block.id,
                "lines": block.primary_lines.iter().map(|line| json!({"text":line.text,"x":line.origin_x,"y":line.origin_y,"width":line.width_px,"font_px":line.font_size_px})).collect::<Vec<_>>(),
                "top": block.bounds.top_px,
                "bottom": block.bounds.bottom_px,
                "secondary": block.secondary_line.as_ref().map(|line| line.text.clone()),
                "truncated_primary": block.truncated_primary,
            })
        }).collect::<Vec<_>>();
        assert!(!blocks.is_empty() && blocks.len() <= 2);
        write_bitmap(&self.path, frame, &pixels);
        self.layout = Some(json!({"width":frame.width(),"height":frame.height(),"blocks":blocks}));
        Ok(())
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        Ok(SpatialReanchorOutcome::Applied)
    }

    fn set_overlay_visible(&mut self, _visible: bool) -> Result<(), OpenVrError> {
        Ok(())
    }
}

#[tokio::test]
async fn windows_graphics_issue403_renders_python_output_corpus_to_readable_images() {
    let corpus = python_cases("corpus");
    let root = std::env::var_os("ISSUE403_NATIVE_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(unique_log_dir("issue403-images")));
    std::fs::create_dir_all(&root).unwrap();
    let mut observations = Vec::new();
    for (name, rows) in corpus.as_object().unwrap() {
        let renderer = CaptionRenderer::new().unwrap();
        assert_eq!(
            renderer.presentation_backend(),
            PresentationBackend::D3d11Hardware
        );
        assert!(renderer.font_initialization_warning().is_none());
        println!(
            "ISSUE403_GPU case={name} backend={:?} adapter={:?} binary={:?}",
            renderer.presentation_backend(),
            renderer.adapter_identity(),
            std::env::current_exe().unwrap()
        );
        for row in rows
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["label"] != "preview")
        {
            let label = row["label"].as_str().unwrap();
            let scene = snapshot(row);
            let mut runtime = OverlayRuntime::new(scene.clone());
            let mut submitter = ImageSubmitter {
                path: root.join(format!("{name}-{label}.bmp")),
                layout: None,
            };
            let log_dir = unique_log_dir("issue403-rendered");
            let logger = OverlayLogger::open(&log_dir).await.unwrap();
            let (mut bridge, server) = connect_test_bridge().await;
            runtime
                .submit_frame_if_needed(&renderer, &mut submitter, &mut bridge, &logger)
                .await
                .unwrap();
            let layout = submitter.layout.as_ref().unwrap();
            let actual = layout["blocks"].as_array().unwrap();
            assert_eq!(actual.len(), scene.blocks.len());
            for (observed, expected) in actual.iter().zip(&scene.blocks) {
                assert_eq!(observed["id"], expected.id);
                let text = observed["lines"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|line| line["text"].as_str().unwrap())
                    .collect::<Vec<_>>()
                    .join("\n");
                assert_eq!(compact(&text), compact(&expected.primary_text));
            }
            observations.push(json!({"case":name,"label":label,"at_ms":row["at_ms"],"input":row["input"],"snapshot":scene,"layout":layout,"image":submitter.path}));
            drop(bridge);
            server.await.unwrap();
            logger.shutdown().unwrap();
        }
    }
    let evidence_path = root.join("rendered-observations.json");
    std::fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&observations).unwrap(),
    )
    .unwrap();
    println!("ISSUE403_IMAGES {}", evidence_path.display());
}
