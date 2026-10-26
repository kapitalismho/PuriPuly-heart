use super::*;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_STAGING,
};

fn read_pixels(frame: &RenderedFrame) -> Vec<u8> {
    let texture = frame.d3d11_texture().unwrap();
    unsafe {
        let device = texture.GetDevice().unwrap();
        let context = device.GetImmediateContext().unwrap();
        let mut description = D3D11_TEXTURE2D_DESC::default();
        texture.GetDesc(&mut description);
        description.Usage = D3D11_USAGE_STAGING;
        description.BindFlags = 0;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        description.MiscFlags = 0;
        let mut staging = None;
        device
            .CreateTexture2D(&description, None, Some(&mut staging))
            .unwrap();
        let staging = staging.unwrap();
        context.CopyResource(&staging, texture);
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context
            .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .unwrap();
        let mut result =
            Vec::with_capacity(description.Width as usize * description.Height as usize * 4);
        for row in 0..description.Height as usize {
            result.extend_from_slice(std::slice::from_raw_parts(
                (mapped.pData as *const u8).add(row * mapped.RowPitch as usize),
                description.Width as usize * 4,
            ));
        }
        context.Unmap(&staging, 0);
        result
    }
}

#[derive(Default)]
struct CapturedFrames {
    text: Vec<String>,
    cache_max: [usize; 4],
    empty: usize,
}

struct ReadbackSubmitter(Arc<Mutex<CapturedFrames>>, bool);

impl OverlayFrameSubmitter for ReadbackSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        let pixels = read_pixels(frame);
        let mut captured = self.0.lock().unwrap();
        if frame.is_fully_transparent() {
            assert!(pixels.iter().all(|byte| *byte == 0));
            captured.empty += 1;
        } else {
            assert!(pixels
                .chunks_exact(4)
                .any(|pixel| pixel[3] != 0 && pixel[..3] != [0, 0, 0]));
            captured.text.push(
                frame
                    .layout()
                    .visible_blocks
                    .iter()
                    .flat_map(|block| {
                        block
                            .primary_lines
                            .iter()
                            .chain(block.secondary_line.iter())
                    })
                    .map(|line| line.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        let diagnostics = frame.diagnostics();
        for (index, count) in [
            diagnostics.text_format_cache_size,
            diagnostics.layout_cache_size,
            diagnostics.line_cache_size,
            diagnostics.block_cache_size,
        ]
        .into_iter()
        .enumerate()
        {
            captured.cache_max[index] = captured.cache_max[index].max(count);
        }
        Ok(())
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        Ok(SpatialReanchorOutcome::Applied)
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.1 = visible;
        Ok(())
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        Some(self.1)
    }
}

#[tokio::test]
async fn windows_graphics_python_snapshots_complete_paced_and_burst_with_readback() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let output = Command::new("uv")
        .args([
            "run",
            "--frozen",
            "python",
            "-m",
            "tests.helpers.overlay_active_freshness",
        ])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let snapshots: Vec<OverlayPresentationSnapshot> =
        serde_json::from_slice(&output.stdout).unwrap();
    for (label, spacing) in [("gpu-paced", 80), ("gpu-burst", 0)] {
        let renderer = CaptionRenderer::new().unwrap();
        assert_eq!(
            renderer.presentation_backend(),
            PresentationBackend::D3d11Hardware
        );
        assert!(
            renderer.font_initialization_warning().is_none(),
            "{:?}",
            renderer.font_initialization_warning()
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let scenes = snapshots.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();
            let _auth = ws.next().await.unwrap().unwrap();
            for (index, scene) in scenes.iter().enumerate() {
                tokio::time::sleep(Duration::from_millis(if index == 6 {
                    650
                } else {
                    spacing
                }))
                .await;
                ws.send(Message::Text(
                    json!({"type":"snapshot", "payload":scene})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
                if index == 0 {
                    wait_for_owner_ready(&mut ws).await;
                }
            }
            tokio::time::sleep(Duration::from_millis(600)).await;
            ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let mut manifest = test_manifest();
        manifest.bridge_url = format!("ws://{address}");
        let (mut bridge, initial) = BridgeClient::connect(&manifest).await.unwrap();
        let captured = Arc::new(Mutex::new(CapturedFrames::default()));
        let mut owner = NativePresentationOwner::new(
            initial,
            renderer,
            ReadbackSubmitter(captured.clone(), false),
        );
        let started = Instant::now();
        owner
            .run(&mut bridge, &test_logger(label).await)
            .await
            .unwrap();
        server.await.unwrap();
        assert!(owner.resources_released());
        assert!(owner.fresh_retry_schedules_for_test().is_empty());
        assert!(owner.runtime().state().blocks().is_empty());
        let records = owner.fresh_retry_evidence_for_test();
        assert!(records
            .iter()
            .any(|record| record["trigger_generation"] == 11
                && record["outcome"] == "completed"
                && record["fresh_rendered"] == true
                && record["producer_ready"] == true
                && record["submission_succeeded"] == true));
        assert_eq!(owner.fresh_retry_audit_dropped_for_test(), 0);
        let attempts = owner.successful_attempt_audit_for_test();
        assert!(attempts
            .windows(2)
            .all(|pair| pair[1].render_generation > pair[0].render_generation));
        let captured = captured.lock().unwrap();
        assert!(
            captured
                .text
                .last()
                .unwrap()
                .contains("synthetic translation"),
            "{:?}",
            captured.text
        );
        assert!(captured.empty > 0);
        assert!(attempts.len() <= 30);
        for (count, cap) in captured.cache_max.into_iter().zip([16, 64, 256, 128]) {
            assert!(count <= cap);
        }
        println!("GPU_OWNER {label} elapsed_ms={} submissions={} nonempty_readbacks={} empty_readbacks={} cache_max={:?} final_text={:?} pending_schedules=0 released=true", started.elapsed().as_millis(), attempts.len(), captured.text.len(), captured.empty, captured.cache_max, captured.text.last());
    }
}

#[tokio::test]
async fn windows_graphics_hidden_suffix_reuses_visible_block_without_owning_caption_identity() {
    let renderer = CaptionRenderer::new().unwrap();
    let prefix = "한국어 日本語 mixed caption ".repeat(30);
    let secondary = "source 原文 e\u{301} 👩\u{200d}💻 ".repeat(30);
    let make = |id: &str, suffix: &str| {
        CaptionBlock::new(id, format!("{prefix}{suffix}"))
            .with_channel(CaptionChannel::PeerChannel)
            .with_primary_language("ko")
            .with_secondary_language("ja")
            .with_speaker_style(SpeakerStyle::Cyan)
            .with_secondary_text(format!("{secondary}{suffix}"), true)
            .with_slot(0, 40.0)
    };
    let first = renderer
        .render_blocks(vec![make("old-occupant", "0000")])
        .unwrap();
    assert_eq!(
        renderer
            .prepare_frame_for_submission(&Default::default())
            .await,
        ReadinessOutcome::Ready
    );
    let first_pixels = read_pixels(&first);
    let replacement = renderer
        .render_blocks(vec![make("new-occupant", "9999")])
        .unwrap();
    assert_eq!(replacement.diagnostics().layout_cache_misses, 1);
    assert_eq!(replacement.diagnostics().block_cache_hits, 1);
    assert_eq!(replacement.diagnostics().block_cache_misses, 0);
    assert_eq!(replacement.layout().visible_blocks[0].id, "new-occupant");
    assert!(replacement.layout().visible_blocks[0].truncated_primary);
    assert!(replacement.layout().visible_blocks[0].truncated_secondary);
    assert_eq!(
        renderer
            .prepare_frame_for_submission(&Default::default())
            .await,
        ReadinessOutcome::Ready
    );
    assert!(read_pixels(&replacement) == first_pixels);
    for block in [
        CaptionBlock::new("new-occupant", "short e\u{301}")
            .with_channel(CaptionChannel::PeerChannel)
            .with_speaker_style(SpeakerStyle::Coral),
        CaptionBlock::new("moved", "日本語")
            .with_primary_language("ja")
            .with_slot(1, 544.0),
    ] {
        let frame = renderer.render_blocks(vec![block.clone()]).unwrap();
        assert_eq!(
            renderer
                .prepare_frame_for_submission(&Default::default())
                .await,
            ReadinessOutcome::Ready
        );
        let fresh = CaptionRenderer::new().unwrap();
        let primed = fresh
            .render_blocks(vec![make("new-occupant", "9999")])
            .unwrap();
        assert_eq!(
            fresh
                .prepare_frame_for_submission(&Default::default())
                .await,
            ReadinessOutcome::Ready
        );
        assert!(primed.layout().visible_blocks[0].truncated_primary);
        let expected = fresh.render_blocks(vec![block]).unwrap();
        assert_eq!(
            fresh
                .prepare_frame_for_submission(&Default::default())
                .await,
            ReadinessOutcome::Ready
        );
        assert_eq!(
            frame.layout().visible_blocks[0].primary_lines[0].font_size_px,
            expected.layout().visible_blocks[0].primary_lines[0].font_size_px,
        );
        assert!(read_pixels(&frame) == read_pixels(&expected));
    }
    let clear = renderer.render_empty_frame().unwrap();
    assert_eq!(
        renderer
            .prepare_frame_for_submission(&Default::default())
            .await,
        ReadinessOutcome::Ready
    );
    assert!(read_pixels(&clear).iter().all(|byte| *byte == 0));
    let fresh_device = CaptionRenderer::new().unwrap();
    let rebuilt = fresh_device
        .render_blocks(vec![make("fresh-device", "9999")])
        .unwrap();
    assert_eq!(rebuilt.diagnostics().block_cache_hits, 0);
    assert_eq!(
        fresh_device
            .prepare_frame_for_submission(&Default::default())
            .await,
        ReadinessOutcome::Ready
    );
    assert!(read_pixels(&rebuilt) == first_pixels);
}
