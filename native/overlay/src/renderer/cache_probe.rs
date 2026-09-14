use super::super::cache::{BoundedLruCache, LayoutCache};
use super::super::types::CaptionBlock;
use super::*;
use windows::Win32::Graphics::Dxgi::{
    IDXGIAdapter3, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    DXGI_QUERY_VIDEO_MEMORY_INFO,
};
use windows::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
use windows::Win32::System::Threading::GetCurrentProcess;

fn memory_sample(renderer: &WindowsCaptionRenderer) -> serde_json::Value {
    unsafe {
        let device: IDXGIDevice = renderer._d3d_device.cast().unwrap();
        let adapter: IDXGIAdapter3 = device.GetAdapter().unwrap().cast().unwrap();
        let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut nonlocal = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let local_available = adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
            .is_ok();
        let nonlocal_available = adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut nonlocal)
            .is_ok();
        let mut memory = PROCESS_MEMORY_COUNTERS_EX::default();
        let size = std::mem::size_of_val(&memory) as u32;
        memory.cb = size;
        let process_available = K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&mut memory as *mut PROCESS_MEMORY_COUNTERS_EX).cast(),
            size,
        )
        .as_bool();
        serde_json::json!({
            "local_bytes": local_available.then_some(local.CurrentUsage),
            "local_budget_bytes": local_available.then_some(local.Budget),
            "nonlocal_bytes": nonlocal_available.then_some(nonlocal.CurrentUsage),
            "private_bytes": process_available.then_some(memory.PrivateUsage),
            "cache_counts": [renderer.caches.text_format_cache.len(), renderer.caches.layout_cache.len(), renderer.caches.line_cache.len(), renderer.caches.block_cache.len()],
        })
    }
}

#[tokio::test]
#[ignore = "Opt-in synthetic renderer memory/performance probe; no HMD or OpenVR submission"]
async fn renderer_cache_memory_probe() {
    let profile =
        std::env::var("PPH_OVERLAY_PROBE_CACHE_PROFILE").unwrap_or_else(|_| "eighth".into());
    let caps = match profile.as_str() {
        "original" => [32, 512, 2048, 1024],
        "eighth" => [16, 64, 256, 128],
        _ => panic!("unsupported probe cache profile"),
    };
    let mut renderer = WindowsCaptionRenderer::new(None).unwrap();
    renderer.caches.text_format_cache = BoundedLruCache::with_capacity(caps[0]);
    renderer.caches.layout_cache = LayoutCache::with_capacity(caps[1]);
    renderer.caches.line_cache = BoundedLruCache::with_capacity(caps[2]);
    renderer.caches.block_cache = BoundedLruCache::with_capacity(caps[3]);
    eprintln!(
        "CACHE_PROBE {}",
        serde_json::json!({"profile": profile, "revision": 0, "memory": memory_sample(&renderer), "adapter": format!("{:?}", renderer.adapter_identity)})
    );
    let mut changed_us = Vec::new();
    let mut repeated_us = Vec::new();
    for revision in 1..=4096 {
        let text = format!("{revision} Hello 日本語 한국어 中文 captions");
        let source_only = revision % 2 == 0;
        let blocks = vec![
            CaptionBlock::new(
                format!("self:{revision}"),
                format!("{revision} hello there"),
            )
            .with_channel(CaptionChannel::SelfChannel),
            CaptionBlock::new(
                format!("peer:{revision}"),
                if source_only { &text } else { "번역 자막" },
            )
            .with_channel(CaptionChannel::PeerChannel)
            .with_secondary_text(if source_only { "" } else { &text }, !source_only),
        ];
        for repeated in [false, true] {
            let started = Instant::now();
            let frame = renderer
                .render(
                    &CaptionLayoutPolicy::default(),
                    &CaptionPresentation::default(),
                    blocks.clone(),
                    DEFAULT_SURFACE_WIDTH_PX,
                    DEFAULT_SURFACE_HEIGHT_PX,
                    None,
                )
                .unwrap();
            let elapsed_us = started.elapsed().as_micros();
            if repeated {
                assert_eq!(frame.diagnostics().line_cache_misses, 0);
                assert_eq!(frame.diagnostics().block_cache_misses, 0);
                repeated_us.push(elapsed_us);
            } else {
                changed_us.push(elapsed_us);
            }
            assert_eq!(
                renderer
                    .prepare_frame_for_submission(&ReadinessCancellation::default())
                    .await,
                ReadinessOutcome::Ready
            );
            let counts = [
                frame.diagnostics().text_format_cache_size,
                frame.diagnostics().layout_cache_size,
                frame.diagnostics().line_cache_size,
                frame.diagnostics().block_cache_size,
            ];
            for (count, cap) in counts.into_iter().zip(caps) {
                assert!(count <= cap);
            }
        }
        if revision % 512 == 0 {
            eprintln!(
                "CACHE_PROBE {}",
                serde_json::json!({"profile": profile, "revision": revision, "memory": memory_sample(&renderer)})
            );
        }
    }
    changed_us.sort_unstable();
    repeated_us.sort_unstable();
    eprintln!(
        "CACHE_PROBE {}",
        serde_json::json!({"profile": profile, "changed_p50_us": changed_us[2048], "changed_p95_us": changed_us[3891], "repeated_p50_us": repeated_us[2048], "repeated_p95_us": repeated_us[3891]})
    );
}
