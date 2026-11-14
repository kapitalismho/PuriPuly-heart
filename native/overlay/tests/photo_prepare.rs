use image::{GenericImageView, Rgb, RgbImage};
use puripuly_heart_overlay::photo::prepare::{
    file_identity, prepare, PreparationCompletion, PreparationWorker, PrepareRequest, SourceRequest,
};
use std::fs::File;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

struct Directory(std::path::PathBuf);
impl Directory {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("puri339-preparation-{}-{name}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn request(&self, width: u32, height: u32) -> PrepareRequest {
        let source = self.0.join("capture.png");
        RgbImage::from_pixel(width, height, Rgb([200, 90, 35]))
            .save(&source)
            .unwrap();
        let output = self.0.join("output");
        std::fs::create_dir(&output).unwrap();
        PrepareRequest {
            photo_id: "synthetic-prep".into(),
            attempt_id: 1,
            source_path: source.to_str().unwrap().into(),
            output_dir: output.to_str().unwrap().into(),
            expected_file_id: file_identity(&File::open(source).unwrap()).unwrap(),
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn local_request(request: &PrepareRequest) -> SourceRequest {
    SourceRequest {
        photo_id: request.photo_id.clone(),
        source_path: request.source_path.clone(),
        expected_file_id: request.expected_file_id.clone(),
        max_size: [320, 180],
    }
}

fn png_header(bytes: &[u8]) -> &[u8] {
    let mut offset = 8;
    loop {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        if &bytes[offset + 4..offset + 8] == b"IDAT" {
            return &bytes[..offset + 8];
        }
        offset += length + 12;
    }
}

fn jpeg_bytes(width: u32, height: u32) -> Vec<u8> {
    let pixels = RgbImage::from_pixel(width, height, Rgb([200, 90, 35]));
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 95)
        .encode_image(&pixels)
        .unwrap();
    bytes
}

fn jpeg_header(bytes: &[u8]) -> &[u8] {
    let mut offset = 2;
    loop {
        assert_eq!(bytes[offset], 0xff);
        let length = usize::from(u16::from_be_bytes(bytes[offset + 2..offset + 4].try_into().unwrap()));
        let end = offset + length + 2;
        if bytes[offset + 1] == 0xda {
            return &bytes[..end];
        }
        offset = end;
    }
}

#[test]
fn native_preparation_preserves_original_and_maps_supported_sizes_and_aspects() {
    for (name, width, height, scale, offset_x, offset_y, canvas) in [
        ("720", 1280, 720, 1.0, 0, 0, (1280, 720)),
        ("1080", 1920, 1080, 2.0 / 3.0, 0, 0, (1280, 720)),
        ("1440", 2560, 1440, 0.5, 0, 0, (1280, 720)),
        ("2160", 3840, 2160, 1.0 / 3.0, 0, 0, (1280, 720)),
        ("portrait", 360, 720, 1280.0 / 720.0, 40, 0, (720, 1280)),
        ("portrait-2160", 2160, 3840, 1.0 / 3.0, 0, 0, (720, 1280)),
        ("wide", 1920, 480, 2.0 / 3.0, 0, 200, (1280, 720)),
        ("custom", 853, 601, 720.0 / 601.0, 129, 0, (1280, 720)),
    ] {
        let directory = Directory::new(name);
        let request = directory.request(width, height);
        let prepared = prepare(&request, &AtomicBool::new(false)).ok().unwrap();
        assert_eq!(prepared.original.dimensions(), (width, height));
        assert_eq!(prepared.scale, scale);
        assert_eq!((prepared.offset_x, prepared.offset_y), (offset_x, offset_y));
        assert_eq!(
            std::fs::read(&prepared.source_path).unwrap(),
            std::fs::read(&request.source_path).unwrap()
        );
        let jpeg = image::open(&prepared.jpeg_path).unwrap().into_rgb8();
        assert_eq!(jpeg.dimensions(), canvas);
        let center = jpeg.get_pixel(canvas.0 / 2, canvas.1 / 2);
        for (actual, expected) in center.0.iter().zip([200i16, 90, 35]) {
            assert!((*actual as i16 - expected).abs() <= 3);
        }
        if offset_x > 0 || offset_y > 0 {
            let border = jpeg.get_pixel(0, 0);
            for (actual, expected) in border.0.iter().zip([16i16, 18, 24]) {
                assert!((*actual as i16 - expected).abs() <= 3);
            }
        }
        assert!(
            prepared.preparation_ms >= prepared.decode_ms + prepared.resize_ms + prepared.jpeg_ms
        );
    }
}

#[test]
fn source_identity_replacement_and_output_ownership_are_preserved() {
    let directory = Directory::new("identity");
    let mut request = directory.request(1280, 720);
    let original = std::path::Path::new(&request.source_path);
    std::fs::rename(original, directory.0.join("old.png")).unwrap();
    RgbImage::new(1280, 720).save(original).unwrap();
    assert_eq!(
        prepare(&request, &AtomicBool::new(false)).err(),
        Some("source_replaced")
    );
    assert!(!std::path::Path::new(&request.output_dir)
        .join("source.png")
        .exists());
    request.expected_file_id = file_identity(&File::open(original).unwrap()).unwrap();
    let jpeg = std::path::Path::new(&request.output_dir).join("send.jpg");
    std::fs::write(&jpeg, b"existing-user-bytes").unwrap();
    assert_eq!(
        prepare(&request, &AtomicBool::new(false)).err(),
        Some("output_unavailable")
    );
    assert_eq!(std::fs::read(&jpeg).unwrap(), b"existing-user-bytes");
}

#[test]
fn incomplete_png_wait_is_bounded_and_cancellation_stops_it() {
    let directory = Directory::new("incomplete");
    let request = directory.request(360, 720);
    let full = std::fs::read(&request.source_path).unwrap();
    std::fs::write(&request.source_path, &full[..24]).unwrap();
    let source = request.source_path.clone();
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(source, full).unwrap();
    });
    let prepared = prepare(&request, &AtomicBool::new(false)).ok().unwrap();
    writer.join().unwrap();
    assert_eq!(prepared.original.dimensions(), (360, 720));
    drop(prepared);
    std::fs::write(&request.source_path, b"incomplete").unwrap();
    let start = std::time::Instant::now();
    assert_eq!(
        prepare(&request, &AtomicBool::new(false)).err(),
        Some("source_incomplete")
    );
    assert!(start.elapsed() >= Duration::from_secs(2));
    assert!(start.elapsed() < Duration::from_secs(5));
    let cancel = Arc::new(AtomicBool::new(false));
    let other = cancel.clone();
    let setter = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(70));
        other.store(true, Ordering::Relaxed);
    });
    assert_eq!(
        prepare(&request, &cancel).err(),
        Some("preparation_cancelled")
    );
    setter.join().unwrap();
}

#[tokio::test]
async fn image_worker_runs_off_async_progress_and_shutdown_drains_started_work() {
    let directory = Directory::new("shutdown");
    let request = directory.request(1280, 720);
    std::fs::write(&request.source_path, b"incomplete").unwrap();
    let mut worker = PreparationWorker::new().ok().unwrap();
    worker.start(request).unwrap();
    tokio::time::timeout(Duration::from_millis(100), async {
        tokio::task::yield_now().await;
    })
    .await
    .unwrap();
    worker.shutdown().await;
    let PreparationCompletion::Prepared(result) = worker.next_result().await.unwrap() else { panic!("Preparation completion required") };
    assert_eq!(result.result.err(), Some("preparation_cancelled"));
    assert!(!directory.0.join("output/source.png").exists());
}

#[test]
fn shared_source_decoder_advances_jpeg_recovery_and_preparation_without_extension_dependency() {
    use puripuly_heart_overlay::photo::prepare::{decode_source, recover_original};
    let directory = Directory::new("shared-jpeg");
    let mut request = directory.request(320, 180);
    let bytes = jpeg_bytes(320, 180);
    let decoded = decode_source(&bytes).unwrap();
    assert_eq!(decoded.dimensions(), (320, 180));
    for (actual, expected) in decoded.get_pixel(0, 0).0.iter().zip([200i16, 90, 35]) {
        assert!((i16::from(*actual) - expected).abs() <= 3);
    }
    let snapshot = directory.0.join("recovery.snapshot");
    std::fs::write(&snapshot, &bytes).unwrap();
    let recovered = recover_original(&snapshot).unwrap();
    assert_eq!(&*recovered, &decoded);
    request.source_path = snapshot.to_string_lossy().into_owned();
    request.expected_file_id = file_identity(&File::open(&snapshot).unwrap()).unwrap();
    let prepared = prepare(&request, &AtomicBool::new(false)).ok().unwrap();
    assert_eq!(&*prepared.original, &decoded);
    assert_eq!(std::fs::read(&prepared.source_path).unwrap(), bytes);
}

#[tokio::test]
async fn grab_source_is_local_and_reused_without_decode_for_authorized_preparation() {
    use puripuly_heart_overlay::photo::prepare::SourceRequest;
    let directory = Directory::new("preview-reuse");
    let request = directory.request(640, 360);
    let mut worker = PreparationWorker::new().unwrap();
    let local = SourceRequest {
        photo_id: request.photo_id.clone(),
        source_path: request.source_path.clone(),
        expected_file_id: request.expected_file_id.clone(),
        max_size: [320, 180],
    };
    worker.start_source(local.clone()).unwrap();
    let PreparationCompletion::Source(completion) = worker.next_result().await.unwrap() else { panic!("Local source completion required") };
    let source = completion.result.unwrap();
    assert_eq!(source.original.dimensions(), (640, 360));
    assert_eq!(source.working.as_ref().unwrap().pixels.dimensions(), (320, 180));
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    worker.start_source(local).unwrap();
    let PreparationCompletion::Source(regrab) = worker.next_result().await.unwrap() else { panic!("Regrab source completion required") };
    assert!(Arc::ptr_eq(&source, &regrab.result.unwrap()));
    worker.start(request.clone()).unwrap();
    let PreparationCompletion::Prepared(completion) = worker.next_result().await.unwrap() else { panic!("Authorized completion required") };
    let prepared = completion.result.unwrap();
    assert!(Arc::ptr_eq(&source.original, &prepared.original));
    assert!(Arc::ptr_eq(source.working.as_ref().unwrap(), prepared.working.as_ref().unwrap()));
    assert_eq!(prepared.decode_ms, 0.0);
    assert_eq!(std::fs::read(&prepared.source_path).unwrap(), std::fs::read(&request.source_path).unwrap());
    assert_eq!(image::open(&prepared.jpeg_path).unwrap().dimensions(), (1280, 720));
    let original = Arc::downgrade(&source.original);
    let working = Arc::downgrade(source.working.as_ref().unwrap());
    drop(source);
    drop(prepared);
    assert!(original.upgrade().is_none());
    assert!(working.upgrade().is_none());
    worker.shutdown().await;
}

#[tokio::test]
async fn authorization_queued_while_grab_decode_is_pending_reuses_the_same_source() {
    use puripuly_heart_overlay::photo::prepare::SourceRequest;
    let directory = Directory::new("preview-pending");
    let request = directory.request(640, 360);
    let full = std::fs::read(&request.source_path).unwrap();
    std::fs::write(&request.source_path, &full[..24]).unwrap();
    let path = request.source_path.clone();
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(path, full).unwrap();
    });
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_source(SourceRequest {
        photo_id: request.photo_id.clone(),
        source_path: request.source_path.clone(),
        expected_file_id: request.expected_file_id.clone(),
        max_size: [320, 180],
    }).unwrap();
    worker.start(request).unwrap();
    let PreparationCompletion::Source(completion) = worker.next_result().await.unwrap() else { panic!("Source must settle before queued preparation") };
    let source = completion.result.unwrap();
    let PreparationCompletion::Prepared(completion) = worker.next_result().await.unwrap() else { panic!("Authorized completion required") };
    let prepared = completion.result.unwrap();
    assert!(Arc::ptr_eq(&source.original, &prepared.original));
    assert!(Arc::ptr_eq(source.working.as_ref().unwrap(), prepared.working.as_ref().unwrap()));
    assert_eq!(prepared.decode_ms, 0.0);
    writer.join().unwrap();
    worker.shutdown().await;
}

#[tokio::test]
async fn selected_preview_source_cannot_silently_switch_on_write_or_replacement() {
    use puripuly_heart_overlay::photo::prepare::SourceRequest;
    for replace in [false, true] {
        let directory = Directory::new(if replace { "preview-replaced" } else { "preview-changed" });
        let request = directory.request(320, 180);
        let mut worker = PreparationWorker::new().unwrap();
        worker.start_source(SourceRequest {
            photo_id: request.photo_id.clone(),
            source_path: request.source_path.clone(),
            expected_file_id: request.expected_file_id.clone(),
            max_size: [320, 180],
        }).unwrap();
        let PreparationCompletion::Source(completion) = worker.next_result().await.unwrap() else { panic!("Source completion required") };
        let source = completion.result.unwrap();
        if replace {
            std::fs::rename(&request.source_path, directory.0.join("old.png")).unwrap();
            RgbImage::new(640, 360).save(&request.source_path).unwrap();
        } else {
            std::fs::write(&request.source_path, b"changed same-file bytes").unwrap();
            assert_eq!(file_identity(&File::open(&request.source_path).unwrap()).unwrap(), request.expected_file_id);
        }
        worker.start(request.clone()).unwrap();
        let PreparationCompletion::Prepared(completion) = worker.next_result().await.unwrap() else { panic!("Preparation completion required") };
        assert_eq!(completion.result.err(), Some(if replace { "source_replaced" } else { "source_changed" }));
        assert_eq!(*source.original.get_pixel(0, 0), Rgb([200, 90, 35]));
        assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
        worker.shutdown().await;
    }
}

#[tokio::test]
async fn incomplete_local_preview_can_fail_then_recover_on_a_new_explicit_grab() {
    use puripuly_heart_overlay::photo::prepare::SourceRequest;
    let directory = Directory::new("preview-recover");
    let request = directory.request(320, 180);
    let full = std::fs::read(&request.source_path).unwrap();
    std::fs::write(&request.source_path, b"incomplete").unwrap();
    let local = SourceRequest {
        photo_id: request.photo_id.clone(),
        source_path: request.source_path.clone(),
        expected_file_id: request.expected_file_id.clone(),
        max_size: [320, 180],
    };
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_source(local.clone()).unwrap();
    let PreparationCompletion::Source(completion) = worker.next_result().await.unwrap() else { panic!("Local failure completion required") };
    assert_eq!(completion.result.err(), Some("source_incomplete"));
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    std::fs::write(&request.source_path, full).unwrap();
    worker.start_source(local).unwrap();
    let PreparationCompletion::Source(completion) = worker.next_result().await.unwrap() else { panic!("Local recovery completion required") };
    assert_eq!(completion.result.unwrap().original.dimensions(), (320, 180));
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    worker.shutdown().await;
}

#[tokio::test]
async fn metadata_reads_complete_png_headers_without_pixel_payload_or_preparation() {
    for (name, width, height) in [("metadata-landscape", 640, 360), ("metadata-portrait", 360, 640)] {
        let directory = Directory::new(name);
        let request = directory.request(width, height);
        let full = std::fs::read(&request.source_path).unwrap();
        let header = png_header(&full);
        assert!(puripuly_heart_overlay::photo::prepare::decode_source(header).is_err());
        std::fs::write(&request.source_path, header).unwrap();
        let local = local_request(&request);
        let mut worker = PreparationWorker::new().unwrap();
        worker.start_metadata(local.clone()).unwrap();
        let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
        let PreparationCompletion::Metadata(completion) = completion else { panic!("Header metadata required") };
        assert_eq!(completion.request, local);
        assert_eq!(completion.result, Ok([width, height]));
        assert!(worker.try_result().is_none());
        assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
        worker.shutdown().await;
    }
}

#[tokio::test]
async fn metadata_reads_landscape_and_portrait_jpeg_headers_despite_png_extension() {
    for (name, width, height) in [("metadata-jpeg-landscape", 640, 360), ("metadata-jpeg-portrait", 360, 640)] {
        let directory = Directory::new(name);
        let request = directory.request(width, height);
        let bytes = jpeg_bytes(width, height);
        std::fs::write(&request.source_path, jpeg_header(&bytes)).unwrap();
        let local = local_request(&request);
        let mut worker = PreparationWorker::new().unwrap();
        worker.start_metadata(local.clone()).unwrap();
        let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
        let PreparationCompletion::Metadata(completion) = completion else { panic!("JPEG header metadata required") };
        assert_eq!(completion.request, local);
        assert_eq!(completion.result, Ok([width, height]));
        assert!(worker.try_result().is_none());
        assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
        worker.shutdown().await;
    }
}

#[tokio::test]
async fn metadata_waits_for_header_completion_and_bounds_incomplete_header_waits() {
    let directory = Directory::new("metadata-incomplete");
    let request = directory.request(360, 640);
    let full = std::fs::read(&request.source_path).unwrap();
    let header = png_header(&full).to_vec();
    std::fs::write(&request.source_path, &header[..24]).unwrap();
    let source = request.source_path.clone();
    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(source, header).unwrap();
    });
    let local = local_request(&request);
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_metadata(local.clone()).unwrap();
    let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
    let PreparationCompletion::Metadata(completion) = completion else { panic!("Completed header metadata required") };
    writer.join().unwrap();
    assert_eq!(completion.result, Ok([360, 640]));
    std::fs::write(&request.source_path, b"incomplete").unwrap();
    let start = std::time::Instant::now();
    worker.start_metadata(local).unwrap();
    let completion = tokio::time::timeout(Duration::from_secs(5), worker.next_result()).await.unwrap().unwrap();
    let PreparationCompletion::Metadata(completion) = completion else { panic!("Bounded metadata failure required") };
    assert_eq!(completion.result, Err("source_incomplete"));
    assert!(start.elapsed() >= Duration::from_secs(2));
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    worker.shutdown().await;
}

#[tokio::test]
async fn metadata_rejects_wrong_identity_and_replacement_while_waiting() {
    let directory = Directory::new("metadata-identity");
    let request = directory.request(360, 640);
    let mut local = local_request(&request);
    local.expected_file_id = ["wrong".into(), "identity".into()];
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_metadata(local).unwrap();
    let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
    let PreparationCompletion::Metadata(completion) = completion else { panic!("Identity failure required") };
    assert_eq!(completion.result, Err("source_replaced"));
    let full = std::fs::read(&request.source_path).unwrap();
    std::fs::write(&request.source_path, &full[..24]).unwrap();
    worker.start_metadata(local_request(&request)).unwrap();
    tokio::time::sleep(Duration::from_millis(70)).await;
    std::fs::rename(&request.source_path, directory.0.join("old.png")).unwrap();
    RgbImage::new(640, 360).save(&request.source_path).unwrap();
    let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
    let PreparationCompletion::Metadata(completion) = completion else { panic!("Replacement failure required") };
    assert_eq!(completion.result, Err("source_replaced"));
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    worker.shutdown().await;
}

#[tokio::test]
async fn metadata_cancellation_replacement_and_shutdown_remain_bounded() {
    let directory = Directory::new("metadata-cancel");
    let request = directory.request(360, 640);
    std::fs::write(&request.source_path, b"incomplete").unwrap();
    let local = local_request(&request);
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_metadata(local.clone()).unwrap();
    worker.cancel_metadata();
    let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
    let PreparationCompletion::Metadata(completion) = completion else { panic!("Metadata cancellation required") };
    assert_eq!(completion.result, Err("preparation_cancelled"));
    worker.start_metadata(local.clone()).unwrap();
    tokio::time::sleep(Duration::from_millis(70)).await;
    let replacement = Directory::new("metadata-new-offer");
    let replacement_request = replacement.request(640, 360);
    let mut latest = local_request(&replacement_request);
    latest.photo_id = "replacement-photo".into();
    worker.start_metadata(latest.clone()).unwrap();
    loop {
        let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
        let PreparationCompletion::Metadata(completion) = completion else { panic!("Metadata replacement completion required") };
        if completion.request == latest {
            assert_eq!(completion.result, Ok([640, 360]));
            break;
        }
        assert_eq!(completion.request, local);
        assert_eq!(completion.result, Err("preparation_cancelled"));
    }
    worker.start_metadata(local).unwrap();
    worker.shutdown().await;
    let PreparationCompletion::Metadata(completion) = worker.next_result().await.unwrap() else { panic!("Shutdown metadata cancellation required") };
    assert_eq!(completion.result, Err("preparation_cancelled"));
    assert!(worker.next_result().await.is_none());
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    assert!(std::fs::read_dir(&replacement_request.output_dir).unwrap().next().is_none());
}

#[tokio::test]
async fn fast_grab_receives_header_metadata_before_incomplete_pixel_decode_settles() {
    let directory = Directory::new("metadata-fast-grab");
    let request = directory.request(360, 640);
    let full = std::fs::read(&request.source_path).unwrap();
    std::fs::write(&request.source_path, png_header(&full)).unwrap();
    let local = local_request(&request);
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_metadata(local.clone()).unwrap();
    worker.start_source(local.clone()).unwrap();
    loop {
        let completion = tokio::time::timeout(Duration::from_secs(1), worker.next_result()).await.unwrap().unwrap();
        let PreparationCompletion::Metadata(completion) = completion else { panic!("Dimensions must precede pixel decoding") };
        assert_eq!(completion.request, local);
        match completion.result {
            Ok(dimensions) => {
                assert_eq!(dimensions, [360, 640]);
                break;
            }
            Err(code) => assert_eq!(code, "preparation_cancelled"),
        }
    }
    worker.shutdown().await;
    let mut source_cancelled = false;
    while let Some(completion) = worker.next_result().await {
        match completion {
            PreparationCompletion::Metadata(completion) => {
                assert_eq!(completion.request, local);
                match completion.result {
                    Ok(dimensions) => assert_eq!(dimensions, [360, 640]),
                    Err(code) => assert_eq!(code, "preparation_cancelled"),
                }
            }
            PreparationCompletion::Source(completion) => {
                assert_eq!(completion.request, local);
                assert_eq!(completion.result.err(), Some("preparation_cancelled"));
                source_cancelled = true;
            }
            PreparationCompletion::Prepared(_) => panic!("Header metadata must not authorize preparation"),
        }
    }
    assert!(source_cancelled);
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
}

#[tokio::test]
async fn metadata_replacements_cannot_fill_the_accepted_grab_and_preparation_queue() {
    let pending = Directory::new("metadata-priority-pending");
    let pending_request = pending.request(360, 640);
    std::fs::write(&pending_request.source_path, b"incomplete").unwrap();
    let directory = Directory::new("metadata-priority-grab");
    let request = directory.request(320, 180);
    let mut worker = PreparationWorker::new().unwrap();
    for index in 0..64 {
        let mut local = local_request(&pending_request);
        local.photo_id = format!("pending-{index}");
        worker.start_metadata(local).unwrap();
    }
    worker.start_source(local_request(&request)).unwrap();
    worker.start(request.clone()).unwrap();
    let source = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            match worker.next_result().await.unwrap() {
                PreparationCompletion::Metadata(completion) => assert_eq!(completion.result, Err("preparation_cancelled")),
                PreparationCompletion::Source(completion) => break completion.result.unwrap(),
                PreparationCompletion::Prepared(_) => panic!("Accepted source must precede preparation"),
            }
        }
    }).await.unwrap();
    let prepared = loop {
        match worker.next_result().await.unwrap() {
            PreparationCompletion::Metadata(completion) => assert_eq!(completion.result, Err("preparation_cancelled")),
            PreparationCompletion::Prepared(completion) => break completion.result.unwrap(),
            PreparationCompletion::Source(_) => panic!("Only one source grab was requested"),
        }
    };
    assert!(Arc::ptr_eq(&source.original, &prepared.original));
    assert_eq!(prepared.decode_ms, 0.0);
    worker.shutdown().await;
}

#[tokio::test]
async fn header_metadata_does_not_cache_or_select_original_pixels() {
    let directory = Directory::new("metadata-no-pixels");
    let request = directory.request(360, 640);
    let full = std::fs::read(&request.source_path).unwrap();
    std::fs::write(&request.source_path, png_header(&full)).unwrap();
    let local = local_request(&request);
    let mut worker = PreparationWorker::new().unwrap();
    worker.start_metadata(local.clone()).unwrap();
    let PreparationCompletion::Metadata(completion) = worker.next_result().await.unwrap() else { panic!("Header metadata required") };
    assert_eq!(completion.result, Ok([360, 640]));
    RgbImage::from_pixel(640, 360, Rgb([20, 40, 60])).save(&request.source_path).unwrap();
    assert_eq!(file_identity(&File::open(&request.source_path).unwrap()).unwrap(), request.expected_file_id);
    worker.start_source(local.clone()).unwrap();
    let PreparationCompletion::Metadata(completion) = worker.next_result().await.unwrap() else { panic!("Grab must publish current header before pixels") };
    assert_eq!(completion.result, Ok([640, 360]));
    let PreparationCompletion::Source(completion) = worker.next_result().await.unwrap() else { panic!("Explicit grab source required") };
    let source = completion.result.unwrap();
    assert_eq!(source.original.dimensions(), (640, 360));
    assert_eq!(*source.original.get_pixel(0, 0), Rgb([20, 40, 60]));
    assert!(std::fs::read_dir(&request.output_dir).unwrap().next().is_none());
    worker.shutdown().await;
}
