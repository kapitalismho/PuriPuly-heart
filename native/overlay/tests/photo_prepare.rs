use image::{Rgb, RgbImage};
use puripuly_heart_overlay::photo::prepare::{
    file_identity, prepare, PreparationWorker, PrepareRequest,
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

#[test]
fn native_preparation_preserves_original_and_maps_supported_sizes_and_aspects() {
    for (name, width, height, scale, offset_x, offset_y) in [
        ("720", 1280, 720, 1.0, 0, 0),
        ("1080", 1920, 1080, 2.0 / 3.0, 0, 0),
        ("1440", 2560, 1440, 0.5, 0, 0),
        ("2160", 3840, 2160, 1.0 / 3.0, 0, 0),
        ("portrait", 360, 720, 1.0, 460, 0),
        ("wide", 1920, 480, 2.0 / 3.0, 0, 200),
        ("custom", 853, 601, 720.0 / 601.0, 129, 0),
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
        assert_eq!(jpeg.dimensions(), (1280, 720));
        let center = jpeg.get_pixel(640, 360);
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
    let result = worker.next_result().await.unwrap();
    assert_eq!(result.result.err(), Some("preparation_cancelled"));
    assert!(!directory.0.join("output/source.png").exists());
}

#[test]
fn shared_source_decoder_advances_jpeg_recovery_and_preparation_without_extension_dependency() {
    use puripuly_heart_overlay::photo::prepare::{decode_source, recover_original};
    let directory = Directory::new("shared-jpeg");
    let mut request = directory.request(320, 180);
    let pixels = RgbImage::from_pixel(320, 180, Rgb([200, 90, 35]));
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 95)
        .encode_image(&pixels)
        .unwrap();
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
