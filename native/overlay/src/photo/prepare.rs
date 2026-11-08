use image::{codecs::jpeg::JpegEncoder, imageops, Rgb, RgbImage};
use serde::Deserialize;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::time::{Duration, Instant, SystemTime};
use super::processing::WorkingPhoto;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareRequest {
    pub photo_id: String,
    pub attempt_id: u64,
    pub source_path: String,
    pub output_dir: String,
    pub expected_file_id: [String; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRequest {
    pub photo_id: String,
    pub source_path: String,
    pub expected_file_id: [String; 2],
    pub max_size: [u32; 2],
}

#[derive(PartialEq, Eq)]
struct SourceSignature {
    length: u64,
    modified: SystemTime,
}

impl SourceSignature {
    fn read(file: &File) -> Result<Self, &'static str> {
        let metadata = file.metadata().map_err(|_| "source_unavailable")?;
        Ok(Self {
            length: metadata.len(),
            modified: metadata.modified().map_err(|_| "source_unavailable")?,
        })
    }
}

pub struct LoadedSource {
    pub original: Arc<RgbImage>,
    pub working: Option<Arc<WorkingPhoto>>,
    bytes: Vec<u8>,
    signature: SourceSignature,
    decode_ms: f64,
}

pub struct SourceCompletion {
    pub request: SourceRequest,
    pub result: Result<Arc<LoadedSource>, &'static str>,
}

pub enum PreparationCompletion {
    Source(SourceCompletion),
    Prepared(PrepareCompletion),
}

pub struct PreparedPhoto {
    pub source_path: PathBuf,
    pub jpeg_path: PathBuf,
    pub original: Arc<RgbImage>,
    pub working: Option<Arc<WorkingPhoto>>,
    pub scale: f64,
    pub offset_x: u32,
    pub offset_y: u32,
    pub preparation_ms: f64,
    pub decode_ms: f64,
    pub resize_ms: f64,
    pub jpeg_ms: f64,
    handed_off: bool,
}

impl PreparedPhoto {
    pub fn hand_off(&mut self) {
        self.handed_off = true;
    }
}

impl Drop for PreparedPhoto {
    fn drop(&mut self) {
        if !self.handed_off {
            let _ = std::fs::remove_file(&self.source_path);
            let _ = std::fs::remove_file(&self.jpeg_path);
        }
    }
}

pub struct PrepareCompletion {
    pub request: PrepareRequest,
    pub result: Result<PreparedPhoto, &'static str>,
}

enum Work {
    Source(SourceRequest, Arc<AtomicBool>),
    Prepare(PrepareRequest, Arc<AtomicBool>),
    Clear,
}

pub struct PreparationWorker {
    sender: Option<mpsc::SyncSender<Work>>,
    results: tokio::sync::mpsc::UnboundedReceiver<PreparationCompletion>,
    thread: Option<std::thread::JoinHandle<()>>,
    cancellation: Option<Arc<AtomicBool>>,
    source_cancellation: Option<(SourceRequest, Arc<AtomicBool>)>,
}

impl PreparationWorker {
    pub fn new() -> Result<Self, &'static str> {
        let (sender, receiver) = mpsc::sync_channel::<Work>(2);
        let (completed, results) = tokio::sync::mpsc::unbounded_channel();
        let thread = std::thread::Builder::new()
            .name("photo-image".into())
            .spawn(move || {
                let mut cached: Option<(SourceRequest, Arc<LoadedSource>, Arc<AtomicBool>)> = None;
                while let Ok(work) = receiver.recv() {
                    if cached.as_ref().is_some_and(|(_, _, cancel)| cancel.load(Ordering::Relaxed)) {
                        cached = None;
                    }
                    match work {
                        Work::Source(request, cancel) => {
                            let result = if let Some((_, source, _)) = cached.as_ref().filter(|(key, _, _)| key == &request) {
                                Ok(source.clone())
                            } else {
                                load_source(&request.source_path, &request.expected_file_id, &cancel)
                                    .and_then(|mut source| {
                                        source.working = Some(Arc::new(
                                            WorkingPhoto::new(source.original.clone(), request.max_size)
                                                .map_err(|_| "source_resize_failed")?
                                        ));
                                        cancelled(&cancel)?;
                                        Ok(Arc::new(source))
                                    })
                            };
                            if let Ok(source) = &result {
                                cached = Some((request.clone(), source.clone(), cancel));
                            }
                            if completed.send(PreparationCompletion::Source(SourceCompletion { request, result })).is_err() {
                                break;
                            }
                        }
                        Work::Prepare(request, cancel) => {
                            let result = if let Some((key, source, _)) = cached.as_ref().filter(|(key, _, _)| key.photo_id == request.photo_id) {
                                if key.source_path != request.source_path || key.expected_file_id != request.expected_file_id {
                                    Err("source_replaced")
                                } else {
                                    prepare_loaded(&request, &cancel, source, true, Instant::now())
                                }
                            } else {
                                prepare(&request, &cancel)
                            };
                            if completed.send(PreparationCompletion::Prepared(PrepareCompletion { request, result })).is_err() {
                                break;
                            }
                        }
                        Work::Clear => cached = None,
                    }
                }
            })
            .map_err(|_| "worker_unavailable")?;
        Ok(Self {
            sender: Some(sender),
            results,
            thread: Some(thread),
            cancellation: None,
            source_cancellation: None,
        })
    }

    pub fn start(&mut self, request: PrepareRequest) -> Result<(), &'static str> {
        self.cancel_prepare();
        let cancel = Arc::new(AtomicBool::new(false));
        self.sender
            .as_ref()
            .ok_or("worker_unavailable")?
            .try_send(Work::Prepare(request, cancel.clone()))
            .map_err(|_| "preparation_busy")?;
        self.cancellation = Some(cancel);
        Ok(())
    }

    pub fn start_source(&mut self, request: SourceRequest) -> Result<(), &'static str> {
        let cancel = match &self.source_cancellation {
            Some((key, cancel)) if key == &request => cancel.clone(),
            _ => {
                self.cancel_source();
                let cancel = Arc::new(AtomicBool::new(false));
                self.source_cancellation = Some((request.clone(), cancel.clone()));
                cancel
            }
        };
        self.sender.as_ref().ok_or("worker_unavailable")?
            .try_send(Work::Source(request, cancel))
            .map_err(|_| "preparation_busy")
    }

    pub fn cancel_source(&mut self) {
        if let Some((_, cancel)) = self.source_cancellation.take() {
            cancel.store(true, Ordering::Relaxed);
            if let Some(sender) = &self.sender {
                let _ = sender.try_send(Work::Clear);
            }
        }
    }

    pub fn cancel(&mut self) {
        self.cancel_prepare();
        self.cancel_source();
    }

    pub fn cancel_prepare(&mut self) {
        if let Some(cancel) = self.cancellation.take() {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    pub async fn next_result(&mut self) -> Option<PreparationCompletion> {
        self.results.recv().await
    }

    pub fn try_result(&mut self) -> Option<PreparationCompletion> {
        self.results.try_recv().ok()
    }

    pub async fn shutdown(&mut self) {
        self.cancel();
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
    }
}

impl Drop for PreparationWorker {
    fn drop(&mut self) {
        self.cancel();
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn cancelled(cancel: &AtomicBool) -> Result<(), &'static str> {
    if cancel.load(Ordering::Relaxed) {
        Err("preparation_cancelled")
    } else {
        Ok(())
    }
}

pub fn file_identity(file: &File) -> Result<[String; 2], &'static str> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::Storage::FileSystem::{
            FileIdInfo, GetFileInformationByHandleEx, FILE_ID_INFO,
        };
        let mut info = FILE_ID_INFO::default();
        unsafe {
            GetFileInformationByHandleEx(
                HANDLE(file.as_raw_handle()),
                FileIdInfo,
                (&mut info as *mut FILE_ID_INFO).cast(),
                std::mem::size_of::<FILE_ID_INFO>() as u32,
            )
        }
        .map_err(|_| "source_identity_unavailable")?;
        Ok([
            info.VolumeSerialNumber.to_string(),
            u128::from_le_bytes(info.FileId.Identifier).to_string(),
        ])
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata().map_err(|_| "source_unavailable")?;
        Ok([metadata.dev().to_string(), metadata.ino().to_string()])
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = file;
        Err("source_identity_unavailable")
    }
}

fn read_source(path: &str, expected_file_id: &[String; 2]) -> Result<(Vec<u8>, SourceSignature), &'static str> {
    let mut file = File::open(path).map_err(|_| "source_unavailable")?;
    if file_identity(&file)? != *expected_file_id {
        return Err("source_replaced");
    }
    let before = SourceSignature::read(&file)?;
    let mut bytes = Vec::with_capacity(usize::try_from(before.length).map_err(|_| "source_invalid")?);
    file.read_to_end(&mut bytes).map_err(|_| "source_unavailable")?;
    let after = SourceSignature::read(&file)?;
    let current = File::open(path).map_err(|_| "source_unavailable")?;
    if file_identity(&file)? != *expected_file_id || file_identity(&current)? != *expected_file_id {
        return Err("source_replaced");
    }
    if before != after || after != SourceSignature::read(&current)? || bytes.len() as u64 != after.length {
        return Err("source_incomplete");
    }
    Ok((bytes, after))
}

fn revalidate_source(request: &PrepareRequest, source: &LoadedSource) -> Result<(), &'static str> {
    let file = File::open(&request.source_path).map_err(|_| "source_unavailable")?;
    let current = File::open(&request.source_path).map_err(|_| "source_unavailable")?;
    if file_identity(&file)? != request.expected_file_id || file_identity(&current)? != request.expected_file_id {
        return Err("source_replaced");
    }
    if SourceSignature::read(&file)? != source.signature || SourceSignature::read(&current)? != source.signature {
        return Err("source_changed");
    }
    Ok(())
}

fn load_source(path: &str, expected_file_id: &[String; 2], cancel: &AtomicBool) -> Result<LoadedSource, &'static str> {
    let start = Instant::now();
    loop {
        cancelled(cancel)?;
        let outcome = read_source(path, expected_file_id).and_then(|(bytes, signature)| {
            let decode_start = Instant::now();
            let original = Arc::new(decode_source(&bytes)?);
            Ok(LoadedSource {
                original,
                working: None,
                bytes,
                signature,
                decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
            })
        });
        match outcome {
            Ok(value) => return Ok(value),
            Err("source_replaced") => return Err("source_replaced"),
            Err(code) if start.elapsed() >= Duration::from_secs(2) => return Err(code),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

pub fn decode_source(bytes: &[u8]) -> Result<RgbImage, &'static str> {
    image::load_from_memory(bytes)
        .map(|image| image.into_rgb8())
        .map_err(|_| "source_incomplete")
}

pub fn recover_original(path: &Path) -> Result<Arc<RgbImage>, &'static str> {
    let bytes = std::fs::read(path).map_err(|_| "source_unavailable")?;
    decode_source(&bytes).map(Arc::new)
}

pub fn prepare(
    request: &PrepareRequest,
    cancel: &AtomicBool,
) -> Result<PreparedPhoto, &'static str> {
    let start = Instant::now();
    let source = load_source(&request.source_path, &request.expected_file_id, cancel)?;
    prepare_loaded(request, cancel, &source, false, start)
}

fn prepare_loaded(
    request: &PrepareRequest,
    cancel: &AtomicBool,
    source: &LoadedSource,
    reused: bool,
    start: Instant,
) -> Result<PreparedPhoto, &'static str> {
    cancelled(cancel)?;
    revalidate_source(request, source)?;
    let original = &source.original;
    let source_path = Path::new(&request.output_dir).join("source.png");
    let jpeg_path = Path::new(&request.output_dir).join("send.jpg");
    let mut source_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source_path)
        .map_err(|_| "output_unavailable")?;
    let mut jpeg_created = false;
    let result = (|| {
        source_file.write_all(&source.bytes).map_err(|_| "output_unavailable")?;
        drop(source_file);
        let resize_start = Instant::now();
        let scale =
            (1280.0 / f64::from(original.width())).min(720.0 / f64::from(original.height()));
        let width = (f64::from(original.width()) * scale)
            .round_ties_even()
            .clamp(1.0, 1280.0) as u32;
        let height = (f64::from(original.height()) * scale)
            .round_ties_even()
            .clamp(1.0, 720.0) as u32;
        let offset_x = (1280 - width) / 2;
        let offset_y = (720 - height) / 2;
        let canvas = if original.width() == 1280 && original.height() == 720 {
            None
        } else {
            let resized =
                imageops::resize(original.as_ref(), width, height, imageops::FilterType::Lanczos3);
            if width == 1280 && height == 720 {
                Some(resized)
            } else {
                let mut canvas = RgbImage::from_pixel(1280, 720, Rgb([16, 18, 24]));
                imageops::replace(
                    &mut canvas,
                    &resized,
                    i64::from(offset_x),
                    i64::from(offset_y),
                );
                Some(canvas)
            }
        };
        let resize_ms = resize_start.elapsed().as_secs_f64() * 1000.0;
        cancelled(cancel)?;
        let jpeg_start = Instant::now();
        let jpeg = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&jpeg_path)
            .map_err(|_| "output_unavailable")?;
        jpeg_created = true;
        JpegEncoder::new_with_quality(jpeg, 85)
            .encode_image(canvas.as_ref().unwrap_or(original.as_ref()))
            .map_err(|_| "jpeg_failed")?;
        let jpeg_ms = jpeg_start.elapsed().as_secs_f64() * 1000.0;
        cancelled(cancel)?;
        revalidate_source(request, source)?;
        Ok(PreparedPhoto {
            source_path: source_path.clone(),
            jpeg_path: jpeg_path.clone(),
            original: original.clone(),
            working: source.working.clone(),
            scale,
            offset_x,
            offset_y,
            preparation_ms: start.elapsed().as_secs_f64() * 1000.0,
            decode_ms: if reused { 0.0 } else { source.decode_ms },
            resize_ms,
            jpeg_ms,
            handed_off: false,
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(source_path);
        if jpeg_created {
            let _ = std::fs::remove_file(jpeg_path);
        }
    }
    result
}
