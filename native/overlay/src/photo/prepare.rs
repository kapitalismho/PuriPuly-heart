use image::{codecs::jpeg::JpegEncoder, imageops, ImageFormat, Rgb, RgbImage};
use serde::Deserialize;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareRequest {
    pub photo_id: String,
    pub attempt_id: u64,
    pub source_path: String,
    pub output_dir: String,
    pub expected_file_id: [String; 2],
}

pub struct PreparedPhoto {
    pub source_path: PathBuf,
    pub jpeg_path: PathBuf,
    pub original: Arc<RgbImage>,
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

struct Work {
    request: PrepareRequest,
    cancel: Arc<AtomicBool>,
}

pub struct PreparationWorker {
    sender: Option<mpsc::SyncSender<Work>>,
    results: tokio::sync::mpsc::UnboundedReceiver<PrepareCompletion>,
    thread: Option<std::thread::JoinHandle<()>>,
    cancellation: Option<Arc<AtomicBool>>,
}

impl PreparationWorker {
    pub fn new() -> Result<Self, &'static str> {
        let (sender, receiver) = mpsc::sync_channel::<Work>(1);
        let (completed, results) = tokio::sync::mpsc::unbounded_channel();
        let thread = std::thread::Builder::new()
            .name("photo-image".into())
            .spawn(move || {
                while let Ok(work) = receiver.recv() {
                    let result = prepare(&work.request, &work.cancel);
                    if completed
                        .send(PrepareCompletion {
                            request: work.request,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|_| "worker_unavailable")?;
        Ok(Self {
            sender: Some(sender),
            results,
            thread: Some(thread),
            cancellation: None,
        })
    }

    pub fn start(&mut self, request: PrepareRequest) -> Result<(), &'static str> {
        self.cancel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.sender
            .as_ref()
            .ok_or("worker_unavailable")?
            .try_send(Work {
                request,
                cancel: cancel.clone(),
            })
            .map_err(|_| "preparation_busy")?;
        self.cancellation = Some(cancel);
        Ok(())
    }

    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancellation.take() {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    pub async fn next_result(&mut self) -> Option<PrepareCompletion> {
        self.results.recv().await
    }

    pub fn try_result(&mut self) -> Option<PrepareCompletion> {
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

fn read_source(request: &PrepareRequest) -> Result<Vec<u8>, &'static str> {
    let mut file = File::open(&request.source_path).map_err(|_| "source_unavailable")?;
    if file_identity(&file)? != request.expected_file_id {
        return Err("source_replaced");
    }
    let before = file.metadata().map_err(|_| "source_unavailable")?;
    let mut bytes =
        Vec::with_capacity(usize::try_from(before.len()).map_err(|_| "source_invalid")?);
    file.read_to_end(&mut bytes)
        .map_err(|_| "source_unavailable")?;
    let after = file.metadata().map_err(|_| "source_unavailable")?;
    let current = File::open(&request.source_path).map_err(|_| "source_unavailable")?;
    if file_identity(&file)? != request.expected_file_id
        || file_identity(&current)? != request.expected_file_id
    {
        return Err("source_replaced");
    }
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("source_incomplete");
    }
    Ok(bytes)
}

pub fn prepare(
    request: &PrepareRequest,
    cancel: &AtomicBool,
) -> Result<PreparedPhoto, &'static str> {
    let start = Instant::now();
    let (bytes, original, decode_ms) = loop {
        cancelled(cancel)?;
        let outcome = read_source(request).and_then(|bytes| {
            let decode_start = Instant::now();
            let image = image::load_from_memory_with_format(&bytes, ImageFormat::Png)
                .map_err(|_| "source_incomplete")?
                .into_rgb8();
            Ok((bytes, image, decode_start.elapsed().as_secs_f64() * 1000.0))
        });
        match outcome {
            Ok(value) => break value,
            Err("source_replaced") => return Err("source_replaced"),
            Err(code) if start.elapsed() >= Duration::from_secs(2) => return Err(code),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    cancelled(cancel)?;
    let source_path = Path::new(&request.output_dir).join("source.png");
    let jpeg_path = Path::new(&request.output_dir).join("send.jpg");
    let mut source = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source_path)
        .map_err(|_| "output_unavailable")?;
    let mut jpeg_created = false;
    let result = (|| {
        source.write_all(&bytes).map_err(|_| "output_unavailable")?;
        drop(source);
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
                imageops::resize(&original, width, height, imageops::FilterType::Lanczos3);
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
            .encode_image(canvas.as_ref().unwrap_or(&original))
            .map_err(|_| "jpeg_failed")?;
        let jpeg_ms = jpeg_start.elapsed().as_secs_f64() * 1000.0;
        cancelled(cancel)?;
        Ok(PreparedPhoto {
            source_path: source_path.clone(),
            jpeg_path: jpeg_path.clone(),
            original: Arc::new(original),
            scale,
            offset_x,
            offset_y,
            preparation_ms: start.elapsed().as_secs_f64() * 1000.0,
            decode_ms,
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
