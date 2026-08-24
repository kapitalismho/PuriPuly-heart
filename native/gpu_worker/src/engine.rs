use serde::Serialize;
use std::path::Path;
use std::time::{Duration, Instant};
use transcribe_cpp::{
    backend_available, devices, init_backends_default, Backend, CancelToken, DeviceType, Error,
    Feature, Model, ModelOptions, RunOptions, Session, TimestampKind,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct VulkanDevice {
    pub device_id: String,
    pub registry_index: usize,
    pub name: String,
    pub description: String,
    pub device_type: String,
    pub memory_total_bytes: u64,
    pub memory_free_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivationResult {
    pub device: VulkanDevice,
    pub model_load_seconds: f64,
    pub warmup_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub detected_language: Option<String>,
    pub audio_seconds: f64,
    pub decode_seconds: f64,
    pub rtf: f64,
}

#[derive(Debug)]
pub struct TranscriptionFailure {
    pub error: EngineError,
    pub audio_seconds: Option<f64>,
    pub decode_seconds: f64,
    pub attempt_started: bool,
}

impl TranscriptionFailure {
    pub fn started_timing(&self) -> Option<(f64, f64, f64)> {
        let audio_seconds = self.audio_seconds?;
        if !self.attempt_started
            || !audio_seconds.is_finite()
            || audio_seconds <= 0.0
            || !self.decode_seconds.is_finite()
            || self.decode_seconds < 0.0
        {
            return None;
        }
        let rtf = self.decode_seconds / audio_seconds;
        rtf.is_finite()
            .then_some((audio_seconds, self.decode_seconds, rtf))
    }
}

#[derive(Debug)]
struct StartedDecode<T> {
    value: T,
    decode_seconds: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("unsupported_capability")]
    UnsupportedCapability,
    #[error("device_unavailable")]
    DeviceUnavailable,
    #[error("model_missing")]
    ModelMissing,
    #[error("model_invalid")]
    ModelInvalid,
    #[error("strict_vulkan_rejected")]
    StrictVulkanRejected,
    #[error("warmup_failed")]
    WarmupFailed,
    #[error("audio_invalid")]
    AudioInvalid,
    #[error("cancelled")]
    Cancelled,
    #[error("out_of_memory")]
    OutOfMemory,
    #[error("backend_failure")]
    BackendFailure,
    #[error("decode_failure")]
    DecodeFailure,
}

impl EngineError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedCapability => "unsupported_capability",
            Self::DeviceUnavailable => "device_unavailable",
            Self::ModelMissing => "model_missing",
            Self::ModelInvalid => "model_invalid",
            Self::StrictVulkanRejected => "strict_vulkan_rejected",
            Self::WarmupFailed => "warmup_failed",
            Self::AudioInvalid => "audio_invalid",
            Self::Cancelled => "cancelled",
            Self::OutOfMemory => "out_of_memory",
            Self::BackendFailure => "backend_failure",
            Self::DecodeFailure => "decode_failure",
        }
    }
}

#[derive(Default)]
pub struct GpuEngine {
    model: Option<Model>,
    session: Option<Session>,
    selected_device: Option<VulkanDevice>,
}

const DISCOVERY_ATTEMPTS: u32 = 8;
const DISCOVERY_RETRY_DELAY: Duration = Duration::from_millis(250);

impl GpuEngine {
    pub fn discover() -> Result<Vec<VulkanDevice>, EngineError> {
        retry_discovery(
            || {
                init_backends_default().map_err(map_backend_error)?;
                if !backend_available(Backend::Vulkan) {
                    return Ok(Vec::new());
                }
                Ok(enumerate_vulkan_devices())
            },
            DISCOVERY_ATTEMPTS,
            DISCOVERY_RETRY_DELAY,
        )
    }

    pub fn activate<F>(
        &mut self,
        model_path: &Path,
        device_id: &str,
        cancel_token: &CancelToken,
        mut progress: F,
    ) -> Result<ActivationResult, EngineError>
    where
        F: FnMut(&'static str, f64),
    {
        self.unload();
        progress("validating", 0.1);
        if !model_path.is_file() {
            return Err(EngineError::ModelMissing);
        }
        if !model_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
        {
            return Err(EngineError::ModelInvalid);
        }
        let available_devices = Self::discover()?;
        let selected_device = if device_id == "auto" {
            available_devices
                .first()
                .cloned()
                .ok_or(EngineError::UnsupportedCapability)?
        } else {
            available_devices
                .iter()
                .find(|device| device.device_id == device_id)
                .cloned()
                .ok_or(EngineError::DeviceUnavailable)?
        };

        progress("loading", 0.25);
        let load_started = Instant::now();
        let model = Model::load_with(
            model_path,
            &ModelOptions {
                backend: Backend::Vulkan,
                gpu_device: selected_device.registry_index as i32,
            },
        )
        .map_err(map_model_load_error)?;
        if !model.backend().to_ascii_lowercase().starts_with("vulkan") {
            return Err(EngineError::StrictVulkanRejected);
        }
        if !model.supports(Feature::Cancellation) {
            return Err(EngineError::StrictVulkanRejected);
        }
        let model_load_seconds = load_started.elapsed().as_secs_f64();
        let mut session = model.session().map_err(map_backend_error)?;
        session.set_cancel_token(cancel_token);

        progress("warming", 0.75);
        let warmup_started = Instant::now();
        let mut run_options = RunOptions::default();
        run_options.timestamps = TimestampKind::None;
        session
            .run(&vec![0.0_f32; 16_000], &run_options)
            .map_err(map_warmup_error)?;
        let warmup_seconds = warmup_started.elapsed().as_secs_f64();
        progress("ready", 1.0);
        self.selected_device = Some(selected_device.clone());
        self.model = Some(model);
        self.session = Some(session);
        Ok(ActivationResult {
            device: selected_device,
            model_load_seconds,
            warmup_seconds,
        })
    }

    pub fn transcribe<F>(
        &mut self,
        audio_path: &Path,
        language_hint: Option<String>,
        cancel_token: &CancelToken,
        on_decode_started: F,
    ) -> Result<TranscriptionResult, TranscriptionFailure>
    where
        F: FnOnce(f64),
    {
        let samples = read_wav(audio_path).map_err(|error| TranscriptionFailure {
            error,
            audio_seconds: None,
            decode_seconds: 0.0,
            attempt_started: false,
        })?;
        let audio_seconds = samples.len() as f64 / 16_000.0;
        if audio_seconds <= 0.0 {
            return Err(TranscriptionFailure {
                error: EngineError::AudioInvalid,
                audio_seconds: Some(audio_seconds),
                decode_seconds: 0.0,
                attempt_started: false,
            });
        }
        let session = self.session.as_mut().ok_or(TranscriptionFailure {
            error: EngineError::BackendFailure,
            audio_seconds: Some(audio_seconds),
            decode_seconds: 0.0,
            attempt_started: false,
        })?;
        session.set_cancel_token(cancel_token);
        let mut run_options = RunOptions::default();
        run_options.timestamps = TimestampKind::None;
        run_options.language = language_hint;
        let decoded = execute_started_decode(audio_seconds, on_decode_started, || {
            session
                .run(&samples, &run_options)
                .map_err(map_decode_error)
        })?;
        Ok(TranscriptionResult {
            text: decoded.value.text,
            detected_language: decoded.value.language,
            audio_seconds,
            decode_seconds: decoded.decode_seconds,
            rtf: decoded.decode_seconds / audio_seconds,
        })
    }

    pub fn unload(&mut self) {
        self.session = None;
        self.model = None;
        self.selected_device = None;
    }
}

fn execute_started_decode<T, F, S>(
    audio_seconds: f64,
    on_decode_started: S,
    decode: F,
) -> Result<StartedDecode<T>, TranscriptionFailure>
where
    F: FnOnce() -> Result<T, EngineError>,
    S: FnOnce(f64),
{
    on_decode_started(audio_seconds);
    let started = Instant::now();
    let value = decode().map_err(|error| TranscriptionFailure {
        error,
        audio_seconds: Some(audio_seconds),
        decode_seconds: started.elapsed().as_secs_f64(),
        attempt_started: true,
    })?;
    Ok(StartedDecode {
        value,
        decode_seconds: started.elapsed().as_secs_f64(),
    })
}

fn read_wav(path: &Path) -> Result<Vec<f32>, EngineError> {
    let mut reader = hound::WavReader::open(path).map_err(|_| EngineError::AudioInvalid)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err(EngineError::AudioInvalid);
    }
    match spec.sample_format {
        hound::SampleFormat::Int if spec.bits_per_sample == 16 => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| value as f32 / 32768.0)
                    .map_err(|_| EngineError::AudioInvalid)
            })
            .collect(),
        hound::SampleFormat::Float if spec.bits_per_sample == 32 => reader
            .samples::<f32>()
            .map(|sample| sample.map_err(|_| EngineError::AudioInvalid))
            .collect(),
        _ => Err(EngineError::AudioInvalid),
    }
}

fn map_model_load_error(error: Error) -> EngineError {
    report_native_error("model_load", &error);
    match error {
        Error::ModelFileNotFound(_) => EngineError::ModelMissing,
        Error::ModelLoad(_) => EngineError::ModelInvalid,
        Error::OutOfMemory(_) => EngineError::OutOfMemory,
        Error::Backend(_) => EngineError::BackendFailure,
        _ => EngineError::ModelInvalid,
    }
}

fn map_warmup_error(error: Error) -> EngineError {
    report_native_error("warmup", &error);
    match error {
        Error::Aborted { .. } => EngineError::Cancelled,
        Error::OutOfMemory(_) => EngineError::OutOfMemory,
        Error::Backend(_) => EngineError::BackendFailure,
        _ => EngineError::WarmupFailed,
    }
}

fn map_decode_error(error: Error) -> EngineError {
    report_native_error("decode", &error);
    match error {
        Error::Aborted { .. } => EngineError::Cancelled,
        Error::OutOfMemory(_) => EngineError::OutOfMemory,
        Error::Backend(_) => EngineError::BackendFailure,
        _ => EngineError::DecodeFailure,
    }
}

fn map_backend_error(error: Error) -> EngineError {
    report_native_error("backend", &error);
    match error {
        Error::OutOfMemory(_) => EngineError::OutOfMemory,
        Error::Backend(_) => EngineError::BackendFailure,
        _ => EngineError::BackendFailure,
    }
}

fn report_native_error(stage: &str, error: &Error) {
    eprintln!(
        "[GPUWorker][Native] stage={stage} raw_status={} error={error}",
        error.raw_status()
    );
}

fn enumerate_vulkan_devices() -> Vec<VulkanDevice> {
    devices()
        .into_iter()
        .filter(|device| device.kind.eq_ignore_ascii_case("vulkan"))
        .filter_map(|device| {
            let registry_index = device.index?;
            Some(VulkanDevice {
                device_id: device
                    .device_id
                    .unwrap_or_else(|| format!("vulkan-index-{registry_index}")),
                registry_index,
                name: device.name,
                description: device.description,
                device_type: match device.device_type {
                    DeviceType::Gpu => "gpu",
                    DeviceType::Igpu => "igpu",
                    DeviceType::Cpu => "cpu",
                    DeviceType::Accel => "accel",
                    DeviceType::Unknown => "unknown",
                }
                .to_string(),
                memory_total_bytes: device.memory_total,
                memory_free_bytes: device.memory_free,
            })
        })
        .collect()
}

fn retry_discovery<F>(
    mut probe: F,
    attempts: u32,
    delay: Duration,
) -> Result<Vec<VulkanDevice>, EngineError>
where
    F: FnMut() -> Result<Vec<VulkanDevice>, EngineError>,
{
    let attempts = attempts.max(1);
    let mut last_error = None;
    for attempt in 0..attempts {
        match probe() {
            Ok(devices) if !devices.is_empty() => return Ok(devices),
            Ok(devices) => {
                last_error = None;
                if attempt + 1 == attempts {
                    return Ok(devices);
                }
            }
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 == attempts {
                    break;
                }
            }
        }
        if delay > Duration::ZERO {
            std::thread::sleep(delay);
        }
    }
    match last_error {
        Some(error) => Err(error),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::{execute_started_decode, retry_discovery, EngineError, GpuEngine, VulkanDevice};
    use std::cell::Cell;
    use std::path::Path;
    use std::time::Duration;
    use transcribe_cpp::CancelToken;

    fn sample_device(device_id: &str) -> VulkanDevice {
        VulkanDevice {
            device_id: device_id.to_string(),
            registry_index: 0,
            name: "GPU 0".to_string(),
            description: "Physical Vulkan GPU".to_string(),
            device_type: "gpu".to_string(),
            memory_total_bytes: 8_000_000_000,
            memory_free_bytes: 4_000_000_000,
        }
    }

    fn assert_started_failure(error: EngineError) {
        let started = Cell::new(false);
        let failure = execute_started_decode(
            1.25,
            |_| started.set(true),
            || -> Result<(), EngineError> { Err(error) },
        )
        .expect_err("decode must fail");

        assert!(started.get());
        assert!(failure.attempt_started);
        assert_eq!(failure.audio_seconds, Some(1.25));
        assert!(failure.decode_seconds.is_finite());
        assert!((failure.decode_seconds / 1.25).is_finite());
    }

    #[test]
    fn invalid_empty_and_prerun_rejections_never_start_decode() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let invalid = temporary.path().join("invalid.wav");
        std::fs::write(&invalid, b"not a wav").expect("invalid fixture");
        let empty = temporary.path().join("empty.wav");
        let writer = hound::WavWriter::create(
            &empty,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("empty fixture");
        writer.finalize().expect("empty fixture finalize");
        let valid = temporary.path().join("valid.wav");
        let mut writer = hound::WavWriter::create(
            &valid,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("valid fixture");
        writer.write_sample(0_i16).expect("valid sample");
        writer.finalize().expect("valid fixture finalize");

        for path in [&invalid, &empty, &valid] {
            let started = Cell::new(false);
            let failure = GpuEngine::default()
                .transcribe(Path::new(path), None, &CancelToken::new(), |_| {
                    started.set(true)
                })
                .expect_err("pre-run input must fail without a loaded session");
            assert!(!started.get());
            assert!(!failure.attempt_started);
            assert_eq!(failure.decode_seconds, 0.0);
        }
    }

    #[test]
    fn started_success_failure_and_cancellation_have_finite_decode_timing() {
        let started = Cell::new(false);
        let success = execute_started_decode(2.0, |_| started.set(true), || Ok("decoded"))
            .expect("decode succeeds");
        assert!(started.get());
        assert_eq!(success.value, "decoded");
        assert!(success.decode_seconds.is_finite());
        assert!((success.decode_seconds / 2.0).is_finite());

        assert_started_failure(EngineError::DecodeFailure);
        assert_started_failure(EngineError::Cancelled);
    }

    #[test]
    fn discovery_retries_transient_empty_and_error_probes() {
        let probes = Cell::new(0);
        let devices = retry_discovery(
            || {
                let attempt = probes.get();
                probes.set(attempt + 1);
                match attempt {
                    0 => Err(EngineError::BackendFailure),
                    1 => Ok(Vec::new()),
                    _ => Ok(vec![sample_device("vulkan-index-0")]),
                }
            },
            3,
            Duration::ZERO,
        )
        .expect("discovery eventually finds a device");

        assert_eq!(probes.get(), 3);
        assert_eq!(devices[0].device_id, "vulkan-index-0");
    }

    #[test]
    fn discovery_empty_list_is_a_completed_answer() {
        let probes = Cell::new(0);
        let devices = retry_discovery(
            || {
                probes.set(probes.get() + 1);
                Ok(Vec::new())
            },
            3,
            Duration::ZERO,
        )
        .expect("empty discovery is not an error");

        assert_eq!(probes.get(), 3);
        assert!(devices.is_empty());
    }

    #[test]
    fn discovery_keeps_init_failure_after_retries() {
        let probes = Cell::new(0);
        let error = retry_discovery(
            || {
                probes.set(probes.get() + 1);
                Err(EngineError::BackendFailure)
            },
            2,
            Duration::ZERO,
        )
        .expect_err("persistent init failure stays an error");

        assert_eq!(probes.get(), 2);
        assert!(matches!(error, EngineError::BackendFailure));
    }
}
