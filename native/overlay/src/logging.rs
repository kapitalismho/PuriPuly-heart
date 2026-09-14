use parking_lot::{Condvar, Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc as std_mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

#[cfg(windows)]
use windows::Win32::Foundation::CloseHandle;
#[cfg(windows)]
use windows::Win32::System::Threading::{GetCurrentThreadId, OpenThread, THREAD_TERMINATE};
#[cfg(windows)]
use windows::Win32::System::IO::CancelSynchronousIo;

type LogStream = Box<dyn Write + Send>;
const MAX_LOG_RECORD_BYTES: usize = 4 * 1024;
const DIAGNOSTIC_QUEUE_CAPACITY: usize = 128;
const RELIABLE_QUEUE_CAPACITY: usize = 8;
const WRITE_TIMEOUT: Duration = Duration::from_millis(25);
const RELIABLE_COMPLETION_TIMEOUT: Duration = Duration::from_millis(50);
const WRITER_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(100);

struct LogRecord {
    stdout: bool,
    line: String,
    completion: Option<oneshot::Sender<io::Result<()>>>,
}

struct ActiveWrite {
    generation: u64,
    started_at: Instant,
    timed_out: bool,
}

#[derive(Default)]
struct WriteWatchState {
    next_generation: u64,
    active: Option<ActiveWrite>,
    shutdown: bool,
}

#[derive(Default)]
struct WriteWatch {
    state: Mutex<WriteWatchState>,
    changed: Condvar,
}

struct WriterOwner {
    writer: Option<JoinHandle<()>>,
    watchdog: Option<JoinHandle<()>>,
    writer_done: std_mpsc::Receiver<()>,
    watchdog_done: std_mpsc::Receiver<()>,
    thread_id: u32,
    #[cfg(test)]
    watch: Arc<WriteWatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverlayLoggingMode {
    #[default]
    Basic,
    Detailed,
}

impl OverlayLoggingMode {
    fn allows_info(self) -> bool {
        matches!(self, Self::Detailed)
    }
}

pub struct OverlayLogger {
    diagnostic_sender: Option<std_mpsc::SyncSender<LogRecord>>,
    reliable_sender: Option<std_mpsc::SyncSender<LogRecord>>,
    mode: RwLock<OverlayLoggingMode>,
    dropped_records: Arc<AtomicU64>,
    shutting_down: Arc<AtomicBool>,
    writer: Mutex<WriterOwner>,
}

impl OverlayLogger {
    pub async fn open(
        _log_dir: impl AsRef<std::path::Path>,
        mode: OverlayLoggingMode,
    ) -> io::Result<Self> {
        Ok(Self::from_streams(
            Box::new(std::io::stdout()),
            Box::new(std::io::stderr()),
            mode,
        ))
    }

    pub async fn info(&self, message: impl AsRef<str>) -> io::Result<()> {
        self.log_line("INFO", message.as_ref())
    }

    pub async fn detailed_info(&self, message: impl AsRef<str>) -> io::Result<bool> {
        if !self.is_detailed() {
            return Ok(false);
        }
        self.enqueue_diagnostic(true, "INFO", message.as_ref());
        Ok(true)
    }

    pub async fn warn(&self, message: impl AsRef<str>) -> io::Result<()> {
        self.log_line("WARN", message.as_ref())
    }

    pub async fn error(&self, message: impl AsRef<str>) -> io::Result<()> {
        self.log_line("ERROR", message.as_ref())
    }

    pub async fn emit_stdout_event(&self, payload: &Value) -> io::Result<()> {
        self.write_reliable(true, format!("EVENT {payload}")).await
    }

    pub async fn emit_stderr_event(&self, payload: &Value) -> io::Result<()> {
        self.write_reliable(false, format!("EVENT {payload}")).await
    }

    pub fn set_mode(&self, mode: OverlayLoggingMode) {
        *self.mode.write() = mode;
    }

    pub fn is_detailed(&self) -> bool {
        matches!(*self.mode.read(), OverlayLoggingMode::Detailed)
    }

    pub fn dropped_records(&self) -> u64 {
        self.dropped_records.load(Ordering::Relaxed)
    }

    pub(crate) fn from_streams(
        stdout: LogStream,
        stderr: LogStream,
        mode: OverlayLoggingMode,
    ) -> Self {
        let (diagnostic_sender, diagnostic_receiver) =
            std_mpsc::sync_channel(DIAGNOSTIC_QUEUE_CAPACITY);
        let (reliable_sender, reliable_receiver) = std_mpsc::sync_channel(RELIABLE_QUEUE_CAPACITY);
        let dropped_records = Arc::new(AtomicU64::new(0));
        let shutting_down = Arc::new(AtomicBool::new(false));
        let watch = Arc::new(WriteWatch::default());
        let (started_sender, started_receiver) = std_mpsc::sync_channel(1);
        let (writer_done_sender, writer_done_receiver) = std_mpsc::sync_channel(1);
        let (watchdog_done_sender, watchdog_done_receiver) = std_mpsc::sync_channel(1);
        let writer_dropped_records = dropped_records.clone();
        let writer_shutting_down = shutting_down.clone();
        let writer_watch = watch.clone();
        let writer = thread::spawn(move || {
            let _ = started_sender.send(current_thread_id());
            run_writer(
                stdout,
                stderr,
                diagnostic_receiver,
                reliable_receiver,
                writer_dropped_records,
                writer_shutting_down,
                writer_watch,
            );
            let _ = writer_done_sender.send(());
        });
        let thread_id = started_receiver
            .recv()
            .expect("overlay logger writer must report its thread identity");
        let watchdog_watch = watch.clone();
        let watchdog = thread::spawn(move || {
            run_write_watchdog(thread_id, watchdog_watch);
            let _ = watchdog_done_sender.send(());
        });
        Self {
            diagnostic_sender: Some(diagnostic_sender),
            reliable_sender: Some(reliable_sender),
            mode: RwLock::new(mode),
            dropped_records,
            shutting_down,
            writer: Mutex::new(WriterOwner {
                writer: Some(writer),
                watchdog: Some(watchdog),
                writer_done: writer_done_receiver,
                watchdog_done: watchdog_done_receiver,
                thread_id,
                #[cfg(test)]
                watch,
            }),
        }
    }

    fn log_line(&self, level: &str, message: &str) -> io::Result<()> {
        if level == "INFO" && !self.mode.read().allows_info() {
            return Ok(());
        }
        self.enqueue_diagnostic(level != "ERROR", level, message);
        Ok(())
    }

    fn enqueue_diagnostic(&self, stdout: bool, level: &str, message: &str) {
        let prefix_len = "[overlay][] ".len().saturating_add(level.len());
        if prefix_len.saturating_add(message.len()).saturating_add(1) > MAX_LOG_RECORD_BYTES {
            increment_saturating(&self.dropped_records);
            return;
        }
        let record = LogRecord {
            stdout,
            line: format!("[overlay][{level}] {message}"),
            completion: None,
        };
        let Some(sender) = self.diagnostic_sender.as_ref() else {
            increment_saturating(&self.dropped_records);
            return;
        };
        if sender.try_send(record).is_err() {
            increment_saturating(&self.dropped_records);
        }
    }

    async fn write_reliable(&self, stdout: bool, line: String) -> io::Result<()> {
        if line.len().saturating_add(1) > MAX_LOG_RECORD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "overlay control record exceeds 4 KiB",
            ));
        }
        let sender = self
            .reliable_sender
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "overlay writer stopped"))?;
        let (completion, receiver) = oneshot::channel();
        let mut record = Some(LogRecord {
            stdout,
            line,
            completion: Some(completion),
        });
        let deadline = Instant::now() + WRITE_TIMEOUT;
        loop {
            match sender.try_send(record.take().expect("reliable record present")) {
                Ok(()) => break,
                Err(std_mpsc::TrySendError::Full(returned)) if Instant::now() < deadline => {
                    record = Some(returned);
                    tokio::task::yield_now().await;
                }
                Err(std_mpsc::TrySendError::Full(_)) => {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "overlay control queue blocked",
                    ));
                }
                Err(std_mpsc::TrySendError::Disconnected(_)) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "overlay writer stopped",
                    ));
                }
            }
        }
        let completion_deadline = Instant::now() + RELIABLE_COMPLETION_TIMEOUT;
        let mut receiver = receiver;
        loop {
            match receiver.try_recv() {
                Ok(result) => return result,
                Err(oneshot::error::TryRecvError::Empty)
                    if Instant::now() < completion_deadline =>
                {
                    tokio::task::yield_now().await;
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "overlay control write blocked",
                    ));
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "overlay writer stopped",
                    ));
                }
            }
        }
    }

    pub fn shutdown(mut self) -> io::Result<()> {
        self.close_and_join()
    }

    fn close_and_join(&mut self) -> io::Result<()> {
        self.shutting_down.store(true, Ordering::Release);
        self.diagnostic_sender.take();
        self.reliable_sender.take();
        let mut owner = self.writer.lock();
        if owner.writer.is_none() {
            return Ok(());
        }
        if owner
            .writer_done
            .recv_timeout(WRITER_SHUTDOWN_TIMEOUT)
            .is_err()
        {
            let _ = cancel_synchronous_io(owner.thread_id);
            if owner
                .writer_done
                .recv_timeout(WRITER_SHUTDOWN_TIMEOUT)
                .is_err()
            {
                owner.writer.take();
                owner.watchdog.take();
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "process-terminal overlay writer cleanup failed",
                ));
            }
        }
        if !thread_finished_within(owner.writer.as_ref(), WRITER_SHUTDOWN_TIMEOUT) {
            owner.writer.take();
            owner.watchdog.take();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "process-terminal overlay writer join failed",
            ));
        }
        if let Some(writer) = owner.writer.take() {
            writer.join().map_err(|_| {
                io::Error::other("process-terminal overlay writer cleanup panicked")
            })?;
        }
        if owner
            .watchdog_done
            .recv_timeout(WRITER_SHUTDOWN_TIMEOUT)
            .is_err()
            || !thread_finished_within(owner.watchdog.as_ref(), WRITER_SHUTDOWN_TIMEOUT)
        {
            owner.watchdog.take();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "process-terminal overlay writer watchdog cleanup failed",
            ));
        }
        if let Some(watchdog) = owner.watchdog.take() {
            watchdog.join().map_err(|_| {
                io::Error::other("process-terminal overlay writer watchdog panicked")
            })?;
        }
        Ok(())
    }
}

impl Drop for OverlayLogger {
    fn drop(&mut self) {
        let _ = self.close_and_join();
    }
}

fn increment_saturating(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_add(1))
    });
}

fn run_writer(
    mut stdout: LogStream,
    mut stderr: LogStream,
    diagnostics: std_mpsc::Receiver<LogRecord>,
    reliable: std_mpsc::Receiver<LogRecord>,
    dropped_records: Arc<AtomicU64>,
    shutting_down: Arc<AtomicBool>,
    watch: Arc<WriteWatch>,
) {
    while !shutting_down.load(Ordering::Acquire) {
        let (record, reliable_disconnected) = match reliable.try_recv() {
            Ok(record) => (Some(record), false),
            Err(std_mpsc::TryRecvError::Disconnected) => (None, true),
            Err(std_mpsc::TryRecvError::Empty) => (None, false),
        };
        let (record, diagnostics_disconnected) = match record {
            Some(record) => (Some(record), false),
            None => match diagnostics.recv_timeout(Duration::from_millis(1)) {
                Ok(record) => (Some(record), false),
                Err(std_mpsc::RecvTimeoutError::Disconnected) => (None, true),
                Err(std_mpsc::RecvTimeoutError::Timeout) => (None, false),
            },
        };
        let Some(record) = record else {
            if reliable_disconnected && diagnostics_disconnected {
                break;
            }
            continue;
        };
        let stream = if record.stdout {
            &mut stdout
        } else {
            &mut stderr
        };
        let generation = begin_write(&watch);
        let mut result = stream
            .write_all(record.line.as_bytes())
            .and_then(|()| stream.write_all(b"\n"))
            .and_then(|()| stream.flush());
        if finish_write(&watch, generation) {
            result = Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "overlay log write exceeded deadline",
            ));
        }
        if let Some(completion) = record.completion {
            let _ = completion.send(result);
        } else if result.is_err() {
            increment_saturating(&dropped_records);
        }
    }
    shutdown_write_watch(&watch);
}

fn begin_write(watch: &WriteWatch) -> u64 {
    let mut state = watch.state.lock();
    state.next_generation = state.next_generation.saturating_add(1);
    let generation = state.next_generation;
    state.active = Some(ActiveWrite {
        generation,
        started_at: Instant::now(),
        timed_out: false,
    });
    watch.changed.notify_one();
    generation
}

fn finish_write(watch: &WriteWatch, generation: u64) -> bool {
    let mut state = watch.state.lock();
    let timed_out = state
        .active
        .as_ref()
        .filter(|active| active.generation == generation)
        .is_some_and(|active| active.timed_out);
    if state
        .active
        .as_ref()
        .is_some_and(|active| active.generation == generation)
    {
        state.active = None;
    }
    watch.changed.notify_one();
    timed_out
}

fn shutdown_write_watch(watch: &WriteWatch) {
    let mut state = watch.state.lock();
    state.active = None;
    state.shutdown = true;
    watch.changed.notify_one();
}

fn thread_finished_within(thread: Option<&JoinHandle<()>>, timeout: Duration) -> bool {
    let Some(thread) = thread else {
        return true;
    };
    let deadline = Instant::now() + timeout;
    while !thread.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    thread.is_finished()
}

fn run_write_watchdog(thread_id: u32, watch: Arc<WriteWatch>) {
    run_write_watchdog_with(watch, || cancel_synchronous_io(thread_id));
}

fn run_write_watchdog_with<F>(watch: Arc<WriteWatch>, mut cancel: F)
where
    F: FnMut() -> io::Result<()>,
{
    let mut state = watch.state.lock();
    loop {
        let Some(active) = state.active.as_ref() else {
            if state.shutdown {
                return;
            }
            watch.changed.wait(&mut state);
            continue;
        };
        if active.timed_out {
            watch.changed.wait(&mut state);
            continue;
        }
        let deadline = active.started_at + WRITE_TIMEOUT;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if !remaining.is_zero() {
            watch.changed.wait_for(&mut state, remaining);
            continue;
        }
        if let Some(active) = state.active.as_mut() {
            active.timed_out = true;
        }
        let _ = cancel();
    }
}

#[cfg(windows)]
fn current_thread_id() -> u32 {
    unsafe { GetCurrentThreadId() }
}

#[cfg(not(windows))]
fn current_thread_id() -> u32 {
    0
}

#[cfg(windows)]
fn cancel_synchronous_io(thread_id: u32) -> io::Result<()> {
    let thread = unsafe { OpenThread(THREAD_TERMINATE, false, thread_id) }?;
    let result = unsafe { CancelSynchronousIo(thread) };
    let _ = unsafe { CloseHandle(thread) };
    result.map_err(io::Error::other)
}

#[cfg(not(windows))]
fn cancel_synchronous_io(_thread_id: u32) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "synchronous writer cancellation is unavailable",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[derive(Clone, Default)]
    struct RecordingSink {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self::default()
        }

        fn bytes(&self) -> Vec<u8> {
            self.buffer.lock().clone()
        }
    }

    impl Write for RecordingSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.buffer.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn unique_log_dir(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos();
        std::env::temp_dir().join(format!("puripuly-heart-overlay-logger-{name}-{nonce}"))
    }

    async fn wait_for_bytes(sink: &RecordingSink, minimum: usize) {
        tokio::time::timeout(Duration::from_millis(100), async {
            while sink.bytes().len() < minimum {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn overlay_logger_does_not_create_dedicated_log_file() {
        let log_dir = unique_log_dir("no-file");
        let log_path = log_dir.join("puripuly_heart_overlay.log");

        let logger = OverlayLogger::open(&log_dir, OverlayLoggingMode::Detailed)
            .await
            .unwrap();
        logger.info("hello").await.unwrap();

        assert!(!log_path.exists());
    }

    #[tokio::test]
    async fn overlay_logger_routes_info_and_error_lines_to_streams_only() {
        let stdout = RecordingSink::new();
        let stderr = RecordingSink::new();
        let logger = OverlayLogger::from_streams(
            Box::new(stdout.clone()),
            Box::new(stderr.clone()),
            OverlayLoggingMode::Detailed,
        );

        logger.info("child line").await.unwrap();
        logger.error("bad line").await.unwrap();
        wait_for_bytes(&stdout, "[overlay][INFO] child line\n".len()).await;
        wait_for_bytes(&stderr, "[overlay][ERROR] bad line\n".len()).await;

        assert_eq!(
            String::from_utf8(stdout.bytes()).unwrap(),
            "[overlay][INFO] child line\n"
        );
        assert_eq!(
            String::from_utf8(stderr.bytes()).unwrap(),
            "[overlay][ERROR] bad line\n"
        );
    }

    #[tokio::test]
    async fn overlay_logger_suppresses_info_lines_in_basic_mode() {
        let stdout = RecordingSink::new();
        let stderr = RecordingSink::new();
        let logger = OverlayLogger::from_streams(
            Box::new(stdout.clone()),
            Box::new(stderr.clone()),
            OverlayLoggingMode::Basic,
        );

        logger.info("hidden").await.unwrap();
        logger.warn("visible").await.unwrap();
        wait_for_bytes(&stdout, "[overlay][WARN] visible\n".len()).await;

        assert_eq!(
            String::from_utf8(stdout.bytes()).unwrap(),
            "[overlay][WARN] visible\n"
        );
        assert_eq!(String::from_utf8(stderr.bytes()).unwrap(), "");
    }

    #[tokio::test]
    async fn overlay_logger_applies_runtime_mode_updates() {
        let stdout = RecordingSink::new();
        let stderr = RecordingSink::new();
        let logger = OverlayLogger::from_streams(
            Box::new(stdout.clone()),
            Box::new(stderr.clone()),
            OverlayLoggingMode::Basic,
        );

        logger.info("hidden").await.unwrap();
        logger.set_mode(OverlayLoggingMode::Detailed);
        logger.info("visible").await.unwrap();
        wait_for_bytes(&stdout, "[overlay][INFO] visible\n".len()).await;

        assert_eq!(
            String::from_utf8(stdout.bytes()).unwrap(),
            "[overlay][INFO] visible\n"
        );
        assert_eq!(String::from_utf8(stderr.bytes()).unwrap(), "");
    }
    struct GatedSink {
        released: Arc<AtomicBool>,
    }

    impl Write for GatedSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            while !self.released.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(1));
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn diagnostic_hot_path_is_nonblocking_and_bounded_when_writer_stalls() {
        let released = Arc::new(AtomicBool::new(false));
        let logger = OverlayLogger::from_streams(
            Box::new(GatedSink {
                released: released.clone(),
            }),
            Box::new(GatedSink {
                released: released.clone(),
            }),
            OverlayLoggingMode::Detailed,
        );

        for index in 0..=(DIAGNOSTIC_QUEUE_CAPACITY + 1) {
            tokio::time::timeout(
                Duration::from_millis(5),
                logger.info(format!("diagnostic {index}")),
            )
            .await
            .unwrap()
            .unwrap();
        }
        assert!(logger.dropped_records() >= 1);

        let event_result = logger
            .emit_stdout_event(&serde_json::json!({"type": "overlay_ready"}))
            .await;
        assert_eq!(event_result.unwrap_err().kind(), io::ErrorKind::TimedOut);
        released.store(true, Ordering::Release);
        logger.shutdown().unwrap();
    }

    #[tokio::test]
    async fn stalled_watchdog_tracks_one_current_write_and_preserves_timeout() {
        let released = Arc::new(AtomicBool::new(false));
        let logger = OverlayLogger::from_streams(
            Box::new(GatedSink {
                released: released.clone(),
            }),
            Box::new(RecordingSink::new()),
            OverlayLoggingMode::Detailed,
        );
        let watch = logger.writer.lock().watch.clone();
        logger.info("blocked").await.unwrap();
        for index in 0..=(DIAGNOSTIC_QUEUE_CAPACITY + 1) {
            logger.info(format!("queued {index}")).await.unwrap();
        }
        tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                let timed_out = watch
                    .state
                    .lock()
                    .active
                    .as_ref()
                    .is_some_and(|active| active.timed_out);
                if timed_out {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        {
            let state = watch.state.lock();
            let active = state.active.as_ref().unwrap();
            assert_eq!(active.generation, 1);
            assert_eq!(state.next_generation, 1);
            assert!(active.timed_out);
        }
        released.store(true, Ordering::Release);
        tokio::time::timeout(Duration::from_millis(100), async {
            while logger.dropped_records() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        logger.shutdown().unwrap();
    }
    #[test]
    fn timeout_cancellation_cannot_cross_from_completed_write_to_successor() {
        let watch = Arc::new(WriteWatch::default());
        let first_generation = begin_write(&watch);
        let (cancel_entered_sender, cancel_entered_receiver) = std_mpsc::sync_channel(1);
        let (cancel_release_sender, cancel_release_receiver) = std_mpsc::sync_channel(1);
        let cancel_count = Arc::new(AtomicU64::new(0));
        let watchdog_watch = watch.clone();
        let watchdog_cancel_count = cancel_count.clone();
        let watchdog = thread::spawn(move || {
            run_write_watchdog_with(watchdog_watch, || {
                watchdog_cancel_count.fetch_add(1, Ordering::Relaxed);
                let _ = cancel_entered_sender.send(());
                let _ = cancel_release_receiver.recv();
                Ok(())
            });
        });
        cancel_entered_receiver
            .recv_timeout(Duration::from_millis(100))
            .unwrap();

        let successor_started = Arc::new(AtomicBool::new(false));
        let racer_watch = watch.clone();
        let racer_successor_started = successor_started.clone();
        let racer = thread::spawn(move || {
            let first_timed_out = finish_write(&racer_watch, first_generation);
            let successor_generation = begin_write(&racer_watch);
            racer_successor_started.store(true, Ordering::Release);
            let successor_timed_out = finish_write(&racer_watch, successor_generation);
            shutdown_write_watch(&racer_watch);
            (first_timed_out, successor_generation, successor_timed_out)
        });

        thread::sleep(Duration::from_millis(10));
        assert!(!successor_started.load(Ordering::Acquire));
        cancel_release_sender.send(()).unwrap();
        let (first_timed_out, successor_generation, successor_timed_out) = racer.join().unwrap();
        watchdog.join().unwrap();

        assert!(first_timed_out);
        assert_eq!(successor_generation, first_generation + 1);
        assert!(!successor_timed_out);
        assert_eq!(cancel_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn dropped_record_counter_saturates() {
        let logger = OverlayLogger::from_streams(
            Box::new(RecordingSink::new()),
            Box::new(RecordingSink::new()),
            OverlayLoggingMode::Detailed,
        );
        logger.dropped_records.store(u64::MAX, Ordering::Relaxed);
        logger.info("x".repeat(MAX_LOG_RECORD_BYTES)).await.unwrap();
        assert_eq!(logger.dropped_records(), u64::MAX);
        logger.shutdown().unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn real_stopped_stdout_and_stderr_pipes_do_not_own_process_shutdown() {
        if let Ok(stream) = std::env::var("PURIPULY_LOG_PIPE_CHILD") {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                let logger =
                    OverlayLogger::open(std::env::temp_dir(), OverlayLoggingMode::Detailed)
                        .await
                        .unwrap();
                let payload = "x".repeat(MAX_LOG_RECORD_BYTES - 32);
                for _ in 0..=DIAGNOSTIC_QUEUE_CAPACITY {
                    if stream == "stdout" {
                        logger.info(&payload).await.unwrap();
                    } else {
                        logger.error(&payload).await.unwrap();
                    }
                }
                thread::sleep(Duration::from_millis(100));
                logger.shutdown().unwrap();
            });
            return;
        }

        for stream in ["stdout", "stderr"] {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "logging::tests::real_stopped_stdout_and_stderr_pipes_do_not_own_process_shutdown",
                    "--nocapture",
                ])
                .env("PURIPULY_LOG_PIPE_CHILD", stream)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "{stream} child failed: {status}");
                    break;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("{stream} child did not terminate with a stopped pipe");
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
