use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::c_void;
#[cfg(any(windows, test))]
use std::ffi::{CStr, CString};

use thiserror::Error;
#[cfg(windows)]
use windows::core::Interface;
#[cfg(windows)]
use windows::Win32::Foundation::LUID;
#[cfg(windows)]
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter, IDXGIAdapter1, IDXGIFactory4, DXGI_ADAPTER_FLAG_SOFTWARE,
};

use crate::presentation::AdapterIdentity;
use crate::renderer::RenderedFrame;
use crate::state::OverlayCalibration;

#[cfg(windows)]
const OVERLAY_KEY_PREFIX: &str = "com.puripuly.heart.overlay.";
#[cfg(windows)]
const OVERLAY_NAME_PREFIX: &str = "PuriPuly Heart Overlay ";
#[cfg(any(windows, test))]
const FN_TABLE_INTERFACE_PREFIX: &str = "FnTable:";
const DEFAULT_OVERLAY_WIDTH_METERS: f32 = 1.0667;
const DEFAULT_OVERLAY_DISTANCE_METERS: f32 = 1.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayAnchorMode {
    HeadLocked,
    SpatialLocked,
}

impl OverlayAnchorMode {
    pub fn from_anchor(anchor: &str) -> Result<Self, OpenVrError> {
        match anchor {
            "head_locked" => Ok(Self::HeadLocked),
            "spatial_locked" => Ok(Self::SpatialLocked),
            _ => Err(OpenVrError::Calibration(format!(
                "unsupported overlay calibration anchor: {anchor}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayPlacementPolicy {
    anchor: OverlayAnchorMode,
    width_meters: f32,
    offset_x_meters: f32,
    offset_y_meters: f32,
    distance_meters: f32,
}

impl Default for OverlayPlacementPolicy {
    fn default() -> Self {
        Self {
            anchor: OverlayAnchorMode::HeadLocked,
            width_meters: DEFAULT_OVERLAY_WIDTH_METERS,
            offset_x_meters: 0.0,
            offset_y_meters: 0.0,
            distance_meters: DEFAULT_OVERLAY_DISTANCE_METERS,
        }
    }
}

impl OverlayPlacementPolicy {
    pub fn is_head_locked(&self) -> bool {
        self.anchor == OverlayAnchorMode::HeadLocked
    }

    pub fn is_spatial_locked(&self) -> bool {
        self.anchor == OverlayAnchorMode::SpatialLocked
    }

    pub fn from_calibration(calibration: &OverlayCalibration) -> Result<Self, OpenVrError> {
        Ok(Self {
            anchor: OverlayAnchorMode::from_anchor(&calibration.anchor)?,
            width_meters: DEFAULT_OVERLAY_WIDTH_METERS * calibration.text_scale.max(0.1),
            offset_x_meters: calibration.offset_x,
            offset_y_meters: calibration.offset_y,
            distance_meters: calibration.distance.max(0.1),
        })
    }

    #[cfg(windows)]
    fn apply(
        &self,
        overlay_api: &openvr_sys::VR_IVROverlay_FnTable,
        overlay_handle: openvr_sys::VROverlayHandle_t,
    ) -> Result<(), OpenVrError> {
        let mut api = OpenVrPlacementApi {
            overlay_api,
            overlay_handle,
        };
        self.apply_with_api(&mut api)
    }

    fn apply_with_api(&self, api: &mut impl OverlayPlacementApi) -> Result<(), OpenVrError> {
        api.set_width(self.width_meters)?;

        if self.is_spatial_locked() {
            return Ok(());
        }

        api.set_hmd_relative_transform(self.hmd_relative_transform())
    }

    fn hmd_relative_transform(&self) -> [[f32; 4]; 3] {
        [
            [1.0, 0.0, 0.0, self.offset_x_meters],
            [0.0, 1.0, 0.0, self.offset_y_meters],
            [0.0, 0.0, 1.0, -self.distance_meters],
        ]
    }
}

trait OverlayPlacementApi {
    fn set_width(&mut self, width_meters: f32) -> Result<(), OpenVrError>;
    fn set_hmd_relative_transform(&mut self, transform: [[f32; 4]; 3]) -> Result<(), OpenVrError>;
}

#[cfg(windows)]
struct OpenVrPlacementApi<'a> {
    overlay_api: &'a openvr_sys::VR_IVROverlay_FnTable,
    overlay_handle: openvr_sys::VROverlayHandle_t,
}

#[cfg(windows)]
impl OverlayPlacementApi for OpenVrPlacementApi<'_> {
    fn set_width(&mut self, width_meters: f32) -> Result<(), OpenVrError> {
        let set_width = self
            .overlay_api
            .SetOverlayWidthInMeters
            .ok_or_else(missing_overlay_method("SetOverlayWidthInMeters"))?;
        let error = unsafe { set_width(self.overlay_handle, width_meters) };
        map_overlay_init_error(self.overlay_api, "SetOverlayWidthInMeters", error)
    }

    fn set_hmd_relative_transform(&mut self, transform: [[f32; 4]; 3]) -> Result<(), OpenVrError> {
        let set_transform = self
            .overlay_api
            .SetOverlayTransformTrackedDeviceRelative
            .ok_or_else(missing_overlay_method(
                "SetOverlayTransformTrackedDeviceRelative",
            ))?;
        let mut transform = openvr_sys::HmdMatrix34_t { m: transform };
        let error = unsafe {
            set_transform(
                self.overlay_handle,
                openvr_sys::k_unTrackedDeviceIndex_Hmd,
                &mut transform,
            )
        };
        map_overlay_init_error(
            self.overlay_api,
            "SetOverlayTransformTrackedDeviceRelative",
            error,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialReanchorOutcome {
    Applied,
    PoseUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpatialTrackingOrigin {
    Seated,
    Standing,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SpatialHmdPose {
    matrix: [[f32; 4]; 3],
    connected: bool,
    valid: bool,
}

trait SpatialReanchorApi {
    fn active_tracking_origin(&mut self) -> Result<Option<SpatialTrackingOrigin>, OpenVrError>;
    fn hmd_pose(
        &mut self,
        origin: SpatialTrackingOrigin,
    ) -> Result<Option<SpatialHmdPose>, OpenVrError>;
    fn set_absolute_transform(
        &mut self,
        origin: SpatialTrackingOrigin,
        transform: [[f32; 4]; 3],
    ) -> Result<(), OpenVrError>;
}

fn reanchor_spatial_locked_with_api(
    policy: &OverlayPlacementPolicy,
    api: &mut impl SpatialReanchorApi,
) -> Result<SpatialReanchorOutcome, OpenVrError> {
    let Some(origin) = api.active_tracking_origin()? else {
        return Ok(SpatialReanchorOutcome::PoseUnavailable);
    };
    let Some(pose) = api.hmd_pose(origin)? else {
        return Ok(SpatialReanchorOutcome::PoseUnavailable);
    };
    if !pose.connected || !pose.valid {
        return Ok(SpatialReanchorOutcome::PoseUnavailable);
    }
    let Some(transform) = spatial_locked_transform(policy, pose.matrix) else {
        return Ok(SpatialReanchorOutcome::PoseUnavailable);
    };
    api.set_absolute_transform(origin, transform)?;
    Ok(SpatialReanchorOutcome::Applied)
}

fn spatial_locked_transform(
    policy: &OverlayPlacementPolicy,
    hmd: [[f32; 4]; 3],
) -> Option<[[f32; 4]; 3]> {
    if !hmd.iter().flatten().all(|value| value.is_finite()) {
        return None;
    }
    let position = [hmd[0][3], hmd[1][3], hmd[2][3]];
    let forward = normalize3([-hmd[0][2], -hmd[1][2], -hmd[2][2]])?;
    let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]))?;
    let up = normalize3(cross3(right, forward))?;
    let anchored = add3(
        add3(
            add3(position, scale3(right, policy.offset_x_meters)),
            scale3(up, policy.offset_y_meters),
        ),
        scale3(forward, policy.distance_meters),
    );
    Some([
        [right[0], up[0], -forward[0], anchored[0]],
        [right[1], up[1], -forward[1], anchored[1]],
        [right[2], up[2], -forward[2], anchored[2]],
    ])
}

fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn scale3(vector: [f32; 3], scale: f32) -> [f32; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize3(vector: [f32; 3]) -> Option<[f32; 3]> {
    let length_squared = vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>();
    if !length_squared.is_finite() || length_squared <= 1.0e-8 {
        return None;
    }
    let inverse_length = length_squared.sqrt().recip();
    Some(scale3(vector, inverse_length))
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OpenVrError {
    #[error("openvr init failed: {0}")]
    Init(String),
    #[error("openvr output adapter selection failed: {0}")]
    AdapterSelection(String),
    #[error("openvr texture submission failed: {0}")]
    Submit(String),
    #[error("openvr calibration failed: {0}")]
    Calibration(String),
}

#[cfg_attr(not(any(windows, test)), allow(dead_code))]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
enum OpenVrBackgroundInitError {
    #[error("SteamVR runtime is not running")]
    NoServerForBackgroundApp,
    #[error("openvr init failed: {0}")]
    Init(String),
}

#[cfg_attr(not(any(windows, test)), allow(dead_code))]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum OpenVrStartupPreflightError {
    #[error("SteamVR/OpenVR runtime is not installed")]
    SteamVrNotInstalled,
    #[error("SteamVR is not running")]
    SteamVrNotRunning,
    #[error("VR headset not found")]
    HmdNotFound,
    #[error("openvr init failed: {0}")]
    Init(String),
}

#[cfg_attr(not(any(windows, test)), allow(dead_code))]
trait OpenVrPreflightApi {
    fn is_runtime_installed(&self) -> bool;
    fn initialize_background_app(&self) -> Result<(), OpenVrBackgroundInitError>;
    fn shutdown_runtime(&self);
    fn is_hmd_present(&self) -> bool;
}

#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn run_startup_preflight(api: &impl OpenVrPreflightApi) -> Result<(), OpenVrStartupPreflightError> {
    if !api.is_runtime_installed() {
        return Err(OpenVrStartupPreflightError::SteamVrNotInstalled);
    }

    match api.initialize_background_app() {
        Ok(()) => api.shutdown_runtime(),
        Err(OpenVrBackgroundInitError::NoServerForBackgroundApp) => {
            return Err(OpenVrStartupPreflightError::SteamVrNotRunning);
        }
        Err(OpenVrBackgroundInitError::Init(message)) => {
            return Err(OpenVrStartupPreflightError::Init(message));
        }
    }

    if !api.is_hmd_present() {
        return Err(OpenVrStartupPreflightError::HmdNotFound);
    }

    Ok(())
}

pub(crate) fn perform_startup_preflight() -> Result<(), OpenVrStartupPreflightError> {
    if std::env::var("PURIPULY_SKIP_VR_PREFLIGHT").is_ok() {
        return Ok(());
    }

    #[cfg(windows)]
    {
        let api = WindowsOpenVrPreflightApi;
        return run_startup_preflight(&api);
    }

    #[cfg(not(windows))]
    {
        Ok(())
    }
}

pub trait OverlayTextureSubmitter {
    fn set_overlay_texture(&self, texture_handle: *mut c_void) -> Result<(), OpenVrError>;
}

pub const OPENVR_EVENT_OVERLAY_SHOWN: u32 = 500;
pub const OPENVR_EVENT_OVERLAY_HIDDEN: u32 = 501;
pub const OPENVR_EVENT_QUIT: u32 = 700;
pub const OPENVR_EVENT_PROCESS_QUIT: u32 = 701;
pub const OPENVR_EVENT_DRIVER_REQUESTED_QUIT: u32 = 704;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenVrEventClass {
    Fatal,
    Reconfigure,
    Ignore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenVrRuntimeEvent {
    OverlayShown,
    OverlayHidden,
    Quit,
    ProcessQuit,
    DriverRequestedQuit,
    Ignored(u32),
}

impl OpenVrRuntimeEvent {
    pub fn from_event_type(event_type: u32) -> Self {
        match event_type {
            OPENVR_EVENT_OVERLAY_SHOWN => Self::OverlayShown,
            OPENVR_EVENT_OVERLAY_HIDDEN => Self::OverlayHidden,
            OPENVR_EVENT_QUIT => Self::Quit,
            OPENVR_EVENT_PROCESS_QUIT => Self::ProcessQuit,
            OPENVR_EVENT_DRIVER_REQUESTED_QUIT => Self::DriverRequestedQuit,
            other => Self::Ignored(other),
        }
    }

    pub fn classify(self) -> OpenVrEventClass {
        match self {
            Self::Quit | Self::ProcessQuit | Self::DriverRequestedQuit => OpenVrEventClass::Fatal,
            Self::OverlayShown | Self::OverlayHidden => OpenVrEventClass::Reconfigure,
            Self::Ignored(_) => OpenVrEventClass::Ignore,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::OverlayShown => "overlay_shown",
            Self::OverlayHidden => "overlay_hidden",
            Self::Quit => "quit",
            Self::ProcessQuit => "process_quit",
            Self::DriverRequestedQuit => "driver_requested_quit",
            Self::Ignored(_) => "ignored",
        }
    }
}

pub trait OverlayFrameSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError>;

    fn display_refresh_rate_hz(&self) -> Option<f32> {
        None
    }

    fn apply_calibration(&mut self, _calibration: &OverlayCalibration) -> Result<(), OpenVrError> {
        Ok(())
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        Ok(SpatialReanchorOutcome::PoseUnavailable)
    }

    fn set_overlay_visible(&mut self, _visible: bool) -> Result<(), OpenVrError> {
        Ok(())
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        None
    }

    fn poll_runtime_events(&mut self, max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        let _ = max_events;
        Vec::new()
    }

    fn take_visibility_api_call_log(&mut self) -> Option<String> {
        None
    }

    fn sample_frame_timing(&self) -> Option<FrameTimingSample> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct OpenVrOutputAdapter {
    identity: AdapterIdentity,
    requested_identity: AdapterIdentity,
    #[cfg(windows)]
    adapter: Option<IDXGIAdapter>,
}

impl OpenVrOutputAdapter {
    pub fn identity(&self) -> AdapterIdentity {
        self.identity
    }

    pub fn requested_identity(&self) -> AdapterIdentity {
        self.requested_identity
    }

    #[cfg(windows)]
    pub(crate) fn adapter(&self) -> Option<&IDXGIAdapter> {
        self.adapter.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTimingSample {
    pub frame_index: u32,
    pub num_frame_presents: u32,
    pub num_mis_presented: u32,
    pub num_dropped_frames: u32,
    pub system_time_seconds: f64,
    pub client_frame_interval_ms: f32,
    pub present_call_cpu_ms: f32,
    pub wait_for_present_cpu_ms: f32,
    pub compositor_render_cpu_ms: f32,
    pub total_render_gpu_ms: f32,
    pub post_submit_gpu_ms: f32,
}

pub fn submit_texture<T: OverlayTextureSubmitter>(
    openvr: &T,
    frame: &RenderedFrame,
) -> Result<(), OpenVrError> {
    let texture_handle = frame
        .texture_ptr()
        .ok_or_else(|| OpenVrError::Submit("renderer returned no texture".into()))?;
    openvr.set_overlay_texture(texture_handle)
}

#[derive(Debug, Default)]
pub struct FakeOpenVr {
    last_call: RefCell<Option<String>>,
    call_sequence: RefCell<Vec<&'static str>>,
    spatial_reanchor_count: Cell<usize>,
    visible: Cell<bool>,
    observed_visible: Cell<Option<bool>>,
    pending_events: RefCell<VecDeque<OpenVrRuntimeEvent>>,
    last_visibility_api_call_log: RefCell<Option<String>>,
}

impl FakeOpenVr {
    pub fn last_call(&self) -> Option<String> {
        self.last_call.borrow().clone()
    }

    pub fn call_sequence(&self) -> Vec<&'static str> {
        self.call_sequence.borrow().clone()
    }

    pub fn spatial_reanchor_count(&self) -> usize {
        self.spatial_reanchor_count.get()
    }

    pub fn set_observed_overlay_visible(&self, visible: Option<bool>) {
        self.observed_visible.set(visible);
    }

    pub fn push_runtime_event(&self, event: OpenVrRuntimeEvent) {
        self.pending_events.borrow_mut().push_back(event);
    }
}

impl OverlayTextureSubmitter for FakeOpenVr {
    fn set_overlay_texture(&self, _texture_handle: *mut c_void) -> Result<(), OpenVrError> {
        self.last_call
            .replace(Some("SetOverlayTexture".to_string()));
        self.call_sequence.borrow_mut().push("SetOverlayTexture");
        Ok(())
    }
}

pub struct OpenVrOverlay {
    backend: OpenVrBackend,
}

impl OpenVrOverlay {
    pub fn new(overlay_instance_id: &str) -> Result<Self, OpenVrError> {
        Ok(Self {
            backend: OpenVrBackend::new(overlay_instance_id)?,
        })
    }

    pub fn output_adapter(&self) -> OpenVrOutputAdapter {
        self.backend.output_adapter()
    }
}

impl OverlayFrameSubmitter for OpenVrOverlay {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        self.backend.submit_frame(frame)
    }

    fn display_refresh_rate_hz(&self) -> Option<f32> {
        self.backend.display_refresh_rate_hz()
    }

    fn apply_calibration(&mut self, calibration: &OverlayCalibration) -> Result<(), OpenVrError> {
        self.backend.apply_calibration(calibration)
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        self.backend.reanchor_spatial_locked()
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.backend.set_overlay_visible(visible)
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        self.backend.observed_overlay_visible()
    }

    fn poll_runtime_events(&mut self, max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        self.backend.poll_runtime_events(max_events)
    }

    fn take_visibility_api_call_log(&mut self) -> Option<String> {
        self.backend.take_visibility_api_call_log()
    }

    fn sample_frame_timing(&self) -> Option<FrameTimingSample> {
        self.backend.sample_frame_timing()
    }
}

enum OpenVrBackend {
    #[cfg(windows)]
    Windows(WindowsOpenVrOverlay),
    #[cfg(not(windows))]
    Test(FakeOpenVr),
}

#[cfg(windows)]
struct WindowsOpenVrPreflightApi;

#[cfg(windows)]
impl OpenVrPreflightApi for WindowsOpenVrPreflightApi {
    fn is_runtime_installed(&self) -> bool {
        unsafe { openvr_sys::VR_IsRuntimeInstalled() }
    }

    fn initialize_background_app(&self) -> Result<(), OpenVrBackgroundInitError> {
        let mut init_error = openvr_sys::EVRInitError_VRInitError_None;
        unsafe {
            openvr_sys::VR_InitInternal(
                &mut init_error,
                openvr_sys::EVRApplicationType_VRApplication_Background,
            );
        }
        if init_error == openvr_sys::EVRInitError_VRInitError_None {
            return Ok(());
        }
        if init_error == openvr_sys::EVRInitError_VRInitError_Init_NoServerForBackgroundApp {
            return Err(OpenVrBackgroundInitError::NoServerForBackgroundApp);
        }
        Err(OpenVrBackgroundInitError::Init(format!(
            "VR_InitInternal failed: {}",
            vr_init_error_name(init_error)
        )))
    }

    fn shutdown_runtime(&self) {
        unsafe {
            openvr_sys::VR_ShutdownInternal();
        }
    }

    fn is_hmd_present(&self) -> bool {
        unsafe { openvr_sys::VR_IsHmdPresent() }
    }
}

impl OpenVrBackend {
    fn new(overlay_instance_id: &str) -> Result<Self, OpenVrError> {
        #[cfg(windows)]
        {
            return WindowsOpenVrOverlay::new(overlay_instance_id).map(Self::Windows);
        }

        #[cfg(not(windows))]
        {
            let _ = overlay_instance_id;
            Ok(Self::Test(FakeOpenVr::default()))
        }
    }

    fn output_adapter(&self) -> OpenVrOutputAdapter {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.output_adapter.clone(),
            #[cfg(not(windows))]
            Self::Test(_) => OpenVrOutputAdapter {
                identity: AdapterIdentity::Test,
                requested_identity: AdapterIdentity::Test,
            },
        }
    }

    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.submit_frame(frame),
            #[cfg(not(windows))]
            Self::Test(openvr) => submit_texture(openvr, frame),
        }
    }

    fn apply_calibration(&mut self, calibration: &OverlayCalibration) -> Result<(), OpenVrError> {
        #[cfg(not(windows))]
        let _ = calibration;

        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.apply_calibration(calibration),
            #[cfg(not(windows))]
            Self::Test(_) => Ok(()),
        }
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.reanchor_spatial_locked(),
            #[cfg(not(windows))]
            Self::Test(openvr) => openvr.reanchor_spatial_locked(),
        }
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.set_overlay_visible(visible),
            #[cfg(not(windows))]
            Self::Test(openvr) => openvr.set_overlay_visible(visible),
        }
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.observed_overlay_visible(),
            #[cfg(not(windows))]
            Self::Test(openvr) => openvr.observed_overlay_visible(),
        }
    }

    fn poll_runtime_events(&mut self, max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.poll_runtime_events(max_events),
            #[cfg(not(windows))]
            Self::Test(openvr) => openvr.poll_runtime_events(max_events),
        }
    }

    fn take_visibility_api_call_log(&mut self) -> Option<String> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.take_visibility_api_call_log(),
            #[cfg(not(windows))]
            Self::Test(openvr) => openvr.take_visibility_api_call_log(),
        }
    }

    fn display_refresh_rate_hz(&self) -> Option<f32> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.display_refresh_rate_hz(),
            #[cfg(not(windows))]
            Self::Test(_) => None,
        }
    }

    fn sample_frame_timing(&self) -> Option<FrameTimingSample> {
        match self {
            #[cfg(windows)]
            Self::Windows(openvr) => openvr.sample_frame_timing(),
            #[cfg(not(windows))]
            Self::Test(_) => None,
        }
    }
}

#[cfg(windows)]
struct WindowsOpenVrOverlay {
    overlay_api: *mut openvr_sys::VR_IVROverlay_FnTable,
    system_api: *mut openvr_sys::VR_IVRSystem_FnTable,
    compositor_api: Option<*mut openvr_sys::VR_IVRCompositor_FnTable>,
    overlay_handle: openvr_sys::VROverlayHandle_t,
    placement_policy: OverlayPlacementPolicy,
    visible: bool,
    last_visibility_api_call_log: Option<String>,
    output_adapter: OpenVrOutputAdapter,
}

#[cfg(windows)]
impl WindowsOpenVrOverlay {
    fn new(overlay_instance_id: &str) -> Result<Self, OpenVrError> {
        let overlay_api = initialize_overlay_api()?;
        let system_api = initialize_system_api()?;
        let compositor_api = initialize_compositor_api().ok();
        let output_adapter = match identify_openvr_output_adapter(unsafe { &*system_api }) {
            Ok(adapter) => adapter,
            Err(error) => {
                unsafe {
                    openvr_sys::VR_ShutdownInternal();
                }
                return Err(error);
            }
        };
        let overlay_handle = create_overlay_handle(overlay_api, overlay_instance_id)?;

        let instance = Self {
            overlay_api,
            system_api,
            compositor_api,
            overlay_handle,
            placement_policy: OverlayPlacementPolicy::default(),
            visible: false,
            last_visibility_api_call_log: None,
            output_adapter,
        };
        instance.configure_overlay()?;
        Ok(instance)
    }

    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        submit_texture(self, frame)
    }

    fn configure_overlay(&self) -> Result<(), OpenVrError> {
        let set_overlay_rendering_pid = self
            .overlay_api()
            .SetOverlayRenderingPid
            .ok_or_else(missing_overlay_method("SetOverlayRenderingPid"))?;
        let error = unsafe { set_overlay_rendering_pid(self.overlay_handle, std::process::id()) };
        map_overlay_init_error(self.overlay_api(), "SetOverlayRenderingPid", error)?;

        let set_overlay_flag = self
            .overlay_api()
            .SetOverlayFlag
            .ok_or_else(missing_overlay_method("SetOverlayFlag"))?;
        let error = unsafe {
            set_overlay_flag(
                self.overlay_handle,
                openvr_sys::VROverlayFlags_IsPremultiplied,
                true,
            )
        };
        map_overlay_init_error(self.overlay_api(), "SetOverlayFlag", error)?;
        self.placement_policy
            .apply(self.overlay_api(), self.overlay_handle)?;
        Ok(())
    }

    fn show_overlay(&self) -> Result<(), OpenVrError> {
        let show_overlay = self
            .overlay_api()
            .ShowOverlay
            .ok_or_else(missing_overlay_method("ShowOverlay"))?;
        let error = unsafe { show_overlay(self.overlay_handle) };
        map_overlay_init_error(self.overlay_api(), "ShowOverlay", error)
    }

    fn hide_overlay(&self) -> Result<(), OpenVrError> {
        let hide_overlay = self
            .overlay_api()
            .HideOverlay
            .ok_or_else(missing_overlay_method("HideOverlay"))?;
        let error = unsafe { hide_overlay(self.overlay_handle) };
        map_overlay_init_error(self.overlay_api(), "HideOverlay", error)
    }

    fn overlay_api(&self) -> &openvr_sys::VR_IVROverlay_FnTable {
        unsafe { &*self.overlay_api }
    }

    fn system_api(&self) -> &openvr_sys::VR_IVRSystem_FnTable {
        unsafe { &*self.system_api }
    }

    fn apply_calibration(&mut self, calibration: &OverlayCalibration) -> Result<(), OpenVrError> {
        self.placement_policy = OverlayPlacementPolicy::from_calibration(calibration)?;
        self.placement_policy
            .apply(self.overlay_api(), self.overlay_handle)
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        let policy = self.placement_policy.clone();
        reanchor_spatial_locked_with_api(&policy, self)
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        let is_visible = self.overlay_api().IsOverlayVisible?;
        Some(unsafe { is_visible(self.overlay_handle) })
    }

    fn poll_runtime_events(&mut self, max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        let mut events = Vec::new();
        if max_events == 0 {
            return events;
        }
        if let Some(poll) = self.overlay_api().PollNextOverlayEvent {
            while events.len() < max_events {
                let mut event = unsafe { std::mem::zeroed::<openvr_sys::VREvent_t>() };
                let got = unsafe {
                    poll(
                        self.overlay_handle,
                        &mut event,
                        std::mem::size_of::<openvr_sys::VREvent_t>() as u32,
                    )
                };
                if !got {
                    break;
                }
                events.push(OpenVrRuntimeEvent::from_event_type(event.eventType));
            }
        }
        if let Some(poll) = self.system_api().PollNextEvent {
            while events.len() < max_events {
                let mut event = unsafe { std::mem::zeroed::<openvr_sys::VREvent_t>() };
                let got = unsafe {
                    poll(
                        &mut event,
                        std::mem::size_of::<openvr_sys::VREvent_t>() as u32,
                    )
                };
                if !got {
                    break;
                }
                events.push(OpenVrRuntimeEvent::from_event_type(event.eventType));
            }
        }
        events
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        let cached_visible_before = self.visible;
        let actual_visible = self.observed_overlay_visible();
        if actual_visible.unwrap_or(self.visible) == visible {
            self.visible = actual_visible.unwrap_or(visible);
            self.last_visibility_api_call_log = Some(format_openvr_visibility_api_call_log(
                visible,
                cached_visible_before,
                "SkipCachedMatch",
                self.visible,
            ));
            return Ok(());
        }
        let api = if visible {
            "ShowOverlay"
        } else {
            "HideOverlay"
        };
        if visible {
            self.show_overlay()?;
        } else {
            self.hide_overlay()?;
        }
        self.visible = visible;
        if let Some(actual_visible) = self.observed_overlay_visible() {
            self.visible = actual_visible;
        }
        self.last_visibility_api_call_log = Some(format_openvr_visibility_api_call_log(
            visible,
            cached_visible_before,
            api,
            self.visible,
        ));
        Ok(())
    }

    fn take_visibility_api_call_log(&mut self) -> Option<String> {
        self.last_visibility_api_call_log.take()
    }

    fn display_refresh_rate_hz(&self) -> Option<f32> {
        const PROP_DISPLAY_FREQUENCY_FLOAT: openvr_sys::ETrackedDeviceProperty = 2002;

        let get_float = self.system_api().GetFloatTrackedDeviceProperty?;
        let mut error = 0;
        let refresh_rate_hz = unsafe {
            get_float(
                openvr_sys::k_unTrackedDeviceIndex_Hmd,
                PROP_DISPLAY_FREQUENCY_FLOAT,
                &mut error,
            )
        };
        if error == 0 && refresh_rate_hz.is_finite() && refresh_rate_hz > 0.0 {
            Some(refresh_rate_hz)
        } else {
            None
        }
    }

    fn sample_frame_timing(&self) -> Option<FrameTimingSample> {
        let compositor_api = self.compositor_api?;
        let get_frame_timing = unsafe { (*compositor_api).GetFrameTiming }?;
        let mut timing: openvr_sys::Compositor_FrameTiming = unsafe { std::mem::zeroed() };
        timing.m_nSize = std::mem::size_of::<openvr_sys::Compositor_FrameTiming>() as u32;
        let ok = unsafe { get_frame_timing(&mut timing, 0) };
        if !ok {
            return None;
        }
        Some(FrameTimingSample {
            frame_index: timing.m_nFrameIndex,
            num_frame_presents: timing.m_nNumFramePresents,
            num_mis_presented: timing.m_nNumMisPresented,
            num_dropped_frames: timing.m_nNumDroppedFrames,
            system_time_seconds: timing.m_flSystemTimeInSeconds,
            client_frame_interval_ms: timing.m_flClientFrameIntervalMs,
            present_call_cpu_ms: timing.m_flPresentCallCpuMs,
            wait_for_present_cpu_ms: timing.m_flWaitForPresentCpuMs,
            compositor_render_cpu_ms: timing.m_flCompositorRenderCpuMs,
            total_render_gpu_ms: timing.m_flTotalRenderGpuMs,
            post_submit_gpu_ms: timing.m_flPostSubmitGpuMs,
        })
    }
}

#[cfg(windows)]
impl SpatialReanchorApi for WindowsOpenVrOverlay {
    fn active_tracking_origin(&mut self) -> Result<Option<SpatialTrackingOrigin>, OpenVrError> {
        let Some(compositor_api) = self.compositor_api else {
            return Ok(None);
        };
        let Some(get_tracking_space) = (unsafe { (*compositor_api).GetTrackingSpace }) else {
            return Ok(None);
        };
        let origin = unsafe { get_tracking_space() };
        if origin == openvr_sys::ETrackingUniverseOrigin_TrackingUniverseSeated {
            return Ok(Some(SpatialTrackingOrigin::Seated));
        }
        if origin == openvr_sys::ETrackingUniverseOrigin_TrackingUniverseStanding {
            return Ok(Some(SpatialTrackingOrigin::Standing));
        }
        Ok(None)
    }

    fn hmd_pose(
        &mut self,
        origin: SpatialTrackingOrigin,
    ) -> Result<Option<SpatialHmdPose>, OpenVrError> {
        let get_pose = self
            .system_api()
            .GetDeviceToAbsoluteTrackingPose
            .ok_or_else(|| {
                OpenVrError::Init(
                    "missing OpenVR system method: GetDeviceToAbsoluteTrackingPose".to_string(),
                )
            })?;
        let mut pose = openvr_sys::TrackedDevicePose_t::default();
        unsafe {
            get_pose(openvr_tracking_origin(origin), 0.0, &mut pose, 1);
        }
        Ok(Some(SpatialHmdPose {
            matrix: pose.mDeviceToAbsoluteTracking.m,
            connected: pose.bDeviceIsConnected,
            valid: pose.bPoseIsValid,
        }))
    }

    fn set_absolute_transform(
        &mut self,
        origin: SpatialTrackingOrigin,
        transform: [[f32; 4]; 3],
    ) -> Result<(), OpenVrError> {
        let set_transform = self
            .overlay_api()
            .SetOverlayTransformAbsolute
            .ok_or_else(missing_overlay_method("SetOverlayTransformAbsolute"))?;
        let mut transform = openvr_sys::HmdMatrix34_t { m: transform };
        let error = unsafe {
            set_transform(
                self.overlay_handle,
                openvr_tracking_origin(origin),
                &mut transform,
            )
        };
        map_overlay_submit_error(self.overlay_api(), "SetOverlayTransformAbsolute", error)
    }
}

#[cfg(windows)]
fn openvr_tracking_origin(origin: SpatialTrackingOrigin) -> openvr_sys::ETrackingUniverseOrigin {
    match origin {
        SpatialTrackingOrigin::Seated => openvr_sys::ETrackingUniverseOrigin_TrackingUniverseSeated,
        SpatialTrackingOrigin::Standing => {
            openvr_sys::ETrackingUniverseOrigin_TrackingUniverseStanding
        }
    }
}

#[cfg(windows)]
fn identify_openvr_output_adapter(
    system_api: &openvr_sys::VR_IVRSystem_FnTable,
) -> Result<OpenVrOutputAdapter, OpenVrError> {
    let get_output_device = system_api
        .GetOutputDevice
        .ok_or_else(|| OpenVrError::AdapterSelection("output adapter method unavailable".into()))?;
    let mut device = 0u64;
    unsafe {
        get_output_device(
            &mut device,
            openvr_sys::ETextureType_TextureType_DirectX,
            std::ptr::null_mut(),
        );
    }
    let requested_identity = requested_adapter_identity(device)?;
    let AdapterIdentity::DxgiLuid { high, low } = requested_identity else {
        unreachable!()
    };
    let luid = LUID {
        LowPart: low,
        HighPart: high,
    };
    let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }
        .map_err(|_| OpenVrError::AdapterSelection("DXGI adapter factory unavailable".into()))?;
    let adapter: IDXGIAdapter = unsafe { factory.EnumAdapterByLuid(luid) }
        .map_err(|_| OpenVrError::AdapterSelection("DXGI adapter lookup failed".into()))?;
    let description = unsafe { adapter.GetDesc() }
        .map_err(|_| OpenVrError::AdapterSelection("output adapter identity unavailable".into()))?;
    let resolved_identity = AdapterIdentity::DxgiLuid {
        high: description.AdapterLuid.HighPart,
        low: description.AdapterLuid.LowPart,
    };
    let adapter1: IDXGIAdapter1 = adapter
        .cast()
        .map_err(|_| OpenVrError::AdapterSelection("adapter classification unavailable".into()))?;
    let description1 = unsafe { adapter1.GetDesc1() }
        .map_err(|_| OpenVrError::AdapterSelection("adapter classification unavailable".into()))?;
    validate_resolved_adapter(
        requested_identity,
        resolved_identity,
        description1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
    )?;
    Ok(OpenVrOutputAdapter {
        identity: resolved_identity,
        requested_identity,
        adapter: Some(adapter),
    })
}

#[cfg(any(windows, test))]
fn split_output_device_luid(device: u64) -> (i32, u32) {
    ((device >> 32) as i32, device as u32)
}

#[cfg(any(windows, test))]
fn requested_adapter_identity(device: u64) -> Result<AdapterIdentity, OpenVrError> {
    if device == 0 {
        return Err(OpenVrError::AdapterSelection(
            "OpenVR returned no output adapter".into(),
        ));
    }
    let (high, low) = split_output_device_luid(device);
    Ok(AdapterIdentity::DxgiLuid { high, low })
}

#[cfg(any(windows, test))]
fn validate_resolved_adapter(
    requested: AdapterIdentity,
    resolved: AdapterIdentity,
    software: bool,
) -> Result<(), OpenVrError> {
    if requested != resolved {
        return Err(OpenVrError::AdapterSelection(
            "resolved adapter LUID mismatch".into(),
        ));
    }
    if software {
        return Err(OpenVrError::AdapterSelection(
            "software output adapter rejected".into(),
        ));
    }
    Ok(())
}

#[cfg(windows)]
impl OverlayTextureSubmitter for WindowsOpenVrOverlay {
    fn set_overlay_texture(&self, texture_handle: *mut c_void) -> Result<(), OpenVrError> {
        let method = self
            .overlay_api()
            .SetOverlayTexture
            .ok_or_else(missing_overlay_method("SetOverlayTexture"))?;
        let mut descriptor = openvr_sys::Texture_t {
            handle: texture_handle,
            eType: openvr_sys::ETextureType_TextureType_DirectX,
            eColorSpace: openvr_sys::EColorSpace_ColorSpace_Auto,
        };
        let error = unsafe { method(self.overlay_handle, &mut descriptor) };
        map_overlay_submit_error(self.overlay_api(), "SetOverlayTexture", error)
    }
}

impl OverlayFrameSubmitter for FakeOpenVr {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        submit_texture(self, frame)
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        self.spatial_reanchor_count
            .set(self.spatial_reanchor_count.get() + 1);
        self.call_sequence
            .borrow_mut()
            .push("ReanchorSpatialLocked");
        Ok(SpatialReanchorOutcome::PoseUnavailable)
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        let cached_visible_before = self.visible.get();
        let actual_visible = self
            .observed_overlay_visible()
            .unwrap_or(cached_visible_before);
        let api = if actual_visible == visible {
            "SkipCachedMatch"
        } else if visible {
            "ShowOverlay"
        } else {
            "HideOverlay"
        };
        if actual_visible != visible {
            self.last_call.replace(Some(api.to_string()));
            self.visible.set(visible);
            self.observed_visible.set(Some(visible));
        } else {
            self.visible.set(visible);
        }
        self.last_visibility_api_call_log
            .replace(Some(format_openvr_visibility_api_call_log(
                visible,
                cached_visible_before,
                api,
                self.visible.get(),
            )));
        Ok(())
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        Some(
            self.observed_visible
                .get()
                .unwrap_or_else(|| self.visible.get()),
        )
    }

    fn poll_runtime_events(&mut self, max_events: usize) -> Vec<OpenVrRuntimeEvent> {
        let mut events = Vec::new();
        let mut pending = self.pending_events.borrow_mut();
        while events.len() < max_events {
            let Some(event) = pending.pop_front() else {
                break;
            };
            events.push(event);
        }
        events
    }

    fn take_visibility_api_call_log(&mut self) -> Option<String> {
        self.last_visibility_api_call_log.borrow_mut().take()
    }
}

pub(crate) fn format_openvr_visibility_api_call_log(
    desired_visible: bool,
    cached_visible_before: bool,
    api: &str,
    cached_visible_after: bool,
) -> String {
    format!(
        "openvr_overlay_visibility_api_call desired_visible={} cached_visible_before={} api={} cached_visible_after={}",
        desired_visible,
        cached_visible_before,
        api,
        cached_visible_after,
    )
}

#[cfg(windows)]
impl Drop for WindowsOpenVrOverlay {
    fn drop(&mut self) {
        if self.overlay_api.is_null() {
            return;
        }
        if let Some(destroy_overlay) = self.overlay_api().DestroyOverlay {
            unsafe {
                destroy_overlay(self.overlay_handle);
            }
        }
        unsafe {
            openvr_sys::VR_ShutdownInternal();
        }
    }
}

#[cfg(windows)]
fn initialize_overlay_api() -> Result<*mut openvr_sys::VR_IVROverlay_FnTable, OpenVrError> {
    let mut init_error = openvr_sys::EVRInitError_VRInitError_None;
    unsafe {
        openvr_sys::VR_InitInternal(
            &mut init_error,
            openvr_sys::EVRApplicationType_VRApplication_Overlay,
        );
    }
    if init_error != openvr_sys::EVRInitError_VRInitError_None {
        return Err(OpenVrError::Init(format!(
            "VR_InitInternal failed: {}",
            vr_init_error_name(init_error)
        )));
    }

    let overlay_interface_version = fn_table_interface_version(openvr_sys::IVROverlay_Version)?;
    let mut interface_error = openvr_sys::EVRInitError_VRInitError_None;
    let overlay_api = unsafe {
        openvr_sys::VR_GetGenericInterface(overlay_interface_version.as_ptr(), &mut interface_error)
    };
    if interface_error != openvr_sys::EVRInitError_VRInitError_None || overlay_api == 0 {
        unsafe {
            openvr_sys::VR_ShutdownInternal();
        }
        return Err(OpenVrError::Init(format!(
            "VR_GetGenericInterface failed: {}",
            vr_init_error_name(interface_error)
        )));
    }

    Ok(overlay_api as *mut openvr_sys::VR_IVROverlay_FnTable)
}

#[cfg(windows)]
fn initialize_system_api() -> Result<*mut openvr_sys::VR_IVRSystem_FnTable, OpenVrError> {
    let system_interface_version = fn_table_interface_version(openvr_sys::IVRSystem_Version)?;
    let mut interface_error = openvr_sys::EVRInitError_VRInitError_None;
    let system_api = unsafe {
        openvr_sys::VR_GetGenericInterface(system_interface_version.as_ptr(), &mut interface_error)
    };
    if interface_error != openvr_sys::EVRInitError_VRInitError_None || system_api == 0 {
        unsafe {
            openvr_sys::VR_ShutdownInternal();
        }
        return Err(OpenVrError::Init(format!(
            "VR_GetGenericInterface failed: {}",
            vr_init_error_name(interface_error)
        )));
    }

    Ok(system_api as *mut openvr_sys::VR_IVRSystem_FnTable)
}

#[cfg(windows)]
fn initialize_compositor_api() -> Result<*mut openvr_sys::VR_IVRCompositor_FnTable, OpenVrError> {
    let compositor_interface_version =
        fn_table_interface_version(openvr_sys::IVRCompositor_Version)?;
    let mut interface_error = openvr_sys::EVRInitError_VRInitError_None;
    let compositor_api = unsafe {
        openvr_sys::VR_GetGenericInterface(
            compositor_interface_version.as_ptr(),
            &mut interface_error,
        )
    };
    if interface_error != openvr_sys::EVRInitError_VRInitError_None || compositor_api == 0 {
        return Err(OpenVrError::Init(format!(
            "VR_GetGenericInterface (compositor) failed: {}",
            vr_init_error_name(interface_error)
        )));
    }

    Ok(compositor_api as *mut openvr_sys::VR_IVRCompositor_FnTable)
}

#[cfg(any(windows, test))]
fn fn_table_interface_version(interface_version: &[u8]) -> Result<CString, OpenVrError> {
    let version = CStr::from_bytes_with_nul(interface_version)
        .map_err(|error| OpenVrError::Init(format!("invalid OpenVR interface version: {error}")))?;
    let mut prefixed =
        Vec::with_capacity(FN_TABLE_INTERFACE_PREFIX.len() + version.to_bytes_with_nul().len());
    prefixed.extend_from_slice(FN_TABLE_INTERFACE_PREFIX.as_bytes());
    prefixed.extend_from_slice(version.to_bytes());
    CString::new(prefixed)
        .map_err(|error| OpenVrError::Init(format!("invalid OpenVR interface version: {error}")))
}

#[cfg(windows)]
fn create_overlay_handle(
    overlay_api: *mut openvr_sys::VR_IVROverlay_FnTable,
    overlay_instance_id: &str,
) -> Result<openvr_sys::VROverlayHandle_t, OpenVrError> {
    let key = CString::new(format!("{OVERLAY_KEY_PREFIX}{overlay_instance_id}"))
        .map_err(|error| OpenVrError::Init(error.to_string()))?;
    let name = CString::new(format!("{OVERLAY_NAME_PREFIX}{overlay_instance_id}"))
        .map_err(|error| OpenVrError::Init(error.to_string()))?;
    let create_overlay = unsafe { (*overlay_api).CreateOverlay }
        .ok_or_else(missing_overlay_method("CreateOverlay"))?;
    let mut handle = 0;
    let error = unsafe {
        create_overlay(
            key.as_ptr().cast_mut(),
            name.as_ptr().cast_mut(),
            &mut handle,
        )
    };
    if error != openvr_sys::EVROverlayError_VROverlayError_None {
        unsafe {
            openvr_sys::VR_ShutdownInternal();
        }
        return Err(OpenVrError::Init(format!(
            "CreateOverlay failed: {}",
            overlay_error_name(unsafe { &*overlay_api }, error)
        )));
    }
    Ok(handle)
}

#[cfg(windows)]
fn missing_overlay_method(method_name: &'static str) -> impl FnOnce() -> OpenVrError {
    move || OpenVrError::Init(format!("missing OpenVR overlay method: {method_name}"))
}

#[cfg(windows)]
fn map_overlay_init_error(
    overlay_api: &openvr_sys::VR_IVROverlay_FnTable,
    method_name: &str,
    error: openvr_sys::EVROverlayError,
) -> Result<(), OpenVrError> {
    if error == openvr_sys::EVROverlayError_VROverlayError_None {
        return Ok(());
    }
    Err(OpenVrError::Init(format!(
        "{method_name} failed: {}",
        overlay_error_name(overlay_api, error)
    )))
}

#[cfg(windows)]
fn map_overlay_submit_error(
    overlay_api: &openvr_sys::VR_IVROverlay_FnTable,
    method_name: &str,
    error: openvr_sys::EVROverlayError,
) -> Result<(), OpenVrError> {
    if error == openvr_sys::EVROverlayError_VROverlayError_None {
        return Ok(());
    }
    Err(OpenVrError::Submit(format!(
        "{method_name} failed: {}",
        overlay_error_name(overlay_api, error)
    )))
}

#[cfg(windows)]
fn overlay_error_name(
    overlay_api: &openvr_sys::VR_IVROverlay_FnTable,
    error: openvr_sys::EVROverlayError,
) -> String {
    let Some(get_error_name) = overlay_api.GetOverlayErrorNameFromEnum else {
        return format!("code {error}");
    };
    let name = unsafe { get_error_name(error) };
    if name.is_null() {
        return format!("code {error}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(windows)]
fn vr_init_error_name(error: openvr_sys::EVRInitError) -> String {
    let name = unsafe { openvr_sys::VR_GetVRInitErrorAsSymbol(error) };
    if name.is_null() {
        return format!("code {error}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{
        fn_table_interface_version, reanchor_spatial_locked_with_api, requested_adapter_identity,
        run_startup_preflight, spatial_locked_transform, split_output_device_luid,
        validate_resolved_adapter, FakeOpenVr, OpenVrBackgroundInitError, OpenVrError,
        OpenVrEventClass, OpenVrPreflightApi, OpenVrRuntimeEvent, OpenVrStartupPreflightError,
        OverlayAnchorMode, OverlayFrameSubmitter, OverlayPlacementApi, OverlayPlacementPolicy,
        SpatialHmdPose, SpatialReanchorApi, SpatialReanchorOutcome, SpatialTrackingOrigin,
        DEFAULT_OVERLAY_WIDTH_METERS, OPENVR_EVENT_OVERLAY_HIDDEN, OPENVR_EVENT_QUIT,
    };
    use crate::state::OverlayCalibration;

    enum FakeBackgroundInitResult {
        Ok,
        NoServer,
        OtherError(&'static str),
    }

    struct FakePreflightApi {
        runtime_installed: bool,
        background_init: FakeBackgroundInitResult,
        hmd_present: bool,
        shutdown_calls: Cell<usize>,
    }

    #[derive(Default)]
    struct FakeSpatialReanchorApi {
        origin: Option<SpatialTrackingOrigin>,
        pose: Option<SpatialHmdPose>,
        calls: Vec<&'static str>,
        pose_origin: Option<SpatialTrackingOrigin>,
        absolute_origin: Option<SpatialTrackingOrigin>,
        absolute_transform: Option<[[f32; 4]; 3]>,
    }

    #[derive(Default)]
    struct FakePlacementApi {
        widths: Vec<f32>,
        relative_transforms: Vec<[[f32; 4]; 3]>,
    }

    impl OverlayPlacementApi for FakePlacementApi {
        fn set_width(&mut self, width_meters: f32) -> Result<(), OpenVrError> {
            self.widths.push(width_meters);
            Ok(())
        }

        fn set_hmd_relative_transform(
            &mut self,
            transform: [[f32; 4]; 3],
        ) -> Result<(), OpenVrError> {
            self.relative_transforms.push(transform);
            Ok(())
        }
    }

    impl SpatialReanchorApi for FakeSpatialReanchorApi {
        fn active_tracking_origin(&mut self) -> Result<Option<SpatialTrackingOrigin>, OpenVrError> {
            self.calls.push("GetTrackingSpace");
            Ok(self.origin)
        }

        fn hmd_pose(
            &mut self,
            origin: SpatialTrackingOrigin,
        ) -> Result<Option<SpatialHmdPose>, OpenVrError> {
            self.calls.push("GetDeviceToAbsoluteTrackingPose");
            self.pose_origin = Some(origin);
            Ok(self.pose)
        }

        fn set_absolute_transform(
            &mut self,
            origin: SpatialTrackingOrigin,
            transform: [[f32; 4]; 3],
        ) -> Result<(), OpenVrError> {
            self.calls.push("SetOverlayTransformAbsolute");
            self.absolute_origin = Some(origin);
            self.absolute_transform = Some(transform);
            Ok(())
        }
    }

    impl FakePreflightApi {
        fn shutdown_calls(&self) -> usize {
            self.shutdown_calls.get()
        }
    }

    impl OpenVrPreflightApi for FakePreflightApi {
        fn is_runtime_installed(&self) -> bool {
            self.runtime_installed
        }

        fn initialize_background_app(&self) -> Result<(), OpenVrBackgroundInitError> {
            match self.background_init {
                FakeBackgroundInitResult::Ok => Ok(()),
                FakeBackgroundInitResult::NoServer => {
                    Err(OpenVrBackgroundInitError::NoServerForBackgroundApp)
                }
                FakeBackgroundInitResult::OtherError(message) => {
                    Err(OpenVrBackgroundInitError::Init(message.to_string()))
                }
            }
        }

        fn shutdown_runtime(&self) {
            self.shutdown_calls.set(self.shutdown_calls.get() + 1);
        }

        fn is_hmd_present(&self) -> bool {
            self.hmd_present
        }
    }

    #[test]
    fn startup_preflight_maps_missing_runtime_to_specific_failure_reason() {
        let api = FakePreflightApi {
            runtime_installed: false,
            background_init: FakeBackgroundInitResult::Ok,
            hmd_present: true,
            shutdown_calls: Cell::new(0),
        };

        let result = run_startup_preflight(&api);

        assert_eq!(
            result,
            Err(OpenVrStartupPreflightError::SteamVrNotInstalled)
        );
        assert_eq!(api.shutdown_calls(), 0);
    }

    #[test]
    fn placement_policy_defaults_to_wider_readable_overlay_width() {
        let policy = OverlayPlacementPolicy::default();

        assert!((policy.width_meters - 1.0667).abs() < 0.0001);
    }

    #[test]
    fn placement_policy_scales_wider_overlay_width_with_text_calibration() {
        let policy = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            text_scale: 1.2,
            ..OverlayCalibration::default()
        })
        .unwrap();

        assert!((policy.width_meters - 1.28004).abs() < 0.001);
    }

    #[test]
    fn placement_policy_parses_supported_anchor_modes() {
        let head_locked =
            OverlayPlacementPolicy::from_calibration(&OverlayCalibration::default()).unwrap();
        let spatial_locked = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "spatial_locked".to_string(),
            ..OverlayCalibration::default()
        })
        .unwrap();

        assert!(head_locked.is_head_locked());
        assert!(spatial_locked.is_spatial_locked());
        assert_eq!(
            OverlayAnchorMode::from_anchor("spatial_locked").unwrap(),
            OverlayAnchorMode::SpatialLocked
        );
    }

    #[test]
    fn placement_policy_rejects_unknown_anchor_mode() {
        let error = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "unsupported".to_string(),
            ..OverlayCalibration::default()
        })
        .unwrap_err();

        assert_eq!(
            error,
            super::OpenVrError::Calibration(
                "unsupported overlay calibration anchor: unsupported".to_string()
            )
        );
    }

    #[test]
    fn spatial_reanchor_uses_active_origin_for_pose_and_absolute_transform() {
        let policy = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "spatial_locked".to_string(),
            offset_x: 0.25,
            offset_y: -0.5,
            distance: 1.5,
            ..OverlayCalibration::default()
        })
        .unwrap();
        let mut api = FakeSpatialReanchorApi {
            origin: Some(SpatialTrackingOrigin::Seated),
            pose: Some(SpatialHmdPose {
                matrix: [
                    [1.0, 0.0, 0.0, 4.0],
                    [0.0, 1.0, 0.0, 5.0],
                    [0.0, 0.0, 1.0, 6.0],
                ],
                connected: true,
                valid: true,
            }),
            ..FakeSpatialReanchorApi::default()
        };

        let outcome = reanchor_spatial_locked_with_api(&policy, &mut api).unwrap();

        assert_eq!(outcome, SpatialReanchorOutcome::Applied);
        assert_eq!(api.pose_origin, Some(SpatialTrackingOrigin::Seated));
        assert_eq!(api.absolute_origin, Some(SpatialTrackingOrigin::Seated));
        assert_eq!(
            api.calls,
            vec![
                "GetTrackingSpace",
                "GetDeviceToAbsoluteTrackingPose",
                "SetOverlayTransformAbsolute"
            ]
        );
        assert_eq!(
            api.absolute_transform,
            Some([
                [1.0, 0.0, 0.0, 4.25],
                [0.0, 1.0, 0.0, 4.5],
                [0.0, 0.0, 1.0, 4.5],
            ])
        );
    }

    #[test]
    fn calibration_placement_calls_split_by_anchor_mode() {
        let spatial = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "spatial_locked".to_string(),
            text_scale: 1.5,
            offset_x: 0.25,
            offset_y: -0.5,
            distance: 1.5,
            ..OverlayCalibration::default()
        })
        .unwrap();
        let mut spatial_api = FakePlacementApi::default();

        spatial.apply_with_api(&mut spatial_api).unwrap();

        assert_eq!(spatial_api.widths, vec![DEFAULT_OVERLAY_WIDTH_METERS * 1.5]);
        assert!(spatial_api.relative_transforms.is_empty());

        let head = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "head_locked".to_string(),
            text_scale: 1.5,
            offset_x: 0.25,
            offset_y: -0.5,
            distance: 1.5,
            ..OverlayCalibration::default()
        })
        .unwrap();
        let mut head_api = FakePlacementApi::default();

        head.apply_with_api(&mut head_api).unwrap();

        assert_eq!(head_api.widths, vec![DEFAULT_OVERLAY_WIDTH_METERS * 1.5]);
        assert_eq!(
            head_api.relative_transforms,
            vec![[
                [1.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 0.0, -0.5],
                [0.0, 0.0, 1.0, -1.5],
            ]]
        );
    }

    #[test]
    fn spatial_reanchor_skips_transform_for_missing_origin_or_invalid_pose() {
        let policy = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "spatial_locked".to_string(),
            ..OverlayCalibration::default()
        })
        .unwrap();
        let mut missing_origin = FakeSpatialReanchorApi::default();

        assert_eq!(
            reanchor_spatial_locked_with_api(&policy, &mut missing_origin).unwrap(),
            SpatialReanchorOutcome::PoseUnavailable
        );
        assert_eq!(missing_origin.calls, vec!["GetTrackingSpace"]);

        let mut invalid_pose = FakeSpatialReanchorApi {
            origin: Some(SpatialTrackingOrigin::Standing),
            pose: Some(SpatialHmdPose {
                matrix: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                connected: true,
                valid: false,
            }),
            ..FakeSpatialReanchorApi::default()
        };

        assert_eq!(
            reanchor_spatial_locked_with_api(&policy, &mut invalid_pose).unwrap(),
            SpatialReanchorOutcome::PoseUnavailable
        );
        assert_eq!(
            invalid_pose.calls,
            vec!["GetTrackingSpace", "GetDeviceToAbsoluteTrackingPose"]
        );
        assert!(invalid_pose.absolute_transform.is_none());
    }

    #[test]
    fn spatial_transform_removes_roll_and_preserves_pitch() {
        let policy = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "spatial_locked".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            distance: 1.0,
            ..OverlayCalibration::default()
        })
        .unwrap();
        let rolled = spatial_locked_transform(
            &policy,
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )
        .unwrap();

        assert_eq!(
            rolled,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        );

        let pitched = spatial_locked_transform(
            &policy,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.8660254, -0.5, 0.0],
                [0.0, 0.5, 0.8660254, 0.0],
            ],
        )
        .unwrap();

        assert!((pitched[1][3] - 0.5).abs() < 0.0001);
        assert!((pitched[2][3] + 0.8660254).abs() < 0.0001);
        assert!((pitched[1][2] + 0.5).abs() < 0.0001);
        assert!((pitched[2][2] - 0.8660254).abs() < 0.0001);
    }

    #[test]
    fn spatial_transform_rejects_non_finite_and_vertical_forward_pose() {
        let policy = OverlayPlacementPolicy::from_calibration(&OverlayCalibration {
            anchor: "spatial_locked".to_string(),
            ..OverlayCalibration::default()
        })
        .unwrap();
        let mut non_finite = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        non_finite[0][3] = f32::NAN;

        assert!(spatial_locked_transform(&policy, non_finite).is_none());
        assert!(spatial_locked_transform(
            &policy,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        .is_none());
    }

    #[test]
    fn fake_openvr_records_explicit_spatial_reanchor_attempts() {
        let mut openvr = FakeOpenVr::default();

        assert_eq!(
            openvr.reanchor_spatial_locked().unwrap(),
            SpatialReanchorOutcome::PoseUnavailable
        );
        assert_eq!(openvr.spatial_reanchor_count(), 1);
        assert_eq!(openvr.call_sequence(), vec!["ReanchorSpatialLocked"]);
    }

    #[test]
    fn startup_preflight_maps_background_no_server_to_runtime_not_running() {
        let api = FakePreflightApi {
            runtime_installed: true,
            background_init: FakeBackgroundInitResult::NoServer,
            hmd_present: true,
            shutdown_calls: Cell::new(0),
        };

        let result = run_startup_preflight(&api);

        assert_eq!(result, Err(OpenVrStartupPreflightError::SteamVrNotRunning));
        assert_eq!(api.shutdown_calls(), 0);
    }

    #[test]
    fn startup_preflight_maps_missing_hmd_after_successful_background_probe() {
        let api = FakePreflightApi {
            runtime_installed: true,
            background_init: FakeBackgroundInitResult::Ok,
            hmd_present: false,
            shutdown_calls: Cell::new(0),
        };

        let result = run_startup_preflight(&api);

        assert_eq!(result, Err(OpenVrStartupPreflightError::HmdNotFound));
        assert_eq!(api.shutdown_calls(), 1);
    }

    #[test]
    fn startup_preflight_preserves_unexpected_background_init_failures() {
        let api = FakePreflightApi {
            runtime_installed: true,
            background_init: FakeBackgroundInitResult::OtherError("unexpected"),
            hmd_present: true,
            shutdown_calls: Cell::new(0),
        };

        let result = run_startup_preflight(&api);

        assert_eq!(
            result,
            Err(OpenVrStartupPreflightError::Init("unexpected".to_string()))
        );
        assert_eq!(api.shutdown_calls(), 0);
    }

    #[test]
    fn startup_preflight_succeeds_after_all_guards_pass() {
        let api = FakePreflightApi {
            runtime_installed: true,
            background_init: FakeBackgroundInitResult::Ok,
            hmd_present: true,
            shutdown_calls: Cell::new(0),
        };

        let result = run_startup_preflight(&api);

        assert_eq!(result, Ok(()));
        assert_eq!(api.shutdown_calls(), 1);
    }

    #[test]
    fn fn_table_interface_version_prefixes_overlay_version_for_flat_api_requests() {
        let request = fn_table_interface_version(b"IVROverlay_028\0").expect("request");

        assert_eq!(request.to_bytes_with_nul(), b"FnTable:IVROverlay_028\0");
    }

    #[test]
    fn openvr_output_device_value_is_split_as_a_dxgi_luid() {
        assert_eq!(
            split_output_device_luid(0xFFFF_FFFE_1234_5678),
            (-2, 0x1234_5678)
        );
    }

    #[test]
    fn adapter_selection_rejects_zero_mismatch_and_software_routes() {
        assert!(matches!(
            requested_adapter_identity(0),
            Err(super::OpenVrError::AdapterSelection(_))
        ));
        let requested = super::AdapterIdentity::DxgiLuid { high: 1, low: 2 };
        let mismatch = super::AdapterIdentity::DxgiLuid { high: 1, low: 3 };
        assert!(validate_resolved_adapter(requested, mismatch, false).is_err());
        assert!(validate_resolved_adapter(requested, requested, true).is_err());
        assert_eq!(
            validate_resolved_adapter(requested, requested, false),
            Ok(())
        );
    }

    #[test]
    fn fake_openvr_visibility_diagnostic_reports_show_and_skip_cached_match() {
        let mut openvr = FakeOpenVr::default();

        openvr.set_overlay_visible(true).expect("show overlay");
        let show_log = openvr
            .take_visibility_api_call_log()
            .expect("show visibility log");
        assert!(show_log.contains("openvr_overlay_visibility_api_call"));
        assert!(show_log.contains("desired_visible=true"));
        assert!(show_log.contains("cached_visible_before=false"));
        assert!(show_log.contains("api=ShowOverlay"));
        assert!(show_log.contains("cached_visible_after=true"));

        openvr
            .set_overlay_visible(true)
            .expect("skip cached visibility match");
        let skip_log = openvr
            .take_visibility_api_call_log()
            .expect("skip visibility log");
        assert!(skip_log.contains("desired_visible=true"));
        assert!(skip_log.contains("cached_visible_before=true"));
        assert!(skip_log.contains("api=SkipCachedMatch"));
        assert!(skip_log.contains("cached_visible_after=true"));
    }

    #[test]
    fn fake_openvr_reasserts_show_when_cached_visible_but_actual_hidden() {
        let mut openvr = FakeOpenVr::default();
        openvr.set_overlay_visible(true).expect("show overlay");
        openvr.set_observed_overlay_visible(Some(false));

        openvr
            .set_overlay_visible(true)
            .expect("reassert show overlay");
        let log = openvr
            .take_visibility_api_call_log()
            .expect("reassert visibility log");
        assert!(log.contains("desired_visible=true"));
        assert!(log.contains("cached_visible_before=true"));
        assert!(log.contains("api=ShowOverlay"));
        assert_eq!(openvr.observed_overlay_visible(), Some(true));
    }

    #[test]
    fn openvr_runtime_events_classify_fatal_reconfigure_and_ignore() {
        assert_eq!(
            OpenVrRuntimeEvent::from_event_type(OPENVR_EVENT_QUIT).classify(),
            OpenVrEventClass::Fatal
        );
        assert_eq!(
            OpenVrRuntimeEvent::from_event_type(OPENVR_EVENT_OVERLAY_HIDDEN).classify(),
            OpenVrEventClass::Reconfigure
        );
        assert_eq!(
            OpenVrRuntimeEvent::from_event_type(1).classify(),
            OpenVrEventClass::Ignore
        );
    }

    #[test]
    fn fake_openvr_poll_runtime_events_are_bounded() {
        let mut openvr = FakeOpenVr::default();
        for _ in 0..8 {
            openvr.push_runtime_event(OpenVrRuntimeEvent::Ignored(1));
        }
        openvr.push_runtime_event(OpenVrRuntimeEvent::Quit);

        let first = openvr.poll_runtime_events(3);
        assert_eq!(first.len(), 3);
        assert!(first
            .iter()
            .all(|event| *event == OpenVrRuntimeEvent::Ignored(1)));
        let rest = openvr.poll_runtime_events(16);
        assert_eq!(rest.len(), 6);
        assert_eq!(rest[5], OpenVrRuntimeEvent::Quit);
    }
}
