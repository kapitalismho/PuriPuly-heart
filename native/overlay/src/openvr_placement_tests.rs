use std::cell::RefCell;
use std::mem::ManuallyDrop;

use super::{
    OpenVrError, OpenVrOutputAdapter, OverlayPlacementPolicy, SpatialReanchorOutcome,
    WindowsOpenVrOverlay, DEFAULT_OVERLAY_WIDTH_METERS,
};
use crate::presentation::AdapterIdentity;
use crate::state::OverlayCalibration;

#[derive(Debug, Clone, PartialEq)]
enum Call {
    Width(u64, f32),
    Relative(u64, u32, [[f32; 4]; 3]),
    Origin,
    Pose(openvr_sys::ETrackingUniverseOrigin),
    Absolute(u64, openvr_sys::ETrackingUniverseOrigin, [[f32; 4]; 3]),
}

struct ControlledApi {
    calls: Vec<Call>,
    counts: [usize; 5],
    fail_width: bool,
    fail_relative: bool,
    pose_valid: bool,
    pose_x: f32,
    width: Option<f32>,
    relative: Option<[[f32; 4]; 3]>,
    absolute: Option<[[f32; 4]; 3]>,
}

impl Default for ControlledApi {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            counts: [0; 5],
            fail_width: false,
            fail_relative: false,
            pose_valid: true,
            pose_x: 4.0,
            width: None,
            relative: None,
            absolute: None,
        }
    }
}

impl ControlledApi {
    fn record(&mut self, index: usize, call: Call) {
        self.counts[index] += 1;
        self.calls.push(call);
    }
}

thread_local! {
    static API: RefCell<ControlledApi> = RefCell::new(ControlledApi::default());
}

unsafe extern "C" fn set_width(handle: u64, width: f32) -> openvr_sys::EVROverlayError {
    API.with_borrow_mut(|api| {
        api.record(0, Call::Width(handle, width));
        if std::mem::take(&mut api.fail_width) {
            return openvr_sys::EVROverlayError_VROverlayError_InvalidHandle;
        }
        api.width = Some(width);
        openvr_sys::EVROverlayError_VROverlayError_None
    })
}

unsafe extern "C" fn set_relative(
    handle: u64,
    device: u32,
    matrix: *mut openvr_sys::HmdMatrix34_t,
) -> openvr_sys::EVROverlayError {
    let matrix = unsafe { (*matrix).m };
    API.with_borrow_mut(|api| {
        api.record(1, Call::Relative(handle, device, matrix));
        if std::mem::take(&mut api.fail_relative) {
            return openvr_sys::EVROverlayError_VROverlayError_InvalidHandle;
        }
        api.relative = Some(matrix);
        openvr_sys::EVROverlayError_VROverlayError_None
    })
}

unsafe extern "C" fn tracking_origin() -> openvr_sys::ETrackingUniverseOrigin {
    API.with_borrow_mut(|api| api.record(2, Call::Origin));
    openvr_sys::ETrackingUniverseOrigin_TrackingUniverseStanding
}

unsafe extern "C" fn get_pose(
    origin: openvr_sys::ETrackingUniverseOrigin,
    _prediction: f32,
    poses: *mut openvr_sys::TrackedDevicePose_t,
    count: u32,
) {
    API.with_borrow_mut(|api| {
        api.record(3, Call::Pose(origin));
        if count > 0 {
            unsafe {
                *poses = openvr_sys::TrackedDevicePose_t {
                    mDeviceToAbsoluteTracking: openvr_sys::HmdMatrix34_t {
                        m: [
                            [1.0, 0.0, 0.0, api.pose_x],
                            [0.0, 1.0, 0.0, 5.0],
                            [0.0, 0.0, 1.0, 6.0],
                        ],
                    },
                    bDeviceIsConnected: true,
                    bPoseIsValid: api.pose_valid,
                    ..Default::default()
                };
            }
        }
    });
}

unsafe extern "C" fn set_absolute(
    handle: u64,
    origin: openvr_sys::ETrackingUniverseOrigin,
    matrix: *mut openvr_sys::HmdMatrix34_t,
) -> openvr_sys::EVROverlayError {
    let matrix = unsafe { (*matrix).m };
    API.with_borrow_mut(|api| {
        api.record(4, Call::Absolute(handle, origin, matrix));
        api.absolute = Some(matrix);
    });
    openvr_sys::EVROverlayError_VROverlayError_None
}

unsafe extern "C" fn set_pid(_handle: u64, _pid: u32) -> openvr_sys::EVROverlayError {
    openvr_sys::EVROverlayError_VROverlayError_None
}

unsafe extern "C" fn set_flag(
    _handle: u64,
    _flag: openvr_sys::VROverlayFlags,
    _enabled: bool,
) -> openvr_sys::EVROverlayError {
    openvr_sys::EVROverlayError_VROverlayError_None
}

struct ControlledOwner {
    owner: ManuallyDrop<WindowsOpenVrOverlay>,
    _overlay: Box<openvr_sys::VR_IVROverlay_FnTable>,
    _system: Box<openvr_sys::VR_IVRSystem_FnTable>,
    _compositor: Box<openvr_sys::VR_IVRCompositor_FnTable>,
}

impl ControlledOwner {
    fn new(handle: u64) -> Self {
        let mut overlay: Box<openvr_sys::VR_IVROverlay_FnTable> =
            Box::new(unsafe { std::mem::zeroed() });
        overlay.SetOverlayWidthInMeters = Some(set_width);
        overlay.SetOverlayTransformTrackedDeviceRelative = Some(set_relative);
        overlay.SetOverlayTransformAbsolute = Some(set_absolute);
        overlay.SetOverlayRenderingPid = Some(set_pid);
        overlay.SetOverlayFlag = Some(set_flag);
        let mut system: Box<openvr_sys::VR_IVRSystem_FnTable> =
            Box::new(unsafe { std::mem::zeroed() });
        system.GetDeviceToAbsoluteTrackingPose = Some(get_pose);
        let mut compositor: Box<openvr_sys::VR_IVRCompositor_FnTable> =
            Box::new(unsafe { std::mem::zeroed() });
        compositor.GetTrackingSpace = Some(tracking_origin);
        let owner = ManuallyDrop::new(WindowsOpenVrOverlay {
            overlay_api: &mut *overlay,
            system_api: &mut *system,
            compositor_api: Some(&mut *compositor),
            overlay_handle: handle,
            placement_policy: OverlayPlacementPolicy::default(),
            applied_placement_policy: None,
            visible: false,
            output_adapter: OpenVrOutputAdapter {
                identity: AdapterIdentity::Unavailable,
                requested_identity: AdapterIdentity::Unavailable,
                adapter: None,
            },
        });
        Self {
            owner,
            _overlay: overlay,
            _system: system,
            _compositor: compositor,
        }
    }
}

fn reset() {
    API.set(ControlledApi::default());
}

fn counts() -> [usize; 5] {
    API.with_borrow(|api| api.counts)
}

fn relative(calibration: &OverlayCalibration) -> [[f32; 4]; 3] {
    [
        [1.0, 0.0, 0.0, calibration.offset_x],
        [0.0, 1.0, 0.0, calibration.offset_y],
        [0.0, 0.0, 1.0, -calibration.distance.max(0.1)],
    ]
}

#[test]
fn placement_owner_skips_only_successful_effective_settings() {
    reset();
    let mut fixture = ControlledOwner::new(71);
    fixture.owner.configure_overlay().unwrap();
    let mut calibration = OverlayCalibration::default();
    fixture.owner.apply_calibration(&calibration).unwrap();
    calibration.background_alpha = 0.75;
    fixture.owner.apply_calibration(&calibration).unwrap();
    assert_eq!(counts(), [1, 1, 0, 0, 0]);

    for change in 0..4 {
        match change {
            0 => calibration.text_scale = 1.5,
            1 => calibration.offset_x = 0.25,
            2 => calibration.offset_y = -0.5,
            _ => calibration.distance = 1.5,
        }
        fixture.owner.apply_calibration(&calibration).unwrap();
        fixture.owner.apply_calibration(&calibration).unwrap();
        API.with_borrow(|api| {
            assert_eq!(
                api.width,
                Some(DEFAULT_OVERLAY_WIDTH_METERS * calibration.text_scale)
            );
            assert_eq!(api.relative, Some(relative(&calibration)));
        });
    }
    assert_eq!(counts(), [5, 5, 0, 0, 0]);
    API.with_borrow(|api| {
        for pair in api.calls.chunks_exact(2) {
            assert!(matches!(pair[0], Call::Width(71, _)));
            assert!(matches!(pair[1], Call::Relative(71, 0, _)));
        }
    });

    calibration.anchor = "unsupported".to_string();
    assert!(matches!(
        fixture.owner.apply_calibration(&calibration),
        Err(OpenVrError::Calibration(_))
    ));
    assert_eq!(counts(), [5, 5, 0, 0, 0]);
}

#[test]
fn placement_owner_retries_width_and_partial_transform_failures() {
    for fail_width in [true, false] {
        reset();
        let mut fixture = ControlledOwner::new(72);
        let calibration = OverlayCalibration::default();
        API.with_borrow_mut(|api| {
            api.fail_width = fail_width;
            api.fail_relative = !fail_width;
        });
        let error = fixture.owner.apply_calibration(&calibration).unwrap_err();
        assert!(matches!(error, OpenVrError::Init(_)));
        let expected_method = if fail_width {
            "SetOverlayWidthInMeters"
        } else {
            "SetOverlayTransformTrackedDeviceRelative"
        };
        assert!(error.to_string().contains(expected_method));
        assert_eq!(counts(), [1, usize::from(!fail_width), 0, 0, 0]);
        fixture.owner.apply_calibration(&calibration).unwrap();
        fixture.owner.apply_calibration(&calibration).unwrap();
        assert_eq!(counts(), [2, 1 + usize::from(!fail_width), 0, 0, 0]);
        API.with_borrow(|api| {
            assert_eq!(api.width, Some(DEFAULT_OVERLAY_WIDTH_METERS));
            assert_eq!(api.relative, Some(relative(&calibration)));
        });
    }
}

#[test]
fn placement_owner_restores_earlier_policy_after_partial_failure() {
    reset();
    let mut fixture = ControlledOwner::new(73);
    let original = OverlayCalibration::default();
    let changed = OverlayCalibration {
        text_scale: 1.5,
        offset_x: 0.25,
        ..OverlayCalibration::default()
    };
    fixture.owner.apply_calibration(&original).unwrap();
    API.with_borrow_mut(|api| api.fail_relative = true);
    assert!(fixture.owner.apply_calibration(&changed).is_err());
    API.with_borrow(|api| {
        assert_eq!(api.width, Some(DEFAULT_OVERLAY_WIDTH_METERS * 1.5));
        assert_eq!(api.relative, Some(relative(&original)));
    });
    fixture.owner.apply_calibration(&original).unwrap();
    fixture.owner.apply_calibration(&original).unwrap();
    assert_eq!(counts(), [3, 3, 0, 0, 0]);
    API.with_borrow(|api| {
        assert_eq!(api.width, Some(DEFAULT_OVERLAY_WIDTH_METERS));
        assert_eq!(api.relative, Some(relative(&original)));
    });
}

#[test]
fn placement_owner_recreation_always_applies_even_for_reused_handle() {
    reset();
    for handle in [74, 75, 75] {
        let mut fixture = ControlledOwner::new(handle);
        fixture.owner.configure_overlay().unwrap();
        fixture
            .owner
            .apply_calibration(&OverlayCalibration::default())
            .unwrap();
    }
    assert_eq!(counts(), [3, 3, 0, 0, 0]);
    API.with_borrow(|api| {
        let handles: Vec<_> = api
            .calls
            .iter()
            .filter_map(|call| match call {
                Call::Width(handle, _) => Some(*handle),
                _ => None,
            })
            .collect();
        assert_eq!(handles, [74, 75, 75]);
    });
}

#[test]
fn placement_owner_keeps_explicit_reanchor_and_pose_recovery_independent() {
    reset();
    let mut fixture = ControlledOwner::new(76);
    let head = OverlayCalibration::default();
    let spatial = OverlayCalibration {
        anchor: "spatial_locked".to_string(),
        offset_x: 0.25,
        offset_y: -0.5,
        distance: 1.5,
        ..OverlayCalibration::default()
    };
    fixture.owner.apply_calibration(&head).unwrap();
    fixture.owner.apply_calibration(&spatial).unwrap();
    fixture.owner.apply_calibration(&spatial).unwrap();
    API.with_borrow_mut(|api| api.pose_valid = false);
    assert_eq!(
        fixture.owner.reanchor_spatial_locked().unwrap(),
        SpatialReanchorOutcome::PoseUnavailable
    );
    assert_eq!(counts(), [2, 1, 1, 1, 0]);
    API.with_borrow_mut(|api| api.pose_valid = true);
    fixture.owner.apply_calibration(&spatial).unwrap();
    assert_eq!(
        fixture.owner.reanchor_spatial_locked().unwrap(),
        SpatialReanchorOutcome::Applied
    );
    API.with_borrow(|api| {
        assert_eq!(
            api.absolute,
            Some([
                [1.0, 0.0, 0.0, 4.25],
                [0.0, 1.0, 0.0, 4.5],
                [0.0, 0.0, 1.0, 4.5],
            ])
        );
    });
    API.with_borrow_mut(|api| api.pose_x = 8.0);
    assert_eq!(
        fixture.owner.reanchor_spatial_locked().unwrap(),
        SpatialReanchorOutcome::Applied
    );
    API.with_borrow(|api| assert_eq!(api.absolute.unwrap()[0][3], 8.25));
    assert_eq!(counts(), [2, 1, 3, 3, 2]);
    fixture.owner.apply_calibration(&head).unwrap();
    assert_eq!(counts(), [3, 2, 3, 3, 2]);
    API.with_borrow(|api| assert_eq!(api.relative, Some(relative(&head))));
}
