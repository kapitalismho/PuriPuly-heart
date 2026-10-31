use super::*;
use std::cell::RefCell;

#[derive(Default)]
struct Calls {
    keys: Vec<String>,
    destroyed: Vec<u64>,
    shutdowns: usize,
    fail_create: bool,
    transforms: Vec<(u32, [[f32; 4]; 3])>,
    widths: Vec<f32>,
    alphas: Vec<f32>,
    fail_alpha: bool,
}

thread_local! { static CALLS: RefCell<Calls> = RefCell::new(Calls::default()); }

unsafe extern "C" fn create(
    key: *mut i8,
    _: *mut i8,
    handle: *mut u64,
) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        if calls.fail_create {
            return openvr_sys::EVROverlayError_VROverlayError_UnknownOverlay;
        }
        calls
            .keys
            .push(unsafe { CStr::from_ptr(key) }.to_str().unwrap().into());
        unsafe {
            *handle = calls.keys.len() as u64;
        }
        openvr_sys::EVROverlayError_VROverlayError_None
    })
}
unsafe extern "C" fn destroy(handle: u64) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| calls.borrow_mut().destroyed.push(handle));
    openvr_sys::EVROverlayError_VROverlayError_None
}
unsafe extern "C" fn pid(_: u64, _: u32) -> openvr_sys::EVROverlayError {
    0
}
unsafe extern "C" fn flag(
    _: u64,
    _: openvr_sys::VROverlayFlags,
    _: bool,
) -> openvr_sys::EVROverlayError {
    0
}
unsafe extern "C" fn width(_: u64, width: f32) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| calls.borrow_mut().widths.push(width));
    0
}
unsafe extern "C" fn transform(
    _: u64,
    device: u32,
    transform: *mut openvr_sys::HmdMatrix34_t,
) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| {
        calls
            .borrow_mut()
            .transforms
            .push((device, unsafe { (*transform).m }))
    });
    0
}
unsafe extern "C" fn alpha(_: u64, alpha: f32) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.alphas.push(alpha);
        if calls.fail_alpha {
            1
        } else {
            0
        }
    })
}
fn shutdown() {
    CALLS.with(|calls| calls.borrow_mut().shutdowns += 1);
}

fn session(api: &mut openvr_sys::VR_IVROverlay_FnTable) -> OpenVrSession {
    OpenVrSession {
        inner: Rc::new(WindowsOpenVrSession {
            overlay_api: api,
            system_api: std::ptr::null_mut(),
            compositor_api: None,
            output_adapter: OpenVrOutputAdapter {
                identity: AdapterIdentity::Test,
                requested_identity: AdapterIdentity::Test,
                adapter: None,
            },
            _runtime: OpenVrRuntimeLease { shutdown },
        }),
    }
}

fn api() -> openvr_sys::VR_IVROverlay_FnTable {
    let mut api: openvr_sys::VR_IVROverlay_FnTable = unsafe { std::mem::zeroed() };
    api.CreateOverlay = Some(create);
    api.DestroyOverlay = Some(destroy);
    api.SetOverlayRenderingPid = Some(pid);
    api.SetOverlayFlag = Some(flag);
    api.SetOverlayWidthInMeters = Some(width);
    api.SetOverlayTransformTrackedDeviceRelative = Some(transform);
    api.SetOverlayAlpha = Some(alpha);
    api
}

#[test]
fn closing_each_surface_preserves_session_until_last_owner_drops() {
    CALLS.with(|calls| *calls.borrow_mut() = Calls::default());
    let mut api = api();
    let session = session(&mut api);
    let icon = OpenVrOverlay::in_session(&session, "photo.instance.icon").unwrap();
    let photo = OpenVrOverlay::in_session(&session, "photo.instance.photo").unwrap();
    drop(icon);
    CALLS.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(calls.destroyed, [1]);
        assert_eq!(calls.shutdowns, 0);
        assert_ne!(calls.keys[0], calls.keys[1]);
    });
    drop(session);
    CALLS.with(|calls| assert_eq!(calls.borrow().shutdowns, 0));
    drop(photo);
    CALLS.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(calls.destroyed, [1, 2]);
        assert_eq!(calls.shutdowns, 1);
    });
}

#[test]
fn failed_surface_creation_does_not_shutdown_other_live_surface() {
    CALLS.with(|calls| *calls.borrow_mut() = Calls::default());
    let mut api = api();
    let session = session(&mut api);
    let icon = OpenVrOverlay::in_session(&session, "photo.instance.icon").unwrap();
    CALLS.with(|calls| calls.borrow_mut().fail_create = true);
    assert!(OpenVrOverlay::in_session(&session, "photo.instance.photo").is_err());
    CALLS.with(|calls| assert_eq!(calls.borrow().shutdowns, 0));
    drop(icon);
    drop(session);
    CALLS.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(calls.destroyed, [1]);
        assert_eq!(calls.shutdowns, 1);
    });
}

#[derive(Default)]
struct LegacyCalls {
    left: u32,
    right: u32,
    grip: bool,
    fail_state: bool,
    invalid_pose: Option<usize>,
    pulses: Vec<(u32, u32, u16)>,
}

thread_local! { static LEGACY: RefCell<LegacyCalls> = RefCell::new(LegacyCalls { left: 1, right: 2, ..Default::default() }); }

unsafe extern "C" fn controller_role(role: openvr_sys::ETrackedControllerRole) -> u32 {
    LEGACY.with(|calls| {
        let calls = calls.borrow();
        if role == openvr_sys::ETrackedControllerRole_TrackedControllerRole_LeftHand {
            calls.left
        } else {
            calls.right
        }
    })
}

unsafe extern "C" fn controller_state(
    _: u32,
    state: *mut openvr_sys::VRControllerState_t,
    size: u32,
) -> bool {
    assert_eq!(
        size as usize,
        std::mem::size_of::<openvr_sys::VRControllerState_t>()
    );
    LEGACY.with(|calls| {
        let calls = calls.borrow();
        unsafe {
            (*state).ulButtonPressed = if calls.grip {
                1u64 << openvr_sys::EVRButtonId_k_EButton_Grip
            } else {
                0
            };
        }
        !calls.fail_state
    })
}

unsafe extern "C" fn poses(
    origin: openvr_sys::ETrackingUniverseOrigin,
    _: f32,
    poses: *mut openvr_sys::TrackedDevicePose_t,
    count: u32,
) {
    assert_eq!(
        origin,
        openvr_sys::ETrackingUniverseOrigin_TrackingUniverseStanding
    );
    LEGACY.with(|calls| {
        let calls = calls.borrow();
        for i in 0..count as usize {
            let pose = unsafe { &mut *poses.add(i) };
            pose.bDeviceIsConnected = true;
            pose.bPoseIsValid = calls.invalid_pose != Some(i);
            pose.mDeviceToAbsoluteTracking.m = [
                [1.0, 0.0, 0.0, i as f32],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ];
        }
    });
}

unsafe extern "C" fn pulse(device: u32, axis: u32, duration: u16) {
    LEGACY.with(|calls| calls.borrow_mut().pulses.push((device, axis, duration)));
}

#[test]
fn legacy_grip_poses_and_haptic_hands_resolve_roles_fresh_and_fail_closed() {
    LEGACY.with(|calls| {
        *calls.borrow_mut() = LegacyCalls {
            left: 1,
            right: 2,
            ..Default::default()
        }
    });
    let mut overlay_api = api();
    let mut system_api: openvr_sys::VR_IVRSystem_FnTable = unsafe { std::mem::zeroed() };
    system_api.GetTrackedDeviceIndexForControllerRole = Some(controller_role);
    system_api.GetControllerState = Some(controller_state);
    system_api.GetDeviceToAbsoluteTrackingPose = Some(poses);
    system_api.TriggerHapticPulse = Some(pulse);
    let mut session = session(&mut overlay_api);
    Rc::get_mut(&mut session.inner).unwrap().system_api = &mut system_api;
    assert!(!session.legacy_input().unwrap().grip);
    LEGACY.with(|calls| calls.borrow_mut().grip = true);
    assert!(session.legacy_input().unwrap().grip);
    use crate::photo::interaction::HapticCue;
    assert!(session.pulse(HapticCue::Grab.hand()));
    assert!(session.pulse(HapticCue::Pull.hand()));
    LEGACY.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.left = 5;
        calls.right = 6;
    });
    let fresh = session.legacy_input().unwrap();
    assert_eq!((fresh.left_index, fresh.right_index), (5, 6));
    assert_eq!(fresh.left[0][3], 5.0);
    assert!(session.pulse(HapticCue::FirstBlock.hand()));
    LEGACY.with(|calls| {
        assert_eq!(
            calls.borrow().pulses,
            [(1, 0, 900), (1, 0, 900), (6, 0, 900)]
        )
    });
    LEGACY.with(|calls| calls.borrow_mut().fail_state = true);
    assert!(session.legacy_input().is_none());
    assert!(!session.pulse(TrackedHand::Left));
    LEGACY.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.fail_state = false;
        calls.invalid_pose = Some(6);
    });
    assert!(session.legacy_input().is_none());
    assert!(!session.pulse(TrackedHand::Right));
    LEGACY.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.invalid_pose = None;
        calls.right = 5;
    });
    assert!(session.legacy_input().is_none());
    LEGACY.with(|calls| assert_eq!(calls.borrow().pulses.len(), 3));
}

#[test]
fn tracked_surface_applies_real_pose_width_alpha_and_only_caches_complete_success() {
    CALLS.with(|calls| *calls.borrow_mut() = Calls::default());
    let mut api = api();
    let session = session(&mut api);
    let mut icon = OpenVrOverlay::in_session(&session, "photo.tracked.icon").unwrap();
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.transforms.clear();
        calls.widths.clear();
        calls.alphas.clear();
    });
    let transform = [
        [1.0, 0.0, 0.0, 0.03],
        [0.0, 1.0, 0.0, 0.04],
        [0.0, 0.0, 1.0, 0.05],
    ];
    icon.set_tracked_presentation(7, transform, 0.03, 0.5)
        .unwrap();
    icon.set_tracked_presentation(7, transform, 0.03, 0.5)
        .unwrap();
    CALLS.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(calls.transforms, [(7, transform)]);
        assert_eq!(calls.widths, [0.03]);
        assert_eq!(calls.alphas, [0.5]);
    });
    CALLS.with(|calls| calls.borrow_mut().fail_alpha = true);
    assert!(icon
        .set_tracked_presentation(7, transform, 0.03, 0.75)
        .is_err());
    CALLS.with(|calls| calls.borrow_mut().fail_alpha = false);
    icon.set_tracked_presentation(7, transform, 0.03, 0.75)
        .unwrap();
    CALLS.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(calls.transforms.len(), 3);
        assert_eq!(calls.alphas, [0.5, 0.75, 0.75]);
    });
}
