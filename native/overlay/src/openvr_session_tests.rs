use super::*;
use std::cell::RefCell;

#[derive(Default)]
struct Calls {
    keys: Vec<String>,
    destroyed: Vec<u64>,
    shutdowns: usize,
    fail_create: bool,
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
unsafe extern "C" fn width(_: u64, _: f32) -> openvr_sys::EVROverlayError {
    0
}
unsafe extern "C" fn transform(
    _: u64,
    _: u32,
    _: *mut openvr_sys::HmdMatrix34_t,
) -> openvr_sys::EVROverlayError {
    0
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
