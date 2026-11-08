use super::*;
use std::cell::RefCell;

#[derive(Default)]
struct Calls {
    keys: Vec<String>,
    destroyed: Vec<u64>,
    shutdowns: usize,
    fail_create: bool,
    fail_create_at: Option<usize>,
    transforms: Vec<(u32, [[f32; 4]; 3])>,
    widths: Vec<f32>,
    surface_widths: Vec<(u64, f32)>,
    alphas: Vec<f32>,
    fail_alpha: bool,
    fail_show: Option<u64>,
    visible: std::collections::HashMap<u64, bool>,
    textures: Vec<(u64, usize)>,
    shown: Vec<u64>,
}

thread_local! { static CALLS: RefCell<Calls> = RefCell::new(Calls::default()); }

unsafe extern "C" fn create(
    key: *mut i8,
    _: *mut i8,
    handle: *mut u64,
) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        if calls.fail_create || calls.fail_create_at == Some(calls.keys.len() + 1) {
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
unsafe extern "C" fn width(handle: u64, width: f32) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        calls.widths.push(width);
        calls.surface_widths.push((handle, width));
    });
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
unsafe extern "C" fn texture(
    handle: u64,
    texture: *mut openvr_sys::Texture_t,
) -> openvr_sys::EVROverlayError {
    let texture = unsafe { &*texture };
    assert!(!texture.handle.is_null());
    assert_eq!(texture.eType, openvr_sys::ETextureType_TextureType_DirectX);
    CALLS.with(|calls| {
        calls
            .borrow_mut()
            .textures
            .push((handle, texture.handle as usize))
    });
    0
}

unsafe extern "C" fn show(handle: u64) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        if calls.fail_show == Some(handle) {
            return 1;
        }
        calls.visible.insert(handle, true);
        calls.shown.push(handle);
        0
    })
}

unsafe extern "C" fn hide(handle: u64) -> openvr_sys::EVROverlayError {
    CALLS.with(|calls| calls.borrow_mut().visible.insert(handle, false));
    0
}

unsafe extern "C" fn visible(handle: u64) -> bool {
    CALLS.with(|calls| {
        calls
            .borrow()
            .visible
            .get(&handle)
            .copied()
            .unwrap_or(false)
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
            hand_references: RefCell::new(PhotoHandReferences::default()),
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
    api.SetOverlayTexture = Some(texture);
    api.ShowOverlay = Some(show);
    api.HideOverlay = Some(hide);
    api.IsOverlayVisible = Some(visible);
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

#[test]
fn diagnostic_surfaces_share_session_and_partial_creation_cleans_every_owned_handle() {
    use crate::photo::manifest::PhotoManifest;
    use crate::photo::runtime::PhotoOpenVrSurfaces;
    let manifest: PhotoManifest = serde_json::from_value(serde_json::json!({
        "protocol_version":2, "role":"photo", "bridge_url":"ws://127.0.0.1:12345",
        "session_token":"synthetic", "instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1, "backend":"openvr", "image":{"debug_controls":true},
        "interaction":{"debug_grab_volume":true,"debug_case_label":"Radius 12 cm\nPosition B fixed"}
    })).unwrap();
    for fail_at in [None, Some(4), Some(6)] {
        CALLS.with(|calls| *calls.borrow_mut() = Calls { fail_create_at: fail_at, ..Default::default() });
        let mut api = api();
        let session = session(&mut api);
        let surfaces = PhotoOpenVrSurfaces::in_session(&manifest, session);
        if fail_at.is_some() {
            assert!(surfaces.is_err());
        } else {
            let mut surfaces = surfaces.unwrap();
            surfaces.close_icon();
            surfaces.close_photo();
            CALLS.with(|calls| {
                let calls = calls.borrow();
                assert_eq!(calls.shutdowns, 0);
                assert_eq!(calls.keys.len(), 6);
                assert_eq!(calls.keys.iter().collect::<std::collections::HashSet<_>>().len(), 6);
                assert_eq!(calls.destroyed, [1, 2]);
            });
            drop(surfaces);
        }
        CALLS.with(|calls| {
            let calls = calls.borrow();
            let mut destroyed = calls.destroyed.clone();
            destroyed.sort_unstable();
            assert_eq!(destroyed, (1..=calls.keys.len() as u64).collect::<Vec<_>>());
            assert_eq!(calls.shutdowns, 1);
        });
    }
}

#[derive(Default)]
struct LegacyCalls {
    left: u32,
    right: u32,
    grip: bool,
    fail_state: bool,
    invalid_pose: Option<usize>,
    interaction_distance: Option<f32>,
    pulses: Vec<(u32, u32, u16)>,
    replacement: bool,
    missing_component: bool,
    invalid_component: bool,
    model_changed: bool,
    component_changed: bool,
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
            if let Some(distance) = calls.interaction_distance {
                pose.mDeviceToAbsoluteTracking.m[0][3] = if i == calls.left as usize {
                    distance
                } else {
                    0.0
                };
                pose.mDeviceToAbsoluteTracking.m[2][3] = if i == 0 { 0.5 } else { 0.0 };
                if i == calls.left as usize {
                    pose.mDeviceToAbsoluteTracking.m[1][3] = 0.03;
                    pose.mDeviceToAbsoluteTracking.m[2][3] = -0.04;
                }
            }
        }
    });
}

unsafe extern "C" fn pulse(device: u32, axis: u32, duration: u16) {
    LEGACY.with(|calls| calls.borrow_mut().pulses.push((device, axis, duration)));
}

unsafe extern "C" fn device_property(
    index: u32,
    property: openvr_sys::ETrackedDeviceProperty,
    value: *mut i8,
    capacity: u32,
    error: *mut openvr_sys::ETrackedPropertyError,
) -> u32 {
    LEGACY.with(|calls| {
        let calls = calls.borrow();
        let text = if property == openvr_sys::ETrackedDeviceProperty_Prop_RenderModelName_String {
            if calls.model_changed {
                "replacement-model"
            } else {
                "actual-model"
            }
        } else if calls.replacement {
            "replacement-device"
        } else if index == calls.left {
            "actual-left"
        } else {
            "actual-right"
        };
        assert!(capacity as usize > text.len());
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr().cast::<i8>(), value, text.len());
            *value.add(text.len()) = 0;
            *error = openvr_sys::ETrackedPropertyError_TrackedProp_Success;
        }
        text.len() as u32 + 1
    })
}

unsafe extern "C" fn handgrip_state(
    model: *mut i8,
    component: *mut i8,
    _: *mut openvr_sys::VRControllerState_t,
    _: *mut openvr_sys::RenderModel_ControllerMode_State_t,
    state: *mut openvr_sys::RenderModel_ComponentState_t,
) -> bool {
    LEGACY.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(unsafe { CStr::from_ptr(component) }, c"handgrip");
        assert_eq!(
            unsafe { CStr::from_ptr(model) },
            if calls.model_changed {
                c"replacement-model"
            } else {
                c"actual-model"
            },
        );
        unsafe {
            (*state).mTrackingToComponentLocal.m = [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, -0.03],
                [0.0, 0.0, 1.0, 0.04],
            ];
            if calls.component_changed {
                (*state).mTrackingToComponentLocal.m[0][0] = 1.0;
                (*state).mTrackingToComponentLocal.m[0][1] = 0.0;
                (*state).mTrackingToComponentLocal.m[1][0] = 0.0;
                (*state).mTrackingToComponentLocal.m[1][1] = 1.0;
            }
            if calls.invalid_component {
                (*state).mTrackingToComponentLocal.m[0][0] = f32::NAN;
            }
        }
        !calls.missing_component
    })
}

fn configure_input(
    session: &mut OpenVrSession,
    system: &mut openvr_sys::VR_IVRSystem_FnTable,
    models: &mut openvr_sys::VR_IVRRenderModels_FnTable,
) {
    system.GetTrackedDeviceIndexForControllerRole = Some(controller_role);
    system.GetControllerState = Some(controller_state);
    system.GetDeviceToAbsoluteTrackingPose = Some(poses);
    system.GetStringTrackedDeviceProperty = Some(device_property);
    system.TriggerHapticPulse = Some(pulse);
    models.GetComponentState = Some(handgrip_state);
    let inner = Rc::get_mut(&mut session.inner).unwrap();
    inner.system_api = system;
    *inner.hand_references.get_mut() = PhotoHandReferences {
        initialized: true,
        render_models: Some(models),
        ..Default::default()
    };
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
    let mut models = unsafe { std::mem::zeroed() };
    let mut session = session(&mut overlay_api);
    configure_input(&mut session, &mut system_api, &mut models);
    assert!(!session.legacy_input().unwrap().grip);
    LEGACY.with(|calls| calls.borrow_mut().grip = true);
    assert!(session.legacy_input().unwrap().grip);
    use crate::photo::interaction::HapticCue;
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
    LEGACY.with(|calls| assert_eq!(calls.borrow().pulses, [(1, 0, 900), (6, 0, 900)]));
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
    LEGACY.with(|calls| assert_eq!(calls.borrow().pulses.len(), 2));
}

#[test]
fn missing_invalid_or_replaced_handgrips_cancel_and_require_fresh_ready_grip() {
    use crate::photo::interaction::{PhotoInteraction, TemporaryInteractionConfig, IDENTITY};
    use crate::photo::state::Photo;
    use std::time::{Duration, Instant};
    for interruption in [
        "missing",
        "invalid",
        "model",
        "device",
        "component",
        "tracking",
    ] {
        LEGACY.with(|calls| {
            *calls.borrow_mut() = LegacyCalls {
                left: 1,
                right: 2,
                interaction_distance: Some(0.0),
                ..Default::default()
            };
        });
        let mut overlay_api = api();
        let mut system_api = unsafe { std::mem::zeroed() };
        let mut models = unsafe { std::mem::zeroed() };
        let mut session = session(&mut overlay_api);
        configure_input(&mut session, &mut system_api, &mut models);
        let photo: Photo = serde_json::from_value(serde_json::json!({
            "photo_id":"reference-loss", "path":"unused.png", "file_size":1,
            "expected_file_id":["1","2"],
            "shutter_unix_ms":1000, "attempt_id":0, "phase":"offer", "offer_open":true,
            "visible":false, "image":null, "preparations":[], "blocks":[],
            "failure_code":null, "first_block_drawn":false
        }))
        .unwrap();
        let start = Instant::now();
        let mut interaction = PhotoInteraction::new(TemporaryInteractionConfig {
            wrist_transform: Some(IDENTITY),
            ..Default::default()
        });
        interaction.snapshot(Some(&photo), start, 1000);
        let mut step = |ms| {
            interaction.step(
                Some(&photo),
                session.legacy_input(),
                start + Duration::from_millis(ms),
            )
        };
        for ms in [0, 50, 150, 350] {
            step(ms);
        }
        let original = session.legacy_input().unwrap();
        assert_eq!(original.left[1][3], 0.03);
        assert_eq!(original.left_handgrip[1][3], -0.03);
        assert_eq!(original.left_handgrip[0][1], -1.0);
        assert_eq!(
            session.legacy_input().unwrap().reference_epoch,
            original.reference_epoch
        );
        LEGACY.with(|calls| calls.borrow_mut().grip = true);
        assert!(step(360).grab);
        LEGACY.with(|calls| {
            let mut calls = calls.borrow_mut();
            calls.interaction_distance = Some(0.2);
            match interruption {
                "missing" => calls.missing_component = true,
                "invalid" => calls.invalid_component = true,
                "model" => calls.model_changed = true,
                "device" => calls.replacement = true,
                "component" => calls.component_changed = true,
                "tracking" => calls.invalid_pose = Some(1),
                _ => unreachable!(),
            }
        });
        assert_eq!(
            session.pulse(TrackedHand::Right),
            matches!(interruption, "model" | "device" | "component"),
        );
        LEGACY.with(|calls| {
            let mut calls = calls.borrow_mut();
            calls.missing_component = false;
            calls.invalid_component = false;
            calls.invalid_pose = None;
        });
        let canceled = step(380);
        assert!(canceled.icon.is_none(), "{interruption}");
        assert!(!canceled.pull, "{interruption}");
        LEGACY.with(|calls| {
            let mut calls = calls.borrow_mut();
            calls.missing_component = false;
            calls.invalid_component = false;
            calls.invalid_pose = None;
            calls.interaction_distance = Some(0.0);
        });
        for ms in [400, 450, 550, 750] {
            let recovered = step(ms);
            assert!(!recovered.grab, "{interruption}");
            assert!(!recovered.pull, "{interruption}");
        }
        assert!(session.legacy_input().unwrap().reference_epoch > original.reference_epoch);
        LEGACY.with(|calls| calls.borrow_mut().grip = false);
        step(760);
        LEGACY.with(|calls| calls.borrow_mut().grip = true);
        assert!(step(770).grab, "{interruption}");
        LEGACY.with(|calls| calls.borrow_mut().interaction_distance = Some(0.121));
        assert!(step(790).pull, "{interruption}");
    }
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

#[tokio::test]
async fn windows_graphics_controlled_held_photo_runtime_reports_first_draw_once_per_attempt() {
    for configured_width in [None, Some(0.64)] {
        CALLS.with(|calls| *calls.borrow_mut() = Calls::default());
        LEGACY.with(|calls| {
            *calls.borrow_mut() = LegacyCalls {
                left: 1,
                right: 2,
                ..Default::default()
            }
        });
        let mut overlay_api = api();
        let mut system_api: openvr_sys::VR_IVRSystem_FnTable = unsafe { std::mem::zeroed() };
        let mut models = unsafe { std::mem::zeroed() };
        let mut session = session(&mut overlay_api);
        configure_input(&mut session, &mut system_api, &mut models);
        let icon = OpenVrOverlay::in_session(&session, "photo.controlled-runtime.icon").unwrap();
        let photo = OpenVrOverlay::in_session(&session, "photo.controlled-runtime.photo").unwrap();
        crate::photo::runtime::PhotoRuntime::assert_controlled_held_render_frame(
            icon,
            photo,
            session,
            configured_width,
            || {
                CALLS.with(|calls| {
                    assert_eq!(
                        calls
                            .borrow()
                            .surface_widths
                            .iter()
                            .rev()
                            .find(|(handle, _)| *handle == 2),
                        Some(&(2, configured_width.unwrap_or(0.375))),
                    );
                })
            },
            |fail| CALLS.with(|calls| calls.borrow_mut().fail_show = fail.then_some(2)),
            |grip, distance| {
                LEGACY.with(|calls| {
                    let mut calls = calls.borrow_mut();
                    calls.grip = grip;
                    calls.interaction_distance = Some(distance);
                })
            },
            |expected| LEGACY.with(|calls| assert_eq!(calls.borrow().pulses, expected)),
        )
        .await;
        CALLS.with(|calls| {
            let calls = calls.borrow();
            assert!(calls.textures.iter().any(|(handle, _)| *handle == 2));
            assert!(calls.shown.iter().filter(|handle| **handle == 2).count() >= 3);
            assert_eq!(calls.visible.get(&2), Some(&false));
            assert_eq!(calls.destroyed.len(), 2);
            assert_eq!(calls.shutdowns, 1);
        });
        LEGACY.with(|calls| {
            assert_eq!(
                calls.borrow().pulses,
                [(1, 0, 900), (2, 0, 900), (2, 0, 900)]
            )
        });
    }
}

#[test]
fn default_photo_session_creates_only_icon_and_photo_surfaces() {
    CALLS.with(|calls| *calls.borrow_mut() = Calls::default());
    let mut api = api();
    let session = session(&mut api);
    let manifest: crate::photo::manifest::PhotoManifest = serde_json::from_value(serde_json::json!({
        "protocol_version":2, "role":"photo", "bridge_url":"ws://127.0.0.1:12345",
        "session_token":"synthetic", "instance_id":"11111111-1111-4111-8111-111111111111",
        "generation":1, "backend":"openvr"
    })).unwrap();
    let surfaces = crate::photo::runtime::PhotoOpenVrSurfaces::in_session(&manifest, session).unwrap();
    CALLS.with(|calls| assert_eq!(calls.borrow().keys.len(), 2));
    drop(surfaces);
    CALLS.with(|calls| {
        let calls = calls.borrow();
        assert_eq!(calls.destroyed.len(), 2);
        assert_eq!(calls.shutdowns, 1);
    });
}
