use serde::{Deserialize, Serialize};

use super::interaction::{compose, inverse, valid_transform, InteractionFrame, Placement, TemporaryInteractionConfig, Transform, IDENTITY};
use crate::openvr::LegacyVrInput;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WristReference {
    pub model: String,
    pub serial: String,
    pub handgrip_transform: Transform,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WristCalibration {
    pub transform: Transform,
    pub reference: WristReference,
}

impl WristCalibration {
    pub fn valid(&self) -> bool {
        valid_transform(self.transform)
            && valid_transform(self.reference.handgrip_transform)
            && [&self.reference.model, &self.reference.serial].into_iter().all(|value| {
                !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control)
            })
    }
}

#[derive(Default)]
pub struct WristAdjustment {
    roles: Option<(u32, u32, u64)>,
    pressed: bool,
    armed: bool,
    relative: Option<Transform>,
}

impl WristAdjustment {
    pub fn step(&mut self, input: Option<LegacyVrInput>, config: &TemporaryInteractionConfig) -> (InteractionFrame, Option<Transform>) {
        let Some(input) = input else {
            *self = Self::default();
            return (InteractionFrame::default(), None);
        };
        let roles = (input.left_index, input.right_index, input.reference_epoch);
        if self.roles != Some(roles) {
            *self = Self { roles: Some(roles), pressed: true, ..Self::default() };
        }
        let wrist_local = config.effective_wrist_transform(input.right_handgrip);
        let wrist = compose(input.right, wrist_local);
        let left = compose(input.left, input.left_handgrip);
        let distance_squared = (0..3).map(|i| (left[i][3] - wrist[i][3]).powi(2)).sum::<f32>();
        if !input.grip {
            self.armed = true;
        }
        if self.armed && input.grip && !self.pressed && distance_squared <= config.grab_m * config.grab_m {
            self.relative = Some(compose(inverse(input.left), wrist));
        }
        self.pressed = input.grip;
        let mut committed = None;
        let (device, transform) = if let Some(relative) = self.relative {
            if !input.grip {
                let world = compose(input.left, relative);
                let right_local = compose(inverse(input.right), world);
                committed = Some(compose(inverse(input.right_handgrip), right_local));
                self.relative = None;
                (input.right_index, right_local)
            } else {
                (input.left_index, relative)
            }
        } else {
            (input.right_index, wrist_local)
        };
        let mut label_offset = IDENTITY;
        label_offset[1][3] = -0.08;
        (InteractionFrame {
            icon: Some(Placement { device, transform, width: config.icon_width_m, alpha: 1.0 }),
            case_label: Some(Placement { device, transform: compose(transform, label_offset), width: 0.24, alpha: 1.0 }),
            glow: distance_squared <= config.grab_m * config.grab_m,
            ..Default::default()
        }, committed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(grip: bool) -> LegacyVrInput {
        LegacyVrInput {
            head: IDENTITY, left: IDENTITY, right: IDENTITY,
            left_handgrip: IDENTITY, right_handgrip: IDENTITY,
            reference_epoch: 1, left_index: 1, right_index: 2, grip, right_grip: false,
        }
    }

    fn close(a: Transform, b: Transform) {
        for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
            assert!((a - b).abs() < 0.00001);
        }
    }

    #[test]
    fn manual_drag_preserves_pickup_offset_and_saves_full_wrist_relative_pose() {
        let mut adjustment = WristAdjustment::default();
        let mut wrist = IDENTITY;
        wrist[0][3] = 0.04;
        let config = TemporaryInteractionConfig { wrist_transform: Some(wrist), ..Default::default() };
        adjustment.step(Some(input(false)), &config);
        let (grab, saved) = adjustment.step(Some(input(true)), &config);
        assert!(saved.is_none());
        assert!(!grab.pull && !grab.grab && grab.photo.is_none());
        close(grab.icon.unwrap().transform, wrist);
        let mut moved = input(true);
        moved.left = [[0.0,-1.0,0.0,0.2],[1.0,0.0,0.0,0.1],[0.0,0.0,1.0,0.03]];
        moved.right[1][3] = 0.05;
        moved.right_handgrip[2][3] = 0.02;
        let (drag, saved) = adjustment.step(Some(moved), &config);
        assert!(saved.is_none() && !drag.pull && drag.photo.is_none());
        moved.grip = false;
        let (released, saved) = adjustment.step(Some(moved), &config);
        let saved = saved.unwrap();
        close(compose(compose(moved.right, moved.right_handgrip), saved), compose(moved.left, wrist));
        close(compose(moved.right, released.icon.unwrap().transform), compose(moved.left, wrist));
        assert!(valid_transform(saved));
    }

    #[test]
    fn held_start_outside_press_and_tracking_loss_never_save_or_authorize() {
        let config = TemporaryInteractionConfig { wrist_transform: Some(IDENTITY), ..Default::default() };
        let mut adjustment = WristAdjustment::default();
        adjustment.step(Some(input(true)), &config);
        assert!(adjustment.step(Some(input(false)), &config).1.is_none());
        let mut outside = input(true);
        outside.left[0][3] = 0.11;
        adjustment.step(Some(outside), &config);
        adjustment.step(Some(input(true)), &config);
        assert!(adjustment.step(Some(input(false)), &config).1.is_none());
        adjustment.step(Some(input(true)), &config);
        assert!(adjustment.step(None, &config).0.icon.is_none());
        adjustment.step(Some(input(true)), &config);
        assert!(adjustment.step(Some(input(false)), &config).1.is_none());
        adjustment.step(Some(input(true)), &config);
        let mut replaced = input(false);
        replaced.reference_epoch += 1;
        assert!(adjustment.step(Some(replaced), &config).1.is_none());
    }
}
