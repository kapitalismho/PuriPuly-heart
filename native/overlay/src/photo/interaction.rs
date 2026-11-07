use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::manifest::PhotoError;
use super::state::{Photo, PhotoPhase};
use crate::openvr::{LegacyVrInput, TrackedHand};

pub type Transform = [[f32; 4]; 3];
pub const IDENTITY: Transform = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemporaryInteractionConfig {
    pub grab_m: f32,
    pub icon_width_m: f32,
    pub photo_width_m: f32,
    pub facing_show_deg: f32,
    pub forward_show_deg: f32,
    pub facing_hide_deg: f32,
    pub forward_hide_deg: f32,
    pub dwell_ms: u64,
    pub wrist_transform: Transform,
    pub photo_transform: Transform,
}

impl Default for TemporaryInteractionConfig {
    fn default() -> Self {
        Self {
            grab_m: 0.07,
            icon_width_m: 0.03,
            photo_width_m: 0.375,
            facing_show_deg: 45.0,
            forward_show_deg: 40.0,
            facing_hide_deg: 55.0,
            forward_hide_deg: 45.0,
            dwell_ms: 50,
            wrist_transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.035],
                [0.0, -1.0, 0.0, 0.08],
            ],
            photo_transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.12],
                [0.0, 0.0, 1.0, -0.08],
            ],
        }
    }
}

impl TemporaryInteractionConfig {
    pub fn validate(&self) -> Result<(), PhotoError> {
        if ![self.grab_m, self.icon_width_m, self.photo_width_m]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
            || self.photo_width_m < self.icon_width_m
            || ![
                (self.facing_show_deg, self.facing_hide_deg),
                (self.forward_show_deg, self.forward_hide_deg),
            ]
            .iter()
            .all(|(show, hide)| {
                show.is_finite() && hide.is_finite() && *show > 0.0 && show < hide && *hide < 90.0
            })
            || self.dwell_ms > 5000
            || !valid_transform(self.wrist_transform)
            || !valid_transform(self.photo_transform)
        {
            return Err(PhotoError::Manifest);
        }
        Ok(())
    }
}

pub fn valid_transform(t: Transform) -> bool {
    t.iter().flatten().all(|v| v.is_finite())
        && (0..3).all(|i| {
            (0..3).all(|j| {
                let dot: f32 = (0..3).map(|k| t[k][i] * t[k][j]).sum();
                (dot - if i == j { 1.0 } else { 0.0 }).abs() < 0.01
            })
        })
        && dot(column(t, 0), cross(column(t, 1), column(t, 2))) > 0.99
}

pub fn compose(a: Transform, b: Transform) -> Transform {
    let mut out = IDENTITY;
    for i in 0..3 {
        for j in 0..4 {
            out[i][j] =
                (0..3).map(|k| a[i][k] * b[k][j]).sum::<f32>() + if j == 3 { a[i][3] } else { 0.0 };
        }
    }
    out
}

pub fn inverse(t: Transform) -> Transform {
    let mut out = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = t[j][i];
        }
        out[i][3] = -(0..3).map(|k| t[k][i] * t[k][3]).sum::<f32>();
    }
    out
}

fn column(t: Transform, i: usize) -> [f32; 3] {
    [t[0][i], t[1][i], t[2][i]]
}
fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn distance(a: Transform, b: Transform) -> f32 {
    let d = subtract(column(a, 3), column(b, 3));
    dot(d, d).sqrt()
}
fn within(a: [f32; 3], b: [f32; 3], degrees: f32) -> bool {
    let norm = (dot(a, a) * dot(b, b)).sqrt();
    norm > 0.000001 && dot(a, b) >= norm * degrees.to_radians().cos()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HapticCue {
    Pull,
    FirstBlock,
}
impl HapticCue {
    pub fn hand(self) -> TrackedHand {
        match self {
            Self::Pull => TrackedHand::Left,
            Self::FirstBlock => TrackedHand::Right,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub device: u32,
    pub transform: Transform,
    pub width: f32,
    pub alpha: f32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct InteractionFrame {
    pub icon: Option<Placement>,
    pub photo: Option<Placement>,
    pub glow: bool,
    pub pull: bool,
    pub hidden: bool,
    pub grab: bool,
}

struct Grab {
    start_distance: f32,
    relative: Transform,
}
struct Return {
    started: Instant,
    from: Transform,
}

pub struct PhotoInteraction {
    pub config: TemporaryInteractionConfig,
    key: Option<String>,
    roles: Option<(u32, u32)>,
    grip: bool,
    armed: bool,
    change_since: Option<Instant>,
    gaze: bool,
    shown_at: Option<Instant>,
    icon_alpha: f32,
    grab: Option<Grab>,
    returning: Option<Return>,
    held: bool,
    photo_alpha: f32,
    last: Option<Instant>,
    deadline: Option<Instant>,
    offer_deadline: Option<Instant>,
    offer_pulled: bool,
    pub frame: InteractionFrame,
}

impl PhotoInteraction {
    pub fn new(config: TemporaryInteractionConfig) -> Self {
        Self {
            config,
            key: None,
            roles: None,
            grip: false,
            armed: false,
            change_since: None,
            gaze: false,
            shown_at: None,
            icon_alpha: 0.0,
            grab: None,
            returning: None,
            held: false,
            photo_alpha: 0.0,
            last: None,
            deadline: None,
            offer_deadline: None,
            offer_pulled: false,
            frame: InteractionFrame::default(),
        }
    }

    pub fn snapshot(&mut self, photo: Option<&Photo>, now: Instant, unix_ms: u64) {
        let key = photo.map(|p| p.photo_id.as_str());
        if self.key.as_deref() != key {
            let config = self.config.clone();
            *self = Self::new(config);
            self.key = key.map(str::to_owned);
        }
        for (deadline, lifetime) in [
            (&mut self.offer_deadline, 10_000),
            (&mut self.deadline, 600_000),
        ] {
            *deadline = photo.map(|p| {
                let remaining = p
                    .shutter_unix_ms
                    .saturating_add(lifetime)
                    .saturating_sub(unix_ms)
                    .min(lifetime);
                let candidate = now + Duration::from_millis(remaining);
                deadline.map_or(candidate, |old| old.min(candidate))
            });
        }
    }

    pub fn active(&self, photo: &Photo, now: Instant) -> bool {
        self.key.as_deref() == Some(photo.photo_id.as_str())
            && photo.phase != PhotoPhase::Expired
            && (if photo.attempt_id == 0 {
                self.offer_deadline
            } else {
                self.deadline
            })
            .is_some_and(|deadline| now < deadline)
    }

    pub fn step(
        &mut self,
        photo: Option<&Photo>,
        input: Option<LegacyVrInput>,
        now: Instant,
    ) -> InteractionFrame {
        let dt = self
            .last
            .replace(now)
            .map_or(0.0, |old| now.saturating_duration_since(old).as_secs_f32());
        let prior_held = self.held;
        let active = photo.is_some_and(|p| self.active(p, now));
        let Some(input) = input.filter(|_| active) else {
            self.armed = false;
            self.grip = true;
            self.grab = None;
            self.returning = None;
            self.held = false;
            self.photo_alpha = 0.0;
            self.gaze = false;
            self.change_since = None;
            self.shown_at = None;
            self.icon_alpha = 0.0;
            self.frame = InteractionFrame {
                hidden: prior_held,
                ..Default::default()
            };
            return self.frame;
        };
        let roles = (input.left_index, input.right_index);
        if self.roles != Some(roles) {
            self.roles = Some(roles);
            self.armed = false;
            self.grip = true;
            self.grab = None;
            self.held = false;
            self.photo_alpha = 0.0;
            self.returning = None;
            self.gaze = false;
            self.change_since = None;
            self.shown_at = None;
            self.icon_alpha = 0.0;
        }
        if !input.grip {
            self.armed = true;
        }
        let pressed = self.armed && input.grip && !self.grip;
        self.grip = input.grip;
        let wrist = compose(input.right, self.config.wrist_transform);
        let head_to_icon = subtract(column(wrist, 3), column(input.head, 3));
        let facing = subtract(column(input.head, 3), column(wrist, 3));
        let forward = column(input.head, 2).map(|v| -v);
        let looking = within(
            column(wrist, 2),
            facing,
            if self.gaze {
                self.config.facing_hide_deg
            } else {
                self.config.facing_show_deg
            },
        ) && within(
            forward,
            head_to_icon,
            if self.gaze {
                self.config.forward_hide_deg
            } else {
                self.config.forward_show_deg
            },
        );
        let want = if self.returning.is_some() {
            true
        } else if self.grab.is_some() {
            self.gaze
        } else {
            looking
        };
        if want == self.gaze {
            self.change_since = None;
        } else {
            let since = self.change_since.get_or_insert(now);
            let wait = if want { self.config.dwell_ms } else { 300 };
            if now.saturating_duration_since(*since) >= Duration::from_millis(wait) {
                self.gaze = want;
                self.change_since = None;
                if want {
                    self.shown_at = Some(now);
                }
            }
        }
        self.icon_alpha = approach(
            self.icon_alpha,
            if self.gaze { 1.0 } else { 0.0 },
            dt.min(0.1) / 0.1,
        );
        let interactive = self.icon_alpha >= 1.0
            && self.shown_at.is_some_and(|shown| {
                now.saturating_duration_since(shown) >= Duration::from_millis(200)
            });
        let photo = photo.unwrap();
        let current_distance = distance(input.left, wrist);
        let near = current_distance <= self.config.grab_m;
        let mut out = InteractionFrame {
            glow: near && interactive && photo.can_pull(),
            hidden: prior_held && !self.held,
            ..Default::default()
        };
        if pressed
            && near
            && interactive
            && photo.can_pull()
            && (photo.attempt_id > 0 || !self.offer_pulled)
            && self.grab.is_none()
            && !self.held
            && self.returning.is_none()
        {
            self.grab = Some(Grab {
                start_distance: current_distance,
                relative: compose(inverse(input.left), wrist),
            });
            out.grab = true;
        }
        if let Some(grab) = &self.grab {
            let icon = compose(input.left, grab.relative);
            let triggered = input.grip && current_distance - grab.start_distance >= 0.12;
            if !input.grip || triggered {
                self.returning = Some(Return {
                    started: now,
                    from: compose(inverse(input.right), icon),
                });
                self.grab = None;
                if triggered {
                    out.pull = true;
                    if photo.attempt_id == 0 {
                        self.offer_pulled = true;
                    }
                    self.held = photo.attempt_id > 0;
                }
            }
        }
        if self.held && (!input.grip || !photo.can_pull()) {
            self.held = false;
            out.hidden = true;
        }
        self.photo_alpha = approach(
            self.photo_alpha,
            if self.held { 1.0 } else { 0.0 },
            if prior_held != self.held {
                0.0
            } else {
                dt / 0.15
            },
        );
        if self.photo_alpha > 0.0 {
            out.photo = Some(Placement {
                device: input.left_index,
                transform: self.config.photo_transform,
                width: self.config.icon_width_m
                    + (self.config.photo_width_m - self.config.icon_width_m) * self.photo_alpha,
                alpha: self.photo_alpha,
            });
        }
        if self.icon_alpha > 0.0 {
            let (device, transform) = if let Some(grab) = &self.grab {
                (input.left_index, grab.relative)
            } else if let Some(returning) = &self.returning {
                let t = (now
                    .saturating_duration_since(returning.started)
                    .as_secs_f32()
                    / 0.225)
                    .min(1.0);
                let ease = 1.0 - (1.0 - t) * (1.0 - t);
                let mut transform = self.config.wrist_transform;
                for i in 0..3 {
                    transform[i][3] =
                        returning.from[i][3] + (transform[i][3] - returning.from[i][3]) * ease;
                }
                if t >= 1.0 {
                    self.returning = None;
                }
                (input.right_index, transform)
            } else {
                (input.right_index, self.config.wrist_transform)
            };
            out.icon = Some(Placement {
                device,
                transform,
                width: self.config.icon_width_m,
                alpha: self.icon_alpha,
            });
        } else if self
            .returning
            .as_ref()
            .is_some_and(|r| now.saturating_duration_since(r.started) >= Duration::from_millis(225))
        {
            self.returning = None;
        }
        self.frame = out;
        out
    }
}

fn approach(value: f32, target: f32, amount: f32) -> f32 {
    if target > value {
        (value + amount).min(target)
    } else {
        (value - amount).max(target)
    }
}

#[cfg(test)]
#[path = "interaction_tests.rs"]
mod tests;
