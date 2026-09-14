use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::renderer::CaptionLayoutPolicy;

const SLOT_COUNT: usize = 2;
const SLOT_TOP_PADDING_PX: f32 = 40.0;
const SLOT_SPACING_PX: f32 = 36.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayPresentationCalibration {
    #[serde(default = "default_anchor")]
    pub anchor: String,
    #[serde(default)]
    pub offset_x: f32,
    #[serde(default)]
    pub offset_y: f32,
    #[serde(default = "default_distance")]
    pub distance: f32,
    #[serde(default = "default_text_scale")]
    pub text_scale: f32,
    #[serde(default = "default_background_alpha")]
    pub background_alpha: f32,
}

impl Default for OverlayPresentationCalibration {
    fn default() -> Self {
        Self {
            anchor: default_anchor(),
            offset_x: 0.0,
            offset_y: 0.0,
            distance: default_distance(),
            text_scale: default_text_scale(),
            background_alpha: default_background_alpha(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverlayPresentationBlockVariant {
    ActiveSelf,
    // Reserved compatibility/fallback variant. Normal product peer rows become
    // primary-visible through translated finalized blocks, not source-only
    // active_peer rows.
    ActivePeer,
    #[default]
    Finalized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OverlayPresentationBlock {
    pub id: String,
    pub occupant_key: String,
    pub appearance_seq: u64,
    pub channel: String,
    pub block_variant: OverlayPresentationBlockVariant,
    pub primary_text: String,
    #[serde(default)]
    pub secondary_text: String,
    #[serde(default = "default_secondary_enabled")]
    pub secondary_enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary_language: Option<String>,
    #[serde(default)]
    pub update_id: Option<String>,
    #[serde(default)]
    pub origin_wall_clock_ms: Option<u64>,
    #[serde(default)]
    pub session_scope: Option<String>,
    #[serde(default)]
    pub publication_scope: Option<String>,
    #[serde(default)]
    pub publication_generation: Option<u64>,
    #[serde(default)]
    pub publication_order: Option<u64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NativeFreshRenderGenerations {
    #[serde(default, rename = "self", skip_serializing_if = "Option::is_none")]
    pub self_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<u64>,
    #[serde(skip)]
    pub self_target: Option<String>,
    #[serde(skip)]
    pub peer_target: Option<String>,
    #[serde(skip)]
    pub quiet_tail_episodes: Option<NativeQuietTailEpisodes>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NativeFreshRenderTargets {
    #[serde(default, rename = "self", skip_serializing_if = "Option::is_none")]
    pub self_target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeQuietTailPhase {
    Stream,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeQuietTailEpisode {
    pub phase: NativeQuietTailPhase,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NativeQuietTailEpisodes {
    #[serde(default, rename = "self", skip_serializing_if = "Option::is_none")]
    pub self_episode: Option<NativeQuietTailEpisode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<NativeQuietTailEpisode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticRetirementFrontier {
    pub scope: String,
    pub generation: u64,
    pub order: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OverlayPresentationSnapshot {
    pub revision: u64,
    pub calibration: OverlayPresentationCalibration,
    pub blocks: Vec<OverlayPresentationBlock>,
    pub native_fresh_render_generations: Option<NativeFreshRenderGenerations>,
    pub semantic_retirement_frontiers: Vec<SemanticRetirementFrontier>,
}
#[derive(Serialize, Deserialize)]
struct OverlayPresentationSnapshotWire {
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    calibration: OverlayPresentationCalibration,
    #[serde(default)]
    blocks: Vec<OverlayPresentationBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_fresh_render_generations: Option<NativeFreshRenderGenerations>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_fresh_render_targets: Option<NativeFreshRenderTargets>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    native_quiet_tail_episodes: Option<NativeQuietTailEpisodes>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    semantic_retirement_frontiers: Vec<SemanticRetirementFrontier>,
}

impl Serialize for OverlayPresentationSnapshot {
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where
        T: serde::Serializer,
    {
        let targets = self
            .native_fresh_render_generations
            .as_ref()
            .and_then(|generations| {
                (generations.self_target.is_some() || generations.peer_target.is_some()).then(
                    || NativeFreshRenderTargets {
                        self_target: generations.self_target.clone(),
                        peer: generations.peer_target.clone(),
                    },
                )
            });
        OverlayPresentationSnapshotWire {
            revision: self.revision,
            calibration: self.calibration.clone(),
            blocks: self.blocks.clone(),
            native_fresh_render_generations: self.native_fresh_render_generations.clone(),
            native_fresh_render_targets: targets,
            native_quiet_tail_episodes: self
                .native_fresh_render_generations
                .as_ref()
                .and_then(|generations| generations.quiet_tail_episodes.clone()),
            semantic_retirement_frontiers: self.semantic_retirement_frontiers.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OverlayPresentationSnapshot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = OverlayPresentationSnapshotWire::deserialize(deserializer)?;
        let mut generations = wire.native_fresh_render_generations;
        if let (Some(generations), Some(targets)) =
            (&mut generations, wire.native_fresh_render_targets)
        {
            generations.self_target = targets.self_target;
            generations.peer_target = targets.peer;
        }
        if let Some(generations) = &mut generations {
            generations.quiet_tail_episodes = wire.native_quiet_tail_episodes;
        }
        Ok(Self {
            revision: wire.revision,
            calibration: wire.calibration,
            blocks: wire.blocks,
            native_fresh_render_generations: generations,
            semantic_retirement_frontiers: wire.semantic_retirement_frontiers,
        })
    }
}

pub type OverlayCalibration = OverlayPresentationCalibration;
pub type OverlayStateSnapshot = OverlayPresentationSnapshot;

#[derive(Debug, Clone, PartialEq)]
pub struct OverlaySlot {
    pub slot_index: usize,
    pub anchor_top_px: f32,
    pub id: String,
    pub occupant_key: String,
    pub appearance_seq: u64,
    pub channel: String,
    pub block_variant: OverlayPresentationBlockVariant,
    pub primary_text: String,
    pub secondary_text: String,
    pub secondary_enabled: bool,
    pub primary_language: Option<String>,
    pub secondary_language: Option<String>,
    pub update_id: Option<String>,
    pub origin_wall_clock_ms: Option<u64>,
    pub session_scope: Option<String>,
    pub slot_entry_order: u64,
}

impl OverlaySlot {
    fn from_block(
        block: &OverlayPresentationBlock,
        slot_index: usize,
        slot_entry_order: u64,
    ) -> Self {
        Self {
            slot_index,
            anchor_top_px: 0.0,
            id: block.id.clone(),
            occupant_key: block.occupant_key.clone(),
            appearance_seq: block.appearance_seq,
            channel: block.channel.clone(),
            block_variant: block.block_variant,
            primary_text: block.primary_text.clone(),
            secondary_text: block.secondary_text.clone(),
            secondary_enabled: block.secondary_enabled,
            primary_language: block.primary_language.clone(),
            secondary_language: block.secondary_language.clone(),
            update_id: block.update_id.clone(),
            origin_wall_clock_ms: block.origin_wall_clock_ms,
            session_scope: block.session_scope.clone(),
            slot_entry_order,
        }
    }

    fn update_from_block(&mut self, block: &OverlayPresentationBlock) {
        self.id = block.id.clone();
        self.occupant_key = block.occupant_key.clone();
        self.appearance_seq = block.appearance_seq;
        self.channel = block.channel.clone();
        self.block_variant = block.block_variant;
        self.primary_text = block.primary_text.clone();
        self.secondary_text = block.secondary_text.clone();
        self.secondary_enabled = block.secondary_enabled;
        self.primary_language = block.primary_language.clone();
        self.secondary_language = block.secondary_language.clone();
        self.update_id = block.update_id.clone();
        self.origin_wall_clock_ms = block.origin_wall_clock_ms;
        self.session_scope = block.session_scope.clone();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OverlayScene {
    slots: [Option<OverlaySlot>; SLOT_COUNT],
    next_slot_entry_order: u64,
}

impl Default for OverlayScene {
    fn default() -> Self {
        Self {
            slots: Default::default(),
            next_slot_entry_order: 0,
        }
    }
}

impl OverlayScene {
    pub fn slots(&self) -> &[Option<OverlaySlot>; SLOT_COUNT] {
        &self.slots
    }

    pub fn slots_mut(&mut self) -> &mut [Option<OverlaySlot>; SLOT_COUNT] {
        &mut self.slots
    }

    fn apply_snapshot(&mut self, blocks: &[OverlayPresentationBlock], text_scale: f32) -> bool {
        let previous = self.slots.clone();
        let mut sorted = blocks.to_vec();
        sorted.sort_by(|left, right| {
            left.appearance_seq
                .cmp(&right.appearance_seq)
                .then_with(|| left.occupant_key.cmp(&right.occupant_key))
        });
        debug_assert!(
            sorted.len() <= SLOT_COUNT,
            "presenter must cap visible blocks to two"
        );

        let mut assigned = self.update_existing_slots(&sorted);
        self.clear_missing_slots(&sorted);
        self.fill_empty_slots(&sorted, &mut assigned);
        self.recompute_slot_anchors(text_scale);

        previous != self.slots
    }

    fn update_existing_slots(&mut self, blocks: &[OverlayPresentationBlock]) -> HashSet<String> {
        let mut assigned = HashSet::new();
        for block in blocks {
            if let Some(slot) = self
                .slots
                .iter_mut()
                .flatten()
                .find(|slot| slot.occupant_key == block.occupant_key)
            {
                slot.update_from_block(block);
                assigned.insert(block.occupant_key.clone());
            }
        }
        assigned
    }

    fn clear_missing_slots(&mut self, blocks: &[OverlayPresentationBlock]) {
        let snapshot_keys = blocks
            .iter()
            .map(|block| block.occupant_key.as_str())
            .collect::<HashSet<_>>();
        for slot in &mut self.slots {
            if slot
                .as_ref()
                .is_some_and(|existing| !snapshot_keys.contains(existing.occupant_key.as_str()))
            {
                *slot = None;
            }
        }
    }

    fn fill_empty_slots(
        &mut self,
        blocks: &[OverlayPresentationBlock],
        assigned: &mut HashSet<String>,
    ) {
        for block in blocks {
            if assigned.contains(&block.occupant_key) {
                continue;
            }
            let Some((slot_index, slot)) = self
                .slots
                .iter_mut()
                .enumerate()
                .find(|(_, slot)| slot.is_none())
            else {
                break;
            };
            *slot = Some(OverlaySlot::from_block(
                block,
                slot_index,
                self.next_slot_entry_order,
            ));
            self.next_slot_entry_order += 1;
            assigned.insert(block.occupant_key.clone());
        }
    }

    fn recompute_slot_anchors(&mut self, text_scale: f32) {
        let slot_height_px =
            CaptionLayoutPolicy::default().stable_block_height_px(true, text_scale);
        for (slot_index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(slot) = slot {
                slot.slot_index = slot_index;
                slot.anchor_top_px =
                    SLOT_TOP_PADDING_PX + slot_index as f32 * (slot_height_px + SLOT_SPACING_PX);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OverlayState {
    snapshot: OverlayPresentationSnapshot,
    scene: OverlayScene,
}

impl OverlayState {
    pub fn seed_snapshot(&mut self, snapshot: &OverlayPresentationSnapshot) -> bool {
        let scene_changed = self
            .scene
            .apply_snapshot(&snapshot.blocks, snapshot.calibration.text_scale);
        let visual_changed = self.snapshot.calibration != snapshot.calibration || scene_changed;
        self.snapshot = snapshot.clone();
        visual_changed
    }

    pub fn apply_snapshot(&mut self, snapshot: &OverlayPresentationSnapshot) -> bool {
        if snapshot.revision <= self.snapshot.revision {
            return false;
        }

        let scene_changed = self
            .scene
            .apply_snapshot(&snapshot.blocks, snapshot.calibration.text_scale);
        let visual_changed = self.snapshot.calibration != snapshot.calibration || scene_changed;
        self.snapshot = snapshot.clone();
        visual_changed
    }

    pub fn snapshot(&self) -> &OverlayPresentationSnapshot {
        &self.snapshot
    }

    pub fn calibration(&self) -> &OverlayPresentationCalibration {
        &self.snapshot.calibration
    }

    pub fn blocks(&self) -> &[OverlayPresentationBlock] {
        &self.snapshot.blocks
    }

    pub fn native_fresh_render_generations(&self) -> Option<&NativeFreshRenderGenerations> {
        self.snapshot.native_fresh_render_generations.as_ref()
    }

    pub fn scene(&self) -> &OverlayScene {
        &self.scene
    }
}

fn default_anchor() -> String {
    "head_locked".to_string()
}

fn default_distance() -> f32 {
    1.1
}

fn default_text_scale() -> f32 {
    1.0
}

fn default_background_alpha() -> f32 {
    0.24
}

fn default_secondary_enabled() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::{NativeFreshRenderGenerations, OverlayPresentationSnapshot, OverlayState};
    use serde_json::json;

    #[test]
    fn native_fresh_render_generations_are_legacy_safe_and_round_trip() {
        let legacy: OverlayPresentationSnapshot = serde_json::from_value(json!({})).unwrap();
        assert!(legacy.native_fresh_render_generations.is_none());

        let snapshot: OverlayPresentationSnapshot = serde_json::from_value(json!({
            "native_fresh_render_generations": {"self": 2, "peer": 7}
        }))
        .unwrap();
        assert_eq!(
            snapshot.native_fresh_render_generations,
            Some(NativeFreshRenderGenerations {
                self_generation: Some(2),
                peer: Some(7),
                self_target: None,
                peer_target: None,
                quiet_tail_episodes: None,
            })
        );
        assert_eq!(
            serde_json::to_value(snapshot).unwrap()["native_fresh_render_generations"],
            json!({"self": 2, "peer": 7})
        );
    }

    #[test]
    fn native_fresh_render_targets_are_optional_and_bound_to_generations() {
        let snapshot: OverlayPresentationSnapshot = serde_json::from_value(json!({
            "native_fresh_render_generations": {"self": 3, "peer": 4},
            "native_fresh_render_targets": {"self": "self:synthetic-a", "peer": "peer:synthetic-b"}
        }))
        .unwrap();
        let generations = snapshot.native_fresh_render_generations.as_ref().unwrap();
        assert_eq!(generations.self_target.as_deref(), Some("self:synthetic-a"));
        assert_eq!(generations.peer_target.as_deref(), Some("peer:synthetic-b"));
        let value = serde_json::to_value(snapshot).unwrap();
        assert_eq!(
            value["native_fresh_render_targets"],
            json!({"self": "self:synthetic-a", "peer": "peer:synthetic-b"})
        );
    }

    #[test]
    fn native_fresh_render_generations_reject_negative_values_and_are_non_visual() {
        assert!(
            serde_json::from_value::<OverlayPresentationSnapshot>(json!({
                "native_fresh_render_generations": {"self": -1}
            }))
            .is_err()
        );
        assert!(serde_json::from_str::<OverlayPresentationSnapshot>(
            r#"{"native_fresh_render_generations":{"peer":18446744073709551616}}"#
        )
        .is_err());
        let mut state = OverlayState::default();
        let first: OverlayPresentationSnapshot = serde_json::from_value(json!({
            "revision": 1,
            "native_fresh_render_generations": {"self": 1}
        }))
        .unwrap();
        let second: OverlayPresentationSnapshot = serde_json::from_value(json!({
            "revision": 2,
            "native_fresh_render_generations": {"self": 2, "peer": 1}
        }))
        .unwrap();
        assert!(!state.apply_snapshot(&first));
        assert!(!state.apply_snapshot(&second));
        assert_eq!(
            state.native_fresh_render_generations(),
            second.native_fresh_render_generations.as_ref()
        );
    }
}
