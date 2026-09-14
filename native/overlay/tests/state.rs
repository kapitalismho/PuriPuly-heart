use serde_json::json;

use puripuly_heart_overlay::{
    OverlayPresentationBlock, OverlayPresentationBlockVariant, OverlayPresentationCalibration,
    OverlayPresentationSnapshot, OverlayState,
};

fn block(
    id: &str,
    channel: &str,
    primary_text: &str,
    secondary_text: &str,
    secondary_enabled: bool,
) -> OverlayPresentationBlock {
    OverlayPresentationBlock {
        id: id.to_string(),
        occupant_key: id.to_string(),
        appearance_seq: 1,
        channel: channel.to_string(),
        block_variant: OverlayPresentationBlockVariant::Finalized,
        primary_text: primary_text.to_string(),
        secondary_text: secondary_text.to_string(),
        secondary_enabled,
        primary_language: None,
        secondary_language: None,
        update_id: None,
        origin_wall_clock_ms: None,
        session_scope: None,
        ..Default::default()
    }
}

fn slot_block(
    id: &str,
    occupant_key: &str,
    appearance_seq: u64,
    channel: &str,
    primary_text: &str,
    secondary_text: &str,
    secondary_enabled: bool,
) -> OverlayPresentationBlock {
    OverlayPresentationBlock {
        id: id.to_string(),
        occupant_key: occupant_key.to_string(),
        appearance_seq,
        channel: channel.to_string(),
        block_variant: OverlayPresentationBlockVariant::Finalized,
        primary_text: primary_text.to_string(),
        secondary_text: secondary_text.to_string(),
        secondary_enabled,
        primary_language: None,
        secondary_language: None,
        update_id: None,
        origin_wall_clock_ms: None,
        session_scope: None,
        ..Default::default()
    }
}

#[test]
fn overlay_state_preserves_snapshot_slot_correlation_observability_metadata() {
    let snapshot: OverlayPresentationSnapshot = serde_json::from_value(json!({
        "revision": 7,
        "calibration": OverlayPresentationCalibration::default(),
        "blocks": [
            {
                "id": "self:1",
                "occupant_key": "self:1",
                "appearance_seq": 1,
                "channel": "self",
                "block_variant": "finalized",
                "primary_text": "hello",
                "secondary_text": "",
                "secondary_enabled": true,
                "update_id": "upd-self-1",
                "origin_wall_clock_ms": 1712345678901u64,
                "session_scope": "session:self"
            }
        ]
    }))
    .unwrap();
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&snapshot));

    let persisted = serde_json::to_value(state.snapshot()).unwrap();
    assert_eq!(persisted["blocks"][0]["update_id"], "upd-self-1");
    assert_eq!(
        persisted["blocks"][0]["origin_wall_clock_ms"],
        1712345678901u64
    );
    assert_eq!(persisted["blocks"][0]["session_scope"], "session:self");
}

#[test]
fn overlay_state_deserializes_active_peer_variant() {
    let snapshot: OverlayPresentationSnapshot = serde_json::from_value(json!({
        "revision": 1,
        "calibration": OverlayPresentationCalibration::default(),
        "blocks": [{
            "id": "peer:turn-1",
            "occupant_key": "peer:turn-1",
            "appearance_seq": 1,
            "channel": "peer",
            "block_variant": "active_peer",
            "primary_text": "",
            "secondary_text": "Can you hear me?",
            "secondary_enabled": true
        }]
    }))
    .unwrap();

    assert_eq!(
        snapshot.blocks[0].block_variant,
        OverlayPresentationBlockVariant::ActivePeer
    );
    assert_eq!(snapshot.blocks[0].primary_text, "");
    assert_eq!(snapshot.blocks[0].secondary_text, "Can you hear me?");
    assert!(snapshot.blocks[0].secondary_enabled);

    let legacy: OverlayPresentationSnapshot = serde_json::from_value(json!({
        "revision": 1,
        "calibration": OverlayPresentationCalibration::default(),
        "blocks": [{
            "id": "self:legacy",
            "occupant_key": "self:legacy",
            "appearance_seq": 1,
            "channel": "self",
            "block_variant": "finalized",
            "primary_text": "hello",
            "secondary_text": "",
            "secondary_enabled": true
        }]
    }))
    .unwrap();

    assert_eq!(legacy.blocks[0].primary_language, None);
    assert_eq!(legacy.blocks[0].secondary_language, None);
}

#[test]
fn overlay_state_treats_language_only_update_as_visual_change_without_reassigning_slot() {
    let initial: OverlayPresentationSnapshot = serde_json::from_value(json!({
        "revision": 1,
        "calibration": OverlayPresentationCalibration::default(),
        "blocks": [{
            "id": "self:language",
            "occupant_key": "self:language",
            "appearance_seq": 1,
            "channel": "self",
            "block_variant": "finalized",
            "primary_text": "こんにちは",
            "secondary_text": "",
            "secondary_enabled": true,
            "primary_language": "ko"
        }]
    }))
    .unwrap();
    let updated: OverlayPresentationSnapshot = serde_json::from_value(json!({
        "revision": 2,
        "calibration": OverlayPresentationCalibration::default(),
        "blocks": [{
            "id": "self:language",
            "occupant_key": "self:language",
            "appearance_seq": 1,
            "channel": "self",
            "block_variant": "finalized",
            "primary_text": "こんにちは",
            "secondary_text": "",
            "secondary_enabled": true,
            "primary_language": "ja"
        }]
    }))
    .unwrap();
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&initial));
    let slot = state.scene().slots()[0].as_ref().unwrap();
    let original_slot_index = slot.slot_index;
    let original_anchor_top_px = slot.anchor_top_px;
    let original_slot_entry_order = slot.slot_entry_order;

    assert!(state.apply_snapshot(&updated));

    let slot = state.scene().slots()[0].as_ref().unwrap();
    assert_eq!(slot.slot_index, original_slot_index);
    assert_eq!(slot.anchor_top_px, original_anchor_top_px);
    assert_eq!(slot.slot_entry_order, original_slot_entry_order);
    assert_eq!(slot.occupant_key, "self:language");
    assert_eq!(slot.appearance_seq, 1);
    assert_eq!(slot.primary_language.as_deref(), Some("ja"));
}

#[test]
fn overlay_state_keeps_snapshot_blocks_in_order() {
    let mut state = OverlayState::default();

    state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "hello", "안녕", true),
            slot_block("peer:2", "peer:2", 2, "peer", "there", "원문", true),
        ],
        ..Default::default()
    });

    assert_eq!(state.blocks().len(), 2);
    assert_eq!(state.blocks()[0].id, "self:1");
    assert_eq!(state.blocks()[1].id, "peer:2");
}

#[test]
fn overlay_state_snapshot_replaces_stale_blocks() {
    let mut state = OverlayState::default();

    state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![block("self:1", "self", "hello", "", true)],
        ..Default::default()
    });

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![block("peer:2", "peer", "there", "원문", false)],
    }));

    assert_eq!(state.blocks().len(), 1);
    assert_eq!(state.blocks()[0].id, "peer:2");
}

#[test]
fn overlay_state_tracks_latest_snapshot_calibration() {
    let mut state = OverlayState::default();

    assert_eq!(state.calibration().distance, 1.1);

    state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 3,
        calibration: OverlayPresentationCalibration {
            anchor: "head_locked".to_string(),
            offset_x: 0.15,
            offset_y: -0.2,
            distance: 1.2,
            text_scale: 1.1,
            background_alpha: 0.4,
        },
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![],
        ..Default::default()
    });

    assert_eq!(state.calibration().distance, 1.2);
    assert_eq!(state.calibration().background_alpha, 0.4);
}

#[test]
fn overlay_state_ignores_lower_revision_snapshots() {
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 3,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![block("self:3", "self", "latest", "", true)],
    }));

    assert!(!state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![block("peer:2", "peer", "stale", "", true)],
    }));

    assert_eq!(state.snapshot().revision, 3);
    assert_eq!(state.blocks()[0].id, "self:3");
}

#[test]
fn overlay_state_treats_equal_revision_snapshots_as_noop() {
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 4,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![block("self:4", "self", "keep", "", true)],
    }));

    assert!(!state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 4,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![block("peer:4", "peer", "ignore", "", true)],
    }));
}

#[test]
fn overlay_state_keeps_slot_two_anchor_when_slot_one_disappears() {
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "one", "", true),
            slot_block("peer:2", "peer:2", 2, "peer", "two", "", true),
        ],
    }));

    let second_top = state.scene().slots()[1].as_ref().unwrap().anchor_top_px;

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![slot_block("peer:2", "peer:2", 2, "peer", "two", "", true)],
    }));

    assert!(state.scene().slots()[0].is_none());
    assert_eq!(
        state.scene().slots()[1].as_ref().unwrap().anchor_top_px,
        second_top
    );
}

#[test]
fn overlay_state_promotes_matching_occupant_key_without_reassigning_slot() {
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![OverlayPresentationBlock {
            id: "self:active".into(),
            occupant_key: "self:merge-1".into(),
            appearance_seq: 1,
            channel: "self".into(),
            block_variant: OverlayPresentationBlockVariant::ActiveSelf,
            primary_text: "hello live".into(),
            secondary_text: String::new(),
            secondary_enabled: true,
            primary_language: None,
            secondary_language: None,
            update_id: None,
            origin_wall_clock_ms: None,
            session_scope: None,
            ..Default::default()
        }],
    }));
    let original_slot = state.scene().slots()[0].as_ref().unwrap().slot_index;
    let original_anchor_top = state.scene().slots()[0].as_ref().unwrap().anchor_top_px;
    let original_entry_order = state.scene().slots()[0].as_ref().unwrap().slot_entry_order;

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![slot_block(
            "self:merge-1",
            "self:merge-1",
            1,
            "self",
            "hello live",
            "",
            true,
        )],
    }));

    let slot = state.scene().slots()[0].as_ref().unwrap();
    assert_eq!(slot.slot_index, original_slot);
    assert_eq!(slot.anchor_top_px, original_anchor_top);
    assert_eq!(slot.slot_entry_order, original_entry_order);
    assert_eq!(slot.occupant_key, "self:merge-1");
    assert_eq!(slot.id, "self:merge-1");
}

#[test]
fn overlay_state_promotes_active_peer_matching_occupant_key_without_reassigning_slot() {
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![OverlayPresentationBlock {
            id: "peer:active".into(),
            occupant_key: "peer:turn-1".into(),
            appearance_seq: 1,
            channel: "peer".into(),
            block_variant: OverlayPresentationBlockVariant::ActivePeer,
            primary_text: "Can you hear me?".into(),
            secondary_text: String::new(),
            secondary_enabled: false,
            primary_language: None,
            secondary_language: None,
            update_id: None,
            origin_wall_clock_ms: None,
            session_scope: None,
            ..Default::default()
        }],
    }));
    let original_slot = state.scene().slots()[0].as_ref().unwrap().slot_index;
    let original_entry_order = state.scene().slots()[0].as_ref().unwrap().slot_entry_order;

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![OverlayPresentationBlock {
            id: "peer:turn-1".into(),
            occupant_key: "peer:turn-1".into(),
            appearance_seq: 1,
            channel: "peer".into(),
            block_variant: OverlayPresentationBlockVariant::Finalized,
            primary_text: "들려?".into(),
            secondary_text: "Can you hear me?".into(),
            secondary_enabled: true,
            primary_language: None,
            secondary_language: None,
            update_id: None,
            origin_wall_clock_ms: None,
            session_scope: None,
            ..Default::default()
        }],
    }));

    let slot = state.scene().slots()[0].as_ref().unwrap();
    assert_eq!(slot.slot_index, original_slot);
    assert_eq!(slot.slot_entry_order, original_entry_order);
    assert_eq!(slot.occupant_key, "peer:turn-1");
    assert_eq!(slot.id, "peer:turn-1");
    assert_eq!(
        slot.block_variant,
        OverlayPresentationBlockVariant::Finalized
    );
}

#[test]
fn overlay_state_fills_first_empty_slot_before_replacing_again() {
    let mut state = OverlayState::default();

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 1,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:1", "self:1", 1, "self", "one", "", true),
            slot_block("peer:2", "peer:2", 2, "peer", "two", "", true),
        ],
    }));

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        revision: 2,
        calibration: OverlayPresentationCalibration::default(),
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        blocks: vec![slot_block("peer:2", "peer:2", 2, "peer", "two", "", true)],
    }));

    assert!(state.apply_snapshot(&OverlayPresentationSnapshot {
        native_fresh_render_generations: None,
        semantic_retirement_frontiers: Vec::new(),
        revision: 3,
        calibration: OverlayPresentationCalibration::default(),
        blocks: vec![
            slot_block("self:3", "self:3", 3, "self", "three", "", true),
            slot_block("peer:2", "peer:2", 2, "peer", "two", "", true),
        ],
    }));

    assert_eq!(
        state
            .scene()
            .slots()
            .iter()
            .map(|slot| slot.as_ref().map(|slot| slot.id.clone()))
            .collect::<Vec<_>>(),
        vec![Some("self:3".to_string()), Some("peer:2".to_string())]
    );
}
