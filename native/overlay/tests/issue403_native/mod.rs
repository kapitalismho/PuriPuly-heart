use super::*;

fn python_cases(mode: &str) -> Value {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let output = Command::new("uv")
        .args([
            "run",
            "--locked",
            "python",
            "-m",
            "tests.fixtures.issue403_native",
            mode,
        ])
        .env("PYTHONIOENCODING", "utf-8")
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn snapshot(row: &Value) -> OverlayPresentationSnapshot {
    serde_json::from_value(row["snapshot"].clone()).unwrap()
}

fn peer_generation(row: &Value) -> u64 {
    row["snapshot"]["native_fresh_render_generations"]["peer"]
        .as_u64()
        .unwrap()
}

fn peer_episode(row: &Value) -> u64 {
    row["snapshot"]["native_quiet_tail_episodes"]["peer"]["generation"]
        .as_u64()
        .unwrap()
}

fn compact(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

#[derive(Default)]
struct SceneFrames {
    frames: Vec<Vec<(String, String)>>,
    visible: bool,
}

struct SceneSubmitter(Arc<Mutex<SceneFrames>>);

impl OverlayFrameSubmitter for SceneSubmitter {
    fn submit_frame(&mut self, frame: &RenderedFrame) -> Result<(), OpenVrError> {
        let blocks = frame
            .layout()
            .visible_blocks
            .iter()
            .map(|block| {
                assert!(!block.truncated_primary, "{}", block.id);
                (
                    block.id.clone(),
                    compact(
                        &block
                            .primary_lines
                            .iter()
                            .map(|line| line.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                )
            })
            .collect();
        self.0.lock().unwrap().frames.push(blocks);
        Ok(())
    }

    fn reanchor_spatial_locked(&mut self) -> Result<SpatialReanchorOutcome, OpenVrError> {
        Ok(SpatialReanchorOutcome::Applied)
    }

    fn set_overlay_visible(&mut self, visible: bool) -> Result<(), OpenVrError> {
        self.0.lock().unwrap().visible = visible;
        Ok(())
    }

    fn observed_overlay_visible(&self) -> Option<bool> {
        Some(self.0.lock().unwrap().visible)
    }
}

#[derive(Clone, Default)]
struct RecordingSink {
    buffer: Arc<Mutex<Vec<u8>>>,
}

impl RecordingSink {
    fn events(&self) -> Vec<Value> {
        String::from_utf8(self.buffer.lock().unwrap().clone())
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("EVENT "))
            .map(|event| serde_json::from_str(event).unwrap())
            .collect()
    }
}

impl std::io::Write for RecordingSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.buffer.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn run_rows(rows: &[Value], label: &str) -> NativePresentationOwner<SceneSubmitter> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let scenes = rows.to_vec();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();
        let _auth = ws.next().await.unwrap().unwrap();
        let started = tokio::time::Instant::now();
        for (index, row) in scenes.iter().enumerate() {
            tokio::time::sleep_until(
                started + Duration::from_millis(row["at_ms"].as_u64().unwrap()),
            )
            .await;
            ws.send(Message::Text(
                json!({"type":"snapshot","payload":row["snapshot"]})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
            if index == 0 {
                wait_for_owner_ready(&mut ws).await;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        ws.send(Message::Text(
            json!({"type":"snapshot","payload":scenes[0]["snapshot"]})
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(550)).await;
        ws.send(Message::Text(json!({"type":"shutdown"}).to_string().into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    });
    let mut manifest = test_manifest();
    manifest.bridge_url = format!("ws://{address}");
    let (mut bridge, initial) = BridgeClient::connect(&manifest).await.unwrap();
    let captured = Arc::new(Mutex::new(SceneFrames::default()));
    let mut owner = NativePresentationOwner::new(
        initial,
        CaptionRenderer::new_for_test().unwrap(),
        SceneSubmitter(captured.clone()),
    );
    let stdout = RecordingSink::default();
    let stderr = RecordingSink::default();
    let logger = OverlayLogger::from_streams(Box::new(stdout.clone()), Box::new(stderr.clone()));
    owner.run(&mut bridge, &logger).await.unwrap();
    server.await.unwrap();
    assert!(owner.resources_released());
    assert!(owner.fresh_retry_schedules_for_test().is_empty());
    assert_eq!(
        owner.runtime().state().snapshot().blocks,
        snapshot(rows.last().unwrap()).blocks
    );
    let attempts = owner.successful_attempt_audit_for_test();
    let captured = captured.lock().unwrap();
    assert!(captured.frames.len() >= attempts.len());
    for (attempt, actual) in attempts.iter().zip(&captured.frames) {
        let row = rows
            .iter()
            .find(|row| row["snapshot"]["revision"] == attempt.scene_generation)
            .unwrap();
        let expected = snapshot(row)
            .blocks
            .iter()
            .map(|block| (block.id.clone(), compact(&block.primary_text)))
            .collect::<Vec<_>>();
        assert_eq!(
            *actual, expected,
            "whole current scene {}",
            attempt.scene_generation
        );
    }
    let final_expected = snapshot(rows.last().unwrap())
        .blocks
        .iter()
        .map(|block| (block.id.clone(), compact(&block.primary_text)))
        .collect::<Vec<_>>();
    assert_eq!(
        captured
            .frames
            .iter()
            .rev()
            .find(|blocks| !blocks.is_empty())
            .unwrap(),
        &final_expected
    );
    let audit = owner.fresh_retry_audit_for_test();
    let evidence = owner.fresh_retry_evidence_for_test();
    for record in evidence
        .iter()
        .filter(|record| record["outcome"] == "completed")
    {
        assert_eq!(record["fresh_rendered"], true);
        assert_eq!(record["producer_ready"], true);
        assert_eq!(record["submission_succeeded"], true);
    }
    assert_eq!(owner.fresh_retry_audit_dropped_for_test(), 0);
    println!(
        "ISSUE403 {label} rows={}",
        serde_json::to_string(rows).unwrap()
    );
    println!(
        "ISSUE403 {label} audit={audit:?} execution={evidence:?} whole_scene_submissions={}",
        attempts.len()
    );
    assert_eq!(logger.reliable_delivery_failures(), 0);
    logger.shutdown().unwrap();
    let events = stdout.events();
    assert_eq!(
        events
            .iter()
            .map(|event| &event["type"])
            .collect::<Vec<_>>(),
        vec!["overlay_ready", "shutdown_complete"]
    );
    for event in &events {
        assert_eq!(event["overlay_instance_id"], manifest.overlay_instance_id);
        assert_eq!(event["runtime_generation"], 1);
    }
    assert_eq!(events[1]["stop_origin"], "parent");
    assert_eq!(events[1]["stop_reason"], "app_requested");
    assert_eq!(events[1]["cleanup_succeeded"], true);
    assert!(stderr.events().is_empty());
    owner
}

fn assert_bounded_episode(
    owner: &NativePresentationOwner<SceneSubmitter>,
    row: &Value,
    maximum: usize,
) {
    let generation = peer_generation(row);
    let audit = owner.fresh_retry_audit_for_test();
    let scheduled = audit
        .iter()
        .filter(|fact| fact.0 == "peer" && fact.1 == generation && fact.2 == "scheduled")
        .collect::<Vec<_>>();
    assert_eq!(scheduled.len(), 1, "{audit:?}");
    let completed = audit
        .iter()
        .filter(|fact| fact.0 == "peer" && fact.1 == generation && fact.2 == "completed")
        .collect::<Vec<_>>();
    assert!(!completed.is_empty(), "{audit:?}");
    assert!(completed.len() <= maximum, "{audit:?}");
    assert!(completed
        .iter()
        .all(|fact| fact.4 - scheduled[0].4 <= Duration::from_millis(500)));
    assert_eq!(NATIVE_FRESH_RETRY_CADENCE, Duration::from_millis(100));
    assert_eq!(NATIVE_FRESH_RETRY_DEADLINE, Duration::from_millis(500));
    assert_eq!(NATIVE_FRESH_RETRY_MAX_COMPLETED, 5);
    assert!(completed
        .windows(2)
        .all(|pair| pair[1].4 - pair[0].4 >= Duration::from_millis(100)));
    assert!(owner.fresh_retry_evidence_for_test().iter().any(|record| {
        record["channel"] == "peer"
            && record["trigger_generation"] == generation
            && record["episode_generation"] == peer_episode(row)
            && record["outcome"] == "completed"
    }));
}

fn assert_current_targets(rows: &[Value]) {
    for row in rows {
        if let Some(target) = row["snapshot"]["native_fresh_render_targets"]["peer"].as_str() {
            assert!(
                row["snapshot"]["blocks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|block| block["channel"] == "peer"
                        && block["id"] == target
                        && !block["primary_text"].as_str().unwrap().trim().is_empty()),
                "{}",
                row["label"]
            );
        }
    }
}

#[tokio::test]
async fn issue403_python_l2_repairs_execute_new_bounded_episodes_without_self() {
    let input = python_cases("l2");
    let rows = input.as_array().unwrap();
    assert_current_targets(rows);
    assert_eq!(rows[1]["at_ms"], 4635);
    assert_eq!(rows[2]["at_ms"], 9875);
    assert!(peer_episode(&rows[0]) < peer_episode(&rows[1]));
    assert!(peer_episode(&rows[1]) < peer_episode(&rows[2]));
    for index in [3, 4] {
        assert_eq!(peer_generation(&rows[index]), peer_generation(&rows[2]));
        assert_eq!(peer_episode(&rows[index]), peer_episode(&rows[2]));
        assert_eq!(rows[index]["expirations"], rows[2]["expirations"]);
    }
    assert_eq!(rows[6]["snapshot"], rows[7]["snapshot"]);
    let owner = run_rows(rows, "issue403-l2").await;
    for row in &rows[..3] {
        assert_bounded_episode(&owner, row, 5);
    }
    let audit = owner.fresh_retry_audit_for_test();
    for index in 0..2 {
        let original_ended = audit.iter().any(|fact| {
            fact.0 == "peer"
                && fact.1 == peer_generation(&rows[index])
                && (fact.2 == "expired" || fact.3 == 5)
                && fact.4 < Duration::from_millis(rows[index + 1]["at_ms"].as_u64().unwrap())
        });
        assert!(original_ended, "{audit:?}");
    }
    assert!(audit
        .iter()
        .filter(|fact| fact.0 == "peer"
            && fact.1 == peer_generation(&rows[2])
            && fact.2 == "completed")
        .all(|fact| fact.4 < Duration::from_millis(10700)));
}

#[tokio::test]
async fn issue403_python_preview_revisions_and_first_final_preserve_episodes() {
    let input = python_cases("previews");
    let rows = input.as_array().unwrap();
    assert_current_targets(rows);
    assert!(peer_episode(&rows[0]) < peer_episode(&rows[1]));
    assert!(peer_episode(&rows[1]) < peer_episode(&rows[2]));
    assert_eq!(
        rows[0]["snapshot"]["native_quiet_tail_episodes"]["peer"]["phase"],
        "stream"
    );
    assert_eq!(
        rows[2]["snapshot"]["native_quiet_tail_episodes"]["peer"]["phase"],
        "final"
    );
    assert_eq!(rows[2]["snapshot"], rows[3]["snapshot"]);
    assert_eq!(rows[4]["snapshot"], rows[5]["snapshot"]);
    let owner = run_rows(rows, "issue403-preview").await;
    for (row, maximum) in rows[..3].iter().zip([4, 4, 5]) {
        assert_bounded_episode(&owner, row, maximum);
    }
}

#[tokio::test]
async fn issue403_python_fragment_targets_and_retirement_keep_current_scene() {
    let input = python_cases("fragments");
    let rows = input.as_array().unwrap();
    assert_current_targets(rows);
    assert_eq!(rows[6]["snapshot"], rows[7]["snapshot"]);
    let owner = run_rows(rows, "issue403-fragments").await;
    let audit = owner.fresh_retry_audit_for_test();
    for row in rows.iter().filter(|row| {
        ["two-fragments", "corrected", "growth", "shrink"].contains(&row["label"].as_str().unwrap())
    }) {
        assert!(
            audit.iter().any(|fact| fact.0 == "peer"
                && fact.1 == peer_generation(row)
                && fact.2 == "scheduled"),
            "{audit:?}"
        );
    }
    let removed = &rows[5];
    assert!(removed["snapshot"]["native_fresh_render_targets"]["peer"].is_null());
    let prior = peer_generation(&rows[5]);
    assert!(audit.iter().any(|fact| fact.0 == "peer"
        && fact.1 == prior
        && (fact.2 == "cancelled" || fact.2 == "replaced")));
    assert!(audit
        .iter()
        .filter(|fact| fact.0 == "peer" && fact.1 == prior && fact.2 == "completed")
        .all(|fact| fact.4 < Duration::from_millis(removed["at_ms"].as_u64().unwrap())));
}

#[tokio::test]
async fn issue403_local_revision_zero_fragments_start_only_when_drawable() {
    let input = python_cases("local");
    let rows = input.as_array().unwrap();
    assert_current_targets(rows);
    let first = &rows[1];
    let queued = &rows[3];
    let continuation = &rows[4];
    assert_eq!(first["caption_revision"], 0);
    assert_ne!(first["capture_parent_id"], first["translation_child_id"]);
    let child_block = format!("peer:{}", first["translation_child_id"].as_str().unwrap());
    assert_eq!(
        first["snapshot"]["native_fresh_render_targets"]["peer"],
        child_block
    );
    let pending = queued["fragments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|fragment| fragment["index"] == 1)
        .unwrap();
    assert_eq!(pending["pending"], true);
    assert_eq!(pending["registered_at"], 0.0);
    assert!(pending["visible_since"].is_null());
    assert!(pending["last_meaningful_visible_at"].is_null());
    assert!(pending["protected_until"].is_null());
    assert!(pending["expires_at"].is_null());
    let tail = pending["id"].as_str().unwrap();
    assert!(!queued["expirations"]
        .as_object()
        .unwrap()
        .contains_key(tail));
    assert_eq!(peer_generation(first), peer_generation(&rows[2]));
    assert_eq!(peer_generation(first), peer_generation(queued));
    assert_eq!(peer_episode(first), peer_episode(queued));
    assert!(peer_generation(continuation) > peer_generation(queued));
    assert!(peer_episode(continuation) > peer_episode(queued));
    assert_eq!(
        continuation["snapshot"]["native_fresh_render_targets"]["peer"],
        tail
    );
    assert_eq!(continuation["expirations"][tail], 9.0);
    let shown_tail = continuation["fragments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|fragment| fragment["index"] == 1)
        .unwrap();
    assert_eq!(shown_tail["visible_since"], 1.0);
    assert_eq!(shown_tail["pending"], false);
    assert_eq!(continuation["snapshot"], rows[5]["snapshot"]);
    let owner = run_rows(rows, "issue403-local").await;
    assert_bounded_episode(&owner, first, 5);
    assert_bounded_episode(&owner, continuation, 5);
    assert!(owner
        .fresh_retry_evidence_for_test()
        .iter()
        .any(|record| record["channel"] == "peer"
            && record["trigger_generation"] == peer_generation(continuation)
            && record["outcome"] == "scheduled"
            && record["scene_generation"] == continuation["snapshot"]["revision"]));
}

#[tokio::test]
async fn issue403_native_accepts_later_ready_older_source_after_expiry_but_not_retired_parent() {
    let input = python_cases("ready-order");
    let rows = input.as_array().unwrap();
    assert_current_targets(rows);
    assert_eq!(rows[0]["source_order"], 2);
    assert_eq!(rows[3]["source_order"], 1);
    assert!(rows[2]["snapshot"]["blocks"].as_array().unwrap().is_empty());
    let b = &rows[0]["snapshot"]["blocks"][0];
    let a = &rows[3]["snapshot"]["blocks"][0];
    assert_eq!(a["primary_text"], "Ordinary local A");
    assert!(a["publication_order"].as_u64().unwrap() > b["publication_order"].as_u64().unwrap());
    assert_eq!(rows[0]["accepted_display_order"], b["publication_order"]);
    assert_eq!(rows[3]["accepted_display_order"], a["publication_order"]);
    assert_eq!(rows[5]["historical_receipt"], "applied");
    assert_eq!(
        rows[5]["historical_receipt_scene_revision"],
        rows[0]["snapshot"]["revision"]
    );
    assert_eq!(rows[4]["snapshot"], rows[5]["snapshot"]);
    assert_eq!(rows[6]["retired_receipt"], "stale");
    assert_eq!(rows[4]["snapshot"], rows[6]["snapshot"]);
    let owner = run_rows(rows, "issue403-ready-order").await;
    assert_bounded_episode(&owner, &rows[0], 5);
    assert_bounded_episode(&owner, &rows[3], 5);
}

#[tokio::test]
async fn issue403_output_generation_retirement_fences_pending_tail_and_allows_recovery() {
    let input = python_cases("generation-retirement");
    let rows = input.as_array().unwrap();
    assert_current_targets(rows);
    let original = &rows[0];
    let root = "peer:00000000-0000-0000-0000-000000000046";
    let tail = original["fragments"]
        .as_array()
        .unwrap()
        .iter()
        .find(|fragment| fragment["index"] == 1)
        .unwrap();
    let tail_id = tail["id"].as_str().unwrap();
    assert_eq!(original["publication_generation"], 1);
    assert_eq!(
        original["snapshot"]["native_fresh_render_targets"]["peer"],
        root
    );
    assert_eq!(tail["pending"], true);
    assert!(tail["visible_since"].is_null());
    assert!(tail["expires_at"].is_null());
    assert_eq!(original["output_has_active_deliveries"], false);
    assert!(original["output_decisions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|reason| reason == "application_applied"));
    for row in &rows[1..4] {
        assert!(row["snapshot"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["id"] != tail_id));
        let episode = row["snapshot"]["native_quiet_tail_episodes"]["peer"]["generation"].as_u64();
        assert!(episode.is_none() || episode == Some(peer_episode(original)));
        let generation = row["snapshot"]["native_fresh_render_generations"]["peer"].as_u64();
        assert!(generation.is_none() || generation == Some(peer_generation(original)));
    }
    for scene in rows[3]["published_while_retired"].as_array().unwrap() {
        assert!(scene["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["id"] != tail_id));
        assert_ne!(scene["native_fresh_render_targets"]["peer"], tail_id);
        let episode = scene["native_quiet_tail_episodes"]["peer"]["generation"].as_u64();
        assert!(episode.is_none() || episode == Some(peer_episode(original)));
        let generation = scene["native_fresh_render_generations"]["peer"].as_u64();
        assert!(generation.is_none() || generation == Some(peer_generation(original)));
    }
    let recovered = &rows[4];
    let current = "peer:00000000-0000-0000-0000-000000000047";
    assert_eq!(recovered["publication_generation"], 2);
    assert_eq!(
        recovered["snapshot"]["native_fresh_render_targets"]["peer"],
        current
    );
    assert!(recovered["snapshot"]["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|block| block["channel"] == "self"));
    assert!(recovered["snapshot"]["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["id"] != root && block["id"] != tail_id));
    assert!(peer_generation(recovered) > peer_generation(original));
    assert!(peer_episode(recovered) > peer_episode(original));
    assert_eq!(rows[5]["snapshot"], recovered["snapshot"]);
    assert_eq!(rows[5]["expirations"], recovered["expirations"]);
    assert!(rows[5]["output_decisions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|reason| reason == "publication_generation_retired"));
    let owner = run_rows(rows, "issue403-generation-retirement").await;
    assert_bounded_episode(&owner, original, 5);
    assert_bounded_episode(&owner, recovered, 5);
    assert!(owner
        .fresh_retry_audit_for_test()
        .iter()
        .filter(|fact| fact.0 == "peer" && fact.2 == "scheduled")
        .all(|fact| fact.1 == peer_generation(original) || fact.1 == peer_generation(recovered)));
}

#[tokio::test]
async fn issue403_translation_toggle_repartitions_current_source_without_tail_revival() {
    let input = python_cases("translation-toggle");
    let rows = input.as_array().unwrap();
    assert_eq!(rows.len(), 4);
    assert_current_targets(rows);
    let original = &rows[0];
    let translated = original["snapshot"]["blocks"].as_array().unwrap();
    assert_eq!(translated.len(), 2);
    assert!(translated.iter().all(|block| block["channel"] == "peer"));
    assert_eq!(
        translated
            .iter()
            .map(|block| compact(block["primary_text"].as_str().unwrap()))
            .collect::<String>(),
        "가".repeat(64)
    );
    assert_eq!(original["source_input"], "One source sentence.");
    let root_id = translated[0]["id"].as_str().unwrap();
    let tail_id = translated[1]["id"].as_str().unwrap();
    let source_only = &rows[1];
    let source_blocks = source_only["snapshot"]["blocks"].as_array().unwrap();
    assert_eq!(source_blocks.len(), 1);
    assert_eq!(source_blocks[0]["id"], root_id);
    assert_eq!(
        compact(source_blocks[0]["primary_text"].as_str().unwrap()),
        "Onesourcesentence."
    );
    assert_eq!(
        source_only["snapshot"]["native_fresh_render_targets"]["peer"],
        root_id
    );
    assert!(peer_generation(source_only) > peer_generation(original));
    assert!(peer_episode(source_only) > peer_episode(original));
    for row in &rows[1..] {
        assert!(row["snapshot"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block["id"] != tail_id));
        assert_eq!(row["snapshot"], source_only["snapshot"]);
        assert_eq!(row["expirations"], source_only["expirations"]);
        if let Some(tail) = row["fragments"]
            .as_array()
            .unwrap()
            .iter()
            .find(|fragment| fragment["id"] == tail_id)
        {
            assert_eq!(tail["dormant"], true);
            assert_eq!(tail["pending"], false);
        }
    }
    let owner = run_rows(rows, "issue403-translation-toggle").await;
    assert_bounded_episode(&owner, original, 5);
    assert_bounded_episode(&owner, source_only, 5);
    assert!(owner
        .fresh_retry_audit_for_test()
        .iter()
        .filter(|fact| fact.0 == "peer" && fact.2 == "scheduled")
        .all(|fact| fact.1 == peer_generation(original) || fact.1 == peer_generation(source_only)));
}

#[cfg(windows)]
mod rendered;
