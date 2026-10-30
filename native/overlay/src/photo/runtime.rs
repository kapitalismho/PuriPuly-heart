use serde_json::json;
use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use super::bridge::{PhotoBridge, PhotoIncoming};
use super::image::{
    PhotoImageBoundary, PhotoImageOutput, PhotoImageResult, PhotoImageWorker, PhotoTexture,
};
use super::manifest::{PhotoBackend, PhotoError, PhotoManifest, PHOTO_PROTOCOL_VERSION};
use super::prepare::{PreparationWorker, PrepareCompletion};
use super::state::{PhotoState, SnapshotAdmission};
use crate::openvr::{
    OpenVrEventClass, OpenVrOverlay, OpenVrSession, OverlayFrameSubmitter, OverlayTextureSubmitter,
};

pub fn surface_identity(manifest: &PhotoManifest, surface: &str) -> String {
    format!(
        "photo.{}.g{}.p{}.{}",
        manifest.instance_id,
        manifest.generation,
        std::process::id(),
        surface
    )
}

pub struct PhotoOpenVrSurfaces {
    icon: Option<OpenVrOverlay>,
    photo: Option<OpenVrOverlay>,
    _session: OpenVrSession,
}

impl PhotoOpenVrSurfaces {
    pub fn new(manifest: &PhotoManifest) -> Result<Self, PhotoError> {
        let session = OpenVrSession::new().map_err(|_| PhotoError::OpenVr)?;
        let icon = OpenVrOverlay::in_session(&session, &surface_identity(manifest, "icon"))
            .map_err(|_| PhotoError::OpenVr)?;
        let photo = OpenVrOverlay::in_session(&session, &surface_identity(manifest, "photo"))
            .map_err(|_| PhotoError::OpenVr)?;
        Ok(Self {
            icon: Some(icon),
            photo: Some(photo),
            _session: session,
        })
    }

    pub fn close_icon(&mut self) {
        self.icon.take();
    }
    pub fn close_photo(&mut self) {
        self.photo.take();
    }
    pub fn icon(&mut self) -> Option<&mut OpenVrOverlay> {
        self.icon.as_mut()
    }
    pub fn photo(&mut self) -> Option<&mut OpenVrOverlay> {
        self.photo.as_mut()
    }

    fn poll(&mut self) -> Result<(), PhotoError> {
        for surface in [&mut self.icon, &mut self.photo].into_iter().flatten() {
            if surface
                .poll_runtime_events(32)
                .iter()
                .any(|event| event.classify() == OpenVrEventClass::Fatal)
            {
                return Err(PhotoError::OpenVr);
            }
        }
        Ok(())
    }
}

enum Backend {
    Probe,
    OpenVr(PhotoOpenVrSurfaces),
}

pub struct PhotoRuntime {
    state: PhotoState,
    images: PhotoImageBoundary,
    backend: Backend,
    pulls: HashSet<String>,
    drawn_reported: bool,
    haptic_granted: bool,
    haptic_accepted: bool,
    composed_blocks: usize,
    texture: Option<PhotoTexture>,
    preparation: PreparationWorker,
    preparation_started: bool,
    live_drawn_block: Option<u64>,
}

impl PhotoRuntime {
    pub fn new(manifest: &PhotoManifest) -> Result<(Self, PhotoImageWorker), PhotoError> {
        manifest.validate()?;
        let backend = match manifest.backend {
            PhotoBackend::Probe => Backend::Probe,
            PhotoBackend::Openvr => Backend::OpenVr(PhotoOpenVrSurfaces::new(manifest)?),
        };
        let (images, worker) = PhotoImageBoundary::channel();
        Ok((
            Self {
                state: PhotoState::default(),
                images,
                backend,
                pulls: HashSet::new(),
                drawn_reported: false,
                live_drawn_block: None,
                haptic_granted: false,
                haptic_accepted: false,
                composed_blocks: 0,
                texture: None,
                preparation: PreparationWorker::new().map_err(|_| PhotoError::Protocol)?,
                preparation_started: false,
            },
            worker,
        ))
    }

    pub fn state(&self) -> &PhotoState {
        &self.state
    }

    pub fn take_haptic_grant(&mut self) -> bool {
        std::mem::take(&mut self.haptic_granted)
    }

    pub async fn run(mut self, manifest: &PhotoManifest) -> Result<(), PhotoError> {
        let (mut bridge, first) =
            tokio::time::timeout(Duration::from_secs(10), PhotoBridge::connect(manifest))
                .await
                .map_err(|_| PhotoError::Bridge)??;
        self.receive(first, &mut bridge).await?;
        bridge.send(json!({"type": "photo_ready"})).await?;
        println!(
            "{}",
            json!({
                "type": "photo_ready", "role": "photo", "instance_id": manifest.instance_id,
                "generation": manifest.generation, "pid": std::process::id(), "backend": manifest.backend,
            })
        );
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        loop {
            tokio::select! {
                incoming = bridge.next() => {
                    if !self.receive(incoming?, &mut bridge).await? { break; }
                }
                Some(result) = self.images.next_result() => {
                    self.image_result(result, &mut bridge).await?;
                }
                Some(completion) = self.preparation.next_result() => {
                    self.prepared(completion, &mut bridge).await?;
                }
                _ = tick.tick() => {
                    if let Backend::OpenVr(surfaces) = &mut self.backend { surfaces.poll()?; }
                }
            }
        }
        self.preparation.shutdown().await;
        while let Some(completion) = self.preparation.try_result() {
            self.prepared(completion, &mut bridge).await?;
        }
        self.images.close();
        self.texture = None;
        if let Backend::OpenVr(surfaces) = &mut self.backend {
            surfaces.close_icon();
            surfaces.close_photo();
        }
        drop(self);
        bridge
            .send(json!({"type": "photo_shutdown_complete"}))
            .await?;
        println!(
            "{}",
            json!({
                "type": "photo_shutdown_complete", "role": "photo", "instance_id": manifest.instance_id,
                "generation": manifest.generation, "pid": std::process::id(),
            })
        );
        let _ = bridge.close().await;
        Ok(())
    }

    async fn receive(
        &mut self,
        incoming: PhotoIncoming,
        bridge: &mut PhotoBridge,
    ) -> Result<bool, PhotoError> {
        match incoming {
            PhotoIncoming::Snapshot { revision, photo } => {
                let prior = self
                    .state
                    .photo()
                    .map(|photo| (photo.photo_id.clone(), photo.attempt_id));
                if self.state.apply(revision, photo)? == SnapshotAdmission::Stale {
                    return Ok(true);
                }
                let current = self
                    .state
                    .photo()
                    .map(|photo| (photo.photo_id.clone(), photo.attempt_id));
                if prior != current {
                    self.pulls.clear();
                    self.preparation.cancel();
                    self.preparation_started = false;
                    self.texture = None;
                    self.haptic_granted = false;
                    self.haptic_accepted = false;
                    self.composed_blocks = 0;
                    self.live_drawn_block = None;
                    self.drawn_reported = self.state.photo().is_some_and(|photo| {
                        photo.first_block_drawn || (prior.is_none() && !photo.blocks.is_empty())
                    });
                }
                let snapshot = self.state.snapshot().ok_or(PhotoError::Protocol)?;
                self.images.publish(snapshot);
                let photo = snapshot.photo.as_deref();
                if let Backend::OpenVr(surfaces) = &mut self.backend {
                    if let Some(surface) = surfaces.photo() {
                        surface
                            .set_overlay_visible(
                                photo.is_some_and(|photo| photo.visible) && self.texture.is_some(),
                            )
                            .map_err(|_| PhotoError::OpenVr)?;
                    }
                }
                bridge.send(json!({
                    "type": "photo_state_received", "photo_id": photo.map(|photo| &photo.photo_id),
                    "attempt_id": photo.map_or(0, |photo| photo.attempt_id), "revision": snapshot.revision,
                    "block_count": photo.map_or(0, |photo| photo.blocks.len()),
                })).await?;
            }
            PhotoIncoming::Prepare(request) => {
                let current = self.state.photo().is_some_and(|photo| {
                    photo.photo_id == request.photo_id
                        && photo.attempt_id == request.attempt_id
                        && photo.attempt_id > 0
                        && photo.phase == super::state::PhotoPhase::Preparing
                        && photo.path == request.source_path
                });
                if !current {
                    bridge
                        .send(
                            json!({"type": "photo_prepare_failed", "photo_id": request.photo_id,
                        "attempt_id": request.attempt_id, "code": "preparation_not_current"}),
                        )
                        .await?;
                    return Ok(true);
                }
                if self.preparation_started {
                    bridge
                        .send(
                            json!({"type": "photo_prepare_failed", "photo_id": request.photo_id,
                        "attempt_id": request.attempt_id, "code": "preparation_already_started"}),
                        )
                        .await?;
                    return Ok(true);
                }
                self.preparation_started = true;
                let photo_id = request.photo_id.clone();
                let attempt_id = request.attempt_id;
                if let Err(code) = self.preparation.start(request) {
                    bridge
                        .send(json!({"type": "photo_prepare_failed", "photo_id": photo_id,
                        "attempt_id": attempt_id, "code": code}))
                        .await?;
                }
            }
            PhotoIncoming::ProbePull { photo_id, event_id } => {
                if !matches!(self.backend, Backend::Probe) {
                    return Err(PhotoError::Protocol);
                }
                if event_id.is_empty() || event_id.len() > 256 {
                    return Err(PhotoError::Protocol);
                }
                if self
                    .state
                    .photo()
                    .is_some_and(|photo| photo.photo_id == photo_id && photo.can_pull())
                    && self.pulls.insert(event_id.clone())
                {
                    bridge.send(json!({"type": "photo_pull", "photo_id": photo_id, "event_id": event_id})).await?;
                }
            }
            PhotoIncoming::ProbeHide { photo_id } => {
                if !matches!(self.backend, Backend::Probe) {
                    return Err(PhotoError::Protocol);
                }
                if self
                    .state
                    .photo()
                    .is_some_and(|photo| photo.photo_id == photo_id && photo.visible)
                {
                    bridge
                        .send(json!({"type": "photo_hidden", "photo_id": photo_id}))
                        .await?;
                }
            }
            PhotoIncoming::Haptic {
                photo_id,
                attempt_id,
                block_id,
            } => {
                self.grant_haptic(&photo_id, attempt_id, block_id);
            }
            PhotoIncoming::Shutdown => return Ok(false),
        }
        Ok(true)
    }
    fn grant_haptic(&mut self, photo_id: &str, attempt_id: u64, block_id: u64) {
        if !self.haptic_accepted
            && self.live_drawn_block == Some(block_id)
            && self.state.photo().is_some_and(|photo| {
                photo.photo_id == photo_id
                    && photo.attempt_id == attempt_id
                    && photo
                        .blocks
                        .first()
                        .is_some_and(|block| block.block_id == block_id)
            })
        {
            self.haptic_granted = true;
            self.haptic_accepted = true;
        }
    }

    async fn prepared(
        &mut self,
        completion: PrepareCompletion,
        bridge: &mut PhotoBridge,
    ) -> Result<(), PhotoError> {
        let request = completion.request;
        if !self.state.photo().is_some_and(|photo| {
            photo.photo_id == request.photo_id
                && photo.attempt_id == request.attempt_id
                && photo.phase == super::state::PhotoPhase::Preparing
        }) {
            drop(completion.result);
            bridge
                .send(
                    json!({"type": "photo_prepare_failed", "photo_id": request.photo_id,
                "attempt_id": request.attempt_id, "code": "preparation_cancelled"}),
                )
                .await?;
            return Ok(());
        }
        match completion.result {
            Ok(mut prepared) => {
                bridge.send(json!({
                    "type": "photo_prepared", "photo_id": request.photo_id, "attempt_id": request.attempt_id,
                    "source_path": prepared.source_path, "jpeg_path": prepared.jpeg_path,
                    "width": prepared.original.width(), "height": prepared.original.height(),
                    "scale": prepared.scale, "offset_x": prepared.offset_x, "offset_y": prepared.offset_y,
                    "preparation_ms": prepared.preparation_ms,
                    "decode_ms": prepared.decode_ms, "resize_ms": prepared.resize_ms,
                    "jpeg_ms": prepared.jpeg_ms,
                })).await?;
                println!(
                    "{}",
                    json!({
                        "type": "photo_image_prepared", "photo_id": request.photo_id, "attempt_id": request.attempt_id,
                        "preparation_ms": prepared.preparation_ms, "decode_ms": prepared.decode_ms,
                        "resize_ms": prepared.resize_ms, "jpeg_ms": prepared.jpeg_ms,
                    })
                );
                self.images.retain_original(
                    request.photo_id,
                    request.attempt_id,
                    prepared.source_path.to_string_lossy().into_owned(),
                    prepared.original.clone(),
                );
                prepared.hand_off();
            }
            Err(code) => {
                bridge
                    .send(
                        json!({"type": "photo_prepare_failed", "photo_id": request.photo_id,
                "attempt_id": request.attempt_id, "code": code}),
                    )
                    .await?
            }
        }
        Ok(())
    }

    async fn image_result(
        &mut self,
        result: PhotoImageResult,
        bridge: &mut PhotoBridge,
    ) -> Result<(), PhotoError> {
        let Some(snapshot) = self.state.snapshot() else {
            return Ok(());
        };
        if !self.images.accept(&result, snapshot) {
            return Ok(());
        }
        let (texture, block_ids, elapsed_ms) = match result.output {
            PhotoImageOutput::Prepared {
                block_id,
                stage,
                elapsed_ms,
            } => {
                println!(
                    "{}",
                    json!({
                        "type": "photo_image_stage", "photo_id": snapshot.photo.as_ref().map(|photo| &photo.photo_id),
                        "attempt_id": snapshot.photo.as_ref().map(|photo| photo.attempt_id),
                        "block_id": block_id, "stage": format!("{stage:?}"), "elapsed_ms": elapsed_ms,
                    })
                );
                return Ok(());
            }
            PhotoImageOutput::Composed {
                texture,
                block_ids,
                elapsed_ms,
            } => (texture, block_ids, elapsed_ms),
        };
        if block_ids.len() < self.composed_blocks {
            return Ok(());
        }
        let Backend::OpenVr(surfaces) = &mut self.backend else {
            return Ok(());
        };
        let Some(surface) = surfaces.photo() else {
            return Ok(());
        };
        #[cfg(windows)]
        surface
            .set_overlay_texture(texture.pointer())
            .map_err(|_| PhotoError::OpenVr)?;
        self.texture = Some(texture);
        self.composed_blocks = block_ids.len();
        let photo = snapshot.photo.as_deref().ok_or(PhotoError::Protocol)?;
        println!(
            "{}",
            json!({
                "type": "photo_image_stage", "photo_id": photo.photo_id,
                "attempt_id": photo.attempt_id, "stage": "composed", "elapsed_ms": elapsed_ms,
            })
        );
        surface
            .set_overlay_visible(photo.visible)
            .map_err(|_| PhotoError::OpenVr)?;
        if photo.visible && !self.drawn_reported && !photo.first_block_drawn {
            if let Some(block_id) = block_ids.first() {
                self.drawn_reported = true;
                self.live_drawn_block = Some(*block_id);
                bridge
                    .send(json!({
                        "type": "photo_first_block_drawn", "photo_id": photo.photo_id,
                        "attempt_id": photo.attempt_id, "block_id": block_id,
                    }))
                    .await?;
            }
        }
        Ok(())
    }
}

pub async fn run_cli(args: &[String]) -> i32 {
    if args.len() == 4
        && args[1] == "--role"
        && args[2] == "photo"
        && args[3] == "--check-startup-contract"
    {
        println!(
            "{}",
            json!({"protocol_version": PHOTO_PROTOCOL_VERSION, "role": "photo", "app_version": env!("CARGO_PKG_VERSION")})
        );
        return 0;
    }
    if args.len() != 5 || args[1] != "--role" || args[2] != "photo" || args[3] != "--config" {
        eprintln!("usage: PuriPulyHeartOverlay --role photo --config <manifest.json>");
        return 2;
    }
    let outcome = async {
        let manifest = PhotoManifest::load(Path::new(&args[4]))?;
        let (runtime, _image_worker) = PhotoRuntime::new(&manifest)?;
        runtime.run(&manifest).await
    }
    .await;
    match outcome {
        Ok(()) => 0,
        Err(error) => {
            eprintln!(
                "{}",
                json!({"type": "photo_runtime_error", "role": "photo", "code": error.code(), "pid": std::process::id()})
            );
            1
        }
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
