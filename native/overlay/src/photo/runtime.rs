#[cfg(windows)]
use super::compositor::PhotoCompositor;
use super::worker::RunningImageWorker;
use serde_json::json;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::bridge::{PhotoBridge, PhotoIncoming};
use super::image::{
    PhotoImageBoundary, PhotoImageOutput, PhotoImageResult, PhotoImageWorker, PhotoTexture,
};
use super::manifest::{PhotoBackend, PhotoError, PhotoManifest, PHOTO_PROTOCOL_VERSION};
use super::prepare::{PreparationCompletion, PreparationWorker, PrepareCompletion, SourceCompletion, SourceRequest};
use super::state::{PhotoState, SnapshotAdmission};
use super::interaction::{HapticCue, PhotoInteraction};
use super::calibration::{WristAdjustment, WristCalibration};
use super::interaction::compose;
#[cfg(windows)]
use super::icon::{self, IconState};
use crate::openvr::{
    OpenVrEventClass, OpenVrOverlay, OpenVrSession, OverlayFrameSubmitter,
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
    grab_volume: [Option<OpenVrOverlay>; 3],
    case_label: Option<OpenVrOverlay>,
    back: Option<OpenVrOverlay>,
    _session: OpenVrSession,
}

impl PhotoOpenVrSurfaces {
    pub fn new(manifest: &PhotoManifest) -> Result<Self, PhotoError> {
        let session = OpenVrSession::new().map_err(|_| PhotoError::OpenVr)?;
        Self::in_session(manifest, session)
    }

    pub(crate) fn in_session(manifest: &PhotoManifest, session: OpenVrSession) -> Result<Self, PhotoError> {
        let icon = OpenVrOverlay::in_session(&session, &surface_identity(manifest, "icon"))
            .map_err(|_| PhotoError::OpenVr)?;
        let photo = OpenVrOverlay::in_session(&session, &surface_identity(manifest, "photo"))
            .map_err(|_| PhotoError::OpenVr)?;
        let grab_volume = ["grab-volume.xy", "grab-volume.xz", "grab-volume.yz"].map(|name| {
            if manifest.interaction.debug_grab_volume {
                OpenVrOverlay::in_session(&session, &surface_identity(manifest, name))
                    .map(Some).map_err(|_| PhotoError::OpenVr)
            } else {
                Ok(None)
            }
        });
        let [xy, xz, yz] = grab_volume;
        let grab_volume = [xy?, xz?, yz?];
        let case_label = if manifest.interaction.debug_case_label.is_some() || manifest.interaction.debug_calibrate_wrist {
            Some(OpenVrOverlay::in_session(&session, &surface_identity(manifest, "case-label"))
                .map_err(|_| PhotoError::OpenVr)?)
        } else {
            None
        };
        let back = OpenVrOverlay::in_session(&session, &surface_identity(manifest, "back"))
            .map_err(|_| PhotoError::OpenVr)?;
        Ok(Self {
            icon: Some(icon),
            photo: Some(photo),
            grab_volume,
            case_label,
            back: Some(back),
            _session: session,
        })
    }

    pub fn close_icon(&mut self) {
        self.icon.take();
    }
    pub fn close_photo(&mut self) {
        self.photo.take();
        self.back.take();
    }
    pub fn icon(&mut self) -> Option<&mut OpenVrOverlay> {
        self.icon.as_mut()
    }
    pub fn photo(&mut self) -> Option<&mut OpenVrOverlay> {
        self.photo.as_mut()
    }

    fn surfaces(&mut self) -> impl Iterator<Item = &mut OpenVrOverlay> {
        [&mut self.icon, &mut self.photo].into_iter()
            .chain([&mut self.back]).chain(self.grab_volume.iter_mut()).chain([&mut self.case_label]).flatten()
    }

    fn close_diagnostics(&mut self) {
        for surface in &mut self.grab_volume {
            surface.take();
        }
        self.case_label.take();
    }

    fn poll(&mut self) -> Result<(), PhotoError> {
        for surface in self.surfaces() {
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
    Offscreen,
}

pub struct PhotoRuntime {
    state: PhotoState,
    images: PhotoImageBoundary,
    backend: Backend,
    pulls: HashSet<String>,
    drawn_reported: bool,
    haptic_granted: bool,
    haptic_accepted: bool,
    texture: Option<PhotoTexture>,
    preparation: PreparationWorker,
    preparation_started: bool,
    source_request: Option<SourceRequest>,
    source_max_size: [u32; 2],
    live_drawn_block: Option<u64>,
    #[cfg(windows)]
    compositor: Option<PhotoCompositor>,
    #[cfg(windows)]
    export_dir: Option<std::path::PathBuf>,
    debug_controls: bool,
    pending_gpu_block: Option<u64>,
    interaction: PhotoInteraction,
    pull_sequence: u64,
    #[cfg(windows)]
    icons: Option<[PhotoTexture; 3]>,
    icon_state: Option<usize>,
    #[cfg(windows)]
    grab_ring: Option<PhotoTexture>,
    #[cfg(windows)]
    case_labels: Option<[PhotoTexture; 2]>,
    diagnostic_submitted: bool,
    label_state: Option<usize>,
    quiesce_requests: HashSet<String>,
    quiesce_request: Option<(String, bool)>,
    wrist_adjustment: Option<WristAdjustment>,
    wrist_reference_epoch: Option<u64>,
    input: Option<crate::openvr::LegacyVrInput>,
    #[cfg(windows)]
    card_material: Option<PhotoTexture>,
    card_submitted: bool,
    #[cfg(windows)]
    calibration_label: Option<PhotoTexture>,
}

impl PhotoRuntime {
    pub fn new(manifest: &PhotoManifest) -> Result<(Self, PhotoImageWorker), PhotoError> {
        manifest.validate()?;
        let mut backend = match manifest.backend {
            PhotoBackend::Probe => Backend::Probe,
            PhotoBackend::Openvr => Backend::OpenVr(PhotoOpenVrSurfaces::new(manifest)?),
            PhotoBackend::Offscreen => Backend::Offscreen,
        };
        #[cfg(windows)]
        let compositor = match &mut backend {
            Backend::Probe => None,
            Backend::OpenVr(surfaces) => Some(PhotoCompositor::new(Some(
                &surfaces.photo().ok_or(PhotoError::OpenVr)?.output_adapter(),
            ))?),
            Backend::Offscreen => Some(PhotoCompositor::new(None)?),
        };
        #[cfg(not(windows))]
        if !matches!(backend, Backend::Probe) {
            return Err(PhotoError::Graphics);
        }
        #[cfg(windows)]
        let icons = if let Some(compositor) = &compositor {
            let textures = [IconState::Idle, IconState::Grab, IconState::Failed]
                .map(|state| compositor.icon_texture(&icon::pixels(state), icon::ICON_SIZE));
            let [idle, grab, failed] = textures;
            Some([idle?, grab?, failed?])
        } else {
            None
        };
        #[cfg(windows)]
        let grab_ring = compositor.as_ref().filter(|_| manifest.interaction.debug_grab_volume)
            .map(|compositor| compositor.icon_texture(&icon::grab_ring_pixels(), super::interaction::GRAB_RING_SIZE))
            .transpose()?;
        #[cfg(windows)]
        let case_labels = if let (Some(compositor), Some(label)) = (&compositor, &manifest.interaction.debug_case_label) {
            Some([compositor.case_label_texture(label, false)?, compositor.case_label_texture(label, true)?])
        } else {
            None
        };
        #[cfg(windows)]
        let card_material = compositor.as_ref().map(|compositor|
            compositor.icon_texture(&[205, 205, 200, 255], 1)
        ).transpose()?;
        #[cfg(windows)]
        let calibration_label = compositor.as_ref().filter(|_| manifest.interaction.debug_calibrate_wrist)
            .map(|compositor| compositor.case_label_texture("WRIST SETUP\nGrip, move, release", false)).transpose()?;
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
                texture: None,
                preparation: PreparationWorker::new().map_err(|_| PhotoError::Protocol)?,
                preparation_started: false,
                source_request: None,
                source_max_size: manifest.image.max_size,
                #[cfg(windows)]
                compositor,
                debug_controls: manifest.image.debug_controls,
                #[cfg(windows)]
                export_dir: manifest.image.export_dir.clone(),
                pending_gpu_block: None,
                interaction: PhotoInteraction::new(manifest.interaction.clone()),
                pull_sequence: 0,
                #[cfg(windows)]
                icons,
                icon_state: None,
                #[cfg(windows)]
                grab_ring,
                #[cfg(windows)]
                case_labels,
                diagnostic_submitted: false,
                label_state: None,
                quiesce_requests: HashSet::new(),
                quiesce_request: None,
                wrist_adjustment: manifest.interaction.debug_calibrate_wrist.then(WristAdjustment::default),
                wrist_reference_epoch: None,
                input: None,
                #[cfg(windows)]
                card_material,
                card_submitted: false,
                #[cfg(windows)]
                calibration_label,
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

    pub async fn run(
        mut self,
        manifest: &PhotoManifest,
        worker: PhotoImageWorker,
    ) -> Result<(), PhotoError> {
        #[cfg(windows)]
        if let Some(compositor) = &mut self.compositor {
            compositor.ready().await?;
        }
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
        let worker = if matches!(self.backend, Backend::Probe) {
            None
        } else {
            Some(RunningImageWorker::start(worker, manifest.image.clone())?)
        };
        let mut outcome: Result<(), PhotoError> = async {
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
                        match completion {
                            PreparationCompletion::Prepared(completion) => self.prepared(completion, &mut bridge).await?,
                            PreparationCompletion::Source(completion) => self.preview_source(completion, &mut bridge).await?,
                        }
                    }
                    _ = tick.tick() => {
                        if let Backend::OpenVr(surfaces) = &mut self.backend { surfaces.poll()?; }
                        self.interaction_tick(&mut bridge).await?;
                        self.render_frame(&mut bridge).await?;
                        self.ack_quiescent(&mut bridge).await?;
                    }
                }
            }
            Ok(())
        }
        .await;
        self.preparation.shutdown().await;
        while let Some(completion) = self.preparation.try_result() {
            let PreparationCompletion::Prepared(completion) = completion else { continue };
            if let Err(error) = self.prepared(completion, &mut bridge).await {
                if outcome.is_ok() {
                    outcome = Err(error);
                }
            }
        }
        self.images.close();
        if let Some(worker) = worker {
            worker.shutdown().await;
        }
        if let Backend::OpenVr(surfaces) = &mut self.backend {
            surfaces.close_diagnostics();
            surfaces.close_icon();
            surfaces.close_photo();
        }
        self.texture = None;
        outcome?;
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
                    .filter(|photo| photo.phase != super::state::PhotoPhase::Expired)
                    .map(|photo| (photo.photo_id.clone(), photo.attempt_id));
                if self.state.apply(revision, photo)? == SnapshotAdmission::Stale {
                    return Ok(true);
                }
                let current = self
                    .state
                    .photo()
                    .filter(|photo| photo.phase != super::state::PhotoPhase::Expired)
                    .map(|photo| (photo.photo_id.clone(), photo.attempt_id));
                if prior != current {
                    self.pulls.clear();
                    self.preparation.cancel_prepare();
                    self.preparation_started = false;
                    if prior.as_ref().map(|key| &key.0) != current.as_ref().map(|key| &key.0) {
                        self.preparation.cancel_source();
                        self.source_request = None;
                        self.texture = None;
                    }
                    #[cfg(windows)]
                    if let Some(compositor) = &mut self.compositor {
                        if prior.as_ref().map(|key| &key.0) == current.as_ref().map(|key| &key.0) {
                            compositor.clear_attempt();
                        } else {
                            compositor.clear_photo();
                        }
                    }
                    self.reset_feedback(prior.is_none());
                }
                self.interaction
                    .snapshot(self.state.photo(), Instant::now(), unix_ms());
                if !self.interaction.quiescing() {
                    self.quiesce_request = None;
                }
                let snapshot = self.state.snapshot().ok_or(PhotoError::Protocol)?;
                self.images.publish(snapshot);
                let photo = snapshot.photo.as_deref();
                if let Backend::OpenVr(surfaces) = &mut self.backend {
                    if current.is_none() {
                        for surface in surfaces.surfaces() {
                            surface
                                .set_overlay_visible(false)
                                .map_err(|_| PhotoError::OpenVr)?;
                        }
                    }
                }
                bridge.send(json!({
                    "type": "photo_state_received", "photo_id": photo.map(|photo| &photo.photo_id),
                    "attempt_id": photo.map_or(0, |photo| photo.attempt_id), "revision": snapshot.revision,
                    "block_count": photo.map_or(0, |photo| photo.blocks.len()),
                })).await?;
                #[cfg(windows)]
                if let Some(compositor) = &mut self.compositor {
                    compositor.invalidate();
                }
                self.render_frame(bridge).await?;
            }
            PhotoIncoming::Prepare(request) => {
                let current = self.state.photo().is_some_and(|photo| {
                    photo.photo_id == request.photo_id
                        && photo.attempt_id == request.attempt_id
                        && photo.attempt_id > 0
                        && photo.phase == super::state::PhotoPhase::Preparing
                        && photo.path == request.source_path
                        && photo.expected_file_id == request.expected_file_id
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
            PhotoIncoming::DebugPreviewSource { photo_id } => {
                if !self.debug_controls || !matches!(self.backend, Backend::Offscreen) {
                    return Err(PhotoError::Protocol);
                }
                if self.state.photo().is_some_and(|photo| photo.photo_id == photo_id) {
                    self.enqueue_preview_source();
                }
            }
            PhotoIncoming::ProbePull { photo_id, event_id } => {
                if !matches!(self.backend, Backend::Probe) && !self.debug_controls {
                    return Err(PhotoError::Protocol);
                }
                if event_id.is_empty() || event_id.len() > 256 {
                    return Err(PhotoError::Protocol);
                }
                if !self.interaction.quiescing() && self.state
                    .photo()
                    .is_some_and(|photo| photo.photo_id == photo_id && photo.can_pull())
                    && self.pulls.insert(event_id.clone())
                {
                    bridge.send(json!({"type": "photo_pull", "photo_id": photo_id, "event_id": event_id})).await?;
                }
            }
            PhotoIncoming::ProbeHide { photo_id } => {
                if !matches!(self.backend, Backend::Probe) && !self.debug_controls {
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
                if self.take_haptic_grant() {
                    self.haptic(HapticCue::FirstBlock);
                }
            }
            PhotoIncoming::DebugExport { photo_id, label } => {
                if !self.debug_controls
                    || label.is_empty()
                    || label.len() > 64
                    || !label
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
                {
                    return Err(PhotoError::Protocol);
                }
                if self
                    .state
                    .photo()
                    .is_some_and(|photo| photo.photo_id == photo_id)
                {
                    #[cfg(windows)]
                    if let (Some(compositor), Some(directory)) =
                        (&mut self.compositor, &self.export_dir)
                    {
                        std::fs::create_dir_all(directory).map_err(|_| PhotoError::Image)?;
                        compositor.render(Instant::now(), false)?;
                        compositor.ready().await?;
                        compositor.save(&directory.join(format!("{label}.png")))?;
                        if let Some(texture) = &self.grab_ring {
                            let size = super::interaction::GRAB_RING_SIZE;
                            compositor.save_texture(texture, size, size, &directory.join(format!("{label}-grab-ring.png")))?;
                        }
                        if let Some(textures) = &self.case_labels {
                            let [width, height] = super::interaction::CASE_LABEL_SIZE;
                            for (texture, status) in textures.iter().zip(["case", "draining"]) {
                                compositor.save_texture(texture, width, height, &directory.join(format!("{label}-{status}.png")))?;
                            }
                        }
                        std::fs::write(
                            directory.join(format!("{label}-interaction.json")),
                            serde_json::to_vec_pretty(&json!({
                                "interaction": self.interaction.config,
                                "frame": self.interaction.frame,
                                "quiescing": self.interaction.quiescing(),
                            })).map_err(|_| PhotoError::Image)?,
                        ).map_err(|_| PhotoError::Image)?;
                        std::fs::write(
                            directory.join(format!("{label}.json")),
                            serde_json::to_vec_pretty(&compositor.metadata())
                                .map_err(|_| PhotoError::Image)?,
                        )
                        .map_err(|_| PhotoError::Image)?;
                    }
                }
            }
            PhotoIncoming::DebugQuiesce { request_id } => {
                if !self.debug_controls
                    || self.interaction.config.debug_case_label.is_none()
                    || request_id.is_empty()
                    || request_id.len() > 128
                    || !request_id.is_ascii()
                {
                    return Err(PhotoError::Protocol);
                }
                if self.quiesce_requests.insert(request_id.clone()) {
                    self.interaction.quiesce();
                    self.quiesce_request = Some((request_id, false));
                }
            }
            PhotoIncoming::Shutdown => return Ok(false),
        }
        Ok(true)
    }

    async fn ack_quiescent(&mut self, bridge: &mut PhotoBridge) -> Result<(), PhotoError> {
        if self.interaction.quiescent() {
            if let Some((request_id, acknowledged)) = &mut self.quiesce_request {
                if !*acknowledged {
                    bridge.send(json!({"type":"photo_debug_quiescent", "request_id":request_id})).await?;
                    *acknowledged = true;
                }
            }
        }
        Ok(())
    }
    fn reset_feedback(&mut self, recovering: bool) {
        self.haptic_granted = false;
        self.haptic_accepted = false;
        self.live_drawn_block = None;
        self.pending_gpu_block = None;
        self.drawn_reported = self.state.photo().is_some_and(|photo| {
            photo.first_block_drawn || (recovering && !photo.blocks.is_empty())
        });
    }

    fn grant_haptic(&mut self, photo_id: &str, attempt_id: u64, block_id: u64) {
        if !self.haptic_accepted
            && self.live_drawn_block == Some(block_id)
            && self.state.photo().is_some_and(|photo| {
                photo.photo_id == photo_id
                    && photo.attempt_id == attempt_id
                    && photo.blocks.iter().any(|block| block.block_id == block_id)
            })
        {
            self.haptic_granted = true;
            self.haptic_accepted = true;
        }
    }

    fn enqueue_preview_source(&mut self) {
        let Some(photo) = self.state.photo().filter(|photo| {
            photo.image.is_none() && photo.can_pull() && self.interaction.active(photo, Instant::now())
        }) else {
            return;
        };
        if self.source_request.is_some() || self.interaction.quiescing() {
            return;
        }
        let request = SourceRequest {
            photo_id: photo.photo_id.clone(),
            source_path: photo.path.clone(),
            expected_file_id: photo.expected_file_id.clone(),
            max_size: self.source_max_size,
        };
        match self.preparation.start_source(request.clone()) {
            Ok(()) => self.source_request = Some(request),
            Err(code) => eprintln!("{}", json!({"type":"photo_image_failed","stage":"local_source","code":code})),
        }
    }

    async fn preview_source(&mut self, completion: SourceCompletion, bridge: &mut PhotoBridge) -> Result<(), PhotoError> {
        if self.source_request.as_ref() != Some(&completion.request)
            || !self.state.photo().is_some_and(|photo| {
                photo.photo_id == completion.request.photo_id
                    && photo.path == completion.request.source_path
                    && photo.expected_file_id == completion.request.expected_file_id
                    && self.interaction.active(photo, Instant::now())
            })
        {
            return Ok(());
        }
        match completion.result {
            Ok(source) => {
                let photo = self.state.photo().ok_or(PhotoError::Protocol)?;
                if photo.image.as_ref().is_none_or(|image| image.path == completion.request.source_path) {
                    self.images.retain_original(
                        completion.request.photo_id,
                        photo.attempt_id,
                        completion.request.source_path,
                        source.original.clone(),
                        source.working.clone(),
                    );
                }
                #[cfg(windows)]
                if let Some(compositor) = &mut self.compositor {
                    compositor.set_original(source.working.as_ref().ok_or(PhotoError::Image)?.clone())?;
                }
                println!("{}", json!({
                    "type":"photo_image_stage", "stage":"display_source_ready", "local_preview":true,
                    "before_authorization":self.state.photo().is_some_and(|photo| photo.attempt_id == 0),
                }));
                self.render_frame(bridge).await?;
            }
            Err(code) => {
                self.source_request = None;
                if code != "preparation_cancelled" {
                    eprintln!("{}", json!({"type":"photo_image_failed","stage":"local_source","code":code}));
                }
            }
        }
        Ok(())
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
                    prepared.working.clone(),
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
        match result.output {
            PhotoImageOutput::Timing {
                block_id,
                stage,
                elapsed_ms,
            } => {
                #[cfg(windows)]
                if let Some(compositor) = &mut self.compositor {
                    compositor.record_timing(block_id, stage, elapsed_ms);
                }
                let photo = snapshot.photo.as_deref().ok_or(PhotoError::Protocol)?;
                bridge
                    .send_timing(
                        &photo.photo_id,
                        photo.attempt_id,
                        block_id,
                        stage,
                        elapsed_ms,
                    )
                    .await?;
                return Ok(());
            }
            PhotoImageOutput::FontReady { font } => {
                println!(
                    "{}",
                    json!({
                        "type":"photo_image_stage","stage":"typesetter_ready",
                        "family":font.family,"source":font.source,"hangul_ink_ratio":font.hangul_ink_ratio,
                    })
                );
                #[cfg(windows)]
                if let Some(compositor) = &mut self.compositor {
                    compositor.set_font(font);
                }
            }
            PhotoImageOutput::Candidate {
                block_id,
                rect,
                elapsed_ms,
            } => {
                println!(
                    "{}",
                    json!({
                        "type": "photo_image_stage", "stage": "bbox_candidate", "block_id": block_id,
                        "elapsed_ms": elapsed_ms, "candidate_width": rect.width, "candidate_height": rect.height,
                    })
                );
            }
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
            }
            PhotoImageOutput::Original { working, recovered } => {
                #[cfg(windows)]
                if let Some(compositor) = &mut self.compositor {
                    compositor.set_original(working)?;
                }
                println!(
                    "{}",
                    json!({"type": "photo_image_stage", "stage": "display_source_ready", "recovered": recovered})
                );
            }
            #[cfg(windows)]
            PhotoImageOutput::Block {
                block_id,
                region,
                typeset,
                typeset_ms,
            } =>
            {
                #[cfg(windows)]
                if let Some(compositor) = &mut self.compositor {
                    let translated = region.is_some();
                    let has_patch = region.as_ref().is_some_and(|region| region.erase.is_some());
                    let roi_ms = compositor.add_block(
                        block_id,
                        region,
                        typeset,
                        typeset_ms,
                        Instant::now(),
                    )?;
                    if translated {
                        self.pending_gpu_block = Some(block_id);
                    }
                    if has_patch {
                        let photo = snapshot.photo.as_deref().ok_or(PhotoError::Protocol)?;
                        bridge
                            .send_timing(
                                &photo.photo_id,
                                photo.attempt_id,
                                block_id,
                                "roi_upload",
                                roi_ms,
                            )
                            .await?;
                    }
                }
            }
            #[cfg(not(windows))]
            PhotoImageOutput::Block { .. } => return Err(PhotoError::Graphics),
            PhotoImageOutput::Failed { code } => {
                eprintln!("{}", json!({"type": "photo_image_failed", "code": code}));
                return Err(PhotoError::Image);
            }
        }
        self.render_frame(bridge).await?;
        Ok(())
    }

    fn haptic(&mut self, cue: HapticCue) {
        if let Backend::OpenVr(surfaces) = &self.backend {
            let delivered = surfaces._session.pulse(cue.hand());
            println!(
                "{}",
                json!({"type":"photo_interaction", "event":"haptic", "cue":format!("{cue:?}"), "hand":format!("{:?}", cue.hand()), "software_pulse":delivered})
            );
        }
    }

    async fn interaction_tick(&mut self, bridge: &mut PhotoBridge) -> Result<(), PhotoError> {
        if !self.state.photo().is_some_and(|photo| self.interaction.active(photo, Instant::now())) {
            self.preparation.cancel_source();
            self.source_request = None;
        }
        let Backend::OpenVr(surfaces) = &self.backend else {
            return Ok(());
        };
        let input = surfaces._session.legacy_input();
        if let Some(input) = input {
            if self.wrist_adjustment.is_none() && self.wrist_reference_epoch != Some(input.reference_epoch) {
                if let Some(calibration) = &self.interaction.config.wrist_calibration {
                    if !surfaces._session.wrist_reference_matches(&calibration.reference) {
                        println!("{}", json!({"type":"photo_runtime_error","code":"wrist_calibration_reference_mismatch"}));
                        return Err(PhotoError::OpenVr);
                    }
                }
                self.wrist_reference_epoch = Some(input.reference_epoch);
            }
        }
        if let Some(adjustment) = &mut self.wrist_adjustment {
            let (frame, transform) = adjustment.step(input, &self.interaction.config);
            self.interaction.frame = frame;
            self.input = input;
            if let Some(transform) = transform {
                let calibration = WristCalibration {
                    transform,
                    reference: surfaces._session.wrist_reference().ok_or(PhotoError::OpenVr)?,
                };
                if !calibration.valid() { return Err(PhotoError::OpenVr); }
                self.interaction.config.wrist_transform = None;
                self.interaction.config.wrist_handgrip_point_m = None;
                self.interaction.config.wrist_calibration = Some(calibration.clone());
                bridge.send(json!({"type":"photo_debug_wrist_calibrated","calibration":calibration})).await?;
                self.wrist_adjustment = None;
                self.label_state = None;
                self.icon_state = None;
            }
            return Ok(());
        }
        self.interaction_input(input, Instant::now(), bridge).await
    }

    async fn interaction_input(
        &mut self,
        input: Option<crate::openvr::LegacyVrInput>,
        now: Instant,
        bridge: &mut PhotoBridge,
    ) -> Result<(), PhotoError> {
        self.input = input;
        #[cfg(windows)]
        if let Some(pixels) = self.compositor.as_ref().and_then(|compositor| compositor.dimensions()) {
            self.interaction.photo_pixels = pixels;
        }
        let frame = self.interaction.step(self.state.photo(), input, now);
        let records: Vec<_> = self.state.photo().map(|photo| {
            self.interaction.notes.iter().map(|note| {
                let mut record = json!(note);
                record["type"] = json!("photo_interaction");
                record["photo_id"] = json!(photo.photo_id);
                record["attempt_id"] = json!(photo.attempt_id);
                record
            }).collect()
        }).unwrap_or_default();
        for record in records {
            println!("{record}");
            bridge.send(record).await?;
        }
        if frame.grab {
            self.enqueue_preview_source();
        }
        if frame.pull {
            self.haptic(HapticCue::Pull);
            self.pull_sequence += 1;
            if let Some(photo) = self.state.photo() {
                bridge.send(json!({"type":"photo_pull","photo_id":photo.photo_id,"event_id":format!("native-{}",self.pull_sequence)})).await?;
            }
        }
        if frame.hidden {
            if let Some(photo) = self.state.photo() {
                bridge
                    .send(json!({"type":"photo_hidden","photo_id":photo.photo_id}))
                    .await?;
            }
        }
        Ok(())
    }
    fn take_draw_report(
        &mut self,
        drawn: bool,
        block_id: Option<u64>,
    ) -> Option<(String, u64, u64)> {
        let photo = self.state.photo()?;
        if !drawn || self.drawn_reported || photo.first_block_drawn {
            return None;
        }
        let block_id = block_id?;
        self.drawn_reported = true;
        self.live_drawn_block = Some(block_id);
        Some((photo.photo_id.clone(), photo.attempt_id, block_id))
    }

    async fn render_frame(&mut self, bridge: &mut PhotoBridge) -> Result<(), PhotoError> {
        #[cfg(windows)]
        {
            let now = Instant::now();
            let start = now;
            let Some(compositor) = &mut self.compositor else {
                return Ok(());
            };
            let updated = compositor.needs_frame(now);
            if updated {
                if let Some(texture) = compositor.render(now, false)? {
                    compositor.ready().await?;
                    self.texture = Some(texture);
                }
            }
            let block_count = compositor.block_count();
            let photo = self.state.photo();
            let mut drawn = false;
            if let Backend::OpenVr(surfaces) = &mut self.backend {
                let frame = &self.interaction.frame;
                let active = self.wrist_adjustment.is_some() || photo.is_some_and(|photo| self.interaction.active(photo, now));
                let icon = surfaces.icon().ok_or(PhotoError::OpenVr)?;
                if let Some(placement) = frame.icon.filter(|_| active) {
                    let state =
                        if photo.is_some_and(|p| p.phase == super::state::PhotoPhase::Failed) {
                            2
                        } else if frame.glow {
                            1
                        } else {
                            0
                        };
                    if self.icon_state != Some(state) {
                        compositor.submit_texture(
                            icon,
                            &self.icons.as_ref().ok_or(PhotoError::Graphics)?[state],
                        )?;
                        self.icon_state = Some(state);
                    }
                    icon.set_tracked_presentation(
                        placement.device,
                        placement.transform,
                        placement.width,
                        placement.alpha,
                    )
                    .map_err(|_| PhotoError::OpenVr)?;
                    icon.set_overlay_visible(true)
                        .map_err(|_| PhotoError::OpenVr)?;
                } else {
                    icon.set_overlay_visible(false)
                        .map_err(|_| PhotoError::OpenVr)?;
                }
                if let Some(placements) = frame.grab_volume.filter(|_| active) {
                    for (surface, placement) in surfaces.grab_volume.iter_mut().zip(placements) {
                        let surface = surface.as_mut().ok_or(PhotoError::OpenVr)?;
                        if !self.diagnostic_submitted {
                            compositor.submit_texture(surface, self.grab_ring.as_ref().ok_or(PhotoError::Graphics)?)?;
                        }
                        surface.set_tracked_presentation(placement.device, placement.transform, placement.width, placement.alpha)
                            .map_err(|_| PhotoError::OpenVr)?;
                        surface.set_overlay_visible(true).map_err(|_| PhotoError::OpenVr)?;
                    }
                    self.diagnostic_submitted = true;
                } else {
                    for surface in surfaces.grab_volume.iter_mut().flatten() {
                        surface.set_overlay_visible(false).map_err(|_| PhotoError::OpenVr)?;
                    }
                }
                if let Some(surface) = &mut surfaces.case_label {
                    if let Some(placement) = frame.case_label.filter(|_| active) {
                        let state = if self.wrist_adjustment.is_some() { 2 } else { usize::from(self.interaction.quiescing()) };
                        if self.label_state != Some(state) {
                            let texture = if state == 2 {
                                self.calibration_label.as_ref().ok_or(PhotoError::Graphics)?
                            } else {
                                &self.case_labels.as_ref().ok_or(PhotoError::Graphics)?[state]
                            };
                            compositor.submit_texture(surface, texture)?;
                            self.label_state = Some(state);
                        }
                        surface.set_tracked_presentation(placement.device, placement.transform, placement.width, placement.alpha)
                            .map_err(|_| PhotoError::OpenVr)?;
                        surface.set_overlay_visible(true).map_err(|_| PhotoError::OpenVr)?;
                    } else {
                        surface.set_overlay_visible(false).map_err(|_| PhotoError::OpenVr)?;
                    }
                }
                let surface = surfaces.photo.as_mut().ok_or(PhotoError::OpenVr)?;
                if updated {
                    if let Some(texture) = &self.texture { compositor.submit_texture(surface, texture)?; }
                }
                if let (Some(placement), Some(_), Some(input), Some(pixels)) =
                    (frame.photo.filter(|_| active), &self.texture, self.input, compositor.dimensions())
                {
                    let device = (!frame.photo_world).then_some(placement.device);
                    let holder = if placement.device == input.right_index { input.right } else { input.left };
                    let world = if frame.photo_world { placement.transform } else { compose(holder, placement.transform) };
                    let front_visible = super::card::front_facing(world, input.head);
                    surface.set_card_presentation(device, placement.transform, placement.width, placement.alpha, 1.0, [0.0, 0.0, frame.photo_extent, 1.0])
                        .map_err(|_| PhotoError::OpenVr)?;
                    surface.set_overlay_visible(front_visible).map_err(|_| PhotoError::OpenVr)?;
                    let height = super::card::card_size(self.interaction.config.photo_width_m, pixels)[1];
                    let back = surfaces.back.as_mut().ok_or(PhotoError::OpenVr)?;
                    if !self.card_submitted {
                        compositor.submit_texture(back, self.card_material.as_ref().ok_or(PhotoError::Graphics)?)?;
                        self.card_submitted = true;
                    }
                    back.set_card_presentation(device, super::card::back(placement.transform), placement.width, placement.alpha, placement.width / height, [0.0, 0.0, 1.0, 1.0])
                        .map_err(|_| PhotoError::OpenVr)?;
                    back.set_overlay_visible(!front_visible).map_err(|_| PhotoError::OpenVr)?;
                    drawn = front_visible && placement.alpha > 0.0;
                } else {
                    surface.set_overlay_visible(false).map_err(|_| PhotoError::OpenVr)?;
                    if let Some(back) = &mut surfaces.back {
                        back.set_overlay_visible(false).map_err(|_| PhotoError::OpenVr)?;
                    }
                }
            }
            let Some(photo) = photo else {
                return Ok(());
            };
            if updated {
                println!(
                    "{}",
                    json!({
                        "type": "photo_image_stage", "photo_id": photo.photo_id,
                        "attempt_id": photo.attempt_id, "stage": "composed", "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0,
                        "block_count": block_count, "original_uploads": compositor.original_uploads,
                        "gpu_ms": compositor.frame_gpu_ms, "gpu_timing_valid": compositor.frame_gpu_ms.is_some(),
                        "submitted": matches!(self.backend, Backend::OpenVr(_)),
                    })
                );
                if let Some(block_id) = self.pending_gpu_block.take() {
                    if let Some(gpu_ms) = compositor.frame_gpu_ms {
                        bridge
                            .send_timing(
                                &photo.photo_id,
                                photo.attempt_id,
                                block_id,
                                "compose_gpu",
                                gpu_ms,
                            )
                            .await?;
                    }
                }
            }
            let visible_block = if drawn && !self.drawn_reported && !photo.first_block_drawn {
                compositor.first_visible_block_in(now, [0.0, 0.0, self.interaction.frame.photo_extent, 1.0])
            } else {
                None
            };
            if let Some((photo_id, attempt_id, block_id)) =
                self.take_draw_report(drawn, visible_block)
            {
                bridge
                    .send(json!({
                        "type": "photo_first_block_drawn", "photo_id": photo_id,
                        "attempt_id": attempt_id, "block_id": block_id,
                    }))
                    .await?;
            }
        }
        Ok(())
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
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
    #[cfg(windows)]
    if args.len() == 5 && args[1] == "--role" && args[2] == "photo" && args[3] == "--offline" {
        return match super::offline::run(Path::new(&args[4])).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!(
                    "{}",
                    json!({"type": "photo_runtime_error", "code": error.code()})
                );
                1
            }
        };
    }
    if args.len() != 5 || args[1] != "--role" || args[2] != "photo" || args[3] != "--config" {
        eprintln!("usage: PuriPulyHeartOverlay --role photo --config <manifest.json>");
        return 2;
    }
    let outcome = async {
        let manifest = PhotoManifest::load(Path::new(&args[4]))?;
        let (runtime, image_worker) = PhotoRuntime::new(&manifest)?;
        runtime.run(&manifest, image_worker).await
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
