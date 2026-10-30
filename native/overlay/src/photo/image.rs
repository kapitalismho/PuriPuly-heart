use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

use super::state::{Photo, PhotoSnapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreparationStage {
    Bbox,
    Source,
}

pub struct PhotoTexture {
    #[cfg(windows)]
    texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
}

impl PhotoTexture {
    #[cfg(windows)]
    pub fn new(texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D) -> Self {
        Self { texture }
    }

    #[cfg(windows)]
    pub fn texture(&self) -> &windows::Win32::Graphics::Direct3D11::ID3D11Texture2D {
        &self.texture
    }

    #[cfg(windows)]
    pub(crate) fn pointer(&self) -> *mut std::ffi::c_void {
        use windows::core::Interface;
        self.texture.as_raw()
    }
}

pub enum PhotoImageOutput {
    Prepared {
        block_id: u64,
        stage: PreparationStage,
        elapsed_ms: f64,
    },
    Composed {
        texture: PhotoTexture,
        block_ids: Vec<u64>,
        elapsed_ms: f64,
    },
}

pub struct PhotoImageResult {
    pub input: Arc<PhotoSnapshot>,
    pub output: PhotoImageOutput,
}

pub struct DecodedOriginal {
    pub photo_id: String,
    pub attempt_id: u64,
    pub path: String,
    pub pixels: Arc<image::RgbImage>,
}

pub struct PhotoImageWorker {
    input: watch::Receiver<Option<Arc<PhotoSnapshot>>>,
    output: mpsc::Sender<PhotoImageResult>,
    original: watch::Receiver<Option<Arc<DecodedOriginal>>>,
}

impl PhotoImageWorker {
    pub fn decoded_original(&self, input: &PhotoSnapshot) -> Option<Arc<DecodedOriginal>> {
        let original = self.original.borrow();
        let original = original.as_ref()?;
        input
            .photo
            .as_ref()
            .filter(|photo| {
                photo.photo_id == original.photo_id
                    && photo.attempt_id == original.attempt_id
                    && photo
                        .image
                        .as_ref()
                        .is_some_and(|image| image.path == original.path)
            })
            .map(|_| original.clone())
    }

    pub async fn next_input(&mut self) -> Option<Arc<PhotoSnapshot>> {
        loop {
            self.input.changed().await.ok()?;
            if let Some(input) = self.input.borrow_and_update().clone() {
                return Some(input);
            }
        }
    }

    pub async fn complete(&self, result: PhotoImageResult) -> Result<(), PhotoImageResult> {
        self.output.send(result).await.map_err(|error| error.0)
    }

    pub fn is_cancelled(&self, input: &PhotoSnapshot) -> bool {
        let latest = self.input.borrow();
        !latest
            .as_deref()
            .is_some_and(|latest| same_attempt(input, latest))
    }
}

pub struct PhotoImageBoundary {
    input: watch::Sender<Option<Arc<PhotoSnapshot>>>,
    output: mpsc::Receiver<PhotoImageResult>,
    prepared: HashSet<(u64, PreparationStage)>,
    original: watch::Sender<Option<Arc<DecodedOriginal>>>,
}

impl PhotoImageBoundary {
    pub fn channel() -> (Self, PhotoImageWorker) {
        let (input, receiver) = watch::channel(None);
        let (sender, output) = mpsc::channel(8);
        let (original, original_receiver) = watch::channel(None);
        (
            Self {
                input,
                output,
                prepared: HashSet::new(),
                original,
            },
            PhotoImageWorker {
                input: receiver,
                output: sender,
                original: original_receiver,
            },
        )
    }

    pub fn retain_original(
        &mut self,
        photo_id: String,
        attempt_id: u64,
        path: String,
        pixels: Arc<image::RgbImage>,
    ) {
        self.original.send_replace(Some(Arc::new(DecodedOriginal {
            photo_id,
            attempt_id,
            path,
            pixels,
        })));
    }

    pub fn publish(&mut self, snapshot: &PhotoSnapshot) {
        let retained = self.original.borrow().clone();
        if let Some(original) = retained {
            match snapshot.photo.as_ref().filter(|photo| {
                photo.photo_id == original.photo_id
                    && photo
                        .image
                        .as_ref()
                        .map_or(photo.attempt_id == original.attempt_id, |image| {
                            image.path == original.path
                        })
            }) {
                Some(photo) if photo.attempt_id != original.attempt_id => {
                    self.retain_original(
                        original.photo_id.clone(),
                        photo.attempt_id,
                        original.path.clone(),
                        original.pixels.clone(),
                    );
                }
                Some(_) => {}
                None => {
                    self.original.send_replace(None);
                }
            }
        }
        if !self
            .input
            .borrow()
            .as_deref()
            .is_some_and(|previous| same_attempt(previous, snapshot))
        {
            self.prepared.clear();
        }
        self.input.send_replace(Some(Arc::new(snapshot.clone())));
    }

    pub async fn next_result(&mut self) -> Option<PhotoImageResult> {
        self.output.recv().await
    }

    pub fn accept(&mut self, result: &PhotoImageResult, current: &PhotoSnapshot) -> bool {
        if result.input.revision > current.revision || !same_attempt(&result.input, current) {
            return false;
        }
        let Some(photo) = current.photo.as_deref() else {
            return false;
        };
        let Some(input_photo) = result.input.photo.as_deref() else {
            return false;
        };
        match &result.output {
            PhotoImageOutput::Prepared {
                block_id,
                stage,
                elapsed_ms,
            } => {
                if !elapsed_ms.is_finite()
                    || *elapsed_ms < 0.0
                    || !has_stage(photo, *block_id, *stage)
                    || !has_stage(input_photo, *block_id, *stage)
                {
                    return false;
                }
                self.prepared.insert((*block_id, *stage))
            }
            PhotoImageOutput::Composed {
                block_ids,
                elapsed_ms,
                ..
            } => {
                elapsed_ms.is_finite()
                    && *elapsed_ms >= 0.0
                    && block_ids.len() <= input_photo.blocks.len()
                    && block_ids.len() <= photo.blocks.len()
                    && block_ids
                        .iter()
                        .enumerate()
                        .all(|(index, id)| photo.blocks[index].block_id == *id)
            }
        }
    }

    pub fn close(&mut self) {
        self.input.send_replace(None);
        self.original.send_replace(None);
        self.output.close();
    }
}

fn has_stage(photo: &Photo, id: u64, stage: PreparationStage) -> bool {
    photo.preparations.iter().any(|preparation| {
        preparation.block_id == id
            && (stage == PreparationStage::Bbox || preparation.source.is_some())
    }) || photo.blocks.iter().any(|block| block.block_id == id)
}

fn same_attempt(left: &PhotoSnapshot, right: &PhotoSnapshot) -> bool {
    match (&left.photo, &right.photo) {
        (Some(left), Some(right)) => {
            left.attempt_id > 0
                && left.photo_id == right.photo_id
                && left.attempt_id == right.attempt_id
                && left.image.is_some()
                && left.image == right.image
                && !matches!(right.phase, super::state::PhotoPhase::Expired)
        }
        _ => false,
    }
}
