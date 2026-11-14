use std::collections::HashSet;
use std::sync::{Arc, Mutex};
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

#[derive(serde::Serialize)]
pub struct PhotoFont {
    pub family: String,
    pub source: String,
    pub hangul_ink_ratio: f32,
}

pub enum PhotoImageOutput {
    Timing {
        block_id: u64,
        stage: &'static str,
        elapsed_ms: f64,
    },
    FontReady {
        font: PhotoFont,
    },
    Candidate {
        block_id: u64,
        rect: super::processing::PixelRect,
        elapsed_ms: f64,
    },
    Prepared {
        block_id: u64,
        stage: PreparationStage,
        elapsed_ms: f64,
    },
    Original {
        working: Arc<super::processing::WorkingPhoto>,
        recovered: bool,
    },
    Block {
        block_id: u64,
        region: Option<super::processing::PreparedRegion>,
        #[cfg(windows)]
        typeset: Option<super::typeset::TypesetBlock>,
        typeset_ms: f64,
    },
    Failed {
        code: &'static str,
    },
}

pub struct PhotoImageResult {
    pub input: Arc<PhotoSnapshot>,
    pub output: PhotoImageOutput,
}

pub enum SourcePixels {
    Original(Arc<image::RgbImage>),
    Working(Arc<super::processing::WorkingPhoto>),
}

pub struct DecodedOriginal {
    pub photo_id: String,
    pub attempt_id: u64,
    pub path: String,
    pub pixels: SourcePixels,
}

pub struct PhotoImageWorker {
    input: watch::Receiver<Option<Arc<PhotoSnapshot>>>,
    output: mpsc::Sender<PhotoImageResult>,
    original: Arc<Mutex<Option<DecodedOriginal>>>,
}

impl PhotoImageWorker {
    pub fn take_original(&self, input: &PhotoSnapshot) -> Option<DecodedOriginal> {
        let mut retained = self.original.lock().unwrap();
        let original = retained.as_ref()?;
        let matches = input.photo.as_ref().is_some_and(|photo| {
            photo.photo_id == original.photo_id
                && photo.attempt_id == original.attempt_id
                && photo.image.as_ref().is_some_and(|image| image.path == original.path)
        });
        if matches { retained.take() } else { None }
    }

    pub async fn next_input(&mut self) -> Option<Arc<PhotoSnapshot>> {
        self.input.changed().await.ok()?;
        self.input.borrow_and_update().clone()
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

    pub(crate) fn complete_blocking(&self, result: PhotoImageResult) -> bool {
        self.output.blocking_send(result).is_ok()
    }
}

pub struct PhotoImageBoundary {
    input: watch::Sender<Option<Arc<PhotoSnapshot>>>,
    output: mpsc::Receiver<PhotoImageResult>,
    prepared: HashSet<(u64, PreparationStage)>,
    original: Arc<Mutex<Option<DecodedOriginal>>>,
}

impl PhotoImageBoundary {
    pub fn channel() -> (Self, PhotoImageWorker) {
        let (input, receiver) = watch::channel(None);
        let (sender, output) = mpsc::channel(8);
        let original = Arc::new(Mutex::new(None));
        (
            Self {
                input,
                output,
                prepared: HashSet::new(),
                original: original.clone(),
            },
            PhotoImageWorker {
                input: receiver,
                output: sender,
                original,
            },
        )
    }

    pub fn retain_original(
        &mut self,
        photo_id: String,
        attempt_id: u64,
        path: String,
        pixels: Arc<image::RgbImage>,
        working: Option<Arc<super::processing::WorkingPhoto>>,
    ) {
        *self.original.lock().unwrap() = Some(DecodedOriginal {
            photo_id,
            attempt_id,
            path,
            pixels: match working {
                Some(working) => SourcePixels::Working(working),
                None => SourcePixels::Original(pixels),
            },
        });
    }

    pub fn publish(&mut self, snapshot: &PhotoSnapshot) {
        {
            let mut retained = self.original.lock().unwrap();
            if let Some(original) = retained.as_mut() {
                match snapshot.photo.as_ref().filter(|photo| {
                    photo.photo_id == original.photo_id
                        && photo.phase != super::state::PhotoPhase::Expired
                        && photo.image.as_ref().map_or(
                            photo.path == original.path || photo.attempt_id == original.attempt_id,
                            |image| image.path == original.path,
                        )
                }) {
                    Some(photo) => original.attempt_id = photo.attempt_id,
                    None => *retained = None,
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
        if let Some(photo) = &snapshot.photo {
            self.prepared.retain(|(id, _)| !photo.excluded_blocks.contains(id));
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
            PhotoImageOutput::Timing {
                block_id,
                stage,
                elapsed_ms,
            } => {
                elapsed_ms.is_finite()
                    && *elapsed_ms >= 0.0
                    && matches!(*stage, "mask" | "erase" | "typeset")
                    && has_stage(photo, *block_id, PreparationStage::Source)
                    && has_stage(input_photo, *block_id, PreparationStage::Source)
            }
            PhotoImageOutput::Candidate {
                block_id,
                elapsed_ms,
                ..
            } => {
                elapsed_ms.is_finite()
                    && *elapsed_ms >= 0.0
                    && has_stage(photo, *block_id, PreparationStage::Bbox)
                    && has_stage(input_photo, *block_id, PreparationStage::Bbox)
                    && self.prepared.insert((*block_id, PreparationStage::Bbox))
            }
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
            PhotoImageOutput::Original { .. }
            | PhotoImageOutput::Failed { .. }
            | PhotoImageOutput::FontReady { .. } => true,
            PhotoImageOutput::Block { block_id, .. } => {
                input_photo
                    .blocks
                    .iter()
                    .any(|block| block.block_id == *block_id)
                    && photo.blocks.iter().any(|block| block.block_id == *block_id)
            }
        }
    }

    pub fn close(&mut self) {
        self.input.send_replace(None);
        *self.original.lock().unwrap() = None;
        self.output.close();
    }
}

fn has_stage(photo: &Photo, id: u64, stage: PreparationStage) -> bool {
    photo.preparations.iter().any(|preparation| {
        preparation.block_id == id
            && (stage == PreparationStage::Bbox || preparation.source.is_some())
    }) || photo.blocks.iter().any(|block| block.block_id == id)
}

pub(crate) fn same_attempt(left: &PhotoSnapshot, right: &PhotoSnapshot) -> bool {
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
