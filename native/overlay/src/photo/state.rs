use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use super::manifest::PhotoError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhotoPhase {
    Offer,
    Preparing,
    Requesting,
    Completed,
    Failed,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhotoImage {
    pub path: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhotoPreparation {
    pub block_id: u64,
    pub bbox: [f64; 4],
    pub source: Option<String>,
    pub source_line_count: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhotoBlock {
    pub block_id: u64,
    pub bbox: [f64; 4],
    pub source: String,
    pub translation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Photo {
    pub photo_id: String,
    pub path: String,
    pub file_size: u64,
    pub shutter_unix_ms: u64,
    pub attempt_id: u64,
    pub phase: PhotoPhase,
    pub offer_open: bool,
    pub visible: bool,
    pub image: Option<PhotoImage>,
    pub preparations: Vec<PhotoPreparation>,
    pub blocks: Vec<PhotoBlock>,
    pub failure_code: Option<String>,
    pub first_block_drawn: bool,
}

impl Photo {
    pub fn can_pull(&self) -> bool {
        self.offer_open
            || (self.attempt_id > 0
                && matches!(
                    self.phase,
                    PhotoPhase::Preparing | PhotoPhase::Requesting | PhotoPhase::Completed
                ))
    }

    fn validate(&self) -> Result<(), PhotoError> {
        if self.photo_id.is_empty()
            || self.photo_id.len() > 256
            || self.attempt_id > 2
            || self.path.is_empty()
            || self.blocks.len() > 4096
            || self.preparations.len() > 4096
            || (self.attempt_id == 0
                && (self.image.is_some()
                    || !self.blocks.is_empty()
                    || !self.preparations.is_empty()
                    || self.first_block_drawn))
            || (self.phase == PhotoPhase::Expired && (self.offer_open || self.visible))
        {
            return Err(PhotoError::Protocol);
        }
        if let Some(image) = &self.image {
            if image.path.is_empty() || image.width == 0 || image.height == 0 {
                return Err(PhotoError::Protocol);
            }
        } else if !self.blocks.is_empty() || !self.preparations.is_empty() {
            return Err(PhotoError::Protocol);
        }
        let mut ids = HashSet::new();
        for preparation in &self.preparations {
            validate_bbox(&preparation.bbox, self.image.as_ref())?;
            if !ids.insert(preparation.block_id)
                || match (&preparation.source, preparation.source_line_count) {
                    (None, None) => false,
                    (Some(source), Some(count)) => count as usize != source.split('\n').count(),
                    _ => true,
                }
            {
                return Err(PhotoError::Protocol);
            }
        }
        ids.clear();
        for block in &self.blocks {
            validate_bbox(&block.bbox, self.image.as_ref())?;
            if !ids.insert(block.block_id) {
                return Err(PhotoError::Protocol);
            }
            if let Some(preparation) = self
                .preparations
                .iter()
                .find(|preparation| preparation.block_id == block.block_id)
            {
                if preparation.bbox != block.bbox
                    || preparation
                        .source
                        .as_ref()
                        .is_some_and(|source| source != &block.source)
                {
                    return Err(PhotoError::Protocol);
                }
            }
        }
        Ok(())
    }
}

fn validate_bbox(bbox: &[f64; 4], image: Option<&PhotoImage>) -> Result<(), PhotoError> {
    let Some(image) = image else {
        return Err(PhotoError::Protocol);
    };
    if bbox.iter().any(|value| !value.is_finite())
        || bbox[0] < 0.0
        || bbox[1] < 0.0
        || bbox[2] <= bbox[0]
        || bbox[3] <= bbox[1]
        || bbox[2] > f64::from(image.width)
        || bbox[3] > f64::from(image.height)
    {
        return Err(PhotoError::Protocol);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhotoSnapshot {
    pub revision: u64,
    pub photo: Option<Arc<Photo>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotAdmission {
    Applied,
    Stale,
}

#[derive(Debug, Default)]
pub struct PhotoState {
    snapshot: Option<PhotoSnapshot>,
    retired_photos: HashSet<String>,
}

impl PhotoState {
    pub fn snapshot(&self) -> Option<&PhotoSnapshot> {
        self.snapshot.as_ref()
    }

    pub fn photo(&self) -> Option<&Photo> {
        self.snapshot.as_ref()?.photo.as_deref()
    }

    pub fn apply(
        &mut self,
        revision: u64,
        photo: Option<Photo>,
    ) -> Result<SnapshotAdmission, PhotoError> {
        if self
            .snapshot
            .as_ref()
            .is_some_and(|current| revision <= current.revision)
        {
            return Ok(SnapshotAdmission::Stale);
        }
        if let Some(next) = &photo {
            next.validate()?;
            if self.retired_photos.contains(&next.photo_id) {
                return Ok(SnapshotAdmission::Stale);
            }
            if let Some(current) = self
                .photo()
                .filter(|current| current.photo_id == next.photo_id)
            {
                if next.attempt_id < current.attempt_id {
                    return Ok(SnapshotAdmission::Stale);
                }
                if current.path != next.path || current.shutter_unix_ms != next.shutter_unix_ms {
                    return Err(PhotoError::Protocol);
                }
                if next.attempt_id == current.attempt_id {
                    validate_cumulative(current, next)?;
                }
            }
        }
        if let Some(current) = self.photo() {
            if photo.as_ref().map(|next| &next.photo_id) != Some(&current.photo_id) {
                self.retired_photos.insert(current.photo_id.clone());
            }
        }
        self.snapshot = Some(PhotoSnapshot {
            revision,
            photo: photo.map(Arc::new),
        });
        Ok(SnapshotAdmission::Applied)
    }
}

fn validate_cumulative(current: &Photo, next: &Photo) -> Result<(), PhotoError> {
    if !next.blocks.starts_with(&current.blocks)
        || (current.first_block_drawn && !next.first_block_drawn)
        || (current.image.is_some() && current.image != next.image)
        || (matches!(current.phase, PhotoPhase::Expired) && next.phase != PhotoPhase::Expired)
    {
        return Err(PhotoError::Protocol);
    }
    for old in &current.preparations {
        let Some(new) = next
            .preparations
            .iter()
            .find(|new| new.block_id == old.block_id)
        else {
            return Err(PhotoError::Protocol);
        };
        if old.bbox != new.bbox
            || (old.source.is_some()
                && (old.source != new.source || old.source_line_count != new.source_line_count))
        {
            return Err(PhotoError::Protocol);
        }
    }
    Ok(())
}
