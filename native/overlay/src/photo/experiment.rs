use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{codecs::jpeg::JpegEncoder, codecs::png::PngEncoder, ImageEncoder, Rgb, RgbImage};
use serde::Deserialize;
use serde_json::{json, Value};

use super::prepare::{decode_source, send_canvas};
use super::processing::{PixelRect, WorkingPhoto};

type Error = Box<dyn std::error::Error>;
pub(super) type QuadRegistry = HashMap<u64, Quad>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrepareJob {
    source_path: PathBuf,
    output_dir: PathBuf,
}

pub(super) fn prepare(path: &Path) -> Result<(), Error> {
    let job: PrepareJob = serde_json::from_slice(&std::fs::read(path)?)?;
    std::fs::create_dir_all(&job.output_dir)?;
    let start = Instant::now();
    let read_start = Instant::now();
    let bytes = std::fs::read(&job.source_path)?;
    let source_read_ms = read_start.elapsed().as_secs_f64() * 1000.0;
    let decode_start = Instant::now();
    let original = decode_source(&bytes).map_err(std::io::Error::other)?;
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let resize_start = Instant::now();
    let canvas = send_canvas(&original);
    let resize_ms = resize_start.elapsed().as_secs_f64() * 1000.0;
    let mut encodings = json!({});
    for (name, filename, quality) in [
        ("jpeg85", "send-jpeg85.jpg", Some(85)),
        ("jpeg95", "send-jpeg95.jpg", Some(95)),
        ("png", "send-png.png", None),
    ] {
        let output = job.output_dir.join(filename);
        let encode_start = Instant::now();
        let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(&output)?);
        if let Some(quality) = quality {
            JpegEncoder::new_with_quality(&mut writer, quality).encode_image(canvas.pixels.as_ref())?;
        } else {
            PngEncoder::new(&mut writer).write_image(
                canvas.pixels.as_raw(), canvas.pixels.width(), canvas.pixels.height(),
                image::ExtendedColorType::Rgb8,
            )?;
        }
        writer.flush()?;
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
        let mut record = json!({
            "path":output,"mime":if quality.is_some() { "image/jpeg" } else { "image/png" },
            "bytes":writer.get_ref().metadata()?.len(),"encode_ms":encode_ms,
            "preparation_ms":source_read_ms+decode_ms+resize_ms+encode_ms,
        });
        if let Some(quality) = quality {
            record["quality"] = json!(quality);
            record["sampling"] = json!("4:4:4");
        } else {
            record["color_type"] = json!("RGB8");
        }
        encodings[name] = record;
    }
    let metadata = json!({
        "source_path":job.source_path,
        "source_dimensions":[original.width(),original.height()],
        "send_dimensions":[canvas.pixels.width(),canvas.pixels.height()],
        "scale":canvas.scale,"offset_x":canvas.offset_x,"offset_y":canvas.offset_y,
        "source_read_ms":source_read_ms,"decode_ms":decode_ms,"resize_ms":resize_ms,
        "preparation_wall_ms":start.elapsed().as_secs_f64()*1000.0,"encodings":encodings,
    });
    serde_json::to_writer_pretty(File::create(job.output_dir.join("preparation.json"))?, &metadata)?;
    println!("{}", json!({"type":"photo_experiment_prepared","output_dir":job.output_dir}));
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Quad {
    pub corners: [[f64; 2]; 4],
}

impl Quad {
    pub fn new(corners: [[f64; 2]; 4], dimensions: [u32; 2]) -> Result<Self, &'static str> {
        if !corners.iter().flatten().all(|v| v.is_finite()) {
            return Err("quad_nonfinite");
        }
        if corners.iter().any(|p| p[0] < 0.0 || p[1] < 0.0
            || p[0] > f64::from(dimensions[0]) || p[1] > f64::from(dimensions[1])) {
            return Err("quad_outside_source");
        }
        for i in 0..4 {
            let [a, b, c] = [corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]];
            if cross(a, b, c) <= 1e-6 {
                return Err("quad_not_convex_clockwise");
            }
        }
        if corners[0][1] + corners[1][1] >= corners[2][1] + corners[3][1]
            || corners[0][0] + corners[3][0] >= corners[1][0] + corners[2][0] {
            return Err("quad_not_tl_tr_br_bl");
        }
        Ok(Self { corners })
    }

    pub fn mapped(self, photo: &WorkingPhoto) -> Self {
        Self { corners: self.corners.map(|p| [p[0] * f64::from(photo.scale_x), p[1] * f64::from(photo.scale_y)]) }
    }

    pub fn contains(self, x: f64, y: f64) -> bool {
        (0..4).all(|i| {
            let a = self.corners[i];
            let b = self.corners[(i + 1) % 4];
            let side = cross(a, b, [x, y]);
            side > 0.0 || (side == 0.0 && (b[1] < a[1] || (b[1] == a[1] && b[0] > a[0])))
        })
    }

    pub fn outside(self, roi: PixelRect) -> bool {
        self.corners.iter().any(|p| p[0] < f64::from(roi.x) || p[1] < f64::from(roi.y)
            || p[0] > f64::from(roi.x + roi.width) || p[1] > f64::from(roi.y + roi.height))
    }
}

fn cross(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}
pub(super) fn load_quads(
    path: &Path, dimensions: [u32; 2], photo: &RgbImage,
    block_ids: &HashSet<u64>, output_dir: &Path,
) -> Result<QuadRegistry, Error> {
    let bytes = std::fs::read(path)?;
    let raw: HashMap<u64, [[f64; 2]; 4]> = match serde_json::from_slice(&bytes) {
        Ok(raw) => raw,
        Err(error) => {
            std::fs::write(output_dir.join("quad-diagnostics.json"), serde_json::to_vec_pretty(&json!({
                "coordinate_space":"original_photo","error":error.to_string(),"sidecar_path":path,
            }))?)?;
            return Err(error.into());
        }
    };
    let mut registry = QuadRegistry::new();
    let mut diagnostics = json!({"coordinate_space":"original_photo","order":["TL","TR","BR","BL"],"blocks":{},"missing_block_ids":[]});
    let mut overlay = photo.clone();
    let mut invalid = false;
    for (&id, &corners) in &raw {
        let result = Quad::new(corners, dimensions);
        let error = if !block_ids.contains(&id) { Some("quad_unknown_block") } else { result.as_ref().err().copied() };
        diagnostics["blocks"][id.to_string()] = json!({"corners":corners,"error":error});
        draw_quad(&mut overlay, corners);
        if error.is_some() {
            invalid = true;
        } else {
            registry.insert(id, result?);
        }
    }
    diagnostics["missing_block_ids"] = json!(block_ids.iter().filter(|id| !raw.contains_key(id)).collect::<Vec<_>>());
    std::fs::write(output_dir.join("quad-diagnostics.json"), serde_json::to_vec_pretty(&diagnostics)?)?;
    overlay.save(output_dir.join("quad-overlay.png"))?;
    if invalid { return Err("invalid quad sidecar; see quad-diagnostics.json".into()); }
    Ok(registry)
}

fn draw_quad(image: &mut RgbImage, corners: [[f64; 2]; 4]) {
    for i in 0..4 {
        let clamp = |p: [f64; 2]| [p[0].clamp(0.0,f64::from(image.width()-1)),p[1].clamp(0.0,f64::from(image.height()-1))];
        let a = clamp(corners[i]);
        let b = clamp(corners[(i+1)%4]);
        let steps = (b[0]-a[0]).abs().max((b[1]-a[1]).abs()).ceil().max(1.0) as u32;
        let color = [Rgb([255,0,0]),Rgb([0,255,0]),Rgb([0,128,255]),Rgb([255,255,0])][i];
        for step in 0..=steps {
            let t = f64::from(step)/f64::from(steps);
            image.put_pixel((a[0]+(b[0]-a[0])*t).round() as u32,(a[1]+(b[1]-a[1])*t).round() as u32,color);
        }
    }
}

#[cfg(windows)]
pub(super) fn block_evidence(
    region: &super::processing::PreparedRegion, quad: Option<Quad>,
    typeset: Option<&super::typeset::TypesetBlock>,
) -> Value {
    json!({
        "ownership":if quad.is_some() { "quad_core_and_owned_line_counts" } else { "aabb" },
        "working_corners":quad.map(|q| q.corners),"quad_outside_padded_roi":quad.map(|q| q.outside(region.roi)),
        "core_pixels":region.core.as_raw().iter().filter(|&&v| v!=0).count(),
        "recovered_pixels":region.recovered.as_raw().iter().filter(|&&v| v!=0).count(),
        "source_line_count":region.source_lines,
        "overlap_area":typeset.map(|t| t.overlap_area),
        "runs":typeset.map(|t| t.runs.iter().map(|r| json!({
            "text":r.text,"size":r.size,"origin":r.origin,
            "ink":[r.ink.left,r.ink.top,r.ink.right,r.ink.bottom],
        })).collect::<Vec<_>>()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn ordered_quads_reject_malformed_corners_and_keep_image_edges() {
        let valid = [[0.0,0.0],[100.0,0.0],[100.0,50.0],[0.0,50.0]];
        let quad = Quad::new(valid,[100,50]).unwrap();
        assert!(quad.contains(0.0,0.0));
        assert!(!quad.contains(100.0,25.0));
        assert!(!quad.contains(50.0,50.0));
        for corners in [
            [[0.0,0.0],[100.0,50.0],[100.0,0.0],[0.0,50.0]],
            [[100.0,0.0],[100.0,50.0],[0.0,50.0],[0.0,0.0]],
            [[0.0,0.0],[0.0,50.0],[100.0,50.0],[100.0,0.0]],
            [[0.0,0.0],[0.0,0.0],[100.0,50.0],[0.0,50.0]],
            [[0.0,0.0],[100.0,0.0],[20.0,10.0],[0.0,50.0]],
            [[-1.0,0.0],[100.0,0.0],[100.0,50.0],[0.0,50.0]],
            [[f64::NAN,0.0],[100.0,0.0],[100.0,50.0],[0.0,50.0]],
        ] { assert!(Quad::new(corners,[100,50]).is_err()); }
    }

    #[test]
    fn original_quads_scale_to_working_pixels_without_send_offsets() {
        let photo = WorkingPhoto::new(Arc::new(RgbImage::new(400,100)),[200,100]).unwrap();
        let quad = Quad::new([[20.0,10.0],[200.0,10.0],[190.0,80.0],[30.0,80.0]],[400,100]).unwrap().mapped(&photo);
        assert_eq!(quad.corners[0],[10.0,5.0]);
        assert!(quad.contains(50.5,20.5));
        assert!(!quad.contains(10.5,39.5));
        assert!(quad.outside(PixelRect { x:20,y:0,width:100,height:50 }));
    }

    #[test]
    fn production_canvas_is_shared_and_png_roundtrips_rgb() {
        let original = RgbImage::from_fn(853,601,|x,y| Rgb([(x%251) as u8,(y%241) as u8,37]));
        let canvas = send_canvas(&original);
        assert_eq!(canvas.pixels.dimensions(),(1280,720));
        assert_eq!((canvas.offset_x,canvas.offset_y),(129,0));
        assert_eq!(canvas.scale,720.0/601.0);
        assert_eq!(*canvas.pixels.get_pixel(0,0),Rgb([16,18,24]));
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes).write_image(canvas.pixels.as_raw(),1280,720,image::ExtendedColorType::Rgb8).unwrap();
        assert_eq!(bytes[25],2);
        assert_eq!(decode_source(&bytes).unwrap(),*canvas.pixels);
        let exact = RgbImage::new(1280,720);
        assert!(matches!(send_canvas(&exact).pixels,std::borrow::Cow::Borrowed(_)));
    }
}

#[cfg(test)]
mod sidecar_tests {
    use super::*;

    #[test]
    fn sidecars_retain_rejected_corner_diagnostics_and_report_missing_blocks() {
        let root = std::env::temp_dir().join(format!("puri438-quads-{}",std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let sidecar = root.join("quads.json");
        let photo = RgbImage::new(20,20);
        let ids = HashSet::from([7,8]);
        std::fs::write(&sidecar,br#"{"7":[[0,0],[20,0],[20,20],[0,20]]}"#).unwrap();
        let registry = load_quads(&sidecar,[20,20],&photo,&ids,&root).unwrap();
        assert_eq!(registry.len(),1);
        let diagnostics: Value = serde_json::from_slice(&std::fs::read(root.join("quad-diagnostics.json")).unwrap()).unwrap();
        assert_eq!(diagnostics["missing_block_ids"],json!([8]));
        assert!(root.join("quad-overlay.png").is_file());
        std::fs::write(&sidecar,br#"{"7":[[0,0],[20,0],[5,5],[0,20]]}"#).unwrap();
        assert!(load_quads(&sidecar,[20,20],&photo,&ids,&root).is_err());
        let diagnostics: Value = serde_json::from_slice(&std::fs::read(root.join("quad-diagnostics.json")).unwrap()).unwrap();
        assert_eq!(diagnostics["blocks"]["7"]["error"],"quad_not_convex_clockwise");
        assert_eq!(diagnostics["blocks"]["7"]["corners"][2],json!([5.0,5.0]));
        std::fs::write(&sidecar,br#"{"7":[[0,0],[20,0],[20,20]]}"#).unwrap();
        assert!(load_quads(&sidecar,[20,20],&photo,&ids,&root).is_err());
        let diagnostics: Value = serde_json::from_slice(&std::fs::read(root.join("quad-diagnostics.json")).unwrap()).unwrap();
        assert!(diagnostics["error"].is_string());
        std::fs::remove_dir_all(root).unwrap();
    }
}
