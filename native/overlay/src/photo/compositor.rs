use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use windows::core::Interface;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct2D::{
    Common::{
        D2D1_ALPHA_MODE_PREMULTIPLIED, D2D1_COLOR_F, D2D1_COMPOSITE_MODE_SOURCE_OVER,
        D2D1_PIXEL_FORMAT, D2D_RECT_F, D2D_SIZE_U,
    },
    D2D1CreateFactory, ID2D1Bitmap1, ID2D1CommandList, ID2D1DeviceContext, ID2D1Factory1,
    ID2D1Layer, ID2D1SolidColorBrush, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE,
    D2D1_BITMAP_OPTIONS_CANNOT_DRAW, D2D1_BITMAP_OPTIONS_NONE, D2D1_BITMAP_OPTIONS_TARGET,
    D2D1_BITMAP_PROPERTIES1, D2D1_DEVICE_CONTEXT_OPTIONS_NONE, D2D1_DRAW_TEXT_OPTIONS_NONE,
    D2D1_FACTORY_TYPE_SINGLE_THREADED, D2D1_INTERPOLATION_MODE_LINEAR, D2D1_LAYER_OPTIONS1_NONE,
    D2D1_LAYER_PARAMETERS1, D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE,
};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0,
    D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::{
    Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
    IDXGIDevice, IDXGISurface,
};
use windows_core::BOOL;
use windows::core::w;
use windows::Win32::Graphics::DirectWrite::{
    DWriteCreateFactory, IDWriteFactory, IDWriteFontCollection, DWRITE_FACTORY_TYPE_SHARED,
    DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT_MEDIUM,
    DWRITE_TEXT_METRICS, DWRITE_WORD_WRAPPING_NO_WRAP,
};
use windows_numerics::{Matrix3x2, Vector2};

use super::gpu::{query, GpuTimer};
use super::image::PhotoTexture;
use super::manifest::PhotoError;
use super::processing::{PixelRect, PreparedRegion, Rect, WorkingPhoto};
use super::typeset::TypesetBlock;
use crate::openvr::{OpenVrOutputAdapter, OverlayTextureSubmitter};
use crate::presentation::AdapterIdentity;
use crate::renderer::glyph_run::render_text_layout_to_command_list;

pub const BLOCK_FADE: Duration = Duration::from_millis(100);

struct RunVisual {
    commands: ID2D1CommandList,
    origin: [f32; 2],
}

struct BlockVisual {
    id: u64,
    erase: Option<(ID2D1Bitmap1, PixelRect)>,
    runs: Vec<RunVisual>,
    typeset: Option<TypesetBlock>,
    start: Instant,
    metadata: Value,
    roi_timer: Option<GpuTimer>,
    plate: Option<(ID2D1SolidColorBrush, Rect)>,
}

pub struct PhotoCompositor {
    device: ID3D11Device,
    d3d: ID3D11DeviceContext,
    factory: ID2D1Factory1,
    d2d: ID2D1DeviceContext,
    opacity_layer: ID2D1Layer,
    working: Option<Arc<WorkingPhoto>>,
    original: Option<ID2D1Bitmap1>,
    target: Option<ID2D1Bitmap1>,
    texture: Option<ID3D11Texture2D>,
    blocks: Vec<BlockVisual>,
    pub original_uploads: u64,
    pub adapter: AdapterIdentity,
    dirty: bool,
    fade_until: Option<Instant>,
    readiness: ID3D11Query,
    frame_timer: GpuTimer,
    pub frame_gpu_ms: Option<f64>,
    font: Option<super::image::PhotoFont>,
    timings: HashMap<u64, [f64; 3]>,
    frame_timing_started: bool,
}

impl PhotoCompositor {
    pub fn new(output: Option<&OpenVrOutputAdapter>) -> Result<Self, PhotoError> {
        let mut device = None;
        let mut d3d = None;
        let adapter = output.and_then(OpenVrOutputAdapter::adapter);
        unsafe {
            D3D11CreateDevice(
                adapter,
                if adapter.is_some() {
                    D3D_DRIVER_TYPE_UNKNOWN
                } else {
                    D3D_DRIVER_TYPE_HARDWARE
                },
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut d3d),
            )
            .map_err(|_| PhotoError::Graphics)?;
        }
        let device = device.ok_or(PhotoError::Graphics)?;
        let d3d = d3d.ok_or(PhotoError::Graphics)?;
        let dxgi: IDXGIDevice = device.cast().map_err(|_| PhotoError::Graphics)?;
        let desc = unsafe { dxgi.GetAdapter().and_then(|adapter| adapter.GetDesc()) }
            .map_err(|_| PhotoError::Graphics)?;
        let identity = AdapterIdentity::DxgiLuid {
            high: desc.AdapterLuid.HighPart,
            low: desc.AdapterLuid.LowPart,
        };
        if let Some(output) = output {
            if output.identity() != identity {
                return Err(PhotoError::Graphics);
            }
        }
        let factory: ID2D1Factory1 =
            unsafe { D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None) }
                .map_err(|_| PhotoError::Graphics)?;
        let d2d = unsafe {
            factory
                .CreateDevice(&dxgi)
                .and_then(|device| device.CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE))
        }
        .map_err(|_| PhotoError::Graphics)?;
        let opacity_layer = unsafe { d2d.CreateLayer(None) }.map_err(|_| PhotoError::Graphics)?;
        unsafe {
            d2d.SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
        }
        let readiness = query(&device, D3D11_QUERY_EVENT)?;
        let frame_timer = GpuTimer::new(&device)?;
        Ok(Self {
            device,
            d3d,
            factory,
            d2d,
            opacity_layer,
            working: None,
            original: None,
            target: None,
            texture: None,
            blocks: Vec::new(),
            original_uploads: 0,
            adapter: identity,
            dirty: false,
            fade_until: None,
            readiness,
            frame_timer,
            frame_gpu_ms: None,
            font: None,
            timings: HashMap::new(),
            frame_timing_started: false,
        })
    }

    pub fn icon_texture(&self, rgba: &[u8], size: u32) -> Result<PhotoTexture, PhotoError> {
        let texture = self.create_texture(size, size, false)?;
        let bgra: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|p| [p[2], p[1], p[0], p[3]])
            .collect();
        unsafe {
            self.d3d
                .UpdateSubresource(&texture, 0, None, bgra.as_ptr().cast(), size * 4, 0);
        }
        Ok(PhotoTexture::new(texture))
    }

    pub fn case_label_texture(&self, label: &str, draining: bool) -> Result<PhotoTexture, PhotoError> {
        use super::interaction::CASE_LABEL_SIZE;
        let factory: IDWriteFactory = unsafe { DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED) }
            .map_err(|_| PhotoError::Graphics)?;
        let layout = |text: &str, size: f32| {
            let format = unsafe {
                factory.CreateTextFormat(
                    w!("Segoe UI"), None::<&IDWriteFontCollection>, DWRITE_FONT_WEIGHT_MEDIUM,
                    DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, size, w!("en-US"),
                )
            }.map_err(|_| PhotoError::Graphics)?;
            unsafe { format.SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP) }
                .map_err(|_| PhotoError::Graphics)?;
            unsafe { factory.CreateTextLayout(&text.encode_utf16().collect::<Vec<_>>(), &format, 8192.0, 256.0) }
                .map_err(|_| PhotoError::Graphics)
        };
        let combined = draining.then(|| format!("{label}\n\nRELEASE TO CONTINUE"));
        let content = combined.as_deref().unwrap_or(label);
        let mut text = layout(content, 36.0)?;
        let mut metrics = DWRITE_TEXT_METRICS::default();
        unsafe { text.GetMetrics(&mut metrics) }.map_err(|_| PhotoError::Graphics)?;
        if metrics.widthIncludingTrailingWhitespace > 976.0 {
            text = layout(content, 36.0 * 976.0 / metrics.widthIncludingTrailingWhitespace)?;
        }
        let fill = unsafe { self.d2d.CreateSolidColorBrush(&color([240, 248, 255], 1.0), None) }
            .map_err(|_| PhotoError::Graphics)?;
        let outline = unsafe { self.d2d.CreateSolidColorBrush(&color([8, 12, 20], 1.0), None) }
            .map_err(|_| PhotoError::Graphics)?;
        let texture = self.create_texture(CASE_LABEL_SIZE[0], CASE_LABEL_SIZE[1], false)?;
        let surface: IDXGISurface = texture.cast().map_err(|_| PhotoError::Graphics)?;
        let target = unsafe {
            self.d2d.CreateBitmapFromDxgiSurface(&surface, Some(&bitmap_properties(
                D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
            )))
        }.map_err(|_| PhotoError::Graphics)?;
        let previous_target = unsafe { self.d2d.GetTarget().ok() };
        let mut previous_transform = Matrix3x2::identity();
        unsafe {
            self.d2d.GetTransform(&mut previous_transform);
            self.d2d.SetTransform(&Matrix3x2::identity());
        }
        let result = (|| {
            let commands = render_text_layout_to_command_list(
                &self.d2d, &self.factory, &text, &fill, &outline, 2.0,
            ).map_err(|_| PhotoError::Graphics)?;
            unsafe {
                self.d2d.SetTarget(&target);
                self.d2d.BeginDraw();
                self.d2d.Clear(Some(&color([0, 0, 0], 0.0)));
                self.d2d.DrawImage(&commands, Some(&Vector2 { X: 24.0, Y: 20.0 }), None,
                    D2D1_INTERPOLATION_MODE_LINEAR, D2D1_COMPOSITE_MODE_SOURCE_OVER);
                self.d2d.EndDraw(None, None).map_err(|_| PhotoError::Graphics)?;
            }
            Ok(PhotoTexture::new(texture))
        })();
        unsafe {
            self.d2d.SetTarget(previous_target.as_ref());
            self.d2d.SetTransform(&previous_transform);
        }
        result
    }

    pub fn submit_texture(
        &self,
        surface: &impl OverlayTextureSubmitter,
        texture: &PhotoTexture,
    ) -> Result<(), PhotoError> {
        surface
            .set_overlay_texture(texture.pointer())
            .map_err(|_| PhotoError::OpenVr)?;
        unsafe {
            self.d3d.Flush();
        }
        Ok(())
    }

    pub fn save_texture(
        &self,
        texture: &PhotoTexture,
        width: u32,
        height: u32,
        path: &Path,
    ) -> Result<(), PhotoError> {
        self.save_d3d_texture(texture.texture(), width, height, path)
    }

    pub fn set_original(&mut self, working: Arc<WorkingPhoto>) -> Result<(), PhotoError> {
        if self
            .working
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, &working))
        {
            return Ok(());
        }
        let (width, height) = working.pixels.dimensions();
        let mut bgra = Vec::with_capacity(width as usize * height as usize * 4);
        for pixel in working.pixels.pixels() {
            bgra.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
        }
        self.original = Some(self.upload(width, height, &bgra)?);
        let texture = self.create_texture(width, height, false)?;
        let surface: IDXGISurface = texture.cast().map_err(|_| PhotoError::Graphics)?;
        let target = unsafe {
            self.d2d.CreateBitmapFromDxgiSurface(
                &surface,
                Some(&bitmap_properties(
                    D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                )),
            )
        }
        .map_err(|_| PhotoError::Graphics)?;
        unsafe {
            self.d2d.SetTarget(&target);
        }
        self.target = Some(target);
        self.texture = Some(texture);
        self.working = Some(working);
        self.blocks.clear();
        self.fade_until = None;
        self.original_uploads += 1;
        self.dirty = true;
        Ok(())
    }

    pub fn clear_attempt(&mut self) {
        self.blocks.clear();
        self.font = None;
        self.timings.clear();
        self.fade_until = None;
        self.dirty = self.original.is_some();
    }

    pub fn clear_photo(&mut self) {
        unsafe {
            self.d2d
                .SetTarget(None::<&windows::Win32::Graphics::Direct2D::ID2D1Image>);
        }
        self.blocks.clear();
        self.original = None;
        self.target = None;
        self.texture = None;
        self.working = None;
        self.font = None;
        self.timings.clear();
        self.dirty = false;
        self.fade_until = None;
    }

    pub fn invalidate(&mut self) {
        self.dirty = self.original.is_some();
    }

    pub fn set_font(&mut self, font: super::image::PhotoFont) {
        self.font = Some(font);
    }

    pub fn add_block(
        &mut self,
        id: u64,
        region: Option<PreparedRegion>,
        typeset: Option<TypesetBlock>,
        typeset_ms: f64,
        now: Instant,
    ) -> Result<f64, PhotoError> {
        if self.blocks.iter().any(|block| block.id == id) {
            return Ok(0.0);
        }
        let upload_start = Instant::now();
        let mut roi_timer = None;
        let erase = region
            .as_ref()
            .and_then(|region| region.erase.as_ref())
            .map(|patch| {
                let mut bgra = Vec::with_capacity(patch.pixels.as_raw().len());
                for pixel in patch.pixels.pixels() {
                    let a = u16::from(pixel[3]);
                    bgra.extend_from_slice(&[
                        ((u16::from(pixel[2]) * a + 127) / 255) as u8,
                        ((u16::from(pixel[1]) * a + 127) / 255) as u8,
                        ((u16::from(pixel[0]) * a + 127) / 255) as u8,
                        pixel[3],
                    ]);
                }
                let timer = GpuTimer::new(&self.device)?;
                timer.begin(&self.d3d);
                let bitmap = self.upload(patch.rect.width, patch.rect.height, &bgra)?;
                timer.end(&self.d3d);
                roi_timer = Some(timer);
                Ok::<_, PhotoError>((bitmap, patch.rect))
            })
            .transpose()?;
        let roi_upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;
        let mut runs = Vec::new();
        if let Some(block) = &typeset {
            let fill = unsafe {
                self.d2d
                    .CreateSolidColorBrush(&color(block.style.fill, 1.0), None)
            }
            .map_err(|_| PhotoError::Graphics)?;
            for run in &block.runs {
                let commands = if let Some(outline) = &block.style.outline {
                    let outline_brush = unsafe {
                        self.d2d
                            .CreateSolidColorBrush(&color(outline.color, 1.0), None)
                    }
                    .map_err(|_| PhotoError::Graphics)?;
                    render_text_layout_to_command_list(
                        &self.d2d,
                        &self.factory,
                        &run.layout,
                        &fill,
                        &outline_brush,
                        outline.width * 2.0,
                    )
                    .map_err(|_| PhotoError::Graphics)?
                } else {
                    let list = unsafe { self.d2d.CreateCommandList() }
                        .map_err(|_| PhotoError::Graphics)?;
                    unsafe {
                        self.d2d.SetTarget(&list);
                        self.d2d.BeginDraw();
                        self.d2d.DrawTextLayout(
                            Vector2 { X: 0.0, Y: 0.0 },
                            &run.layout,
                            &fill,
                            D2D1_DRAW_TEXT_OPTIONS_NONE,
                        );
                        let result = self.d2d.EndDraw(None, None);
                        if let Some(target) = &self.target {
                            self.d2d.SetTarget(target);
                        }
                        result.map_err(|_| PhotoError::Graphics)?;
                        list.Close().map_err(|_| PhotoError::Graphics)?;
                    }
                    list
                };
                runs.push(RunVisual {
                    commands,
                    origin: run.origin,
                });
            }
        }
        let plate = typeset
            .as_ref()
            .and_then(|block| block.style.plate.as_ref())
            .map(|plate| {
                unsafe {
                    self.d2d
                        .CreateSolidColorBrush(&color(plate.color, plate.alpha), None)
                }
                .map(|brush| (brush, plate.rect))
                .map_err(|_| PhotoError::Graphics)
            })
            .transpose()?;
        let costs = self.timings.remove(&id);
        let metadata = json!({
            "block_id": id, "kept": region.is_none(), "plate": region.as_ref().is_some_and(|region| region.plate),
            "mask_ms": costs.map_or_else(||region.as_ref().map_or(0.0,|region|region.mask_ms),|cost|cost[0]),
            "erase_ms": costs.map_or_else(||region.as_ref().map_or(0.0,|region|region.erase_ms),|cost|cost[1]),
            "typeset_ms": costs.map_or(typeset_ms,|cost|cost[2]), "roi_upload_ms": roi_upload_ms,
            "roi_gpu_ms": null, "roi_gpu_timing_valid": false,
            "bbox": region.as_ref().map(|region| rect_array(region.bbox)),
            "roi": region.as_ref().map(|region| [region.roi.x, region.roi.y, region.roi.width, region.roi.height]),
            "collision": typeset.as_ref().is_some_and(|block| block.collision),
            "selected_size": typeset.as_ref().map(|block| block.selected_size), "source_cap": typeset.as_ref().map(|block| block.source_cap),
            "floor": typeset.as_ref().map(|block| block.floor), "kind": typeset.as_ref().map(|block| format!("{:?}", block.kind)),
            "source_vertical": typeset.as_ref().map(|block| block.source_vertical), "angle_degrees": typeset.as_ref().map(|block| block.angle_degrees),
            "region_overflow": typeset.as_ref().is_some_and(|block| block.region_overflow),
        });
        println!(
            "{}",
            json!({
                "type":"photo_image_stage","stage":"block_ready","block_id":id,
                "mask_ms":metadata["mask_ms"],"erase_ms":metadata["erase_ms"],
                "typeset_ms":typeset_ms,"roi_upload_ms":roi_upload_ms,
                "kept":metadata["kept"],"plate":metadata["plate"],"collision":metadata["collision"],
            })
        );
        if erase.is_some() || plate.is_some() || !runs.is_empty() {
            self.fade_until = Some(now + BLOCK_FADE);
        }
        self.blocks.push(BlockVisual {
            id,
            erase,
            runs,
            typeset,
            start: now,
            metadata,
            roi_timer,
            plate,
        });
        self.dirty = true;
        Ok(roi_upload_ms)
    }

    pub fn record_timing(&mut self, id: u64, stage: &str, duration_ms: f64) {
        let index = match stage {
            "mask" => 0,
            "erase" => 1,
            "typeset" => 2,
            _ => return,
        };
        self.timings.entry(id).or_insert([0.0; 3])[index] += duration_ms;
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn dimensions(&self) -> Option<[u32; 2]> {
        self.working.as_ref().map(|photo| [photo.pixels.width(), photo.pixels.height()])
    }

    pub fn first_visible_block_in(&self, now: Instant, shown: [f32; 4]) -> Option<u64> {
        let [width, height] = self.dimensions()?.map(|value| value as f32);
        let [left, top, right, bottom] = [shown[0] * width, shown[1] * height, shown[2] * width, shown[3] * height];
        self.blocks.iter().find(|block| {
            !block.runs.is_empty() && fade_opacity(block.start, now) > 0.0
                && block.typeset.as_ref().is_some_and(|text| text.collision_boxes.iter().any(|bounds| {
                    bounds.left < right && bounds.right > left && bounds.top < bottom && bounds.bottom > top
                }))
        }).map(|block| block.id)
    }

    pub fn needs_frame(&self, _now: Instant) -> bool {
        self.dirty || self.fade_until.is_some()
    }

    pub fn render(
        &mut self,
        now: Instant,
        erased_only: bool,
    ) -> Result<Option<PhotoTexture>, PhotoError> {
        let Some(original) = &self.original else {
            return Ok(None);
        };
        let working = self.working.as_ref().ok_or(PhotoError::Graphics)?;
        let bounds = D2D_RECT_F {
            left: 0.0,
            top: 0.0,
            right: working.pixels.width() as f32,
            bottom: working.pixels.height() as f32,
        };
        self.frame_timer.begin(&self.d3d);
        self.frame_timing_started = true;
        unsafe {
            self.d2d
                .SetTarget(self.target.as_ref().ok_or(PhotoError::Graphics)?);
            self.d2d.SetTransform(&Matrix3x2::identity());
            self.d2d.BeginDraw();
            self.d2d.Clear(Some(&D2D1_COLOR_F {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            }));
            self.d2d.DrawBitmap(
                original,
                Some(&bounds),
                1.0,
                D2D1_INTERPOLATION_MODE_LINEAR,
                None,
                None,
            );
            for block in &self.blocks {
                let opacity = fade_opacity(block.start, now);
                if let Some((bitmap, rect)) = &block.erase {
                    let destination = D2D_RECT_F {
                        left: rect.x as f32,
                        top: rect.y as f32,
                        right: (rect.x + rect.width) as f32,
                        bottom: (rect.y + rect.height) as f32,
                    };
                    self.d2d.DrawBitmap(
                        bitmap,
                        Some(&destination),
                        opacity,
                        D2D1_INTERPOLATION_MODE_LINEAR,
                        None,
                        None,
                    );
                }
                if let Some((brush, rect)) = &block.plate {
                    brush.SetOpacity(opacity);
                    self.d2d.FillRectangle(&d2d_rect(*rect), brush);
                }
            }
            if !erased_only {
                for block in &self.blocks {
                    let Some(typeset) = &block.typeset else {
                        continue;
                    };
                    let opacity = fade_opacity(block.start, now);
                    if opacity <= 0.0 {
                        continue;
                    }
                    self.d2d.PushLayer(
                        &D2D1_LAYER_PARAMETERS1 {
                            contentBounds: bounds,
                            geometricMask: ManuallyDrop::new(None),
                            maskAntialiasMode: D2D1_ANTIALIAS_MODE_PER_PRIMITIVE,
                            maskTransform: Matrix3x2::identity(),
                            opacity,
                            opacityBrush: ManuallyDrop::new(None),
                            layerOptions: D2D1_LAYER_OPTIONS1_NONE,
                        },
                        &self.opacity_layer,
                    );
                    for run in &block.runs {
                        self.d2d.SetTransform(&run_transform(
                            run.origin,
                            typeset.center,
                            typeset.angle_degrees,
                        ));
                        self.d2d.DrawImage(
                            &run.commands,
                            None,
                            None,
                            D2D1_INTERPOLATION_MODE_LINEAR,
                            D2D1_COMPOSITE_MODE_SOURCE_OVER,
                        );
                    }
                    self.d2d.SetTransform(&Matrix3x2::identity());
                    self.d2d.PopLayer();
                }
            }
            self.d2d
                .EndDraw(None, None)
                .map_err(|_| PhotoError::Graphics)?;
        }
        self.frame_timer.end(&self.d3d);
        self.dirty = false;
        if self.fade_until.is_some_and(|until| now >= until) {
            self.fade_until = None;
        }
        Ok(self
            .texture
            .as_ref()
            .map(|texture| PhotoTexture::new(texture.clone())))
    }

    pub async fn ready(&mut self) -> Result<(), PhotoError> {
        unsafe {
            self.d3d.End(&self.readiness);
            self.d3d.Flush();
        }
        let start = Instant::now();
        loop {
            let mut ready = BOOL::default();
            unsafe {
                self.d3d.GetData(
                    &self.readiness,
                    Some((&mut ready as *mut BOOL).cast()),
                    std::mem::size_of::<BOOL>() as u32,
                    D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
                )
            }
            .map_err(|_| PhotoError::Graphics)?;
            if ready.as_bool() {
                self.frame_gpu_ms = if self.frame_timing_started {
                    self.frame_timer.elapsed_ms(&self.d3d)?
                } else {
                    None
                };
                for block in &mut self.blocks {
                    if let Some(timer) = block.roi_timer.take() {
                        let elapsed = timer.elapsed_ms(&self.d3d)?;
                        block.metadata["roi_gpu_ms"] = json!(elapsed);
                        block.metadata["roi_gpu_timing_valid"] = json!(elapsed.is_some());
                        println!(
                            "{}",
                            json!({
                                "type":"photo_image_stage","stage":"roi_updated","block_id":block.id,
                                "gpu_ms":elapsed,"gpu_timing_valid":elapsed.is_some(),
                            })
                        );
                    }
                }
                return Ok(());
            }
            if start.elapsed() >= Duration::from_millis(50) {
                return Err(PhotoError::Graphics);
            }
            tokio::task::yield_now().await;
        }
    }

    pub fn metadata(&self) -> Value {
        json!({ "width": self.working.as_ref().map(|photo| photo.pixels.width()), "height": self.working.as_ref().map(|photo| photo.pixels.height()), "reference_scale": self.working.as_ref().map(|photo| photo.reference_scale), "blocks": self.blocks.iter().map(|block| &block.metadata).collect::<Vec<_>>(), "original_uploads": self.original_uploads, "adapter": format!("{:?}", self.adapter), "composition_gpu_ms": self.frame_gpu_ms, "gpu_timing_valid": self.frame_gpu_ms.is_some(), "font": self.font })
    }

    pub fn save(&self, path: &Path) -> Result<(), PhotoError> {
        let working = self.working.as_ref().ok_or(PhotoError::Graphics)?;
        let width = working.pixels.width();
        let height = working.pixels.height();
        self.save_d3d_texture(
            self.texture.as_ref().ok_or(PhotoError::Graphics)?,
            width,
            height,
            path,
        )
    }

    fn save_d3d_texture(
        &self,
        texture: &ID3D11Texture2D,
        width: u32,
        height: u32,
        path: &Path,
    ) -> Result<(), PhotoError> {
        let staging = self.create_texture(width, height, true)?;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.d3d.CopyResource(&staging, texture);
            self.d3d
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|_| PhotoError::Graphics)?;
        }
        let mut pixels = image::RgbaImage::new(width, height);
        for y in 0..height {
            let row = unsafe {
                std::slice::from_raw_parts(
                    (mapped.pData as *const u8).add(y as usize * mapped.RowPitch as usize),
                    width as usize * 4,
                )
            };
            for (x, pixel) in row.chunks_exact(4).enumerate() {
                pixels.put_pixel(
                    x as u32,
                    y,
                    image::Rgba([pixel[2], pixel[1], pixel[0], pixel[3]]),
                );
            }
        }
        unsafe {
            self.d3d.Unmap(&staging, 0);
        }
        pixels.save(path).map_err(|_| PhotoError::Image)
    }

    fn upload(&self, width: u32, height: u32, bgra: &[u8]) -> Result<ID2D1Bitmap1, PhotoError> {
        unsafe {
            self.d2d.CreateBitmap(
                D2D_SIZE_U { width, height },
                Some(bgra.as_ptr().cast()),
                width * 4,
                &bitmap_properties(D2D1_BITMAP_OPTIONS_NONE),
            )
        }
        .map_err(|_| PhotoError::Graphics)
    }

    fn create_texture(
        &self,
        width: u32,
        height: u32,
        staging: bool,
    ) -> Result<ID3D11Texture2D, PhotoError> {
        let mut texture = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: if staging {
                D3D11_USAGE_STAGING
            } else {
                D3D11_USAGE_DEFAULT
            },
            BindFlags: if staging {
                0
            } else {
                (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32
            },
            CPUAccessFlags: if staging {
                D3D11_CPU_ACCESS_READ.0 as u32
            } else {
                0
            },
            MiscFlags: 0,
        };
        unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut texture)) }
            .map_err(|_| PhotoError::Graphics)?;
        texture.ok_or(PhotoError::Graphics)
    }
}

pub fn fade_opacity(start: Instant, now: Instant) -> f32 {
    (now.saturating_duration_since(start).as_secs_f32() / BLOCK_FADE.as_secs_f32()).clamp(0.0, 1.0)
}

fn color(rgb: [u8; 3], alpha: f32) -> D2D1_COLOR_F {
    D2D1_COLOR_F {
        r: rgb[0] as f32 / 255.0,
        g: rgb[1] as f32 / 255.0,
        b: rgb[2] as f32 / 255.0,
        a: alpha,
    }
}
fn rect_array(rect: Rect) -> [f32; 4] {
    [rect.left, rect.top, rect.right, rect.bottom]
}
fn d2d_rect(rect: Rect) -> D2D_RECT_F {
    D2D_RECT_F {
        left: rect.left,
        top: rect.top,
        right: rect.right,
        bottom: rect.bottom,
    }
}
fn bitmap_properties(
    options: windows::Win32::Graphics::Direct2D::D2D1_BITMAP_OPTIONS,
) -> D2D1_BITMAP_PROPERTIES1 {
    D2D1_BITMAP_PROPERTIES1 {
        pixelFormat: D2D1_PIXEL_FORMAT {
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED,
        },
        dpiX: 96.0,
        dpiY: 96.0,
        bitmapOptions: options,
        colorContext: ManuallyDrop::new(None),
    }
}
fn run_transform(origin: [f32; 2], center: [f32; 2], degrees: f32) -> Matrix3x2 {
    let (s, c) = degrees.to_radians().sin_cos();
    Matrix3x2 {
        M11: c,
        M12: s,
        M21: -s,
        M22: c,
        M31: center[0] + (origin[0] - center[0]) * c - (origin[1] - center[1]) * s,
        M32: center[1] + (origin[0] - center[0]) * s + (origin[1] - center[1]) * c,
    }
}

#[cfg(test)]
#[path = "compositor_tests.rs"]
mod tests;
