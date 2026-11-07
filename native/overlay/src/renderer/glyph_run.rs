#[cfg(windows)]
use super::types::CaptionRenderError;
#[cfg(windows)]
use windows::core::implement;
#[cfg(windows)]
use windows::Win32::Graphics::Direct2D::{
    Common::ID2D1SimplifiedGeometrySink, ID2D1CommandList, ID2D1DeviceContext, ID2D1Factory1,
    ID2D1SolidColorBrush,
};
#[cfg(windows)]
use windows::Win32::Graphics::DirectWrite::{
    IDWriteInlineObject, IDWritePixelSnapping_Impl, IDWriteTextLayout, IDWriteTextRenderer,
    IDWriteTextRenderer_Impl, DWRITE_GLYPH_RUN, DWRITE_GLYPH_RUN_DESCRIPTION, DWRITE_MATRIX,
    DWRITE_MEASURING_MODE, DWRITE_STRIKETHROUGH, DWRITE_UNDERLINE,
};
#[cfg(windows)]
use windows_core::Interface;
#[cfg(windows)]
use windows_core::BOOL;
#[cfg(windows)]
use windows_numerics::Matrix3x2;

#[cfg(windows)]
#[implement(IDWriteTextRenderer)]
struct GeometryTextRenderer {
    d2d_context: ID2D1DeviceContext,
    d2d_factory: ID2D1Factory1,
    fill_brush: ID2D1SolidColorBrush,
    outline_brush: ID2D1SolidColorBrush,
    outline_stroke_width_px: f32,
}

#[cfg(windows)]
impl IDWritePixelSnapping_Impl for GeometryTextRenderer_Impl {
    fn IsPixelSnappingDisabled(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
    ) -> windows::core::Result<BOOL> {
        Ok(false.into())
    }

    fn GetCurrentTransform(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
        transform: *mut DWRITE_MATRIX,
    ) -> windows::core::Result<()> {
        unsafe {
            transform.write(DWRITE_MATRIX {
                m11: 1.0,
                m12: 0.0,
                m21: 0.0,
                m22: 1.0,
                dx: 0.0,
                dy: 0.0,
            });
        }
        Ok(())
    }

    fn GetPixelsPerDip(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
    ) -> windows::core::Result<f32> {
        Ok(1.0)
    }
}

#[cfg(windows)]
impl IDWriteTextRenderer_Impl for GeometryTextRenderer_Impl {
    fn DrawGlyphRun(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
        baselineoriginx: f32,
        baselineoriginy: f32,
        _measuringmode: DWRITE_MEASURING_MODE,
        glyphrun: *const DWRITE_GLYPH_RUN,
        _glyphrundescription: *const DWRITE_GLYPH_RUN_DESCRIPTION,
        _clientdrawingeffect: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        let Some(glyph_run) = (unsafe { glyphrun.as_ref() }) else {
            return Ok(());
        };
        let Some(font_face) = glyph_run.fontFace.as_ref() else {
            return Ok(());
        };

        let path = unsafe { self.d2d_factory.CreatePathGeometry()? };
        let sink = unsafe { path.Open()? };
        let simplified_sink: ID2D1SimplifiedGeometrySink = sink.cast()?;
        let is_right_to_left = glyph_run.bidiLevel % 2 == 1;
        unsafe {
            font_face.GetGlyphRunOutline(
                glyph_run.fontEmSize,
                glyph_run.glyphIndices,
                Some(glyph_run.glyphAdvances),
                Some(glyph_run.glyphOffsets),
                glyph_run.glyphCount,
                glyph_run.isSideways.as_bool(),
                is_right_to_left,
                &simplified_sink,
            )?;
            sink.Close()?;
        }

        let transform = Matrix3x2 {
            M11: 1.0,
            M12: 0.0,
            M21: 0.0,
            M22: 1.0,
            M31: baselineoriginx,
            M32: baselineoriginy,
        };
        unsafe {
            self.d2d_context.SetTransform(&transform);
            self.d2d_context.DrawGeometry(
                &path,
                &self.outline_brush,
                self.outline_stroke_width_px,
                None,
            );
            self.d2d_context.FillGeometry(&path, &self.fill_brush, None);
            self.d2d_context.SetTransform(&identity_transform());
        }

        Ok(())
    }

    fn DrawUnderline(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _underline: *const DWRITE_UNDERLINE,
        _clientdrawingeffect: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        Ok(())
    }

    fn DrawStrikethrough(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _strikethrough: *const DWRITE_STRIKETHROUGH,
        _clientdrawingeffect: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        Ok(())
    }

    fn DrawInlineObject(
        &self,
        _clientdrawingcontext: *const core::ffi::c_void,
        _originx: f32,
        _originy: f32,
        _inlineobject: windows::core::Ref<IDWriteInlineObject>,
        _issideways: BOOL,
        _isrighttoleft: BOOL,
        _clientdrawingeffect: windows::core::Ref<windows::core::IUnknown>,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

#[cfg(windows)]
pub(crate) fn render_text_layout_to_command_list(
    cache_context: &ID2D1DeviceContext,
    d2d_factory: &ID2D1Factory1,
    text_layout: &IDWriteTextLayout,
    fill_brush: &ID2D1SolidColorBrush,
    outline_brush: &ID2D1SolidColorBrush,
    outline_stroke_width_px: f32,
) -> Result<ID2D1CommandList, CaptionRenderError> {
    let previous_target = unsafe { cache_context.GetTarget().ok() };
    let command_list = unsafe {
        cache_context
            .CreateCommandList()
            .map_err(|error| CaptionRenderError::Draw(error.to_string()))?
    };
    unsafe {
        cache_context.SetTarget(&command_list);
        cache_context.BeginDraw();
    }

    let renderer = GeometryTextRenderer {
        d2d_context: cache_context.clone(),
        d2d_factory: d2d_factory.clone(),
        fill_brush: fill_brush.clone(),
        outline_brush: outline_brush.clone(),
        outline_stroke_width_px,
    };
    let text_renderer: IDWriteTextRenderer = renderer.into();
    let draw_result = unsafe {
        text_layout
            .Draw(None, &text_renderer, 0.0, 0.0)
            .map_err(|error| CaptionRenderError::Draw(error.to_string()))
    };
    let end_draw_result = unsafe {
        cache_context
            .EndDraw(None, None)
            .map_err(|error| CaptionRenderError::Draw(error.to_string()))
    };
    unsafe {
        cache_context.SetTarget(previous_target.as_ref());
    }
    match (draw_result, end_draw_result) {
        (Err(error), _) => return Err(error),
        (Ok(()), Err(error)) => return Err(error),
        (Ok(()), Ok(())) => unsafe {
            command_list
                .Close()
                .map_err(|error| CaptionRenderError::Draw(error.to_string()))?;
        },
    }

    Ok(command_list)
}

#[cfg(windows)]
fn identity_transform() -> Matrix3x2 {
    Matrix3x2 {
        M11: 1.0,
        M12: 0.0,
        M21: 0.0,
        M22: 1.0,
        M31: 0.0,
        M32: 0.0,
    }
}
