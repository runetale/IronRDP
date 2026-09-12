use core::num::NonZeroU32;

#[cfg(target_arch = "wasm32")]
use anyhow::anyhow;
use ironrdp::pdu::geometry::InclusiveRectangle;
#[cfg(target_arch = "wasm32")]
use ironrdp::pdu::geometry::Rectangle as _;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{Clamped, JsCast as _};
#[cfg(target_arch = "wasm32")]
use web_sys::ImageData;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, OffscreenCanvas, OffscreenCanvasRenderingContext2d};

/// Where a session draws.
///
/// An `OffscreenCanvas` is the same surface handed to a worker, so a session
/// running off the main thread neither blocks paint nor is blocked by it. The
/// two context types are distinct in web-sys despite offering the same methods
/// here, hence the enum rather than a trait object.
pub(crate) enum RenderTarget {
    Onscreen(HtmlCanvasElement),
    Offscreen(OffscreenCanvas),
}

impl Clone for RenderTarget {
    fn clone(&self) -> Self {
        match self {
            Self::Onscreen(canvas) => Self::Onscreen(canvas.clone()),
            Self::Offscreen(canvas) => Self::Offscreen(canvas.clone()),
        }
    }
}

enum Context {
    Onscreen(CanvasRenderingContext2d),
    Offscreen(OffscreenCanvasRenderingContext2d),
}

/// Web render surface: blits each dirty region to the canvas with `put_image_data`.
pub(crate) struct Canvas {
    canvas: RenderTarget,
    ctx: Context,
}

impl Canvas {
    pub(crate) fn new(render_canvas: RenderTarget, width: NonZeroU32, height: NonZeroU32) -> anyhow::Result<Self> {
        set_size(&render_canvas, width, height);
        let ctx = context_2d(&render_canvas)?;

        Ok(Self {
            canvas: render_canvas,
            ctx,
        })
    }

    /// Resizes the backing store. Note: this also clears the canvas and resets 2D context state;
    /// the cached `ctx` stays valid.
    pub(crate) fn resize(&mut self, width: NonZeroU32, height: NonZeroU32) {
        set_size(&self.canvas, width, height);
    }

    /// Blits a dirty region with `put_image_data`. Forces alpha opaque first: the framebuffer isn't
    /// guaranteed opaque (zero-init columns, QOI-RGBA) and `put_image_data` stores alpha verbatim.
    pub(crate) fn draw(&self, buffer: &mut [u8], region: InclusiveRectangle) -> anyhow::Result<()> {
        for pixel in buffer.chunks_exact_mut(4) {
            pixel[3] = 0xFF;
        }

        #[cfg(target_arch = "wasm32")]
        {
            let image = ImageData::new_with_u8_clamped_array_and_sh(
                Clamped(&*buffer),
                u32::from(region.width()),
                u32::from(region.height()),
            )
            .map_err(|err| anyhow!("ImageData::new failed: {err:?}"))?;
            match &self.ctx {
                Context::Onscreen(ctx) => ctx.put_image_data(&image, f64::from(region.left), f64::from(region.top)),
                Context::Offscreen(ctx) => ctx.put_image_data(&image, f64::from(region.left), f64::from(region.top)),
            }
            .map_err(|err| anyhow!("put_image_data failed: {err:?}"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (&self.ctx, buffer, region);
            unimplemented!("web canvas is only available on wasm32")
        }
    }
}

fn set_size(canvas: &RenderTarget, width: NonZeroU32, height: NonZeroU32) {
    match canvas {
        RenderTarget::Onscreen(canvas) => {
            canvas.set_width(width.get());
            canvas.set_height(height.get());
        }
        RenderTarget::Offscreen(canvas) => {
            canvas.set_width(width.get());
            canvas.set_height(height.get());
        }
    }
}

/// Acquires the canvas 2D context (wasm only; panics on other targets).
fn context_2d(canvas: &RenderTarget) -> anyhow::Result<Context> {
    #[cfg(target_arch = "wasm32")]
    {
        match canvas {
            RenderTarget::Onscreen(canvas) => canvas
                .get_context("2d")
                .map_err(|err| anyhow!("get_context(\"2d\") failed: {err:?}"))?
                .ok_or_else(|| anyhow!("canvas has no 2d context"))?
                .dyn_into::<CanvasRenderingContext2d>()
                .map(Context::Onscreen)
                .map_err(|_| anyhow!("2d context is not a CanvasRenderingContext2d")),
            RenderTarget::Offscreen(canvas) => canvas
                .get_context("2d")
                .map_err(|err| anyhow!("get_context(\"2d\") failed: {err:?}"))?
                .ok_or_else(|| anyhow!("canvas has no 2d context"))?
                .dyn_into::<OffscreenCanvasRenderingContext2d>()
                .map(Context::Offscreen)
                .map_err(|_| anyhow!("2d context is not an OffscreenCanvasRenderingContext2d")),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = canvas;
        unimplemented!("web canvas is only available on wasm32")
    }
}
