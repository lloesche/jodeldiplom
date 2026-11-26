//! A cover: the art fitted into its place, its corners rounded, a soft
//! shadow under it on the ground; with text over its bottom, the cover
//! darkened under the text so it reads on any cover. It does not take the
//! pointer: what is under it does.
//!
//! The places come in pixels ([`Spot`]), as the old canvas lays them out.
//! The art is scaled on the processor to the pixels it shows at (finer
//! than the GPU's scaling of a large image, which keeps no smaller copies),
//! once for each art and size: [`Covers`] keeps it by art id.

use std::f32::consts::PI;
use std::sync::{Arc, OnceLock};

use iced::advanced::image::{Allocation, FilterMethod, Handle, Renderer as _};
use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{self, Tree, Widget};
use iced::advanced::{mouse, renderer};
use iced::gradient::Linear;
use iced::widget::{container, pin, space, Image, Stack};
use iced::{
    border, Background, Border, Color, ContentFit, Element, Length, Rectangle, Renderer, Shadow,
    Size, Theme, Vector, Widget as _,
};

use crate::art::{self, Art};
use crate::gui::look::{self, radius};
use crate::gui::pixels::PixelRect;

/// How dark the cover gets at its bottom under text: 80 % (as the eye
/// sees it).
const DARKEST: f32 = 0.8;

/// How a cover sits in its place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    /// At the place's top left.
    TopLeft,
    /// In the middle of it.
    Center,
}

/// Where a cover goes in the window, in pixels.
#[derive(Clone, Debug)]
pub struct Spot {
    /// The place it is fitted into (its aspect kept).
    pub place: PixelRect,
    /// How it sits there.
    pub align: Align,
    /// Text over it, if any.
    pub caption: Option<Caption>,
}

/// Text over a cover's bottom, in pixels.
#[derive(Clone, Debug)]
pub struct Caption {
    /// Where the text is.
    pub rect: PixelRect,
    /// How far above the text the cover starts to darken.
    pub lead: usize,
    /// The text on a clear ground, at the window's pixels.
    pub text: Handle,
}

/// The art fitted to a place: for which art and place size, the image,
/// its size and its offset that centres it there (pixels).
#[derive(Clone, Debug)]
struct Fitted {
    id: u64,
    place: (u32, u32),
    image: Handle,
    size: (u32, u32),
    offset: (u32, u32),
    /// The image on the GPU, once drawn ([`Loaded`]).
    loaded: Arc<OnceLock<Option<Allocation>>>,
}

impl Fitted {
    fn new(art: &Art, place: (u32, u32)) -> Fitted {
        let (scaled, x, y) = art::fit(&art.image, place.0.max(1), place.1.max(1));
        let size = scaled.dimensions();
        let rgba = image::DynamicImage::ImageRgb8(scaled)
            .into_rgba8()
            .into_raw();
        Fitted {
            id: art.id,
            place,
            image: Handle::from_rgba(size.0, size.1, rgba),
            size,
            offset: (x, y),
            loaded: Arc::default(),
        }
    }
}

/// An image drawn the first time it is drawn: iced sends a large image
/// to the GPU in the background and leaves it out until it is there,
/// which would leave a new cover's place empty for a frame or more (each
/// step of a window being resized); this one waits for it once, and
/// keeps it there while the cover shows.
struct Loaded<'a> {
    image: Image<Handle>,
    handle: Handle,
    loaded: &'a OnceLock<Option<Allocation>>,
}

impl widget::Meta for Loaded<'_> {}

impl<M> Widget<M, Theme, Renderer> for Loaded<'_> {
    fn size(&self) -> Size<Length> {
        Widget::<M, Theme, Renderer>::size(&self.image)
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &Renderer, limits: &layout::Limits) {
        Widget::<M, Theme, Renderer>::layout(&mut self.image, tree, renderer, limits);
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.loaded
            .get_or_init(|| renderer.load_image(&self.handle).ok());
        Widget::<M, Theme, Renderer>::draw(
            &self.image,
            tree,
            renderer,
            theme,
            style,
            layout,
            cursor,
            viewport,
        );
    }
}

/// The covers that show, each with the art fitted to its place, at the
/// window's scale.
#[derive(Debug)]
pub struct Covers {
    shown: Vec<(Spot, Fitted)>,
    /// Pixels a point.
    scale: f32,
}

impl Default for Covers {
    fn default() -> Covers {
        Covers {
            shown: Vec::new(),
            scale: 1.0,
        }
    }
}

impl Covers {
    /// `art` in `spots` at `scale` pixels a point (nothing without art).
    /// Art fitted to a place of the same size before is kept as it was.
    pub fn show(&mut self, art: Option<&Art>, spots: &[Spot], scale: f32) {
        let before = std::mem::take(&mut self.shown);
        self.scale = scale;
        let Some(art) = art else {
            return;
        };
        for spot in spots {
            let place = (spot.place.2 as u32, spot.place.3 as u32);
            let fitted = before
                .iter()
                .map(|(_, f)| f)
                .find(|f| (f.id, f.place) == (art.id, place))
                .cloned()
                .unwrap_or_else(|| Fitted::new(art, place));
            self.shown.push((spot.clone(), fitted));
        }
    }

    /// The images of the covers shown, in order.
    pub fn images(&self) -> impl Iterator<Item = &Handle> {
        self.shown.iter().map(|(_, f)| &f.image)
    }

    /// The covers, each pinned where it goes in the window.
    pub fn view<'a, M: 'a>(&'a self) -> Element<'a, M> {
        if self.shown.is_empty() {
            return space().boxed();
        }
        Stack::with_children(
            self.shown
                .iter()
                .flat_map(|(spot, fitted)| self.cover(spot, fitted)),
        )
        .boxed()
    }

    /// `px` pixels in points.
    fn points(&self, px: usize) -> f32 {
        px as f32 / self.scale
    }

    /// One cover's layers: the cover, and the darkening and the text
    /// over it.
    fn cover<'a, M: 'a>(&self, spot: &Spot, fitted: &'a Fitted) -> Vec<Element<'a, M>> {
        let (px, py, _, _) = spot.place;
        let (ox, oy) = match spot.align {
            Align::TopLeft => (0, 0),
            Align::Center => (fitted.offset.0 as usize, fitted.offset.1 as usize),
        };
        let bounds = Rectangle {
            x: self.points(px + ox),
            y: self.points(py + oy),
            width: self.points(fitted.size.0 as usize),
            height: self.points(fitted.size.1 as usize),
        };
        let mut layers = vec![pinned(picture(fitted, bounds), bounds)];
        if let Some(caption) = &spot.caption {
            let (cx, cy, cw, ch) = caption.rect;
            let from = self
                .points(cy.saturating_sub(caption.lead))
                .clamp(bounds.y, bounds.y + bounds.height);
            let shade = Rectangle {
                y: from,
                height: bounds.y + bounds.height - from,
                ..bounds
            };
            let text = Rectangle {
                x: self.points(cx),
                y: self.points(cy),
                width: self.points(cw),
                height: self.points(ch),
            };
            // Text all under the cover needs no darkening on it.
            if text.y < bounds.y + bounds.height {
                layers.push(pinned(darkened(shade, from > bounds.y), shade));
            }
            let words = Image::new(caption.text.clone())
                .width(text.width)
                .height(text.height)
                .content_fit(ContentFit::Fill)
                .filter_method(FilterMethod::Nearest);
            layers.push(pinned(words.boxed(), text));
        }
        layers
    }
}

/// `content` at `bounds`' top left in the window.
fn pinned<'a, M: 'a>(content: Element<'a, M>, bounds: Rectangle) -> Element<'a, M> {
    pin(content).x(bounds.x).y(bounds.y).boxed()
}

/// The `fitted` art filling `bounds`, its corners rounded, with a soft
/// shadow under it.
fn picture<'a, M: 'a>(fitted: &'a Fitted, bounds: Rectangle) -> Element<'a, M> {
    let art = Loaded {
        image: Image::new(fitted.image.clone())
            .width(bounds.width)
            .height(bounds.height)
            .content_fit(ContentFit::Fill)
            .border_radius(radius::PANEL),
        handle: fitted.image.clone(),
        loaded: &fitted.loaded,
    };
    container(art.boxed())
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            border: Border {
                radius: radius::PANEL.into(),
                ..Border::default()
            },
            shadow: Shadow {
                color: look::alpha(Color::BLACK, 0.5),
                offset: Vector::new(0.0, 4.0),
                blur_radius: 18.0,
            },
            ..container::Style::default()
        })
        .boxed()
}

/// `bounds` darkened from clear at its top to [`DARKEST`] at its bottom,
/// easing in and out; its bottom corners rounded as the cover's are, and
/// its top ones too where it starts at the cover's top (`below_top`
/// false).
fn darkened<'a, M: 'a>(bounds: Rectangle, below_top: bool) -> Element<'a, M> {
    // Blended in linear light: as dark as the eye sees the share of black
    // mixed into the cover's own colours.
    let shade = |t: f32| {
        let eased = t * t * (3.0 - 2.0 * t);
        let keep = 1.0 - DARKEST * eased;
        look::alpha(Color::BLACK, 1.0 - keep.powf(2.2))
    };
    // As many stops as a gradient takes, so the curve has no kinks to see.
    let gradient = (0..8u8).fold(Linear::new(PI), |g, i| {
        let t = f32::from(i) / 7.0;
        g.add_stop(t, shade(t))
    });
    let top = if below_top { 0.0 } else { radius::PANEL };
    container(space())
        .width(bounds.width)
        .height(bounds.height)
        .style(move |_| container::Style {
            background: Some(Background::Gradient(gradient.into())),
            border: Border {
                radius: border::Radius {
                    top_left: top,
                    top_right: top,
                    bottom_right: radius::PANEL,
                    bottom_left: radius::PANEL,
                },
                ..Border::default()
            },
            ..container::Style::default()
        })
        .boxed()
}
