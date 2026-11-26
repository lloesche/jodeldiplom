//! A cover: the art fitted into its place, its corners rounded, a soft
//! shadow under it on the ground. It does not take the pointer: what is
//! under it does. Text over its bottom goes on a shade ([`shade`]), the
//! cover darkened under it so it reads on any cover.
//!
//! The places' sizes come in pixels, as the screens make room for their
//! covers ([`crate::gui::screens::covers`]); the screens lay the covers
//! out in their places ([`Covers::art`]).
//! The art is scaled on the processor to the pixels it shows at (finer
//! than the GPU's scaling of a large image, which keeps no smaller copies),
//! once for each art and size: [`Covers`] keeps it by art id.
//!
//! A cover taken apart, as a song's preparation shows it
//! ([`crate::view::prep::CoverView`]), is its plate with its subject over
//! it ([`Covers::cut`]): the subject lifted off a little and swaying, the
//! plate still. Both are scaled the same way, once for each cover and
//! place ([`Covers::show_cut`]).

use std::f32::consts::PI;
use std::sync::{Arc, OnceLock};

use iced::advanced::image::{Allocation, Handle, Renderer as _};
use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{self, Tree, Widget};
use iced::advanced::{mouse, renderer};
use iced::gradient::Linear;
use iced::widget::{container, Container, Image};
use iced::{
    border, Background, Border, Color, ContentFit, Element, Length, Rectangle, Renderer, Shadow,
    Size, Theme, Vector, Widget as _,
};

use crate::core::prep::CoverCut;
use crate::gui::look::{self, radius};
use crate::work::art::{self, Art};
use image::imageops::FilterType;

/// How dark the cover gets at its bottom under text: 80 % (as the eye
/// sees it).
const DARKEST: f32 = 0.8;

/// The art fitted to a place: for which art and place size, the image
/// and its size (pixels).
#[derive(Clone, Debug)]
struct Fitted {
    id: u64,
    place: (u32, u32),
    image: Handle,
    size: (u32, u32),
    /// The image on the GPU, once drawn ([`Loaded`]).
    loaded: Arc<OnceLock<Option<Allocation>>>,
}

impl Fitted {
    fn new(art: &Art, place: (u32, u32)) -> Fitted {
        let (scaled, _, _) = art::fit(&art.image, place.0.max(1), place.1.max(1));
        let size = scaled.dimensions();
        let rgba = image::DynamicImage::ImageRgb8(scaled)
            .into_rgba8()
            .into_raw();
        Fitted {
            id: art.id,
            place,
            image: Handle::from_rgba(size.0, size.1, rgba),
            size,
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

/// A cover taken apart fitted to a place: which cover result and place,
/// its plate and its subject, and their size (pixels).
#[derive(Clone, Debug)]
struct Cut {
    key: String,
    place: (u32, u32),
    plate: Handle,
    subject: Option<Handle>,
    size: (u32, u32),
}

impl Cut {
    fn new(cut: &CoverCut, place: (u32, u32)) -> Cut {
        let layers = &cut.0.layers;
        let (plate, _, _) = art::fit(&layers.plate, place.0.max(1), place.1.max(1));
        let size = plate.dimensions();
        let plate = image::DynamicImage::ImageRgb8(plate).into_rgba8();
        let subject = layers.subject.as_ref().map(|s| {
            let scaled = image::imageops::resize(s, size.0, size.1, FilterType::CatmullRom);
            Handle::from_rgba(size.0, size.1, scaled.into_raw())
        });
        Cut {
            key: cut.0.key.clone(),
            place,
            plate: Handle::from_rgba(size.0, size.1, plate.into_raw()),
            subject,
            size,
        }
    }
}

/// The covers that show, each with the art fitted to its place, at the
/// window's scale; and a cover taken apart, fitted to the place of the
/// first.
#[derive(Debug)]
pub struct Covers {
    shown: Vec<Fitted>,
    cut: Option<Cut>,
    /// A story's scenes as drawn so far, each by its picture's address.
    scenes: Vec<(usize, Handle)>,
    /// Pixels a point.
    scale: f32,
}

impl Default for Covers {
    fn default() -> Covers {
        Covers {
            shown: Vec::new(),
            cut: None,
            scenes: Vec::new(),
            scale: 1.0,
        }
    }
}

impl Covers {
    /// `art` in places of `places` (each a width and a height in pixels)
    /// at `scale` pixels a point (nothing without art). Art fitted to a
    /// place of the same size before is kept as it was.
    pub fn show(&mut self, art: Option<&Art>, places: &[(u32, u32)], scale: f32) {
        let before = std::mem::take(&mut self.shown);
        self.scale = scale;
        let Some(art) = art else {
            return;
        };
        for &place in places {
            let fitted = before
                .iter()
                .find(|f| (f.id, f.place) == (art.id, place))
                .cloned()
                .unwrap_or_else(|| Fitted::new(art, place));
            self.shown.push(fitted);
        }
    }

    /// `cut` (none: no cover taken apart shows) fitted to `place` (in
    /// pixels): kept as it was for the same cover and place.
    pub fn show_cut(&mut self, cut: Option<&CoverCut>, place: Option<(u32, u32)>) {
        let (Some(cut), Some(place)) = (cut, place) else {
            self.cut = None;
            return;
        };
        if (self.cut.as_ref()).is_some_and(|c| c.key == cut.0.key && c.place == place) {
            return;
        }
        self.cut = Some(Cut::new(cut, place));
    }

    /// The cover taken apart, its subject `lift`ed off (0 to 1) and swayed
    /// across by `sway` (-1 to 1); `None` while none is fitted.
    pub fn cut<'a, M: 'a>(&'a self, sway: f32, lift: f32) -> Option<Element<'a, M>> {
        let cut = self.cut.as_ref()?;
        let size = Size::new(
            cut.size.0 as f32 / self.scale,
            cut.size.1 as f32 / self.scale,
        );
        let lifted = Lifted {
            plate: cut.plate.clone(),
            subject: cut.subject.clone(),
            sway,
            lift,
        };
        Some(
            iced::widget::canvas::Canvas::new(lifted)
                .width(size.width)
                .height(size.height)
                .boxed(),
        )
    }

    /// A story's `scenes` as drawn so far, in order: the pictures seen
    /// before kept as they were.
    pub fn show_scenes(&mut self, scenes: &[Arc<image::RgbaImage>]) {
        let before = std::mem::take(&mut self.scenes);
        for picture in scenes {
            let id = Arc::as_ptr(picture) as usize;
            let handle = (before.iter().find(|(k, _)| *k == id))
                .map(|(_, h)| h.clone())
                .unwrap_or_else(|| {
                    let (w, h) = picture.dimensions();
                    Handle::from_rgba(w, h, picture.as_raw().clone())
                });
            self.scenes.push((id, handle));
        }
    }

    /// Scene `index`'s picture, `side` points across at `opacity`.
    pub fn scene<'a, M: 'a>(
        &'a self,
        index: usize,
        opacity: f32,
        side: f32,
    ) -> Option<Element<'a, M>> {
        let (_, handle) = self.scenes.get(index)?;
        let picture: Image<Handle> = Image::new(handle.clone())
            .width(side)
            .height(side)
            .opacity(opacity)
            .border_radius(radius::CONTROL);
        Some(picture.boxed())
    }

    /// The images of the covers shown, in order.
    pub fn images(&self) -> impl Iterator<Item = &Handle> {
        self.shown.iter().map(|f| &f.image)
    }

    /// The cover of the `index`th place, at the size its art was fitted
    /// to, for the screen to lay out in its place; `None` without art.
    pub fn art<'a, M: 'a>(&'a self, index: usize) -> Option<Element<'a, M>> {
        let fitted = self.shown.get(index)?;
        Some(picture(fitted, self.size(fitted)))
    }

    /// The size `fitted` shows at, in points.
    fn size(&self, fitted: &Fitted) -> Size {
        Size::new(
            fitted.size.0 as f32 / self.scale,
            fitted.size.1 as f32 / self.scale,
        )
    }
}

/// How far the subject sways across, in points.
const SWAY: f32 = 4.0;
/// How far it lifts up off the plate, in points.
const LIFT: f32 = 7.0;
/// How much larger the subject grows as it lifts.
const GROW: f32 = 0.04;

/// A cover taken apart: its plate still, its subject over it lifted off
/// and swaying, drawn on a canvas.
struct Lifted {
    plate: Handle,
    subject: Option<Handle>,
    sway: f32,
    lift: f32,
}

impl<M> iced::widget::canvas::Program<M> for Lifted {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<iced::widget::canvas::Geometry> {
        use iced::advanced::image::Image as Picture;
        let mut frame = iced::widget::canvas::Frame::new(renderer, bounds.size());
        let whole = Rectangle::new(iced::Point::ORIGIN, bounds.size());
        let mut plate = Picture::new(self.plate.clone());
        plate.border_radius = radius::PANEL.into();
        frame.draw_image(whole, plate);
        if let Some(subject) = &self.subject {
            let grow = 1.0 + GROW * self.lift;
            let size = Size::new(bounds.width * grow, bounds.height * grow);
            let at = iced::Point::new(
                (bounds.width - size.width) / 2.0 + SWAY * self.sway,
                (bounds.height - size.height) / 2.0 - LIFT * self.lift,
            );
            frame.draw_image(Rectangle::new(at, size), Picture::new(subject.clone()));
        }
        vec![frame.into_geometry()]
    }
}

/// A place of `size` points in pixels at `scale` pixels a point, for
/// [`Covers::show`].
pub fn pixels(size: Size, scale: f32) -> (u32, u32) {
    let px = |points: f32| (points * scale).max(0.0).round() as u32;
    (px(size.width), px(size.height))
}

/// The `fitted` art at `size`, its corners rounded, with a soft shadow
/// under it.
fn picture<'a, M: 'a>(fitted: &'a Fitted, size: Size) -> Element<'a, M> {
    let art = Loaded {
        image: Image::new(fitted.image.clone())
            .width(size.width)
            .height(size.height)
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

/// `content` on a shade for a cover's bottom: the cover under it darkened
/// from clear at its top to [`DARKEST`] at its bottom, easing in and out,
/// its bottom corners rounded as the cover's are. Its top padding is how
/// far above the content the darkening starts.
pub fn shade<'a, M: 'a>(content: Element<'a, M>) -> Container<'a, Element<'a, M>> {
    // Blended in linear light: as dark as the eye sees the share of black
    // mixed into the cover's own colours.
    let dark = |t: f32| {
        let eased = t * t * (3.0 - 2.0 * t);
        let keep = 1.0 - DARKEST * eased;
        look::alpha(Color::BLACK, 1.0 - keep.powf(2.2))
    };
    // As many stops as a gradient takes, so the curve has no kinks to see.
    let gradient = (0..8u8).fold(Linear::new(PI), |g, i| {
        let t = f32::from(i) / 7.0;
        g.add_stop(t, dark(t))
    });
    container(content).style(move |_| container::Style {
        background: Some(Background::Gradient(gradient.into())),
        border: Border {
            radius: border::Radius {
                top_left: 0.0,
                top_right: 0.0,
                bottom_right: radius::PANEL,
                bottom_left: radius::PANEL,
            },
            ..Border::default()
        },
        ..container::Style::default()
    })
}
