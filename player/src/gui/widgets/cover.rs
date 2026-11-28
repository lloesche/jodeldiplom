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
//! The preparation view's card is the cover of the song it shows
//! ([`Covers::show_card`], [`Covers::card`]), and while that song's story
//! is made, the story built on it ([`Covers::building`]): the cover
//! darkened, each scene drawn so far over its tile.

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

use crate::gui::look::{self, radius};
use crate::view::prep::StoryView;
use crate::work::art::{self, Art};

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

/// The covers that show, each with the art fitted to its place, at the
/// window's scale; the cover of the song whose preparation shows, fitted
/// to the preparation view's card; and its story's scenes as drawn so far.
#[derive(Debug)]
pub struct Covers {
    shown: Vec<Fitted>,
    /// The card's place (pixels), and the cover fitted to it.
    card: Option<((u32, u32), Option<Fitted>)>,
    /// A story's scenes as drawn so far, each by its picture (held, so its
    /// address is its own while it is a key).
    scenes: Vec<(Arc<image::RgbaImage>, Handle)>,
    /// Pixels a point.
    scale: f32,
}

impl Default for Covers {
    fn default() -> Covers {
        Covers {
            shown: Vec::new(),
            card: None,
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

    /// `art` (none: no cover for the card) fitted to the preparation view's
    /// card at `place` (in pixels; none: no card shows): kept as it was
    /// for the same art and place.
    pub fn show_card(&mut self, art: Option<&Art>, place: Option<(u32, u32)>) {
        let Some(place) = place else {
            self.card = None;
            return;
        };
        let same = |f: &Fitted| art.is_some_and(|a| (f.id, f.place) == (a.id, place));
        let kept = (self.card.take()).and_then(|(_, f)| f.filter(same));
        let fitted = kept.or_else(|| art.map(|a| Fitted::new(a, place)));
        self.card = Some((place, fitted));
    }

    /// The card's cover; `None` while none is fitted.
    pub fn card<'a, M: 'a>(&'a self) -> Option<Element<'a, M>> {
        let fitted = self.card.as_ref()?.1.as_ref()?;
        Some(picture(fitted, self.size(fitted)))
    }

    /// The card's cover (the card's raised ground without one) with
    /// `story`'s scenes taking its place tile by tile as they are drawn,
    /// darkened under them (the story's words go over it); `None` while
    /// no card shows.
    pub fn building<'a, M: 'a>(&'a self, story: &StoryView) -> Option<Element<'a, M>> {
        let (place, fitted) = self.card.as_ref()?;
        let size = fitted.as_ref().map_or_else(
            || {
                let side = place.0.min(place.1) as f32 / self.scale;
                Size::new(side, side)
            },
            |f| self.size(f),
        );
        // Each scene's picture as made ready for the GPU, by its picture.
        let handle = |picture: &Arc<image::RgbaImage>| {
            (self.scenes.iter()).find_map(|(k, h)| Arc::ptr_eq(k, picture).then(|| h.clone()))
        };
        let tiles = (story.scenes.iter())
            .filter_map(|(index, picture, shown)| {
                let (w, h) = picture.dimensions();
                let shape = w as f32 / h.max(1) as f32;
                Some((*index, handle(picture)?, *shown, shape))
            })
            .collect();
        let building = Building {
            cover: fitted.as_ref().map(|f| f.image.clone()),
            tiles,
            of: story.of,
        };
        Some(
            iced::widget::canvas::Canvas::new(building)
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
            let handle = (before.iter().find(|(k, _)| Arc::ptr_eq(k, picture)))
                .map(|(_, h)| h.clone())
                .unwrap_or_else(|| {
                    let (w, h) = picture.dimensions();
                    Handle::from_rgba(w, h, picture.as_raw().clone())
                });
            self.scenes.push((Arc::clone(picture), handle));
        }
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

/// How dark the cover is under a story being built on it (0 clear, 1
/// black).
const UNDER_STORY: f32 = 0.6;
/// The gap between two of the story's tiles, in points.
const TILE_GAP: f32 = 1.0;

/// The shape of a story's scene pictures, its width over its height.
const SCENE_SHAPE: f32 =
    crate::core::story::pictures::SCENE.0 as f32 / crate::core::story::pictures::SCENE.1 as f32;

/// The rows and columns of a grid of `of` tiles over a picture of
/// `size`: as many columns as keep the tiles nearest to the `shape` (width
/// over height) of the pictures drawn in them.
fn grid(of: usize, size: Size, shape: f32) -> (usize, usize) {
    let of = of.max(1);
    let off = |columns: usize| {
        let rows = of.div_ceil(columns) as f32;
        let tile = (size.width / columns as f32) / (size.height.max(1.0) / rows);
        (tile / shape).ln().abs()
    };
    let columns = (1..=of)
        .min_by(|a, b| off(*a).total_cmp(&off(*b)))
        .unwrap_or(1);
    (of.div_ceil(columns), columns)
}

/// The largest place of `shape` (width over height) inside `tile`,
/// centred: a picture drawn there whole and unstretched.
fn fitted(tile: Rectangle, shape: f32) -> Rectangle {
    let (width, height) = if tile.width / tile.height.max(1.0) > shape {
        (tile.height * shape, tile.height)
    } else {
        (tile.width, tile.width / shape)
    };
    Rectangle::new(
        iced::Point::new(
            tile.x + (tile.width - width) / 2.0,
            tile.y + (tile.height - height) / 2.0,
        ),
        Size::new(width, height),
    )
}

/// A song's story being built on its cover: the cover darkened (none: the
/// card's ground), each scene drawn so far over its tile (the `of` tiles
/// in the song's order, row by row), fading in as it came.
struct Building {
    cover: Option<Handle>,
    /// Each scene drawn: which, its picture, how far it has faded in, and
    /// its picture's shape (width over height).
    tiles: Vec<(usize, Handle, f32, f32)>,
    of: usize,
}

impl<M> iced::widget::canvas::Program<M> for Building {
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
        if let Some(cover) = &self.cover {
            let mut cover = Picture::new(cover.clone());
            cover.border_radius = radius::PANEL.into();
            frame.draw_image(whole, cover);
        }
        let shade = look::alpha(Color::BLACK, UNDER_STORY);
        frame.fill_rectangle(iced::Point::ORIGIN, bounds.size(), shade);
        let (rows, columns) = grid(self.of, bounds.size(), SCENE_SHAPE);
        let tile = Size::new(bounds.width / columns as f32, bounds.height / rows as f32);
        for (index, handle, shown, shape) in &self.tiles {
            let (row, column) = (index / columns, index % columns);
            let at = iced::Point::new(
                column as f32 * tile.width + TILE_GAP / 2.0,
                row as f32 * tile.height + TILE_GAP / 2.0,
            );
            let size = Size::new(tile.width - TILE_GAP, tile.height - TILE_GAP);
            let mut scene = Picture::new(handle.clone());
            scene.opacity = *shown;
            frame.draw_image(fitted(Rectangle::new(at, size), *shape), scene);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The story's tiles are as near to its pictures' shape as their count
    /// lets them be, every one of them in the grid, none of it a row too
    /// many.
    #[test]
    fn a_story_s_tiles_are_near_its_pictures_shape_over_its_cover() {
        let wide = Size::new(1600.0, 900.0);
        let (rows, columns) = grid(23, wide, 16.0 / 9.0);
        assert_eq!((rows, columns), (5, 5), "16:9 tiles of 320 × 180");
        assert!(rows * columns >= 23 && (rows - 1) * columns < 23);
        assert_eq!(grid(4, Size::new(300.0, 300.0), 1.0), (2, 2));
        assert_eq!(grid(1, wide, SCENE_SHAPE), (1, 1));
        assert_eq!(grid(0, wide, SCENE_SHAPE), (1, 1), "none told yet: one");
        let (rows, columns) = grid(23, Size::new(900.0, 1600.0), SCENE_SHAPE);
        assert!(rows > columns && rows * columns >= 23);
        // A square cover of 18 scenes: tiles wider than tall.
        let (rows, columns) = grid(18, Size::new(576.0, 576.0), SCENE_SHAPE);
        assert!(columns < rows, "{rows} × {columns}");
    }

    /// A scene is drawn whole in its tile, its shape kept, centred.
    #[test]
    fn a_scene_keeps_its_shape_in_its_tile() {
        let square = Rectangle::new(iced::Point::new(10.0, 20.0), Size::new(100.0, 100.0));
        let at = fitted(square, 2.0);
        assert_eq!(
            at,
            Rectangle::new(iced::Point::new(10.0, 45.0), Size::new(100.0, 50.0))
        );
        let tall = Rectangle::new(iced::Point::ORIGIN, Size::new(100.0, 40.0));
        let at = fitted(tall, 2.0);
        assert_eq!(
            at,
            Rectangle::new(iced::Point::new(10.0, 0.0), Size::new(80.0, 40.0))
        );
    }
}
