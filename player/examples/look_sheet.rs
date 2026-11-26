//! The native look on one sheet ([`jodeldiplom::gui::look`] and
//! [`jodeldiplom::gui::widgets`]): every colour, the type scale, the
//! spacing and the corners, every button in every state, the surfaces, the
//! progress bars and the icons. Drawn as the window draws (iced's headless
//! renderer, the player's own fonts) into `look-sheet.png` at scale 1 and
//! `look-sheet-hidpi.png` at scale 2.
//!
//! ```text
//! cargo run -p jodeldiplom --example look_sheet -- OUT_DIR
//! ```

mod headless;

use std::path::PathBuf;
use std::time::Instant;

use iced::widget::{column, container, row, text, Column, Row, Space};
use iced::{Alignment, Background, Border, Color, Element, Fill, Font, Size, Widget as _};
use jodeldiplom::gui::look::{self, height, radius, size, space};
use jodeldiplom::gui::widgets::button::{self, Kind, Look};
use jodeldiplom::gui::widgets::{self, badge, card, note, panel, progress, Icon};
use jodeldiplom::gui::window::Message;

use headless::Headless;

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

/// The sheet's size in logical pixels.
const SHEET: Size = Size::new(1280.0, 1180.0);

fn hex(c: Color) -> String {
    let [r, g, b, _] = c.into_rgba8();
    format!("#{r:02x}{g:02x}{b:02x}")
}

/// A section: its label over its content.
fn section<'a>(label: &'a str, content: Element<'a, Message>) -> Element<'a, Message> {
    column![
        text(label)
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_DIM),
        content,
    ]
    .spacing(space::M)
    .boxed()
}

fn caption(s: String) -> Element<'static, Message> {
    text(s)
        .size(size::CAPTION)
        .font(look::font::numbers())
        .color(look::TEXT_FAINT)
        .boxed()
}

fn colours() -> Element<'static, Message> {
    let all = [
        ("bg", look::BG),
        ("surface", look::SURFACE),
        ("raised", look::RAISED),
        ("selected", look::SELECTED),
        ("edge", look::EDGE),
        ("text", look::TEXT),
        ("text_dim", look::TEXT_DIM),
        ("text_faint", look::TEXT_FAINT),
        ("accent", look::ACCENT),
        ("accent_ink", look::ACCENT_INK),
        ("good", look::GOOD),
        ("warn", look::WARN),
        ("fail", look::FAIL),
        ("folder", look::FOLDER),
        ("source", look::SOURCE),
    ];
    let mut grid = Row::new().spacing(space::M);
    for (name, colour) in all {
        let swatch =
            container(Space::new().width(64).height(40)).style(move |_| container::Style {
                background: Some(Background::Color(colour)),
                border: Border {
                    color: look::EDGE,
                    width: look::line::EDGE,
                    radius: radius::CONTROL.into(),
                },
                ..container::Style::default()
            });
        grid = grid.push(
            column![
                swatch,
                text(name)
                    .size(size::CAPTION)
                    .font(look::font::label())
                    .color(look::TEXT),
                caption(hex(colour)),
            ]
            .spacing(space::XS)
            .boxed(),
        );
    }
    section("Colours", grid.boxed())
}

fn type_scale() -> Element<'static, Message> {
    let lines: [(&str, f32, Font, Color); 6] = [
        (
            "Display 28 · Getting the player ready",
            size::DISPLAY,
            look::font::title(),
            look::TEXT,
        ),
        (
            "Title 18 · Downloading the word-timing model",
            size::TITLE,
            look::font::title(),
            look::TEXT,
        ),
        (
            "Body 14 · It hears where each sung word falls.",
            size::BODY,
            look::font::body(),
            look::TEXT,
        ),
        (
            "Label 13 · All of it",
            size::LABEL,
            look::font::label(),
            look::TEXT_DIM,
        ),
        (
            "Caption 12 · Quitting keeps what is downloaded whole.",
            size::CAPTION,
            look::font::body(),
            look::TEXT_FAINT,
        ),
        (
            "Numbers · 0123456789 · 11:11 · 88:88 · 806 MB of 1.88 GB · 43 %",
            size::BODY,
            look::font::numbers(),
            look::TEXT,
        ),
    ];
    let mut col = Column::new().spacing(space::S);
    for (s, px, font, colour) in lines {
        col = col.push(text(s).size(px).font(font).color(colour).boxed());
    }
    section("Type", col.boxed())
}

fn spacing_and_corners() -> Element<'static, Message> {
    let mut bars = Row::new().spacing(space::L).align_y(Alignment::End);
    for (name, w) in [
        ("XS", space::XS),
        ("S", space::S),
        ("M", space::M),
        ("L", space::L),
        ("XL", space::XL),
        ("XXL", space::XXL),
    ] {
        let bar = container(Space::new().width(w).height(w)).style(|_| container::Style {
            background: Some(Background::Color(look::alpha(look::ACCENT, 0.5))),
            ..container::Style::default()
        });
        bars = bars.push(
            column![bar, caption(format!("{name} {w}"))]
                .spacing(space::XS)
                .align_x(Alignment::Center)
                .boxed(),
        );
    }
    let mut corners = Row::new().spacing(space::L);
    for (name, r) in [
        ("control 6", radius::CONTROL),
        ("panel 10", radius::PANEL),
        ("pill", radius::PILL),
    ] {
        let shape = container(Space::new().width(88).height(40)).style(move |_| container::Style {
            background: Some(Background::Color(look::RAISED)),
            border: Border {
                color: look::EDGE,
                width: look::line::EDGE,
                radius: r.into(),
            },
            ..container::Style::default()
        });
        corners = corners.push(
            column![shape, caption(name.to_owned())]
                .spacing(space::XS)
                .boxed(),
        );
    }
    let heights = caption(format!(
        "row {} · dense row {} · control {} · bar {} · track {} · icon {}",
        height::ROW,
        height::DENSE_ROW,
        height::CONTROL,
        height::BAR,
        height::TRACK,
        height::ICON
    ));
    row![
        section("Spacing", bars.boxed()),
        section("Corners", corners.boxed()),
        section("Heights", heights),
    ]
    .spacing(space::XXL)
    .boxed()
}

fn buttons() -> Element<'static, Message> {
    let states = [
        ("rest", Look::default()),
        (
            "hover",
            Look {
                hover: 1.0,
                ..Look::default()
            },
        ),
        (
            "pressed",
            Look {
                hover: 1.0,
                pressed: true,
                ..Look::default()
            },
        ),
        (
            "focused",
            Look {
                focused: true,
                ..Look::default()
            },
        ),
        (
            "disabled",
            Look {
                disabled: true,
                ..Look::default()
            },
        ),
    ];
    let cell = |e: Element<'static, Message>, width: f32| container(e).width(width).boxed();
    let mut head = Row::new().push(cell(Space::new().boxed(), 90.0));
    for (name, _) in states {
        head = head.push(cell(caption(name.to_owned()), 190.0));
    }
    let mut rows = Column::new().spacing(space::L).push(head.boxed());
    for kind in [Kind::Primary, Kind::Secondary, Kind::Ghost, Kind::Icon] {
        let mut line = Row::new()
            .align_y(Alignment::Center)
            .push(cell(caption(format!("{kind:?}").to_lowercase()), 90.0));
        for (_, shown) in states {
            let made = match kind {
                Kind::Primary => widgets::primary("Try again")
                    .leading(Icon::Retry)
                    .key("Enter"),
                Kind::Secondary => widgets::secondary("Quit").key("Q"),
                Kind::Ghost => widgets::ghost("Quit"),
                Kind::Icon => button::icon(Icon::Retry),
            };
            let made = if shown.disabled {
                made
            } else {
                made.on_press(Message::Wake)
            };
            line = line.push(cell(Element::from(made.shown(shown)), 190.0));
        }
        rows = rows.push(line.boxed());
    }
    section("Buttons", rows.boxed())
}

fn surfaces() -> Element<'static, Message> {
    let sample = |s: &'static str| {
        text(s)
            .size(size::BODY)
            .font(look::font::body())
            .color(look::TEXT)
    };
    let marks = row![
        badge(
            widgets::icon(Icon::Check).size(14.0).color(look::GOOD),
            22.0,
            look::tint(look::GOOD, look::SURFACE, 0.14),
            Color::TRANSPARENT,
        ),
        badge(
            text("2")
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::ACCENT_INK),
            22.0,
            look::ACCENT,
            Color::TRANSPARENT,
        ),
        badge(
            text("3")
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::TEXT_FAINT),
            22.0,
            Color::TRANSPARENT,
            look::EDGE,
        ),
        badge(
            widgets::icon(Icon::Cross).size(12.0).color(look::FAIL),
            22.0,
            look::tint(look::FAIL, look::SURFACE, 0.16),
            Color::TRANSPARENT,
        ),
    ]
    .spacing(space::S);
    let on_panel = column![
        sample("A panel: surface, padding 16, radius 10"),
        card(sample("A card on it: raised")).width(Fill),
        note(
            row![
                widgets::icon(Icon::Alert).size(18.0).color(look::FAIL),
                sample("A note in a status tone"),
            ]
            .spacing(space::M),
            look::FAIL,
        )
        .width(Fill),
        marks,
    ]
    .spacing(space::M);
    let bars = column![
        caption("0 %".to_owned()),
        progress(0.0, look::ACCENT),
        caption("43 %, accent".to_owned()),
        progress(0.43, look::ACCENT),
        caption("100 %, good".to_owned()),
        progress(1.0, look::GOOD),
    ]
    .spacing(space::S)
    .width(320);
    let mut icons = Row::new().spacing(space::XL);
    for icon in Icon::ALL {
        icons = icons.push(
            column![
                row![
                    widgets::icon(icon).color(look::TEXT),
                    widgets::icon(icon).size(24.0).color(look::TEXT),
                    widgets::icon(icon).size(32.0).color(look::ACCENT),
                ]
                .spacing(space::M)
                .align_y(Alignment::End),
                caption(icon.name().to_owned()),
            ]
            .spacing(space::XS)
            .boxed(),
        );
    }
    row![
        section("Surfaces", panel(on_panel).width(420).boxed()),
        column![
            section("Progress", bars.boxed()),
            section("Icons", icons.boxed()),
        ]
        .spacing(space::XL),
    ]
    .spacing(space::XXL)
    .boxed()
}

fn sheet() -> Element<'static, Message> {
    container(
        column![
            text("The look")
                .size(size::DISPLAY)
                .font(look::font::title())
                .color(look::TEXT),
            colours(),
            type_scale(),
            spacing_and_corners(),
            buttons(),
            surfaces(),
        ]
        .spacing(space::XL),
    )
    .padding(space::XXL)
    .width(Fill)
    .height(Fill)
    .style(|_| container::Style {
        background: Some(Background::Color(look::BG)),
        ..container::Style::default()
    })
    .boxed()
}

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("give an output folder"));
    std::fs::create_dir_all(&out).unwrap();
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    let mut headless = Headless::new();
    for (name, scale) in [("look-sheet", 1.0), ("look-sheet-hidpi", 2.0)] {
        headless.draw(sheet(), SHEET, Instant::now());
        let path = out.join(format!("{name}.png"));
        let mut png = durable::AtomicFile::create(&path).expect("write the PNG");
        headless
            .screenshot(scale)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("a PNG");
        png.commit().expect("write the PNG");
        println!("{} (drawn by {})", path.display(), headless.name());
    }
}
