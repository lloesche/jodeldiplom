//! Writes `assets/jodeldiplom.ico` and `assets/jodeldiplom.icns` from
//! `jodeldiplom::icon` (after a change to the drawing; the tests check the
//! files match it). With a folder as the argument, also writes
//! `icon-preview.png` (the Windows sizes, enlarged, side by side) and
//! `icon-preview-macos.png` (the macOS style at 1024 and 128 pixels) there.

use std::path::PathBuf;

use image::{imageops, RgbaImage};
use jodeldiplom::icon::{self, Style};

fn main() {
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    for (name, data) in [
        ("jodeldiplom.ico", icon::ico()),
        ("jodeldiplom.icns", icon::icns()),
    ] {
        let path = assets.join(name);
        std::fs::write(&path, data).expect("write the icon");
        println!("wrote {}", path.display());
    }
    let Some(dir) = std::env::args_os().nth(1).map(PathBuf::from) else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("create the folder");
    let light = image::Rgba([235, 235, 235, 255]);
    let scale = 4;
    let width: u32 = icon::ICO_SIZES.iter().map(|s| s * scale + 16).sum::<u32>() + 16;
    let mut preview = RgbaImage::from_pixel(width, 256 * scale + 32, light);
    let mut x = 16;
    for size in icon::ICO_SIZES {
        let image = RgbaImage::from_raw(size, size, icon::rgba(Style::Windows, size)).unwrap();
        let big = imageops::resize(
            &image,
            size * scale,
            size * scale,
            imageops::FilterType::Nearest,
        );
        imageops::overlay(&mut preview, &big, i64::from(x), 16);
        x += size * scale + 16;
    }
    let path = dir.join("icon-preview.png");
    preview.save(&path).expect("write the preview");
    println!("wrote {}", path.display());
    // The macOS style on a light and a dark backdrop, as the Dock shows it.
    let mut mac = RgbaImage::from_pixel(2 * 1024 + 3 * 32 + 2 * 128, 1024 + 64, light);
    for (i, backdrop) in [light, image::Rgba([40, 40, 46, 255])]
        .into_iter()
        .enumerate()
    {
        let x0 = 32 + i as u32 * (1024 + 128 + 32);
        for y in 0..1024 + 64 {
            for x in x0 - 16..x0 + 1024 + 128 + 16 {
                if x < mac.width() {
                    mac.put_pixel(x, y, backdrop);
                }
            }
        }
        let big = RgbaImage::from_raw(1024, 1024, icon::rgba(Style::MacOs, 1024)).unwrap();
        imageops::overlay(&mut mac, &big, i64::from(x0), 32);
        let small = RgbaImage::from_raw(128, 128, icon::rgba(Style::MacOs, 128)).unwrap();
        imageops::overlay(&mut mac, &small, i64::from(x0 + 1024), 32 + 1024 - 128);
    }
    let path = dir.join("icon-preview-macos.png");
    mac.save(&path).expect("write the preview");
    println!("wrote {}", path.display());
}
