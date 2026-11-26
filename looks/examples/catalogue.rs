//! Writes the looks' catalogue, the menu a model is given to choose a
//! song's look from, made from the passes' own definitions: Markdown into
//! `docs/looks-catalogue.md` (a test holds it to the passes), and with
//! `--json FILE` the same as JSON into `FILE`. Each written whole.
//!
//! ```text
//! cargo run -p jodel-looks --example catalogue [-- --json FILE]
//! ```

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/looks-catalogue.md");
    durable::write_atomic(&docs, jodel_looks::catalogue::markdown().as_bytes())
        .expect("the catalogue written");
    println!("{}", docs.display());
    if let Some(file) = args.iter().skip_while(|a| *a != "--json").nth(1) {
        durable::write_atomic(Path::new(file), jodel_looks::catalogue::json().as_bytes())
            .expect("the JSON written");
        println!("{file}");
    }
}
