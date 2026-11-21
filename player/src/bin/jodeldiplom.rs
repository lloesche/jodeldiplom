//! The terminal player. `jodeldiplom --help` for the options.

fn main() -> std::process::ExitCode {
    jodeldiplom::main(jodeldiplom::Frontend::Terminal)
}
