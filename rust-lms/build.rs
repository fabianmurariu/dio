use std::env;

const SUPPORTED_TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-pc-windows-msvc",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
];

fn main() {
    let target = env::var("TARGET").expect("Cargo must provide TARGET to rust-lms/build.rs");
    if !SUPPORTED_TARGETS.contains(&target.as_str()) {
        panic!(
            "rust-lms does not support target {target}; supported targets are: {}",
            SUPPORTED_TARGETS.join(", ")
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
}
