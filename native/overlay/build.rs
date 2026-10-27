use std::env;
use std::path::PathBuf;

#[path = "../windows_version.rs"]
mod windows_version;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let version = env!("CARGO_PKG_VERSION");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is required"));
    let rc_path = out_dir.join("version.rc");
    std::fs::write(
        &rc_path,
        windows_version::render_version_rc(version, "PuriPuly <3 Overlay", "PuriPulyHeartOverlay"),
    )
    .expect("failed to write version.rc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=../windows_version.rs");
    embed_resource::compile(&rc_path, embed_resource::NONE)
        .manifest_optional()
        .expect("failed to embed Windows version resource");
}
