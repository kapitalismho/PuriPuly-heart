use std::env;
use std::path::PathBuf;

#[path = "../windows_version.rs"]
mod windows_version;

fn record_runtime_sources(output: &std::path::Path) {
    let mut directories = std::collections::BTreeSet::new();
    for key in [
        "DEP_TRANSCRIBE_CPP_RUNTIME_DIR",
        "DEP_TRANSCRIBE_CPP_BIN_DIR",
        "DEP_TRANSCRIBE_CPP_MODULE_DIR",
    ] {
        let directory =
            PathBuf::from(env::var_os(key).unwrap_or_else(|| panic!("{key} is required")));
        directories.insert(
            directory
                .canonicalize()
                .expect("runtime directory must exist"),
        );
        println!("cargo:rerun-if-changed={}", directory.display());
    }
    let lib_dir = PathBuf::from(
        env::var_os("DEP_TRANSCRIBE_CPP_LIB_DIR")
            .expect("transcribe library directory is required"),
    );
    let link_manifest = lib_dir.join("transcribe-link.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&link_manifest).expect("transcribe link manifest must exist"),
    )
    .expect("transcribe link manifest must be JSON");
    assert_eq!(
        manifest["shared"], true,
        "GPU worker requires shared backends"
    );
    let mut files = std::collections::BTreeMap::new();
    for directory in &directories {
        for entry in std::fs::read_dir(directory).expect("runtime directory must be readable") {
            let path = entry.expect("runtime entry must be readable").path();
            if path.extension().and_then(|value| value.to_str()) != Some("dll") {
                continue;
            }
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            assert!(files.insert(name, path).is_none(), "duplicate runtime DLL");
        }
    }
    for name in [
        "transcribe.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-vulkan.dll",
        "ggml-cpu-x64.dll",
    ] {
        assert!(
            files.contains_key(name),
            "required runtime DLL is missing: {name}"
        );
    }
    assert!(
        files
            .keys()
            .any(|name| name.starts_with("ggml-cpu-") && name != "ggml-cpu-x64.dll"),
        "optimized CPU runtime modules are required"
    );
    let profile = output.ancestors().nth(3).expect("Cargo profile directory");
    for (name, path) in &files {
        std::fs::copy(path, profile.join(name)).expect("runtime DLL staging must succeed");
    }
    let record = serde_json::json!({
        "schema_version": 1,
        "directories": directories,
        "link_manifest": link_manifest,
        "files": files,
    });
    std::fs::write(
        profile.join("gpu-worker-runtime-sources.json"),
        serde_json::to_vec_pretty(&record).expect("runtime inventory must serialize"),
    )
    .expect("runtime inventory must be written");
}

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let sdk = PathBuf::from(env::var_os("VULKAN_SDK").expect("VULKAN_SDK is required"));
    let source = sdk.join("Lib").join("vulkan-1.lib");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is required"));
    let alias = output.join("vulkan.lib");
    record_runtime_sources(&output);
    std::fs::copy(&source, &alias).expect("failed to stage Vulkan import library alias");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed={}", source.display());
    let version = env!("CARGO_PKG_VERSION");
    let rc_path = output.join("version.rc");
    std::fs::write(
        &rc_path,
        windows_version::render_version_rc(version, "PuriPuly <3 GPU Worker", "PuriPulyHeartGpuWorker"),
    )
    .expect("failed to write version.rc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=../windows_version.rs");
    embed_resource::compile(&rc_path, embed_resource::NONE)
        .manifest_optional()
        .expect("failed to embed Windows version resource");
}
