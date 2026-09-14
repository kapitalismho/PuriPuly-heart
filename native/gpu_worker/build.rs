use std::env;
use std::path::PathBuf;

const PRODUCT_NAME: &str = "PuriPuly <3";
const COMPANY_NAME: &str = "salee";
const FILE_DESCRIPTION: &str = "PuriPuly <3 GPU Worker";
const INTERNAL_NAME: &str = "PuriPulyHeartGpuWorker";
const ORIGINAL_FILENAME: &str = "PuriPulyHeartGpuWorker.exe";

fn version_tuple(version: &str) -> (u16, u16, u16, u16) {
    let parts: Vec<u16> = version
        .trim()
        .split('.')
        .map(|part| {
            part.parse::<u16>()
                .unwrap_or_else(|_| panic!("non-numeric version part {part:?} in {version:?}"))
        })
        .collect();
    assert!(
        (1..=4).contains(&parts.len()),
        "version must have 1-4 numeric parts: {version:?}"
    );
    let mut padded = parts;
    while padded.len() < 4 {
        padded.push(0);
    }
    (padded[0], padded[1], padded[2], padded[3])
}

fn render_version_rc(version: &str) -> String {
    assert_eq!(PRODUCT_NAME, "PuriPuly <3");
    let (major, minor, patch, build) = version_tuple(version);
    format!(
        concat!(
            "1 VERSIONINFO\r\n",
            " FILEVERSION {major},{minor},{patch},{build}\r\n",
            " PRODUCTVERSION {major},{minor},{patch},{build}\r\n",
            " FILEFLAGSMASK 0x3fL\r\n",
            " FILEFLAGS 0x0L\r\n",
            " FILEOS 0x40004L\r\n",
            " FILETYPE 0x1L\r\n",
            " FILESUBTYPE 0x0L\r\n",
            "BEGIN\r\n",
            "    BLOCK \"StringFileInfo\"\r\n",
            "    BEGIN\r\n",
            "        BLOCK \"040904B0\"\r\n",
            "        BEGIN\r\n",
            "            VALUE \"CompanyName\", \"{company}\"\r\n",
            "            VALUE \"FileDescription\", \"{description}\"\r\n",
            "            VALUE \"FileVersion\", \"{version}\"\r\n",
            "            VALUE \"InternalName\", \"{internal}\"\r\n",
            "            VALUE \"OriginalFilename\", \"{filename}\"\r\n",
            "            VALUE \"ProductName\", \"{product}\"\r\n",
            "            VALUE \"ProductVersion\", \"{version}\"\r\n",
            "        END\r\n",
            "    END\r\n",
            "    BLOCK \"VarFileInfo\"\r\n",
            "    BEGIN\r\n",
            "        VALUE \"Translation\", 0x409, 1200\r\n",
            "    END\r\n",
            "END\r\n"
        ),
        major = major,
        minor = minor,
        patch = patch,
        build = build,
        company = COMPANY_NAME,
        description = FILE_DESCRIPTION,
        internal = INTERNAL_NAME,
        filename = ORIGINAL_FILENAME,
        product = PRODUCT_NAME,
        version = version,
    )
}

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let sdk = PathBuf::from(env::var_os("VULKAN_SDK").expect("VULKAN_SDK is required"));
    let source = sdk.join("Lib").join("vulkan-1.lib");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is required"));
    let alias = output.join("vulkan.lib");
    std::fs::copy(&source, &alias).expect("failed to stage Vulkan import library alias");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed={}", source.display());
    let version = env!("CARGO_PKG_VERSION");
    let rc_path = output.join("version.rc");
    std::fs::write(&rc_path, render_version_rc(version)).expect("failed to write version.rc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    embed_resource::compile(&rc_path, embed_resource::NONE)
        .manifest_optional()
        .expect("failed to embed Windows version resource");
}
