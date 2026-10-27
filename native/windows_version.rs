fn version_tuple(version: &str) -> [u16; 4] {
    let mut parts = [0; 4];
    for (index, part) in version.trim().split('.').enumerate() {
        assert!(index < parts.len(), "version must have 1-4 numeric parts: {version:?}");
        parts[index] = part
            .parse::<u16>()
            .unwrap_or_else(|_| panic!("non-numeric version part {part:?} in {version:?}"));
    }
    parts
}

pub fn render_version_rc(version: &str, description: &str, internal: &str) -> String {
    let [major, minor, patch, build] = version_tuple(version);
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
            "            VALUE \"CompanyName\", \"salee\"\r\n",
            "            VALUE \"FileDescription\", \"{description}\"\r\n",
            "            VALUE \"FileVersion\", \"{version}\"\r\n",
            "            VALUE \"InternalName\", \"{internal}\"\r\n",
            "            VALUE \"OriginalFilename\", \"{internal}.exe\"\r\n",
            "            VALUE \"ProductName\", \"PuriPuly <3\"\r\n",
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
        description = description,
        internal = internal,
        version = version,
    )
}
