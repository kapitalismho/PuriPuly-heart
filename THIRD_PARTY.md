# Third-Party Software

This document lists the major open-source software components directly used by PuriPuly Heart.

| Priority | Component | Version | License | Official Repository | Purpose and Usage |
|---:|---|---|---|---|---|
| 1 | **soxr (Python-SoXR)** | 1.1.0 | LGPL-2.1-or-later | https://github.com/dofuuz/python-soxr | Audio resampling / used as a runtime library |
| 2 | **PyInstaller** | 6.21.0 | GPL-2.0-or-later WITH Bootloader-exception | https://github.com/pyinstaller/pyinstaller | Windows executable packaging / used as a build tool |
| 3 | **sounddevice** | 0.5.5 | MIT | https://github.com/spatialaudio/python-sounddevice | Audio device input and output / used as a runtime library |
| 4 | **ONNX Runtime** | 1.28.0 | MIT | https://github.com/microsoft/onnxruntime | ONNX model inference / used as a runtime library |
| 5 | **llama.cpp** | b10423 (`a94d563ed801d1da1b8c2432946de07d0231bb3d`) | MIT | https://github.com/ggml-org/llama.cpp | Local translation model inference / bundled as a local CPU and Vulkan runtime |
| 6 | **sherpa-onnx + sherpa-onnx-core** | 1.13.4 | Apache-2.0 | https://github.com/k2-fsa/sherpa-onnx | Local speech recognition / used as runtime libraries |
| 7 | **Flet + flet-desktop** | 0.86.1 | Apache-2.0 | https://github.com/flet-dev/flet | Desktop user interface / used as the application UI framework |
| 8 | **transcribe-cpp** | 0.1.3 | MIT | https://github.com/FluidInference/transcribe-rs | GPU speech transcription / used as a Rust library with the Vulkan backend |
| 9 | **Hono** | 4.12.12 | MIT | https://github.com/honojs/hono | Broker HTTP service / used as the web framework |
| 10 | **OpenVR Runtime** | v2.15.6 | BSD-3-Clause | https://github.com/ValveSoftware/openvr | SteamVR integration / bundled as the OpenVR client runtime |

## Additional Notices

The table above is a summary of the major direct dependencies for the competition submission.

Full third-party license notices and redistribution details are available in
`src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`.