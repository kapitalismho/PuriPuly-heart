# llama.cpp Windows runtime

Source: https://github.com/ggml-org/llama.cpp

Release: https://github.com/ggml-org/llama.cpp/releases/tag/b10423

Build: `b10423`

Commit: `a94d563ed801d1da1b8c2432946de07d0231bb3d`

The Windows CPU and Vulkan release archives are fetched only while building the PuriPuly Heart application. Their pinned sizes and SHA-256 digests are enforced before the `llama-server.exe` runtime and its DLLs are staged. Model files are not part of these packaged runtime inputs.
