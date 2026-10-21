# llama.cpp Windows runtime

Source: https://github.com/ggml-org/llama.cpp

Release: https://github.com/ggml-org/llama.cpp/releases/tag/b11537

Build: `b11537`

Commit: `8e2d31e0eb658d0382d3106a61369dff81f0c080`

The Windows CPU and Vulkan release archives are fetched only while building the PuriPuly Heart application. Their pinned sizes and SHA-256 digests are enforced before the `llama-server.exe` runtime and its DLLs are staged. Model files are not part of these packaged runtime inputs.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `llama-b11537-bin-win-cpu-x64.zip` | 19513247 | `a05c2d35ebe9787489277d6e039b391fd006384a6c49d069395f3a7b47186fd2` |
| `llama-b11537-bin-win-vulkan-x64.zip` | 33458882 | `ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe` |

The distribution includes `libomp.dll` and its upstream `LICENSE-LLVM-OpenMP`; the older `libomp140.x86_64.dll` is not the dependency for this build.
