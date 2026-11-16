# llama.cpp Windows runtime

Source: https://github.com/ggml-org/llama.cpp

Release: https://github.com/ggml-org/llama.cpp/releases/tag/b11537

Build: `b11537`

Commit: `8e2d31e0eb658d0382d3106a61369dff81f0c080`

The build downloads and verifies the pinned CPU/Vulkan archives, then stages `llama-server.exe` and its DLLs. Model files are separate.

Managed launches run offline with explicit app-owned settings. They ignore inherited `LLAMA_ARG_*`, Hugging Face tokens, and llama.cpp configuration.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `llama-b11537-bin-win-cpu-x64.zip` | 19513247 | `a05c2d35ebe9787489277d6e039b391fd006384a6c49d069395f3a7b47186fd2` |
| `llama-b11537-bin-win-vulkan-x64.zip` | 33458882 | `ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe` |

The distribution includes `libomp.dll` and `LICENSE-LLVM-OpenMP`.
