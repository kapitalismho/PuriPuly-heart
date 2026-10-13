# PuriPuly desktop preview / Windows 测试版

This is a portable, bundled CPython + pinned Flet desktop preview, **not an official native installer or Release**. The original native build requires additional pinned Flutter/Python toolchains unavailable in this environment. Runtime packages come from the existing `uv.lock`; Flet and llama.cpp binaries retain upstream checksum/provenance validation.

## 使用

1. 完整解压测试包到一个新目录，不要覆盖正式安装。
2. 双击 `PuriPulyHeartPreview.exe`。不需要另外安装 Python，不弹 CMD。
3. 标题 `[TEST]` 表示独立测试配置。配置、日志、模型写入包旁边的 `preview-data`；系统密钥存储使用 `puripuly-heart.test.desktop-preview`，不读取正式 Key，也不从环境变量导入生产 Key。
4. 标题栏下载图标：选择 Hugging Face、`hf-mirror.com` 或本地导入。导入可选择两个固定 GGUF 文件或所在目录；文件名、大小和 SHA256 必须完全匹配。镜像若跳到境外会明确拒绝，请切换官方源或手动导入。
5. 标题栏翻译图标：打开独立快捷窗口；Enter 发送，Shift+Enter 换行。沿用主窗口翻译设置；请先配置测试后端并启用翻译。保持置顶可关闭；X 或“隐藏”不退出主程序。独占全屏不保证覆盖。
6. 设置 → 常规 → 麦克风测试：录制 10 秒后回放采集 PCM 或声道转换/重采样后的 VAD/STT 前 PCM。可停止、重录。基础测试完全本地，不调用识别/翻译服务，不保存 WAV。
7. 关闭功能窗口后，到日志页查看 `[ModelDownload]`、`[QuickTranslate]`、`[AudioTest]`。日志文件在 `preview-data/puripuly_heart.log`，沿用现有轮转。

没有捆绑数 GB 的模型，也没有复制个人配置、QQ 身份、激活凭证、API Key 或录音。测试配置仍需您自行设置可用的测试后端；不要再次领取已有一次性 QQ 额度。所有测试包副本默认共享这一测试密钥命名空间，但与正式程序隔离。不要把运行后生成的 `preview-data` 分享给他人。

## Build

From a clean feature checkout with `.venv` populated by `uv sync --locked --extra dev --extra build`, prepare existing pinned Flet and llama.cpp resources with repository build helpers. Then:

```powershell
.venv/Scripts/python.exe scripts/preview/build_portable.py --output D:/path/to/fresh-preview-directory
```

The output directory must not exist. `BUILD.json` identifies the exact branch/commit. Packaging copies only application/runtime inputs, never a user's profile. This preview does not validate official Setup, VR native overlay, real ASR models, or game integration by itself.
