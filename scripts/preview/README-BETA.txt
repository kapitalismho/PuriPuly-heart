PuriPuly Heart 2.8.2 Beta — RICHARDwuxiaofei Fork

这是独立测试版，不是原作者正式 Release。
安装后打开“PuriPuly Heart Beta”。无需自行安装 Python。
保留正式版，不要覆盖其安装目录。

本包包含本分支的 GPU 语音组件、Vulkan/CPU DLL、VR Overlay、
Gemma 翻译运行库和锁定版本的 Python/Flet 运行环境。
包含各独立进程所需的 app-local Visual C++ 运行库，不改系统运行库。
源码与版本标识在 SOURCE.zip 和 BUILD.json。
不包含数 GB 的识别/翻译模型，首次使用请在下载页勾选需要的模型。
千问 0.6B 可选 ModelScope；Gemma 的社区镜像可能跳回境外。

配置、模型和日志保存在 %LOCALAPPDATA%\PuriPulyHeartBeta。
密钥使用独立的 puripuly-heart.test.desktop-beta 系统存储空间。
不会读取正式版密钥，不附带开发者配置、Key、QQ 凭证或录音。
请自行配置使用后端；已经领取过 QQ 资格的用户不要重复领取。
卸载安装包不会主动清除这份独立用户配置。

功能窗口关闭后可在日志页查看 ModelDownload、QuickTranslate、AudioTest。
核显被枚举不代表一定能运行大型模型；实际兼容性与驱动、内存有关。
游戏内显示、真实语音/VR 与完整模型运行仍需实机验收。
反馈时提供 BUILD.json 中的提交号和脱敏错误日志，不要发送 API Key。
