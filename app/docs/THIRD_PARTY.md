# 内置源码与第三方许可记录

根目录 LICENSE 保留 RVC 原始 MIT 版权声明；各组件继续适用自己的许可证。角色权重、索引、预训练模型、用户音视频、Python/CUDA 运行时和原生二进制均不进入源码仓库。

## 音乐分离源码

| 本地目录 | 发布包依据 | 来源 | 本地许可证 |
| --- | --- | --- | --- |
| `tools/pymss/` | `pymss 2.0.14` 的安装元数据 | https://github.com/pymss-project/pymss | `tools/pymss/LICENSE`，KitsuneX07，2025，MIT |
| `tools/pymss_core/` | `pymss-core 0.1.4` 的安装元数据 | https://github.com/pymss-project/pymss-core | `tools/pymss_core/LICENSE`，KitsuneX07，2025，MIT |

本机两个发行包的 METADATA 声明 `License-Expression: MIT`，包内许可证与这些本地 LICENSE 一致。工作台使用内置副本，并有本地适配；requirements 中对应包还提供其运行依赖，不能假设升级 PyPI 包会替换内置代码。

`pymss_core/modules/` 包含 Bandit、Bandit v2、BS-Roformer、Demucs、SCNet、MDX23C、Apollo、Look2Hear、UVR 及 MLX 实现。这些文件随上述包获得，本地副本未完整保留各架构最初来源的 commit 与单独许可证清单。因此这里只记录可核对的发行包依据，不编造架构级 commit 或把包级声明视为所有外部代码的独立权利核验。原有源码及许可证保留，模型权重不上传。

工作台的 `tools/uvr5/bsroformer.py` 保留 `https://github.com/ZFTurbo/` 来源注释；相关实现参考见 [SOURCES.md](SOURCES.md)。

## Rubber Band 与原生工具

- `tools/pitch_shift/rubberband-src/`：以 Git 子模块固定 Rubber Band `v4.0.0`，源 commit `1d95888bec3ae0a17c0c4af791810d5a63f6bc35`，GPL-2.0-or-later；递归克隆会取得完整对应源码及 `COPYING`。
- 内嵌 KissFFT 与 Speex：保留 `src/ext/kissfft/COPYING`、`src/ext/speex/COPYING`。
- 工作台桥接代码：`rubberband_bridge.cpp`；编译方式见 [BUILD.md](../tools/pitch_shift/BUILD.md)。GPL 组件不由根目录 MIT 许可证覆盖。
- MinGW winpthreads：保留 `WINPTHREADS-COPYING.txt`。本机使用的 `libwinpthread-1.dll` 为 52,224 字节，SHA-256 为 `5bbef249a0d00e2d32c699d0bbe89f714ebeb872b3990a5cbeccb1d89f63e5e8`；构建工具线索为 MinGW GCC 8.1.0。原始 DLL 来源构建记录未完整保留，此校验只识别当前文件，不代表可复现的来源证明。DLL 不进入源码主分支。
- FFmpeg：不上传二进制。`tools/media/manifest.json`、`download.json`、`SOURCE.txt`、构建信息与 `LICENSE.txt` 保留当前已验证 GPL 构建的版本、来源和校验值。

## 模型与其余依赖

Whisper large-v3-turbo 保留官方模型仓库 MIT 许可证和固定 revision/文件校验清单；权重需从原来源另行下载。角色音色模型来源、HuBERT、RMVPE、RVC、Bili23 与其他依赖见 [SOURCES.md](SOURCES.md)。本项目不复制 Bili23 的登录状态，也不声称拥有用户音视频或角色模型。
