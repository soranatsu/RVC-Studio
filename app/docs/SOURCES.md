# 声音工作台：来源与引用

本工作台在 RVC 上游代码基础上加入桌面任务协调、智能翻唱、文件整理、混音、字幕和 Bili23 下载连接。保留已有版权声明及对应组件许可证。这里的代码许可不自动适用于角色音色模型或用户输入音视频。

## 角色音色模型

用户指定的模型集合来源：**[Bilibili BV1mqKq6PE49](https://www.bilibili.com/video/BV1mqKq6PE49/)**。

当前显示名称按本机 `configs/model_labels.json` 整理：爱音、灯、乐奈、立希、素世、初华、睦、祥子、海玲、喵梦、莫提斯。模型原文件来自用户已有文件夹，本项目未训练或声称拥有这些角色模型。

该链接的作者名称及逐模型分发许可尚未独立核实，不编造作者或许可证。本源码发布只附来源和导入方法；角色 `.pth`、`.index` 文件不直接加入源码仓库。其他未映射模型保留原文件名，其单独来源尚待核实。

## 上游代码与推理组件

| 组件 | 本项目用途 | 来源与许可依据 |
| --- | --- | --- |
| RVC | 音色转换、实时及文件推理 | [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，MIT；保留根目录 `LICENSE` 及 `MIT协议暨相关引用库协议` |
| HuBERT / ContentVec | 人声特征提取 | [RVC 模型资产](https://huggingface.co/lj1995/VoiceConversionWebUI)、[ContentVec](https://github.com/auspicious3000/contentvec)；沿用原文件许可 |
| RMVPE | 人声音高检测 | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE)、[yxlllc/RMVPE](https://github.com/yxlllc/RMVPE)；RVC 已有实现及预训练模型 |
| BS-Roformer | 精修人声分离 | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)、[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)；本机 BS-Roformer 推理文件保留上游来源注释 |
| UVR / pymss | 快速分离及分离模型支持 | [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)、[pymss-project/pymss](https://github.com/pymss-project/pymss)；许可见 `tools/pymss/LICENSE`、`tools/pymss_core/LICENSE` |
| Whisper large-v3-turbo | 中文、英语、日语、粤语字幕 | [OpenAI 模型](https://huggingface.co/openai/whisper-large-v3-turbo)、[Whisper](https://github.com/openai/whisper)，MIT；固定 revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`；校验记录与许可证在 `assets/asr/whisper-large-v3-turbo/` |
| Transformers | Whisper 本地推理接口 | [Hugging Face Transformers](https://github.com/huggingface/transformers)，运行时 4.49.0；保留包内许可证 |
| FFmpeg | 解码、移调、响度测量与导出 | [FFmpeg](https://ffmpeg.org/)、[BtbN Windows builds](https://github.com/BtbN/FFmpeg-Builds)；当前精修工具为 GPL 构建，版本、二进制校验及对应源码说明见 `tools/media/` |
| Rubber Band | 同速移调、高音恢复 | [Rubber Band](https://breakfastquay.com/rubberband/)，4.0.0、GPL-2.0-or-later；源码、桥接实现、构建方法与许可证见 `tools/pitch_shift/` |
| Bili23 | B站链接下载 | [Bili23-Downloader](https://github.com/ScottSloan/Bili23-Downloader)、[官方 MCP 接口](https://bili23.scott-sloan.cn/doc/mcp-server.html)；调用用户本机已安装程序，不复制账号、登录状态或程序进入源码仓库 |
| FreeSimpleGUI / sounddevice / PyTorch | 桌面界面、音频设备和推理 | 沿用安装包内原始 LICENSE/NOTICE；不将整个 Python/CUDA 运行环境上传源码仓库 |

来源记录不等于对原模型训练数据、字符形象或音视频内容的权利声明。版本、硬件和主观听感验收以本机实际测试报告为准。
