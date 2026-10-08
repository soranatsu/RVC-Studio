在线安装器使用的简体中文 Inno Setup 翻译文件。

来源：Inno Setup 用户贡献翻译下载页
https://jrsoftware.org/files/istrans/

翻译维护者：Zhenghan Yang (Kira)
翻译项目：https://github.com/kira-96/Inno-Setup-Chinese-Simplified-Translation
许可证：MIT；原文保存在本目录 LICENSE.txt。

文件头保留原维护者和来源说明。在线安装器只引用本目录的副本，
不依赖被 Git 忽略的本机 packaging/vendor 目录。

构建：在 app/ 目录运行 Inno Setup 6 的 ISCC.exe packaging\online_setup.iss。
校验：运行 python packaging\test_online_setup.py；检查固定发布数据、原生下载、
缓存复用、损坏文件拒绝和顺序重组。INSTALL-DATA.json 为固定版本的分块清单。
发布数据总 SHA-256 通过后才启用完整安装器；不修改已有的完整安装包。

原生下载接口：https://jrsoftware.org/ishelp/topic_isxfunc_createdownloadpage.htm
