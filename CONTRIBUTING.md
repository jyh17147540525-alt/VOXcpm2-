# 贡献指南

感谢你对 VOXcpm2 的关注！我们欢迎所有形式的贡献，包括但不限于：报告 Bug、提出功能建议、改进文档、提交代码。

## 行为准则

请保持友善、尊重他人。我们希望社区对每一位贡献者都友好包容。

## 如何贡献

### 报告 Bug

1. 先搜索 [Issues](../../issues) 确认是否已有相同问题；
2. 使用「Bug 报告」模板创建新 Issue；
3. 尽量提供：运行环境（OS / Python 版本 / GPU 型号）、复现步骤、期望结果与实际结果、相关日志。

### 提出功能建议

使用「功能建议」模板，说明需求背景、期望行为和可能的实现思路。

### 提交代码（Pull Request）

1. **Fork** 本仓库，克隆到你本地；
2. 基于 `main` 分支创建新分支：`git checkout -b feat/your-feature`；
3. 编写代码与测试，确保改动可运行；
4. 提交前自检：`python -m py_compile` 校验语法，运行 `examples/test_infer.py` 做冒烟测试；
5. 提交并推送，发起 Pull Request，填写 PR 模板。

## 开发环境

```bash
# 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
# CUDA 版 PyTorch
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

> 模型权重需单独下载，见 [README](./README.md#4-下载预训练权重)。

## 代码规范

- 遵循 [PEP 8](https://pep8.org/)；
- Python 代码使用 4 空格缩进；
- 变量与函数命名清晰，中文注释说明关键逻辑；
- 新增功能请补充必要的说明文档；
- **不要提交**：模型权重、`credentials.json`、`env/`、缓存文件、生成的音频（这些已加入 `.gitignore`）。

## 提交信息规范

建议使用约定式提交（Conventional Commits）：

```
feat: 新增音色包导出功能
fix: 修复长文本音色漂移
docs: 更新安装说明
refactor: 重构音频后处理引擎
```

## 目录约定

```
server.py                主服务
audio_edit.py            音频后处理引擎
voice_packs.py           音色包管理
voice_clone/             克隆增强工具包
scripts/                 启动脚本
examples/                示例脚本
```

- 与语音克隆核心逻辑相关的改动放在 `voice_clone/` 或根目录对应模块；
- 独立的小工具/脚本放在 `examples/`；
- 跨平台的启动脚本放在 `scripts/`。

## 许可

贡献的代码默认以 [Apache-2.0](./LICENSE) 许可发布。提交即表示你同意该许可条款。

再次感谢你的贡献！
