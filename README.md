# VOXcpm2

一个开箱即用的本地语音克隆（Voice Cloning）与文本转语音（TTS）服务。基于 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)（`voxcpm 2.0.3`）核心模型，在其之上封装了一套完整的 Web 应用与克隆增强工具链，让语音克隆爱好者能**点开即用**，也便于二次开发与贡献。

> 本仓库仅包含**应用层代码与配置文件**，预训练模型权重体积较大（约 4.7 GB），需按下方说明单独下载。

---

## ✨ 功能特性

- **三种合成模式**
  - **设计模式（Design）**：零样本 TTS，无需参考音频，直接文字转语音，支持 `（括号内设计文本）` 自动剥离。
  - **克隆模式（Clone）**：上传参考音频（0.3s–10min）克隆目标音色。
  - **极致克隆（HiFi）**：参考音频 + 逐字文本，实现更强的音色还原。
- **音色包管理**：把参考音色提取、保存为可复用的"音色包"，后续克隆一键选用，无需重复上传长音频；支持**视频拖拽导入**（自动提取人声）。
- **编辑音频（后处理引擎）**
  - 音调 / 语速 / 音量独立调节（时域 WSOLA 算法，无相位声码器金属感）。
  - 情绪预设（高兴/悲伤/严肃/温柔/愤怒等），情绪只改韵律、不改变音色。
  - 自然停顿 + 换气声、呼吸声合成、SSML 标签解析。
  - 发音校正（多音字检测）。
- **长文本稳定合成**：自动分句、块级独立生成（参考锚定，杜绝音色漂移）、逗号/句号分级停顿、情绪统一控制（默认中性平稳），适合有声书、长段朗读。
- **音频导出**：WAV / MP3 / M4A 多格式导出。
- **Web 界面**：FastAPI + 令牌验证，浏览器一键登录，内置播放器与生成历史。

---

## 📦 项目结构

```
.
├── server.py                 # 主服务（FastAPI Web + API）
├── audio_edit.py             # 音频后处理引擎（音调/语速/音量/情绪/呼吸/SSML）
├── voice_packs.py            # 音色包管理
├── tokenization_voxcpm2.py   # 分词器
├── voice_clone/              # 克隆增强工具包
│   ├── pipeline.py           #   参考音频预处理管线
│   ├── preprocess.py         #   降噪 / 去背景 / 分段融合
│   ├── length_adapter.py     #   长音频适配
│   ├── synthesis_stab.py     #   长文本稳定合成 + 情绪控制
│   └── cli.py                #   命令行入口
├── config.json               # 模型配置（voxcpm2 架构）
├── tokenizer.json            # 分词器词表
├── tokenizer_config.json     # 分词器配置
├── special_tokens_map.json   # 特殊 token 映射
├── scripts/                  # 一键启动脚本（Windows .bat）
├── examples/                 # 示例脚本（推理自检 / 管线测试 / 诊断）
└── .github/                  # Issue / PR 模板
```

---

## 🔧 安装步骤

### 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows / Linux / macOS |
| Python | 3.10 – 3.12（推荐 3.11） |
| GPU | 推荐 NVIDIA（显存 ≥ 12GB，CUDA 12.x）；CPU 可运行但较慢 |
| 磁盘 | 预留 ≥ 10GB（含模型权重） |

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **CUDA 版 PyTorch 需单独安装**（PyPI 默认是 CPU 版）：
> ```bash
> pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
> ```

### 4. 下载预训练权重

模型权重**不包含在本仓库中**，请从以下任一来源下载，并放到项目根目录：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | ~4.3 GB | 主模型权重 |
| `audiovae.pth` | ~360 MB | 音频 VAE 权重 |

**方式一（推荐，ModelScope）**：

```bash
pip install modelscope
# 下载到本地后，把权重文件复制到项目根目录
git lfs install
git clone https://www.modelscope.cn/OpenBMB/VoxCPM2.git
```

**方式二（HuggingFace）**：

```bash
pip install huggingface_hub
huggingface-cli download OpenBMB/VoxCPM2 --local-dir .
```

> 下载完成后，请确认项目根目录下存在 `model.safetensors`、`audiovae.pth`、`config.json`、`tokenizer.json` 等文件。

---

## 🚀 使用方法

### 一键启动（Windows）

```bash
scripts\start.bat
```

启动后浏览器访问 `http://localhost:8808`，访问令牌会自动生成在项目根目录的 `credentials.json` 中（首次启动时创建）。

### 命令行启动

```bash
python server.py
```

常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOXCPM_PORT` | `8808` | 服务端口 |
| `VOXCPM_HOST` | `127.0.0.1` | 监听地址（局域网访问改 `0.0.0.0`） |
| `VOXCPM_HOME` | 项目根目录 | 权重所在目录（支持权重与代码分离） |
| `VOXCPM_DEVICE` | `auto` | 推理设备（`auto`/`cuda`/`cpu`） |
| `HF_HUB_OFFLINE` | — | 设为 `1` 可离线加载本地权重 |

### 命令行推理自检

```bash
python examples/test_infer.py
```

### 直接调用 Python API

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(".", load_denoiser=False, device="auto")
wav = model.generate(
    text="你好，欢迎使用语音合成。",
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=True,
)
```

### 调用 HTTP API

```bash
# 设计模式（零样本 TTS）
curl -X POST http://127.0.0.1:8808/api/generate \
  -H "X-API-Key: 你的令牌" \
  -F "text=你好，这是测试" -F "mode=design"

# 克隆模式（上传参考音频）
curl -X POST http://127.0.0.1:8808/api/generate \
  -H "X-API-Key: 你的令牌" \
  -F "text=你好" -F "mode=clone" -F "reference=@ref.wav"
```

完整 API 说明见 [docs/API.md](docs/API.md)（如存在）。

---

## 🎓 训练与推理流程

本仓库为**推理与克隆应用层**，模型本身由 OpenBMB 团队训练，本仓库不涉及训练。

**推理流程（本仓库核心）**：

1. **加载**：`VoxCPM.from_pretrained()` 加载 `model.safetensors` + `audiovae.pth`；
2. **参考处理**（克隆模式）：`voice_clone.pipeline` 对参考音频做降噪 / 去背景 / 长音频分段融合，得到干净代表参考；
3. **生成**：`model.generate()` 或 `generate_with_prompt_cache()` 分块生成；
4. **后处理**：`audio_edit` 应用音调 / 语速 / 音量 / 情绪 / 呼吸等；
5. **拼接**：长文本按句子 + 分级停顿自然拼接，保证音色一致、情绪平稳。

> 若需训练或微调 VoxCPM 模型，请参考上游 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 的说明。

---

## 🤝 参与贡献

欢迎提交 Issue 与 Pull Request！请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本仓库代码采用 [Apache-2.0](LICENSE) 许可证。核心模型 `voxcpm` 及其预训练权重归 [OpenBMB](https://github.com/OpenBMB/VoxCPM) 所有，使用请遵循其 Apache-2.0 许可。

## 🙏 致谢

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — 底层 TTS 模型与预训练权重
- [librosa](https://librosa.org/)、[soundfile](https://pypi.org/project/SoundFile/)、[SciPy](https://scipy.org/) — 音频处理
- [FastAPI](https://fastapi.tiangolo.com/) — Web 框架
