# VOXcpm2

A ready-to-use local voice cloning and text-to-speech (TTS) service. Built on top of [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) (`voxcpm 2.0.3`), it wraps the core model with a complete web application and a voice-cloning toolchain, so voice-cloning enthusiasts can **run it out of the box** and contribute easily.

> This repository contains only the **application-layer code and config files**. The pretrained model weights are large (~4.7 GB) and must be downloaded separately (see below).

---

## ✨ Features

- **Three synthesis modes**
  - **Design**: zero-shot TTS — text to speech without any reference audio. `(design text in parentheses)` is automatically stripped.
  - **Clone**: upload a reference audio (0.3s–10min) to clone a target voice.
  - **HiFi clone**: reference audio + verbatim transcript for stronger voice restoration.
- **Voice pack management**: extract and save a reference voice as a reusable "voice pack", then reuse it for cloning with one click — no need to re-upload long audio. Supports **drag-and-drop video import** (voice is extracted automatically).
- **Audio editing (post-processing engine)**
  - Independent pitch / speed / volume control (time-domain WSOLA algorithm, no phase-vocoder artifacts).
  - Emotion presets (happy / sad / serious / gentle / angry, etc.) — emotion only changes prosody, never the timbre.
  - Natural pauses, breath sounds, SSML tag parsing.
  - Pronunciation correction (polyphone detection).
- **Long-text stable synthesis**: automatic sentence splitting, chunk-wise independent generation (reference-anchored, no timbre drift), graded pauses for commas/periods, unified emotion control (neutral & stable by default). Great for audiobooks and long passages.
- **Audio export**: WAV / MP3 / M4A.
- **Beta module — multi-role dialogue**: a dynamic panel builder for multi-speaker / multi-turn scripts. Mark a speaker with `(@Name)` and an emotion with `(emotion)` (full-width Chinese parentheses such as `（情绪）` are normalized automatically). The UI generates an **independent, collapsible control panel for every single participation** (labelled "角色-第N次参与"), so the same character appearing multiple times gets separate, non-interfering panels. Each panel tunes tone / lines / action / emotion / volume in real time while the rest of the cast stays untouched.
- **Web UI**: FastAPI + token auth, one-click login in the browser, built-in player and generation history.

---

## 🖥️ Platform support

The code is cross-platform by design — pure Python, no OS-specific shell commands, `pathlib` for all paths, and ffmpeg discovery that falls back to the bundled `imageio-ffmpeg`. However, it has been **tested primarily on Windows (CUDA)**. Other platforms are expected to work but have not been fully verified; feedback and fixes for Linux/macOS are very welcome.

| Platform | Status |
|---|---|
| Windows (CUDA) | ✅ Tested |
| Linux | ⚠️ Expected to work, not yet verified |
| macOS | ⚠️ Expected to work, not yet verified |
| CPU-only | ⚠️ Should work, but slower |

## 🔌 Compatibility with the `voxcpm` package

The code talks to the `voxcpm` model through a small adapter layer (`voice_clone/synthesis_stab.py`). It uses the **public API** (`VoxCPM.from_pretrained`, `model.generate`) as the primary path, and only opts into the optional prompt-cache fast path (the internal `tts_model.build_prompt_cache` / `generate_with_prompt_cache`) when those methods are detected at runtime. If they are absent or fail, generation **falls back cleanly to the public API**, so the project does not hard-depend on voxcpm's private internals.

---

## 📦 Project structure

```
.
├── server.py                 # Main service (FastAPI web + API)
├── audio_edit.py             # Audio post-processing engine (pitch/speed/volume/emotion/breath/SSML)
├── voice_packs.py            # Voice pack management
├── tokenization_voxcpm2.py   # Tokenizer
├── voice_clone/              # Voice-clone enhancement toolkit
│   ├── pipeline.py           #   Reference audio preprocessing pipeline
│   ├── preprocess.py         #   Denoise / remove background / segment fusion
│   ├── length_adapter.py     #   Long-audio adaptation
│   ├── synthesis_stab.py     #   Long-text stable synthesis + emotion control
│   └── cli.py                #   CLI entry point
├── config.json               # Model config (voxcpm2 architecture)
├── tokenizer.json            # Tokenizer vocabulary
├── tokenizer_config.json     # Tokenizer config
├── special_tokens_map.json   # Special token mapping
├── scripts/                  # One-click launch scripts (Windows .bat)
├── examples/                 # Example scripts (inference self-test / pipeline test / diagnostics)
└── .github/                  # Issue / PR templates
```

---

## 🔧 Installation

### 1. Requirements

| Item | Requirement |
|---|---|
| OS | Windows / Linux / macOS |
| Python | 3.10 – 3.12 (3.11 recommended) |
| GPU | NVIDIA recommended (VRAM ≥ 12GB, CUDA 12.x); CPU works but slower |
| Disk | ≥ 10GB free (including model weights) |

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **The CUDA build of PyTorch must be installed separately** (PyPI defaults to CPU):
> ```bash
> pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
> ```

### 4. Download pretrained weights

The model weights are **not included in this repository**. Download them from one of the sources below and place them in the project root:

| File | Size | Description |
|---|---|---|
| `model.safetensors` | ~4.3 GB | Main model weights |
| `audiovae.pth` | ~360 MB | Audio VAE weights |

**Option 1 (recommended, ModelScope)**:

```bash
pip install modelscope
git lfs install
git clone https://www.modelscope.cn/OpenBMB/VoxCPM2.git
# then copy the weight files into the project root
```

**Option 2 (HuggingFace)**:

```bash
pip install huggingface_hub
huggingface-cli download OpenBMB/VoxCPM2 --local-dir .
```

> After downloading, make sure the project root contains `model.safetensors`, `audiovae.pth`, `config.json`, `tokenizer.json`, etc.

---

## 🚀 Usage

### One-click launch (Windows)

```bash
scripts\start.bat
```

Then open `http://localhost:8808` in your browser. The access token is auto-generated in `credentials.json` (in the project root) on first launch.

### Command line

```bash
python server.py
```

Common environment variables:

| Variable | Default | Description |
|---|---|---|
| `VOXCPM_PORT` | `8808` | Service port |
| `VOXCPM_HOST` | `127.0.0.1` | Bind address (set `0.0.0.0` for LAN access) |
| `VOXCPM_HOME` | project root | Directory containing the weights (allows separating weights from code) |
| `VOXCPM_DEVICE` | `auto` | Inference device (`auto` / `cuda` / `cpu`) |
| `HF_HUB_OFFLINE` | — | Set `1` to load local weights offline |

### Inference self-test

```bash
python examples/test_infer.py
```

### Python API

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(".", load_denoiser=False, device="auto")
wav = model.generate(
    text="Hello, welcome to speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
    normalize=True,
)
```

### HTTP API

```bash
# Design mode (zero-shot TTS)
curl -X POST http://127.0.0.1:8808/api/generate \
  -H "X-API-Key: YOUR_TOKEN" \
  -F "text=Hello, this is a test" -F "mode=design"

# Clone mode (upload reference audio)
curl -X POST http://127.0.0.1:8808/api/generate \
  -H "X-API-Key: YOUR_TOKEN" \
  -F "text=Hello" -F "mode=clone" -F "reference=@ref.wav"
```

See [docs/API.md](docs/API.md) for the full API reference (if present).

---

## 🎓 Training and inference

This repository is an **inference and cloning application layer** — the model itself is trained by OpenBMB, and this repository does not involve training.

**Inference pipeline (core of this repo)**:

1. **Load**: `VoxCPM.from_pretrained()` loads `model.safetensors` + `audiovae.pth`.
2. **Reference processing** (clone mode): `voice_clone.pipeline` denoises / removes background / fuses long-audio segments into a clean representative reference.
3. **Generate**: `model.generate()` or `generate_with_prompt_cache()` generates chunk by chunk.
4. **Post-process**: `audio_edit` applies pitch / speed / volume / emotion / breath, etc.
5. **Concatenate**: long text is joined per sentence with graded pauses, keeping timbre consistent and emotion stable.

> To train or fine-tune the VoxCPM model, refer to the upstream [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

---

## 🤝 Contributing

Issues and pull requests are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

The code in this repository is licensed under [Apache-2.0](LICENSE). The core model `voxcpm` and its pretrained weights belong to [OpenBMB](https://github.com/OpenBMB/VoxCPM) and are used under its Apache-2.0 license.

## 🙏 Acknowledgements

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — the underlying TTS model and pretrained weights
- [librosa](https://librosa.org/), [soundfile](https://pypi.org/project/SoundFile/), [SciPy](https://scipy.org/) — audio processing
- [FastAPI](https://fastapi.tiangolo.com/) — web framework
