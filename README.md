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
- **Beta module — multi-role dialogue**: a dynamic panel builder for multi-speaker / multi-turn scripts. Mark a speaker with `(@Name)` and an emotion with `(emotion)` (full-width Chinese parentheses such as `（情绪）` are normalized automatically). The UI generates an **independent, collapsible control panel for every single participation** (labelled "角色-第N次参与"), so the same character appearing multiple times gets separate, non-interfering panels. Every panel carries its **own set of 5 sliders — volume, pitch, speed, inter-sentence pause and breath intensity** — adjusted in real time for that turn only, while all other turns keep their settings untouched.
- **Dark / light theme**: the whole UI is built on CSS custom properties, so switching themes repaints instantly. Follows the OS preference by default, remembers your choice in `localStorage`, and can be forced per-URL with `?theme=dark` / `?theme=light` (handy for screenshots and shared links).
- **Neural vocal separation (MDX-NET)**: strips background music from a reference recording using UVR's MDX-NET ONNX weights. Unlike the built-in DSP fallback (REPET-lite / HPSS), a trained model recovers the vocal even where it overlaps the accompaniment — measured **correlation 0.99 vs 0.85** and **+19.3 dB vs +3.9 dB** against a clean reference. Weights download on demand; if absent, the pipeline degrades gracefully to the DSP path.
- **Long-audio auto-transcribe → training samples**: upload a long recording, it is auto-segmented by whisper (with Silero VAD) into 1–30 s clips and each clip is transcribed. Paste the **full verbatim transcript** once and the text is matched onto the segments automatically — no line-by-line editing.
- **Tested**: a layered `pytest` suite covers the transcript-alignment algorithm and the MDX separation pipeline, plus CI for Python syntax and the inlined front-end JavaScript. Quality tests skip cleanly when weights or audio are unavailable. See [Testing](#-testing).
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
│   ├── preprocess.py         #   Denoise / remove background / MDX dispatch / segment fusion
│   ├── mdx_separator.py      #   Neural vocal separation (UVR MDX-NET ONNX inference)
│   ├── transcriber.py        #   Long-audio transcription + transcript alignment
│   ├── training_store.py     #   Training-sample store (text↔audio pairs)
│   ├── trainer.py            #   LoRA fine-tuning runner
│   ├── length_adapter.py     #   Long-audio adaptation
│   ├── synthesis_stab.py     #   Long-text stable synthesis + emotion control
│   └── cli.py                #   CLI entry point
├── config.json               # Model config (voxcpm2 architecture)
├── tokenizer.json            # Tokenizer vocabulary
├── tokenizer_config.json     # Tokenizer config
├── special_tokens_map.json   # Special token mapping
├── scripts/                  # One-click launch scripts (Windows .bat + helper tools)
│   ├── start.bat             #   Launch the service
│   ├── check_inline_js.py    #   Validate the inlined front-end JS (node --check)
│   └── fetch_mdx_models.py   #   Download MDX-NET vocal-separation weights
├── tests/                    # Regression test suite (pytest)
│   ├── fixtures.py           #   Synthetic + real-audio test fixtures & quality metrics
│   ├── test_transcriber_align.py  # Transcript-alignment algorithm
│   └── test_mdx_separator.py      # MDX chunk math, engine contract, separation quality
├── examples/                 # Example scripts (inference self-test / pipeline test / diagnostics)
└── .github/                  # CI workflow + Issue / PR templates
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

**Optional — transcription model (faster-whisper)**

Only needed for *Long-audio transcription*. `faster-whisper` downloads the weights on first use; to pre-seed them offline, place these four files in `models/faster-whisper-small/`:

```
config.json  model.bin  tokenizer.json  vocabulary.txt
# from: https://hf-mirror.com/Systran/faster-whisper-small/tree/main
```

**Optional — vocal-separation model (MDX-NET)**

Only needed for *Removing background music*. See [the section below](#-removing-background-music-neural-vocal-separation); one command fetches it:

```bash
python scripts/fetch_mdx_models.py
```

---

## 🧪 Testing

A regression suite lives in `tests/`. It is split into layers so it stays useful
on machines without weights or a GPU:

```bash
pip install -r requirements-dev.txt
python -m pytest tests -v
```

| Layer | What it covers | Requirements |
|---|---|---|
| **Transcript alignment** | `_fold` / `_text_atoms` normalization, Needleman–Wunsch anchoring, gap-midpoint segment cuts, proportional fallback | `numpy` only |
| **MDX chunk math** | STFT frame formula, the 4-channel `(L_re, L_im, R_re, R_im)` contract, `dim_f = n_fft/2` | `numpy` only |
| **MDX engine contract** | model discovery, ONNX session load, I/O shape `[B, 4, dim_f, dim_t]`, degenerate inputs | weights in `models/mdx/` |
| **Separation quality** | correlation against the clean source, SDR, spectral centroid ("muffled" detector), amplitude fidelity, residual reconstruction | weights **+** a real speech `.wav` |

Tests in the last two layers **skip** (not fail) when weights or audio are absent,
so CI stays green on a bare runner. To exercise the quality layer locally, point
`VOXCPM2_FIXTURE_WAV` at a speech file, or just leave some `.wav`s in `outputs/`.

> **Note on synthetic fixtures.** The separator's quality tests deliberately use
> *real* speech as the vocal source. A purely synthetic harmonic tone stack reads
> as a musical instrument to MDX-NET and gets separated *out* — measuring quality
> against it produces misleading numbers. See the warning in `tests/fixtures.py`.

CI (`.github/workflows/ci.yml`) runs three jobs on Linux + Windows across
Python 3.10–3.12:

1. **tests** — the pytest suite
2. **syntax check** — `compileall` over every Python source
3. **inline JS syntax** — extracts the front-end `<script>` blocks inlined in
   `server.py` and validates them with `node --check`. A JS syntax error there
   makes the *entire* page non-interactive while Python-side checks stay green,
   so this is checked explicitly.

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

See the API section of this document for the common endpoints; the FastAPI app also exposes interactive docs at `/docs` while the service is running.

---

## 🎤 Removing background music (neural vocal separation)

Reference recordings often carry BGM. The pipeline ships **two** strategies and picks the best one available:

| Tier | Engine | Quality vs clean reference | Notes |
|---|---|---|---|
| 1 | **MDX-NET** (UVR ONNX) | corr **0.99** · SDR **+19.3 dB** | Needs weights (65 MB); CPU or CUDA |
| 2 | demucs `htdemucs` | good | Only if `demucs` is installed |
| 3 | REPET-lite / HPSS | corr ~0.85 · SDR +3.9 dB | Pure DSP, always available |

> **Why not just use the DSP path?** REPET-lite and HPSS are *assumption-driven*: they assume the background is predictable and subtract it. When the accompaniment **overlaps** the voice in the time–frequency plane, they attenuate the voice too — heard as a **dull, "cotton-wrapped" timbre**. A trained separator is *data-driven* and can pull the vocal back out of the overlap. This is the difference between `HF retention 0.50` and `0.66`, and between a spectral centroid of `1857 Hz` and `2211 Hz` (clean reference: `2220 Hz`).

**Install the weights:**

```bash
python scripts/fetch_mdx_models.py          # default: UVR-MDX-NET_Main_340 (recommended)
python scripts/fetch_mdx_models.py all      # every listed model
```

Files land in `models/mdx/` (excluded from git — they are ~30–65 MB each). Tries HF-Mirror first, then HuggingFace.

**Use it:**

```python
from voice_clone import preprocess as pp

y, sr = pp.load_audio("song_with_bgm.wav", sr=44100)
vocals = pp.isolate_vocals(y, sr, method="auto")   # 'auto' → MDX if available, else DSP
```

In the web UI this is the **🎤 Keep vocals only (remove BGM)** checkbox on the training-sample and transcription panels. No weights installed? Everything still works — it just falls back to DSP and says so.

---

## 📝 Long-audio transcription → training samples

Turning a long recording into training data is *not* a matter of raising the upload limit — training pairs are short (the packer truncates over-long audio at `max_len`). So the correct move is to **cut the audio into sentences** and transcribe each one.

**Workflow**

1. Upload a recording (up to 10 minutes) on the training page.
2. whisper (`faster-whisper`, CPU int8) segments it with **Silero VAD**, drops non-speech, and discards clips outside 1–30 s.
3. Paste the **full verbatim transcript** and hit *Match transcript to segments*.
4. Review each clip (edit text, audition it), tick the ones you want, and import them as samples.

**How the text matching works**

The naive approach — distributing the transcript across segments in proportion to their duration — assumes a constant speaking rate and drifts badly whenever the pace changes, music plays, or the speaker repeats themselves. Instead:

1. Transcription runs with `word_timestamps=True`, so every word carries a real timestamp. Each word is expanded into **character-level anchors** (one anchor per CJK character; one per Latin word).
2. The transcript is tokenized into the **same granularity** and aligned to those anchors with a **Needleman–Wunsch global edit alignment** (match 0 / substitute 1 / insert-delete 1).
3. Only *exactly matching* pairs become **hard anchors** (transcript character → real timestamp); everything else is linearly interpolated between the two nearest hard anchors, yielding a monotone text→time map.
4. Segment boundaries are the **midpoints of the gaps** between speech regions, so bisecting the time map gives each segment the words it actually contains.

Transcript text that runs past the end of the audio is reported rather than crammed into the last segment, and any segment the transcript cannot cover keeps whisper's original text (flagged so you can drop it). If word timestamps are unavailable the code falls back to the proportional method and tells you.

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
- [UVR (Ultimate Vocal Remover)](https://github.com/Anjok07/ultimatevocalremovergui) — MDX-NET vocal-separation models
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / [OpenAI Whisper](https://github.com/openai/whisper) — transcription and word-level timestamps
- [librosa](https://librosa.org/), [soundfile](https://pypi.org/project/SoundFile/), [SciPy](https://scipy.org/) — audio processing
- [FastAPI](https://fastapi.tiangolo.com/) — web framework
