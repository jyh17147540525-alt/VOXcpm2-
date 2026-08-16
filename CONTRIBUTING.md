# Contributing Guide

Thank you for your interest in VOXcpm2! We welcome all kinds of contributions, including bug reports, feature suggestions, documentation improvements, and code.

## Code of conduct

Please be kind and respectful to others. We want the community to be welcoming and inclusive for every contributor.

## How to contribute

### Report a bug

1. Search [Issues](../../issues) to see if the same problem has already been reported.
2. Open a new Issue using the "Bug report" template.
3. Provide as much detail as possible: environment (OS / Python version / GPU model), reproduction steps, expected vs. actual behavior, and relevant logs.

### Suggest a feature

Use the "Feature request" template, and describe the background, expected behavior, and any implementation ideas.

### Submit code (Pull Request)

1. **Fork** this repository and clone it locally.
2. Create a branch from `main`: `git checkout -b feat/your-feature`.
3. Write code and tests; make sure your changes run.
4. Self-check before committing: run `python -m py_compile` and `examples/test_infer.py` as a smoke test.
5. Commit, push, and open a Pull Request using the PR template.

## Development environment

```bash
# Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# CUDA build of PyTorch
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

> Model weights must be downloaded separately — see [README](./README.md#4-download-pretrained-weights).

## Code style

- Follow [PEP 8](https://pep8.org/).
- Use 4-space indentation for Python.
- Name variables and functions clearly; keep comments concise.
- Document new features where appropriate.
- **Do not commit**: model weights, `credentials.json`, `env/`, cache files, or generated audio (these are already in `.gitignore`).

## Commit message convention

Use Conventional Commits:

```
feat: add voice pack export
fix: fix long-text timbre drift
docs: update installation instructions
refactor: refactor the audio post-processing engine
```

## Directory conventions

```
server.py                Main service
audio_edit.py            Audio post-processing engine
voice_packs.py           Voice pack management
voice_clone/             Voice-clone toolkit
scripts/                 Launch scripts
examples/                Example scripts
```

- Core voice-cloning logic goes in `voice_clone/` or the corresponding root modules.
- Standalone tools/scripts go in `examples/`.
- Cross-platform launch scripts go in `scripts/`.

## License

Contributions are released under the [Apache-2.0](./LICENSE) license by default. Submitting a contribution means you agree to those terms.

Thank you again for contributing!
