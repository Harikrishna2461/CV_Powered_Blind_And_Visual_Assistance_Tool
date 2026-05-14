# Backend dependency policy

This folder contains two requirements files to support both day-to-day development and exact reproducibility.

Files
- `requirements.txt` - minimal top-level packages for development (small and human-maintainable).
- `requirements-pinned.txt` - full pip freeze snapshot for exact reproduction (used in CI or releases).

Installation workflows

1) Development (minimal)

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

2) Reproducible install (pinned)

```cmd
python -m venv .venv_pinned
.venv_pinned\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-pinned.txt
```

3) Hybrid (recommended): install top-level packages but constrain versions using the pinned file

```cmd
python -m venv .venv_constrained
.venv_constrained\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c requirements-pinned.txt
```

Special notes
- `torch` / `torchvision` / `torchaudio` are platform/CUDA-specific and may be pinned in `requirements-pinned.txt` with `+cuXXX` tags. Install them manually following the official PyTorch instructions for your platform if needed.
- `TTS` is omitted from the minimal list due to historical numpy version conflicts. Install it separately if required: `python -m pip install TTS==0.22.0`.
- The runtime import name for `groundingdino-py` is `groundingdino`.

Updating pinned snapshot
- After intentional dependency upgrades in development, update the pinned snapshot:

```cmd
python -m pip freeze > requirements-pinned.txt
```

