# Code Release Manifest

Included:

- Python source code for the main model, ablations, baselines, and speaker-probe diagnostics.
- Modular `ser_cbm/` package used by all runner scripts; CREMA-D and IEMOCAP are both wired in.
- Minimal dependency list in `requirements.txt`.
- Reproduction notes and command examples in `README.md`.
- `.gitignore` rules that keep local data and generated artifacts out of the repository.

Excluded:

- CREMA-D and IEMOCAP audio.
- Generated outputs, checkpoints, cached embeddings/features, predictions, and result tables.
- Paper figures and rendering scripts.
- Local virtual environments and machine-specific files.

Before uploading to GitHub, initialize git from this folder:

```bash
git init
git add .
git commit -m "Initial code release"
```
