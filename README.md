# Disentangled Affective-Style Concept Bottleneck SER

Code release for speaker-disjoint speech emotion recognition experiments on **CREMA-D** and **IEMOCAP**.

This repository intentionally contains code only. The two datasets, trained checkpoints, cached features, generated tables, and paper figures are not included.

## What Is Included

- `main_egemaps_baseline_deviation_cbm.py` - thin entrypoint for the main experiment.
- `ser_cbm/` - reusable package with config, data loading, feature extraction, model, training, diagnostics, and experiment loop. The package is dataset-aware and picks CREMA-D or IEMOCAP through `cfg.DATASET`.
- `ablation_runner.py` - controlled neural ablations.
- `baseline_comparison_runner.py` - eGeMAPS LR/SVM, Plain CRNN, CBM variants, and summary tables.
- `plain_encoder_speaker_probe.py` - post-hoc speaker probe for the Plain CRNN encoder representation.
- `wav2vec2_frozen_baseline.py` - frozen `facebook/wav2vec2-base` embeddings with linear SER and speaker probe.

## Code Layout

```text
ser_cbm/
  config.py       # experiment config, label maps (CREMA-D 6-way, IEMOCAP 4-way), concept names
  data.py         # CREMA-D + IEMOCAP discovery, audio loading, log-mel/eGeMAPS/librosa features
  concepts.py     # baseline/deviation concept targets
  datasets.py     # PyTorch dataset
  models.py       # CRNN encoder and affective/style CBM
  metrics.py      # SER metrics, calibration, GRL schedule
  training.py     # train/eval loop helpers and fold dataloaders
  diagnostics.py  # speaker probes, intervention diagnostics, result summaries
  experiment.py   # full speaker-disjoint CV experiment
```

## Data

Both datasets must be downloaded and arranged separately. The repository does not redistribute audio.

### CREMA-D

Place wav files under:

```text
AudioWAV/
```

Expected filename format:

```text
1001_DFA_ANG_XX.wav
1001_IEO_HAP_HI.wav
```

### IEMOCAP

Place the standard IEMOCAP release tree under:

```text
iemocap/
  Session1/
    dialog/EmoEvaluation/*.txt
    sentences/wav/<dialog>/<turn>.wav
  Session2/
  ...
  Session5/
```

We follow the canonical 4-class benchmark protocol (`angry`, `happy`, `neutral`, `sad`) and fold `exc` (excited) into `happy`. Speaker identity is `Ses0X_F` or `Ses0X_M`, giving 10 distinct speakers across the five sessions.

`AudioWAV/` and `iemocap/` are gitignored because the datasets are not redistributed with this code release.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA-specific PyTorch installs, follow the official PyTorch wheel selector, then install the remaining requirements.

## Main Runs

The dataset is selected per command with `--dataset {cremad,iemocap}`. Output directories are auto-prefixed with the dataset name when running on IEMOCAP, so CREMA-D outputs are never overwritten.

Run the full proposed model:

```bash
# CREMA-D
python main_egemaps_baseline_deviation_cbm.py

# IEMOCAP: set ser_cbm.CFG.DATASET = "iemocap" before calling, or use the runner scripts below.
```

Run the paper comparison table:

```bash
python baseline_comparison_runner.py --dataset cremad
python baseline_comparison_runner.py --dataset iemocap
```

Run the neural ablations:

```bash
python ablation_runner.py --dataset cremad --variants plain_ser_encoder concept_bottleneck_only dual_branch_no_adversaries dual_branch_orthogonality full no_style_branch
python ablation_runner.py --dataset iemocap --variants plain_ser_encoder concept_bottleneck_only dual_branch_no_adversaries dual_branch_orthogonality full no_style_branch
```

Run the Plain CRNN encoder speaker probe:

```bash
python plain_encoder_speaker_probe.py --dataset cremad
python plain_encoder_speaker_probe.py --dataset iemocap
```

Run the frozen wav2vec2-base baseline:

```bash
python wav2vec2_frozen_baseline.py --dataset cremad
python wav2vec2_frozen_baseline.py --dataset iemocap
```

## Smoke Tests

Tiny settings to confirm that the environment and paths are correct:

```bash
python baseline_comparison_runner.py --quick --models B1_egemaps_lr --dataset cremad
python ablation_runner.py --quick --variants plain_ser_encoder --dataset iemocap
python plain_encoder_speaker_probe.py --quick --max-folds 1 --dataset cremad
python wav2vec2_frozen_baseline.py --quick --dataset iemocap
```

## Outputs

All generated outputs are written to local `*_outputs/` directories and ignored by git. This includes checkpoints, cached features, predictions, CSV summaries, and markdown tables.

For non-CREMA-D runs the runner scripts automatically prepend the dataset name, so the two corpora write to disjoint trees by default:

```text
baseline_comparison_outputs/          # CREMA-D
iemocap_baseline_comparison_outputs/  # IEMOCAP
ablation_outputs/                     # CREMA-D
iemocap_ablation_outputs/             # IEMOCAP
plain_encoder_speaker_probe_outputs/          # CREMA-D
iemocap_plain_encoder_speaker_probe_outputs/  # IEMOCAP
wav2vec2_frozen_outputs/              # CREMA-D
iemocap_wav2vec2_frozen_outputs/      # IEMOCAP
```

## Evaluation Protocol

The experiments use speaker-disjoint `GroupKFold` splits by speaker identity. Reported metrics include UAR, Macro-F1, accuracy, speaker-probe accuracy, uniform/majority chance baselines, intervention diagnostics, and mean/std summaries across folds where applicable.
