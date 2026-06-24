# Self-Supervised Implicit CEST Reconstruction via Physics-Informed Lorentz Encoding

PyTorch implementation for the MICCAI 2026 work:

> **Self-Supervised Implicit CEST Reconstruction via Physics-Informed Lorentz Encoding**

This repository provides code for self-supervised CEST MRI Z-spectrum reconstruction using implicit neural representations (INRs). The method combines multi-resolution hash spatial encoding with a physics-informed Lorentz spectral encoder to model the Z-spectrum continuously in both space and frequency, enabling reconstruction from sparsely sampled ppm offsets without paired ground-truth labels.

## Highlights

- **Self-supervised training** on undersampled Z-spectrum offsets
- **Physics-informed Lorentz encoding (LSE)** for ppm-axis spectral modeling
- **Multi-resolution hash encoding** for spatial coordinates
- Flexible offset split patterns for train/test evaluation

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Data layout

Prepare the following files:

| File | Description |
|------|-------------|
| `data/cest.nii.gz` | CEST volume, shape `(H, W, slices, offsets)` or similar |
| `data/mask.nii.gz` | Binary brain mask |
| `data/offsets_dense.csv` | CSV with an `offset` column listing ppm values |

A de-identified example dataset is provided under `data/example/`. See `data/example/README.md` for details.

## Quick start

Run from the `CEST/` directory (using the bundled example data):

```bash
cd CEST
python train.py \
  --img_path ../data/example/cest_volume.nii.gz \
  --msk_path ../data/example/brain_mask.nii.gz \
  --offset_path ../data/example/offsets_dense.csv \
  --pos_encoder LSE \
  --split_method pattern2 \
  --log_dir ../runs
```

For the full Lorentz-encoding model described in the paper, use `--pos_encoder LSE` (hash spatial encoding + Lorentz ppm encoding). To use hash encoding without Lorentz ppm encoding, add `--pos_encoder MultiresEncoding --ppm_encoding`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

## Offset split patterns

- `pattern1`, `pattern2`, `pattern3`: hold out a subset of ppm offsets for testing

See `CEST/INR_dataset.py` for the exact train/test ppm lists.

## Outputs

Each experiment writes to `runs/<experiment_name>/<timestamp>/`:

- `args.json`: full training configuration
- `logs/`: TensorBoard scalars and images
- `models/`: best checkpoint
- `<experiment_name>.nii.gz`: reconstructed volume

For all-slice reconstruction:

```bash
python train.py --all_slices --output_path ../output/reconstruction.nii.gz
```

## Project structure

```
CEST/
  train.py              # CLI entry point
  setup_model.py        # model/dataloader setup
  INR_dataset.py        # dataset and offset splits
  networks.py           # INR backbone and encoders
  loss_function.py      # optional weighted MSE
  multires_hash_encoding/
  Trainer/
```

## Quantitative CEST fitting (not included)

This repository contains the **INR reconstruction code only**. Quantitative CEST map fitting used in some paper comparisons (e.g., Lorentzian/levenberg-marquardt fitting of Z-spectra) is **not part of this release**, as fitting parameters depend on the specific MR sequence, scanner, and acquisition settings.

For CEST quantitative analysis, we refer readers to the open-source MATLAB toolbox [CEST_EVAL](https://github.com/cest-sources/CEST_EVAL/) from [cest-sources](https://github.com/cest-sources/CEST_EVAL/). If you use that pipeline, please cite the original CEST-sources toolbox and follow its GPL license terms.
