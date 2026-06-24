# Example data

This folder contains a de-identified in vivo brain CEST example used in the paper reproduction demo.

| File | Description |
|------|-------------|
| `cest_volume.nii.gz` | B0-corrected CEST stack, shape `(128, 128, 97)` |
| `brain_mask.nii.gz` | Binary brain mask, shape `(128, 128)` |
| `offsets_dense.csv` | ppm offsets for each spectral frame |

Anonymization steps:

- Removed patient/study identifiers from filenames and NIfTI header fields
- Replaced the affine matrix with diagonal spacing only (no origin orientation metadata)
