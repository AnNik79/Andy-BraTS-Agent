# Initial prototype verification record

This records the earlier synthetic-only implementation check. For the subsequent real BraTS audit and CNN run, see [real_data_workflow.md](real_data_workflow.md) and `outputs/brats2023_gli_audit/report.json`.

Verified on CPU with Python 3.12.7, PyTorch 2.10.0, nibabel 5.4.2, MONAI 1.6.0, and einops 0.8.2. CUDA and MPS were not available for execution.

- **22 unit tests passed.** Coverage includes NIfTI loading, classic/modern modality aliases, CSV manifests, unlabeled inference input, spatial alignment rejection, source label validation, patient split overlap rejection, normalization, crop/pad reconstruction, geometric targets, all four expert output shapes and loss gradients, entropy/margin, MC dropout, disagreement, controller weight normalization and abstention, gate gradients/size, overlapping patch reconstruction, metrics/HD95, disagreement/error enrichment, cache/checkpoint leakage checks, reasoning fallback, and original-space label export.
- **Synthetic end-to-end run passed** at `outputs/smoke_final/`: eight artificial 16³ multimodal NIfTI patients; three expert-training, two controller-training, two validation, and one test patient. Each expert and the controller trained for one epoch. The initial CNN/transformer evaluation ran before specialist training. Validation and test reports contain all seven requested methods.
- **MONAI Swin UNETR forward and backward passed** on a synthetic `[1,4,64,64,64]` input, producing `[1,4,64,64,64]` class probabilities and compact features, with nonzero-connectivity gradient to the segmentation head. This is a single-step architecture check, not a Swin training experiment.
- **Unlabeled-patient CLI inference passed** using a copied synthetic patient with its segmentation omitted.
- **Artifact inspection passed:** 14 exported NIfTI maps for the synthetic test patient retain its original 16³ dimensions and affine; exported expert weights sum to one within `1e-6`. The visualization was inspected and includes MRI, ground truth, four experts, final segmentation, disagreement, and expert weight maps.
- Source compilation and Git whitespace checks passed.

The session used existing scientific packages plus locally downloaded nibabel/MONAI/einops under ignored `.venv-deps/`. The exact software-check commands were:

```bash
PYTHONPATH=src:.venv-deps MPLCONFIGDIR=/private/tmp/brats-mpl pytest -q
PYTHONPATH=src:.venv-deps MPLCONFIGDIR=/private/tmp/brats-mpl \
  python scripts/smoke_test.py --output outputs/smoke_final
PYTHONPATH=src:.venv-deps \
  python scripts/audit_dataset.py --config outputs/smoke_final/synthetic.yaml
PYTHONPATH=src:.venv-deps MPLCONFIGDIR=/private/tmp/brats-mpl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/run_inference.py --config outputs/smoke_final/synthetic.yaml \
  --patient outputs/smoke_final/unlabeled_case
```

For normal use, install into the virtual environment as documented in README.md; these session-specific environment variables are unnecessary. Use a new empty output directory to repeat the synthetic workflow.

No real BraTS dataset was found or used. Real CNN/transformer error diversity, disagreement/error association, controller benefit, convergence, GPU/MPS execution, large-volume throughput, and research generalization remain unmeasured. No synthetic scores are presented as BraTS results. The code is ready for configuring and auditing a real dataset, not for claiming the research hypotheses have been established.
