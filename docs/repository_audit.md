# Initial repository audit

The initial workspace contained only `.git` metadata. There were no source files, project instructions, dependency manifests, tests, datasets, or existing architecture to reuse. The workspace and a shallow scan of nearby home-directory dataset locations did not reveal BraTS data. This is not a claim that every disk location was searched.

Python already provided PyTorch, NumPy, pandas, matplotlib, SciPy, scikit-learn, PyYAML, and pytest. nibabel and MONAI were absent. CUDA and MPS were unavailable to the session. The implementation adds a local Python research package and optional MONAI support; it needs no backend, cloud account, or LLM API.

Implementation followed data loading/preprocessing, independent CNN and transformer, standardized outputs and uncertainty, explicit disagreement, boundary and patch specialists, held-out controller caches, learned gating, comparative evaluation, and structured reasoning. With real data absent, the executable smoke workflow enforces the requested initial CNN/transformer comparison before training specialists and the controller. Real-data training and research conclusions remain pending.

No BraTS version or hidden metadata was inferred. Source label mapping, modality aliases, biological patient identity, and dataset location must be confirmed for a real experiment. See README.md for configuration and commands.
