# CFGS

CFGS is the public-release codebase for the paper's RF spectrum synthesis experiments. This repository keeps only the minimal training, inference, rendering, and CUDA extension sources needed to reproduce the core workflow.

## Repository Scope

- Included: `train.py`, `inference.py`, `arguments/`, `scene/`, `gaussian_renderer/`, `utils/`, `submodules/`
- Included: `dataset/` for dataset construction and quick verification
- Not included: real datasets, checkpoints, logs, build outputs, IDE metadata, internal planning files

## Environment

Create the conda environment and install the CUDA extensions:

```bash
conda env create -f environment.yml
conda activate cfgs

pip install -e ./submodules/simple-knn
pip install -e ./submodules/complex-gaussian-tracer
```

## Dataset Layout

We provide a method for generating the CFGS dataset.
And a small dataset is included to help quickly verify the code.

Please place the generated dataset, or the provided small verification dataset, under:

```text
dataset/<dataset_name>/
```

The current loader expects at least:

```text
dataset/<dataset_name>/
|- spectrum/
|- tx_pos.csv
|- gateway_info.yml
\- freq.txt
```

Optional user-provided split files can be passed through `--train_index_path` and `--test_index_path`.

Derived artifacts such as generated split indices and initialized `points3D.ply` are written under:

```text
logs/<dataset_name>/<exp_name>/dataset_artifacts/
```

This avoids modifying the raw dataset directory.

## Training

Basic training:

```bash
python train.py --dataset <dataset_name>
```

Training outputs are stored under:

```text
logs/<dataset_name>/<exp_name>/
```

Examples with common parameters:

Use a custom experiment name:

```bash
python train.py --dataset <dataset_name> --exp_name cfgs_baseline
```

Use custom dataset and log roots:

```bash
python train.py \
  --dataset <dataset_name> \
  --input_data_folder /path/to/datasets \
  --log_base_folder /path/to/logs
```

Train with explicit split files:

```bash
python train.py \
  --dataset <dataset_name> \
  --train_index_path splits/train_index.txt \
  --test_index_path splits/test_index.txt
```

Regenerate the initialization point cloud and change the initialization mode:

```bash
python train.py \
  --dataset <dataset_name> \
  --gene_init_point \
  --point_init_mode random \
  --voxel_size_scale 1.2
```

Resume training from a checkpoint:

```bash
python train.py \
  --dataset <dataset_name> \
  --exp_name cfgs_baseline \
  --iterations 30000 \
  --start_checkpoint logs/<dataset_name>/cfgs_baseline/chkpnt30000.pth
```

Useful parameters:

- `--dataset`: dataset name under `dataset/` or under `--input_data_folder`
- `--exp_name`: experiment name and output subdirectory
- `--input_data_folder`: dataset root directory
- `--log_base_folder`: output root directory
- `--iterations`: total number of training iterations
- `--start_checkpoint`: checkpoint path for resuming training
- `--train_index_path`: custom training split file
- `--test_index_path`: custom test split file
- `--point_init_mode`: point initialization mode, currently `cube` or `random`
- `--voxel_size_scale`: scale factor for initialization voxel size
- `--data_device`: runtime device such as `cuda:0`

## Inference

```bash
python inference.py --dataset <dataset_name> --exp_name <exp_name>
```

If `--start_checkpoint` is not provided, inference first looks for `chkpnt<iterations>.pth`, then falls back to the numerically latest checkpoint in the experiment directory.


## Acknowledgments

This codebase is adapted from GSRF ([nesl/GSRF](https://github.com/nesl/GSRF)).

It also builds upon 3D Gaussian Splatting (3DGS) by the GraphDECO research group at Inria ([graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)).
