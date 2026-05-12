# Dataset

This folder hosts XfreqGS datasets and a generator for building them from a 3D scene.

Two independent concerns live side by side:

- `scene_builder/` — an optional generator that renders RF power angular spectra with Sionna RT + Mitsuba 3. Use it only if you want to build a new dataset.
- `<dataset_name>/` — one folder per dataset, consumed by `train.py` / `inference.py` at the repo root. A reference dataset `scene01/` is included so you can verify the training pipeline without running the generator.

Do not commit raw datasets, generated splits, checkpoints, or experiment outputs. Derived artifacts (train/test indices, initialized point clouds) are written to `logs/<dataset_name>/<exp_name>/dataset_artifacts/` by `train.py` and never modify the dataset directory.

---

## 1. Expected dataset layout

The loader (`scene/dataset_readers.py::readSpectrumImage`) expects:

```text
dataset/<dataset_name>/
├── spectrum/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── tx_pos.csv
├── gateway_info.yml
└── freq.txt
```

**Row alignment.** After sorting `spectrum/*.png` by filename, the i-th image corresponds to row i of both `tx_pos.csv` and `freq.txt` (1-indexed here to match the zero-padded filenames). All three lists must have the same length.

### `spectrum/*.png`
Bartlett power angular spectrum images, `H x W` grayscale (default `90 x 360`). Pixel values are the per-sample magnitude normalized to `[0, 255]` (`uint8`). Filenames use zero-padded indices (`00001.png`, `00002.png`, ...).

### `tx_pos.csv`
Transmitter position per sample, in world coordinates (meters):

```csv
x,y,z
0.0,0.0,1.0
0.0,0.0,1.1
...
```

### `freq.txt`
One frequency per sample, in **GHz**, one value per line:

```text
1
1
...
2.4
2.4
...
```

The loader normalizes by a hard-coded `max_freq = 94.0` GHz (see `scene/dataset_readers.py::readSpectrumImage`). If you go higher, update that constant accordingly.

### `gateway_info.yml`
Single-receiver ("gateway") metadata:

```yaml
dataset_name: scene01
gateway1:
  position: [2.0, 1.0, 0.0]        # meters, world frame
  orientation: [0.0, 0.0, 0.0, 1.0]  # quaternion [x, y, z, w]
```

### Optional splits
You can provide your own train/test index files and pass them via `--train_index_path` / `--test_index_path` to `train.py`. If absent, `train.py` generates a random 80/20 split inside the experiment directory and never writes to the dataset folder.

---

## 2. Using the included reference dataset

`scene01/` is a small 270-sample dataset (10 frequencies x 27 transmitter positions) provided for end-to-end verification. From the repo root:

```bash
python train.py --dataset scene01
```

---

## 3. Building a dataset with `scene_builder/`

`scene_builder/` is a thin wrapper around Sionna RT that:

1. loads a Mitsuba 3 scene (`scene.xml` + `meshes/*.ply`),
2. filters a user-defined 3D transmitter grid (or reads positions from a CSV),
3. runs `PathSolver` per (frequency, transmitter) pair,
4. projects the CFR onto a Bartlett PAS (`theta in [0, pi/2] x phi in [-pi, pi]`),
5. saves PNGs plus `tx_pos.csv`, `freq.txt`, `gateway_info.yml`.

### 3.1 Layout

```text
scene_builder/
├── build_dataset.py          # entry point, reads YAML
├── utils.py                  # scene bounds, grid filter, Bartlett PAS
├── requirements.txt
├── configs/
│   └── scene01.yml           # example config
└── assets/
    └── scene01/              # example scene geometry
        ├── scene.xml
        └── meshes/*.ply
```

### 3.2 Install

Sionna RT and Mitsuba 3 must be available (the root `environment.yml` already provides them). If installing separately:

```bash
pip install -r scene_builder/requirements.txt
```

### 3.3 Reproduce the reference dataset

From `scene_builder/`:

```bash
python build_dataset.py --config configs/scene01.yml
```

Output is written to `dataset/<dataset_name>/` (controlled by `output_root` in the config).

### 3.4 Adapt to your own scene

1. **Bring your geometry.** Place your Mitsuba 3 scene under `scene_builder/assets/<your_scene>/` (an XML referencing `.ply` meshes with ITU material BSDFs). The included `assets/scene01/` is a minimal example.
2. **Copy and edit a config.** Duplicate `configs/scene01.yml` and set:
   - `dataset_name` — output subdirectory name
   - `scene_xml` — path to your scene (relative to the config file)
   - `frequencies_hz` — list of carrier frequencies in Hz
   - `rx.position` / `rx.orientation_euler` — receiver pose (Euler angles use Sionna's intrinsic ZYX convention in radians; the quaternion in `gateway_info.yml` is derived from them)
   - `tx_array` / `rx_array` — antenna geometry (`sionna.rt.PlanarArray` parameters)
   - `tx_positions` — either `csv: path/to/positions.csv` (columns `x,y,z`) **or** `grid: {start, counts, steps, safety_radius}` to sweep a 3D lattice. Grid points outside the auto-detected building bounds or too close to geometry are skipped.
   - `solver` — `PathSolver` knobs (`max_depth`, `samples_per_src`, `diffraction`, ...)
   - `pas` — Bartlett spectrum grid resolution (`grid_h`, `grid_w`)
3. **Run.**
   ```bash
   python build_dataset.py --config configs/<your_scene>.yml
   ```
4. **Verify.** The output directory should contain `spectrum/*.png`, `tx_pos.csv`, `freq.txt`, `gateway_info.yml`, and a `build_metadata.yml` snapshot of the config used.
5. **Train.** From the repo root: `python train.py --dataset <dataset_name>`.

### 3.5 Notes and conventions

- Wavelength is computed with `c = 2.99792458e8 m/s`.
- Spectrum order = outer loop over frequencies, inner loop over transmitter positions. Both `tx_pos.csv` and `freq.txt` follow that order and align 1:1 with the sorted `spectrum/*.png`.
- `utils.get_building_bounds` ignores shapes whose z-extent is `< 0.2 m` or whose name contains `floor`/`ground`/`plane`/`terrain`, to keep flat ground planes out of the bounding box. Rename or flag your ground mesh accordingly.
- For a purely CSV-driven placement, omit the `grid:` block — bounds checking and collision filtering are skipped for CSV positions, so ensure they are valid in your scene.

---

## Acknowledgments

`scene_builder/` builds on [Sionna RT](https://github.com/NVlabs/sionna) and [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3).
