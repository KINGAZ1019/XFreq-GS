"""Parameterized RF spectrum dataset builder.

Usage:
    python build_dataset.py --config configs/scene01.yml

Reads all scene / antenna / solver / sampling parameters from a YAML file,
runs Sionna RT on every (frequency, tx_position) pair, and writes:

    <output_root>/<dataset_name>/
    |-- spectrum/*.png
    |-- tx_pos.csv
    |-- freq.txt
    |-- gateway_info.yml
    +-- build_metadata.yml   (snapshot of the config used)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import imageio
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

from utils import bartlett_pas_algorithm_numpy, generate_3d_grid, get_building_bounds


SPEED_OF_LIGHT = 2.99792458e8


def _resolve(base: Path, value: str | os.PathLike) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _require(cfg: dict, key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: '{key}'")
    return cfg[key]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}.")
    return cfg


def build_planar_array(name: str, spec: dict) -> PlanarArray:
    return PlanarArray(
        num_rows=int(spec.get("num_rows", 1)),
        num_cols=int(spec.get("num_cols", 1)),
        vertical_spacing=float(spec.get("vertical_spacing", 0.5)),
        horizontal_spacing=float(spec.get("horizontal_spacing", 0.5)),
        pattern=str(spec.get("pattern", "iso")),
        polarization=str(spec.get("polarization", "V")),
    )


def euler_to_quaternion_xyzw(euler_rad: Iterable[float]) -> list[float]:
    """Sionna orientation is (yaw, pitch, roll) intrinsic ZYX in radians."""
    yaw, pitch, roll = (float(x) for x in euler_rad)
    q = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_quat()
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def load_tx_positions(scene, tx_cfg: dict, base: Path) -> list[list[float]]:
    if "csv" in tx_cfg and tx_cfg["csv"]:
        csv_path = _resolve(base, tx_cfg["csv"])
        df = pd.read_csv(csv_path)
        missing = {"x", "y", "z"} - set(df.columns)
        if missing:
            raise ValueError(f"tx csv '{csv_path}' is missing columns {sorted(missing)}.")
        return df[["x", "y", "z"]].to_numpy(dtype=float).tolist()

    grid_cfg = tx_cfg.get("grid")
    if not grid_cfg:
        raise ValueError("tx_positions must define either 'csv' or 'grid'.")

    bounds_min, bounds_max = get_building_bounds(scene)
    return generate_3d_grid(
        scene,
        start_point=_require(grid_cfg, "start"),
        counts=_require(grid_cfg, "counts"),
        steps=_require(grid_cfg, "steps"),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        safety_radius=float(grid_cfg.get("safety_radius", 0.4)),
    )


def reset_output_dir(output_dir: Path, spectrum_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if spectrum_dir.exists():
        shutil.rmtree(spectrum_dir)
    spectrum_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ("freq.txt", "tx_pos.csv", "gateway_info.yml", "build_metadata.yml"):
        stale = output_dir / file_name
        if stale.exists():
            stale.unlink()


def write_gateway_info(
    output_dir: Path,
    dataset_name: str,
    rx_position: list[float],
    rx_quat_xyzw: list[float],
) -> None:
    gateway = {
        "dataset_name": dataset_name,
        "gateway1": {
            "position": [float(v) for v in rx_position],
            "orientation": rx_quat_xyzw,
        },
    }
    with (output_dir / "gateway_info.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(gateway, handle, sort_keys=False)


def write_dataset_metadata(
    output_dir: Path, sample_positions: list[list[float]], sample_frequencies_ghz: list[float]
) -> None:
    pd.DataFrame(sample_positions, columns=["x", "y", "z"]).to_csv(
        output_dir / "tx_pos.csv", index=False
    )
    with (output_dir / "freq.txt").open("w", encoding="utf-8") as handle:
        for freq_ghz in sample_frequencies_ghz:
            handle.write(f"{freq_ghz:g}\n")


def write_build_metadata(output_dir: Path, config_source: Path, cfg: dict) -> None:
    snapshot = {"config_source": str(config_source), "config": cfg}
    with (output_dir / "build_metadata.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False, allow_unicode=True)


def validate_generation(
    sample_positions: list, sample_frequencies_ghz: list, spectrum_dir: Path, expected: int
) -> None:
    image_count = len(list(spectrum_dir.glob("*.png")))
    if len(sample_positions) != expected:
        raise RuntimeError(
            f"Expected {expected} transmitter entries, got {len(sample_positions)}."
        )
    if len(sample_frequencies_ghz) != expected:
        raise RuntimeError(
            f"Expected {expected} frequency entries, got {len(sample_frequencies_ghz)}."
        )
    if image_count != expected:
        raise RuntimeError(f"Expected {expected} spectrum images, found {image_count}.")


def build(config_path: Path) -> None:
    config_path = config_path.resolve()
    cfg = load_config(config_path)
    base = config_path.parent

    dataset_name = str(_require(cfg, "dataset_name"))
    scene_xml = _resolve(base, _require(cfg, "scene_xml"))
    output_root = _resolve(base, cfg.get("output_root", "../"))
    output_dir = output_root / dataset_name
    spectrum_dir = output_dir / "spectrum"

    frequencies_hz = [float(f) for f in _require(cfg, "frequencies_hz")]
    if not frequencies_hz:
        raise ValueError("frequencies_hz must be a non-empty list.")

    rx_cfg = _require(cfg, "rx")
    rx_position = [float(v) for v in _require(rx_cfg, "position")]
    rx_euler = [float(v) for v in rx_cfg.get("orientation_euler", [0.0, 0.0, 0.0])]
    rx_quat_xyzw = euler_to_quaternion_xyzw(rx_euler)

    tx_array_cfg = cfg.get("tx_array", {"num_rows": 1, "num_cols": 1, "pattern": "tr38901"})
    rx_array_cfg = cfg.get(
        "rx_array",
        {
            "num_rows": 4,
            "num_cols": 4,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "dipole",
            "polarization": "V",
        },
    )

    solver_cfg = cfg.get("solver", {})
    pas_cfg = cfg.get("pas", {})

    reset_output_dir(output_dir, spectrum_dir)

    print(f"--- loading scene: {scene_xml} ---")
    scene = load_scene(str(scene_xml))
    scene.transmitters.clear()
    scene.receivers.clear()

    scene.tx_array = build_planar_array("tx", tx_array_cfg)
    scene.rx_array = build_planar_array("rx", rx_array_cfg)

    tx_coords = load_tx_positions(scene, _require(cfg, "tx_positions"), base)

    expected_samples = len(frequencies_hz) * len(tx_coords)
    print(
        f"Generating {expected_samples} samples "
        f"({len(frequencies_hz)} frequencies x {len(tx_coords)} transmitter positions)."
    )

    tx = Transmitter(name="tx", position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0])
    scene.add(tx)
    tx.array = scene.tx_array

    rx = Receiver(name="rx", position=rx_position, orientation=rx_euler)
    scene.add(rx)
    rx.array = scene.rx_array

    solver = PathSolver()
    solver_kwargs = {
        "max_depth": int(solver_cfg.get("max_depth", 5)),
        "samples_per_src": int(solver_cfg.get("samples_per_src", int(1e5))),
        "synthetic_array": bool(solver_cfg.get("synthetic_array", False)),
        "specular_reflection": bool(solver_cfg.get("specular_reflection", True)),
        "diffraction": bool(solver_cfg.get("diffraction", True)),
        "diffuse_reflection": bool(solver_cfg.get("diffuse_reflection", False)),
    }

    grid_h = int(pas_cfg.get("grid_h", 90))
    grid_w = int(pas_cfg.get("grid_w", 360))

    write_gateway_info(output_dir, dataset_name, rx_position, rx_quat_xyzw)

    sample_frequencies_ghz: list[float] = []
    sample_positions: list[list[float]] = []
    img_idx = 1

    print(f"Frequency settings: {[f / 1e9 for f in frequencies_hz]} GHz")
    print("Starting RF dataset generation...")

    for freq in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / freq
        rx_array_pos_np = scene.rx_array.positions(wavelength).numpy()
        print(f"--- solving paths at {freq / 1e9:g} GHz ---")

        for pos in tx_coords:
            scene.transmitters["tx"].position = pos
            paths = solver(scene, **solver_kwargs)

            h_freq = paths.cfr(freq, normalize=True, out_type="numpy")
            h_vec = np.asarray(h_freq).reshape(-1)

            pas_img = bartlett_pas_algorithm_numpy(
                h_vec, rx_array_pos_np, wavelength, grid_h=grid_h, grid_w=grid_w
            )
            png_path = spectrum_dir / f"{img_idx:05d}.png"
            imageio.imwrite(png_path, (np.clip(pas_img, 0.0, 1.0) * 255).astype(np.uint8))

            sample_frequencies_ghz.append(freq / 1e9)
            sample_positions.append(list(pos))
            img_idx += 1

    validate_generation(sample_positions, sample_frequencies_ghz, spectrum_dir, expected_samples)
    write_dataset_metadata(output_dir, sample_positions, sample_frequencies_ghz)
    write_build_metadata(output_dir, config_path, cfg)

    print("==========================================")
    print("Dataset generation finished")
    print(f"Output directory: {output_dir.resolve()}")
    print("==========================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an XfreqGS RF spectrum dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file (see configs/scene01.yml).",
    )
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
