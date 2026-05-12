"""Scene-bound helpers and PAS imaging used by build_dataset.py."""

from __future__ import annotations

import mitsuba as mi
import numpy as np


def _get_mitsuba_scene(scene):
    try:
        return scene._scene
    except AttributeError:
        return scene.mitsuba_scene


def _shape_identifier(shape) -> str:
    try:
        return str(shape.id()).lower()
    except Exception:
        return ""


def _is_inside_bounds(point, min_b, max_b, padding: float = 0.0) -> bool:
    return bool(
        np.all(point >= np.asarray(min_b, dtype=float) - padding)
        and np.all(point <= np.asarray(max_b, dtype=float) + padding)
    )


def get_building_bounds(scene):
    """Estimate usable scene bounds, ignoring flat ground-like geometry."""
    mi_scene = _get_mitsuba_scene(scene)

    bounds_min = np.full(3, np.inf, dtype=float)
    bounds_max = np.full(3, -np.inf, dtype=float)
    found_building = False

    print("--- [scene] estimating usable scene bounds ---")

    for shape in mi_scene.shapes():
        bbox = shape.bbox()
        b_min = np.array(bbox.min, dtype=float)
        b_max = np.array(bbox.max, dtype=float)
        size = b_max - b_min
        shape_name = _shape_identifier(shape)

        is_flat = size[2] < 0.2
        is_ground = any(
            token in shape_name for token in ("floor", "ground", "plane", "terrain")
        )
        if is_flat or is_ground:
            continue

        bounds_min = np.minimum(bounds_min, b_min)
        bounds_max = np.maximum(bounds_max, b_max)
        found_building = True

    if not found_building:
        print("  [warn] no building-like geometry found; using full scene bounds.")
        full_bbox = mi_scene.bbox()
        return np.array(full_bbox.min, dtype=float), np.array(full_bbox.max, dtype=float)

    print(
        "  [ok] bounds "
        f"X[{bounds_min[0]:.2f}, {bounds_max[0]:.2f}] "
        f"Y[{bounds_min[1]:.2f}, {bounds_max[1]:.2f}] "
        f"Z[{bounds_min[2]:.2f}, {bounds_max[2]:.2f}]"
    )
    return bounds_min, bounds_max


def validate_start_point(point, min_b, max_b) -> bool:
    point = np.asarray(point, dtype=float)
    min_b = np.asarray(min_b, dtype=float)
    max_b = np.asarray(max_b, dtype=float)

    if _is_inside_bounds(point, min_b, max_b, padding=0.05):
        print(f"  [ok] start point {point.tolist()} is inside the valid scene bounds.")
        return True

    print(f"  [error] start point {point.tolist()} is outside the valid scene bounds.")
    print(f"          X: {min_b[0]:.2f} ~ {max_b[0]:.2f}")
    print(f"          Y: {min_b[1]:.2f} ~ {max_b[1]:.2f}")
    print(f"          Z: {min_b[2]:.2f} ~ {max_b[2]:.2f}")
    return False


def is_position_safe(scene, position, safety_radius: float = 0.5) -> bool:
    """Reject a point whose nearest geometry along any axis is within safety_radius."""
    mi_scene = _get_mitsuba_scene(scene)
    px, py, pz = map(float, position)

    directions = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )

    for dx, dy, dz in directions:
        ray = mi.Ray3f(mi.Point3f(px, py, pz), mi.Vector3f(dx, dy, dz))
        si = mi_scene.ray_intersect(ray)
        if bool(si.is_valid()) and float(si.t) < float(safety_radius):
            return False

    return True


def generate_3d_grid(
    scene,
    start_point,
    counts,
    steps,
    bounds_min=None,
    bounds_max=None,
    safety_radius: float = 0.4,
):
    """Build a filtered 3D transmitter grid inside the scene bounds."""
    start = np.asarray(start_point, dtype=float)
    counts = np.asarray(counts, dtype=int)
    steps = np.asarray(steps, dtype=float)

    if counts.shape != (3,):
        raise ValueError(f"counts must contain exactly 3 integers, got {counts}.")
    if steps.shape != (3,):
        raise ValueError(f"steps must contain exactly 3 values, got {steps}.")
    if np.any(counts <= 0):
        raise ValueError(f"counts must be positive, got {counts.tolist()}.")
    if np.any(steps <= 0):
        raise ValueError(f"steps must be positive, got {steps.tolist()}.")

    if bounds_min is None or bounds_max is None:
        bounds_min, bounds_max = get_building_bounds(scene)

    bounds_min = np.asarray(bounds_min, dtype=float)
    bounds_max = np.asarray(bounds_max, dtype=float)
    if not validate_start_point(start, bounds_min, bounds_max):
        raise ValueError("The transmitter grid start point is outside the scene bounds.")

    nx, ny, nz = counts.tolist()
    sx, sy, sz = steps.tolist()
    total_attempts = int(np.prod(counts))
    valid_positions = []
    skipped_out_of_bounds = 0
    skipped_collision = 0

    print("--- generating 3D transmitter grid ---")
    print(f"grid: {nx}x{ny}x{nz} ({total_attempts} candidates)")
    print(f"step: X={sx}m, Y={sy}m, Z={sz}m")

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                candidate = start + np.array([i * sx, j * sy, k * sz], dtype=float)
                if not _is_inside_bounds(candidate, bounds_min, bounds_max, padding=1e-6):
                    skipped_out_of_bounds += 1
                    continue

                candidate_list = np.round(candidate, 6).tolist()
                if is_position_safe(scene, candidate_list, safety_radius=safety_radius):
                    valid_positions.append(candidate_list)
                else:
                    skipped_collision += 1

    print(
        f"grid generation finished: {len(valid_positions)} valid, "
        f"{skipped_out_of_bounds} out-of-bounds, {skipped_collision} blocked."
    )

    if not valid_positions:
        raise ValueError(
            "No valid transmitter positions were generated. "
            "Adjust start, counts, steps, or safety_radius."
        )

    return valid_positions


def bartlett_pas_algorithm_numpy(
    h_agg, antenna_pos, wavelength, grid_h: int = 90, grid_w: int = 360
) -> np.ndarray:
    """Bartlett power angular spectrum on a (theta in [0, pi/2], phi in [-pi, pi]) grid."""
    h_agg = np.asarray(h_agg).reshape(-1)
    antenna_pos = np.asarray(antenna_pos, dtype=float)

    if antenna_pos.ndim != 2 or antenna_pos.shape[1] != 3:
        raise ValueError(f"antenna_pos must have shape [N_ant, 3], got {antenna_pos.shape}.")
    if h_agg.size != antenna_pos.shape[0]:
        raise ValueError(
            f"Channel vector size {h_agg.size} does not match antenna count {antenna_pos.shape[0]}."
        )
    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength}.")

    phi_range = np.linspace(-np.pi, np.pi, grid_w)
    theta_range = np.linspace(0.0, np.pi / 2, grid_h)
    phi_grid, theta_grid = np.meshgrid(phi_range, theta_range)

    u = np.sin(theta_grid) * np.cos(phi_grid)
    v = np.sin(theta_grid) * np.sin(phi_grid)
    w = np.cos(theta_grid)

    d_k_flat = np.stack([u, v, w], axis=0).reshape(3, -1)
    delta_r = antenna_pos.T @ d_k_flat
    k = 2.0 * np.pi / float(wavelength)
    phi_theo = -k * delta_r

    psi_raw = np.angle(h_agg)
    psi_meas = psi_raw - np.mean(psi_raw)
    delta_k = phi_theo - psi_meas[:, np.newaxis]

    v_k = np.sum(np.exp(1j * delta_k), axis=0)
    s_raw = np.abs(v_k)
    peak = float(np.max(s_raw))

    if peak <= 0.0 or not np.isfinite(peak):
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    return (s_raw / peak).reshape(grid_h, grid_w).astype(np.float32)
