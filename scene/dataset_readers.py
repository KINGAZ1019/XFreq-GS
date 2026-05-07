#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import sys
from typing import NamedTuple

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from scene.gaussian_model import BasicPointCloud


class SpectrumInfo(NamedTuple):
    R: np.array
    T_rx: np.array
    T_tx: np.array
    spectrum: np.array
    spectrum_path: str
    spectrum_name: str
    width: int
    height: int
    freq: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_spectrums: list
    test_spectrums: list
    nerf_normalization: dict
    ply_path: str


DEFAULT_TRAIN_INDEX_PATH = "train_index.txt"
DEFAULT_TEST_INDEX_PATH = "test_index.txt"
DEFAULT_AUXILIARY_DIR_NAME = "dataset_artifacts"


def ensure_parent_dir(file_path):
    folder_path = os.path.dirname(file_path)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)


def get_auxiliary_data_root(args_model):
    aux_root = getattr(args_model, "aux_data_folder", "")
    if aux_root:
        aux_root = os.path.abspath(aux_root)
    else:
        model_path = getattr(args_model, "model_path", "")
        if not model_path:
            raise ValueError(
                "args.model_path is required to store generated dataset artifacts."
            )
        aux_root = os.path.join(
            os.path.abspath(model_path), DEFAULT_AUXILIARY_DIR_NAME
        )

    os.makedirs(aux_root, exist_ok=True)
    return aux_root


def resolve_auxiliary_path(aux_root, relative_or_absolute_path):
    if os.path.isabs(relative_or_absolute_path):
        return relative_or_absolute_path
    return os.path.join(aux_root, relative_or_absolute_path)


def split_dataset_train_v2(datadir, train_path, test_path, ratio=0.8):
    # Match the current split logic exactly: shuffle first, then split.
    spectrum_dir = os.path.join(datadir, "spectrum")
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith(".png")])
    image_names = [x.split(".")[0] for x in spt_names]
    len_image = len(image_names)

    random.seed(1994)
    np.random.seed(1994)

    # Keep zero-based indices so they align with the sorted spectrum list.
    indices = list(range(len_image))
    random.shuffle(indices)

    train_len = int(len_image * ratio)
    train_list = np.array(indices[:train_len], dtype=int)
    test_list = np.array(indices[train_len:], dtype=int)

    print(
        "\n [Modified] Ratio: {}....  Train set: {}....  Test set: {}....\n".format(
            ratio, len(train_list), len(test_list)
        )
    )

    ensure_parent_dir(train_path)
    ensure_parent_dir(test_path)
    np.savetxt(train_path, train_list, fmt="%d")
    np.savetxt(test_path, test_list, fmt="%d")


def readSpectrumImage(data_dir_path):
    data_infos = []

    tx_pos_path = os.path.join(data_dir_path, "tx_pos.csv")
    tx_pos = pd.read_csv(tx_pos_path).values

    gateway_pos_path = os.path.join(data_dir_path, "gateway_info.yml")
    spectrum_dir = os.path.join(data_dir_path, "spectrum")
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith(".png")])

    freq_path = os.path.join(data_dir_path, "freq.txt")
    try:
        freq = np.loadtxt(freq_path)
        max_freq = 94.0
        freq_norm = freq / max_freq
        print(
            f"Loaded frequencies from {freq_path}. Range: {freq.min()} - {freq.max()} GHz"
        )
    except Exception as e:
        print(f"ERROR: Failed to load or process '{freq_path}'.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    with open(gateway_pos_path) as f_loader:
        gateway_info = yaml.safe_load(f_loader)
        gateway_pos = gateway_info["gateway1"]["position"]
        gateway_quaternion = gateway_info["gateway1"]["orientation"]

    for image_idx, image_name in enumerate(spt_names):
        qvec = np.array(gateway_quaternion)
        rotation_matrix = torch.from_numpy(Rotation.from_quat(qvec).as_matrix()).float()

        tvec_rx = torch.from_numpy(np.array(gateway_pos)).float()
        tvec_tx = torch.from_numpy(np.array(tx_pos[image_idx])).float()

        image_path = os.path.join(spectrum_dir, os.path.basename(image_name))
        image_name_t = os.path.basename(image_path).split(".")[0]

        image = imageio.imread(image_path).astype(np.float32) / 255.0
        height = image.shape[0]
        width = image.shape[1]
        resized_image = torch.from_numpy(np.array(image)).float()
        freq_value = float(freq_norm[image_idx % len(freq_norm)])

        spec_info = SpectrumInfo(
            R=rotation_matrix,
            T_rx=tvec_rx,
            T_tx=tvec_tx,
            spectrum=resized_image,
            spectrum_path=image_path,
            spectrum_name=image_name_t,
            height=height,
            width=width,
            freq=freq_value,
        )
        data_infos.append(spec_info)

    sys.stdout.write("\n")
    return data_infos


def getNorm_3d(specs_info, scale):
    def get_center_and_diag(gatewa_pos_t, cam_center):
        gatewa_pos_t = gatewa_pos_t.unsqueeze(1)
        cam_center = torch.stack(cam_center, dim=1)

        dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
        radius = torch.max(dists) * scale

        deviations = cam_center - gatewa_pos_t
        positive_deviations = deviations.clone()
        negative_deviations = deviations.clone()

        positive_deviations[positive_deviations < 0] = 0
        negative_deviations[negative_deviations > 0] = 0

        max_positive = positive_deviations.max(dim=1).values
        max_negative = negative_deviations.min(dim=1).values.abs()

        epsilon = 1e-6
        max_positive[max_positive < epsilon] = 1.0
        max_negative[max_negative < epsilon] = 1.0

        return {
            "max_positive": max_positive * scale,
            "max_negative": max_negative * scale,
        }, radius.item()

    cam_centers = []
    gatewa_pos = specs_info[0].T_rx

    for cam in specs_info:
        cam_centers.append(cam.T_tx)

    diagonal, radius = get_center_and_diag(gatewa_pos, cam_centers)
    translate = -gatewa_pos

    return {"translate": translate, "radius": radius, "extent": diagonal}


def getNorm_3d_v2(specs_info, scale, margin=1.5):
    gateway_pos = specs_info[0].T_rx
    all_positions = [gateway_pos]
    for cam in specs_info:
        all_positions.append(cam.T_tx)

    all_positions_stack = torch.stack(all_positions)
    world_min = torch.min(all_positions_stack, dim=0).values
    world_max = torch.max(all_positions_stack, dim=0).values

    scene_min = world_min - margin
    scene_max = world_max + margin

    dist_min = torch.norm(scene_min - gateway_pos)
    dist_max = torch.norm(scene_max - gateway_pos)
    radius = max(dist_min.item(), dist_max.item()) * scale

    translate = -gateway_pos

    return {
        "translate": translate,
        "radius": radius,
        "scene_min": scene_min,
        "scene_max": scene_max,
        "extent": None,
    }


def obtain_train_test_idx(args_model, len_list):
    path = args_model.source_path
    llffhold = args_model.llffhold
    llffhold_flag = args_model.llffhold_flag

    train_index = os.path.join(path, args_model.train_index_path)
    test_index = os.path.join(path, args_model.test_index_path)

    if llffhold_flag:
        print("\nUSING LLFFHOLD INDEX FILE\n")
        i_test = np.arange(int(len_list))[::llffhold]
        i_train = np.array([j for j in np.arange(int(len_list)) if (j not in i_test)])
    elif "knn" in train_index:
        print("\nUSING KNN INDEX FILE\n")
        i_train = np.loadtxt(train_index, dtype=int)
        i_test = np.loadtxt(test_index, dtype=int)
    else:
        print("\nUSING INDEX FILE\n")
        i_train = np.loadtxt(train_index, dtype=int)
        i_test = np.loadtxt(test_index, dtype=int)

    return i_train, i_test


def should_generate_split_indices(args_model, train_index_path, test_index_path):
    if args_model.llffhold_flag:
        return False

    if "knn" in os.path.basename(train_index_path).lower():
        return False

    train_index_rel = os.path.normpath(str(args_model.train_index_path))
    test_index_rel = os.path.normpath(str(args_model.test_index_path))

    uses_default_split_files = (
        train_index_rel == os.path.normpath(DEFAULT_TRAIN_INDEX_PATH)
        and test_index_rel == os.path.normpath(DEFAULT_TEST_INDEX_PATH)
    )

    train_exists = os.path.exists(train_index_path)
    test_exists = os.path.exists(test_index_path)

    if uses_default_split_files:
        return not (train_exists and test_exists)

    missing_paths = []
    if not train_exists:
        missing_paths.append(train_index_path)
    if not test_exists:
        missing_paths.append(test_index_path)

    if missing_paths:
        missing_text = ", ".join(f"'{p}'" for p in missing_paths)
        raise ValueError(
            "Explicit split index file(s) not found: "
            f"{missing_text}. Generate them first instead of falling back to random splitting."
        )

    return False


def readRFSceneInfo(args_model):
    path = args_model.source_path
    eval_mode = args_model.eval
    camera_scale = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale
    point_init_mode = getattr(args_model, "point_init_mode", "cube").lower()
    cube_sampling_density_factor = getattr(
        args_model, "cube_sampling_density_factor", 2
    )
    ratio_train = args_model.ratio_train

    spectrums_infos_unsorted = readSpectrumImage(path)

    aux_root = get_auxiliary_data_root(args_model)
    train_index_path = resolve_auxiliary_path(aux_root, args_model.train_index_path)
    test_index_path = resolve_auxiliary_path(aux_root, args_model.test_index_path)

    should_generate_random_split = should_generate_split_indices(
        args_model, train_index_path, test_index_path
    )
    if should_generate_random_split:
        split_dataset_train_v2(path, train_index_path, test_index_path, ratio=ratio_train)
    i_train, i_test = obtain_train_test_idx(args_model, len(spectrums_infos_unsorted))

    spectrums_infos = sorted(
        spectrums_infos_unsorted.copy(), key=lambda x: int(x.spectrum_name)
    )

    if eval_mode:
        train_infos = [spectrums_infos[idx] for idx in i_train]
        test_infos = [spectrums_infos[idx] for idx in i_test]
    else:
        train_infos = spectrums_infos
        test_infos = []

    nerf_normalization = getNorm_3d_v2(spectrums_infos, camera_scale, margin=1.5)
    ply_path = resolve_auxiliary_path(aux_root, args_model.init_point_cloud_path)

    if (not os.path.exists(ply_path)) or (args_model.gene_init_point):
        avg_freq_ghz = np.mean([s.freq for s in spectrums_infos])
        avg_freq = avg_freq_ghz * 1e9
        cube_size = round((3.00e8 / avg_freq) * voxel_size_scale, 2)

        num_pos = init_ply_v2(
            ply_path,
            nerf_normalization["scene_min"],
            nerf_normalization["scene_max"],
            cube_size,
            point_init_mode=point_init_mode,
            density_factor=cube_sampling_density_factor,
        )
        print(
            f"\nInitialized point clouds with mode '{point_init_mode}'. "
            f"Cube size: {cube_size} meters, Number of points: {num_pos}\n"
        )

    try:
        pcd = fetch_init_ply(ply_path)
    except (FileNotFoundError, OSError, ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Failed to load initial point cloud from '{ply_path}'.") from exc

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_spectrums=train_infos,
        test_spectrums=test_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

    return scene_info


def fetch_init_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, attris=None, normals=normals)


def init_ply_v2(
    ply_path, scene_min, scene_max, cube_size, point_init_mode="cube", density_factor=2
):
    print(f"\nInitializing point cloud with Cube Size: {cube_size}")
    ensure_parent_dir(ply_path)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]

    point_init_mode = point_init_mode.lower()
    if point_init_mode == "cube":
        xyz = generate_cube_coordinates_v2(
            scene_min, scene_max, cube_size, density_factor=density_factor
        )
    elif point_init_mode == "random":
        target_num_points = estimate_cube_point_count(
            scene_min, scene_max, cube_size, density_factor=density_factor
        )
        xyz = generate_random_coordinates_v2(scene_min, scene_max, target_num_points)
    else:
        raise ValueError(f"Unsupported point_init_mode: {point_init_mode}")

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    return xyz.shape[0]


def generate_cube_coordinates(receiver_pos, camera_extent, cube_size):
    x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
    x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()
    y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
    y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()
    z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
    z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()

    num_cubes_x = int(np.ceil((x_max - x_min) / cube_size))
    num_cubes_y = int(np.ceil((y_max - y_min) / cube_size))
    num_cubes_z = int(np.ceil((z_max - z_min) / cube_size))

    x_coords = (
        np.linspace(x_min, x_max, num_cubes_x)
        if num_cubes_x > 1
        else np.array([(x_min + x_max) / 2])
    )
    y_coords = (
        np.linspace(y_min, y_max, num_cubes_y)
        if num_cubes_y > 1
        else np.array([(y_min + y_max) / 2])
    )
    z_coords = (
        np.linspace(z_min, z_max, num_cubes_z)
        if num_cubes_z > 1
        else np.array([(z_min + z_max) / 2])
    )

    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    return cube_points


def generate_random_coordinates(receiver_pos, camera_extent, num_points):
    x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
    x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()
    y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
    y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()
    z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
    z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()

    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(z_min, z_max, num_points)

    return np.column_stack((x, y, z))


def estimate_cube_grid_shape(scene_min, scene_max, cube_size, density_factor=2):
    x_min, y_min, z_min = scene_min[0].item(), scene_min[1].item(), scene_min[2].item()
    x_max, y_max, z_max = scene_max[0].item(), scene_max[1].item(), scene_max[2].item()
    x_min, x_max, z_min, z_max = z_min, z_max, x_min, x_max

    num_cubes_x = max(1, int(np.ceil((x_max - x_min) / cube_size))) * density_factor
    num_cubes_y = max(1, int(np.ceil((y_max - y_min) / cube_size))) * density_factor
    num_cubes_z = max(1, int(np.ceil((z_max - z_min) / cube_size))) * density_factor

    return (x_min, x_max, y_min, y_max, z_min, z_max), (
        num_cubes_x,
        num_cubes_y,
        num_cubes_z,
    )


def estimate_cube_point_count(scene_min, scene_max, cube_size, density_factor=2):
    _, grid_shape = estimate_cube_grid_shape(
        scene_min, scene_max, cube_size, density_factor=density_factor
    )
    return int(np.prod(grid_shape))


def generate_random_coordinates_v2(scene_min, scene_max, num_points):
    x_min, y_min, z_min = scene_min[0].item(), scene_min[1].item(), scene_min[2].item()
    x_max, y_max, z_max = scene_max[0].item(), scene_max[1].item(), scene_max[2].item()
    x_min, x_max, z_min, z_max = z_min, z_max, x_min, x_max

    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(z_min, z_max, num_points)

    return np.column_stack((x, y, z))


def generate_cube_coordinates_v2(scene_min, scene_max, cube_size, density_factor=2):
    """Generate a uniform grid of initialization points inside scene bounds."""
    bounds, grid_shape = estimate_cube_grid_shape(
        scene_min, scene_max, cube_size, density_factor=density_factor
    )
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    print(
        f"Generating RF Sources in World Bounds: "
        f"X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]"
    )

    num_cubes_x, num_cubes_y, num_cubes_z = grid_shape
    xs = np.linspace(x_min, x_max, num_cubes_x)
    ys = np.linspace(y_min, y_max, num_cubes_y)
    zs = np.linspace(z_min, z_max, num_cubes_z)

    x_grid, y_grid, z_grid = np.meshgrid(xs, ys, zs, indexing="ij")
    cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    return cube_points
