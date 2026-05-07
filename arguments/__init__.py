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

import ast
import os
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime

import yaml


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value)
            value = value if not fill_none else None

            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])

        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(curr_dir)
        current_time = datetime.now().strftime("%m%d%Y%H%M%S")

        self.input_data_folder = os.path.join(base_dir, "dataset")

        self.dataset = "rfid"
        self.exp_name = f"cfgs_{current_time}"
        self.log_base_folder = os.path.join(base_dir, "logs")
        self.aux_data_folder = ""

        self.llffhold = 8
        self.llffhold_flag = False
        self.ratio_train = 0.8

        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._white_background = False
        self.data_device = "cuda:0"
        self.eval = True

        self.gene_init_point = False
        self.camera_scale = 1.5
        self.voxel_size_scale = 1.0
        self.point_init_mode = "cube"
        self.cube_sampling_density_factor = 2

        self.train_index_path = "train_index.txt"
        self.test_index_path = "test_index.txt"
        self.init_point_cloud_path = "points3D.ply"

        self.max_freq_log2 = 9
        self.num_freqs = 10

        self.hidden_dim_1 = 256
        self.hidden_dim_2 = 64
        self.output_dim = 3
        self.point_code_dim = 8
        self.point_code_targets = "all"
        self.freq_sh_scale = 0.10
        self.freq_sh_bias_scale = 0.01
        self.freq_modulate_dc_only = True
        self.freq_att_scale = 0.5
        self.freq_splat_scale = 0.10
        self.freq_sh_head_mode = "channel"
        self.freq_sh_max_order = 3

        self.ablate_freq_mod = False
        self.ablate_freq_splat = False

        self.input_dim_emd = (3 + 2 * 3 * self.num_freqs) * 2

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = True
        self.debug = False
        self.radius_rx = 1.1
        self.aos_mode = "adaptive"
        self.ablate_aos = False

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        max_iter = 30_000

        self.iterations = max_iter
        self.position_lr_init = 0.00016
        self.position_lr_final = 1.6e-06
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = max_iter
        self.feature_lr = 0.0025
        self.opacity_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.02

        self.lambda_dssim = 0.4
        self.lambda_dfourier = 0.0
        self.lambda_dssim_schedule = "linear"
        self.lambda_dssim_init = 0.0
        self.lambda_dssim_warmup_iters = 5000
        self.ssim_domain = "linear"
        self.ssim_log_eps = 1e-6
        self.lambda_freq_reg = 5e-4

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = max_iter // 2
        self.densify_grad_threshold = 0.0001

        self.min_attenuation_threshold = 0.004

        self.raddi_size_threshold = 10

        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


def _load_cfg_namespace(cfg_text: str) -> Namespace:
    cfg_text = str(cfg_text).strip()
    if not cfg_text:
        return Namespace()

    try:
        cfg_data = yaml.safe_load(cfg_text)
    except yaml.YAMLError:
        cfg_data = None

    if isinstance(cfg_data, dict):
        return Namespace(**cfg_data)

    try:
        expr = ast.parse(cfg_text, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(
            "Unsupported cfg_args format. Use YAML or legacy Namespace(...)."
        ) from exc

    if (
        not isinstance(expr, ast.Call)
        or not isinstance(expr.func, ast.Name)
        or expr.func.id != "Namespace"
        or expr.args
    ):
        raise ValueError(
            "Unsupported cfg_args format. Use YAML or legacy Namespace(...)."
        )

    cfg_values = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            raise ValueError("cfg_args does not support **kwargs expansion.")
        cfg_values[keyword.arg] = ast.literal_eval(keyword.value)

    return Namespace(**cfg_values)


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    args_cfgfile = Namespace()

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)

        with open(cfgfilepath, encoding="utf-8") as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            args_cfgfile = _load_cfg_namespace(cfg_file.read())
    except (TypeError, FileNotFoundError):
        print("Config file not found at")
        pass

    merged_dict = vars(args_cfgfile).copy()

    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)
