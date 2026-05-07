import os
import sys
import warnings
from argparse import ArgumentParser
from random import randint

import torch
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.pos_encoder import Embedder
from utils.general_utils import safe_state
from utils.loss_utils import compute_ssim_loss, fourier_loss, l1_loss
from utils.system_utils import extract_trailing_integer
from utils.train_utils import training_report, prepare_output_and_logger

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)


def resolve_lambda_dssim(opt_args, iteration):
    schedule = str(getattr(opt_args, "lambda_dssim_schedule", "constant")).lower()
    lambda_final = float(opt_args.lambda_dssim)
    lambda_init = float(getattr(opt_args, "lambda_dssim_init", lambda_final))
    warmup_iters = max(1, int(getattr(opt_args, "lambda_dssim_warmup_iters", 1)))

    if schedule == "constant":
        return lambda_final

    progress = min(max(iteration, 0) / warmup_iters, 1.0)

    if schedule == "linear":
        return lambda_init + (lambda_final - lambda_init) * progress

    raise ValueError(f"Unsupported lambda_dssim_schedule: {schedule}")


def get_ssim_config(opt_args):
    return {
        "domain": str(getattr(opt_args, "ssim_domain", "linear")).lower(),
        "eps": float(getattr(opt_args, "ssim_log_eps", 1e-6)),
    }


def training(
    model_para_args,
    optimization_para_args,
    pipeline_para_args,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_para_args)

    gaussians = GaussianModel(model_para_args)

    tx_pos_encoder = Embedder(
        input_dims=3,
        include_input=True,
        max_freq_log2=model_para_args.max_freq_log2,
        num_freqs=model_para_args.num_freqs,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )

    if not checkpoint:
        scene = Scene(model_para_args, gaussians, load_iteration=None, shuffle=True)
    else:
        file_name = os.path.basename(checkpoint)
        extracted_number = extract_trailing_integer(file_name)
        if extracted_number is None:
            raise ValueError(
                f"Unable to infer iteration number from checkpoint: {checkpoint}"
            )
        scene = Scene(
            model_para_args, gaussians, load_iteration=extracted_number, shuffle=True
        )

    gaussians.training_setup(optimization_para_args)

    if checkpoint:
        print("\nLoading saved trained model from path: {}\n".format(checkpoint))
        model_params, first_iter = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, optimization_para_args)

    bg_color = [1, 1, 1] if model_para_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = (
        torch.rand((3), device="cuda")
        if optimization_para_args.random_background
        else background
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ssim_cfg = get_ssim_config(optimization_para_args)

    progress_bar = tqdm(
        range(first_iter, optimization_para_args.iterations), desc="Training progress"
    )
    first_iter += 1

    for iteration in range(first_iter, optimization_para_args.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainSpectrums().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipeline_para_args.debug = True

        render_pkg = render(
            viewpoint_cam, gaussians, tx_pos_encoder, pipeline_para_args, bg
        )

        spectrum, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        gt_spectrum = viewpoint_cam.spectrum.cuda()
        freq_reg_loss = render_pkg.get("freq_reg_loss")
        if freq_reg_loss is None:
            freq_reg_loss = torch.tensor(0.0, device=gt_spectrum.device)

        lambda_dssim = resolve_lambda_dssim(optimization_para_args, iteration)

        ll1 = l1_loss(spectrum, gt_spectrum)
        ssim_loss = compute_ssim_loss(
            spectrum,
            gt_spectrum,
            clamp=True,
            domain=ssim_cfg["domain"],
            eps=ssim_cfg["eps"],
        )

        pred = spectrum.unsqueeze(dim=0)
        gt = gt_spectrum.unsqueeze(dim=0)
        fourier_lo = fourier_loss(pred, gt)

        loss = (
            (1.0 - lambda_dssim - optimization_para_args.lambda_dfourier) * ll1
            + lambda_dssim * ssim_loss
            + optimization_para_args.lambda_dfourier * fourier_lo
            + optimization_para_args.lambda_freq_reg * freq_reg_loss
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            if iteration == optimization_para_args.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                tx_pos_encoder,
                pipeline_para_args,
                model_para_args,
                bg,
                lambda_dssim,
                ssim_cfg,
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians Points".format(iteration))
                scene.save(iteration)

            if iteration < optimization_para_args.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                gaussians.add_densification_stats(gaussians.get_xyz, visibility_filter)

                if (
                    iteration >= optimization_para_args.densify_from_iter
                    and iteration % optimization_para_args.densification_interval == 0
                ):
                    size_threshold = (
                        optimization_para_args.raddi_size_threshold
                        if iteration > optimization_para_args.opacity_reset_interval
                        else None
                    )

                    gaussians.densify_and_prune(
                        optimization_para_args.densify_grad_threshold,
                        optimization_para_args.min_attenuation_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % optimization_para_args.opacity_reset_interval == 0 or (
                    model_para_args.white_background
                    and iteration == optimization_para_args.densify_from_iter
                ):
                    gaussians.reset_attenuation()

            if iteration < optimization_para_args.iterations:
                torch.nn.utils.clip_grad_norm_(
                    gaussians.freq_modulator.parameters(), max_norm=1.0
                )

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                gaussians.net_optimizer.step()
                gaussians.net_scheduler.step()
                gaussians.net_optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                chkpnt_path = os.path.join(
                    scene.model_path, f"chkpnt{str(iteration)}.pth"
                )

                print(
                    "\n[ITER {}] Saving Checkpoint in Path: {}".format(
                        iteration, chkpnt_path
                    )
                )

                torch.save((gaussians.capture(), iteration), chkpnt_path)


if __name__ == "__main__":
    random_seed_num_t = 1994

    parser = ArgumentParser(description="Training script parameters")

    model_para_cls = ModelParams(parser)
    optimization_para_cls = OptimizationParams(parser)
    pipeline_para_cls = PipelineParams(parser)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--iteration", dest="iterations", type=int)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", "--checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    default_iter = 7_000

    dataset_name = args.dataset
    exp_name = args.exp_name
    log_base_folder = args.log_base_folder
    input_data_folder = args.input_data_folder

    args.source_path = os.path.join(input_data_folder, dataset_name)

    basedir_out = os.path.join(log_base_folder, dataset_name)
    os.makedirs(basedir_out, exist_ok=True)

    model_path_dir = os.path.join(basedir_out, exp_name)
    os.makedirs(model_path_dir, exist_ok=True)
    args.model_path = model_path_dir
    args.aux_data_folder = os.path.join(args.model_path, "dataset_artifacts")

    args.save_iterations.append(default_iter)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations = args.save_iterations
    args.test_iterations = args.save_iterations

    args.densify_until_iter = args.iterations // 2
    args.position_lr_max_steps = args.iterations

    print(f"\n\tData path: {args.source_path}\n")
    print(f"\tModel path: {args.model_path}\n")
    print(f"\tLoading checkpoint path: {args.start_checkpoint}\n")

    safe_state(args.quiet, random_seed_num_t, torch.device(args.data_device))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    f_path = os.path.join(args.model_path, "config.yml")
    with open(f_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(vars(args), file, sort_keys=True, allow_unicode=True)

    training(
        model_para_cls.extract(args),
        optimization_para_cls.extract(args),
        pipeline_para_cls.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    print("\nTraining complete\n")
