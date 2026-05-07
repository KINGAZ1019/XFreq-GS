import os
import sys
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import skimage
import torch

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.pos_encoder import Embedder
from utils.general_utils import safe_state
from utils.system_utils import extract_trailing_integer, find_latest_checkpoint

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)


def testing(
    model_para_args,
    optimization_para_args,
    pipeline_para_args,
    checkpointpath_inference,
    timing_warmup=10,
):
    if not checkpointpath_inference:
        raise ValueError(
            "No checkpoint was provided for inference. "
            "Pass --start_checkpoint explicitly or ensure the experiment directory contains a checkpoint."
        )

    gaussians = GaussianModel(model_para_args)

    tx_pos_encoder = Embedder(
        input_dims=3,
        include_input=True,
        max_freq_log2=model_para_args.max_freq_log2,
        num_freqs=model_para_args.num_freqs,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )

    file_name = os.path.basename(checkpointpath_inference)
    extracted_number = extract_trailing_integer(file_name)
    if extracted_number is None:
        raise ValueError(
            f"Unable to infer iteration number from checkpoint: {checkpointpath_inference}"
        )

    scene = Scene(
        model_para_args, gaussians, load_iteration=extracted_number, shuffle=True
    )

    model_params, _ = torch.load(checkpointpath_inference, weights_only=False)
    gaussians.restore(model_params, optimization_para_args, restore_optimizers=False)

    bg_color = [1, 1, 1] if model_para_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = (
        torch.rand((3), device="cuda")
        if optimization_para_args.random_background
        else background
    )

    viewpoint_stack = scene.getTestSpectrums().copy()

    psnr_list = []
    ssim_list = []
    name_list = []
    inference_times_ms = []
    render_times_ms = []
    total_times_ms = []

    output_dir = os.path.dirname(checkpointpath_inference)
    warmup_iters = min(timing_warmup, len(viewpoint_stack))
    for warmup_idx in range(warmup_iters):
        with torch.no_grad():
            _ = render(
                viewpoint_stack[warmup_idx],
                gaussians,
                tx_pos_encoder,
                pipeline_para_args,
                bg,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for viewpoint_cam in viewpoint_stack:
        with torch.no_grad():
            total_start = time.perf_counter()
            render_pkg = render(
                viewpoint_cam, gaussians, tx_pos_encoder, pipeline_para_args, bg
            )
            total_ms_fallback = (time.perf_counter() - total_start) * 1000.0

        timing = render_pkg.get("timing")
        if timing is not None:
            inference_ms = timing.get("inference_ms", total_ms_fallback)
            render_ms = timing.get("render_ms", 0.0)
            total_ms = timing.get("total_ms", inference_ms + render_ms)
        else:
            inference_ms = total_ms_fallback
            render_ms = 0.0
            total_ms = total_ms_fallback

        inference_times_ms.append(inference_ms)
        render_times_ms.append(render_ms)
        total_times_ms.append(total_ms)

        spectrum = render_pkg["render"].detach().cpu().numpy()
        gt_spectrum = viewpoint_cam.spectrum.cpu().numpy()

        psnr_value = skimage.metrics.peak_signal_noise_ratio(
            spectrum, gt_spectrum, data_range=1
        )
        ssim_value = skimage.metrics.structural_similarity(
            spectrum, gt_spectrum, data_range=1
        )

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        name_list.append(viewpoint_cam.spectrum_name)

    avg_psnr = np.mean(psnr_list)
    med_psnr = np.median(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_infer_ms = np.mean(inference_times_ms)
    avg_render_ms = np.mean(render_times_ms)
    avg_total_ms = np.mean(total_times_ms)
    med_total_ms = np.median(total_times_ms)
    p90_total_ms = np.percentile(total_times_ms, 90)
    fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0
    sum_infer_ms = np.sum(inference_times_ms)
    sum_render_ms = np.sum(render_times_ms)
    sum_total_ms = np.sum(total_times_ms)

    print(
        f"\n[FINAL RESULT] Average PSNR: {avg_psnr:.4f} | Median PSNR: {med_psnr:.4f} | "
        f"Average SSIM: {avg_ssim:.4f}"
    )
    print(
        f"[PERFORMANCE] Infer Avg: {avg_infer_ms:.2f} ms | Render Avg: {avg_render_ms:.2f} ms | "
        f"Total Avg: {avg_total_ms:.2f} ms | Median: {med_total_ms:.2f} ms | "
        f"P90: {p90_total_ms:.2f} ms | FPS: {fps:.2f}"
    )

    txt_path = os.path.join(output_dir, f"evaluation_iter{extracted_number}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Evaluation Report - Iteration {extracted_number}\n")
        f.write(f"Checkpoint: {checkpointpath_inference}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"{'View Name':<20} | {'PSNR':<15} | {'SSIM':<15} | "
            f"{'Infer(ms)':<12} | {'Render(ms)':<12} | {'Total(ms)':<12}\n"
        )
        f.write("-" * 60 + "\n")

        for n, p, s, it, rt, tt in zip(
            name_list,
            psnr_list,
            ssim_list,
            inference_times_ms,
            render_times_ms,
            total_times_ms,
        ):
            f.write(
                f"{n:<20} | {p:<15.4f} | {s:<15.4f} | {it:<12.2f} | {rt:<12.2f} | {tt:<12.2f}\n"
            )

        f.write("-" * 60 + "\n")
        f.write(
            f"{'AVERAGE':<20} | {avg_psnr:<15.4f} | {avg_ssim:<15.4f} | "
            f"{avg_infer_ms:<12.2f} | {avg_render_ms:<12.2f} | {avg_total_ms:<12.2f}\n"
        )
        f.write(f"Median PSNR: {med_psnr:.4f}\n")
        f.write(f"Sum Inference Time (ms): {sum_infer_ms:.2f}\n")
        f.write(f"Sum Render Time (ms): {sum_render_ms:.2f}\n")
        f.write(f"Sum Total Time (ms): {sum_total_ms:.2f}\n")
        f.write(f"Median Total Time (ms): {med_total_ms:.2f}\n")
        f.write(f"P90 Total Time (ms): {p90_total_ms:.2f}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write("=" * 60 + "\n")
    print(f"Detailed report (txt) saved to: {txt_path}")


def main_inference():
    random_seed_num_t = 1994

    parser = ArgumentParser(description="Training script parameters")

    model_para_cls = ModelParams(parser)
    optimization_para_cls = OptimizationParams(parser)
    pipeline_para_cls = PipelineParams(parser)

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--timing_warmup", type=int, default=10)

    args = parser.parse_args(sys.argv[1:])

    dataset_name = args.dataset
    exp_name = args.exp_name
    log_base_folder = args.log_base_folder
    input_data_folder = args.input_data_folder

    args.source_path = os.path.join(input_data_folder, dataset_name)

    basedir_out = os.path.join(log_base_folder, dataset_name)
    args.model_path = os.path.join(basedir_out, exp_name)
    args.aux_data_folder = os.path.join(args.model_path, "dataset_artifacts")

    if args.start_checkpoint is None:
        checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.iterations}.pth")
        if os.path.exists(checkpoint_path):
            args.start_checkpoint = checkpoint_path

    if args.start_checkpoint is None:
        args.start_checkpoint = find_latest_checkpoint(args.model_path)

    print(f"\n\tData path: {args.source_path}\n")
    print(f"\tModel path: {args.model_path}\n")
    print(f"\tLoading checkpoint path: {args.start_checkpoint}\n")

    safe_state(args.quiet, random_seed_num_t, torch.device(args.data_device))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    testing(
        model_para_cls.extract(args),
        optimization_para_cls.extract(args),
        pipeline_para_cls.extract(args),
        args.start_checkpoint,
        args.timing_warmup,
    )


if __name__ == "__main__":
    main_inference()
