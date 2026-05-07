import os
import torch
from random import randint
import uuid
import random

import yaml

from .data_painter import paint_spectrum
from .loss_utils import psnr, compute_ssim
import torch.nn as nn

# import lpips
# loss_lpips_t = lpips.LPIPS(net='alex')


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene,
    renderFunc,
    tx_pos_encoder_func,
    pipe_args,
    dataset_args,
    background,
    lambda_dssim,
    ssim_cfg,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/lambda_dssim", lambda_dssim, iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    image_path = os.path.join(dataset_args.model_path, "spectrums")
    os.makedirs(image_path, exist_ok=True)

    number_of_samples = 5
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = (
            {
                "name": "test",
                "spectrums": random.sample(scene.getTestSpectrums(), number_of_samples),
            },
            {
                "name": "train",
                "spectrums": random.sample(
                    scene.getTrainSpectrums(), number_of_samples
                ),
            },
        )

        for config in validation_configs:
            if config["spectrums"] and len(config["spectrums"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0

                for idx, viewpoint in enumerate(config["spectrums"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        tx_pos_encoder_func,
                        pipe_args,
                        background,
                    )["render"]
                    gt_image = viewpoint.spectrum.to("cuda")

                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)

                    image_metric = image.unsqueeze(0)
                    gt_image_metric = gt_image.unsqueeze(0)

                    # if tb_writer and (idx < 5):
                    if tb_writer:
                        file_name_test = config["name"] + "_view_{}/render".format(
                            viewpoint.spectrum_name
                        )

                        # add_images expects N-C-H-W; spectrum tensors here are H-W.
                        tb_writer.add_images(
                            file_name_test, image[None, None], global_step=iteration
                        )

                        if (
                            iteration == testing_iterations[0]
                        ):  # testing_iterations = [7000, 30000]
                            file_name_gt = config[
                                "name"
                            ] + "_view_{}/ground_truth".format(viewpoint.spectrum_name)
                            tb_writer.add_images(
                                file_name_gt,
                                gt_image[None, None],
                                global_step=iteration,
                            )

                    l1_test += l1_loss(image_metric, gt_image_metric).mean().double()
                    psnr_test += psnr(image_metric, gt_image_metric).mean().double()
                    ssim_test += (
                        compute_ssim(
                            image_metric,
                            gt_image_metric,
                            clamp=True,
                            domain=ssim_cfg["domain"],
                            eps=ssim_cfg["eps"],
                        )
                        .mean()
                        .double()
                    )

                    filename = os.path.join(
                        image_path,
                        f"ite_{iteration:06d}_{config['name']}_{idx:06d}.png",
                    )
                    paint_spectrum(
                        gt_image.cpu().squeeze().numpy(),
                        image.cpu().squeeze().numpy(),
                        save_path=filename,
                    )

                l1_test /= len(config["spectrums"])
                psnr_test /= len(config["spectrums"])
                ssim_test /= len(config["spectrums"])

                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(
                        iteration, config["name"], l1_test, psnr_test, ssim_test
                    )
                )

                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_attenuation, iteration
            )

            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )

        torch.cuda.empty_cache()


def prepare_output_and_logger(args):

    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")

        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./output/", unique_str[0:10])

    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        yaml.safe_dump(vars(args), cfg_log_f, sort_keys=True, allow_unicode=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("\nTensorboard not available: not logging progress!\n")

    return tb_writer


def initialize_weights(module):

    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
