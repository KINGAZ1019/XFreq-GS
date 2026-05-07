import time

import numpy as np
import torch
import torch.nn.functional as F

from complex_gaussian_tracer import ComplexGaussianTracerSettings, ComplexGaussianTracer
from scene.gaussian_model import GaussianModel
from scene.pos_encoder import Embedder
from utils.sh_utils import eval_sh


class RenderCache:
    """Cache render-time tensors derived from the view sampling grid."""

    _instance = None

    def __init__(self, device):
        self.device = device
        self._r_d_fine = None
        self._idx_tensor = None
        self._radius_rx = None
        self._n_azimuth = None
        self._n_elevation = None
        self._step = None

    @classmethod
    def get(cls, device):
        """Return the cache instance for the requested device."""
        if cls._instance is None or cls._instance.device != device:
            cls._instance = cls(device)
        return cls._instance

    @classmethod
    def clear(cls):
        """Drop cached tensors, e.g. after tests or device changes."""
        cls._instance = None

    def get_ray_directions(
        self, n_azimuth=360, n_elevation=90, radius_rx=0.5, dtype=torch.float32
    ):
        """Return cached ray directions for the current sampling grid."""
        need_recompute = (
            self._r_d_fine is None
            or self._radius_rx != radius_rx
            or self._n_azimuth != n_azimuth
            or self._n_elevation != n_elevation
            or self._r_d_fine.dtype != dtype
        )

        if need_recompute:
            self._r_d_fine = create_ray_direction_fine_v2(
                n_azimuth=n_azimuth, n_elevation=n_elevation, radius=radius_rx
            ).to(self.device, dtype=dtype)
            self._radius_rx = radius_rx
            self._n_azimuth = n_azimuth
            self._n_elevation = n_elevation

        return self._r_d_fine

    def get_coarse_indices(self, n_azimuth=360, n_elevation=90, step=16):
        """Return cached representative direction indices for coarse AOS."""
        need_recompute = (
            self._idx_tensor is None
            or self._step != step
            or self._n_azimuth != n_azimuth
            or self._n_elevation != n_elevation
        )

        if need_recompute:
            idx_list = select_representative_directions_idx(
                n_azimuth, n_elevation, step=step
            )
            self._idx_tensor = torch.tensor(
                idx_list, dtype=torch.long, device=self.device
            )
            self._step = step
            if self._n_azimuth is None:
                self._n_azimuth = n_azimuth
            if self._n_elevation is None:
                self._n_elevation = n_elevation

        return self._idx_tensor


def resolve_aos_mode(pipe):
    aos_mode = getattr(pipe, "aos_mode", "adaptive")
    aos_mode = str(aos_mode).lower()
    if aos_mode not in {"adaptive", "analytic"}:
        raise ValueError(f"Unsupported aos_mode: {aos_mode}")
    return aos_mode


def should_use_coarse_aos(pipe):
    if bool(getattr(pipe, "ablate_aos", False)):
        return False

    aos_mode = resolve_aos_mode(pipe)
    # `use_aos` in the CUDA tracer selects the legacy coarse-direction path.
    # Swap only the public mode mapping here.
    return aos_mode == "analytic"


def create_ray_direction_fine_v2(n_azimuth=360, n_elevation=90, radius=0.5):
    azimuth = torch.linspace(1, 360, n_azimuth) / 180 * np.pi
    elevation = torch.linspace(1, 90, n_elevation) / 180 * np.pi

    azimuth = torch.tile(azimuth, (n_elevation,))
    elevation = torch.repeat_interleave(elevation, n_azimuth)

    x = radius * torch.cos(elevation) * torch.cos(azimuth)
    y = radius * torch.cos(elevation) * torch.sin(azimuth)
    z = radius * torch.sin(elevation)

    return torch.stack([x, y, z], dim=0)


def select_representative_directions_idx(n_azimuth=360, n_elevation=90, step=16):
    def calculate_centers(n, step):
        centers = []
        for start in range(0, n, step):
            end = min(start + step, n)
            center = (start + end - 1) // 2
            centers.append(center)
        return centers

    azimuth_centers = calculate_centers(n_azimuth, step)
    elevation_centers = calculate_centers(n_elevation, step)

    representatives = []
    for elevation_idx in elevation_centers:
        for azimuth_idx in azimuth_centers:
            index = elevation_idx * n_azimuth + azimuth_idx
            representatives.append(index)

    return representatives


def render(
    viewpoint, pc: GaussianModel, pos_enc: Embedder, pipe, bg_color: torch.Tensor
):
    use_cuda_timer = torch.cuda.is_available()
    if use_cuda_timer:
        infer_start_event = torch.cuda.Event(enable_timing=True)
        infer_end_event = torch.cuda.Event(enable_timing=True)
        render_start_event = torch.cuda.Event(enable_timing=True)
        render_end_event = torch.cuda.Event(enable_timing=True)
        infer_start_event.record()
    else:
        infer_start_time = time.perf_counter()

    scaling_modifier = 1.0
    radius_rx = pipe.radius_rx

    means_3d = pc.get_xyz
    freq_effects = pc.apply_frequency_modulation(viewpoint)
    shs_coeffs_mod = freq_effects["shs_coeffs"]
    attenuation_mod = freq_effects["attenuation"]
    s_a = freq_effects["amp_scale"]
    s_p = freq_effects["phase_shift"]
    splat_scale = freq_effects["splat_scale"]
    freq_reg_loss = freq_effects["freq_reg_loss"]

    cov3d_precomp, _ = pc.get_covariance(scaling_modifier)
    cov3d_precomp = cov3d_precomp * splat_scale[:, None]

    tvec_rx = viewpoint.T_tx.to(means_3d.device, dtype=means_3d.dtype)
    tvec_tx = viewpoint.T_rx.to(means_3d.device, dtype=means_3d.dtype)

    cache = RenderCache.get(means_3d.device)
    r_d_fine_ori = cache.get_ray_directions(
        n_azimuth=int(viewpoint.width),
        n_elevation=int(viewpoint.height),
        radius_rx=radius_rx,
        dtype=means_3d.dtype,
    )

    r_d_w_fine = (r_d_fine_ori + tvec_rx[:, None]).permute(1, 0)

    shs_view = shs_coeffs_mod.transpose(1, 2).view(
        -1, pc.num_channels, (pc.max_sh_degree + 1) ** 2
    )

    dir_pp = means_3d - tvec_rx.repeat(means_3d.shape[0], 1)
    dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-6)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    use_aos = should_use_coarse_aos(pipe)
    if use_aos:
        idx_tensor = cache.get_coarse_indices(
            n_azimuth=int(viewpoint.width), n_elevation=int(viewpoint.height), step=16
        )
        r_d_w_coarse = r_d_w_fine[idx_tensor]
    else:
        r_d_w_coarse = r_d_w_fine

    tvec_tx_embd = pos_enc(tvec_tx.reshape(-1, 3).float()).squeeze(0)

    raster_settings_t = ComplexGaussianTracerSettings(
        height=int(viewpoint.height),
        width=int(viewpoint.width),
        sh_degree_active=pc.active_sh_degree,
        spectrum_3d_coarse=r_d_w_coarse,
        spectrum_3d_fine=r_d_w_fine,
        rx_pos=tvec_rx,
        radius_rx=radius_rx,
        tx_pos=tvec_tx_embd,
        bg=bg_color,
        use_aos=use_aos,
        debug=pipe.debug,
    )

    rasterizer = ComplexGaussianTracer(raster_settings=raster_settings_t)

    if use_cuda_timer:
        infer_end_event.record()
        render_start_event.record()
    else:
        infer_end_time = time.perf_counter()
        render_start_time = infer_end_time

    singal_amp, singal_pha = colors_precomp[:, 0], colors_precomp[:, 1]
    singal_amp = abs(F.leaky_relu(singal_amp)) * s_a
    singal_pha = torch.sigmoid(singal_pha) * np.pi * 2 + s_p
    stacked_signal = torch.stack((singal_amp, singal_pha), dim=1)

    rendered_image_complex = rasterizer(
        means_3d=means_3d,
        cov3d_precomp=cov3d_precomp,
        signal_precomp=stacked_signal,
        attenuation=attenuation_mod,
    )

    real_part = rendered_image_complex[0, :, :]
    imaginary_part = rendered_image_complex[1, :, :]
    rendered_image = torch.sqrt(real_part**2 + imaginary_part**2 + 1e-6)

    num_gaussians = means_3d.shape[0]
    visibility_filter = torch.ones(
        num_gaussians, dtype=torch.bool, device=means_3d.device
    )

    if use_cuda_timer:
        render_end_event.record()
        torch.cuda.synchronize()
        inference_ms = infer_start_event.elapsed_time(infer_end_event)
        render_ms = render_start_event.elapsed_time(render_end_event)
    else:
        render_end_time = time.perf_counter()
        inference_ms = (infer_end_time - infer_start_time) * 1000.0
        render_ms = (render_end_time - render_start_time) * 1000.0

    return {
        "render": rendered_image,
        "visibility_filter": visibility_filter,
        "radii": torch.ones(num_gaussians, device=means_3d.device),
        "timing": {
            "inference_ms": inference_ms,
            "render_ms": render_ms,
            "total_ms": inference_ms + render_ms,
        },
        "freq_reg_loss": freq_reg_loss,
    }
