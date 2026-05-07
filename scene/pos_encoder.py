import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]

        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        num_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=num_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FreqEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0

        if self.kwargs.get("include_input", True):
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.kwargs.get("include_physical", True):
            embed_fns.append(lambda x: 1.0 / (x + 1e-6))
            embed_fns.append(lambda x: torch.log10(x + 1e-6))
            out_dim += d * 2

        max_freq = self.kwargs["max_freq_log2"]
        num_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=num_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class Update_SH_Coeffs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
        )

    def forward(self, tvec_tx, shs_coeffs):
        num_points = shs_coeffs.size(0)
        tvec_tx = tvec_tx.repeat(num_points, 1)

        sig_amp, sig_pha = shs_coeffs.split(1, dim=-1)
        sig_amp = torch.cat((tvec_tx, sig_amp.squeeze(dim=-1)), dim=-1)
        sig_pha = torch.cat((tvec_tx, sig_pha.squeeze(dim=-1)), dim=-1)
        sig_amp = self.model(sig_amp)
        sig_pha = self.model(sig_pha)

        return torch.stack((sig_amp, sig_pha), dim=-1)


class WideFreqNetwork(nn.Module):
    """
    Frequency-conditioned modulation network for Gaussian signal properties.

    The frequency branch is computed once per spectrum and reused across points
    through FiLM-style modulation.
    """

    def __init__(
        self,
        input_dim_pos,
        input_dim_freq,
        input_dim_pts,
        num_sh_channels,
        num_sh_coeffs,
        point_code_dim=0,
        point_code_targets="all",
        sh_head_mode="channel",
        freq_sh_max_order=3,
        hidden_dim=256,
        D=8,
        skips=[1, 3, 5, 7],
    ):
        super().__init__()

        self.D = D
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.film_dim = hidden_dim // 2

        self.input_dim_freq = input_dim_freq
        self.input_dim_pos = input_dim_pos
        self.input_dim_pts = input_dim_pts
        self.num_sh_channels = num_sh_channels
        self.num_sh_coeffs = num_sh_coeffs
        self.point_code_dim = point_code_dim
        self.point_code_targets = str(point_code_targets).lower()
        self.sh_head_mode = str(sh_head_mode).lower()
        if self.point_code_targets not in {"all", "amp_att"}:
            raise ValueError(
                f"Unsupported point_code_targets: {self.point_code_targets}"
            )
        if self.sh_head_mode not in {"channel", "full", "order"}:
            raise ValueError(f"Unsupported sh_head_mode: {self.sh_head_mode}")

        self.point_code_in_trunk = (
            self.point_code_dim > 0 and self.point_code_targets == "all"
        )

        all_order_counts = [1, 3, 5, 7]
        self.freq_sh_max_order = min(freq_sh_max_order, 3)
        self.sh_order_counts = all_order_counts[: self.freq_sh_max_order + 1]
        self.num_orders = len(self.sh_order_counts)

        self.sh_output_dim = self.num_sh_channels
        if self.sh_head_mode == "full":
            self.sh_output_dim *= self.num_sh_coeffs
        elif self.sh_head_mode == "order":
            self.sh_output_dim = self.num_sh_channels * self.num_sh_coeffs

        self.freq_branch = nn.Sequential(
            nn.Linear(input_dim_freq, self.film_dim),
            nn.ReLU(),
            nn.Linear(self.film_dim, self.film_dim),
        )
        self.film_gamma = nn.Linear(self.film_dim, self.film_dim)
        self.film_beta = nn.Linear(self.film_dim, self.film_dim)

        self.spatial_branch = nn.Sequential(
            nn.Linear(input_dim_pos + input_dim_pts, self.film_dim),
            nn.ReLU(),
            nn.Linear(self.film_dim, self.film_dim),
        )

        self.point_code_proj = None
        if self.point_code_in_trunk:
            self.point_code_proj = nn.Sequential(
                nn.Linear(self.point_code_dim, self.film_dim // 2),
                nn.ReLU(),
                nn.Linear(self.film_dim // 2, self.film_dim // 2),
            )

        self.point_code_amp_att_head = None
        if self.point_code_dim > 0 and self.point_code_targets == "amp_att":
            self.point_code_amp_att_head = nn.Sequential(
                nn.Linear(self.point_code_dim, self.film_dim // 2),
                nn.ReLU(),
                nn.Linear(self.film_dim // 2, 2),
            )

        self.point_freq_head = None
        if self.point_code_dim > 0:
            self.point_freq_head = nn.Bilinear(self.point_code_dim, self.film_dim, 7)

        self.raw_input_dim = (
            input_dim_pos
            + input_dim_pts
            + input_dim_freq
            + (self.point_code_dim if self.point_code_in_trunk else 0)
        )
        point_feat_dim = 0 if self.point_code_proj is None else self.film_dim // 2
        self.combined_input_dim = self.raw_input_dim + self.film_dim + point_feat_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.combined_input_dim, hidden_dim))

        for i in range(D - 1):
            layer_idx = i + 1
            if layer_idx in skips:
                self.layers.append(
                    nn.Linear(hidden_dim + self.combined_input_dim, hidden_dim)
                )
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.out_layer = nn.Linear(hidden_dim, 3)
        self.splat_scale_head = nn.Linear(hidden_dim, 1)
        self.sh_scale_head = nn.Linear(hidden_dim, self.sh_output_dim)
        self.sh_bias_head = nn.Linear(hidden_dim, self.sh_output_dim)
        self._init_weights()

        self._cached_freq_feat = None
        self._cached_freq_emb_hash = None
        self._cached_gamma = None
        self._cached_beta = None

    def _init_weights(self):
        """Initialize output heads close to identity behavior."""
        nn.init.zeros_(self.out_layer.bias)
        with torch.no_grad():
            self.out_layer.bias[0] = 1.0
            self.out_layer.bias[2] = 1.0

        nn.init.zeros_(self.sh_scale_head.weight)
        nn.init.zeros_(self.sh_scale_head.bias)
        nn.init.zeros_(self.sh_bias_head.weight)
        nn.init.zeros_(self.sh_bias_head.bias)
        nn.init.zeros_(self.splat_scale_head.weight)
        nn.init.zeros_(self.splat_scale_head.bias)

        if self.point_freq_head is not None:
            nn.init.zeros_(self.point_freq_head.weight)
            nn.init.zeros_(self.point_freq_head.bias)

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        if self.point_code_amp_att_head is not None:
            last_linear = self.point_code_amp_att_head[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def _compute_freq_features(self, freq_emb):
        """
        Compute cached frequency features and FiLM parameters.

        Args:
            freq_emb: `[1, D_freq]` or `[D_freq]` frequency embedding.
        """
        if freq_emb.dim() == 1:
            freq_emb = freq_emb.unsqueeze(0)

        if self.training:
            freq_feat = self.freq_branch(freq_emb)
            gamma = self.film_gamma(freq_feat)
            beta = self.film_beta(freq_feat)
            return freq_feat, gamma, beta

        current_hash = (freq_emb.data_ptr(), freq_emb.shape, freq_emb.device)
        if (
            self._cached_freq_emb_hash == current_hash
            and self._cached_gamma is not None
        ):
            return self._cached_freq_feat, self._cached_gamma, self._cached_beta

        freq_feat = self.freq_branch(freq_emb)
        gamma = self.film_gamma(freq_feat)
        beta = self.film_beta(freq_feat)

        self._cached_freq_emb_hash = current_hash
        self._cached_freq_feat = freq_feat
        self._cached_gamma = gamma
        self._cached_beta = beta

        return freq_feat, gamma, beta

    def forward(self, tx_emb, pts_emb, freq_emb, point_code=None):
        """Forward pass for frequency-conditioned point modulation."""
        device = next(self.parameters()).device
        tx_emb = tx_emb.to(device)
        pts_emb = pts_emb.to(device)
        freq_emb = freq_emb.to(device)

        ablate_pc = point_code is None
        if self.point_code_dim > 0 and not ablate_pc:
            point_code = point_code.to(device)

        num_points = pts_emb.size(0)

        freq_feat, gamma, beta = self._compute_freq_features(freq_emb)

        if tx_emb.dim() == 1:
            tx_emb = tx_emb.unsqueeze(0)
        tx_exp = tx_emb.expand(num_points, -1) if tx_emb.shape[0] != num_points else tx_emb

        spatial_input = torch.cat([tx_exp, pts_emb], dim=-1)
        spatial_feat = self.spatial_branch(spatial_input)
        film_feat = gamma * spatial_feat + beta

        if freq_emb.dim() == 1:
            freq_emb = freq_emb.unsqueeze(0)
        freq_exp = (
            freq_emb.expand(num_points, -1)
            if freq_emb.shape[0] != num_points
            else freq_emb
        )

        raw_input_parts = [tx_exp, pts_emb, freq_exp]
        combined_parts = []

        if self.point_code_proj is not None and not ablate_pc:
            point_feat = self.point_code_proj(point_code)
            raw_input_parts.append(point_code)
            combined_parts.append(point_feat)
        elif self.point_code_proj is not None and ablate_pc:
            zero_pc = torch.zeros(num_points, self.point_code_dim, device=device)
            zero_feat = torch.zeros(num_points, self.film_dim // 2, device=device)
            raw_input_parts.append(zero_pc)
            combined_parts.append(zero_feat)

        raw_inputs = torch.cat(raw_input_parts, dim=-1)
        combined_inputs = torch.cat([raw_inputs, film_feat] + combined_parts, dim=-1)

        h = combined_inputs
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([combined_inputs, h], dim=-1)
            h = layer(h)
            h = F.relu(h)

        out = self.out_layer(h)
        splat_scale = self.splat_scale_head(h)
        sh_scale = self.sh_scale_head(h)
        sh_bias = self.sh_bias_head(h)

        if self.point_code_amp_att_head is not None and not ablate_pc:
            point_residual = self.point_code_amp_att_head(point_code)
            out = out.clone()
            out[:, 0] = out[:, 0] + point_residual[:, 0]
            out[:, 2] = out[:, 2] + point_residual[:, 1]

        if self.point_freq_head is not None and not ablate_pc:
            freq_feat_exp = (
                freq_feat.expand(num_points, -1)
                if freq_feat.shape[0] != num_points
                else freq_feat
            )
            point_freq_residual = self.point_freq_head(point_code, freq_feat_exp)
            out = out.clone()
            out[:, 0:3] = out[:, 0:3] + point_freq_residual[:, 0:3]
            if self.sh_head_mode == "channel":
                sh_scale = sh_scale + point_freq_residual[:, 3:5]
                sh_bias = sh_bias + point_freq_residual[:, 5:7]
            else:
                sh_scale = sh_scale.clone()
                sh_bias = sh_bias.clone()
                sh_scale[:, : self.num_sh_channels] = (
                    sh_scale[:, : self.num_sh_channels] + point_freq_residual[:, 3:5]
                )
                sh_bias[:, : self.num_sh_channels] = (
                    sh_bias[:, : self.num_sh_channels] + point_freq_residual[:, 5:7]
                )

        return {
            "amp_scale": out[:, 0:1],
            "phase_shift": out[:, 1:2],
            "att_scale": out[:, 2:3],
            "splat_scale": splat_scale,
            "sh_scale": sh_scale,
            "sh_bias": sh_bias,
        }

    def clear_cache(self):
        """Clear the cached frequency features."""
        self._cached_freq_feat = None
        self._cached_freq_emb_hash = None
        self._cached_gamma = None
        self._cached_beta = None
