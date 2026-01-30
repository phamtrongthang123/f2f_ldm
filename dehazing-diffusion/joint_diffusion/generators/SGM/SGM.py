"""SGM: Score-Based Generative Modeling
Ported from TensorFlow/Keras to PyTorch.
Inspired from: https://github.com/yang-song/score_sde
Author(s): Tristan Stevens
"""
import torch
import torch.nn as nn
import tqdm

from generators.layers import (
    ConvBlock,
    RefineBlock,
    ResidualBlock,
    get_activation,
    get_normalization,
)
from generators.SGM import sde_lib
from generators.SGM.sampling import ScoreSampler


class NCSNv2(nn.Module):
    """NCSNv2 score network with ResNet backbone and RefineNet decoder.

    Accepts config with attributes:
        image_shape: (C, H, W) — PyTorch channel-first format
        channels: int — base feature channels (nf)
        activation: str — e.g. 'elu'
        normalization: str — e.g. 'instance'
        kernel_size: int — e.g. 3
    """

    def __init__(self, config):
        super().__init__()
        # Support both (C, H, W) and (H, W, C) config formats
        image_shape = config.image_shape
        if len(image_shape) == 3:
            in_channels = image_shape[0]
        else:
            in_channels = image_shape[-1]

        act_name = config.activation
        norm_name = config.normalization
        nf = config.channels

        act = get_activation(act_name)
        norm_fn = get_normalization(norm_name)
        if norm_name.lower() == "layer":
            norm = lambda ch: norm_fn(1, ch)
        else:
            norm = norm_fn

        # Initial conv
        self.begin_conv = ConvBlock(
            in_channels, nf, kernel_size=3, stride=1,
            activation=act_name, normalization=None, bias=True,
        )

        # ResNet backbone (4 stages)
        self.res1 = nn.ModuleList([
            ResidualBlock(nf, nf, resample=None, act=act,
                          normalization=norm),
            ResidualBlock(nf, nf, resample=None, act=act,
                          normalization=norm),
        ])

        self.res2 = nn.ModuleList([
            ResidualBlock(nf, 2 * nf, resample="down", act=act,
                          normalization=norm),
            ResidualBlock(2 * nf, 2 * nf, resample=None, act=act,
                          normalization=norm),
        ])

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * nf, 2 * nf, resample="down", act=act,
                          normalization=norm, dilation=2),
            ResidualBlock(2 * nf, 2 * nf, resample=None, act=act,
                          normalization=norm, dilation=2),
        ])

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * nf, 2 * nf, resample="down", act=act,
                          normalization=norm, dilation=4),
            ResidualBlock(2 * nf, 2 * nf, resample=None, act=act,
                          normalization=norm, dilation=4),
        ])

        # RefineNet decoder
        self.refine1 = RefineBlock(
            [2 * nf], 2 * nf, act=act, start=True
        )
        self.refine2 = RefineBlock(
            [2 * nf, 2 * nf], 2 * nf, act=act
        )
        self.refine3 = RefineBlock(
            [2 * nf, 2 * nf], nf, act=act
        )
        self.refine4 = RefineBlock(
            [nf, nf], nf, act=act, end=True
        )

        # Output
        self.normalizer = norm(nf)
        self.act_out = act
        self.end_conv = nn.Conv2d(nf, in_channels, 3, stride=1, padding=1, bias=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x):
        h = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, h)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act_out(output)
        output = self.end_conv(output)

        return output


class ScoreNet(nn.Module):
    """A time-dependent score-based model.

    Wraps a backbone (e.g. NCSNv2) with SDE, loss computation, and sampling.

    Config attributes:
        image_shape: (C, H, W)
        sde: str — 'vesde', 'vpsde', 'subvpsde', 'simple'
        sigma_min, sigma_max (for VESDE)
        beta_min, beta_max (for VPSDE)
        num_scales: int — number of discretization steps
        score_backbone: str — backbone class name (default: 'NCSNv2')
        sampling_method: str — 'pc' or 'ode'
        predictor: str — predictor name
        corrector: str — corrector name
        snr: float — corrector SNR
        num_img: int — number of images for sampling shape
        image_range: tuple or None — clipping range for samples
        reduce_mean: bool — reduce loss by mean (else sum)
        likelihood_weighting: bool
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_shape = config.image_shape

        self.set_sde(config)

        score_backbone = getattr(config, "score_backbone", "NCSNv2")
        # Resolve backbone class
        backbone_cls = {"NCSNv2": NCSNv2}.get(score_backbone)
        if backbone_cls is None:
            raise ValueError(f"Unknown score backbone: {score_backbone}")
        self.model = backbone_cls(config)

        # Loss settings
        self.reduce_mean = getattr(config, "reduce_mean", True)
        self.likelihood_weighting = getattr(config, "likelihood_weighting", False)

    def set_sde(self, config):
        """Setup SDEs."""
        sde_name = config.sde.lower()
        if sde_name == "vpsde":
            self.sde = sde_lib.VPSDE(
                beta_min=config.beta_min,
                beta_max=config.beta_max,
                N=config.num_scales,
            )
        elif sde_name == "subvpsde":
            self.sde = sde_lib.subVPSDE(
                beta_min=config.beta_min,
                beta_max=config.beta_max,
                N=config.num_scales,
            )
        elif sde_name == "vesde":
            self.sde = sde_lib.VESDE(
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                N=config.num_scales,
            )
        elif sde_name == "simple":
            self.sde = sde_lib.simple(
                sigma=config.sigma,
                N=config.num_scales,
            )
        else:
            raise NotImplementedError(f"SDE {config.sde} unknown.")

    def forward(self, x, t):
        """Forward pass: return score for (x, t)."""
        score = self.get_score(x, t)
        return score

    def score_loss(self, batch):
        """Compute the denoising score matching loss.

        Args:
            batch: A mini-batch of training data, shape (B, C, H, W).

        Returns:
            loss: scalar loss value.
        """
        eps = 1e-5
        device = batch.device

        t = torch.rand(batch.shape[0], device=device) * (self.sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)

        # Expand std for broadcasting
        while std.dim() < batch.dim():
            std = std.unsqueeze(-1)

        perturbed_data = mean + std * z
        score = self.get_score(perturbed_data, t)

        if not self.likelihood_weighting:
            losses = torch.square(score * std + z)
            if self.reduce_mean:
                losses = torch.mean(
                    losses.reshape(losses.shape[0], -1), dim=-1
                )
            else:
                losses = 0.5 * torch.sum(
                    losses.reshape(losses.shape[0], -1), dim=-1
                )
        else:
            g2 = self.sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std)
            if self.reduce_mean:
                losses = torch.mean(
                    losses.reshape(losses.shape[0], -1), dim=-1
                ) * g2
            else:
                losses = 0.5 * torch.sum(
                    losses.reshape(losses.shape[0], -1), dim=-1
                ) * g2

        loss = torch.mean(losses)
        return loss

    def get_score(self, x, t, training=True):
        """Compute the score s(x, t) = model(x) / std(t).

        Args:
            x: input tensor (B, C, H, W)
            t: time tensor (B,)
            training: unused (kept for API compatibility)

        Returns:
            score: tensor same shape as x
        """
        score = self.model(x)
        _, std = self.sde.marginal_prob(x, t)
        while std.dim() < x.dim():
            std = std.unsqueeze(-1)
        score = score / std
        return score

    def get_latent_vector(self, batch_size):
        z = self.sde.prior_sampling([batch_size, *self.image_shape])
        return z

    @torch.no_grad()
    def sample(self, sampler=None, z=None, **kwargs):
        """Generate samples using the sampler.

        Args:
            sampler: a ScoreSampler instance. If None, must be set up externally.
            z: optional latent code.

        Returns:
            samples tensor.
        """
        if sampler is None:
            raise ValueError("Sampler must be provided for sampling.")
        samples = sampler(z=z, **kwargs)
        image_range = getattr(self.config, "image_range", None)
        if image_range is not None:
            samples = torch.clamp(samples, image_range[0], image_range[1])
        return samples

    def get_eval_loss(self, dataloader, n_batches=None, device="cpu"):
        """Compute average evaluation loss over a dataloader."""
        self.eval()
        losses = []

        if n_batches is None:
            n_batches = len(dataloader)
        else:
            n_batches = min(n_batches, len(dataloader))

        gen = iter(dataloader)
        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_batches), desc="Score eval loss"):
                batch = next(gen)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                loss = self.score_loss(batch)
                losses.append(loss.item())

        self.train()
        return sum(losses) / len(losses)
