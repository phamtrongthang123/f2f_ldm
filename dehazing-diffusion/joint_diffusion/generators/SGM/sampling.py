"""Sampling functionality for score-based diffusion models.
Ported from TensorFlow to PyTorch.
Author(s): Tristan Stevens
"""
import abc
import warnings

import numpy as np
import torch
from tqdm import tqdm

from generators.SGM import sde_lib
from generators.SGM.guidance import get_guidance

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered predictor with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered corrector with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    """Get predictor class for given name."""
    return _PREDICTORS[name]


def get_corrector(name):
    """Get corrector class for given name."""
    return _CORRECTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False, compute_grad=True):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn
        self.compute_grad = compute_grad
        self.grad_x0_xt = None

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
            x: current state tensor
            t: current time step tensor

        Returns:
            x: next state
            x_mean: next state without noise (denoised)
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
            x: current state tensor
            t: current time step tensor

        Returns:
            x: next state
            x_mean: next state without noise
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama diffusion sampler."""

    def update_fn(self, x, t):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)

        if self.compute_grad:
            # Enable gradient tracking for PIGDM / DPS guidance
            # Use torch.enable_grad() to ensure gradients are computed even in no_grad context
            with torch.enable_grad():
                x_input = x.detach().requires_grad_(True)
                drift, diffusion = self.rsde.sde(x_input, t)
                x_mean = x_input + drift * dt
                # Compute dx_mean/dx for guidance
                self.grad_x0_xt = torch.autograd.grad(
                    x_mean, x_input, grad_outputs=torch.ones_like(x_mean),
                    create_graph=False,
                )[0]
            x_mean = x_mean.detach()
            diffusion = diffusion.detach() if isinstance(diffusion, torch.Tensor) else diffusion
        else:
            drift, diffusion = self.rsde.sde(x, t)
            x_mean = x + drift * dt

        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    """Reverse diffusion sampler."""

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    """Langevin diffusion corrector."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(
            sde,
            (sde_lib.VPSDE, sde_lib.VESDE, sde_lib.subVPSDE, sde_lib.simple),
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE, sde_lib.simple)):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(
                grad.reshape(grad.shape[0], -1), dim=-1
            ).mean()
            noise_norm = torch.norm(
                noise.reshape(noise.shape[0], -1), dim=-1
            ).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(
            sde, (sde_lib.VPSDE, sde_lib.VESDE, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE)):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False, compute_grad=False):
        pass

    def update_fn(self, x, t):
        return x, x


class ScoreSampler:
    """Sampler class for score-based generative models.

    Supports both unconditional and conditional (posterior) sampling
    using predictor-corrector (PC) methods.
    """

    def __init__(
        self,
        model,
        image_shape,
        sde: sde_lib.SDE,
        sampling_method: str,
        predictor: str = None,
        corrector: str = None,
        guidance: str = None,
        corruptor=None,
        keep_track: bool = False,
        n_corrector_steps: int = 1,
        corrector_snr: float = 0.15,
        lambda_coeff: float = 0.1,
        kappa_coeff: float = 0.1,
        noise_model=None,
        noise_shape=None,
        start_diffusion: float = None,
        sampling_eps: float = None,
        early_stop: int = None,
        patch_overlap: int = 0,
        full_image_shape: tuple = None,
    ):
        assert sampling_method in ["pc", "ode"], f"{sampling_method} is not supported"

        self.image_shape = image_shape
        self.model = model
        self.sde = sde
        self.sampling_method = sampling_method
        self.corruptor = corruptor
        self.batch_size = None

        self.score_fn = lambda x, t: self.model.get_score(x, t, training=False)
        if noise_model:
            self.noise_score_fn = lambda x, t: noise_model.get_score(
                x, t, training=False
            )
            self.noise_model = noise_model
        else:
            self.noise_model = None

        self.keep_track = keep_track
        self.n_corrector_steps = n_corrector_steps
        self.corrector_snr = corrector_snr
        self.lambda_coeff = lambda_coeff
        self.kappa_coeff = kappa_coeff
        self.noise_shape = self.image_shape if noise_shape is None else noise_shape
        self.start_diffusion = start_diffusion
        self.eps = sampling_eps if sampling_eps is not None else 1e-3
        self.early_stop = early_stop
        self.patch_overlap = patch_overlap
        self.full_image_shape = full_image_shape
        self.guidance = guidance
        self.compute_grad = False

        if self.sampling_method == "pc":
            if self.guidance:
                if guidance.lower() in ["pigdm", "dps"]:
                    self.compute_grad = True
                    assert (
                        predictor == "euler_maruyama"
                    ), "PIGDM/DPS only supported by Euler-Maruyama predictor."
                self.guidance = get_guidance(guidance)(
                    self.sde, self.corruptor, self.lambda_coeff, self.kappa_coeff
                )
                print("Using guidance model: ", guidance)

            self.predictor, self.corrector = self.get_predictor_corrector_fn(
                predictor, corrector, self.score_fn,
            )
            if self.noise_model:
                (
                    self.noise_predictor,
                    self.noise_corrector,
                ) = self.get_predictor_corrector_fn(
                    predictor, corrector, self.noise_score_fn,
                )
        elif self.sampling_method == "ode":
            raise NotImplementedError("ODE not supported")

    def get_predictor_corrector_fn(self, predictor_name, corrector_name, score_fn):
        """Return predictor and corrector instances."""
        if predictor_name is None:
            predictor = NonePredictor(
                self.sde, score_fn, probability_flow=False
            )
        else:
            predictor_cls = get_predictor(predictor_name.lower())
            predictor = predictor_cls(
                self.sde, score_fn, probability_flow=False,
                compute_grad=self.compute_grad,
            )

        if corrector_name is None:
            corrector = NoneCorrector(
                self.sde, score_fn, self.corrector_snr, self.n_corrector_steps,
            )
        else:
            corrector_cls = get_corrector(corrector_name.lower())
            corrector = corrector_cls(
                self.sde, score_fn, self.corrector_snr, self.n_corrector_steps,
            )
        return predictor, corrector

    def _extract_patches(self, image, patch_h, patch_w, overlap):
        """Extract overlapping patches from a full image.

        Args:
            image: (B, C, H, W) tensor
            patch_h: patch height
            patch_w: patch width
            overlap: number of overlapping pixels

        Returns:
            patches: (B*N*M, C, patch_h, patch_w) tensor
            grid_info: dict with grid dimensions and parameters
        """
        B, C, H, W = image.shape
        stride_h = patch_h - overlap
        stride_w = patch_w - overlap

        # Number of patches in each dimension
        n_rows = max(1, (H - patch_h) // stride_h + 1)
        n_cols = max(1, (W - patch_w) // stride_w + 1)

        patches = []
        for i in range(n_rows):
            for j in range(n_cols):
                top = min(i * stride_h, H - patch_h)
                left = min(j * stride_w, W - patch_w)
                patch = image[:, :, top:top + patch_h, left:left + patch_w]
                patches.append(patch)

        patches = torch.cat(patches, dim=0)  # (B*N*M, C, patch_h, patch_w)

        grid_info = {
            "batch_size": B,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "overlap": overlap,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "H": H,
            "W": W,
        }
        return patches, grid_info

    def _interleave_patches(self, patches, grid_info):
        """Copy overlapping pixels from each patch to adjacent patches (Algorithm 1, lines 48-51).

        For each patch (n, m), copy its overlap region to neighbors:
          (n, m-1), (n-1, m), (n-1, m-1)

        Args:
            patches: (B*N*M, C, patch_h, patch_w) tensor
            grid_info: dict from _extract_patches

        Returns:
            patches: updated tensor with interleaved overlap regions
        """
        B = grid_info["batch_size"]
        n_rows = grid_info["n_rows"]
        n_cols = grid_info["n_cols"]
        overlap = grid_info["overlap"]
        patch_h = grid_info["patch_h"]
        patch_w = grid_info["patch_w"]

        if overlap <= 0:
            return patches

        num_patches = n_rows * n_cols

        def idx(b, r, c):
            """Get flat index for batch b, row r, col c."""
            return b * num_patches + r * n_cols + c

        for b in range(B):
            for n in range(n_rows):
                for m in range(n_cols):
                    src_idx = idx(b, n, m)

                    # Copy left overlap to (n, m-1)
                    if m > 0:
                        dst_idx = idx(b, n, m - 1)
                        # Left side of src = right side of dst
                        patches[dst_idx, :, :, patch_w - overlap:] = \
                            patches[src_idx, :, :, :overlap].clone()

                    # Copy top overlap to (n-1, m)
                    if n > 0:
                        dst_idx = idx(b, n - 1, m)
                        # Top side of src = bottom side of dst
                        patches[dst_idx, :, patch_h - overlap:, :] = \
                            patches[src_idx, :, :overlap, :].clone()

                    # Copy top-left corner to (n-1, m-1)
                    if n > 0 and m > 0:
                        dst_idx = idx(b, n - 1, m - 1)
                        patches[dst_idx, :, patch_h - overlap:, patch_w - overlap:] = \
                            patches[src_idx, :, :overlap, :overlap].clone()

        return patches

    def _stitch_patches(self, patches, grid_info):
        """Reassemble patches into full image. Last-write-wins for overlap regions.

        Args:
            patches: (B*N*M, C, patch_h, patch_w) tensor
            grid_info: dict from _extract_patches

        Returns:
            image: (B, C, H, W) tensor
        """
        B = grid_info["batch_size"]
        n_rows = grid_info["n_rows"]
        n_cols = grid_info["n_cols"]
        patch_h = grid_info["patch_h"]
        patch_w = grid_info["patch_w"]
        H = grid_info["H"]
        W = grid_info["W"]
        stride_h = grid_info["stride_h"]
        stride_w = grid_info["stride_w"]
        num_patches = n_rows * n_cols

        C = patches.shape[1]
        image = torch.zeros(B, C, H, W, device=patches.device, dtype=patches.dtype)

        for b in range(B):
            for i in range(n_rows):
                for j in range(n_cols):
                    p_idx = b * num_patches + i * n_cols + j
                    top = min(i * stride_h, H - patch_h)
                    left = min(j * stride_w, W - patch_w)
                    image[b, :, top:top + patch_h, left:left + patch_w] = patches[p_idx]

        return image

    def __call__(self, y=None, **kwargs):
        if y is None:
            x = self._sample(**kwargs)
        else:
            x = self._conditional_sample(y, **kwargs)
        return x

    def _sample(self, z=None, shape=None, progress_bar=True):
        if self.sampling_method == "pc":
            x = self.pc_sampler(z=z, shape=shape, progress_bar=progress_bar)
        elif self.sampling_method == "ode":
            raise NotImplementedError("ODE not supported")
        return x

    def _conditional_sample(self, y, progress_bar=True):
        if self.sampling_method == "pc":
            x = self.pc_sampler(y=y, progress_bar=progress_bar)
        elif self.sampling_method == "ode":
            raise NotImplementedError("ODE not supported")
        return x

    def pc_sampler(self, y=None, z=None, shape=None, progress_bar=True):
        """The PC sampler function.

        Note: @torch.no_grad() removed to allow gradient computation for PIGDM/DPS guidance.

        Args:
            y: measurement for conditional sampling. None = unconditional.
            z: latent code for unconditional sampling.
            shape: shape for prior sampling.
            progress_bar: whether to show progress bar.

        Returns:
            samples (and noise estimates for joint inference).
        """
        device = next(self.model.parameters()).device
        use_patches = self.patch_overlap > 0 and self.full_image_shape is not None

        # Initialize
        if y is None:
            if z is None:
                x = self.sde.prior_sampling(shape).to(device)
            else:
                x = z.float().to(device)
            self.batch_size = x.shape[0]
        else:
            y = y.float().to(device)
            self.batch_size = y.shape[0]
            shape = (self.batch_size, *self.image_shape)
            self.guidance.batch_size = self.batch_size
            self.guidance.image_shape = self.image_shape

            if self.noise_model is not None:
                noise_shape = (self.batch_size, *self.noise_shape)
                self.guidance.noise_shape = self.noise_shape

            if (self.start_diffusion is not None) and self.start_diffusion > 0:
                t_start = torch.tensor(
                    self.sde.T - self.start_diffusion, device=device
                )
                t_batch = t_start.expand(self.batch_size)
                x = self.sde.forward_diffuse(y, t_batch)
            else:
                x = self.sde.prior_sampling(shape).to(device)

        # Patch extraction (Algorithm 1)
        grid_info = None
        if use_patches and y is not None:
            patch_h, patch_w = self.image_shape[-2], self.image_shape[-1]
            y_patches, grid_info = self._extract_patches(
                y, patch_h, patch_w, self.patch_overlap
            )
            x_patches, _ = self._extract_patches(
                x, patch_h, patch_w, self.patch_overlap
            )
            if self.noise_model is not None:
                # Initialize noise at full image size, then extract patches
                n_full_shape = (self.batch_size, self.noise_shape[0], y.shape[2], y.shape[3])
                n_full = self.sde.prior_sampling(n_full_shape).to(device)
                n_patches, _ = self._extract_patches(
                    n_full, patch_h, patch_w, self.patch_overlap
                )
            # Update batch size to number of total patches
            total_patches = x_patches.shape[0]
            self.batch_size = total_patches
            self.guidance.batch_size = total_patches
            x = x_patches
            y = y_patches
            if self.noise_model is not None:
                n = n_patches
        elif self.noise_model is not None and y is not None:
            n = self.sde.prior_sampling(noise_shape).to(device)

        # Tracking
        if self.keep_track:
            x_list = [x.clone()]
            if self.noise_model and y is not None:
                n_list = [n.clone()]

        # Diffusion timeline
        if self.start_diffusion:
            timesteps = torch.linspace(
                self.sde.T - self.start_diffusion, self.eps, self.sde.N,
                device=device,
            )
        else:
            timesteps = torch.linspace(
                self.sde.T, self.eps, self.sde.N, device=device
            )

        if self.early_stop:
            timesteps = timesteps[: self.early_stop]

        iterator = tqdm(timesteps, desc="Sampling") if progress_bar else timesteps

        # Main reverse diffusion loop
        for t in iterator:
            vec_t = torch.ones(self.batch_size, device=device) * t

            x, x_mean = self.corrector.update_fn(x, vec_t)
            x, x_mean = self.predictor.update_fn(x, vec_t)

            # Data consistency steps
            if y is not None:
                assert self.guidance is not None, (
                    "Please select a guidance model for conditional sampling."
                )

                if self.noise_model is not None:
                    n, n_mean = self.noise_corrector.update_fn(n, vec_t)
                    n, n_mean = self.noise_predictor.update_fn(n, vec_t)

                    x, n = self.guidance.joint_update_fn(
                        y, x, n, vec_t, x_mean, n_mean,
                        self.predictor.grad_x0_xt,
                        self.noise_predictor.grad_x0_xt,
                    )
                else:
                    x = self.guidance.update_fn(
                        y, x, vec_t, x_mean, self.predictor.grad_x0_xt
                    )

            # Patch interleaving (Algorithm 1, lines 47-51)
            if grid_info is not None:
                x = self._interleave_patches(x, grid_info)
                if self.noise_model is not None and y is not None:
                    n = self._interleave_patches(n, grid_info)

            if self.keep_track:
                x_list.append(x_mean.clone())
                if self.noise_model and y is not None:
                    n_list.append(n_mean.clone())

            if torch.isnan(x.sum()):
                warnings.warn("NaN in intermediate solution, breaking out...")
                break

        # Stitch patches back together
        if grid_info is not None:
            x_mean = self._stitch_patches(x_mean, grid_info)
            if self.noise_model and y is not None:
                n_mean = self._stitch_patches(n_mean, grid_info)
            if self.keep_track:
                x_list = [self._stitch_patches(xi, grid_info) for xi in x_list]
                if self.noise_model and y is not None:
                    n_list = [self._stitch_patches(ni, grid_info) for ni in n_list]

        if self.noise_model and y is not None:
            return (
                x_list if self.keep_track else x_mean,
                n_list if self.keep_track else n_mean,
            )
        else:
            return x_list if self.keep_track else x_mean
