"""Inverse tasks (PyTorch version)
Author(s): Tristan Stevens
Ported to PyTorch: Jan 2026
"""
import abc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from easydict import EasyDict as edict

from generators.models import get_model
from generators.SGM.sampling import ScoreSampler
from utils.checkpoints import ModelCheckpoint
from utils.corruptors import get_corruptor
from utils.utils import get_date_filename, save_animation, timefunc, torch_to_numpy

# Optional imports
try:
    from utils.metrics import Metrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    Metrics = None

try:
    from bm3d import BM3DStages, bm3d
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False

try:
    from skimage.restoration import denoise_nl_means
    HAS_NLM = True
except ImportError:
    HAS_NLM = False

_DENOISERS = {}
_MODEL_NAMES = {
    "gt": "Ground\nTruth",
    "noisy": "Input",
    "noise": "Noise",
    "gan": "GAN",
    "glow": "FLOW",
    "sgm": "DIFFUSION",
    "sgm_dps": "DPS",
    "sgm_proj": "Proj.",
    "sgm_pigdm": r"$\Pi$GDM",
    "bm3d": "BM3D",
    "nlm": "NLM",
    "wvtcs": "LASSO",
}


def register_denoiser(cls=None, *, name=None):
    """A decorator for registering denoiser classes."""
    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _DENOISERS:
            raise ValueError(f"Already registered denoiser with name: {local_name}")
        _DENOISERS[local_name] = cls
        cls.name = local_name
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_denoiser(name):
    """Retrieve a denoiser class by name."""
    return _DENOISERS[name]


def get_list_of_denoisers():
    """Get all registered denoiser names."""
    return list(_DENOISERS.keys()) + ["sgm_proj", "sgm_dps", "sgm_pigdm"]


class Denoiser(abc.ABC):
    """Base denoiser abstract class."""

    @abc.abstractmethod
    def __init__(
        self,
        config,
        dataset=None,
        num_img: int = None,
        metrics: list = None,
        keep_track: bool = None,
        sweep_id: str = None,
        corruptor=None,
        verbose: bool = True,
    ):
        """Initialize denoiser.
        
        Args:
            config: Configuration dict/object
            dataset: Optional DataLoader for data
            num_img: Number of images to process
            metrics: List of metric names to compute
            keep_track: Whether to track intermediate steps
            sweep_id: Optional sweep identifier
            corruptor: Optional corruptor instance
            verbose: Whether to print progress
        """
        config.denoiser = self.name
        self.config = config
        self.dataset = dataset
        self.vmin, self.vmax = config.image_range
        self.verbose = verbose

        self.model = None
        self.target_samples = None
        self.noisy_samples = None
        self.denoised_samples = None

        # Set attributes from config
        self.num_img = num_img or config.get("num_img", 8)
        self.keep_track = keep_track if keep_track is not None else config.get("keep_track", False)
        self.sweep_id = sweep_id or config.get("sweep_id")
        self.metrics_list = metrics or config.get("metrics")

        # Batch size
        self.batch_size = None
        if self.name in config:
            if "batch_size" in config[self.name]:
                self.batch_size = config[self.name].batch_size
                config.batch_size = self.batch_size

        # Corruptor
        if corruptor is None:
            corruptor_name = config.get("corruptor", "gaussian")
            self.corruptor = get_corruptor(corruptor_name)(config, train=False)
        else:
            self.corruptor = corruptor

        # Metrics
        self.metrics = None
        if self.metrics_list and HAS_METRICS:
            if config.get("paired_data", True):
                self.metrics = Metrics(self.metrics_list, config.image_range)

        self.eval_noisy = None
        self.eval_denoised = None
        self.model_names = _MODEL_NAMES

    def __call__(
        self,
        noisy_samples=None,
        target_samples=None,
        plot=True,
        save=True,
    ):
        """Run denoising.
        
        Args:
            noisy_samples: Input noisy samples (optional)
            target_samples: Ground truth samples (optional)
            plot: Whether to plot results
            save: Whether to save plot
        
        Returns:
            Denoised samples
        """
        # Set data
        self.set_data(noisy_samples, target_samples)

        # Run denoising
        self._call()

        # Evaluate metrics
        if self.metrics:
            self.get_metrics()

        # Plot results
        if plot:
            self.plot(save=save, figsize=self.config.get("figsize"))

        return self.denoised_samples

    def _call(self):
        """Internal call method."""
        self.denoised_samples = self._denoise(self.noisy_samples)

    @abc.abstractmethod
    def _denoise(self, images):
        """Abstract denoising method."""
        return images

    def set_data(self, noisy_samples=None, target_samples=None):
        """Set input data for denoising.
        
        Args:
            noisy_samples: Optional noisy input
            target_samples: Optional ground truth
        
        Returns:
            Tuple of (noisy_samples, target_samples)
        """
        if target_samples is None:
            if self.dataset is not None:
                batch = next(iter(self.dataset))
                if isinstance(batch, (tuple, list)):
                    self.target_samples = batch[0][:self.num_img]
                else:
                    self.target_samples = batch[:self.num_img]
            else:
                raise ValueError("Provide dataset or target_samples")
        else:
            self.target_samples = target_samples
            self.num_img = len(target_samples)

        # Ensure tensor
        if isinstance(self.target_samples, np.ndarray):
            self.target_samples = torch.from_numpy(self.target_samples)

        if self.config.get("paired_data", True):
            if noisy_samples is None:
                self.noisy_samples = self.corruptor.corrupt(self.target_samples)
            else:
                self.noisy_samples = noisy_samples
        else:
            self.noisy_samples = self.target_samples

        return self.noisy_samples, self.target_samples

    def get_metrics(self):
        """Compute evaluation metrics."""
        if self.metrics is None:
            return

        denoised = self.denoised_samples
        if self.keep_track and isinstance(denoised, list):
            denoised = denoised[-1]

        # Handle joint model output
        if isinstance(denoised, tuple):
            denoised = denoised[0]
            if self.keep_track and isinstance(denoised, list):
                denoised = denoised[-1]

        # Convert to numpy for metrics
        if isinstance(denoised, torch.Tensor):
            denoised = denoised.detach().cpu().numpy()
        if isinstance(self.target_samples, torch.Tensor):
            target = self.target_samples.detach().cpu().numpy()
        else:
            target = self.target_samples

        self.eval_denoised = self.metrics.eval_metrics(target, denoised)

        if self.verbose:
            self.metrics.print_results(self.eval_denoised)

    def plot(self, save=True, zoom=None, figsize=None, dpi=300):
        """Plot denoising results."""
        display_bmode = self.config.get("display_bmode", False)

        denoised = self.denoised_samples
        if self.keep_track and isinstance(denoised, list):
            denoised = denoised[-1]

        # Handle joint model output
        noise_samples = None
        if isinstance(denoised, tuple):
            denoised, noise_samples = denoised
            if self.keep_track and isinstance(denoised, list):
                denoised = denoised[-1]
                noise_samples = noise_samples[-1] if noise_samples else None

        # Convert tensors to numpy (keep BCHW for B-mode conversion)
        def to_numpy(x, keep_bchw=False):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if not keep_bchw:
                # Convert from (B, C, H, W) to (B, H, W, C)
                if x.ndim == 4 and x.shape[1] in [1, 3]:
                    x = x.transpose(0, 2, 3, 1)
                return np.clip(x, self.vmin, self.vmax)
            return x

        if display_bmode:
            from utils.bmode import rf_to_bmode, extent_mm, undo_normalization

            dynamic_range = tuple(self.config.get("dynamic_range", [-50, 0]))

            # Get original data scale from dataset for full denormalization
            ds = getattr(self.dataset, "dataset", None)  # unwrap DataLoader
            data_min = getattr(ds, "data_min", None)
            data_max = getattr(ds, "data_max", None)

            def to_bmode(x, label=""):
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                print(f"[{label}] raw normalized: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}")
                x = np.clip(x, self.vmin, self.vmax)
                x = undo_normalization(
                    x, image_range=(self.vmin, self.vmax),
                    data_min=data_min, data_max=data_max,
                )
                print(f"[{label}] after undo_norm: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
                bmode = rf_to_bmode(x, dynamic_range=dynamic_range)
                b0 = np.asarray(bmode[0])
                print(f"[{label}] bmode[0]: min={b0.min()}, max={b0.max()}, mean={b0.mean():.1f}")
                return bmode

            print(f"[diag] data_min={data_min}, data_max={data_max}, vmin={self.vmin}, vmax={self.vmax}")
            target_imgs = to_bmode(self.target_samples, "target") if self.target_samples is not None else None
            noisy_imgs = to_bmode(self.noisy_samples, "noisy")
            denoised_imgs = to_bmode(denoised, "denoised")
        else:
            target = to_numpy(self.target_samples) if self.target_samples is not None else None
            noisy = to_numpy(self.noisy_samples)
            denoised = to_numpy(denoised)

        # Setup figure
        n_cols = 3 if (self.target_samples is not None) else 2
        if noise_samples is not None:
            n_cols += 1
        if display_bmode and self.target_samples is not None:
            n_cols += 1  # diff column

        num_img = len(self.noisy_samples)
        if figsize is None:
            figsize = (n_cols * 3, num_img * 2)

        fig, axs = plt.subplots(num_img, n_cols, figsize=figsize)
        if num_img == 1:
            axs = axs.reshape(1, -1)

        titles = []
        samples_list = []

        if display_bmode:
            if target_imgs is not None:
                titles.append("Ground Truth")
                samples_list.append(target_imgs)
            titles.append("Noisy")
            samples_list.append(noisy_imgs)
            titles.append(self.model_names.get(self.name, self.name))
            samples_list.append(denoised_imgs)
            if noise_samples is not None:
                titles.append("Noise Posterior")
                samples_list.append(to_bmode(noise_samples, "noise_post"))
            # Add diff column: |GT - Denoised| in B-mode pixel space
            if target_imgs is not None:
                diff_imgs = []
                for t, d in zip(target_imgs, denoised_imgs):
                    ta, da = np.asarray(t, dtype=np.float32), np.asarray(d, dtype=np.float32)
                    diff = np.abs(ta - da).clip(0, 255).astype(np.uint8)
                    diff_imgs.append(diff)
                titles.append("|GT - Denoised|")
                samples_list.append(diff_imgs)
        else:
            if target is not None:
                titles.append("Ground Truth")
                samples_list.append(target)
            titles.append("Noisy")
            samples_list.append(noisy)
            titles.append(self.model_names.get(self.name, self.name))
            samples_list.append(denoised)
            if noise_samples is not None:
                titles.append("Noise Posterior")
                samples_list.append(to_numpy(noise_samples))

        for n in range(num_img):
            for i, (sample, title) in enumerate(zip(samples_list, titles)):
                if display_bmode:
                    img = np.asarray(sample[n])
                    is_diff = title.startswith("|")
                    cmap = "hot" if is_diff else "gray"
                    axs[n, i].imshow(img, cmap=cmap, vmin=0, vmax=255, extent=extent_mm)
                    axs[n, i].set_xlabel("X (mm)")
                    axs[n, i].set_ylabel("Z (mm)")
                else:
                    img = np.squeeze(sample[n])
                    axs[n, i].imshow(img, cmap="gray", vmin=self.vmin, vmax=self.vmax)
                    axs[n, i].axis("off")
                if n == 0:
                    axs[n, i].set_title(title)

        fig.tight_layout()

        if save:
            self.savefig(fig, dpi=dpi, path=save if isinstance(save, (str, Path)) else None)

        # Save raw normalized-space diff figure for debugging scale issues
        if display_bmode and self.target_samples is not None:
            target_np = self.target_samples.detach().cpu().numpy() if isinstance(self.target_samples, torch.Tensor) else self.target_samples
            denoised_np = denoised.detach().cpu().numpy() if isinstance(denoised, torch.Tensor) else denoised
            num = min(len(target_np), 3)  # show up to 3 samples
            fig_diff, axes = plt.subplots(num, 4, figsize=(16, num * 3))
            if num == 1:
                axes = axes.reshape(1, -1)
            for n in range(num):
                # Pick first channel for display
                gt = target_np[n, 0] if target_np.ndim == 4 else target_np[n]
                dn = denoised_np[n, 0] if denoised_np.ndim == 4 else denoised_np[n]
                diff = gt - dn

                im0 = axes[n, 0].imshow(gt, cmap="gray", vmin=self.vmin, vmax=self.vmax)
                axes[n, 0].set_title(f"GT [{gt.min():.3f}, {gt.max():.3f}]" if n == 0 else f"[{gt.min():.3f}, {gt.max():.3f}]")
                fig_diff.colorbar(im0, ax=axes[n, 0], fraction=0.046)

                im1 = axes[n, 1].imshow(dn, cmap="gray", vmin=self.vmin, vmax=self.vmax)
                axes[n, 1].set_title(f"Denoised [{dn.min():.3f}, {dn.max():.3f}]" if n == 0 else f"[{dn.min():.3f}, {dn.max():.3f}]")
                fig_diff.colorbar(im1, ax=axes[n, 1], fraction=0.046)

                im2 = axes[n, 2].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
                axes[n, 2].set_title(f"GT-Denoised [{diff.min():.3f}, {diff.max():.3f}]" if n == 0 else f"[{diff.min():.3f}, {diff.max():.3f}]")
                fig_diff.colorbar(im2, ax=axes[n, 2], fraction=0.046)

                im3 = axes[n, 3].imshow(np.abs(diff), cmap="hot", vmin=0, vmax=0.5)
                axes[n, 3].set_title(f"|diff| [{np.abs(diff).min():.3f}, {np.abs(diff).max():.3f}]" if n == 0 else f"[{np.abs(diff).min():.3f}, {np.abs(diff).max():.3f}]")
                fig_diff.colorbar(im3, ax=axes[n, 3], fraction=0.046)

                for ax in axes[n]:
                    ax.axis("off")

            fig_diff.suptitle("Raw normalized-space comparison (channel 0)", fontsize=12)
            fig_diff.tight_layout()
            diff_path = str(save if isinstance(save, (str, Path)) else "").replace(".png", "_diff.png") if save else None
            if not diff_path:
                diff_path = "figures/debug_diff.png"
            Path(diff_path).parent.mkdir(parents=True, exist_ok=True)
            fig_diff.savefig(diff_path, dpi=150, bbox_inches="tight")
            print(f"Saved diff plot to {diff_path}")
            plt.close(fig_diff)

        return fig

    def savefig(self, fig, dpi=300, path=None):
        """Save figure."""
        if path is None:
            filename = f"{self.name.lower()}_{self.config.dataset_name}_{self.corruptor.task}"
            folder = "figures"
            if self.sweep_id:
                folder = f"{folder}/{self.sweep_id}"
            path = get_date_filename(f"{folder}/{filename}.png")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot to {path}")

    def animate(self, duration=5, dpi=150, save=True):
        """Create animation from denoising history.
        
        Args:
            duration: Duration of animation in seconds
            dpi: Figure DPI
            save: Whether to save the animation
        
        Returns:
            matplotlib animation or None
        """
        if not self.keep_track:
            print("No history tracked, skipping animation")
            return None
        
        denoised = self.denoised_samples
        if not isinstance(denoised, list):
            print("No history available for animation")
            return None
        
        # Handle joint output
        if isinstance(denoised[0], tuple):
            denoised = [d[0] for d in denoised]
        
        num_frames = len(denoised)
        if num_frames < 2:
            print("Not enough frames for animation")
            return None
        
        try:
            from matplotlib.animation import FuncAnimation
            
            # Convert to numpy helper
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                if x.ndim == 4 and x.shape[1] in [1, 3]:
                    x = x.transpose(0, 2, 3, 1)
                return np.clip(x, self.vmin, self.vmax)
            
            # Setup figure
            num_img = min(4, len(to_numpy(denoised[0])))
            fig, axs = plt.subplots(1, num_img, figsize=(num_img * 3, 3))
            if num_img == 1:
                axs = [axs]
            
            # Initialize images
            frame0 = to_numpy(denoised[0])
            ims = []
            for n in range(num_img):
                img = np.squeeze(frame0[n])
                im = axs[n].imshow(img, cmap="gray", vmin=self.vmin, vmax=self.vmax)
                axs[n].axis("off")
                ims.append(im)
            
            def update(frame_idx):
                frame = to_numpy(denoised[frame_idx])
                for n in range(num_img):
                    img = np.squeeze(frame[n])
                    ims[n].set_array(img)
                return ims
            
            interval = (duration * 1000) / num_frames
            anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
            
            if save:
                filename = f"{self.name.lower()}_{self.config.dataset_name}_{self.corruptor.task}_animation"
                path = get_date_filename(f"figures/{filename}.gif")
                anim.save(path, writer="pillow", fps=num_frames / duration, dpi=dpi)
                print(f"Saved animation to {path}")
            
            plt.close(fig)
            return anim
        except Exception as e:
            print(f"Could not create animation: {e}")
            return None


@register_denoiser(name="none")
class NoneDenoiser(Denoiser):
    """Identity denoiser (does nothing)."""

    def __init__(self, config, dataset=None, num_img=None, metrics=None, **kwargs):
        super().__init__(config, dataset, num_img, metrics, keep_track=False, **kwargs)

    def _denoise(self, images):
        if self.verbose:
            print("\nNone Denoiser (identity)...")
        return images


@register_denoiser(name="sgm")
class SGMDenoiser(Denoiser):
    """Score-based generative model (diffusion) denoiser."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img: int = None,
        metrics: list = None,
        keep_track: bool = None,
        device=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            num_img=num_img,
            metrics=metrics,
            keep_track=keep_track,
            **kwargs,
        )
        
        # Merge SGM-specific config
        if "sgm" in config:
            self.config = edict({**self.config, **self.config.sgm})

        # Device
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = get_model(self.config, training=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load checkpoint
        ckpt = ModelCheckpoint(self.model, config=self.config)
        ckpt.restore(self.config.get("checkpoint_file"))

        # Load corruptor model for joint inference (e.g., haze model for dehazing)
        self._load_corruptor_model()

        # Setup sampler
        self.set_sampler()

    def _load_corruptor_model(self):
        """Load corruptor noise model from corruptor_run_id if specified."""
        from utils.runs import init_config
        
        corruptor_run_id = self.config.get("corruptor_run_id")
        if corruptor_run_id is None:
            return
        
        print(f"Loading corruptor model from: {corruptor_run_id}")
        
        # Load corruptor model config
        corr_config = init_config(corruptor_run_id, just_dataset=False, verbose=False)
        
        # Handle image_size vs image_shape naming inconsistency
        if "image_size" in corr_config and "image_shape" not in corr_config:
            # image_size is [H, W], image_shape should be [C, H, W]
            img_size = corr_config.image_size
            # For ZEA datasets, get channels from the main model's image_shape
            # (both tissue and haze have same number of transmits)
            if hasattr(self.config, 'image_shape'):
                channels = self.config.image_shape[0]
            else:
                color_mode = corr_config.get("color_mode", "grayscale")
                channels = 1 if color_mode == "grayscale" else 3
            corr_config.image_shape = [channels, *img_size]
        
        # Build and load the corruptor model
        noise_model = get_model(corr_config, training=False)
        noise_model = noise_model.to(self.device)
        noise_model.eval()
        
        # Load checkpoint
        corr_ckpt = ModelCheckpoint(noise_model, config=corr_config)
        corr_ckpt.restore()
        
        # Attach to corruptor
        self.corruptor.model = noise_model
        self.corruptor.image_shape = corr_config.image_shape
        
        print(f"Loaded corruptor model: {type(noise_model).__name__}")

    def set_sampler(self):
        """Initialize the score sampler."""
        # Get noise model for joint inference
        noise_model = getattr(self.corruptor, "model", None)
        noise_shape = getattr(self.corruptor, "image_shape", None)

        self.sampler = ScoreSampler(
            model=self.model,
            sde=self.model.sde,
            image_shape=self.config.image_shape,
            sampling_method=self.config.get("sampling_method", "pc"),
            predictor=self.config.get("predictor", "euler_maruyama"),
            corrector=self.config.get("corrector", "none"),
            guidance=self.config.get("guidance"),
            corruptor=self.corruptor,
            keep_track=self.keep_track,
            corrector_snr=self.config.get("snr", 0.16),
            lambda_coeff=self.config.get("lambda_coeff"),
            kappa_coeff=self.config.get("kappa_coeff"),
            noise_model=noise_model,
            noise_shape=noise_shape,
            start_diffusion=self.config.get("ccdf"),
            sampling_eps=self.config.get("sampling_eps"),
            early_stop=self.config.get("early_stop"),
            patch_overlap=self.config.get("patch_overlap", 0),
            full_image_shape=tuple(self.config.image_shape) if self.config.get("patch_overlap", 0) > 0 else None,
        )

    @timefunc
    def _call(self):
        self.denoised_samples = self._denoise(self.noisy_samples)

    def _denoise(self, images):
        """Run diffusion sampling for denoising.
        
        Args:
            images: Noisy input images (tensor)
        
        Returns:
            Denoised samples
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.to(self.device)

        with torch.no_grad():
            denoised = self.sampler(y=images, progress_bar=self.verbose)

        # Clamp to training range (same as SGM.sample but that path is bypassed here)
        lo, hi = self.config.image_range
        def _clamp(d):
            if d is None:
                return d
            if isinstance(d, list):
                return [torch.clamp(t, lo, hi) for t in d]
            return torch.clamp(d, lo, hi)

        if isinstance(denoised, tuple):
            denoised = tuple(_clamp(d) for d in denoised)
        else:
            denoised = _clamp(denoised)

        return denoised


@register_denoiser(name="bm3d")
class BM3DDenoiser(Denoiser):
    """Block-matching 3D denoiser."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img=None,
        metrics=None,
        stage="all_stages",
        **kwargs,
    ):
        super().__init__(config, dataset, num_img, metrics, keep_track=False, **kwargs)

        if not HAS_BM3D:
            raise ImportError("bm3d package not installed")

        self.stddev = self.corruptor.noise_stddev

        str_to_stage = {
            "hard_thresholding": BM3DStages.HARD_THRESHOLDING,
            "all_stages": BM3DStages.ALL_STAGES,
        }
        self.stage = str_to_stage.get(stage, BM3DStages.ALL_STAGES)

    @timefunc
    def _denoise(self, images):
        if self.verbose:
            print("\nBM3D Denoiser...")

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        # Convert from (B, C, H, W) to (B, H, W, C)
        if images.ndim == 4 and images.shape[1] in [1, 3]:
            images = images.transpose(0, 2, 3, 1)

        denoised = []
        for image in images:
            result = bm3d(np.squeeze(image), self.stddev, stage_arg=self.stage)
            denoised.append(result)

        denoised = np.stack(denoised)

        if self.config.get("color_mode") == "grayscale" and denoised.ndim == 3:
            denoised = denoised[..., np.newaxis]

        # Convert back to (B, C, H, W)
        if denoised.ndim == 4:
            denoised = denoised.transpose(0, 3, 1, 2)

        return torch.from_numpy(denoised)


@register_denoiser(name="nlm")
class NLMDenoiser(Denoiser):
    """Non-local means denoiser."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img=None,
        metrics=None,
        patch_size=6,
        patch_distance=5,
        **kwargs,
    ):
        super().__init__(config, dataset, num_img, metrics, keep_track=False, **kwargs)

        if not HAS_NLM:
            raise ImportError("skimage not installed")

        self.stddev = self.corruptor.noise_stddev

        channel_axis = -1 if self.config.get("color_mode") == "rgb" else None
        self.patch_kw = dict(
            patch_size=patch_size,
            patch_distance=patch_distance,
            channel_axis=channel_axis,
        )

    def _denoise(self, images):
        if self.verbose:
            print("\nNLM Denoiser...")

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        # Convert from (B, C, H, W) to (B, H, W, C)
        if images.ndim == 4 and images.shape[1] in [1, 3]:
            images = images.transpose(0, 2, 3, 1)

        denoised = []
        for image in images:
            result = denoise_nl_means(
                np.squeeze(image),
                h=0.6 * self.stddev,
                sigma=self.stddev,
                fast_mode=True,
                **self.patch_kw,
            )
            denoised.append(result)

        denoised = np.stack(denoised)

        if self.config.get("color_mode") == "grayscale" and denoised.ndim == 3:
            denoised = denoised[..., np.newaxis]

        # Convert back to (B, C, H, W)
        if denoised.ndim == 4:
            denoised = denoised.transpose(0, 3, 1, 2)

        return torch.from_numpy(denoised)


def plot_multiple_denoisers(denoisers, dpi=300, show_metrics=True, save=True, figsize=None):
    """Plot results from multiple denoisers side by side.
    
    Args:
        denoisers: List of Denoiser objects (already run)
        dpi: Figure DPI
        show_metrics: Whether to show metrics on plot
        save: Whether to save figure
        figsize: Optional figure size
    
    Returns:
        matplotlib Figure
    """
    if not denoisers:
        return None

    denoiser = denoisers[0]

    # Collect denoised samples
    samples_list = []
    for d in denoisers:
        ds = d.denoised_samples
        if isinstance(ds, tuple):
            ds = ds[0]
        if d.keep_track and isinstance(ds, list):
            ds = ds[-1]
        samples_list.append(ds)

    # Convert to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] in [1, 3]:
            x = x.transpose(0, 2, 3, 1)
        return np.clip(x, denoiser.vmin, denoiser.vmax)

    target = to_numpy(denoiser.target_samples) if denoiser.target_samples is not None else None
    noisy = to_numpy(denoiser.noisy_samples)
    samples_list = [to_numpy(s) for s in samples_list]

    # Setup figure
    titles = []
    plot_samples = []

    if target is not None:
        titles.append("Ground Truth")
        plot_samples.append(target)

    titles.append("Noisy")
    plot_samples.append(noisy)

    for d, s in zip(denoisers, samples_list):
        titles.append(_MODEL_NAMES.get(d.name, d.name))
        plot_samples.append(s)

    num_img = len(noisy)
    n_cols = len(plot_samples)

    if figsize is None:
        figsize = (n_cols * 2.5, num_img * 2)

    fig, axs = plt.subplots(num_img, n_cols, figsize=figsize)
    if num_img == 1:
        axs = axs.reshape(1, -1)

    for n in range(num_img):
        for i, (sample, title) in enumerate(zip(plot_samples, titles)):
            img = np.squeeze(sample[n])
            axs[n, i].imshow(img, cmap="gray", vmin=denoiser.vmin, vmax=denoiser.vmax)
            axs[n, i].axis("off")
            if n == 0:
                axs[n, i].set_title(title)

    fig.tight_layout()

    if save:
        names = "-".join([d.name for d in denoisers])
        filename = f"{names}_{denoiser.config.dataset_name}_{denoiser.corruptor.task}"
        path = get_date_filename(f"figures/{filename}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved comparison plot to {path}")

    return fig


def animate_multiple_denoisers(denoisers, duration=5, dpi=150, save=True):
    """Create animation from denoising history of multiple denoisers.
    
    Args:
        denoisers: List of Denoiser objects (already run with keep_track=True)
        duration: Duration of animation in seconds
        dpi: Figure DPI
        save: Whether to save the animation
    
    Returns:
        matplotlib animation or None if animation not possible
    """
    from matplotlib.animation import FuncAnimation
    
    if not denoisers:
        return None
    
    # Find denoisers with history
    denoisers_with_history = [d for d in denoisers if d.keep_track and hasattr(d, 'denoised_samples')]
    if not denoisers_with_history:
        print("No denoisers with tracked history, skipping animation")
        return None
    
    denoiser = denoisers_with_history[0]
    
    # Get history from first denoiser with tracking
    history = denoiser.denoised_samples
    if not isinstance(history, list):
        print("No history available for animation")
        return None
    
    num_frames = len(history)
    if num_frames < 2:
        print("Not enough frames for animation")
        return None
    
    # Convert to numpy helper
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] in [1, 3]:
            x = x.transpose(0, 2, 3, 1)
        return np.clip(x, denoiser.vmin, denoiser.vmax)
    
    # Setup figure
    num_img = min(4, len(to_numpy(history[0])))  # Limit to 4 images for animation
    n_cols = len(denoisers_with_history)
    
    fig, axs = plt.subplots(num_img, n_cols, figsize=(n_cols * 3, num_img * 3))
    if num_img == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif num_img == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    # Initialize images
    ims = []
    for d_idx, d in enumerate(denoisers_with_history):
        d_history = d.denoised_samples if isinstance(d.denoised_samples, list) else [d.denoised_samples]
        frame0 = to_numpy(d_history[0])
        row_ims = []
        for n in range(num_img):
            img = np.squeeze(frame0[n])
            im = axs[n, d_idx].imshow(img, cmap="gray", vmin=denoiser.vmin, vmax=denoiser.vmax)
            axs[n, d_idx].axis("off")
            if n == 0:
                axs[n, d_idx].set_title(_MODEL_NAMES.get(d.name, d.name))
            row_ims.append(im)
        ims.append(row_ims)
    
    def update(frame_idx):
        for d_idx, d in enumerate(denoisers_with_history):
            d_history = d.denoised_samples if isinstance(d.denoised_samples, list) else [d.denoised_samples]
            # Handle different history lengths
            f_idx = min(frame_idx, len(d_history) - 1)
            frame = to_numpy(d_history[f_idx])
            for n in range(num_img):
                img = np.squeeze(frame[n])
                ims[d_idx][n].set_array(img)
        return [im for row in ims for im in row]
    
    interval = (duration * 1000) / num_frames  # milliseconds per frame
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    
    if save:
        names = "-".join([d.name for d in denoisers_with_history])
        filename = f"{names}_{denoiser.config.dataset_name}_{denoiser.corruptor.task}_animation"
        path = get_date_filename(f"figures/{filename}.gif")
        anim.save(path, writer="pillow", fps=num_frames / duration, dpi=dpi)
        print(f"Saved animation to {path}")
    
    plt.close(fig)
    return anim
