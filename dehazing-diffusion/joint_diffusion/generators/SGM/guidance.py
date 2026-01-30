"""Guidance class for joint posterior sampling.
Ported from TensorFlow to PyTorch.
Author(s): Tristan Stevens
"""
import abc

import torch

_GUIDANCE = {}


def register_guidance(cls=None, *, name=None):
    """A decorator for registering guidance classes."""

    def _register(cls):
        local_name = name if name is not None else cls.__name__
        if local_name in _GUIDANCE:
            raise ValueError(f"Already registered guidance with name: {local_name}")
        _GUIDANCE[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_guidance(name):
    """Get guidance class for given name."""
    return _GUIDANCE[name]


class Guidance(abc.ABC):
    """The abstract class for guidance / conditional sampling."""

    def __init__(self, sde, corruptor, lambda_coeff=None, kappa_coeff=None):
        self.sde = sde
        self.corruptor = corruptor
        self.lambda_coeff = lambda_coeff
        self.kappa_coeff = kappa_coeff
        self.A = self.corruptor.A if self.corruptor is not None else None
        if self.A is not None:
            if isinstance(self.A, torch.Tensor):
                self.A_T = self.A.t()
            else:
                import numpy as np
                self.A = torch.from_numpy(np.array(self.A)).float()
                self.A_T = self.A.t()

    def update_fn(self, y, x, t, x_mean, grad_x0_xt=None):
        """One update for guidance."""
        if self.corruptor.name in ["gaussian", "mnist", "cs", "cs_sine", "haze"]:
            x = self.denoise_update(y, x, t, x_mean, grad_x0_xt)
        else:
            raise ValueError(f"Unknown corruptor: {self.corruptor.name}")
        return x

    def joint_update_fn(self, y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt):
        """One update for joint guidance."""
        if self.corruptor.name in ["gaussian", "mnist", "cs", "cs_sine", "haze"]:
            x, n = self.joint_denoise_update(
                y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt
            )
        else:
            raise ValueError(f"Unknown corruptor: {self.corruptor.name}")
        return x, n


@register_guidance(name="pigdm")
class PIGDM(Guidance):
    """Pseudo inverse guidance.
    https://openreview.net/forum?id=9_gsMA8MRKQ
    """

    def rt_squared(self, x, t):
        """rt^2 = sigma / (sigma + 1)"""
        sigma_t_squared = self.sde.marginal_prob(x, t)[1] ** 2
        r_t_squared = sigma_t_squared / (sigma_t_squared + 1)
        return r_t_squared

    def denoise_update(self, y, x, t, x_mean, grad_x0_xt):
        """Compute denoising data consistency step for Gaussian noise."""
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"

        r_t_squared = self.rt_squared(x, t)
        # Expand to spatial dims
        while r_t_squared.dim() < x.dim():
            r_t_squared = r_t_squared.unsqueeze(-1)

        # compressed sensing y = Ax + n
        if self.A is not None:
            A = self.A.to(x.device)
            A_T = self.A_T.to(x.device)
            m, d = A.shape
            I = torch.eye(m, device=x.device)

            # flatten x_mean and grad
            x_mean_flat = x_mean.reshape(self.batch_size, -1)
            grad_flat = grad_x0_xt.reshape(self.batch_size, -1)

            # posterior mean: A x_0
            mu_t = x_mean_flat @ A_T
            # posterior variance
            sigma_t = (
                r_t_squared.reshape(-1, 1, 1) * (A @ A_T).unsqueeze(0)
                + self.corruptor.noise_stddev**2 * I.unsqueeze(0)
            )
            sigma_t_inv = torch.linalg.inv(sigma_t)
            sigma_inv_A = sigma_t_inv @ A.unsqueeze(0)

            y_flat = y.reshape(self.batch_size, -1)
            grad_p_y_xt = ((y_flat - mu_t).unsqueeze(1) @ sigma_inv_A).squeeze(1) * grad_flat
            grad_p_y_xt = grad_p_y_xt.reshape(self.batch_size, *self.image_shape)

        # denoising y = x + n
        else:
            sigma_t = self.corruptor.noise_stddev**2 + r_t_squared
            grad_p_y_xt = grad_x0_xt * (y - x_mean) / sigma_t

        x = x + self.lambda_coeff * r_t_squared * grad_p_y_xt
        return x

    def joint_denoise_update(self, y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt):
        """Compute denoising data consistency step for structured noise."""
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"
        assert grad_n0_nt is not None, "Gradients for p(n_0|t | n_t) were not provided"

        r_t_squared = self.rt_squared(x, t)
        q_t_squared = r_t_squared

        while r_t_squared.dim() < x.dim():
            r_t_squared = r_t_squared.unsqueeze(-1)
        while q_t_squared.dim() < n.dim():
            q_t_squared = q_t_squared.unsqueeze(-1)

        if self.A is not None:
            A = self.A.to(x.device)
            A_T = self.A_T.to(x.device)
            m, d = A.shape
            I = torch.eye(m, device=x.device)

            x_mean_flat = x_mean.reshape(self.batch_size, -1)
            n_mean_flat = n_mean.reshape(self.batch_size, -1)
            grad_x_flat = grad_x0_xt.reshape(self.batch_size, -1)
            grad_n_flat = grad_n0_nt.reshape(self.batch_size, -1)

            mu_t = x_mean_flat @ A_T + n_mean_flat
            sigma_t = (
                r_t_squared.reshape(-1, 1, 1) * (A @ A_T).unsqueeze(0)
                + q_t_squared.reshape(-1, 1, 1) * I.unsqueeze(0)
            )
            sigma_t_inv = torch.linalg.inv(sigma_t).transpose(-2, -1)

            sigma_inv_A = sigma_t_inv @ A.unsqueeze(0)

            y_flat = y.reshape(self.batch_size, -1)
            diff = (y_flat - mu_t)

            grad_p_y_xt = grad_x_flat * (diff.unsqueeze(1) @ sigma_inv_A).squeeze(1)
            grad_p_y_xt = grad_p_y_xt.reshape(self.batch_size, *self.image_shape)

            grad_p_y_nt = grad_n_flat * (diff.unsqueeze(1) @ sigma_t_inv).squeeze(1)
            grad_p_y_nt = grad_p_y_nt.reshape(self.batch_size, *self.noise_shape)

        else:
            # y = beta * x + alpha * n
            alpha = self.corruptor.blend_factor
            beta = 1 - alpha

            sigma_t = r_t_squared + q_t_squared

            grad_p_y_xt = (
                -1 * grad_x0_xt
                * (beta**2 * x_mean - beta * y + alpha * beta * n_mean)
                / sigma_t
            )
            grad_p_y_nt = (
                -1 * grad_n0_nt
                * (alpha**2 * n_mean - alpha * y + alpha * beta * x_mean)
                / sigma_t
            )

        x = x + self.lambda_coeff * grad_p_y_xt * r_t_squared
        n = n + self.kappa_coeff * grad_p_y_nt * q_t_squared

        return x, n


@register_guidance(name="dps")
class DPS(Guidance):
    """Diffusion Posterior Sampling.
    https://arxiv.org/pdf/2209.14687.pdf
    """

    def update_fn(self, y, x, t, x_mean, grad_x0_xt, *args):
        """Compute denoising data consistency step."""
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"

        x_mean_var = x_mean.detach().requires_grad_(True)
        if self.A is not None:
            A_T = self.A_T.to(x.device)
            Ax = x_mean_var.reshape(self.batch_size, -1) @ A_T
            y_flat = y.reshape(self.batch_size, -1)
            norm = torch.linalg.norm(y_flat - Ax)
        else:
            norm = torch.linalg.norm(y - x_mean_var)

        grad_norm = torch.autograd.grad(norm, x_mean_var)[0]
        grad_p_y_xt = -1 * grad_norm * grad_x0_xt

        x = x + self.lambda_coeff * grad_p_y_xt
        return x

    def joint_denoise_update(self, y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt):
        """Compute denoising data consistency step for structured noise."""
        assert grad_x0_xt is not None
        assert grad_n0_nt is not None

        x_mean_var = x_mean.detach().requires_grad_(True)
        n_mean_var = n_mean.detach().requires_grad_(True)

        if self.A is not None:
            A_T = self.A_T.to(x.device)
            x_flat = x_mean_var.reshape(self.batch_size, -1)
            Ax = x_flat @ A_T
            y_flat = y.reshape(self.batch_size, -1)
            norm = torch.linalg.norm(y_flat - Ax - n_mean_var.reshape(self.batch_size, -1))
        else:
            alpha = self.corruptor.blend_factor
            beta = 1 - alpha
            norm = torch.linalg.norm(y - beta * x_mean_var - alpha * n_mean_var)

        grad_x = torch.autograd.grad(norm, x_mean_var, retain_graph=True)[0]
        grad_n = torch.autograd.grad(norm, n_mean_var)[0]

        grad_p_y_xt = -1 * grad_x * grad_x0_xt
        grad_p_y_nt = -1 * grad_n * grad_n0_nt

        x = x + self.lambda_coeff * grad_p_y_xt
        n = n + self.kappa_coeff * grad_p_y_nt

        return x, n


@register_guidance(name="companded_projection")
class CompandedProjection(Guidance):
    """Companding-aware projection sampling (Paper Eq. 11).

    Uses mu-law companding nonlinearity for data consistency:
        loss = ||y_hat_t - C(C^{-1}(x_t) + gamma * C^{-1}(h_t))||^2
    Gradients are computed via torch.autograd.grad.
    """

    def __init__(self, sde, corruptor, lambda_coeff=None, kappa_coeff=None, mu=255):
        super().__init__(sde, corruptor, lambda_coeff, kappa_coeff)
        self.mu = mu

    @staticmethod
    def mu_law_compress(x, mu=255):
        """Mu-law companding compression: C(x) = sign(x) * log1p(mu * |x|) / log1p(mu)"""
        return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, dtype=x.dtype, device=x.device))

    @staticmethod
    def mu_law_expand(x, mu=255):
        """Mu-law companding expansion: C^{-1}(x) = sign(x) * ((1 + mu)^|x| - 1) / mu"""
        return torch.sign(x) * ((1 + mu) ** torch.abs(x) - 1) / mu

    def denoise_update(self, y, x, t, *args):
        """Single-model data consistency: ||y_hat - x||^2."""
        t_batch = t if t.dim() > 0 else t.expand(self.batch_size)
        mu = self.mu

        with torch.enable_grad():
            y_hat = self.sde.forward_diffuse(y, t_batch)
            x_var = x.detach().requires_grad_(True)

            # Clamp to C^{-1} domain [-1, 1] (paper Eq 220)
            x_clamped = x_var.clamp(-1, 1)
            x_rf = self.mu_law_expand(x_clamped, mu)
            x_pred = self.mu_law_compress(x_rf, mu)

            loss = torch.sum((y_hat - x_pred) ** 2)
            grad_x = torch.autograd.grad(loss, x_var)[0]

        x = x - self.lambda_coeff * grad_x
        return x

    def joint_denoise_update(self, y, x, n, t, *args):
        """Companding-aware joint data consistency (Eq. 11).

        loss = ||y_hat_t - C(C^{-1}(x_t) + gamma * C^{-1}(h_t))||^2
        """
        t_batch = t if t.dim() > 0 else t.expand(self.batch_size)

        gamma = self.corruptor.noise_stddev
        mu = self.mu

        with torch.enable_grad():
            y_hat = self.sde.forward_diffuse(y, t_batch)

            x_var = x.detach().requires_grad_(True)
            n_var = n.detach().requires_grad_(True)

            # Clamp to C^{-1} domain [-1, 1] (paper Eq 220)
            x_clamped = x_var.clamp(-1, 1)
            n_clamped = n_var.clamp(-1, 1)

            # Expand from companded to RF domain, combine, compress back
            x_rf = self.mu_law_expand(x_clamped, mu)
            h_rf = self.mu_law_expand(n_clamped, mu)
            y_pred = self.mu_law_compress(x_rf + gamma * h_rf, mu)

            loss = torch.sum((y_hat - y_pred) ** 2)

            grad_x, grad_n = torch.autograd.grad(loss, [x_var, n_var])

        x = x - self.lambda_coeff * grad_x
        n = n - self.kappa_coeff * grad_n

        return x, n


@register_guidance(name="projection")
class Projection(Guidance):
    """Projection sampling.
    https://arxiv.org/pdf/2111.08005.pdf
    """

    def denoise_update(self, y, x, t, *args):
        """Compute data consistency for compressed sensing task."""
        t_batch = t if t.dim() > 0 else t.expand(self.batch_size)
        y_hat = self.sde.forward_diffuse(y, t_batch)

        if self.A is not None:
            A = self.A.to(x.device)
            A_T = self.A_T.to(x.device)

            x_flat = x.reshape(self.batch_size, -1)
            y_flat = y.reshape(self.batch_size, -1)

            # A^T(Ax - y)
            residual = x_flat @ A_T - y_flat
            grad_y_hat_xt = -(residual @ A).reshape(self.batch_size, *self.image_shape)
        else:
            grad_y_hat_xt = y_hat - x

        x = x + self.lambda_coeff * grad_y_hat_xt
        return x

    def joint_denoise_update(self, y, x, n, t, *args):
        """Compute data consistency for denoising using score models."""
        t_batch = t if t.dim() > 0 else t.expand(self.batch_size)
        y_hat = self.sde.forward_diffuse(y, t_batch)

        if self.A is not None:
            A = self.A.to(x.device)
            A_T = self.A_T.to(x.device)

            x_flat = x.reshape(self.batch_size, -1)
            n_flat = n.reshape(self.batch_size, -1)
            y_flat = y_hat.reshape(self.batch_size, -1)

            # (Ax - y_hat + n)
            residual = x_flat @ A_T - y_flat + n_flat
            grad_y_hat_xt = -(residual @ A).reshape(self.batch_size, *self.image_shape)
            grad_y_hat_nt = -residual.reshape(self.batch_size, *self.noise_shape)
        else:
            alpha = self.corruptor.blend_factor
            beta = 1 - alpha

            grad_y_hat_xt = -(beta**2 * x - beta * y_hat + alpha * beta * n)
            grad_y_hat_nt = -(alpha**2 * n - alpha * y_hat + alpha * beta * x)

        x = x + self.lambda_coeff * grad_y_hat_xt
        n = n + self.kappa_coeff * grad_y_hat_nt

        return x, n
