"""Sanity test for PyTorch-ported score-based diffusion code.

Run from joint_diffusion/:
    python test_sanity.py

Tests:
  1. Imports work without TensorFlow
  2. SDE classes (marginal_prob, prior_sampling, discretize)
  3. Layer building blocks (ResidualBlock, RefineBlock, etc.)
  4. NCSNv2 forward pass
  5. ScoreNet loss + backward
  6. Sampling (few steps)
"""
import sys
import traceback

import torch
import numpy as np


class SimpleConfig:
    """Minimal config object for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_imports():
    """Test 1: All imports work without TensorFlow."""
    print("=" * 60)
    print("Test 1: Imports")
    print("=" * 60)

    from generators.SGM.sde_lib import SDE, VPSDE, VESDE, subVPSDE, simple
    print("  [OK] from generators.SGM.sde_lib import VPSDE, VESDE, ...")

    from generators.SGM.SGM import NCSNv2, ScoreNet
    print("  [OK] from generators.SGM.SGM import NCSNv2, ScoreNet")

    from generators.SGM.sampling import (
        ScoreSampler, get_predictor, get_corrector,
        EulerMaruyamaPredictor, ReverseDiffusionPredictor,
        LangevinCorrector, AnnealedLangevinDynamics,
    )
    print("  [OK] from generators.SGM.sampling import ScoreSampler, ...")

    from generators.SGM.guidance import PIGDM, DPS, Projection, get_guidance
    print("  [OK] from generators.SGM.guidance import PIGDM, DPS, Projection")

    from generators.layers import (
        ConvBlock, ResidualBlock, RefineBlock, RCUBlock, MSFBlock, CRPBlock,
        get_activation, get_normalization,
    )
    print("  [OK] from generators.layers import ResidualBlock, RefineBlock, ...")

    from utils.corruptors import GaussianCorruptor, get_corruptor
    print("  [OK] from utils.corruptors import GaussianCorruptor")

    # Verify NO tensorflow
    assert "tensorflow" not in sys.modules, "TensorFlow was imported!"
    print("  [OK] No TensorFlow in sys.modules")
    print()


def test_sde_classes():
    """Test 2: SDE classes work correctly."""
    print("=" * 60)
    print("Test 2: SDE Classes")
    print("=" * 60)

    from generators.SGM.sde_lib import VPSDE, VESDE, subVPSDE, simple

    B, C, H, W = 2, 1, 64, 64
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)

    for SDE_cls, kwargs in [
        (VESDE, dict(sigma_min=0.01, sigma_max=50, N=100)),
        (VPSDE, dict(beta_min=0.1, beta_max=20, N=100)),
        (subVPSDE, dict(beta_min=0.1, beta_max=20, N=100)),
        (simple, dict(sigma=25.0, N=100)),
    ]:
        name = SDE_cls.__name__
        sde = SDE_cls(**kwargs)

        # marginal_prob
        mean, std = sde.marginal_prob(x, t)
        assert mean.shape == x.shape, f"{name}: mean shape mismatch"
        print(f"  [OK] {name}.marginal_prob: mean={mean.shape}, std={std.shape}")

        # prior_sampling
        z = sde.prior_sampling([B, C, H, W])
        assert z.shape == (B, C, H, W), f"{name}: prior_sampling shape mismatch"
        print(f"  [OK] {name}.prior_sampling: {z.shape}")

        # forward_diffuse
        x_noisy = sde.forward_diffuse(x, t)
        assert x_noisy.shape == x.shape, f"{name}: forward_diffuse shape mismatch"
        print(f"  [OK] {name}.forward_diffuse: {x_noisy.shape}")

        # discretize
        f, G = sde.discretize(x, t)
        print(f"  [OK] {name}.discretize: f={f.shape}, G={G.shape}")

        # reverse
        score_fn = lambda x, t: torch.randn_like(x)
        rsde = sde.reverse(score_fn, probability_flow=False)
        drift, diff = rsde.sde(x, t)
        assert drift.shape == x.shape, f"{name}: reverse drift shape mismatch"
        print(f"  [OK] {name}.reverse: drift={drift.shape}")
        print()


def test_layers():
    """Test 3: Layer building blocks."""
    print("=" * 60)
    print("Test 3: Layer Building Blocks")
    print("=" * 60)

    from generators.layers import (
        ConvBlock, ResidualBlock, RefineBlock, RCUBlock, MSFBlock, CRPBlock,
        get_activation, get_normalization,
    )
    import torch.nn as nn

    B, C, H, W = 2, 32, 64, 64

    # ConvBlock
    cb = ConvBlock(C, C, kernel_size=3, stride=1, activation="elu",
                   normalization="instance", bias=True)
    x = torch.randn(B, C, H, W)
    y = cb(x)
    assert y.shape == x.shape, f"ConvBlock shape mismatch: {y.shape}"
    print(f"  [OK] ConvBlock: {x.shape} -> {y.shape}")

    # ResidualBlock (no resample)
    norm_fn = get_normalization("instance")
    act = get_activation("elu")
    rb = ResidualBlock(C, C, resample=None, act=act, normalization=norm_fn)
    y = rb(x)
    assert y.shape == x.shape, f"ResidualBlock shape mismatch: {y.shape}"
    print(f"  [OK] ResidualBlock (no resample): {x.shape} -> {y.shape}")

    # ResidualBlock (downsample)
    rb_down = ResidualBlock(C, 2 * C, resample="down", act=act, normalization=norm_fn)
    y = rb_down(x)
    assert y.shape[1] == 2 * C, f"ResidualBlock down channels mismatch"
    print(f"  [OK] ResidualBlock (down): {x.shape} -> {y.shape}")

    # ResidualBlock (dilation)
    rb_dil = ResidualBlock(C, C, resample="down", act=act, normalization=norm_fn,
                           dilation=2)
    y = rb_dil(x)
    print(f"  [OK] ResidualBlock (dilation=2): {x.shape} -> {y.shape}")

    # RCUBlock
    rcu = RCUBlock(C, 2, 2, act)
    y = rcu(x)
    assert y.shape == x.shape, f"RCUBlock shape mismatch"
    print(f"  [OK] RCUBlock: {x.shape} -> {y.shape}")

    # CRPBlock
    crp = CRPBlock(C, 2, act)
    y = crp(x)
    assert y.shape == x.shape, f"CRPBlock shape mismatch"
    print(f"  [OK] CRPBlock: {x.shape} -> {y.shape}")

    # MSFBlock
    msf = MSFBlock([C, C], C)
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H // 2, W // 2)
    y = msf([x1, x2], (H, W))
    assert y.shape == (B, C, H, W), f"MSFBlock shape mismatch: {y.shape}"
    print(f"  [OK] MSFBlock: [{x1.shape}, {x2.shape}] -> {y.shape}")

    # RefineBlock
    ref = RefineBlock([C, C], C, act=act, start=False, end=False)
    y = ref([x1, x2], (H, W))
    assert y.shape == (B, C, H, W), f"RefineBlock shape mismatch: {y.shape}"
    print(f"  [OK] RefineBlock: [{x1.shape}, {x2.shape}] -> {y.shape}")

    # RefineBlock (start=True, single input)
    ref_start = RefineBlock([C], C, act=act, start=True)
    y = ref_start([x1], (H, W))
    assert y.shape == (B, C, H, W)
    print(f"  [OK] RefineBlock (start): [{x1.shape}] -> {y.shape}")
    print()


def test_ncsnv2_forward():
    """Test 4: NCSNv2 forward pass."""
    print("=" * 60)
    print("Test 4: NCSNv2 Forward Pass")
    print("=" * 60)

    from generators.SGM.SGM import NCSNv2

    config = SimpleConfig(
        image_shape=(1, 64, 64),
        channels=32,
        activation="elu",
        normalization="instance",
        kernel_size=3,
    )

    model = NCSNv2(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  NCSNv2 parameters: {n_params:,}")

    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        score = model(x)
    assert score.shape == x.shape, f"NCSNv2 output shape mismatch: {score.shape}"
    print(f"  [OK] NCSNv2 forward: {x.shape} -> {score.shape}")
    print()


def test_scorenet_loss():
    """Test 5: ScoreNet loss computation and backward."""
    print("=" * 60)
    print("Test 5: ScoreNet Loss + Backward")
    print("=" * 60)

    from generators.SGM.SGM import ScoreNet

    config = SimpleConfig(
        image_shape=(1, 64, 64),
        channels=32,
        activation="elu",
        normalization="instance",
        kernel_size=3,
        sde="vesde",
        sigma_min=0.01,
        sigma_max=50,
        num_scales=100,
        score_backbone="NCSNv2",
        reduce_mean=True,
        likelihood_weighting=False,
    )

    score_net = ScoreNet(config)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-4)

    batch = torch.randn(2, 1, 64, 64)
    loss = score_net.score_loss(batch)
    print(f"  Loss value: {loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check gradients exist
    grad_norms = []
    for p in score_net.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    assert len(grad_norms) > 0, "No gradients computed!"
    print(f"  [OK] Backward pass: {len(grad_norms)} params with gradients")
    print(f"  Mean grad norm: {np.mean(grad_norms):.6f}")

    # Second step to verify training loop works
    batch2 = torch.randn(2, 1, 64, 64)
    loss2 = score_net.score_loss(batch2)
    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()
    print(f"  [OK] Second training step: loss={loss2.item():.6f}")
    print()


def test_sampling():
    """Test 6: Sampling runs for a few steps."""
    print("=" * 60)
    print("Test 6: Sampling (Few Steps)")
    print("=" * 60)

    from generators.SGM.SGM import ScoreNet
    from generators.SGM.sampling import ScoreSampler

    config = SimpleConfig(
        image_shape=(1, 32, 32),
        channels=16,
        activation="elu",
        normalization="instance",
        kernel_size=3,
        sde="vesde",
        sigma_min=0.01,
        sigma_max=50,
        num_scales=10,  # Very few steps for speed
        score_backbone="NCSNv2",
        reduce_mean=True,
        likelihood_weighting=False,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
        snr=0.16,
        num_img=2,
        image_range=None,
    )

    score_net = ScoreNet(config)
    score_net.eval()

    sampler = ScoreSampler(
        model=score_net,
        sde=score_net.sde,
        image_shape=config.image_shape,
        sampling_method="pc",
        predictor="euler_maruyama",
        corrector="none",
        corrector_snr=0.16,
    )

    samples = sampler(z=None, shape=(2, *config.image_shape), progress_bar=False)
    assert samples.shape == (2, 1, 32, 32), f"Sample shape mismatch: {samples.shape}"
    assert not torch.isnan(samples).any(), "NaN in samples!"
    print(f"  [OK] Unconditional sampling: {samples.shape}")
    print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    print()


def main():
    print("\n" + "=" * 60)
    print("SANITY TEST: PyTorch Score-Based Diffusion")
    print("=" * 60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("SDE Classes", test_sde_classes),
        ("Layer Building Blocks", test_layers),
        ("NCSNv2 Forward Pass", test_ncsnv2_forward),
        ("ScoreNet Loss + Backward", test_scorenet_loss),
        ("Sampling", test_sampling),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            traceback.print_exc()
            print()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {name}")
        if status == "FAIL":
            print(f"         {result}")

    print()
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
