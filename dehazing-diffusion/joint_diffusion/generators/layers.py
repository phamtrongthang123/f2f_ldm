"""PyTorch layer building blocks for NCSNv2 / RefineNet architecture.
Ported from TensorFlow/Keras to PyTorch.
Reference: ncsnv2/models/layers.py (Yang Song)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def get_activation(activation: str = None):
    """Return an activation module given its name."""
    if activation is None:
        return nn.Identity()
    act = activation.lower()
    if act == "relu":
        return nn.ReLU()
    elif act in ("leakyrelu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.2)
    elif act == "elu":
        return nn.ELU()
    elif act == "swish" or act == "silu":
        return nn.SiLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def get_normalization(normalization: str):
    """Return a normalization class (not instantiated) given its name.

    Returns a callable that takes (num_features) and returns an nn.Module.
    """
    if normalization is None:
        return None
    norm = normalization.lower()
    if norm == "batch":
        return nn.BatchNorm2d
    elif norm == "instance":
        return nn.InstanceNorm2d
    elif norm == "layer":
        return nn.GroupNorm  # LayerNorm for conv = GroupNorm with 1 group
    elif norm.startswith("group"):
        num_groups = int(norm.replace("group", ""))
        return partial(nn.GroupNorm, num_groups)
    else:
        raise ValueError(f"Unknown normalization layer: {normalization}")


def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)


def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False,
            dilation=1):
    """3x3 convolution with padding."""
    padding = dilation  # keeps spatial dims with dilation
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, bias=bias,
    )
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    """1x1 convolution."""
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride,
        padding=0, bias=bias,
    )
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True, spec_norm=False):
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, padding=dilation,
        dilation=dilation, bias=bias,
    )
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


class ConvMeanPool(nn.Module):
    """Conv followed by mean pooling (stride-2 downsampling)."""

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True,
                 adjust_padding=False, spec_norm=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride=1,
                padding=kernel_size // 2, bias=biases,
            )
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = conv
        else:
            conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride=1,
                padding=kernel_size // 2, bias=biases,
            )
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = (
            output[:, :, ::2, ::2]
            + output[:, :, 1::2, ::2]
            + output[:, :, ::2, 1::2]
            + output[:, :, 1::2, 1::2]
        ) / 4.0
        return output


class ConvBlock(nn.Module):
    """Conv2d + optional normalization + optional activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, normalization=None, activation=None,
                 spec_norm=False, bias=False):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias,
        )
        if spec_norm:
            conv = spectral_norm(conv)

        layers = [conv]
        if normalization is not None:
            norm_fn = get_normalization(normalization)
            if normalization.lower() == "layer":
                layers.append(norm_fn(1, out_channels))
            else:
                layers.append(norm_fn(out_channels))
        if activation is not None:
            layers.append(get_activation(activation))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRPBlock(nn.Module):
    """Chained Residual Pooling Block.

    Captures background context from a large image region via chained
    pooling blocks with residual connections.
    """

    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True,
                 spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_stages):
            self.convs.append(
                conv3x3(features, features, stride=1, bias=False,
                        spec_norm=spec_norm)
            )
        self.n_stages = n_stages
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class RCUBlock(nn.Module):
    """Residual Conv Unit Block."""

    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU(),
                 spec_norm=False):
        super().__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    f"{i + 1}_{j + 1}_conv",
                    conv3x3(features, features, stride=1, bias=False,
                            spec_norm=spec_norm),
                )
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, f"{i + 1}_{j + 1}_conv")(x)
            x = x + residual
        return x


class MSFBlock(nn.Module):
    """Multi-Scale Fusion Block.

    Fuses multi-resolution feature maps by adapting channel dims and
    upsampling to a common resolution.
    """

    def __init__(self, in_planes, features, spec_norm=False):
        """
        Args:
            in_planes: list/tuple of input channel counts for each path.
            features: output channel count.
        """
        super().__init__()
        assert isinstance(in_planes, (list, tuple))
        self.convs = nn.ModuleList()
        self.features = features
        for inp in in_planes:
            self.convs.append(
                conv3x3(inp, features, stride=1, bias=True, spec_norm=spec_norm)
            )

    def forward(self, xs, shape):
        """
        Args:
            xs: list of tensors from different resolution paths.
            shape: target (H, W) to upsample to.
        """
        sums = torch.zeros(
            xs[0].shape[0], self.features, *shape, device=xs[0].device
        )
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums = sums + h
        return sums


class RefineBlock(nn.Module):
    """A RefineNet Block.

    Combines ResidualConvUnits, fuses feature maps using MultiResolutionFusion,
    then gets large-scale context with ChainedResidualPooling.
    """

    def __init__(self, in_planes, features, act=nn.ReLU(), start=False,
                 end=False, maxpool=True, spec_norm=False):
        super().__init__()
        assert isinstance(in_planes, (list, tuple))
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm)
            )

        self.output_convs = RCUBlock(
            features, 3 if end else 1, 2, act, spec_norm=spec_norm
        )

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm)

    def forward(self, xs, output_shape):
        """
        Args:
            xs: list of input tensors.
            output_shape: target (H, W) for fusion.
        """
        assert isinstance(xs, (list, tuple))
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)
        return h


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling via ConvMeanPool or dilation."""

    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
                 normalization=nn.InstanceNorm2d, adjust_padding=False,
                 dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample

        if resample == "down":
            if dilation is not None:
                self.conv1 = dilated_conv3x3(
                    input_dim, input_dim, dilation=dilation, spec_norm=spec_norm
                )
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(
                    input_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
                conv_shortcut = partial(
                    dilated_conv3x3, dilation=dilation, spec_norm=spec_norm
                )
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3,
                    adjust_padding=adjust_padding, spec_norm=spec_norm,
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1,
                    adjust_padding=adjust_padding, spec_norm=spec_norm,
                )
        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(
                    dilated_conv3x3, dilation=dilation, spec_norm=spec_norm
                )
                self.conv1 = dilated_conv3x3(
                    input_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(
                    output_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
            else:
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise ValueError(f"invalid resample value: {resample}")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output
