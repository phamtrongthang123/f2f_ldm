#!/bin/bash
set -euo pipefail

ROOT_DIR="/scrfs/storage/tp030/home/f2f_ldm"
JD_DIR="$ROOT_DIR/dehazing-diffusion/joint_diffusion"

source "$ROOT_DIR/.venv_joint/bin/activate"
cd "$JD_DIR"

echo "=== Layer 1: Import smoke test ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

from datasets import get_dataset
from generators.models import get_model
from generators.layers import ConvBlock, ResidualBlock, DownSample, UpSample
from generators.SGM.SGM import NCSNv2
from generators.SGM.sde_lib import get_sde
from generators.SGM.sampling import get_predictor, get_corrector
from utils.corruptors import get_corruptor
from utils.inverse import get_denoiser

print('All imports OK')
"

echo "=== Layer 2: Component shape tests ==="
python -c "
import torch

from generators.layers import ConvBlock, ResidualBlock
x = torch.randn(2, 1, 128, 64)
conv = ConvBlock(in_channels=1, out_channels=32, kernel_size=3)
out = conv(x)
assert out.shape == (2, 32, 128, 64), f'ConvBlock: expected (2,32,128,64), got {out.shape}'
print(f'ConvBlock OK: {x.shape} -> {out.shape}')

from generators.SGM.SGM import NCSNv2
from utils.utils import AttrDict
config = AttrDict({
    'channels': 32,
    'image_size': [128, 64],
    'num_scales': 10,
    'sigma': 25.0,
    'normalization': 'batch',
    'kernel_size': 3,
    'activation': 'relu',
    'drop_prob': None,
    'upmode': 'upconv',
    'embed_dim': 256,
})
model = NCSNv2(config)
x = torch.randn(2, 1, 128, 64)
t = torch.randint(0, 10, (2,))
score = model(x, t)
assert score.shape == x.shape, f'NCSNv2: expected {x.shape}, got {score.shape}'
print(f'NCSNv2 OK: input {x.shape} -> score {score.shape}')

from generators.SGM.sde_lib import get_sde
sde = get_sde(config)
t = torch.rand(2)
mean, std = sde.marginal_prob(x, t)
assert mean.shape == x.shape, f'SDE marginal_prob mean: expected {x.shape}, got {mean.shape}'
print(f'SDE OK: marginal_prob shapes correct')

print()
print('All component shape tests PASSED')
"

echo "=== Layer 3: Dataset loading round-trip ==="
# Requires synthesized data at $ROOT_DIR/data/zea_synth_test
if [ -f "$ROOT_DIR/data/zea_synth_test/tissue/train.npz" ]; then
    python -c "
import torch
from datasets import get_dataset
from utils.utils import AttrDict

for name in ['zea_tissue', 'zea_haze']:
    config = AttrDict({
        'dataset_name': name,
        'data_root': '$ROOT_DIR/data/zea_synth_test/..',
        'batch_size': 4,
        'image_range': [0, 1],
        'shuffle': True,
        'seed': 42,
        'npz_key': 'rf',
        'image_size': [1024, 64],
    })
    train, val = get_dataset(config)
    batch = next(iter(train))
    assert isinstance(batch, torch.Tensor), f'Expected torch.Tensor, got {type(batch)}'
    print(f'{name}: batch={batch.shape}, range=[{batch.min():.3f}, {batch.max():.3f}] OK')

print()
print('Dataset loading tests PASSED')
"
else
    echo "SKIP: No synthesized data found at $ROOT_DIR/data/zea_synth_test"
    echo "Run zea_synth_run.sh first to generate test data."
fi

echo "=== Sanity tests complete ==="
