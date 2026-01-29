#!/usr/bin/env python3
"""Visualize cyst phantom."""

import numpy as np
import matplotlib.pyplot as plt

xlims = (-20e-3, 20e-3)
zlims = (10e-3, 35e-3)

rng = np.random.default_rng(42)

n_scat = 800
x = rng.uniform(*xlims, n_scat)
z = rng.uniform(*zlims, n_scat)

# Cyst center and radius
cx = rng.uniform(xlims[0] * 0.5, xlims[1] * 0.5)
cz = rng.uniform(zlims[0] * 1.2, zlims[1] * 0.8)
cr = rng.uniform(3e-3, 8e-3)

# Remove scatterers inside cyst
dist = np.sqrt((x - cx)**2 + (z - cz)**2)
mask = dist > cr
x_out, z_out = x[mask], z[mask]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before removing cyst
axes[0].scatter(x * 1e3, z * 1e3, s=1, c='blue', alpha=0.5)
circle = plt.Circle((cx * 1e3, cz * 1e3), cr * 1e3, fill=False, color='red', linewidth=2)
axes[0].add_patch(circle)
axes[0].set_xlabel('X (mm)')
axes[0].set_ylabel('Z (mm)')
axes[0].set_title('Before: All scatterers + cyst boundary')
axes[0].set_aspect('equal')
axes[0].invert_yaxis()

# After removing cyst (anechoic region)
axes[1].scatter(x_out * 1e3, z_out * 1e3, s=1, c='blue', alpha=0.5)
circle = plt.Circle((cx * 1e3, cz * 1e3), cr * 1e3, fill=False, color='red', linewidth=2, linestyle='--')
axes[1].add_patch(circle)
axes[1].set_xlabel('X (mm)')
axes[1].set_ylabel('Z (mm)')
axes[1].set_title('After: Cyst = empty region (anechoic)')
axes[1].set_aspect('equal')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('/home/tp030/f2f_ldm/dehazing-diffusion/reproduce_helpers/cyst_visualization.png', dpi=150)
plt.show()
print("Saved to cyst_visualization.png")
