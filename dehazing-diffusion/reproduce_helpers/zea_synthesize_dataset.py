#!/usr/bin/env python3
"""Synthesize ultrasound RF datasets (tissue + haze) with ZEA.

Outputs raw beamformed RF data (float32) for training diffusion models,
following the methodology in "Dehazing Ultrasound using Diffusion Models".
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("ZEA_DISABLE_CACHE", "1")
