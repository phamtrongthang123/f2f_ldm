"""
00_download_data.py — Download CAMUS dataset before running on HPC.

Downloads the CAMUS sample dataset from HuggingFace via ZEA's Dataset API.
Run this on a node with internet before submitting the SLURM job.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from zea.data import Dataset

print("Downloading CAMUS sample dataset from HuggingFace...")
dataset = Dataset("hf://zeahub/camus-sample/val", key="image")

# Iterate to trigger full download and cache
count = 0
for i in range(len(dataset)):
    _ = dataset[i]
    count += 1

dataset.close()
print(f"Downloaded {count} samples.")
print("Dataset cached locally. Ready for offline use.")
