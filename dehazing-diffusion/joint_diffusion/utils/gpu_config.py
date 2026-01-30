"""GPU utilities (PyTorch version)
Author(s): Tristan Stevens, Ben Luijten
Ported to PyTorch: Jan 2026
"""
import os
import subprocess as sp
import warnings

import numpy as np
import torch


def get_gpu_memory(verbose=True):
    """Retrieve memory allocation information of all gpus.
    Arguments
        verbose: prints output if True.
    Returns
        memory_free_values: list of available memory for each gpu in MiB.
    """
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    try:
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    except (FileNotFoundError, sp.CalledProcessError):
        if verbose:
            print("nvidia-smi not found, cannot query GPU memory")
        return []

    # only show enabled devices
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        gpus = [int(gpu.strip()) for gpu in gpus.split(",") if gpu.strip()]
        if verbose and len(gpus) < len(memory_free_values):
            print(
                f"{len(memory_free_values) - len(gpus)}/{len(memory_free_values)} "
                "GPUs were disabled via CUDA_VISIBLE_DEVICES"
            )
        memory_free_values = [
            memory_free_values[gpu] for gpu in gpus if gpu < len(memory_free_values)
        ]

    if verbose and memory_free_values:
        print("GPU Memory Available (MiB):")
        for i, mem in enumerate(memory_free_values):
            print(f"  GPU {i}: {mem} MiB")
    return memory_free_values


def set_gpu_usage(device=None):
    """Select GPU device for PyTorch.
    
    Args:
        device (str/int/list): GPU number to select. 
            - If None, choose GPU with most available memory.
            - If 'cpu', use CPU only.
            - If int, use that specific GPU.
            - If list of ints, sets CUDA_VISIBLE_DEVICES (PyTorch will use first).
    
    Returns:
        Selected device string (e.g., 'cuda:0' or 'cpu')
    """
    if device == "cpu":
        print("Setting device to CPU based on config.")
        return "cpu"

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs available, using CPU")
        return "cpu"

    print(f"{n_gpus} GPU(s) available via PyTorch")

    # If device is None, auto-select based on memory
    if device is None:
        mem = get_gpu_memory(verbose=False)
        if mem:
            device = int(np.argmax(mem))
            print(f"Auto-selected GPU {device} with {mem[device]} MiB free")
        else:
            device = 0
            print(f"Using default GPU {device}")

    # Handle list of devices
    if isinstance(device, list):
        if len(device) > 0:
            device = device[0]
            print(f"Using first GPU from list: {device}")
        else:
            device = 0

    # Validate device index
    if isinstance(device, int):
        if device >= n_gpus:
            warnings.warn(
                f"Requested GPU {device} but only {n_gpus} available. Using GPU 0."
            )
            device = 0
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(device)
        print(f"Selected GPU {device}: {device_name}")
        return f"cuda:{device}"

    # If device is already a string like 'cuda:0'
    if isinstance(device, str) and device.startswith("cuda"):
        return device

    return f"cuda:{device}"


def get_device(config=None):
    """Get the appropriate device for PyTorch operations.
    
    Args:
        config: Optional config object with 'device' attribute.
    
    Returns:
        torch.device object
    """
    if config is not None and hasattr(config, "device"):
        device_str = config.device
    else:
        device_str = None

    if device_str == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        if device_str is None:
            return torch.device("cuda")
        elif isinstance(device_str, int):
            return torch.device(f"cuda:{device_str}")
        elif isinstance(device_str, str) and device_str.startswith("cuda"):
            return torch.device(device_str)
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    ## Example on how to use gpu config functions
    print("=" * 50)
    print("GPU Configuration Test")
    print("=" * 50)
    device = set_gpu_usage()
    print(f"\nFinal device: {device}")
    
    # Test tensor creation on selected device
    if device != "cpu":
        x = torch.randn(10, 10, device=device)
        print(f"Test tensor created on: {x.device}")
