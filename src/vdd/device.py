"""Explicit device selection and synchronized accelerator timing."""

import torch


def resolve_device(requested="auto"):
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda or mps.")
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise ValueError("CUDA unavailable. Enable a GPU runtime or use --device cpu.")
    if requested == "mps" and not mps:
        raise ValueError("Apple MPS unavailable. Use --device cpu.")
    return torch.device(requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
