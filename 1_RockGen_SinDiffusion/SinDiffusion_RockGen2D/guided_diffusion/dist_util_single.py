"""
Helpers for single-process training (no MPI required).
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


def setup_dist():
    """
    Setup for single-process training (no distributed setup needed).
    """
    # Do nothing for single process
    pass


def dev():
    """
    Get the device to use for training.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file for single process.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    No-op for single process (no synchronization needed).
    """
    pass


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsname()[1]
    finally:
        s.close()
