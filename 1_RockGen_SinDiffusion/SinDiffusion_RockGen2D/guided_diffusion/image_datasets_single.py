"""
Image dataset loading for single process training (no MPI required).
"""

import os
import numpy as np
import torch as th
from PIL import Image
import torchvision.transforms as transforms
from guided_diffusion import dist_util


def load_data(
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    scale_init=0.75,
    scale_factor=0.75,
    stop_scale=8,
    current_scale=8,
):
    """
    Load image data for single process training.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # Load single image
    image_path = data_dir
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Create a simple dataset that returns the same image
    class SingleImageDataset:
        def __init__(self, image_tensor, batch_size):
            self.image_tensor = image_tensor
            self.batch_size = batch_size
            self.length = 1000  # Arbitrary length for training
            
        def __iter__(self):
            return self
            
        def __next__(self):
            # Return the same image with batch_size
            return self.image_tensor.repeat(self.batch_size, 1, 1, 1)
            
        def __len__(self):
            return self.length
    
    return SingleImageDataset(image_tensor, batch_size)
