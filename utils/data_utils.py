"""
Data loading and preprocessing utilities for colorization
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ColorizationDataset(Dataset):
    """
    Dataset for colorization task.
    Converts RGB images to LAB color space and uses L channel as input,
    AB channels as target.
    """
    
    def __init__(self, image_paths, transform=None, size=(256, 256)):
        """
        Args:
            image_paths: List of paths to color images
            transform: Optional transform to be applied
            size: Target size for images (height, width)
        """
        self.image_paths = image_paths
        self.transform = transform
        self.size = size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.size)
        
        # Convert RGB to LAB color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Normalize LAB values
        img_lab = img_lab.astype(np.float32)
        img_lab[:, :, 0] = img_lab[:, :, 0] / 100.0  # L channel: [0, 100] -> [0, 1]
        img_lab[:, :, 1] = (img_lab[:, :, 1] + 128) / 255.0  # A channel: [-128, 127] -> [0, 1]
        img_lab[:, :, 2] = (img_lab[:, :, 2] + 128) / 255.0  # B channel: [-128, 127] -> [0, 1]
        
        # Extract L channel (grayscale input)
        l_channel = img_lab[:, :, 0]
        
        # Extract AB channels (color target)
        ab_channels = img_lab[:, :, 1:3]
        
        # Convert to tensors
        l_channel = torch.from_numpy(l_channel).unsqueeze(0)  # Add channel dimension
        ab_channels = torch.from_numpy(ab_channels).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return l_channel, ab_channels


def get_image_paths(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Get all image paths from a directory
    
    Args:
        directory: Path to directory containing images
        extensions: Tuple of valid image extensions
    
    Returns:
        List of image paths
    """
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def lab_to_rgb(l_channel, ab_channels):
    """
    Convert LAB channels back to RGB image
    
    Args:
        l_channel: L channel tensor (1, H, W) with values in [0, 1]
        ab_channels: AB channels tensor (2, H, W) with values in [0, 1]
    
    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255]
    """
    # Denormalize
    l = l_channel.squeeze().cpu().numpy() * 100.0
    ab = ab_channels.permute(1, 2, 0).cpu().numpy()
    ab[:, :, 0] = (ab[:, :, 0] * 255.0) - 128
    ab[:, :, 1] = (ab[:, :, 1] * 255.0) - 128
    
    # Combine LAB channels
    lab = np.zeros((l.shape[0], l.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = l
    lab[:, :, 1:] = ab
    
    # Convert LAB to RGB
    lab = lab.astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return rgb


def create_dataloader(image_dir, batch_size=16, shuffle=True, size=(256, 256), num_workers=4):
    """
    Create a DataLoader for training
    
    Args:
        image_dir: Directory containing training images
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        size: Target image size
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    image_paths = get_image_paths(image_dir)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    dataset = ColorizationDataset(image_paths, size=size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

