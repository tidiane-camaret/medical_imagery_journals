import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def display_samples(ds_dir, sample_dirs, n_rows):
    # Define modalities
    modalities = ['slice_orig.nii.gz', 'slice_norm.nii.gz', 
                  'slice_seg4.nii.gz', 'slice_seg24.nii.gz']
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
    
    # If only one row, make axes indexable like 2D array
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Display images for each sample
    for row, sample_dir in enumerate(sample_dirs[:n_rows]):
        for col, modality in enumerate(modalities):
            # Load and process image
            img_path = f"{ds_dir}/{sample_dir}/{modality}"
            if os.path.exists(img_path):
                img = np.squeeze(nib.load(img_path).get_fdata())
                
                # Choose appropriate colormap
                cmap = 'gray' if col < 2 else ('viridis' if col == 2 else 'nipy_spectral')
                
                # Display image
                axes[row, col].imshow(img, cmap=cmap)
                axes[row, col].set_title(f"{sample_dir}\n{modality.split('.')[0]}")
                axes[row, col].axis('off')
            else:
                axes[row, col].text(0.5, 0.5, f"No {modality} for {sample_dir}", 
                                   ha='center', va='center')
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


import torch
from torch.utils.data import Dataset


class BrainSegmentationDataset(Dataset):
    def __init__(self, ds_dir, sample_dirs, transform=None, target_labels=None):
        """
        PyTorch Dataset for brain MRI segmentation with selectable labels.
        
        Args:
            ds_dir (str): Path to the dataset directory
            sample_dirs (list): List of sample directories to use
            transform (callable, optional): Transform to apply to the data
            target_labels (list, optional): List of label IDs to keep in the segmentation.
                                           If None, all labels are kept.
                                           If provided, other labels will be set to 0.
        """
        self.ds_dir = ds_dir
        self.sample_dirs = sample_dirs
        self.transform = transform
        self.target_labels = target_labels
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # Load the input image (normalized MRI)
        img_path = os.path.join(self.ds_dir, sample_dir, 'slice_norm.nii.gz')
        img = np.squeeze(nib.load(img_path).get_fdata()).astype(np.float32)
        
        # Load the segmentation mask with 24 labels
        mask_path = os.path.join(self.ds_dir, sample_dir, 'slice_seg24.nii.gz')
        mask = np.squeeze(nib.load(mask_path).get_fdata()).astype(np.float32)
        
        # Filter the mask to only include target labels
        if self.target_labels is not None:
            # Create a binary mask for each label and combine them
            filtered_mask = np.zeros_like(mask)
            for label in self.target_labels:
                filtered_mask[mask == label] = label
            mask = filtered_mask
        
        # Add channel dimension if needed (C×H×W format)
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
            mask = mask[np.newaxis, :, :]
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)
        
        return img_tensor, mask_tensor
