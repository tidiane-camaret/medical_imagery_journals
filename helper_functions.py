import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def display_samples(ds_dir, sample_dirs, n_rows):
    # Define modalities
    modalities = [
        "slice_orig.nii.gz",
        "slice_norm.nii.gz",
        "slice_seg4.nii.gz",
        "slice_seg24.nii.gz",
    ]

    # Create figure
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

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
                cmap = (
                    "gray" if col < 2 else ("viridis" if col == 2 else "nipy_spectral")
                )

                # Display image
                axes[row, col].imshow(img, cmap=cmap)
                axes[row, col].set_title(f"{sample_dir}\n{modality.split('.')[0]}")
                axes[row, col].axis("off")
            else:
                axes[row, col].text(
                    0.5,
                    0.5,
                    f"No {modality} for {sample_dir}",
                    ha="center",
                    va="center",
                )
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


import torch
from sympy import im
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
                                           If one label is provided, its value will be set to 1.
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
        img_path = os.path.join(self.ds_dir, sample_dir, "slice_norm.nii.gz")
        img = np.squeeze(nib.load(img_path).get_fdata()).astype(np.float32)

        # Load the segmentation mask with 24 labels
        mask_path = os.path.join(self.ds_dir, sample_dir, "slice_seg24.nii.gz")
        mask = np.squeeze(nib.load(mask_path).get_fdata()).astype(np.float32)

        # Filter the mask to only include target labels
        if self.target_labels is not None:

            filtered_mask = np.zeros_like(mask)
            if len(self.target_labels) == 1:
                # If only one label is provided, set it to 1
                filtered_mask[mask == self.target_labels[0]] = 1
            else:
                # Create a binary mask for each label and combine them
                for label in self.target_labels:
                    filtered_mask[mask == label] = label
            mask = filtered_mask

        # Add channel dimension if needed (C×H×W format)
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
            mask = mask[np.newaxis, :, :]

        # Apply transformations if any

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor


import os

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import io, transform
from tqdm import tqdm


def evaluate_medsam_on_dataset(
    MedSAM_CKPT_PATH,
    val_loader,
    # ds_dir, val_dirs, target_labels, checkpoint_path,
    device="cuda",
    max_samples=None,
    plot_results=False,
    save_plots_dir=None,
):
    """
    Evaluate MedSAM performance on validation dataset.

    Args:
        ds_dir: Base directory containing the dataset
        val_dirs: List of validation sample directories
        target_labels: Labels to include in ground truth mask
        checkpoint_path: Path to MedSAM checkpoint
        device: Computation device
        max_samples: Maximum number of samples to evaluate (None for all)
        plot_results: Whether to plot original, ground truth and prediction
        save_plots_dir: Directory to save plots (None for no saving)

    Returns:
        mean_dice: Mean Dice coefficient across validation set
    """
    """
    # Load MedSAM model
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    
    # Limit the number of samples if specified
    if max_samples is not None:
        val_dirs = val_dirs[:max_samples]
    """

    MedSAM_CKPT_PATH = "model_checkpoints/medsam_vit_b.pth"
    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    print(f"Evaluating {len(val_loader)} samples with MedSAM...")
    # Create directory for saving plots if needed
    if plot_results and save_plots_dir is not None:
        os.makedirs(save_plots_dir, exist_ok=True)

    dice_scores = []
    sample_count = 0

    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating MedSAM")):
            if max_samples is not None and sample_count >= max_samples:
                break
            print(len(batch))
            images, masks = batch
            images = images.to(device)[0]
            masks = masks.to(device)[0]
            """
            # Load original image
            img_path = os.path.join(ds_dir, sample_dir, 'slice_norm.nii.gz')
            img_np = np.asarray(nib.load(img_path).dataobj)
            """
            img_np = images.cpu().numpy().squeeze()
            if len(img_np.shape) > 2:
                img_np = np.squeeze(img_np)

            # Convert to 3 channels if grayscale
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np

            # Normalize image to 0-255 for visualization
            img_3c = (
                (img_3c - img_3c.min()) / (img_3c.max() - img_3c.min()) * 255
            ).astype(np.uint8)
            """
            # Load ground truth mask
            mask_path = os.path.join(ds_dir, sample_dir, 'slice_seg24.nii.gz')
            gt_mask = np.asarray(nib.load(mask_path).dataobj)
            if len(gt_mask.shape) > 2:
                gt_mask = np.squeeze(gt_mask)
            
            # Create binary mask for target labels
            binary_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            for label in target_labels:
                binary_mask[gt_mask == label] = 1
                
            # If no pixels match the target labels, skip this sample
            if binary_mask.sum() == 0:
                print(f"Skipping {sample_dir}: No pixels match target labels")
                continue
            """
            binary_mask = masks.cpu().numpy().squeeze()
            if len(binary_mask.shape) > 2:
                binary_mask = np.squeeze(binary_mask)
            # Get bounding box from ground truth mask
            # Format: [x_min, y_min, x_max, y_max]
            y_indices, x_indices = np.where(binary_mask > 0)




            if len(y_indices) == 0:
                continue  # Skip if mask is empty

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # Add small padding to bounding box (5% of dimensions)
            h, w = binary_mask.shape
            pad_x = int(0.05 * (x_max - x_min))
            pad_y = int(0.05 * (y_max - y_min))

            x_min = max(0, x_min - pad_x)
            x_max = min(w - 1, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h - 1, y_max + pad_y)

            box_np = np.array([[x_min, y_min, x_max, y_max]])

            # MedSAM preprocessing
            H, W = img_3c.shape[:2]
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)

            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )

            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            )

            # Convert box to 1024x1024 scale
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            # Run MedSAM inference
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor)
                medsam_seg = medsam_inference(
                    medsam_model, image_embedding, box_1024, H, W
                )

            # Calculate Dice score
            intersection = np.logical_and(medsam_seg, binary_mask).sum()
            union = medsam_seg.sum() + binary_mask.sum()

            if union > 0:
                dice = (2.0 * intersection) / union
                dice_scores.append(dice)
                print( f"Dice score = {dice:.4f}")

            # Plot results if requested
            if plot_results:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image with bounding box
                axes[0].imshow(img_3c)
                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                axes[0].add_patch(rect)
                axes[0].set_title(f"Original Image + Box")
                axes[0].axis("off")

                # Ground truth mask
                axes[1].imshow(img_3c)
                axes[1].imshow(binary_mask, alpha=0.5, cmap="viridis")
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                # MedSAM prediction
                axes[2].imshow(img_3c)
                axes[2].imshow(medsam_seg, alpha=0.5, cmap="plasma")
                axes[2].set_title(f"MedSAM Prediction\nDice: {dice:.4f}")
                axes[2].axis("off")

                plt.tight_layout()

                # Save plot if directory is provided
                if save_plots_dir is not None:
                    plt.savefig(
                        os.path.join(
                            save_plots_dir,
                            f"{sample_idx}_dice_{dice:.4f}.png",
                        )
                    )
                    plt.close()
                else:
                    plt.show()

                sample_count += 1

    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"Evaluation complete. Mean Dice score: {mean_dice:.4f}")
    return mean_dice, dice_scores


# Use the medsam_inference function from the provided code
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def evaluate_trained_model(
    model,
    val_loader,
    device="cuda",
    max_samples=None,
    plot_results=False,
    save_plots_dir=None,
):
    """
    Evaluate trained segmentation model on validation dataset.

    Args:
        model: Trained PyTorch segmentation model
        val_loader: Validation DataLoader
        device: Computation device
        max_samples: Maximum number of samples to evaluate (None for all)
        plot_results: Whether to plot original, ground truth and prediction
        save_plots_dir: Directory to save plots (None for no saving)

    Returns:
        mean_dice: Mean Dice coefficient across validation set
        dice_scores: List of individual Dice scores
    """
    model.eval()

    # Create directory for saving plots if needed
    if plot_results and save_plots_dir is not None:
        os.makedirs(save_plots_dir, exist_ok=True)

    dice_scores = []
    sample_count = 0

    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating model")):
            if max_samples is not None and sample_count >= max_samples:
                break

            # Get images and masks from batch
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

            # Process each item in the batch
            for i in range(images.shape[0]):
                if max_samples is not None and sample_count >= max_samples:
                    break

                # Get single image and mask
                image = images[i : i + 1]
                mask = masks[i : i + 1]

                # Forward pass
                with torch.amp.autocast(device_type=device):
                    output = model(image)
                    pred = torch.sigmoid(output) > 0.5

                # Calculate Dice score
                pred_np = pred.cpu().numpy().squeeze().astype(np.uint8)
                mask_np = mask.cpu().numpy().squeeze().astype(np.uint8)

                intersection = np.logical_and(pred_np, mask_np).sum()
                union = pred_np.sum() + mask_np.sum()

                if union > 0:
                    dice = (2.0 * intersection) / union
                    dice_scores.append(dice)
                    print(f"Sample {sample_count}: Dice score = {dice:.4f}")

                # Plot results if requested
                if plot_results:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # Original image
                    img_np = image.cpu().numpy().squeeze()
                    if len(img_np.shape) > 2 and img_np.shape[0] in [
                        1,
                        3,
                    ]:  # Handle channel-first format
                        img_np = np.transpose(img_np, (1, 2, 0))
                        if img_np.shape[2] == 1:  # Single channel
                            img_np = img_np.squeeze(2)

                    axes[0].imshow(img_np, cmap="gray")
                    axes[0].set_title(f"Original Image\nSample {sample_count}")
                    axes[0].axis("off")

                    # Ground truth mask
                    axes[1].imshow(img_np, cmap="gray")
                    axes[1].imshow(mask_np, alpha=0.5, cmap="viridis")
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis("off")

                    # Model prediction
                    axes[2].imshow(img_np, cmap="gray")
                    axes[2].imshow(pred_np, alpha=0.5, cmap="plasma")
                    axes[2].set_title(f"Model Prediction\nDice: {dice:.4f}")
                    axes[2].axis("off")

                    plt.tight_layout()

                    # Save plot if directory is provided
                    if save_plots_dir is not None:
                        plt.savefig(
                            os.path.join(
                                save_plots_dir,
                                f"sample_{sample_count}_dice_{dice:.4f}.png",
                            )
                        )
                        plt.close()
                    else:
                        plt.show()

                sample_count += 1

    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"Evaluation complete. Mean Dice score: {mean_dice:.4f}")
    return mean_dice, dice_scores


# Example usage
"""
# First load your trained model
model = YourModelClass()
model.load_state_dict(torch.load("path/to/model.pth"))
model.to(device)

# Create validation dataset and dataloader
val_dataset = BrainSegmentationDataset(DS_DIR, val_dirs, target_labels=[1])
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Evaluate the model
mean_dice, all_dice_scores = evaluate_trained_model(
    model=model,
    val_loader=val_loader,
    device="cuda",
    max_samples=5,  # Only evaluate the first 5 samples
    plot_results=True,  # Generate plots
    save_plots_dir="./model_results"  # Save plots to this directory
)
"""
