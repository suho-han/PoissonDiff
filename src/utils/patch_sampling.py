"""
Patch-based sampling utilities for processing large images.
"""

import math

import torch


def patch_sample(
    sample_fn,
    model,
    image,
    prior,
    input_size,
    patches_per_dim=None,
    overlap=None,
    model_kwargs=None,
):
    """
    Sample large images by dividing them into patches with configurable overlap.

    Args:
        sample_fn: The sampling function (p_sample_loop or ddim_sample_loop)
        model: The model to use for sampling
        image: Input image tensor of shape (B, C, H, W)
        prior: Prior tensor of shape (B, C, H, W)
        input_size: Size of patches to feed into the model
        patches_per_dim: Desired number of patches along each dimension (height/width)
        overlap: Optional overlap between adjacent patches (used only when patches_per_dim is not set)
        model_kwargs: Additional model keyword arguments
        batch_size: Batch size for processing

    Returns:
        Combined sampled image and intermediates
    """
    device = image.device
    B, C, H, W = image.shape

    # Derive stride from desired patch count; fall back to overlap-based stride.
    if patches_per_dim is not None and patches_per_dim > 1:
        stride_h = math.ceil(max(H - input_size, 0) / (patches_per_dim - 1)) if H > input_size else input_size
        stride_w = math.ceil(max(W - input_size, 0) / (patches_per_dim - 1)) if W > input_size else input_size
    else:
        overlap_val = overlap if overlap is not None else 0
        stride_h = stride_w = max(1, input_size - overlap_val)

    # Calculate number of patches needed per dimension
    num_patches_h = math.ceil((H - input_size) / stride_h) + 1 if H > input_size else 1
    num_patches_w = math.ceil((W - input_size) / stride_w) + 1 if W > input_size else 1

    # Initialize output tensor
    output = torch.zeros((B, C, H, W), device=device)
    weight_map = torch.zeros((B, C, H, W), device=device)
    all_intermediates = []

    # Process each patch
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            print(f"Sampling patch ({i+1}/{num_patches_h}, {j+1}/{num_patches_w})")
            # Calculate patch coordinates
            start_h = min(i * stride_h, H - input_size)
            start_w = min(j * stride_w, W - input_size)
            end_h = start_h + input_size
            end_w = start_w + input_size

            # Extract patches
            image_patch = image[:, :, start_h:end_h, start_w:end_w]
            prior_patch = prior[:, :, start_h:end_h, start_w:end_w]

            # Update model_kwargs for this patch
            patch_kwargs = {
                "img": image_patch,
                "prior": prior_patch
            }
            if "y" in model_kwargs:
                patch_kwargs["y"] = model_kwargs["y"]

            # Sample this patch
            sample_patch, intermediates = sample_fn(
                model,
                (B, C, input_size, input_size),
                model_kwargs=patch_kwargs,
                return_intermediates=True,
            )

            # Accumulate the patch into output with weighted averaging
            output[:, :, start_h:end_h, start_w:end_w] += sample_patch
            weight_map[:, :, start_h:end_h, start_w:end_w] += 1.0

            # Store intermediates from first patch only to save memory
            if i == 0 and j == 0:
                all_intermediates = intermediates

    # Average overlapping regions
    output = output / weight_map.clamp(min=1.0)

    return output, all_intermediates


if __name__ == "__main__":
    """
    Test patch sampling with 512x512 image divided into 256x256 patches with overlap.
    """
    import numpy as np

    # Create a test 512x512 image (B=1, C=1, H=512, W=512)
    test_image = torch.randn(1, 1, 584, 565)
    test_prior = torch.randn(1, 1, 584, 565)

    # Test parameters
    input_size = 256  # Patch size
    target_size_h = 584  # Full image height
    target_size_w = 565  # Full image width
    overlap = 64  # Overlap between patches

    print(f"Image shape: {test_image.shape}")
    print(f"Patch size: {input_size}x{input_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Stride: {input_size - overlap} pixels")
    print()

    # Calculate expected number of patches
    stride = input_size - overlap
    H, W = 584, 565
    num_patches_h = math.ceil((H - input_size) / stride) + 1 if H > input_size else 1
    num_patches_w = math.ceil((W - input_size) / stride) + 1 if W > input_size else 1

    print(f"Expected patches: {num_patches_h} x {num_patches_w} = {num_patches_h * num_patches_w} total")
    print()

    # Simulate patch extraction to visualize the coordinates
    print("Patch coordinates:")
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = min(i * stride, H - input_size)
            start_w = min(j * stride, W - input_size)
            end_h = start_h + input_size
            end_w = start_w + input_size
            print(f"  Patch ({i},{j}): [{start_h}:{end_h}, {start_w}:{end_w}]")
    print()

    # Visualize overlap regions
    weight_map = torch.zeros((1, 1, H, W))
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = min(i * stride, H - input_size)
            start_w = min(j * stride, W - input_size)
            end_h = start_h + input_size
            end_w = start_w + input_size
            weight_map[:, :, start_h:end_h, start_w:end_w] += 1.0

    print("Weight map statistics (number of overlapping patches per pixel):")
    print(f"  Min: {weight_map.min().item()}")
    print(f"  Max: {weight_map.max().item()}")
    print(f"  Mean: {weight_map.mean().item():.2f}")

    # Show unique values in weight map
    unique_weights = torch.unique(weight_map).numpy()
    print(f"  Unique weights: {unique_weights}")

    # Count pixels by coverage
    for w in unique_weights:
        count = (weight_map == w).sum().item()
        percentage = count / (H * W) * 100
        print(f"    {int(w)}x covered: {count} pixels ({percentage:.1f}%)")
