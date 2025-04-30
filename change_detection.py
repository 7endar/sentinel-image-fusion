import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform


def save_image_tif(image, path, transform, dtype=np.float32):
    with rasterio.open(
        path, 'w', driver='GTiff', count=image.shape[0] if len(image.shape) == 3 else 1,
        dtype=dtype, height=image.shape[-2], width=image.shape[-1], transform=transform
    ) as dst:
        if len(image.shape) == 3:
            dst.write(image)
        else:
            dst.write(image, 1)


def mask_change_detection(pre_mask, post_mask, pre_transform, method_name):
    diff_mask = np.abs(post_mask - pre_mask)
    diff_mask_normalized = (diff_mask - diff_mask.min()) / (diff_mask.max() - diff_mask.min() + 1e-6)
    diff_mask_normalized = np.clip(diff_mask_normalized, 0, 1)

    threshold = 0.01
    mask_change_map = (diff_mask_normalized > threshold).astype(np.uint8)

    rgb_image = np.zeros((pre_mask.shape[-2], pre_mask.shape[-1], 3))
    rgb_image[..., 0] = mask_change_map * 1.0
    rgb_image[..., 1] = pre_mask * (1 - mask_change_map) * 0.5
    rgb_image[..., 2] = post_mask * (1 - mask_change_map) * 0.5
    plt.imsave(f"change_detections/{method_name.lower()}_mask.png", np.clip(rgb_image, 0, 1))

    # SAVE AS TIFF
    os.makedirs("change_detections/tif", exist_ok=True)
    save_image_tif(rgb_image.transpose(2, 0, 1), f"change_detections/tif/{method_name.lower()}_mask.tif", pre_transform, dtype=np.float32)

    # Damaged Areas
    post_unchanged_image = np.zeros((pre_mask.shape[-2], pre_mask.shape[-1], 3))
    unchanged_mask = (1 - mask_change_map) * post_mask
    post_unchanged_image[..., 2] = unchanged_mask * (rgb_image[..., 0] == 0) * (rgb_image[..., 1] == 0)
    plt.imsave(f"change_detections/damage_masks/{method_name.lower()}_damage_mask.png", np.clip(post_unchanged_image, 0, 1))


fusion_methods = {
    'Brovey': ("urban_masks/tif/Brovey_pre_non_combined_urban_mask.tif", "urban_masks/tif/Brovey_post_non_combined_urban_mask.tif"),
    'PCA': ("urban_masks/tif/PCA_pre_non_combined_urban_mask.tif", "urban_masks/tif/PCA_post_non_combined_urban_mask.tif"),
    'Wavelet': ("urban_masks/tif/Wavelet_pre_non_combined_urban_mask.tif", "urban_masks/tif/Wavelet_post_non_combined_urban_mask.tif"),
    'Default': ("urban_masks/tif/Default_pre_non_combined_urban_mask.tif", "urban_masks/tif/Default_post_non_combined_urban_mask.tif")
}

for method_name, (pre_mask_path, post_mask_path) in fusion_methods.items():
    pre_mask, pre_mask_transform = read_tif(pre_mask_path)
    post_mask, post_mask_transform = read_tif(post_mask_path)
    if pre_mask.shape != post_mask.shape:
        raise ValueError(f"Mask shapes do not match for {method_name}: {pre_mask.shape} vs {post_mask.shape}")

    mask_change_detection(pre_mask, post_mask, pre_mask_transform, method_name)
