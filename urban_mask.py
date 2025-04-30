import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform

def save_mask_tif(mask, transform, out_path):
    with rasterio.open(
        out_path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
        count=1, dtype=mask.dtype, transform=transform
    ) as dst:
        dst.write(mask, 1)

def save_mask_png(mask, out_path):
    plt.imsave(out_path, mask, cmap='gray')

def sar_threshold_mask(sar_image, threshold=-8):
    return (sar_image > threshold).astype(np.uint8)

def fused_kmeans_mask(fused_image, n_clusters=2):
    h, w = fused_image.shape
    labels = KMeans(n_clusters=n_clusters, random_state=0).fit(fused_image.reshape(-1, 1)).labels_.reshape(h, w)
    city_cluster = np.argmax([fused_image[labels == i].mean() for i in range(n_clusters)])
    return (labels == city_cluster).astype(np.uint8)


sar_paths = {
    "pre": "data/S1_Pre_VV.tif",
    "post": "data/S1_Post_VV.tif"
}

fusion_methods = {
    'Brovey': ("fusion_results/tif/brovey_pre.tif", "fusion_results/tif/brovey_post.tif"),
    'PCA': ("fusion_results/tif/pca_pre.tif", "fusion_results/tif/pca_post.tif"),
    'Wavelet': ("fusion_results/tif/wavelet_pre.tif", "fusion_results/tif/wavelet_post.tif"),
    'Default': ("data/S2_Pre_RGB.tif", "data/S2_Post_RGB.tif")
}

for method, (pre_path, post_path) in fusion_methods.items():
    for period, path in zip(["pre", "post"], [pre_path, post_path]):

        sar_image, sar_transform = read_tif(sar_paths[period])
        sar_mask = sar_threshold_mask(sar_image)

        fused_image, fused_transform = read_tif(path)
        fused_mask = fused_kmeans_mask(fused_image)
        combined_mask = sar_mask * fused_mask

        base_name = f"{method.lower()}_{period}"
        tif_path = f"urban_masks/tif/{base_name}_urban_mask.tif"
        png_path = f"urban_masks/{base_name}_urban_mask.png"

        save_mask_tif(combined_mask, fused_transform, tif_path)
        save_mask_png(combined_mask, png_path)

        # Fusion maskesinin SAR ile birleşmeden direkt halini kaydet (non_combined)
        non_combined_tif_path = f"urban_masks/tif/{base_name}_non_combined_urban_mask.tif"
        non_combined_png_path = f"urban_masks/{base_name}_non_combined_urban_mask.png"
        save_mask_tif(fused_mask, fused_transform, non_combined_tif_path)
        save_mask_png(fused_mask, non_combined_png_path)
