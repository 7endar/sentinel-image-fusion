import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt


def read_image(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    return img


def normalize_image(image):
    p2, p98 = np.percentile(image, (2, 98))
    norm_img = np.clip((image - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (norm_img * 255).astype(np.uint8)


def save_image_as_tif(image, path, reference_path):
    with rasterio.open(reference_path) as ref:
        profile = ref.profile
        profile.update(dtype=rasterio.float32, count=image.shape[0])
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(image.astype(np.float32))


def save_image_as_png(image, path):
    norm_img = normalize_image(image)
    plt.imsave(path, norm_img.transpose(1, 2, 0))


# FUSION FUNCTIONS

def brovey_fusion(s1, s2):
    s1_norm = (s1 - np.percentile(s1, 2)) / (np.percentile(s1, 98) - np.percentile(s1, 2) + 1e-6)
    sum_rgb = np.sum(s2, axis=0) + 1e-6
    return np.array([s2[i] * s1_norm / sum_rgb for i in range(3)])


def pca_fusion(s1, s2):
    combined = np.vstack((s2, s1[np.newaxis, ...])).reshape(4, -1).T
    pca = PCA(n_components=3)
    fused = pca.fit_transform(combined)
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-6) * 255 
    return fused.T.reshape(3, s2.shape[1], s2.shape[2])


def wavelet_fusion(s1, s2):
    coeffs_s2 = [pywt.dwt2(s2[i], 'haar') for i in range(3)]
    coeffs_s1 = pywt.dwt2(s1, 'haar')

    fused_low = (coeffs_s2[0][0] + coeffs_s2[1][0] + coeffs_s2[2][0] + coeffs_s1[0]) / 4
    fused_high = (
        np.maximum(np.abs(coeffs_s2[0][1][0]), np.abs(coeffs_s1[1][0])),
        np.maximum(np.abs(coeffs_s2[0][1][1]), np.abs(coeffs_s1[1][1])),
        np.maximum(np.abs(coeffs_s2[0][1][2]), np.abs(coeffs_s1[1][2]))
    )

    return np.array([pywt.idwt2((fused_low, fused_high), 'haar') for _ in range(3)])


s1_post = read_image("data/S1_Post_VV.tif")[0]
s2_post = read_image("data/S2_Post_RGB.tif")
s1_pre = read_image("data/S1_Pre_VV.tif")[0]
s2_pre = read_image("data/S2_Pre_RGB.tif")

post_brovey_result = brovey_fusion(s1_post, s2_post)
pre_brovey_result = brovey_fusion(s1_pre, s2_pre)
post_pca_result = pca_fusion(s1_post, s2_post)
pre_pca_result = pca_fusion(s1_pre, s2_pre)
post_wavelet_result = wavelet_fusion(s1_post, s2_post)
pre_wavelet_result = wavelet_fusion(s1_pre, s2_pre)

# Kaydet (Post)
save_image_as_tif(post_brovey_result, "fusion_results/tif/brovey_post.tif", "data/S2_Post_RGB.tif")
save_image_as_tif(post_pca_result, "fusion_results/tif/pca_post.tif", "data/S2_Post_RGB.tif")
save_image_as_tif(post_wavelet_result, "fusion_results/tif/wavelet_post.tif", "data/S2_Post_RGB.tif")
save_image_as_png(post_brovey_result, "fusion_results/brovey_post.png")
save_image_as_png(post_pca_result, "fusion_results/pca_post.png")
save_image_as_png(post_wavelet_result, "fusion_results/wavelet_post.png")

# Kaydet (Pre)
save_image_as_tif(pre_brovey_result, "fusion_results/tif/brovey_pre.tif", "data/S2_Pre_RGB.tif")
save_image_as_tif(pre_pca_result, "fusion_results/tif/pca_pre.tif", "data/S2_Pre_RGB.tif")
save_image_as_tif(pre_wavelet_result, "fusion_results/tif/wavelet_pre.tif", "data/S2_Pre_RGB.tif")
save_image_as_png(pre_brovey_result, "fusion_results/brovey_pre.png")
save_image_as_png(pre_pca_result, "fusion_results/pca_pre.png")
save_image_as_png(pre_wavelet_result, "fusion_results/wavelet_pre.png")