import os
import rasterio
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(fused, reference):
    min_h = min(fused.shape[1], reference.shape[1])
    min_w = min(fused.shape[2], reference.shape[2])

    fused = fused[:, :min_h, :min_w]
    reference = reference[:, :min_h, :min_w]

    # Normalize
    fused_norm = (fused - fused.min()) / (fused.max() - fused.min())
    ref_norm = (reference - reference.min()) / (reference.max() - reference.min())

    ssim_val = ssim(ref_norm.transpose(1, 2, 0),
                    fused_norm.transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=1.0)

    psnr_val = psnr(ref_norm, fused_norm, data_range=1.0)
    rmse_val = np.sqrt(mean_squared_error(ref_norm.flatten(), fused_norm.flatten()))
    mae_val = mean_absolute_error(ref_norm.flatten(), fused_norm.flatten())
    mse_val = mean_squared_error(ref_norm.flatten(), fused_norm.flatten())

    return ssim_val, psnr_val, rmse_val, mae_val, mse_val


fusion_results_dir = 'fusion_results/tif'
pre_path = 'data/S2_Pre_RGB.tif'  
post_path = 'data/S2_Post_RGB.tif' 

with rasterio.open(pre_path) as src:
    reference_image1 = src.read()

with rasterio.open(post_path) as src:
    reference_image2 = src.read()


fusion_files = [f for f in os.listdir(fusion_results_dir) if f.endswith('.tif')]
results = []

for fusion_file in fusion_files:
    fusion_image_path = os.path.join(fusion_results_dir, fusion_file)

    with rasterio.open(fusion_image_path) as src:
        fused_image = src.read()

    ssim_val1, psnr_val1, rmse_val1, mae_val1, mse_val1 = calculate_metrics(fused_image, reference_image1)
    ssim_val2, psnr_val2, rmse_val2, mae_val2, mse_val2 = calculate_metrics(fused_image, reference_image2)

    avg_ssim = (ssim_val1 + ssim_val2) / 2
    avg_psnr = (psnr_val1 + psnr_val2) / 2
    avg_rmse = (rmse_val1 + rmse_val2) / 2
    avg_mae = (mae_val1 + mae_val2) / 2
    avg_mse = (mse_val1 + mse_val2) / 2

    results.append({
        'Fusion File': fusion_file,
        'SSIM Pre': ssim_val1,
        'PSNR Pre': psnr_val1,
        'RMSE Pre': rmse_val1,
        'MAE Pre': mae_val1,
        'MSE Pre': mse_val1,
        'SSIM Post': ssim_val2,
        'PSNR Post': psnr_val2,
        'RMSE Post': rmse_val2,
        'MAE Post': mae_val2,
        'MSE Post': mse_val2,
        'Avg SSIM': avg_ssim,
        'Avg PSNR': avg_psnr,
        'Avg RMSE': avg_rmse,
        'Avg MAE': avg_mae,
        'Avg MSE': avg_mse
    })

df = pd.DataFrame(results)
df.to_csv('fusion_metrics.csv', index=False)
