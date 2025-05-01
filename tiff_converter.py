import os
import numpy as np
from PIL import Image
import rasterio


def create_transparent_overlay_from_png(png_path, reference_tif_path, output_tif_path):
    # PNG'yi RGB olarak oku
    img = Image.open(png_path).convert("RGB")
    rgb = np.array(img)

    # Saydamlık kanalı oluştur: siyah (0,0,0) olan yerler tamamen saydam (alpha = 0)
    alpha = np.where(np.all(rgb == [0, 0, 0], axis=-1), 0, 255).astype(np.uint8)

    # RGBA dizisi oluştur
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    with rasterio.open(reference_tif_path) as ref:
        profile = ref.profile
        profile.update({
            'count': 4,
            'dtype': 'uint8',
            'driver': 'GTiff'
        })
        transform = ref.transform
        crs = ref.crs

    profile.update({'transform': transform, 'crs': crs})

    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)
        dst.write(alpha, 4)


input_folder = "change_detections"
output_folder = os.path.join(input_folder, "gee_tif")
os.makedirs(output_folder, exist_ok=True)
reference_tif_path = "data/S1_Pre_VV.tif"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".tif"
        output_path = os.path.join(output_folder, output_filename)
        create_transparent_overlay_from_png(png_path, reference_tif_path, output_path)
