import os
import numpy as np
from PIL import Image
import rasterio

def create_red_transparent_overlay(png_path, reference_tif_path, output_tif_path):
    # PNG'yi grayscale olarak yükle
    img = Image.open(png_path).convert("L")
    mask = np.array(img)

    # RGBA görüntü oluştur (kırmızı alanlar opak, diğer yerler tam saydam)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask > 0] = [255, 0, 0, 255]  # kırmızı ve opak
    rgba[mask == 0] = [0, 0, 0, 0]     # tam saydam

    # Kanalları ayır
    r, g, b, a = [rgba[:, :, i] for i in range(4)]

    # Referans TIF'ten coğrafi bilgi al
    with rasterio.open(reference_tif_path) as ref:
        profile = ref.profile
        profile.update({
            'count': 4,
            'dtype': 'uint8',
            'driver': 'GTiff'
        })

        transform = ref.transform
        crs = ref.crs

    # GeoTIFF olarak kaydet
    profile.update({
        'transform': transform,
        'crs': crs
    })

    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)
        dst.write(a, 4)

    print(f"GeoTIFF oluşturuldu: {output_tif_path}")

# Klasör yolları
input_folder = "change_detections/damage_masks"
output_folder = os.path.join(input_folder, "tif")
reference_tif_path = "data/S1_Pre_VV.tif"

# Kayıt klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# Tüm PNG dosyaları için işlemi yap
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".tif"
        output_path = os.path.join(output_folder, output_filename)
        create_red_transparent_overlay(png_path, reference_tif_path, output_path)
