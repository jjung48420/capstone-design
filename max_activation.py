import os
import numpy as np
import matplotlib.pyplot as plt

# Paths (EDIT THESE)
INPUT_NPY = "/data/jun3700/sftp_code/feature_maps/noise/clear_rainy.npy"
OUTPUT_PNG = "/data/jun3700/sftp_code/max_activation/clear_rainy_activation.png"

# Load the NumPy feature map
feat = np.load(INPUT_NPY)
if feat.ndim != 3:
    raise ValueError(f"Expected a 3D array [C,H,W], but got shape {feat.shape}")

# Compute the max over channels map (H, W)
max_activations = feat.max(axis=0)

# Normalize to [0, 255] (uint8)
min_val = max_activations.min()
max_activations = max_activations - min_val
max_val = max_activations.max()
if max_val > 0:
    max_activations = max_activations / max_val
heatmap_uint8 = (max_activations * 255).astype(np.uint8)

# Save as a grayscale PNG
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
plt.imsave(OUTPUT_PNG, heatmap_uint8, cmap="inferno", vmin=0, vmax=255)
