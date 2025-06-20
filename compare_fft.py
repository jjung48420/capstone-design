import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Config
ROOT = "./"
OUTDIR = "./out"
DOMAINS = ["origin", "foggy", "rainy"]
NORM = True

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "spectra"), exist_ok=True)

def load_gray(path, norm=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")
    return img.astype(np.float32) / 255.0 if norm else img.astype(np.float32)

# Collect shapes for padding reference
file_list = []
for d in DOMAINS:
    file_list += sorted(glob.glob(os.path.join(ROOT, d, "*")))[:10]

shapes = [load_gray(fp, norm=False).shape for fp in file_list]
max_h = max(h for h, w in shapes)
max_w = max(w for h, w in shapes)
TARGET_SHAPE = (max_h, max_w)

def pad_to_canvas(img, target_shape=TARGET_SHAPE):
    # Pad image to center in target_shape
    h, w = img.shape
    H, W = target_shape
    pad_top = (H - h) // 2
    pad_bottom = H - h - pad_top
    pad_left = (W - w) // 2
    pad_right = W - w - pad_left
    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

def radial_profile(img):
    # Radially average a 2D array
    h, w = img.shape
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    r = np.hypot(x - center[1], y - center[0])
    r_flat = r.flatten().astype(np.int32)
    img_flat = img.flatten()
    r_max = r_flat.max()
    sums = np.bincount(r_flat, weights=img_flat, minlength=r_max + 1)
    counts = np.bincount(r_flat, minlength=r_max + 1)
    psd = sums / np.maximum(counts, 1)
    return np.arange(r_max + 1), psd

def fft_mag(img):
    # Centered FFT magnitude
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.abs(F)

spectra_mean = {}
psd_curves = {}

# Main processing loop
for domain in DOMAINS:
    files = sorted(glob.glob(os.path.join(ROOT, domain, "*")))
    assert len(files) >= 10, f"Need ≥10 images per domain, found {len(files)} in {domain}"
    spectra_sum = None
    out_sub = os.path.join(OUTDIR, "spectra", domain)
    os.makedirs(out_sub, exist_ok=True)

    for fp in files[:10]:
        name = os.path.splitext(os.path.basename(fp))[0]
        img = load_gray(fp, norm=NORM)
        img = pad_to_canvas(img)
        if img is None:
            raise RuntimeError(f"Could not read {fp}")
        if NORM:
            img = img.astype(np.float32) / 255.0

        # FFT
        mag = fft_mag(img)
        spectra_sum = mag if spectra_sum is None else spectra_sum + mag

        # Save spectrum image
        mag_log = np.log1p(mag)
        mag_norm = (mag_log - mag_log.min()) / (np.ptp(mag_log) + 1e-12)
        cv2.imwrite(os.path.join(out_sub, f"{name}_fft.png"), (mag_norm * 255).astype(np.uint8))

    # Mean spectrum and radial PSD
    spectra_mean[domain] = spectra_sum / 10.0
    r, psd = radial_profile(spectra_mean[domain] ** 2)
    psd_curves[domain] = (r, psd)

# Plot PSD curves for all domains
plt.figure(figsize=(6, 4))
for domain, (r, psd) in psd_curves.items():
    plt.plot(r, 10 * np.log10(psd + 1e-12), label=domain)
plt.xlabel("Spatial frequency radius (pixels)")
plt.ylabel("Power (dB)")
plt.title("Radially Averaged Power Spectral Density")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "psd_compare.png"), dpi=300)
