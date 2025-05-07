import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ======================
# 1. Load and Prepare CT
# ======================
def load_ct(filepath):
    """Load DICOM CT scan and normalize to [0, 1]"""
    ds = pydicom.dcmread(filepath)
    ct = ds.pixel_array.astype(float)
    return (ct - np.min(ct)) / (np.max(ct) - np.min(ct)), ds

# ======================
# 2. Noise Functions
# ======================
def add_gaussian_noise(image, noise_level=0.1):
    """Add Gaussian noise to image"""
    noisy = image + np.random.normal(0, noise_level, image.shape)
    return np.clip(noisy, 0, 1)

def save_noisy_image(image, filename):
    """Save noisy image as PNG"""
    Image.fromarray((image * 255).astype(np.uint8)).save(filename)

# ======================
# 3. Anisotropic Diffusion
# ======================
def anisotropic_diffusion(img, iterations=0, k=0, lambda_=0):
    """Edge-preserving denoising filter"""
    img = img.copy().astype(float)
    for _ in range(iterations):
        gradN = np.roll(img, -1, axis=0) - img  # North
        gradS = np.roll(img,  1, axis=0) - img  # South
        gradE = np.roll(img, -1, axis=1) - img  # East
        gradW = np.roll(img,  1, axis=1) - img  # West
        
        cN = np.exp(-(gradN/k)**2)  # Diffusion coefficients
        cS = np.exp(-(gradS/k)**2)
        cE = np.exp(-(gradE/k)**2)
        cW = np.exp(-(gradW/k)**2)
        
        img += lambda_ * (cN*gradN + cS*gradS + cE*gradE + cW*gradW)
    return np.clip(img, 0, 1)

# ======================
# 4. Main Workflow
# ======================
# Load CT scan (replace with your file path)
ct_clean, ds = load_ct("ct_scan.dcm")

# Add noise and save
noisy_ct = add_gaussian_noise(ct_clean, noise_level=0.15)
save_noisy_image(noisy_ct, "noisy_ct.png")

# Denoise
denoised_ct = anisotropic_diffusion(noisy_ct, iterations=500, k=0.1, lambda_=0.1)

# ======================
# 5. Visualization
# ======================
plt.figure(figsize=(15, 5))
#plt.subplot(131), plt.imshow(ct_clean), plt.title("Old CT")
plt.subplot(132), plt.imshow(noisy_ct), plt.title("Original CT")
plt.subplot(133), plt.imshow(denoised_ct), plt.title("Filtered CT")
plt.show()

# ======================
# 6. Quantitative Analysis
# ======================
print(f"SNR (Original CT): {psnr(ct_clean, noisy_ct):.2f} dB")
print(f"SNR (Filtered CT): {psnr(ct_clean, denoised_ct):.2f} dB")
#print(f"SSIM (Denoised vs Original): {ssim(ct_clean, denoised_ct):.3f}")

# Save results
# Image.fromarray((denoised_ct * 255).astype(np.uint8)).save("denoised_ct.png")

# After creating the noisy image
noisy_ct = add_gaussian_noise(ct_clean, noise_level=0.15)

# Save noisy DICOM
ds.PixelData = (noisy_ct * (2**ds.BitsStored - 1)).astype(ds.pixel_array.dtype).tobytes()
ds.save_as("noisy_ct.dcm")

# (Optional) Also save as PNG for quick viewing
save_noisy_image(noisy_ct, "noisy_ct.png")