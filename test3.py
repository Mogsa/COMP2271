import cv2
import numpy as np
import matplotlib.pyplot as plt
import bm3d  # BM3D denoising package
from enum import IntFlag

class BM3DStage(IntFlag):
    HARD_THRESHOLDING = 1
    WIENER = 2
    ALL = HARD_THRESHOLDING | WIENER

def lab_color_denoise(image):
    """
    Denoises an image by converting it to LAB color space and applying
    fast Non-Local Means denoising selectively on the A and B channels,
    while preserving the L channel (luminance).

    Parameters:
      image: Input color image (BGR) as a NumPy array.

    Returns:
      denoised: The denoised image in BGR color space.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    a_denoised = cv2.fastNlMeansDenoising(a_channel, None, h=10, 
                                           templateWindowSize=7, searchWindowSize=21)
    b_denoised = cv2.fastNlMeansDenoising(b_channel, None, h=10, 
                                           templateWindowSize=7, searchWindowSize=21)
    denoised_lab = cv2.merge((l_channel, a_denoised, b_denoised))
    denoised = cv2.cvtColor(denoised_lab, cv2.COLOR_LAB2BGR)
    return denoised

def bm3d_denoise(image, sigma=10):
    """
    Applies BM3D as a secondary noise filter.

    Parameters:
      image: Input color image (BGR) as a NumPy array.
      sigma: Estimated noise standard deviation (in intensity units, e.g., 10).
             BM3D requires the noise level to be normalized relative to 1.

    Returns:
      denoised: The denoised image in BGR color space.
    """
    # BM3D works in float32 [0, 1]
    image_float = image.astype(np.float32) / 255.0
    # Convert from BGR to RGB because BM3D generally expects RGB.
    image_rgb = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
    # Normalize sigma to [0,1]
    sigma_normalized = sigma / 255.0
    # Use our BM3DStage enum instead of bm3d.BM3DStage
    denoised_rgb = bm3d.bm3d(image_rgb, sigma_psd=sigma_normalized, stage_arg=BM3DStage.ALL)
    # Convert back to BGR and uint8.
    denoised_rgb = np.clip(denoised_rgb * 255, 0, 255).astype(np.uint8)
    denoised_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)
    return denoised_bgr

def main():
    image_path = "driving_images/im001-snow.jpg"
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Apply the first denoising (LAB-based).
    denoised_lab = lab_color_denoise(original)
    
    # Apply BM3D as the secondary noise filter.
    denoised_final = bm3d_denoise(denoised_lab, sigma=10)
    
    # Convert images to RGB for display with matplotlib.
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised_final, cv2.COLOR_BGR2RGB)
    
    # Display the original and the final denoised image.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(denoised_rgb)
    axs[1].set_title("Denoised Image (BM3D Secondary Filter)")
    axs[1].axis("off")
    plt.show()

if __name__ == "__main__":
    main()