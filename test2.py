import cv2
import numpy as np
import matplotlib.pyplot as plt

def lab_color_denoise(image):
    """
    Denoises an image by converting it to LAB color space and applying
    fast Non-Local Means denoising selectively on the A and B channels,
    while preserving the L channel.
    """
    # Convert the image to LAB color space.
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_image)
    a_denoised = cv2.fastNlMeansDenoising(a, None, h=10, templateWindowSize=7, searchWindowSize=21)
    b_denoised = cv2.fastNlMeansDenoising(b, None, h=10, templateWindowSize=7, searchWindowSize=21)
    denoised_lab = cv2.merge((L, a_denoised, b_denoised))
    denoised = cv2.cvtColor(denoised_lab, cv2.COLOR_LAB2BGR)
    return denoised

def lab_outlier_denoise(image, threshold=10):
    """
    Applies a Lab-based outlier denoising filter:
      1. Converts the image to Lab.
      2. Applies median filtering to the a and b channels.
      3. Detects outlier pixels in a and b (i.e. those that deviate by more than
         'threshold' from the neutral value 128).
      4. Replaces outlier pixels with the neutral value (128).
      5. Recombines channels and converts back to BGR.
    Optionally, histograms and a visual mask can be displayed for debugging.
    """
    # Convert to Lab space.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    
    # Apply median filtering to chrominance channels.
    a_filtered = cv2.medianBlur(a, 3)  # Kernel size 3; adjust if necessary.
    b_filtered = cv2.medianBlur(b, 3)
    
    # Detect outliers: pixels with a or b values away from 128 by more than threshold.
    mask_a = np.abs(a_filtered - 128) > threshold
    mask_b = np.abs(b_filtered - 128) > threshold
    mask = mask_a | mask_b  # Combined boolean mask.
    
    # (Optional) Visualize histograms and outlier mask.
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.hist(a.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
    plt.axvline(x=128 - threshold, color='black', linestyle='--', label='Threshold')
    plt.axvline(x=128 + threshold, color='black', linestyle='--')
    plt.title('Histogram of a Channel')
    plt.xlabel('Value')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(b.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.axvline(x=128 - threshold, color='black', linestyle='--', label='Threshold')
    plt.axvline(x=128 + threshold, color='black', linestyle='--')
    plt.title('Histogram of b Channel')
    plt.xlabel('Value')
    plt.legend()
    
    mask_visual = (mask.astype(np.uint8)) * 255
    colored_mask = cv2.applyColorMap(mask_visual, cv2.COLORMAP_JET)
    
    plt.subplot(2, 2, 3)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(colored_mask)
    plt.title('Outlier Noise Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Replace outlier pixels with neutral value (128).
    a_denoised = np.where(np.abs(a_filtered - 128) > threshold, 128, a_filtered).astype(np.uint8)
    b_denoised = np.where(np.abs(b_filtered - 128) > threshold, 128, b_filtered).astype(np.uint8)
    
    # Merge the processed channels with the original L channel.
    lab_denoised = cv2.merge([L, a_denoised, b_denoised])
    denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
    return denoised

def main():
    # -------------------------
    # Step 1: Load the image.
    # -------------------------
    input_path = 'driving_images/im001-snow.jpg'
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image at {input_path}")
        return

    # -------------------------
    # Step 2: Apply primary denoising.
    # -------------------------
    denoised_image = lab_color_denoise(image)
    
    # -------------------------
    # Step 3: Apply Lab-based outlier denoising.
    # -------------------------
    denoised_outlier = lab_outlier_denoise(denoised_image, threshold=10)
    
    # -------------------------
    # Step 4: Display and save the result.
    # -------------------------
    cv2.imshow("Denoised Outlier Removal", denoised_outlier)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output_image.jpg', denoised_outlier)
    print("Processing complete. Output saved as 'output_image.jpg'.")

if __name__ == "__main__":
    main()