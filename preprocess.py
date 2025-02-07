"""
Run the script from the command line with:
    python vector_median_filter.py --input_dir=/path/to/your/images --ksize 3

This script applies a vector median filter to the first valid image found in the specified input directory.
It is designed to reduce multicolored salt-and-pepper noise by treating each pixel as a vector (in BGR color space)
and replacing it with the vector median in a ksize x ksize window if it differs significantly (using a hard-coded
threshold of 30) from its neighbors.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def vector_median_filter(image, ksize=3):
    """
    Applies a vector median filter to a color image.
    
    For each pixel, a window of size ksize x ksize is examined. The function computes the
    vector median (the pixel whose sum of Euclidean distances to all other pixels in the window is minimal).
    If the Euclidean distance between the center pixel and this vector median exceeds a hard-coded threshold (30),
    the pixel is replaced by the vector median; otherwise, it is left unchanged.
    
    Parameters:
      image : Input color image (BGR) as a NumPy array.
      ksize : Window size (must be an odd integer, e.g., 3 or 5).
      
    Returns:
      The filtered image as a NumPy array (uint8).
    """
    # Hard-coded threshold for replacement.
    threshold = 30
    
    pad = ksize // 2
    # Pad the image to handle border pixels (using reflection).
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    H, W = image.shape[:2]
    output = image.copy().astype(np.float32)
    
    # Iterate over each pixel in the original image.
    for i in range(H):
        for j in range(W):
            # Extract the window centered at (i, j)
            window = padded[i:i+ksize, j:j+ksize].astype(np.float32)
            # Flatten the window to a list of pixels; each pixel is a 3D vector.
            pixels = window.reshape(-1, 3)
            n = pixels.shape[0]
            # Compute the sum of distances from each pixel in the window to every other pixel.
            sums = np.zeros(n, dtype=np.float32)
            for a in range(n):
                for b in range(n):
                    if a != b:
                        sums[a] += np.linalg.norm(pixels[a] - pixels[b])
            # Find the index of the vector median.
            median_idx = np.argmin(sums)
            vector_median = pixels[median_idx]
            # Get the center pixel (from the padded image).
            center_pixel = padded[i+pad, j+pad].astype(np.float32)
            # Replace the center pixel if it differs by more than the threshold.
            if np.linalg.norm(center_pixel - vector_median) > threshold:
                output[i, j] = vector_median
            else:
                output[i, j] = center_pixel
    # Clip the output values to [0, 255] and convert to uint8.
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def load_first_image(input_dir):
    """
    Loads the first valid image (with extension .jpg, .jpeg, or .png) found in the input directory.
    
    Returns:
      image : The loaded image (BGR).
      fname : The filename of the image.
    """
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.jpg')):
            img_path = os.path.join(input_dir, f)
            image = cv2.imread(img_path)
            if image is not None:
                return image, f
    return None, None

def main(input_dir, ksize):
    # Load the first valid image from the input directory.
    image, fname = load_first_image(input_dir)
    if image is None:
        print("No valid images found in the specified directory.")
        return

    # Apply the vector median filter.
    filtered_image = vector_median_filter(image, ksize=ksize)

    # Display the original and filtered images side by side.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Vector Median Filtered\n(ksize={ksize}, threshold=30)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a vector median filter to remove multicolored salt-and-pepper noise from the first image in the input directory."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--ksize", type=int, default=3,
                        help="Kernel size for the filter (odd integer, e.g., 3 or 5).")
    args = parser.parse_args()
    main(args.input_dir, args.ksize)