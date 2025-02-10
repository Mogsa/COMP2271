#!/usr/bin/env python
"""
Run the script from the command line with:
    python /Users/morgan/Documents/GitHub/COMP2271/exploration.py --input_dir=/Users/morgan/Documents/GitHub/COMP2271/driving_images --output_dir=/Users/morgan/Documents/GitHub/COMP2271/Output_images

This script processes all valid images in the input directory using a two-stage noise filtering algorithm designed for
multicolored salt-and-pepper noise that appears in blobs:
  
  Stage 1 (Initial Filtering):  
    A median filter with a relatively large kernel (e.g., 5x5) is applied to remove the most obvious noise.

  Stage 2 (Secondary Filtering):  
    A noise mask is computed by comparing the original image to the initially filtered image. For each pixel, if the
    Euclidean distance (in color space) between the original and the median-filtered pixel exceeds a hard-coded
    threshold (e.g., 30), that pixel is marked as noisy. OpenCV's inpainting (using the TELEA method) is then applied
    to the original image using this mask, which fills in the noisy areas more precisely.

The processed (refined) images are saved in the output directory.
"""

import os
import cv2
import numpy as np
import argparse

def denoise_image(image):
    """
    Applies OpenCV's fastNlMeansDenoisingColored to remove noise from a color image.
    
    Parameters:
      image : Input color image (BGR) as a NumPy array.
    
    Returns:
      denoised : The denoised image.
    """
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

def initial_noise_filter(image, ksize=5):
    """
    Applies a median filter to remove the most obvious noise.

    Parameters:
      image : Input color image (BGR) as a NumPy array.
      ksize : Kernel size for the median filter (e.g., 5).

    Returns:
      The initially filtered image.
    """
    return cv2.medianBlur(image, ksize)

def compute_noise_mask(original, filtered, diff_threshold=30):
    """
    Computes a binary noise mask by comparing the original image to the filtered image.

    For each pixel, the Euclidean distance in color space between the original and filtered pixel is computed.
    If the distance exceeds diff_threshold, the pixel is marked as noisy (value 255), otherwise 0.

    Parameters:
      original      : Original color image (BGR).
      filtered      : Initially filtered image (BGR).
      diff_threshold: Threshold value for marking a pixel as noisy.

    Returns:
      A binary mask (uint8) where noisy pixels are 255.
    """
    # Compute absolute difference per channel
    diff = cv2.absdiff(original, filtered).astype(np.float32)
    # Compute the Euclidean norm of the difference for each pixel
    diff_norm = np.sqrt(np.sum(diff**2, axis=2))
    # Create mask: pixels with difference greater than the threshold are marked as noise (255)
    mask = np.uint8((diff_norm > diff_threshold) * 255)
    return mask

def secondary_noise_filter(original, mask, inpaintRadius=3):
    """
    Applies inpainting on the original image using the given mask to remove remaining noise.

    Parameters:
      original     : Original color image (BGR).
      mask         : Binary noise mask where noisy pixels are 255.
      inpaintRadius: Inpainting radius.

    Returns:
      The refined (noise-reduced) image.
    """
    return cv2.inpaint(original, mask, inpaintRadius, cv2.INPAINT_TELEA)

def process_image(image, diff_threshold=30):
    """
    Processes a single image using the two-stage noise filtering algorithm.

    Stage 1: Apply an initial median filter.
    Stage 2: Compute a noise mask and apply inpainting for precise noise removal.

    Parameters:
      image         : Input color image (BGR).
      diff_threshold: Threshold for detecting noisy pixels.

    Returns:
      refined_image: The final noise-filtered image.
      mask         : The computed noise mask.
      initial_filtered: The image after initial median filtering.
    """
    # Stage 1: Initial noise filtering using median filter
    initial_filtered = initial_noise_filter(image, ksize=5)
    # Stage 2: Compute noise mask from difference between original and filtered image
    mask = compute_noise_mask(image, initial_filtered, diff_threshold=diff_threshold)
    # Apply inpainting using the computed noise mask
    refined_image = secondary_noise_filter(image, mask, inpaintRadius=3)
    return refined_image, mask, initial_filtered

def process_all_images(input_dir, output_dir, diff_threshold=30):
    """
    Processes all valid images in the input directory using the two-stage noise filtering algorithm,
    and saves the refined images to the output directory.

    Parameters:
      input_dir     : Path to the directory containing images.
      output_dir    : Path to the directory where processed images will be saved.
      diff_threshold: Threshold for the noise mask.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No valid images found in the specified directory.")
        return

    failed_images = []
    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load {filename}.")
            failed_images.append(filename)
            continue
        
        try:
            refined, mask, initial_filtered = process_image(image, diff_threshold=diff_threshold)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, refined)
            print(f"Processed {filename} saved to {output_path}.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_images.append(filename)
    
    if failed_images:
        print("\nThe following images failed to be processed:")
        for f in failed_images:
            print(f"  - {f}")
    else:
        print("\nAll images were processed successfully.")

def main(input_dir, output_dir, diff_threshold):
    process_all_images(input_dir, output_dir, diff_threshold=diff_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a two-stage noise filtering algorithm to remove multicolored salt-and-pepper noise from all images in the input directory, then save the refined images to the output directory."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where processed images will be saved.")
    parser.add_argument("--diff_threshold", type=int, default=30,
                        help="Threshold for detecting noisy pixels based on color difference (default=30).")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.diff_threshold)