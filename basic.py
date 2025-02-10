#!/usr/bin/env python
"""
Run the script from the command line with:
    python basic.py --input_dir=/path/to/your/images --output_dir=/path/to/output/images

This script processes all valid images in the input directory using OpenCV's fastNlMeansDenoisingColored
to remove noise and saves the refined images to the output directory.
"""

import os
import cv2
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

def process_all_images(input_dir, output_dir):
    """
    Processes all valid images in the input directory using OpenCV's fastNlMeansDenoisingColored,
    and saves the refined images to the output directory.

    Parameters:
      input_dir  : Path to the directory containing images.
      output_dir : Path to the directory where processed images will be saved.
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
            denoised = denoise_image(image)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, denoised)
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

def main(input_dir, output_dir):
    process_all_images(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply OpenCV's fastNlMeansDenoisingColored to remove noise from all images in the input directory, then save the refined images to the output directory."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where processed images will be saved.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
    