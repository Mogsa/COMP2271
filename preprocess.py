#!/usr/bin/env python
"""
Run the script from the command line with:
    python preprocess.py --input_dir=/path/to/your/images --output_dir=/path/to/output/images

This script processes all valid images in the input directory:
  - Applies OpenCV's fastNlMeansDenoisingColored to remove noise.
  - Applies robust document corner detection and perspective correction.
  - Saves the refined images to the output directory.
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
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=20, hForColorComponents=20, templateWindowSize=11, searchWindowSize=21)
    return denoised

def order_points(pts):
    """
    Orders four points in the following order: top-left, top-right, bottom-right, bottom-left.
    
    Parameters:
      pts: A NumPy array of shape (4,2).
    
    Returns:
      rect: A NumPy array of shape (4,2) with points ordered as above.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def detect_document_corners(image, area_threshold_ratio=0.3):
    """
    Detects the four corners of the document (or object) in the image using a robust approach.
    
    Steps:
      1. Convert to grayscale and blur the image.
      2. Apply Canny edge detection.
      3. Dilate the edges to close gaps.
      4. Find external contours and sort them by area.
      5. For contours above the area threshold, approximate the contour:
         - If a polygon with 4 vertices is found, return those points.
    
    Parameters:
      image              : Input color image (BGR).
      area_threshold_ratio: Minimum ratio of image area for the candidate polygon.
    
    Returns:
      A NumPy array of shape (4,2) containing the corner points, or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    edged = cv2.dilate(edged, None, iterations=2)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = image.shape[0] * image.shape[1]
    min_area = area_threshold_ratio * img_area
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    return None

def correct_perspective(image, pts):
    """
    Applies a perspective transformation to the image using the provided corner points.
    
    Parameters:
      image: Input image (BGR).
      pts  : A NumPy array of shape (4,2) representing the four corner points.
    
    Returns:
      warped: The perspective-corrected (unwarped) image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image.
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image.
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Set up destination points for the "birds-eye view."
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_all_images(input_dir, output_dir):
    """
    Processes all valid images in the input directory:
      - Loads each image.
      - Applies noise removal using fastNlMeansDenoisingColored.
      - Applies robust document corner detection and perspective correction.
      - Saves the corrected images to the output directory.
      - Logs filenames that fail processing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    failed_images = []
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No valid images found in the specified directory.")
        return

    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load {filename}.")
            failed_images.append(filename)
            continue
        
        try:
            # Apply noise removal
            denoised = denoise_image(image)
            
            # Detect document corners
            pts = detect_document_corners(denoised)
            if pts is None:
                print(f"Could not detect document corners for {filename}.")
                failed_images.append(filename)
                continue
            
            # Correct perspective
            unwarped = correct_perspective(denoised, pts)
            
            # Save the corrected image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, unwarped)
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
        description="Apply noise removal and perspective correction to all images in the input directory, then save the refined images to the output directory."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where processed images will be saved.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)