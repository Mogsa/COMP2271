#!/usr/bin/env python
"""
Run the script from the command line with:
    python /Users/morgan/Documents/GitHub/COMP2271/exploration.py --input_dir=/Users/morgan/Documents/GitHub/COMP2271/driving_images --output_dir=/Users/morgan/Documents/GitHub/COMP2271/Output_images

This script processes all valid images in the specified input directory.
For each image, it detects the four corners of the warped document (assuming the document is against a black background)
using a robust edge/contour detection approach, applies a projective transformation to correct the perspective warp,
and saves the resulting unwarped image to the specified output directory.

Any images that fail to be processed (for example, if the corners cannot be detected) are logged.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
      1. Convert to grayscale and apply Gaussian blur.
      2. Use Canny edge detection to get strong edges.
      3. Dilate the edges to close gaps.
      4. Find external contours and sort them by area.
      5. For each large contour (above a given area threshold), approximate the contour with a polygon.
         - If the approximation has 4 points, return these points.
         - Otherwise, try the convex hull and re‑approximate.
         - If that still does not yield 4 points, use the minimum area rectangle as a fallback.
    
    Parameters:
      image               : Input color image (BGR).
      area_threshold_ratio: Minimum ratio of the image area for a candidate contour.
    
    Returns:
      A NumPy array of shape (4,2) with the corner points, or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = image.shape[0] * image.shape[1]
    min_area = area_threshold_ratio * img_area

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
        else:
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
            else:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                return box.reshape(4, 2)
    return None

def correct_perspective(image, pts):
    """
    Applies a perspective transformation to the image using the provided corner points.
    
    Parameters:
      image: Input image (BGR).
      pts  : A NumPy array of shape (4,2) representing the corner points.
    
    Returns:
      warped: The perspective-corrected (unwarped) image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def load_first_image(input_dir):
    """
    Loads the first valid image (with extension .jpg, .jpeg, or .png) found in the input directory.
    
    Returns:
      image : The loaded image in BGR format.
      fname : The filename of the image.
    """
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, f)
            image = cv2.imread(img_path)
            if image is not None:
                return image, f
    return None, None

def process_image(image, filename, output_dir, failed_images):
    """
    Processes one image: detects document corners, corrects perspective, and saves the result.
    If corner detection fails, logs the filename.
    
    Parameters:
      image      : Input image (BGR).
      filename   : Filename of the image.
      output_dir : Directory where the processed image will be saved.
      failed_images: List to accumulate filenames of images that fail processing.
    """
    pts = detect_document_corners(image, area_threshold_ratio=0.3)
    if pts is None:
        print(f"Could not detect corners for {filename}. Skipping...")
        failed_images.append(filename)
        return
    unwarped = correct_perspective(image, pts)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, unwarped)
    print(f"Processed {filename} saved to {output_path}.")

def process_all_images(input_dir, output_dir):
    """
    Processes all valid images in the input directory:
      - Loads each image.
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
        process_image(image, filename, output_dir, failed_images)

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
        description="Detect document edges and correct the perspective warp for all images in the input directory, saving the results to the output directory. Images that fail processing will be logged."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where processed images will be saved.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)