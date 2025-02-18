"""
This script processes all valid images in the specified input directory.
For each image, it detects the four corners of a warped document (assuming the document
is against a dark background) using a robust edge/contour detection approach.
If the initial method fails, a fallback method using adaptive thresholding is attempted.
Once the corners are detected, a perspective transformation is applied to "unwarp" the document,
and the resulting image is saved to the output directory.
Any images that fail to be processed (for example, if the corners cannot be detected)
are logged.
"""

import os
import cv2
import numpy as np
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
    Detects the four corners of the document in the image using a robust approach.

    Steps:
      1. Convert to grayscale and apply Gaussian blur.
      2. Use Canny edge detection.
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

def fallback_detect_document_corners(image, area_threshold_ratio=0.3):
    """
    Fallback method: Uses adaptive thresholding (instead of Canny) to generate a mask,
    then finds contours from that mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding with inversion so that the document appears white.
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(adaptive_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def process_image(image, filename, output_dir, failed_images):
    """
    Processes one image: detects document corners, corrects perspective, and saves the result.
    If corner detection fails, logs the filename.

    Parameters:
      image       : Input image (BGR).
      filename    : Filename of the image.
      output_dir  : Directory where the processed image will be saved.
      failed_images: List to accumulate filenames of images that fail processing.
    """
    pts = detect_document_corners(image, area_threshold_ratio=0.3)
    if pts is None:
        print(f"Could not detect corners for {filename} using the initial method. Trying fallback...")
        pts = fallback_detect_document_corners(image, area_threshold_ratio=0.3)
    if pts is None:
        print(f"Could not detect corners for {filename} even with fallback. Skipping...")
        failed_images.append(filename)
        return

    unwarped = correct_perspective(image, pts)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, unwarped)
    print(f"Processed and saved {filename}.")

def main():
    parser = argparse.ArgumentParser(description="Document Perspective Correction")
    parser.add_argument("input_dir", help="Path to the input directory containing images.")
    parser.add_argument("output_dir", help="Path to the output directory for processed images.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory '{output_dir}'.")

    failed_images = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {filename}. Skipping...")
                failed_images.append(filename)
                continue
            process_image(image, filename, output_dir, failed_images)

    if failed_images:
        print("\nThe following images failed to process:")
        for f in failed_images:
            print(f)
    else:
        print("\nAll images processed successfully.")

if __name__ == "__main__":
    main()