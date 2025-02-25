import cv2
import os
import numpy as np
import argparse


def remove_multicolored_noise(image, strength=3):
    """
    Specialized function to remove multicolored noise patterns while preserving details
    
    Args:
        image: Input image with multicolored noise
        strength: Denoising strength (1-5)
        
    Returns:
        Denoised image
    """
    # Make a copy to avoid modifying the original
    result = image.copy()
    
    # Step 1: Color space conversion to work in a more noise-separable space
    # LAB color space separates luminance from color information
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Step 2: Apply targeted filtering to luminance channel
    # This preserves edges while removing noise
    l_denoised = cv2.fastNlMeansDenoising(l, None, h=10*strength, templateWindowSize=7, searchWindowSize=21)
    
    # Step 3: Apply stronger filtering to color channels where human eye is less sensitive
    # Use a series of filters for maximum effectiveness
    
    # 3.1 First, apply small median filter to remove extreme outliers (salt & pepper)
    a_filtered = cv2.medianBlur(a, 3)
    b_filtered = cv2.medianBlur(b, 3)
    
    # 3.2 Then apply bilateral filter to smooth while preserving edges
    a_filtered = cv2.bilateralFilter(a_filtered, 5, 25*strength, 5*strength)
    b_filtered = cv2.bilateralFilter(b_filtered, 5, 25*strength, 5*strength)
    
    # 3.3 Finally, apply non-local means to the color channels
    a_denoised = cv2.fastNlMeansDenoising(a_filtered, None, h=15*strength, templateWindowSize=7, searchWindowSize=21)
    b_denoised = cv2.fastNlMeansDenoising(b_filtered, None, h=15*strength, templateWindowSize=7, searchWindowSize=21)
    
    # Step 4: Merge channels and convert back to BGR
    lab_denoised = cv2.merge([l_denoised, a_denoised, b_denoised])
    result = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
    
    return result


def remove_multicolored_noise_detail_preserving(image, strength=3):
    """
    Enhanced version of multicolored noise removal with better detail preservation
    
    Args:
        image: Input image with multicolored noise
        strength: Denoising strength (1-5)
        
    Returns:
        Denoised image with preserved details
    """
    # Make a copy to avoid modifying the original
    result = image.copy()
    
    # Step 1: Edge detection to create a detail mask before denoising
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly to ensure all details are covered
    kernel = np.ones((2, 2), np.uint8)
    detail_mask = cv2.dilate(edges, kernel, iterations=1)
    
    # Create a more sensitive detail mask using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_mask = np.uint8(np.absolute(laplacian) > 5)
    
    # Combine both masks for comprehensive detail detection
    combined_mask = cv2.bitwise_or(detail_mask, laplacian_mask * 255)
    
    # Step 2: Color space conversion to work in a more noise-separable space
    # LAB color space separates luminance from color information
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Step 3: Apply different filtering strengths to luminance vs. color channels
    
    # 3.1: For luminance channel - use gentler filtering to preserve details
    # Reduce strength for luminance to preserve more details
    l_strength = max(1, strength - 1)  # Reduce strength for luminance channel
    
    # First use bilateral filter which is excellent for preserving edges
    l_filtered = cv2.bilateralFilter(l, 5, 20*l_strength, 7)
    
    # Then apply gentle non-local means for areas without strong details
    l_denoised = cv2.fastNlMeansDenoising(l_filtered, None, 
                                         h=7*l_strength,  # Reduced h parameter
                                         templateWindowSize=5,  # Smaller window
                                         searchWindowSize=15)
    
    # 3.2: For color channels - use stronger filtering as human eye is less sensitive to color details
    # Apply small median filter to remove extreme outliers (salt & pepper)
    a_filtered = cv2.medianBlur(a, 3)
    b_filtered = cv2.medianBlur(b, 3)
    
    # Then apply bilateral filter to smooth while preserving edges
    a_filtered = cv2.bilateralFilter(a_filtered, 5, 25*strength, 5*strength)
    b_filtered = cv2.bilateralFilter(b_filtered, 5, 25*strength, 5*strength)
    
    # Finally, apply non-local means to the color channels
    a_denoised = cv2.fastNlMeansDenoising(a_filtered, None, h=15*strength, 
                                          templateWindowSize=7, searchWindowSize=21)
    b_denoised = cv2.fastNlMeansDenoising(b_filtered, None, h=15*strength, 
                                          templateWindowSize=7, searchWindowSize=21)
    
    # Step 4: Preserve the original luminance in areas with strong details
    combined_mask_normalized = combined_mask.astype(float) / 255.0
    
    # Create a weight map based on the detail mask (values between 0.3 and 1.0)
    # This allows partial denoising even in detail areas
    detail_weight = 0.3 + (0.7 * combined_mask_normalized)
    
    # Apply weighted combination of original and denoised luminance
    l_result = np.uint8(l * detail_weight + l_denoised * (1 - detail_weight))
    
    # Step 5: Merge channels and convert back to BGR
    lab_denoised = cv2.merge([l_result, a_denoised, b_denoised])
    result = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
    
    return result


def apply_enhanced_denoising(image, strength=3, detail_preservation=0.6):
    """
    Enhanced multi-stage approach for noise removal with detail preservation
    
    Args:
        image: Input image with mixed noise
        strength: Denoising strength (1-5)
        detail_preservation: Level of detail to preserve (0.0-1.0)
                            Higher values preserve more detail but may keep more noise
        
    Returns:
        Denoised image with preserved details
    """
    # Adjust parameters to reasonable ranges
    strength = max(1, min(5, strength))
    detail_preservation = max(0.0, min(1.0, detail_preservation))
    
    # Stage 1: Extract details from original image that we want to preserve
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a detail mask using multiple methods for better coverage
    # Sobel for detecting strong edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_mask = np.uint8(sobel_magnitude > 20)
    
    # Laplacian for detecting fine details
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_mask = np.uint8(np.absolute(laplacian) > 5)
    
    # Combine masks
    detail_mask = cv2.bitwise_or(sobel_mask, laplacian_mask) * 255
    
    # Dilate to include neighboring pixels around details
    kernel = np.ones((2, 2), np.uint8)
    detail_mask = cv2.dilate(detail_mask, kernel, iterations=1)
    
    # Stage 2: First handle color noise with initial HSV filtering
    # This targets the multicolored noise while preserving structure
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply adaptive median filter to saturation channel
    # This reduces color noise while preserving edges
    s_filtered = cv2.medianBlur(s, 3)
    
    # Use bilateral filter for value channel to preserve edges
    v_filtered = cv2.bilateralFilter(v, 5, 30, 30)
    
    # Recombine and convert back to BGR
    hsv_filtered = cv2.merge([h, s_filtered, v_filtered])
    color_filtered = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
    
    # Stage 3: Apply detail-preserving denoising
    denoised = remove_multicolored_noise_detail_preserving(color_filtered, strength)
    
    # Stage 4: Combine original image details with denoised image
    # Create a normalized weight map from detail mask
    detail_weight_map = cv2.GaussianBlur(detail_mask.astype(float), (5, 5), 0) / 255.0
    
    # Adjust weight map based on detail_preservation parameter
    detail_weight_map = detail_weight_map * detail_preservation
    
    # Convert maps to 3-channel for weighted blending
    detail_weight_map_3ch = cv2.merge([detail_weight_map, detail_weight_map, detail_weight_map])
    
    # Combine original and denoised using the weight map
    result = np.uint8(image * detail_weight_map_3ch + denoised * (1 - detail_weight_map_3ch))
    
    # Stage 5: Final refinement to smooth any artifacts while preserving edges
    # Use a guided filter for edge-aware smoothing
    refined = cv2.bilateralFilter(result, 5, 10, 10)
    
    # Additional texture-preserving step: 
    # Only apply refined result where detail weight is low
    # Use original with denoising where detail weight is high
    low_detail_mask = 1 - (detail_weight_map_3ch > 0.3)
    final_result = np.uint8(refined * low_detail_mask + result * (1 - low_detail_mask))
    
    return final_result


def preserve_black_areas(original_image, processed_image, threshold=20):
    """
    Preserve black areas (like holes) from the original image
    
    Args:
        original_image: Original image with holes
        processed_image: Processed image where holes need to be preserved
        threshold: Brightness threshold to identify black areas
        
    Returns:
        Image with black areas preserved
    """
    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for very dark areas
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate the mask slightly to ensure all of the black area is covered
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Convert the mask to 3 channels for merging
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Combine the images: use original where mask is white, processed elsewhere
    result = np.where(mask_3ch > 0, original_image, processed_image)
    
    return result


def apply_brightness_contrast(image, brightness=0, contrast=1.0, use_clahe=False):
    """
    Apply brightness and contrast adjustment to an image
    
    Args:
        image: Input image (BGR format)
        brightness: Brightness adjustment value (-100 to 100)
        contrast: Contrast adjustment factor (0.0 to 3.0)
        use_clahe: Whether to apply CLAHE instead of linear adjustments
        
    Returns:
        Adjusted image
    """
    if use_clahe:
        # Convert to LAB color space (better for CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # Apply linear brightness and contrast
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def auto_brightness_contrast(image, clip_hist_percent=1):
    """
    Automatically adjust brightness and contrast based on image histogram
    
    Args:
        image: Input image
        clip_hist_percent: Percentage of histogram to clip at minimum and maximum
        
    Returns:
        Automatically adjusted image
    """
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    # Apply brightness/contrast correction
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def auto_perspective_correction(image):
    """Ensure output is always a perfect square with 90 degree angles"""
    result = try_contour_detection(image)
    if result is None:
        result = try_hough_lines(image)
    if result is None:
        # Final fallback - just make it square
        result = make_square(image)
    
    return result


def try_contour_detection(image):
    """Enhanced contour detection with adaptive parameters"""
    # Enhanced preprocessing for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Adaptive parameter selection based on image characteristics
    mean_brightness = np.mean(enhanced)
    std_brightness = np.std(enhanced)
    
    # Choose parameters based on image characteristics
    if std_brightness < 30:  # Low contrast image
        blur_sizes = [(7, 7)]  # More blur for low contrast
        thresholds = [(30, 100), (50, 150)]
    else:  # Normal or high contrast image
        blur_sizes = [(5, 5)]
        thresholds = [(50, 150), (75, 200)]
    
    # Edge detection with selected parameters
    edges = None
    for blur_size in blur_sizes:
        for threshold1, threshold2 in thresholds:
            blur = cv2.GaussianBlur(enhanced, blur_size, 0)
            current_edges = cv2.Canny(blur, threshold1, threshold2, apertureSize=3)
            
            if edges is None:
                edges = current_edges
            else:
                # Combine edge detection results
                edges = cv2.bitwise_or(edges, current_edges)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter tiny contours
    min_area = gray.shape[0] * gray.shape[1] * 0.01  # 1% of image area
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not contours:
        return None
    
    # Find largest contour    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Try different epsilon values to find exactly 4 points
    approx = None
    peri = cv2.arcLength(largest_contour, True)
    
    # More comprehensive epsilon values to increase chances of finding 4 points
    for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
        current_approx = cv2.approxPolyDP(largest_contour, eps_factor * peri, True)
        if len(current_approx) == 4:
            approx = current_approx
            break
    
    if approx is None or len(approx) != 4:
        return None
    
    return create_square_from_points(image, approx)


def try_hough_lines(image):
    """Improved Hough lines method with adaptive thresholds"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement for better line detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Multi-scale edge detection
    edges1 = cv2.Canny(enhanced, 50, 150, apertureSize=3)
    edges2 = cv2.Canny(cv2.GaussianBlur(enhanced, (5, 5), 0), 30, 100, apertureSize=3)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Adaptive threshold based on image size
    img_area = gray.shape[0] * gray.shape[1]
    threshold = max(50, int(img_area / 20000))  # Scale threshold with image size
    
    # Try multiple Hough parameters to handle different image types
    for hough_threshold in [threshold, threshold//2, threshold*2]:
        lines = cv2.HoughLines(edges, 1, np.pi/180, hough_threshold)
        
        if lines is None or len(lines) < 4:
            continue
        
        # Improved line grouping with angle binning
        h_lines = []
        v_lines = []
        
        for line in lines:
            rho, theta = line[0]
            angle_deg = theta * 180 / np.pi
            
            # More refined grouping for better accuracy
            if (angle_deg < 30 or angle_deg > 150):
                h_lines.append(line[0])
            elif (60 <= angle_deg <= 120):
                v_lines.append(line[0])
        
        # Need at least 2 lines in each direction
        if len(h_lines) < 2 or len(v_lines) < 2:
            continue
        
        # Sort lines by distance from origin for more stable corner detection
        h_lines.sort(key=lambda x: abs(x[0]))
        v_lines.sort(key=lambda x: abs(x[0]))
        
        # Find corners by intersection of lines
        corners = []
        for h in h_lines[:2]:  # Use first two horizontal lines
            for v in v_lines[:2]:  # Use first two vertical lines
                corner = line_intersection(h, v)
                if corner is not None:
                    corners.append(corner)
        
        if len(corners) == 4:
            return create_square_from_points(image, np.array(corners))
    
    return None


def line_intersection(line1, line2):
    """Find intersection point of two lines with improved numerical stability"""
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Check if lines are nearly parallel to avoid numerical issues
    if abs(np.sin(theta1 - theta2)) < 1e-10:
        return None
    
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    
    try:
        x, y = np.linalg.solve(A, b)
        return [int(x), int(y)]
    except np.linalg.LinAlgError:
        return None


def create_square_from_points(image, points):
    """Create a perfect square from 4 points with improved border handling"""
    # Get original image dimensions
    orig_height, orig_width = image.shape[:2]
    max_orig_dimension = max(orig_height, orig_width)
    
    # Reshape points to correct format
    pts = np.float32(points.reshape(4, 2))
    rect = order_points(pts)
    
    # Calculate the maximum possible dimension based on detected quadrilateral
    width = max(
        int(np.linalg.norm(rect[1] - rect[0])),  # Top edge
        int(np.linalg.norm(rect[2] - rect[3]))   # Bottom edge
    )
    
    height = max(
        int(np.linalg.norm(rect[3] - rect[0])),  # Left edge
        int(np.linalg.norm(rect[2] - rect[1]))   # Right edge
    )
    
    # Create a square of the original image's maximum dimension
    square_size = max_orig_dimension
    
    # Set destination points to fill the entire square
    dst = np.array([
        [0, 0],
        [square_size-1, 0],
        [square_size-1, square_size-1],
        [0, square_size-1]
    ], dtype="float32")
    
    # Apply perspective transform with better border handling
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (square_size, square_size), 
                               borderMode=cv2.BORDER_REPLICATE)
    
    return warped


def order_points(pts):
    """Order points in: top-left, top-right, bottom-right, bottom-left order"""
    # Sort by sum of coordinates (x+y)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]  # Top-left has smallest sum
    br = pts[np.argmax(s)]  # Bottom-right has largest sum
    
    # Sort by difference of coordinates (y-x)
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]  # Top-right has smallest difference
    bl = pts[np.argmax(diff)]  # Bottom-left has largest difference
    
    return np.array([tl, tr, br, bl], dtype="float32")


def make_square(image):
    """Improved fallback to make image square with better border color"""
    h, w = image.shape[:2]
    square_size = max(h, w)
    
    # Get average color from the edges for better blending
    edge_pixels = []
    edge_pixels.extend(image[0, :].reshape(-1, 3))  # Top edge
    edge_pixels.extend(image[-1, :].reshape(-1, 3))  # Bottom edge
    edge_pixels.extend(image[:, 0].reshape(-1, 3))  # Left edge
    edge_pixels.extend(image[:, -1].reshape(-1, 3))  # Right edge
    
    bg_color = np.median(np.array(edge_pixels), axis=0).astype(np.uint8)
    
    # Create square with edge color as background
    square = np.full((square_size, square_size, 3), bg_color, dtype=np.uint8)
    
    # Calculate position to center the original image
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    
    # Copy original image to center of square
    square[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return square


def process_image(image, fix_perspective=True, fix_brightness=True, fix_noise=True, 
                 auto_adjust=True, brightness=30, contrast=1.5, use_clahe=False, 
                 denoise_strength=3, detail_preservation=0.6):
    """
    Complete image processing pipeline
    
    Args:
        image: Input image
        fix_perspective: Whether to apply perspective correction
        fix_brightness: Whether to apply brightness/contrast correction
        fix_noise: Whether to apply noise reduction
        auto_adjust: Use automatic brightness/contrast adjustment
        brightness: Manual brightness value
        contrast: Manual contrast value
        use_clahe: Use CLAHE instead of linear adjustments
        denoise_strength: Strength of denoising (1-5)
        detail_preservation: Level of detail to preserve (0.0-1.0)
        
    Returns:
        Processed image
    """
    # Save a copy of the original for preserving black areas
    original = image.copy()
    result = image.copy()
    
    # Step 1: Perspective correction
    if fix_perspective:
        result = auto_perspective_correction(result)
    
    # Step 2: Apply enhanced denoising with detail preservation
    if fix_noise:
        result = apply_enhanced_denoising(result, denoise_strength, detail_preservation)
    
    # Step 3: Brightness and contrast correction
    if fix_brightness:
        if auto_adjust:
            result = auto_brightness_contrast(result)
        else:
            result = apply_brightness_contrast(result, brightness, contrast, use_clahe)
    
    # Step 4: Ensure any black holes from the original image are preserved
    # Create a mask for very dark areas
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray_original, 20, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate the mask slightly to ensure full coverage
    kernel = np.ones((3, 3), np.uint8)
    black_mask = cv2.dilate(black_mask, kernel, iterations=1)
    
    # Apply the mask to preserve black areas from original
    black_mask_3ch = cv2.merge([black_mask, black_mask, black_mask])
    result = np.where(black_mask_3ch > 0, original, result)
    
    return result


def process_images(input_dir, output_dir, fix_perspective=True, fix_brightness=True, fix_noise=True,
                  auto_adjust=True, brightness=30, contrast=1.5, use_clahe=False, 
                  denoise_strength=3, detail_preservation=0.6):
    """Process all images in input_dir and save results to output_dir"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Valid image extensions to process
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in directory: {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images sequentially
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Could not read {filename}. Skipping...")
                continue
            
            print(f"Processing: {filename} ({i+1}/{len(image_files)})")
            
            # Apply the complete processing pipeline
            processed = process_image(
                image, 
                fix_perspective=fix_perspective,
                fix_brightness=fix_brightness,
                fix_noise=fix_noise,
                auto_adjust=auto_adjust,
                brightness=brightness,
                contrast=contrast,
                use_clahe=use_clahe,
                denoise_strength=denoise_strength,
                detail_preservation=detail_preservation
            )
            
            cv2.imwrite(output_path, processed)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


def main():
    """Parse command line arguments and process images"""
    parser = argparse.ArgumentParser(
        description="Advanced image processing pipeline with detail-preserving denoising."
    )
    parser.add_argument("--input_dir", default="driving_images", 
                       help="Directory containing images to process")
    parser.add_argument("--output_dir", default="Results", 
                       help="Directory to save processed images")
    parser.add_argument("--skip_perspective", action="store_true",
                       help="Skip perspective correction")
    parser.add_argument("--skip_brightness", action="store_true",
                       help="Skip brightness/contrast correction")
    parser.add_argument("--skip_noise", action="store_true",
                       help="Skip noise reduction")
    parser.add_argument("--manual_adjust", action="store_true",
                       help="Use manual brightness/contrast values instead of auto adjustment")
    parser.add_argument("--brightness", type=int, default=30,
                       help="Brightness adjustment (-100 to 100, default: 30)")
    parser.add_argument("--contrast", type=float, default=1.5,
                       help="Contrast adjustment (0.0 to 3.0, default: 1.5)")
    parser.add_argument("--use_clahe", action="store_true",
                       help="Use CLAHE for brightness/contrast correction")
    parser.add_argument("--denoise_strength", type=int, choices=range(1, 6), default=3,
                       help="Strength of denoising (1-5, default: 3)")
    parser.add_argument("--detail_preservation", type=float, default=0.6,
                       help="Level of detail to preserve (0.0-1.0, default: 0.6)")
    
    args = parser.parse_args()
    
    print(f"Processing images from {args.input_dir} to {args.output_dir}")
    process_images(
        args.input_dir, 
        args.output_dir,
        fix_perspective=not args.skip_perspective,
        fix_brightness=not args.skip_brightness,
        fix_noise=not args.skip_noise,
        auto_adjust=not args.manual_adjust,
        brightness=args.brightness,
        contrast=args.contrast,
        use_clahe=args.use_clahe,
        denoise_strength=args.denoise_strength,
        detail_preservation=args.detail_preservation
    )
    print("Processing complete!")


if __name__ == "__main__":
    main()