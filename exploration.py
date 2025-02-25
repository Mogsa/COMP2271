import cv2
import os
import numpy as np
import argparse

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


def process_image(image, fix_perspective=True, fix_brightness=True, auto_adjust=True,
                 brightness=30, contrast=1.5, use_clahe=False):
    """
    Complete image processing pipeline
    
    Args:
        image: Input image
        fix_perspective: Whether to apply perspective correction
        fix_brightness: Whether to apply brightness/contrast correction
        auto_adjust: Use automatic brightness/contrast adjustment
        brightness: Manual brightness value
        contrast: Manual contrast value
        use_clahe: Use CLAHE instead of linear adjustments
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Step 1: Perspective correction
    if fix_perspective:
        result = auto_perspective_correction(result)
    
    # Step 2: Brightness and contrast correction
    if fix_brightness:
        if auto_adjust:
            result = auto_brightness_contrast(result)
        else:
            result = apply_brightness_contrast(result, brightness, contrast, use_clahe)
    
    return result

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

# Update the main processing function to use the new pipeline
def process_images(input_dir, output_dir, fix_perspective=True, fix_brightness=True, 
                  auto_adjust=True, brightness=30, contrast=1.5, use_clahe=False):
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
                auto_adjust=auto_adjust,
                brightness=brightness,
                contrast=contrast,
                use_clahe=use_clahe
            )
            
            cv2.imwrite(output_path, processed)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


def main():
    """Parse command line arguments and process images"""
    parser = argparse.ArgumentParser(
        description="Image processing pipeline with perspective and brightness/contrast correction."
    )
    parser.add_argument("--input_dir", default="driving_images", 
                       help="Directory containing images to process")
    parser.add_argument("--output_dir", default="Results", 
                       help="Directory to save processed images")
    parser.add_argument("--skip_perspective", action="store_true",
                       help="Skip perspective correction")
    parser.add_argument("--skip_brightness", action="store_true",
                       help="Skip brightness/contrast correction")
    parser.add_argument("--manual_adjust", action="store_true",
                       help="Use manual brightness/contrast values instead of auto adjustment")
    parser.add_argument("--brightness", type=int, default=30,
                       help="Brightness adjustment (-100 to 100, default: 30)")
    parser.add_argument("--contrast", type=float, default=1.5,
                       help="Contrast adjustment (0.0 to 3.0, default: 1.5)")
    parser.add_argument("--use_clahe", action="store_true",
                       help="Use CLAHE for brightness/contrast correction")
    
    args = parser.parse_args()
    
    print(f"Processing images from {args.input_dir} to {args.output_dir}")
    process_images(
        args.input_dir, 
        args.output_dir,
        fix_perspective=not args.skip_perspective,
        fix_brightness=not args.skip_brightness,
        auto_adjust=not args.manual_adjust,
        brightness=args.brightness,
        contrast=args.contrast,
        use_clahe=args.use_clahe
    )
    print("Processing complete!")


if __name__ == "__main__":
    main()