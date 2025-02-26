import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def detect_black_circle(image, threshold=30, min_radius=20):
    """
    Detect a black circle in the top right corner of the image.
    
    Args:
        image: Input image
        threshold: Intensity threshold for black
        min_radius: Minimum radius to consider
        
    Returns:
        mask: Binary mask where 1 indicates the circle region
    """
    # Create a copy of the image for processing
    img_copy = image.copy()
    
    # Consider only the top-right quadrant
    height, width = img_copy.shape[:2]
    top_right = img_copy[0:height//2, width//2:width]
    
    # Convert to grayscale if it's not already
    if len(img_copy.shape) == 3:
        gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)
    else:
        gray = top_right
    
    # Threshold to find dark regions
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros_like(image[:,:,0]) if len(image.shape) == 3 else np.zeros_like(image)
    
    # Find the largest circular contour
    max_circularity = 0
    best_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Avoid division by zero
        if perimeter == 0:
            continue
            
        # Circularity = 4π(area/perimeter²)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius) 
        
        # Adjust coordinates to full image space
        x += width // 2
        
        # Check if it's reasonably circular and big enough
        if circularity > max_circularity and radius > min_radius:
            max_circularity = circularity
            best_contour = contour
            center = (int(x), int(y))
            radius = int(radius)
    
    # Draw the detected circle on the mask
    if best_contour is not None:
        cv2.circle(mask, center, radius, 1, -1)  # -1 means filled
        
    return mask


class ExemplarInpainting:
    """
    Implementation of the Criminisi algorithm for exemplar-based inpainting
    as described in the paper:
    "Region Filling and Object Removal by Exemplar-Based Image Inpainting"
    """
    
    def __init__(self, image, mask, patch_size=9):
        """
        Initialize the inpainting algorithm.
        
        Args:
            image: Input image to be inpainted
            mask: Binary mask (1 indicates regions to be filled)
            patch_size: Size of patches (must be odd)
        """
        if patch_size % 2 == 0:
            patch_size += 1  # Ensure patch size is odd
            
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        
        # Create working copies
        if len(image.shape) == 3:
            self.image = image.copy().astype(np.float32)
            self.mask = mask.copy().astype(np.float32)
            self.working_image = image.copy().astype(np.float32)
            self.fill_front = np.zeros_like(mask)
            self.confidence = (1.0 - mask).astype(np.float32)
            self.data = np.zeros_like(mask, dtype=np.float32)
            self.priorities = np.zeros_like(mask, dtype=np.float32)
            self.normals = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
            self.gradients = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
        else:
            # Convert grayscale to 3-channel
            self.image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR).astype(np.float32)
            self.mask = mask.copy().astype(np.float32)
            self.working_image = self.image.copy()
            self.fill_front = np.zeros_like(mask)
            self.confidence = (1.0 - mask).astype(np.float32)
            self.data = np.zeros_like(mask, dtype=np.float32)
            self.priorities = np.zeros_like(mask, dtype=np.float32)
            self.normals = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
            self.gradients = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
    
    def update_fill_front(self):
        """Update the fill front (boundary of the target region)"""
        # Dilate the mask and subtract the original to get the boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(self.mask, kernel)
        self.fill_front = dilated - self.mask
        
    def compute_normals(self):
        """Compute normal vectors along the fill front"""
        # Compute gradients of the mask
        grad_x = cv2.Sobel(self.mask, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.mask, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize to get normal vectors
        norm = np.sqrt(grad_x**2 + grad_y**2)
        norm[norm == 0] = 1  # Avoid division by zero
        
        self.normals[:,:,0] = grad_x / norm
        self.normals[:,:,1] = grad_y / norm
        
    def compute_gradients(self):
        """Compute image gradients (for the data term)"""
        # Convert image to grayscale for gradient computation
        if self.image.shape[2] == 3:
            gray = cv2.cvtColor(self.working_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = self.working_image[:,:,0].astype(np.uint8)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Store gradients
        self.gradients[:,:,0] = grad_x
        self.gradients[:,:,1] = grad_y
    
    def compute_priorities(self):
        """Compute priorities for all pixels on the fill front"""
        self.update_fill_front()
        self.compute_normals()
        self.compute_gradients()
        
        # Prepare priority map
        self.priorities = np.zeros_like(self.mask, dtype=np.float32)
        
        # Find fill front pixels
        front_pixels = np.where(self.fill_front > 0)
        front_coords = list(zip(front_pixels[0], front_pixels[1]))
        
        for y, x in front_coords:
            # Skip pixels too close to the boundary for patch extraction
            if (y < self.half_patch or y >= self.image.shape[0] - self.half_patch or
                x < self.half_patch or x >= self.image.shape[1] - self.half_patch):
                continue
            
            # Compute confidence term C(p)
            patch_mask = self.mask[y-self.half_patch:y+self.half_patch+1, 
                                   x-self.half_patch:x+self.half_patch+1]
            
            confidence_patch = self.confidence[y-self.half_patch:y+self.half_patch+1, 
                                              x-self.half_patch:x+self.half_patch+1]
            
            confidence_term = np.sum(confidence_patch * (1 - patch_mask)) / (self.patch_size**2)
            
            # Compute data term D(p)
            normal = self.normals[y, x]
            
            # Find perpendicular (isophote)
            isophote = np.array([-self.gradients[y, x, 1], self.gradients[y, x, 0]])
            
            # Normalize isophote
            isophote_magnitude = np.linalg.norm(isophote)
            if isophote_magnitude > 0:
                isophote = isophote / isophote_magnitude
            
            data_term = abs(np.dot(isophote, normal))
            
            # Compute priority P(p) = C(p) * D(p)
            priority = confidence_term * data_term
            
            # Store computed priority
            self.priorities[y, x] = priority
            
            # Update confidence value
            self.confidence[y, x] = confidence_term
        
        return front_coords
    
    def find_best_patch(self, point):
        """
        Find the best matching source patch for a target patch centered at point.
        
        Args:
            point: (y, x) coordinates of the center of the target patch
        
        Returns:
            best_match: (y, x) coordinates of the center of the best matching source patch
        """
        y, x = point
        
        # Extract the target patch
        target_patch = self.working_image[y-self.half_patch:y+self.half_patch+1, 
                                         x-self.half_patch:x+self.half_patch+1]
        target_mask = self.mask[y-self.half_patch:y+self.half_patch+1,
                               x-self.half_patch:x+self.half_patch+1]
        
        # Valid pixels in the target patch (pixels that are already filled)
        valid_mask = 1 - target_mask
        
        best_match = None
        best_error = float('inf')
        
        # Iterate through all possible source patches
        for src_y in range(self.half_patch, self.image.shape[0] - self.half_patch):
            for src_x in range(self.half_patch, self.image.shape[1] - self.half_patch):
                # Skip if source patch overlaps with the target region
                source_mask = self.mask[src_y-self.half_patch:src_y+self.half_patch+1,
                                      src_x-self.half_patch:src_x+self.half_patch+1]
                
                if np.sum(source_mask) > 0:
                    continue
                
                # Extract the source patch
                source_patch = self.working_image[src_y-self.half_patch:src_y+self.half_patch+1,
                                                src_x-self.half_patch:src_x+self.half_patch+1]
                
                # Compute SSD error for valid pixels
                if len(self.image.shape) == 3:
                    error = np.sum(((target_patch - source_patch)**2) * np.expand_dims(valid_mask, axis=2))
                else:
                    error = np.sum(((target_patch - source_patch)**2) * valid_mask)
                
                # Normalize by the number of valid pixels
                error = error / np.sum(valid_mask)
                
                if error < best_error:
                    best_error = error
                    best_match = (src_y, src_x)
        
        return best_match
    
    def copy_patch(self, target_point, source_point):
        """
        Copy the source patch to the target patch, but only for masked pixels.
        
        Args:
            target_point: (y, x) coordinates of the center of the target patch
            source_point: (y, x) coordinates of the center of the source patch
        """
        y, x = target_point
        src_y, src_x = source_point
        
        # Extract patches
        target_mask = self.mask[y-self.half_patch:y+self.half_patch+1,
                               x-self.half_patch:x+self.half_patch+1].copy()
        
        source_patch = self.working_image[src_y-self.half_patch:src_y+self.half_patch+1,
                                        src_x-self.half_patch:src_x+self.half_patch+1].copy()
        
        # Create a 3D mask if needed
        if len(self.image.shape) == 3:
            target_mask_3d = np.repeat(target_mask[:, :, np.newaxis], 3, axis=2)
        else:
            target_mask_3d = target_mask
        
        # Update the working image
        target_region = self.working_image[y-self.half_patch:y+self.half_patch+1,
                                          x-self.half_patch:x+self.half_patch+1]
        
        target_region[target_mask > 0] = source_patch[target_mask > 0]
        
        # Update the mask
        self.mask[y-self.half_patch:y+self.half_patch+1,
                 x-self.half_patch:x+self.half_patch+1] = 0
        
        # Update the confidence term
        self.confidence[y-self.half_patch:y+self.half_patch+1,
                       x-self.half_patch:x+self.half_patch+1] = self.confidence[y, x]
    
    def inpaint(self, max_iterations=1000):
        """
        Perform exemplar-based inpainting.
        
        Args:
            max_iterations: Maximum number of iterations to perform
            
        Returns:
            The inpainted image
        """
        iteration = 0
        
        while np.sum(self.mask) > 0 and iteration < max_iterations:
            # Compute priorities for all pixels on the fill front
            front_coords = self.compute_priorities()
            
            if not front_coords:
                break
            
            # Find the pixel with the highest priority
            priorities = [self.priorities[y, x] for y, x in front_coords]
            if not priorities:
                break
                
            best_idx = np.argmax(priorities)
            best_point = front_coords[best_idx]
            
            # Find the best matching patch
            source_point = self.find_best_patch(best_point)
            
            if source_point is None:
                # If no good match is found, mark this pixel as processed and continue
                y, x = best_point
                self.mask[y, x] = 0
                continue
            
            # Copy the patch
            self.copy_patch(best_point, source_point)
            
            iteration += 1
            
            # Optional: print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, remaining pixels: {np.sum(self.mask)}")
        
        return self.working_image.astype(np.uint8)


def process_image(image_path, output_path=None):
    """
    Process an image by detecting a black circle and inpainting it.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (optional)
        
    Returns:
        Original image, mask, and inpainted image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert from BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect the black circle
    mask = detect_black_circle(image)
    
    # Create the inpainting object
    inpainter = ExemplarInpainting(image, mask, patch_size=9)
    
    # Perform inpainting
    result = inpainter.inpaint()
    
    # Convert result from BGR to RGB for display
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, result)
    
    return image_rgb, mask, result_rgb


def display_results(image, mask, result):
    """
    Display the original image, mask, and inpainted result.
    
    Args:
        image: Original image
        mask: Binary mask
        result: Inpainted result
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title('Inpainted Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "Results/im004-snow.jpg"
    output_path = "inpainted_result.jpg"
    
    try:
        image, mask, result = process_image(image_path, output_path)
        display_results(image, mask, result)
    except Exception as e:
        print(f"Error: {e}")