import cv2
import numpy as np
import matplotlib.pyplot as plt

def denoise_image(input_path, output_path):
    """
    Apply Non-Local Means denoising to an image using OpenCV.
    
    Parameters:
    input_path (str): Path to the input noisy image
    output_path (str): Path to save the denoised image
    
    Returns:
    tuple: Original image and denoised image
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        raise FileNotFoundError(f"Could not read image from {input_path}")
    
    # Convert to RGB for display purposes (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply Non-Local Means Denoising
    # Parameters:
    # h: Filter strength (higher values remove more noise but might blur details)
    # templateWindowSize: Size of template patch for similarity calculation
    # searchWindowSize: Size of window for searching similar patches
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10,         # Filter strength for luminance
        hColor=10,    # Filter strength for color components
        templateWindowSize=7,  # Size of template patch
        searchWindowSize=21    # Size of window for searching similar patches
    )
    
    # Convert denoised to RGB for display
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    # Save the denoised image
    cv2.imwrite(output_path, denoised)
    
    return img_rgb, denoised_rgb

def show_comparison(original, denoised):
    """
    Display the original and denoised images side by side.
    
    Parameters:
    original (ndarray): Original noisy image
    denoised (ndarray): Denoised image
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Noisy Image')
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Denoised Image (Non-Local Means)')
    plt.imshow(denoised)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with actual input and output paths
    input_image_path = "driving_images/im060-rain.jpg"  
    output_image_path = "denoised_image.jpg"
    
    # Apply denoising
    original, denoised = denoise_image(input_image_path, output_image_path)
    
    # Show comparison
    show_comparison(original, denoised)

# Advanced usage with parameter tuning
def denoise_with_parameters(input_path, h_values=None, template_sizes=None, search_sizes=None):
    """
    Test different denoising parameters and compare results.
    
    Parameters:
    input_path (str): Path to the input noisy image
    h_values (list): List of filter strength values to test
    template_sizes (list): List of template window sizes to test
    search_sizes (list): List of search window sizes to test
    """
    if h_values is None:
        h_values = [5, 10, 15]
    
    if template_sizes is None:
        template_sizes = [5, 7]
    
    if search_sizes is None:
        search_sizes = [15, 21]
    
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {input_path}")
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a grid of denoised images with different parameters
    rows = len(h_values)
    cols = len(template_sizes) * len(search_sizes)
    
    plt.figure(figsize=(4*cols, 4*rows))
    
    # Add original image
    plt.subplot(rows, cols, 1)
    plt.title('Original')
    plt.imshow(img_rgb)
    plt.axis('off')
    
    idx = 1
    for h in h_values:
        for template_size in template_sizes:
            for search_size in search_sizes:
                idx += 1
                
                # Apply denoising with current parameters
                denoised = cv2.fastNlMeansDenoisingColored(
                    img,
                    None,
                    h=h,
                    hColor=h,
                    templateWindowSize=template_size,
                    searchWindowSize=search_size
                )
                
                # Convert to RGB for display
                denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
                
                # Display
                plt.subplot(rows, cols, idx)
                plt.title(f'h={h}, t={template_size}, s={search_size}')
                plt.imshow(denoised_rgb)
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png')
    plt.show()

# To use the parameter tuning function:
# denoise_with_parameters("noisy_image.jpg")