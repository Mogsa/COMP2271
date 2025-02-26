import os
import cv2
import argparse
from advanced_denoise import AdvancedDenoiser
from tqdm import tqdm

def process_directory(input_dir, output_dir, strength=0.9, preserve_details=True):
    """
    Process all images in input_dir and save results to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Create denoiser instance
    denoiser = AdvancedDenoiser(
        preserve_details=preserve_details,
        strength=strength
    )
    
    # Process each image with progress bar
    for filename in tqdm(image_files, desc="Denoising images"):
        # Load image
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Warning: Could not read image {input_path}")
            continue
        
        # Apply denoising
        denoised = denoiser.denoise(image)
        
        # Save result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, denoised)
    
    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch Image Denoising')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for denoised images')
    parser.add_argument('--strength', '-s', type=float, default=0.5, 
                        help='Denoising strength (0.0 to 1.0)')
    parser.add_argument('--preserve-details', '-p', action='store_true', 
                        help='Preserve fine details')
    
    args = parser.parse_args()
    
    process_directory(
        args.input_dir, 
        args.output_dir,
        strength=args.strength,
        preserve_details=args.preserve_details
    )

if __name__ == "__main__":
    main()