import os
import cv2
import numpy as np  # Needed for absolute value computations

# run the script: python /Users/morgan/Documents/GitHub/COMP2271/preprocess.py --input_dir=/Users/morgan/Documents/GitHub/COMP2271/driving_images --output_dir=/Users/morgan/Documents/GitHub/COMP2271/Output_images

def copy_images(input_dir, output_dir):
    """
    Reads images from the input directory and writes them unchanged to the output directory.
    
    Parameters:
      input_dir  : Directory containing the input images.
      output_dir : Directory where the images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg")):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                # Simply save the image without any modifications.
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, image)
                print(f"Image saved as {output_path}")
            else:
                print(f"Warning: Could not load image {img_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Copy images from input directory to output directory with no processing."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory to save images.")
    args = parser.parse_args()

    copy_images(args.input_dir, args.output_dir)