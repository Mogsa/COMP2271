import cv2
import numpy as np
import argparse
import subprocess

def denoise_nlm(image_path, output_path):
    """ Step 1: Apply Non-Local Means (NLM) Denoising to Preserve Edges. """
    image = cv2.imread(image_path)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    cv2.imwrite(output_path, denoised)
    print(f"Step 1: NLM Denoising Done! Saved to {output_path}")
    return output_path

def detect_snowy_regions(image_path, output_mask_path):
    """ Step 2: Detect Snowy/Low-Texture Areas for Targeted Inpainting. """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian to detect texture variations
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_map = np.abs(laplacian)

    # Threshold for low texture (smooth areas)
    texture_threshold = 15  # Lower = More Aggressive (captures more flat areas)
    smooth_mask = (texture_map < texture_threshold).astype(np.uint8) * 255

    # Brightness threshold to detect white areas (snow)
    brightness_threshold = 200  # Adjust based on how bright the snow is
    bright_mask = (gray > brightness_threshold).astype(np.uint8) * 255

    # Combine both masks (low-texture & bright = Snowy Area)
    snow_mask = cv2.bitwise_and(smooth_mask, bright_mask)

    cv2.imwrite(output_mask_path, snow_mask)
    print(f"Step 2: Snow Region Detection Done! Mask saved to {output_mask_path}")
    return output_mask_path

def run_lama_inpainting(image_path, mask_path, output_path):
    """ Step 3: Apply LaMa (Deep Learning) Inpainting on the Snowy Regions. """
    lama_script = "scripts/inpaint.py"  # Make sure you have LaMa installed
    command = ["python3", lama_script, "--image", image_path, "--mask", mask_path, "--output", output_path]
    try:
        subprocess.run(command, check=True)
        print(f"Step 3: LaMa Inpainting Done! Saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running LaMa: {e}")

def inpaint_snowy_regions(image_path, mask_path, output_path):
    """ Step 3 (Alternative): Use OpenCV’s Inpainting on the Snowy Areas. """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8) * 255

    # Apply OpenCV inpainting
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(output_path, inpainted)
    print(f"Step 3: OpenCV Inpainting Done! Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Denoise ocular noise from an image.")
    parser.add_argument("--method", type=str, choices=["lama", "opencv"], default="lama",
                        help="Choose inpainting method: 'lama' (Deep Learning) or 'opencv' (PatchMatch).")
    args = parser.parse_args()

    # Input and Output Paths
    input_image = "driving_images/im001-snow.jpg"
    nlm_output = "denoised_nlm.jpg"
    snow_mask_path = "snow_mask.jpg"
    final_output = "final_cleaned.jpg"

    # Step 1: Apply Non-Local Means Denoising (Preserves Objects)
    nlm_output = denoise_nlm(input_image, nlm_output)

    # Step 2: Detect Snowy Areas
    snow_mask_path = detect_snowy_regions(nlm_output, snow_mask_path)

    # Step 3: Apply Inpainting for Snow Regions
    if args.method == "lama":
        run_lama_inpainting(nlm_output, snow_mask_path, final_output)
    else:
        inpaint_snowy_regions(nlm_output, snow_mask_path, final_output)

    print(f"All steps completed! Final output saved as {final_output}")

if __name__ == "__main__":
    main()
