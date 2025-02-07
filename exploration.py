import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_histogram(image, title="Histogram"):
    """
    Converts the image to grayscale, computes its histogram,
    and plots it using matplotlib.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate histogram (256 bins for pixel values 0-255)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.show()

def show_fourier_transform(image, title="Fourier Transform"):
    """
    Converts the image to grayscale, computes the 2D Fourier transform,
    shifts the zero frequency component to the center, and displays the
    magnitude spectrum on a logarithmic scale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the 2D FFT of the image.
    f = np.fft.fft2(gray)
    # Shift the zero frequency component to the center.
    fshift = np.fft.fftshift(f)
    # Compute the magnitude spectrum and take log for better visualization.
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # add 1 to avoid log(0)
    
    plt.figure()
    plt.title(title)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.colorbar()
    plt.show()

def show_edge_detection(image, title="Edge Detection"):
    """
    Converts the image to grayscale and applies the Canny edge detector.
    The result is then displayed using matplotlib.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Parameters for Canny: adjust thresholds as needed.
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    
    plt.figure()
    plt.title(title)
    plt.imshow(edges, cmap='gray')
    plt.show()

def process_image(image, filename):
    """
    Runs experimental analysis on a single image:
      - Displays the original image.
      - Shows the histogram.
      - Displays the Fourier transform magnitude spectrum.
      - Displays the result of edge detection.
    """
    print(f"Processing {filename} ...")
    
    # Display the original image.
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Display histogram.
    show_histogram(image, f"Histogram: {filename}")
    
    # Display Fourier transform.
    show_fourier_transform(image, f"Fourier Transform: {filename}")
    
    # Display edge detection results.
    show_edge_detection(image, f"Edge Detection: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experimental Image Analysis: Histogram, Fourier Transform, and Edge Detection for the first image in the input directory."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    args = parser.parse_args()

    # Get a sorted list of image files.
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    
    if not image_files:
        print("No image files found in the specified directory.")
    else:
        # Process only the first image.
        first_image = image_files[0]
        img_path = os.path.join(args.input_dir, first_image)
        image = cv2.imread(img_path)
        if image is not None:
            process_image(image, first_image)
        else:
            print(f"Could not load image: {img_path}")