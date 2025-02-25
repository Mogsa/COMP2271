import cv2
import numpy as np
from bm3d import bm3d, BM3DProfile

# Load the image (BGR format) and check if loaded.
image = cv2.imread('driving_images/im001-snow.jpg')
if image is None:
    raise ValueError("Image not found. Check your path.")

# Convert BGR image to RGB.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize image to [0, 1].
image_rgb = image_rgb.astype(np.float32) / 255.0

# Define noise standard deviation (sigma).
sigma = 0.1  # Adjust as needed.

# Apply BM3D on each channel separately.
denoised_channels = []
for channel in range(3):
    denoised_channel = bm3d(image_rgb[..., channel], sigma_psd=sigma, profile=BM3DProfile())
    denoised_channels.append(denoised_channel)

# Stack the denoised channels back together.
denoised_rgb = np.stack(denoised_channels, axis=2)

# Convert back to 8-bit and then to BGR for saving with OpenCV.
denoised_rgb = np.clip(denoised_rgb * 255, 0, 255).astype(np.uint8)
denoised_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)

# Save the denoised image.
cv2.imwrite('denoised_image.jpg', denoised_bgr)

# ---------------------------
# Intermediate Purple Noise Filter
# ---------------------------
# Convert the denoised BGR image to HSV.
hsv_denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)

# Define the HSV range for purple noise.
# Adjust these thresholds as needed to best capture your purple noise.
lower_purple = np.array([140, 40, 40])
upper_purple = np.array([179, 255, 255])
mask_purple = cv2.inRange(hsv_denoised, lower_purple, upper_purple)

# (Optional) Clean up the mask with morphological operations.
kernel = np.ones((3, 3), np.uint8)
mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)

# Create an image that contains only purple noise.
purple_noise = cv2.bitwise_and(denoised_bgr, denoised_bgr, mask=mask_purple)

# ---------------------------
# Display the images side by side
# ---------------------------
# Concatenate original and denoised for comparison.
comparison = np.hstack((image, denoised_bgr))
cv2.imshow("Original vs Denoised", comparison)

# Also show the isolated purple noise.
cv2.imshow("Purple Noise Filter", purple_noise)

cv2.waitKey(0)
cv2.destroyAllWindows()