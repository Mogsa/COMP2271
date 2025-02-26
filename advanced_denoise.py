import cv2
import numpy as np
import os
from scipy import ndimage
from PIL import Image, ImageEnhance
import argparse
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle, denoise_wavelet
from skimage.util import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class AdvancedDenoiser:
    def __init__(self, preserve_details=True, strength=0.5):
        """
        Initialize the denoiser with configurable parameters
        
        Args:
            preserve_details: Whether to focus on preserving fine details
            strength: Strength of denoising (0.0 to 1.0)
        """
        self.preserve_details = preserve_details
        self.strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        
        # Parameters tuned based on strength
        self.nlm_h = 5 + (self.strength * 6)  # NLM filter strength
        self.tv_weight = 0.1 + (self.strength * 0.2)  # Total Variation weight
        self.bilateral_d = 7  # Bilateral filter diameter
        self.bilateral_sigma_color = 30 * self.strength  # Controls filter strength
        self.bilateral_sigma_space = 5 + (self.strength * 5)  # Spatial sigma
    
    def _convert_to_float(self, image):
        """Convert image to float format for processing"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    
    def _convert_from_float(self, image):
        """Convert float image back to uint8"""
        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)
    
    def _analyze_noise(self, image):
        """Analyze noise characteristics to guide denoising strategy"""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Estimate noise level (newer versions of scikit-image use channel_axis instead of multichannel)
        noise_sigma = np.mean(estimate_sigma(gray))
        
        # Analyze local variance to identify noisy regions
        # Create a kernel for local variance calculation
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # Calculate local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Calculate local variance 
        local_var = cv2.filter2D(np.square(gray.astype(np.float32)), -1, kernel) - np.square(local_mean)
        high_var_ratio = np.sum(local_var > np.mean(local_var) * 1.5) / gray.size
        
        # Analyze edge content
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        return {
            'noise_level': noise_sigma,
            'high_var_ratio': high_var_ratio,
            'edge_ratio': edge_ratio
        }
    
    def _edge_aware_smoothing(self, image):
        """Apply edge-preserving smoothing"""
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        return smoothed
    
    def _detail_enhancing(self, image):
        """Enhance details in the image"""
        if self.preserve_details:
            # Convert to PIL for enhancing
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(pil_img)
            enhanced = enhancer.enhance(1.2)
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        return image
    
    def _apply_nlm(self, image, noise_info):
        """Apply Non-Local Means denoising"""
        # Use OpenCV's NLM implementation which is faster
        if len(image.shape) == 3:
            # For color images
            h = self.nlm_h * max(0.1, noise_info['noise_level'])
            
            # OpenCV NLM parameters
            h_luminance = h
            h_color = h * 1.5
            template_size = 7
            search_size = 21
            
            # Apply NLM with OpenCV (faster than scikit-image for color)
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h_luminance,
                h_color,
                template_size,
                search_size
            )
            return denoised
        else:
            # For grayscale images
            # Fallback to scikit-image implementation which gives better quality for grayscale
            float_img = img_as_float(image)
            denoised = denoise_nl_means(
                float_img, 
                h=self.nlm_h * noise_info['noise_level'],
                patch_size=7, 
                patch_distance=6,
                fast_mode=True
            )
            return img_as_ubyte(denoised)
    
    def _apply_wavelet(self, image, noise_info):
        """Apply Wavelet denoising"""
        # Ensure image is properly scaled between 0 and 1 for processing
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.copy()
            
        # Apply wavelet denoising - works well for Gaussian noise
        if len(image.shape) == 3:
            denoised = denoise_wavelet(
                image_float,
                channel_axis=2,  # For color images, channel axis is 2
                convert2ycbcr=True,
                method='BayesShrink',
                mode='soft',
                rescale_sigma=True
            )
        else:
            denoised = denoise_wavelet(
                image_float,
                channel_axis=None,  # For grayscale images
                method='BayesShrink',
                mode='soft',
                rescale_sigma=True
            )
        
        # Convert back to uint8
        return (denoised * 255).astype(np.uint8)
    
    def _apply_tv(self, image, noise_info):
        """Apply Total Variation denoising"""
        # Ensure image is properly scaled between 0 and 1 for processing
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.copy()
            
        # Apply TV denoising - good for salt and pepper noise
        if len(image.shape) == 3:
            denoised = denoise_tv_chambolle(
                image_float, 
                weight=self.tv_weight,
                channel_axis=2  # For color images, channel axis is 2
            )
        else:
            denoised = denoise_tv_chambolle(
                image_float, 
                weight=self.tv_weight,
                channel_axis=None  # For grayscale images
            )
            
        # Convert back to uint8
        return (denoised * 255).astype(np.uint8)
    
    def _blend_results(self, original, nlm, wavelet, tv, edge_aware, noise_info):
        """Intelligently blend results from different methods"""
        # Determine weights based on noise analysis
        nlm_weight = 0.4
        wavelet_weight = 0.3
        tv_weight = 0.2
        edge_weight = 0.1
        
        # Adjust weights based on noise characteristics
        if noise_info['noise_level'] > 0.15:
            # Higher noise - give more weight to stronger denoising methods
            nlm_weight = 0.5
            wavelet_weight = 0.3
            tv_weight = 0.15
            edge_weight = 0.05
        
        if noise_info['edge_ratio'] > 0.1:
            # More edges - prioritize edge preservation
            nlm_weight = 0.45
            wavelet_weight = 0.2
            tv_weight = 0.1
            edge_weight = 0.25
        
        # Normalize weights
        total = nlm_weight + wavelet_weight + tv_weight + edge_weight
        nlm_weight /= total
        wavelet_weight /= total
        tv_weight /= total
        edge_weight /= total
        
        # Blend the results
        blended = cv2.addWeighted(nlm, nlm_weight, wavelet, wavelet_weight, 0)
        blended = cv2.addWeighted(blended, 1.0, tv, tv_weight, 0)
        blended = cv2.addWeighted(blended, 1.0, edge_aware, edge_weight, 0)
        
        return blended
    
    def evaluate_quality(self, original, denoised):
        """Evaluate denoising quality"""
        # Convert to same format
        if original.dtype != denoised.dtype:
            if original.dtype == np.uint8:
                original = img_as_float(original)
                denoised = img_as_float(denoised)
            else:
                original = img_as_ubyte(original)
                denoised = img_as_ubyte(denoised)
        
        # Calculate PSNR (higher is better)
        psnr_value = psnr(original, denoised)
        
        # Calculate SSIM (higher is better)
        if len(original.shape) == 3:
            ssim_value = ssim(original, denoised, channel_axis=2)
        else:
            ssim_value = ssim(original, denoised)
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value
        }
    
    def denoise(self, image, return_metrics=False, original=None):
        """
        Main denoising method that applies a customized combination of algorithms
        
        Args:
            image: Input image (numpy array)
            return_metrics: Whether to return quality metrics
            original: Original image for metrics calculation (if different from input)
            
        Returns:
            Denoised image, and optionally quality metrics
        """
        # Clone the image to avoid modifying the original
        image_copy = image.copy()
        
        # Analyze noise characteristics
        noise_info = self._analyze_noise(image_copy)
        
        # Apply edge-aware smoothing
        edge_aware = self._edge_aware_smoothing(image_copy)
        
        # Apply different denoising methods
        nlm_result = self._apply_nlm(image_copy, noise_info)
        wavelet_result = self._apply_wavelet(image_copy, noise_info)
        tv_result = self._apply_tv(image_copy, noise_info)
        
        # Blend the results
        denoised = self._blend_results(
            image_copy, nlm_result, wavelet_result, tv_result, edge_aware, noise_info
        )
        
        # Enhance details if needed
        if self.preserve_details:
            denoised = self._detail_enhancing(denoised)
        
        # Calculate metrics if requested
        if return_metrics:
            # If original is not provided, use the input image as reference
            reference = original if original is not None else image_copy
            metrics = self.evaluate_quality(reference, denoised)
            return denoised, metrics
        
        return denoised


def main():
    parser = argparse.ArgumentParser(description='Advanced Image Denoising')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--strength', '-s', type=float, default=0.5, 
                        help='Denoising strength (0.0 to 1.0)')
    parser.add_argument('--preserve-details', '-p', action='store_true', 
                        help='Preserve fine details')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='Evaluate and display quality metrics')
    
    args = parser.parse_args()
    
    # Load input image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image {args.input}")
        return
    
    # Create denoiser instance
    denoiser = AdvancedDenoiser(
        preserve_details=args.preserve_details,
        strength=args.strength
    )
    
    # Apply denoising
    if args.evaluate:
        denoised, metrics = denoiser.denoise(image, return_metrics=True)
        print(f"Denoising Results:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    else:
        denoised = denoiser.denoise(image)
    
    # Save or show the result
    if args.output:
        cv2.imwrite(args.output, denoised)
        print(f"Denoised image saved to {args.output}")
    else:
        # Display images
        cv2.imshow('Original', image)
        cv2.imshow('Denoised', denoised)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()