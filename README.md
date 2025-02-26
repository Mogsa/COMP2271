# Advanced Image Denoising

This repository contains tools for image denoising and processing, with a focus on maintaining image quality while removing noise.

## Advanced Denoiser

The `advanced_denoise.py` script provides a comprehensive solution for image denoising that combines multiple techniques:

- Non-Local Means (NLM) denoising for edge preservation
- Wavelet denoising for texture areas
- Total Variation (TV) denoising for removing artifacts
- Edge-aware bilateral filtering

The algorithm automatically analyzes the noise characteristics of each image and applies an optimized blend of techniques.

### Features

- Intelligent noise analysis to guide denoising strategy
- Dynamic parameter adjustment based on noise level
- Edge preservation focus
- Quality metrics calculation (PSNR, SSIM)
- Configurable strength and detail preservation

### Usage

Process a single image:

```bash
python advanced_denoise.py input_image.jpg -o output_image.jpg [options]
```

Options:
- `--strength` / `-s`: Denoising strength (0.0 to 1.0, default: 0.5)
- `--preserve-details` / `-p`: Enable detail preservation mode
- `--evaluate` / `-e`: Calculate and display quality metrics

## Batch Processing

The `batch_denoise.py` script allows processing multiple images at once:

```bash
python batch_denoise.py -i input_directory -o output_directory [options]
```

Options:
- `--strength` / `-s`: Denoising strength (0.0 to 1.0, default: 0.5)
- `--preserve-details` / `-p`: Enable detail preservation mode

## Requirements

Install dependencies with:

```bash
pip install opencv-python numpy scipy pillow scikit-image tqdm
```

## Examples

To denoise an image while preserving details:

```bash
python advanced_denoise.py driving_images/im060-rain.jpg -o cleaned_img.jpg --strength 0.6 --preserve-details
```

To process all images in the driving_images folder:

```bash
python batch_denoise.py -i driving_images -o Results --preserve-details
```