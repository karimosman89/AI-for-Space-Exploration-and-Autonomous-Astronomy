"""
Image processing utilities for satellite and astronomical imagery.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import albumentations as A


def load_image(
    image_path: str,
    color_mode: str = 'RGB'
) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        color_mode: Color mode ('RGB', 'BGR', 'GRAY')
        
    Returns:
        Image as numpy array
    """
    if color_mode == 'RGB':
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == 'BGR':
        image = cv2.imread(image_path)
    elif color_mode == 'GRAY':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"Unknown color mode: {color_mode}")
    
    return image


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image
    """
    # Resize
    image = cv2.resize(image, target_size[::-1])
    
    # Normalize
    if normalize:
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
    
    return image


def augment_data(
    image: np.ndarray,
    augmentation_type: str = 'training'
) -> np.ndarray:
    """
    Apply data augmentation to image.
    
    Args:
        image: Input image
        augmentation_type: Type of augmentation ('training', 'validation', 'test')
        
    Returns:
        Augmented image
    """
    if augmentation_type == 'training':
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.ElasticTransform(p=1),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
                A.RGBShift(p=1),
            ], p=0.5),
        ])
    else:
        transform = A.Compose([])
    
    augmented = transform(image=image)
    return augmented['image']


def normalize_image(
    image: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image
        method: Normalization method ('minmax', 'zscore', 'imagenet')
        
    Returns:
        Normalized image
    """
    if method == 'minmax':
        image = (image - image.min()) / (image.max() - image.min())
    elif method == 'zscore':
        image = (image - image.mean()) / image.std()
    elif method == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return image


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 2:  # Grayscale
        return clahe.apply(image)
    else:  # Color
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def remove_noise(
    image: np.ndarray,
    method: str = 'gaussian',
    **kwargs
) -> np.ndarray:
    """
    Remove noise from image.
    
    Args:
        image: Input image
        method: Denoising method ('gaussian', 'bilateral', 'nlmeans')
        **kwargs: Additional parameters for the denoising method
        
    Returns:
        Denoised image
    """
    if method == 'gaussian':
        ksize = kwargs.get('ksize', 5)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    elif method == 'nlmeans':
        h = kwargs.get('h', 10)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    else:
        raise ValueError(f"Unknown denoising method: {method}")
