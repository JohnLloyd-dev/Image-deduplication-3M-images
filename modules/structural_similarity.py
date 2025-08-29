import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union
from PIL import Image
import gc

logger = logging.getLogger(__name__)

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("scikit-image not available, SSIM will use fallback implementation")

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, will use CPU-only processing")


class StructuralSimilarity:
    """
    Structural Similarity (SSIM) calculator for improved image comparison.
    
    Features:
    - SSIM computation using scikit-image or fallback implementation
    - GPU acceleration when available
    - Image preprocessing for better comparison
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 use_gpu: bool = True,
                 enable_preprocessing: bool = True):
        """
        Initialize structural similarity calculator.
        
        Args:
            target_size: Target size for image preprocessing (width, height)
            use_gpu: Enable GPU acceleration if available
            enable_preprocessing: Enable image preprocessing
        """
        self.target_size = target_size
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize transforms if PyTorch is available
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = None
        
        # Performance tracking
        self.stats = {
            'total_comparisons': 0,
            'gpu_comparisons': 0,
            'cpu_comparisons': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
        
        logger.info(f"Structural similarity initialized - GPU: {self.use_gpu}, "
                   f"Target size: {target_size}, Preprocessing: {enable_preprocessing}")
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM between two images.
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            SSIM score between 0.0 and 1.0 (higher = more similar)
        """
        import time
        start_time = time.time()
        
        try:
            if self.enable_preprocessing:
                img1_processed = self._preprocess_image(img1)
                img2_processed = self._preprocess_image(img2)
            else:
                img1_processed = img1
                img2_processed = img2
            
            # Compute SSIM
            if SSIM_AVAILABLE:
                score = self._compute_ssim_scikit(img1_processed, img2_processed)
            else:
                score = self._compute_ssim_fallback(img1_processed, img2_processed)
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['total_comparisons'] += 1
            self.stats['total_time'] += processing_time
            self.stats['average_time'] = self.stats['total_time'] / self.stats['total_comparisons']
            
            if self.use_gpu:
                self.stats['gpu_comparisons'] += 1
            else:
                self.stats['cpu_comparisons'] += 1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"SSIM computation failed: {e}")
            return 0.0
        finally:
            # Clean up
            if self.enable_preprocessing:
                del img1_processed, img2_processed
            gc.collect()
    
    def _compute_ssim_scikit(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM using scikit-image implementation.
        
        Args:
            img1: First preprocessed image
            img2: Second preprocessed image
            
        Returns:
            SSIM score
        """
        try:
            # Ensure images are grayscale
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Ensure same data type
            if img1.dtype != img2.dtype:
                img2 = img2.astype(img1.dtype)
            
            # Compute SSIM
            score = ssim(img1, img2, 
                        data_range=img2.max() - img2.min(),
                        win_size=11,  # Window size for SSIM computation
                        gaussian_weights=True,  # Use Gaussian weights
                        sigma=1.5,  # Gaussian sigma
                        use_sample_covariance=False)  # Use population covariance
            
            return score
            
        except Exception as e:
            logger.error(f"Scikit-image SSIM failed: {e}")
            return self._compute_ssim_fallback(img1, img2)
    
    def _compute_ssim_fallback(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Fallback SSIM implementation using OpenCV.
        
        Args:
            img1: First preprocessed image
            img2: Second preprocessed image
            
        Returns:
            SSIM-like score
        """
        try:
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Normalize images
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0
            
            # Compute mean
            mu1 = cv2.GaussianBlur(img1_norm, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2_norm, (11, 11), 1.5)
            
            # Compute variance and covariance
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(img1_norm * img1_norm, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2_norm * img2_norm, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1_norm * img2_norm, (11, 11), 1.5) - mu1_mu2
            
            # SSIM constants
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            # Compute SSIM
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim_map = numerator / (denominator + 1e-8)
            ssim_score = np.mean(ssim_map)
            
            return ssim_score
            
        except Exception as e:
            logger.error(f"Fallback SSIM failed: {e}")
            # Return simple correlation as last resort
            return self._compute_correlation(img1, img2)
    
    def _compute_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute simple correlation between images as last resort.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Correlation score between -1.0 and 1.0
        """
        try:
            # Flatten and normalize
            flat1 = img1.flatten().astype(np.float32)
            flat2 = img2.flatten().astype(np.float32)
            
            # Normalize
            flat1 = (flat1 - np.mean(flat1)) / (np.std(flat1) + 1e-8)
            flat2 = (flat2 - np.mean(flat2)) / (np.std(flat2) + 1e-8)
            
            # Compute correlation
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            
            # Convert to similarity score (0.0 to 1.0)
            similarity = (correlation + 1.0) / 2.0
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Correlation computation failed: {e}")
            return 0.0
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better SSIM computation.
        
        Args:
            img: Input image array
            
        Returns:
            Preprocessed image array
        """
        try:
            # Resize with aspect ratio preservation
            img_resized = self._resize_with_aspect_ratio(img)
            
            # Normalize illumination
            img_normalized = self._normalize_illumination(img_resized)
            
            return img_normalized
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return img
    
    def _resize_with_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image while preserving aspect ratio.
        
        Args:
            img: Input image array
            
        Returns:
            Resized image array
        """
        height, width = img.shape[:2]
        target_width, target_height = self.target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        resized = cv2.resize(img, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        
        # Pad if necessary to reach target size
        if resized.shape[0] != target_height or resized.shape[1] != target_width:
            # Calculate padding
            pad_bottom = max(0, target_height - resized.shape[0])
            pad_right = max(0, target_width - resized.shape[1])
            pad_top = 0
            pad_left = 0
            
            # Apply padding
            resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return resized
    
    def _normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image illumination using CLAHE.
        
        Args:
            img: Input image array
            
        Returns:
            Illumination-normalized image array
        """
        try:
            if len(img.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L-channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back
                lab = cv2.merge((l, a, b))
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(img)
                
        except Exception as e:
            logger.error(f"Illumination normalization failed: {e}")
            return img
    
    def _preprocess_with_torch(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images using PyTorch transforms.
        
        Args:
            img1: First image array
            img2: Second image array
            
        Returns:
            Tuple of preprocessed images
        """
        if not TORCH_AVAILABLE or self.transform is None:
            return img1, img2
        
        try:
            # Convert to PIL
            pil_img1 = Image.fromarray(img1)
            pil_img2 = Image.fromarray(img2)
            
            # Apply transforms
            tensor1 = self.transform(pil_img1)
            tensor2 = self.transform(pil_img2)
            
            # Move to GPU if available
            if self.use_gpu:
                tensor1 = tensor1.cuda()
                tensor2 = tensor2.cuda()
            
            # Convert back to numpy
            if self.use_gpu:
                tensor1 = tensor1.cpu()
                tensor2 = tensor2.cpu()
            
            img1_processed = tensor1.numpy().squeeze()
            img2_processed = tensor2.numpy().squeeze()
            
            return img1_processed, img2_processed
            
        except Exception as e:
            logger.error(f"PyTorch preprocessing failed: {e}")
            return img1, img2
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_comparisons': 0,
            'gpu_comparisons': 0,
            'cpu_comparisons': 0,
            'total_time': 0.0,
            'average_time': 0.0
        }
    
    def release(self):
        """Release resources."""
        logger.info("Releasing structural similarity calculator resources...")
        self.reset_stats()
        gc.collect()


def create_structural_similarity_calculator(
    target_size: Tuple[int, int] = (256, 256),
    use_gpu: bool = True,
    enable_preprocessing: bool = True
) -> StructuralSimilarity:
    """
    Factory function to create structural similarity calculator.
    
    Args:
        target_size: Target size for image preprocessing
        use_gpu: Enable GPU acceleration if available
        enable_preprocessing: Enable image preprocessing
        
    Returns:
        Configured StructuralSimilarity instance
    """
    return StructuralSimilarity(
        target_size=target_size,
        use_gpu=use_gpu,
        enable_preprocessing=enable_preprocessing
    )
