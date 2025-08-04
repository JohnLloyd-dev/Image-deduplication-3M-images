import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List, Union
import kornia.feature as KF
import timm
from PIL import Image
import pywt
import logging
import kornia as K
import os
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

class FeatureExtractor:
    # Precomputed mean/std for EfficientNet
    EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
    EFFICIENTNET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize EfficientNet-B7 with proper normalization
        self.global_model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
        
        # Initialize kornia feature extractors
        self.local_feature = KF.KeyNetAffNetHardNet(5000).to(self.device)  # Increase keypoints
        self.local_feature.eval()
        
        # Initialize LoFTR for geometric verification
        self.loftr = KF.LoFTR(pretrained="outdoor").to(self.device)
        self.loftr.eval()
        
    def decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image bytes to numpy array."""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of images for GPU processing."""
        try:
            # Resize all images to same size
            resized = [cv2.resize(img, (224, 224)) for img in images]
            
            # Convert to torch tensors
            tensors = [torch.from_numpy(img).float() for img in resized]
            
            # Stack into batch
            batch = torch.stack(tensors)
            
            # Normalize
            batch = batch / 255.0
            
            # Move to GPU
            batch = batch.to(self.device)
            
            return batch
        except Exception as e:
            logger.error(f"Error preprocessing batch: {str(e)}")
            raise
    
    def extract_features_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Extract features from a batch of images."""
        try:
            # Preprocess batch
            batch = self.preprocess_batch(images)
            
            # Extract features
            with torch.no_grad():
                # Color histogram
                color_hist = self._extract_color_histogram_batch(batch)
                
                # Wavelet hash
                wavelet_hash = self._extract_wavelet_hash_batch(batch)
                
                # Edge features
                edge_features = self._extract_edge_features_batch(batch)
            
            # Convert to list of dictionaries
            features = []
            for i in range(len(images)):
                features.append({
                    'color_histogram': color_hist[i].cpu().numpy(),
                    'wavelet_hash': wavelet_hash[i].cpu().numpy(),
                    'edge_features': edge_features[i].cpu().numpy()
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features batch: {str(e)}")
            raise
    
    def _extract_color_histogram_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract color histograms for a batch of images."""
        try:
            # Convert to HSV
            hsv = batch.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Calculate histograms
            hist = torch.zeros(batch.size(0), 256, 3, device=self.device)
            for i in range(3):
                hist[:, :, i] = torch.histc(hsv[:, i], bins=256, min=0, max=1)
            
            # Normalize
            hist = hist / hist.sum(dim=1, keepdim=True)
            
            return hist
            
        except Exception as e:
            logger.error(f"Error extracting color histogram batch: {str(e)}")
            raise
    
    def _extract_wavelet_hash_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract wavelet hashes for a batch of images."""
        try:
            # Convert to grayscale
            gray = 0.299 * batch[:, :, :, 0] + 0.587 * batch[:, :, :, 1] + 0.114 * batch[:, :, :, 2]
            
            # Apply DWT
            coeffs = F.avg_pool2d(gray.unsqueeze(1), kernel_size=2, stride=2)
            
            # Calculate hash
            hash = (coeffs > coeffs.mean()).float()
            
            return hash
            
        except Exception as e:
            logger.error(f"Error extracting wavelet hash batch: {str(e)}")
            raise
    
    def _extract_edge_features_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract edge features for a batch of images."""
        try:
            # Convert to grayscale
            gray = 0.299 * batch[:, :, :, 0] + 0.587 * batch[:, :, :, 1] + 0.114 * batch[:, :, :, 2]
            
            # Apply Sobel
            sobel_x = F.conv2d(gray.unsqueeze(1), 
                              torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                         dtype=torch.float32, device=self.device).view(1, 1, 3, 3))
            sobel_y = F.conv2d(gray.unsqueeze(1),
                              torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                         dtype=torch.float32, device=self.device).view(1, 1, 3, 3))
            
            # Calculate magnitude
            magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
            
            # Calculate histogram
            hist = torch.histc(magnitude, bins=32, min=0, max=1)
            hist = hist / hist.sum()
            
            return hist
            
        except Exception as e:
            logger.error(f"Error extracting edge features batch: {str(e)}")
            raise
        
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract all features from an image array.
        
        Args:
            image: Input image array (BGR format from OpenCV)
            
        Returns:
            Dictionary containing features
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features
            global_features = self.extract_global_features(image_rgb)
            local_features = self.extract_local_features(image_rgb)
            wavelet_hash = self.compute_wavelet_hash(image)
            
            return {
                'global': global_features,
                'local': local_features,
                'wavelet': wavelet_hash
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'global': None,
                'local': None,
                'wavelet': None
            }
    
    def extract_global_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract global features with proper normalization.
        """
        try:
            # Convert to tensor with normalization
            transform = K.augmentation.Normalize(
                mean=torch.tensor(self.EFFICIENTNET_MEAN), 
                std=torch.tensor(self.EFFICIENTNET_STD)
            )
            
            # Resize and convert to tensor
            img_tensor = K.image_to_tensor(
                cv2.resize(image, (600, 600)), 
                False
            ).float() / 255.0
            
            # Apply normalization
            img_tensor = transform(img_tensor.to(self.device))
            
            # Extract features with mixed precision
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                features = self.global_model.forward_features(img_tensor)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.squeeze(-1).squeeze(-1)
                features = F.normalize(features, p=2, dim=1)
                
            return features.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            logger.error(f"Global feature error: {e}")
            return None
    
    def extract_local_features(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract local features with consistent processing.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Resize and convert to tensor
            img_tensor = K.image_to_tensor(
                cv2.resize(gray, (800, 800)),  # Higher resolution for better keypoints
                False
            ).float() / 255.0
            img_tensor = img_tensor.to(self.device)
            
            # Extract features with mixed precision
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                keypoints, _, descriptors = self.local_feature(img_tensor)
                
                # Convert to numpy
                keypoints = keypoints.cpu().numpy().squeeze(0)
                descriptors = descriptors.cpu().numpy().squeeze(0)
                
                # Handle single descriptor case
                if descriptors.ndim == 1:
                    descriptors = descriptors.reshape(1, -1)
                
            return {
                'keypoints': keypoints,
                'descriptors': descriptors
            }
            
        except Exception as e:
            logger.error(f"Local feature error: {e}")
            return None
    
    def compute_wavelet_hash(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhanced wavelet hash with adaptive thresholding.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Resize to target size
            resized = cv2.resize(gray, (64, 64))
            
            # Multi-level wavelet decomposition
            coeffs = pywt.wavedec2(resized, 'bior6.8', level=4)  # Better wavelet for images
            
            # Combine approximation and detail coefficients
            combined = []
            for i, coeff in enumerate(coeffs):
                if i == 0:
                    # Approximation coefficients
                    combined.append(coeff.flatten())
                else:
                    # Detail coefficients (horizontal, vertical, diagonal)
                    for c in coeff:
                        combined.append(c.flatten())
            
            combined = np.concatenate(combined)
            
            # Adaptive binarization
            hash_array = (combined > np.percentile(combined, 60)).astype(np.uint8)
            
            return hash_array
            
        except Exception as e:
            logger.error(f"Wavelet error: {e}")
            return None
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> Optional[np.ndarray]:
        """
        Optimized feature matching with symmetry check.
        """
        try:
            # Validate inputs
            if desc1 is None or desc2 is None or desc1.size == 0 or desc2.size == 0:
                return None
                
            # Ensure proper shape and type
            desc1 = desc1.astype(np.float32).reshape(-1, 128)
            desc2 = desc2.astype(np.float32).reshape(-1, 128)
            
            # Normalize descriptors
            desc1 /= np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-7
            desc2 /= np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-7
            
            # Create matcher optimized for HardNet
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=4),  # Fewer trees for faster matching
                dict(checks=128)             # More checks for accuracy
            )
            
            # Forward matching
            matches_forward = matcher.knnMatch(desc1, desc2, k=2)
            # Backward matching for symmetry
            matches_backward = matcher.knnMatch(desc2, desc1, k=2)
            
            # Ratio test with symmetry check
            good_matches = []
            forward_matches = {}
            
            # Forward pass
            for m, n in matches_forward:
                if m.distance < 0.8 * n.distance:
                    forward_matches[m.queryIdx] = m.trainIdx
                    
            # Backward pass for symmetry
            for m, n in matches_backward:
                if m.distance < 0.8 * n.distance:
                    if m.trainIdx in forward_matches and forward_matches[m.trainIdx] == m.queryIdx:
                        good_matches.append(cv2.DMatch(m.trainIdx, m.queryIdx, m.distance))
            
            if len(good_matches) < 8:
                return None
                
            return np.array([[m.queryIdx, m.trainIdx] for m in good_matches])
            
        except Exception as e:
            logger.error(f"Matching failed: {e}")
            return None
    
    def geometric_verification(self, img1: np.ndarray, img2: np.ndarray, 
                              kps1: np.ndarray, kps2: np.ndarray,
                              matches: np.ndarray) -> Tuple[float, int]:
        """
        In-memory geometric verification with adaptive thresholding.
        
        Returns:
            Tuple (inlier_ratio, num_inliers)
        """
        try:
            if matches is None or len(matches) < 8:
                return 0.0, 0
                
            # Convert keypoints to coordinates
            pts1 = kps1[matches[:, 0]]
            pts2 = kps2[matches[:, 1]]
            
            # Estimate fundamental matrix
            F, mask = cv2.findFundamentalMat(
                pts1, pts2, cv2.FM_RANSAC, 
                ransacReprojThreshold=1.0, 
                confidence=0.99
            )
            
            if mask is None:
                return 0.0, 0
                
            inliers = mask.ravel().astype(bool)
            num_inliers = np.sum(inliers)
            inlier_ratio = num_inliers / len(matches)
            
            return inlier_ratio, num_inliers
            
        except Exception as e:
            logger.error(f"Geometric verification failed: {e}")
            return 0.0, 0
    
    def batch_extract(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Batch feature extraction for efficiency.
        """
        features = []
        for img in tqdm(images, desc="Extracting features"):
            features.append(self.extract_features(img))
        return features
    
    def release(self):
        """Release resources with thorough cleanup."""
        try:
            del self.global_model
            del self.local_feature
            del self.loftr
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def optimize_for_2060(self):
        """Optimize model for RTX 2060 GPU with 8GB VRAM."""
        try:
            if hasattr(self, 'global_model') and self.global_model is not None:
                # Convert model to half precision
                self.global_model = self.global_model.half()
                # Move to GPU if not already there
                if self.device == "cuda":
                    self.global_model = self.global_model.cuda()
                logger.info("Model optimized for RTX 2060")
            else:
                logger.warning("No model available for optimization")
        except Exception as e:
            logger.error(f"Failed to optimize model for RTX 2060: {e}") 