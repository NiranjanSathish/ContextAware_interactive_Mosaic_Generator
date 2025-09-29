"""
ImagePreprocessor.py
Step 1: Image preprocessing for mosaic generation.
Handles image loading, resizing, cropping, and color quantization.
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional
from sklearn.cluster import MiniBatchKMeans
import warnings

class ImagePreprocessor:
    """
    Handles image preprocessing for mosaic generation.
    Features: Smart resizing, grid-perfect cropping, fast color quantization.
    """
    
    def __init__(self, target_resolution: Tuple[int, int] = (800, 600), 
                 grid_size: Tuple[int, int] = (20, 15)):
        """
        Initialize the preprocessor.
        
        Args:
            target_resolution: Target (width, height) for processed images
            grid_size: Grid dimensions (cols, rows) for mosaic
        """
        self.target_resolution = target_resolution
        self.grid_size = grid_size
        
        # Calculate tile size for perfect grid alignment
        self.tile_width = target_resolution[0] // grid_size[0]
        self.tile_height = target_resolution[1] // grid_size[1]
        
        # Adjust target resolution to fit grid perfectly
        self.adjusted_width = self.tile_width * grid_size[0]
        self.adjusted_height = self.tile_height * grid_size[1]
        
        print(f"Target resolution: {self.adjusted_width}x{self.adjusted_height}")
        print(f"Grid size: {grid_size[0]}x{grid_size[1]}")
        print(f"Tile size: {self.tile_width}x{self.tile_height}")
    
    def load_and_preprocess_image(self, image_path: str, 
                                 apply_quantization: bool = False,
                                 n_colors: int = 16) -> Optional[np.ndarray]:
        """
        Load and preprocess image from file path.
        
        Args:
            image_path: Path to the image file
            apply_quantization: Whether to apply color quantization
            n_colors: Number of colors for quantization
            
        Returns:
            Preprocessed image as numpy array (RGB) or None if failed
        """
        try:
            # Load and convert image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize and crop to fit grid
            processed_image = self._resize_and_crop(image)
            
            # Apply color quantization if requested
            if apply_quantization:
                processed_image = self._apply_color_quantization(processed_image, n_colors)
            
            return processed_image
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def preprocess_numpy_image(self, image: np.ndarray,
                              apply_quantization: bool = False,
                              n_colors: int = 16) -> Optional[np.ndarray]:
        """
        Preprocess numpy image array (for Gradio integration).
        
        Args:
            image: Input image as numpy array
            apply_quantization: Whether to apply color quantization
            n_colors: Number of colors for quantization
            
        Returns:
            Preprocessed image as numpy array (RGB) or None if failed
        """
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Image must be RGB format with shape (H, W, 3)")
            
            processed_image = image.copy()
            processed_image = self._resize_and_crop(processed_image)
            
            if apply_quantization:
                processed_image = self._apply_color_quantization(processed_image, n_colors)
            
            return processed_image
            
        except Exception as e:
            print(f"Error processing numpy image: {str(e)}")
            return None
    
    def _resize_and_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and crop image to fit target resolution while maintaining aspect ratio.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.adjusted_width, self.adjusted_height
        
        # Scale to fill target size
        scale_w = target_w / w
        scale_h = target_h / h
        scale = max(scale_w, scale_h)
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center crop to exact target size
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
        
        return cropped
    
    def _apply_color_quantization(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """
        Apply color quantization using Mini-Batch K-means for speed.
        3-4x faster than regular K-means with similar quality.
        """
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Adaptive batch size based on image size
        total_pixels = len(pixels)
        batch_size = min(max(total_pixels // 100, 1000), 10000)
        
        print(f"Applying Mini-Batch K-means quantization:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Batch size: {batch_size:,}")
        print(f"  Target colors: {n_colors}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_colors, 
                batch_size=batch_size,
                random_state=42, 
                n_init=3,
                max_iter=100
            )
            labels = kmeans.fit_predict(pixels)
        
        # Replace pixels with cluster centers
        quantized_pixels = kmeans.cluster_centers_[labels]
        quantized_image = quantized_pixels.reshape(h, w, c).astype(np.uint8)
        
        return quantized_image
    
    def save_preprocessed_image(self, image: np.ndarray, output_path: str):
        """Save preprocessed image to disk."""
        try:
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, quality=95)
            print(f"Saved preprocessed image to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {str(e)}")

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor(
        target_resolution=(1280, 1280),
        grid_size=(32, 32)
    )
    
    test_image = "EmmaPotrait.jpg"
    
    if os.path.exists(test_image):
        processed = preprocessor.load_and_preprocess_image(
            test_image, 
            apply_quantization=True,
            n_colors=8
        )
        
        if processed is not None:
            preprocessor.save_preprocessed_image(processed, "processed_quantized.jpg")
            print("Image preprocessing completed!")
    else:
        print(f"Test image not found: {test_image}")