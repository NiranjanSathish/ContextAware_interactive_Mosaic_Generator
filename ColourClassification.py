import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import seaborn as sns

class ColorClassificationMethod(Enum):
    """Different methods for classifying cell colors."""
    DOMINANT_COLOR = "dominant_color"
    AVERAGE_COLOR = "average_color"
    HISTOGRAM_BINS = "histogram_bins"
    HSV_QUANTIZATION = "hsv_quantization"

@dataclass
class GridCell:
    """Represents a single grid cell with its properties."""
    row: int
    col: int
    average_color: np.ndarray
    dominant_color: np.ndarray
    brightness: float
    saturation: float
    hue: float
    color_category: int
    pixel_data: np.ndarray

class ImageGridAnalyzer:
    """
    Analyzes images by dividing them into grids and classifying each cell's color properties.
    Uses vectorized NumPy operations for high performance.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (32, 32),
                 classification_method: ColorClassificationMethod = ColorClassificationMethod.DOMINANT_COLOR,
                 n_color_categories: int = 16):
        """
        Initialize the grid analyzer.
        
        Args:
            grid_size: (rows, cols) for the grid division
            classification_method: Method to classify cell colors
            n_color_categories: Number of color categories for classification
        """
        self.grid_size = grid_size
        self.classification_method = classification_method
        self.n_color_categories = n_color_categories
        self.color_classifier = None
        self.category_colors = None
        
    def divide_image_into_grid(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Divide image into a grid using vectorized operations.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Grid of cells (grid_rows, grid_cols, tile_height, tile_width, channels)
            Tuple of (tile_height, tile_width)
        """
        h, w, c = image.shape
        grid_rows, grid_cols = self.grid_size
        
        # Calculate tile dimensions
        tile_h = h // grid_rows
        tile_w = w // grid_cols
        
        # Adjust image size to fit grid perfectly (crop if necessary)
        adjusted_h = tile_h * grid_rows
        adjusted_w = tile_w * grid_cols
        image = image[:adjusted_h, :adjusted_w]
        
        # Vectorized grid division using reshape and transpose
        # This is much faster than nested loops
        grid = image.reshape(grid_rows, tile_h, grid_cols, tile_w, c)
        grid = grid.transpose(0, 2, 1, 3, 4)  # (grid_rows, grid_cols, tile_h, tile_w, c)
        
        return grid, (tile_h, tile_w)
    
    def analyze_grid_colors_vectorized(self, grid: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze color properties of all grid cells using vectorized operations.
        
        Args:
            grid: Grid of cells (grid_rows, grid_cols, tile_h, tile_w, c)
            
        Returns:
            Dictionary containing vectorized analysis results
        """
        grid_rows, grid_cols, tile_h, tile_w, c = grid.shape
        
        # Reshape for vectorized operations: (total_cells, pixels_per_cell, channels)
        cells_flat = grid.reshape(grid_rows * grid_cols, tile_h * tile_w, c)
        
        # Calculate average colors for all cells at once
        average_colors = np.mean(cells_flat, axis=1)  # (total_cells, c)
        
        # Calculate dominant colors using vectorized approach
        dominant_colors = self._calculate_dominant_colors_vectorized(cells_flat)
        
        # Convert to HSV for additional analysis
        hsv_averages = self._rgb_to_hsv_vectorized(average_colors)
        
        # Calculate brightness (V in HSV)
        brightness = hsv_averages[:, 2]
        
        # Calculate saturation
        saturation = hsv_averages[:, 1]
        
        # Calculate hue
        hue = hsv_averages[:, 0]
        
        # Reshape results back to grid format
        results = {
            'average_colors': average_colors.reshape(grid_rows, grid_cols, c),
            'dominant_colors': dominant_colors.reshape(grid_rows, grid_cols, c),
            'brightness': brightness.reshape(grid_rows, grid_cols),
            'saturation': saturation.reshape(grid_rows, grid_cols),
            'hue': hue.reshape(grid_rows, grid_cols),
            'cells_data': grid  # Keep original cell data
        }
        
        return results
    
    def _calculate_dominant_colors_vectorized(self, cells_flat: np.ndarray) -> np.ndarray:
        """
        Calculate dominant color for each cell using vectorized operations.
        
        Args:
            cells_flat: Flattened cells (total_cells, pixels_per_cell, channels)
            
        Returns:
            Dominant colors for all cells (total_cells, channels)
        """
        import warnings
        
        total_cells, pixels_per_cell, c = cells_flat.shape
        dominant_colors = np.zeros((total_cells, c))
        
        # Process cells in batches for memory efficiency
        batch_size = 100
        for i in range(0, total_cells, batch_size):
            end_idx = min(i + batch_size, total_cells)
            batch = cells_flat[i:end_idx]
            
            for j, cell_pixels in enumerate(batch):
                # Check for color diversity first
                unique_pixels = np.unique(cell_pixels, axis=0)
                
                if len(unique_pixels) >= 3 and pixels_per_cell > 100:
                    # Use k-means for larger cells with sufficient color diversity
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
                        
                        kmeans = KMeans(n_clusters=min(3, len(unique_pixels)), 
                                      random_state=42, n_init=5)
                        labels = kmeans.fit_predict(cell_pixels)
                        # Get the most frequent cluster center
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        dominant_idx = unique_labels[np.argmax(counts)]
                        dominant_colors[i + j] = kmeans.cluster_centers_[dominant_idx]
                elif len(unique_pixels) >= 2:
                    # Use most frequent color for limited diversity
                    unique_colors, counts = np.unique(cell_pixels, axis=0, return_counts=True)
                    dominant_colors[i + j] = unique_colors[np.argmax(counts)]
                else:
                    # Use simple average for uniform cells
                    dominant_colors[i + j] = np.mean(cell_pixels, axis=0)
        
        return dominant_colors
    
    def _rgb_to_hsv_vectorized(self, rgb_colors: np.ndarray) -> np.ndarray:
        """
        Convert RGB colors to HSV using vectorized operations.
        
        Args:
            rgb_colors: RGB colors (N, 3)
            
        Returns:
            HSV colors (N, 3)
        """
        # Normalize to 0-1 range
        rgb_normalized = rgb_colors / 255.0
        
        # Create a dummy image for cv2 conversion
        dummy_img = rgb_normalized.reshape(-1, 1, 3).astype(np.float32)
        hsv_img = cv2.cvtColor(dummy_img, cv2.COLOR_RGB2HSV)
        hsv_colors = hsv_img.reshape(-1, 3)
        
        return hsv_colors
    
    def classify_colors(self, color_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Classify each grid cell into color categories.
        
        Args:
            color_data: Dictionary containing color analysis results
            
        Returns:
            Color categories for each grid cell (grid_rows, grid_cols)
        """
        import warnings
        
        if self.classification_method == ColorClassificationMethod.AVERAGE_COLOR:
            features = color_data['average_colors']
        elif self.classification_method == ColorClassificationMethod.DOMINANT_COLOR:
            features = color_data['dominant_colors']
        elif self.classification_method == ColorClassificationMethod.HSV_QUANTIZATION:
            # Combine HSV features
            h = color_data['hue']
            s = color_data['saturation']
            v = color_data['brightness']
            features = np.stack([h, s, v], axis=-1)
        else:
            features = color_data['average_colors']
        
        # Flatten for clustering
        grid_rows, grid_cols = features.shape[:2]
        features_flat = features.reshape(-1, features.shape[-1])
        
        # Check for sufficient diversity before clustering
        unique_features = np.unique(features_flat, axis=0)
        effective_clusters = min(self.n_color_categories, len(unique_features))
        
        if effective_clusters < 2:
            # Handle case with very limited color diversity
            print(f"Warning: Only {len(unique_features)} unique colors found. Using simple classification.")
            categories = np.zeros(len(features_flat), dtype=int)
            categories_grid = categories.reshape(grid_rows, grid_cols)
            self.category_colors = unique_features[:1] if len(unique_features) > 0 else np.array([[128, 128, 128]])
            return categories_grid
        
        # Fit color classifier with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
            
            self.color_classifier = KMeans(n_clusters=effective_clusters, 
                                         random_state=42, n_init=10)
            categories = self.color_classifier.fit_predict(features_flat)
        
        # Store category representative colors
        self.category_colors = self.color_classifier.cluster_centers_
        
        # Reshape back to grid
        categories_grid = categories.reshape(grid_rows, grid_cols)
        
        return categories_grid
    
    def apply_thresholding(self, color_data: Dict[str, np.ndarray], 
                          brightness_threshold: float = 0.5,
                          saturation_threshold: float = 0.3) -> Dict[str, np.ndarray]:
        """
        Apply thresholding to create binary masks for different criteria.
        
        Args:
            color_data: Color analysis results
            brightness_threshold: Threshold for bright/dark classification
            saturation_threshold: Threshold for saturated/desaturated classification
            
        Returns:
            Dictionary containing various threshold masks
        """
        brightness = color_data['brightness']
        saturation = color_data['saturation']
        
        # Normalize brightness and saturation to 0-1 range
        brightness_norm = brightness / 255.0 if brightness.max() > 1.0 else brightness
        saturation_norm = saturation / 255.0 if saturation.max() > 1.0 else saturation
        
        thresholds = {
            'bright_mask': brightness_norm > brightness_threshold,
            'dark_mask': brightness_norm <= brightness_threshold,
            'saturated_mask': saturation_norm > saturation_threshold,
            'desaturated_mask': saturation_norm <= saturation_threshold,
            'bright_saturated': (brightness_norm > brightness_threshold) & 
                              (saturation_norm > saturation_threshold),
            'dark_saturated': (brightness_norm <= brightness_threshold) & 
                            (saturation_norm > saturation_threshold)
        }
        
        return thresholds
    
    def analyze_image_complete(self, image: np.ndarray) -> Dict:
        """
        Complete analysis pipeline for an image.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Complete analysis results
        """
        print(f"Analyzing image with {self.grid_size[0]}x{self.grid_size[1]} grid...")
        
        # Step 1: Divide into grid
        grid, tile_size = self.divide_image_into_grid(image)
        print(f"Created grid with tile size: {tile_size}")
        
        # Step 2: Analyze colors (vectorized)
        color_data = self.analyze_grid_colors_vectorized(grid)
        print("Completed color analysis")
        
        # Step 3: Classify colors
        color_categories = self.classify_colors(color_data)
        print(f"Classified into {self.n_color_categories} color categories")
        
        # Step 4: Apply thresholding
        thresholds = self.apply_thresholding(color_data)
        print("Applied thresholding")
        
        # Combine all results
        results = {
            'grid': grid,
            'tile_size': tile_size,
            'color_data': color_data,
            'color_categories': color_categories,
            'thresholds': thresholds,
            'category_colors': self.category_colors
        }
        
        return results
    
    def visualize_analysis(self, results: Dict, original_image: np.ndarray):
        """
        Create comprehensive visualizations of the analysis results.
        
        Args:
            results: Analysis results from analyze_image_complete
            original_image: Original input image
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Average colors
        avg_colors = results['color_data']['average_colors'].astype(np.uint8)
        axes[0, 1].imshow(avg_colors)
        axes[0, 1].set_title('Average Colors per Cell')
        axes[0, 1].axis('off')
        
        # Dominant colors
        dom_colors = results['color_data']['dominant_colors'].astype(np.uint8)
        axes[0, 2].imshow(dom_colors)
        axes[0, 2].set_title('Dominant Colors per Cell')
        axes[0, 2].axis('off')
        
        # Color categories
        categories = results['color_categories']
        im_cat = axes[0, 3].imshow(categories, cmap='tab20')
        axes[0, 3].set_title(f'Color Categories ({self.n_color_categories} classes)')
        axes[0, 3].axis('off')
        plt.colorbar(im_cat, ax=axes[0, 3])
        
        # Brightness
        brightness = results['color_data']['brightness']
        im_bright = axes[1, 0].imshow(brightness, cmap='gray')
        axes[1, 0].set_title('Brightness Values')
        axes[1, 0].axis('off')
        plt.colorbar(im_bright, ax=axes[1, 0])
        
        # Saturation
        saturation = results['color_data']['saturation']
        im_sat = axes[1, 1].imshow(saturation, cmap='viridis')
        axes[1, 1].set_title('Saturation Values')
        axes[1, 1].axis('off')
        plt.colorbar(im_sat, ax=axes[1, 1])
        
        # Threshold: Bright areas
        axes[1, 2].imshow(results['thresholds']['bright_mask'], cmap='gray')
        axes[1, 2].set_title('Bright Areas (Threshold)')
        axes[1, 2].axis('off')
        
        # Threshold: Saturated areas
        axes[1, 3].imshow(results['thresholds']['saturated_mask'], cmap='gray')
        axes[1, 3].set_title('Saturated Areas (Threshold)')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show color category palette
        self._visualize_color_palette(results['category_colors'])
    
    def _visualize_color_palette(self, category_colors: np.ndarray):
        """
        Visualize the color category palette.
        
        Args:
            category_colors: Color palette (n_categories, channels)
        """
        if category_colors is None:
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        
        # Normalize colors if needed
        colors = category_colors.copy()
        if colors.max() > 1.0:
            colors = colors / 255.0
        
        # Create color swatches
        palette = colors.reshape(1, -1, 3)
        ax.imshow(palette, aspect='auto')
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(colors)))
        ax.set_xticklabels([f'Cat {i}' for i in range(len(colors))])
        ax.set_title(f'Color Category Palette ({len(colors)} categories)')
        ax.set_ylabel('Color Categories')
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_stats(self, results: Dict) -> Dict:
        """
        Calculate performance and analysis statistics.
        
        Args:
            results: Analysis results
            
        Returns:
            Dictionary containing statistics
        """
        grid_shape = results['color_categories'].shape
        total_cells = np.prod(grid_shape)
        
        # Color diversity
        unique_categories = len(np.unique(results['color_categories']))
        
        # Brightness statistics
        brightness = results['color_data']['brightness']
        
        # Saturation statistics
        saturation = results['color_data']['saturation']
        
        stats = {
            'grid_size': f"{grid_shape[0]}x{grid_shape[1]}",
            'total_cells': total_cells,
            'unique_color_categories': unique_categories,
            'category_utilization': unique_categories / self.n_color_categories,
            'avg_brightness': np.mean(brightness),
            'brightness_std': np.std(brightness),
            'avg_saturation': np.mean(saturation),
            'saturation_std': np.std(saturation),
            'bright_cells_percent': np.mean(results['thresholds']['bright_mask']) * 100,
            'saturated_cells_percent': np.mean(results['thresholds']['saturated_mask']) * 100
        }
        
        return stats

# Example usage and testing
def main():
    """
    Example usage of the ImageGridAnalyzer.
    """
    # Create sample test image (or load your own)
    def create_test_image():
        # Create a colorful test image with gradients and patterns
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add gradients
        for i in range(256):
            for j in range(256):
                img[i, j, 0] = i  # Red gradient
                img[i, j, 1] = j  # Green gradient
                img[i, j, 2] = (i + j) % 255  # Blue pattern
        
        return img
    
    # Initialize analyzer
    analyzer = ImageGridAnalyzer(
        grid_size=(32, 32),  # 32x32 grid = 1024 cells
        classification_method=ColorClassificationMethod.DOMINANT_COLOR,
        n_color_categories=16
    )
    
    # Create or load test image
    # test_image = create_test_image()
    
    # Or load real image (uncomment and modify path):
    test_image = cv2.imread('processed_quantized.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    print("Starting analysis...")
    
    # Analyze the image
    results = analyzer.analyze_image_complete(test_image)
    
    # Get performance statistics
    stats = analyzer.get_performance_stats(results)
    
    print("\n=== Analysis Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Visualize results
    # analyzer.visualize_analysis(results, test_image)
    
    return results, analyzer

if __name__ == "__main__":
    results, analyzer = main()