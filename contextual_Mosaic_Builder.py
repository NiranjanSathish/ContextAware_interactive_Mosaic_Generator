"""
contextual_mosaic.py
Advanced contextual mosaic generator with face detection and rotation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import pickle
import warnings
import matplotlib.pyplot as plt

@dataclass
class ImageContext:
    """Stores contextual analysis results."""
    has_faces: bool
    face_regions: List[Tuple[int, int, int, int]]
    is_portrait: bool
    is_landscape: bool
    dominant_colors: np.ndarray
    brightness_map: np.ndarray
    edge_density_map: np.ndarray
    content_complexity: float

class ContextualMosaicGenerator:
    """Advanced mosaic generator with contextual awareness."""
    
    def __init__(self, cache_file: str, tile_folder: str = "extracted_images", 
                 tile_size: Tuple[int, int] = (32, 32),
                 colour_bins: int = 8,
                 enable_rotation: bool = True):
        self.cache_file = cache_file
        self.tile_folder = tile_folder
        self.target_tile_size = tile_size
        self.target_colour_bins = colour_bins
        self.target_enable_rotation = enable_rotation
        
        # Data containers
        self.tile_images = None
        self.tile_colours = None
        self.tile_names = None
        self.colour_palette = None
        self.colour_groups = None
        self.colour_indices = None
        self.tile_size = None
        self.enable_rotation = False
        
        # Setup face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("Warning: Face detection not available")
            self.face_cascade = None
        
        self._load_or_build_cache()
    
    def _load_or_build_cache(self):
        """Load existing cache or build new one."""
        if Path(self.cache_file).exists():
            try:
                if self._load_and_validate_cache():
                    print("Cache loaded successfully")
                    return
            except Exception as e:
                print(f"Cache error: {e}")
        
        print("Building new cache...")
        self._build_cache_automatically()
    
    def _load_and_validate_cache(self) -> bool:
        """Load and validate cache parameters."""
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
        
        required_keys = ['tile_images', 'tile_colours', 'tile_names', 'colour_palette', 'colour_groups', 'colour_indices']
        if not all(key in data for key in required_keys):
            return False
        
        # Validate parameters
        if (data.get('tile_size') != self.target_tile_size or
            data.get('colour_bins') != self.target_colour_bins or
            data.get('enable_rotation') != self.target_enable_rotation):
            print("Cache parameters don't match")
            return False
        
        # Load data
        self.tile_images = data['tile_images']
        self.tile_colours = data['tile_colours']
        self.tile_names = data['tile_names']
        self.colour_palette = data['colour_palette']
        self.colour_groups = data['colour_groups']
        self.colour_indices = data['colour_indices']
        self.tile_size = data['tile_size']
        self.enable_rotation = data.get('enable_rotation', False)
        
        print(f"Loaded: {len(self.tile_images)} tiles")
        return True
    
    def _build_cache_automatically(self):
        """Build cache using cache builder."""
        try:
            from Cache_Builder import TileCacheBuilder
            
            print("Auto-building cache...")
            builder = TileCacheBuilder(
                tile_folder=self.tile_folder,
                tile_size=self.target_tile_size,
                colour_bins=self.target_colour_bins,
                enable_rotation=self.target_enable_rotation
            )
            
            success = builder.build_cache(self.cache_file, force_rebuild=True)
            if not success:
                raise RuntimeError("Cache builder failed")
            
            if not self._load_and_validate_cache():
                raise RuntimeError("Failed to load new cache")
            
            print("Cache built successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Auto-cache failed: {e}")
    
    def analyze_image_context(self, image: np.ndarray) -> ImageContext:
        """Analyze image content for contextual tile selection."""
        print("Analyzing context...")
        
        # Face detection
        faces = []
        has_faces = False
        if self.face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = [(x, y, w, h) for (x, y, w, h) in detected_faces]
            has_faces = len(faces) > 0
        
        # Scene classification
        aspect_ratio = image.shape[1] / image.shape[0]
        is_portrait = aspect_ratio < 1.2 and has_faces
        is_landscape = aspect_ratio > 1.5
        
        # Dominant colors
        pixels = image.reshape(-1, 3)
        sample_size = min(10000, len(pixels))
        sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=1000)
            kmeans.fit(sampled_pixels)
            dominant_colors = kmeans.cluster_centers_
        
        # Analysis maps
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness_map = cv2.resize(gray, (32, 32)) / 255.0
        edges = cv2.Canny(gray, 50, 150)
        edge_density_map = cv2.resize(edges, (32, 32)) / 255.0
        content_complexity = np.mean(edge_density_map)
        
        context = ImageContext(
            has_faces=has_faces,
            face_regions=faces,
            is_portrait=is_portrait,
            is_landscape=is_landscape,
            dominant_colors=dominant_colors,
            brightness_map=brightness_map,
            edge_density_map=edge_density_map,
            content_complexity=content_complexity
        )
        
        print(f"Context: {len(faces)} faces, {'Portrait' if is_portrait else 'Landscape' if is_landscape else 'General'}")
        return context
    
    def get_adaptive_tile_strategy(self, context: ImageContext, grid_pos: Tuple[int, int], grid_size: Tuple[int, int]) -> Dict:
        """Get adaptive strategy based on content and position."""
        row, col = grid_pos
        grid_rows, grid_cols = grid_size
        
        strategy = {
            'precision_factor': 1.0,
            'diversity_factor': 0.1,
            'allow_rotation': self.enable_rotation,
            'search_candidates': 5
        }
        
        # Face area adjustments
        if context.has_faces:
            for face_x, face_y, face_w, face_h in context.face_regions:
                face_grid_x = int((face_x / 1024) * grid_cols)
                face_grid_y = int((face_y / 1024) * grid_rows)
                face_grid_w = max(1, int((face_w / 1024) * grid_cols))
                face_grid_h = max(1, int((face_h / 1024) * grid_rows))
                
                if (face_grid_x <= col <= face_grid_x + face_grid_w and
                    face_grid_y <= row <= face_grid_y + face_grid_h):
                    strategy['precision_factor'] = 2.0
                    strategy['diversity_factor'] = 0.05
                    strategy['allow_rotation'] = False
                    strategy['search_candidates'] = 10
                    break
        
        # Edge adjustments
        edge_row = min(int((row / grid_rows) * 32), 31)
        edge_col = min(int((col / grid_cols) * 32), 31)
        edge_density = context.edge_density_map[edge_row, edge_col]
        
        if edge_density > 0.3:
            strategy['precision_factor'] *= 1.5
            strategy['search_candidates'] = min(strategy['search_candidates'] * 2, 15)
        
        return strategy
    
    def find_contextual_best_tile(self, target_colour: np.ndarray, context: ImageContext,
                                 grid_pos: Tuple[int, int], grid_size: Tuple[int, int],
                                 usage_count: Dict = None) -> int:
        """Find best tile using contextual analysis."""
        strategy = self.get_adaptive_tile_strategy(context, grid_pos, grid_size)
        
        # Find color bin
        distances = euclidean_distances(target_colour.reshape(1, -1), self.colour_palette)[0]
        target_bin = np.argmin(distances)
        
        if target_bin in self.colour_indices:
            index, tile_indices = self.colour_indices[target_bin]
            
            n_candidates = min(strategy['search_candidates'], len(tile_indices))
            _, indices = index.kneighbors(target_colour.reshape(1, -1), n_neighbors=n_candidates)
            
            best_score = float('inf')
            best_tile_idx = tile_indices[indices[0][0]]
            
            for local_idx in indices[0]:
                global_tile_idx = tile_indices[local_idx]
                tile_name = self.tile_names[global_tile_idx]
                
                # Skip rotation in face areas
                if not strategy['allow_rotation'] and '_rot' in tile_name:
                    continue
                
                # Calculate score
                tile_colour = self.tile_colours[global_tile_idx]
                color_distance = np.linalg.norm(target_colour - tile_colour) * strategy['precision_factor']
                
                usage_penalty = 0
                if usage_count is not None:
                    usage_penalty = usage_count.get(tile_name, 0) * strategy['diversity_factor']
                
                total_score = color_distance + usage_penalty
                
                if total_score < best_score:
                    best_score = total_score
                    best_tile_idx = global_tile_idx
            
            return best_tile_idx
        
        return 0
    
    def create_contextual_mosaic(self, image: np.ndarray, grid_size: Tuple[int, int],
                               diversity_factor: float = 0.1) -> Tuple[np.ndarray, ImageContext]:
        """Create mosaic with contextual awareness."""
        print("Creating contextual mosaic...")
        
        # Analyze context
        context = self.analyze_image_context(image)
        
        # Setup grid
        h, w = image.shape[:2]
        grid_rows, grid_cols = grid_size
        cell_h = h // grid_rows
        cell_w = w // grid_cols
        
        # Initialize mosaic
        tile_h, tile_w = self.tile_size[1], self.tile_size[0]
        mosaic = np.zeros((grid_rows * tile_h, grid_cols * tile_w, 3), dtype=np.uint8)
        
        usage_count = {} if diversity_factor > 0 else None
        
        print(f"Building {grid_rows}x{grid_cols} mosaic...")
        
        for i in tqdm(range(grid_rows), desc="Creating mosaic"):
            for j in range(grid_cols):
                # Get cell color
                start_y = i * cell_h
                end_y = min(start_y + cell_h, h)
                start_x = j * cell_w
                end_x = min(start_x + cell_w, w)
                
                cell = image[start_y:end_y, start_x:end_x]
                target_colour = np.mean(cell, axis=(0, 1))
                
                # Find best tile
                best_tile_idx = self.find_contextual_best_tile(
                    target_colour, context, (i, j), (grid_rows, grid_cols), usage_count
                )
                
                # Place tile
                tile_start_row = i * tile_h
                tile_end_row = tile_start_row + tile_h
                tile_start_col = j * tile_w
                tile_end_col = tile_start_col + tile_w
                
                mosaic[tile_start_row:tile_end_row, tile_start_col:tile_end_col] = self.tile_images[best_tile_idx]
                
                # Update usage
                if usage_count is not None:
                    tile_name = self.tile_names[best_tile_idx]
                    usage_count[tile_name] = usage_count.get(tile_name, 0) + 1
        
        print("Contextual mosaic completed")
        return mosaic, context
    
    def visualize_context_analysis(self, image: np.ndarray, context: ImageContext):
        """Display contextual analysis visualization."""
        print("Showing context analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original with face overlay
        img_with_faces = image.copy()
        for (x, y, w, h) in context.face_regions:
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        axes[0, 0].imshow(img_with_faces)
        axes[0, 0].set_title(f'Face Detection\n{len(context.face_regions)} faces')
        axes[0, 0].axis('off')
        
        # Brightness map
        axes[0, 1].imshow(context.brightness_map, cmap='gray')
        axes[0, 1].set_title('Brightness Map')
        axes[0, 1].axis('off')
        
        # Edge density
        axes[0, 2].imshow(context.edge_density_map, cmap='hot')
        axes[0, 2].set_title(f'Edge Density\nComplexity: {context.content_complexity:.3f}')
        axes[0, 2].axis('off')
        
        # Dominant colors
        colors_display = context.dominant_colors.reshape(1, -1, 3).astype(np.uint8)
        axes[1, 0].imshow(colors_display)
        axes[1, 0].set_title('Dominant Colors')
        axes[1, 0].axis('off')
        
        # Scene info
        scene_text = (f"Scene Analysis:\n\n"
                     f"Portrait: {context.is_portrait}\n"
                     f"Landscape: {context.is_landscape}\n"
                     f"Has Faces: {context.has_faces}\n"
                     f"Complexity: {context.content_complexity:.3f}")
        
        axes[1, 1].text(0.5, 0.5, scene_text, ha='center', va='center', 
                        fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Scene Classification')
        axes[1, 1].axis('off')
        
        # Color palette
        if self.colour_palette is not None:
            palette_display = self.colour_palette.reshape(1, -1, 3).astype(np.uint8)
            axes[1, 2].imshow(palette_display)
            axes[1, 2].set_title('Color Palette')
            axes[1, 2].axis('off')
        
        plt.suptitle('Contextual Analysis for Intelligent Tile Selection', fontsize=16)
        plt.tight_layout()
        plt.show()