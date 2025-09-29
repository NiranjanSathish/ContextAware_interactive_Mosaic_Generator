"""
cache_builder.py
Builds optimized tile caches for fast mosaic generation.
Run this once to create cache files, then use for instant tile loading.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import pickle
import time
import warnings
from collections import defaultdict

class TileCacheBuilder:
    """Builds optimized tile caches with rotation variants and color subgrouping."""
    
    def __init__(self, tile_folder: str, 
                 tile_size: Tuple[int, int] = (32, 32),
                 colour_bins: int = 8,
                 enable_rotation: bool = True):
        """
        Initialize cache builder.
        
        Args:
            tile_folder: Path to folder containing tile images
            tile_size: Target tile size (width, height)
            colour_bins: Number of colour categories for subgrouping
            enable_rotation: Whether to create rotated variants
        """
        self.tile_folder = Path(tile_folder)
        self.tile_size = tile_size
        self.colour_bins = colour_bins
        self.enable_rotation = enable_rotation
        self.rotation_angles = [0, 90, 180, 270] if enable_rotation else [0]
        
        # Data containers
        self.tile_images = []
        self.tile_colours = []
        self.tile_names = []
        self.colour_palette = None
        self.colour_groups = defaultdict(list)
        self.colour_indices = {}
        
        print(f"Tile Cache Builder")
        print(f"Folder: {tile_folder}")
        print(f"Tile size: {tile_size[0]}x{tile_size[1]}")
        print(f"Colour bins: {colour_bins}")
        print(f"Rotation: {enable_rotation}")
    
    def build_cache(self, output_file: str, force_rebuild: bool = False) -> bool:
        """Build complete optimized tile cache."""
        if Path(output_file).exists() and not force_rebuild:
            print(f"Cache exists: {output_file} (use force_rebuild=True to rebuild)")
            return False
        
        print("Building comprehensive tile cache...")
        total_start = time.time()
        
        try:
            self._load_base_tiles()
            if self.enable_rotation:
                self._create_rotated_variants()
            self._analyze_tile_colors()
            self._create_color_subgroups()
            self._save_cache(output_file)
            
            total_time = time.time() - total_start
            print(f"Cache built in {total_time:.2f} seconds: {output_file}")
            return True
            
        except Exception as e:
            print(f"Cache building failed: {e}")
            return False
    
    def _load_base_tiles(self):
        """Load base tiles from folder."""
        if not self.tile_folder.exists():
            raise ValueError(f"Tile folder not found: {self.tile_folder}")
        
        # Find all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(self.tile_folder.glob(f'*{ext}'))
            image_files.extend(self.tile_folder.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No images found in {self.tile_folder}")
        
        print(f"Loading {len(image_files)} base tiles...")
        
        for img_path in tqdm(image_files, desc="Loading tiles"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size
                    if img_rgb.shape[:2] != (self.tile_size[1], self.tile_size[0]):
                        img_rgb = cv2.resize(img_rgb, self.tile_size, interpolation=cv2.INTER_AREA)
                    
                    self.tile_images.append(img_rgb)
                    self.tile_names.append(img_path.name)
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Loaded {len(self.tile_images)} base tiles")
    
    def _create_rotated_variants(self):
        """Create 90, 180, 270 degree rotated variants."""
        print("Creating rotated variants...")
        
        base_count = len(self.tile_images)
        rotated_images = []
        rotated_names = []
        
        for i in range(base_count):
            base_image = self.tile_images[i]
            base_name = self.tile_names[i]
            
            # Create 3 rotated versions
            for angle in [90, 180, 270]:
                if angle == 90:
                    rotated = cv2.rotate(base_image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated = cv2.rotate(base_image, cv2.ROTATE_180)
                elif angle == 270:
                    rotated = cv2.rotate(base_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                rotated_images.append(rotated)
                rotated_names.append(f"{base_name}_rot{angle}")
        
        # Add to main collections
        self.tile_images.extend(rotated_images)
        self.tile_names.extend(rotated_names)
        
        print(f"Expanded to {len(self.tile_images)} tiles with rotation")
    
    def _analyze_tile_colors(self):
        """Calculate average colors for all tiles."""
        print("Analyzing tile colors...")
        
        for tile_image in tqdm(self.tile_images, desc="Color analysis"):
            avg_colour = np.mean(tile_image, axis=(0, 1))
            self.tile_colours.append(avg_colour)
        
        self.tile_colours = np.array(self.tile_colours)
        print(f"Color analysis complete: {len(self.tile_colours)} tiles")
    
    def _create_color_subgroups(self):
        """Create color palette and subgroup tiles for fast searching."""
        print(f"Creating {self.colour_bins}-color palette...")
        
        # Create color palette using Mini-Batch K-means
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            batch_size = min(max(len(self.tile_colours) // 10, 100), 1000)
            kmeans = MiniBatchKMeans(
                n_clusters=self.colour_bins, 
                batch_size=batch_size,
                random_state=42, 
                n_init=3
            )
            kmeans.fit(self.tile_colours)
            self.colour_palette = kmeans.cluster_centers_
        
        # Assign tiles to color bins
        for i, tile_colour in enumerate(self.tile_colours):
            distances = euclidean_distances(
                tile_colour.reshape(1, -1), 
                self.colour_palette
            )[0]
            closest_bin = np.argmin(distances)
            self.colour_groups[closest_bin].append(i)
        
        # Create search indices for each group
        for bin_id, tile_indices in self.colour_groups.items():
            if len(tile_indices) > 0:
                group_colours = self.tile_colours[tile_indices]
                
                index = NearestNeighbors(
                    n_neighbors=min(10, len(tile_indices)),
                    metric='euclidean',
                    algorithm='kd_tree'
                )
                index.fit(group_colours)
                self.colour_indices[bin_id] = (index, tile_indices)
        
        print(f"Created {len(self.colour_groups)} color subgroups")
    
    def _save_cache(self, output_file: str):
        """Save processed data to cache file."""
        cache_data = {
            'tile_images': np.array(self.tile_images),
            'tile_colours': self.tile_colours,
            'tile_names': self.tile_names,
            'colour_palette': self.colour_palette,
            'colour_groups': dict(self.colour_groups),
            'colour_indices': self.colour_indices,
            'tile_size': self.tile_size,
            'colour_bins': self.colour_bins,
            'enable_rotation': self.enable_rotation,
            'build_timestamp': time.time()
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        cache_size_mb = Path(output_file).stat().st_size / 1024 / 1024
        print(f"Cache saved: {cache_size_mb:.1f}MB")

if __name__ == "__main__":
    # Build cache with your settings
    builder = TileCacheBuilder(
        tile_folder="extracted_images",
        tile_size=(32, 32),
        colour_bins=8,
        enable_rotation=True
    )
    
    success = builder.build_cache("tiles_cache.pkl", force_rebuild=True)
    
    if success:
        print("Cache building completed! You can now run main.py")
    else:
        print("Cache building failed!")