"""
main.py
Complete advanced mosaic generator with contextual awareness.
Integrates all components for professional mosaic creation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Optional
import time
import matplotlib.pyplot as plt

# Import modular components
from ImagePreprocessor import ImagePreprocessor
from ColourClassification import ImageGridAnalyzer
from contextual_Mosaic_Builder import ContextualMosaicGenerator, ImageContext
from Performance_metrics import PerformanceEvaluator

def create_advanced_mosaic(image_path: str,
                          tile_folder: str = "extracted_images", 
                          grid_size: Tuple[int, int] = (64, 64),
                          tile_size: Tuple[int, int] = (32, 32),
                          diversity_factor: float = 0.15,
                          enable_rotation: bool = True,
                          colour_bins: int = 8,
                          apply_quantization: bool = True,
                          n_colors: int = 12,
                          evaluate_quality: bool = True,
                          show_visualizations: bool = True) -> Tuple[np.ndarray, Optional[Dict], Optional[ImageContext]]:
    """
    Create advanced contextual mosaic with all optimizations and analysis.
    
    Args:
        image_path: Path to input image
        tile_folder: Path to tile images folder
        grid_size: Grid dimensions (rows, cols)
        tile_size: Individual tile size (width, height)
        diversity_factor: Tile diversity factor (0.0-0.5)
        enable_rotation: Whether to enable 4-way tile rotation
        colour_bins: Number of color bins for subgrouping optimization
        apply_quantization: Whether to apply color quantization to input
        n_colors: Number of colors for quantization
        evaluate_quality: Whether to calculate quality metrics
        show_visualizations: Whether to display analysis and results
        
    Returns:
        Tuple of (mosaic_image, quality_metrics, context_analysis)
    """
    print("Advanced Contextual Mosaic Generator")
    print(f"Image: {image_path}")
    print(f"Grid: {grid_size[0]}x{grid_size[1]} = {np.prod(grid_size)} tiles")
    print(f"Tile size: {tile_size[0]}x{tile_size[1]}")
    print(f"Rotation: {enable_rotation}")
    
    total_start = time.time()
    
    # Step 1: Initialize contextual generator (auto-builds cache if needed)
    cache_filename = f"cache_{tile_size[0]}x{tile_size[1]}_bins{colour_bins}{'_rot' if enable_rotation else ''}.pkl"
    
    generator = ContextualMosaicGenerator(
        cache_file=cache_filename,
        tile_folder=tile_folder,
        tile_size=tile_size,
        colour_bins=colour_bins,
        enable_rotation=enable_rotation
    )
    
    # Step 2: Preprocess image with smart sizing
    print("\n=== Step 1: Image Preprocessing ===")
    target_width = grid_size[1] * tile_size[0]
    target_height = grid_size[0] * tile_size[1]
    
    preprocessor = ImagePreprocessor(
        target_resolution=(target_width, target_height),
        grid_size=grid_size
    )
    
    processed_image = preprocessor.load_and_preprocess_image(
        image_path,
        apply_quantization=apply_quantization,
        n_colors=n_colors
    )
    
    if processed_image is None:
        raise ValueError("Failed to preprocess image")
    
    # Step 3: Generate contextual mosaic
    print("\n=== Steps 2-3: Contextual Mosaic Generation ===")
    mosaic, context = generator.create_contextual_mosaic(
        processed_image, grid_size, diversity_factor
    )
    
    # Step 4: Show context analysis
    if show_visualizations:
        generator.visualize_context_analysis(processed_image, context)
    
    # Step 5: Quality evaluation
    metrics = None
    if evaluate_quality:
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate_mosaic_quality(processed_image, mosaic, image_path)
        
        if show_visualizations:
            evaluator.visualize_quality_comparison(processed_image, mosaic, metrics)
    
    # Final results display
    if not show_visualizations:
        # Simple results without plots
        print(f"\nMosaic created: {mosaic.shape[1]}x{mosaic.shape[0]} pixels")
        if metrics:
            print(f"Quality score: {metrics['overall_quality']:.1f}/100")
    
    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    return mosaic, metrics, context

def create_mosaic_for_gradio(image_array: np.ndarray,
                           tile_folder: str = "extracted_images",
                           grid_size: int = 64,
                           tile_size: int = 32,
                           diversity_factor: float = 0.15,
                           enable_rotation: bool = True) -> Tuple[np.ndarray, str]:
    """
    Gradio-optimized mosaic creation function.
    Designed for web interface integration with status reporting.
    
    Args:
        image_array: Input image as numpy array (from Gradio)
        tile_folder: Path to tile folder
        grid_size: Grid size (single value for square grid)
        tile_size: Tile size (single value for square tiles)
        diversity_factor: Tile diversity factor
        enable_rotation: Whether to enable rotation
        
    Returns:
        Tuple of (mosaic_image, status_message)
    """
    try:
        grid_tuple = (grid_size, grid_size)
        tile_tuple = (tile_size, tile_size)
        
        # Save temporary image for processing
        temp_path = "temp_gradio_input.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        # Create mosaic with visualizations disabled for Gradio
        mosaic, metrics, context = create_advanced_mosaic(
            image_path=temp_path,
            tile_folder=tile_folder,
            grid_size=grid_tuple,
            tile_size=tile_tuple,
            diversity_factor=diversity_factor,
            enable_rotation=enable_rotation,
            evaluate_quality=True,
            show_visualizations=False
        )
        
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)
        
        # Create status message
        status = f"""Mosaic Generated Successfully!

Configuration:
• Grid: {grid_size}×{grid_size} = {grid_size**2} tiles
• Tile size: {tile_size}×{tile_size} pixels
• Rotation: {'Enabled' if enable_rotation else 'Disabled'}
• Faces detected: {len(context.face_regions) if context else 0}
• Quality score: {metrics['overall_quality']:.1f}/100"""
        
        return mosaic, status
        
    except Exception as e:
        error_msg = f"Error creating mosaic: {str(e)}"
        return None, error_msg

if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "Images/Batman.jpg"
    TILE_FOLDER = "extracted_images"
    
    # Validate paths
    if not Path(TILE_FOLDER).exists():
        print(f"Tile folder not found: {TILE_FOLDER}")
        exit(1)
    
    if not Path(IMAGE_PATH).exists():
        print(f"Image not found: {IMAGE_PATH}")
        exit(1)
    
    # Create advanced mosaic with full analysis and visualizations
    print("Creating advanced contextual mosaic with full analysis...")
    
    mosaic, metrics, context = create_advanced_mosaic(
        image_path=IMAGE_PATH,
        tile_folder=TILE_FOLDER,
        grid_size=(64, 64),
        tile_size=(16, 16),
        diversity_factor=0.15,
        enable_rotation=True,
        apply_quantization=True,
        n_colors=12,
        evaluate_quality=True,
        show_visualizations=True  # Enable all visualizations
    )
    
    if mosaic is not None:
        print("Advanced mosaic generation completed successfully!")
        if metrics:
            print(f"Final quality score: {metrics['overall_quality']:.1f}/100")
        
        # Save result
        cv2.imwrite("output_mosaic.jpg", cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
        print("Mosaic saved as 'output_mosaic.jpg'")
    else:
        print("Mosaic generation failed.")