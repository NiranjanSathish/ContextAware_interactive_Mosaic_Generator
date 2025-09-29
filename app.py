"""
app.py
Deployment-only Gradio interface with grid segmentation visualization.
Uses only pre-built cache files for fast cloud deployment.
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import time
import pickle
import warnings
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from skimage.metrics import structural_similarity as ssim

# Deployment configuration
TILE_FOLDER = "extracted_images"

# Pre-built cache files that should be uploaded with the app
AVAILABLE_CACHES = {
    16: "cache_16x16_bins8_rot.pkl",
    32: "cache_32x32_bins8_rot.pkl", 
    64: "cache_64x64_bins8_rot.pkl"
}

@dataclass
class ImageContext:
    """Contextual analysis results."""
    has_faces: bool
    face_regions: List[Tuple[int, int, int, int]]
    is_portrait: bool
    is_landscape: bool
    content_complexity: float

class DeploymentMosaicGenerator:
    """Deployment-optimized mosaic generator that NEVER builds caches."""
    
    def __init__(self, cache_file: str):
        """Initialize with existing cache file only."""
        self.cache_file = cache_file
        
        # Load cache data
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.tile_images = data['tile_images']
            self.tile_colours = data['tile_colours']
            self.tile_names = data['tile_names']
            self.colour_palette = data['colour_palette']
            self.colour_groups = data['colour_groups']
            self.colour_indices = data['colour_indices']
            self.tile_size = data['tile_size']
            
            print(f"Loaded cache: {len(self.tile_images)} tiles, size {self.tile_size}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load cache {cache_file}: {e}")
        
        # Face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = None
    
    def analyze_context(self, image: np.ndarray) -> ImageContext:
        """Basic context analysis for deployment."""
        faces = []
        has_faces = False
        
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                faces = [(x, y, w, h) for (x, y, w, h) in detected_faces]
                has_faces = len(faces) > 0
            except:
                pass
        
        # Basic scene classification
        aspect_ratio = image.shape[1] / image.shape[0]
        is_portrait = aspect_ratio < 1.2 and has_faces
        is_landscape = aspect_ratio > 1.5
        
        # Simple complexity measure
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        content_complexity = np.mean(edges) / 255.0
        
        return ImageContext(
            has_faces=has_faces,
            face_regions=faces,
            is_portrait=is_portrait,
            is_landscape=is_landscape,
            content_complexity=content_complexity
        )
    
    def find_best_tile(self, target_colour: np.ndarray) -> int:
        """Find best matching tile using color subgroups."""
        distances = euclidean_distances(target_colour.reshape(1, -1), self.colour_palette)[0]
        target_bin = np.argmin(distances)
        
        if target_bin in self.colour_indices:
            index, tile_indices = self.colour_indices[target_bin]
            _, indices = index.kneighbors(target_colour.reshape(1, -1), n_neighbors=1)
            return tile_indices[indices[0][0]]
        
        return 0
    
    def create_mosaic_with_preprocessing(self, image: np.ndarray, grid_size: int, diversity_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray, ImageContext]:
        """Create mosaic with preprocessing visualization."""
        print(f"Creating {grid_size}x{grid_size} mosaic...")
        
        # Analyze context
        context = self.analyze_context(image)
        
        # Preprocess image to fit grid
        target_size = grid_size * self.tile_size[0]
        h, w = image.shape[:2]
        scale = max(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Center crop
        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        processed_image = resized[start_y:start_y + target_size, start_x:start_x + target_size]
        
        # Create grid visualization
        grid_visualization = self.create_grid_overlay(processed_image, grid_size)
        
        # Create mosaic
        cell_size = target_size // grid_size
        mosaic = np.zeros((grid_size * self.tile_size[1], grid_size * self.tile_size[0], 3), dtype=np.uint8)
        
        usage_count = {} if diversity_factor > 0 else None
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Get cell color
                start_y = i * cell_size
                end_y = start_y + cell_size
                start_x = j * cell_size
                end_x = start_x + cell_size
                
                cell = processed_image[start_y:end_y, start_x:end_x]
                target_colour = np.mean(cell, axis=(0, 1))
                
                # Find best tile
                best_tile_idx = self.find_best_tile(target_colour)
                
                # Apply diversity if enabled
                if usage_count is not None:
                    tile_name = self.tile_names[best_tile_idx]
                    usage_penalty = usage_count.get(tile_name, 0) * diversity_factor
                    
                    # Try alternative tiles if this one is overused
                    if usage_penalty > 2.0:
                        distances = euclidean_distances(target_colour.reshape(1, -1), self.tile_colours)[0]
                        sorted_indices = np.argsort(distances)
                        for idx in sorted_indices[1:6]:
                            alt_tile_name = self.tile_names[idx]
                            if usage_count.get(alt_tile_name, 0) < usage_count[tile_name]:
                                best_tile_idx = idx
                                break
                    
                    usage_count[self.tile_names[best_tile_idx]] = usage_count.get(self.tile_names[best_tile_idx], 0) + 1
                
                # Place tile
                tile_start_y = i * self.tile_size[1]
                tile_end_y = tile_start_y + self.tile_size[1]
                tile_start_x = j * self.tile_size[0]
                tile_end_x = tile_start_x + self.tile_size[0]
                
                mosaic[tile_start_y:tile_end_y, tile_start_x:tile_end_x] = self.tile_images[best_tile_idx]
        
        return mosaic, grid_visualization, context
    
    def create_grid_overlay(self, image: np.ndarray, grid_size: int) -> np.ndarray:
        """Create visualization showing grid segmentation with enhanced styling."""
        h, w = image.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        # Create a copy for drawing
        grid_image = image.copy()
        
        # Draw grid lines with better visibility
        line_color = (255, 255, 255)  # White lines
        shadow_color = (0, 0, 0)     # Black shadow
        line_thickness = max(1, min(w, h) // 500)  # Adaptive thickness
        
        # Draw vertical lines
        for i in range(1, grid_size):
            x = i * cell_w
            # Draw shadow first
            cv2.line(grid_image, (x-1, 0), (x-1, h), shadow_color, line_thickness)
            cv2.line(grid_image, (x+1, 0), (x+1, h), shadow_color, line_thickness)
            # Draw main line
            cv2.line(grid_image, (x, 0), (x, h), line_color, line_thickness)
        
        # Draw horizontal lines
        for i in range(1, grid_size):
            y = i * cell_h
            # Draw shadow first
            cv2.line(grid_image, (0, y-1), (w, y-1), shadow_color, line_thickness)
            cv2.line(grid_image, (0, y+1), (w, y+1), shadow_color, line_thickness)
            # Draw main line
            cv2.line(grid_image, (0, y), (w, y), line_color, line_thickness)
        
        # Add border around entire image
        border_thickness = max(2, line_thickness * 2)
        cv2.rectangle(grid_image, (0, 0), (w-1, h-1), line_color, border_thickness)
        cv2.rectangle(grid_image, (border_thickness, border_thickness), 
                     (w-border_thickness-1, h-border_thickness-1), shadow_color, 1)
        
        # Add informative text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 800)
        thickness = max(1, int(font_scale * 2))
        
        # Main grid info
        text_main = f"Grid: {grid_size}x{grid_size}"
        text_sub = f"{grid_size**2} cells total"
        text_cell = f"Cell size: {cell_w}x{cell_h}px"
        
        # Calculate text dimensions
        (main_w, main_h), _ = cv2.getTextSize(text_main, font, font_scale, thickness)
        (sub_w, sub_h), _ = cv2.getTextSize(text_sub, font, font_scale * 0.7, thickness)
        (cell_w_text, cell_h_text), _ = cv2.getTextSize(text_cell, font, font_scale * 0.6, thickness)
        
        # Create background for text
        padding = 10
        bg_width = max(main_w, sub_w, cell_w_text) + padding * 2
        bg_height = main_h + sub_h + cell_h_text + padding * 4
        
        # Draw semi-transparent background
        overlay = grid_image.copy()
        cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, grid_image, 0.3, 0, grid_image)
        
        # Draw border around text area
        cv2.rectangle(grid_image, (10, 10), (10 + bg_width, 10 + bg_height), line_color, 1)
        
        # Draw text
        y_offset = 10 + padding + main_h
        cv2.putText(grid_image, text_main, (10 + padding, y_offset), 
                   font, font_scale, line_color, thickness)
        
        y_offset += sub_h + padding // 2
        cv2.putText(grid_image, text_sub, (10 + padding, y_offset), 
                   font, font_scale * 0.7, (200, 200, 200), thickness)
        
        y_offset += cell_h_text + padding // 2
        cv2.putText(grid_image, text_cell, (10 + padding, y_offset), 
                   font, font_scale * 0.6, (180, 180, 180), thickness)
        
        # Add corner indicators for first few cells
        if grid_size <= 16:  # Only for smaller grids to avoid clutter
            corner_size = max(15, min(cell_w, cell_h) // 10)
            for i in range(min(3, grid_size)):
                for j in range(min(3, grid_size)):
                    x = j * cell_w + 5
                    y = i * cell_h + 15
                    cell_num = i * grid_size + j + 1
                    
                    # Small background for cell number
                    cv2.circle(grid_image, (x + 10, y), 12, (0, 0, 0), -1)
                    cv2.circle(grid_image, (x + 10, y), 12, line_color, 1)
                    
                    # Cell number
                    cv2.putText(grid_image, str(cell_num), (x + 5, y + 4), 
                               font, 0.4, line_color, 1)
        
        return grid_image

def create_grid_visualization(image: np.ndarray, grid_size: int) -> np.ndarray:
    """Create visualization showing grid segmentation on preprocessed image."""
    h, w = image.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # Create a copy for drawing grid lines
    grid_image = image.copy()
    
    # Draw vertical lines
    for i in range(1, grid_size):
        x = i * cell_w
        cv2.line(grid_image, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(grid_image, (x-1, 0), (x-1, h), (0, 0, 0), 1)  # Black shadow for visibility
    
    # Draw horizontal lines
    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(grid_image, (0, y), (w, y), (255, 255, 255), 1)
        cv2.line(grid_image, (0, y-1), (w, y-1), (0, 0, 0), 1)  # Black shadow for visibility
    
    # Add grid info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(w, h) / 800)  # Scale font based on image size
    thickness = max(1, int(font_scale * 2))
    
    # Add background rectangle for text
    text = f"Grid: {grid_size}x{grid_size} = {grid_size**2} cells"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(grid_image, (5, 5), (text_w + 15, text_h + baseline + 10), (0, 0, 0), -1)
    cv2.rectangle(grid_image, (5, 5), (text_w + 15, text_h + baseline + 10), (255, 255, 255), 1)
    cv2.putText(grid_image, text, (10, text_h + 8), font, font_scale, (255, 255, 255), thickness)
    
    return grid_image

def calculate_global_ssim(original: np.ndarray, mosaic: np.ndarray) -> float:
    """
    Calculate Global SSIM by treating each image as a single entity.
    Computes global statistics for the entire image rather than using sliding windows.
    """
    if original.shape != mosaic.shape:
        original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
    else:
        original_resized = original
    
    # Convert to float for calculations
    orig_float = original_resized.astype(np.float64)
    mosaic_float = mosaic.astype(np.float64)
    
    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    global_ssim_values = []
    
    # Calculate global SSIM for each channel
    for channel in range(3):
        orig_channel = orig_float[:, :, channel]
        mosaic_channel = mosaic_float[:, :, channel]
        
        # Global means (entire image)
        mu1 = np.mean(orig_channel)
        mu2 = np.mean(mosaic_channel)
        
        # Global variances and covariance (entire image)
        sigma1_sq = np.var(orig_channel)
        sigma2_sq = np.var(mosaic_channel)
        sigma12 = np.mean((orig_channel - mu1) * (mosaic_channel - mu2))
        
        # Global SSIM calculation
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        channel_ssim = numerator / denominator
        global_ssim_values.append(channel_ssim)
    
    # Average across channels
    return float(np.mean(global_ssim_values))

def calculate_global_color_similarity(original: np.ndarray, mosaic: np.ndarray) -> float:
    """Calculate global color distribution similarity."""
    if original.shape != mosaic.shape:
        original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
    else:
        original_resized = original
    
    # Calculate global color statistics
    orig_mean = np.mean(original_resized, axis=(0, 1))  # Mean color per channel
    mosaic_mean = np.mean(mosaic, axis=(0, 1))
    
    orig_std = np.std(original_resized, axis=(0, 1))    # Std per channel
    mosaic_std = np.std(mosaic, axis=(0, 1))
    
    # Color mean similarity
    color_mean_diff = np.linalg.norm(orig_mean - mosaic_mean)
    color_mean_sim = 1.0 / (1.0 + color_mean_diff / 255.0)
    
    # Color variance similarity
    color_std_diff = np.linalg.norm(orig_std - mosaic_std)
    color_std_sim = 1.0 / (1.0 + color_std_diff / 255.0)
    
    # Combined color similarity
    global_color_sim = 0.6 * color_mean_sim + 0.4 * color_std_sim
    
    return float(global_color_sim)

def calculate_metrics(original: np.ndarray, mosaic: np.ndarray) -> Dict[str, float]:
    """Calculate enhanced quality metrics with global SSIM."""
    # Resize for comparison
    if original.shape != mosaic.shape:
        original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
    else:
        original_resized = original
    
    # MSE
    orig_float = original_resized.astype(np.float64)
    mosaic_float = mosaic.astype(np.float64)
    mse = float(np.mean((orig_float - mosaic_float) ** 2))
    
    # PSNR
    psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else float('inf')
    
    # Global SSIM (replaces standard SSIM)
    global_ssim = calculate_global_ssim(original, mosaic)
    
    # Global Color Similarity
    global_color_sim = calculate_global_color_similarity(original, mosaic)
    
    # Histogram similarity
    correlations = []
    for channel in range(3):
        hist_orig = cv2.calcHist([original_resized], [channel], None, [256], [0, 256])
        hist_mosaic = cv2.calcHist([mosaic], [channel], None, [256], [0, 256])
        corr = cv2.compareHist(hist_orig, hist_mosaic, cv2.HISTCMP_CORREL)
        correlations.append(corr)
    histogram_similarity = float(np.mean(correlations))
    
    # Enhanced overall score calculation with global metrics
    ssim_norm = (global_ssim + 1) / 2  # Normalize global SSIM to 0-1
    psnr_norm = min(psnr / 50.0, 1.0)
    
    # Weighted combination emphasizing global consistency
    overall = (
        0.4 * ssim_norm +                    # Global SSIM gets highest weight
        0.25 * global_color_sim +            # Global color consistency
        0.2 * histogram_similarity +         # Color distribution
        0.15 * psnr_norm                     # Signal quality
    ) * 100
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': global_ssim,
        'global_color_similarity': global_color_sim,
        'histogram_similarity': histogram_similarity,
        'overall_quality': float(overall)
    }

def get_best_available_cache(requested_tile_size: int) -> Optional[str]:
    """Get the best available cache for requested tile size."""
    # Check exact match first
    if requested_tile_size in AVAILABLE_CACHES:
        cache_file = AVAILABLE_CACHES[requested_tile_size]
        if Path(cache_file).exists():
            return cache_file
    
    # Find closest available cache
    available_sizes = []
    for size, cache_file in AVAILABLE_CACHES.items():
        if Path(cache_file).exists():
            available_sizes.append(size)
    
    if not available_sizes:
        return None
    
    # Return closest match
    closest_size = min(available_sizes, key=lambda x: abs(x - requested_tile_size))
    return AVAILABLE_CACHES[closest_size]

def create_mosaic_interface(image, grid_size, tile_size, diversity_factor, enable_rotation, apply_quantization, n_colors):
    """Main interface function with grid visualization - deployment optimized."""
    if image is None:
        return None, None, None, "Please upload an image first.", "Please upload an image first."
    
    try:
        start_time = time.time()
        
        # Get appropriate cache file
        cache_file = get_best_available_cache(tile_size)
        if cache_file is None:
            error_msg = f"No cache available for tile size {tile_size}x{tile_size}"
            return None, None, None, error_msg, error_msg
        
        # Initialize generator with existing cache
        generator = DeploymentMosaicGenerator(cache_file)
        
        # Basic color quantization if requested
        if apply_quantization:
            pixels = image.reshape(-1, 3)
            sample_size = min(5000, len(pixels))
            sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                kmeans = MiniBatchKMeans(n_clusters=n_colors, batch_size=500, random_state=42)
                kmeans.fit(sampled_pixels)
                labels = kmeans.predict(pixels)
            
            quantized_pixels = kmeans.cluster_centers_[labels]
            image = quantized_pixels.reshape(image.shape).astype(np.uint8)
        
        # Create mosaic with grid visualization
        mosaic, grid_viz, context = generator.create_mosaic_with_preprocessing(image, grid_size, diversity_factor)
        
        # Calculate metrics
        metrics = calculate_metrics(image, mosaic)
        
        total_time = time.time() - start_time
        
        # Create comparison
        comparison = np.hstack([
            cv2.resize(image, (mosaic.shape[1]//2, mosaic.shape[0])),
            cv2.resize(mosaic, (mosaic.shape[1]//2, mosaic.shape[0]))
        ])
        
        # Metrics display - enhanced with global metrics
        metrics_text = f"""ENHANCED PERFORMANCE METRICS

Mean Squared Error (MSE): {metrics['mse']:.2f}

Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB

Global Structural Similarity (SSIM): {metrics['ssim']:.4f}

Global Color Similarity: {metrics['global_color_similarity']:.4f}

Color Histogram Similarity: {metrics['histogram_similarity']:.4f}

Overall Quality Score: {metrics['overall_quality']:.1f}/100"""
        
        # Status
        status = f"""Generation Successful!

Grid: {grid_size}x{grid_size} = {grid_size**2} tiles
Tile Size: {tile_size}x{tile_size} pixels
Processing Time: {total_time:.2f} seconds
Cache Used: {cache_file}

Contextual Analysis:
• Faces Detected: {len(context.face_regions)}
• Scene Type: {'Portrait' if context.is_portrait else 'Landscape' if context.is_landscape else 'General'}
• Content Complexity: {context.content_complexity:.3f}"""
        
        return mosaic, comparison, grid_viz, metrics_text, status
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, None, None, error_msg, error_msg

def verify_deployment_setup():
    """Check deployment setup without building anything."""
    available_caches = {}
    total_size_mb = 0
    
    for size, cache_file in AVAILABLE_CACHES.items():
        if Path(cache_file).exists():
            size_mb = Path(cache_file).stat().st_size / 1024 / 1024
            available_caches[size] = size_mb
            total_size_mb += size_mb
    
    setup_msg = f"Found {len(available_caches)} cache files ({total_size_mb:.1f}MB total)"
    
    return len(available_caches) > 0, setup_msg, available_caches

def get_system_status():
    """System status for deployment."""
    setup_ok, setup_msg, available_caches = verify_deployment_setup()
    
    cache_list = ""
    for size, size_mb in available_caches.items():
        cache_list += f"  {size}x{size}: {size_mb:.1f}MB\n"
    
    status = f"""DEPLOYMENT STATUS
{'='*30}

Cache System: {'✅' if setup_ok else '❌'}
{setup_msg}

Available Caches:
{cache_list if cache_list else "  None found"}

Smart Selection: System automatically uses the best 
available cache for your chosen tile size.

INNOVATIONS INCLUDED
{'='*30}

Contextual Awareness: Face detection, scene classification
Multi-Orientation: Rotation variants in cache files  
Performance: Color subgrouping, Mini-Batch K-means
Enhanced Metrics: Global SSIM, Color similarity analysis
Grid Visualization: Shows preprocessing segmentation

DEPLOYMENT OPTIMIZED
{'='*30}

• No cache building during startup
• Fast initialization with pre-built caches
• Lightweight processing for cloud deployment
• Maintains all core innovations
• Global quality assessment"""
    
    return status

def create_interface():
    """Create deployment-optimized Gradio interface with grid visualization."""
    
    css = """
    .gradio-container { max-width: 100% !important; padding: 0 20px; }
    .left-panel { 
        flex: 0 0 350px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .right-panel { 
        flex: 1; background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metrics-display {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 10px;
        font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.6;
    }
    """
    
    with gr.Blocks(css=css, title="Advanced Mosaic Generator") as demo:
        
        gr.Markdown("# Advanced Contextual Mosaic Generator")
        gr.Markdown("AI-powered mosaic creation with contextual awareness, grid visualization, and performance metrics.")
        
        with gr.Accordion("System Status", open=False):
            status_display = gr.Textbox(value=get_system_status(), lines=22, show_label=False)
            gr.Button("Refresh Status").click(fn=get_system_status, outputs=status_display)
        
        with gr.Row():
            # Left Panel: Controls
            with gr.Column(scale=0, min_width=350, elem_classes=["left-panel"]):
                gr.Markdown("## Configuration")
                
                # Generate button at top
                generate_btn = gr.Button("Generate Mosaic", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                # Image upload
                input_image = gr.Image(type="numpy", label="Upload Image", height=200)
                
                # Controls
                grid_size = gr.Slider(16, 128, 32, step=8, label="Grid Size", info="Number of tiles per side")
                tile_size = gr.Slider(16, 32, 32, step=16, label="Tile Size", info="Must match available cache")
                diversity_factor = gr.Slider(0.0, 0.5, 0.15, step=0.05, label="Diversity", info="Tile variety")
                enable_rotation = gr.Checkbox(label="Enable Rotation", value=False, info="Uses rotation variants if available")
                apply_quantization = gr.Checkbox(label="Color Quantization", value=True)
                n_colors = gr.Slider(4, 24, 12, step=2, label="Colors")

                # Examples
                gr.Markdown("### Or, try with an example:")
                gr.Examples(
                    examples=[
                        ["EmmaPotrait.jpg", 64, 32, 0.15, False, True, 12],
                        ["Batman.jpg", 128, 16, 0.05, False, False, 16],
                        ["Indian_Dog.jpg", 56, 32, 0.2, False, True, 16],
                    ],
                    inputs=[input_image, grid_size, tile_size, diversity_factor, enable_rotation, apply_quantization, n_colors]
                )
                
                # Presets
                gr.Markdown("### Quick Presets")
                with gr.Row():
                    preset_fast = gr.Button("Fast", size="sm")
                    preset_quality = gr.Button("Quality", size="sm")
            
            # Right Panel: Results
            with gr.Column(scale=2, elem_classes=["right-panel"]):
                gr.Markdown("## Results & Analysis")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generated Mosaic")
                        mosaic_output = gr.Image(height=300, show_label=False)
                    
                    with gr.Column():
                        gr.Markdown("### Comparison (Original | Mosaic)")
                        comparison_output = gr.Image(height=300, show_label=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Grid Segmentation Visualization")
                        grid_viz_output = gr.Image(height=300, show_label=False)
                        gr.Markdown("*Shows how the image is divided into cells for tile placement*")
                    
                    with gr.Column():
                        gr.Markdown("### Performance Metrics")
                        metrics_output = gr.Textbox(lines=8, elem_classes=["metrics-display"], show_label=False)
                
                status_output = gr.Textbox(label="Generation Status", lines=6)
        
        # Connect functions
        def fast_preset():
            return 24, 32, 0.1, False, True, 8
        
        def quality_preset():
            return 128, 16, 0.0, True, False, 24
        
        generate_btn.click(
            fn=create_mosaic_interface,
            inputs=[input_image, grid_size, tile_size, diversity_factor, enable_rotation, apply_quantization, n_colors],
            outputs=[mosaic_output, comparison_output, grid_viz_output, metrics_output, status_output]
        )
        
        preset_fast.click(fn=fast_preset, outputs=[grid_size, tile_size, diversity_factor, enable_rotation, apply_quantization, n_colors])
        preset_quality.click(fn=quality_preset, outputs=[grid_size, tile_size, diversity_factor, enable_rotation, apply_quantization, n_colors])
    
    return demo

if __name__ == "__main__":
    print("Advanced Mosaic Generator - Deployment Version with Grid Visualization")
    print("Checking deployment setup...")
    
    setup_ok, setup_msg, caches = verify_deployment_setup()
    if setup_ok:
        print(f"Deployment ready: {setup_msg}")
        demo = create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    else:
        print(f"Deployment not ready: {setup_msg}")
        print("Please upload the required cache files")