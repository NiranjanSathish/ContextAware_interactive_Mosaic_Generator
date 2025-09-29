# Advanced Contextual Mosaic Generator

An AI-powered mosaic generation system with contextual awareness, face detection, and performance optimization. Creates high-quality photo mosaics using intelligent tile selection and advanced computer vision techniques.

Check 
[Gradio Implementation](https://huggingface.co/spaces/NiranjanSathish/Interactive_Mosaic_Generator) for a live interactive Demo.

## Features

### Core Capabilities
- **Contextual Awareness**: Face detection and scene classification for intelligent tile placement
- **Multi-Scale Processing**: Support for various grid sizes (16×16 to 128×128)
- **Rotation Variants**: 4-way tile rotation for increased diversity
- **Color Optimization**: Mini-Batch K-means clustering for fast color matching
- **Performance Metrics**: Comprehensive quality evaluation (MSE, PSNR, SSIM, histogram similarity)
- **Web Interface**: Production-ready Gradio deployment

### Advanced Features
- **Smart Preprocessing**: Automatic image resizing with grid-perfect alignment
- **Color Quantization**: Optional color reduction for artistic effects
- **Diversity Control**: Anti-repetition algorithms for natural-looking results
- **Cache System**: Pre-built tile caches for instant deployment
- **Grid Visualization**: Visual feedback showing image segmentation

## Installation


### Quick Setup
```bash
# Clone the repository
git clone <your-repository-url>
cd advanced-mosaic-generator

# Install dependencies
pip install -r requirements.txt

# Create tile folder and add your tile images
mkdir extracted_images
# Add your tile images to this folder
```

### Dependencies
```text
gradio>=4.0.0
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
matplotlib>=3.5.0
pillow>=9.0.0
tqdm>=4.64.0
fastapi>=0.68.0
uvicorn>=0.15.0
seaborn==0.13.2
```

## Usage

### 1. Web Interface (Recommended)
```bash
python app.py
```
Access the interface at `http://localhost:7860`

**Features:**
- Drag-and-drop image upload
- Real-time parameter adjustment
- Grid visualization preview
- Performance metrics display
- One-click presets (Fast/Quality)

### 2. Command Line Interface
```bash
python main.py
```
Edit configuration in `main.py` for custom settings.

### 3. Building Custom Caches
```bash
# Build optimized tile caches
python cache_builder.py
```

## File Structure

```
├── Images                      # Sample Images
├── app.py                      # Gradio web interface (deployment-ready)
├── main.py                     # Complete CLI mosaic generator
├── cache_builder.py            # Tile cache builder with rotation variants
├── contextual_mosaic.py        # Advanced contextual mosaic engine
├── ImagePreprocessor.py        # Image preprocessing and quantization
├── ColourClassification.py     # Grid analysis and color classification
├── performance_metrics.py      # Quality evaluation metrics
├── extracted_images/           # Your tile images folder
├── cache_*.pkl                 # Pre-built tile caches
└── requirements.txt           # Python dependencies
```

## Configuration Options

### Grid Settings
- **Grid Size**: 16×16 to 128×128 tiles
- **Tile Size**: 16×16 or 32×32
- **Diversity Factor**: 0.0-0.5 (controls tile repetition)

### Quality Settings
- **Rotation**: Enable 4-way tile rotation
- **Color Bins**: 4-24 color categories for grouping
- **Quantization**: Optional color reduction

### Performance Modes
- **Fast Mode**: 24×24 grid, 32×32 tiles, minimal diversity
- **Quality Mode**: 128×128 grid, 16×16 tiles, maximum precision

## Technical Innovations

### Contextual Intelligence
- **Face Detection**: Preserves facial features with higher precision
- **Scene Classification**: Automatic portrait/landscape detection
- **Content Analysis**: Edge density and complexity mapping

### Performance Optimizations
- **Color Subgrouping**: 10x faster tile matching using K-means clustering
- **Vectorized Operations**: NumPy-optimized grid processing
- **Mini-Batch Processing**: Memory-efficient color quantization
- **Pre-built Caches**: Instant loading for deployment

### Quality Metrics
- **MSE**: Mean Squared Error measurement
- **PSNR**: Peak Signal-to-Noise Ratio (>30dB = good)
- **SSIM**: Structural Similarity Index (>0.8 = very good)
- **Histogram Similarity**: Color distribution matching
- **Overall Score**: Composite quality rating (0-100)

## Examples

### Basic Usage
```python
from main import create_advanced_mosaic

mosaic, metrics, context = create_advanced_mosaic(
    image_path="your_image.jpg",
    grid_size=(64, 64),
    tile_size=(32, 32),
    diversity_factor=0.15,
    enable_rotation=True
)
```

### Gradio Integration
```python
from main import create_mosaic_for_gradio

mosaic, status = create_mosaic_for_gradio(
    image_array=image_data,
    grid_size=64,
    tile_size=32,
    diversity_factor=0.15
)
```

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
1. Upload pre-built cache files (`cache_*.pkl`)
2. Ensure `extracted_images/` folder is accessible
3. Deploy with your preferred hosting service
4. The system automatically uses available caches

### Cache Management
- **16×16 tiles**: Best for detailed, high-resolution mosaics
- **32×32 tiles**: Balanced quality and performance


## Performance Benchmarks

### Memory Requirements
- **Base System**: ~500MB
- **With Caches**: ~1-3GB (depending on tile count)
- **Processing Peak**: ~2-5GB

## Troubleshooting

### Common Issues
1. **"No cache available"**: Run `cache_builder.py` first
2. **"Tile folder not found"**: Create `extracted_images/` folder
3. **Memory errors**: Reduce grid size or enable quantization
4. **Poor quality**: Increase grid size or add more diverse tiles

### Cache Rebuilding
```bash
# Force rebuild all caches
python -c "from cache_builder import TileCacheBuilder; 
TileCacheBuilder('extracted_images', (32,32), 8, True).build_cache('cache_32x32_bins8_rot.pkl', True)"
```

## Contributing

The system is designed with modular architecture:
- Add new tile selection algorithms in `contextual_mosaic.py`
- Implement additional metrics in `performance_metrics.py`
- Extend preprocessing in `ImagePreprocessor.py`
- Enhance UI features in `app.py`

## Academic Context

This project demonstrates advanced concepts in:
- **Computer Vision**: Face detection, edge analysis, color spaces
- **Machine Learning**: K-means clustering, Mini-Batch optimization
- **Image Processing**: Multi-scale analysis, quality metrics
- **Software Engineering**: Modular design, caching, deployment

