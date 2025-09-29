"""
performance_metrics.py
Performance evaluation with global SSIM that compares images as complete entities.
"""

import numpy as np
import cv2
from typing import Dict
import matplotlib.pyplot as plt

class PerformanceEvaluator:
    """Comprehensive mosaic quality evaluation with global SSIM."""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_mse(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if original.shape != mosaic.shape:
            original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
        else:
            original_resized = original
        
        orig_float = original_resized.astype(np.float64)
        mosaic_float = mosaic.astype(np.float64)
        mse = np.mean((orig_float - mosaic_float) ** 2)
        return float(mse)
    
    def calculate_psnr(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = self.calculate_mse(original, mosaic)
        if mse == 0:
            return float('inf')
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return float(psnr)
    
    def calculate_global_ssim(self, original: np.ndarray, mosaic: np.ndarray) -> float:
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
    
    def calculate_ssim(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """Calculate global SSIM score."""
        return self.calculate_global_ssim(original, mosaic)
    
    def calculate_global_structural_similarity(self, original: np.ndarray, mosaic: np.ndarray) -> Dict[str, float]:
        """
        Calculate multiple global structural similarity metrics.
        All metrics treat images as complete entities.
        """
        if original.shape != mosaic.shape:
            original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
        else:
            original_resized = original
        
        results = {}
        
        # 1. Global SSIM (already calculated)
        results['global_ssim'] = self.calculate_global_ssim(original_resized, mosaic)
        
        # 2. Global Luminance Similarity
        orig_luminance = 0.299 * original_resized[:,:,0] + 0.587 * original_resized[:,:,1] + 0.114 * original_resized[:,:,2]
        mosaic_luminance = 0.299 * mosaic[:,:,0] + 0.587 * mosaic[:,:,1] + 0.114 * mosaic[:,:,2]
        
        mu1_lum = np.mean(orig_luminance)
        mu2_lum = np.mean(mosaic_luminance)
        C1 = (0.01 * 255) ** 2
        
        luminance_sim = (2 * mu1_lum * mu2_lum + C1) / (mu1_lum**2 + mu2_lum**2 + C1)
        results['global_luminance'] = float(luminance_sim)
        
        # 3. Global Contrast Similarity
        sigma1_lum = np.std(orig_luminance)
        sigma2_lum = np.std(mosaic_luminance)
        C2 = (0.03 * 255) ** 2
        
        contrast_sim = (2 * sigma1_lum * sigma2_lum + C2) / (sigma1_lum**2 + sigma2_lum**2 + C2)
        results['global_contrast'] = float(contrast_sim)
        
        # 4. Global Structure Similarity (correlation coefficient)
        sigma12_lum = np.mean((orig_luminance - mu1_lum) * (mosaic_luminance - mu2_lum))
        structure_sim = (sigma12_lum + C2/2) / (sigma1_lum * sigma2_lum + C2/2)
        results['global_structure'] = float(structure_sim)
        
        return results
    
    def calculate_color_histogram_similarity(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """Calculate color histogram similarity."""
        if original.shape != mosaic.shape:
            original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
        else:
            original_resized = original
        
        correlations = []
        for channel in range(3):
            hist_orig = cv2.calcHist([original_resized], [channel], None, [256], [0, 256])
            hist_mosaic = cv2.calcHist([mosaic], [channel], None, [256], [0, 256])
            corr = cv2.compareHist(hist_orig, hist_mosaic, cv2.HISTCMP_CORREL)
            correlations.append(corr)
        
        return float(np.mean(correlations))
    
    def calculate_global_color_similarity(self, original: np.ndarray, mosaic: np.ndarray) -> float:
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
    
    def evaluate_mosaic_quality(self, original: np.ndarray, mosaic: np.ndarray, 
                               original_path: str = None) -> Dict[str, float]:
        """Comprehensive quality evaluation with global metrics."""
        print("Calculating global quality metrics...")
        
        # Basic metrics
        mse_score = self.calculate_mse(original, mosaic)
        psnr_score = self.calculate_psnr(original, mosaic)
        
        # Global structural analysis
        structural_metrics = self.calculate_global_structural_similarity(original, mosaic)
        
        # Color metrics
        histogram_sim = self.calculate_color_histogram_similarity(original, mosaic)
        global_color_sim = self.calculate_global_color_similarity(original, mosaic)
        
        metrics = {
            'mse': mse_score,
            'psnr': psnr_score,
            'ssim': structural_metrics['global_ssim'],
            'global_luminance': structural_metrics['global_luminance'],
            'global_contrast': structural_metrics['global_contrast'],
            'global_structure': structural_metrics['global_structure'],
            'histogram_similarity': histogram_sim,
            'global_color_similarity': global_color_sim
        }
        
        # Global quality score calculation
        ssim_norm = (metrics['ssim'] + 1) / 2  # Normalize SSIM to 0-1
        psnr_norm = min(metrics['psnr'] / 50.0, 1.0)
        
        # Weight global metrics appropriately
        overall = (
            0.4 * ssim_norm +                              # Global SSIM
            0.2 * metrics['global_color_similarity'] +     # Global color consistency
            0.2 * metrics['histogram_similarity'] +        # Color distribution
            0.2 * psnr_norm                                # Signal quality
        ) * 100
        
        metrics['overall_quality'] = float(overall)
        
        self.metrics = metrics
        self._print_global_report(metrics, original_path)
        return metrics
    
    def _print_global_report(self, metrics: Dict[str, float], original_path: str = None):
        """Print global quality assessment report."""
        print(f"\n{'='*50}")
        print("GLOBAL QUALITY ASSESSMENT")
        print(f"{'='*50}")
        
        if original_path:
            print(f"Image: {original_path}")
        
        print("\nGlobal Structure Metrics:")
        print(f"  Global SSIM: {metrics['ssim']:.4f} (>0.8 = very good)")
        print(f"  Luminance: {metrics['global_luminance']:.4f}")
        print(f"  Contrast: {metrics['global_contrast']:.4f}")
        print(f"  Structure: {metrics['global_structure']:.4f}")
        
        print("\nGlobal Color Metrics:")
        print(f"  Color Similarity: {metrics['global_color_similarity']:.4f}")
        print(f"  Histogram Corr: {metrics['histogram_similarity']:.4f}")
        
        print("\nBasic Metrics:")
        print(f"  MSE: {metrics['mse']:.2f} (lower = better)")
        print(f"  PSNR: {metrics['psnr']:.2f} dB (>30 = good)")
        
        print(f"\nOverall Quality Score: {metrics['overall_quality']:.1f}/100")
        
        score = metrics['overall_quality']
        if score >= 85:
            quality = "Excellent"
        elif score >= 75:
            quality = "Very Good"
        elif score >= 65:
            quality = "Good"
        else:
            quality = "Fair"
        
        print(f"Assessment: {quality}")
        print(f"{'='*50}")
    
    def visualize_quality_comparison(self, original: np.ndarray, mosaic: np.ndarray, 
                                   metrics: Dict[str, float] = None):
        """Create visual comparison with global metrics."""
        if metrics is None:
            metrics = self.metrics
        
        print("Showing global quality comparison...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title(f'Original\n{original.shape[1]}×{original.shape[0]}')
        axes[0, 0].axis('off')
        
        # Mosaic
        axes[0, 1].imshow(mosaic)
        axes[0, 1].set_title(f'Mosaic\n{mosaic.shape[1]}×{mosaic.shape[0]}')
        axes[0, 1].axis('off')
        
        # Difference map
        if original.shape == mosaic.shape:
            diff_img = np.abs(original.astype(np.int16) - mosaic.astype(np.int16))
        else:
            original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
            diff_img = np.abs(original_resized.astype(np.int16) - mosaic.astype(np.int16))
        
        diff_enhanced = np.clip(diff_img * 3, 0, 255).astype(np.uint8)
        axes[0, 2].imshow(diff_enhanced)
        axes[0, 2].set_title('Difference Map\n(Enhanced 3x)')
        axes[0, 2].axis('off')
        
        # Global metrics display
        if metrics:
            global_metrics_text = (
                f"Global Structure:\n\n"
                f"SSIM: {metrics['ssim']:.4f}\n"
                f"Luminance: {metrics['global_luminance']:.4f}\n"
                f"Contrast: {metrics['global_contrast']:.4f}\n"
                f"Structure: {metrics['global_structure']:.4f}\n\n"
                f"Global Color:\n"
                f"Similarity: {metrics['global_color_similarity']:.4f}\n"
                f"Histogram: {metrics['histogram_similarity']:.4f}"
            )
            
            basic_metrics_text = (
                f"Basic Metrics:\n\n"
                f"MSE: {metrics['mse']:.1f}\n"
                f"PSNR: {metrics['psnr']:.1f} dB\n\n"
                f"Overall Score:\n{metrics['overall_quality']:.1f}/100\n\n"
                f"Assessment:\n"
                f"{'Excellent' if metrics['overall_quality'] >= 85 else 'Very Good' if metrics['overall_quality'] >= 75 else 'Good' if metrics['overall_quality'] >= 65 else 'Fair'}"
            )
            
            axes[1, 0].text(0.05, 0.95, global_metrics_text, fontsize=11, fontfamily='monospace', 
                           va='top', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Global Structural Metrics')
            axes[1, 0].axis('off')
            
            axes[1, 1].text(0.05, 0.95, basic_metrics_text, fontsize=12, fontfamily='monospace', 
                           va='top', transform=axes[1, 1].transAxes, weight='bold')
            axes[1, 1].set_title('Overall Assessment')
            axes[1, 1].axis('off')
        
        # Global metrics visualization
        global_metrics = ['SSIM', 'Luminance', 'Contrast', 'Structure', 'Color', 'Histogram']
        global_values = [
            metrics['ssim'],
            metrics['global_luminance'],
            metrics['global_contrast'], 
            metrics['global_structure'],
            metrics['global_color_similarity'],
            metrics['histogram_similarity']
        ]
        
        # Normalize values for visualization (handle negative SSIM values)
        normalized_values = [(val + 1) / 2 if val < 0 else val for val in global_values]
        
        colors = ['navy', 'darkblue', 'mediumblue', 'blue', 'cornflowerblue', 'lightblue']
        bars = axes[1, 2].bar(range(len(global_metrics)), normalized_values, color=colors, alpha=0.7)
        axes[1, 2].set_xticks(range(len(global_metrics)))
        axes[1, 2].set_xticklabels(global_metrics, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Similarity Score')
        axes[1, 2].set_title('Global Metrics Breakdown')
        axes[1, 2].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value, orig_value in zip(bars, normalized_values, global_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{orig_value:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
        
        plt.suptitle('Global Quality Assessment: Whole Image Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()