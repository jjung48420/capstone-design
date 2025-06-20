import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import random
from pathlib import Path
import torch

class SimpleNoiseAnalyzer:
    def __init__(self, feature_dirs):
        # Simple analyzer for comparing (foggy-clear) vs (rainy-clear) noise
        self.feature_dirs = feature_dirs
        self.clear_features = None
        self.foggy_features = None
        self.rainy_features = None
        self.foggy_noise = None
        self.rainy_noise = None
        
    def load_features(self, max_samples=1500):
        # Load features from .npy files (clear, foggy, rainy)
        clear_files = list(Path(self.feature_dirs['clear']).glob('*.npy'))
        if len(clear_files) > max_samples:
            clear_files = random.sample(clear_files, max_samples)
        clear_features = []
        for npy_file in tqdm(clear_files, desc="Processing clear"):
            feature_map = np.load(npy_file)
            if len(feature_map.shape) == 4:
                feature_map = feature_map.squeeze(0)
            processed = np.mean(feature_map, axis=(1, 2))
            clear_features.append(processed)
        self.clear_features = np.array(clear_features)

        foggy_files = list(Path(self.feature_dirs['foggy']).glob('*.npy'))
        if len(foggy_files) > max_samples:
            foggy_files = random.sample(foggy_files, max_samples)
        foggy_features = []
        for npy_file in tqdm(foggy_files, desc="Processing foggy"):
            feature_map = np.load(npy_file)
            if len(feature_map.shape) == 4:
                feature_map = feature_map.squeeze(0)
            processed = np.mean(feature_map, axis=(1, 2))
            foggy_features.append(processed)
        self.foggy_features = np.array(foggy_features)

        rainy_files = list(Path(self.feature_dirs['rainy']).glob('*.npy'))
        if len(rainy_files) > max_samples:
            rainy_files = random.sample(rainy_files, max_samples)
        rainy_features = []
        for npy_file in tqdm(rainy_files, desc="Processing rainy"):
            feature_map = np.load(npy_file)
            if len(feature_map.shape) == 4:
                feature_map = feature_map.squeeze(0)
            processed = np.mean(feature_map, axis=(1, 2))
            rainy_features.append(processed)
        self.rainy_features = np.array(rainy_features)
        
    def compute_noise_vectors(self):
        # Compute noise as simple subtraction: adverse_weather - clear
        min_samples = min(len(self.clear_features), len(self.foggy_features), len(self.rainy_features))
        self.foggy_noise = self.foggy_features[:min_samples] - self.clear_features[:min_samples]
        self.rainy_noise = self.rainy_features[:min_samples] - self.clear_features[:min_samples]
        self.combined_noise = np.vstack([self.foggy_noise, self.rainy_noise])
        self.noise_labels = np.concatenate([
            np.zeros(len(self.foggy_noise)),
            np.ones(len(self.rainy_noise))
        ])
        
    def analyze_noise_statistics(self):
        # Compute basic statistics about the noise vectors
        foggy_magnitudes = np.linalg.norm(self.foggy_noise, axis=1)
        rainy_magnitudes = np.linalg.norm(self.rainy_noise, axis=1)
        foggy_mean = np.mean(self.foggy_noise, axis=0)
        rainy_mean = np.mean(self.rainy_noise, axis=0)
        cos_sim = cosine_similarity([foggy_mean], [rainy_mean])[0, 0]
        foggy_abs_mean = np.mean(np.abs(self.foggy_noise), axis=0)
        rainy_abs_mean = np.mean(np.abs(self.rainy_noise), axis=0)
        channel_corr = np.corrcoef(foggy_abs_mean, rainy_abs_mean)[0, 1]
        top_foggy_channels = np.argsort(foggy_abs_mean)[-10:]
        top_rainy_channels = np.argsort(rainy_abs_mean)[-10:]
        overlap = len(set(top_foggy_channels) & set(top_rainy_channels))
        print(f"Foggy noise: mean={np.mean(foggy_magnitudes):.4f} ± {np.std(foggy_magnitudes):.4f}")
        print(f"Rainy noise: mean={np.mean(rainy_magnitudes):.4f} ± {np.std(rainy_magnitudes):.4f}")
        print(f"Cosine similarity: {cos_sim:.4f}, Channel corr: {channel_corr:.4f}, Top 10 overlap: {overlap}/10")
        return {
            'foggy_magnitudes': foggy_magnitudes,
            'rainy_magnitudes': rainy_magnitudes,
            'cosine_similarity': cos_sim,
            'channel_correlation': channel_corr,
            'channel_overlap': overlap
        }
        
    def run_tsne(self, perplexity=30, max_iter=1000, use_pca=True, n_components_pca=50):
        # Run t-SNE on the noise vectors
        if use_pca and self.combined_noise.shape[1] > n_components_pca:
            pca = PCA(n_components=n_components_pca, random_state=42)
            noise_data = pca.fit_transform(self.combined_noise)
        else:
            noise_data = self.combined_noise.copy()
        scaler = StandardScaler()
        noise_data = scaler.fit_transform(noise_data)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            learning_rate='auto',
            random_state=42,
            verbose=0
        )
        self.tsne_results = tsne.fit_transform(noise_data)
        
    def visualize_noise_comparison(self, save_path='noise_comparison.png'):
        # Create visualization comparing the two noise types
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax1 = axes[0, 0]
        foggy_mask = self.noise_labels == 0
        rainy_mask = self.noise_labels == 1
        ax1.scatter(self.tsne_results[foggy_mask, 0], self.tsne_results[foggy_mask, 1],
                   c='#e74c3c', label='Foggy Noise', alpha=0.6, s=20)
        ax1.scatter(self.tsne_results[rainy_mask, 0], self.tsne_results[rainy_mask, 1],
                   c='#3498db', label='Rainy Noise', alpha=0.6, s=20)
        ax1.set_title('t-SNE: Noise Vector Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        foggy_magnitudes = np.linalg.norm(self.foggy_noise, axis=1)
        rainy_magnitudes = np.linalg.norm(self.rainy_noise, axis=1)
        ax2.hist(foggy_magnitudes, bins=40, alpha=0.7, color='#e74c3c', 
                label=f'Foggy (μ={np.mean(foggy_magnitudes):.3f})', density=True)
        ax2.hist(rainy_magnitudes, bins=40, alpha=0.7, color='#3498db', 
                label=f'Rainy (μ={np.mean(rainy_magnitudes):.3f})', density=True)
        ax2.set_title('Noise Magnitude Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Noise Vector Magnitude')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        foggy_impact = np.mean(np.abs(self.foggy_noise), axis=0)
        rainy_impact = np.mean(np.abs(self.rainy_noise), axis=0)
        ax3.scatter(foggy_impact, rainy_impact, alpha=0.6, s=15, c='gray')
        max_val = max(np.max(foggy_impact), np.max(rainy_impact))
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal Impact')
        corr_coef = np.corrcoef(foggy_impact, rainy_impact)[0, 1]
        ax3.set_title(f'Channel Impact Correlation (r={corr_coef:.3f})', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Foggy Channel Impact')
        ax3.set_ylabel('Rainy Channel Impact')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        foggy_mean = np.mean(self.foggy_noise, axis=0)
        rainy_mean = np.mean(self.rainy_noise, axis=0)
        channels = np.arange(min(50, len(foggy_mean)))
        ax4.plot(channels, foggy_mean[:len(channels)], 'o-', color='#e74c3c', 
                label='Foggy Mean', alpha=0.7, markersize=3)
        ax4.plot(channels, rainy_mean[:len(channels)], 'o-', color='#3498db', 
                label='Rainy Mean', alpha=0.7, markersize=3)
        cos_sim = cosine_similarity([foggy_mean], [rainy_mean])[0, 0]
        ax4.set_title(f'Mean Noise Patterns (cos_sim={cos_sim:.3f})', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Channel Index')
        ax4.set_ylabel('Mean Noise Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to: {save_path}")
        plt.close()
        
    def analyze_separability(self):
        # Quantify how separable the two noise types are
        foggy_mask = self.noise_labels == 0
        rainy_mask = self.noise_labels == 1
        foggy_centroid = np.mean(self.tsne_results[foggy_mask], axis=0)
        rainy_centroid = np.mean(self.tsne_results[rainy_mask], axis=0)
        centroid_distance = np.linalg.norm(foggy_centroid - rainy_centroid)
        foggy_spread = np.mean([np.linalg.norm(point - foggy_centroid) 
                               for point in self.tsne_results[foggy_mask]])
        rainy_spread = np.mean([np.linalg.norm(point - rainy_centroid) 
                               for point in self.tsne_results[rainy_mask]])
        separability_ratio = centroid_distance / (foggy_spread + rainy_spread)
        print(f"Centroid distance: {centroid_distance:.2f}, Foggy spread: {foggy_spread:.2f}, Rainy spread: {rainy_spread:.2f}, Separability ratio: {separability_ratio:.2f}")
        return separability_ratio

def main():
    # Update these paths to your feature directories
    feature_dirs = {
        'clear': '/data/jun3700/feature_maps/origin/',
        'foggy': '/data/jun3700/feature_maps/foggy/',
        'rainy': '/data/jun3700/feature_maps/rainy/'
    }
    analyzer = SimpleNoiseAnalyzer(feature_dirs)
    analyzer.load_features(max_samples=1500)
    analyzer.compute_noise_vectors()
    stats = analyzer.analyze_noise_statistics()
    analyzer.run_tsne(perplexity=30, max_iter=1000, use_pca=True, n_components_pca=50)
    analyzer.visualize_noise_comparison(save_path='noise_comparison_analysis.png')
    separability = analyzer.analyze_separability()
    print(f"Cosine similarity: {stats['cosine_similarity']:.3f}, Channel corr: {stats['channel_correlation']:.3f}, Separability: {separability:.2f}")

if __name__ == "__main__":
    main()
