"""
Visualization Utilities for Audio Analysis

This module provides comprehensive visualization functions for audio analysis
results. It creates publication-quality plots and charts that help understand
the musical characteristics and relationships in synthesizer music.

Key visualization types:
- Phase timeline plots showing musical structure
- Cluster analysis visualizations with PCA projections
- Feature distribution plots and correlation matrices
- Sequence recommendation visualizations
- Mood and character analysis charts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

# Configure matplotlib for better-looking plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class Visualizer:
    """
    Comprehensive visualization system for audio analysis results.
    
    This class provides methods for creating various types of visualizations
    that help understand and communicate the results of audio analysis.
    All plots are designed to be publication-quality and informative.
    """
    
    def __init__(self, style: str = 'default', color_palette: str = 'viridis'):
        """
        Initialize the visualizer with style settings.
        
        Args:
            style: matplotlib style to use
            color_palette: seaborn color palette
        """
        self.style = style
        self.color_palette = color_palette
        
        # Set up plotting style
        if style != 'default':
            plt.style.use(style)
        
        # Define color schemes for different plot types
        self.phase_colors = {
            'Intro/Ambient': '#E8F4FD',
            'Introduction': '#4A90E2',
            'Build-up/Energetic': '#F5A623',
            'Rhythmic/Percussive': '#D0021B',
            'Bright/Melodic': '#F8E71C',
            'Climax/Peak': '#8B0000',
            'Development': '#7ED321',
            'Breakdown/Quiet': '#D3D3D3',
            'Conclusion': '#9013FE',
            'Outro/Fade': '#1E3A8A'
        }
        
        # Set up seaborn style
        sns.set_palette(color_palette)
        
    def create_phase_timeline(self, phase_data: List[Dict[str, Any]], 
                            save_path: Optional[Path] = None,
                            show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create a timeline visualization showing musical phases for each track.
        
        This visualization shows the structural analysis of each track,
        displaying how the music evolves through different phases over time.
        Each phase is color-coded by type and shows duration and characteristics.
        
        Args:
            phase_data: List of phase analysis results for each file
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object or None
        """
        if not phase_data:
            print("No phase data provided")
            return None
        
        # Set up the figure
        fig, axes = plt.subplots(
            len(phase_data), 1, 
            figsize=(16, 4 * len(phase_data)),
            squeeze=False
        )
        
        # Handle single file case
        if len(phase_data) == 1:
            axes = [axes[0]]
        else:
            axes = axes.flatten()
        
        # Create timeline for each file
        for i, file_info in enumerate(phase_data):
            ax = axes[i]
            self._plot_single_phase_timeline(ax, file_info)
        
        # Add overall title
        fig.suptitle('Musical Phase Timeline Analysis', fontsize=16, fontweight='bold')
        
        # Create legend
        self._add_phase_legend(fig)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_single_phase_timeline(self, ax: plt.Axes, file_info: Dict[str, Any]):
        """
        Plot timeline for a single file.
        
        Args:
            ax: matplotlib Axes object
            file_info: Phase information for the file
        """
        filename = file_info['filename']
        total_duration = file_info['total_duration']
        phases = file_info['phases']
        
        # Plot each phase as a colored bar
        for phase in phases:
            start_time = phase['start_time']
            duration = phase['duration']
            phase_type = phase['phase_type']
            
            # Get color for this phase type
            color = self.phase_colors.get(phase_type, '#808080')
            
            # Draw phase bar
            ax.barh(
                0, duration, left=start_time, height=0.6,
                color=color, alpha=0.8, edgecolor='black', linewidth=1
            )
            
            # Add phase label
            mid_time = start_time + duration / 2
            label_text = f"P{phase['phase_number']}\n{phase_type.split('/')[0]}"
            ax.text(
                mid_time, 0, label_text,
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        
        # Format axes
        ax.set_xlim(0, total_duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_title(f"{filename} - {len(phases)} phases", fontsize=12, fontweight='bold')
        ax.set_yticks([])
        
        # Add time markers for longer tracks
        if total_duration > 120:  # 2 minutes
            time_markers = np.arange(0, total_duration, 60)  # Every minute
            ax.set_xticks(time_markers)
            ax.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in time_markers])
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='x')
    
    def _add_phase_legend(self, fig: plt.Figure):
        """
        Add legend for phase types.
        
        Args:
            fig: matplotlib Figure object
        """
        # Create legend elements
        legend_elements = []
        for phase_type, color in self.phase_colors.items():
            legend_elements.append(
                patches.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, edgecolor='black')
            )
        
        # Add legend
        fig.legend(
            legend_elements, list(self.phase_colors.keys()),
            loc='upper center', bbox_to_anchor=(0.5, 0.02),
            ncol=5, fontsize=10, frameon=True
        )
    
    def create_cluster_visualization(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                                   features_scaled: np.ndarray, feature_names: List[str],
                                   save_path: Optional[Path] = None,
                                   show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create comprehensive cluster analysis visualization.
        
        This creates a multi-panel visualization showing:
        - PCA projection of clusters in 2D space
        - Feature comparison plots
        - Cluster characteristics
        
        Args:
            df: Original DataFrame with features
            cluster_labels: Cluster assignments
            features_scaled: Standardized features
            feature_names: Names of features used for clustering
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object or None
        """
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. PCA visualization (main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_pca_clusters(ax1, features_scaled, cluster_labels, df['filename'])
        
        # 2. Feature comparison plots
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_feature_comparison(ax2, df, cluster_labels, 'tempo', 'rms_mean')
        
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_feature_comparison(ax3, df, cluster_labels, 'spectral_centroid_mean', 'spectral_bandwidth_mean')
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_feature_comparison(ax4, df, cluster_labels, 'duration', 'tempo')
        
        # 3. Cluster characteristics
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_cluster_characteristics(ax5, df, cluster_labels)
        
        # 4. Key distribution
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_key_distribution(ax6, df, cluster_labels)
        
        # Add overall title
        fig.suptitle('Audio Cluster Analysis', fontsize=16, fontweight='bold')
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_pca_clusters(self, ax: plt.Axes, features_scaled: np.ndarray, 
                          cluster_labels: np.ndarray, filenames: pd.Series):
        """
        Plot PCA projection of clusters.
        
        Args:
            ax: matplotlib Axes object
            features_scaled: Standardized features
            cluster_labels: Cluster assignments
            filenames: File names for annotations
        """
        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # Create scatter plot
        scatter = ax.scatter(
            features_pca[:, 0], features_pca[:, 1],
            c=cluster_labels, cmap='viridis', alpha=0.7, s=100
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Add labels
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Clusters in PCA Space')
        
        # Add file name annotations
        for i, filename in enumerate(filenames):
            ax.annotate(
                filename[:15], (features_pca[i, 0], features_pca[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7
            )
        
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_comparison(self, ax: plt.Axes, df: pd.DataFrame, 
                                cluster_labels: np.ndarray, feature_x: str, feature_y: str):
        """
        Plot feature comparison colored by cluster.
        
        Args:
            ax: matplotlib Axes object
            df: DataFrame with features
            cluster_labels: Cluster assignments
            feature_x: X-axis feature name
            feature_y: Y-axis feature name
        """
        if feature_x not in df.columns or feature_y not in df.columns:
            ax.text(0.5, 0.5, f'Features {feature_x} or {feature_y} not available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        scatter = ax.scatter(
            df[feature_x], df[feature_y],
            c=cluster_labels, cmap='viridis', alpha=0.7
        )
        
        ax.set_xlabel(feature_x.replace('_', ' ').title())
        ax.set_ylabel(feature_y.replace('_', ' ').title())
        ax.set_title(f'{feature_x.replace("_", " ").title()} vs {feature_y.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
    
    def _plot_cluster_characteristics(self, ax: plt.Axes, df: pd.DataFrame, 
                                     cluster_labels: np.ndarray):
        """
        Plot cluster characteristics.
        
        Args:
            ax: matplotlib Axes object
            df: DataFrame with features
            cluster_labels: Cluster assignments
        """
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df[cluster_labels == cluster_id]
            cluster_stats.append({
                'cluster': cluster_id,
                'count': len(cluster_data),
                'avg_tempo': cluster_data['tempo'].mean() if 'tempo' in df.columns else 0,
                'avg_energy': cluster_data['rms_mean'].mean() if 'rms_mean' in df.columns else 0
            })
        
        # Create bar plot
        clusters = [stat['cluster'] for stat in cluster_stats]
        counts = [stat['count'] for stat in cluster_stats]
        
        bars = ax.bar(clusters, counts, color=plt.cm.viridis(np.linspace(0, 1, len(clusters))))
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Tracks')
        ax.set_title('Cluster Sizes')
        ax.set_xticks(clusters)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
    
    def _plot_key_distribution(self, ax: plt.Axes, df: pd.DataFrame, 
                              cluster_labels: np.ndarray):
        """
        Plot key distribution by cluster.
        
        Args:
            ax: matplotlib Axes object
            df: DataFrame with features
            cluster_labels: Cluster assignments
        """
        # Support both 'key' (spec-compliant) and legacy 'detected_key' field name
        _key_col = 'key' if 'key' in df.columns else ('detected_key' if 'detected_key' in df.columns else None)
        if _key_col is None:
            ax.text(0.5, 0.5, 'Key information not available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Create DataFrame for plotting
        df_plot = df.copy()
        df_plot['cluster'] = cluster_labels

        # Create crosstab
        key_cluster_counts = pd.crosstab(df_plot[_key_col], df_plot['cluster'])
        
        # Create stacked bar plot
        key_cluster_counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        ax.set_xlabel('Detected Key')
        ax.set_ylabel('Number of Tracks')
        ax.set_title('Key Distribution by Cluster')
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    def create_mood_distribution_plot(self, mood_data: Dict[str, Any],
                                     save_path: Optional[Path] = None,
                                     show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create visualization of mood distribution across tracks.
        
        Args:
            mood_data: Mood distribution analysis results
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object or None
        """
        if not mood_data or 'mood_counts' not in mood_data:
            print("No mood data provided")
            return None
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Mood count bar chart
        ax1 = axes[0, 0]
        moods = list(mood_data['mood_counts'].keys())
        counts = list(mood_data['mood_counts'].values())
        
        bars = ax1.bar(moods, counts, color=plt.cm.viridis(np.linspace(0, 1, len(moods))))
        ax1.set_title('Mood Distribution')
        ax1.set_xlabel('Mood')
        ax1.set_ylabel('Number of Tracks')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # 2. Mood percentage pie chart
        ax2 = axes[0, 1]
        if 'mood_percentages' in mood_data:
            percentages = list(mood_data['mood_percentages'].values())
            ax2.pie(percentages, labels=moods, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Mood Distribution (%)')
        
        # 3. Dominant moods
        ax3 = axes[1, 0]
        if 'dominant_moods' in mood_data:
            dominant_moods = mood_data['dominant_moods'][:5]  # Top 5
            dom_moods = [mood for mood, count in dominant_moods]
            dom_counts = [count for mood, count in dominant_moods]
            
            ax3.barh(dom_moods, dom_counts, color=plt.cm.plasma(np.linspace(0, 1, len(dom_moods))))
            ax3.set_title('Top 5 Dominant Moods')
            ax3.set_xlabel('Number of Tracks')
        
        # 4. Mood diversity metrics
        ax4 = axes[1, 1]
        metrics = {
            'Unique Moods': mood_data.get('unique_moods', 0),
            'Total Tracks': mood_data.get('total_tracks', 0),
            'Mood Diversity': mood_data.get('mood_diversity', 0) * 100
        }
        
        ax4.axis('off')
        y_pos = 0.8
        for metric, value in metrics.items():
            if metric == 'Mood Diversity':
                ax4.text(0.1, y_pos, f'{metric}: {value:.1f}%', fontsize=12, transform=ax4.transAxes)
            else:
                ax4.text(0.1, y_pos, f'{metric}: {value}', fontsize=12, transform=ax4.transAxes)
            y_pos -= 0.2
        
        ax4.set_title('Mood Analysis Metrics')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def create_sequence_visualization(self, sequence_data: List[Dict[str, Any]],
                                     save_path: Optional[Path] = None,
                                     show_plot: bool = False) -> Optional[plt.Figure]:
        """
        Create visualization of recommended song sequence.
        
        Args:
            sequence_data: Sequence recommendation results
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object or None
        """
        if not sequence_data:
            print("No sequence data provided")
            return None
        
        # Set up the figure
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Extract data
        positions = [item['position'] for item in sequence_data]
        tempos = [item['tempo'] for item in sequence_data]
        energies = [item['energy'] for item in sequence_data]
        filenames = [item['filename'] for item in sequence_data]
        
        # 1. Tempo progression
        ax1 = axes[0]
        ax1.plot(positions, tempos, marker='o', linewidth=2, markersize=8, color='blue')
        ax1.set_title('Tempo Progression Through Sequence')
        ax1.set_xlabel('Position in Sequence')
        ax1.set_ylabel('Tempo (BPM)')
        ax1.grid(True, alpha=0.3)
        
        # Add filename labels
        for i, (pos, tempo, filename) in enumerate(zip(positions, tempos, filenames)):
            if i % 2 == 0:  # Show every other label to avoid crowding
                ax1.annotate(filename[:15], (pos, tempo), 
                           xytext=(0, 10), textcoords='offset points',
                           fontsize=8, ha='center', rotation=45)
        
        # 2. Energy progression
        ax2 = axes[1]
        ax2.plot(positions, energies, marker='s', linewidth=2, markersize=8, color='red')
        ax2.set_title('Energy Progression Through Sequence')
        ax2.set_xlabel('Position in Sequence')
        ax2.set_ylabel('Energy Level')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sequence overview
        ax3 = axes[2]
        
        # Create a timeline view
        colors = plt.cm.viridis(np.linspace(0, 1, len(sequence_data)))
        
        for i, (item, color) in enumerate(zip(sequence_data, colors)):
            duration = item['duration']
            ax3.barh(0, duration, left=sum(seq['duration'] for seq in sequence_data[:i]), 
                    height=0.5, color=color, alpha=0.8, edgecolor='black')
            
            # Add track label
            start_pos = sum(seq['duration'] for seq in sequence_data[:i])
            mid_pos = start_pos + duration / 2
            ax3.text(mid_pos, 0, f"{item['position']}\n{item['filename'][:10]}", 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax3.set_title('Sequence Timeline')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('')
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Format x-axis for timeline
        total_duration = sum(item['duration'] for item in sequence_data)
        if total_duration > 600:  # More than 10 minutes
            time_markers = np.arange(0, total_duration, 300)  # Every 5 minutes
            ax3.set_xticks(time_markers)
            ax3.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in time_markers])
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig


def create_phase_timeline(phase_data: List[Dict[str, Any]], 
                         save_path: Optional[Path] = None,
                         show_plot: bool = False) -> Optional[plt.Figure]:
    """
    Standalone function to create phase timeline visualization.
    
    Args:
        phase_data: Phase analysis results
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object or None
    """
    visualizer = Visualizer()
    return visualizer.create_phase_timeline(phase_data, save_path, show_plot)


def create_cluster_plot(df: pd.DataFrame, cluster_labels: np.ndarray,
                       features_scaled: np.ndarray, feature_names: List[str],
                       save_path: Optional[Path] = None,
                       show_plot: bool = False) -> Optional[plt.Figure]:
    """
    Standalone function to create cluster visualization.
    
    Args:
        df: DataFrame with features
        cluster_labels: Cluster assignments
        features_scaled: Standardized features
        feature_names: Feature names
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object or None
    """
    visualizer = Visualizer()
    return visualizer.create_cluster_visualization(
        df, cluster_labels, features_scaled, feature_names, save_path, show_plot
    )