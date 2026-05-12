"""
Audio Clustering Module

This module implements K-means clustering specifically designed for grouping
synthesizer music tracks based on their extracted audio features. The clustering
approach is tailored for creative music analysis rather than generic machine learning.

The analytical approach follows these principles:
1. Use only musically meaningful features for clustering
2. Standardize features to ensure equal weighting
3. Automatically adjust cluster count based on dataset size
4. Provide interpretable cluster descriptions for composers

Key Concepts:
- Feature standardization: Ensures all features contribute equally to clustering
- Optimal cluster count: Balances granularity with interpretability
- Cluster interpretation: Translates technical metrics into musical meaning
- Similarity measurement: Uses Euclidean distance in standardized feature space

The clustering is particularly effective for:
- Grouping tracks with similar energy levels and moods
- Identifying distinct synthesis techniques or sound palettes
- Creating themed playlists or DJ sets
- Understanding compositional patterns across a body of work
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt


class AudioClusterer:
    """
    K-means clustering system optimized for synthesizer music analysis.
    
    This class implements intelligent clustering that considers the unique
    characteristics of electronic music. It handles feature selection,
    standardization, and automatic cluster count optimization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the audio clusterer.
        
        Args:
            random_state: Random seed for reproducible clustering results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.features_scaled = None
        self.feature_names = None
        self.cluster_labels = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare audio features for clustering by selecting and cleaning numeric features.
        
        This method filters the feature set to include only numeric features
        that are meaningful for clustering. It excludes categorical features
        and metadata that don't contribute to musical similarity.
        
        The feature selection process considers:
        1. Numeric data types only (floats, ints)
        2. Exclusion of metadata (filename, duration)
        3. Inclusion of all spectral, temporal, and harmonic features
        4. Handling of missing values and outliers
        
        Args:
            df: DataFrame containing extracted audio features
            
        Returns:
            DataFrame with cleaned numeric features suitable for clustering
        """
        # Select only numeric columns for clustering
        # This excludes string features like filename and key
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        clustering_features = df[numeric_columns].copy()
        
        # Remove features that aren't useful for clustering
        # Duration is excluded because it's more of a metadata than a musical characteristic
        # Filename-related columns are excluded as they're identifiers, not features
        columns_to_exclude = ['duration']
        clustering_features = clustering_features.drop(columns_to_exclude, axis=1, errors='ignore')
        
        # Handle missing values by filling with feature means
        # This is important for robust clustering when some features fail to extract
        clustering_features = clustering_features.fillna(clustering_features.mean())
        
        # Remove any remaining non-finite values (inf, -inf, nan)
        clustering_features = clustering_features.replace([np.inf, -np.inf], np.nan)
        clustering_features = clustering_features.fillna(clustering_features.mean())
        
        return clustering_features
    
    def determine_optimal_clusters(self, features: pd.DataFrame, max_clusters: int = 10) -> int:
        """
        Determine the optimal number of clusters using multiple methods.
        
        This method combines several approaches to find the best cluster count:
        1. Elbow method: Find the point where adding clusters gives diminishing returns
        2. Silhouette analysis: Maximize the quality of cluster separation
        3. Practical constraints: Ensure clusters are interpretable and useful
        
        The approach is designed for creative music analysis where too many clusters
        become difficult to interpret, while too few clusters lose meaningful distinctions.
        
        Args:
            features: Prepared features for clustering
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        n_samples = len(features)
        
        # Ensure we don't have more clusters than samples
        max_clusters = min(max_clusters, n_samples)
        
        # For very small datasets, use simple rules
        if n_samples <= 2:
            return 1
        elif n_samples <= 5:
            return min(2, n_samples)
        
        # Standardize features for evaluation
        features_scaled = self.scaler.fit_transform(features)
        
        # Calculate metrics for different cluster counts
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            # Fit K-means with k clusters
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate inertia (within-cluster sum of squares)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score (quality of clustering)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal k using elbow method
        # Look for the point where the rate of decrease in inertia slows down
        if len(inertias) > 1:
            # Calculate second derivative to find elbow point
            deltas = np.diff(inertias)
            second_deltas = np.diff(deltas)
            
            # Find the elbow point (where second derivative is maximum)
            if len(second_deltas) > 0:
                elbow_idx = np.argmax(second_deltas) + 2  # +2 because we started from k=2
            else:
                elbow_idx = 2
        else:
            elbow_idx = 2
        
        # Find optimal k using silhouette score
        if silhouette_scores:
            silhouette_idx = np.argmax(silhouette_scores) + 2  # +2 because we started from k=2
        else:
            silhouette_idx = 2
        
        # Combine both methods with a preference for interpretability
        # For small datasets, prefer fewer clusters
        # For larger datasets, balance between the two methods
        if n_samples <= 10:
            optimal_k = min(elbow_idx, silhouette_idx, 3)
        else:
            # Weight the two methods and consider practical limits
            optimal_k = min(max(elbow_idx, silhouette_idx), 5)
        
        return max(2, optimal_k)  # Ensure at least 2 clusters
    
    def perform_clustering(self, df: pd.DataFrame, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Perform K-means clustering on audio features.
        
        This method implements the complete clustering pipeline:
        1. Feature preparation and standardization
        2. Optimal cluster count determination (if not specified)
        3. K-means clustering with proper initialization
        4. Cluster validation and quality assessment
        
        The clustering process is designed to handle the unique characteristics
        of synthesizer music, including the wide range of feature values and
        the need for musically meaningful groupings.
        
        Args:
            df: DataFrame containing audio features
            n_clusters: Number of clusters (if None, automatically determined)
            
        Returns:
            Tuple containing:
            - Cluster labels for each track
            - Cluster centers in original feature space
            - List of feature names used for clustering
        """
        # Stage 1: Feature Preparation
        # Prepare features by selecting only numeric columns relevant for clustering
        features = self.prepare_features(df)
        self.feature_names = features.columns.tolist()
        
        # Stage 2: Cluster Count Determination
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(features)
        
        # Ensure we don't have more clusters than samples
        n_samples = len(features)
        if n_clusters > n_samples:
            n_clusters = max(1, n_samples)
            print(f"Warning: Reduced cluster count to {n_clusters} due to limited samples ({n_samples}).")
        
        # Stage 3: Feature Standardization
        # Standardize features to ensure equal weighting in clustering
        # This is crucial because features have different scales and units
        self.features_scaled = self.scaler.fit_transform(features)
        
        # Stage 4: K-means Clustering
        # Perform K-means with multiple initializations for stability
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state, 
            n_init=10,  # Multiple initializations for stability
            max_iter=300  # Sufficient iterations for convergence
        )
        
        # Final safety check: ensure no NaN values before clustering
        if np.isnan(self.features_scaled).any():
            print("Warning: NaN values detected before clustering. Filling with column means...")
            # Use sklearn's SimpleImputer as a final fallback
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            self.features_scaled = imputer.fit_transform(self.features_scaled)
        
        # Fit the clustering model and get cluster labels
        self.cluster_labels = self.kmeans.fit_predict(self.features_scaled)
        
        # Stage 5: Transform Cluster Centers Back to Original Space
        # Convert standardized cluster centers back to original feature space
        # This allows for interpretable cluster descriptions
        cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        return self.cluster_labels, cluster_centers, self.feature_names
    
    def analyze_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze and describe the characteristics of each cluster.
        
        This method provides detailed analysis of each cluster, translating
        technical metrics into musically meaningful descriptions. It focuses
        on characteristics that composers and producers care about.
        
        The analysis includes:
        1. Basic cluster statistics (size, member tracks)
        2. Musical characteristics (tempo, key, energy)
        3. Compositional patterns (phases, structure)
        4. Creative descriptors (mood, character)
        
        Args:
            df: Original DataFrame with all features
            cluster_labels: Cluster assignments for each track
            
        Returns:
            Dictionary with detailed cluster analysis
        """
        cluster_analysis = {}
        
        # Add cluster labels to the dataframe for analysis
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Analyze each cluster
        for cluster_id in sorted(np.unique(cluster_labels)):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Basic cluster information
            cluster_size = len(cluster_data)
            track_names = cluster_data['filename'].tolist()
            
            # Musical characteristics analysis
            # These metrics help understand the "musical personality" of each cluster
            
            # Tempo analysis - important for DJ sets and playlists
            avg_tempo = float(cluster_data['tempo'].mean())
            tempo_std = float(cluster_data['tempo'].std())
            
            # Duration analysis - helps understand track length patterns
            avg_duration = float(cluster_data['duration'].mean())
            duration_std = float(cluster_data['duration'].std())
            
            # Energy analysis - captures the "intensity" of the cluster
            avg_energy = float(cluster_data['rms_mean'].mean())
            energy_std = float(cluster_data['rms_mean'].std())
            
            # Brightness analysis - captures the "color" of the sound
            avg_brightness = float(cluster_data['spectral_centroid_mean'].mean())
            brightness_std = float(cluster_data['spectral_centroid_mean'].std())
            
            # Key analysis - important for harmonic compatibility
            # Support both 'key' (spec-compliant) and legacy 'detected_key' field name
            _key_col = 'key' if 'key' in cluster_data.columns else ('detected_key' if 'detected_key' in cluster_data.columns else None)
            if _key_col:
                key_counts = cluster_data[_key_col].value_counts()
                common_key = key_counts.index[0] if not key_counts.empty else 'Unknown'
                key_diversity = len(key_counts)
            else:
                common_key = 'Unknown'
                key_diversity = 0

            # Structural analysis - if phase information is available
            if 'num_phases' in cluster_data.columns:
                avg_phases = float(cluster_data['num_phases'].mean())
                has_climax_percent = float(cluster_data['has_climax'].sum() / cluster_size * 100) if 'has_climax' in cluster_data.columns else 0
                has_breakdown_percent = float(cluster_data['has_breakdown'].sum() / cluster_size * 100) if 'has_breakdown' in cluster_data.columns else 0
            else:
                avg_phases = 0
                has_climax_percent = 0
                has_breakdown_percent = 0
            
            # Creative descriptor analysis
            if 'primary_mood' in cluster_data.columns:
                mood_counts = cluster_data['primary_mood'].value_counts()
                dominant_mood = mood_counts.index[0] if not mood_counts.empty else 'Unknown'
                mood_diversity = len(mood_counts)
            else:
                dominant_mood = 'Unknown'
                mood_diversity = 0
            
            if 'primary_character' in cluster_data.columns:
                character_counts = cluster_data['primary_character'].value_counts()
                dominant_character = character_counts.index[0] if not character_counts.empty else 'Unknown'
                character_diversity = len(character_counts)
            else:
                dominant_character = 'Unknown'
                character_diversity = 0
            
            # Compile comprehensive cluster analysis
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                # Basic information
                'count': cluster_size,
                'files': track_names,
                
                # Musical characteristics
                'avg_tempo': avg_tempo,
                'tempo_std': tempo_std,
                'avg_duration': avg_duration,
                'duration_std': duration_std,
                'avg_energy': avg_energy,
                'energy_std': energy_std,
                'avg_brightness': avg_brightness,
                'brightness_std': brightness_std,
                
                # Harmonic characteristics
                'common_key': common_key,
                'key_diversity': key_diversity,
                
                # Structural characteristics
                'avg_phases': avg_phases,
                'has_climax_percent': has_climax_percent,
                'has_breakdown_percent': has_breakdown_percent,
                
                # Creative characteristics
                'dominant_mood': dominant_mood,
                'mood_diversity': mood_diversity,
                'dominant_character': dominant_character,
                'character_diversity': character_diversity,
                
                # Cluster quality metrics
                'homogeneity': self._calculate_cluster_homogeneity(cluster_data),
                'musical_coherence': self._assess_musical_coherence(cluster_data)
            }
        
        return cluster_analysis
    
    def _calculate_cluster_homogeneity(self, cluster_data: pd.DataFrame) -> float:
        """
        Calculate the homogeneity of a cluster based on feature variance.
        
        This metric measures how similar the tracks in a cluster are to each other.
        Lower values indicate more homogeneous clusters where tracks share
        similar characteristics.
        
        Args:
            cluster_data: DataFrame containing tracks in this cluster
            
        Returns:
            Homogeneity score (0.0 = perfectly homogeneous, higher = more diverse)
        """
        if len(cluster_data) <= 1:
            return 0.0
        
        # Select numeric features for homogeneity calculation
        numeric_features = cluster_data.select_dtypes(include=[np.number])
        
        # Calculate coefficient of variation for each feature
        # CV = std / mean, normalized measure of variability
        cv_values = []
        for column in numeric_features.columns:
            if column in ['duration', 'filename']:  # Skip metadata
                continue
            
            values = numeric_features[column].dropna()
            if len(values) > 1 and values.mean() != 0:
                cv = values.std() / abs(values.mean())
                cv_values.append(cv)
        
        # Return mean coefficient of variation
        return np.mean(cv_values) if cv_values else 0.0
    
    def _assess_musical_coherence(self, cluster_data: pd.DataFrame) -> str:
        """
        Assess the musical coherence of a cluster using creative descriptors.
        
        This method evaluates whether the tracks in a cluster make sense
        together from a musical perspective, considering tempo, key, mood,
        and character compatibility.
        
        Args:
            cluster_data: DataFrame containing tracks in this cluster
            
        Returns:
            String describing the musical coherence level
        """
        if len(cluster_data) <= 1:
            return "Single Track"
        
        coherence_score = 0
        max_score = 0
        
        # Tempo coherence - check if tempos are compatible
        if 'tempo' in cluster_data.columns:
            tempo_range = cluster_data['tempo'].max() - cluster_data['tempo'].min()
            if tempo_range < 20:  # Within 20 BPM
                coherence_score += 2
            elif tempo_range < 40:  # Within 40 BPM
                coherence_score += 1
            max_score += 2
        
        # Key coherence - check if keys are compatible
        # Support both 'key' (spec-compliant) and legacy 'detected_key' field name
        _key_col_coh = 'key' if 'key' in cluster_data.columns else ('detected_key' if 'detected_key' in cluster_data.columns else None)
        if _key_col_coh:
            unique_keys = cluster_data[_key_col_coh].nunique()
            if unique_keys <= 2:  # At most 2 different keys
                coherence_score += 2
            elif unique_keys <= 3:  # At most 3 different keys
                coherence_score += 1
            max_score += 2
        
        # Mood coherence - check if moods are compatible
        if 'primary_mood' in cluster_data.columns:
            unique_moods = cluster_data['primary_mood'].nunique()
            if unique_moods <= 2:  # At most 2 different moods
                coherence_score += 2
            elif unique_moods <= 3:  # At most 3 different moods
                coherence_score += 1
            max_score += 2
        
        # Energy coherence - check if energy levels are similar
        if 'rms_mean' in cluster_data.columns:
            energy_cv = cluster_data['rms_mean'].std() / cluster_data['rms_mean'].mean()
            if energy_cv < 0.2:  # Low coefficient of variation
                coherence_score += 2
            elif energy_cv < 0.5:  # Moderate coefficient of variation
                coherence_score += 1
            max_score += 2
        
        # Calculate coherence percentage
        if max_score > 0:
            coherence_percent = (coherence_score / max_score) * 100
            
            if coherence_percent >= 75:
                return "High Coherence"
            elif coherence_percent >= 50:
                return "Moderate Coherence"
            else:
                return "Low Coherence"
        
        return "Unable to Assess"
    
    def visualize_clusters_2d(self, df: pd.DataFrame, cluster_labels: np.ndarray, 
                             save_path: Optional[str] = None, show_plot: bool = True) -> None:
        """
        Create a 2D visualization of clusters using PCA.
        
        This method projects the high-dimensional feature space into 2D
        using Principal Component Analysis, allowing for visual inspection
        of cluster quality and separation.
        
        Args:
            df: Original DataFrame with features
            cluster_labels: Cluster assignments
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
        """
        if self.features_scaled is None:
            print("Error: No scaled features available. Run clustering first.")
            return
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features_scaled)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.7, s=100)
        plt.colorbar(scatter, label='Cluster')
        
        # Add labels and title
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Audio Clusters in 2D PCA Space')
        
        # Add file names as annotations
        for i, filename in enumerate(df['filename']):
            plt.annotate(filename[:15], (features_pca[i, 0], features_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_cluster_recommendations(self, cluster_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate creative recommendations for each cluster.
        
        This method provides actionable suggestions for how to use each cluster
        in creative contexts like playlist creation, DJ sets, or composition.
        
        Args:
            cluster_analysis: Analysis results from analyze_clusters()
            
        Returns:
            Dictionary mapping cluster names to recommendation strings
        """
        recommendations = {}
        
        for cluster_name, analysis in cluster_analysis.items():
            recommendations_list = []
            
            # Tempo-based recommendations
            if analysis['avg_tempo'] > 140:
                recommendations_list.append("High-energy tracks suitable for peak-time sets")
            elif analysis['avg_tempo'] > 120:
                recommendations_list.append("Mid-tempo tracks perfect for building energy")
            else:
                recommendations_list.append("Ambient/downtempo tracks ideal for chill sessions")
            
            # Energy-based recommendations
            if analysis['avg_energy'] > 0.1:
                recommendations_list.append("High-energy cluster suitable for dance floors")
            elif analysis['avg_energy'] > 0.05:
                recommendations_list.append("Moderate energy - versatile for various contexts")
            else:
                recommendations_list.append("Low energy - perfect for background or meditation")
            
            # Structural recommendations
            if analysis['has_climax_percent'] > 60:
                recommendations_list.append("Dynamic tracks with clear peaks - great for storytelling")
            
            if analysis['has_breakdown_percent'] > 60:
                recommendations_list.append("Tracks with breakdowns - excellent for DJ mixing")
            
            # Coherence-based recommendations
            if analysis['musical_coherence'] == "High Coherence":
                recommendations_list.append("Highly coherent cluster - perfect for seamless playlists")
            
            recommendations[cluster_name] = ". ".join(recommendations_list)
        
        return recommendations