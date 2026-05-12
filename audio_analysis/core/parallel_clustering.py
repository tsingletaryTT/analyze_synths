"""
Parallel Clustering Module

This module implements distributed clustering algorithms optimized for
parallel processing and hardware acceleration, including future Tenstorrent
processor support.

Key Features:
1. Distributed K-means clustering across multiple cores/devices
2. Tensor-optimized clustering operations
3. Memory-efficient processing for large datasets
4. Hierarchical clustering for improved scalability
5. Hardware-agnostic clustering interface

The implementation separates clustering logic from hardware-specific
optimizations, allowing for easy adaptation to different processing units.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import time
from pathlib import Path

# Import tensor operations
from .tensor_operations import TensorBatch, TensorProcessor, TensorProcessorFactory

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for parallel clustering operations."""
    max_workers: int = field(default_factory=lambda: mp.cpu_count())
    batch_size: int = 1000
    use_mini_batch: bool = True
    mini_batch_size: int = 100
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    device: str = "cpu"
    enable_tensor_optimization: bool = True
    memory_limit_mb: int = 2048
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_workers <= 0:
            self.max_workers = mp.cpu_count()
        if self.batch_size <= 0:
            self.batch_size = 1000
        if self.mini_batch_size <= 0:
            self.mini_batch_size = 100


@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    feature_names: List[str]
    n_clusters: int
    inertia: float
    silhouette_score: float
    calinski_harabasz_score: float
    processing_time: float
    convergence_iterations: int
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive clustering statistics."""
        unique_labels = np.unique(self.cluster_labels)
        cluster_sizes = [np.sum(self.cluster_labels == label) for label in unique_labels]
        
        return {
            'n_clusters': self.n_clusters,
            'n_samples': len(self.cluster_labels),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'avg_cluster_size': np.mean(cluster_sizes),
            'cluster_size_std': np.std(cluster_sizes),
            'inertia': self.inertia,
            'silhouette_score': self.silhouette_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'processing_time': self.processing_time,
            'convergence_iterations': self.convergence_iterations
        }


class ParallelKMeansClusterer:
    """
    Parallel K-means clustering implementation optimized for audio features.
    
    This class implements distributed K-means clustering with support for
    different hardware backends and memory-efficient processing.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize parallel K-means clusterer.
        
        Args:
            config: Clustering configuration parameters
        """
        self.config = config or ClusteringConfig()
        self.scaler = StandardScaler()
        self.tensor_processor = None
        
        # Initialize tensor processor if enabled
        if self.config.enable_tensor_optimization:
            self.tensor_processor = TensorProcessorFactory.create_processor(
                device=self.config.device
            )
        
        logger.info(f"Initialized ParallelKMeansClusterer with {self.config.max_workers} workers")
    
    def fit_predict(self, features: pd.DataFrame, 
                   n_clusters: Optional[int] = None) -> ClusteringResult:
        """
        Fit K-means clustering and predict cluster labels.
        
        This method implements parallel K-means clustering with automatic
        cluster count determination and comprehensive result analysis.
        
        Args:
            features: DataFrame with numeric features
            n_clusters: Number of clusters (None = auto-determine)
            
        Returns:
            ClusteringResult with comprehensive clustering information
        """
        start_time = time.time()
        
        # Stage 1: Feature preparation
        logger.info("Preparing features for clustering...")
        prepared_features = self._prepare_features(features)
        
        if prepared_features.empty:
            raise ValueError("No valid features available for clustering")
        
        # Stage 2: Feature standardization
        logger.info("Standardizing features...")
        standardized_features = self._standardize_features(prepared_features)
        
        # Stage 3: Optimal cluster count determination
        if n_clusters is None:
            logger.info("Determining optimal cluster count...")
            n_clusters = self._determine_optimal_clusters(standardized_features)
        
        logger.info(f"Clustering with {n_clusters} clusters...")
        
        # Stage 4: Parallel clustering
        clustering_result = self._perform_parallel_clustering(
            standardized_features, n_clusters, prepared_features.columns.tolist()
        )
        
        # Stage 5: Post-processing and validation
        processing_time = time.time() - start_time
        clustering_result.processing_time = processing_time
        
        logger.info(f"Clustering completed in {processing_time:.2f}s")
        
        return clustering_result
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for clustering by selecting numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned numeric features
        """
        # Select only numeric columns
        numeric_features = df.select_dtypes(include=[np.number]).copy()
        
        # Remove metadata columns
        columns_to_exclude = ['duration', 'sample_rate']
        numeric_features = numeric_features.drop(columns_to_exclude, axis=1, errors='ignore')
        
        # Handle missing values
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Remove infinite values
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Remove columns with zero variance
        zero_variance_cols = numeric_features.columns[numeric_features.var() == 0]
        if len(zero_variance_cols) > 0:
            logger.warning(f"Removing {len(zero_variance_cols)} zero-variance columns")
            numeric_features = numeric_features.drop(zero_variance_cols, axis=1)
        
        return numeric_features
    
    def _standardize_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Standardize features using parallel processing.
        
        Args:
            features: DataFrame with numeric features
            
        Returns:
            Standardized feature array
        """
        if self.config.enable_tensor_optimization and self.tensor_processor:
            # Use tensor-optimized standardization
            return self._standardize_features_tensor(features)
        else:
            # Use standard sklearn standardization
            return self.scaler.fit_transform(features)
    
    def _standardize_features_tensor(self, features: pd.DataFrame) -> np.ndarray:
        """
        Standardize features using tensor operations.
        
        Args:
            features: DataFrame with numeric features
            
        Returns:
            Standardized feature array
        """
        # Convert to numpy array
        feature_array = features.values.astype(np.float32)
        
        # Compute mean and std using tensor operations
        mean = np.mean(feature_array, axis=0, keepdims=True)
        std = np.std(feature_array, axis=0, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        # Standardize
        standardized = (feature_array - mean) / std
        
        # Store scaler parameters for inverse transform
        self.scaler.mean_ = mean.flatten()
        self.scaler.scale_ = std.flatten()
        
        return standardized
    
    def _determine_optimal_clusters(self, features: np.ndarray, 
                                  max_clusters: int = 10) -> int:
        """
        Determine optimal number of clusters using parallel evaluation.
        
        Args:
            features: Standardized feature array
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        n_samples = features.shape[0]
        max_clusters = min(max_clusters, n_samples // 2)
        
        if n_samples <= 2:
            return 1
        elif n_samples <= 5:
            return 2
        
        # Evaluate different cluster counts in parallel
        cluster_range = range(2, max_clusters + 1)
        
        if self.config.max_workers > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_k = {
                    executor.submit(self._evaluate_cluster_count, features, k): k 
                    for k in cluster_range
                }
                
                evaluation_results = {}
                for future in as_completed(future_to_k):
                    k = future_to_k[future]
                    try:
                        result = future.result()
                        evaluation_results[k] = result
                    except Exception as e:
                        logger.error(f"Error evaluating k={k}: {e}")
        else:
            # Sequential evaluation
            evaluation_results = {}
            for k in cluster_range:
                try:
                    result = self._evaluate_cluster_count(features, k)
                    evaluation_results[k] = result
                except Exception as e:
                    logger.error(f"Error evaluating k={k}: {e}")
        
        # Select optimal k based on evaluation results
        return self._select_optimal_k(evaluation_results, n_samples)
    
    def _evaluate_cluster_count(self, features: np.ndarray, k: int) -> Dict[str, float]:
        """
        Evaluate clustering quality for a specific number of clusters.
        
        Args:
            features: Standardized feature array
            k: Number of clusters
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Use MiniBatchKMeans for faster evaluation
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.config.random_state,
            batch_size=self.config.mini_batch_size,
            max_iter=self.config.max_iter // 2  # Faster evaluation
        )
        
        labels = kmeans.fit_predict(features)
        
        # Calculate evaluation metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        
        return {
            'inertia': inertia,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz
        }
    
    def _select_optimal_k(self, evaluation_results: Dict[int, Dict[str, float]], 
                         n_samples: int) -> int:
        """
        Select optimal number of clusters from evaluation results.
        
        Args:
            evaluation_results: Results from cluster evaluation
            n_samples: Number of samples
            
        Returns:
            Optimal number of clusters
        """
        if not evaluation_results:
            return 2
        
        # Extract metrics
        k_values = sorted(evaluation_results.keys())
        silhouette_scores = [evaluation_results[k]['silhouette_score'] for k in k_values]
        calinski_harabasz_scores = [evaluation_results[k]['calinski_harabasz_score'] for k in k_values]
        
        # Find best k using silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_k_silhouette = k_values[best_silhouette_idx]
        
        # Find best k using Calinski-Harabasz score
        best_calinski_idx = np.argmax(calinski_harabasz_scores)
        best_k_calinski = k_values[best_calinski_idx]
        
        # Combine results with preference for interpretability
        if n_samples <= 20:
            # For small datasets, prefer fewer clusters
            optimal_k = min(best_k_silhouette, best_k_calinski, 3)
        else:
            # For larger datasets, balance between metrics
            optimal_k = int(np.mean([best_k_silhouette, best_k_calinski]))
        
        return max(2, optimal_k)  # Ensure at least 2 clusters
    
    def _perform_parallel_clustering(self, features: np.ndarray, n_clusters: int, 
                                   feature_names: List[str]) -> ClusteringResult:
        """
        Perform parallel K-means clustering.
        
        Args:
            features: Standardized feature array
            n_clusters: Number of clusters
            feature_names: List of feature names
            
        Returns:
            ClusteringResult with clustering information
        """
        if self.config.enable_tensor_optimization and self.tensor_processor:
            # Use tensor-optimized clustering
            return self._cluster_with_tensor_processor(features, n_clusters, feature_names)
        else:
            # Use standard sklearn clustering
            return self._cluster_with_sklearn(features, n_clusters, feature_names)
    
    def _cluster_with_tensor_processor(self, features: np.ndarray, n_clusters: int,
                                     feature_names: List[str]) -> ClusteringResult:
        """
        Perform clustering using tensor processor.
        
        Args:
            features: Standardized feature array
            n_clusters: Number of clusters
            feature_names: List of feature names
            
        Returns:
            ClusteringResult with clustering information
        """
        # Use tensor processor for clustering
        labels, centers = self.tensor_processor.cluster_features(features, n_clusters)
        
        # Calculate quality metrics
        inertia = self._calculate_inertia(features, labels, centers)
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        
        return ClusteringResult(
            cluster_labels=labels,
            cluster_centers=centers,
            feature_names=feature_names,
            n_clusters=n_clusters,
            inertia=inertia,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            processing_time=0.0,  # Will be set by caller
            convergence_iterations=0  # Not available from tensor processor
        )
    
    def _cluster_with_sklearn(self, features: np.ndarray, n_clusters: int,
                            feature_names: List[str]) -> ClusteringResult:
        """
        Perform clustering using sklearn implementation.
        
        Args:
            features: Standardized feature array
            n_clusters: Number of clusters
            feature_names: List of feature names
            
        Returns:
            ClusteringResult with clustering information
        """
        # Choose clustering algorithm based on dataset size
        if features.shape[0] > self.config.batch_size and self.config.use_mini_batch:
            # Use MiniBatchKMeans for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.config.random_state,
                batch_size=self.config.mini_batch_size,
                max_iter=self.config.max_iter,
                tol=self.config.tol
            )
        else:
            # Use regular KMeans for smaller datasets
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
                tol=self.config.tol
            )
        
        # Fit clustering model
        labels = kmeans.fit_predict(features)
        
        # Calculate quality metrics
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        
        return ClusteringResult(
            cluster_labels=labels,
            cluster_centers=kmeans.cluster_centers_,
            feature_names=feature_names,
            n_clusters=n_clusters,
            inertia=kmeans.inertia_,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            processing_time=0.0,  # Will be set by caller
            convergence_iterations=getattr(kmeans, 'n_iter_', 0)
        )
    
    def _calculate_inertia(self, features: np.ndarray, labels: np.ndarray, 
                          centers: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        Args:
            features: Feature array
            labels: Cluster labels
            centers: Cluster centers
            
        Returns:
            Inertia value
        """
        inertia = 0.0
        
        for i in range(len(centers)):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                cluster_features = features[cluster_mask]
                center = centers[i]
                distances_squared = np.sum((cluster_features - center) ** 2, axis=1)
                inertia += np.sum(distances_squared)
        
        return inertia
    
    def analyze_clusters(self, df: pd.DataFrame, clustering_result: ClusteringResult) -> Dict[str, Any]:
        """
        Analyze clustering results and provide detailed cluster descriptions.
        
        Args:
            df: Original DataFrame with features
            clustering_result: Result from clustering operation
            
        Returns:
            Dictionary with detailed cluster analysis
        """
        cluster_analysis = {}
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clustering_result.cluster_labels
        
        # Analyze each cluster
        for cluster_id in range(clustering_result.n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Basic cluster information
            cluster_analysis[f'Cluster_{cluster_id}'] = self._analyze_single_cluster(
                cluster_data, cluster_id, clustering_result
            )
        
        return cluster_analysis
    
    def _analyze_single_cluster(self, cluster_data: pd.DataFrame, cluster_id: int,
                               clustering_result: ClusteringResult) -> Dict[str, Any]:
        """
        Analyze a single cluster in detail.
        
        Args:
            cluster_data: Data for this cluster
            cluster_id: Cluster identifier
            clustering_result: Overall clustering result
            
        Returns:
            Dictionary with cluster analysis
        """
        cluster_size = len(cluster_data)
        
        # Basic information
        analysis = {
            'cluster_id': cluster_id,
            'size': cluster_size,
            'percentage': (cluster_size / len(clustering_result.cluster_labels)) * 100
        }
        
        # File information
        if 'filename' in cluster_data.columns:
            analysis['files'] = cluster_data['filename'].tolist()
        
        # Feature analysis
        numeric_features = cluster_data.select_dtypes(include=[np.number])
        
        # Musical characteristics
        if 'tempo' in numeric_features.columns:
            analysis['avg_tempo'] = float(numeric_features['tempo'].mean())
            analysis['tempo_std'] = float(numeric_features['tempo'].std())
        
        if 'rms_mean' in numeric_features.columns:
            analysis['avg_energy'] = float(numeric_features['rms_mean'].mean())
            analysis['energy_std'] = float(numeric_features['rms_mean'].std())
        
        if 'spectral_centroid_mean' in numeric_features.columns:
            analysis['avg_brightness'] = float(numeric_features['spectral_centroid_mean'].mean())
            analysis['brightness_std'] = float(numeric_features['spectral_centroid_mean'].std())
        
        # Key analysis — support both 'key' (spec-compliant) and legacy 'detected_key'
        _key_col = 'key' if 'key' in cluster_data.columns else ('detected_key' if 'detected_key' in cluster_data.columns else None)
        if _key_col:
            key_counts = cluster_data[_key_col].value_counts()
            analysis['common_key'] = key_counts.index[0] if not key_counts.empty else 'Unknown'
            analysis['key_diversity'] = len(key_counts)
        
        # Mood analysis
        if 'primary_mood' in cluster_data.columns:
            mood_counts = cluster_data['primary_mood'].value_counts()
            analysis['dominant_mood'] = mood_counts.index[0] if not mood_counts.empty else 'Unknown'
            analysis['mood_diversity'] = len(mood_counts)
        
        # Cluster quality metrics
        analysis['homogeneity'] = self._calculate_cluster_homogeneity(numeric_features)
        analysis['cohesion'] = self._calculate_cluster_cohesion(
            cluster_data, cluster_id, clustering_result
        )
        
        return analysis
    
    def _calculate_cluster_homogeneity(self, cluster_features: pd.DataFrame) -> float:
        """
        Calculate homogeneity of a cluster based on feature variance.
        
        Args:
            cluster_features: Numeric features for the cluster
            
        Returns:
            Homogeneity score (lower = more homogeneous)
        """
        if len(cluster_features) <= 1:
            return 0.0
        
        # Calculate coefficient of variation for each feature
        cv_values = []
        for column in cluster_features.columns:
            if column in ['duration', 'sample_rate', 'cluster']:
                continue
            
            values = cluster_features[column].dropna()
            if len(values) > 1 and values.mean() != 0:
                cv = values.std() / abs(values.mean())
                cv_values.append(cv)
        
        return np.mean(cv_values) if cv_values else 0.0
    
    def _calculate_cluster_cohesion(self, cluster_data: pd.DataFrame, cluster_id: int,
                                  clustering_result: ClusteringResult) -> float:
        """
        Calculate cohesion of a cluster based on distance to cluster center.
        
        Args:
            cluster_data: Data for this cluster
            cluster_id: Cluster identifier
            clustering_result: Overall clustering result
            
        Returns:
            Cohesion score (lower = more cohesive)
        """
        if len(cluster_data) <= 1:
            return 0.0
        
        # Get cluster center
        cluster_center = clustering_result.cluster_centers[cluster_id]
        
        # Calculate distances from cluster center
        numeric_features = cluster_data.select_dtypes(include=[np.number])
        
        # Remove non-feature columns
        feature_columns = [col for col in numeric_features.columns 
                          if col not in ['duration', 'sample_rate', 'cluster']]
        
        if not feature_columns:
            return 0.0
        
        cluster_features = numeric_features[feature_columns]
        
        # Standardize features (using stored scaler)
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            # Filter to features that were actually used in clustering
            available_features = [col for col in feature_columns 
                                if col in clustering_result.feature_names]
            
            if available_features:
                cluster_features = cluster_features[available_features]
                
                # Apply same standardization as used in clustering
                feature_indices = [clustering_result.feature_names.index(col) 
                                 for col in available_features]
                
                standardized_features = (cluster_features - self.scaler.mean_[feature_indices]) / self.scaler.scale_[feature_indices]
                
                # Calculate distances to cluster center
                center_subset = cluster_center[feature_indices]
                distances = np.linalg.norm(standardized_features - center_subset, axis=1)
                
                return float(np.mean(distances))
        
        return 0.0
    
    def get_clustering_recommendations(self, cluster_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate recommendations for each cluster.
        
        Args:
            cluster_analysis: Analysis results from analyze_clusters()
            
        Returns:
            Dictionary mapping cluster names to recommendation strings
        """
        recommendations = {}
        
        for cluster_name, analysis in cluster_analysis.items():
            recommendations_list = []
            
            # Size-based recommendations
            if analysis['size'] == 1:
                recommendations_list.append("Single track - consider for unique/special use")
            elif analysis['size'] >= 10:
                recommendations_list.append("Large cluster - suitable for extended playlists")
            
            # Tempo-based recommendations
            if 'avg_tempo' in analysis:
                if analysis['avg_tempo'] > 140:
                    recommendations_list.append("High-energy cluster - peak-time material")
                elif analysis['avg_tempo'] > 120:
                    recommendations_list.append("Mid-tempo cluster - versatile for building energy")
                else:
                    recommendations_list.append("Low-tempo cluster - ambient/downtempo material")
            
            # Homogeneity-based recommendations
            if analysis['homogeneity'] < 0.3:
                recommendations_list.append("Highly consistent cluster - seamless transitions")
            elif analysis['homogeneity'] > 0.7:
                recommendations_list.append("Diverse cluster - interesting variety")
            
            # Mood-based recommendations
            if 'dominant_mood' in analysis and analysis['dominant_mood'] != 'Unknown':
                recommendations_list.append(f"Dominated by {analysis['dominant_mood']} mood")
            
            recommendations[cluster_name] = ". ".join(recommendations_list) if recommendations_list else "General purpose cluster"
        
        return recommendations