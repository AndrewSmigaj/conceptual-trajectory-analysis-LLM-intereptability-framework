"""
Centralized data loader for transformation analyses.
Provides consistent data loading, caching, and utilities.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)


class TransformationDataLoader:
    """
    Centralized data loader for all transformation analyses.
    
    Handles:
    - Trajectory loading from unified experiment results
    - Activation loading from pickle files
    - Token metadata loading (frequencies, types, labels)
    - Cluster model loading for mapping new data
    - Caching to avoid redundant I/O
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base path for experiments directory
        """
        if base_path is None:
            # Default to current experiment directory structure
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
            
        # Cache containers
        self._trajectories_cache = {}
        self._activations_cache = {}
        self._metadata_cache = {}
        self._cluster_models_cache = {}
        
        # Paths
        self.unified_results_path = self.base_path / "results_unified"
        self.token_data_path = self.base_path.parent / "all_tokens"
        
    @lru_cache(maxsize=1)
    def load_unified_trajectories(self, k: int = 10) -> Dict[str, Any]:
        """
        Load unified trajectories from experiment results.
        
        Args:
            k: Number of clusters used in the experiment
            
        Returns:
            Dictionary with trajectories and metadata
        """
        cache_key = f"trajectories_k{k}"
        if cache_key in self._trajectories_cache:
            return self._trajectories_cache[cache_key]
            
        trajectory_file = self.unified_results_path / f"unified_trajectories_k{k}.json"
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
            
        logger.info(f"Loading trajectories from {trajectory_file}")
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
            
        self._trajectories_cache[cache_key] = data
        return data
        
    def load_unified_activations(self) -> Dict[str, np.ndarray]:
        """
        Load unified activations from pickle file.
        
        Returns:
            Dictionary mapping case_idx to activation arrays
        """
        if 'activations' in self._activations_cache:
            return self._activations_cache['activations']
            
        activation_file = self.unified_results_path / "unified_activations.pkl"
        if not activation_file.exists():
            raise FileNotFoundError(f"Activation file not found: {activation_file}")
            
        logger.info(f"Loading activations from {activation_file}")
        with open(activation_file, 'rb') as f:
            activations = pickle.load(f)
            
        self._activations_cache['activations'] = activations
        return activations
        
    @lru_cache(maxsize=1)
    def load_token_frequencies(self) -> Dict[str, int]:
        """
        Load token frequency data from Brown corpus analysis.
        
        Returns:
            Dictionary mapping token ID to frequency count
        """
        freq_file = self.token_data_path / "gpt2_token_frequencies_brown.json"
        if not freq_file.exists():
            logger.warning(f"Frequency file not found: {freq_file}")
            return {}
            
        with open(freq_file, 'r') as f:
            data = json.load(f)
            
        return data.get('token_frequencies', {})
        
    @lru_cache(maxsize=1)
    def load_token_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load comprehensive token metadata.
        
        Returns:
            Dictionary mapping token ID to metadata dict
        """
        meta_file = self.token_data_path / "top_10k_tokens_full.json"
        if not meta_file.exists():
            logger.warning(f"Metadata file not found: {meta_file}")
            return {}
            
        with open(meta_file, 'r') as f:
            tokens_data = json.load(f)
            
        # Convert list to dict indexed by token ID
        metadata = {
            str(t['token_id']): t for t in tokens_data
        }
        
        return metadata
        
    @lru_cache(maxsize=1)
    def load_token_labels(self) -> Dict[str, Dict[str, Any]]:
        """
        Load comprehensive token labels (POS tags, semantic categories).
        
        Returns:
            Dictionary mapping token ID to label information
        """
        label_file = self.token_data_path / "token_labels/comprehensive_token_labels.json"
        if not label_file.exists():
            logger.warning(f"Label file not found: {label_file}")
            return {}
            
        with open(label_file, 'r') as f:
            labels = json.load(f)
            
        return labels
        
    def load_cluster_models(self, k: int = 10) -> Dict[int, Any]:
        """
        Load saved KMeans models for each layer.
        
        Args:
            k: Number of clusters
            
        Returns:
            Dictionary mapping layer index to sklearn KMeans model
        """
        cache_key = f"models_k{k}"
        if cache_key in self._cluster_models_cache:
            return self._cluster_models_cache[cache_key]
            
        models = {}
        model_dir = self.unified_results_path / f"cluster_models_k{k}"
        
        if not model_dir.exists():
            logger.warning(f"Cluster model directory not found: {model_dir}")
            return {}
            
        for layer in range(12):  # GPT-2 has 12 layers
            model_file = model_dir / f"kmeans_layer_{layer}.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    models[layer] = pickle.load(f)
                    
        self._cluster_models_cache[cache_key] = models
        return models
        
    def get_trajectories_by_context(self, context: str, k: int = 10) -> Dict[int, List[int]]:
        """
        Get all trajectories for a specific context.
        
        Args:
            context: Context frame name
            k: Number of clusters
            
        Returns:
            Dictionary mapping token index to trajectory
        """
        data = self.load_unified_trajectories(k)
        trajectories = data['trajectories']
        
        result = {}
        for key, traj_data in trajectories.items():
            if traj_data['context_frame'] == context:
                token_idx = traj_data['token_idx']
                result[token_idx] = traj_data['path']
                
        return result
        
    def get_context_types(self, k: int = 10) -> List[str]:
        """
        Get list of all context types in the experiment.
        
        Args:
            k: Number of clusters
            
        Returns:
            Sorted list of context type names
        """
        data = self.load_unified_trajectories(k)
        trajectories = data['trajectories']
        
        contexts = set()
        for traj_data in trajectories.values():
            contexts.add(traj_data['context_frame'])
            
        return sorted(list(contexts))
        
    def get_token_indices(self, k: int = 10) -> List[int]:
        """
        Get list of all token indices in the experiment.
        
        Args:
            k: Number of clusters
            
        Returns:
            Sorted list of token indices
        """
        data = self.load_unified_trajectories(k)
        trajectories = data['trajectories']
        
        indices = set()
        for traj_data in trajectories.values():
            indices.add(traj_data['token_idx'])
            
        return sorted(list(indices))
        
    def get_trajectory_pairs(self, token_idx: int, context1: str, context2: str, 
                           k: int = 10) -> Optional[Tuple[List[int], List[int]]]:
        """
        Get trajectory pair for a token under two different contexts.
        
        Args:
            token_idx: Token index
            context1: First context
            context2: Second context
            k: Number of clusters
            
        Returns:
            Tuple of (trajectory1, trajectory2) or None if not found
        """
        data = self.load_unified_trajectories(k)
        trajectories = data['trajectories']
        
        key1 = f"{token_idx}_{context1}"
        key2 = f"{token_idx}_{context2}"
        
        if key1 in trajectories and key2 in trajectories:
            return (trajectories[key1]['path'], trajectories[key2]['path'])
        else:
            return None
            
    def stratify_tokens(self, stratify_by: str = 'frequency', 
                       quantiles: List[float] = [0.33, 0.67]) -> Dict[str, List[int]]:
        """
        Stratify tokens by frequency or type.
        
        Args:
            stratify_by: 'frequency' or 'type'
            quantiles: Quantile boundaries for frequency stratification
            
        Returns:
            Dictionary mapping strata names to token indices
        """
        token_indices = self.get_token_indices()
        
        if stratify_by == 'frequency':
            frequencies = self.load_token_frequencies()
            
            # Get frequencies for our tokens
            token_freqs = []
            for idx in token_indices:
                freq = int(frequencies.get(str(idx), 0))
                token_freqs.append((idx, freq))
                
            # Sort by frequency
            token_freqs.sort(key=lambda x: x[1])
            
            # Calculate quantile boundaries
            n = len(token_freqs)
            boundaries = [int(n * q) for q in quantiles]
            
            # Create strata
            strata = defaultdict(list)
            for i, (idx, freq) in enumerate(token_freqs):
                if i < boundaries[0]:
                    strata['low'].append(idx)
                elif i < boundaries[1]:
                    strata['medium'].append(idx)
                else:
                    strata['high'].append(idx)
                    
            return dict(strata)
            
        elif stratify_by == 'type':
            metadata = self.load_token_metadata()
            labels = self.load_token_labels()
            
            strata = defaultdict(list)
            for idx in token_indices:
                # Try metadata first
                meta = metadata.get(str(idx), {})
                token_type = meta.get('token_type', '')
                
                # Classify
                if 'punctuation' in token_type or meta.get('is_punctuation', False):
                    strata['punctuation'].append(idx)
                elif meta.get('is_subword', False) or 'subword' in token_type:
                    strata['subword'].append(idx)
                else:
                    # Check POS tag from labels
                    label = labels.get(str(idx), {})
                    pos = label.get('grammatical', {}).get('pos', '')
                    
                    if pos in ['DET', 'PRON', 'ADP', 'CONJ', 'AUX']:
                        strata['function'].append(idx)
                    else:
                        strata['content'].append(idx)
                        
            return dict(strata)
            
        else:
            raise ValueError(f"Unknown stratification method: {stratify_by}")
            
    def get_activation_pairs(self, token_idx: int, context1: str, 
                           context2: str, layer: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get activation vectors for a token under two contexts at a specific layer.
        
        Args:
            token_idx: Token index
            context1: First context
            context2: Second context  
            layer: Layer index
            
        Returns:
            Tuple of (activation1, activation2) or None if not found
        """
        # Load trajectories to get case indices
        data = self.load_unified_trajectories()
        trajectories = data['trajectories']
        
        key1 = f"{token_idx}_{context1}"
        key2 = f"{token_idx}_{context2}"
        
        if key1 not in trajectories or key2 not in trajectories:
            return None
            
        case_idx1 = trajectories[key1]['case_idx']
        case_idx2 = trajectories[key2]['case_idx']
        
        # Load activations
        activations = self.load_unified_activations()
        
        # Extract specific layer activations
        # Activations are stored as {case_idx: {layer: array}}
        if case_idx1 in activations and case_idx2 in activations:
            act1 = activations[case_idx1][layer]
            act2 = activations[case_idx2][layer]
            return (act1, act2)
        else:
            return None
            
    def clear_cache(self):
        """Clear all cached data."""
        self._trajectories_cache.clear()
        self._activations_cache.clear()
        self._metadata_cache.clear()
        self._cluster_models_cache.clear()
        
        # Clear LRU caches
        self.load_unified_trajectories.cache_clear()
        self.load_token_frequencies.cache_clear()
        self.load_token_metadata.cache_clear()
        self.load_token_labels.cache_clear()
        
        logger.info("All caches cleared")