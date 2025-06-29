"""
Tests for the TransformationDataLoader.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import pickle
import tempfile
import shutil

from ..data_loader import TransformationDataLoader


class TestDataLoaderBasics:
    """Test basic data loader functionality."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test data loader initialization."""
        loader = TransformationDataLoader()
        
        assert isinstance(loader.base_path, Path)
        assert loader.unified_results_path.name == "results_unified"
        assert loader.token_data_path.name == "all_tokens"
        
    @pytest.mark.unit
    def test_initialization_with_path(self, temp_dir):
        """Test initialization with custom path."""
        loader = TransformationDataLoader(base_path=temp_dir)
        
        assert loader.base_path == temp_dir
        assert loader.unified_results_path == temp_dir / "results_unified"


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.mark.unit
    def test_get_context_types(self, mock_data_loader):
        """Test getting context types."""
        contexts = mock_data_loader.get_context_types()
        
        assert len(contexts) == 3
        assert "baseline" in contexts
        assert "determiner_the" in contexts
        assert "determiner_a" in contexts
        
    @pytest.mark.unit
    def test_get_token_indices(self, mock_data_loader):
        """Test getting token indices."""
        indices = mock_data_loader.get_token_indices()
        
        assert len(indices) == 3
        assert indices == [0, 1, 2]
        
    @pytest.mark.unit
    def test_get_trajectories_by_context(self, mock_data_loader):
        """Test getting trajectories for specific context."""
        trajectories = mock_data_loader.get_trajectories_by_context("baseline")
        
        assert len(trajectories) == 3
        assert 0 in trajectories
        assert trajectories[0] == [1, 1, 1, 1, 7, 9, 1, 1, 7, 1, 1, 6]
        
    @pytest.mark.unit
    def test_get_trajectory_pairs(self, mock_data_loader):
        """Test getting trajectory pairs."""
        pair = mock_data_loader.get_trajectory_pairs(
            token_idx=0,
            context1="baseline",
            context2="determiner_the"
        )
        
        assert pair is not None
        assert len(pair) == 2
        assert pair[0] == [1, 1, 1, 1, 7, 9, 1, 1, 7, 1, 1, 6]
        assert pair[1] == [9, 9, 4, 7, 6, 0, 0, 2, 5, 5, 0, 3]
        
    @pytest.mark.unit
    def test_get_trajectory_pairs_missing(self, mock_data_loader):
        """Test getting trajectory pairs when one is missing."""
        pair = mock_data_loader.get_trajectory_pairs(
            token_idx=999,  # Non-existent token
            context1="baseline",
            context2="determiner_the"
        )
        
        assert pair is None


class TestTokenStratification:
    """Test token stratification functionality."""
    
    @pytest.mark.unit
    def test_stratify_by_frequency(self, mock_data_loader):
        """Test stratifying tokens by frequency."""
        strata = mock_data_loader.stratify_tokens(stratify_by="frequency")
        
        # With 3 tokens, we get at least 2 strata (possibly 3)
        assert len(strata) >= 2
        
        # All tokens should be assigned
        all_tokens = []
        for indices in strata.values():
            all_tokens.extend(indices)
        assert sorted(all_tokens) == [0, 1, 2]
        
        # Check relative ordering - highest frequency token should not be in lowest stratum
        if "low" in strata and "high" in strata:
            assert 0 not in strata["low"]  # Token 0 has highest frequency
            assert 2 not in strata["high"]  # Token 2 has lowest frequency
        
    @pytest.mark.unit
    def test_stratify_by_type(self, mock_data_loader):
        """Test stratifying tokens by type."""
        strata = mock_data_loader.stratify_tokens(stratify_by="type")
        
        assert len(strata) > 0
        # All test tokens are subwords
        assert "subword" in strata
        assert len(strata["subword"]) == 3
        
    @pytest.mark.unit
    def test_stratify_invalid_method(self, mock_data_loader):
        """Test invalid stratification method."""
        with pytest.raises(ValueError, match="Unknown stratification"):
            mock_data_loader.stratify_tokens(stratify_by="invalid")


class TestCaching:
    """Test caching functionality."""
    
    @pytest.mark.unit
    def test_trajectory_caching(self, mock_data_loader):
        """Test that trajectories are cached."""
        # First call
        traj1 = mock_data_loader.load_unified_trajectories(k=10)
        
        # Second call should return cached version
        traj2 = mock_data_loader.load_unified_trajectories(k=10)
        
        # Should be the same object
        assert traj1 is traj2
        
    @pytest.mark.unit
    def test_clear_cache(self):
        """Test clearing cache."""
        # Create a real data loader instance (not mocked)
        from ..data_loader import TransformationDataLoader
        loader = TransformationDataLoader()
        
        # Add some data to internal caches
        loader._trajectories_cache[10] = {"test": "data"}
        loader._metadata_cache["test"] = {"meta": "data"}
        
        # Clear cache
        loader.clear_cache()
        
        # Caches should be empty
        assert len(loader._trajectories_cache) == 0
        assert len(loader._metadata_cache) == 0
        assert len(loader._activations_cache) == 0


class TestFileHandling:
    """Test file handling and error cases."""
    
    @pytest.mark.unit
    def test_missing_trajectory_file(self, temp_dir):
        """Test handling missing trajectory file."""
        loader = TransformationDataLoader(base_path=temp_dir)
        
        # Create the directory structure
        (temp_dir / "results_unified").mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError, match="Trajectory file not found"):
            loader.load_unified_trajectories(k=10)
            
    @pytest.mark.unit
    def test_missing_activation_file(self, temp_dir):
        """Test handling missing activation file."""
        loader = TransformationDataLoader(base_path=temp_dir)
        
        # Create the directory structure
        (temp_dir / "results_unified").mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError, match="Activation file not found"):
            loader.load_unified_activations()
            
    @pytest.mark.unit
    def test_load_with_real_files(self, temp_dir, sample_trajectories, sample_activations):
        """Test loading with real files."""
        loader = TransformationDataLoader(base_path=temp_dir)
        
        # Create directory structure
        results_dir = temp_dir / "results_unified"
        results_dir.mkdir(parents=True)
        
        # Write trajectory file
        traj_file = results_dir / "unified_trajectories_k10.json"
        with open(traj_file, 'w') as f:
            json.dump(sample_trajectories, f)
            
        # Write activation file
        act_file = results_dir / "unified_activations.pkl"
        with open(act_file, 'wb') as f:
            pickle.dump(sample_activations, f)
            
        # Test loading
        trajectories = loader.load_unified_trajectories(k=10)
        assert trajectories["metadata"]["num_tokens"] == 3
        
        activations = loader.load_unified_activations()
        assert len(activations) == 9


class TestActivationPairs:
    """Test activation pair extraction."""
    
    @pytest.mark.unit  
    def test_get_activation_pairs(self, mock_data_loader):
        """Test getting activation pairs."""
        # This test requires both trajectories and activations
        pair = mock_data_loader.get_activation_pairs(
            token_idx=0,
            context1="baseline",
            context2="determiner_the",
            layer=0
        )
        
        assert pair is not None
        assert len(pair) == 2
        assert isinstance(pair[0], np.ndarray)
        assert isinstance(pair[1], np.ndarray)
        assert pair[0].shape == (768,)  # GPT-2 hidden size
        
    @pytest.mark.unit
    def test_get_activation_pairs_missing(self, mock_data_loader):
        """Test getting activation pairs when token is missing."""
        pair = mock_data_loader.get_activation_pairs(
            token_idx=999,  # Non-existent
            context1="baseline", 
            context2="determiner_the",
            layer=0
        )
        
        assert pair is None