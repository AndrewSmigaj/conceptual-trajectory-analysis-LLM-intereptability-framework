"""
Shared test fixtures for transformation analysis tests.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil

from ..output_schema import (
    UnifiedAnalysisOutput, AnalysisMetadata, QualityMetrics,
    TransformationMatrix, SignificanceTests, EffectSizes,
    Visualization
)


@pytest.fixture
def sample_trajectories():
    """Sample trajectory data for testing."""
    return {
        "metadata": {
            "experiment": "test_experiment",
            "timestamp": "2025-06-23T10:00:00",
            "num_tokens": 3,
            "num_contexts": 3,
            "num_test_cases": 9,
            "k_clusters": 10
        },
        "trajectories": {
            "0_baseline": {
                "token_idx": 0,
                "token_str": " the",
                "context_frame": "baseline",
                "path": [1, 1, 1, 1, 7, 9, 1, 1, 7, 1, 1, 6],
                "case_idx": 0
            },
            "0_determiner_the": {
                "token_idx": 0,
                "token_str": " the",
                "context_frame": "determiner_the",
                "path": [9, 9, 4, 7, 6, 0, 0, 2, 5, 5, 0, 3],
                "case_idx": 1
            },
            "0_determiner_a": {
                "token_idx": 0,
                "token_str": " the",
                "context_frame": "determiner_a",
                "path": [9, 9, 4, 7, 6, 0, 8, 4, 4, 5, 3, 3],
                "case_idx": 2
            },
            "1_baseline": {
                "token_idx": 1,
                "token_str": " and",
                "context_frame": "baseline",
                "path": [2, 2, 2, 2, 8, 8, 2, 2, 8, 2, 2, 7],
                "case_idx": 3
            },
            "1_determiner_the": {
                "token_idx": 1,
                "token_str": " and",
                "context_frame": "determiner_the",
                "path": [8, 8, 3, 6, 5, 1, 1, 3, 6, 6, 1, 4],
                "case_idx": 4
            },
            "1_determiner_a": {
                "token_idx": 1,
                "token_str": " and",
                "context_frame": "determiner_a",
                "path": [8, 8, 3, 6, 5, 1, 9, 5, 5, 6, 4, 4],
                "case_idx": 5
            },
            "2_baseline": {
                "token_idx": 2,
                "token_str": " of",
                "context_frame": "baseline",
                "path": [3, 3, 3, 3, 9, 7, 3, 3, 9, 3, 3, 8],
                "case_idx": 6
            },
            "2_determiner_the": {
                "token_idx": 2,
                "token_str": " of",
                "context_frame": "determiner_the",
                "path": [7, 7, 2, 5, 4, 2, 2, 4, 7, 7, 2, 5],
                "case_idx": 7
            },
            "2_determiner_a": {
                "token_idx": 2,
                "token_str": " of",
                "context_frame": "determiner_a",
                "path": [7, 7, 2, 5, 4, 2, 7, 6, 6, 7, 5, 5],
                "case_idx": 8
            }
        }
    }


@pytest.fixture
def sample_activations():
    """Sample activation data for testing."""
    np.random.seed(42)
    activations = {}
    
    for case_idx in range(9):
        activations[case_idx] = {}
        for layer in range(12):
            # GPT-2 hidden size is 768
            activations[case_idx][layer] = np.random.randn(768)
            
    return activations


@pytest.fixture
def sample_token_metadata():
    """Sample token metadata for testing."""
    return {
        "0": {
            "token_id": 0,
            "token_str": " the",
            "token_type": "word_with_space",
            "is_subword": True,
            "subword_type": "prefix_space",
            "is_alphabetic": True
        },
        "1": {
            "token_id": 1,
            "token_str": " and",
            "token_type": "word_with_space", 
            "is_subword": True,
            "subword_type": "prefix_space",
            "is_alphabetic": True
        },
        "2": {
            "token_id": 2,
            "token_str": " of",
            "token_type": "word_with_space",
            "is_subword": True,
            "subword_type": "prefix_space",
            "is_alphabetic": True
        }
    }


@pytest.fixture
def sample_token_frequencies():
    """Sample token frequency data for testing."""
    return {
        "token_frequencies": {
            "0": 50000,  # High frequency
            "1": 5000,   # Medium frequency
            "2": 500     # Low frequency
        }
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_output():
    """Sample unified analysis output for testing."""
    metadata = AnalysisMetadata(
        analysis_type="test_analysis",
        timestamp=datetime.now(),
        parameters={"test": True}
    )
    
    return UnifiedAnalysisOutput(
        metadata=metadata,
        data={"test_data": [1, 2, 3]},
        statistics={"mean": 2.0},
        summary={
            "key_findings": ["Test finding"],
            "interpretation": "Test interpretation",
            "next_steps": ["Test next step"]
        }
    )


@pytest.fixture
def mock_data_loader(monkeypatch, sample_trajectories, sample_activations, 
                    sample_token_metadata, sample_token_frequencies):
    """Mock the data loader for testing."""
    from ..data_loader import TransformationDataLoader
    
    def mock_load_trajectories(self, k=10):
        return sample_trajectories
        
    def mock_load_activations(self):
        return sample_activations
        
    def mock_load_metadata(self):
        return sample_token_metadata
        
    def mock_load_frequencies(self):
        return sample_token_frequencies["token_frequencies"]
        
    monkeypatch.setattr(TransformationDataLoader, "load_unified_trajectories", 
                       mock_load_trajectories)
    monkeypatch.setattr(TransformationDataLoader, "load_unified_activations",
                       mock_load_activations)
    monkeypatch.setattr(TransformationDataLoader, "load_token_metadata",
                       mock_load_metadata)
    monkeypatch.setattr(TransformationDataLoader, "load_token_frequencies",
                       mock_load_frequencies)
    
    return TransformationDataLoader()