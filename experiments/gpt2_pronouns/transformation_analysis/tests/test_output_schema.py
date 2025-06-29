"""
Tests for the unified output schema.
"""

import pytest
import json
import numpy as np
from datetime import datetime

from ..output_schema import (
    UnifiedAnalysisOutput, AnalysisMetadata, QualityMetrics,
    TransformationMatrix, SignificanceTests, EffectSizes,
    InformationMetrics, StratifiedResults, Visualization
)


class TestAnalysisMetadata:
    """Test AnalysisMetadata dataclass."""
    
    @pytest.mark.unit
    def test_metadata_creation(self):
        """Test creating metadata object."""
        metadata = AnalysisMetadata(
            analysis_type="test_analysis",
            timestamp=datetime.now()
        )
        
        assert metadata.analysis_type == "test_analysis"
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.version == "1.0.0"
        assert metadata.parameters == {}
        
    @pytest.mark.unit
    def test_metadata_with_parameters(self):
        """Test metadata with custom parameters."""
        params = {"k_clusters": 10, "n_tokens": 1000}
        metadata = AnalysisMetadata(
            analysis_type="test",
            timestamp=datetime.now(),
            parameters=params
        )
        
        assert metadata.parameters == params


class TestQualityMetrics:
    """Test QualityMetrics dataclass."""
    
    @pytest.mark.unit
    def test_quality_metrics(self):
        """Test creating quality metrics."""
        metrics = QualityMetrics(
            r2_score=0.85,
            mse=0.02,
            cosine_similarity=0.92
        )
        
        assert metrics.r2_score == 0.85
        assert metrics.mse == 0.02
        assert metrics.cosine_similarity == 0.92


class TestUnifiedAnalysisOutput:
    """Test UnifiedAnalysisOutput dataclass."""
    
    @pytest.mark.unit
    def test_output_creation(self, sample_output):
        """Test creating output object."""
        assert sample_output.metadata.analysis_type == "test_analysis"
        assert sample_output.data == {"test_data": [1, 2, 3]}
        assert sample_output.statistics == {"mean": 2.0}
        assert len(sample_output.summary["key_findings"]) == 1
        
    @pytest.mark.unit
    def test_output_to_dict(self, sample_output):
        """Test converting output to dictionary."""
        output_dict = sample_output.to_dict()
        
        assert "metadata" in output_dict
        assert "data" in output_dict
        assert "statistics" in output_dict
        assert "summary" in output_dict
        
        # Check metadata conversion
        assert output_dict["metadata"]["analysis_type"] == "test_analysis"
        assert isinstance(output_dict["metadata"]["timestamp"], str)
        
    @pytest.mark.unit
    def test_output_json_serializable(self, sample_output):
        """Test that output can be JSON serialized."""
        output_dict = sample_output.to_dict()
        
        # Should not raise exception
        json_str = json.dumps(output_dict, indent=2)
        assert isinstance(json_str, str)
        
        # Can deserialize
        loaded = json.loads(json_str)
        assert loaded["metadata"]["analysis_type"] == "test_analysis"
        
    @pytest.mark.unit
    def test_output_with_numpy_arrays(self):
        """Test output with numpy arrays (transition matrices)."""
        metadata = AnalysisMetadata(
            analysis_type="test",
            timestamp=datetime.now()
        )
        
        output = UnifiedAnalysisOutput(
            metadata=metadata,
            data={},
            statistics={},
            summary={},
            transition_matrices={
                0: {
                    "context1": np.random.rand(10, 10),
                    "context2": np.random.rand(10, 10)
                }
            }
        )
        
        # Convert to dict
        output_dict = output.to_dict()
        
        # Check numpy arrays converted to lists
        assert isinstance(output_dict["data"]["transition_matrices"]["layer_0"]["context1"], list)
        assert len(output_dict["data"]["transition_matrices"]["layer_0"]["context1"]) == 10
        
    @pytest.mark.unit
    def test_output_from_dict(self):
        """Test creating output from dictionary."""
        data = {
            "metadata": {
                "analysis_type": "test",
                "timestamp": "2025-06-23T10:00:00",
                "version": "1.0.0",
                "parameters": {}
            },
            "data": {"test": 123},
            "statistics": {"mean": 1.5},
            "summary": {
                "key_findings": ["finding"],
                "interpretation": "interp",
                "next_steps": []
            }
        }
        
        output = UnifiedAnalysisOutput.from_dict(data)
        
        assert output.metadata.analysis_type == "test"
        assert output.data["test"] == 123
        assert output.statistics["mean"] == 1.5
        
    @pytest.mark.unit
    def test_optional_fields(self):
        """Test output with optional fields."""
        metadata = AnalysisMetadata(
            analysis_type="test",
            timestamp=datetime.now()
        )
        
        sig_tests = SignificanceTests(
            permutation_p_values={"ctx1": {0: 0.01}},
            corrected_p_values={"ctx1": {0: 0.05}}
        )
        
        effect_sizes = EffectSizes(
            cohens_d={"comparison1": {0: 1.2}},
            confidence_intervals={"comparison1": {0: (1.0, 1.4)}}
        )
        
        viz = Visualization(
            name="test_viz",
            path="path/to/viz.png",
            type="heatmap",
            description="Test visualization"
        )
        
        output = UnifiedAnalysisOutput(
            metadata=metadata,
            data={},
            statistics={},
            summary={},
            significance_tests=sig_tests,
            effect_sizes=effect_sizes,
            visualizations=[viz]
        )
        
        # Convert to dict
        output_dict = output.to_dict()
        
        # Check visualizations converted properly
        assert "visualizations" in output_dict
        assert len(output_dict["visualizations"]["figures"]) == 1
        assert output_dict["visualizations"]["figures"][0]["name"] == "test_viz"