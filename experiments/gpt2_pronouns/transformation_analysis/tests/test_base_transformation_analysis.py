"""
Tests for the BaseTransformationAnalysis abstract class.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from ..base_transformation_analysis import BaseTransformationAnalysis
from ..output_schema import UnifiedAnalysisOutput


class ConcreteAnalysis(BaseTransformationAnalysis):
    """Concrete implementation for testing."""
    
    def validate_data(self):
        """Simple validation."""
        if not self.trajectories:
            raise ValueError("No trajectories loaded")
            
    def analyze(self):
        """Simple analysis."""
        return {
            "data": {"n_tokens": len(self.get_token_indices())},
            "statistics": {"test": True},
            "summary": {
                "key_findings": ["Test finding"],
                "interpretation": "Test interpretation",
                "next_steps": ["Test next step"]
            }
        }
        
    def validate_results(self):
        """Simple result validation."""
        if not self.output:
            raise ValueError("No output generated")


class TestBaseTransformationAnalysis:
    """Test BaseTransformationAnalysis functionality."""
    
    @pytest.mark.unit
    def test_initialization(self, temp_dir):
        """Test initialization of concrete analysis."""
        analysis = ConcreteAnalysis(
            analysis_name="test_analysis",
            output_dir=str(temp_dir),
            config={"k_clusters": 10}
        )
        
        assert analysis.analysis_name == "test_analysis"
        assert analysis.output_dir == temp_dir
        assert analysis.config["k_clusters"] == 10
        assert analysis.data_loader is not None
        
    @pytest.mark.unit
    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""
        
        class IncompleteAnalysis(BaseTransformationAnalysis):
            pass
            
        # Should raise TypeError for missing abstract methods
        with pytest.raises(TypeError):
            IncompleteAnalysis("test")
            
    @pytest.mark.integration
    def test_run_pipeline(self, temp_dir, mock_data_loader, monkeypatch):
        """Test running the complete analysis pipeline."""
        # Patch the data loader
        def mock_init(self):
            self.base_path = temp_dir
            
        monkeypatch.setattr(
            "transformation_analysis.data_loader.TransformationDataLoader.__init__",
            mock_init
        )
        
        analysis = ConcreteAnalysis(
            analysis_name="test_analysis",
            output_dir=str(temp_dir),
            config={"k_clusters": 10}
        )
        
        # Mock the data loader
        analysis.data_loader = mock_data_loader
        
        # Run analysis
        output = analysis.run()
        
        assert isinstance(output, UnifiedAnalysisOutput)
        assert output.metadata.analysis_type == "test_analysis"
        assert output.data["n_tokens"] == 3
        
        # Check that results were saved
        json_file = temp_dir / "test_analysis_results.json"
        assert json_file.exists()
        
    @pytest.mark.unit
    def test_load_data(self, mock_data_loader):
        """Test data loading."""
        analysis = ConcreteAnalysis(
            analysis_name="test",
            config={"k_clusters": 10, "load_activations": True}
        )
        
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        assert analysis.trajectories is not None
        assert len(analysis.trajectories) == 9
        assert analysis.metadata is not None
        assert analysis.token_frequencies is not None
        assert analysis.token_metadata is not None
        assert analysis.activations is not None
        
    @pytest.mark.unit
    def test_utility_methods(self, mock_data_loader):
        """Test utility methods."""
        analysis = ConcreteAnalysis(
            analysis_name="test",
            config={"k_clusters": 10}
        )
        
        analysis.data_loader = mock_data_loader
        
        # Test get_context_types
        contexts = analysis.get_context_types()
        assert len(contexts) == 3
        assert "baseline" in contexts
        
        # Test get_token_indices
        indices = analysis.get_token_indices()
        assert indices == [0, 1, 2]
        
        # Test get_trajectories_by_context
        trajectories = analysis.get_trajectories_by_context("baseline")
        assert len(trajectories) == 3
        
        # Test stratify_tokens
        strata = analysis.stratify_tokens("frequency")
        # With 3 tokens, stratification creates 2 groups based on quantiles
        # The actual groups depend on the frequency distribution
        assert len(strata) >= 2  # Should have at least 2 strata
        assert sum(len(indices) for indices in strata.values()) == 3  # All tokens assigned
        
    @pytest.mark.unit
    def test_validation_called(self, mock_data_loader):
        """Test that validation methods are called."""
        
        class ValidationTestAnalysis(ConcreteAnalysis):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.data_validated = False
                self.results_validated = False
                
            def validate_data(self):
                self.data_validated = True
                super().validate_data()
                
            def validate_results(self):
                self.results_validated = True
                super().validate_results()
                
        analysis = ValidationTestAnalysis(
            analysis_name="test",
            config={"k_clusters": 10}
        )
        
        analysis.data_loader = mock_data_loader
        
        # Run analysis
        analysis.run()
        
        # Check validations were called
        assert analysis.data_validated
        assert analysis.results_validated
        
    @pytest.mark.unit
    def test_error_handling(self, mock_data_loader):
        """Test error handling in pipeline."""
        
        class ErrorAnalysis(ConcreteAnalysis):
            def analyze(self):
                raise RuntimeError("Test error")
                
        analysis = ErrorAnalysis(
            analysis_name="test",
            config={"k_clusters": 10}
        )
        
        analysis.data_loader = mock_data_loader
        
        # Should raise the error
        with pytest.raises(RuntimeError, match="Test error"):
            analysis.run()
            
    @pytest.mark.unit
    def test_output_creation(self):
        """Test creating unified output from results."""
        analysis = ConcreteAnalysis("test")
        
        results = {
            "data": {"test": 123},
            "statistics": {"mean": 1.5},
            "summary": {
                "key_findings": ["finding"],
                "interpretation": "interp",
                "next_steps": ["next"]
            },
            "transition_matrices": {0: {"ctx": np.eye(10)}},
            "visualizations": []
        }
        
        output = analysis._create_output(results, datetime.now())
        
        assert isinstance(output, UnifiedAnalysisOutput)
        assert output.metadata.analysis_type == "test"
        assert output.data["test"] == 123
        assert output.transition_matrices is not None
        
    @pytest.mark.unit
    def test_save_results(self, temp_dir, sample_output):
        """Test saving results."""
        analysis = ConcreteAnalysis(
            analysis_name="test",
            output_dir=str(temp_dir)
        )
        
        analysis.output = sample_output
        analysis.save_results()
        
        # Check JSON file
        json_file = temp_dir / "test_results.json"
        assert json_file.exists()
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        assert data["metadata"]["analysis_type"] == "test_analysis"
        
        # Check pickle file
        pkl_file = temp_dir / "test_results.pkl"
        assert pkl_file.exists()