"""
Tests for Publication Figures generation.
"""

import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ..publication_figures import PublicationFigures


class TestPublicationFigures:
    """Test publication figure generation."""
    
    @pytest.mark.unit
    def test_initialization(self, temp_dir):
        """Test initialization."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'dpi': 150, 'formats': ['png'], 'enable_logging': False}
        )
        
        assert analysis.analysis_name == "publication_figures"
        assert analysis.config['dpi'] == 150
        assert analysis.config['formats'] == ['png']
        
    @pytest.mark.unit
    def test_style_configuration(self):
        """Test that style is properly configured."""
        analysis = PublicationFigures(config={'enable_logging': False})
        
        # Check some key style parameters
        assert plt.rcParams['figure.dpi'] == 300
        assert plt.rcParams['font.size'] == 8
        assert plt.rcParams['axes.linewidth'] == 0.8
        
    @pytest.mark.unit
    def test_color_palette(self):
        """Test color palette is defined."""
        analysis = PublicationFigures(config={'enable_logging': False})
        
        # Check key colors exist
        assert 'baseline' in analysis.COLORS
        assert 'determiner_the' in analysis.COLORS
        assert 'function' in analysis.COLORS
        assert 'content' in analysis.COLORS
        
        # Check colors are valid hex
        for color in analysis.COLORS.values():
            assert color.startswith('#')
            assert len(color) == 7
            
    @pytest.mark.unit
    def test_find_token_index(self, mock_data_loader):
        """Test token finding by string."""
        analysis = PublicationFigures(config={'enable_logging': False})
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Mock token metadata
        analysis.token_metadata = {
            'token_str': {0: ' the', 1: ' light', 2: ' and'}
        }
        
        # Find existing token
        idx = analysis._find_token_index('light')
        assert idx == 1
        
        # Find with different case
        idx = analysis._find_token_index('THE')
        assert idx == 0
        
        # Non-existent token
        idx = analysis._find_token_index('nonexistent')
        assert idx is None
        
    @pytest.mark.unit
    def test_save_figure(self, temp_dir):
        """Test figure saving functionality."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'formats': ['png', 'pdf'], 'enable_logging': False}
        )
        
        # Create simple figure
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Save figure
        saved = analysis._save_figure(fig, 'test_figure')
        
        # Check files created
        assert 'png' in saved
        assert 'pdf' in saved
        assert Path(saved['png']).exists()
        assert Path(saved['pdf']).exists()
        
        # Check file paths are correct
        assert saved['png'].endswith('test_figure.png')
        assert saved['pdf'].endswith('test_figure.pdf')
        
    @pytest.mark.unit
    def test_figure_manifest(self):
        """Test figure manifest creation."""
        analysis = PublicationFigures(config={'enable_logging': False})
        
        # Mock figures
        figures = {
            'trajectory_fan': {'png': 'path/to/fan.png'},
            'token_type_metrics': {'png': 'path/to/metrics.png'}
        }
        
        manifest = analysis._create_figure_manifest(figures)
        
        # Check structure
        assert 'trajectory_fan' in manifest
        assert 'token_type_metrics' in manifest
        
        # Check each entry has required fields
        for fig_name, info in manifest.items():
            assert 'description' in info
            assert 'caption' in info
            assert 'files' in info
            
    @pytest.mark.unit
    def test_trajectory_fan_plot(self, mock_data_loader, temp_dir):
        """Test trajectory fan plot creation."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'formats': ['png'], 'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Create plot
        fig = analysis._create_trajectory_fan_plot()
        
        # Basic checks
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        
        # Check plot elements
        assert ax.get_xlabel() == 'Layer'
        assert ax.get_ylabel() == 'Cluster Assignment (stacked)'
        assert len(ax.lines) > 0  # Should have plotted lines
        
        plt.close(fig)
        
    @pytest.mark.unit
    def test_token_type_metrics_plot(self, temp_dir):
        """Test token type metrics plot."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'formats': ['png'], 'enable_logging': False}
        )
        
        # Create plot
        fig = analysis._create_token_type_metrics_plot()
        
        # Check structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two subplots
        
        # Check axes
        ax1, ax2 = fig.axes
        assert 'Entropy' in ax1.get_ylabel()
        assert 'Sparsity' in ax2.get_ylabel()
        
        plt.close(fig)
        
    @pytest.mark.unit
    def test_single_token_showcase(self, mock_data_loader, temp_dir):
        """Test single token visualization."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'formats': ['png'], 'example_token': 'light', 'enable_logging': False}
        )
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        
        # Create showcase
        fig = analysis._create_single_token_showcase()
        
        # Check structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Four subplots
        
        # Check title
        assert 'Context Effects' in fig._suptitle.get_text()
        
        plt.close(fig)
        
    @pytest.mark.unit
    def test_transformation_geometry(self, temp_dir):
        """Test transformation geometry visualization."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'formats': ['png'], 'enable_logging': False}
        )
        
        # Create plot
        fig = analysis._create_transformation_geometry()
        
        # Check structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 grid
        
        # Check title
        assert 'Geometric Transformations' in fig._suptitle.get_text()
        
        plt.close(fig)
        
    @pytest.mark.integration
    def test_full_analysis(self, mock_data_loader, temp_dir):
        """Test complete figure generation pipeline."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={
                'formats': ['png'],
                'dpi': 150,  # Lower for testing
                'k_clusters': 10,
                'enable_logging': False
            }
        )
        
        analysis.data_loader = mock_data_loader
        output = analysis.run()
        
        # Check output structure
        assert hasattr(output, 'data')
        assert hasattr(output, 'statistics')
        assert hasattr(output, 'summary')
        
        # Check figures generated
        figures = output.data
        expected_figures = [
            'trajectory_fan', 'token_type_metrics', 'context_dendrogram',
            'single_token', 'transformation_geometry', 'layer_evolution'
        ]
        
        for fig_name in expected_figures:
            assert fig_name in figures
            assert 'png' in figures[fig_name]
            
        # Check files exist
        assert (temp_dir / "publication_figures_results.json").exists()
        
        # Check manifest created
        assert 'manifest' in output.summary
        
    @pytest.mark.unit
    def test_validation(self, mock_data_loader):
        """Test data validation."""
        analysis = PublicationFigures(config={'enable_logging': False})
        
        # Should fail without data
        with pytest.raises(ValueError, match="No trajectory data"):
            analysis.validate_data()
            
        # Load data and validate
        analysis.data_loader = mock_data_loader
        analysis.load_data()
        analysis.validate_data()  # Should pass
        
    @pytest.mark.unit
    def test_layer_evolution_plot(self, temp_dir):
        """Test layer evolution figure."""
        analysis = PublicationFigures(
            output_dir=str(temp_dir),
            config={'formats': ['png'], 'enable_logging': False}
        )
        
        # Create plot
        fig = analysis._create_layer_evolution_figure()
        
        # Check structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 grid
        
        # Check subplot titles
        titles = [ax.get_title() for ax in fig.axes]
        assert any('Entropy' in t for t in titles)
        assert any('Divergence' in t for t in titles)
        
        plt.close(fig)