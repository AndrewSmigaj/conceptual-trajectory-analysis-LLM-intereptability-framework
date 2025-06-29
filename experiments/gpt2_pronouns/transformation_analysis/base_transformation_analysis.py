"""
Base class for transformation analysis experiments.
Extends the concept_fragmentation BaseExperiment pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime
import numpy as np

from .output_schema import UnifiedAnalysisOutput, AnalysisMetadata
from .data_loader import TransformationDataLoader

logger = logging.getLogger(__name__)


class BaseTransformationAnalysis(ABC):
    """
    Abstract base class for all transformation analyses.
    
    Provides consistent interface for loading data, running analyses,
    and saving results in the unified output format.
    """
    
    def __init__(self, 
                 analysis_name: str,
                 output_dir: str = "results_transformation",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transformation analysis.
        
        Args:
            analysis_name: Name of the analysis (e.g., 'stratified_transition')
            output_dir: Directory for saving results
            config: Analysis-specific configuration
        """
        self.analysis_name = analysis_name
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Initialize data loader
        self.data_loader = TransformationDataLoader()
        
        # Results container
        self.output = None
        
        # Data attributes (populated by load_data)
        self.trajectories = None
        self.metadata = None
        self.activations = None
        self.token_frequencies = None
        self.token_metadata = None
        self.token_labels = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger (always create instance attribute)
        self.logger = logger
        
        # Set up logging (can be disabled for tests)
        if self.config.get('enable_logging', True):
            self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Set up analysis-specific logging."""
        log_file = self.output_dir / f"{self.analysis_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Store file handler reference for cleanup
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(self.file_handler)
        
    def __del__(self):
        """Clean up logging handlers to avoid file lock issues."""
        if hasattr(self, 'file_handler'):
            logger.removeHandler(self.file_handler)
            self.file_handler.close()
        
    def run(self) -> UnifiedAnalysisOutput:
        """
        Run the complete analysis pipeline.
        
        Returns:
            UnifiedAnalysisOutput with all results
        """
        logger.info(f"Starting analysis: {self.analysis_name}")
        start_time = datetime.now()
        
        try:
            # Load data
            logger.info("Loading data...")
            self.load_data()
            
            # Validate data
            logger.info("Validating data...")
            self.validate_data()
            
            # Run analysis
            logger.info("Running analysis...")
            results = self.analyze()
            
            # Create output object
            self.output = self._create_output(results, start_time)
            
            # Validate results
            logger.info("Validating results...")
            self.validate_results()
            
            # Save results
            logger.info("Saving results...")
            self.save_results()
            
            duration = datetime.now() - start_time
            logger.info(f"Analysis completed successfully in {duration}")
            
            return self.output
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise
            
    def load_data(self) -> None:
        """Load trajectories and optionally activations using data loader."""
        k = self.config.get('k_clusters', 10)
        
        # Load trajectories
        logger.info(f"Loading trajectories for k={k}")
        trajectories_data = self.data_loader.load_unified_trajectories(k)
        self.trajectories = trajectories_data['trajectories']
        self.metadata = trajectories_data.get('metadata', {})
        
        # Load activations if needed
        if self.config.get('load_activations', False):
            logger.info("Loading activations")
            self.activations = self.data_loader.load_unified_activations()
                
        # Load token metadata
        logger.info("Loading token metadata")
        self.token_frequencies = self.data_loader.load_token_frequencies()
        self.token_metadata = self.data_loader.load_token_metadata()
        self.token_labels = self.data_loader.load_token_labels()
            
    @abstractmethod
    def validate_data(self) -> None:
        """
        Validate loaded data.
        Should raise ValueError if data is invalid.
        """
        pass
        
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Run the main analysis.
        
        Returns:
            Dictionary of analysis results
        """
        pass
        
    @abstractmethod
    def validate_results(self) -> None:
        """
        Validate analysis results.
        Should raise ValueError if results are invalid.
        """
        pass
        
    def _create_output(self, results: Dict[str, Any], start_time: datetime) -> UnifiedAnalysisOutput:
        """
        Create unified output object from results.
        
        Args:
            results: Raw analysis results
            start_time: Analysis start time
            
        Returns:
            UnifiedAnalysisOutput object
        """
        metadata = AnalysisMetadata(
            analysis_type=self.analysis_name,
            timestamp=start_time,
            parameters=self.config
        )
        
        # Extract components from results
        data = results.get('data', {})
        statistics = results.get('statistics', {})
        summary = results.get('summary', {})
        
        # Create output object
        output = UnifiedAnalysisOutput(
            metadata=metadata,
            data=data,
            statistics=statistics,
            summary=summary
        )
        
        # Add optional fields if present in results
        for field in ['transition_matrices', 'transformation_matrices', 
                     'stability_metrics', 'predictive_results',
                     'significance_tests', 'effect_sizes',
                     'information_metrics', 'stratification',
                     'visualizations']:
            if field in results:
                setattr(output, field, results[field])
                
        return output
        
    def save_results(self) -> None:
        """Save analysis results in multiple formats."""
        # Save as JSON
        json_path = self.output_dir / f"{self.analysis_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.output.to_dict(), f, indent=2)
        logger.info(f"Saved JSON results to {json_path}")
        
        # Save as pickle (preserves numpy arrays)
        pkl_path = self.output_dir / f"{self.analysis_name}_results.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.output, f)
        logger.info(f"Saved pickle results to {pkl_path}")
        
    def get_context_types(self) -> List[str]:
        """Get list of unique context types from trajectories."""
        k = self.config.get('k_clusters', 10)
        return self.data_loader.get_context_types(k)
        
    def get_token_indices(self) -> List[int]:
        """Get list of unique token indices."""
        k = self.config.get('k_clusters', 10)
        return self.data_loader.get_token_indices(k)
        
    def get_trajectories_by_context(self, context: str) -> Dict[int, List[int]]:
        """
        Get trajectories for a specific context.
        
        Args:
            context: Context frame name
            
        Returns:
            Dictionary mapping token index to trajectory
        """
        k = self.config.get('k_clusters', 10)
        return self.data_loader.get_trajectories_by_context(context, k)
        
    def stratify_tokens(self, stratify_by: str = 'frequency') -> Dict[str, List[int]]:
        """
        Stratify tokens by frequency or type.
        
        Args:
            stratify_by: 'frequency' or 'type'
            
        Returns:
            Dictionary mapping strata to token indices
        """
        return self.data_loader.stratify_tokens(stratify_by)