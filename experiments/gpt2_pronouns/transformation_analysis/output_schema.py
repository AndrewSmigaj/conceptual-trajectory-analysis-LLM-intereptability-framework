"""
Unified output schema for transformation analysis results.
Provides type-safe dataclasses matching the JSON schema.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import numpy as np


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric"""
    lower: Union[float, np.ndarray]
    upper: Union[float, np.ndarray]
    confidence_level: float = 0.95
    method: str = "percentile"
    n_bootstrap: int = 1000
    
    def contains(self, value: Union[float, np.ndarray]) -> bool:
        """Check if value is within the confidence interval"""
        return np.all(self.lower <= value) and np.all(value <= self.upper)
    
    def width(self) -> Union[float, np.ndarray]:
        """Calculate width of the confidence interval"""
        return self.upper - self.lower
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "lower": float(self.lower) if np.isscalar(self.lower) else self.lower.tolist(),
            "upper": float(self.upper) if np.isscalar(self.upper) else self.upper.tolist(),
            "confidence_level": self.confidence_level,
            "method": self.method,
            "n_bootstrap": self.n_bootstrap
        }


@dataclass
class MetricWithCI:
    """A metric value with optional confidence interval"""
    value: Union[float, np.ndarray]
    ci: Optional[ConfidenceInterval] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "value": float(self.value) if np.isscalar(self.value) else self.value.tolist()
        }
        if self.ci:
            result["confidence_interval"] = self.ci.to_dict()
        return result


@dataclass
class QualityMetrics:
    """Quality metrics for transformation matrices"""
    r2_score: float
    mse: float
    cosine_similarity: float


@dataclass
class TransformationMatrix:
    """Linear transformation matrix with quality metrics"""
    matrix: np.ndarray
    quality_metrics: QualityMetrics


@dataclass
class StabilityMetrics:
    """Clustering stability metrics for a layer"""
    mean_ari: float
    std_ari: float
    transition_variance: float


@dataclass
class PredictiveResults:
    """Results from predictive modeling"""
    accuracy: float
    confusion_matrix: np.ndarray
    per_context_accuracy: Dict[str, float]
    feature_importance: Optional[np.ndarray] = None


@dataclass
class SignificanceTests:
    """Statistical significance test results"""
    permutation_p_values: Dict[str, Dict[int, float]]  # context -> layer -> p-value
    corrected_p_values: Dict[str, Dict[int, float]]
    correction_method: str = "fdr_bh"


@dataclass
class EffectSizes:
    """Effect size measurements"""
    cohens_d: Dict[str, Dict[int, float]]  # comparison -> layer -> effect size
    confidence_intervals: Dict[str, Dict[int, tuple]]  # comparison -> layer -> (lower, upper)


@dataclass
class InformationMetrics:
    """Information-theoretic metrics"""
    mutual_information: Dict[int, float]  # layer -> MI
    entropy: Dict[str, Dict[int, float]]  # context -> layer -> entropy
    kl_divergence: Dict[str, Dict[int, float]]  # context -> layer -> KL


@dataclass
class StratifiedResults:
    """Results for a specific stratification group"""
    transition_matrices: Dict[int, Dict[str, np.ndarray]]
    summary_metrics: Dict[str, float]
    n_tokens: int


@dataclass
class Visualization:
    """Reference to a generated visualization"""
    name: str
    path: str
    type: str  # 'heatmap', 'sankey', 'scatter', etc.
    description: str


@dataclass
class AnalysisMetadata:
    """Metadata for the analysis run"""
    analysis_type: str
    timestamp: datetime
    version: str = "1.0.0"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedAnalysisOutput:
    """
    Unified output structure for all transformation analyses.
    
    This class provides a consistent interface for storing and accessing
    results from different analysis scripts, ensuring compatibility
    across the entire pipeline.
    """
    # Required fields
    metadata: AnalysisMetadata
    data: Dict[str, Any]  # Flexible for analysis-specific data
    statistics: Dict[str, Any]  # Statistical test results
    summary: Dict[str, Any]  # High-level findings
    
    # Optional fields for specific analyses
    transition_matrices: Optional[Dict[int, Dict[str, np.ndarray]]] = None
    transformation_matrices: Optional[Dict[int, Dict[str, TransformationMatrix]]] = None
    stability_metrics: Optional[Dict[int, StabilityMetrics]] = None
    predictive_results: Optional[PredictiveResults] = None
    significance_tests: Optional[SignificanceTests] = None
    effect_sizes: Optional[EffectSizes] = None
    information_metrics: Optional[InformationMetrics] = None
    stratification: Optional[Dict[str, Dict[str, StratifiedResults]]] = None
    visualizations: Optional[List[Visualization]] = None
    confidence_intervals: Optional[Dict[str, Any]] = None  # Metric name -> CI data
    
    def _serialize_value(self, obj: Any) -> Any:
        """Recursively serialize values for JSON compatibility."""
        if isinstance(obj, MetricWithCI):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_value(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "metadata": {
                "analysis_type": self.metadata.analysis_type,
                "timestamp": self.metadata.timestamp.isoformat(),
                "version": self.metadata.version,
                "parameters": self.metadata.parameters
            },
            "data": self._serialize_value(self.data),
            "statistics": self._serialize_value(self.statistics),
            "summary": self._serialize_value(self.summary)
        }
        
        # Add optional fields if present
        if self.transition_matrices:
            result["data"]["transition_matrices"] = {
                f"layer_{l}": {
                    ctx: mat.tolist() for ctx, mat in contexts.items()
                } for l, contexts in self.transition_matrices.items()
            }
            
        if self.visualizations:
            result["visualizations"] = {
                "figures": [
                    {
                        "name": viz.name,
                        "path": viz.path,
                        "type": viz.type,
                        "description": viz.description
                    } for viz in self.visualizations
                ]
            }
            
        if self.confidence_intervals:
            result["confidence_intervals"] = self.confidence_intervals
            
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedAnalysisOutput":
        """Create from dictionary (e.g., loaded from JSON)"""
        metadata = AnalysisMetadata(
            analysis_type=data["metadata"]["analysis_type"],
            timestamp=datetime.fromisoformat(data["metadata"]["timestamp"]),
            version=data["metadata"]["version"],
            parameters=data["metadata"]["parameters"]
        )
        
        return cls(
            metadata=metadata,
            data=data["data"],
            statistics=data["statistics"],
            summary=data["summary"]
        )