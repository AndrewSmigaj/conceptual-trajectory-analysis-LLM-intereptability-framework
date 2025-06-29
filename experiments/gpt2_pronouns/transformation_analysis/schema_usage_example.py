"""
Example usage of the unified output schema for transformation analyses.
Shows how different analysis scripts should format their outputs.
"""

import numpy as np
from datetime import datetime
from output_schema import (
    UnifiedAnalysisOutput, AnalysisMetadata, QualityMetrics,
    TransformationMatrix, SignificanceTests, EffectSizes,
    Visualization
)


def example_transition_analysis_output():
    """Example output from stratified_transition_analysis.py"""
    
    # Build transition matrices (simplified example)
    transition_matrices = {
        0: {  # Layer 0
            "determiner_the": np.random.rand(10, 10),
            "determiner_a": np.random.rand(10, 10),
            "baseline": np.eye(10)  # Identity for baseline
        }
    }
    
    # Statistical tests
    significance_tests = SignificanceTests(
        permutation_p_values={
            "determiner_the": {0: 0.001, 1: 0.002},
            "determiner_a": {0: 0.001, 1: 0.003}
        },
        corrected_p_values={
            "determiner_the": {0: 0.01, 1: 0.02},
            "determiner_a": {0: 0.01, 1: 0.03}
        }
    )
    
    # Effect sizes
    effect_sizes = EffectSizes(
        cohens_d={
            "determiner_vs_baseline": {0: 1.2, 1: 0.8},
        },
        confidence_intervals={
            "determiner_vs_baseline": {0: (1.0, 1.4), 1: (0.6, 1.0)}
        }
    )
    
    # Create unified output
    output = UnifiedAnalysisOutput(
        metadata=AnalysisMetadata(
            analysis_type="stratified_transition_analysis",
            timestamp=datetime.now(),
            parameters={
                "k_clusters": 10,
                "n_tokens": 1000,
                "contexts": ["determiner_the", "determiner_a"],
                "stratify_by": ["frequency", "type"]
            }
        ),
        data={
            "n_tokens_analyzed": 1000,
            "n_contexts": 9,
            "layers_analyzed": list(range(12))
        },
        statistics={
            "all_significant": True,
            "mean_effect_size": 1.0,
            "n_comparisons": 108  # 9 contexts × 12 layers
        },
        summary={
            "key_findings": [
                "Context effects are systematic, not random (p < 0.001)",
                "Transition matrices show sparse, structured patterns",
                "Effect sizes are large (Cohen's d > 0.8) across all layers"
            ],
            "interpretation": "Context creates predictable transformations of the representation space",
            "next_steps": [
                "Test linear transformation hypothesis",
                "Analyze layer-wise evolution of transformations"
            ]
        },
        transition_matrices=transition_matrices,
        significance_tests=significance_tests,
        effect_sizes=effect_sizes,
        visualizations=[
            Visualization(
                name="transition_heatmaps_layer_0",
                path="visualizations/transition_heatmaps_layer_0.png",
                type="heatmap",
                description="Transition matrices for all contexts at layer 0"
            )
        ]
    )
    
    return output


def example_procrustes_analysis_output():
    """Example output from procrustes_cv_analysis.py"""
    
    # Transformation matrices with quality metrics
    transformation_matrices = {
        0: {  # Layer 0
            "determiner_the": TransformationMatrix(
                matrix=np.random.rand(768, 768),  # GPT-2 hidden size
                quality_metrics=QualityMetrics(
                    r2_score=0.85,
                    mse=0.02,
                    cosine_similarity=0.92
                )
            )
        }
    }
    
    output = UnifiedAnalysisOutput(
        metadata=AnalysisMetadata(
            analysis_type="procrustes_transformation_analysis",
            timestamp=datetime.now(),
            parameters={
                "cv_folds": 5,
                "regularization": "l2",
                "alpha": 0.01
            }
        ),
        data={
            "mean_r2_score": 0.82,
            "transformation_type": "orthogonal",
            "condition_numbers": {"layer_0": 1.2}
        },
        statistics={
            "cv_scores": [0.80, 0.82, 0.83, 0.81, 0.84],
            "stability_score": 0.95
        },
        summary={
            "key_findings": [
                "Linear transformations explain 82% of variance",
                "Transformations are near-orthogonal (condition number ~1)",
                "Cross-validation shows stable results"
            ],
            "interpretation": "Context effects can be modeled as linear transformations",
            "next_steps": ["Analyze transformation properties across layers"]
        },
        transformation_matrices=transformation_matrices
    )
    
    return output


if __name__ == "__main__":
    # Example 1: Transition analysis output
    transition_output = example_transition_analysis_output()
    print("Transition Analysis Output:")
    print(f"  Analysis type: {transition_output.metadata.analysis_type}")
    print(f"  Key findings: {transition_output.summary['key_findings'][0]}")
    
    # Example 2: Procrustes analysis output
    procrustes_output = example_procrustes_analysis_output()
    print("\nProcrustes Analysis Output:")
    print(f"  Analysis type: {procrustes_output.metadata.analysis_type}")
    print(f"  Mean R² score: {procrustes_output.data['mean_r2_score']}")
    
    # Convert to dict for JSON serialization
    output_dict = transition_output.to_dict()
    print(f"\nJSON-serializable: {isinstance(output_dict, dict)}")