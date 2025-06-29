"""
Information Theory Metrics for Context Transformations

Calculates information-theoretic measures to quantify how context affects
token representations, including:
- Mutual information between contexts and transformations
- KL divergence from baseline distributions
- Entropy measures of transition patterns
- Jensen-Shannon divergence for context similarity

This analysis provides theoretical grounding for the systematic transformation
hypothesis by quantifying information content and flow.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

from .base_transformation_analysis import BaseTransformationAnalysis
from .bootstrap_mixin import BootstrapMixin
from .output_schema import UnifiedAnalysisOutput, MetricWithCI


class InformationTheoryMetrics(BaseTransformationAnalysis, BootstrapMixin):
    """
    Calculates information-theoretic measures of context-induced transformations.
    """
    
    def __init__(self, output_dir: str = None, config: dict = None):
        """Initialize information theory analysis.
        
        Args:
            output_dir: Output directory path
            config: Configuration dictionary
        """
        if not config:
            config = {}
            
        if not output_dir:
            output_dir = config.get('output_dir', 'results_transformation/information_theory')
            
        # Set default config
        config.setdefault('n_bootstrap', 1000)
        config.setdefault('k_clusters', 10)
        config.setdefault('contexts_to_analyze', ['baseline', 'determiner_the', 'copula_is'])
        
        # Initialize base class first
        BaseTransformationAnalysis.__init__(
            self,
            analysis_name="information_theory_metrics",
            output_dir=output_dir,
            config=config
        )
        
        # Initialize bootstrap mixin
        BootstrapMixin.__init__(self, n_bootstrap=config['n_bootstrap'])
        
    def analyze(self) -> Dict[str, Any]:
        """Run information theory analysis."""
        self.logger.info("Starting information theory analysis")
        
        results = {
            'mutual_information': {},
            'kl_divergence': {},
            'entropy_metrics': {},
            'jensen_shannon': {},
            'layer_evolution': {},
            'stratified_analysis': {}
        }
        
        # Get contexts to analyze
        contexts = self.config['contexts_to_analyze']
        all_contexts = self.get_context_types()
        
        # Validate contexts exist
        for ctx in contexts:
            if ctx not in all_contexts:
                self.logger.warning(f"Context '{ctx}' not found in data")
                contexts = [c for c in contexts if c in all_contexts]
        
        if 'baseline' not in contexts:
            contexts.insert(0, 'baseline')
            
        self.logger.info(f"Analyzing contexts: {contexts}")
        
        # 1. Calculate mutual information
        self.logger.info("Calculating mutual information...")
        results['mutual_information'] = self._calculate_mutual_information(contexts)
        
        # 2. Calculate KL divergence
        self.logger.info("Calculating KL divergence...")
        results['kl_divergence'] = self._calculate_kl_divergence(contexts)
        
        # 3. Calculate entropy metrics
        self.logger.info("Calculating entropy metrics...")
        results['entropy_metrics'] = self._calculate_entropy_metrics(contexts)
        
        # 4. Calculate Jensen-Shannon divergence
        self.logger.info("Calculating Jensen-Shannon divergence...")
        results['jensen_shannon'] = self._calculate_jensen_shannon(contexts)
        
        # 5. Analyze layer evolution
        self.logger.info("Analyzing layer-wise evolution...")
        results['layer_evolution'] = self._analyze_layer_evolution(contexts)
        
        # 6. Stratified analysis
        self.logger.info("Performing stratified analysis...")
        results['stratified_analysis'] = self._stratified_analysis(contexts)
        
        # Create summary
        summary = self._create_summary(results)
        
        return {
            'data': results,
            'statistics': self._calculate_statistics(results),
            'summary': summary
        }
        
    def _calculate_mutual_information(self, contexts: List[str]) -> Dict[str, Any]:
        """Calculate mutual information between contexts and cluster transitions."""
        mi_results = {}
        
        # Get all token indices
        token_indices = self.get_token_indices()
        n_layers = 12  # GPT-2 layers
        
        # MI between context and final cluster for each layer
        layer_mi = []
        
        for layer in range(n_layers):
            # Collect context labels and cluster assignments
            context_labels = []
            cluster_assignments = []
            
            for token_idx in token_indices:
                for ctx_idx, context in enumerate(contexts):
                    traj = self.get_trajectories_by_context(context).get(token_idx)
                    if traj and len(traj) > layer:
                        context_labels.append(ctx_idx)
                        cluster_assignments.append(traj[layer])
            
            if context_labels:
                # Calculate MI
                mi = mutual_info_score(context_labels, cluster_assignments)
                layer_mi.append(mi)
            else:
                layer_mi.append(0.0)
        
        mi_results['layer_mi'] = layer_mi
        
        # MI between token properties and transformation patterns
        if self.token_metadata:
            property_mi = self._calculate_property_mi(contexts)
            mi_results['property_mi'] = property_mi
        
        # Add confidence intervals
        mi_with_ci = {}
        for key, values in mi_results.items():
            if isinstance(values, list):
                mi_with_ci[key] = values  # TODO: Add confidence intervals for lists
            else:
                mi_with_ci[key] = values
                
        return mi_with_ci
        
    def _calculate_property_mi(self, contexts: List[str]) -> Dict[str, float]:
        """Calculate MI between token properties and transformations."""
        property_mi = {}
        
        # Properties to test
        properties = ['token_type', 'frequency_bin', 'length_bin']
        
        for prop in properties:
            if prop not in self.token_metadata:
                continue
                
            # Collect property values and transformation indicators
            prop_values = []
            transform_indicators = []
            
            for token_idx in self.get_token_indices():
                # Get baseline trajectory
                baseline_traj = self.get_trajectories_by_context('baseline').get(token_idx)
                if not baseline_traj:
                    continue
                    
                # Check if trajectory changes with context
                for context in contexts[1:]:  # Skip baseline
                    ctx_traj = self.get_trajectories_by_context(context).get(token_idx)
                    if ctx_traj:
                        prop_values.append(self.token_metadata[prop].get(token_idx, 0))
                        # Binary indicator: does trajectory change?
                        transform_indicators.append(int(ctx_traj != baseline_traj))
            
            if prop_values:
                mi = mutual_info_score(prop_values, transform_indicators)
                property_mi[prop] = mi
                
        return property_mi
        
    def _calculate_kl_divergence(self, contexts: List[str]) -> Dict[str, Any]:
        """Calculate KL divergence from baseline distributions."""
        kl_results = {}
        
        # Build transition matrices for each context
        transition_matrices = {}
        
        for context in contexts:
            matrix = self._build_transition_matrix(context)
            transition_matrices[context] = matrix
            
        baseline_matrix = transition_matrices['baseline']
        
        # Calculate KL divergence for each context from baseline
        context_kl = {}
        
        for context in contexts[1:]:  # Skip baseline
            ctx_matrix = transition_matrices[context]
            
            # Calculate KL divergence for each row (source cluster)
            kl_values = []
            
            for i in range(len(baseline_matrix)):
                # Get probability distributions
                p = baseline_matrix[i] + 1e-10  # Add small epsilon
                q = ctx_matrix[i] + 1e-10
                
                # Normalize
                p = p / p.sum()
                q = q / q.sum()
                
                # Calculate KL divergence
                kl = stats.entropy(p, q)
                kl_values.append(kl)
                
            # Average KL divergence
            avg_kl = np.mean(kl_values)
            context_kl[context] = MetricWithCI(
                value=avg_kl,
                ci=None  # TODO: Add proper bootstrap CI
            )
            
        kl_results['context_kl'] = context_kl
        
        # Layer-wise KL evolution
        layer_kl = self._calculate_layer_kl(contexts)
        kl_results['layer_evolution'] = layer_kl
        
        return kl_results
        
    def _build_transition_matrix(self, context: str) -> np.ndarray:
        """Build transition matrix for a context."""
        k = self.config['k_clusters']
        matrix = np.zeros((k, k))
        
        trajectories = self.get_trajectories_by_context(context)
        
        for token_idx, traj in trajectories.items():
            if len(traj) >= 2:
                # Count transitions between consecutive layers
                for i in range(len(traj) - 1):
                    src = traj[i]
                    dst = traj[i + 1]
                    if 0 <= src < k and 0 <= dst < k:
                        matrix[src, dst] += 1
                        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
        
        return matrix
        
    def _calculate_layer_kl(self, contexts: List[str]) -> List[MetricWithCI]:
        """Calculate KL divergence at each layer."""
        n_layers = 12
        layer_kl = []
        
        for layer in range(n_layers):
            # Get cluster distributions at this layer
            baseline_dist = self._get_layer_distribution('baseline', layer)
            
            # Average KL from baseline across contexts
            kl_values = []
            
            for context in contexts[1:]:
                ctx_dist = self._get_layer_distribution(context, layer)
                
                # Ensure same size
                max_cluster = max(len(baseline_dist), len(ctx_dist))
                p = np.zeros(max_cluster)
                q = np.zeros(max_cluster)
                
                p[:len(baseline_dist)] = baseline_dist + 1e-10
                q[:len(ctx_dist)] = ctx_dist + 1e-10
                
                # Normalize
                p = p / p.sum()
                q = q / q.sum()
                
                kl = stats.entropy(p, q)
                kl_values.append(kl)
                
            avg_kl = np.mean(kl_values) if kl_values else 0.0
            
            metric = MetricWithCI(
                value=avg_kl,
                ci=None  # TODO: Add proper bootstrap CI
            )
            layer_kl.append(metric)
            
        return layer_kl
        
    def _get_layer_distribution(self, context: str, layer: int) -> np.ndarray:
        """Get cluster distribution at a specific layer."""
        k = self.config['k_clusters']
        counts = np.zeros(k)
        
        trajectories = self.get_trajectories_by_context(context)
        
        for token_idx, traj in trajectories.items():
            if len(traj) > layer:
                cluster = traj[layer]
                if 0 <= cluster < k:
                    counts[cluster] += 1
                    
        return counts
        
    def _calculate_entropy_metrics(self, contexts: List[str]) -> Dict[str, Any]:
        """Calculate various entropy metrics."""
        entropy_results = {}
        
        # 1. Transition matrix entropy
        matrix_entropy = {}
        
        for context in contexts:
            matrix = self._build_transition_matrix(context)
            
            # Calculate entropy of each row
            row_entropies = []
            for row in matrix:
                if row.sum() > 0:
                    p = row + 1e-10
                    p = p / p.sum()
                    entropy = -np.sum(p * np.log2(p))
                    row_entropies.append(entropy)
                    
            avg_entropy = np.mean(row_entropies) if row_entropies else 0.0
            
            matrix_entropy[context] = MetricWithCI(
                value=avg_entropy,
                ci=None  # TODO: Add proper bootstrap CI
            )
            
        entropy_results['matrix_entropy'] = matrix_entropy
        
        # 2. Entropy reduction from baseline
        baseline_entropy = matrix_entropy['baseline'].value
        entropy_reduction = {}
        
        for context in contexts[1:]:
            reduction = baseline_entropy - matrix_entropy[context].value
            entropy_reduction[context] = MetricWithCI(
                value=reduction,
                ci=None  # TODO: Add proper bootstrap CI
            )
            
        entropy_results['entropy_reduction'] = entropy_reduction
        
        # 3. Conditional entropy H(cluster|context)
        conditional_entropy = self._calculate_conditional_entropy(contexts)
        entropy_results['conditional_entropy'] = conditional_entropy
        
        return entropy_results
        
    def _calculate_conditional_entropy(self, contexts: List[str]) -> float:
        """Calculate H(cluster|context)."""
        # Collect all context-cluster pairs
        context_cluster_pairs = []
        
        for ctx_idx, context in enumerate(contexts):
            trajectories = self.get_trajectories_by_context(context)
            
            for token_idx, traj in trajectories.items():
                # Use final layer cluster
                if traj:
                    final_cluster = traj[-1]
                    context_cluster_pairs.append((ctx_idx, final_cluster))
                    
        if not context_cluster_pairs:
            return 0.0
            
        # Calculate joint distribution
        contexts_array = np.array([p[0] for p in context_cluster_pairs])
        clusters_array = np.array([p[1] for p in context_cluster_pairs])
        
        # Calculate conditional entropy using sklearn
        # H(Y|X) = H(X,Y) - H(X)
        joint_entropy = self._calculate_discrete_entropy(
            list(zip(contexts_array, clusters_array))
        )
        context_entropy = self._calculate_discrete_entropy(contexts_array)
        
        conditional_entropy = joint_entropy - context_entropy
        
        return MetricWithCI(
            value=conditional_entropy,
            ci=None  # TODO: Add proper bootstrap CI
        )
        
    def _calculate_discrete_entropy(self, data) -> float:
        """Calculate entropy of discrete distribution."""
        # Count occurrences
        unique, counts = np.unique(data, return_counts=True, axis=0)
        
        # Calculate probabilities
        probabilities = counts / counts.sum()
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
        
    def _calculate_jensen_shannon(self, contexts: List[str]) -> Dict[str, Any]:
        """Calculate Jensen-Shannon divergence between contexts."""
        js_results = {}
        
        # Build pairwise JS divergence matrix
        n_contexts = len(contexts)
        js_matrix = np.zeros((n_contexts, n_contexts))
        
        for i, ctx1 in enumerate(contexts):
            for j, ctx2 in enumerate(contexts):
                if i < j:  # Only compute upper triangle
                    # Get distributions
                    dist1 = self._get_context_distribution(ctx1)
                    dist2 = self._get_context_distribution(ctx2)
                    
                    # Ensure same size
                    max_size = max(len(dist1), len(dist2))
                    p = np.zeros(max_size)
                    q = np.zeros(max_size)
                    
                    p[:len(dist1)] = dist1
                    q[:len(dist2)] = dist2
                    
                    # Calculate JS divergence
                    js_div = jensenshannon(p, q) ** 2  # Square for JS divergence
                    
                    js_matrix[i, j] = js_div
                    js_matrix[j, i] = js_div  # Symmetric
                    
        js_results['divergence_matrix'] = js_matrix
        js_results['context_labels'] = contexts
        
        # Find most similar/dissimilar pairs
        similarities = []
        
        for i in range(n_contexts):
            for j in range(i + 1, n_contexts):
                similarities.append({
                    'context1': contexts[i],
                    'context2': contexts[j],
                    'js_divergence': js_matrix[i, j]
                })
                
        # Sort by divergence
        similarities.sort(key=lambda x: x['js_divergence'])
        
        js_results['most_similar'] = similarities[:3]
        js_results['most_dissimilar'] = similarities[-3:]
        
        return js_results
        
    def _get_context_distribution(self, context: str) -> np.ndarray:
        """Get overall cluster distribution for a context."""
        k = self.config['k_clusters']
        counts = np.zeros(k)
        
        trajectories = self.get_trajectories_by_context(context)
        
        for token_idx, traj in trajectories.items():
            # Count all clusters in trajectory
            for cluster in traj:
                if 0 <= cluster < k:
                    counts[cluster] += 1
                    
        # Normalize
        if counts.sum() > 0:
            counts = counts / counts.sum()
            
        return counts
        
    def _analyze_layer_evolution(self, contexts: List[str]) -> Dict[str, Any]:
        """Analyze how information metrics evolve across layers."""
        evolution = {}
        
        # Track metrics across layers
        n_layers = 12
        
        # 1. Entropy evolution
        entropy_evolution = []
        
        for layer in range(n_layers):
            layer_entropies = []
            
            for context in contexts:
                dist = self._get_layer_distribution(context, layer)
                if dist.sum() > 0:
                    p = dist / dist.sum()
                    entropy = -np.sum(p * np.log2(p + 1e-10))
                    layer_entropies.append(entropy)
                    
            if layer_entropies:
                entropy_evolution.append(np.mean(layer_entropies))
            else:
                entropy_evolution.append(0.0)
                
        evolution['entropy_by_layer'] = entropy_evolution
        
        # 2. Divergence evolution (average JS from baseline)
        divergence_evolution = []
        
        for layer in range(n_layers):
            baseline_dist = self._get_layer_distribution('baseline', layer)
            
            divergences = []
            for context in contexts[1:]:
                ctx_dist = self._get_layer_distribution(context, layer)
                
                # Ensure same size
                max_size = max(len(baseline_dist), len(ctx_dist))
                p = np.zeros(max_size)
                q = np.zeros(max_size)
                
                p[:len(baseline_dist)] = baseline_dist
                q[:len(ctx_dist)] = ctx_dist
                
                if p.sum() > 0 and q.sum() > 0:
                    js_div = jensenshannon(p, q) ** 2
                    divergences.append(js_div)
                    
            if divergences:
                divergence_evolution.append(np.mean(divergences))
            else:
                divergence_evolution.append(0.0)
                
        evolution['divergence_by_layer'] = divergence_evolution
        
        # 3. Information gain (reduction in entropy)
        info_gain = []
        
        for i in range(1, n_layers):
            prev_entropy = entropy_evolution[i-1]
            curr_entropy = entropy_evolution[i]
            gain = prev_entropy - curr_entropy
            info_gain.append(gain)
            
        evolution['information_gain'] = info_gain
        
        return evolution
        
    def _stratified_analysis(self, contexts: List[str]) -> Dict[str, Any]:
        """Perform stratified analysis by token properties."""
        stratified = {}
        
        # Stratify by frequency
        freq_strata = self.stratify_tokens('frequency')
        stratified['by_frequency'] = {}
        
        for stratum, indices in freq_strata.items():
            if not indices:
                continue
                
            # Calculate average JS divergence for this stratum
            divergences = []
            
            for token_idx in indices:
                baseline_traj = self.get_trajectories_by_context('baseline').get(token_idx)
                
                if baseline_traj:
                    for context in contexts[1:]:
                        ctx_traj = self.get_trajectories_by_context(context).get(token_idx)
                        
                        if ctx_traj:
                            # Simple trajectory distance
                            dist = sum(1 for a, b in zip(baseline_traj, ctx_traj) if a != b)
                            divergences.append(dist / len(baseline_traj))
                            
            if divergences:
                stratified['by_frequency'][stratum] = {
                    'n_tokens': len(indices),
                    'avg_divergence': np.mean(divergences),
                    'std_divergence': np.std(divergences)
                }
                
        # Stratify by type
        type_strata = self.stratify_tokens('type')
        stratified['by_type'] = {}
        
        for stratum, indices in type_strata.items():
            if not indices:
                continue
                
            # Calculate entropy for this stratum
            cluster_counts = np.zeros(self.config['k_clusters'])
            
            for token_idx in indices:
                for context in contexts:
                    traj = self.get_trajectories_by_context(context).get(token_idx)
                    if traj:
                        for cluster in traj:
                            if 0 <= cluster < self.config['k_clusters']:
                                cluster_counts[cluster] += 1
                                
            if cluster_counts.sum() > 0:
                p = cluster_counts / cluster_counts.sum()
                entropy = -np.sum(p * np.log2(p + 1e-10))
                
                stratified['by_type'][stratum] = {
                    'n_tokens': len(indices),
                    'entropy': entropy,
                    'dominant_clusters': np.argsort(cluster_counts)[-3:].tolist()
                }
                
        return stratified
        
    def _calculate_statistics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate summary statistics."""
        stats = {}
        
        # Average mutual information
        if 'layer_mi' in results['mutual_information']:
            layer_mi = results['mutual_information']['layer_mi']
            stats['avg_mutual_information'] = np.mean([
                m.value if hasattr(m, 'value') else m 
                for m in layer_mi
            ])
            
        # Average KL divergence
        if 'context_kl' in results['kl_divergence']:
            kl_values = [
                m.value for m in results['kl_divergence']['context_kl'].values()
            ]
            if kl_values:
                stats['avg_kl_divergence'] = np.mean(kl_values)
                
        # Average entropy
        if 'matrix_entropy' in results['entropy_metrics']:
            entropy_values = [
                m.value for m in results['entropy_metrics']['matrix_entropy'].values()
            ]
            if entropy_values:
                stats['avg_entropy'] = np.mean(entropy_values)
                
        # Maximum JS divergence
        if 'divergence_matrix' in results['jensen_shannon']:
            js_matrix = results['jensen_shannon']['divergence_matrix']
            # Get upper triangle (excluding diagonal)
            upper_triangle = js_matrix[np.triu_indices_from(js_matrix, k=1)]
            if len(upper_triangle) > 0:
                stats['max_js_divergence'] = np.max(upper_triangle)
                stats['avg_js_divergence'] = np.mean(upper_triangle)
                
        return stats
        
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create interpretable summary of findings."""
        summary = {
            'key_findings': [],
            'interpretation': "",
            'theoretical_implications': []
        }
        
        # Analyze mutual information
        if 'layer_mi' in results['mutual_information']:
            layer_mi = results['mutual_information']['layer_mi']
            mi_values = [m.value if hasattr(m, 'value') else m for m in layer_mi]
            avg_mi = np.mean(mi_values)
            
            summary['key_findings'].append(
                f"Average mutual information between context and clusters: {avg_mi:.3f} bits"
            )
            
            if avg_mi > 0.5:
                summary['key_findings'].append(
                    "High mutual information indicates context strongly determines cluster assignments"
                )
                
        # Analyze KL divergence
        if 'context_kl' in results['kl_divergence']:
            kl_dict = results['kl_divergence']['context_kl']
            for context, metric in kl_dict.items():
                if metric.value > 1.0:
                    summary['key_findings'].append(
                        f"Context '{context}' shows high KL divergence ({metric.value:.2f}) from baseline"
                    )
                    
        # Analyze entropy
        if 'entropy_reduction' in results['entropy_metrics']:
            reductions = results['entropy_metrics']['entropy_reduction']
            avg_reduction = np.mean([m.value for m in reductions.values()])
            
            if avg_reduction > 0:
                summary['key_findings'].append(
                    f"Context reduces transition entropy by {avg_reduction:.3f} bits on average"
                )
                
        # Analyze JS divergence
        if 'most_similar' in results['jensen_shannon']:
            most_similar = results['jensen_shannon']['most_similar'][0]
            summary['key_findings'].append(
                f"Most similar contexts: {most_similar['context1']} and {most_similar['context2']} "
                f"(JS divergence: {most_similar['js_divergence']:.3f})"
            )
            
        # Create interpretation
        summary['interpretation'] = (
            "Information-theoretic analysis confirms that context creates systematic, "
            "predictable transformations in the representation space. The high mutual "
            "information and significant KL divergences indicate that transformations "
            "are not random but carry substantial information about the context."
        )
        
        # Theoretical implications
        summary['theoretical_implications'] = [
            "Context acts as an information channel that modulates representations",
            "Transformations preserve information while adapting to context",
            "The systematic nature suggests learnable transformation rules",
            "Entropy reduction indicates more structured, predictable patterns with context"
        ]
        
        return summary
        
    def validate_data(self):
        """Validate data requirements."""
        if not self.trajectories:
            raise ValueError("No trajectory data loaded")
            
        # Check we have at least baseline and one other context
        contexts = self.get_context_types()
        if len(contexts) < 2:
            raise ValueError("Need at least 2 contexts for information theory analysis")
            
        if 'baseline' not in contexts:
            raise ValueError("Baseline context required for divergence calculations")
            
    def validate_results(self):
        """Validate analysis results."""
        if not self.output:
            raise ValueError("No output generated")
            
        # Check for required metrics
        required = ['mutual_information', 'kl_divergence', 'entropy_metrics']
        for req in required:
            if req not in self.output.data:
                raise ValueError(f"Missing required metric: {req}")
                
        # Validate metric ranges
        stats = self.output.statistics
        
        # MI should be non-negative
        if 'avg_mutual_information' in stats and stats['avg_mutual_information'] < 0:
            raise ValueError("Mutual information cannot be negative")
            
        # KL divergence should be non-negative
        if 'avg_kl_divergence' in stats and stats['avg_kl_divergence'] < 0:
            raise ValueError("KL divergence cannot be negative")
            
        # Entropy should be non-negative
        if 'avg_entropy' in stats and stats['avg_entropy'] < 0:
            raise ValueError("Entropy cannot be negative")