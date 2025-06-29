"""
Linguistic Grouping Analysis

Tests if tokens with similar linguistic properties (POS tags, semantic categories)
have similar transformation patterns when context is added.

This analysis:
1. Groups tokens by linguistic properties
2. Analyzes transformation patterns within/between groups  
3. Tests statistical significance of group differences
4. Identifies which linguistic properties predict transformation behavior
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from .base_transformation_analysis import BaseTransformationAnalysis
from .output_schema import UnifiedAnalysisOutput, AnalysisMetadata


class LinguisticGroupingAnalysis(BaseTransformationAnalysis):
    """
    Analyzes whether tokens with similar linguistic properties exhibit similar
    transformation patterns under context.
    """
    
    def __init__(self, output_dir: str = None, config: dict = None, config_path: str = None):
        """Initialize the linguistic grouping analysis.
        
        Args:
            output_dir: Output directory path
            config: Configuration dictionary  
            config_path: Path to configuration file (legacy)
        """
        if config_path:
            # Legacy support
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                
        if not config:
            config = {}
            
        if not output_dir:
            output_dir = config.get('output_dir', 'results_transformation/linguistic_grouping')
            
        super().__init__(
            analysis_name="linguistic_grouping_analysis",
            output_dir=output_dir,
            config=config
        )
        self.linguistic_groups = {}
        self.transformation_vectors = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Run linguistic grouping analysis"""
        self.logger.info("Starting linguistic grouping analysis")
        
        results = {
            'group_analysis': {},
            'statistical_tests': {},
            'pattern_discovery': {},
            'group_similarities': {},
            'predictive_properties': {}
        }
        
        # Load token metadata
        token_metadata = self.data_loader.load_token_metadata()
        if not token_metadata:
            self.logger.warning("No token metadata available")
            return results
        
        # Get context types to analyze
        context_types = self.config.get('context_types', ['determiner_the', 'function_have'])
        
        # Prepare transformation vectors for each context
        for context_type in context_types:
            self.logger.info(f"Analyzing context: {context_type}")
            
            # Get transformation vectors
            transform_data = self._prepare_transformation_vectors('baseline', context_type)
            if not transform_data:
                continue
                
            self.transformation_vectors[context_type] = transform_data
            
            # Analyze different linguistic groupings
            grouping_results = {}
            
            # Group by POS tags
            if self._has_pos_tags(token_metadata):
                pos_analysis = self._analyze_by_linguistic_property(
                    transform_data, token_metadata, 'pos_tag', context_type
                )
                grouping_results['pos_tags'] = pos_analysis
            
            # Group by semantic categories
            if self._has_semantic_categories(token_metadata):
                semantic_analysis = self._analyze_by_linguistic_property(
                    transform_data, token_metadata, 'semantic_category', context_type
                )
                grouping_results['semantic_categories'] = semantic_analysis
            
            # Group by token type (function/content/subword)
            type_analysis = self._analyze_by_linguistic_property(
                transform_data, token_metadata, 'token_type', context_type
            )
            grouping_results['token_types'] = type_analysis
            
            # Group by frequency bins
            freq_analysis = self._analyze_by_frequency_bins(
                transform_data, token_metadata, context_type
            )
            grouping_results['frequency_bins'] = freq_analysis
            
            results['group_analysis'][context_type] = grouping_results
            
            # Statistical tests for group differences
            stats_results = self._perform_statistical_tests(grouping_results)
            results['statistical_tests'][context_type] = stats_results
            
            # Calculate between-group similarities
            similarities = self._calculate_group_similarities(grouping_results)
            results['group_similarities'][context_type] = similarities
        
        # Discover patterns across contexts
        if len(context_types) > 1:
            patterns = self._discover_cross_context_patterns(results['group_analysis'])
            results['pattern_discovery'] = patterns
        
        # Identify most predictive properties
        predictive = self._identify_predictive_properties(results)
        results['predictive_properties'] = predictive
        
        return results
    
    def _prepare_transformation_vectors(self, baseline_context: str, 
                                      target_context: str) -> Dict[int, Dict[str, Any]]:
        """Prepare transformation vectors from baseline to context"""
        transform_data = {}
        
        # Load trajectories
        trajectories = self.data_loader.load_unified_trajectories(k=self.config.get('k_clusters', 10))
        if not trajectories or 'trajectories' not in trajectories:
            return transform_data
        
        # Get tokens
        tokens = self.data_loader.get_all_tokens()[:self.config.get('max_tokens', 1000)]
        
        # Build transformation data for each token
        for token_idx, token in enumerate(tokens):
            baseline_path = None
            context_path = None
            
            # Find trajectories
            for key, traj_data in trajectories['trajectories'].items():
                if traj_data.get('token_str') == token:
                    if traj_data.get('context_frame') == baseline_context:
                        baseline_path = traj_data.get('path')
                    elif traj_data.get('context_frame') == target_context:
                        context_path = traj_data.get('path')
            
            if baseline_path and context_path:
                # Convert to cluster transition vector
                transition_vector = self._compute_transition_vector(
                    baseline_path, context_path
                )
                
                transform_data[token_idx] = {
                    'token': token,
                    'baseline_path': baseline_path,
                    'context_path': context_path,
                    'transition_vector': transition_vector,
                    'divergence_layer': self._find_divergence_layer(baseline_path, context_path)
                }
        
        return transform_data
    
    def _compute_transition_vector(self, baseline_path: List[int], 
                                 context_path: List[int]) -> np.ndarray:
        """Compute transition vector representation"""
        # Create a vector encoding the transformation
        # Options: concatenate paths, compute differences, or transition probabilities
        
        # Method 1: Concatenated one-hot encoding
        n_clusters = self.config.get('k_clusters', 10)
        n_layers = len(baseline_path)
        
        vector = []
        for layer in range(n_layers):
            # One-hot encode transition from baseline to context cluster
            transition = np.zeros(n_clusters * n_clusters)
            from_cluster = baseline_path[layer]
            to_cluster = context_path[layer]
            transition_idx = from_cluster * n_clusters + to_cluster
            transition[transition_idx] = 1
            vector.extend(transition)
        
        return np.array(vector)
    
    def _find_divergence_layer(self, path1: List[int], path2: List[int]) -> int:
        """Find first layer where paths diverge"""
        for i, (c1, c2) in enumerate(zip(path1, path2)):
            if c1 != c2:
                return i
        return len(path1)  # Paths are identical
    
    def _has_pos_tags(self, metadata: Dict) -> bool:
        """Check if POS tag data is available"""
        return any('pos_tag' in str(v) or 'pos_tags' in metadata for v in metadata.values())
    
    def _has_semantic_categories(self, metadata: Dict) -> bool:
        """Check if semantic category data is available"""
        return any('semantic_category' in str(v) or 'semantic_categories' in metadata 
                  for v in metadata.values())
    
    def _analyze_by_linguistic_property(self, transform_data: Dict[int, Dict],
                                      token_metadata: Dict,
                                      property_name: str,
                                      context: str) -> Dict[str, Any]:
        """Analyze transformations grouped by a linguistic property"""
        # Group tokens by property
        groups = defaultdict(list)
        
        for token_idx, transform_info in transform_data.items():
            # Get property value for this token
            prop_value = self._get_token_property(token_idx, property_name, token_metadata)
            if prop_value:
                groups[prop_value].append(token_idx)
        
        # Analyze each group
        group_analysis = {}
        
        for group_name, token_indices in groups.items():
            if len(token_indices) < 3:  # Skip small groups
                continue
                
            # Get transformation vectors for group
            vectors = [transform_data[idx]['transition_vector'] for idx in token_indices]
            vectors = np.array(vectors)
            
            # Calculate within-group statistics
            analysis = {
                'n_tokens': len(token_indices),
                'mean_vector': np.mean(vectors, axis=0),
                'std_vector': np.std(vectors, axis=0),
                'cohesion': self._calculate_group_cohesion(vectors),
                'divergence_layers': [transform_data[idx]['divergence_layer'] 
                                    for idx in token_indices],
                'mean_divergence_layer': np.mean([transform_data[idx]['divergence_layer'] 
                                                for idx in token_indices])
            }
            
            group_analysis[group_name] = analysis
        
        # Calculate between-group distances
        if len(group_analysis) > 1:
            distances = self._calculate_between_group_distances(group_analysis)
            
            return {
                'groups': group_analysis,
                'between_group_distances': distances,
                'n_groups': len(group_analysis),
                'property': property_name
            }
        
        return {'groups': group_analysis, 'property': property_name}
    
    def _get_token_property(self, token_idx: int, property_name: str, 
                          metadata: Dict) -> Optional[str]:
        """Get linguistic property value for a token"""
        # Handle different metadata structures
        token_key = str(token_idx)
        
        if property_name == 'pos_tag':
            if 'pos_tags' in metadata and token_key in metadata['pos_tags']:
                return metadata['pos_tags'][token_key]
            elif token_key in metadata and 'pos_tag' in metadata[token_key]:
                return metadata[token_key]['pos_tag']
                
        elif property_name == 'semantic_category':
            if 'semantic_categories' in metadata and token_key in metadata['semantic_categories']:
                return metadata['semantic_categories'][token_key]
            elif token_key in metadata and 'semantic_category' in metadata[token_key]:
                return metadata[token_key]['semantic_category']
                
        elif property_name == 'token_type':
            if 'types' in metadata and token_key in metadata['types']:
                return metadata['types'][token_key]
            elif token_key in metadata and 'token_type' in metadata[token_key]:
                return metadata[token_key]['token_type']
        
        return None
    
    def _calculate_group_cohesion(self, vectors: np.ndarray) -> float:
        """Calculate cohesion (inverse of average pairwise distance) within group"""
        if len(vectors) < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = pairwise_distances(vectors, metric='euclidean')
        
        # Get upper triangle (excluding diagonal)
        n = len(vectors)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_dists = distances[upper_indices]
        
        # Cohesion is inverse of mean distance
        mean_dist = np.mean(pairwise_dists)
        cohesion = 1.0 / (1.0 + mean_dist)  # Normalized to [0,1]
        
        return float(cohesion)
    
    def _calculate_between_group_distances(self, group_analysis: Dict) -> Dict[str, float]:
        """Calculate distances between group centroids"""
        distances = {}
        group_names = list(group_analysis.keys())
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                centroid1 = group_analysis[group1]['mean_vector']
                centroid2 = group_analysis[group2]['mean_vector']
                
                dist = np.linalg.norm(centroid1 - centroid2)
                distances[f'{group1}_vs_{group2}'] = float(dist)
        
        return distances
    
    def _analyze_by_frequency_bins(self, transform_data: Dict[int, Dict],
                                  token_metadata: Dict,
                                  context: str) -> Dict[str, Any]:
        """Analyze transformations grouped by frequency bins"""
        # Get token frequencies
        frequencies = {}
        
        for token_idx in transform_data.keys():
            freq = self._get_token_frequency(token_idx, token_metadata)
            if freq is not None:
                frequencies[token_idx] = freq
        
        if not frequencies:
            return {}
        
        # Create frequency bins (quartiles)
        freq_values = list(frequencies.values())
        quartiles = np.percentile(freq_values, [25, 50, 75])
        
        # Group tokens by frequency bin
        groups = {
            'very_low': [],
            'low': [],
            'medium': [],
            'high': []
        }
        
        for token_idx, freq in frequencies.items():
            if freq <= quartiles[0]:
                groups['very_low'].append(token_idx)
            elif freq <= quartiles[1]:
                groups['low'].append(token_idx)
            elif freq <= quartiles[2]:
                groups['medium'].append(token_idx)
            else:
                groups['high'].append(token_idx)
        
        # Analyze each frequency group
        freq_analysis = {}
        
        for bin_name, token_indices in groups.items():
            if len(token_indices) < 3:
                continue
                
            vectors = [transform_data[idx]['transition_vector'] for idx in token_indices]
            vectors = np.array(vectors)
            
            freq_analysis[bin_name] = {
                'n_tokens': len(token_indices),
                'mean_vector': np.mean(vectors, axis=0),
                'cohesion': self._calculate_group_cohesion(vectors),
                'mean_divergence_layer': np.mean([transform_data[idx]['divergence_layer'] 
                                                for idx in token_indices]),
                'frequency_range': self._get_frequency_range(token_indices, frequencies)
            }
        
        return {
            'groups': freq_analysis,
            'property': 'frequency_bin',
            'bin_thresholds': quartiles.tolist()
        }
    
    def _get_token_frequency(self, token_idx: int, metadata: Dict) -> Optional[float]:
        """Get frequency value for a token"""
        token_key = str(token_idx)
        
        if 'frequencies' in metadata and token_key in metadata['frequencies']:
            return float(metadata['frequencies'][token_key])
        elif 'token_frequencies' in metadata and token_key in metadata['token_frequencies']:
            return float(metadata['token_frequencies'][token_key])
        elif token_key in metadata and 'frequency' in metadata[token_key]:
            return float(metadata[token_key]['frequency'])
        
        return None
    
    def _get_frequency_range(self, token_indices: List[int], 
                           frequencies: Dict[int, float]) -> Tuple[float, float]:
        """Get min and max frequency for a group of tokens"""
        freqs = [frequencies[idx] for idx in token_indices if idx in frequencies]
        if freqs:
            return (min(freqs), max(freqs))
        return (0.0, 0.0)
    
    def _perform_statistical_tests(self, grouping_results: Dict) -> Dict[str, Any]:
        """Perform statistical tests for group differences"""
        test_results = {}
        
        for property_name, analysis in grouping_results.items():
            if 'groups' not in analysis or len(analysis['groups']) < 2:
                continue
            
            # Prepare data for statistical tests
            groups_data = []
            group_labels = []
            
            for group_name, group_info in analysis['groups'].items():
                if 'cohesion' in group_info:
                    # Use cohesion as test statistic
                    groups_data.append([group_info['cohesion']])
                    group_labels.append(group_name)
            
            if len(groups_data) < 2:
                continue
            
            # Kruskal-Wallis test (non-parametric)
            try:
                h_stat, p_value = stats.kruskal(*groups_data)
                
                test_results[property_name] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': float(h_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'n_groups': len(groups_data)
                }
                
                # If significant, perform post-hoc pairwise tests
                if p_value < 0.05 and len(groups_data) > 2:
                    pairwise = self._perform_pairwise_tests(analysis['groups'])
                    test_results[property_name]['pairwise_tests'] = pairwise
                    
            except Exception as e:
                self.logger.warning(f"Statistical test failed for {property_name}: {e}")
        
        return test_results
    
    def _perform_pairwise_tests(self, groups: Dict) -> Dict[str, Dict]:
        """Perform pairwise Mann-Whitney U tests between groups"""
        pairwise_results = {}
        group_names = list(groups.keys())
        
        for i, group1 in enumerate(group_names):
            for j, group2 in enumerate(group_names[i+1:], i+1):
                # For now, compare mean divergence layers
                data1 = [groups[group1]['mean_divergence_layer']]
                data2 = [groups[group2]['mean_divergence_layer']]
                
                try:
                    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    pairwise_results[f'{group1}_vs_{group2}'] = {
                        'statistic': float(u_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                except:
                    continue
        
        return pairwise_results
    
    def _calculate_group_similarities(self, grouping_results: Dict) -> Dict[str, Any]:
        """Calculate similarity matrix between linguistic groups"""
        similarities = {}
        
        for property_name, analysis in grouping_results.items():
            if 'groups' not in analysis:
                continue
                
            groups = analysis['groups']
            if len(groups) < 2:
                continue
            
            # Build similarity matrix
            group_names = list(groups.keys())
            n_groups = len(group_names)
            sim_matrix = np.zeros((n_groups, n_groups))
            
            for i, group1 in enumerate(group_names):
                for j, group2 in enumerate(group_names):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                    else:
                        # Calculate cosine similarity between mean vectors
                        vec1 = groups[group1]['mean_vector']
                        vec2 = groups[group2]['mean_vector']
                        
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        
                        if norm1 > 0 and norm2 > 0:
                            cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                            sim_matrix[i, j] = cos_sim
            
            similarities[property_name] = {
                'similarity_matrix': sim_matrix.tolist(),
                'group_names': group_names,
                'mean_similarity': float(np.mean(sim_matrix[np.triu_indices(n_groups, k=1)]))
            }
        
        return similarities
    
    def _discover_cross_context_patterns(self, all_group_analyses: Dict) -> Dict[str, Any]:
        """Discover patterns that hold across different contexts"""
        patterns = {
            'consistent_groupings': [],
            'context_specific_effects': [],
            'universal_patterns': []
        }
        
        # Find properties that show consistent grouping across contexts
        property_names = set()
        for context_analysis in all_group_analyses.values():
            property_names.update(context_analysis.keys())
        
        for prop in property_names:
            prop_data = []
            
            for context, analysis in all_group_analyses.items():
                if prop in analysis and 'groups' in analysis[prop]:
                    prop_data.append({
                        'context': context,
                        'groups': analysis[prop]['groups']
                    })
            
            if len(prop_data) > 1:
                # Check consistency
                consistency = self._check_grouping_consistency(prop_data)
                
                if consistency['is_consistent']:
                    patterns['consistent_groupings'].append({
                        'property': prop,
                        'consistency_score': consistency['score'],
                        'description': f"{prop} shows consistent grouping across contexts"
                    })
        
        return patterns
    
    def _check_grouping_consistency(self, prop_data: List[Dict]) -> Dict[str, Any]:
        """Check if grouping patterns are consistent across contexts"""
        # Simple consistency check: do the same groups appear?
        all_groups = []
        for data in prop_data:
            all_groups.append(set(data['groups'].keys()))
        
        # Calculate Jaccard similarity
        intersection = set.intersection(*all_groups)
        union = set.union(*all_groups)
        
        jaccard = len(intersection) / len(union) if union else 0
        
        return {
            'is_consistent': jaccard > 0.7,
            'score': jaccard,
            'common_groups': list(intersection)
        }
    
    def _identify_predictive_properties(self, results: Dict) -> Dict[str, Any]:
        """Identify which linguistic properties best predict transformation patterns"""
        predictive_scores = {}
        
        # Analyze statistical test results
        for context, stats_results in results.get('statistical_tests', {}).items():
            for prop, test_result in stats_results.items():
                if test_result.get('significant', False):
                    # Use p-value inverse as predictive score
                    score = -np.log10(test_result['p_value'] + 1e-10)
                    
                    if prop not in predictive_scores:
                        predictive_scores[prop] = []
                    predictive_scores[prop].append(score)
        
        # Average scores across contexts
        avg_scores = {}
        for prop, scores in predictive_scores.items():
            avg_scores[prop] = np.mean(scores)
        
        # Rank properties
        ranked_properties = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'ranked_properties': ranked_properties,
            'most_predictive': ranked_properties[0] if ranked_properties else None,
            'predictive_scores': avg_scores
        }
    
    def validate_data(self) -> None:
        """Validate loaded data"""
        if not hasattr(self, 'data_loader') or self.data_loader is None:
            raise ValueError("Data loader not initialized")
        
        # Check we have trajectory data
        trajectories = self.data_loader.load_unified_trajectories(k=self.config.get('k_clusters', 10))
        if not trajectories:
            raise ValueError("No trajectory data found")
        
        # Check we have token metadata
        metadata = self.data_loader.load_token_metadata()
        if not metadata:
            raise ValueError("No token metadata found")
        
        self.logger.info("Data validation passed")
    
    def validate_results(self) -> None:
        """Validate analysis results"""
        if not hasattr(self, 'output') or self.output is None:
            raise ValueError("No output generated")
        
        # Check required fields
        if 'group_analysis' not in self.output.data:
            raise ValueError("Missing group analysis results")
        
        if 'statistical_tests' not in self.output.data:
            raise ValueError("Missing statistical test results")
        
        self.logger.info("Results validation passed")
    
    def _create_visualizations(self) -> List[Dict[str, Any]]:
        """Create visualizations for linguistic grouping analysis"""
        viz_list = []
        
        # Group cohesion plot
        viz_list.append({
            'name': 'group_cohesion',
            'path': str(self.output_dir / 'group_cohesion.png'),
            'type': 'bar_chart',
            'description': 'Transformation cohesion by linguistic group'
        })
        
        # Similarity heatmap
        viz_list.append({
            'name': 'group_similarity',
            'path': str(self.output_dir / 'group_similarity.png'),
            'type': 'heatmap',
            'description': 'Similarity matrix between linguistic groups'
        })
        
        # Divergence layer distribution
        viz_list.append({
            'name': 'divergence_layers',
            'path': str(self.output_dir / 'divergence_layers.png'),
            'type': 'box_plot',
            'description': 'Distribution of divergence layers by group'
        })
        
        return viz_list
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        key_findings = []
        
        # Analyze predictive properties
        predictive = results.get('predictive_properties', {})
        if predictive.get('most_predictive'):
            prop, score = predictive['most_predictive']
            key_findings.append(f"{prop} is the most predictive linguistic property")
        
        # Analyze statistical tests
        significant_props = []
        for context, tests in results.get('statistical_tests', {}).items():
            for prop, test in tests.items():
                if test.get('significant'):
                    significant_props.append(prop)
        
        if significant_props:
            key_findings.append(f"Significant group differences found for: {', '.join(set(significant_props))}")
        
        # Analyze patterns
        patterns = results.get('pattern_discovery', {})
        if patterns.get('consistent_groupings'):
            key_findings.append(f"{len(patterns['consistent_groupings'])} properties show consistent patterns across contexts")
        
        return {
            'key_findings': key_findings,
            'interpretation': self._generate_interpretation(results),
            'next_steps': [
                "Investigate specific transformation patterns for each linguistic group",
                "Test if linguistic properties can predict transformation trajectories",
                "Compare findings across different model architectures"
            ]
        }
    
    def _generate_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate interpretation of results"""
        interpretation = ""
        
        # Check if linguistic properties matter
        stats_results = results.get('statistical_tests', {})
        n_significant = sum(1 for tests in stats_results.values() 
                          for test in tests.values() 
                          if test.get('significant', False))
        
        if n_significant > 0:
            interpretation = (
                "The analysis reveals that linguistic properties significantly influence "
                "how tokens transform under context. This suggests that transformers learn "
                "systematic mappings that respect linguistic structure, with different "
                "grammatical categories undergoing distinct types of transformations."
            )
            
            # Add specific insights
            predictive = results.get('predictive_properties', {})
            if predictive.get('most_predictive'):
                prop, _ = predictive['most_predictive']
                interpretation += (
                    f" The {prop} appears to be particularly important in determining "
                    "transformation patterns, indicating that this linguistic dimension "
                    "is fundamental to how context modifies representations."
                )
        else:
            interpretation = (
                "The analysis did not find significant differences in transformation "
                "patterns based on linguistic properties. This suggests that context "
                "effects may operate more uniformly across token types, or that the "
                "linguistic categorizations used do not capture the relevant distinctions "
                "for transformation behavior."
            )
        
        return interpretation


if __name__ == "__main__":
    # Example usage
    analysis = LinguisticGroupingAnalysis("config_unified.yaml")
    analysis.run()