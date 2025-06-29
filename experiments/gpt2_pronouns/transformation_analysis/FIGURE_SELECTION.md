# Figure Selection for Paper

## Recommended Figures (4 Essential + 1 Optional)

### Figure 1: Trajectory Fan Plot ✅ **ESSENTIAL**
- **File**: `trajectory_fan_plot.png` (1.56 MB)
- **Purpose**: Shows the core phenomenon - systematic divergence
- **Why Include**: This is your main result visualization. It immediately shows that context causes systematic, not random, divergence
- **Placement**: First figure in Results section

### Figure 2: Layer Evolution ✅ **ESSENTIAL**
- **File**: `layer_evolution.png` (277 KB)
- **Purpose**: Shows how metrics evolve through the network
- **Why Include**: Demonstrates that transformations develop systematically across layers, not randomly
- **Placement**: Second figure, after establishing the divergence pattern

### Figure 3: Context Similarity Dendrogram ✅ **ESSENTIAL**
- **File**: `context_similarity_dendrogram.png` (127 KB)
- **Purpose**: Shows which contexts create similar transformations
- **Why Include**: Proves transformations are linguistically structured (determiners cluster together)
- **Placement**: Third figure, supporting systematic nature

### Figure 4: Token Type Metrics ✅ **ESSENTIAL**
- **File**: `token_type_metrics.png` (52 KB)
- **Purpose**: Compares behavior across token types
- **Why Include**: Shows transformations respect linguistic categories
- **Placement**: Fourth figure, demonstrating generality

### Figure 5: Single Token Deep Dive ⭕ **OPTIONAL**
- **File**: `single_token_showcase.png` (351 KB)
- **Purpose**: Detailed view of one token's transformations
- **Why Include**: If space allows, provides concrete example
- **Alternative**: Could go in supplementary materials

### Not Recommended:
- **Transformation Geometry**: Too technical for main paper, better for supplementary

## Figure References in Text

```latex
% In Introduction
"As shown in Figure~\ref{fig:trajectory_fan}, context causes immediate and systematic divergence in token trajectories..."

% In Results 
"Figure~\ref{fig:layer_evolution} reveals how transformation metrics evolve across GPT-2's 12 layers..."

"The hierarchical clustering in Figure~\ref{fig:context_dendrogram} demonstrates that linguistically similar contexts..."

"Figure~\ref{fig:token_metrics} shows that different token types exhibit characteristic transformation patterns..."

% In Discussion
"The systematic patterns evident in Figures~\ref{fig:trajectory_fan}--\ref{fig:token_metrics} challenge..."
```

## LaTeX Figure Code

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/trajectory_fan_plot.pdf}
\caption{Trajectory divergence induced by linguistic context. Fifty representative tokens show systematic divergence from baseline trajectories (gray) when preceded by different context types. Determiners (blue), copulas (green), and modals (orange) create characteristic transformation patterns that emerge immediately at layer 0 and amplify through the network.}
\label{fig:trajectory_fan}
\end{figure*}

\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figures/layer_evolution.pdf}
\caption{Evolution of transformation metrics across GPT-2's layers. (a) Entropy decreases in deeper layers, indicating increasing structure. (b) Trajectory divergence accumulates monotonically. (c) Cluster stability shows a characteristic dip in middle layers. (d) Mutual information between context and transformations peaks in early-middle layers where linguistic processing occurs.}
\label{fig:layer_evolution}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\columnwidth]{figures/context_similarity_dendrogram.pdf}
\caption{Hierarchical clustering of linguistic contexts based on induced transformation patterns. Grammatically similar contexts (e.g., determiners "the" and "a") produce nearly identical transformations, while sentence-initial positions create unique patterns. Distance represents Jensen-Shannon divergence between transformation signatures.}
\label{fig:context_dendrogram}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figures/token_type_metrics.pdf}
\caption{Transformation characteristics by token type. Function words exhibit lower entropy but higher sparsity than content words, indicating more deterministic but selective transformations. Error bars show 95\% confidence intervals from bootstrap resampling (n=1000).}
\label{fig:token_metrics}
\end{figure}
```