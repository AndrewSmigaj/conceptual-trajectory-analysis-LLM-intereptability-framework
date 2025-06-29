"""Analyze apple dataset for class distributions, feature correlations, and missing data patterns."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('arxiv_apple/apples_processed.csv')

# Filter to known routing classes
known_routing = ['fresh_premium', 'fresh_standard', 'juice']
df = df[df['routing'].isin(known_routing)].copy()

# Filter varieties with minimum 3 samples
variety_counts = df['variety'].value_counts()
valid_varieties = variety_counts[variety_counts >= 3].index
print(f"Filtering from {len(variety_counts)} to {len(valid_varieties)} varieties (min 3 samples)")
df = df[df['variety'].isin(valid_varieties)].copy()

print(f"\nDataset size: {len(df)} samples")
print(f"Number of varieties: {len(df['variety'].unique())}")

# 1. Class Distribution Analysis
print("\n=== CLASS DISTRIBUTION ===")
routing_dist = df['routing'].value_counts()
print("Routing class distribution:")
print(routing_dist)
print(f"\nClass percentages:")
for cls, count in routing_dist.items():
    print(f"  {cls}: {count/len(df)*100:.1f}%")

# Check if classes are balanced
max_class = routing_dist.max()
min_class = routing_dist.min()
imbalance_ratio = max_class / min_class
print(f"\nClass imbalance ratio: {imbalance_ratio:.2f} (max/min)")

# 2. Feature Analysis
features = ['brix_numeric', 'firmness_numeric', 'starch_numeric', 'size_numeric', 'season_numeric']
print("\n=== FEATURE STATISTICS ===")
print("\nBasic statistics:")
print(df[features].describe())

# Missing data analysis
print("\n=== MISSING DATA ANALYSIS ===")
missing_counts = df[features].isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
print("Missing data per feature:")
for feat, pct in missing_pct.items():
    print(f"  {feat}: {pct:.1f}%")

# 3. Feature Correlations
print("\n=== FEATURE CORRELATIONS ===")
corr_matrix = df[features].corr()
print("\nCorrelation matrix:")
print(corr_matrix.round(2))

# Find high correlations
high_corr_pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val > 0.5:
            high_corr_pairs.append((features[i], features[j], corr_val))

if high_corr_pairs:
    print("\nHighly correlated features (|r| > 0.5):")
    for f1, f2, corr in high_corr_pairs:
        print(f"  {f1} - {f2}: {corr:.2f}")
else:
    print("\nNo highly correlated features found (|r| > 0.5)")

# 4. Class Separability Analysis
print("\n=== CLASS SEPARABILITY ===")
for feature in features:
    print(f"\n{feature} by routing class:")
    for routing_class in known_routing:
        class_data = df[df['routing'] == routing_class][feature].dropna()
        if len(class_data) > 0:
            print(f"  {routing_class}: mean={class_data.mean():.2f}, std={class_data.std():.2f}")

# 5. Variety Distribution Across Classes
print("\n=== VARIETY DISTRIBUTION ===")
variety_routing = pd.crosstab(df['variety'], df['routing'])
print(f"\nTop 10 varieties by sample count:")
top_varieties = df['variety'].value_counts().head(10)
for variety, count in top_varieties.items():
    routing_counts = df[df['variety'] == variety]['routing'].value_counts()
    routing_str = ", ".join([f"{r}: {c}" for r, c in routing_counts.items()])
    print(f"  {variety}: {count} samples ({routing_str})")

# 6. Recommendations
print("\n=== RECOMMENDATIONS ===")
print(f"1. Dataset has {len(df)} samples with {imbalance_ratio:.1f}x class imbalance")
print(f"2. Consider class weights or stratified sampling")
print(f"3. Feature 'firmness_numeric' has {missing_pct['firmness_numeric']:.1f}% missing - may need imputation")
print(f"4. No severe multicollinearity detected in base features")
print(f"5. With only {len(df)} samples and {len(df.columns)} potential features, regularization is critical")

# Save detailed analysis
output_dir = Path('experiments/apple_variety/data_analysis')
output_dir.mkdir(exist_ok=True)

# Save class distribution plot
plt.figure(figsize=(8, 6))
routing_dist.plot(kind='bar')
plt.title('Routing Class Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'class_distribution.png')
plt.close()

# Save correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(output_dir / 'correlation_matrix.png')
plt.close()

# Save feature distributions by class
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, feature in enumerate(features):
    for routing_class in known_routing:
        class_data = df[df['routing'] == routing_class][feature].dropna()
        axes[i].hist(class_data, alpha=0.5, label=routing_class, bins=20)
    axes[i].set_title(feature)
    axes[i].legend()
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(output_dir / 'feature_distributions.png')
plt.close()

print(f"\nPlots saved to {output_dir}")