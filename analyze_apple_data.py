import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv('arxiv_apple/apples_processed.csv')

print("=== DATASET OVERVIEW ===")
print(f"Total samples: {len(df)}")
print(f"Number of features: {df.shape[1]}")

# Focus on the 5 numeric features mentioned
numeric_features = ['brix_numeric', 'firmness_numeric', 'starch_numeric', 'size_numeric', 'season_numeric']
target = 'routing'

print(f"\nTarget classes:")
print(df[target].value_counts())
print(f"\nTarget class proportions:")
print(df[target].value_counts(normalize=True))

# Check for missing values
print("\n=== MISSING VALUES ===")
for col in numeric_features + [target]:
    missing = df[col].isna().sum()
    print(f"{col}: {missing} ({missing/len(df)*100:.1f}%)")

# Basic statistics for numeric features
print("\n=== BASIC STATISTICS ===")
for col in numeric_features:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.3f}")
    print(f"  Std: {df[col].std():.3f}")
    print(f"  Min: {df[col].min():.3f}")
    print(f"  Max: {df[col].max():.3f}")
    print(f"  Unique values: {df[col].nunique()}")

# Analyze variation within each feature
print("\n=== FEATURE VARIATION ANALYSIS ===")
for col in numeric_features:
    cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
    print(f"{col}: CV = {cv:.3f}")

# Check correlation between features
print("\n=== FEATURE CORRELATIONS ===")
corr_matrix = df[numeric_features].corr()
print(corr_matrix)

# Analyze discriminative power - check if features differ between classes
print("\n=== DISCRIMINATIVE POWER ANALYSIS ===")
for col in numeric_features:
    print(f"\n{col} by routing class:")
    for route in df[target].unique():
        if pd.notna(route):
            subset = df[df[target] == route][col].dropna()
            print(f"  {route}: mean={subset.mean():.3f}, std={subset.std():.3f}, n={len(subset)}")
    
    # ANOVA test
    groups = [df[df[target] == route][col].dropna() for route in df[target].unique() if pd.notna(route)]
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  ANOVA: F={f_stat:.3f}, p={p_value:.3f}")

# Check for extreme class imbalance within feature values
print("\n=== FEATURE VALUE DISTRIBUTIONS BY CLASS ===")
for col in numeric_features:
    print(f"\n{col}:")
    # Check if most values are the same
    value_counts = df[col].value_counts()
    top_value = value_counts.index[0]
    top_count = value_counts.iloc[0]
    print(f"  Most common value: {top_value} ({top_count/len(df)*100:.1f}% of data)")
    
    # Check distribution overlap
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for route in df[target].unique():
        if pd.notna(route):
            subset = df[df[target] == route][col].dropna()
            if len(subset) > 0:
                ax.hist(subset, alpha=0.5, label=route, bins=20)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title(f'Distribution of {col} by routing class')
    plt.tight_layout()
    plt.savefig(f'apple_dist_{col}.png')
    plt.close()

# Feature engineering ideas
print("\n=== FEATURE ENGINEERING INSIGHTS ===")
# Check if size_numeric has very limited variation
size_values = df['size_numeric'].value_counts()
print(f"\nsize_numeric distribution:")
print(size_values)

# Check if season_numeric has very limited variation
season_values = df['season_numeric'].value_counts()
print(f"\nseason_numeric distribution:")
print(season_values)

# Look for patterns in unknown routing
unknown_mask = df[target] == 'unknown'
print(f"\nUnknown routing samples: {unknown_mask.sum()}")
if unknown_mask.sum() > 0:
    print("Features for unknown routing:")
    for col in numeric_features:
        unknown_vals = df[unknown_mask][col].dropna()
        if len(unknown_vals) > 0:
            print(f"  {col}: mean={unknown_vals.mean():.3f}")

# Create correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('apple_correlation_matrix.png')
plt.close()

# Summary insights
print("\n=== KEY INSIGHTS ===")
print("1. Check if features have sufficient variation")
print("2. Look for high correlations between features")
print("3. Examine if features can distinguish between routing classes")
print("4. Identify any data quality issues")