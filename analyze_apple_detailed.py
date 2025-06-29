import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the data
df = pd.read_csv('arxiv_apple/apples_processed.csv')

# Focus on the 5 numeric features and valid routing classes
numeric_features = ['brix_numeric', 'firmness_numeric', 'starch_numeric', 'size_numeric', 'season_numeric']
valid_routes = ['fresh_premium', 'fresh_standard', 'juice']

# Filter out unknown routing for this analysis
df_known = df[df['routing'].isin(valid_routes)].copy()

print("=== CRITICAL ISSUES ANALYSIS ===")
print(f"\n1. CLASS IMBALANCE:")
print(f"- Unknown routing: {(df['routing'] == 'unknown').sum()} samples (69.7% of data!)")
print(f"- Known routing: {len(df_known)} samples (30.3% of data)")
print(f"\nKnown class distribution:")
print(df_known['routing'].value_counts())

# Check for outliers
print(f"\n2. OUTLIER ANALYSIS:")
# Check brix_numeric outlier
brix_outlier = df['brix_numeric'].max()
print(f"- brix_numeric max value: {brix_outlier} (likely data entry error)")
brix_without_outlier = df[df['brix_numeric'] < 100]['brix_numeric']
print(f"- brix_numeric without outlier: mean={brix_without_outlier.mean():.2f}, std={brix_without_outlier.std():.2f}")

# Remove outliers for clean analysis
df_clean = df_known[df_known['brix_numeric'] < 100].copy()

# Check starch_numeric outlier
starch_outlier = df['starch_numeric'].max()
print(f"- starch_numeric max value: {starch_outlier} (likely data entry error)")
df_clean = df_clean[df_clean['starch_numeric'] < 50].copy()

print(f"\n3. FEATURE VALUE CONCENTRATION:")
# Size numeric
size_dist = df_clean['size_numeric'].value_counts(normalize=True)
print(f"\nsize_numeric:")
for val, pct in size_dist.items():
    print(f"  Value {val}: {pct*100:.1f}%")
print(f"  -> 88.9% of samples have size=3, very little variation!")

# Season numeric
season_dist = df_clean['season_numeric'].value_counts(normalize=True)
print(f"\nseason_numeric:")
for val, pct in season_dist.items():
    print(f"  Value {val}: {pct*100:.1f}%")
print(f"  -> 78.8% of samples have season=3, limited variation!")

print(f"\n4. FEATURE OVERLAP BETWEEN CLASSES:")
# For each feature, calculate overlap between class distributions
for feature in numeric_features:
    print(f"\n{feature}:")
    # Get ranges for each class
    for route in valid_routes:
        subset = df_clean[df_clean['routing'] == route][feature].dropna()
        if len(subset) > 0:
            print(f"  {route}: [{subset.min():.2f}, {subset.max():.2f}] (mean={subset.mean():.2f})")
    
    # Calculate pairwise overlap
    premium = df_clean[df_clean['routing'] == 'fresh_premium'][feature].dropna()
    standard = df_clean[df_clean['routing'] == 'fresh_standard'][feature].dropna()
    juice = df_clean[df_clean['routing'] == 'juice'][feature].dropna()
    
    # KS test for distribution differences
    if len(premium) > 0 and len(standard) > 0:
        ks_stat, p_val = stats.ks_2samp(premium, standard)
        print(f"  KS test premium vs standard: p={p_val:.3f}")
    if len(premium) > 0 and len(juice) > 0:
        ks_stat, p_val = stats.ks_2samp(premium, juice)
        print(f"  KS test premium vs juice: p={p_val:.3f}")

# Visualize feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    for route in valid_routes:
        subset = df_clean[df_clean['routing'] == route][feature].dropna()
        if len(subset) > 0:
            ax.violinplot([subset], positions=[valid_routes.index(route)], widths=0.6, 
                         showmeans=True, showextrema=True)
    ax.set_xticks(range(len(valid_routes)))
    ax.set_xticklabels(valid_routes)
    ax.set_title(f'{feature} Distribution by Class')
    ax.set_ylabel(feature)

plt.tight_layout()
plt.savefig('apple_violin_plots.png', dpi=150)
plt.close()

# PCA Analysis
print(f"\n5. DIMENSIONALITY REDUCTION ANALYSIS:")
# Prepare data
X = df_clean[numeric_features].fillna(df_clean[numeric_features].mean())
y = df_clean['routing']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Explained Variance Ratios:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.1f}%")
print(f"  Cumulative (first 2): {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")

# Plot PCA
plt.figure(figsize=(10, 8))
colors = {'fresh_premium': 'blue', 'fresh_standard': 'green', 'juice': 'red'}
for route in valid_routes:
    mask = y == route
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[route], label=route, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Apple Quality Classes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('apple_pca.png', dpi=150)
plt.close()

# Feature importance from PCA
print(f"\nPCA Feature Loadings (PC1):")
loadings = pd.DataFrame(
    pca.components_[0],
    columns=['Loading'],
    index=numeric_features
).sort_values('Loading', key=abs, ascending=False)
print(loadings)

print(f"\n6. INFORMATION CONTENT ANALYSIS:")
# Calculate entropy for each feature
for feature in numeric_features:
    # Discretize into bins
    bins = pd.qcut(df_clean[feature].dropna(), q=10, duplicates='drop')
    probs = bins.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    print(f"{feature}: entropy = {entropy:.2f} bits")

print(f"\n7. MUTUAL INFORMATION WITH TARGET:")
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_filled = X.fillna(X.mean())

mi_scores = mutual_info_classif(X_filled, y_encoded, random_state=42)
mi_df = pd.DataFrame({
    'Feature': numeric_features,
    'MI Score': mi_scores
}).sort_values('MI Score', ascending=False)
print(mi_df)

print("\n=== SUMMARY OF PROBLEMS ===")
print("1. MASSIVE CLASS IMBALANCE: 69.7% of data has 'unknown' routing")
print("2. DATA QUALITY: Extreme outliers in brix_numeric (1601) and starch_numeric (154)")
print("3. LOW VARIATION: size_numeric (89% are '3') and season_numeric (79% are '3')")
print("4. POOR SEPARATION: Features show high overlap between classes")
print("5. LOW INFORMATION: Most features have similar distributions across classes")
print("6. INSUFFICIENT FEATURES: Only 5 numeric features to distinguish 3 complex quality classes")