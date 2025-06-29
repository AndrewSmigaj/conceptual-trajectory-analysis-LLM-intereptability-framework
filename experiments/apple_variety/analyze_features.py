"""Analyze if features have predictive power for routing classification."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('arxiv_apple/apples_processed.csv')

# Filter to known routing classes
known_routing = ['fresh_premium', 'fresh_standard', 'juice']
df = df[df['routing'].isin(known_routing)].copy()

print(f"Total samples with known routing: {len(df)}")
print(f"\nClass distribution:")
print(df['routing'].value_counts())

# Get features
feature_cols = ['brix_numeric', 'firmness_numeric', 'starch_numeric', 'size_numeric', 'season_numeric']

# Check missing values
print(f"\nMissing values per feature:")
for col in feature_cols:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")

# Prepare data
X = df[feature_cols].fillna(df[feature_cols].mean()).values
routing_encoder = LabelEncoder()
y = routing_encoder.fit_transform(df['routing'])

# 1. Feature statistics by class
print("\n" + "="*50)
print("FEATURE STATISTICS BY CLASS")
print("="*50)

for feature in feature_cols:
    print(f"\n{feature}:")
    for routing in known_routing:
        values = df[df['routing'] == routing][feature].dropna()
        if len(values) > 0:
            print(f"  {routing}: mean={values.mean():.2f}, std={values.std():.2f}, "
                  f"min={values.min():.2f}, max={values.max():.2f}")

# 2. Feature correlations with target
print("\n" + "="*50)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*50)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importances:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# 3. Compare different models
print("\n" + "="*50)
print("MODEL COMPARISON (5-fold CV)")
print("="*50)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for name, model in models.items():
    # Use scaled features for logistic regression
    X_input = X_scaled if name == 'Logistic Regression' else X
    scores = cross_val_score(model, X_input, y, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 4. Decision Tree Analysis (most interpretable)
print("\n" + "="*50)
print("DECISION TREE ANALYSIS")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

print(f"Train accuracy: {dt.score(X_train, y_train):.3f}")
print(f"Test accuracy: {dt.score(X_test, y_test):.3f}")

# 5. Feature separability visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (feat1, feat2) in enumerate([
    ('brix_numeric', 'firmness_numeric'),
    ('brix_numeric', 'size_numeric'),
    ('firmness_numeric', 'size_numeric'),
    ('brix_numeric', 'starch_numeric'),
    ('firmness_numeric', 'starch_numeric'),
    ('size_numeric', 'starch_numeric')
]):
    ax = axes[i]
    for routing, color in zip(known_routing, ['green', 'blue', 'orange']):
        mask = df['routing'] == routing
        ax.scatter(df[mask][feat1], df[mask][feat2], 
                  c=color, label=routing, alpha=0.6, s=30)
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.legend()
    ax.set_title(f'{feat1} vs {feat2}')

plt.tight_layout()
plt.savefig('experiments/apple_variety/feature_separability.png', dpi=150)
plt.close()

# 6. Engineered features
print("\n" + "="*50)
print("ENGINEERED FEATURES ANALYSIS")
print("="*50)

# Add same engineered features as in the NN
df['sweetness_ratio'] = df['brix_numeric'] / (df['firmness_numeric'] + 1e-6)
df['quality_index'] = (
    0.3 * df['brix_numeric'] / df['brix_numeric'].max() +
    0.3 * df['firmness_numeric'] / df['firmness_numeric'].max() +
    0.2 * df['size_numeric'] / df['size_numeric'].max() +
    0.2 * (1 - df['starch_numeric'] / df['starch_numeric'].max())
)
df['firmness_sugar_interaction'] = df['firmness_numeric'] * df['brix_numeric']

engineered_cols = ['sweetness_ratio', 'quality_index', 'firmness_sugar_interaction']
all_features = feature_cols + engineered_cols

X_eng = df[all_features].fillna(df[all_features].mean()).values

rf_eng = RandomForestClassifier(n_estimators=100, random_state=42)
scores_eng = cross_val_score(rf_eng, X_eng, y, cv=5, scoring='accuracy')
print(f"Random Forest with engineered features: {scores_eng.mean():.3f} (+/- {scores_eng.std():.3f})")

# Check if any single feature is predictive
print("\n" + "="*50)
print("SINGLE FEATURE PREDICTIVE POWER")
print("="*50)

for feature in all_features:
    X_single = df[[feature]].fillna(df[feature].mean()).values
    scores = cross_val_score(DecisionTreeClassifier(max_depth=2), X_single, y, cv=5)
    print(f"{feature}: {scores.mean():.3f}")

print("\nConclusion: Features may not contain enough information to reliably predict routing class.")