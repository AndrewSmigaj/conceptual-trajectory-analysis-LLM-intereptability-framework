"""Create a synthetic apple quality dataset with clear routing patterns."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Number of samples per class
n_premium = 500
n_standard = 400
n_juice = 300
total_samples = n_premium + n_standard + n_juice

# Define clear patterns for each routing class
# Fresh Premium: High sugar (16-20), high firmness (3.2-3.6), low starch (8-12), large size (4-5)
# Fresh Standard: Medium sugar (14-17), medium firmness (2.8-3.3), medium starch (10-14), medium size (2-4)
# Juice: Lower sugar (12-16), lower firmness (2.5-3.0), high starch (12-16), any size (1-5)

def generate_samples(n_samples, routing_class):
    """Generate samples for a specific routing class with realistic patterns."""
    
    if routing_class == 'fresh_premium':
        # Premium apples: high quality across the board
        brix = np.random.normal(18, 1.2, n_samples)  # High sugar
        firmness = np.random.normal(3.4, 0.15, n_samples)  # High firmness
        starch = np.random.normal(10, 1.5, n_samples)  # Low starch (ripe)
        size = np.random.normal(4, 0.5, n_samples)  # Large
        season = np.random.choice([3, 4], n_samples, p=[0.7, 0.3])  # Late season
        
    elif routing_class == 'fresh_standard':
        # Standard fresh: good but not premium
        brix = np.random.normal(15.5, 1.5, n_samples)  # Medium sugar
        firmness = np.random.normal(3.0, 0.2, n_samples)  # Medium firmness
        starch = np.random.normal(12, 1.8, n_samples)  # Medium starch
        size = np.random.normal(3, 0.8, n_samples)  # Medium size
        season = np.random.choice([2, 3, 4], n_samples, p=[0.3, 0.5, 0.2])  # Mixed season
        
    else:  # juice
        # Juice apples: lower quality, overripe or damaged
        brix = np.random.normal(14, 1.8, n_samples)  # Lower sugar
        firmness = np.random.normal(2.7, 0.25, n_samples)  # Soft
        starch = np.random.normal(14, 2, n_samples)  # High starch or overripe
        size = np.random.uniform(1, 5, n_samples)  # Any size
        season = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.3, 0.3, 0.1, 0.1])  # Any season
    
    # Add some noise and overlap to make it realistic
    noise_factor = 0.1
    brix += np.random.normal(0, noise_factor * brix.std(), n_samples)
    firmness += np.random.normal(0, noise_factor * firmness.std(), n_samples)
    
    # Clip to realistic ranges
    brix = np.clip(brix, 10, 22)
    firmness = np.clip(firmness, 2.5, 3.8)
    starch = np.clip(starch, 8, 18)
    size = np.clip(size, 1, 5)
    season = np.clip(season, 1, 5).astype(int)
    
    return pd.DataFrame({
        'brix_numeric': brix,
        'firmness_numeric': firmness,
        'starch_numeric': starch,
        'size_numeric': size,
        'season_numeric': season,
        'routing': routing_class
    })

# Generate data for each class
premium_data = generate_samples(n_premium, 'fresh_premium')
standard_data = generate_samples(n_standard, 'fresh_standard')
juice_data = generate_samples(n_juice, 'juice')

# Combine all data
df = pd.concat([premium_data, standard_data, juice_data], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add variety names (random assignment)
varieties = ['Honeycrisp', 'Gala', 'Fuji', 'Granny Smith', 'Red Delicious', 
             'Golden Delicious', 'Braeburn', 'Pink Lady', 'Cosmic Crisp', 'Jazz',
             'Ambrosia', 'Cameo', 'Empire', 'McIntosh', 'Cortland']
df['variety'] = np.random.choice(varieties, len(df))

# Add some missing values to be realistic
missing_prob = 0.02
for col in ['firmness_numeric', 'starch_numeric']:
    missing_mask = np.random.random(len(df)) < missing_prob
    df.loc[missing_mask, col] = np.nan

# Add other columns to match original format
df['is_premium'] = df['routing'] == 'fresh_premium'

# Save the synthetic dataset
df.to_csv('experiments/apple_variety/synthetic_apples.csv', index=False)

# Print statistics
print("Synthetic Apple Dataset Created!")
print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['routing'].value_counts())

# Visualize the data
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Feature distributions by class
features = ['brix_numeric', 'firmness_numeric', 'starch_numeric', 'size_numeric', 'season_numeric']
for i, feature in enumerate(features):
    ax = axes[i]
    for routing in ['fresh_premium', 'fresh_standard', 'juice']:
        data = df[df['routing'] == routing][feature].dropna()
        ax.hist(data, alpha=0.5, label=routing, bins=20, density=True)
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(f'{feature} by routing class')

# Scatter plot of two most important features
ax = axes[5]
for routing, color in zip(['fresh_premium', 'fresh_standard', 'juice'], ['green', 'blue', 'orange']):
    mask = df['routing'] == routing
    ax.scatter(df[mask]['brix_numeric'], df[mask]['firmness_numeric'], 
              c=color, label=routing, alpha=0.5, s=20)
ax.set_xlabel('Brix (sugar content)')
ax.set_ylabel('Firmness')
ax.legend()
ax.set_title('Brix vs Firmness by routing class')

plt.tight_layout()
plt.savefig('experiments/apple_variety/synthetic_data_visualization.png', dpi=150)
plt.close()

# Test with simple models to verify predictability
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*50)
print("TESTING PREDICTABILITY")
print("="*50)

X = df[features].fillna(df[features].mean())
le = LabelEncoder()
y = le.fit_transform(df['routing'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest accuracy: {rf.score(X_test, y_test):.3f}")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree accuracy: {dt.score(X_test, y_test):.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature importances:")
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

print("\nSynthetic dataset saved to: experiments/apple_variety/synthetic_apples.csv")