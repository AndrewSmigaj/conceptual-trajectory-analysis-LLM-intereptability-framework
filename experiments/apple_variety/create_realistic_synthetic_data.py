"""Create a realistic synthetic apple quality dataset with true variety-specific characteristics and pricing."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

np.random.seed(42)

# Define realistic variety characteristics based on agricultural data
VARIETY_CHARACTERISTICS = {
    # Premium varieties
    'Honeycrisp': {
        'brix_range': (14, 16),
        'firmness_range': (3.6, 3.8),
        'starch_range': (12, 14),
        'harvest_season': 3,
        'market_position': 'premium',
        'price_per_lb': 3.24  # Average of $2.49-3.99
    },
    'Cosmic Crisp': {
        'brix_range': (13, 16),
        'firmness_range': (3.8, 4.0),
        'starch_range': (10, 12),
        'harvest_season': 4,
        'market_position': 'premium',
        'price_per_lb': 2.93  # Average of $2.35-3.50
    },
    'Pink Lady': {
        'brix_range': (13, 15),
        'firmness_range': (3.5, 3.8),
        'starch_range': (14, 16),
        'harvest_season': 5,
        'market_position': 'premium',
        'price_per_lb': 3.00
    },
    'Jazz': {
        'brix_range': (14, 16),
        'firmness_range': (3.6, 3.9),
        'starch_range': (12, 14),
        'harvest_season': 4,
        'market_position': 'premium',
        'price_per_lb': 3.00
    },
    'Ambrosia': {
        'brix_range': (14, 16),
        'firmness_range': (3.4, 3.7),
        'starch_range': (11, 13),
        'harvest_season': 4,
        'market_position': 'premium',
        'price_per_lb': 2.50
    },
    
    # Standard varieties
    'Gala': {
        'brix_range': (12, 14),
        'firmness_range': (2.9, 3.2),
        'starch_range': (11, 13),
        'harvest_season': 3,
        'market_position': 'standard',
        'price_per_lb': 1.75
    },
    'Fuji': {
        'brix_range': (15, 18),
        'firmness_range': (3.9, 4.0),
        'starch_range': (12, 15),
        'harvest_season': 4,
        'market_position': 'standard',
        'price_per_lb': 2.00
    },
    'Granny Smith': {
        'brix_range': (10, 12),
        'firmness_range': (3.7, 4.0),
        'starch_range': (15, 17),
        'harvest_season': 4,
        'market_position': 'standard',
        'price_per_lb': 1.75
    },
    'Golden Delicious': {
        'brix_range': (13, 15),
        'firmness_range': (3.0, 3.3),
        'starch_range': (10, 12),
        'harvest_season': 4,
        'market_position': 'standard',
        'price_per_lb': 1.50
    },
    'Braeburn': {
        'brix_range': (12, 14),
        'firmness_range': (3.5, 3.8),
        'starch_range': (13, 15),
        'harvest_season': 4,
        'market_position': 'standard',
        'price_per_lb': 1.85
    },
    'Cameo': {
        'brix_range': (13, 15),
        'firmness_range': (3.4, 3.7),
        'starch_range': (11, 13),
        'harvest_season': 4,
        'market_position': 'standard',
        'price_per_lb': 2.15
    },
    'Empire': {
        'brix_range': (12, 14),
        'firmness_range': (3.2, 3.5),
        'starch_range': (10, 12),
        'harvest_season': 3,
        'market_position': 'standard',
        'price_per_lb': 1.75
    },
    
    # Often used for juice
    'Red Delicious': {
        'brix_range': (11, 13),
        'firmness_range': (2.8, 3.2),
        'starch_range': (9, 11),
        'harvest_season': 4,
        'market_position': 'juice',
        'price_per_lb': 1.25
    },
    'McIntosh': {
        'brix_range': (11, 13),
        'firmness_range': (2.5, 2.8),
        'starch_range': (9, 11),
        'harvest_season': 3,
        'market_position': 'juice',
        'price_per_lb': 1.50
    },
    'Cortland': {
        'brix_range': (11, 13),
        'firmness_range': (2.7, 3.0),
        'starch_range': (10, 12),
        'harvest_season': 3,
        'market_position': 'juice',
        'price_per_lb': 1.50
    }
}

# Pricing structure
ROUTING_PRICES = {
    'fresh_premium': 2.80,   # Average premium price
    'fresh_standard': 1.75,  # Average standard price
    'juice': 0.06           # Processing price per pound
}

def determine_routing(variety, brix, firmness, starch, size):
    """Determine routing based on variety and quality metrics."""
    variety_info = VARIETY_CHARACTERISTICS[variety]
    base_position = variety_info['market_position']
    
    # Quality thresholds based on actual metrics
    has_premium_sugar = brix >= 13.5
    has_premium_firmness = firmness >= 3.4
    has_good_size = size >= 3.5
    is_ripe = starch <= 14
    
    # Premium varieties can be downgraded
    if base_position == 'premium':
        if has_premium_sugar and has_premium_firmness and has_good_size and is_ripe:
            return 'fresh_premium'
        elif firmness >= 3.0 and size >= 2.5:
            return 'fresh_standard'
        else:
            return 'juice'
    
    # Standard varieties can be upgraded or downgraded
    elif base_position == 'standard':
        # Exceptional standard apples can be premium
        if (has_premium_sugar and has_premium_firmness and has_good_size and 
            is_ripe and variety in ['Fuji', 'Cameo', 'Braeburn']):
            return 'fresh_premium'
        elif firmness >= 2.8 and size >= 2.0:
            return 'fresh_standard'
        else:
            return 'juice'
    
    # Juice varieties can sometimes be upgraded
    else:
        if firmness >= 3.0 and has_good_size and is_ripe:
            return 'fresh_standard'
        else:
            return 'juice'

def generate_variety_samples(variety, n_samples):
    """Generate samples for a specific variety with realistic characteristics."""
    chars = VARIETY_CHARACTERISTICS[variety]
    
    # Generate base characteristics with some variation
    brix = np.random.uniform(*chars['brix_range'], n_samples)
    firmness = np.random.uniform(*chars['firmness_range'], n_samples)
    starch = np.random.uniform(*chars['starch_range'], n_samples)
    
    # Size distribution (1-5 scale) varies by variety and individual apple
    if chars['market_position'] == 'premium':
        size = np.random.normal(4.0, 0.5, n_samples)
    elif chars['market_position'] == 'standard':
        size = np.random.normal(3.5, 0.6, n_samples)
    else:
        size = np.random.normal(3.0, 0.8, n_samples)
    
    # Add quality variation (storage time, growing conditions)
    storage_days = np.random.exponential(10, n_samples)  # Days in storage
    firmness -= storage_days * 0.01  # Firmness decreases with storage
    
    # Weather/growing condition effects
    weather_effect = np.random.normal(0, 0.1, n_samples)
    brix += weather_effect * 2
    
    # Clip to realistic ranges
    brix = np.clip(brix, 10, 20)
    firmness = np.clip(firmness, 2.5, 4.0)
    starch = np.clip(starch, 8, 18)
    size = np.clip(size, 1, 5)
    
    # Determine routing for each apple
    routing = []
    for i in range(n_samples):
        routing.append(determine_routing(variety, brix[i], firmness[i], starch[i], size[i]))
    
    return pd.DataFrame({
        'variety': variety,
        'brix_numeric': brix,
        'firmness_numeric': firmness,
        'starch_numeric': starch,
        'size_numeric': size,
        'season_numeric': chars['harvest_season'],
        'routing': routing,
        'variety_base_price': chars['price_per_lb'],
        'storage_days': storage_days
    })

# Generate samples for each variety (proportional to market share)
variety_samples = {
    'Honeycrisp': 120,      # Popular premium
    'Cosmic Crisp': 80,     # Newer premium
    'Pink Lady': 60,
    'Jazz': 60,
    'Ambrosia': 70,
    'Gala': 200,           # Very popular standard
    'Fuji': 150,
    'Granny Smith': 120,
    'Golden Delicious': 100,
    'Braeburn': 80,
    'Cameo': 60,
    'Empire': 50,
    'Red Delicious': 80,    # Declining popularity
    'McIntosh': 50,
    'Cortland': 40
}

# Generate the dataset
all_data = []
for variety, n_samples in variety_samples.items():
    variety_data = generate_variety_samples(variety, n_samples)
    all_data.append(variety_data)

df = pd.concat(all_data, ignore_index=True)

# Calculate actual prices based on routing
df['price_per_lb'] = df['routing'].map(ROUTING_PRICES)
df['economic_loss'] = np.where(
    df['routing'] == 'juice',
    df['variety_base_price'] - ROUTING_PRICES['juice'],
    0
)

# Add some missing values to be realistic
missing_prob = 0.02
for col in ['starch_numeric']:
    missing_mask = np.random.random(len(df)) < missing_prob
    df.loc[missing_mask, col] = np.nan

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the dataset
output_path = 'experiments/apple_variety/realistic_synthetic_apples.csv'
df.to_csv(output_path, index=False)

print("Realistic Synthetic Apple Dataset Created!")
print(f"Total samples: {len(df)}")
print(f"\nVariety distribution:")
print(df['variety'].value_counts())
print(f"\nRouting distribution:")
print(df['routing'].value_counts())
print(f"\nAverage prices by routing:")
for routing in ['fresh_premium', 'fresh_standard', 'juice']:
    avg_price = df[df['routing'] == routing]['price_per_lb'].mean()
    print(f"  {routing}: ${avg_price:.2f}/lb")

# Calculate economic impact
juice_apples = df[df['routing'] == 'juice']
total_loss = juice_apples['economic_loss'].sum()
avg_loss = juice_apples['economic_loss'].mean()
print(f"\nEconomic impact of juice routing:")
print(f"  Total potential loss: ${total_loss:.2f}")
print(f"  Average loss per juice apple: ${avg_loss:.2f}/lb")
print(f"  Varieties most often routed to juice:")
juice_variety_counts = juice_apples['variety'].value_counts().head(5)
for variety, count in juice_variety_counts.items():
    pct = count / variety_samples[variety] * 100
    print(f"    {variety}: {count}/{variety_samples[variety]} ({pct:.1f}%)")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# 1. Routing distribution by variety
ax = axes[0]
variety_routing = pd.crosstab(df['variety'], df['routing'], normalize='index') * 100
variety_routing.plot(kind='bar', stacked=True, ax=ax, 
                    color=['#2ecc71', '#3498db', '#e67e22'])
ax.set_title('Routing Distribution by Variety (%)')
ax.set_xlabel('Variety')
ax.set_ylabel('Percentage')
ax.legend(title='Routing', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. Brix vs Firmness colored by routing
ax = axes[1]
for routing, color in zip(['fresh_premium', 'fresh_standard', 'juice'], 
                         ['#2ecc71', '#3498db', '#e67e22']):
    mask = df['routing'] == routing
    ax.scatter(df[mask]['brix_numeric'], df[mask]['firmness_numeric'], 
              c=color, label=routing, alpha=0.6, s=30)
ax.set_xlabel('Brix (sugar content)')
ax.set_ylabel('Firmness')
ax.set_title('Quality Metrics by Routing')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Economic loss by variety
ax = axes[2]
variety_loss = df.groupby('variety')['economic_loss'].mean().sort_values(ascending=False)
variety_loss.plot(kind='bar', ax=ax, color='#c0392b')
ax.set_title('Average Economic Loss by Variety')
ax.set_xlabel('Variety')
ax.set_ylabel('Loss ($/lb)')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 4. Feature distributions by routing
features = ['brix_numeric', 'firmness_numeric', 'size_numeric']
for i, feature in enumerate(features):
    ax = axes[3 + i]
    df.boxplot(column=feature, by='routing', ax=ax)
    ax.set_title(f'{feature.replace("_numeric", "").title()} by Routing')
    ax.set_xlabel('Routing')
    ax.set_ylabel(feature.replace('_numeric', '').title())

plt.tight_layout()
plt.savefig('experiments/apple_variety/realistic_synthetic_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# Test predictability
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("\n" + "="*50)
print("TESTING PREDICTABILITY")
print("="*50)

features = ['brix_numeric', 'firmness_numeric', 'starch_numeric', 'size_numeric', 'season_numeric']
X = df[features].fillna(df[features].mean())
le = LabelEncoder()
y = le.fit_transform(df['routing'])

# Add variety as a categorical feature
variety_encoded = pd.get_dummies(df['variety'], prefix='variety')
X_with_variety = pd.concat([X, variety_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_with_variety, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print(f"Random Forest accuracy (with variety): {accuracy:.3f}")

# Feature importance (top 10)
feature_names = list(X.columns) + list(variety_encoded.columns)
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\nTop 10 feature importances:")
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

print(f"\nRealistic synthetic dataset saved to: {output_path}")