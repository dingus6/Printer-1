import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_preprocessor import FinancialDataPreprocessor

# Create directories for analysis
os.makedirs('data_analysis', exist_ok=True)
os.makedirs('data_analysis/liquidations', exist_ok=True)

# Initialise the preprocessor with the new liquidations data
preprocessor = FinancialDataPreprocessor(
    hourly_data_path='/root/hlmmnn/hourly_data.csv',
    fear_greed_data_path='/root/hlmmnn/fear_greed_data/fear_greed_index_enhanced.csv',
    liquidations_data_path='/root/hlmmnn/liquidations_data.csv'  # Path to your new CSV
)

# Get the merged data with all datasets
merged_data = preprocessor._merge_all_data()

# Analyse the "IsUp" distribution
is_up_count = merged_data['IsUp'].value_counts()
print("\nDirection distribution in full dataset:")
print(is_up_count)
print(f"Percentage of 'Up' (1): {100 * is_up_count.get(1, 0) / len(merged_data):.2f}%")
print(f"Percentage of 'Down' (0): {100 * is_up_count.get(0, 0) / len(merged_data):.2f}%")

# Calculate class weights
total = len(merged_data)
pos_weight = total / (2 * is_up_count.get(1, 0.5 * total))
neg_weight = total / (2 * is_up_count.get(0, 0.5 * total))
print(f"\nRecommended class weights:")
print(f"Positive class weight: {pos_weight:.4f}")
print(f"Negative class weight: {neg_weight:.4f}")

# Plot the direction distribution
plt.figure(figsize=(10, 6))
merged_data['IsUp'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Direction Distribution (0=Down, 1=Up)')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.savefig('data_analysis/direction_distribution.png')

# Analyse price changes
merged_data['PriceChangePercent'] = (merged_data['Close'] - merged_data['Open']) / merged_data['Open']
price_changes = merged_data['PriceChangePercent']
print(f"\nPrice change statistics:")
print(f"Mean: {price_changes.mean():.6f}")
print(f"Median: {price_changes.median():.6f}")
print(f"Min: {price_changes.min():.6f}")
print(f"Max: {price_changes.max():.6f}")

# Plot price change distribution
plt.figure(figsize=(12, 6))
plt.hist(price_changes, bins=50, alpha=0.75)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Price Change Percentage Distribution')
plt.xlabel('Price Change %')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('data_analysis/price_change_distribution.png')

# Analyse the relationship between window size and price movements
window_sizes = [4, 8, 12, 24]
price_direction_stats = []
for window in window_sizes:
    # Create sequences
    data_dict = preprocessor.prepare_data(window_size=window, horizon=1, train_ratio=0.8)
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    
    # Get direction labels
    directions = y_train[:, 0]  # First column is direction
    
    # Calculate stats
    up_pct = np.mean(directions) * 100
    price_direction_stats.append({
        'Window Size': window, 
        'Up_pct': up_pct, 
        'Down_pct': 100 - up_pct,
        'Count': len(directions)
    })

# Print window stats
print("\nDirection distribution by window size:")
for stat in price_direction_stats:
    print(f"Window Size {stat['Window Size']}: {stat['Up_pct']:.2f}% Up, {stat['Down_pct']:.2f}% Down (n={stat['Count']})")

# ============ LIQUIDATION DATA ANALYSIS ============

# Check if liquidation data is available
if preprocessor.liquidations_data is not None:
    liq_data = preprocessor.liquidations_data
    
    print("\n===== LIQUIDATIONS DATA ANALYSIS =====")
    print(f"Liquidations data shape: {liq_data.shape}")
    print(f"Time range: {liq_data.index.min()} to {liq_data.index.max()}")
    
    # Basic statistics for liquidations
    print("\nBasic statistics for key liquidations metrics:")
    stats_cols = ['short_liq', 'long_liq', 'total_liquidations', 
                  'liquidation_intensity', 'open_interest', 'ratio',
                  'oi_momentum_4h', 'oi_momentum_8h', 'oi_momentum_16h']
    
    print(liq_data[stats_cols].describe())
    
    # Analyse correlation between liquidations and price movement
    print("\nCorrelation between liquidations and price movement:")
    correlation_data = merged_data[['IsUp', 'PriceChangePercent', 'Returns', 'Volatility']].copy()
    
    # Add liquidation features
    for col in ['short_liq', 'long_liq', 'liquidation_intensity', 
                'oi_momentum_4h', 'oi_momentum_8h', 'oi_momentum_16h',
                'ratio', 'price_oi_divergence']:
        if col in merged_data.columns:
            correlation_data[col] = merged_data[col]
    
    print(correlation_data.corr()[['IsUp', 'PriceChangePercent', 'Returns', 'Volatility']].round(3))

# Print completion message
print("\nData analysis complete. Check the data_analysis directory for plots.")

#add new data sources
