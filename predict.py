import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime, timedelta

from data_preprocessor import FinancialDataPreprocessor
from transformer_model import FinancialTransformerModel

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("financial_model")

def load_model(model_path, device):
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model on.
        
    Returns:
        FinancialTransformerModel: Loaded model.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparams']
    
    # Check if model was trained with liquidations data
    with_liquidations = hyperparams.get('with_liquidations', False)
    if with_liquidations:
        logger.info("Model was trained with liquidations data")
    else:
        logger.info("Model was trained without liquidations data")
    
    # Initialize model with same hyperparameters
    model = FinancialTransformerModel(
        input_dim=hyperparams['input_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        transformer_layers=hyperparams['transformer_layers'],
        num_heads=hyperparams['num_heads'],
        dropout=hyperparams['dropout'],
        direction_threshold=hyperparams['direction_threshold']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, with_liquidations

def make_predictions(
    model, 
    hourly_data_path, 
    fear_greed_data_path,
    liquidations_data_path=None,
    window_size=6,
    start_date=None,
    end_date=None,
    direction_threshold=None,
    calibrate_threshold=False,
    device='cuda',
    future_prediction=False,
    with_liquidations=False
):
    """
    Make predictions using a trained model.
    
    Args:
        model (FinancialTransformerModel): Trained model.
        hourly_data_path (str): Path to the hourly financial data CSV.
        fear_greed_data_path (str): Path to the fear and greed index enhanced CSV.
        liquidations_data_path=args.liquidations_data,
        window_size=args.window_size,
        start_date=args.start_date,
        end_date=args.end_date,
        direction_threshold=args.direction_threshold,
        calibrate_threshold=args.calibrate_threshold,
        device=args.device,
        future_prediction=args.future,
        with_liquidations=with_liquidations
    )
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_info = f"w{args.window_size}"
    liq_info = "_liq" if with_liquidations else ""
    
    # Generate output filenames
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_filename = f"predictions_{window_info}{liq_info}_{timestamp}"
    
    # Save predictions
    if args.output_format == "json" or args.output_format == "both":
        save_predictions(
            predictions=predictions,
            output_file=os.path.join(args.output_dir, f"{base_filename}.json"),
            output_format="json"
        )
    
    if args.output_format == "csv" or args.output_format == "both":
        save_predictions(
            predictions=predictions,
            output_file=os.path.join(args.output_dir, f"{base_filename}.csv"),
            output_format="csv"
        )
    
    # Only evaluate and visualize if not doing future prediction
    if not args.future:
        # Evaluate predictions
        metrics = evaluate_predictions(predictions)
        
        if metrics:
            logger.info("Prediction Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Save metrics
            with open(os.path.join(args.output_dir, f"metrics_{window_info}{liq_info}_{timestamp}.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Generate visualizations
        if args.visualize:
            visualize_predictions(
                predictions=predictions,
                save_path=os.path.join(args.output_dir, f"visualizations_{window_info}{liq_info}_{timestamp}.png")
            )
        
        # Analyze confidence vs accuracy if requested
        if args.analyze_confidence:
            analyze_confidence_vs_accuracy(
                predictions=predictions,
                save_path=os.path.join(args.output_dir, f"confidence_analysis_{window_info}{liq_info}_{timestamp}.png")
            )
        
        # Analyze liquidations impact if requested
        if args.analyze_liquidations and with_liquidations:
            analyze_liquidations_impact(
                predictions=predictions,
                save_path=os.path.join(args.output_dir, f"liquidations_impact_{window_info}{liq_info}_{timestamp}.png")
            )
    else:
        logger.info("\nPrediction for next period:")
        logger.info(f"Timestamp: {predictions.index[0]}")
        logger.info(f"Direction probability: {predictions['direction_prob'].iloc[0]:.4f}")
        logger.info(f"Predicted direction: {'UP' if predictions['direction'].iloc[0] == 1 else 'DOWN'}")
        logger.info(f"Confidence: {predictions['confidence'].iloc[0]:.4f}")
        logger.info(f"Expected volatility: {predictions['volatility'].iloc[0]:.4f}")
        logger.info(f"Expected price change: {predictions['price_change'].iloc[0]:.4f}")
        logger.info(f"Expected spread: {predictions['spread'].iloc[0]:.4f}")path (str, optional): Path to the liquidations data CSV.
        window_size (int): Number of hours of historical data to use (default: 6).
        start_date (str): Start date for predictions (format: YYYY-MM-DD).
        end_date (str): End date for predictions (format: YYYY-MM-DD).
        direction_threshold (float): Threshold for direction prediction. If None, use model's default.
        calibrate_threshold (bool): Whether to calibrate the direction threshold.
        device (str): Device to use for predictions ('cuda' or 'cpu').
        future_prediction (bool): Whether to make a prediction for the next period instead of historical evaluation.
        with_liquidations (bool): Whether the model was trained with liquidations data.
        
    Returns:
        dict: Dictionary containing predictions.
    """
    # Verify that liquidations data is consistent with the model's training
    if with_liquidations and liquidations_data_path is None:
        logger.warning("Model was trained with liquidations data, but no liquidations data provided for prediction.")
        logger.warning("This may lead to inconsistent results. Please provide liquidations data.")
    
    if not with_liquidations and liquidations_data_path is not None:
        logger.warning("Model was trained without liquidations data, but liquidations data was provided.")
        logger.warning("The liquidations data will be ignored for predictions.")
        liquidations_data_path = None
    
    # Initialize data preprocessor
    logger.info("Loading and preprocessing data...")
    preprocessor = FinancialDataPreprocessor(
        hourly_data_path, 
        fear_greed_data_path,
        liquidations_data_path
    )
    
    # Merge all data
    merged_data = preprocessor._merge_all_data()
    
    # Filter data by date range if specified and not doing future prediction
    if not future_prediction:
        if start_date:
            start_datetime = pd.to_datetime(start_date)
            merged_data = merged_data[merged_data.index >= start_datetime]
        
        if end_date:
            end_datetime = pd.to_datetime(end_date)
            merged_data = merged_data[merged_data.index <= end_datetime]
    
    # Create features
    feature_cols = [col for col in merged_data.columns if col not in ['IsUp', 'Returns', 'PriceChangePercent', 'Volatility', 'SpreadEstimate']]
    
    # Normalize the features
    normalized_features = preprocessor.hourly_scaler.fit_transform(merged_data[feature_cols].values)
    normalized_data = pd.DataFrame(normalized_features, index=merged_data.index, columns=feature_cols)
    
    # Log feature count
    logger.info(f"Using {len(feature_cols)} features for prediction")
    
    if future_prediction:
        # For future prediction, use the most recent window of real data
        if len(normalized_data) < window_size:
            raise ValueError(f"Not enough data for prediction. Need at least {window_size} data points.")
        
        # Use the last window_size points for prediction
        last_window = normalized_data.iloc[-window_size:][feature_cols].values
        X = np.array([last_window])
        timestamps = [normalized_data.index[-1] + pd.Timedelta(hours=1)]  # Next hour
        
        logger.info(f"Using real market data window from {normalized_data.index[-window_size]} to {normalized_data.index[-1]}")
        logger.info(f"Making prediction for {timestamps[0]}")
    else:
        # Create sequences for historical prediction
        X = []
        timestamps = []
        for i in range(len(normalized_data) - window_size):
            X.append(normalized_data.iloc[i:i+window_size][feature_cols].values)
            timestamps.append(normalized_data.index[i+window_size])
        X = np.array(X)
    
    # Check if model's input dimension matches data dimension
    input_dim = model.input_dim
    feature_count = X.shape[2]
    
    if input_dim != feature_count:
        logger.error(f"Model input dimension ({input_dim}) doesn't match data feature count ({feature_count}).")
        logger.error("This might be because the model was trained with a different feature set.")
        logger.error("Make sure you're using the correct combination of model and data sources.")
        raise ValueError(f"Feature dimension mismatch: model expects {input_dim}, got {feature_count}")
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Create batches for prediction
    batch_size = 64 if not future_prediction else 1
    predictions = []
    
    logger.info("Making predictions...")
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            batch_preds = model(batch)
            
            # Append predictions to list
            for j in range(len(batch)):
                idx = i + j
                if idx < len(timestamps):
                    pred_dict = {
                        'timestamp': timestamps[idx],
                        'direction_prob': batch_preds['direction_prob'][j].item(),
                        'volatility': batch_preds['volatility'][j].item(),
                        'price_change': batch_preds['price_change'][j].item(),
                        'spread': batch_preds['spread'][j].item()
                    }
                    predictions.append(pred_dict)
                    
                    if future_prediction:
                        logger.info(f"Generated prediction for {timestamps[idx]}:")
                        logger.info(f"  Price Change: {pred_dict['price_change']:.4f}")
                        logger.info(f"  Volatility: {pred_dict['volatility']:.4f}")
                        logger.info(f"  Direction Prob: {pred_dict['direction_prob']:.4f}")
    
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
    pred_df.set_index('timestamp', inplace=True)
    
    # Calibrate direction threshold if requested and not doing future prediction
    if calibrate_threshold and not future_prediction:
        logger.info("Calibrating direction threshold...")
        # Use validation data to find optimal threshold
        direction_probs = pred_df['direction_prob'].values
        
        # Try different thresholds
        thresholds = np.linspace(0.3, 0.7, 41)
        
        best_threshold = 0.5
        best_balance = float('inf')
        
        for threshold in thresholds:
            pred_positive = (direction_probs >= threshold).mean()
            pred_negative = 1 - pred_positive
            
            # Calculate imbalance (0 is perfect balance)
            balance_diff = abs(pred_positive - 0.5)
            
            if balance_diff < best_balance:
                best_balance = balance_diff
                best_threshold = threshold
        
        logger.info(f"Calibrated direction threshold: {best_threshold:.4f}")
        direction_threshold = best_threshold
    
    # Use provided or calibrated threshold
    if direction_threshold is not None:
        model.direction_threshold = direction_threshold
    
    # Make final predictions
    pred_df['direction'] = (pred_df['direction_prob'] >= model.direction_threshold).astype(int)
    pred_df['confidence'] = (abs(pred_df['direction_prob'] - 0.5) * 2).values
    
    if not future_prediction:
        # Join with actual values if available and not doing future prediction
        pred_df = pd.merge(
            pred_df,
            merged_data[['IsUp', 'Volatility', 'PriceChangePercent', 'SpreadEstimate', 'Open', 'High', 'Low', 'Close']],
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Rename columns
        pred_df.rename(columns={
            'IsUp': 'actual_direction',
            'Volatility': 'actual_volatility',
            'PriceChangePercent': 'actual_price_change',
            'SpreadEstimate': 'actual_spread'
        }, inplace=True)
    
    return pred_df

def evaluate_predictions(predictions):
    """
    Evaluate prediction performance.
    
    Args:
        predictions (pd.DataFrame): DataFrame containing predictions and actual values.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Check if actual values are available
    if 'actual_direction' not in predictions.columns:
        logger.warning("No actual values available for evaluation")
        return None
    
    # Direction prediction metrics
    direction_preds = predictions['direction'].values
    direction_targets = predictions['actual_direction'].values
    
    # Only evaluate samples with valid targets
    valid_idx = ~np.isnan(direction_targets)
    if np.sum(valid_idx) == 0:
        logger.warning("No valid targets for evaluation")
        return None
    
    direction_preds = direction_preds[valid_idx]
    direction_targets = direction_targets[valid_idx]
    
    # Calculate confusion matrix
    cm = confusion_matrix(direction_targets, direction_preds)
    
    # Calculate metrics
    tp = cm[1][1]
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate class distribution
    positive_pct = np.mean(direction_preds) * 100
    
    # Regression metrics
    volatility_mae = np.mean(np.abs(predictions['volatility'] - predictions['actual_volatility']))
    price_change_mae = np.mean(np.abs(predictions['price_change'] - predictions['actual_price_change']))
    spread_mae = np.mean(np.abs(predictions['spread'] - predictions['actual_spread']))
    
    metrics = {
        'direction_accuracy': accuracy,
        'direction_precision': precision,
        'direction_recall': recall,
        'direction_f1': f1,
        'positive_prediction_pct': positive_pct,
        'volatility_mae': volatility_mae,
        'price_change_mae': price_change_mae,
        'spread_mae': spread_mae
    }
    
    return metrics

def visualize_predictions(predictions, save_path=None):
    """
    Visualize prediction performance.
    
    Args:
        predictions (pd.DataFrame): DataFrame containing predictions and actual values.
        save_path (str): Path to save visualizations.
    """
    # Check if actual values are available
    if 'actual_direction' not in predictions.columns:
        logger.warning("No actual values available for visualization")
        return
    
    # Only use samples with valid targets
    valid_predictions = predictions.dropna(subset=['actual_direction'])
    
    if len(valid_predictions) == 0:
        logger.warning("No valid targets for visualization")
        return
    
    # Plot direction predictions over time
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(valid_predictions.index, valid_predictions['direction_prob'], label='Direction Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.scatter(
        valid_predictions[valid_predictions['direction'] == valid_predictions['actual_direction']].index,
        valid_predictions[valid_predictions['direction'] == valid_predictions['actual_direction']]['direction_prob'],
        color='green', marker='o', alpha=0.5, label='Correct'
    )
    plt.scatter(
        valid_predictions[valid_predictions['direction'] != valid_predictions['actual_direction']].index,
        valid_predictions[valid_predictions['direction'] != valid_predictions['actual_direction']]['direction_prob'],
        color='red', marker='x', alpha=0.5, label='Incorrect'
    )
    plt.title('Direction Prediction Probabilities')
    plt.xlabel('Timestamp')
    plt.ylabel('Probability')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(3, 1, 2)
    direction_preds = valid_predictions['direction'].values
    direction_targets = valid_predictions['actual_direction'].values
    cm = confusion_matrix(direction_targets, direction_preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix for Direction Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot high confidence vs low confidence prediction accuracy
    plt.subplot(3, 1, 3)
    
    # Bin confidence scores
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    bin_labels = ['Very Low', 'Low', 'Medium', 'High']
    
    valid_predictions['confidence_bin'] = pd.cut(valid_predictions['confidence'], bins=bins, labels=bin_labels)
    
    # Calculate accuracy by confidence bin
    confidence_accuracy = valid_predictions.groupby('confidence_bin').apply(
        lambda x: (x['direction'] == x['actual_direction']).mean()
    )
    
    # Count samples in each bin
    bin_counts = valid_predictions['confidence_bin'].value_counts()
    
    # Plot bar chart
    ax = confidence_accuracy.plot(kind='bar', color='skyblue')
    plt.title('Prediction Accuracy by Confidence Level')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy')
    
    # Add sample counts as text on bars
    for i, v in enumerate(confidence_accuracy):
        if not np.isnan(v):  # Skip if NaN
            count = bin_counts.get(confidence_accuracy.index[i], 0)
            plt.text(i, v + 0.02, f'n={count}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Visualizations saved to {save_path}")
    
    plt.close()

def save_predictions(predictions, output_file, output_format='json'):
    """
    Save predictions to a file.
    
    Args:
        predictions (pd.DataFrame): DataFrame containing predictions.
        output_file (str): Path to save predictions.
        output_format (str): Output format ('json' or 'csv').
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    if output_format == 'json':
        # Convert datetime index to ISO format strings
        pred_dict = predictions.reset_index().to_dict(orient='records')
        
        # Convert datetime objects to strings
        for record in pred_dict:
            if isinstance(record['timestamp'], pd.Timestamp):
                record['timestamp'] = record['timestamp'].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(pred_dict, f, indent=2)
    
    elif output_format == 'csv':
        predictions.to_csv(output_file)
    
    else:
        logger.error(f"Unsupported output format: {output_format}")
        return
    
    logger.info(f"Predictions saved to {output_file}")

def analyze_confidence_vs_accuracy(predictions, save_path=None):
    """
    Analyze the relationship between prediction confidence and accuracy.
    
    Args:
        predictions (pd.DataFrame): DataFrame containing predictions and actual values.
        save_path (str): Path to save analysis results.
    """
    if 'actual_direction' not in predictions.columns:
        logger.warning("No actual values available for confidence analysis")
        return
    
    # Only use samples with valid targets
    valid_predictions = predictions.dropna(subset=['actual_direction'])
    
    if len(valid_predictions) == 0:
        logger.warning("No valid targets for confidence analysis")
        return
    
    # Create confidence bins
    bins = np.linspace(0, 1, 11)  # 10 bins
    valid_predictions['confidence_bin'] = pd.cut(valid_predictions['confidence'], bins=bins)
    
    # Calculate accuracy by confidence bin
    accuracy_by_confidence = valid_predictions.groupby('confidence_bin').apply(
        lambda x: {
            'accuracy': (x['direction'] == x['actual_direction']).mean(),
            'count': len(x),
            'percent': len(x) / len(valid_predictions) * 100
        }
    )
    
    # Convert to DataFrame for easier analysis
    accuracy_df = pd.DataFrame([
        {
            'confidence_min': float(bin.left),
            'confidence_max': float(bin.right),
            'accuracy': details['accuracy'],
            'count': details['count'],
            'percent': details['percent']
        }
        for bin, details in accuracy_by_confidence.items()
    ])
    
    logger.info("Confidence vs. Accuracy Analysis:")
    logger.info(f"{'Confidence Range':<20} {'Accuracy':<10} {'Count':<10} {'Percent':<10}")
    for _, row in accuracy_df.iterrows():
        logger.info(f"{row['confidence_min']:.2f}-{row['confidence_max']:.2f} {row['accuracy']:.4f} {row['count']:<10d} {row['percent']:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy by confidence bin
    plt.bar(
        accuracy_df.index,
        accuracy_df['accuracy'],
        alpha=0.7,
        color='skyblue'
    )
    
    # Add count labels
    for i, row in accuracy_df.iterrows():
        plt.text(
            i,
            row['accuracy'] + 0.02,
            f"n={row['count']}",
            ha='center',
            fontsize=9
        )
    
    # Add confidence range labels
    plt.xticks(
        accuracy_df.index,
        [f"{row['confidence_min']:.1f}-{row['confidence_max']:.1f}" for _, row in accuracy_df.iterrows()],
        rotation=45,
        fontsize=9
    )
    
    plt.title('Prediction Accuracy by Confidence Range')
    plt.xlabel('Confidence Range')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confidence analysis visualization saved to {save_path}")
    
    plt.close()
    
    return accuracy_df

def analyze_liquidations_impact(predictions, save_path=None):
    """
    Analyze the impact of liquidations metrics on prediction performance.
    Only works if the model was trained with liquidations data and predictions have actual values.
    
    Args:
        predictions (pd.DataFrame): DataFrame containing predictions and actual values.
        save_path (str): Path to save analysis results.
    """
    liquidation_columns = [
        'liquidation_intensity', 'oi_momentum_4h', 'oi_momentum_8h',
        'oi_momentum_16h', 'price_oi_divergence'
    ]
    
    # Check if predictions have liquidations data and actual values
    has_liquidations = any(col in predictions.columns for col in liquidation_columns)
    has_actuals = 'actual_direction' in predictions.columns
    
    if not has_liquidations or not has_actuals:
        logger.warning("Liquidations impact analysis requires liquidations data and actual values")
        return
    
    # Only use samples with valid targets
    valid_predictions = predictions.dropna(subset=['actual_direction'])
    
    if len(valid_predictions) == 0:
        logger.warning("No valid targets for liquidations impact analysis")
        return
    
    # Check which liquidation metrics are available
    available_metrics = [col for col in liquidation_columns if col in valid_predictions.columns]
    
    if not available_metrics:
        logger.warning("No liquidation metrics available for analysis")
        return
    
    logger.info(f"Analyzing impact of liquidations metrics: {', '.join(available_metrics)}")
    
    # Calculate correlation between liquidation metrics and prediction performance
    for metric in available_metrics:
        # Create bins for the metric
        try:
            bins = pd.qcut(valid_predictions[metric], 5)
            valid_predictions[f'{metric}_bin'] = bins
            
            # Calculate accuracy by bin
            accuracy_by_bin = valid_predictions.groupby(f'{metric}_bin').apply(
                lambda x: {
                    'accuracy': (x['direction'] == x['actual_direction']).mean(),
                    'count': len(x),
                    'avg_value': x[metric].mean()
                }
            )
            
            logger.info(f"\nImpact of {metric} on prediction accuracy:")
            logger.info(f"{'Range':<30} {'Accuracy':<10} {'Count':<10} {'Avg Value':<15}")
            
            for bin, details in accuracy_by_bin.items():
                logger.info(f"{bin} {details['accuracy']:.4f} {details['count']:<10d} {details['avg_value']:.4f}")
            
            # Calculate correlation
            correct_predictions = (valid_predictions['direction'] == valid_predictions['actual_direction']).astype(int)
            correlation = np.corrcoef(valid_predictions[metric], correct_predictions)[0, 1]
            
            logger.info(f"Correlation between {metric} and prediction accuracy: {correlation:.4f}")
            
            # Plot relationship
            plt.figure(figsize=(10, 6))
            
            # Convert bin information to DataFrame for plotting
            plot_data = [
                {
                    'bin': str(bin),
                    'accuracy': details['accuracy'],
                    'count': details['count'],
                    'avg_value': details['avg_value']
                }
                for bin, details in accuracy_by_bin.items()
            ]
            plot_df = pd.DataFrame(plot_data)
            
            # Sort by average value
            plot_df = plot_df.sort_values('avg_value')
            
            # Plot bar chart
            plt.bar(
                plot_df.index,
                plot_df['accuracy'],
                alpha=0.7,
                color='skyblue'
            )
            
            # Add count labels
            for i, row in plot_df.iterrows():
                plt.text(
                    i,
                    row['accuracy'] + 0.02,
                    f"n={row['count']}",
                    ha='center',
                    fontsize=9
                )
            
            # Add bin labels
            plt.xticks(
                plot_df.index,
                [row['bin'] for _, row in plot_df.iterrows()],
                rotation=45,
                fontsize=8
            )
            
            plt.title(f'Prediction Accuracy by {metric} (Corr: {correlation:.4f})')
            plt.xlabel(f'{metric} Range')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                metric_path = save_path.replace('.png', f'_{metric}.png')
                plt.savefig(metric_path)
                logger.info(f"Liquidations impact analysis for {metric} saved to {metric_path}")
            
            plt.close()
        except Exception as e:
            logger.warning(f"Could not analyze {metric}: {str(e)}")
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Predictions with Financial Transformer Model")
    
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--hourly_data", type=str, default="/root/hlmmnn/hourly_data.csv", 
                        help="Path to hourly financial data CSV")
    parser.add_argument("--fear_greed_data", type=str, default="/root/hlmmnn/fear_greed_data/fear_greed_index_enhanced.csv", 
                        help="Path to fear and greed index enhanced CSV")
    parser.add_argument("--liquidations_data", type=str, default=None,
                       help="Path to liquidations and open interest data CSV")
    parser.add_argument("--window_size", type=int, default=6, 
                        help="Number of hours of historical data to use")
    parser.add_argument("--start_date", type=str, default=None, 
                        help="Start date for predictions (format: YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, 
                        help="End date for predictions (format: YYYY-MM-DD)")
    parser.add_argument("--direction_threshold", type=float, default=None, 
                        help="Threshold for direction prediction")
    parser.add_argument("--calibrate_threshold", action="store_true", 
                        help="Calibrate direction threshold")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "csv", "both"], 
                        help="Output format (json, csv, or both)")
    parser.add_argument("--output_dir", type=str, default="/root/hlmmnn/new_model/predictions", 
                        help="Directory to save predictions")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualizations")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="Device to use for inference")
    parser.add_argument("--future", action="store_true",
                        help="Make prediction for next period instead of historical evaluation")
    parser.add_argument("--analyze_confidence", action="store_true",
                        help="Analyze the relationship between prediction confidence and accuracy")
    parser.add_argument("--analyze_liquidations", action="store_true",
                        help="Analyze the impact of liquidations metrics on prediction performance")
    
    args = parser.parse_args()
    
    # Use CPU if CUDA is not available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU instead")
        args.device = "cpu"
    
    device = torch.device(args.device)
    
    # Load model
    model, with_liquidations = load_model(args.model_path, device)
    
    # Warning if model/data mismatch
    if with_liquidations and args.liquidations_data is None:
        logger.warning(
            "This model was trained with liquidations data, but no liquidations data was provided. "
            "This will likely result in prediction errors."
        )
    elif not with_liquidations and args.liquidations_data is not None:
        logger.warning(
            "This model was trained without liquidations data, but liquidations data was provided. "
            "The liquidations data will be ignored."
        )
    
    # Make predictions
    predictions = make_predictions(
        model=model,
        hourly_data_path=args.hourly_data,
        fear_greed_data_path=args.fear_greed_data,
        liquidations_data_
