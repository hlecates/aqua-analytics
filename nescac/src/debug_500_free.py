#!/usr/bin/env python3
"""
Debug script for 500 Freestyle prediction issues.
Analyzes feature importance, creates blended predictions, and provides detailed diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import PredictionEngine
from feature_modeling import train_models
from engineer_features import FeatureEngineer

class Debug500Free:
    def __init__(self):
        self.engine = PredictionEngine()
        self.feature_engineer = FeatureEngineer()
        
    def load_data(self):
        """Load all relevant data for 500 Free analysis."""
        print("Loading data...")
        
        # Load feature data
        self.winning_features = pd.read_csv('../data/processed/features/winning_features.csv')
        self.cutoff_features = pd.read_csv('../data/processed/features/cutoff_features.csv')
        
        # Filter for 500 Freestyle
        self.free_500_winning = self.winning_features[
            (self.winning_features['stroke'] == 'Freestyle') & 
            (self.winning_features['distance'] == 500)
        ].copy()
        
        self.free_500_cutoff = self.cutoff_features[
            (self.cutoff_features['stroke'] == 'Freestyle') & 
            (self.cutoff_features['distance'] == 500)
        ].copy()
        
        print(f"Found {len(self.free_500_winning)} 500 Free winning records")
        print(f"Found {len(self.free_500_cutoff)} 500 Free cutoff records")
        
    def analyze_2024_features(self):
        """Analyze the 2024 500 Free features vs historical data."""
        print("\n=== 2024 500 Free Feature Analysis ===")
        
        # Get 2024 data
        data_2024 = self.free_500_winning[self.free_500_winning['year'] == 2024]
        if len(data_2024) == 0:
            print("No 2024 data found!")
            return
            
        # Get historical data (excluding 2024)
        historical = self.free_500_winning[self.free_500_winning['year'] != 2024]
        
        # Key features to analyze
        key_features = [
            'prelim_mean', 'prelim_std', 'prelim_skewness', 'prelim_kurtosis',
            'field_size', 'seed_mean', 'seed_std', 'seed_cv',
            'fastest_seed', 'slowest_seed', 'seed_range'
        ]
        
        print("\n2024 vs Historical Feature Comparison:")
        print("-" * 60)
        
        for feature in key_features:
            if feature in data_2024.columns and feature in historical.columns:
                val_2024 = data_2024[feature].iloc[0]
                hist_mean = historical[feature].mean()
                hist_std = historical[feature].std()
                z_score = (val_2024 - hist_mean) / hist_std if hist_std > 0 else 0
                
                print(f"{feature:20s}: 2024={val_2024:8.3f} | Hist={hist_mean:8.3f} | Z-score={z_score:6.2f}")
                
                if abs(z_score) > 2:
                    print(f"  ⚠️  OUTLIER: {feature} is {z_score:.1f} standard deviations from mean")
        
        # Check for missing or extreme values
        print(f"\n2024 Actual Winning Time: {data_2024['winning_time_sec'].iloc[0]:.2f}s")
        print("2024 Predicted Time: Not available in CSV (would be calculated by model)")
        
    def create_feature_importance_plot(self):
        """Create feature importance plot for 500 Free models."""
        print("\n=== Creating Feature Importance Analysis ===")
        
        # Prepare data for feature importance
        historical = self.free_500_winning[self.free_500_winning['year'] != 2024].copy()
        
        # Select features for modeling
        feature_cols = [
            'prelim_mean', 'prelim_std', 'prelim_skewness', 'prelim_kurtosis',
            'field_size', 'seed_mean', 'seed_median', 'seed_std', 'seed_cv',
            'fastest_seed', 'slowest_seed', 'seed_range', 'seed_skewness',
            'year', 'distance'
        ]
        
        # Filter to available features
        available_features = [col for col in feature_cols if col in historical.columns]
        
        X = historical[available_features].dropna()
        y = historical.loc[X.index, 'winning_time_sec']
        
        if len(X) < 10:
            print("Not enough data for feature importance analysis")
            return
            
        # Train a Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('500 Freestyle - Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('../output/plots/500_free_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature importance plot saved to: ../output/plots/500_free_feature_importance.png")
        print("\nTop 5 Most Important Features:")
        for i, row in importance_df.tail(5).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")
            
    def create_blended_prediction(self):
        """Create a blended prediction using multiple models."""
        print("\n=== Creating Blended Prediction ===")
        
        # Get historical data
        historical = self.free_500_winning[self.free_500_winning['year'] != 2024].copy()
        
        # Prepare features
        feature_cols = [
            'prelim_mean', 'prelim_std', 'prelim_skewness', 'prelim_kurtosis',
            'field_size', 'seed_mean', 'seed_median', 'seed_std', 'seed_cv',
            'fastest_seed', 'slowest_seed', 'seed_range', 'year', 'distance'
        ]
        
        available_features = [col for col in feature_cols if col in historical.columns]
        
        X = historical[available_features].dropna()
        y = historical.loc[X.index, 'winning_time_sec']
        
        if len(X) < 10:
            print("Not enough data for blended prediction")
            return
            
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Simple Average': None  # Will use historical average
        }
        
        predictions = {}
        
        # Train and predict with ML models
        for name, model in models.items():
            if model is not None:
                model.fit(X, y)
                # Use 2024 features for prediction
                data_2024 = self.free_500_winning[self.free_500_winning['year'] == 2024]
                if len(data_2024) > 0:
                    X_2024 = data_2024[available_features].iloc[0:1]
                    pred = model.predict(X_2024)[0]
                    predictions[name] = pred
                    print(f"{name:20s}: {pred:.2f}s")
        
        # Simple average prediction
        if len(predictions) > 0:
            avg_pred = np.mean(list(predictions.values()))
            predictions['Simple Average'] = avg_pred
            print(f"{'Simple Average':20s}: {avg_pred:.2f}s")
        
        # Get actual 2024 time
        data_2024 = self.free_500_winning[self.free_500_winning['year'] == 2024]
        actual_2024 = data_2024['winning_time_sec'].iloc[0] if len(data_2024) > 0 else None
        
        print(f"\nActual 2024 Time: {actual_2024:.2f}s")
        
        # Calculate errors
        if actual_2024:
            print("\nPrediction Errors:")
            for name, pred in predictions.items():
                error = pred - actual_2024
                error_pct = (error / actual_2024) * 100
                print(f"{name:20s}: {pred:.2f}s (Δ{error:+.2f}s, {error_pct:+.1f}%)")
        
        return predictions, actual_2024
        
    def analyze_trends(self):
        """Analyze historical trends for 500 Free."""
        print("\n=== Historical Trend Analysis ===")
        
        # Get all 500 Free data
        all_data = self.free_500_winning.sort_values('year')
        
        if len(all_data) < 5:
            print("Not enough historical data for trend analysis")
            return
            
        # Plot winning times over years
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(all_data['year'], all_data['actual_winning_time'], 'o-', label='Actual')
        plt.plot(all_data['year'], all_data['predicted_winning_time'], 's--', label='Predicted')
        plt.xlabel('Year')
        plt.ylabel('Winning Time (seconds)')
        plt.title('500 Free Winning Times Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot prediction errors
        plt.subplot(2, 2, 2)
        errors = all_data['predicted_winning_time'] - all_data['actual_winning_time']
        error_pct = (errors / all_data['actual_winning_time']) * 100
        plt.plot(all_data['year'], error_pct, 'ro-')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Year')
        plt.ylabel('Prediction Error (%)')
        plt.title('500 Free Prediction Errors Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot field size trends
        plt.subplot(2, 2, 3)
        plt.plot(all_data['year'], all_data['field_size'], 'go-')
        plt.xlabel('Year')
        plt.ylabel('Field Size')
        plt.title('500 Free Field Size Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot seed time trends
        plt.subplot(2, 2, 4)
        plt.plot(all_data['year'], all_data['seed_mean'], 'mo-')
        plt.xlabel('Year')
        plt.ylabel('Average Seed Time (seconds)')
        plt.title('500 Free Average Seed Times Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../output/plots/500_free_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Trend analysis plot saved to: ../output/plots/500_free_trends.png")
        
    def run_full_analysis(self):
        """Run complete analysis."""
        print("=== 500 Freestyle Prediction Debug Analysis ===\n")
        
        # Load data
        self.load_data()
        
        # Run analyses
        self.analyze_2024_features()
        self.create_feature_importance_plot()
        self.create_blended_prediction()
        self.analyze_trends()
        
        print("\n=== Analysis Complete ===")
        print("Check the output plots for detailed visualizations.")
        print("Feature importance: ../output/plots/500_free_feature_importance.png")
        print("Trend analysis: ../output/plots/500_free_trends.png")

if __name__ == "__main__":
    debugger = Debug500Free()
    debugger.run_full_analysis() 