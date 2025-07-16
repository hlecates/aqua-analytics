#!/usr/bin/env python3
"""
NESCAC Swimming Prediction Script

Usage: python predict.py <year>

This script generates predictions for all NESCAC swimming events for a given year.
It uses both simple and advanced models when possible, with automatic fallbacks.

Features:
- Automatic model training if models don't exist
- Intelligent year range detection for advanced model usage
- Comparison with actual times when available
- Comprehensive output with error metrics
"""

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Local imports
from simple_modeling import Preprocessor, ModelTrainer, ModelSaver
from engineer_features import FeatureEngineer
from feature_modeling import (
    train_improved_models, 
    EventSpecificModelManager,
    LinearEnsemble,
    DistanceSpecializedModel,
    StandardEnsemble
)
from pipeline import MeetDataPipeline


class PredictionEngine:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "data"
        self.output_path = self.base_path / "output"
        
        # Model paths
        self.simple_models_path = self.output_path / "models" / "simple_modeling"
        self.advanced_models_path = self.output_path / "advanced_model" / "enhanced_ultra_precise_models.pkl"
        
        # Feature paths
        self.features_path = self.data_path / "processed" / "features"
        self.cutoff_features_path = self.features_path / "cutoff_features.csv"
        self.winning_features_path = self.features_path / "winning_features.csv"
        
        # Data path
        self.combined_data_path = self.data_path / "processed" / "clean" / "combined_individual_events.csv"
        
        # Event definitions
        self.events = [
            ("Men", 50, "Freestyle"),
            ("Men", 100, "Freestyle"),
            ("Men", 200, "Freestyle"),
            ("Men", 500, "Freestyle"),
            ("Men", 50, "Backstroke"),
            ("Men", 100, "Backstroke"),
            ("Men", 200, "Backstroke"),
            ("Men", 50, "Breaststroke"),
            ("Men", 100, "Breaststroke"),
            ("Men", 200, "Breaststroke"),
            ("Men", 50, "Butterfly"),
            ("Men", 100, "Butterfly"),
            ("Men", 200, "Butterfly"),
            ("Men", 200, "IM"),
            ("Men", 400, "IM")
        ]
        
        # Final types for cutoff predictions
        self.final_types = ['A', 'B', 'C']
        
        self.simple_models_loaded = False
        self.advanced_models_loaded = False
        self.simple_models = None
        self.advanced_models = None
        
    def check_data_availability(self) -> Dict[str, bool]:
        """Check what data and models are available."""
        return {
            'combined_data': self.combined_data_path.exists(),
            'cutoff_features': self.cutoff_features_path.exists(),
            'winning_features': self.winning_features_path.exists(),
            'simple_models': (self.simple_models_path / "nescac_models.pkl").exists(),
            'advanced_models': self.advanced_models_path.exists()
        }
    
    def get_year_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Get valid year ranges for different components."""
        ranges = {}
        
        # Combined data range
        if self.combined_data_path.exists():
            df = pd.read_csv(self.combined_data_path)
            ranges['data'] = (df['year'].min(), df['year'].max())
        
        # Feature ranges
        if self.cutoff_features_path.exists():
            df = pd.read_csv(self.cutoff_features_path)
            ranges['cutoff_features'] = (df['year'].min(), df['year'].max())
            
        if self.winning_features_path.exists():
            df = pd.read_csv(self.winning_features_path)
            ranges['winning_features'] = (df['year'].min(), df['year'].max())
            
        return ranges
    
    def ensure_features_exist(self):
        """Ensure feature files exist, create them if necessary."""
        if not (self.cutoff_features_path.exists() and self.winning_features_path.exists()):
            print("Feature files not found. Running feature engineering...")
            pipeline = MeetDataPipeline(self.data_path)
            features_path = pipeline.run_feature_engineering()
            if not features_path:
                raise RuntimeError("Failed to engineer features")
            print("✅ Features created successfully")
    
    def ensure_simple_models_exist(self):
        """Ensure simple models exist, train them if necessary."""
        simple_model_file = self.simple_models_path / "nescac_models.pkl"
        if not simple_model_file.exists():
            print("Simple models not found. Training simple models...")
            
            # Ensure we have the data
            if not self.combined_data_path.exists():
                raise RuntimeError(f"Combined data file not found: {self.combined_data_path}")
            
            # Initialize components
            preprocessor = Preprocessor()
            trainer = ModelTrainer()
            saver = ModelSaver(str(self.simple_models_path))
            
            # Load and prepare data
            df = preprocessor.load_data(str(self.combined_data_path))
            
            # Train cutoff models
            X_cutoff, y_cutoff = preprocessor.prepare_cutoff_data(df)
            cutoff_results = trainer.train_models(X_cutoff, y_cutoff, 'cutoff', temporal_split=True)
            
            # Train winning time models
            X_winning, y_winning = preprocessor.prepare_winning_time_data(df)
            winning_results = trainer.train_models(X_winning, y_winning, 'winning', temporal_split=True)
            
            # Save models
            all_results = {'cutoff': cutoff_results, 'winning': winning_results}
            saver.save_models_and_preprocessing(trainer, preprocessor, all_results)
            
            print("✅ Simple models trained successfully")
    
    def retrain_advanced_models_excluding_year(self, exclude_year: int):
        """Retrain advanced models excluding data from the specified year to prevent data leakage."""
        print(f"Retraining advanced models excluding {exclude_year} to prevent data leakage...")
        
        # Import the modeling modules
        from feature_modeling import train_improved_models
        
        # Create temporary feature files excluding the target year
        import tempfile
        import shutil
        
        # Create temporary directories for filtered data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_cutoff_path = Path(temp_dir) / "temp_cutoff_features.csv"
            temp_winning_path = Path(temp_dir) / "temp_winning_features.csv"
            
            # Filter out the target year from both feature files
            if self.cutoff_features_path.exists():
                cutoff_df = pd.read_csv(self.cutoff_features_path)
                filtered_cutoff = cutoff_df[cutoff_df['year'] != exclude_year]
                filtered_cutoff.to_csv(temp_cutoff_path, index=False)
                print(f"Filtered cutoff features: {len(filtered_cutoff)} records (excluded {exclude_year})")
            
            if self.winning_features_path.exists():
                winning_df = pd.read_csv(self.winning_features_path)
                filtered_winning = winning_df[winning_df['year'] != exclude_year]
                filtered_winning.to_csv(temp_winning_path, index=False)
                print(f"Filtered winning features: {len(filtered_winning)} records (excluded {exclude_year})")
            
            # Retrain models using filtered data
            train_improved_models(temp_cutoff_path, temp_winning_path)
        
        # Reload the newly trained models
        self.load_advanced_models()
        
        print(f"Advanced models retrained successfully, excluding {exclude_year}")
    
    def ensure_advanced_models_exist(self):
        """Ensure advanced models exist, train them if necessary."""
        if not self.advanced_models_path.exists():
            print("Advanced models not found. Training advanced models...")
            
            # Ensure features exist first
            self.ensure_features_exist()
            
            # Train advanced models
            results = train_improved_models(self.cutoff_features_path, self.winning_features_path)
            
            print("✅ Advanced models trained successfully")
    
    def load_simple_models(self):
        """Load simple models and preprocessing objects."""
        if self.simple_models_loaded:
            return
            
        self.ensure_simple_models_exist()
        
        models_file = self.simple_models_path / "nescac_models.pkl"
        preprocessing_file = self.simple_models_path / "nescac_preprocessing.pkl"
        
        with open(models_file, 'rb') as f:
            self.simple_models_data = pickle.load(f)
            
        with open(preprocessing_file, 'rb') as f:
            self.simple_preprocessing = pickle.load(f)
            
        self.simple_models_loaded = True
        print("✅ Simple models loaded")
    
    def load_advanced_models(self):
        """Load advanced models."""
        if self.advanced_models_loaded:
            return
            
        self.ensure_advanced_models_exist()
        
        with open(self.advanced_models_path, 'rb') as f:
            self.advanced_models = pickle.load(f)
            
        self.advanced_models_loaded = True
        print("✅ Advanced models loaded")
    
    def predict_simple_cutoff(self, year: int, stroke: str, distance: int, final_type: str) -> float:
        """Make cutoff prediction using simple model."""
        # Get best model for cutoff
        best_models = {}
        for model_name, model_data in self.simple_models_data['cutoff'].items():
            if hasattr(model_data, 'predict'):  # Direct model object
                best_models[model_name] = model_data
            else:  # Wrapped model object
                best_models[model_name] = model_data['model']
        
        # Find best model by getting results (we'll use gradient_boosting as default good performer)
        model_name = 'gradient_boosting' if 'gradient_boosting' in best_models else list(best_models.keys())[0]
        model = best_models[model_name]
        
        # Encode features
        stroke_encoded = self.simple_preprocessing['label_encoders']['cutoff']['stroke'].transform([stroke])[0]
        final_type_encoded = self.simple_preprocessing['label_encoders']['cutoff']['final_type'].transform([final_type])[0]
        
        # Create feature array
        features = np.array([[year, distance, stroke_encoded, final_type_encoded]])
        
        # Scale features
        features_scaled = self.simple_preprocessing['scalers']['cutoff'].transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        return prediction
    
    def predict_simple_winning(self, year: int, stroke: str, distance: int) -> float:
        """Make winning time prediction using simple model."""
        # Get best model for winning
        best_models = {}
        for model_name, model_data in self.simple_models_data['winning'].items():
            if hasattr(model_data, 'predict'):  # Direct model object
                best_models[model_name] = model_data
            else:  # Wrapped model object
                best_models[model_name] = model_data['model']
        
        # Find best model by getting results (we'll use gradient_boosting as default good performer)
        model_name = 'gradient_boosting' if 'gradient_boosting' in best_models else list(best_models.keys())[0]
        model = best_models[model_name]
        
        # Encode features
        stroke_encoded = self.simple_preprocessing['label_encoders']['winning']['stroke'].transform([stroke])[0]
        
        # Create feature array
        features = np.array([[year, distance, stroke_encoded]])
        
        # Scale features
        features_scaled = self.simple_preprocessing['scalers']['winning'].transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        return prediction
    
    def get_advanced_features_for_event(self, year: int, stroke: str, distance: int) -> Dict:
        """Get advanced features for a specific event, with outlier handling."""
        # Load feature data
        winning_features = pd.read_csv('../data/processed/features/winning_features.csv')
        cutoff_features = pd.read_csv('../data/processed/features/cutoff_features.csv')
        
        # Filter for the specific event
        winning_data = winning_features[
            (winning_features['year'] == year) & 
            (winning_features['stroke'] == stroke) & 
            (winning_features['distance'] == distance)
        ]
        
        cutoff_data = cutoff_features[
            (cutoff_features['year'] == year) & 
            (cutoff_features['stroke'] == stroke) & 
            (cutoff_features['distance'] == distance)
        ]
        
        # Apply outlier handling to the prediction data
        outlier_features = [
            'prelim_skewness', 'prelim_kurtosis', 'prelim_mean', 'prelim_std',
            'seed_skewness', 'seed_kurtosis', 'seed_mean', 'seed_std',
            'field_size', 'seed_cv', 'fastest_seed', 'slowest_seed', 'seed_range'
        ]
        
        # Get historical stats for outlier clipping
        historical_winning = winning_features[
            (winning_features['stroke'] == stroke) & 
            (winning_features['distance'] == distance) &
            (winning_features['year'] != year)  # Exclude target year
        ]
        
        historical_cutoff = cutoff_features[
            (cutoff_features['stroke'] == stroke) & 
            (cutoff_features['distance'] == distance) &
            (cutoff_features['year'] != year)  # Exclude target year
        ]
        
        # Apply outlier clipping to prediction data
        if not winning_data.empty and not historical_winning.empty:
            for feat in outlier_features:
                if feat in winning_data.columns and feat in historical_winning.columns:
                    hist_mean = historical_winning[feat].mean()
                    hist_std = historical_winning[feat].std()
                    if hist_std > 0:
                        current_val = winning_data[feat].values[0] if hasattr(winning_data[feat], 'values') else winning_data[feat].iloc[0]
                        z_score = (current_val - hist_mean) / hist_std
                        if abs(z_score) > 2.0:  # Clip outliers
                            if z_score > 2.0:
                                winning_data.loc[winning_data.index[0], feat] = hist_mean + 2.0 * hist_std
                            else:
                                winning_data.loc[winning_data.index[0], feat] = hist_mean - 2.0 * hist_std
        
        if not cutoff_data.empty and not historical_cutoff.empty:
            for feat in outlier_features:
                if feat in cutoff_data.columns and feat in historical_cutoff.columns:
                    hist_mean = historical_cutoff[feat].mean()
                    hist_std = historical_cutoff[feat].std()
                    if hist_std > 0:
                        current_val = cutoff_data[feat].values[0] if hasattr(cutoff_data[feat], 'values') else cutoff_data[feat].iloc[0]
                        z_score = (current_val - hist_mean) / hist_std
                        if abs(z_score) > 2.0:  # Clip outliers
                            if z_score > 2.0:
                                cutoff_data.loc[cutoff_data.index[0], feat] = hist_mean + 2.0 * hist_std
                            else:
                                cutoff_data.loc[cutoff_data.index[0], feat] = hist_mean - 2.0 * hist_std
        
        return {
            'winning': winning_data.iloc[0].to_dict() if not winning_data.empty else None,
            'cutoff': cutoff_data.iloc[0].to_dict() if not cutoff_data.empty else None
        }
    
    def predict_advanced_cutoff(self, year: int, stroke: str, distance: int, final_type: str) -> float:
        """Make cutoff prediction using event-specific model manager."""
        try:
            # Use the event-specific model manager
            manager = self.advanced_models  # This should be the EventSpecificModelManager
            
            # Make prediction using the manager
            prediction = manager.predict(year, distance, stroke, final_type, 'cutoff')
            return prediction
            
        except Exception as e:
            print(f"Warning: Advanced cutoff prediction failed for {year} {stroke} {distance} {final_type}: {e}")
            # Fall back to simple model
            return self.predict_simple_cutoff(year, stroke, distance, final_type)
    
    def predict_advanced_winning(self, year: int, stroke: str, distance: int) -> float:
        """Make winning time prediction using event-specific model manager."""
        try:
            # Use the event-specific model manager
            manager = self.advanced_models  # This should be the EventSpecificModelManager
            
            # Make prediction using the manager
            prediction = manager.predict(year, distance, stroke, None, 'winning')
            return prediction
            
        except Exception as e:
            print(f"Warning: Advanced winning prediction failed for {year} {stroke} {distance}: {e}")
            # Fall back to simple model
            return self.predict_simple_winning(year, stroke, distance)
    
    def get_actual_times(self, year: int) -> Dict[str, Dict]:
        """Get actual times for the specified year if available."""
        if not self.combined_data_path.exists():
            return {}
            
        df = pd.read_csv(self.combined_data_path)
        year_data = df[df['year'] == year]
        
        if year_data.empty:
            return {}
        
        actual_times = {}
        
        for _, row in year_data.iterrows():
            event_key = f"{row['gender']} {row['distance']} {row['stroke']}"
            
            actual_times[event_key] = {
                'winning_time': row['winning_time_sec'],
                'a_cutoff': row['a_final_cutoff_sec'] if pd.notna(row['a_final_cutoff_sec']) else None,
                'b_cutoff': row['b_final_cutoff_sec'] if pd.notna(row['b_final_cutoff_sec']) else None,
                'c_cutoff': row['c_final_cutoff_sec'] if pd.notna(row['c_final_cutoff_sec']) else None
            }
        
        return actual_times
    
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS.SS format."""
        if pd.isna(seconds) or seconds is None:
            return "N/A"
        
        minutes = int(seconds // 60)
        secs = seconds % 60
        
        if minutes > 0:
            return f"{minutes}:{secs:05.2f}"
        else:
            return f"{secs:.2f}"
    
    def calculate_error_metrics(self, predicted: float, actual: float) -> Tuple[float, float]:
        """Calculate error metrics between predicted and actual times."""
        if pd.isna(actual) or actual is None or pd.isna(predicted) or predicted is None:
            return None, None
        
        diff_seconds = predicted - actual
        percent_diff = (diff_seconds / actual) * 100
        
        return diff_seconds, percent_diff
    
    def generate_predictions(self, year: int, retrain_advanced: bool = False) -> Dict:
        """Generate all predictions for the specified year."""
        print(f"\nGenerating predictions for {year}...")
        
        # Check what models we can use
        year_ranges = self.get_year_ranges()
        availability = self.check_data_availability()
        
        can_use_advanced = (
            year_ranges.get('cutoff_features', (9999, 0))[0] <= year <= year_ranges.get('cutoff_features', (0, -1))[1] and
            year_ranges.get('winning_features', (9999, 0))[0] <= year <= year_ranges.get('winning_features', (0, -1))[1] and
            availability['advanced_models']
        )
        
        can_use_simple = availability['combined_data']
        
        print(f"Year {year} analysis:")
        print(f"  - Can use simple models: {can_use_simple}")
        print(f"  - Can use advanced models: {can_use_advanced}")
        
        # Load appropriate models
        if can_use_simple:
            self.load_simple_models()
        
        # For advanced models, retrain excluding the target year to prevent data leakage
        if can_use_advanced:
            if retrain_advanced:
                print(f"Retraining advanced models excluding {year} to prevent data leakage...")
                self.retrain_advanced_models_excluding_year(year)
            else:
                self.load_advanced_models()
        
        # Get actual times for comparison
        actual_times = self.get_actual_times(year)
        
        # Generate predictions
        predictions = {}
        
        for gender, distance, stroke in self.events:
            event_key = f"{gender} {distance} {stroke}"
            predictions[event_key] = {}
            
            # Winning time predictions
            if can_use_simple:
                try:
                    simple_winning = self.predict_simple_winning(year, stroke, distance)
                    predictions[event_key]['simple_winning'] = simple_winning
                except Exception as e:
                    print(f"Warning: Simple winning prediction failed for {event_key}: {e}")
                    predictions[event_key]['simple_winning'] = None
            
            if can_use_advanced:
                try:
                    advanced_winning = self.predict_advanced_winning(year, stroke, distance)
                    predictions[event_key]['advanced_winning'] = advanced_winning
                except Exception as e:
                    print(f"Warning: Advanced winning prediction failed for {event_key}: {e}")
                    predictions[event_key]['advanced_winning'] = None
            
            # Cutoff predictions
            for final_type in self.final_types:
                if can_use_simple:
                    try:
                        simple_cutoff = self.predict_simple_cutoff(year, stroke, distance, final_type)
                        predictions[event_key][f'simple_{final_type.lower()}_cutoff'] = simple_cutoff
                    except Exception as e:
                        print(f"Warning: Simple {final_type} cutoff prediction failed for {event_key}: {e}")
                        predictions[event_key][f'simple_{final_type.lower()}_cutoff'] = None
                
                if can_use_advanced:
                    try:
                        advanced_cutoff = self.predict_advanced_cutoff(year, stroke, distance, final_type)
                        predictions[event_key][f'advanced_{final_type.lower()}_cutoff'] = advanced_cutoff
                    except Exception as e:
                        print(f"Warning: Advanced {final_type} cutoff prediction failed for {event_key}: {e}")
                        predictions[event_key][f'advanced_{final_type.lower()}_cutoff'] = None
            
            # Add actual times
            if event_key in actual_times:
                predictions[event_key]['actual'] = actual_times[event_key]
            else:
                predictions[event_key]['actual'] = {
                    'winning_time': None,
                    'a_cutoff': None,
                    'b_cutoff': None,
                    'c_cutoff': None
                }
        
        return predictions
    
    def write_predictions_report(self, year: int, predictions: Dict, output_file: str):
        """Write comprehensive predictions report to file."""
        with open(output_file, 'w') as f:
            f.write(f"NESCAC Swimming Predictions Report for {year}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary information
            availability = self.check_data_availability()
            year_ranges = self.get_year_ranges()
            
            f.write("PREDICTION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Target Year: {year}\n")
            f.write(f"Data Available: {availability}\n")
            f.write(f"Year Ranges: {year_ranges}\n")
            f.write("\n")
            
            # Detailed predictions for each event
            f.write("DETAILED PREDICTIONS\n")
            f.write("-" * 40 + "\n\n")
            
            for event_key, event_data in predictions.items():
                f.write(f"{event_key}\n")
                f.write("=" * len(event_key) + "\n")
                
                # Winning time section
                f.write("Winning Time:\n")
                
                actual_winning = event_data['actual']['winning_time']
                if actual_winning:
                    f.write(f"  Actual:           {self.format_time(actual_winning)}\n")
                
                if 'simple_winning' in event_data and event_data['simple_winning']:
                    simple_winning = event_data['simple_winning']
                    f.write(f"  Simple Model:     {self.format_time(simple_winning)}")
                    if actual_winning:
                        diff_sec, diff_pct = self.calculate_error_metrics(simple_winning, actual_winning)
                        if diff_sec is not None:
                            f.write(f" (Δ{diff_sec:+.2f}s, {diff_pct:+.1f}%)")
                    f.write("\n")
                
                if 'advanced_winning' in event_data and event_data['advanced_winning']:
                    advanced_winning = event_data['advanced_winning']
                    f.write(f"  Advanced Model:   {self.format_time(advanced_winning)}")
                    if actual_winning:
                        diff_sec, diff_pct = self.calculate_error_metrics(advanced_winning, actual_winning)
                        if diff_sec is not None:
                            f.write(f" (Δ{diff_sec:+.2f}s, {diff_pct:+.1f}%)")
                    f.write("\n")
                
                f.write("\n")
                
                # Cutoff times section
                for final_type in self.final_types:
                    f.write(f"{final_type} Final Cutoff:\n")
                    
                    actual_cutoff_key = f"{final_type.lower()}_cutoff"
                    actual_cutoff = event_data['actual'][actual_cutoff_key]
                    
                    if actual_cutoff:
                        f.write(f"  Actual:           {self.format_time(actual_cutoff)}\n")
                    
                    simple_key = f"simple_{final_type.lower()}_cutoff"
                    if simple_key in event_data and event_data[simple_key]:
                        simple_cutoff = event_data[simple_key]
                        f.write(f"  Simple Model:     {self.format_time(simple_cutoff)}")
                        if actual_cutoff:
                            diff_sec, diff_pct = self.calculate_error_metrics(simple_cutoff, actual_cutoff)
                            if diff_sec is not None:
                                f.write(f" (Δ{diff_sec:+.2f}s, {diff_pct:+.1f}%)")
                        f.write("\n")
                    
                    advanced_key = f"advanced_{final_type.lower()}_cutoff"
                    if advanced_key in event_data and event_data[advanced_key]:
                        advanced_cutoff = event_data[advanced_key]
                        f.write(f"  Advanced Model:   {self.format_time(advanced_cutoff)}")
                        if actual_cutoff:
                            diff_sec, diff_pct = self.calculate_error_metrics(advanced_cutoff, actual_cutoff)
                            if diff_sec is not None:
                                f.write(f" (Δ{diff_sec:+.2f}s, {diff_pct:+.1f}%)")
                        f.write("\n")
                    
                    f.write("\n")
                
                f.write("-" * 60 + "\n\n")
            
            # Summary statistics
            f.write("PREDICTION ACCURACY SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Calculate summary statistics for each model type
            simple_winning_errors = []
            advanced_winning_errors = []
            simple_cutoff_errors = []
            advanced_cutoff_errors = []
            
            for event_key, event_data in predictions.items():
                actual = event_data['actual']
                
                # Winning time errors
                if actual['winning_time']:
                    if 'simple_winning' in event_data and event_data['simple_winning']:
                        _, pct_error = self.calculate_error_metrics(event_data['simple_winning'], actual['winning_time'])
                        if pct_error is not None:
                            simple_winning_errors.append(abs(pct_error))
                    
                    if 'advanced_winning' in event_data and event_data['advanced_winning']:
                        _, pct_error = self.calculate_error_metrics(event_data['advanced_winning'], actual['winning_time'])
                        if pct_error is not None:
                            advanced_winning_errors.append(abs(pct_error))
                
                # Cutoff errors
                for final_type in self.final_types:
                    actual_cutoff_key = f"{final_type.lower()}_cutoff"
                    if actual[actual_cutoff_key]:
                        simple_key = f"simple_{final_type.lower()}_cutoff"
                        if simple_key in event_data and event_data[simple_key]:
                            _, pct_error = self.calculate_error_metrics(event_data[simple_key], actual[actual_cutoff_key])
                            if pct_error is not None:
                                simple_cutoff_errors.append(abs(pct_error))
                        
                        advanced_key = f"advanced_{final_type.lower()}_cutoff"
                        if advanced_key in event_data and event_data[advanced_key]:
                            _, pct_error = self.calculate_error_metrics(event_data[advanced_key], actual[actual_cutoff_key])
                            if pct_error is not None:
                                advanced_cutoff_errors.append(abs(pct_error))
            
            # Write summary statistics
            if simple_winning_errors:
                f.write(f"Simple Model - Winning Times (n={len(simple_winning_errors)}):\n")
                f.write(f"  Mean Absolute Error: {np.mean(simple_winning_errors):.2f}%\n")
                f.write(f"  Median Absolute Error: {np.median(simple_winning_errors):.2f}%\n")
                f.write(f"  Max Absolute Error: {np.max(simple_winning_errors):.2f}%\n\n")
            
            if advanced_winning_errors:
                f.write(f"Advanced Model - Winning Times (n={len(advanced_winning_errors)}):\n")
                f.write(f"  Mean Absolute Error: {np.mean(advanced_winning_errors):.2f}%\n")
                f.write(f"  Median Absolute Error: {np.median(advanced_winning_errors):.2f}%\n")
                f.write(f"  Max Absolute Error: {np.max(advanced_winning_errors):.2f}%\n\n")
            
            if simple_cutoff_errors:
                f.write(f"Simple Model - Cutoff Times (n={len(simple_cutoff_errors)}):\n")
                f.write(f"  Mean Absolute Error: {np.mean(simple_cutoff_errors):.2f}%\n")
                f.write(f"  Median Absolute Error: {np.median(simple_cutoff_errors):.2f}%\n")
                f.write(f"  Max Absolute Error: {np.max(simple_cutoff_errors):.2f}%\n\n")
            
            if advanced_cutoff_errors:
                f.write(f"Advanced Model - Cutoff Times (n={len(advanced_cutoff_errors)}):\n")
                f.write(f"  Mean Absolute Error: {np.mean(advanced_cutoff_errors):.2f}%\n")
                f.write(f"  Median Absolute Error: {np.median(advanced_cutoff_errors):.2f}%\n")
                f.write(f"  Max Absolute Error: {np.max(advanced_cutoff_errors):.2f}%\n\n")
            
            f.write(f"\nReport generated successfully!\n")
            f.write(f"Total events predicted: {len(self.events)}\n")
            f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NESCAC swimming predictions for a given year",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py 2025          # Predict for 2025 (uses both models)
  python predict.py 2000          # Predict for 2000 (uses only simple model)
  python predict.py 2024          # Predict for 2024 and compare with actual
        """
    )
    
    parser.add_argument(
        'year', 
        type=int, 
        help='Year to generate predictions for'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file path (default: predictions_YEAR.txt)'
    )
    
    args = parser.parse_args()
    
    # Validate year
    if args.year < 1990 or args.year > 2030:
        print(f"Error: Year {args.year} is outside reasonable range (1990-2030)")
        return 1
    
    # Setup output file
    if args.output is None:
        # Create prediction directory in output folder
        prediction_dir = Path(__file__).parent.parent / "output" / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        output_file = prediction_dir / f"predictions_{args.year}.txt"
    else:
        output_file = args.output
    
    try:
        # Initialize prediction engine
        print("Initializing NESCAC Prediction Engine...")
        engine = PredictionEngine()
        
        # Generate predictions
        predictions = engine.generate_predictions(args.year)
        
        # Write report
        print(f"\nWriting predictions report to: {output_file}")
        engine.write_predictions_report(args.year, predictions, output_file)
        
        print(f"\n✅ Predictions completed successfully!")
        print(f"📄 Report saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 