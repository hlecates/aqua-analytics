#!/usr/bin/env python3
"""
Test script for the new EventSpecificModelManager implementation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from feature_modeling import EventSpecificModelManager, train_improved_models

def test_event_specific_model_manager():
    """Test the EventSpecificModelManager functionality."""
    print("Testing EventSpecificModelManager...")
    
    # Initialize the manager
    manager = EventSpecificModelManager()
    
    # Test event group classification
    test_cases = [
        (50, 'Freestyle', 'sprint'),
        (100, 'Backstroke', 'short_distance'),
        (200, 'IM', 'medium_distance'),
        (500, 'Freestyle', 'distance'),
        (400, 'IM', 'distance')
    ]
    
    print("\nTesting event group classification:")
    for distance, stroke, expected_group in test_cases:
        actual_group = manager._get_event_group(distance, stroke)
        status = "✅" if actual_group == expected_group else "❌"
        print(f"{status} {distance} {stroke} -> {actual_group} (expected: {expected_group})")
    
    # Test feature calculation methods
    print("\nTesting feature calculation methods:")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'seed_mean': [30.0, 31.0, 32.0],
        'seed_std': [2.0, 2.5, 3.0],
        'seed_range': [10.0, 12.0, 15.0],
        'avg_gap': [0.5, 0.6, 0.7],
        'seed_median': [30.0, 31.0, 32.0],
        'seed_iqr': [3.0, 3.5, 4.0],
        'seed_iqr_ratio': [0.1, 0.11, 0.125],
        'seed_skewness': [0.0, 0.1, -0.1],
        'seed_cv': [0.067, 0.081, 0.094]
    })
    
    # Test sprint features
    field_density = manager._calculate_field_density(sample_data)
    print(f"Sprint field density: {field_density.values}")
    
    # Test distance features
    pacing_spread = manager._calculate_pacing_spread(sample_data)
    print(f"Distance pacing spread: {pacing_spread.values}")
    
    # Test medium distance features
    balance_metrics = manager._calculate_balance_metrics(sample_data)
    print(f"Medium distance balance: {balance_metrics.values}")
    
    print("\n✅ EventSpecificModelManager tests completed!")

def test_model_training():
    """Test the model training functionality."""
    print("\nTesting model training...")
    
    # Check if feature files exist
    data_path = Path(__file__).parent / "data" / "processed" / "features"
    cutoff_features_path = data_path / "cutoff_features.csv"
    winning_features_path = data_path / "winning_features.csv"
    
    if not (cutoff_features_path.exists() and winning_features_path.exists()):
        print("❌ Feature files not found. Please run feature engineering first.")
        return
    
    print("✅ Feature files found. Testing model training...")
    
    try:
        # Train models
        results = train_improved_models(cutoff_features_path, winning_features_path)
        
        print(f"✅ Model training completed!")
        print(f"Results keys: {list(results.keys())}")
        
        # Test prediction
        print("\nTesting predictions...")
        
        # Load the trained manager
        output_path = Path(__file__).parent / "output" / "advanced_model"
        manager_file = output_path / "enhanced_ultra_precise_models.pkl"
        
        if manager_file.exists():
            import pickle
            with open(manager_file, 'rb') as f:
                manager = pickle.load(f)
            
            # Test some predictions
            test_predictions = [
                (2025, 50, 'Freestyle', 'A', 'cutoff'),
                (2025, 100, 'Backstroke', 'B', 'cutoff'),
                (2025, 200, 'IM', 'C', 'cutoff'),
                (2025, 500, 'Freestyle', 'A', 'cutoff'),
                (2025, 50, 'Freestyle', None, 'winning'),
                (2025, 100, 'Backstroke', None, 'winning'),
                (2025, 200, 'IM', None, 'winning'),
                (2025, 500, 'Freestyle', None, 'winning')
            ]
            
            print("\nPrediction test results:")
            for year, distance, stroke, final_type, task in test_predictions:
                try:
                    if task == 'cutoff':
                        prediction = manager.predict(year, distance, stroke, final_type, task)
                    else:
                        prediction = manager.predict(year, distance, stroke, None, task)
                    
                    print(f"✅ {year} {distance} {stroke} {task}: {prediction:.2f}s")
                except Exception as e:
                    print(f"❌ {year} {distance} {stroke} {task}: {e}")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")

if __name__ == "__main__":
    print("Event-Specific Model Manager Test Suite")
    print("=" * 50)
    
    test_event_specific_model_manager()
    test_model_training()
    
    print("\n🎉 All tests completed!") 