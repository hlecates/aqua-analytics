import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFECV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy import stats
from itertools import combinations


class OutlierDetector:
    def __init__(self, method='iqr', threshold=2.5):
        self.method = method
        self.threshold = threshold
        self.outlier_bounds = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task: str):
        if self.method == 'iqr':
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif self.method == 'zscore':
            mean = y.mean()
            std = y.std()
            lower_bound = mean - self.threshold * std
            upper_bound = mean + self.threshold * std
        else:
            # Use percentile method
            lower_bound = y.quantile(0.01)
            upper_bound = y.quantile(0.99)
        
        self.outlier_bounds[task] = (lower_bound, upper_bound)
        return self
    
    def remove_outliers(self, X: pd.DataFrame, y: pd.Series, task: str) -> Tuple[pd.DataFrame, pd.Series]:
        if task not in self.outlier_bounds:
            return X, y
        
        lower_bound, upper_bound = self.outlier_bounds[task]
        mask = (y >= lower_bound) & (y <= upper_bound)
        
        n_outliers = (~mask).sum()
        print(f"Removed {n_outliers} outliers ({n_outliers/len(y)*100:.1f}%) from {task} data")
        
        return X[mask], y[mask]


class AdvancedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        
    def create_stroke_distance_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        
        # Stroke-distance interactions
        if 'stroke_encoded' in X.columns and 'distance' in X.columns:
            X_new['stroke_distance_interaction'] = X['stroke_encoded'] * X['distance']
            
            # Stroke-specific distance categories
            stroke_names = ['free', 'back', 'breast', 'fly', 'im']
            for i, stroke in enumerate(stroke_names):
                stroke_mask = (X['stroke_encoded'] == i).astype(int)
                X_new[f'{stroke}_distance'] = X['distance'] * stroke_mask
                X_new[f'{stroke}_distance_sq'] = (X['distance'] ** 2) * stroke_mask
                X_new[f'{stroke}_distance_log'] = np.log(X['distance'] + 1) * stroke_mask
                
                # Distance-normalized features by stroke
                stroke_data = X[X['stroke_encoded'] == i]
                if len(stroke_data) > 0:
                    stroke_mean_distance = stroke_data['distance'].mean()
                    X_new[f'{stroke}_distance_norm'] = X['distance'] / stroke_mean_distance * stroke_mask
        
        # Year-stroke interactions
        if 'year' in X.columns and 'stroke_encoded' in X.columns:
            X_new['year_stroke_interaction'] = X['year'] * X['stroke_encoded']
            
        # Final type interactions (for cutoff task)
        if 'final_type_encoded' in X.columns:
            if 'distance' in X.columns:
                X_new['final_distance_interaction'] = X['final_type_encoded'] * X['distance']
            if 'stroke_encoded' in X.columns:
                X_new['final_stroke_interaction'] = X['final_type_encoded'] * X['stroke_encoded']
            if 'year' in X.columns:
                X_new['final_year_interaction'] = X['final_type_encoded'] * X['year']
        
        return X_new
        
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2, 
                                 interaction_only: bool = True) -> pd.DataFrame:
        # Only use numerical features for polynomials
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X
        
        # Fill NaN values before polynomial features
        X_clean = X.copy()
        for col in numerical_cols:
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=False
        )
        
        X_poly = poly.fit_transform(X_clean[numerical_cols])
        poly_feature_names = poly.get_feature_names_out(numerical_cols)
        
        # Create new dataframe with polynomial features
        X_new = X_clean.copy()
        for i, name in enumerate(poly_feature_names):
            if name not in X_clean.columns:  # Don't duplicate existing features
                X_new[name] = X_poly[:, i]
        
        return X_new
    
    def create_ratio_features(self, X: pd.DataFrame, task: str) -> pd.DataFrame:
        X_new = X.copy()
        
        # Distance-based ratios
        if 'distance' in X.columns:
            # Distance relative to common distances
            common_distances = [50, 100, 200, 500, 1000, 1650]
            for dist in common_distances:
                X_new[f'distance_ratio_{dist}'] = X['distance'] / dist
                X_new[f'distance_diff_{dist}'] = abs(X['distance'] - dist)
        
        # Year-based features
        if 'year' in X.columns:
            current_year = X['year'].max()
            X_new['years_since_latest'] = current_year - X['year']
            X_new['year_progression'] = (X['year'] - X['year'].min()) / (X['year'].max() - X['year'].min())
        
        return X_new
    
    def engineer_features(self, X: pd.DataFrame, task: str) -> pd.DataFrame:
        print(f"Starting feature engineering for {task}...")
        print(f"Initial features: {X.shape[1]}")
        
        # Step 1: Stroke-distance interactions
        X = self.create_stroke_distance_interactions(X)
        print(f"After stroke-distance interactions: {X.shape[1]}")
        
        # Step 2: Ratio features
        X = self.create_ratio_features(X, task)
        print(f"After ratio features: {X.shape[1]}")
        
        # Step 3: Polynomial features (selective)
        # Only apply to most important numerical features to avoid explosion
        key_features = ['year', 'distance', 'stroke_encoded']
        if task == 'cutoff':
            key_features.append('final_type_encoded')
        
        key_features = [f for f in key_features if f in X.columns]
        if key_features:
            X_key = X[key_features].copy()
            X_poly = self.create_polynomial_features(X_key, degree=2, interaction_only=True)
            
            # Add only the new polynomial features
            for col in X_poly.columns:
                if col not in X.columns:
                    X[col] = X_poly[col]
        
        print(f"After polynomial features: {X.shape[1]}")
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        return X


class SmartEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.individual_scores = {}
        
    def fit(self, X_train, y_train, X_val, y_val, task: str):
        # Define models with optimized parameters
        base_models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, 
                max_depth=6, 
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            ),
            'ridge': Ridge(alpha=10.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=8, 
                min_samples_split=5,
                random_state=42
            )
        }
        
        # Train individual models and get validation scores
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            
            # Calculate MAPE for weight determination
            mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
            self.individual_scores[name] = mape
            self.models[name] = model
            
            print(f"{name} validation MAPE: {mape:.3f}%")
        
        # Calculate weights based on inverse MAPE (better models get higher weight)
        inverse_scores = {name: 1.0 / score for name, score in self.individual_scores.items()}
        total_inverse = sum(inverse_scores.values())
        self.weights = {name: score / total_inverse for name, score in inverse_scores.items()}
        
        print(f"\nOptimal ensemble weights:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.3f}")
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions += self.weights[name] * pred
        
        return predictions
    
    def get_feature_importance(self, feature_names):
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                weight = self.weights[name]
                
                for i, feature in enumerate(feature_names):
                    if feature not in importance_dict:
                        importance_dict[feature] = 0
                    importance_dict[feature] += weight * importances[i]
        
        return importance_dict


class FeatureSelector:
    def __init__(self):
        self.selected_features = {}
        self.feature_scores = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, task: str, 
                       max_features: int = 50) -> pd.DataFrame:
        
        # Essential features that must be preserved
        essential_features = {
            'cutoff': ['final_type_encoded', 'distance', 'stroke_encoded', 'year'],
            'winning': ['distance', 'stroke_encoded', 'year']
        }
        
        essential = [f for f in essential_features.get(task, []) if f in X.columns]
        
        # Calculate feature scores using multiple methods
        print(f"Selecting features for {task} task...")
        print(f"Starting with {X.shape[1]} features")
        
        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Method 2: Correlation with target
        corr_scores = X.corrwith(y).abs()
        corr_ranking = corr_scores.sort_values(ascending=False)
        
        # Method 3: Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_scores = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Combine rankings (average rank across methods)
        combined_scores = {}
        for feature in X.columns:
            mi_rank = list(mi_ranking.index).index(feature) + 1
            corr_rank = list(corr_ranking.index).index(feature) + 1 if feature in corr_ranking.index else len(X.columns)
            rf_rank = list(rf_scores.index).index(feature) + 1
            
            combined_scores[feature] = (mi_rank + corr_rank + rf_rank) / 3
        
        # Sort by combined rank (lower is better)
        ranked_features = sorted(combined_scores.keys(), key=lambda x: combined_scores[x])
        
        # Select top features, ensuring essential features are included
        selected = essential.copy()
        
        for feature in ranked_features:
            if feature not in selected:
                selected.append(feature)
                if len(selected) >= max_features:
                    break
        
        self.selected_features[task] = selected
        self.feature_scores[task] = {
            'mutual_info': mi_ranking,
            'correlation': corr_ranking,
            'random_forest': rf_scores,
            'combined_rank': combined_scores
        }
        
        print(f"Selected {len(selected)} features")
        print(f"Top 10 features: {selected[:10]}")
        
        return X[selected]


def main():
    print("ENHANCED ULTRA-PRECISE NESCAC Swimming Prediction Model")
    print("=" * 70)
    
    # Configuration
    cutoff_features_file = Path(__file__).parent.parent / "data" / "processed" / "features" / "cutoff_features.csv"
    winning_features_file = Path(__file__).parent.parent / "data" / "processed" / "features" / "winning_features.csv"
    models_output_dir = Path(__file__).parent.parent / "output" / "models"
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer()
    outlier_detector = OutlierDetector(method='iqr')  # Use IQR method for conservative outlier removal
    feature_selector = FeatureSelector()
    
    # Load feature data
    print(f"Loading cutoff features from: {cutoff_features_file}")
    cutoff_df = pd.read_csv(cutoff_features_file)
    print(f"Loaded {len(cutoff_df)} cutoff records")
    
    print(f"Loading winning features from: {winning_features_file}")
    winning_df = pd.read_csv(winning_features_file)
    print(f"Loaded {len(winning_df)} winning records")
    
    # Process cutoff prediction
    print("\n" + "="*80)
    print("ENHANCED ULTRA-PRECISE CUTOFF PREDICTION")
    print("="*80)
    
    # Prepare cutoff data by expanding final types
    cutoff_expanded_data = []
    for _, row in cutoff_df.iterrows():
        for final_type, cutoff_col in [('A', 'a_final_cutoff_sec'), ('B', 'b_final_cutoff_sec'), ('C', 'c_final_cutoff_sec')]:
            if pd.notna(row[cutoff_col]):
                cutoff_expanded_data.append({
                    'year': row['year'],
                    'distance': row['distance'],
                    'stroke': row['stroke'],
                    'final_type': final_type,
                    'cutoff_sec': row[cutoff_col],
                    **{col: row[col] for col in cutoff_df.columns if col.startswith(('field_size', 'seed_', 'fastest_', 'slowest_', 'avg_gap', 'hhi_'))}
                })
    
    cutoff_expanded_df = pd.DataFrame(cutoff_expanded_data)
    
    # Prepare features and target
    cutoff_feature_cols = [
        'year', 'distance', 'stroke', 'final_type',
        'field_size', 'seed_mean', 'seed_median', 'seed_std', 'seed_cv',
        'fastest_seed', 'slowest_seed', 'seed_range', 'seed_skewness', 'seed_kurtosis',
        'seed_iqr', 'seed_iqr_ratio', 'avg_gap', 'hhi_seed_times'
    ]
    
    X_cutoff = cutoff_expanded_df[cutoff_feature_cols].copy()
    y_cutoff = cutoff_expanded_df['cutoff_sec']
    
    # Encode categorical variables
    le_stroke = LabelEncoder()
    le_final_type = LabelEncoder()
    X_cutoff['stroke_encoded'] = le_stroke.fit_transform(X_cutoff['stroke'])
    X_cutoff['final_type_encoded'] = le_final_type.fit_transform(X_cutoff['final_type'])
    
    # Drop original categorical columns
    X_cutoff = X_cutoff.drop(['stroke', 'final_type'], axis=1)
    
    # Enhanced feature engineering
    print("Creating enhanced features...")
    X_cutoff = feature_engineer.engineer_features(X_cutoff, 'cutoff')
    
    # Remove outliers
    outlier_detector.fit(X_cutoff, y_cutoff, 'cutoff')
    X_cutoff, y_cutoff = outlier_detector.remove_outliers(X_cutoff, y_cutoff, 'cutoff')
    
    # Feature selection
    X_cutoff = feature_selector.select_features(X_cutoff, y_cutoff, 'cutoff', max_features=40)
    
    print(f"Final cutoff data shape: {X_cutoff.shape}")
    
    # Temporal split for validation
    train_mask = X_cutoff['year'] < 2024
    val_mask = X_cutoff['year'] >= 2024
    
    X_train_cutoff = X_cutoff[train_mask]
    X_val_cutoff = X_cutoff[val_mask]
    y_train_cutoff = y_cutoff[train_mask]
    y_val_cutoff = y_cutoff[val_mask]
    
    print(f"Training set: {len(X_train_cutoff)} samples")
    print(f"Validation set: {len(X_val_cutoff)} samples")
    
    # Scale features
    scaler_cutoff = RobustScaler()
    X_train_cutoff_scaled = scaler_cutoff.fit_transform(X_train_cutoff)
    X_val_cutoff_scaled = scaler_cutoff.transform(X_val_cutoff)
    
    # Train smart ensemble
    print("Training smart ensemble for cutoff prediction...")
    ensemble_cutoff = SmartEnsemble()
    ensemble_cutoff.fit(X_train_cutoff_scaled, y_train_cutoff, X_val_cutoff_scaled, y_val_cutoff, 'cutoff')
    
    # Evaluate ensemble
    cutoff_pred = ensemble_cutoff.predict(X_val_cutoff_scaled)
    cutoff_mape = np.mean(np.abs((y_val_cutoff - cutoff_pred) / y_val_cutoff)) * 100
    cutoff_rmse = np.sqrt(mean_squared_error(y_val_cutoff, cutoff_pred))
    cutoff_r2 = r2_score(y_val_cutoff, cutoff_pred)
    
    print(f"\nCutoff Ensemble Results:")
    print(f"  MAPE: {cutoff_mape:.3f}%")
    print(f"  RMSE: {cutoff_rmse:.3f} seconds")
    print(f"  R²: {cutoff_r2:.4f}")
    
    # Process winning time prediction
    print("\n" + "="*80)
    print("ENHANCED ULTRA-PRECISE WINNING TIME PREDICTION")
    print("="*80)
    
    # Prepare winning features
    winning_feature_cols = [
        'year', 'distance', 'stroke',
        'field_size', 'prelim_mean', 'prelim_median', 'prelim_std', 'prelim_cv',
        'prelim_skewness', 'prelim_kurtosis', 'prelim_iqr', 'prelim_iqr_ratio',
        'max_gap', 'avg_gap', 'gap_1st_2nd', 'max_gap_position', 'hhi_prelim_times',
        'pressure_index', 'dark_horse_potential', 'competitive_bandwidth', 'competitive_bandwidth_pct',
        'seed_prelim_correlation', 'avg_improvement_pct', 'improvement_std'
    ]
    
    X_winning = winning_df[winning_feature_cols].copy()
    y_winning = winning_df['winning_time_sec']
    
    # Encode categorical variables
    le_stroke_winning = LabelEncoder()
    X_winning['stroke_encoded'] = le_stroke_winning.fit_transform(X_winning['stroke'])
    X_winning = X_winning.drop(['stroke'], axis=1)
    
    # Enhanced feature engineering
    print("Creating enhanced features...")
    X_winning = feature_engineer.engineer_features(X_winning, 'winning')
    
    # Remove outliers
    outlier_detector.fit(X_winning, y_winning, 'winning')
    X_winning, y_winning = outlier_detector.remove_outliers(X_winning, y_winning, 'winning')
    
    # Feature selection
    X_winning = feature_selector.select_features(X_winning, y_winning, 'winning', max_features=40)
    
    print(f"Final winning data shape: {X_winning.shape}")
    
    # Temporal split for validation
    train_mask = X_winning['year'] < 2024
    val_mask = X_winning['year'] >= 2024
    
    X_train_winning = X_winning[train_mask]
    X_val_winning = X_winning[val_mask]
    y_train_winning = y_winning[train_mask]
    y_val_winning = y_winning[val_mask]
    
    print(f"Training set: {len(X_train_winning)} samples")
    print(f"Validation set: {len(X_val_winning)} samples")
    
    # Scale features
    scaler_winning = RobustScaler()
    X_train_winning_scaled = scaler_winning.fit_transform(X_train_winning)
    X_val_winning_scaled = scaler_winning.transform(X_val_winning)
    
    # Train smart ensemble
    print("Training smart ensemble for winning time prediction...")
    ensemble_winning = SmartEnsemble()
    ensemble_winning.fit(X_train_winning_scaled, y_train_winning, X_val_winning_scaled, y_val_winning, 'winning')
    
    # Evaluate ensemble
    winning_pred = ensemble_winning.predict(X_val_winning_scaled)
    winning_mape = np.mean(np.abs((y_val_winning - winning_pred) / y_val_winning)) * 100
    winning_rmse = np.sqrt(mean_squared_error(y_val_winning, winning_pred))
    winning_r2 = r2_score(y_val_winning, winning_pred)
    
    print(f"\nWinning Time Ensemble Results:")
    print(f"  MAPE: {winning_mape:.3f}%")
    print(f"  RMSE: {winning_rmse:.3f} seconds")
    print(f"  R²: {winning_r2:.4f}")
    
    # Feature importance analysis
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    for task, ensemble, feature_names in [
        ('cutoff', ensemble_cutoff, X_cutoff.columns),
        ('winning', ensemble_winning, X_winning.columns)
    ]:
        print(f"\nTop 10 most important features for {task} prediction:")
        importance_dict = ensemble.get_feature_importance(feature_names)
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_importance[:10]):
            print(f"  {i+1:2d}. {feature:<30} {importance:.4f}")
    
    # Final summary
    print("\n" + "="*80)
    print("ENHANCED ULTRA-PRECISE MODEL SUMMARY")
    print("="*80)
    
    print(f"\nCUTOFF prediction:")
    print(f"  Smart Ensemble MAPE: {cutoff_mape:.3f}%")
    print(f"  RMSE: {cutoff_rmse:.3f} seconds")
    print(f"  R²: {cutoff_r2:.4f}")
    cutoff_target_achieved = cutoff_mape < 1.0
    cutoff_status = "🎯 TARGET ACHIEVED" if cutoff_target_achieved else f"⚠️  {cutoff_mape - 1.0:.3f}% away from target"
    print(f"  Status: {cutoff_status}")
    
    print(f"\nWINNING TIME prediction:")
    print(f"  Smart Ensemble MAPE: {winning_mape:.3f}%")
    print(f"  RMSE: {winning_rmse:.3f} seconds")
    print(f"  R²: {winning_r2:.4f}")
    winning_target_achieved = winning_mape < 1.0
    winning_status = "🎯 TARGET ACHIEVED" if winning_target_achieved else f"⚠️  {winning_mape - 1.0:.3f}% away from target"
    print(f"  Status: {winning_status}")
    
    # Save ensemble models
    ensemble_results = {
        'cutoff': {
            'ensemble': ensemble_cutoff,
            'scaler': scaler_cutoff,
            'feature_selector': feature_selector,
            'label_encoders': {'stroke': le_stroke, 'final_type': le_final_type},
            'mape': cutoff_mape,
            'rmse': cutoff_rmse,
            'r2': cutoff_r2
        },
        'winning': {
            'ensemble': ensemble_winning,
            'scaler': scaler_winning,
            'feature_selector': feature_selector,
            'label_encoders': {'stroke': le_stroke_winning},
            'mape': winning_mape,
            'rmse': winning_rmse,
            'r2': winning_r2
        }
    }
    
    output_file = models_output_dir / 'enhanced_ultra_precise_models.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(ensemble_results, f)
    
    print(f"\nModels saved to: {output_file}")
    
    # Example predictions with real value comparisons
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS vs REAL VALUES")
    print("="*80)
    
    # Cutoff prediction examples
    print("\nCUTOFF PREDICTION EXAMPLES:")
    print("-" * 50)
    
    # Get some test examples for cutoff
    cutoff_examples = []
    for i in range(min(10, len(X_val_cutoff))):
        actual_time = y_val_cutoff.iloc[i]
        predicted_time = cutoff_pred[i]
        
        # Get original features for context
        year = X_val_cutoff.iloc[i]['year']
        distance = X_val_cutoff.iloc[i]['distance']
        
        # Decode stroke and final type
        stroke_encoded = X_val_cutoff.iloc[i]['stroke_encoded']
        final_type_encoded = X_val_cutoff.iloc[i]['final_type_encoded']
        
        stroke_names = ['Backstroke', 'Breaststroke', 'Butterfly', 'Freestyle', 'IM']
        final_types = ['A', 'B', 'C']
        
        stroke = stroke_names[int(stroke_encoded)] if int(stroke_encoded) < len(stroke_names) else f"Stroke_{int(stroke_encoded)}"
        final_type = final_types[int(final_type_encoded)] if int(final_type_encoded) < len(final_types) else f"Final_{int(final_type_encoded)}"
        
        error_pct = abs(predicted_time - actual_time) / actual_time * 100
        
        cutoff_examples.append({
            'year': int(year),
            'stroke': stroke,
            'distance': int(distance),
            'final_type': final_type,
            'actual': actual_time,
            'predicted': predicted_time,
            'error_pct': error_pct
        })
    
    for i, example in enumerate(cutoff_examples[:5]):
        print(f"Example {i+1}: {example['year']} {example['stroke']} {example['distance']}m {example['final_type']}-Final")
        print(f"  Actual cutoff:    {example['actual']:.2f} seconds")
        print(f"  Predicted cutoff: {example['predicted']:.2f} seconds")
        print(f"  Error: {example['error_pct']:.3f}%")
        print()
    
    # Winning time prediction examples
    print("\nWINNING TIME PREDICTION EXAMPLES:")
    print("-" * 50)
    
    # Get some test examples for winning times
    winning_examples = []
    for i in range(min(10, len(X_val_winning))):
        actual_time = y_val_winning.iloc[i]
        predicted_time = winning_pred[i]
        
        # Get original features for context
        year = X_val_winning.iloc[i]['year']
        distance = X_val_winning.iloc[i]['distance']
        stroke_encoded = X_val_winning.iloc[i]['stroke_encoded']
        
        stroke_names = ['Backstroke', 'Breaststroke', 'Butterfly', 'Freestyle', 'IM']
        stroke = stroke_names[int(stroke_encoded)] if int(stroke_encoded) < len(stroke_names) else f"Stroke_{int(stroke_encoded)}"
        
        error_pct = abs(predicted_time - actual_time) / actual_time * 100
        
        winning_examples.append({
            'year': int(year),
            'stroke': stroke,
            'distance': int(distance),
            'actual': actual_time,
            'predicted': predicted_time,
            'error_pct': error_pct
        })
    
    for i, example in enumerate(winning_examples[:5]):
        print(f"Example {i+1}: {example['year']} {example['stroke']} {example['distance']}m")
        print(f"  Actual winning time:    {example['actual']:.2f} seconds")
        print(f"  Predicted winning time: {example['predicted']:.2f} seconds")
        print(f"  Error: {example['error_pct']:.3f}%")
        print()
    
    # Summary statistics for examples
    print("PREDICTION ACCURACY SUMMARY:")
    print("-" * 40)
    
    cutoff_errors = [ex['error_pct'] for ex in cutoff_examples]
    winning_errors = [ex['error_pct'] for ex in winning_examples]
    
    print(f"Cutoff predictions (n={len(cutoff_errors)}):")
    print(f"  Mean error: {np.mean(cutoff_errors):.3f}%")
    print(f"  Max error:  {np.max(cutoff_errors):.3f}%")
    print(f"  Min error:  {np.min(cutoff_errors):.3f}%")
    
    print(f"\nWinning time predictions (n={len(winning_errors)}):")
    print(f"  Mean error: {np.mean(winning_errors):.3f}%")
    print(f"  Max error:  {np.max(winning_errors):.3f}%")
    print(f"  Min error:  {np.min(winning_errors):.3f}%")
    
    print("\n" + "="*80)
    print("Enhanced ultra-precise modeling completed!")
    print("="*80)


if __name__ == "__main__":
    main() 