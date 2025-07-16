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
from engineer_features import ImprovedFeatureEngineer, ImprovedFeatureSelector, ImprovedOutlierHandler


class LinearEnsemble:
    """Simple, fast ensemble for sprint events."""
    
    def __init__(self):
        self.models = [
            LinearRegression(),
            Ridge(alpha=1.0),
            Lasso(alpha=0.1)
        ]
        self.weights = [0.4, 0.4, 0.2]  # Simple weighted average
        
    def fit(self, X, y):
        self.fitted_models = []
        for model in self.models:
            model.fit(X, y)
            self.fitted_models.append(model)
    
    def predict(self, X):
        predictions = []
        for model in self.fitted_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            weighted_pred += weight * pred
            
        return weighted_pred


class DistanceSpecializedModel:
    """Complex model for distance events with pacing considerations."""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class StandardEnsemble:
    """Standard ensemble for medium-distance events."""
    
    def __init__(self):
        self.model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=100, random_state=42))
        ])
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class EventSpecificModelManager:
    def __init__(self):
        self.event_groups = {
            'sprint': {
                'events': [(50, 'Freestyle'), (50, 'Backstroke'), (50, 'Breaststroke'), (50, 'Butterfly')],
                'features': ['field_density', 'seed_clustering', 'top_gap_analysis'],
                'model_type': 'linear_ensemble'
            },
            'short_distance': {
                'events': [(100, 'Freestyle'), (100, 'Backstroke'), (100, 'Breaststroke'), (100, 'Butterfly')],
                'features': ['speed_endurance_balance', 'tactical_positioning', 'moderate_spreads'],
                'model_type': 'gradient_boosting'
            },
            'medium_distance': {
                'events': [(200, 'Freestyle'), (200, 'Backstroke'), (200, 'Breaststroke'), 
                          (200, 'Butterfly'), (200, 'IM')],
                'features': ['pacing_indicators', 'endurance_markers', 'tactical_elements'],
                'model_type': 'random_forest'
            },
            'distance': {
                'events': [(500, 'Freestyle'), (400, 'IM')],
                'features': ['endurance_dominance', 'large_spreads', 'strategy_indicators'],
                'model_type': 'specialized_distance'
            }
        }
        
        self.models = {}
        self.feature_engineers = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        
    def create_event_specific_features(self, df: pd.DataFrame, event_group: str, task: str) -> pd.DataFrame:
        """Create features tailored to specific event groups."""
        features_df = df.copy()
        if task == 'winning':
            # For winning, do not add event-specific features (columns like seed_std are not present)
            return features_df
        
        if event_group == 'sprint':
            features_df['field_density'] = self._calculate_field_density(df)
            features_df['seed_clustering'] = self._calculate_seed_clustering(df)
            features_df['top_gap_ratio'] = self._calculate_top_gap_ratio(df)
            features_df['reaction_importance'] = 0.8
        elif event_group == 'distance':
            features_df['pacing_spread'] = self._calculate_pacing_spread(df)
            features_df['endurance_indicators'] = self._calculate_endurance_indicators(df)
            features_df['large_gap_handling'] = self._calculate_large_gaps(df)
            features_df['strategy_variance'] = self._calculate_strategy_variance(df)
            features_df['reaction_importance'] = 0.2
        elif event_group == 'medium_distance':
            features_df['speed_endurance_balance'] = self._calculate_balance_metrics(df)
            features_df['tactical_elements'] = self._calculate_tactical_elements(df)
            features_df['moderate_spreads'] = self._calculate_moderate_spreads(df)
            features_df['stroke_efficiency'] = self._calculate_stroke_efficiency(df)
        elif event_group == 'short_distance':
            features_df['speed_endurance_balance'] = self._calculate_balance_metrics(df)
            features_df['tactical_positioning'] = self._calculate_tactical_positioning(df)
            features_df['moderate_spreads'] = self._calculate_moderate_spreads(df)
            features_df['reaction_importance'] = 0.6
        return features_df
    
    def train_event_group_model(self, df: pd.DataFrame, event_group: str, task: str) -> Dict[str, Any]:
        """Train a model specific to an event group."""
        print(f"Training {event_group} model for {task} prediction...")
        
        # Filter data for this event group
        group_events = self.event_groups[event_group]['events']
        group_data = df[df.apply(lambda row: (row['distance'], row['stroke']) in group_events, axis=1)]
        
        if len(group_data) == 0:
            print(f"No data found for {event_group} events")
            return {}
        
        print(f"Found {len(group_data)} samples for {event_group}")
        
        # Create event-specific features (now passes task)
        features_df = self.create_event_specific_features(group_data, event_group, task)
        
        # Prepare features and target
        if task == 'cutoff':
            # Expand final types for cutoff prediction
            expanded_data = []
            for _, row in features_df.iterrows():
                for final_type, cutoff_col in [('A', 'a_final_cutoff_sec'), ('B', 'b_final_cutoff_sec'), ('C', 'c_final_cutoff_sec')]:
                    if not pd.isna(row[cutoff_col]):
                        expanded_data.append({
                            'year': row['year'],
                            'distance': row['distance'],
                            'stroke': row['stroke'],
                            'final_type': final_type,
                            'cutoff_sec': row[cutoff_col],
                            **{col: row[col] for col in features_df.columns if col not in ['a_final_cutoff_sec', 'b_final_cutoff_sec', 'c_final_cutoff_sec', 'meet', 'gender', 'event_name', 'distance_category', 'stroke_category']}
                        })
            
            features_df = pd.DataFrame(expanded_data)
            X = features_df.drop(['cutoff_sec'], axis=1)
            y = features_df['cutoff_sec']
            
        else:  # winning
            # Filter out non-numeric columns for winning prediction
            exclude_cols = ['meet', 'gender', 'event_name', 'distance_category', 'stroke_category']
            features_df = features_df.drop(columns=[col for col in exclude_cols if col in features_df.columns])
            X = features_df.drop(['winning_time_sec'], axis=1)
            y = features_df['winning_time_sec']
        
        # Encode categorical variables
        le_stroke = LabelEncoder()
        X['stroke_encoded'] = le_stroke.fit_transform(X['stroke'])
        X = X.drop(['stroke'], axis=1)
        
        if task == 'cutoff':
            le_final_type = LabelEncoder()
            X['final_type_encoded'] = le_final_type.fit_transform(X['final_type'])
            X = X.drop(['final_type'], axis=1)
            self.label_encoders[f'{event_group}_{task}'] = {
                'stroke': le_stroke,
                'final_type': le_final_type
            }
        else:
            self.label_encoders[f'{event_group}_{task}'] = {
                'stroke': le_stroke
            }
        
        # Enhanced feature engineering
        feature_engineer = ImprovedFeatureEngineer()
        # Only apply feature engineering to numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        X_engineered = feature_engineer.engineer_features(X_numeric, task)
        
        # Combine engineered features with original categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        X_categorical = X[categorical_columns]
        X = pd.concat([X_engineered, X_categorical], axis=1)
        
        # Remove outliers
        outlier_handler = ImprovedOutlierHandler(method='iqr', threshold=3.0)
        outlier_handler.fit(X, y, task)
        X, y = outlier_handler.remove_outliers(X, y, task)
        
        # Feature selection
        feature_selector = ImprovedFeatureSelector()
        X = feature_selector.select_features(X, y, task, max_features=35)
        
        # Temporal split for validation
        train_mask = X['year'] < 2024
        val_mask = X['year'] >= 2024
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Use appropriate model type
        model_type = self.event_groups[event_group]['model_type']
        
        if model_type == 'linear_ensemble':
            model = LinearEnsemble()
        elif model_type == 'specialized_distance':
            model = DistanceSpecializedModel()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        else:
            model = StandardEnsemble()
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val_scaled)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        print(f"{event_group} {task} model - MAPE: {mape:.3f}%, RMSE: {rmse:.3f}, R²: {r2:.4f}")
        
        # Store results
        self.models[f'{event_group}_{task}'] = model
        self.scalers[f'{event_group}_{task}'] = scaler
        self.feature_engineers[f'{event_group}_{task}'] = feature_engineer
        self.feature_selectors[f'{event_group}_{task}'] = feature_selector
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_engineer': feature_engineer,
            'mape': mape,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict(self, year: int, distance: int, stroke: str, final_type: str = None, task: str = 'cutoff') -> float:
        """Make prediction using appropriate event-specific model."""
        # Determine which event group this belongs to
        event_group = self._get_event_group(distance, stroke)
        model_key = f'{event_group}_{task}'
        if model_key not in self.models:
            raise ValueError(f"No trained model found for {event_group} {task}")
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        feature_engineer = self.feature_engineers[model_key]
        feature_selector = self.feature_selectors[model_key]
        label_encoders = self.label_encoders[model_key]
        
        # Get realistic historical data for this event
        features = self._get_historical_features_for_event(year, distance, stroke, final_type, task)
        pred_df = pd.DataFrame([features])
        
        # Encode categorical variables as in training
        if 'stroke' in label_encoders:
            pred_df['stroke_encoded'] = label_encoders['stroke'].transform([stroke])
            pred_df = pred_df.drop(['stroke'], axis=1)
        if task == 'cutoff' and 'final_type' in label_encoders and final_type:
            pred_df['final_type_encoded'] = label_encoders['final_type'].transform([final_type])
            pred_df = pred_df.drop(['final_type'], axis=1)
        # Apply feature engineering as in training
        pred_df = feature_engineer.engineer_features(pred_df, task)
        # Use stored feature names from the feature selector (do not re-fit or re-select)
        if hasattr(feature_selector, 'selected_features'):
            selected_features = feature_selector.selected_features
            if isinstance(selected_features, dict):
                selected_features = selected_features.get(task, list(selected_features.values())[0])
            # Ensure all required columns are present
            for col in selected_features:
                if col not in pred_df.columns:
                    pred_df[col] = 0  # Fill with 0 instead of NaN
            pred_df = pred_df[selected_features]
            pred_df = pred_df.fillna(0)  # Fill any remaining NaNs with 0
        # Scale features
        pred_scaled = scaler.transform(pred_df)
        return model.predict(pred_scaled)[0]
    
    def _get_historical_features_for_event(self, year: int, distance: int, stroke: str, final_type: str = None, task: str = 'cutoff') -> Dict:
        """Get realistic historical features for a specific event."""
        # Load historical data
        if task == 'cutoff':
            df = pd.read_csv('data/processed/features/cutoff_features.csv')
        else:
            df = pd.read_csv('data/processed/features/winning_features.csv')
        
        # Filter for this specific event (excluding the target year)
        event_data = df[(df['distance'] == distance) & 
                       (df['stroke'] == stroke) & 
                       (df['year'] != year)]
        
        if len(event_data) == 0:
            # Fallback to all data for this event type
            event_data = df[(df['distance'] == distance) & (df['stroke'] == stroke)]
        
        if len(event_data) == 0:
            # Ultimate fallback - use reasonable defaults based on event type
            return self._get_reasonable_defaults(distance, stroke, final_type, task)
        
        # Get recent historical data (last 5 years if available)
        recent_data = event_data.sort_values('year', ascending=False).head(5)
        
        # Calculate realistic features based on historical data
        features = {
            'year': year,
            'distance': distance,
            'stroke': stroke,
        }
        
        if task == 'cutoff' and final_type:
            features['final_type'] = final_type
        
        # Use median values from recent historical data for realistic features
        numeric_columns = recent_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['year', 'distance']:
                features[col] = recent_data[col].median()
        
        event_group = self._get_event_group(distance, stroke)
        # Event-specific features: use only available columns for each task
        if event_group == 'sprint':
            if task == 'cutoff':
                seed_mean = recent_data['seed_mean'].median() if 'seed_mean' in recent_data else 0.067
                seed_std = recent_data['seed_std'].median() if 'seed_std' in recent_data else 0.067
                seed_range = recent_data['seed_range'].median() if 'seed_range' in recent_data else 1.0
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.017
                features['field_density'] = seed_std / seed_mean if seed_mean else 0.067
                features['seed_clustering'] = seed_std / seed_range if seed_range else 0.2
                features['top_gap_ratio'] = avg_gap / seed_mean if seed_mean else 0.017
            else:  # winning
                prelim_mean = recent_data['prelim_mean'].median() if 'prelim_mean' in recent_data else 0.067
                prelim_std = recent_data['prelim_std'].median() if 'prelim_std' in recent_data else 0.067
                prelim_range = (recent_data['prelim_mean'].median() - recent_data['prelim_median'].median()) if 'prelim_median' in recent_data and 'prelim_mean' in recent_data else 1.0
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.017
                features['field_density'] = prelim_std / prelim_mean if prelim_mean else 0.067
                features['seed_clustering'] = prelim_std / prelim_range if prelim_range else 0.2
                features['top_gap_ratio'] = avg_gap / prelim_mean if prelim_mean else 0.017
            features['reaction_importance'] = 0.8
        elif event_group == 'distance':
            if task == 'cutoff':
                seed_iqr = recent_data['seed_iqr'].median() if 'seed_iqr' in recent_data else 0.1
                seed_median = recent_data['seed_median'].median() if 'seed_median' in recent_data else 1.0
                seed_range = recent_data['seed_range'].median() if 'seed_range' in recent_data else 0.33
                seed_mean = recent_data['seed_mean'].median() if 'seed_mean' in recent_data else 1.0
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.25
                seed_std = recent_data['seed_std'].median() if 'seed_std' in recent_data else 1.0
                seed_skewness = recent_data['seed_skewness'].median() if 'seed_skewness' in recent_data else 0.0
                features['pacing_spread'] = seed_iqr / seed_median if seed_median else 0.1
                features['endurance_indicators'] = seed_range / seed_mean if seed_mean else 0.33
                features['large_gap_handling'] = avg_gap / seed_std if seed_std else 0.25
                features['strategy_variance'] = abs(seed_skewness)
            else:  # winning
                prelim_iqr = recent_data['prelim_iqr'].median() if 'prelim_iqr' in recent_data else 0.1
                prelim_median = recent_data['prelim_median'].median() if 'prelim_median' in recent_data else 1.0
                prelim_range = (recent_data['prelim_mean'].median() - recent_data['prelim_median'].median()) if 'prelim_median' in recent_data and 'prelim_mean' in recent_data else 0.33
                prelim_mean = recent_data['prelim_mean'].median() if 'prelim_mean' in recent_data else 1.0
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.25
                prelim_std = recent_data['prelim_std'].median() if 'prelim_std' in recent_data else 1.0
                prelim_skewness = recent_data['prelim_skewness'].median() if 'prelim_skewness' in recent_data else 0.0
                features['pacing_spread'] = prelim_iqr / prelim_median if prelim_median else 0.1
                features['endurance_indicators'] = prelim_range / prelim_mean if prelim_mean else 0.33
                features['large_gap_handling'] = avg_gap / prelim_std if prelim_std else 0.25
                features['strategy_variance'] = abs(prelim_skewness)
            features['reaction_importance'] = 0.2
        elif event_group == 'medium_distance':
            if task == 'cutoff':
                seed_cv = recent_data['seed_cv'].median() if 'seed_cv' in recent_data else 0.067
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.067
                seed_std = recent_data['seed_std'].median() if 'seed_std' in recent_data else 0.067
                seed_median = recent_data['seed_median'].median() if 'seed_median' in recent_data else 1.0
                seed_iqr_ratio = recent_data['seed_iqr_ratio'].median() if 'seed_iqr_ratio' in recent_data else 0.1
                seed_mean = recent_data['seed_mean'].median() if 'seed_mean' in recent_data else 1.0
                features['speed_endurance_balance'] = seed_cv * avg_gap
                features['tactical_elements'] = seed_std / seed_median if seed_median else 0.067
                features['moderate_spreads'] = seed_iqr_ratio
                features['stroke_efficiency'] = seed_mean / seed_median if seed_median else 1.0
            else:  # winning
                prelim_cv = recent_data['prelim_cv'].median() if 'prelim_cv' in recent_data else 0.067
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.067
                prelim_std = recent_data['prelim_std'].median() if 'prelim_std' in recent_data else 0.067
                prelim_median = recent_data['prelim_median'].median() if 'prelim_median' in recent_data else 1.0
                prelim_iqr_ratio = recent_data['prelim_iqr_ratio'].median() if 'prelim_iqr_ratio' in recent_data else 0.1
                prelim_mean = recent_data['prelim_mean'].median() if 'prelim_mean' in recent_data else 1.0
                features['speed_endurance_balance'] = prelim_cv * avg_gap
                features['tactical_elements'] = prelim_std / prelim_median if prelim_median else 0.067
                features['moderate_spreads'] = prelim_iqr_ratio
                features['stroke_efficiency'] = prelim_mean / prelim_median if prelim_median else 1.0
        elif event_group == 'short_distance':
            if task == 'cutoff':
                seed_cv = recent_data['seed_cv'].median() if 'seed_cv' in recent_data else 0.067
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.067
                seed_std = recent_data['seed_std'].median() if 'seed_std' in recent_data else 0.067
                seed_range = recent_data['seed_range'].median() if 'seed_range' in recent_data else 1.0
                seed_iqr_ratio = recent_data['seed_iqr_ratio'].median() if 'seed_iqr_ratio' in recent_data else 0.1
                features['speed_endurance_balance'] = seed_cv * avg_gap
                features['tactical_positioning'] = seed_std / seed_range if seed_range else 0.2
                features['moderate_spreads'] = seed_iqr_ratio
            else:  # winning
                prelim_cv = recent_data['prelim_cv'].median() if 'prelim_cv' in recent_data else 0.067
                avg_gap = recent_data['avg_gap'].median() if 'avg_gap' in recent_data else 0.067
                prelim_std = recent_data['prelim_std'].median() if 'prelim_std' in recent_data else 0.067
                prelim_range = (recent_data['prelim_mean'].median() - recent_data['prelim_median'].median()) if 'prelim_median' in recent_data and 'prelim_mean' in recent_data else 1.0
                prelim_iqr_ratio = recent_data['prelim_iqr_ratio'].median() if 'prelim_iqr_ratio' in recent_data else 0.1
                features['speed_endurance_balance'] = prelim_cv * avg_gap
                features['tactical_positioning'] = prelim_std / prelim_range if prelim_range else 0.2
                features['moderate_spreads'] = prelim_iqr_ratio
            features['reaction_importance'] = 0.6
        return features
    
    def _get_reasonable_defaults(self, distance: int, stroke: str, final_type: str = None, task: str = 'cutoff') -> Dict:
        """Get reasonable default values based on event type when no historical data is available."""
        # Base times for different events (in seconds)
        base_times = {
            50: {'Freestyle': 22.0, 'Backstroke': 24.0, 'Breaststroke': 26.0, 'Butterfly': 23.0},
            100: {'Freestyle': 47.0, 'Backstroke': 51.0, 'Breaststroke': 58.0, 'Butterfly': 50.0},
            200: {'Freestyle': 100.0, 'Backstroke': 110.0, 'Breaststroke': 125.0, 'Butterfly': 110.0, 'IM': 105.0},
            500: {'Freestyle': 270.0},
            400: {'IM': 280.0}
        }
        
        base_time = base_times.get(distance, {}).get(stroke, 50.0)
        
        features = {
            'year': 2025,
            'distance': distance,
            'stroke': stroke,
            'field_size': 24,
            'seed_mean': base_time,
            'seed_median': base_time,
            'seed_std': base_time * 0.05,  # 5% variation
            'seed_cv': 0.05,
            'fastest_seed': base_time * 0.95,
            'slowest_seed': base_time * 1.05,
            'seed_range': base_time * 0.1,
            'seed_skewness': 0.0,
            'seed_kurtosis': 0.0,
            'seed_iqr': base_time * 0.03,
            'seed_iqr_ratio': 0.03,
            'avg_gap': base_time * 0.01,
            'hhi_seed_times': 0.1,
            'prelim_mean': base_time,
            'prelim_median': base_time,
            'prelim_std': base_time * 0.05,
            'prelim_cv': 0.05,
            'prelim_skewness': 0.0,
            'prelim_kurtosis': 0.0,
            'prelim_iqr': base_time * 0.03,
            'prelim_iqr_ratio': 0.03,
            'max_gap': base_time * 0.02,
            'gap_1st_2nd': base_time * 0.01,
            'max_gap_position': 1.0,
            'hhi_prelim_times': 0.1,
            'pressure_index': 0.1,
            'dark_horse_potential': 0.1,
            'competitive_bandwidth': 0.1,
            'competitive_bandwidth_pct': 0.1,
            'seed_prelim_correlation': 0.1,
            'avg_improvement_pct': 0.1,
            'improvement_std': 0.1
        }
        
        if task == 'cutoff' and final_type:
            features['final_type'] = final_type
        
        return features
    
    def _get_event_group(self, distance: int, stroke: str) -> str:
        """Determine which event group an event belongs to."""
        if distance == 50:
            return 'sprint'
        elif distance == 100:
            return 'short_distance'
        elif distance == 200:
            return 'medium_distance'
        elif distance >= 400:
            return 'distance'
        else:
            raise ValueError(f"Unknown distance: {distance}")
    
    def _calculate_field_density(self, df: pd.DataFrame) -> pd.Series:
        """Calculate field density metrics for sprint events."""
        return df['seed_std'] / df['seed_mean']
    
    def _calculate_seed_clustering(self, df: pd.DataFrame) -> pd.Series:
        """Calculate seed clustering for sprint events."""
        return df['seed_std'] / df['seed_range']
    
    def _calculate_top_gap_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate top gap ratio for sprint events."""
        return df['avg_gap'] / df['seed_mean']
    
    def _calculate_pacing_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate pacing spread for distance events."""
        return df['seed_iqr'] / df['seed_median']
    
    def _calculate_endurance_indicators(self, df: pd.DataFrame) -> pd.Series:
        """Calculate endurance indicators for distance events."""
        return df['seed_range'] / df['seed_mean']
    
    def _calculate_large_gaps(self, df: pd.DataFrame) -> pd.Series:
        """Calculate large gap handling for distance events."""
        return df['avg_gap'] / df['seed_std']
    
    def _calculate_strategy_variance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate strategy variance for distance events."""
        return df['seed_skewness'].abs()
    
    def _calculate_balance_metrics(self, df: pd.DataFrame) -> pd.Series:
        """Calculate speed-endurance balance for medium distance."""
        return df['seed_cv'] * df['avg_gap']
    
    def _calculate_tactical_elements(self, df: pd.DataFrame) -> pd.Series:
        """Calculate tactical elements for medium distance."""
        return df['seed_std'] / df['seed_median']
    
    def _calculate_moderate_spreads(self, df: pd.DataFrame) -> pd.Series:
        """Calculate moderate spreads for medium distance."""
        return df['seed_iqr_ratio'].astype(float)
    
    def _calculate_stroke_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate stroke efficiency for medium distance."""
        return (df['seed_mean'] / df['seed_median']).astype(float)
    
    def _calculate_tactical_positioning(self, df: pd.DataFrame) -> pd.Series:
        """Calculate tactical positioning for short distance."""
        return df['seed_std'] / df['seed_range']
    
    def _create_prediction_features(self, year: int, distance: int, stroke: str, 
                                  event_group: str, final_type: str = None, task: str = 'cutoff') -> pd.DataFrame:
        """Create features for prediction."""
        # This is a simplified version - in practice, you'd need to create
        # appropriate features based on historical data patterns
        features = {
            'year': year,
            'distance': distance,
            'stroke': stroke,
            'field_size': 24,  # Default field size
            'seed_mean': 30.0,  # Default seed mean
            'seed_median': 30.0,
            'seed_std': 2.0,
            'seed_cv': 0.067,
            'fastest_seed': 25.0,
            'slowest_seed': 35.0,
            'seed_range': 10.0,
            'seed_skewness': 0.0,
            'seed_kurtosis': 0.0,
            'seed_iqr': 3.0,
            'seed_iqr_ratio': 0.1,
            'avg_gap': 0.5,
            'hhi_seed_times': 0.1
        }
        
        if task == 'cutoff' and final_type:
            features['final_type'] = final_type
        
        # Add event-specific features
        if event_group == 'sprint':
            features['field_density'] = 0.067
            features['seed_clustering'] = 0.2
            features['top_gap_ratio'] = 0.017
            features['reaction_importance'] = 0.8
        elif event_group == 'distance':
            features['pacing_spread'] = 0.1
            features['endurance_indicators'] = 0.33
            features['large_gap_handling'] = 0.25
            features['strategy_variance'] = 0.0
            features['reaction_importance'] = 0.2
        elif event_group == 'medium_distance':
            features['speed_endurance_balance'] = 0.067
            features['tactical_elements'] = 0.067
            features['moderate_spreads'] = 0.1
            features['stroke_efficiency'] = 1.0
        elif event_group == 'short_distance':
            features['speed_endurance_balance'] = 0.067
            features['tactical_positioning'] = 0.2
            features['moderate_spreads'] = 0.1
            features['reaction_importance'] = 0.6
        
        return pd.DataFrame([features])


def train_improved_models(cutoff_features_file: Path, winning_features_file: Path) -> Dict[str, Any]:
    """Train improved models using event-specific grouping."""
    
    print("EVENT-SPECIFIC NESCAC Swimming Prediction Model")
    print("=" * 70)
    
    # Initialize the event-specific model manager
    manager = EventSpecificModelManager()
    
    # Load feature data
    print(f"Loading cutoff features from: {cutoff_features_file}")
    cutoff_df = pd.read_csv(cutoff_features_file)
    print(f"Loaded {len(cutoff_df)} cutoff records")
    
    print(f"Loading winning features from: {winning_features_file}")
    winning_df = pd.read_csv(winning_features_file)
    print(f"Loaded {len(winning_df)} winning records")
    
    results = {}
    
    # Train models for each event group
    for event_group in manager.event_groups.keys():
        print(f"\nTraining {event_group} models...")
        
        # Train cutoff models
        cutoff_results = manager.train_event_group_model(cutoff_df, event_group, 'cutoff')
        if cutoff_results:
            results[f'{event_group}_cutoff'] = cutoff_results
        
        # Train winning models
        winning_results = manager.train_event_group_model(winning_df, event_group, 'winning')
        if winning_results:
            results[f'{event_group}_winning'] = winning_results
    
    # Save the manager with all trained models
    output_path = Path(__file__).parent.parent / "output" / "advanced_model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    manager_file = output_path / "enhanced_ultra_precise_models.pkl"
    with open(manager_file, 'wb') as f:
        pickle.dump(manager, f)
    
    print(f"\n✅ All event-specific models trained and saved to {manager_file}")
    
    return manager


if __name__ == "__main__":
    # Define file paths
    cutoff_features_file = Path(__file__).parent.parent / "data" / "processed" / "features" / "cutoff_features.csv"
    winning_features_file = Path(__file__).parent.parent / "data" / "processed" / "features" / "winning_features.csv"
    
    # Run the modeling
    results = train_improved_models(cutoff_features_file, winning_features_file)
    
    print("\n" + "="*80)
    print("IMPROVED MODELING COMPLETED!")
    print("="*80)
    print(f"Cutoff MAPE: {results['cutoff']['mape']:.3f}%")
    print(f"Winning MAPE: {results['winning']['mape']:.3f}%") 