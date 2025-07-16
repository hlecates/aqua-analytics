import pandas as pd
import numpy as np
from pathlib import Path
import ast
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class TimeConverter:
    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        if pd.isna(time_str):
            return np.nan
        
        time_str = str(time_str).strip()

        # Handle "NT" (No Time) values and empty strings
        if time_str.upper() == 'NT' or time_str == '':
            return np.nan

        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            try:
             return float(time_str)
            except ValueError:
                return np.nan
        

    @staticmethod
    def seconds_to_time(seconds: float) -> str:
        if pd.isna(seconds):
            return ''
        
        if seconds >= 60:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}:{seconds:05.2f}"
        else:
            return f"{seconds:05.2f}"
        

class SeedTimeAnalyzer:
    def __init__(self):
        self.time_converter = TimeConverter()

    
    def extract_seed_times(self, entries: List[Dict]) -> List[float]:
        seed_times = []

        for entry in entries:
            seed_time = self.time_converter.time_to_seconds(entry['seed_time'])
            if not pd.isna(seed_time):
                seed_times.append(seed_time)

        return sorted(seed_times)
    

    def calculate_cutoff_features(self, seed_times: List[float]) -> Dict[str, float]:
        features = {}

        if len(seed_times) == 0:
            return self._empty_cutoff_features()
        
        # Basic statistics
        features['field_size'] = len(seed_times)
        features['seed_mean'] = np.mean(seed_times)
        features['seed_median'] = np.median(seed_times)
        features['seed_std'] = np.std(seed_times)
        features['seed_cv'] = features['seed_std'] / features['seed_mean'] if features['seed_mean'] > 0 else 0

        # Simple range features
        features['fastest_seed'] = seed_times[0]  # First (fastest) time
        features['slowest_seed'] = seed_times[-1]  # Last (slowest) time
        features['seed_range'] = features['slowest_seed'] - features['fastest_seed']

        # Distribution features
        if len(seed_times) >= 3:
            features['seed_skewness'] = stats.skew(seed_times)
            features['seed_kurtosis'] = stats.kurtosis(seed_times)
        else:
            features['seed_skewness'] = 0
            features['seed_kurtosis'] = 0

        # Percentile features
        if len(seed_times) >= 4:
            q25, q75 = np.percentile(seed_times, [25, 75])
            features['seed_iqr'] = q75 - q25
            features['seed_iqr_ratio'] = features['seed_iqr'] / features['seed_median']
        else:
            features['seed_iqr'] = 0
            features['seed_iqr_ratio'] = 0

        # Average gap between consecutive times (holistic measure)
        if len(seed_times) >= 2:
            gaps = [seed_times[i+1] - seed_times[i] for i in range(len(seed_times)-1)]
            features['avg_gap'] = np.mean(gaps)
        else:
            features['avg_gap'] = 0

        # Herfindahl-Hirschman Index (HHI) for seed times
        # Convert times to shares by inverting and normalizing
        if len(seed_times) > 1:
            inverted_times = [1/t for t in seed_times]
            total = sum(inverted_times)
            shares = [t/total for t in inverted_times]
            features['hhi_seed_times'] = sum(s**2 for s in shares)
        else:
            features['hhi_seed_times'] = 1.0
        
        return features
    

    def _empty_cutoff_features(self) -> Dict[str, float]:
        return {
            'field_size': 0,
            'seed_mean': np.nan,
            'seed_median': np.nan,
            'seed_std': np.nan,
            'seed_cv': np.nan,
            'fastest_seed': np.nan,
            'slowest_seed': np.nan,
            'seed_range': np.nan,
            'seed_skewness': np.nan,
            'seed_kurtosis': np.nan,
            'seed_iqr': np.nan,
            'seed_iqr_ratio': np.nan,
            'avg_gap': np.nan,
            'hhi_seed_times': np.nan
        }
    

class WinningTimeAnalyzer:
    def __init__(self):
        self.time_converter = TimeConverter()

    
    def extract_top_eight_results(self, entries: List[Dict]) -> List[Dict]:
        # Get prelim times for top 8 swimmers
        prelim_results = []
        
        for entry in entries:
            prelim_time = self.time_converter.time_to_seconds(entry['prelim_time'])
            seed_time = self.time_converter.time_to_seconds(entry['seed_time'])
            
            if not pd.isna(prelim_time):
                prelim_results.append({
                    'name': entry['name'],
                    'prelim_time': prelim_time,
                    'seed_time': seed_time,
                    'rank': entry['rank']
                })
        
        # Sort by prelim time and take top 8
        prelim_results.sort(key=lambda x: x['prelim_time'])
        return prelim_results[:8]
    

    def calculate_winning_features(self, top_eight: List[Dict]) -> Dict[str, float]:
        features = {}

        if len(top_eight) == 0:
            return self._empty_winning_features()
        
        prelim_times = [entry['prelim_time'] for entry in top_eight]
        seed_times = [entry['seed_time'] for entry in top_eight if not pd.isna(entry['seed_time'])]
        
        # Basic field depth statistics
        features['field_size'] = len(top_eight)
        features['prelim_mean'] = np.mean(prelim_times)
        features['prelim_median'] = np.median(prelim_times)
        features['prelim_std'] = np.std(prelim_times)
        features['prelim_cv'] = features['prelim_std'] / features['prelim_mean'] if features['prelim_mean'] > 0 else 0

        # Distribution features
        if len(prelim_times) >= 3:
            features['prelim_skewness'] = stats.skew(prelim_times)
            features['prelim_kurtosis'] = stats.kurtosis(prelim_times)
        else:
            features['prelim_skewness'] = 0
            features['prelim_kurtosis'] = 0

        # Percentile features
        if len(prelim_times) >= 4:
            q25, q75 = np.percentile(prelim_times, [25, 75])
            features['prelim_iqr'] = q75 - q25
            features['prelim_iqr_ratio'] = features['prelim_iqr'] / features['prelim_median']
        else:
            features['prelim_iqr'] = 0
            features['prelim_iqr_ratio'] = 0

        # Gap analysis
        if len(prelim_times) >= 2:
            gaps = [prelim_times[i+1] - prelim_times[i] for i in range(len(prelim_times)-1)]
            features['max_gap'] = max(gaps)
            features['avg_gap'] = np.mean(gaps)
            features['gap_1st_2nd'] = gaps[0]
            
            # Find biggest gap position
            max_gap_idx = gaps.index(max(gaps))
            features['max_gap_position'] = max_gap_idx + 1
        else:
            features['max_gap'] = 0
            features['avg_gap'] = 0
            features['gap_1st_2nd'] = 0
            features['max_gap_position'] = 0

        # Herfindahl-Hirschman Index for prelim times
        if len(prelim_times) > 1:
            inverted_times = [1/t for t in prelim_times]
            total = sum(inverted_times)
            shares = [t/total for t in inverted_times]
            features['hhi_prelim_times'] = sum(s**2 for s in shares)
        else:
            features['hhi_prelim_times'] = 1.0

        # Pressure Index: Gap between 1st and 2nd prelim (dominance measure)
        if len(prelim_times) >= 2:
            features['pressure_index'] = (prelim_times[1] - prelim_times[0]) / prelim_times[0]
        else:
            features['pressure_index'] = np.nan

        # Dark Horse Potential: Strength of mid-field relative to top 2
        if len(prelim_times) >= 8:
            top_2_avg = np.mean(prelim_times[:2])
            mid_field_avg = np.mean(prelim_times[2:8])
            features['dark_horse_potential'] = (mid_field_avg - top_2_avg) / top_2_avg
        else:
            features['dark_horse_potential'] = np.nan

        # Competitive Bandwidth: Time spread of middle 50% of field
        if len(prelim_times) >= 4:
            q25_idx = len(prelim_times) // 4
            q75_idx = 3 * len(prelim_times) // 4
            competitive_bandwidth = prelim_times[q75_idx] - prelim_times[q25_idx]
            features['competitive_bandwidth'] = competitive_bandwidth
            features['competitive_bandwidth_pct'] = competitive_bandwidth / prelim_times[0]
        else:
            features['competitive_bandwidth'] = np.nan
            features['competitive_bandwidth_pct'] = np.nan

        # Seed vs Prelim analysis
        if seed_times:
            features['seed_prelim_correlation'] = np.corrcoef(seed_times, prelim_times[:len(seed_times)])[0, 1] if len(seed_times) > 1 else np.nan
            
            # Average improvement from seed to prelim
            improvements = []
            for i, prelim_time in enumerate(prelim_times[:len(seed_times)]):
                if not pd.isna(seed_times[i]):
                    improvement = (seed_times[i] - prelim_time) / seed_times[i]
                    improvements.append(improvement)
            
            if improvements:
                features['avg_improvement_pct'] = np.mean(improvements)
                features['improvement_std'] = np.std(improvements)
            else:
                features['avg_improvement_pct'] = np.nan
                features['improvement_std'] = np.nan
        else:
            features['seed_prelim_correlation'] = np.nan
            features['avg_improvement_pct'] = np.nan
            features['improvement_std'] = np.nan

        return features
    

    def _empty_winning_features(self) -> Dict[str, float]:
        return {
            'field_size': 0,
            'prelim_mean': np.nan,
            'prelim_median': np.nan,
            'prelim_std': np.nan,
            'prelim_cv': np.nan,
            'prelim_skewness': np.nan,
            'prelim_kurtosis': np.nan,
            'prelim_iqr': np.nan,
            'prelim_iqr_ratio': np.nan,
            'max_gap': np.nan,
            'avg_gap': np.nan,
            'gap_1st_2nd': np.nan,
            'max_gap_position': np.nan,
            'hhi_prelim_times': np.nan,
            'pressure_index': np.nan,
            'dark_horse_potential': np.nan,
            'competitive_bandwidth': np.nan,
            'competitive_bandwidth_pct': np.nan,
            'seed_prelim_correlation': np.nan,
            'avg_improvement_pct': np.nan,
            'improvement_std': np.nan
        }
    

class ImprovedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        
    def create_distance_normalized_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create distance-normalized features to avoid polynomial explosion"""
        X_new = X.copy()
        
        if 'distance' in X.columns:
            # Normalize distance by common swimming distances
            common_distances = [50, 100, 200, 500, 1000, 1650]
            for dist in common_distances:
                X_new[f'distance_ratio_{dist}'] = X['distance'] / dist
                X_new[f'distance_diff_{dist}'] = abs(X['distance'] - dist) / dist  # Normalized difference
            
            # Log-scale distance features to reduce magnitude
            X_new['distance_log'] = np.log(X['distance'])
            X_new['distance_sqrt'] = np.sqrt(X['distance'])
            
            # Distance categories for different competitive dynamics
            X_new['is_sprint'] = (X['distance'] <= 100).astype(int)
            X_new['is_middle'] = ((X['distance'] > 100) & (X['distance'] <= 400)).astype(int)
            X_new['is_distance'] = (X['distance'] > 400).astype(int)
        
        return X_new
    
    def create_stroke_distance_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create stroke-distance interactions with normalized values"""
        X_new = X.copy()
        
        if 'stroke_encoded' in X.columns and 'distance' in X.columns:
            # Use normalized distance for interactions
            distance_normalized = X['distance'] / 500  # Normalize to 500m scale
            X_new['stroke_distance_interaction'] = X['stroke_encoded'] * distance_normalized
            
            # Stroke-specific distance features (normalized)
            stroke_names = ['free', 'back', 'breast', 'fly', 'im']
            for i, stroke in enumerate(stroke_names):
                stroke_mask = (X['stroke_encoded'] == i).astype(int)
                X_new[f'{stroke}_distance_norm'] = distance_normalized * stroke_mask
                X_new[f'{stroke}_distance_log'] = np.log(X['distance'] + 1) * stroke_mask
                
                # Distance-specific competitive features
                if i == 0:  # Freestyle
                    X_new[f'{stroke}_is_distance_event'] = (X['distance'] > 400).astype(int) * stroke_mask
                    X_new[f'{stroke}_is_sprint_event'] = (X['distance'] <= 100).astype(int) * stroke_mask
        
        # Year-stroke interactions (normalized)
        if 'year' in X.columns and 'stroke_encoded' in X.columns:
            year_normalized = (X['year'] - 2000) / 25  # Normalize year to 0-1 scale
            X_new['year_stroke_interaction'] = year_normalized * X['stroke_encoded']
            
        # Final type interactions (for cutoff task)
        if 'final_type_encoded' in X.columns:
            if 'distance' in X.columns:
                distance_normalized = X['distance'] / 500
                X_new['final_distance_interaction'] = X['final_type_encoded'] * distance_normalized
            if 'stroke_encoded' in X.columns:
                X_new['final_stroke_interaction'] = X['final_type_encoded'] * X['stroke_encoded']
            if 'year' in X.columns:
                year_normalized = (X['year'] - 2000) / 25
                X_new['final_year_interaction'] = X['final_type_encoded'] * year_normalized
        
        return X_new
        
    def create_selective_polynomial_features(self, X: pd.DataFrame, task: str) -> pd.DataFrame:
        """Create polynomial features selectively to avoid explosion"""
        X_new = X.copy()
        
        # Only use key features for polynomials, with normalized values
        key_features = ['year', 'distance', 'stroke_encoded']
        if task == 'cutoff':
            key_features.append('final_type_encoded')
        
        key_features = [f for f in key_features if f in X.columns]
        if key_features:
            X_key = X[key_features].copy()
            
            # Normalize features before polynomial expansion
            if 'year' in X_key.columns:
                X_key['year_norm'] = (X_key['year'] - 2000) / 25
                X_key = X_key.drop('year', axis=1)
            
            if 'distance' in X_key.columns:
                X_key['distance_norm'] = X_key['distance'] / 500
                X_key = X_key.drop('distance', axis=1)
            
            # Create polynomial features with degree 2, interaction only
            poly = PolynomialFeatures(
                degree=2, 
                interaction_only=True,
                include_bias=False
            )
            
            # Fill NaN values before polynomial features
            X_key_clean = X_key.fillna(X_key.median())
            
            X_poly = poly.fit_transform(X_key_clean)
            poly_feature_names = poly.get_feature_names_out(X_key_clean.columns)
            
            # Add only the new polynomial features
            for i, name in enumerate(poly_feature_names):
                if name not in X_new.columns:
                    X_new[name] = X_poly[:, i]
        
        return X_new
    
    def create_distance_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to distance events like 500 Free"""
        X_new = X.copy()
        
        if 'distance' in X.columns:
            # Distance-specific competitive dynamics
            X_new['is_500_free'] = ((X['distance'] == 500) & 
                                   (X['stroke_encoded'] == 0)).astype(int)  # 0 = freestyle
            
            # Distance event categories
            X_new['is_sprint'] = (X['distance'] <= 100).astype(int)
            X_new['is_middle'] = ((X['distance'] > 100) & (X['distance'] <= 400)).astype(int)
            X_new['is_distance'] = (X['distance'] > 400).astype(int)
            
            # Distance-specific pacing features
            X_new['distance_pacing_factor'] = np.log(X['distance'] + 1) / np.log(500 + 1)
            
            # Competitive bandwidth features for distance events
            if 'field_size' in X.columns:
                X_new['distance_competition_density'] = X['field_size'] / X['distance'] * 100
        
        return X_new
    
    def create_ratio_features(self, X: pd.DataFrame, task: str) -> pd.DataFrame:
        """Create ratio features with better normalization"""
        X_new = X.copy()
        
        # Distance-based ratios (normalized)
        if 'distance' in X.columns:
            common_distances = [50, 100, 200, 500, 1000, 1650]
            for dist in common_distances:
                X_new[f'distance_ratio_{dist}'] = X['distance'] / dist
                X_new[f'distance_diff_{dist}'] = abs(X['distance'] - dist) / dist
        
        # Year-based features (normalized)
        if 'year' in X.columns:
            current_year = X['year'].max()
            X_new['years_since_latest'] = (current_year - X['year']) / 25  # Normalized
            X_new['year_progression'] = (X['year'] - X['year'].min()) / (X['year'].max() - X['year'].min())
        
        return X_new
    
    def engineer_features(self, X: pd.DataFrame, task: str) -> pd.DataFrame:
        """Engineer features with improved handling for distance events"""
        print(f"Starting improved feature engineering for {task}...")
        print(f"Initial features: {X.shape[1]}")
        
        # Step 1: Distance-normalized features
        X = self.create_distance_normalized_features(X)
        print(f"After distance normalization: {X.shape[1]}")
        
        # Step 2: Stroke-distance interactions (normalized)
        X = self.create_stroke_distance_interactions(X)
        print(f"After stroke-distance interactions: {X.shape[1]}")
        
        # Step 3: Distance-specific features
        X = self.create_distance_specific_features(X)
        print(f"After distance-specific features: {X.shape[1]}")
        
        # Step 4: Ratio features
        X = self.create_ratio_features(X, task)
        print(f"After ratio features: {X.shape[1]}")
        
        # Step 5: Selective polynomial features
        X = self.create_selective_polynomial_features(X, task)
        print(f"After selective polynomial features: {X.shape[1]}")
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        return X


class ImprovedOutlierHandler:
    def __init__(self, method='iqr', threshold=3.0):
        self.method = method
        self.threshold = threshold
        self.outlier_bounds = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task: str):
        """Fit outlier detection with distance-specific handling"""
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
    
    def remove_outliers(self, X: pd.DataFrame, y: pd.Series, task: str):
        """Remove outliers with distance-specific considerations"""
        if task not in self.outlier_bounds:
            self.fit(X, y, task)
        
        lower_bound, upper_bound = self.outlier_bounds[task]
        
        # More conservative outlier handling for distance events
        if 'distance' in X.columns:
            distance_mask = X['distance'] > 400
            if distance_mask.any():
                # Use wider bounds for distance events
                distance_lower = lower_bound - (upper_bound - lower_bound) * 0.5
                distance_upper = upper_bound + (upper_bound - lower_bound) * 0.5
                
                # Apply different bounds for distance vs non-distance events
                outlier_mask = (
                    ((y < lower_bound) | (y > upper_bound)) & ~distance_mask |
                    ((y < distance_lower) | (y > distance_upper)) & distance_mask
                )
            else:
                outlier_mask = (y < lower_bound) | (y > upper_bound)
        else:
            outlier_mask = (y < lower_bound) | (y > upper_bound)
        
        # Remove outliers
        X_clean = X[~outlier_mask].copy()
        y_clean = y[~outlier_mask].copy()
        
        print(f"Removed {outlier_mask.sum()} outliers from {task} data")
        return X_clean, y_clean


class ImprovedFeatureSelector:
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


class FeatureEngineer:
    def __init__(self):
        self.time_converter = TimeConverter()
        self.seed_time_analyzer = SeedTimeAnalyzer()
        self.winning_time_analyzer = WinningTimeAnalyzer()
        self.advanced_engineer = ImprovedFeatureEngineer()
        self.feature_selector = ImprovedFeatureSelector()

    
    def parse_entries(self, entries_str: str) -> List[Dict]:
        if pd.isna(entries_str) or entries_str == '[]':
            return []
        
        try:
            entries_list = ast.literal_eval(entries_str)
            parsed_entries = []
            for entry in entries_list:
                parsed_entries.append({
                    'rank': entry.get('rank', ''),
                    'name': entry.get('name', ''),
                    'yr': entry.get('yr', ''),
                    'school': entry.get('school', ''),
                    'seed_time': entry.get('seed_time', ''),
                    'prelim_time': entry.get('prelim_time', ''),
                    'finals_time': entry.get('finals_time', '')
                })
            return parsed_entries
        except:
            return []


    def create_cutoff_features(self, row: pd.Series) -> Optional[Dict]:
        features = {}

        # Add basic event information
        features['year'] = row['year']
        features['meet'] = str(row['meet']).strip()
        features['stroke'] = row['stroke']
        features['gender'] = row['gender']
        features['distance'] = row['distance']
        features['event_name'] = row['event_name']

        # Parse entries
        entries = self.parse_entries(str(row['results']))

        # Get seed times
        seed_times = self.seed_time_analyzer.extract_seed_times(entries)

        # Only include events with valid seed times
        if len(seed_times) == 0:
            return None

        # Add cutoff features
        features.update(self.seed_time_analyzer.calculate_cutoff_features(seed_times))

        # Add target features (cutoff times)
        features['a_final_cutoff_sec'] = row['a_final_cutoff_sec']
        features['b_final_cutoff_sec'] = row['b_final_cutoff_sec']
        features['c_final_cutoff_sec'] = row['c_final_cutoff_sec']

        return features
    

    def create_winning_features(self, row: pd.Series) -> Optional[Dict]:
        features = {}

        # Add basic event information
        features['year'] = row['year']
        features['meet'] = str(row['meet']).strip()
        features['stroke'] = row['stroke']
        features['gender'] = row['gender']
        features['distance'] = row['distance']
        features['event_name'] = row['event_name']

        # Parse entries
        entries = self.parse_entries(str(row['results']))

        # Get top 8 prelim results
        top_eight = self.winning_time_analyzer.extract_top_eight_results(entries)

        # Only include events with valid prelim results
        if len(top_eight) == 0:
            return None

        # Add winning features
        features.update(self.winning_time_analyzer.calculate_winning_features(top_eight))

        # Add target features (winning times)
        features['winning_time_sec'] = row['winning_time_sec']

        return features


    def engineer_cutoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_rows = []
        for _, row in df.iterrows():
            event_features = self.create_cutoff_features(row)
            if event_features is not None:
                feature_rows.append(event_features)
        features_df = pd.DataFrame(feature_rows)
        if not features_df.empty:
            features_df = self._add_event_level_features(features_df)
        return features_df if isinstance(features_df, pd.DataFrame) else pd.DataFrame(features_df)

    
    def engineer_winning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_rows = []
        for _, row in df.iterrows():
            event_features = self.create_winning_features(row)
            if event_features is not None:
                feature_rows.append(event_features)
        features_df = pd.DataFrame(feature_rows)
        if not features_df.empty:
            features_df = self._add_event_level_features(features_df)
        return features_df if isinstance(features_df, pd.DataFrame) else pd.DataFrame(features_df)

    
    def _add_event_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['distance_category'] = pd.cut(df['distance'], 
                                       bins=[0, 100, 200, 400, 800, 2000],
                                       labels=['sprint', 'short', 'middle', 'distance', 'long_distance'])
        
        # Stroke encoding
        stroke_map = {
            'Freestyle': 'free',
            'Backstroke': 'back',
            'Breaststroke': 'breast', 
            'Butterfly': 'fly',
            'IM': 'im'
        }
        df['stroke_category'] = df['stroke'].map(lambda x: stroke_map.get(x, x))
        
        return df


    def handle_outliers(self, df, features=None, method='clip', z_thresh=2.0):
        """
        Clip or replace outlier features in the dataframe.
        - features: list of feature names to process (default: all numeric columns)
        - method: 'clip' (default) or 'mean' (replace outliers with mean)
        - z_thresh: Z-score threshold for outlier detection
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        for feat in features:
            if feat not in df.columns:
                continue
            vals = df[feat]
            mean = vals.mean()
            std = vals.std()
            if std == 0 or np.isnan(std):
                continue
            z = (vals - mean) / std
            if method == 'clip':
                df[feat] = np.where(z > z_thresh, mean + z_thresh * std,
                                    np.where(z < -z_thresh, mean - z_thresh * std, vals))
            elif method == 'mean':
                df[feat] = np.where(np.abs(z) > z_thresh, mean, vals)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

    def engineer(self, df, event_type=None):
        # After all other feature engineering steps, handle outliers:
        outlier_features = [
            'prelim_skewness', 'prelim_kurtosis', 'prelim_mean', 'prelim_std',
            'seed_skewness', 'seed_kurtosis', 'seed_mean', 'seed_std',
            'field_size', 'seed_cv', 'fastest_seed', 'slowest_seed', 'seed_range'
        ]
        features_to_clip = [f for f in outlier_features if f in df.columns]
        if features_to_clip:
            df = self.handle_outliers(df, features=features_to_clip, method='clip', z_thresh=2.0)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def load_data() -> pd.DataFrame:
    data_file = Path(__file__).parent.parent / "data" / "processed" / "clean" / "combined_individual_events.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} does not exist.")
    
    df = pd.read_csv(data_file)
    print(f"Loaded data with shape {df.shape}")
    
    return df


def main():
    print("Engineering NESCAC features")

    df = load_data()

    feature_engineer = FeatureEngineer()

    # Engineer cutoff features
    print("Creating cutoff features...")
    cutoff_features_df = feature_engineer.engineer_cutoff_features(df)
    
    # Engineer winning features
    print("Creating winning features...")
    winning_features_df = feature_engineer.engineer_winning_features(df)

    # Save features
    output_dir = Path(__file__).parent.parent / "data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_output_path = output_dir / "cutoff_features.csv"
    winning_output_path = output_dir / "winning_features.csv"

    cutoff_features_df.to_csv(cutoff_output_path, index=False)
    winning_features_df.to_csv(winning_output_path, index=False)
    
    print(f"Cutoff features saved to {cutoff_output_path}")
    print(f"Winning features saved to {winning_output_path}")
    print(f"Cutoff features shape: {cutoff_features_df.shape}")
    print(f"Winning features shape: {winning_features_df.shape}")


if __name__ == "__main__":
    main()
