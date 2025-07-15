import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        return df
    
    def prepare_cutoff_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        cutoff_data = []
        
        for _, row in df.iterrows():
            # A final cutoff
            if pd.notna(row['a_final_cutoff_sec']):
                cutoff_data.append({
                    'year': row['year'],
                    'stroke': row['stroke'],
                    'distance': row['distance'],
                    'final_type': 'A',
                    'cutoff_sec': row['a_final_cutoff_sec']
                })
            
            # B final cutoff
            if pd.notna(row['b_final_cutoff_sec']):
                cutoff_data.append({
                    'year': row['year'],
                    'stroke': row['stroke'],
                    'distance': row['distance'],
                    'final_type': 'B',
                    'cutoff_sec': row['b_final_cutoff_sec']
                })
            
            # C final cutoff
            if pd.notna(row['c_final_cutoff_sec']):
                cutoff_data.append({
                    'year': row['year'],
                    'stroke': row['stroke'],
                    'distance': row['distance'],
                    'final_type': 'C',
                    'cutoff_sec': row['c_final_cutoff_sec']
                })
        
        cutoff_df = pd.DataFrame(cutoff_data)
        cutoff_df = cutoff_df.dropna()
        
        # Prepare features and target
        X = self._prepare_features(cutoff_df, 'cutoff')
        y = cutoff_df['cutoff_sec']
        
        return X, y
    
    def prepare_winning_time_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Filter for rows with winning times
        winning_df = df.dropna(subset=['winning_time_sec']).copy()
        
        # Prepare features and target
        X = self._prepare_features(winning_df, 'winning')
        y = winning_df['winning_time_sec']
        
        return X, y
    
    def _prepare_features(self, df: pd.DataFrame, task: str) -> pd.DataFrame:
        # Encode categorical variables
        le_stroke = LabelEncoder()
        le_final_type = LabelEncoder()
        
        X = df[['year', 'distance', 'stroke']].copy()
        X['stroke_encoded'] = le_stroke.fit_transform(X['stroke'])
        
        if task == 'cutoff':
            X['final_type'] = df['final_type']
            X['final_type_encoded'] = le_final_type.fit_transform(X['final_type'])
            feature_cols = ['year', 'distance', 'stroke_encoded', 'final_type_encoded']
        else:
            feature_cols = ['year', 'distance', 'stroke_encoded']
        
        # Store encoders for later use
        if task not in self.label_encoders:
            self.label_encoders[task] = {}
        
        self.label_encoders[task]['stroke'] = le_stroke
        if task == 'cutoff':
            self.label_encoders[task]['final_type'] = le_final_type
        
        return X[feature_cols]


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scalers = {}
    
    def define_models(self) -> Dict[str, Any]:
        return {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
    
    def _temporal_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                                  test_year_threshold: int = 2024) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        # Assuming 'year' is the first column in your feature matrix
        train_mask = X.iloc[:, 0] < test_year_threshold
        test_mask = X.iloc[:, 0] >= test_year_threshold
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, task: str, 
                    test_year_threshold: int = 2024, temporal_split: bool = True) -> Dict[str, Dict]:
        models = self.define_models()
        
        if temporal_split:
            # Do temporal split on DataFrame BEFORE scaling
            X_train, X_test, y_train, y_test = self._temporal_train_test_split(
                X, y, test_year_threshold
            )
            
            print(f"Temporal split - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            print(f"Train years: {X_train.iloc[:, 0].min()}-{X_train.iloc[:, 0].max()}")
            print(f"Test years: {X_test.iloc[:, 0].min()}-{X_test.iloc[:, 0].max()}")
            
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"Random split - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Scale the split data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for later use
        self.scalers[task] = scaler
        
        self.models[task] = {}
        self.results[task] = {}
        
        for name, model in models.items():
            print(f"\nTraining {name} for {task} prediction...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation on training data
            if temporal_split:
                # Use TimeSeriesSplit for temporal data
                cv = TimeSeriesSplit(n_splits=5)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
            
            # Store results
            self.models[task][name] = model
            self.results[task][name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.3f}")
            print(f"CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.results[task]
    
    def get_best_model(self, task: str) -> Tuple[str, Any]:
        if task not in self.results:
            raise ValueError(f"No results found for task: {task}")
        
        # Find best model by R² score
        best_model_name = max(self.results[task].keys(), 
                            key=lambda x: self.results[task][x]['r2'])
        best_model = self.models[task][best_model_name]
        
        return best_model_name, best_model


class ModelSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_models_and_preprocessing(self, trainer: ModelTrainer, 
                                    preprocessor: Preprocessor,
                                    results: Dict[str, Dict]) -> None:
        
        # Save preprocessing objects
        preprocessing_objects = {
            'scalers': trainer.scalers,  # Now stored in trainer
            'label_encoders': preprocessor.label_encoders
        }
        
        with open(self.output_dir / 'nescac_preprocessing.pkl', 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        # Save models
        all_models = {}
        for task, task_models in trainer.models.items():
            all_models[task] = {}
            for model_name, model in task_models.items():
                all_models[task][model_name] = {
                    'model': model,
                    'model_type': model_name
                }
        
        with open(self.output_dir / 'nescac_models.pkl', 'wb') as f:
            pickle.dump(all_models, f)
        
        # Save results
        with open(self.output_dir / 'nescac_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save metadata
        metadata = {
            'tasks': list(trainer.models.keys()),
            'models_per_task': {task: list(task_models.keys()) 
                              for task, task_models in trainer.models.items()},
            'timestamp': pd.Timestamp.now().isoformat(),
            'preprocessing_info': {
                'scalers': list(trainer.scalers.keys()),
                'label_encoders': {task: list(encoders.keys()) 
                                 for task, encoders in preprocessor.label_encoders.items()}
            }
        }
        
        with open(self.output_dir / 'nescac_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nModels and preprocessing saved to: {self.output_dir}")


def create_prediction_functions(trainer: ModelTrainer, 
                              preprocessor: Preprocessor) -> Dict[str, callable]:
    
    def predict_cutoff(year: int, stroke: str, distance: int, final_type: str) -> float:
        # Encode features
        stroke_encoded = preprocessor.label_encoders['cutoff']['stroke'].transform([stroke])[0]
        final_type_encoded = preprocessor.label_encoders['cutoff']['final_type'].transform([final_type])[0]
        
        # Create feature array
        features = np.array([[year, distance, stroke_encoded, final_type_encoded]])
        
        # Scale features
        features_scaled = trainer.scalers['cutoff'].transform(features)
        
        # Get best model and predict
        best_model_name, best_model = trainer.get_best_model('cutoff')
        prediction = best_model.predict(features_scaled)[0]
        
        return prediction
    
    def predict_winning_time(year: int, stroke: str, distance: int) -> float:
        # Encode features
        stroke_encoded = preprocessor.label_encoders['winning']['stroke'].transform([stroke])[0]
        
        # Create feature array
        features = np.array([[year, distance, stroke_encoded]])
        
        # Scale features
        features_scaled = trainer.scalers['winning'].transform(features)
        
        # Get best model and predict
        best_model_name, best_model = trainer.get_best_model('winning')
        prediction = best_model.predict(features_scaled)[0]
        
        return prediction
    
    return {
        'predict_cutoff': predict_cutoff,
        'predict_winning_time': predict_winning_time
    }


def main():
    
    # Configuration
    data_file = Path(__file__).parent.parent / "data" / "processed" / "clean" / "combined_individual_events.csv"
    models_output_dir = Path(__file__).parent.parent / "output" / "models"
    
    print("NESCAC Swimming Prediction Model")
    print("=" * 50)
    
    # Initialize components
    preprocessor = Preprocessor()
    trainer = ModelTrainer()
    saver = ModelSaver(str(models_output_dir))
    
    # Load data
    print(f"Loading data from: {data_file}")
    df = preprocessor.load_data(str(data_file))
    print(f"Loaded {len(df)} records")
    
    # Prepare data for cutoff prediction
    print("\nPreparing cutoff prediction data...")
    X_cutoff, y_cutoff = preprocessor.prepare_cutoff_data(df)
    print(f"Cutoff data shape: {X_cutoff.shape}")
    
    # Train cutoff models (DataFrame passed directly, scaling handled internally)
    print("\nTraining cutoff prediction models...")
    cutoff_results = trainer.train_models(X_cutoff, y_cutoff, 'cutoff', 
                                         test_year_threshold=2024, temporal_split=True)
    
    # Prepare data for winning time prediction
    print("\nPreparing winning time prediction data...")
    X_winning, y_winning = preprocessor.prepare_winning_time_data(df)
    print(f"Winning time data shape: {X_winning.shape}")
    
    # Train winning time models
    print("\nTraining winning time prediction models...")
    winning_results = trainer.train_models(X_winning, y_winning, 'winning', 
                                          test_year_threshold=2024, temporal_split=True)
    
    # Combine results
    all_results = {
        'cutoff': cutoff_results,
        'winning': winning_results
    }
    
    # Save models and preprocessing
    saver.save_models_and_preprocessing(trainer, preprocessor, all_results)
    
    # Create prediction functions
    prediction_functions = create_prediction_functions(trainer, preprocessor)
    
    # Print best models
    print("\nBest Models:")
    print("=" * 30)
    for task in ['cutoff', 'winning']:
        best_name, _ = trainer.get_best_model(task)
        best_r2 = trainer.results[task][best_name]['r2']
        print(f"{task.capitalize()} prediction: {best_name} (R² = {best_r2:.3f})")
    
    # Example predictions
    print("\nExample Predictions:")
    print("=" * 30)
    
    # Cutoff predictions
    examples_cutoff = [
        (2025, 'Freestyle', 100, 'A'),
        (2025, 'Backstroke', 200, 'B'),
        (2025, 'Breaststroke', 100, 'C')
    ]
    
    print("\nCutoff Time Predictions:")
    for year, stroke, distance, final_type in examples_cutoff:
        try:
            pred = prediction_functions['predict_cutoff'](year, stroke, distance, final_type)
            minutes = int(pred // 60)
            seconds = pred % 60
            time_str = f"{minutes}:{seconds:05.2f}" if minutes > 0 else f"{seconds:.2f}"
            print(f"{year} {stroke} {distance} {final_type} Final: {time_str}")
        except Exception as e:
            print(f"Error predicting {year} {stroke} {distance} {final_type}: {e}")
    
    # Winning time predictions
    examples_winning = [
        (2025, 'Freestyle', 100),
        (2025, 'Backstroke', 200),
        (2025, 'Breaststroke', 100)
    ]
    
    print("\nWinning Time Predictions:")
    for year, stroke, distance in examples_winning:
        try:
            pred = prediction_functions['predict_winning_time'](year, stroke, distance)
            minutes = int(pred // 60)
            seconds = pred % 60
            time_str = f"{minutes}:{seconds:05.2f}" if minutes > 0 else f"{seconds:.2f}"
            print(f"{year} {stroke} {distance}: {time_str}")
        except Exception as e:
            print(f"Error predicting {year} {stroke} {distance}: {e}")


if __name__ == "__main__":
    main()