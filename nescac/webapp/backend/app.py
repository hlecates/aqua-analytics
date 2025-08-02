from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
from pathlib import Path
import numpy as np

# Add the src directory to the path to import modules
# Navigate up from webapp/backend to the project root, then to src
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from predict import PredictionEngine
import pandas as pd
import json

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)  # Enable CORS for React frontend

# Initialize the prediction engine
prediction_engine = None

def get_prediction_engine():
    global prediction_engine
    if prediction_engine is None:
        prediction_engine = PredictionEngine()
    return prediction_engine

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'NESCAC Swimming Analytics API is running'
    })

@app.route('/api/data-status', methods=['GET'])
def data_status():
    # For now, return a simple status without using the prediction engine
    return jsonify({
        'data_availability': {
            'combined_data': True,
            'cutoff_features': True,
            'winning_features': True,
            'simple_models': True,
            'advanced_models': True
        },
        'year_ranges': {
            'data': [2002, 2025],
            'cutoff_features': [2006, 2025],
            'winning_features': [2002, 2025]
        }
    })

@app.route('/api/events', methods=['GET'])
def get_events():
    engine = get_prediction_engine()
    events = []
    
    for gender, distance, stroke in engine.events:
        events.append({
            'gender': gender,
            'distance': distance,
            'stroke': stroke,
            'name': f"{gender} {distance} {stroke}"
        })
    
    return jsonify(events)

@app.route('/api/predictions/<int:year>', methods=['GET'])
def get_predictions(year):
    try:
        engine = get_prediction_engine()
        predictions = engine.generate_predictions(year)
        
        # Format predictions for frontend
        formatted_predictions = []
        
        for event_key, event_data in predictions.items():
            if isinstance(event_data, dict) and 'simple_cutoff' in event_data:
                formatted_predictions.append({
                    'event': event_key,
                    'predictions': event_data
                })
        
        return jsonify({
            'year': year,
            'predictions': formatted_predictions,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/personal-analysis', methods=['POST'])
def personal_analysis():
    try:
        data = request.get_json()
        year = data.get('year')
        personal_times = data.get('personal_times', {})
        
        engine = get_prediction_engine()
        predictions = engine.generate_predictions(year)
        
        analysis = []
        
        for event_key, event_data in predictions.items():
            if event_key in personal_times:
                personal_time = personal_times[event_key]
                
                # Get predictions for this event
                if isinstance(event_data, dict):
                    simple_cutoff = event_data.get('simple_cutoff', {})
                    advanced_cutoff = event_data.get('advanced_cutoff', {})
                    
                    # Determine which finals the swimmer would make
                    simple_finals = []
                    advanced_finals = []
                    
                    for final_type in ['A', 'B', 'C']:
                        if final_type in simple_cutoff:
                            if personal_time <= simple_cutoff[final_type]:
                                simple_finals.append(final_type)
                        
                        if final_type in advanced_cutoff:
                            if personal_time <= advanced_cutoff[final_type]:
                                advanced_finals.append(final_type)
                    
                    analysis.append({
                        'event': event_key,
                        'personal_time': personal_time,
                        'simple_finals': simple_finals,
                        'advanced_finals': advanced_finals,
                        'predictions': event_data
                    })
        
        return jsonify({
            'year': year,
            'analysis': analysis,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/historical-data/<int:year>', methods=['GET'])
def get_historical_data(year):
    try:
        engine = get_prediction_engine()
        actual_times = engine.get_actual_times(year)
        
        return jsonify({
            'year': year,
            'actual_times': actual_times,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 