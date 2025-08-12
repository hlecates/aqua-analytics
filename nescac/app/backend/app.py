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

from nescac_src.modeling.predict import PredictionEngine
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

# Paths for data and plots
nescac_root = project_root
data_path = nescac_root / "data"
processed_features_path = data_path / "processed" / "features"
winning_features_csv = processed_features_path / "winning_features.csv"
cutoff_features_csv = processed_features_path / "cutoff_features.csv"
school_fastest_csv = data_path / "school-specific" / "fastest_times_by_school.csv"
plots_root = nescac_root / "output" / "plots"
school_individual_event_dir = plots_root / "schools" / "individual-event"
school_fastest_counts_dir = plots_root / "schools" / "event-fastest-counts"
event_cutoffs_dir = plots_root / "event_cutoffs"
winning_times_dir = plots_root / "winning_times"

# Simple in-memory cache for dataframes
_cached_dfs = {
    "winning": None,
    "cutoff": None,
    "school_fastest": None,
}


def _load_df(kind: str):
    if kind == "winning":
        if _cached_dfs["winning"] is None and winning_features_csv.exists():
            _cached_dfs["winning"] = pd.read_csv(winning_features_csv)
        return _cached_dfs["winning"]
    if kind == "cutoff":
        if _cached_dfs["cutoff"] is None and cutoff_features_csv.exists():
            _cached_dfs["cutoff"] = pd.read_csv(cutoff_features_csv)
        return _cached_dfs["cutoff"]
    if kind == "school_fastest":
        if _cached_dfs["school_fastest"] is None and school_fastest_csv.exists():
            _cached_dfs["school_fastest"] = pd.read_csv(school_fastest_csv)
        return _cached_dfs["school_fastest"]
    return None


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
    # Prefer real availability and ranges from engine when possible
    try:
        engine = get_prediction_engine()
        availability_raw = engine.check_data_availability()
        year_ranges_raw = engine.get_year_ranges()

        # Normalize availability to plain Python bools
        availability = {}
        if isinstance(availability_raw, dict):
            availability = {str(k): bool(v) for k, v in availability_raw.items()}

        # Normalize year ranges to lists of built-in ints
        year_ranges = {}
        if isinstance(year_ranges_raw, dict):
            for k, v in year_ranges_raw.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    if len(v) >= 2:
                        a, b = v[0], v[1]
                        year_ranges[str(k)] = [int(a), int(b)]
                    else:
                        year_ranges[str(k)] = [int(x) for x in v]
                else:
                    # Fallback: if provided as a scalar or unsupported type, coerce safely
                    try:
                        year_ranges[str(k)] = [int(v)]
                    except Exception:
                        year_ranges[str(k)] = []
    except Exception:
        availability = {
            'combined_data': True,
            'cutoff_features': True,
            'winning_features': True,
            'simple_models': True,
            'advanced_models': True
        }
        year_ranges = {
            'data': [2002, 2025],
            'cutoff_features': [2006, 2025],
            'winning_features': [2002, 2025]
        }

    return jsonify({
        'data_availability': availability,
        'year_ranges': year_ranges
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
            'name': f"{gender} {distance} {stroke}",
            'event_key': f"{gender}_{distance}_{stroke}"
        })
    
    return jsonify(events)

@app.route('/api/schools', methods=['GET'])
def list_schools():
    # Prefer filesystem, fallback to CSV
    schools = []
    try:
        if school_individual_event_dir.exists():
            for item in sorted(school_individual_event_dir.iterdir()):
                if item.is_dir():
                    schools.append(item.name)
        if not schools:
            df = _load_df("school_fastest")
            if df is not None:
                schools = sorted(df['school'].dropna().unique().tolist())
    except Exception:
        pass
    return jsonify(schools)

@app.route('/api/plots/<path:subpath>', methods=['GET'])
def serve_plot(subpath):
    # Serve images located under output/plots
    safe_root = plots_root
    requested_path = (safe_root / subpath).resolve()
    # Prevent path traversal
    if safe_root not in requested_path.parents and requested_path != safe_root:
        return jsonify({'error': 'Invalid path'}), 400
    if not requested_path.exists() or not requested_path.is_file():
        return jsonify({'error': 'File not found'}), 404
    # Determine directory and filename
    directory = str(requested_path.parent)
    filename = requested_path.name
    return send_from_directory(directory, filename)

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

# -------- Stats Endpoints --------

@app.route('/api/stats/overall', methods=['GET'])
def overall_stats():
    winning_df = _load_df("winning")
    cutoff_df = _load_df("cutoff")
    if winning_df is None:
        return jsonify({'error': 'Winning features not available'}), 500

    # Only Men for now as per dataset/plots
    winning_df = winning_df[winning_df['gender'] == 'Men']
    if cutoff_df is not None:
        cutoff_df = cutoff_df[cutoff_df['gender'] == 'Men']

    result = {}
    for (stroke, distance), grp in winning_df.groupby(['stroke', 'distance']):
        grp_sorted = grp.sort_values('year')
        event_name = f"{int(distance)}_{stroke}" if isinstance(distance, (int, float)) else f"{distance}_{stroke}"
        start_year = int(grp_sorted['year'].iloc[0])
        end_year = int(grp_sorted['year'].iloc[-1])
        start_win = float(grp_sorted['winning_time_sec'].iloc[0])
        end_win = float(grp_sorted['winning_time_sec'].iloc[-1])
        delta_win = end_win - start_win
        years_span = max(1, end_year - start_year)
        avg_annual_change_win = delta_win / years_span

        entry = {
            'event_name': event_name,
            'start_year': start_year,
            'end_year': end_year,
            'winning_time_start': start_win,
            'winning_time_end': end_win,
            'winning_time_delta': delta_win,
            'winning_time_avg_annual_change': avg_annual_change_win,
        }

        if cutoff_df is not None:
            cgrp = cutoff_df[(cutoff_df['stroke'] == stroke) & (cutoff_df['distance'] == distance)]
            if not cgrp.empty:
                cgrp_sorted = cgrp.sort_values('year')
                for final_col in ['a_final_cutoff_sec', 'b_final_cutoff_sec', 'c_final_cutoff_sec']:
                    if final_col in cgrp_sorted.columns and cgrp_sorted[final_col].notna().any():
                        start_val = float(cgrp_sorted[final_col].dropna().iloc[0])
                        end_val = float(cgrp_sorted[final_col].dropna().iloc[-1])
                        entry[f'{final_col}_start'] = start_val
                        entry[f'{final_col}_end'] = end_val
                        entry[f'{final_col}_delta'] = end_val - start_val
        result[event_name] = entry

    return jsonify(result)

@app.route('/api/stats/event', methods=['GET'])
def event_stats():
    gender = request.args.get('gender', 'Men')
    stroke = request.args.get('stroke')
    distance = request.args.get('distance', type=int)
    if not stroke or distance is None:
        return jsonify({'error': 'Missing stroke or distance'}), 400

    winning_df = _load_df("winning")
    cutoff_df = _load_df("cutoff")
    if winning_df is None:
        return jsonify({'error': 'Winning features not available'}), 500

    w = winning_df[(winning_df['gender'] == gender) & (winning_df['stroke'] == stroke) & (winning_df['distance'] == distance)]
    w = w.sort_values('year')
    series = {
        'years': w['year'].astype(int).tolist(),
        'winning_time_sec': w['winning_time_sec'].astype(float).tolist(),
    }

    stats = {}
    if len(w) >= 2:
        stats['winning_time_start'] = float(w['winning_time_sec'].iloc[0])
        stats['winning_time_end'] = float(w['winning_time_sec'].iloc[-1])
        stats['winning_time_delta'] = stats['winning_time_end'] - stats['winning_time_start']
        span = int(w['year'].iloc[-1]) - int(w['year'].iloc[0])
        stats['winning_time_avg_annual_change'] = stats['winning_time_delta'] / max(1, span)

    if cutoff_df is not None:
        c = cutoff_df[(cutoff_df['gender'] == gender) & (cutoff_df['stroke'] == stroke) & (cutoff_df['distance'] == distance)]
        c = c.sort_values('year')
        for col in ['a_final_cutoff_sec', 'b_final_cutoff_sec', 'c_final_cutoff_sec']:
            if col in c.columns:
                series[col] = c[col].astype(float).fillna(pd.NA).tolist()
        if len(c) >= 2 and c['a_final_cutoff_sec'].notna().any():
            a_nonnull = c['a_final_cutoff_sec'].dropna()
            stats['a_final_cutoff_start'] = float(a_nonnull.iloc[0])
            stats['a_final_cutoff_end'] = float(a_nonnull.iloc[-1])
            stats['a_final_cutoff_delta'] = stats['a_final_cutoff_end'] - stats['a_final_cutoff_start']

    return jsonify({'series': series, 'stats': stats})

@app.route('/api/stats/school', methods=['GET'])
def school_stats():
    school = request.args.get('school')
    event_name = request.args.get('event_name')  # e.g., "100_Backstroke"
    if not school:
        return jsonify({'error': 'Missing school'}), 400

    df = _load_df("school_fastest")
    if df is None:
        return jsonify({'error': 'School fastest data not available'}), 500

    sdf = df[df['school'] == school]
    result = {
        'school': school,
        'total_fastest_count': int(len(sdf))
    }

    if event_name:
        edf = sdf[sdf['event_name'] == event_name].sort_values('year')
        years = edf['year'].astype(int).tolist()
        times = edf['fastest_time_sec'].astype(float).tolist()
        result['event'] = event_name
        result['years'] = years
        result['fastest_time_sec'] = times
        if len(times) >= 2:
            result['start_time'] = float(times[0])
            result['end_time'] = float(times[-1])
            result['delta_time'] = result['end_time'] - result['start_time']
            span = years[-1] - years[0]
            result['avg_annual_change'] = result['delta_time'] / max(1, span)
        result['fastest_count_for_event'] = int(len(edf))
    else:
        # Aggregate counts per event
        counts = sdf.groupby('event_name').size().sort_values(ascending=False)
        result['fastest_counts_by_event'] = counts.to_dict()

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 