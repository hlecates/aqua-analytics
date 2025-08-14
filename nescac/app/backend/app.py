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
import pickle

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

@app.route('/api/predict-simple/<int:year>', methods=['GET'])
def predict_simple_excluding_year(year):
    try:
        engine = get_prediction_engine()
        
        # For now, use the existing prediction engine but only return simple model results
        # This avoids the complexity of retraining models in the API
        print(f"Generating simple model predictions for {year}...")
        
        # Ensure simple models are available
        print("Checking data availability...")
        availability = engine.check_data_availability()
        print(f"Data availability: {availability}")
        
        # Check if data files exist
        print(f"Combined data path exists: {engine.combined_data_path.exists()}")
        print(f"Combined data path: {engine.combined_data_path}")
        
        # If simple models don't exist, train them
        if not availability.get('simple_models', False):
            print("Simple models not found. Training them now...")
            try:
                engine.ensure_simple_models_exist()
                print("Simple models training completed.")
            except Exception as e:
                print(f"Error training simple models: {e}")
                import traceback
                traceback.print_exc()
        
        # Check what years are available in the data
        year_ranges = engine.get_year_ranges()
        print(f"Available year ranges: {year_ranges}")
        
        # Use the existing prediction engine but filter to only simple models
        predictions = engine.generate_predictions(year)
        
        print(f"Raw predictions keys: {list(predictions.keys())}")
        if predictions:
            sample_event = list(predictions.keys())[0]
            print(f"Sample event data: {predictions[sample_event]}")
            print(f"Sample event actual times: {predictions[sample_event].get('actual', {})}")
        
        # Format predictions for frontend, only including simple model results
        formatted_predictions = []
        
        for event_key, event_data in predictions.items():
            print(f"Processing event: {event_key}")
            print(f"Event data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'Not a dict'}")
            print(f"Actual times for {event_key}: {event_data.get('actual', {})}")
            
            if isinstance(event_data, dict) and 'simple_winning' in event_data:
                # Only include simple model predictions
                filtered_event = {
                    'event': event_key,
                    'predictions': {
                        'simple_winning': event_data.get('simple_winning'),
                        'simple_cutoff': {
                            'A': event_data.get('simple_a_cutoff'),
                            'B': event_data.get('simple_b_cutoff'),
                            'C': event_data.get('simple_c_cutoff')
                        },
                        'actual': event_data.get('actual', {
                            'winning_time': None,
                            'a_cutoff': None,
                            'b_cutoff': None,
                            'c_cutoff': None
                        })
                    }
                }
                formatted_predictions.append(filtered_event)
                print(f"Added event: {event_key}")
            else:
                print(f"Skipping event {event_key} - no simple_winning found")
        
        print(f"Total formatted predictions: {len(formatted_predictions)}")
        
        return jsonify({
            'year': year,
            'predictions': formatted_predictions,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in predict_simple_excluding_year: {e}")
        import traceback
        traceback.print_exc()
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

# -------- History Endpoints (DB-backed) --------
from sqlalchemy import select, func
from nescac_src.db.session import get_session, init_db
from nescac_src.db.models import Event, Meet, Athlete, ResultIndividual, School

@app.route('/api/history/years', methods=['GET'])
def history_years():
    try:
        init_db()
        with get_session() as s:
            years = s.execute(select(Meet.year).distinct().order_by(Meet.year.asc())).scalars().all()
        return jsonify({'years': [int(y) for y in years]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/events', methods=['GET'])
def history_events():
    try:
        init_db()
        year = request.args.get('year', type=int)
        gender = request.args.get('gender')
        include_relay = request.args.get('include_relay', default='false').lower() == 'true'
        with get_session() as s:
            if year is not None:
                q = (
                    select(Event.id, Event.gender, Event.distance, Event.stroke, Event.is_relay)
                    .join(ResultIndividual, ResultIndividual.event_id == Event.id)
                    .join(Meet, Meet.id == ResultIndividual.meet_id)
                    .where(Meet.year == year)
                )
            else:
                q = select(Event.id, Event.gender, Event.distance, Event.stroke, Event.is_relay)
            if gender:
                q = q.where(Event.gender == gender)
            if not include_relay:
                q = q.where(Event.is_relay == False)  # noqa: E712
            q = q.distinct().order_by(Event.gender.asc(), Event.distance.asc(), Event.stroke.asc())
            rows = s.execute(q).all()
            events = []
            for eid, g, d, st, rel in rows:
                events.append({
                    'id': int(eid),
                    'gender': g,
                    'distance': int(d),
                    'stroke': st,
                    'is_relay': bool(rel),
                    'name': f"{g} {int(d)} {st}"
                })
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/event-years', methods=['GET'])
def history_event_years():
    try:
        init_db()
        event_id = request.args.get('event_id', type=int)
        gender = request.args.get('gender')
        distance = request.args.get('distance', type=int)
        stroke = request.args.get('stroke')
        if event_id is None and not (gender and distance is not None and stroke):
            return jsonify({'error': 'Missing event specifier'}), 400
        with get_session() as s:
            if event_id is not None:
                ev = s.get(Event, event_id)
                if ev is None:
                    return jsonify({'error': 'Event not found'}), 404
                ev_id = ev.id
            else:
                ev = s.execute(
                    select(Event).where(
                        Event.gender == gender,
                        Event.distance == distance,
                        Event.stroke == stroke,
                    )
                ).scalar_one_or_none()
                if ev is None:
                    return jsonify({'error': 'Event not found'}), 404
                ev_id = ev.id
            years = s.execute(
                select(Meet.year).distinct()
                .join(ResultIndividual, ResultIndividual.meet_id == Meet.id)
                .where(ResultIndividual.event_id == ev_id)
                .order_by(Meet.year.asc())
            ).scalars().all()
        return jsonify({'event_id': int(ev_id), 'years': [int(y) for y in years]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/results', methods=['GET'])
def history_results():
    try:
        init_db()
        year = request.args.get('year', type=int)
        event_id = request.args.get('event_id', type=int)
        gender = request.args.get('gender')
        distance = request.args.get('distance', type=int)
        stroke = request.args.get('stroke')
        if year is None:
            return jsonify({'error': 'Missing year'}), 400
        with get_session() as s:
            ev = None
            if event_id is not None:
                ev = s.get(Event, event_id)
                if ev is None:
                    return jsonify({'error': 'Event not found'}), 404
            elif gender and distance is not None and stroke:
                ev = s.execute(
                    select(Event).where(
                        Event.gender == gender,
                        Event.distance == distance,
                        Event.stroke == stroke,
                    )
                ).scalar_one_or_none()
                if ev is None:
                    return jsonify({'error': 'Event not found'}), 404
            else:
                return jsonify({'error': 'Missing event specifier'}), 400

            base_q = (
                select(
                    ResultIndividual.id,
                    ResultIndividual.athlete_name,
                    ResultIndividual.school_name,
                    ResultIndividual.round,
                    ResultIndividual.place,
                    ResultIndividual.time_seconds,
                    ResultIndividual.time_raw,
                )
                .where(
                    ResultIndividual.event_id == ev.id,
                    ResultIndividual.meet_id == select(Meet.id).where(Meet.year == year).scalar_subquery(),
                )
            )
            finals_q = base_q.where(ResultIndividual.round == 'Final').order_by(ResultIndividual.time_seconds.asc())
            prelims_q = base_q.where(ResultIndividual.round == 'Prelim').order_by(ResultIndividual.time_seconds.asc())
            with get_session() as s2:
                finals = [
                    {
                        'id': int(r.id),
                        'athlete_name': r.athlete_name,
                        'school_name': r.school_name,
                        'round': r.round,
                        'place': int(r.place) if r.place is not None else None,
                        'time_seconds': float(r.time_seconds) if r.time_seconds is not None else None,
                        'time_raw': r.time_raw,
                    }
                    for r in s2.execute(finals_q).all()
                ]
                prelims = [
                    {
                        'id': int(r.id),
                        'athlete_name': r.athlete_name,
                        'school_name': r.school_name,
                        'round': r.round,
                        'place': int(r.place) if r.place is not None else None,
                        'time_seconds': float(r.time_seconds) if r.time_seconds is not None else None,
                        'time_raw': r.time_raw,
                    }
                    for r in s2.execute(prelims_q).all()
                ]
            event_info = {
                'id': int(ev.id),
                'gender': ev.gender,
                'distance': int(ev.distance),
                'stroke': ev.stroke,
                'is_relay': bool(ev.is_relay),
                'name': f"{ev.gender} {int(ev.distance)} {ev.stroke}",
            }
            return jsonify({'year': int(year), 'event': event_info, 'finals': finals, 'prelims': prelims})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/athlete', methods=['GET'])
def history_athlete():
    try:
        init_db()
        q = request.args.get('q', default='', type=str).strip()
        if not q:
            return jsonify({'results': []})
        with get_session() as s:
            # Find matching athletes (case-insensitive contains)
            ath_ids = s.execute(
                select(Athlete.id).where(func.lower(Athlete.name).like(f"%{q.lower()}%"))
            ).scalars().all()
            if not ath_ids:
                return jsonify({'results': []})
            rows = s.execute(
                select(
                    Athlete.name.label('athlete_name'),
                    Meet.year.label('year'),
                    Event.gender, Event.distance, Event.stroke,
                    ResultIndividual.round, ResultIndividual.place,
                    ResultIndividual.time_seconds, ResultIndividual.time_raw,
                    ResultIndividual.school_name
                )
                .join(ResultIndividual, ResultIndividual.athlete_id == Athlete.id)
                .join(Meet, Meet.id == ResultIndividual.meet_id)
                .join(Event, Event.id == ResultIndividual.event_id)
                .where(Athlete.id.in_(ath_ids))
                .order_by(Meet.year.asc(), Event.gender.asc(), Event.distance.asc(), Event.stroke.asc(), ResultIndividual.round.desc(), ResultIndividual.time_seconds.asc())
            ).all()
            results = [
                {
                    'athlete_name': r.athlete_name,
                    'year': int(r.year) if r.year is not None else None,
                    'gender': r.gender,
                    'distance': int(r.distance) if r.distance is not None else None,
                    'stroke': r.stroke,
                    'round': r.round,
                    'place': int(r.place) if r.place is not None else None,
                    'time_seconds': float(r.time_seconds) if r.time_seconds is not None else None,
                    'time_raw': r.time_raw,
                    'school_name': r.school_name,
                }
                for r in rows
            ]
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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