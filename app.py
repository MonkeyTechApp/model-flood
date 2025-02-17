
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Step 1: Define Your Prediction Function
def predict_for_date(target_date, forecast_data, static_features):
    end_date = pd.to_datetime(target_date) - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=29)

    sequence = forecast_data[
        (forecast_data['date'] >= start_date) & 
        (forecast_data['date'] <= end_date)
    ].sort_values('date')

    sequence = sequence[['precip', 'temp', 'humid', 'soil_m', 'vegetation']].values

    # Ensure 30 time steps
    if len(sequence) < 30:
        padding = np.zeros((30 - len(sequence), sequence.shape[1]))
        sequence = np.vstack([sequence, padding])

    # Ensure 26 features
    num_missing_features = 26 - sequence.shape[1]
    if num_missing_features > 0:
        placeholder = np.zeros((sequence.shape[0], num_missing_features))
        sequence = np.hstack([sequence, placeholder])

    static_input = np.array([
        static_features['elevation'],
        static_features['slope'],
        *static_features['soil_type'][:3]
    ], dtype='float32')

    input_data = {
        'temporal': sequence.astype('float32'),
        'static': static_input
    }

    model = load_model('flood_drought_model.h5', compile=False)

    flood_prob, drought_prob = model.predict([
        np.expand_dims(input_data['temporal'], axis=0),
        np.expand_dims(input_data['static'], axis=0)
    ])

    return {
        'target_date': target_date,
        'flood_risk': float(flood_prob[0][0]),
        'drought_risk': float(drought_prob[0][0]),
        'validity_window': {
            'flood': f"{(end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {end_date + pd.Timedelta(days=7)}",
            'drought': f"{(end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {end_date + pd.Timedelta(days=30)}"
        }
    }

# Step 2: Create API Routes
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    # data = request.get_json()
    # target_date = data.get("target_date")
    # forecast_data = pd.DataFrame(data.get("forecast_data"))
    # static_features = data.get("static_features")

    # if not target_date or forecast_data.empty or not static_features:
    #     return jsonify({"error": "Invalid input data"}), 400
    
    target_date='2024-03-25',
    # Corrected code with consistent 40-day forecast
    forecast_data = pd.DataFrame({
        'date': pd.date_range(start='2024-03-01', periods=30),  # 40 days
        'precip': np.random.uniform(0, 20, 40),
        'temp': np.random.uniform(15, 35, 40),
        'humid': np.random.uniform(30, 80, 40),
        'soil_m': np.random.uniform(50, 120, 40),
        'vegetation': np.random.uniform(0.3, 0.7, 40)
    })
    # Static features for a location
    static_features = {
        'elevation': 245.6,
        'slope': 3.8,
        'soil_type': [0, 0, 1, 0, 0]  # Loam soil
    }

    try:
        result = predict_for_date(target_date, forecast_data, static_features)
    except Exception as e :
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    return jsonify(result)

# Step 3: Assign Flask app to Passenger
application = app