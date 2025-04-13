from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
import os
import sys
import time
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and scaler
MODEL = None
SCALER = None

def load_models():
    """Load model and scaler once at startup"""
    global MODEL, SCALER
    
    model_path = 'model.h5'
    scaler_path = 'scaler.joblib'
    
    try:
        logger.info(f"Loading model from {model_path}")
        MODEL = load_model(model_path)
        logger.info(f"Model loaded successfully with input shape: {MODEL.input_shape}")
        
        logger.info(f"Loading scaler from {scaler_path}")
        SCALER = joblib.load(scaler_path)
        logger.info("Scaler loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_and_preprocess_json_data(weather_json, hydro_json, geo_json):
    """Load and preprocess new data for prediction from JSON inputs"""
    try:
        # Convert JSON data to DataFrames
        weather = pd.DataFrame(weather_json)
        hydro = pd.DataFrame(hydro_json)
        geo = pd.DataFrame(geo_json)
        
        logger.info(f"Received data: {len(weather)} weather records, {len(hydro)} hydro records, {len(geo)} geo records")
        
        # Convert date fields to datetime
        weather['date'] = pd.to_datetime(weather['date'])
        hydro['measurement_date'] = pd.to_datetime(hydro['measurement_date'])
        
        # Standardize column names and merge
        hydro = hydro.rename(columns={'measurement_date': 'date'})
        merged = pd.merge(weather, hydro, on=['date'], how='inner')
        merged = pd.merge(merged, geo, on='location', how='left')
        
        logger.info(f"After merging: {len(merged)} records")
        
        # Sort by location and date
        merged = merged.sort_values(['location', 'date'])
        
        # Extract metadata
        metadata = merged[['date', 'location']].copy()
        
        # Preprocess data (same steps as during training)
        merged['reservoir_level'] = merged.groupby('location')['reservoir_level'].ffill()
        merged['precipitation'] = merged['precipitation'].fillna(0)
        merged['temp_avg'] = (merged['temp_max'] + merged['temp_min']) / 2
        merged = merged.dropna(subset=['reservoir_level', 'river_flow', 'groundwater_level'])
        
        # Create features
        merged['day_of_year'] = merged['date'].dt.dayofyear
        merged['month'] = merged['date'].dt.month
        
        for window in [7, 30, 90]:
            merged[f'reservoir_{window}d_mean'] = merged.groupby('location')['reservoir_level'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            merged[f'precip_{window}d_sum'] = merged.groupby('location')['precipitation'].transform(
                lambda x: x.rolling(window, min_periods=1).sum())
            merged[f'river_flow_{window}d_mean'] = merged.groupby('location')['river_flow'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            merged[f'groundwater_{window}d_mean'] = merged.groupby('location')['groundwater_level'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
        
        # Other features
        merged['reservoir_7d_change'] = merged.groupby('location')['reservoir_level'].transform(
            lambda x: x.pct_change(7).fillna(0))
        merged['groundwater_7d_change'] = merged.groupby('location')['groundwater_level'].transform(
            lambda x: x.pct_change(7).fillna(0))
        merged['flow_3d_avg'] = merged.groupby('location')['river_flow'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        
        # Avoid division by zero in temperature diff calculation
        temp_diff = (merged['temp_max'] - merged['temp_min']).replace(0, 0.1)
        merged['evapotranspiration'] = 0.0023 * (merged['temp_avg'] + 17.8) * np.sqrt(temp_diff)
        merged['water_balance'] = merged[f'precip_30d_sum'] - merged['evapotranspiration']
        
        # Cyclical encoding
        merged['month_sin'] = np.sin(2 * np.pi * merged['month']/12)
        merged['month_cos'] = np.cos(2 * np.pi * merged['month']/12)
        merged['day_sin'] = np.sin(2 * np.pi * merged['day_of_year']/365)
        merged['day_cos'] = np.cos(2 * np.pi * merged['day_of_year']/365)
        
        # Handle soil_type to match training (expecting 'soil_type_silty')
        if 'soil_type' in merged.columns:
            merged['soil_type_silty'] = (merged['soil_type'] == 'silty').astype(int)
            merged = merged.drop(columns=['soil_type'])
        elif 'soil_Clay' in merged.columns:
            merged['soil_type_silty'] = 0  # Assuming clay is not silty
            merged = merged.drop(columns=['soil_Clay'])
        
        # Handle Watershed - ensure one-hot encoding exists
        if 'Watershed' in merged.columns:
            if isinstance(merged['Watershed'].iloc[0], (int, float)):
                # Create one-hot encoding if Watershed is numeric
                merged['Watershed_0'] = (merged['Watershed'] == 0).astype(int)
                merged['Watershed_1'] = (merged['Watershed'] == 1).astype(int)
                merged = merged.drop(columns=['Watershed'])
            else:
                # Handle categorical Watershed if needed
                pass
        else:
            # Create Watershed one-hot columns if they don't exist
            merged['Watershed_0'] = 0
            merged['Watershed_1'] = 1  # Or adjust based on your data
        
        # Handle location to match training (expecting 'location_Yagoua')
        if 'location' in merged.columns:
            if pd.api.types.is_object_dtype(merged['location']):
                merged['location_Yagoua'] = (merged['location'] == 'Yagoua').astype(int)
                merged['location'] = pd.Categorical(merged['location']).codes
            else:
                # If already encoded, create the Yagoua dummy
                merged['location_Yagoua'] = 0  # Adjust based on your actual data
        
        # Add required features that might be missing
        for req_feature in ['flood', 'drought']:
            if req_feature not in merged.columns:
                merged[req_feature] = 0
        
        # Remove original temporal columns
        merged = merged.drop(columns=['month', 'day_of_year'], errors='ignore')
        
        # Remove any remaining non-numeric columns except location and date
        for col in merged.columns:
            if col not in ['date', 'location'] and pd.api.types.is_object_dtype(merged[col]):
                merged = merged.drop(columns=[col])
        
        # Get expected features from scaler
        expected_features = SCALER.feature_names_in_
        
        # Ensure all expected features exist
        for feature in expected_features:
            if feature not in merged.columns:
                logger.info(f"Adding missing feature: {feature}")
                merged[feature] = 0
        
        # Select ONLY the expected features (don't add date here)
        final_features = list(expected_features)
        
        # Special case: if 'location' is in expected_features but not in merged,
        # use the encoded version we created
        if 'location' in expected_features and 'location' not in merged.columns:
            # Use the categorical codes we created earlier
            merged['location'] = pd.Categorical(merged['location']).codes
        
        # Now select only the expected features
        merged = merged[final_features]
        
        logger.info(f"Final features count: {len(merged.columns)}")
        
        return merged, metadata
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_sequences(data, metadata, time_steps=30):
    """Create sequences for LSTM prediction, preserving location grouping"""
    unique_locations = data['location'].unique()
    
    X_sequences = []
    sequence_metadata = []
    
    for loc in unique_locations:
        loc_data = data[data['location'] == loc].copy()
        loc_metadata = metadata.iloc[loc_data.index].copy()
        
        # Remove non-feature columns - ensure we're only keeping numeric data
        feature_data = loc_data.select_dtypes(include=[np.number])
        
        # Only create sequences if we have enough data points
        if len(feature_data) >= time_steps:
            for i in range(len(feature_data) - time_steps + 1):
                X_sequences.append(feature_data.iloc[i:i+time_steps].values)
                sequence_metadata.append(loc_metadata.iloc[i+time_steps-1])
    
    if len(X_sequences) > 0:
        logger.info(f"Created sequences with shape: {np.array(X_sequences).shape}")
    
    return np.array(X_sequences), pd.DataFrame(sequence_metadata)

def predict_reservoir_levels(data, metadata, time_steps=30):
    """Make predictions using the LSTM model, handling feature name mismatches"""
    global MODEL, SCALER
    
    # Get the expected feature names from the scaler
    expected_features = SCALER.feature_names_in_
    
    # Create a DataFrame with exactly the expected features
    aligned_data = pd.DataFrame(index=data.index)
    
    # Copy existing expected features
    for feature in expected_features:
        if feature in data.columns:
            # Handle case where feature might be a duplicate
            if isinstance(data[feature], pd.DataFrame):
                aligned_data[feature] = data[feature].iloc[:, 0]  # Take first column if it's a DataFrame
            else:
                aligned_data[feature] = data[feature]
        else:
            logger.info(f"Adding missing feature with zeros: {feature}")
            aligned_data[feature] = 0.0
    
    # Now transform using the properly aligned features
    aligned_data_transformed = SCALER.transform(aligned_data)
    
    # Create a new DataFrame with scaled values
    scaled_data = pd.DataFrame(aligned_data_transformed, columns=expected_features, index=data.index)
    
    # Add back location (needed for sequence creation)
    if 'location' in data.columns:
        scaled_data['location'] = data['location'].values

    # Create sequences
    X_sequences, seq_metadata = create_sequences(scaled_data, metadata, time_steps)
    
    if len(X_sequences) == 0:
        logger.warning("No sequences could be created - not enough data points")
        return pd.DataFrame()
    
    # Verify input shape matches model expectations
    logger.info(f"Model input shape expectation: {MODEL.input_shape}")
    logger.info(f"Actual input shape: {X_sequences.shape}")
    
    if X_sequences.shape[2] != MODEL.input_shape[2]:
        error_msg = f"Feature count mismatch! Model expects {MODEL.input_shape[2]} features, got {X_sequences.shape[2]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Make predictions
    predictions = MODEL.predict(X_sequences)
    
    # Create results DataFrame
    results = seq_metadata.copy()
    
    # Add prediction columns
    target_names = ['reservoir_7d', 'reservoir_14d', 'reservoir_30d', 
                   'reservoir_change_7d', 'reservoir_change_14d', 'reservoir_change_30d',
                   'reservoir_7d_mean', 'reservoir_30d_mean', 'reservoir_7d_change']
    
    if predictions.shape[1] <= len(target_names):
        used_targets = target_names[:predictions.shape[1]]
    else:
        used_targets = [f'target_{i}' for i in range(predictions.shape[1])]
    
    for i, name in enumerate(used_targets):
        results[name] = predictions[:, i]
    
    return results

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    if load_models() == False :
        return jsonify({"status": "healthy", "model_loaded": True}) 
    
    load_models()
    if MODEL is not None and SCALER is not None:
        return jsonify({"status": "healthy", "model_loaded": True})
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint that accepts JSON data"""
    start_time = time.time()
    
    try:
        # Check if models are loaded
        if MODEL is None or SCALER is None:
            if not load_models():
                return jsonify({
                    "status": "error",
                    "message": "Failed to load model or scaler",
                    "predictions": []
                }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided. Please send JSON data with 'weather', 'hydro', and 'geo' fields.",
                "predictions": []
            }), 400
        
        # Extract data components
        weather_json = data.get('weather')
        hydro_json = data.get('hydro')
        geo_json = data.get('geo')
        
        # Validate input data
        if not weather_json or not hydro_json or not geo_json:
            return jsonify({
                "status": "error",
                "message": "Missing required data fields: 'weather', 'hydro', and 'geo' are all required",
                "predictions": []
            }), 400
        
        # Get optional parameters
        time_steps = int(data.get('time_steps', 30))
        
        # Log request summary
        logger.info(f"Prediction request with {len(weather_json)} weather records, " + 
                    f"{len(hydro_json)} hydro records, {len(geo_json)} geo records")
        
        # Preprocess data
        data, metadata = load_and_preprocess_json_data(weather_json, hydro_json, geo_json)
        
        # Make predictions
        results = predict_reservoir_levels(data, metadata, time_steps)
        
        if not results.empty:
            # Convert datetime columns to string for JSON serialization
            if 'date' in results.columns:
                results['date'] = results['date'].dt.strftime('%Y-%m-%d')
                
            # Convert to JSON
            results_json = results.to_dict(orient='records')
            
            elapsed_time = time.time() - start_time
            logger.info(f"Prediction completed successfully in {elapsed_time:.2f} seconds with {len(results_json)} predictions")
            
            return jsonify({
                "status": "success",
                "predictions": results_json,
                "num_predictions": len(results_json),
                "processing_time_seconds": elapsed_time
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No predictions could be made. Check if you have enough data points.",
                "predictions": []
            }), 400
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "message": str(e),
            "predictions": [],
            "processing_time_seconds": elapsed_time
        }), 500

@app.route('/predict-custom-horizons', methods=['POST'])
def predict_custom_horizons():
    """
    Advanced endpoint that allows specifying custom forecast horizons
    This requires retraining or special handling for the model
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract data components and horizons
        weather_json = data.get('weather')
        hydro_json = data.get('hydro')
        geo_json = data.get('geo')
        horizons = data.get('horizons', [7, 14, 30])
        
        # Validate input data
        if not weather_json or not hydro_json or not geo_json:
            return jsonify({
                "status": "error",
                "message": "Missing required data fields"
            }), 400
        
        # TODO: Implement custom horizon prediction logic
        # This would require model adaptation or multiple model support
        
        return jsonify({
            "status": "error",
            "message": "Custom horizons not yet implemented. Currently supported horizons: 7, 14, 30 days"
        }), 501
        
    except Exception as e:
        logger.error(f"Error in custom prediction: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# if __name__ == '__main__':
#     # Load models at startup
#     load_models()
    
#     # Get port from environment variable or use default
#     port = int(os.environ.get('PORT', 5000))
    
#     # Run app
#     app.run(host='0.0.0.0', port=port, debug=False)

# Step 3: Assign Flask app to Passenger
application = app