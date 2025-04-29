# ---------- Backend (app.py) ----------
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_stress_model.pkl')

# Feature columns in correct order
FEATURE_COLUMNS = [
    'body_temperature', 
    'heart_rate',
    'respiration_rate',
    'activity_level',
    'feeding_frequency',
    'environment_temp',
    'vocalization_freq'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'body_temperature': float(request.form['body_temp']),
            'heart_rate': int(request.form['heart_rate']),
            'respiration_rate': int(request.form['respiration_rate']),
            'activity_level': float(request.form['activity_level']),
            'feeding_frequency': int(request.form['feeding_freq']),
            'environment_temp': float(request.form['env_temp']),
            'vocalization_freq': int(request.form['vocalization_freq'])
        }
        
        # Create DataFrame and ensure correct feature order
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Format results
        result = {
            'status': 'Stressed' if prediction == 1 else 'Normal',
            'probability': f"{probability:.2%}",
            'risk_level': 'high' if probability > 0.7 else 
                          'medium' if probability > 0.4 else 'low'
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'Stressed' if prediction == 1 else 'Normal'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)