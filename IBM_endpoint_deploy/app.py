from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load all saved models and preprocessing objects
try:
    # Load the trained model
    model = pickle.load(open('/Users/akondiathreya/Documents/Development/Projects/APSCHE/IBM_endpoint_deploy/rainfall.pkl', 'rb'))
    
    # Load preprocessing objects
    scaler = pickle.load(open('/Users/akondiathreya/Documents/Development/Projects/APSCHE/IBM_endpoint_deploy/scale.pkl', 'rb'))
    imputer = pickle.load(open('/Users/akondiathreya/Documents/Development/Projects/APSCHE/IBM_endpoint_deploy/impter.pkl', 'rb'))
    categorical_imputer = pickle.load(open('/Users/akondiathreya/Documents/Development/Projects/APSCHE/IBM_endpoint_deploy/cat_impter.pkl', 'rb'))
    label_encoders = pickle.load(open('/Users/akondiathreya/Documents/Development/Projects/APSCHE/IBM_endpoint_deploy/encoder.pkl', 'rb'))
    feature_names = pickle.load(open('/Users/akondiathreya/Documents/Development/Projects/APSCHE/IBM_endpoint_deploy/feature_names.pkl', 'rb'))
    
    print("✓ Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define feature order (as expected by the model)
feature_order = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 
    'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
    'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday'
]

@app.route('/')
def index():
    """Home page - display the input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    """
    try:
        # Get input data from form
        data = request.form.to_dict()
        
        # Create a dictionary with all features
        input_data = {
            'MinTemp': float(data.get('MinTemp', 0)),
            'MaxTemp': float(data.get('MaxTemp', 0)),
            'Rainfall': float(data.get('Rainfall', 0)),
            'WindGustDir': data.get('WindGustDir', 'N'),
            'WindGustSpeed': float(data.get('WindGustSpeed', 0)),
            'WindDir9am': data.get('WindDir9am', 'N'),
            'WindDir3pm': data.get('WindDir3pm', 'N'),
            'WindSpeed9am': float(data.get('WindSpeed9am', 0)),
            'WindSpeed3pm': float(data.get('WindSpeed3pm', 0)),
            'Humidity9am': float(data.get('Humidity9am', 50)),
            'Humidity3pm': float(data.get('Humidity3pm', 50)),
            'Pressure9am': float(data.get('Pressure9am', 1013)),
            'Pressure3pm': float(data.get('Pressure3pm', 1013)),
            'Temp9am': float(data.get('Temp9am', 0)),
            'Temp3pm': float(data.get('Temp3pm', 0)),
            'RainToday': data.get('RainToday', 'No')
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except:
                    # If label not in training set, use mode or default
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        
        # Ensure column order matches training data
        df = df[feature_order]
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
        
        # Decode prediction
        prediction_label = 'Rain' if prediction == 1 else 'No Rain'
        rain_probability = probability[1] * 100
        no_rain_probability = probability[0] * 100
        
        # Return appropriate template based on prediction
        if prediction == 1:  # Rain predicted
            return render_template('chance.html',
                                 probability=f"{rain_probability:.2f}",
                                 no_rain_prob=f"{no_rain_probability:.2f}")
        else:  # No rain predicted
            return render_template('noChance.html',
                                 probability=f"{no_rain_probability:.2f}",
                                 rain_prob=f"{rain_probability:.2f}")
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for JSON predictions
    """
    try:
        data = request.get_json()
        
        # Create input data
        input_data = {
            'MinTemp': float(data.get('MinTemp', 0)),
            'MaxTemp': float(data.get('MaxTemp', 0)),
            'Rainfall': float(data.get('Rainfall', 0)),
            'WindGustDir': data.get('WindGustDir', 'N'),
            'WindGustSpeed': float(data.get('WindGustSpeed', 0)),
            'WindDir9am': data.get('WindDir9am', 'N'),
            'WindDir3pm': data.get('WindDir3pm', 'N'),
            'WindSpeed9am': float(data.get('WindSpeed9am', 0)),
            'WindSpeed3pm': float(data.get('WindSpeed3pm', 0)),
            'Humidity9am': float(data.get('Humidity9am', 50)),
            'Humidity3pm': float(data.get('Humidity3pm', 50)),
            'Pressure9am': float(data.get('Pressure9am', 1013)),
            'Pressure3pm': float(data.get('Pressure3pm', 1013)),
            'Temp9am': float(data.get('Temp9am', 0)),
            'Temp3pm': float(data.get('Temp3pm', 0)),
            'RainToday': data.get('RainToday', 'No')
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except:
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        
        # Ensure column order
        df = df[feature_order]
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
        
        return jsonify({
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'confidence': float(max(probability)) * 100,
            'rain_probability': float(probability[1]) * 100,
            'no_rain_probability': float(probability[0]) * 100
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
