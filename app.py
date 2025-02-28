from flask import Flask, render_template, request, jsonify
import sqlite3
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load ML models
recommendation_model = load_model('ml_models/recommendation_model.h5')
price_prediction_model = joblib.load('ml_models/price_prediction_model.pkl')

# Database connection
def get_db_connection():
    conn = sqlite3.connect('travel.db')
    conn.row_factory = sqlite3.Row
    return conn

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Personalized Recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_preferences = request.json  # e.g., {'budget': 1000, 'interests': ['beach', 'history']}
    
    # Preprocess input for the ML model
    budget = user_preferences.get('budget', 1000)
    interests = user_preferences.get('interests', [])
    input_data = np.array([[budget, len(interests)]])  # Example preprocessing
    
    # Predict destinations
    predictions = recommendation_model.predict(input_data)
    recommended_destinations = ["Paris", "Bali", "New York"]  # Replace with actual predictions
    
    return jsonify({'destinations': recommended_destinations})

# Flight Price Prediction
@app.route('/predict_price', methods=['POST'])
def predict_price():
    flight_details = request.json  # e.g., {'origin': 'NYC', 'destination': 'LAX', 'days_to_departure': 30}
    
    # Preprocess input for the ML model
    origin = flight_details.get('origin', 'NYC')
    destination = flight_details.get('destination', 'LAX')
    days_to_departure = flight_details.get('days_to_departure', 30)
    input_data = [[days_to_departure]]  # Example preprocessing
    
    # Predict price
    predicted_price = price_prediction_model.predict(input_data)[0]
    
    return jsonify({'predicted_price': round(predicted_price, 2)})

# Itinerary Planning
@app.route('/plan_itinerary', methods=['POST'])
def plan_itinerary():
    trip_details = request.json  # e.g., {'destination': 'Paris', 'duration': 5}
    
    destination = trip_details.get('destination', 'Paris')
    duration = trip_details.get('duration', 5)
    
    # Generate a simple itinerary (replace with ML-based optimization)
    itinerary = {
        'Day 1': 'Arrival and city tour',
        'Day 2': 'Visit Eiffel Tower',
        'Day 3': 'Louvre Museum',
        'Day 4': 'Seine River Cruise',
        'Day 5': 'Departure'
    }
    
    return jsonify({'itinerary': itinerary})

if __name__ == '__main__':
    app.run(debug=True)
    