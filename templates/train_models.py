import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
import joblib

# Example dataset for recommendations
X_recommendation = np.random.rand(100, 2)  # Features: [budget, number_of_interests]
y_recommendation = np.random.randint(0, 10, size=(100,))  # Labels: destination indices

# Train recommendation model
recommendation_model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # Assuming 10 destinations
])
recommendation_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
recommendation_model.fit(X_recommendation, y_recommendation, epochs=10)
recommendation_model.save('ml_models/recommendation_model.h5')

# Example dataset for price prediction
X_price = np.random.rand(100, 1)  # Feature: [days_to_departure]
y_price = np.random.rand(100) * 1000  # Labels: flight prices

# Train price prediction model
price_prediction_model = RandomForestRegressor()
price_prediction_model.fit(X_price, y_price)
joblib.dump(price_prediction_model, 'ml_models/price_prediction_model.pkl')