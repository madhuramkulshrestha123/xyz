import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load the dataset
csv_path = os.path.join(os.path.dirname(__file__), 'test_sample - Sheet1.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

def train_sklearn_model(single_row_data):
    # Prepare the data
    X = single_row_data[['Day_0', 'Day_1', 'Day_2']].values.reshape(1, -1)
    y = single_row_data['Day_3']

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(np.array([y]).reshape(-1, 1)).flatten()

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_scaled, y_scaled)

    return model, scaler_X, scaler_y

def forecast_next_days(model, scaler_X, scaler_y, initial_values, num_days=10):
    forecasts = []
    last_data = initial_values[['Day_0', 'Day_1', 'Day_2']].values.reshape(1, -1)

    for _ in range(num_days):
        last_data_scaled = scaler_X.transform(last_data)
        next_price_scaled = model.predict(last_data_scaled)[0]
        next_price = scaler_y.inverse_transform([[next_price_scaled]])[0][0]
        forecasts.append(next_price)

        # Shift the window for the next prediction
        last_data = np.array([last_data[0][1], last_data[0][2], next_price]).reshape(1, -1)

    return forecasts

def evaluate_deal(avg_predicted_price, discounted_price):
    if avg_predicted_price > discounted_price:
        return "This is the best Deal you're getting Today, Buy Now!!"
    else:
        return "I think you should Wait for some more time."

@app.route('/')
def index():
    return "Price Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    product_name = data.get('product_name')
    
    if not product_name:
        return jsonify({"error": "Invalid input"}), 400

    product_data = df[df['product_name'] == product_name]

    if product_data.empty:
        return jsonify({"error": "Product data not found"}), 400

    # Extract MRP and discounted price
    MRP = product_data['MRP'].iloc[0]
    discounted_price = product_data['discounted_price'].iloc[0]

    # Train the model on the single row of data
    sklearn_model, scaler_X, scaler_y = train_sklearn_model(product_data.iloc[0])

    # Forecast the next 10 days using the trained model
    future_prices = forecast_next_days(sklearn_model, scaler_X, scaler_y, product_data.iloc[0], num_days=10)
    avg_predicted_price = np.mean(future_prices)
    
    # Evaluate if it's a good deal
    advice_message = evaluate_deal(avg_predicted_price, discounted_price)

    # Retrieve product URL from the dataset
    product_url = product_data['product_url'].iloc[0] if 'product_url' in product_data.columns else None
    
    return jsonify({
    "product_name": product_name,
    "product_url": product_url,
    "MRP": int(MRP) if isinstance(MRP, (np.int64, np.int32)) else MRP,
    "discounted_price": int(discounted_price) if isinstance(discounted_price, (np.int64, np.int32)) else discounted_price,
    "average_predicted_price": int(avg_predicted_price) if isinstance(avg_predicted_price, (np.int64, np.int32)) else avg_predicted_price,
    "advice_message": advice_message
})


if __name__ == '__main__':
    app.run(debug=True)
