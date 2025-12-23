from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from model import DeliveryTimePredictor, _create_nested_defaultdict

app = Flask(__name__)
CORS(app)

# Global variables to simulate a database and hold the model
model_components = None
customer_db = {} 

def load_data_and_model():
    """
    Load the trained model and build a simple in-memory database 
    from the history file to look up customer names/addresses.
    """
    global model_components, customer_db
    
    # 1. Load Model
    MODEL_PATH = 'models/delivery_time_predictor.pkl'
    if os.path.exists(MODEL_PATH):
        try:
            model_components = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            model_components = None
    else:
        print("⚠️ No trained model found. Please train the model first.")

    # 2. Load Customer Database (Simulation from CSV)
    DATA_PATH = 'data/delivery_history.csv'
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Create a lookup dictionary: { 'CUST_0001': {'name': 'John Doe', 'postal_code': '10001'} }
            # We take the most recent record for each customer to get their latest details
            if 'delivery_date' in df.columns:
                df = df.sort_values('delivery_date')
            
            df_unique = df.groupby('customer_id').tail(1)
            
            customer_db = {}
            for _, row in df_unique.iterrows():
                customer_db[row['customer_id']] = {
                    'name': row.get('customer_name', 'Unknown'),
                    'postal_code': str(row.get('postal_code', ''))
                }
            print(f"✅ Loaded details for {len(customer_db)} customers from history.")
        except Exception as e:
            print(f"⚠️ Error loading customer history: {e}")
    else:
        print("⚠️ No data file found. Customer lookup will not work.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_time_slot():
    """
    Predict preferred delivery time slot with auto-lookup for customer details
    """
    try:
        if model_components is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        data = request.json
        customer_id = data.get('customer_id')
        
        # 1. Auto-lookup customer details
        cust_info = customer_db.get(customer_id, {})
        
        # Use DB name if available, otherwise default
        customer_name = cust_info.get('name', 'New Customer')
        
        # Logic: If postal code not provided in request, use one from DB
        postal_code = data.get('postal_code')
        if not postal_code or postal_code == "":
            postal_code = cust_info.get('postal_code', '10001') # Default fallback
            
        delivery_date = data.get('delivery_date', datetime.now().strftime('%Y-%m-%d'))
        order_type = data.get('order_type', 'standard')
        order_value = float(data.get('order_value', 50.0))
        
        # 2. Make Prediction
        predictor = DeliveryTimePredictor()
        predictor.model_components = model_components
        
        predictions = predictor.predict_time_slots(
            customer_id=customer_id,
            customer_name=customer_name,
            delivery_date=delivery_date,
            postal_code=str(postal_code),
            order_type=order_type,
            order_value=order_value,
            top_n=3
        )
        
        return jsonify({
            'success': True,
            'customer_id': customer_id,
            'customer_name': customer_name, # Return name for UI display
            'used_postal_code': postal_code,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/api/feedback', methods=['POST'])
def record_feedback():
    """
    Record whether the predicted time slot was accepted
    """
    try:
        data = request.json
        
        # Save feedback for model retraining
        feedback_data = {
            'customer_id': data.get('customer_id'),
            'predicted_slot': data.get('predicted_slot'),
            'actual_slot': data.get('actual_slot'),
            'delivery_date': data.get('delivery_date'),
            'was_successful': data.get('was_successful', True),
            'timestamp': datetime.now().isoformat()
        }
        
        # Append to feedback log
        feedback_file = 'data/feedback_log.csv'
        df_feedback = pd.DataFrame([feedback_data])
        
        if os.path.exists(feedback_file):
            df_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            df_feedback.to_csv(feedback_file, mode='w', header=True, index=False)
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Trigger model training and reload data
    """
    try:
        predictor = DeliveryTimePredictor()
        metrics = predictor.train('data/delivery_history.csv')
        
        # Reload the model and DB after training so new data is active immediately
        load_data_and_model()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get model performance statistics
    """
    try:
        if model_components is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        metrics = model_components.get('metrics', {})
        
        stats = {
            'total_customers': len(customer_db), # Now reflects real DB count
            'model_type': metrics.get('model_type', 'Unknown'),
            'accuracy': metrics.get('accuracy', 0),
            'top_3_accuracy': metrics.get('top_3_accuracy', 0)
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Initialize logic on startup
    load_data_and_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)