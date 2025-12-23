import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from collections import defaultdict
import warnings
import os
import sys

# Import advanced boosting libraries
import xgboost as xgb
import lightgbm as lgb

# --- 1. Aggressive Warning Suppression (Clean Logs) ---
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def _create_nested_defaultdict():
    """Helper function for pickling defaultdicts."""
    return defaultdict(int)

class DeliveryTimePredictor:
    """
    Hybrid recommendation system for predicting optimal delivery time slots.
    Supports GPU-accelerated training with XGBoost and LightGBM.
    """
    
    def __init__(self):
        self.model_components = {
            'main_model': None,
            'customer_encoder': LabelEncoder(),
            'time_slot_encoder': LabelEncoder(),
            'postal_code_encoder': LabelEncoder(),
            'order_type_encoder': LabelEncoder(),
            'customer_preferences': defaultdict(_create_nested_defaultdict),
            'temporal_patterns': defaultdict(_create_nested_defaultdict),
            'time_slot_success_rate': defaultdict(float),
            'metrics': {}
        }
        
        self.standard_time_slots = [
            '09:00-12:00', '12:00-15:00', '15:00-18:00', 
            '18:00-21:00', '21:00-23:00'
        ]
    
    def extract_temporal_features(self, date_str):
        date = pd.to_datetime(date_str)
        return {
            'day_of_week': date.dayofweek,
            'is_weekend': 1 if date.dayofweek >= 5 else 0,
            'month': date.month,
            'day_of_month': date.day
        }
    
    def build_customer_profiles(self, df):
        customer_preferences = defaultdict(_create_nested_defaultdict)
        temporal_patterns = defaultdict(_create_nested_defaultdict)
        time_slot_success = defaultdict(list)
        
        for _, row in df.iterrows():
            customer_id = row['customer_id']
            time_slot = row['time_slot']
            was_successful = row.get('was_successful', 1)
            
            customer_preferences[customer_id][time_slot] += 1
            temporal_features = self.extract_temporal_features(row['delivery_date'])
            dow = temporal_features['day_of_week']
            temporal_patterns[dow][time_slot] += 1
            time_slot_success[time_slot].append(was_successful)
        
        time_slot_success_rate = {
            slot: np.mean(successes) if successes else 0.5
            for slot, successes in time_slot_success.items()
        }
        return customer_preferences, temporal_patterns, time_slot_success_rate
    
    def prepare_features(self, df):
        features = []
        for _, row in df.iterrows():
            temporal_features = self.extract_temporal_features(row['delivery_date'])
            features.append({
                'customer_id': row['customer_id'],
                'day_of_week': temporal_features['day_of_week'],
                'is_weekend': temporal_features['is_weekend'],
                'month': temporal_features['month'],
                'day_of_month': temporal_features['day_of_month'],
                'postal_code': row.get('postal_code', 'unknown'),
                'order_type': row.get('order_type', 'standard'),
                'order_value': row.get('order_value', 50.0)
            })
        return pd.DataFrame(features)
    
    def train(self, data_path):
        print("Loading training data...")
        df = pd.read_csv(data_path)
        
        if 'postal_code' not in df.columns: df['postal_code'] = 'unknown'
        df['postal_code'] = df['postal_code'].astype(str)
        if 'order_type' not in df.columns: df['order_type'] = 'standard'
        if 'order_value' not in df.columns: df['order_value'] = 50.0
        
        print(f"Loaded {len(df)} delivery records")
        print("Building profiles and features...")
        
        customer_prefs, temporal_patterns, slot_success = self.build_customer_profiles(df)
        self.model_components['customer_preferences'] = customer_prefs
        self.model_components['temporal_patterns'] = temporal_patterns
        self.model_components['time_slot_success_rate'] = slot_success
        
        X_df = self.prepare_features(df)
        y = df['time_slot'].values
        
        X_df['customer_id_encoded'] = self.model_components['customer_encoder'].fit_transform(X_df['customer_id'])
        X_df['postal_code_encoded'] = self.model_components['postal_code_encoder'].fit_transform(X_df['postal_code'])
        X_df['order_type_encoded'] = self.model_components['order_type_encoder'].fit_transform(X_df['order_type'])
        y_encoded = self.model_components['time_slot_encoder'].fit_transform(y)
        
        feature_columns = [
            'customer_id_encoded', 'day_of_week', 'is_weekend', 
            'month', 'day_of_month', 'postal_code_encoded',
            'order_type_encoded', 'order_value'
        ]
        
        X = X_df[feature_columns].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print("\nStarting GPU Model Tournament...")
        
        models = {
            'RandomForest (CPU)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'XGBoost (GPU)': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, 
                device='cuda', tree_method='hist', random_state=42, verbosity=0
            ),
            'LightGBM (GPU)': lgb.LGBMClassifier(
                n_estimators=100, max_depth=10, learning_rate=0.1, 
                device='gpu', random_state=42, verbose=-1, silent=True
            )
        }
        
        best_name = None
        best_score = 0
        best_model_obj = None
        
        for name, model in models.items():
            try:
                # Redirect stderr to suppress C++ warnings from GPU libraries
                stderr_old = sys.stderr
                sys.stderr = open(os.devnull, 'w')
                
                model.fit(X_train, y_train)
                
                # Restore stderr
                sys.stderr = stderr_old
                
                # Calculate metrics
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                # Calculate Top-3 immediately for visibility
                y_pred_proba = model.predict_proba(X_test)
                top_3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]
                top_3_acc = np.mean([y_test[i] in top_3_preds[i] for i in range(len(y_test))])
                
                print(f"   [{name}]")
                print(f"      Exact Match:  {acc:.2%}")
                print(f"      Top-3 Accuracy: {top_3_acc:.2%}")
                
                if acc > best_score:
                    best_score = acc
                    best_name = name
                    best_model_obj = model
            except Exception as e:
                sys.stderr = sys.__stderr__ # Restore just in case
                print(f"   [Failed] {name}: {str(e)}")
        
        if best_model_obj is None: raise RuntimeError("All models failed.")
            
        print(f"\nWinner: {best_name}")
        self.model_components['main_model'] = best_model_obj
        
        # Final Metrics Calculation
        y_pred_proba = best_model_obj.predict_proba(X_test)
        top_3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]
        top_3_acc = np.mean([y_test[i] in top_3_preds[i] for i in range(len(y_test))])
        
        self.model_components['metrics'] = {
            'model_type': best_name,
            'accuracy': float(best_score),
            'top_3_accuracy': float(top_3_acc)
        }
        
        print("-" * 40)
        print("PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Best Model:         {best_name}")
        print(f"Top-1 Accuracy:     {best_score:.2%} (Exact Match)")
        print(f"Top-3 Accuracy:     {top_3_acc:.2%} (Recommendation Hit)")
        print(f"Recommendation Lift: +{top_3_acc - best_score:.2%}")
        print("-" * 40)
        
        joblib.dump(self.model_components, 'models/delivery_time_predictor.pkl')
        print("Model saved successfully!")
        return self.model_components['metrics']
    
    def predict_time_slots(self, customer_id, customer_name, delivery_date, 
                          postal_code='unknown', order_type='standard', 
                          order_value=50.0, top_n=3):
        predictions = []
        temporal_features = self.extract_temporal_features(delivery_date)
        
        customer_prefs = self.model_components['customer_preferences'].get(customer_id, {})
        dow = temporal_features['day_of_week']
        temporal_prefs = self.model_components['temporal_patterns'].get(dow, {})
        
        if self.model_components['main_model'] is not None:
            try:
                c_enc = self._safe_encode(self.model_components['customer_encoder'], customer_id)
                p_enc = self._safe_encode(self.model_components['postal_code_encoder'], str(postal_code))
                o_enc = self._safe_encode(self.model_components['order_type_encoder'], order_type)
                
                feature_vector = np.array([[
                    c_enc, temporal_features['day_of_week'], temporal_features['is_weekend'], 
                    temporal_features['month'], temporal_features['day_of_month'],
                    p_enc, o_enc, order_value
                ]])
                
                all_probs = self.model_components['main_model'].predict_proba(feature_vector)[0]
                
                for time_slot in self.standard_time_slots:
                    try:
                        slot_idx = self.model_components['time_slot_encoder'].transform([time_slot])[0]
                        ml_score = all_probs[slot_idx] if slot_idx < len(all_probs) else 0.0
                    except: ml_score = 0.0
                    
                    total_cust = sum(customer_prefs.values())
                    cf_score = customer_prefs.get(time_slot, 0) / max(total_cust, 1)
                    
                    total_temp = sum(temporal_prefs.values())
                    temporal_score = temporal_prefs.get(time_slot, 0) / max(total_temp, 1)
                    
                    success_rate = self.model_components['time_slot_success_rate'].get(time_slot, 0.5)
                    
                    if total_cust > 0:
                        final_score = 0.4 * ml_score + 0.4 * cf_score + 0.1 * temporal_score + 0.1 * success_rate
                    else:
                        final_score = 0.5 * ml_score + 0.3 * temporal_score + 0.2 * success_rate
                    
                    predictions.append({
                        'time_slot': time_slot,
                        'confidence': float(final_score),
                        'reason': self._get_reason(customer_prefs, temporal_prefs, time_slot)
                    })
            except Exception as e:
                predictions = self._fallback_prediction(temporal_features, temporal_prefs)
        else:
            predictions = self._fallback_prediction(temporal_features, temporal_prefs)
        
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:top_n]

    def _safe_encode(self, encoder, value):
        try: return encoder.transform([value])[0]
        except: return 0 
    
    def _get_reason(self, customer_prefs, temporal_prefs, time_slot):
        if customer_prefs and time_slot in customer_prefs:
            return f"Based on your {customer_prefs[time_slot]} previous deliveries"
        elif temporal_prefs and time_slot in temporal_prefs:
            return "Popular time for this day"
        else: return "Suggested based on order details"

    def _fallback_prediction(self, temporal_features, temporal_prefs):
        predictions = []
        if temporal_features['is_weekend']:
            preferred_slots = ['12:00-15:00', '15:00-18:00', '09:00-12:00']
        else:
            preferred_slots = ['18:00-21:00', '15:00-18:00', '12:00-15:00']
        for i, slot in enumerate(preferred_slots):
            predictions.append({'time_slot': slot, 'confidence': 0.8-(i*0.1), 'reason': 'General Recommendation'})
        return predictions

if __name__ == '__main__':
    predictor = DeliveryTimePredictor()
    predictor.train('data/delivery_history.csv')