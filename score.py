
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    '''
    Cette fonction est appelée une seule fois au démarrage du service.
    Elle charge le modèle et le scaler en mémoire.
    '''
    global model, scaler, feature_names
    
    try:
        # Charger le modèle
        model_path = Model.get_model_path('fraud-detection-model')
        model = joblib.load(model_path)
        print(f"✅ Modèle chargé: {type(model).__name__}")
        
        # Charger le scaler
        scaler_path = Model.get_model_path('fraud-detection-scaler')
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler chargé: {type(scaler).__name__}")
        
        # Charger les noms de features (optionnel)
        try:
            metadata_path = Model.get_model_path('fraud-detection-metadata')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                feature_names = metadata.get('feature_names', [])
                print(f"✅ Métadonnées chargées ({len(feature_names)} features)")
        except:
            feature_names = []
            print("⚠️ Métadonnées non disponibles")
        
        print("✅ Initialisation réussie!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {str(e)}")
        raise

def run(raw_data):
    '''
    Cette fonction est appelée pour chaque requête HTTP.
    Elle reçoit les données, fait la prédiction et retourne le résultat.
    
    Format d'entrée attendu:
    {
        "data": [[feature1, feature2, ..., featureN]],
        "transaction_ids": ["TXN_001", "TXN_002"]  # optionnel
    }
    
    Format de sortie:
    {
        "predictions": [
            {
                "transaction_id": "TXN_001",
                "is_fraud": true/false,
                "fraud_probability": 0.85,
                "confidence": 0.85,
                "risk_level": "HIGH/MEDIUM/LOW"
            }
        ],
        "model_info": {
            "model_name": "XGBoost",
            "version": "1.0"
        },
        "status": "success"
    }
    '''
    try:
        # Parser les données JSON
        data = json.loads(raw_data)
        
        # Extraire les features
        if 'data' not in data:
            return json.dumps({
                'error': 'Missing "data" field in request',
                'status': 'error'
            })
        
        input_data = np.array(data['data'])
        transaction_ids = data.get('transaction_ids', 
                                   [f'TXN_{i:04d}' for i in range(len(input_data))])
        
        # Validation de la forme des données
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Preprocessing: Scaling
        scaled_data = scaler.transform(input_data)
        
        # Prédiction
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)
        
        # Formater les résultats
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            fraud_prob = float(proba[1])
            
            # Déterminer le niveau de risque
            if fraud_prob >= 0.7:
                risk_level = "HIGH"
            elif fraud_prob >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            results.append({
                'transaction_id': transaction_ids[i],
                'is_fraud': bool(pred == 1),
                'fraud_probability': round(fraud_prob, 4),
                'legitimate_probability': round(float(proba[0]), 4),
                'confidence': round(float(max(proba)), 4),
                'risk_level': risk_level,
                'recommendation': (
                    'BLOCK - Fraude probable' if fraud_prob >= 0.7 else
                    'REVIEW - Investigation recommandée' if fraud_prob >= 0.4 else
                    'APPROVE - Transaction sûre'
                )
            })
        
        # Réponse complète
        response = {
            'predictions': results,
            'model_info': {
                'model_name': type(model).__name__,
                'version': '1.0',
                'features_count': input_data.shape[1]
            },
            'metadata': {
                'total_transactions': len(results),
                'fraud_detected': sum(1 for r in results if r['is_fraud']),
                'avg_fraud_probability': round(
                    sum(r['fraud_probability'] for r in results) / len(results), 4
                )
            },
            'status': 'success'
        }
        
        return json.dumps(response)
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__,
            'status': 'error',
            'message': 'Une erreur est survenue lors de la prédiction'
        }
        return json.dumps(error_response)
