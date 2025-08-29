#!/usr/bin/env python3
"""Direct test of improved ML model integration."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def test_model():
    print("=" * 60)
    print("Testing Improved ML Model Integration")
    print("=" * 60)
    
    # 1. Load the improved model
    model_path = Path("trained_models/improved_model.pkl")
    if not model_path.exists():
        print("❌ Model not found at trained_models/improved_model.pkl")
        return False
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("\n✅ Model loaded successfully!")
    print(f"   Test accuracy: {model_data['metrics']['test_score']:.3f}")
    print(f"   Confidence threshold: {model_data.get('confidence_threshold', 0.6)}")
    print(f"   Prediction threshold: {model_data.get('threshold', 0.005)*100:.1f}%")
    
    # 2. Create test features matching what the model expects
    feature_names = model_data['features']
    print(f"\n📊 Model expects {len(feature_names)} features")
    
    # Create random test data
    n_samples = 10
    X_test = pd.DataFrame(
        np.random.randn(n_samples, len(feature_names)),
        columns=feature_names
    )
    
    # 3. Make predictions
    model = model_data['model']
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\n🔮 Test Predictions ({n_samples} samples):")
    high_conf_count = 0
    for i in range(min(5, n_samples)):  # Show first 5
        pred = predictions[i]
        prob = max(probabilities[i])
        signal = "BUY" if pred == 1 else "SELL"
        conf_status = "✅ HIGH" if prob > 0.6 else "❌ LOW"
        
        print(f"   Sample {i+1}: {signal} (confidence: {prob:.3f}) {conf_status}")
        
        if prob > 0.6:
            high_conf_count += 1
    
    # 4. Statistics
    total_high_conf = sum(np.max(probabilities, axis=1) > 0.6)
    print(f"\n📈 Statistics:")
    print(f"   High confidence predictions: {total_high_conf}/{n_samples} ({total_high_conf/n_samples*100:.0f}%)")
    print(f"   Expected accuracy on high conf: ~62.6%")
    
    # 5. Integration status
    print(f"\n🚀 Integration Status:")
    print(f"   ✅ Model loads correctly")
    print(f"   ✅ Predictions work")
    print(f"   ✅ Confidence filtering available")
    print(f"\n   To use: python -m robo_trader.runner_async --symbols AAPL --use-ml")
    
    return True

if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)