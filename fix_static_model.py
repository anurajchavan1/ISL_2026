"""
fix_static_model.py
Converts the static model to .h5 format which is version-independent.
Run once, then update live_inference.py to use the new file.
"""
import tensorflow as tf
import pickle
import numpy as np

# Load with custom object handling
try:
    model = tf.keras.models.load_model(
        'isl_landmark_model.keras',
        compile=False
    )
    print("Loaded successfully")
except Exception as e:
    print(f"Direct load failed: {e}")
    print("Trying legacy format...")
    # Rebuild model architecture manually and load weights
    model = None

if model:
    # Save as .h5 — universally compatible
    model.save('isl_landmark_model.h5')
    print("Saved as isl_landmark_model.h5")
    
    # Verify it loads back
    m2 = tf.keras.models.load_model('isl_landmark_model.h5', compile=False)
    print(f"Verified — input shape: {m2.input_shape}")
    print(f"           output shape: {m2.output_shape}")