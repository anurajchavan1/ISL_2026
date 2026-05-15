import tensorflow as tf
import numpy as np
import zipfile
import shutil
import h5py

# Step 1 — Rebuild exact architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(84,), name='input_layer'),
    tf.keras.layers.Dense(128, activation='relu', name='dense'),
    tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization'),
    tf.keras.layers.Dropout(0.2, name='dropout'),
    tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
    tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_1'),
    tf.keras.layers.Dropout(0.2, name='dropout_1'),
    tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(35, activation='softmax', name='dense_3'),
], name='sequential')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build((None, 84))
print("Architecture rebuilt successfully")

# Step 2 — Manually load weights from h5 file
weights_path = '_keras_extracted/model.weights.h5'

with h5py.File(weights_path, 'r') as f:

    # Dense layer weights: vars/0 = kernel, vars/1 = bias
    model.get_layer('dense').set_weights([
        f['layers/dense/vars/0'][:],
        f['layers/dense/vars/1'][:]
    ])

    # BatchNormalization: vars/0=gamma, 1=beta, 2=moving_mean, 3=moving_var
    model.get_layer('batch_normalization').set_weights([
        f['layers/batch_normalization/vars/0'][:],
        f['layers/batch_normalization/vars/1'][:],
        f['layers/batch_normalization/vars/2'][:],
        f['layers/batch_normalization/vars/3'][:]
    ])

    model.get_layer('dense_1').set_weights([
        f['layers/dense_1/vars/0'][:],
        f['layers/dense_1/vars/1'][:]
    ])

    model.get_layer('batch_normalization_1').set_weights([
        f['layers/batch_normalization_1/vars/0'][:],
        f['layers/batch_normalization_1/vars/1'][:],
        f['layers/batch_normalization_1/vars/2'][:],
        f['layers/batch_normalization_1/vars/3'][:]
    ])

    model.get_layer('dense_2').set_weights([
        f['layers/dense_2/vars/0'][:],
        f['layers/dense_2/vars/1'][:]
    ])

    model.get_layer('dense_3').set_weights([
        f['layers/dense_3/vars/0'][:],
        f['layers/dense_3/vars/1'][:]
    ])

print("All weights loaded successfully!")

# Step 3 — Sanity check
dummy = np.zeros((1, 84), dtype=np.float32)
out   = model(dummy, training=False).numpy()
print(f"Output shape : {out.shape}")
print(f"Output sum   : {out.sum():.6f}  (should be exactly 1.0)")
print(f"Top class    : {np.argmax(out)} with confidence {out.max():.4f}")

# Step 4 — Save as .h5
model.save('isl_landmark_model.h5')
print("\nSaved as isl_landmark_model.h5")

# Step 5 — Verify the saved .h5 loads cleanly
m2 = tf.keras.models.load_model('isl_landmark_model.h5', compile=False)
out2 = m2(dummy, training=False).numpy()
print(f"Reload verify: sum={out2.sum():.6f} — OK")

# Cleanup
shutil.rmtree('_keras_extracted', ignore_errors=True)
print("\nDone! Now update live_inference.py:")
print("  STATIC_MODEL_FILE = 'isl_landmark_model.h5'")