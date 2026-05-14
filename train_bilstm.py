import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional,
                                     BatchNormalization, Input, Layer)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH          = os.path.join(os.getcwd(), 'extracted_data')
SEQUENCE_LENGTH    = 30
FEATURES_PER_FRAME = 720
EPOCHS             = 150
BATCH_SIZE         = 32

ACTIONS = np.array([n for n in sorted(os.listdir(DATA_PATH))
                    if os.path.isdir(os.path.join(DATA_PATH, n))])
np.save('classes.npy', ACTIONS)
print(f"Found {len(ACTIONS)} classes.")


# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    sequences, labels = [], []
    label_map = {l: i for i, l in enumerate(ACTIONS)}
    skipped = 0
    for action in ACTIONS:
        path = os.path.join(DATA_PATH, action)
        if not os.path.exists(path): continue
        for f in os.listdir(path):
            if not f.endswith('.npy'): continue
            seq = np.load(os.path.join(path, f))
            if seq.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                skipped += 1; continue
            sequences.append(seq)
            labels.append(label_map[action])
    if skipped:
        print(f"  Skipped {skipped} bad-shape files. Re-run extract_features.py if large.")
    return np.array(sequences), np.array(labels)


# ==========================================
# AUGMENTATION
# ==========================================
def augment_train_only(X, y, num_classes):
    """
    Four augmentation strategies applied to training split only.

    NEW: MixUp augmentation — interpolates between pairs of training
    sequences and their labels. This is the key fix for the "too rigid"
    problem. By training on blended sequences the model learns a smoother
    decision boundary that tolerates natural signer-to-signer variation
    instead of memorising one exact version of each sign.

    The other three (noise, reversal, speed jitter) are kept from before.
    """
    print("Augmenting training set (noise + reversal + jitter + mixup)...")

    # 1. Gaussian noise (coords + velocity dims only, not angles)
    noise = np.zeros_like(X)
    noise[:, :, :690] = np.random.normal(0, 0.012, X[:, :, :690].shape)
    X_noisy = X + noise

    # 2. Time reversal
    X_rev = X[:, ::-1, :]

    # 3. Speed jitter
    jittered = []
    for seq in X:
        factor  = np.random.uniform(0.85, 1.15)
        new_len = max(10, int(SEQUENCE_LENGTH * factor))
        idx     = np.linspace(0, SEQUENCE_LENGTH-1, new_len).astype(int)
        r       = seq[idx]
        if new_len >= SEQUENCE_LENGTH:
            jittered.append(r[:SEQUENCE_LENGTH])
        else:
            jittered.append(np.concatenate([r, np.zeros((SEQUENCE_LENGTH-new_len, FEATURES_PER_FRAME))]))
    X_jit = np.array(jittered)

    # 4. MixUp — blend random pairs
    # alpha controls interpolation strength; 0.2 keeps blends close to originals
    alpha = 0.2
    lam   = np.random.beta(alpha, alpha, size=len(X))
    perm  = np.random.permutation(len(X))
    lam_s = lam[:, np.newaxis, np.newaxis]  # shape for sequence broadcasting

    X_mix = lam_s * X + (1 - lam_s) * X[perm]

    # MixUp labels are soft blends too
    y_onehot     = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    y_onehot_mix = (lam[:, np.newaxis] * y_onehot
                    + (1 - lam[:, np.newaxis]) * y_onehot[perm])

    # Combine all augmented sets
    X_all = np.concatenate([X,       X_noisy, X_rev,   X_jit,   X_mix])
    # For non-mixup sets, use standard one-hot
    y_all = np.concatenate([y_onehot, y_onehot, y_onehot, y_onehot, y_onehot_mix])

    perm2 = np.random.permutation(len(X_all))
    return X_all[perm2], y_all[perm2]


# ==========================================
# LOAD AND SPLIT
# ==========================================
print("Loading data...")
X, y = load_data()
print(f"  {len(X)} sequences, shape {X.shape[1:]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train, y_train_cat = augment_train_only(X_train, y_train, len(ACTIONS))
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(ACTIONS))
print(f"  Training samples after augmentation: {len(X_train)}")

# Class weights on original integer labels (before one-hot)
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))


# ==========================================
# MODEL
# ==========================================
# Two changes from the previous version:
#
# 1. SpatialDropout1D equivalent via recurrent_dropout — drops entire timestep
#    connections rather than individual units. This forces the model to be
#    robust to missing or noisy frames, directly improving tolerance of
#    signer variation.
#
# 2. Reduced label_smoothing from 0.1 to 0.08 — 0.1 was slightly too
#    aggressive and was part of why confidence values were low at inference.
#    Lower smoothing → sharper softmax peaks → faster confident predictions.

inputs = Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME))

x = Bidirectional(LSTM(256, return_sequences=True, activation='tanh',
                        recurrent_dropout=0.1))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh',
                        recurrent_dropout=0.1))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(64, return_sequences=False, activation='tanh'))(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
outputs = Dense(len(ACTIONS), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
    metrics=['accuracy']
)
model.summary()


# ==========================================
# TRAINING
# ==========================================
callbacks = [
    ModelCheckpoint('isl_bilstm_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6,
                      verbose=1, min_lr=1e-6)
]

print("\nStarting training...")
model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
print("\nDone. Model saved as isl_bilstm_model.h5")