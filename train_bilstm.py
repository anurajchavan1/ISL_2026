"""
train_bilstm.py

Bi-directional LSTM model tuned for:
  ~160 classes (after cleaning)
  ~13-21 videos per class
  720-dim feature vectors
  30-frame sequences

Key accuracy improvements over naive training:
  - MixUp augmentation  : smooths decision boundaries between similar signs
  - Recurrent dropout   : forces robustness to signer variation
  - Label smoothing 0.05: prevents overconfident predictions
  - Class weighting     : handles imbalanced video counts per class
  - Stratified split    : ensures every class in both train and test
  - Augment AFTER split : no data leakage into test set
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH          = os.path.join(os.getcwd(), 'extracted_data')
SEQUENCE_LENGTH    = 30
FEATURES_PER_FRAME = 720
EPOCHS             = 200
BATCH_SIZE         = 16   # smaller batch = better gradient estimates for small dataset

ACTIONS = np.array([
    n for n in sorted(os.listdir(DATA_PATH))
    if os.path.isdir(os.path.join(DATA_PATH, n))
])
np.save('classes.npy', ACTIONS)
print(f"Classes: {len(ACTIONS)}")


# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    sequences, labels = [], []
    label_map = {l: i for i, l in enumerate(ACTIONS)}
    skipped   = 0

    for action in ACTIONS:
        path = os.path.join(DATA_PATH, action)
        if not os.path.isdir(path):
            continue
        for f in sorted(os.listdir(path)):
            if not f.endswith('.npy'):
                continue
            seq = np.load(os.path.join(path, f))
            if seq.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                skipped += 1
                continue
            sequences.append(seq)
            labels.append(label_map[action])

    if skipped:
        print(f"  Skipped {skipped} bad-shape files — re-run extract_features.py")

    return np.array(sequences, dtype=np.float32), np.array(labels)


# ==========================================
# AUGMENTATION (training fold only)
# ==========================================
def augment(X, y, num_classes):
    """
    5 augmentation strategies for small datasets:

    1. Gaussian noise      — simulates measurement noise and signer variability
    2. Time reversal       — temporal mirror of the sign
    3. Speed jitter        — 85-115% speed variation
    4. Spatial scaling     — slight zoom in/out on hand positions
    5. MixUp               — blend pairs of sequences, smooths decision boundary
                             This is the most important one for similar-sign confusion
    """
    print("  Augmenting (noise + reversal + jitter + scale + mixup)...")
    y_oh = tf.keras.utils.to_categorical(y, num_classes)

    # 1. Noise (coord + velocity dims only, not angles)
    noise        = np.zeros_like(X)
    noise[:,:,:690] = np.random.normal(0, 0.01, X[:,:,:690].shape)
    X_noise      = X + noise

    # 2. Time reversal
    X_rev        = X[:, ::-1, :]

    # 3. Speed jitter
    jit = []
    for seq in X:
        f   = np.random.uniform(0.80, 1.20)
        n   = max(8, int(SEQUENCE_LENGTH * f))
        idx = np.linspace(0, SEQUENCE_LENGTH-1, n).astype(int)
        r   = seq[idx]
        if n >= SEQUENCE_LENGTH:
            jit.append(r[:SEQUENCE_LENGTH])
        else:
            jit.append(np.concatenate(
                [r, np.zeros((SEQUENCE_LENGTH-n, FEATURES_PER_FRAME))]))
    X_jit = np.array(jit, dtype=np.float32)

    # 4. Spatial scaling (scale hand coords by 0.9-1.1)
    scale        = np.random.uniform(0.90, 1.10, (len(X), 1, 1))
    X_scale      = X.copy()
    X_scale[:,:,:345] *= scale  # scale coordinate dims only

    # 5. MixUp
    alpha        = 0.3
    lam          = np.random.beta(alpha, alpha, len(X))
    perm         = np.random.permutation(len(X))
    lam_s        = lam[:,None,None]
    X_mix        = lam_s*X + (1-lam_s)*X[perm]
    y_mix        = lam[:,None]*y_oh + (1-lam[:,None])*y_oh[perm]

    X_all = np.concatenate([X, X_noise, X_rev, X_jit, X_scale, X_mix])
    y_all = np.concatenate([y_oh, y_oh, y_oh, y_oh, y_oh, y_mix])

    p = np.random.permutation(len(X_all))
    return X_all[p], y_all[p]


# ==========================================
# LOAD AND SPLIT
# ==========================================
print("Loading data...")
X, y = load_data()
print(f"  {len(X)} sequences | {len(ACTIONS)} classes | shape {X.shape[1:]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, y_train_cat = augment(X_train, y_train, len(ACTIONS))
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(ACTIONS))
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))


# ==========================================
# MODEL
# ==========================================
# Architecture notes for this dataset size:
#   - 3 Bi-LSTM layers with recurrent_dropout forces robustness
#   - Reduced units (192/96/48) vs previous (256/128/64) because
#     ~160 classes with 13-21 samples is a small dataset — large
#     models overfit here
#   - BatchNorm after each LSTM stabilises training
#   - Dense bottleneck 192→96 before softmax
inp = Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME))

x = Bidirectional(LSTM(192, return_sequences=True, activation='tanh',
                        recurrent_dropout=0.15))(inp)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(96, return_sequences=True, activation='tanh',
                        recurrent_dropout=0.15))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(48, return_sequences=False, activation='tanh',
                        recurrent_dropout=0.10))(x)
x = Dropout(0.3)(x)

x = Dense(192, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(96, activation='relu')(x)
out = Dense(len(ACTIONS), activation='softmax')(x)

model = Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    # Lower label_smoothing (0.05) than before — gives sharper softmax
    # peaks at inference, faster confident predictions
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.summary()

# ==========================================
# TRAINING
# ==========================================
callbacks = [
    ModelCheckpoint('isl_bilstm_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=25,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, verbose=1, min_lr=1e-6)
]

print("\nTraining Bi-LSTM...")
model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
print("Done — isl_bilstm_model.h5")