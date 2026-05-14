"""
train_tcn.py — Temporal Convolutional Network alternative to Bi-LSTM

WHY TCN OVER BI-LSTM FOR THIS PROBLEM:
  - Signs like Green vs Hello share the same handshape and start position
    but differ in their motion TRAJECTORY (the arc/path over time).
  - Bi-LSTM processes frames sequentially and can struggle to capture the
    global shape of a motion path — it tends to weight recent frames more.
  - TCN uses dilated convolutions at multiple receptive field sizes
    simultaneously (frames 1-2, 1-4, 1-8, 1-16) so it sees both the
    fine-grained frame differences AND the full motion arc in one pass.
  - TCN is also more robust to signer variation because dilated convolutions
    act as learned motion filters, not position memorisers.
  - Trains ~2x faster than Bi-LSTM on the same data.
  - Typically matches or beats Bi-LSTM on trajectory-discriminating tasks.

HOW TO USE:
  python train_tcn.py
  This saves isl_tcn_model.h5. Then in live_inference.py change:
      model = tf.keras.models.load_model('isl_tcn_model.h5')
  Run diagnose_confusion.py on both models and pick the better one.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout,
                                     BatchNormalization, Activation, Add,
                                     GlobalAveragePooling1D, SpatialDropout1D)
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
        print(f"  Skipped {skipped} bad-shape files.")
    return np.array(sequences), np.array(labels)


# ==========================================
# AUGMENTATION (same as train_bilstm.py)
# ==========================================
def augment_train_only(X, y, num_classes):
    print("Augmenting (noise + reversal + jitter + mixup)...")

    noise = np.zeros_like(X)
    noise[:, :, :690] = np.random.normal(0, 0.012, X[:, :, :690].shape)
    X_noisy = X + noise

    X_rev = X[:, ::-1, :]

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

    alpha = 0.2
    lam   = np.random.beta(alpha, alpha, size=len(X))
    perm  = np.random.permutation(len(X))
    X_mix = lam[:, np.newaxis, np.newaxis]*X + (1-lam[:, np.newaxis, np.newaxis])*X[perm]

    y_onehot     = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    y_onehot_mix = lam[:, np.newaxis]*y_onehot + (1-lam[:, np.newaxis])*y_onehot[perm]

    X_all = np.concatenate([X, X_noisy, X_rev, X_jit, X_mix])
    y_all = np.concatenate([y_onehot, y_onehot, y_onehot, y_onehot, y_onehot_mix])

    perm2 = np.random.permutation(len(X_all))
    return X_all[perm2], y_all[perm2]


# ==========================================
# TCN BUILDING BLOCK
# ==========================================
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """
    One TCN residual block.

    Architecture per block:
      Conv1D (dilated, causal) → BN → ReLU → SpatialDropout
      Conv1D (dilated, causal) → BN → ReLU → SpatialDropout
      + residual connection (1x1 conv to match channels if needed)

    Causal padding ensures the model only looks at past frames,
    matching how real-time inference works.

    Dilation rate controls the receptive field:
      dilation=1  → sees adjacent frames (fine motion)
      dilation=2  → sees every other frame (medium motion)
      dilation=4  → sees quarter-sequence span (coarse motion arc)
      dilation=8  → sees half the sequence (full trajectory shape)
    """
    residual = x

    x = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)

    x = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # Residual: 1x1 conv to match channel dimensions if different
    if residual.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding='same')(residual)

    return Add()([x, residual])


# ==========================================
# BUILD TCN MODEL
# ==========================================
def build_tcn(sequence_length, features, num_classes):
    """
    Full TCN stack with 4 dilation levels.

    Receptive field = 2 * kernel_size * sum(dilation_rates)
                    = 2 * 3 * (1+2+4+8) = 90 frames

    Since our sequences are 30 frames, every block already sees
    the full sequence — this is intentional. The stacked dilations
    learn motion at progressively coarser timescales simultaneously.
    """
    inputs = Input(shape=(sequence_length, features))

    # Initial projection to 128 channels
    x = Conv1D(128, 1, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Four TCN blocks with increasing dilation
    for dilation in [1, 2, 4, 8]:
        x = tcn_block(x, filters=128, kernel_size=3,
                      dilation_rate=dilation, dropout_rate=0.2)

    # Additional block at 256 channels for classification capacity
    x = Conv1D(256, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tcn_block(x, filters=256, kernel_size=3,
                  dilation_rate=1, dropout_rate=0.2)

    # Global average pooling collapses the time dimension
    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# ==========================================
# LOAD, SPLIT, AUGMENT
# ==========================================
print("Loading data...")
X, y = load_data()
print(f"  {len(X)} sequences, shape {X.shape[1:]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train, y_train_cat = augment_train_only(X_train, y_train, len(ACTIONS))
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(ACTIONS))
print(f"  Training after augmentation: {len(X_train)}")

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))

# ==========================================
# TRAIN
# ==========================================
model = build_tcn(SEQUENCE_LENGTH, FEATURES_PER_FRAME, len(ACTIONS))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    ModelCheckpoint('isl_tcn_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6,
                      verbose=1, min_lr=1e-6)
]

print("\nStarting TCN training...")
model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
print("\nDone. TCN model saved as isl_tcn_model.h5")
print("\nTo compare:")
print("  1. Run diagnose_confusion.py — change MODEL_PATH to 'isl_tcn_model.h5'")
print("  2. Compare val_accuracy and confusion matrix to isl_bilstm_model.h5")
print("  3. Use whichever model scores higher in live_inference.py")