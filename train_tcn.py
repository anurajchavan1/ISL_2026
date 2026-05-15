"""
train_tcn.py

Temporal Convolutional Network tuned for:
  ~160 classes after cleaning
  ~13-21 videos per class
  720-dim feature vectors
  30-frame sequences

TCN advantages over Bi-LSTM for this dataset:
  - Dilated convolutions capture motion at multiple timescales simultaneously
  - Better at distinguishing signs with same handshape but different motion arc
    (e.g. Green vs Hello, Good Morning vs Good Afternoon)
  - Trains faster than Bi-LSTM
  - SpatialDropout1D drops entire feature channels — stronger regularisation
    for small datasets than unit dropout

Same augmentation as train_bilstm.py so results are directly comparable.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout,
                                     BatchNormalization, Activation, Add,
                                     GlobalAveragePooling1D, SpatialDropout1D)
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
BATCH_SIZE         = 16

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
        if not os.path.isdir(path): continue
        for f in sorted(os.listdir(path)):
            if not f.endswith('.npy'): continue
            seq = np.load(os.path.join(path, f))
            if seq.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                skipped += 1; continue
            sequences.append(seq)
            labels.append(label_map[action])
    if skipped:
        print(f"  Skipped {skipped} bad-shape files")
    return np.array(sequences, dtype=np.float32), np.array(labels)


# ==========================================
# AUGMENTATION (identical to train_bilstm.py)
# ==========================================
def augment(X, y, num_classes):
    print("  Augmenting (noise + reversal + jitter + scale + mixup)...")
    y_oh = tf.keras.utils.to_categorical(y, num_classes)

    noise = np.zeros_like(X)
    noise[:,:,:690] = np.random.normal(0, 0.01, X[:,:,:690].shape)
    X_noise = X + noise

    X_rev = X[:, ::-1, :]

    jit = []
    for seq in X:
        f   = np.random.uniform(0.80, 1.20)
        n   = max(8, int(SEQUENCE_LENGTH * f))
        idx = np.linspace(0, SEQUENCE_LENGTH-1, n).astype(int)
        r   = seq[idx]
        jit.append(r[:SEQUENCE_LENGTH] if n >= SEQUENCE_LENGTH
                   else np.concatenate(
                       [r, np.zeros((SEQUENCE_LENGTH-n, FEATURES_PER_FRAME))]))
    X_jit = np.array(jit, dtype=np.float32)

    scale = np.random.uniform(0.90, 1.10, (len(X), 1, 1))
    X_scale = X.copy()
    X_scale[:,:,:345] *= scale

    alpha = 0.3
    lam   = np.random.beta(alpha, alpha, len(X))
    perm  = np.random.permutation(len(X))
    X_mix = lam[:,None,None]*X + (1-lam[:,None,None])*X[perm]
    y_mix = lam[:,None]*y_oh  + (1-lam[:,None])*y_oh[perm]

    X_all = np.concatenate([X, X_noise, X_rev, X_jit, X_scale, X_mix])
    y_all = np.concatenate([y_oh, y_oh, y_oh, y_oh, y_oh, y_mix])
    p = np.random.permutation(len(X_all))
    return X_all[p], y_all[p]


# ==========================================
# TCN BLOCK
# ==========================================
def tcn_block(x, filters, kernel_size, dilation, dropout=0.2):
    """
    Residual TCN block with causal dilated convolutions.

    Causal padding: model only sees past frames (matches real-time inference).
    Dilation rates 1,2,4,8 give receptive fields of 2,4,8,16 frames,
    so the stack simultaneously models fine motion and full trajectory arc.
    """
    res = x

    x = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout)(x)

    x = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout)(x)

    if res.shape[-1] != filters:
        res = Conv1D(filters, 1, padding='same')(res)

    return Add()([x, res])


# ==========================================
# BUILD TCN
# ==========================================
def build_tcn(seq_len, features, num_classes):
    inp = Input(shape=(seq_len, features))

    # Input projection
    x = Conv1D(128, 1, padding='same',
               kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 4 dilated blocks — receptive field covers full 30-frame sequence
    for d in [1, 2, 4, 8]:
        x = tcn_block(x, filters=128, kernel_size=3,
                      dilation=d, dropout=0.15)

    # Wider block for classification capacity
    x = Conv1D(256, 1, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tcn_block(x, filters=256, kernel_size=3,
                  dilation=1, dropout=0.15)

    x = GlobalAveragePooling1D()(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(96, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inp, out)


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
# TRAIN
# ==========================================
model = build_tcn(SEQUENCE_LENGTH, FEATURES_PER_FRAME, len(ACTIONS))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    ModelCheckpoint('isl_tcn_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=25,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, verbose=1, min_lr=1e-6)
]

print("\nTraining TCN...")
model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
print("Done — isl_tcn_model.h5")