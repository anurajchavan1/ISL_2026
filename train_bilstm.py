import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
DATA_PATH = os.path.join(os.getcwd(), 'extracted_data') 

ACTIONS = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
np.save('classes.npy', ACTIONS) 
print(f"Found {len(ACTIONS)} sign classes.")

SEQUENCE_LENGTH = 30 
FEATURES_PER_FRAME = 690 # UPDATED FOR HYBRID MODEL (345 coords + 345 velocity)
EPOCHS = 150 
BATCH_SIZE = 32 

# ==========================================
# 2. DATA LOADING & AUGMENTATION
# ==========================================
def load_data():
    sequences, labels = [], []
    label_map = {label:num for num, label in enumerate(ACTIONS)}
    
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path): continue
            
        for sequence_file in os.listdir(action_path):
            res = np.load(os.path.join(action_path, sequence_file))
            sequences.append(res)
            labels.append(label_map[action])
            
    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS))
    return X, y

def augment_data(X, y):
    """Injects random spatial noise to prevent overfitting."""
    print("Applying Data Augmentation (Adding Noise)...")
    noise = np.random.normal(0, 0.03, X.shape)
    X_augmented = X + noise
    
    X_combined = np.concatenate((X, X_augmented))
    y_combined = np.concatenate((y, y))
    
    return X_combined, y_combined

print("Loading Data...")
X, y = load_data() 
X, y = augment_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. BUILD THE EXPANDED BI-LSTM ARCHITECTURE
# ==========================================
model = Sequential()

model.add(Bidirectional(LSTM(256, return_sequences=True, activation='tanh'), 
                        input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)))
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(128, return_sequences=False, activation='tanh')))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))

model.add(Dense(len(ACTIONS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 4. TRAINING WITH ADVANCED CALLBACKS
# ==========================================
callbacks = [
    ModelCheckpoint('isl_bilstm_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=0.00001)
]

print("Starting Advanced Training on RTX A1000...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

print("Training Complete! Model saved as 'isl_bilstm_model.h5'")