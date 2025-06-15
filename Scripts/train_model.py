import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from data_augmentation import augment_keypoints, temporal_augmentation
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

set_global_policy('mixed_float16')

PATH = os.path.join('data')

actions = np.array(os.listdir(PATH))

sequences = 10
frames = 10

# Create a label map to map each action label to a numeric value
label_map = {label:num for num, label in enumerate(actions)}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

# Augment the dataset
augmented_landmarks, augmented_labels = [], []
for i in range(len(landmarks)):
    original_landmark = np.array(landmarks[i])
    for _ in range(5):  # Generate 5 augmented samples per original sample
        augmented_landmarks.append(augment_keypoints(original_landmark))
        augmented_labels.append(labels[i])

# Combine original and augmented data
landmarks = np.array([np.array(l) for l in landmarks])  # Ensure all landmarks are NumPy arrays
labels = np.array(labels)
landmarks = np.concatenate((landmarks, np.array(augmented_landmarks)))
labels = np.concatenate((labels, np.array(augmented_labels)))

# Convert landmarks and labels to numpy arrays
X, Y = np.array(landmarks), to_categorical(labels).astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.20, random_state=34, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=34, stratify=Y_temp)

def lr_scheduler(epoch, lr):
    if epoch > 20:
        return lr * 0.5
    return lr

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(Y, axis=1)),
    y=np.argmax(Y, axis=1)
)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10, 126)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax', dtype='float32'))  # Use float32 for mixed precision

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stopping, LearningRateScheduler(lr_scheduler)]
)

model.save('/models/SLR.keras')

predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(metrics.classification_report(test_labels, predictions, target_names=actions))