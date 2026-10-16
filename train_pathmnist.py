import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset унших
data = np.load("pathmnist_64.npz")

x_train = data["train_images"].astype("float32") / 255.0
y_train = data["train_labels"].flatten()

x_val = data["val_images"].astype("float32") / 255.0
y_val = data["val_labels"].flatten()

# CNN model
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(9, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=64
)

model.save("pathmnist_model.keras")

print("PathMNIST model saved!")