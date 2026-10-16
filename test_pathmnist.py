import numpy as np
import tensorflow as tf

label_names = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium"
}

data = np.load("pathmnist_64.npz")

# Test dataset-аас эхний зураг
image = data["test_images"][0].astype("float32") / 255.0
true_label = int(data["test_labels"][0, 0])

# Model ачаалах
model = tf.keras.models.load_model("pathmnist_model.keras")

# Prediction
prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)

predicted_label = int(np.argmax(prediction[0]))
confidence = float(np.max(prediction[0])) * 100

print("True label:", label_names[true_label])
print("AI prediction:", label_names[predicted_label])
print(f"Confidence: {confidence:.2f}%")
x_test = data["test_images"].astype("float32") / 255.0
y_test = data["test_labels"].flatten()

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")