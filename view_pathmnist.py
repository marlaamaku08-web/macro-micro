import numpy as np
from PIL import Image

data = np.load("pathmnist_64.npz")

images = data["train_images"]
labels = data["train_labels"].flatten()

label_names = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth_muscle",
    6: "normal_mucosa",
    7: "cancer_stroma",
    8: "adenocarcinoma"
}

for label in range(9):
    index = np.where(labels == label)[0][0]
    image = Image.fromarray(images[index])

    filename = f"pathmnist_{label}_{label_names[label]}.png"
    image.save(filename)

print("9 sample images saved!")