import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model

# IMAGE PATH
IMAGE_PATH = r"C:\projects\main project svm journal paddy\Dataset\rgb_dieases\test_images\blast\100154.jpg"

# LOAD TRAINED MODEL
MODEL_PATH = "disease_classifier_svm.joblib"
LE_PATH = "label_encoder.joblib"

classifier = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LE_PATH)

# LOAD CNN FEATURE EXTRACTOR
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# READ IMAGE
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Image not found")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224,224))

# CNN FEATURE EXTRACTION
x = np.expand_dims(img_resized, axis=0)
x = preprocess_input(x)

features = feature_extractor.predict(x)
features = features.flatten().reshape(1,-1)

# CLASSIFICATION
pred = classifier.predict(features)[0]
disease = label_encoder.inverse_transform([pred])[0]

# DISPLAY RESULT
plt.figure(figsize=(6,4))

plt.imshow(img_resized)
plt.title(f"Predicted Disease: {disease}")
plt.axis("off")

plt.show()