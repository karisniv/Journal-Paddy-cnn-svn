import os, cv2, joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model

RGB_TRAIN_DIR = r"C:\projects\main project svm journal paddy\Dataset\rgb_dieases\train_images"

MODEL_DISEASE = "disease_classifier_svm.joblib"
MODEL_STRESS = "stress_classifier_svm.joblib"
LE_PATH = "label_encoder.joblib"

print("🔄 Loading MobileNetV2 (fast)...")
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def list_images(folder):
    return [f for f in os.listdir(folder)
            if f.lower().endswith((".jpg",".jpeg",".png"))]

def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    x = preprocess_input(np.expand_dims(img,0))
    return feature_extractor.predict(x, verbose=0).flatten()

print("📂 Reading dataset...")
classes = [d for d in os.listdir(RGB_TRAIN_DIR)
           if os.path.isdir(os.path.join(RGB_TRAIN_DIR,d))]

le = LabelEncoder()
le.fit(classes)
joblib.dump(le, LE_PATH)

X, y = [], []
for cls in classes:
    cls_dir = os.path.join(RGB_TRAIN_DIR, cls)
    print(f"➡ Processing class: {cls}")
    for f in list_images(cls_dir):
        feat = extract_features(os.path.join(cls_dir,f))
        if feat is not None:
            X.append(feat)
            y.append(cls)

X = np.array(X)
y = le.transform(y)

print("🧠 Training SVM disease model...")
disease_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])
disease_model.fit(X, y)
joblib.dump(disease_model, MODEL_DISEASE)

print("🌱 Training stress model (conceptual)...")
stress_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])
stress_model.fit(X, y)
joblib.dump(stress_model, MODEL_STRESS)

print("✅ TRAINING FINISHED SUCCESSFULLY")
