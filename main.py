import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import threading
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix
)

from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================
RGB_TEST_DIR = r"C:\projects\main project svm journal paddy\Dataset\rgb_dieases\test_images"
MODEL_PATH_DISEASE = "disease_classifier_svm.joblib"
LE_PATH = "label_encoder.joblib"

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading SVM model...")
disease_classifier = joblib.load(MODEL_PATH_DISEASE)
label_encoder = joblib.load(LE_PATH)

# =====================================================
# LOAD CNN FEATURE EXTRACTOR
# =====================================================
print("Loading MobileNetV2...")
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_svm_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    feat = feature_extractor.predict(x, verbose=0)
    return feat.flatten()

# =====================================================
# DISEASE PREDICTION
# =====================================================
def predict_disease(image_path):
    feat = extract_svm_features(image_path)
    if feat is None:
        return "Unknown"
    pred = disease_classifier.predict([feat])[0]
    return label_encoder.inverse_transform([pred])[0]

# =====================================================
# RGB → THERMAL CONVERSION (IMPROVED)
# =====================================================
def rgb_to_thermal(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return thermal

# =====================================================
# STRESS PREDICTION
# =====================================================
def predict_stress(image_path):
    thermal = rgb_to_thermal(image_path)
    if thermal is None:
        return 0.0
    gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)

def get_stress_category(v):
    if v < 0.25:
        return "Non-Stress"
    elif v < 0.5:
        return "Mild Stress"
    elif v < 0.75:
        return "Moderate Stress"
    else:
        return "Severe Stress"

# =====================================================
# GUI SETUP
# =====================================================
root = tk.Tk()
root.title("Rice Disease + Stress Detection (CNN–SVM)")
root.geometry("950x700")

panel = tk.Label(root)
panel.pack()

status_label = tk.Label(root, text="Status: Models Ready", anchor="w")
status_label.pack(fill="x", padx=10)

result_label = tk.Label(root, text="Prediction:", font=("Arial", 14), justify="left")
result_label.pack()

# =====================================================
# IMAGE UPLOAD
# =====================================================
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Show RGB image
    img = Image.open(file_path).resize((400, 300))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Disease Prediction
    disease = predict_disease(file_path)

    # Thermal Image
    thermal = rgb_to_thermal(file_path)
    if thermal is not None:
        thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
        thermal_img = Image.fromarray(thermal_rgb).resize((400, 300))
        thermal_tk = ImageTk.PhotoImage(thermal_img)

        thermal_window = tk.Toplevel(root)
        thermal_window.title("Thermal Visualization")

        tk.Label(thermal_window, text="Thermal Image").pack()
        thermal_label = tk.Label(thermal_window, image=thermal_tk)
        thermal_label.image = thermal_tk
        thermal_label.pack()

    # Stress Prediction
    stress_val = predict_stress(file_path)
    stress_cat = get_stress_category(stress_val)

    result_label.config(
        text=f"Disease: {disease}\n"
             f"Stress Value: {stress_val:.2f}\n"
             f"Stress Level: {stress_cat}"
    )

# =====================================================
# CONFUSION MATRIX
# =====================================================
def plot_confusion_matrix_gui():

    loading = tk.Toplevel(root)
    loading.title("Processing...")
    tk.Label(loading, text="Generating confusion matrix... Please wait.").pack(padx=20, pady=20)
    loading.update()

    def worker():
        X_test, y_test = [], []

        for cls in label_encoder.classes_:
            cls_dir = os.path.join(RGB_TEST_DIR, cls)
            if not os.path.isdir(cls_dir):
                continue

            for f in os.listdir(cls_dir)[:30]:
                img_path = os.path.join(cls_dir, f)
                feat = extract_svm_features(img_path)
                if feat is not None:
                    X_test.append(feat)
                    y_test.append(cls)

        if len(X_test) == 0:
            root.after(0, lambda: (
                loading.destroy(),
                messagebox.showerror("Error", "No test images found!")
            ))
            return

        X_test = np.array(X_test)
        y_test_enc = label_encoder.transform(y_test)
        y_pred = disease_classifier.predict(X_test)

        acc = accuracy_score(y_test_enc, y_pred)
        prec = precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test_enc, y_pred)
        cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-8)

        # TP TN FP FN
        mcm = multilabel_confusion_matrix(y_test_enc, y_pred)

        print("\nCLASSIFICATION RESULTS (TP / TN / FP / FN)")
        print("-------------------------------------------------------------")
        print(f"{'Class':25s} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
        print("-------------------------------------------------------------")

        for i, class_name in enumerate(label_encoder.classes_):
            tn, fp, fn, tp = mcm[i].ravel()
            print(f"{class_name:25s} {tp:6d} {tn:6d} {fp:6d} {fn:6d}")

        print("-------------------------------------------------------------")

        def show_plot():
            loading.destroy()

            win = tk.Toplevel(root)
            win.title("Confusion Matrix")
            win.state("zoomed")

            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax
            )

            ax.set_title(
                f"Confusion Matrix\nAcc={acc:.2f}, Prec={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}"
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig)

        root.after(0, show_plot)

    threading.Thread(target=worker, daemon=True).start()

# =====================================================
# BUTTONS
# =====================================================
tk.Button(root, text="Upload Image", command=upload_image).pack(pady=5)
tk.Button(root, text="Show Confusion Matrix", command=plot_confusion_matrix_gui).pack(pady=5)

root.mainloop()