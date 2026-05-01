import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# INPUT IMAGE
# ===============================
IMAGE_PATH = r"C:\projects\main project svm journal paddy\Dataset\rgb_dieases\test_images\blast\100154.jpg"

# ===============================
# READ IMAGE
# ===============================
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Image not found. Check path.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ===============================
# RGB → GRAYSCALE
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# HISTOGRAM EQUALIZATION
# ===============================
gray_eq = cv2.equalizeHist(gray)

# ===============================
# THERMAL VISUALIZATION
# ===============================
thermal = cv2.applyColorMap(gray_eq, cv2.COLORMAP_JET)
thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

# ===============================
# STRESS ESTIMATION
# ===============================
stress_value = np.mean(gray_eq) / 255.0

# Stress category
if stress_value < 0.25:
    stress_level = "Non-Stress"
elif stress_value < 0.5:
    stress_level = "Mild Stress"
elif stress_value < 0.75:
    stress_level = "Moderate Stress"
else:
    stress_level = "Severe Stress"

# ===============================
# DISPLAY RESULTS
# ===============================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("RGB Leaf Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(thermal_rgb)
plt.title(f"Pseudo Thermal Image\nStress = {stress_value:.2f} ({stress_level})")
plt.axis("off")

plt.tight_layout()

# Save high-quality figure for journal
plt.savefig("thermal_stress_output.png", dpi=300, bbox_inches="tight")

plt.show()

print("Stress Value:", round(stress_value,3))
print("Stress Level:", stress_level)