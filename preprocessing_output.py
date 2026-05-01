import cv2
import os
import matplotlib.pyplot as plt

# Input image
INPUT_IMAGE = r"C:\projects\main project svm journal paddy\Dataset\rgb_dieases\test_images\blast\100154.jpg"

# Output folder
OUTPUT_DIR = "preprocessing_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read image
img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise ValueError("Image not found. Check the path.")

# Convert BGR → RGB
before = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# AFTER PREPROCESSING (resize to CNN input size)
after = cv2.resize(before, (224, 224))

# Save images
cv2.imwrite(os.path.join(OUTPUT_DIR, "before_preprocessing.png"), img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "after_preprocessing.png"),
            cv2.cvtColor(after, cv2.COLOR_RGB2BGR))

# Show comparison
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(before)
plt.title("Before Preprocessing")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(after)
plt.title("After Preprocessing")
plt.axis("off")

plt.tight_layout()

# Save final figure for journal
plt.savefig(os.path.join(OUTPUT_DIR,"preprocessing_before_after.png"), dpi=300)

plt.show()

print("Preprocessing output saved in:", OUTPUT_DIR)