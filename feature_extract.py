import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from matplotlib.gridspec import GridSpec

# IMAGE PATH
IMAGE_PATH = r"C:\projects\main project svm journal paddy\Dataset\rgb_dieases\test_images\blast\100154.jpg"

# LOAD MODEL
base_model = MobileNetV2(weights="imagenet", include_top=False)

layer_name = "block_1_expand_relu"

feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer(layer_name).output
)

# LOAD IMAGE
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("Image not found")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224,224))

x = np.expand_dims(img_resized, axis=0)
x = preprocess_input(x)

# EXTRACT FEATURES
feature_maps = feature_model.predict(x)[0]

# CREATE LAYOUT
fig = plt.figure(figsize=(12,5))
gs = GridSpec(3,5, figure=fig)

# INPUT IMAGE
ax1 = fig.add_subplot(gs[:, :2])
ax1.imshow(img_resized)
ax1.set_title("Input Leaf Image")
ax1.axis("off")

labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']

# FEATURE MAPS
for i in range(9):
    row = i // 3
    col = i % 3 + 2
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(feature_maps[:,:,i], cmap="viridis")
    ax.axis("off")

    # label BELOW the image
    ax.text(0.5, -0.15, labels[i],
            transform=ax.transAxes,
            ha='center',
            fontsize=11)

plt.suptitle("CNN Feature Extraction")
plt.tight_layout()
plt.show()