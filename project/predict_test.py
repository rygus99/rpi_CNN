# predict_test.py

import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("shape_model.h5")

labels = ["circle", "triangle", "square"]

img = cv2.imread("test_images/sample1.jpg")
img_resized = cv2.resize(img, (64, 64))
img_norm = img_resized.astype("float32") / 255.0
img_norm = np.expand_dims(img_norm, axis=0)

pred = model.predict(img_norm)
idx = np.argmax(pred)

print("예측 결과:", labels[idx])
