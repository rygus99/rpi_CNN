# train.py (improved)

import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (96, 96)
BATCH = 32
DATASET_PATH = "dataset"  # dataset/circle ... triangle ... square

# ---------------------------
# 1) 컨투어 기반 도형 ROI 추출 함수
# ---------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 컨투어
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return cv2.resize(img, IMAGE_SIZE)

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi = img[y:y+h, x:x+w]

    return cv2.resize(roi, IMAGE_SIZE)


# ---------------------------
# 2) 커스텀 데이터 제너레이터 구성
# ---------------------------
class CustomGen(tf.keras.utils.Sequence):
    def __init__(self, file_list, labels, batch_size):
        self.file_list = file_list
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.file_list[idx*self.batch_size : (idx+1)*self.batch_size]

        images = []
        targets = []

        for fpath in batch_files:
            img = preprocess_image(fpath)
            img = img.astype("float32") / 255.0
            images.append(img)

            cname = fpath.split("/")[-2]   # 폴더명 = 클래스명
            targets.append(self.labels[cname])

        return np.array(images), tf.keras.utils.to_categorical(targets, 3)


# 모든 파일 수집
labels = {"circle":0, "triangle":1, "square":2}
file_list = []

for cls in labels.keys():
    folder = os.path.join(DATASET_PATH, cls)
    for f in os.listdir(folder):
        file_list.append(os.path.join(folder, f))

# train / val split
np.random.shuffle(file_list)
split = int(len(file_list) * 0.8)
train_files = file_list[:split]
val_files = file_list[split:]

train_data = CustomGen(train_files, labels, BATCH)
val_data = CustomGen(val_files, labels, BATCH)

# ---------------------------
# 3) 강화된 CNN 모델
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(*IMAGE_SIZE,3)),
    tf.keras.layers.MaxPool2D(),
    
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

model.save("shape_model.h5")
print("모델 저장 완료!")
