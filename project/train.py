# train.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (64, 64)
BATCH = 32
DATASET_PATH = "dataset"   # 폴더 이름

# 1. 데이터 증강 설정
train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=25,
    zoom_range=0.3,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

# 2. 학습 데이터 로더
train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

# 3. 검증 데이터 로더
val_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)

# 4. CNN 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # circle / triangle / square
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. 학습 시작
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25
)

# 6. 모델 저장
model.save("shape_model.h5")
print("모델 저장 완료: shape_model.h5")
