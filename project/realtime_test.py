import cv2
import numpy as np
from picamera2 import Picamera2
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load trained model
model = load_model("shape_model.h5")

# Class labels
labels = ["circle", "triangle", "square"]

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration({'size': (640, 480)}))
picam2.start()

def predict_shape(frame):
    # 이미지 크기 조정 및 정규화
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = pred[0][idx]

    return labels[idx], conf

while True:
    frame = picam2.capture_array()

    # 예측
    shape, confidence = predict_shape(frame)

    # 화면 크기 기준 bounding box 좌표
    h, w, _ = frame.shape
    x1, y1, x2, y2 = int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)

    # 박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 텍스트 표시
    text = f"{shape} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Real-time Shape Detection", frame)

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
