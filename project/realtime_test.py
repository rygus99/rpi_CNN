import cv2
import numpy as np
from picamera2 import Picamera2
from tensorflow.keras.models import load_model

model = load_model("shape_model.h5")
labels = ["circle", "triangle", "square"]

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration({'size': (640, 480), 'format': 'RGB888'}))
picam2.start()


def extract_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return frame, (0,0,frame.shape[1], frame.shape[0])

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi = frame[y:y+h, x:x+w]
    return roi, (x, y, w, h)


def predict_shape(frame):
    roi, box = extract_roi(frame)

    img = cv2.resize(roi, (96, 96))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = pred[0][idx]

    return labels[idx], conf, box


while True:
    frame = picam2.capture_array()

    shape, conf, (x, y, w, h) = predict_shape(frame)

    # draw
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(frame, f"{shape} ({conf*100:.1f}%)",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Shape Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
