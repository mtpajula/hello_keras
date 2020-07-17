import numpy as np
from tensorflow import keras
import cv2
import time
from read_images import flatten_and_normalize

# Read model from folder
model = keras.models.load_model('data/model')
# Webcam
cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled_frame = cv2.resize(grayFrame, (20, 20), interpolation=cv2.INTER_AREA)
    cv2.imshow('BW', scaled_frame)

    # Predict with model
    prediction = model.predict(np.array([flatten_and_normalize(scaled_frame)]))

    print(prediction, end=' ')
    if prediction[0] > 0.5:
        print('Something in image')
    else:
        print('nothing...')

    time.sleep(2)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
