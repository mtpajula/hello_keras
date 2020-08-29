import numpy as np
from tensorflow import keras
import cv2
import time

# Read model from folder
model = keras.models.load_model('data/best')
# Webcam
cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read()
    scaled_frame = cv2.resize(frame, (250, 250), interpolation=cv2.INTER_AREA)
    cv2.imshow('CNN', scaled_frame)

    # image format convert for neural network
    img_array = np.array(scaled_frame)
    # Add dimensions to match model requirements
    img_array = np.expand_dims(img_array, axis=0)
    # print(img_array.shape)

    # Predict with model
    prediction = model.predict(img_array)

    print(prediction, end=' ')
    if prediction[0][0] > 0.7:
        print('Something in image')
    else:
        print('nothing...')

    time.sleep(2)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
