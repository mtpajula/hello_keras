import cv2
import time

cap = cv2.VideoCapture(0)

"""
Capture 20x20 size images from webcam every 2 seconds.
Save those images to data/raw/ folder.
"""
while True:
    rect, frame = cap.read()
    # To B&W
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Scale to 20x20
    scaled_frame = cv2.resize(grayFrame, (20, 20), interpolation=cv2.INTER_AREA)
    # Show image
    cv2.imshow('BW', scaled_frame)
    # Name image as [timestamp].jpg and save to raw-folder
    ts = int(time.time())
    print(ts)
    cv2.imwrite('data/raw/' + str(ts) + '.jpg', scaled_frame)

    # Wait 2 seconds
    time.sleep(2)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

