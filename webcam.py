import numpy as np
import cv2

dest = 'webcam_calib'

cap = cv2.VideoCapture(0)

calibration_images = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print('Exiting')
        break

    if key == ord('c'):
        print('Saving image')
        calibration_images.append(frame)

for i, img in zip(range(len(calibration_images)), calibration_images):
    filename = '{}/asuswebcam_{}.png'.format(dest, i)
    print('Saving image {} at {}'.format(i, dest))
    cv2.imwrite(filename, img)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
