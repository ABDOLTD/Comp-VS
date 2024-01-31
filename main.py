# how to create background Substraction


"""
import cv2 as cv


video = cv.VideoCapture('people.mp4')

subtractor = cv.createBackgroundSubtractorMOG2(300, 50)

while True:
    ret, frame = video.read()

    if ret:
        mask = subtractor.apply(frame)
        cv.imshow("Masked", mask)

        if cv.waitKey(1) == ord('x'):
            break
    else:
        break

cv.destroyAllWindows()
video.release()
"""


# how to output the frame of your images
"""
import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

while True:
    _, frame = camera.read()

    cv.imshow("frame", frame)

    if cv.waitKey(5) == ord('x'):
        break

camera.release()
cv.destroyAllWindows()
"""

# edge detection
"""
import cv2 as cv
import numpy as np

camera = cv.VideoCapture('people.mp4')

while True:
    _, frame = camera.read()

    cv.imshow('camera', frame)

    laplacian = cv.Laplacian(frame, cv.CV_64F)
    laplacian = np.uint8(laplacian)
    cv.imshow('Laplacian', laplacian)

    edges = cv.Canny(frame, 50, 50)
    cv.imshow('Canny', edges)

    if cv.waitKey(5) == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
"""
"""
import cv2 as cv
import matplotlip import pyplot as plt
import numpy as np
import imutlis
import easyocr
"""

# Face Detection application

import cv2 as cv
import numpy as np 


image = cv.imread()