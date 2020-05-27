import cv2
import os

curdir = os.listdir()

for video in curdir:
    if video.endswith(".mp4"):

        capture = cv2.VideoCapture(video)
        counter = 0

        while capture.isOpened():
            vid, frame = capture.read()

            if vid:
                cv2.imwrite(video+'___{:d}.jpg'.format(counter), frame)
                counter += 500 
                capture.set(1, counter)
            else:
                capture.release()
                break

