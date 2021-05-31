import os
import cv2
import time
from tqdm import tqdm


time.sleep(2)
ImageCount = 200
pathname = 'Images'
basename = 'img_'

if not os.path.exists(pathname):
    os.mkdir(pathname)


# Makes Json folder for utility
if not os.path.exists("Json"):
    os.mkdir("Json")

Camera = cv2.VideoCapture(0)
Camera.set(3, 1280)
Camera.set(4, 720)

imgList = os.listdir(pathname)
imgList = [int(os.path.splitext(value)[0][4:]) for value in imgList]
counter = (max(imgList) + 1) if (len(imgList) > 0) else 0

for i in tqdm(range(ImageCount)):
    _, img = Camera.read()
    cv2.imshow("Preview", img)
    cv2.imwrite(os.path.join(pathname, basename + str(counter) + '.png'), img)
    time.sleep(0.2)
    cv2.waitKey(1)
    counter += 1
