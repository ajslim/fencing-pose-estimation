import argparse
import cv2
from Pose import detectPosesInImage
from FencingMovements import detectLunge

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
image1 = cv2.imread(args.image)

positions = detectPosesInImage(image1)


for i, point in enumerate(positions[0]):
    cv2.circle(image1, (int(point[0]), int(point[1])), 12, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image1, str(i), (int(point[0])-6, int(point[1])+6), font, .5, (0, 0, 0), 2, cv2.LINE_AA)

for i, point in enumerate(positions[1]):
    cv2.circle(image1, (int(point[0]), int(point[1])), 12, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(image1, str(i), (int(point[0])-6, int(point[1])+6), font, .5, (0, 0, 0), 2, cv2.LINE_AA)


if detectLunge(positions[0]):
    print('fencer 1 lunge')

if detectLunge(positions[1]):
    print('fencer 2 lunge')

cv2.imshow("Detected Pose", image1)
cv2.waitKey(0)