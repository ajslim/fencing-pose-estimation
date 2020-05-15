import numpy as np
import cv2 as cv
import argparse
import timeit
from pprint import pprint


def run():
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                  The example file can be downloaded from: \
                                                  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    parser.add_argument('image', type=str, help='path to image file')
    args = parser.parse_args()

    topCrop = 40
    bottomCrop = 350
    threshold = .1

    cap = cv.VideoCapture(args.image)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    maxFrames = round(length * .7)

    print(maxFrames)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.2,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, fullFrame = cap.read()

    old_frame = fullFrame[topCrop:bottomCrop, 0:640]
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)


    totalDisplacement = 0
    totalMovement = 0
    frameCount = 0
    averageX = 0
    cleanAverageX = 0
    directionChangeCount = 0

    output = ''
    output = output + '{"cameraMovement":['
    oneCycle = False
    while 1:
        ret, fullFrame = cap.read()

        if not ret:
            break

        frameCount = frameCount + 1

        if frameCount >= maxFrames:
            break

        frame = fullFrame[topCrop:bottomCrop, 0:640]
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if p0 is None:
            break

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is None:
            continue

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # get the average change
        totalX = 0
        pointCount = 0
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            totalX = (c - a) + totalX
            pointCount = pointCount + 1

        lastAverageX = averageX
        averageX = 0
        if pointCount > 0:
            averageX = totalX / pointCount

        if abs(averageX) > 0.1 and np.sign(averageX) != np.sign(lastAverageX):
            directionChangeCount = directionChangeCount + 1


        cleanTotalX = 0
        cleanPointCount = 0

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()

            usePoint = False
            if averageX > 0:
                if (c - a) * (1 + threshold) > averageX > (c - a) * (1 - threshold):
                    usePoint = True
            else:
                if (c - a) * (1 + threshold) < averageX < (c - a) * (1 - threshold):
                    usePoint = True

            if usePoint:
                cleanTotalX = cleanTotalX + (c - a)
                cleanPointCount = cleanPointCount + 1
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

        if cleanPointCount > 0:
            cleanAverageX = cleanTotalX / pointCount

        totalDisplacement = totalDisplacement + cleanAverageX
        totalMovement = totalMovement + abs(cleanAverageX)

        output = output + str(round(cleanAverageX, 3)) + ","

        img = cv.add(frame,mask)

        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
           break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        oneCycle = True

    if oneCycle:
        output = output[:(len(output) - 1)]

    averageVelocity = totalDisplacement / frameCount
    averageSpeed = totalMovement / frameCount
    output = output + '],'

    output = output + '"displacement":' + str(totalDisplacement) + ','
    output = output + '"distance":' + str(totalMovement) + ','
    output = output + '"frames":' + str(frameCount) + '}'

    print(output)

run()