import json
import numpy as np
import cv2
import argparse
import time
from Pose import detectPosesInImage
from FencingMovements import detectLunge

cameraTopCrop = 40
cameraBottomCrop = 100
cameraThreshold = .1

fencersTopCrop = 60
fencersBottomCrop = 290
start = time.time()

display = False

def run():
    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                  The example file can be downloaded from: \
                                                  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('display', type=bool, help='Display output')
    args = parser.parse_args()

    if args.display:
        display = True

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.2,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    cap = cv2.VideoCapture(args.image)

    # Start 3 seconds in
    start_frame_number = 70
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    # Take first frame and find corners in it
    ret, fullFrame = cap.read()
    fencerFrame = fullFrame[fencersTopCrop:fencersBottomCrop, 0:640]
    cameraFrame = fullFrame[cameraTopCrop:cameraBottomCrop, 0:640]

    fshape = fullFrame.shape
    fheight = fshape[0]
    fwidth = fshape[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth, fheight))

    fencer_old_frame = fencerFrame.copy()
    fencer_old_gray = cv2.cvtColor(fencer_old_frame, cv2.COLOR_BGR2GRAY)

    camera_old_frame = cameraFrame.copy()
    camera_old_gray = cv2.cvtColor(camera_old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(fullFrame)
    mask2 = np.zeros_like(fullFrame)

    # get fencer points from pose detection
    allPoints = detectPosesInImage(fencerFrame)

    # nest all the points so that calcOpticalFlowPyrLK works
    fencer1p0 = np.zeros(shape=(18, 1, 2), dtype=np.float32)
    fencer2p0 = np.zeros(shape=(18, 1, 2), dtype=np.float32)
    for i in range(len(allPoints[0])):
        fencer1p0[i][0] = allPoints[0][i]
    for i in range(len(allPoints[1])):
        fencer2p0[i][0] = allPoints[1][i]

    # Get camera points from good features to track
    camerap0 = cv2.goodFeaturesToTrack(camera_old_gray, mask=None, **feature_params)

    totalDisplacement = 0
    totalMovement = 0
    frameCount = 0
    cleanAverageX = 0

    fencer1Positions = []
    fencer1PistePositions = []
    fencer2Positions = []
    fencer2PistePositions = []
    cameraPositions = []

    averageFencerXPositionDifference = 0

    while 1:
        ret, fullFrame = cap.read()

        if not ret:
            break

        fencerFrame = fullFrame[fencersTopCrop:fencersBottomCrop, 0:640]
        cameraFrame = fullFrame[cameraTopCrop:cameraBottomCrop, 0:640]

        frameCount = frameCount + 1

        fencer_frame_gray = cv2.cvtColor(fencerFrame, cv2.COLOR_BGR2GRAY)
        camera_frame_gray = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2GRAY)

        if fencer1p0 is None:
            break
        if fencer2p0 is None:
            break
        if camerap0 is None:
            break

        # calculate optical flow
        fencer1p1, st1, err1 = cv2.calcOpticalFlowPyrLK(fencer_old_gray, fencer_frame_gray, fencer1p0, None, **lk_params)
        fencer2p1, st2, err2 = cv2.calcOpticalFlowPyrLK(fencer_old_gray, fencer_frame_gray, fencer2p0, None, **lk_params)
        camerap1, cameraStatus, err3 = cv2.calcOpticalFlowPyrLK(camera_old_gray, camera_frame_gray, camerap0, None, **lk_params)

        img = np.zeros_like(fullFrame)
        img = cv2.add(img, fullFrame)

        if camerap1 is not None:
            # Select good points
            camera_good_new = camerap1[cameraStatus==1]
            camera_good_old = camerap0[cameraStatus==1]

            # get the average change
            totalX = 0
            pointCount = 0
            for i, (new, old) in enumerate(zip(camera_good_new, camera_good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                totalX = (c - a) + totalX
                pointCount = pointCount + 1

            averageX = 0
            if pointCount > 0:
                averageX = totalX / pointCount

            cleanTotalX = 0
            cleanPointCount = 0
            for i, (new, old) in enumerate(zip(camera_good_new, camera_good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                usePoint = False
                if averageX > 0:
                    if (c - a) * (1 + cameraThreshold) > averageX > (c - a) * (1 - cameraThreshold):
                        usePoint = True
                else:
                    if (c - a) * (1 + cameraThreshold) < averageX < (c - a) * (1 - cameraThreshold):
                        usePoint = True

                if usePoint:
                    cleanTotalX = cleanTotalX + (c - a)
                    cleanPointCount = cleanPointCount + 1

            if cleanPointCount > 0:
                cleanAverageX = cleanTotalX / pointCount

            totalDisplacement = totalDisplacement + cleanAverageX
            totalMovement = totalMovement + abs(cleanAverageX)

            cameraPositions.append(int(totalDisplacement))

            cameraMask = np.zeros_like(fullFrame)
            cv2.circle(cameraMask, (int(-totalDisplacement + 320), 30), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

            img = cv2.add(img, cameraMask)

        else:
            cameraPositions.append(-1)
            camera_good_new = camerap0[cameraStatus == 1]

        if fencer1p1 is None or fencer2p1 is None:
            continue

            # Select good points
        fencer1_good_new = fencer1p1[st1 == 1]

        fencer1_good_old = fencer1p0[st1 == 1]

        fencer2_good_new = fencer2p1[st2 == 1]
        fencer2_good_old = fencer2p0[st2 == 1]

        lungeMask = np.zeros_like(fullFrame)

        # draw the tracks
        fencer1Points = []
        fencer1PistePoints = []
        for i, (new, old) in enumerate(zip(fencer1_good_new, fencer1_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            b = np.int(b + fencersTopCrop)
            d = np.int(d + fencersTopCrop)
            fencer1Points.append([int(a), int(b)])
            fencer1PistePoints.append([int(a + totalDisplacement), int(b)])
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)


        fencer1Positions.append(fencer1Points)
        fencer1PistePositions.append(fencer1Points)

        # draw the tracks
        fencer2Points = []
        fencer2PistePoints = []
        for i, (new, old) in enumerate(zip(fencer2_good_new, fencer2_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            b = np.int(b + fencersTopCrop)
            d = np.int(d + fencersTopCrop)
            fencer2Points.append([int(a), int(b)])
            fencer2PistePoints.append([int(a + totalDisplacement), int(b)])
            mask2 = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)

        fencer2Positions.append(fencer2Points)
        fencer2PistePositions.append(fencer2Points)

        # X position of the fencers necks
        averageFencerXPositionDifference = (averageFencerXPositionDifference + (
                    fencer1Points[1][0] - fencer1Points[1][0])) / 2

        if detectLunge(fencer1Points):
            cv2.circle(lungeMask, (10, 10), 8, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)

        if detectLunge(fencer2Points):
            cv2.circle(lungeMask, (40, 10), 8, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)

        img = cv2.add(img, mask)
        img = cv2.add(img, mask2)
        img = cv2.add(img, lungeMask)

        if display:
            cropMask = np.zeros_like(fullFrame)
            cv2.line(cropMask, (0, cameraTopCrop), (640, cameraTopCrop), (0, 255, 255), 2)
            cv2.line(cropMask, (0, cameraBottomCrop), (640, cameraBottomCrop), (0, 255, 255), 2)
            cv2.line(cropMask, (0, fencersTopCrop), (640, fencersTopCrop), (255, 0, 255), 2)
            cv2.line(cropMask, (0, fencersBottomCrop), (640, fencersBottomCrop), (255, 0, 255), 2)

            cv2.line(cropMask, (320, 0), (320, 360), (255, 255, 0), 2)
            cv2.line(cropMask, (160, 0), (160, 360), (255, 255, 0), 2)
            cv2.line(cropMask, (480, 0), (480, 360), (255, 255, 0), 2)

            img = cv2.add(img, cropMask)

            out.write(img)
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
               break

        # Now update the previous frame and previous points
        fencer_old_gray = fencer_frame_gray.copy()
        camera_old_gray = camera_frame_gray.copy()
        fencer1p0 = fencer1_good_new.reshape(-1,1,2)
        fencer2p0 = fencer2_good_new.reshape(-1,1,2)
        camerap0 = camera_good_new.reshape(-1,1,2)

    cap.release()
    out.release()

    if averageFencerXPositionDifference < 0:
        leftFencerPositions = fencer1Positions
        leftFencerPistePositions = fencer1PistePositions
        rightFencerPositions = fencer2Positions
        rightFencerPistePositions = fencer2PistePositions
    else:
        leftFencerPositions = fencer2Positions
        leftFencerPistePositions = fencer2PistePositions
        rightFencerPositions = fencer1Positions
        rightFencerPistePositions = fencer1PistePositions

    outputJson = ''

    outputJson = outputJson + '{'

    outputJson = outputJson + '"leftFencer":'
    outputJson = outputJson + '{'
    # outputJson = outputJson + '"positions":'
    # outputJson = outputJson + json.dumps(leftFencerPositions)
    # outputJson = outputJson + ','
    outputJson = outputJson + '"pistePositions":'
    outputJson = outputJson + json.dumps(leftFencerPistePositions)
    outputJson = outputJson + '}'
    outputJson = outputJson + ','

    outputJson = outputJson + '"rightFencer":'
    outputJson = outputJson + '{'
    # outputJson = outputJson + '"positions":'
    # outputJson = outputJson + json.dumps(rightFencerPositions)
    # outputJson = outputJson + ','
    outputJson = outputJson + '"pistePositions":'
    outputJson = outputJson + json.dumps(rightFencerPistePositions)
    outputJson = outputJson + '}'
    outputJson = outputJson + ','

    outputJson = outputJson + '"camera":'
    outputJson = outputJson + json.dumps(cameraPositions)

    outputJson = outputJson + '}'

    outputJson = outputJson.replace(' ', '')

    print(outputJson)
run()
# print("Time: {} ".format(time.time() - start))