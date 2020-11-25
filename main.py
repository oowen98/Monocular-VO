from cv2 import cv2
import time
import numpy as np
import tracking as ft
import imutils as im
import csv
import pandas as pd

import logging
import datetime


vid_path = 'videos/drivingfootage.mp4'
vid_path2 = 'videos/minecraft_circle.gif'
vid_path3 = 'videos/drivingfootage2.mov'
vid_path4 = 'videos/GH011027.mp4'
vid_path5 = 'videos/GH011028.mp4'
CMAT_GOPRO = 'Camera Calibrations/Gopro_Camera_Matrix_test.csv'
CMAT_IPH = 'Camera Calibrations/Iphone_Camera_Matrix.csv'
CMAT_MC = 'Camera Calibrations/Minecraft_Camera_Matrix.csv'
GPS_data = 'videos/GH011028_HERO8 Black-GPS5.csv'

def refillFeatures(frame, featureList, FASTDetector, minFeatures):
    pushed = False
    # thr = min feature threshold per quadrant
    half_width = int(np.size(frame, 1)/2)
    half_height = int(np.size(frame, 0)/2)
    quad1 = frame[0:half_height, 0:half_width, :]
    quad2 = frame[0:half_height, half_width:, :]
    quad3 = frame[half_height:, 0:half_width, :]
    quad4 = frame[half_height:, half_width:, :]
    quads = [quad1, quad2, quad3, quad4]
    fcounts = [0, 0, 0, 0]

    # count features in each quadrant
    for f in featureList.list:
        if f.pos[0] < half_width and f.pos[1] < half_height:
            fcounts[0] += 1

        elif f.pos[0] >= half_width and f.pos[1] < half_height:
            fcounts[1] += 1

        elif f.pos[0] < half_width and f.pos[1] >= half_height:
            fcounts[2] += 1
        else:
            fcounts[3] += 1

    for q in range(0,4):
        if (fcounts[q] <= minFeatures):
            # set offsets for each quadrant
            # also use higher threshold for treeline (high contrast), lower for road (low contrast)
            if q == 0:  
                offset = (0, 0)
                FASTDetector.setThreshold(70)
            elif q == 1:
                offset = (half_width, 0)
                FASTDetector.setThreshold(70)
            elif q == 2:
                offset = (0, half_height)
                FASTDetector.setThreshold(35)
            else:
                offset = (half_width, half_height)
                FASTDetector.setThreshold(35)

            kp = FASTDetector.detect(quads[q], None) #Returns a list of Keypoints
            points = cv2.KeyPoint_convert(kp) #(x,y) cooridinates of the detected corners

            for p in points: 
                featureList.pushToList(ft.Feature(frame, np.add(p, offset)), 10)

            pushed = True
    return pushed
        

def getTransRot(pts1, pts2):
    #Calculate the Essential Matrix from the prev, current points and the Camera Matrix from calibration using the RANSAC method        
    EssentialMatrix, _ = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.LMEDS, prob=0.999)

    #Get the Rotation Matrix and Translation vector from the essential matrix
    #retval, rot_mat, trans_vec, mask = cv2.recoverPose(EssentialMatrix, prev_pts_norm, curr_pts_norm)
    rot1, rot2, trans = cv2.decomposeEssentialMat(EssentialMatrix)

    if trans[2,0] < 0:
        trans = np.multiply(trans, -1)

    if not update_flag:
        testvec1 = rot1@np.array([1, 0, 0])
        testvec2 = rot2@np.array([1, 0, 0])
        theta1 = np.arctan2(testvec1[2], testvec1[0])
        theta2 = np.arctan2(testvec2[2], testvec2[0])
    
    if not update_flag:
        if abs(theta1) > abs(theta2):
            rot_mat = rot2
            theta = theta2
        else:
            rot_mat = rot1
            theta = theta1

    if abs(trans[0]) > 0.55 or abs(trans[1]) > 0.55 :
        trans = np.array([[0], [0], [0]])
        # if the translation is sideways or vertical, don't move

    if abs(theta) > 0.5:
        rot_mat = np.eye(3)
        # if the rotation is greater than pi/2 rad, don't rotate at all

    return trans, rot_mat

def ReadCameraMat(fileName):
    matrix = []
    with open(fileName) as file:
        reader = csv.reader(file, delimiter=',')
        count = 0
        
        for row in reader:
            if count != 0:
                matrix.append(row)
            count +=1
    matrix = np.array(matrix, dtype = np.float32)

    return matrix

STEPSIZE = 4
FPS_MULT = 1
SCALE = 1
FRAME_SIZE = (800,600)
CMAT = ReadCameraMat(CMAT_GOPRO)

FPS = 30

if __name__ == '__main__':
    cap = cv2.VideoCapture(vid_path5) #Change video path for different video
    fast = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True, type=1) #Feature Detector
    frame_counter = 1
    trajectory_map = np.zeros((800,800,3), dtype=np.uint8)
    featureList = ft.FeatureList([]) #List of actively Tracked Features
    
    print(CMAT)
    kp = []
    update_flag = True
    camera_tf = np.eye(3)
    trans_f = [0, 0, 0]
    trans_f_prev = trans_f
    diff = trans_f
    trans = np.array([[0],[0],[0]])
    column_names = ["date", "speed"]
    df = pd.read_csv(GPS_data, usecols=column_names)
    speed = df.speed.to_list() #List of 
    count = 0
    distance = 0
    prev_frame_counter = 0
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE)

    total_frames = 1807 #For GH011027.mp4
    total_datapts = 1071
    total_frames = 3460 #For GH011028.mp4
    total_datapts = 2032
    increment = 0
    # pre-read one frame
    _, frame = cap.read()

    frame = cv2.resize(frame, FRAME_SIZE)
    i = 0
    Pushing = False
    frames_since_last_push = 0
    start_time = time.time()

    while True:
        lastframe = frame
        for i in range(0, FPS_MULT):
            success, frame = cap.read()
            frame_counter += 1
        if(success == 0):
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        j = int(np.floor(total_datapts*frame_counter/(total_frames)))
        increment = speed[j]/FPS

        Pushing = refillFeatures(frame, featureList, fast, 40)  

        if Pushing: 
            frames_since_last_push = 0
        
        if (featureList.len > 0):
            featureList.updatePopList(frame)
            
            for f in featureList.list:    
                bbox = f.getBBoxI()
                p1 = (bbox[0], bbox[1])
                p2 = (bbox[2] + bbox[0] , bbox[3] + bbox[1])
                
                #Draw the bounding box
                if f.isActive:
                    #cv2.rectangle(frame, p1, p2, (0,0,255), 2,1)
                    cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,255), -1)
                else:
                    #cv2.rectangle(frame, p1, p2, (255,0,0), 2,1)
                    cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,0), -1)
            
            # update every step frames
            if frame_counter % (STEPSIZE) == 0:
                #print("Updating...")
                update_flag = True
            
            if update_flag:
                prev_points = []
                curr_points = []
                for f in featureList.getActiveFeatures(STEPSIZE):
                    prev_points.append(f.getPrevPos(STEPSIZE))
                    curr_points.append(f.pos)

                # only try to reproject if we have more than 10 active features
                if (len(prev_points) < 10):
                    continue

                update_flag = False # clear the feature check flag

                prev_pts_norm = cv2.undistortPoints(np.expand_dims(prev_points, axis=1), cameraMatrix=CMAT, distCoeffs=None)
                curr_pts_norm = cv2.undistortPoints(np.expand_dims(curr_points, axis=1), cameraMatrix=CMAT, distCoeffs=None)
                                
                trans, rot_mat = getTransRot(prev_pts_norm, curr_pts_norm)

                # rotate current camera
                camera_tf = rot_mat@camera_tf
                camera_tf = np.multiply(camera_tf, 1/np.linalg.norm(camera_tf@np.array([1,0,0]), 2)) # normalize
                trans_cam = (camera_tf@trans)*increment*STEPSIZE*FPS_MULT

                # transform camera location
                trans_f = np.add(trans_f, trans_cam[:,0])
                diff = trans_f - trans_f_prev

                x = int(trans_f[0]*SCALE) + 400
                y = int(trans_f[2]*SCALE) + 400  #z coordinate is the one that is changing during the video
                cv2.circle(trajectory_map, (x,y), 1, (255,255,255), 2)

                velocity = speed[j]*3600.0/1000.0 #km/h
                distance = distance + increment*(frame_counter - prev_frame_counter)
                prev_frame_counter = frame_counter
                text = 'Cooridinates: x: {} y: {} z: {} \n Speed: {} km/hr Distance Travelled: {:.2f} m'.format(int(trans_f[0]), int(trans_f[1]), int(trans_f[2]), int(velocity), distance)
                trajectory_map[0:60, 0:800] = 0 #Clear the text on the screen
                
                cv2.putText(trajectory_map, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, )
                trans_f_prev = trans_f
            
        #print('Frame #: {}  trans_vector: x: {:.4f} y: {:.4f} z: {:.4f} FAST Threshold: {} # Active Features {} Feature List Length {} Pushing {}'.format(
        #       frame_counter, float(trans[0]), float(trans[1]), float(trans[2]), fast.getThreshold(), len(curr_points), featureList.len, Pushing ))
        Pushing = False

        cv2.imshow('frame', frame) #Display Frame on window  
        cv2.imshow('Trajectory', trajectory_map) 
        i += 1
        #Close program when key 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) == ord('p'): #press p to pause
            cv2.waitKey(-1) 
        '''
        #Play video repeatedly.
        if(frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        '''
        count += 1
        frames_since_last_push += 1
    
duration = time.time() - start_time
print('Duration: ', str(datetime.timedelta(seconds=duration)))
print('Total Frames: ', frame_counter)
print('Total While Loop Iterations: ', count)
cv2.imwrite('map.png', trajectory_map)
cap.release()
cv2.destroyAllWindows()

