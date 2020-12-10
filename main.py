from cv2 import cv2
import time
import numpy as np
import tracking as ft
import imutils as im
import csv
import pandas as pd

import logging
import datetime
import pickle

VIDEO_NAME = 'GH011028'

#Paths of all the different video files
vid_path = 'videos/drivingfootage.mp4'
vid_path2 = 'videos/minecraft_circle.gif'
vid_path3 = 'videos/drivingfootage2.mov'
vid_path4 = 'videos/'+ VIDEO_NAME +'.mp4'

#Camera Matrices
CMAT_GOPRO = 'Camera Calibrations/Gopro_Camera_Matrix_test.csv'
CMAT_IPH = 'Camera Calibrations/Iphone_Camera_Matrix.csv'
CMAT_MC = 'Camera Calibrations/Minecraft_Camera_Matrix.csv'
GPS_data = 'videos/' + VIDEO_NAME + '_HERO8 Black-GPS5.csv'

#Function to detect features in frame when # of actively tracked features falls below a threshold
def refillFeatures(frame, featureList, FASTDetector, minFeatures):
    pushed = False
    # thr = min feature threshold per quadrant
    half_width = int(np.size(frame, 1)/2)
    half_height = int(np.size(frame, 0)/2)
    quad1 = frame[0:half_height, 0:half_width,:]
    quad2 = frame[0:half_height, half_width:,:]
    quad3 = frame[half_height:, 0:half_width,:]
    quad4 = frame[half_height:, half_width:,:]
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
                featureList.pushToList(ft.Feature(frame, np.add(p, offset), 30, tracker="MOSSE"), 10)

            pushed = True
    return pushed
        
#Getting the Essential Matrix from two sets of points
def getTransRot(pts1, pts2):
    #Apply LMedS solver    
    EssentialMatrix, _ = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.LMEDS, prob=0.999)

    #Automatic selection, may not work well
    #_, rot_mat, trans_vec, _ = cv2.recoverPose(EssentialMatrix, prev_pts_norm, curr_pts_norm)

    #Get the Rotation Matrix and Translation vectors from the essential matrix
    rot1, rot2, trans = cv2.decomposeEssentialMat(EssentialMatrix)

    #Get the positive translation
    if trans[2,0] < 0:
        trans = np.multiply(trans, -1)

    #Calculate azimuth of rotations
    if not update_flag:
        testvec1 = rot1@np.array([1, 0, 0])
        testvec2 = rot2@np.array([1, 0, 0])
        theta1 = np.arctan2(testvec1[2], testvec1[0])
        theta2 = np.arctan2(testvec2[2], testvec2[0])
    
    #Use the smaller rotation
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

#Captures features on the road
def getRoadFeatures(frame, roadFeatureList):
    frame_center = (int(frame.shape[0]/2), int(frame.shape[1]/2))
    if roadFeatureList.len < 4:
        roadPoints = []
        for i in range(0,30):
            x = frame_center[0] + 400 * (np.random.random() - 1/2) + 100
            y = frame_center[0] + 80 * (np.random.random()) + 140
            roadPoints.append( (x,y) ) 

        for p in roadPoints: 
            roadFeatureList.pushToList(ft.Feature(frame, p, 25, tracker="KCF", maxLife=30), 1)

#Calculating the relative speed - visual speedometry 
def getRelSpeed(roadFeatureList, Cx, Cy, nframes):
    speeds = []
    for roadPoint in roadFeatureList.list:
        #If the feature doesn't have enough history states, skip
        if len(roadPoint.poshist) < nframes:
            continue
        # calculate transformed velocity
        depth1 = Cy / abs(Cy - roadPoint.pos[1])
        depth2 = Cy / abs(Cy - roadPoint.getPrevPos(nframes)[1])
        speeds.append(abs(depth2 - depth1))
    if len(speeds) == 0:
        return None
    else:
        return np.amax(speeds)


#Reading the Camera Matrix from the csv file
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

#Calibration parameters for algorithm
STEPSIZE = 4
FPS_MULT = 1 #For very high FPS videos
SCALE = 1
FRAME_SIZE = (800,600)
CMAT = ReadCameraMat(CMAT_GOPRO)

FPS = 30

if __name__ == '__main__':
    cap = cv2.VideoCapture(vid_path4) #Change video path for different video
    fast = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True, type=1) #Feature Detector
    frame_counter = 1
    trajectory_map = np.zeros((800,800,3), dtype=np.uint8)
    featureList = ft.FeatureList([]) #List of actively Tracked Features
    roadFeatureList = ft.FeatureList([])

    # Initializations
    print(CMAT)
    update_flag = True
    camera_tf = np.eye(3)
    trans_f = [0, 0, 0]
    trans_f1 = trans_f
    trans_f_prev = trans_f
    trans = np.array([[0],[0],[0]])

    column_names = ["date", "speed"] #Reading GPS Data for speed
    df = pd.read_csv(GPS_data, usecols=column_names)
    speed = df.speed.to_list() 
    x_cord = []
    y_cord = []
    x_cord1 = []
    y_cord1 = []
    increment_MA = 0
    increment_list = []
    speedFactor = 1     # visual speedometry speed
    speedFactors = [1.0]
    iteration_count = 0
    distance = 0
    prev_frame_counter = 0
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE)

    if VIDEO_NAME == 'GH011027':
        TOTAL_FRAMES = 1807 #For GH011027.mp4
        SPEED_DATA_PTS = 1071
    elif VIDEO_NAME == 'GH011028':
        TOTAL_FRAMES = 3460 #For GH011028.mp4
        SPEED_DATA_PTS = 2032
    elif VIDEO_NAME == 'GH011029':
        TOTAL_FRAMES = 2073 #For GH011028.mp4
        SPEED_DATA_PTS = 1240
    else:
        print('Check Video Name for Speed file')


    increment = 1
    num_elements = 1
    start_time = time.time()
    
    ###################################### MAIN LOOP ##########################################
    while True:
        # read ever FPS_MULTth frame
        for i in range(0, FPS_MULT):
            success, frame = cap.read()
            frame_counter += 1
        if(success == 0):
            break

        # downscale the frame to process faster
        frame = cv2.resize(frame, FRAME_SIZE)

        # find center of frame
        Cx = int(frame.shape[0]/2)
        Cy = int(frame.shape[1]/2)

        # find linear interpolation of GPS telemetry speed data
        j = int(np.floor(SPEED_DATA_PTS*frame_counter/(TOTAL_FRAMES)))
        increment = speed[j]/FPS
        increment_list.append(increment)
        increment_MA = np.sum(increment_list[max(j-4, 0):j])/4 # moving average

        # do feature detection
        refillFeatures(frame, featureList, fast, 40)  
        getRoadFeatures(frame, roadFeatureList)

        # update road feature trackers and calculate relative speed
        if(roadFeatureList.len > 0):
            roadFeatureList.updatePopList(frame)
        
        if(roadFeatureList.len > 0):
            relSpeed = getRelSpeed(roadFeatureList, Cx, Cy, 4)
            if (relSpeed is not None):
                speedFactors.append(speedFactor)
                if (abs(relSpeed - speedFactors[-2]) < 5 ):
                    speedFactor = relSpeed
            
            # draw road markers
            for f in roadFeatureList.list:
                cv2.circle(frame, tuple(f.getPosI()), 7, (0,255,255), -1)       

        # update general feature trackers
        if (featureList.len > 0):
            featureList.updatePopList(frame)
            
            for f in featureList.list:    
                bbox = f.getBBoxI()
                p1 = (bbox[0], bbox[1])
                p2 = (bbox[2] + bbox[0] , bbox[3] + bbox[1])
                
                #Draw feature markers
                if f.isActive:
                    #cv2.rectangle(frame, p1, p2, (0,0,255), 2,1)
                    cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,255), -1)
                else:
                    #cv2.rectangle(frame, p1, p2, (255,0,0), 2,1)
                    cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,0), -1)
            
            # update every STEP frames
            if frame_counter % (STEPSIZE) == 0:
                update_flag = True
            
            if update_flag:
                #Grab current positions and nth history positions
                prev_points = []
                curr_points = []
                for f in featureList.getActiveFeatures(STEPSIZE):
                    prev_points.append(f.getPrevPos(STEPSIZE))
                    curr_points.append(f.pos)

                # only try to reproject if we have more than 10 active features
                if (len(prev_points) < 10):
                    continue

                update_flag = False # clear the feature check flag

                # normalizing using camera matrix
                prev_pts_norm = cv2.undistortPoints(np.expand_dims(prev_points, axis=1), cameraMatrix=CMAT, distCoeffs=None)
                curr_pts_norm = cv2.undistortPoints(np.expand_dims(curr_points, axis=1), cameraMatrix=CMAT, distCoeffs=None)
                                
                trans, rot_mat = getTransRot(prev_pts_norm, curr_pts_norm)

                # rotate current camera
                camera_tf = rot_mat@camera_tf
                camera_tf = np.multiply(camera_tf, 1/np.linalg.norm(camera_tf@np.array([1,0,0]), 2)) # normalize
                trans_cam = (camera_tf@trans)*increment*STEPSIZE*FPS_MULT
                trans_cam1 = (camera_tf@trans)*speedFactor*STEPSIZE*FPS_MULT

                # transform camera location
                trans_f = np.add(trans_f, trans_cam[:,0])
                trans_f1 = np.add(trans_f1, trans_cam1[:,0])

                # data logging code
                x = int(trans_f[0]*SCALE) + 600
                y = int(trans_f[2]*SCALE) + 400
                x1 = int(trans_f1[0]*SCALE) + 600
                y1 = int(trans_f1[2]*SCALE) + 400
                x_cord.append(x)
                y_cord.append(y)
                x_cord1.append(x1)
                y_cord1.append(y1)

                # plotting code
                #cv2.circle(trajectory_map, (x,y), 1, (255,0,255), 2)
                cv2.circle(trajectory_map, (x1,y1), 1, (255,255,255), 2)
                
                # labelling code
                velocity = speed[j]*3600.0/1000.0 #km/h
                distance = distance + increment*(frame_counter - prev_frame_counter)
                prev_frame_counter = frame_counter
                text = 'Cooridinates: x: {} y: {} '.format(x-600, y-400)
                trajectory_map[0:60, 0:800] = 0 #Clear the text on the screen
                
                cv2.putText(trajectory_map, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, )
        
        # window updating code
        cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)),7, (255,0,255), -1)
        cv2.imshow('frame', frame) #Display Frame on window  
        cv2.imshow('Trajectory', trajectory_map) 
        
        #Close program when key 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) == ord('p'): #press p to pause
            cv2.waitKey(-1) 

        # increment the iteration counter and repeat
        iteration_count += 1

# save trajectory data
duration = time.time() - start_time
coords = list(zip(x_cord,y_cord,x_cord1,y_cord1, speedFactors, increment_list))
df = pd.DataFrame(coords, columns = ['x', 'y', 'x1', 'y1','Relative Scale', 'increment list'])
df.to_csv('Factor and SCALE_coordinates7'+ VIDEO_NAME + '.csv', index=True)
pickle.dump(trajectory_map, open("Factor and SCALE_trajectory_map7_" + VIDEO_NAME + ".bin", "wb"))
print('Duration: ', str(datetime.timedelta(seconds=duration)))
print('Total Frames: ', frame_counter)
print('Total While Loop Iterations: ', iteration_count)
cv2.imwrite(VIDEO_NAME + '_map7.png', trajectory_map)
cap.release()
cv2.destroyAllWindows()

