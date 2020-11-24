from cv2 import cv2
import time
import numpy as np
import tracking as ft
import imutils as im
import csv
import pandas as pd

vid_path = 'videos/drivingfootage.mp4'
vid_path2 = 'videos/minecraft_circle.gif'
vid_path3 = 'videos/drivingfootage2.mov'
vid_path4 = 'videos/GH011027.mp4'
vid_path5 = 'videos/GH011028.mp4'
image = 'videos/minecraft.png'
camera_matrix_gopro = 'Camera Calibrations/Gopro_Camera_Matrix.csv'
camera_matrix_iphone = 'Camera Calibrations/Iphone_Camera_Matrix.csv'
camera_matrix_minecraft = 'Camera Calibrations/Minecraft_Camera_Matrix.csv'
GPS_data = 'videos/GH011028_HERO8 Black-GPS5.csv'
def FeatureTracking(prev_frame,current_frame, prev_points, LK_parameters):

    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None, **LK_parameters)
    return new_points 

def ReadCameraMat(fileName):
    cameraMatrix = []
    with open(fileName) as file:
        reader = csv.reader(file, delimiter=',')
        count = 0
        
        for row in reader:
            if count != 0:
                cameraMatrix.append(row)
            count +=1
    cameraMatrix = np.array(cameraMatrix, dtype = np.float32)

    return cameraMatrix

STEPSIZE = 1
FPS_MULT = 2
SCALE = 2

if __name__ == '__main__':

    cap = cv2.VideoCapture(vid_path5) #Change video path for different video
    fast = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True, type=2) #Feature Detector
    frame_counter = 1
    trajectory_map = np.zeros((800,800,3), dtype=np.uint8)
    featureList = ft.FeatureList([]) #List of actively Tracked Features
    cameraMatrix = ReadCameraMat(camera_matrix_gopro)
    print(cameraMatrix)
    kp = []
    update_flag = True
    camera_tf = np.eye(3)
    trans_f = [0, 0, 0]

    column_names = ["date", "speed"]
    df = pd.read_csv(GPS_data, usecols=column_names)
    speed = df.speed.to_list() #List of 
    fps = 30
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
    #frame = im.resize(frame, width=600)
    frame = cv2.resize(frame, (640,360)) #Reisze to 640,360 for iphone video
    i = 0
    while True:
        lastframe = frame
        for i in range(0, FPS_MULT):
            success, frame = cap.read()
            frame_counter += 1
        if(success == 0):
            break
        #frame = im.resize(frame, width=600)
        frame = cv2.resize(frame, (640,360))
        j= int(np.floor(total_datapts*frame_counter/(total_frames)))
        print('j: ', j)
      
        increment = speed[j]/fps
        #print('Speed: ', speed[x], ' Scale: ', SCALE)
        #print('Scale: ', SCALE)
        # min feature threshold
        if (featureList.len <= 80):
            kp = fast.detect(frame, None) #Returns a list of Keypoints

            points = cv2.KeyPoint_convert(kp) #(x,y) cooridinates of the detected corners
            
            # debug code for one feature
            for p in points: #Push all the features detected from FAST algorithm to the feature list
                featureList.pushToList(ft.Feature(frame, p), 10)
                #print('Pushing, # Points: ', len(points))
            
        
        if (featureList.len > 0):
            featureList.updatePopList(frame)
            
            for f in featureList.list:    
                bbox = f.getBBoxI()
                p1 = (bbox[0], bbox[1])
                p2 = (bbox[2] + bbox[0] , bbox[3] + bbox[1])
                
                #Draw the bounding box
                if f.isActive:
                    cv2.rectangle(frame, p1, p2, (0,0,255), 2,1)
                else:
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2,1)
                cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,255), -1)
            
            # update every Mth frame
            if frame_counter % STEPSIZE == 0:
                print("Updating...")
                update_flag = True
            
            if update_flag:
                prev_points = []
                curr_points = []
                for f in featureList.getActiveFeatures(STEPSIZE):
                    prev_points.append(f.getPrevPos(STEPSIZE))
                    curr_points.append(f.pos)
                #print(len(curr_points))
                # only try to reproject if we have more than 10 active features
                if (len(curr_points) > 10):
                    update_flag = False # clear the feature check flag

                    prev_pts_norm = cv2.undistortPoints(np.expand_dims(prev_points, axis=1), cameraMatrix=cameraMatrix, distCoeffs=None)
                    curr_pts_norm = cv2.undistortPoints(np.expand_dims(curr_points, axis=1), cameraMatrix=cameraMatrix, distCoeffs=None)
                
                    #Calculate the Essential Matrix from the prev, current points and the Camera Matrix from calibration using the RANSAC method        
                    EssentialMatrix, mask = cv2.findEssentialMat(prev_pts_norm, curr_pts_norm, focal=1.0, pp=(0., 0.), method=cv2.LMEDS, prob=0.999, threshold=1.0)

                    #Get the Rotation Matrix and Translation vector from the essential matrix
                    #retval, rot_mat, trans_vec, mask = cv2.recoverPose(EssentialMatrix, prev_pts_norm, curr_pts_norm)
                    rot1, rot2, trans = cv2.decomposeEssentialMat(EssentialMatrix)
                    print(trans)
                    if trans[2,0] < 0:
                        trans = np.multiply(trans, -1)
                    
                    if abs(trans[0]) > 0.50 or abs(trans[1]) > 0.47:
                        update_flag = True  # if translation is sideways, retry the step
                        print("Translation abnormal, retrying next frame")
                    
                    if not update_flag:
                        testvec1 = rot1@np.array([1, 0, 0])
                        testvec2 = rot2@np.array([1, 0, 0])

                        theta1 = np.arctan2(testvec1[2], testvec1[0])
                        theta2 = np.arctan2(testvec2[2], testvec2[0])

                        if min(abs(theta1), abs(theta2)) > 0.15: #15 degrees
                            update_flag = True # if rotation is over 90 degrees, retry the step
                            print("Rotation abnormal, retrying next frame")
                    
                    if not update_flag:
                        if abs(theta1) > abs(theta2):
                            rot_mat = rot2
                            #print(theta2*180/np.pi)
                        else:
                            rot_mat = rot1
                            #print(theta1*180/np.pi)
                        #print(trans[:,0])

                    if not update_flag:
                        # rotate current camera
                        camera_tf = rot_mat@camera_tf
                        camera_tf = np.multiply(camera_tf, 1/np.linalg.norm(camera_tf@np.array([1,0,0]), 2)) # normalize
                        trans_cam = (camera_tf@trans)*increment

                        # transform camera location
                        trans_f = np.add(trans_f, trans_cam[:,0])

                        x = int(trans_f[0]*SCALE) + 600
                        y = int(trans_f[2]*SCALE) + 400  #z coordinate is the one that is changing during the video
                        cv2.circle(trajectory_map, (x,y), 1, (255,255,255), 2)

                        velocity = speed[j]*3600.0/1000.0 #km/h
                        distance = distance + increment*(frame_counter - prev_frame_counter)
                        prev_frame_counter = frame_counter
                        text = 'Cooridinates: x: {} y: {} z: {} \n Speed: {} km/hr Distance Travelled: {:.2f} m'.format(int(trans_f[0]), int(trans_f[1]), int(trans_f[2]), int(velocity), distance)
                        trajectory_map[0:120, 0:800] = 0 #Clear the text on the screen
                        
                        cv2.putText(trajectory_map, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, )
            
        
        cv2.imshow('frame', frame) #Display Frame on window  
        cv2.imshow('Trajectory', trajectory_map) 
        i += 1
        #Close program when key 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''
        #Play video repeatedly.
        if(frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        '''
        count += 1

print('Total Frames: ', frame_counter)
print('Total While Loop Iterations: ', count)
cv2.imwrite('map.png', trajectory_map)
cap.release()
cv2.destroyAllWindows()

