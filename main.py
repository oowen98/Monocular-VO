from cv2 import cv2
import time
import numpy as np
import tracking as ft
import imutils as im
import csv

vid_path = 'videos/drivingfootage.mp4'
vid_path2 = 'videos/minecraft_left.gif'
vid_path3 = 'videos/drivingfootage2.mov'
image = 'videos/minecraft.png'
camera_matrix_gopro = 'Camera Calibrations/Gopro_Camera_Matrix.csv'
camera_matrix_minecraft = 'Camera Calibrations/Minecraft_Camera_Matrix.csv'
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

if __name__ == '__main__':

    cap = cv2.VideoCapture(vid_path2) #Change video path for different video
    fast = cv2.FastFeatureDetector_create(threshold=100, nonmaxSuppression=True, type=2) #Feature Detector
    frame_counter = 0

    featureList = ft.FeatureList([]) #List of actively Tracked Features
    cameraMatrix = ReadCameraMat(camera_matrix_minecraft)
    print(cameraMatrix)
    kp = []
    trans_sum = np.zeros((3,1), dtype=np.float32)
    rot_sum = np.eye(3, dtype=np.float32)

    count = 0
    while True:
        success, frame = cap.read()
        frame = im.resize(frame, width=600)

        frame_counter += 1
        if(success == 0):
            break

        # min feature threshold
        if (len(featureList.getActiveFeatures()) <= 10):
            kp = fast.detect(frame, None) #Returns a list of Keypoints

            points = cv2.KeyPoint_convert(kp) #(x,y) cooridinates of the detected corners
            
            # debug code for one feature
            for p in points: #Push all the features detected from FAST algorithm to the feature list
                featureList.pushToList(ft.Feature(frame, p), 10)
            
        
        if (featureList.len > 0):
            featureList.updatePopList(frame)
            for f in featureList.list:    
                bbox = f.getBBoxI()
                p1 = (bbox[0], bbox[1])
                p2 = (bbox[2] + bbox[0] , bbox[3] + bbox[1])
                
                #Draw the bounding box
                cv2.rectangle(frame, p1, p2, (255,0,0), 2,1)
                cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,255), -1)

            prev_points = []
            curr_points = []
            for f in featureList.getActiveFeatures():
                prev_points.append(f.lastpos)
                curr_points.append(f.pos)

            if (len(curr_points) > 5):
                prev_pts_norm = cv2.undistortPoints(np.expand_dims(prev_points, axis=1), cameraMatrix=cameraMatrix, distCoeffs=None)
                curr_pts_norm = cv2.undistortPoints(np.expand_dims(curr_points, axis=1), cameraMatrix=cameraMatrix, distCoeffs=None)
            
                #Calculate the Essential Matrix from the prev, current points and the Camera Matrix from calibration using the RANSAC method        
                EssentialMatrix, mask = cv2.findEssentialMat(prev_pts_norm, curr_pts_norm, focal=1.0, pp=(0., 0.), method=cv2.LMEDS, prob=0.999, threshold=1.0)

                #Get the Rotation Matrix and Translation vector from the essential matrix
                retval, rot_mat, trans_vec, mask = cv2.recoverPose(EssentialMatrix, prev_pts_norm, curr_pts_norm)
                #print('Rot Matrix: ', rot_mat)
                #print('trans vec: ', trans_vec)
                trans_sum += trans_vec
                rot_sum = np.multiply(rot_mat, rot_sum)
            
            
        
        cv2.imshow('frame', frame) #Display Frame on window  
       
        #Close program when key 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print(trans_sum)
            print(rot_sum)
            break

        #Play video repeatedly.
        if(frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            #print('Frames: ', i)
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        count += 1

cap.release()
cv2.destroyAllWindows()

