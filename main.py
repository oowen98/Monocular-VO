import cv2
import time
import numpy as np
import tracking as ft

vid_path = 'C:/Users/Owen/Desktop/School/ECE_2020_2021/Term 1/ELEC 421/Project/Video Data/drivingfootage.mp4'
vid_path2 = 'C:/Users/Owen/Desktop/School/ECE_2020_2021/Term 1/ELEC 421/Project/Video Data/minecraft1.gif'
image = 'C:/Users/Owen/Desktop/School/ECE_2020_2021/Term 1/ELEC 421/Project/Video Data/minecraft.png'

def FeatureTracking(prev_frame,current_frame, prev_points, LK_parameters):

    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None, **LK_parameters)
    return new_points 


if __name__ == '__main__':

    cap = cv2.VideoCapture(vid_path2)
    fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True, type=2) #Feature Detector
    LK_parameters = dict(winSize = (15,15), maxLevel = 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))
    frame_counter = 0
   
    #featureList - ft.FeatureList([]) #List of actively Tracked Features

    ret,prev_frame = cap.read()
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
   
    prev_points = fast.detect(prev_frame, None)
    prev_points = cv2.KeyPoint_convert(prev_points)

    while True:
        success, frame = cap.read()
        frame_counter += 1
        if(success == 0):
            break

        kp = fast.detect(frame, None) #Returns a list of Keypoints
        points = cv2.KeyPoint_convert(kp) #(x,y) cooridinates of the detected corners
       
        #new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_points, None, **LK_parameters)
        #new_points = cv2.KeyPoint_convert(new_points[:,0], new_points[:,1])      

        frame = cv2.drawKeypoints(frame, kp, None, color = (255,0,255))

        #good_new = new_points[status.flatten()==1]
        #good_old = prev_points[status.flatten()==1]
    
        prev_frame = np.copy(frame)
        prev_points = np.copy(points)
        
        cv2.imshow('frame', frame) #Display Frame on window
        
        #Close program when key 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        #Play video repeatedly.
        if(frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            #print('Frames: ', i)
            frame_counter = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()

