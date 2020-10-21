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
    frame_counter = 0
   
    featureList = ft.FeatureList([]) #List of actively Tracked Features
    
    while True:
        success, frame = cap.read()
        frame_counter += 1
        if(success == 0):
            break

        kp = fast.detect(frame, None) #Returns a list of Keypoints
        points = cv2.KeyPoint_convert(kp) #(x,y) cooridinates of the detected corners

        #frame = cv2.drawKeypoints(frame, kp[0], None, color = (255,0,255)) #Drawing all detected corners on frame
        cv2.circle(frame, tuple(points[0]), 7, (255,0,255), -1)

        if(len(kp)>0):
            feature1 = ft.Feature(frame, points[0])        
    
        track_success = feature1.update(frame)
        if (track_success): #Successfully tracked feature
            bbox = feature1.return_bbox()
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]) , int(bbox[3]))
            #Draw the bounding boxq
            cv2.rectangle(frame, p1, p2, (255,0,0), 2,1) 
        else: 
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        
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

