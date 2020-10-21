from cv2 import cv2
import time
import numpy as np
import tracking as ft

vid_path = 'videos/drivingfootage.mp4'
vid_path2 = 'videos/minecraft1.gif'
image = 'videos/minecraft.png'

def FeatureTracking(prev_frame,current_frame, prev_points, LK_parameters):

    new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None, **LK_parameters)
    return new_points 


if __name__ == '__main__':

    cap = cv2.VideoCapture(vid_path2)
    fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True, type=2) #Feature Detector
    frame_counter = 0
   
    featureList = ft.FeatureList([]) #List of actively Tracked Features
    
    kp = []
    while True:
        success, frame = cap.read()
        frame_counter += 1
        if(success == 0):
            break
        
        # min feature threshold
        if (featureList.len <= 10):
            kp = fast.detect(frame, None) #Returns a list of Keypoints
            points = cv2.KeyPoint_convert(kp) #(x,y) cooridinates of the detected corners

            # debug code for one feature
            for p in points:
                featureList.pushToList(ft.Feature(frame, p), 16)
        
        if (featureList.len > 0):
            featureList.updatePopList(frame)
            for f in featureList.list:    
                bbox = f.getBBoxI()
                p1 = (bbox[0], bbox[1])
                p2 = (bbox[2] + bbox[0] , bbox[3] + bbox[1])
                
                #Draw the bounding box
                cv2.rectangle(frame, p1, p2, (255,0,0), 2,1)
                cv2.circle(frame, tuple(f.getPosI()), 7, (255,0,255), -1)

        
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

