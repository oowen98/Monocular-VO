import cv2 
import time
import threading
from Play_Video_Thread import Play_Video_Thread

if __name__ == '__main__':
    vid_path = 'C:/Users/Owen/Desktop/School/ECE_2020_2021/Term 1/ELEC 421/Project/Video Data/drivingfootage.mp4'
    video = Play_Video_Thread(path=vid_path2).start()

   
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    time.sleep(.5)
    i = 0
    while True:

        frame = video.read()
        i += 1
        frame = cv2.resize(frame,(640,480), interpolation=cv2.INTER_LINEAR) #Bilinear interpolation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp = fast.detect(gray, None)
        frame = cv2.drawKeypoints(gray, kp, None, color = (255,0,0))
        cv2.imshow('frame', frame) 
        
        vid_bool = video.vid_success()
        if vid_bool == 0:
            cv2.destroyAllWindows()
            print('Camera Loops', i)
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            video.stop()
            time.sleep(1)
            print('Main Loops', i)
            break

