import cv2
import time
import threading

class Play_Video_Thread:
    def __init__(self,path):
        self.cap = cv2.VideoCapture(path)
        self.frame = None
        self.success = False
        self.stopped = False
        print("Initializing Video Thread")

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def read(self):
        return self.frame

    def vid_success(self):
        return self.success

    def update(self):
        j = 0
        while True:
            self.success, self.frame = self.cap.read()
            j+=1
            if (self.success == 0):
                self.cap.release()
                print('Video Thread Loops: ', j)
                break

            if self.stopped:
                print("Stopping Video")
                print('Video Thread Loops: ', j)
                self.cap.release()
                break

    def stop(self): 
        self.stopped = True
