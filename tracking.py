from cv2 import cv2
import numpy as np
class Feature:
    # Radius of tracking
    FEATURESIZE = 24
    def __init__(self, frame, pos):
        self.pos = pos
        self.lastpos = (None,None)
        # define a bounding box centered at pos
        self.featureBB = (pos[0] - self.FEATURESIZE, pos[1] - self.FEATURESIZE, 2*self.FEATURESIZE, 2*self.FEATURESIZE)
        self.tracker = cv2.TrackerMOSSE_create()
        self.tracker.init(frame, self.featureBB)
        self.stationaryFrames = 0
    
    # updates the position of the feature using a new frame
    # returns true if the feature is still tractable, false if tracking has failed
    # the feature should be popped if this function returns false
    def update(self, frame):
        self.lastpos = tuple(self.pos)
        
        (succ, bbox) = self.tracker.update(frame)
        if (succ):
            self.pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            self.featureBB = bbox
            
            if (self.pos == self.lastpos):
                self.stationaryFrames += 1
            else:
                self.stationaryFrames = 0
            return True
        else:
            return False

    def getBBoxI(self): #Returns bounding box
        return tuple([int(i) for i in self.featureBB])
    
    def getPosI(self):
        return tuple([int(i) for i in self.pos])
    
class FeatureList:
    def __init__(self, featureList):
        self.list = featureList
        self.len = len(featureList)
        self.current_points = []
        self.previous_points = []
    
    # try to push a feature given some minimum separation
    # returns true if push is successful, false if rejected
    def pushToList(self, feature, minDist):
        if self.len > 0:
            for f in self.list: #If current feature is within distance of a feature already in the list, don't append it to the list
                if((feature.pos[0] - f.pos[0])**2 + (feature.pos[1] - f.pos[1])**2 < minDist**2):
                    return False

        self.list.append(feature) #otherwise append the feature to the list
        self.current_points.append(tuple(feature.pos)) #Creating list containing all the coordinates of the tracked points in current frame
        self.previous_points.append(tuple(feature.lastpos)) #Creating list containing all the coordinates of the tracked points in previous frame
        self.len = len(self.list)
        return True
    
    # update all stored features and pop the untractable ones
    def updatePopList(self, frame):
        for index, f in enumerate(self.list):
            ret = f.update(frame)
            self.previous_points[index] = tuple(f.lastpos) #Updating the coordinates from the previous frame
            if ret == False:
                del self.current_points[index] #Delete coordinates if it is no longer tracked
                del self.previous_points[index]
                self.list.remove(f)

            if f.stationaryFrames > 0:
                del self.current_points[index]
                del self.previous_points[index]
                self.list.remove(f)

            self.len = len(self.list)
         