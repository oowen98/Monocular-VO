from cv2 import cv2
import numpy as np
class Feature:
    # Radius of tracking
    FEATURESIZE = 24
    STEPSIZE = 5
    def __init__(self, frame, pos):
        self.pos = pos
        self.lastpos = None
        self.poshist = []
        self.isActive = True
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
        self.poshist.insert(0, self.pos)
        
        (succ, bbox) = self.tracker.update(frame)
        if (succ):
            self.pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            self.featureBB = bbox
            
            # if the feature has moved less than 0.5 px, mark it as stationary
            if max(np.abs(np.subtract(self.pos, self.lastpos))) < 0.5:
                self.stationaryFrames += 1
            else:
                self.stationaryFrames = 0

            # if the feature has moved more than n px, mark it as active
            if (max(np.abs(np.subtract(self.pos, self.lastpos))) > 1.0):
                self.isActive = True
            else:
                self.isActive = False
            return True
        else:
            return False

    def getBBoxI(self): #Returns bounding box
        return tuple([int(i) for i in self.featureBB])
    
    def getPosI(self):
        return tuple([int(i) for i in self.pos])

    def getPrevPos(self, n):
        return self.poshist[n]
    
class FeatureList:
    def __init__(self, featureList):
        self.list = featureList
        self.len = len(featureList)
    
    # try to push a feature given some minimum separation
    # returns true if push is successful, false if rejected
    def pushToList(self, feature, minDist):
        if self.len > 0:
            for f in self.list: #If current feature is within distance of a feature already in the list, don't append it to the list
                if((feature.pos[0] - f.pos[0])**2 + (feature.pos[1] - f.pos[1])**2 < minDist**2):
                    return False
        self.list.append(feature) #otherwise append the feature to the list
        self.len = len(self.list)
        return True
    
    # update all stored features and pop the untractable ones
    def updatePopList(self, frame):
        for f in self.list:
            if not f.update(frame):
                self.list.remove(f)
                continue

            if f.stationaryFrames > 4:
                self.list.remove(f)
                continue
            
            # remove if near edge
            if (abs(f.pos[0]) >= np.size(frame, 1) - 1) or (abs(f.pos[1]) >= np.size(frame, 0) - 1):
                #print(f.pos)
                #print( np.size(frame, 1) - 1)
                self.list.remove(f)
                continue

        self.len = len(self.list)

    def getActiveFeatures(self, minlife):
        filteredList = []
        for f in self.list:
            if f.lastpos is not None and (len(f.poshist) > minlife):
                # if the feature is active and has existed for the last n frames, use it to reproject
                if f.isActive:
                    filteredList.append(f)
        return filteredList
         