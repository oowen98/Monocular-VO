from cv2 import cv2

class Feature:
    # Radius of tracking
    FEATURESIZE = 16
    def __init__(self, frame, pos):
        self.pos = pos
        self.lastpos = None
        # define a bounding box centered at pos
        #self.featureBB = (pos[0] - self.FEATURESIZE, pos[1] - self.FEATURESIZE, 2*self.FEATURESIZE, 2*self.FEATURESIZE)
        self.featureBB = (pos[0] - self.FEATURESIZE, pos[1] - self.FEATURESIZE, pos[0] + self.FEATURESIZE, pos[1] + self.FEATURESIZE)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.featureBB)
    
    # updates the position of the feature using a new frame
    # returns true if the feature is still tractable, false if tracking has failed
    # the feature should be popped if this function returns false
    def update(self, frame):
        self.lastpos = self.pos
        (succ, bbox) = self.tracker.update(frame)
        if (succ):
            self.pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            return True
        else:
            return False

    def return_bbox(self): #Returns bounding box
        return self.featureBB
    
class FeatureList:
    def __init__(self, featureList):
        self.list = featureList
        self.len = len(featureList)
    
    # try to push a feature given some minimum separation
    # returns true if push is successful, false if rejected
    def pushToList(self, feature, minDist):
        for f in self.list:
            if((feature.pos[1] - f.pos[1])**2 + (feature.pos[2] - f.pos[2])**2 < minDist**2):
                return False
            else:
                self.list.append(feature)
                self.len += 1
                return True
    
    # update all stored features and pop the untractable ones
    def updatePopList(self, frame):
        for f in self.list:
            if not f.update(frame):
                self.list.remove(f)
                self.len -= 1
