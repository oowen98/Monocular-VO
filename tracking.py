from cv2 import cv2

class Feature:
    # Radius of tracking
    FEATURESIZE = 8
    def __init__(self, frame, pos):
        self.pos = pos
        self.lastpos = None
        # define a bounding box centered at pos
        self.featureBB = (pos[0] - self.FEATURESIZE, pos[1] - self.FEATURESIZE, 2*self.FEATURESIZE, 2*self.FEATURESIZE)
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
    
