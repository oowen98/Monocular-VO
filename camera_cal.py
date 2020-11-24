#Camera Calibration script
import cv2
import numpy as np
import glob
import time
import imutils as im
import os

Iphone = glob.glob('C:/Users/Owen/Desktop/School/ECE_2020_2021/Term 1/ELEC 421/Project/Code/videos/phone_camera_calibration/*.JPG') #location of either iphone or gopro images
Gopro = glob.glob('C:/Users/Owen/Desktop/School/ECE_2020_2021/Term 1/ELEC 421/Project/Code/videos/gopro_camera_calibration/*.JPG')

camera_type = 'Gopro' #Either Iphone or Gopro depending on which camera we want to calibrate for

def image_resize(frame):
    if(camera_type == 'Gopro'):
        size = (800,600) #Gopro Image ratio is 4:3
    else:
        size = (640, 360) #Iphone image Ratio is 16:9
    img = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC) #Change resize shape
    return img

if __name__ == '__main__':

    checkerBoard = (8,6) #6 x 8 checkerboard for camera calibration
    corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #termination criteria 

    object_points = [] #3D points for each checkerboard image
    image_points = [] #2D points for each checkerboard image

    #Defining world cooridinates for 3d points
    objp = np.zeros((1, checkerBoard[0] * checkerBoard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerBoard[0], 0:checkerBoard[1]].T.reshape(-1, 2)

    #cv2.namedWindow('Checkerboard Detection', cv2.WINDOW_AUTOSIZE)
    #Extracting images from glob and detecting the checkerboard pattern of the images

    if(camera_type == 'Gopro'):
        images = Gopro
    else:
        images = Iphone

    for image in images:
        filename = os.path.basename(image)
        
        img = cv2.imread(image)
        print(img.shape)
        img = image_resize(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Makes it easier to find corners

        #Find checkerboard corners, ret = true when desired number of corners are found
        patternfound, corners = cv2.findChessboardCorners(gray_img, checkerBoard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        

        if (patternfound):
            print(filename)
            object_points.append(objp)
            #Finds the more refined pixel cooridinates given the corners from cv2.findChessboardCorners
            refined_corners = cv2.cornerSubPix(gray_img, corners, (11,11), (-1,-1), corner_criteria)
            image_points.append(refined_corners)
        
            #Draw and display the corners onto image
            img = cv2.drawChessboardCorners(img, checkerBoard, refined_corners, patternfound)
            img = cv2.putText(img, filename, (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2, cv2.LINE_AA)
        cv2.imshow('Checkerboard Detection', img)
        cv2.waitKey(0)
        
    print("Calibrating images...")

    error, camera_matrix, distortion_coeff, rot_vec, trans_vec = cv2.calibrateCamera(object_points, image_points, gray_img.shape[::-1], None, None)

    print("Reprojection Error: ", error)
    print("Camera Matrix", camera_matrix)
    print("Distortion Coefficients: ", distortion_coeff)
    #cv2.namedWindow('Undistorted Image', cv2.WINDOW_AUTOSIZE)

    #Get a new optimal Camera Matrix
    img = cv2.imread(images[0])
    img = image_resize(img)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 0, (w,h))

    #Compute undistortion and refctification transformation map
    mapX, mapY = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff, np.eye(3),new_camera_matrix, gray_img.shape[::-1], cv2.CV_32FC1)

    #Save Camera Parameters to .csv file
    np.savetxt(('Camera Calibrations/'+ camera_type + '_Camera_Matrix.csv'), camera_matrix, delimiter=',',newline = '\n', fmt='%f', header=(camera_type + ' Camera Matrix'))
    np.savetxt(('Camera Calibrations/'+ camera_type + '_Camera_Dist_Coeff.csv'), distortion_coeff, delimiter=',',newline = '\n', fmt='%f', header=(camera_type + ' Distortion Coefficients'))
    
    #Show corrected images
    for image in images: 
        filename = os.path.basename(image)
        img = cv2.imread(image)
        img = image_resize(img)
        img_undistorted = cv2.remap(img, mapX, mapY, cv2.INTER_LINEAR) #Undistorted Image
        img_undistorted = cv2.putText(img_undistorted, filename, (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2, cv2.LINE_AA)
        print(filename)

        cv2.imshow('Undistorted Image', img_undistorted)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

