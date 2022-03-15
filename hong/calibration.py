from glob import glob
import cv2
import numpy as np
import pandas as pd
import sys 
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def cailbration_save():

    cail_path_list = glob(os.path.join(os.getcwd(), 'data/calibration/*.jpg'))

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for filename in cail_path_list:

        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        pattern_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if True:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, pattern_found)
   

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    import pickle

    data = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs':rvecs,
        'tvecs':tvecs   
    }

    # save
    with open('./camera_matrix.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def undistort(frame, mtx, dist, img_plot=False):
    """
    Undistort a frame given camera matrix and distortion coefficients.
    :param frame: input frame
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param verbose: if True, show frame before/after distortion correction
    :return: undistorted frame
    """
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    if img_plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 12))
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_undistorted

if __name__ == '__main__':
    # camera matrix extact
    cailbration_save()