import os
import cv2
import numpy as np


def find_chessboard(dir_with_imgs):
    imgs = []
    for img in os.listdir(dir_with_imgs):
        imgs.append(cv2.imread(os.path.join(dir_with_imgs, img)))
    for img in imgs:
        retval, corners = cv2.findChessboardCorners(img, (8, 5))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(img, (8, 5), corners2, retval)
        # window = cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow(window, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cameraMatrix, distCoeffs = None, None
        objp = np.zeros((8 * 5, 3), np.float32)
        objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2) * 30
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera([objp], [corners2], img_gray.shape[::-1], cameraMatrix,
                                                             distCoeffs)
        print(retval, cameraMatrix, distCoeffs, rvecs, tvecs)
        break



if __name__ == '__main__':
    find_chessboard('data')
