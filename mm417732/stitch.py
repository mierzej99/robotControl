import numpy as np
import cv2
import os

CAMERA_MATRIX = np.array([
    [9.13658341e+02, 0.00000000e+00, 6.57214974e+02],
    [0.00000000e+00, 4.30822512e+03, -5.18255037e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

DIST_COEFFS = np.array([
    [-0.24130662, 0.23226936, -0.00119309, -0.20247147, -0.03785273]
])

def task1(dir_with_imgs):
    imgs = []
    for img in os.listdir(dir_with_imgs):
        imgs.append(cv2.imread(os.path.join(dir_with_imgs, img)))
    for img in imgs:
        size = (img.shape[1], img.shape[0])
        alpha = 0  # TODO: try 0 and 1
        rect_camera_matrix = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS, size, alpha)[0]
        print(rect_camera_matrix)

def task2():
    pass


def task3():
    pass


def task4():
    pass


def task5():
    pass


def task6():
    pass


def task7():
    pass


if __name__ == '__main__':
    task1('data')
