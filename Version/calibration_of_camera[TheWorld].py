# __author__ = 'lenovo'
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
from numpy import *
import computeExtrinsics

'''
README
使用时，只需调用WorldCR即可。
输入二维点，即可得到相对应的世界坐标。
那么，愿君使用愉快。

示范：
input:
point = [1920, 1080]
print(WorldCR(point))

out:
[230.07460985940585, 123.50374964342024]
'''



def WorldCR(CameraPoint):
    # 输入去畸变图片地址（图不在多，一张就行）
    UImage = cv2.imread('D:\\calibration_of_life\\undistortedImage1.jpg')
    w = 9
    h = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    # 输入相机参数（内外参矩阵）
    rotationMatrix = np.array([0.9736, 0.2206, 0.0586,
                               -0.2211, 0.9752, 0.0028,
                               -0.0565, -0.0156, 0.9983]).reshape([3, 3])
    tvecs = np.array([-56.5005495948635, -34.2788952166047, 165.498421195206])
    IntrinsicMatrix = np.array([1.483103141133464e+03, 0, 0,
                                0, 1.482484166341801e+03, 0,
                                9.756763152717728e+02, 5.707759201957292e+02, 1]).reshape([3, 3])
    '''
    imageFileNames = 'D:\calibration_of_life\WIN_20180128_13_58_40_Pro.jpg'
    originalImage = cv2.imread(imageFileNames)
    H, W = originalImage.shape[:2]
    # dist = np.array([-0.393139024611420, 0.145398896979296, 0.0453941885614018, -0.00721261506810390, -0.00307266272556008])
    # 去畸变化
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(IntrinsicMatrix, dist, (W, H), 0.7, (W, H))
    UImage = cv2.undistort(originalImage, IntrinsicMatrix, dist, None, newcameramtx)
    cv2.imwrite('D:\calibration_of_life\Image_undistorted.png', UImage)
    '''

    # 寻找角点
    gray = cv2.cvtColor(UImage, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(UImage, (w, h), corners, ret)
        cv2.imshow('find corners', UImage)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    # print('imgp', imgpoints)
    imgpoints = np.array(imgpoints).reshape(54, 2)
    Opoint = imgpoints[0, :]
    tV = Opoint
    theta = 0
    for m in range(1, 8):
        testpoint = imgpoints[m, :]
        theta_s = np.arctan((testpoint[1] - Opoint[1]) / (testpoint[0] - Opoint[0]))
        theta = theta + theta_s

    theta = theta / (m + 1)
    TRM = np.array(
        [cos(theta), -sin(theta), sin(theta), cos(theta)]).reshape(2, 2)
    CameraPoint = np.array(CameraPoint)
    TimgP = np.dot(TRM, CameraPoint) + tV
    WorldPoint = pointsToWorld(rotationMatrix, tvecs, IntrinsicMatrix, TRM, TimgP)
    return WorldPoint



def pointsToWorld(rotationMatrix, tvecs, IntrinsicMatrix, TRM, CameraPoint ):
    # 构造投影变换矩阵
    r11 = rotationMatrix[0, 0]
    r12 = rotationMatrix[0, 1]
    r13 = rotationMatrix[0, 2]
    r21 = rotationMatrix[1, 0]
    r22 = rotationMatrix[1, 1]
    r23 = rotationMatrix[1, 2]
    t1 = tvecs[0]
    t2 = tvecs[1]
    t3 = tvecs[2]
    RT = np.array([r11, r12, r13, r21, r22, r23, t1, t2, t3]).reshape([3, 3])
    # print('RT\n', RT)
    T = np.dot(RT, IntrinsicMatrix)
    # print('T\n', T)
    invT = np.linalg.inv(T)
    # print('invT\n', invT)
    # 所求点的图像坐标
    U = np.hstack((CameraPoint, np.array([1])))
    X = np.dot(U, invT)
    Xa = X[0] / X[2]
    Xb = X[1] / X[2]
    X = [Xa, Xb]
    return X
