import cv2
import imutils
import argparse
import datetime
import imutils
import time
import numpy as np


cap = cv2.VideoCapture(0)
while (1):

    (grabbed, img) = cap.read()
    text = "Unoccupied"

    # 如果不能抓取到一帧，说明我们到了视频的结尾
    if not grabbed:
        break

        # 调整该帧的大小，转换为灰阶图像并且对其进行高斯模糊
    frame = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (25, 25), 0)  # 高斯模糊

    rea, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, None, iterations=2)  # 膨胀

    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        # compute the center of the contour
        if M['m00'] == 0:
            continue
        else:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            if cX >= 480 or cY >= 480:
                continue
            else:
                #   b,g,r=gray[cX,cY]
                print(gray1[cX, cY])
                if gray1[cX, cY] <= 50:
                    cv2.circle(img, (cX, cY), 15, (255, 0, 0), 1)  # 圈住蓝色物体中心
                if gray1[cX, cY] >= 60:
                    cv2.circle(img, (cX, cY), 15, (0, 0, 255), 1)  # 圈住红色物体中心

                cv2.drawContours(img, contours, -1, (0, 0, 255), 3)  # 画轮廓
                cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)  # 圈住中心

                # cv2.imshow("binary", binary)  # 二值图
                cv2.imshow("img", img)  # 识别图
                # cv2.imshow("gray", gray)  # 灰度图

                k = cv2.waitKey(5) & 0xFF   # 按ESC退出
                if k == 27:
                    break
                    # 关闭窗口
cv2.destroyAllWindows()
