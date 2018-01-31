import math
import numpy as np


def delta(R=210, r=50, Lb=620, La=880, x=0, y=0, z=-407.4):
    # 逆运动学
    # # 参数
    # R = 210
    # r = 50
    # Lb = 620
    # La = 880
    # # 输入
    # x = 0
    # y = 0
    # z = -407.4
    # 方程
    K_1 = (La ** 2 - Lb ** 2 - x ** 2 - y ** 2 - z ** 2 - (R - r) ** 2 + (R - r) * (np.sqrt(3) * x + y)) / Lb + 2 * z
    U_1 = -2 * (2 * (R - r) - np.sqrt(3) * x - y)
    V_1 = (La ** 2 - Lb ** 2 - x ** 2 - y ** 2 - z ** 2 - (R - r) ** 2 + (R - r) * (np.sqrt(3) * x + y)) / Lb - 2 * z
    K_2 = (La ** 2 - Lb ** 2 - x ** 2 - y ** 2 - z ** 2 - (R - r) ** 2 - (R - r) * (np.sqrt(3) * x - y)) / Lb + 2 * z
    U_2 = -2 * (2 * (R - r) + np.sqrt(3) * x - y)
    V_2 = (La ** 2 - Lb ** 2 - x ** 2 - y ** 2 - z ** 2 - (R - r) ** 2 - (R - r) * (np.sqrt(3) * x - y)) / Lb - 2 * z
    K_3 = (La ** 2 - Lb ** 2 - x ** 2 - y ** 2 - z ** 2 - (R - r) ** 2 - (R - r) * 2 * y) / (2 * Lb) + z
    U_3 = -2 * (R - r + y)
    V_3 = (La ** 2 - Lb ** 2 - x ** 2 - y ** 2 - z ** 2 - (R - r) ** 2 - (R - r) * 2 * y) / (2 * Lb) - z
    theta_1 = 2 * math.atan((-U_1 - np.sqrt(U_1 ** 2 - 4 * K_1 * V_1)) / (2 * K_1))
    theta_2 = 2 * math.atan((-U_2 - np.sqrt(U_2 ** 2 - 4 * K_2 * V_2)) / (2 * K_2))
    theta_3 = 2 * math.atan((-U_3 - np.sqrt(U_3 ** 2 - 4 * K_3 * V_3)) / (2 * K_3))
    print('theta:', theta_1, theta_2, theta_3, '\t')
    return theta_1, theta_2, theta_3
