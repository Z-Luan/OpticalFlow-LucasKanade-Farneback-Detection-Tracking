import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d


def Gaussian(sigma, x, y):
    a = 1 / (np.sqrt(2 * np.pi) * sigma)
    b = math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    return a * b


# 高斯卷积核
def Gaussian_kernel():
    G = np.zeros((5,5))
    for i in range(-2,3):
        for j in range(-2,3):
            G[i + 1, j + 1] = Gaussian(1.5, i, j)
    return G

# 预测运动轨迹
def Expand(img):

    width, heigh = img.shape
    new_width = int(width * 2)
    new_heigh = int(heigh * 2)
    newimg = np.zeros((new_width, new_heigh))
    # 插入0值行, 0值列
    newimg[::2, ::2] = img
    G = Gaussian_kernel()
    for i in range(2, newimg.shape[0] - 2, 2):
        for j in range(2, newimg.shape[1] - 2, 2):
            # 高斯滤波
            newimg[i, j] = np.sum(newimg[i - 2 : i + 3, j - 2 : j + 3] * G)

    return newimg

def LK_Expand(img, Level):
    # 不采用高斯金字塔
    if Level == 0:
        img = cv2.imread(img,0)
        return img
    i = 0
    # 实参为0, 表示读入灰度图像
    newimg = cv2.imread(img, 0)
    while(i < Level):
        newimg = Expand(newimg)
        i = i + 1
    return newimg

# 处理目标快速运动
def Reduce(img):
    weith, heigth = img.shape
    new_width = int(weith / 2)
    new_heigth = int(heigth / 2)
    G = Gaussian_kernel()
    newimg = np.ones((new_width, new_heigth))
    for i in range(2, img.shape[0] - 2, 2): 
        for j in range(2, img.shape[1] - 2, 2):
            # 高斯滤波 + 下采样
            newimg[int(i / 2), int(j / 2)] = np.sum(img[i - 2 : i + 3, j - 2 : j + 3] * G)

    return newimg

def LK_Reduce(img,Level):
    # 不采用高斯金字塔
    if Level == 0:
        img = cv2.imread(img,0)
        return img
    i = 0
    # 实参为0, 表示读入灰度图像
    newimg = cv2.imread(img,0)
    while(i < Level):
        newimg = Reduce(newimg)
        i = i + 1
    return newimg


def Lucas_Kanade(image1, I1, image2, I2, Level, Reduce_Expand):
    oldframe = cv2.imread(image1)
    oldframe_gray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)

    newframe = cv2.imread(image2)

    color = np.random.randint(0, 255, (100, 3))
    # Roberts算子求图像沿 x 方向的梯度
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))
    # Roberts算子求图像沿 y 方向的梯度
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))


    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2
    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 
    It = convolve2d(I1, Gt1) + convolve2d(I2, Gt2) 

    # Shi-Tomasi 算法是 Harris 算法的改进,角点响应函数为 lambda1 + lambda2
    # minDistance：对于初选出的角点而言，如果在其周围 minDistance 范围内存在其他更强角点，则将此角点删除
    # blockSize：计算协方差矩阵时的窗口大小
    Shi_Tomasi_feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    Shi_Tomasi_feature = np.int32(cv2.goodFeaturesToTrack(oldframe_gray, mask = None, **Shi_Tomasi_feature_params))

    if Reduce_Expand == "Reduce":
        Shi_Tomasi_feature = np.int32(Shi_Tomasi_feature / (2 ** Level))
    else:
        Shi_Tomasi_feature = np.int32(Shi_Tomasi_feature * (2 ** Level))

    Shi_Tomasi_feature = np.reshape(Shi_Tomasi_feature, newshape = [-1, 2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    motion_line = np.zeros_like(oldframe)
    new_Shi_Tomasi_feature=np.zeros_like(Shi_Tomasi_feature)

    # LK算法：假设 3 * 3 的领域窗口内所有像素点有相同的光流矢量
    # 最小二乘解：U = A_inverse * B
    # U 是沿 x(U[0]) 方向 和沿 y(U[1]) 方向的光流矢量
    # A = [[sum_neighbor(fx ** 2), sum_neighbor(fx * fy)] , [sum_neighbor(fx * fy), sum_neighbor(fy ** 2)]]
    # B = -[[sum_neighbor(fx * ft)], [sum_neighbor(fy * ft)]]
    for num , i in enumerate(Shi_Tomasi_feature):

        heigh, width = i

        # 构造 A matrix
        A[0, 0] = np.sum((Ix[width - 1 : width + 2, heigh - 1 : heigh + 2]) ** 2)
        A[1, 1] = np.sum((Iy[width - 1 : width + 2, heigh - 1 : heigh + 2]) ** 2)
        A[0, 1] = np.sum(Ix[width - 1 : width + 2, heigh - 1 : heigh + 2] * Iy[width - 1 : width + 2, heigh - 1 : heigh + 2])
        A[1, 0] = np.sum(Ix[width - 1 : width + 2, heigh - 1 : heigh + 2] * Iy[width - 1 : width + 2, heigh - 1 : heigh + 2])
        # np.linalg.pinv 矩阵求逆
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[width - 1 : width + 2, heigh - 1 : heigh + 2] * It[width - 1 : width + 2, heigh - 1 : heigh + 2])
        B[1, 0] = -np.sum(Iy[width - 1 : width + 2, heigh - 1 : heigh + 2] * It[width - 1 : width + 2, heigh - 1 : heigh + 2])
        optical_flow_vector = np.matmul(Ainv, B)

        u[width, heigh] = optical_flow_vector[0]
        v[width, heigh] = optical_flow_vector[1]

        new_Shi_Tomasi_feature[num] = [np.int32(heigh + u[width, heigh]), np.int32(width + v[width, heigh])]

    if Reduce_Expand == "Reduce":
        new_Shi_Tomasi_feature = np.int32(new_Shi_Tomasi_feature * (2 ** Level))
        Shi_Tomasi_feature = np.int32(Shi_Tomasi_feature * (2 ** Level))
    else:
        new_Shi_Tomasi_feature = np.int32(new_Shi_Tomasi_feature / (2 ** Level))
        Shi_Tomasi_feature = np.int32(Shi_Tomasi_feature / (2 ** Level))

    # 只保留部分, 效果更好
    for i, (new, old) in enumerate(zip(new_Shi_Tomasi_feature, Shi_Tomasi_feature)):
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        motion_line = cv2.line(motion_line, (x1, y1), (x2, y2), color[i].tolist(), 2)
        newframe = cv2.circle(newframe, (x1, y1), 5, color[i].tolist(), -1)
    img = cv2.add(newframe, motion_line)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

IMG1 = "data/Rapid_Minion1.jpg"
IMG2 = "data/Rapid_Minion2.jpg"
# I1_reduce = LK_Reduce(IMG1, 0)
# I2_reduce = LK_Reduce(IMG2, 0)
# FinalReduceImage1 = Lucas_Kanade(IMG1, I1_reduce, IMG2, I2_reduce, 0, "Reduce")
# plt.imshow(FinalReduceImage1)
# plt.show()

I1_expand = LK_Expand(IMG1, 1)
I2_expand = LK_Expand(IMG2, 1)
FinalExpandImage1 = Lucas_Kanade(IMG1, I1_expand, IMG2, I2_expand, 1, "Expand")
plt.imshow(FinalExpandImage1)
plt.show()