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


def Gaussian_Pyramid(Img, Level):
    gaussian_pyramid = []
    count = 0
    gaussian_pyramid.append(Img)
    while count < Level - 1:
        weith, heigth = Img.shape
        newWidth = int(weith / 2)
        newHeigth = int(heigth / 2)
        G = Gaussian_kernel()
        newImg = np.ones((newWidth, newHeigth))
        for i in range(2, Img.shape[0] - 2, 2):
            for j in range(2, Img.shape[1] - 2, 2):
                newImg[int(i / 2), int(j / 2)] = np.sum(Img[i - 2 : i + 3, j - 2 : j + 3] * G)
        Img = newImg
        gaussian_pyramid.append(Img)
        count += 1
    return gaussian_pyramid


def Lucas_Kanade(I1, I2):


    Gx = np.asarray([[-1, 1], [-1, 1]])
    Gy = np.asarray([[-1, -1], [1, 1]])
    Gt1 = np.asarray([[-1, -1], [-1, -1]])
    Gt2 = np.asarray([[1, 1], [1, 1]])

    # scipy.signal.convolve2d(image , kernel) 卷积操作, 默认离散线性卷积，边界用0填充
    # 卷积时，卷积核先旋转180度，若卷积盒行列数为奇数，则以卷积核的中心为锚点，若卷积盒行列数为偶数，则以卷积盒右下角的点为锚点
    # Roberts算子求图像沿 x 方向的梯度
    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 

    # Roberts算子求图像沿 y 方向的梯度
    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2
    # 求图像沿 t 方向的梯度
    It = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)

    # 存储特征点沿 x 方向的光流矢量
    u = np.ones(Ix.shape)

    # 存储特征点沿 y 方向的光流矢量
    v = np.ones(Ix.shape)
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    # LK算法：假设 3 * 3 的领域窗口内所有像素点有相同的光流矢量
    # 最小二乘解：U = A_inverse * B
    # U 是沿 x(U[0]) 方向 和沿 y(U[1]) 方向的光流矢量
    # A = [[sum_neighbor(fx ** 2), sum_neighbor(fx * fy)] , [sum_neighbor(fx * fy), sum_neighbor(fy ** 2)]]
    # B = -[[sum_neighbor(fx * ft)], [sum_neighbor(fy * ft)]]
    for num , i in enumerate(new_Shi_Tomasi_feature):

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

        new_Shi_Tomasi_feature[num] = [(heigh + u[width, heigh]) * 2, (width + v[width, heigh]) * 2]


def Compute_LK(gaussian_pyramid_img1, gaussian_pyramid_img2):
    global new_Shi_Tomasi_feature
    for I1 , I2 in zip(gaussian_pyramid_img1, gaussian_pyramid_img2):
        Lucas_Kanade(I1, I2)
    new_Shi_Tomasi_feature = np.int32(new_Shi_Tomasi_feature / 2)


def Draw():
    global newframe, new_Shi_Tomasi_feature, Shi_Tomasi_feature
    motion_line = np.zeros_like(oldframe)
    color = np.random.randint(0, 255, (100, 3))
    # 绘制光流矢量
    for i, (new, old) in enumerate(zip(new_Shi_Tomasi_feature, Shi_Tomasi_feature)):
        # np.ravel() 将 array 数组降为一维 
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        motion_line = cv2.line(motion_line, (x1, y1), (x2, y2), color[i].tolist(), 2)
        newframe = cv2.circle(newframe, (x1, y1), 5, color[i].tolist(), -1)
    img = cv2.add(newframe, motion_line)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

Level = 2
oldframe = cv2.imread("Rapid_Minion1.jpg")
img1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
newframe = cv2.imread("Rapid_Minion2.jpg")
img2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi 算法是 Harris 算法的改进,角点响应函数为 lambda1 + lambda2
# minDistance：对于初选出的角点而言，如果在其周围 minDistance 范围内存在其他更强角点，则将此角点删除
# blockSize：计算协方差矩阵时的窗口大小
Shi_Tomasi_feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
Shi_Tomasi_feature = np.int32(cv2.goodFeaturesToTrack(img1, mask = None, **Shi_Tomasi_feature_params))
Shi_Tomasi_feature = np.reshape(Shi_Tomasi_feature, newshape=[-1, 2])
new_Shi_Tomasi_feature = np.int32(Shi_Tomasi_feature / (Level ** (Level - 1)))

gaussian_pyramid_img1 = Gaussian_Pyramid(img1, Level)
gaussian_pyramid_img2 = Gaussian_Pyramid(img2, Level)
gaussian_pyramid_img1.reverse()
gaussian_pyramid_img2.reverse()

Compute_LK(gaussian_pyramid_img1, gaussian_pyramid_img2)
Draw()