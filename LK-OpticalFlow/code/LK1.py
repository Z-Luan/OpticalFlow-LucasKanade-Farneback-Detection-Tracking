import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


# cv2.imread(filename, flags) 以BGR格式读入图片
# flags: {cv2.IMREAD_COLOR，cv2.IMREAD_GRAYSCALE，cv2.IMREAD_UNCHANGED}
# cv2.IMREAD_COLOR: 默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
# cv2.IMREAD_GRAYSCALE: 读入灰度图片，可用0作为实参替代
# cv2.IMREAD_UNCHANGED: 读入完整图片，包括alpha通道，可用-1作为实参替代
# 【补充】alpha通道, 一个8位的灰度通道, 该通道用256级灰度来记录图像中的透明度信息，定义了透明，不透明和半透明区域，其中黑表示全透明，白表示不透明，灰表示半透明
# cv2.cvtColor(input_image, flag) 颜色空间转换
# flags: {cv2.COLOR_BGR2GRAY，cv2.COLOR_BGR2RGB，cv2.COLOR_BGR2HSV}
def Lucas_Kanade(I1, I2):

    global newframe, Lambda

    # Shi-Tomasi 算法是 Harris 算法的改进,角点响应函数为 lambda1 + lambda2
    # minDistance：对于初选出的角点而言，如果在其周围 minDistance 范围内存在其他更强角点，则将此角点删除
    # blockSize：计算协方差矩阵时的窗口大小
    Shi_Tomasi_feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    Shi_Tomasi_feature = np.int32(cv2.goodFeaturesToTrack(I1, mask = None, **Shi_Tomasi_feature_params))
    Shi_Tomasi_feature = np.reshape(Shi_Tomasi_feature, newshape=[-1, 2])

    # 光照不变性
    I1 = pre_processing(I1)
    I2 = pre_processing(I2)

    color = np.random.randint(0, 255, (100, 3))
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
    motion_line = np.zeros_like(oldframe)
    new_Shi_Tomasi_feature = np.zeros_like(Shi_Tomasi_feature)

    # LK算法：假设 3 * 3 的领域窗口内所有像素点有相同的光流矢量
    # 最小二乘解：U = A_inverse * B
    # U 是沿 x(U[0]) 方向 和沿 y(U[1]) 方向的光流矢量
    # A = [[sum_neighbor(fx ** 2), sum_neighbor(fx * fy)] , [sum_neighbor(fx * fy), sum_neighbor(fy ** 2)]]
    # B = -[[sum_neighbor(fx * ft)], [sum_neighbor(fy * ft)]]

    # 引入正则项之后, 可以处理局部区域像素点光流矢量可能不一致的现象
    # A = [[sum_neighbor(fx ** 2), sum_neighbor(fx * fy)] , [sum_neighbor(fx * fy), sum_neighbor(fy ** 2)]] + lambda * Identity_matrix
    for num , i in enumerate(Shi_Tomasi_feature):

        heigh, width = i

        # 构造 A matrix
        A[0, 0] = np.sum((Ix[width - 1 : width + 2, heigh - 1 : heigh + 2]) ** 2 + Lambda * 1)
        A[1, 1] = np.sum((Iy[width - 1 : width + 2, heigh - 1 : heigh + 2]) ** 2 + Lambda * 1)
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

    # 绘制光流矢量
    for i, (new, old) in enumerate(zip(new_Shi_Tomasi_feature, Shi_Tomasi_feature)):
        # np.ravel() 将 array 数组降为一维 
        x1, y1 = new.ravel()
        x2, y2 = old.ravel()
        motion_line = cv2.line(motion_line, (x1, y1), (x2, y2), color[i].tolist(), 2)
        newframe = cv2.circle(newframe, (x1, y1), 5, color[i].tolist(), -1)
    img = cv2.add(newframe, motion_line)
    return img

def pre_processing(img):
    # MOSSE 没有改变数据的分布,即数据之间的相对值,数据的平均数变成0, 标准差变成1，规范化来减少光照影响
    # img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    
    # SIFT 归一化处理
    # L2范数归一化，欧式距离与余弦相似度一致
    # img = img.astype(np.float32)
    # img /= np.linalg.norm(img)
    # 最大最小归一化
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

# 容许误差范围
Tolerance = 1e-7
Lambda = 0.

oldframe = cv2.imread("data/Brightness1.jpg")
img1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
newframe = cv2.imread("data/Brightness2.jpg")
img2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)

basketball_image = Lucas_Kanade(img1, img2)
basketball_image = cv2.cvtColor(basketball_image,cv2.COLOR_BGR2RGB)
plt.imshow(basketball_image)
plt.show()