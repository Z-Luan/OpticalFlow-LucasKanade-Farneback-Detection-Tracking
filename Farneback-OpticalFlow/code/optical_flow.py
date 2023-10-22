import numpy as np
import scipy.ndimage



def poly_exp(f, c, sigma):
    """
    Calculates the local polynomial expansion of a 2D signal, as described by Farneback
    计算2D信号的局部多项式展开

    Uses separable normalized correlation
    使用可分离的归一化相关性

    f ~ x^T A x + B^T x + C

    If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
    A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
    is a 2-element array representing the linear term, and C[i, j] is a scalar
    representing the constant term.
    f[i，j] 和 c[i，j] 对应像素点 (i，j) 的信号值(灰度值)和确定性，则 A[i，j] 是表示多项式二次项的 2x2 数组,
    B[i, j] 是表示多项式一次项的 1x2 数组，C[i，j] 是标量, 表示常数项

    Parameters
    ----------
    f：输入信号
    c：信号的准确性
    sigma：利用最小二乘法求解时, 并非邻域内每个像素点样本误差都对中心点有着同样的影响力, 因此利用二维高斯分布将影响力赋予权重

    Returns
    -------
    A：多项式展开的二次项
    B：多项式展开的一次项
    C：多项式展开的常数项
    """

    # 产生二维高斯分布的基础是一维高斯分布
    n = int(4 * sigma + 1)
    x = np.arange(-n, n + 1, dtype=np.int)
    a = np.exp(-(x ** 2) / (2 * sigma ** 2))

    # 分离计算 x , y 维度, 降低计算量
    # b.shape (n, 6)
    bx = np.stack([np.ones(a.shape), x,                np.ones(a.shape), x ** 2,           np.ones(a.shape), x], axis = -1)
    by = np.stack([np.ones(a.shape), np.ones(a.shape), x,                np.ones(a.shape), x ** 2,           x], axis = -1)

    # 预先计算确定性和信号值的乘积
    cf = c * f

    # G and v are used to calculate "r" from the paper: v = G * r
    G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    # 可分离的互相关
    ab = np.einsum("i,ij->ij", a, bx)
    abb = np.einsum("ij,ik->ijk", ab, bx)

    # 计算具有互相关的每个像素点的 G 和 v
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[...,i, j] = scipy.ndimage.correlate1d(c, abb[..., i, j], axis=0, mode="constant", cval=0)

        v[..., i] = scipy.ndimage.correlate1d(cf, ab[..., i], axis=0, mode="constant", cval=0)

    ab = np.einsum("i,ij->ij", a, by)
    abb = np.einsum("ij,ik->ijk", ab, by)

    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0)

        v[..., i] = scipy.ndimage.correlate1d(v[..., i], ab[..., i], axis=1, mode="constant", cval=0)

    # 每个像素点的 r
    r = np.linalg.solve(G, v)

    # 二次项系数
    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    # 一次项系数
    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    # 常数项系数
    C = r[..., 0]

    return A, B, C


def flow_iterative( f1, f2, sigma, c1, c2, sigma_flow, num_iter=1, d=None, model="constant", mu=None ):

    """
    Parameters
    ----------
    f1: 上一帧图像
    f2: 下一帧图像
    sigma: 用于平滑多项式展开的高斯模板参数
           利用最小二乘法求解时, 并非邻域内每个像素点样本误差都对中心点有着同样的影响力, 因此利用二维高斯分布将影响力赋予权重
    c1：Certainty of first image
    c2：Certainty of second image
    sigma_flow：计算光流时拟合邻域内各像素点对应的多项式展开系数的高斯模板参数
    num_iter：迭代次数
    d: (optional)：初始化光流矢量
    p: (optional)：初始运动模型参数
    model: ['constant', 'affine', 'eight_param'] Optical flow parametrization to use
    mu: (optional): Weighting term for usage of global parametrization. Defaults to using value recommended in Farneback's thesis

    Returns
    -------
    d:返回图像各像素点的位移大小，即光流矢量
    """
    # 添加初始噪声参数作为可选输入

    # 计算图像中每个像素点的多项式展开
    A1, B1, C1 = poly_exp(f1, c1, sigma)
    A2, B2, C2 = poly_exp(f2, c2, sigma)

    # 图像中每个像素点的坐标
    x = np.stack(np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])), axis = -1).astype(np.int32)

    # 初始化光流矢量, 沿 x 方向以及沿 y 方向
    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    # 设置高斯滤波模板
    n_flow = int(4 * sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-(xw ** 2) / (2 * sigma_flow **2 ))

    if model == "constant":
        S = np.eye(2)

    elif model in ("affine", "eight_param"):
        S = np.empty(list(x.shape) + [6 if model == "affine" else 8])

        S[..., 0, 0] = 1
        S[..., 0, 1] = x[..., 0]
        S[..., 0, 2] = x[..., 1]
        S[..., 0, 3] = 0
        S[..., 0, 4] = 0
        S[..., 0, 5] = 0

        S[..., 1, 0] = 0
        S[..., 1, 1] = 0
        S[..., 1, 2] = 0
        S[..., 1, 3] = 1
        S[..., 1, 4] = x[..., 0]
        S[..., 1, 5] = x[..., 1]

        if model == "eight_param":
            S[..., 0, 6] = x[..., 0] ** 2
            S[..., 0, 7] = x[..., 0] * x[..., 1]

            S[..., 1, 6] = x[..., 0] * x[..., 1]
            S[..., 1, 7] = x[..., 1] ** 2

    else:
        raise ValueError("参数化模型无效")

    # .swapaxes 交换轴的位置
    S_T = S.swapaxes(-1, -2)

    # 迭代 估计光流矢量
    for _ in range(num_iter):
        
        # 迭代位移
        d_ = d.astype(np.int)
        x_ = x + d_

        x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
        off_img = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        # 将 off-image 的像素点确定性置为0
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_img] = 0

        # 计算 A, B
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        # 论文中的建议 通过应用于 A 和 delB 来增加确定性
        A *= c_[..., None, None]  
        # np.dot 与 @ 作用一致  
        delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        delB *= c_[ ..., None] 

        A_T = A.swapaxes(-1, -2)
        # np.dot 与 @ 作用一致
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]

        # 如果 mu 为0，则表示不应计算全局/平均参数化扭曲，参数化应该应用于局部计算
        if mu == 0:
            G = scipy.ndimage.correlate1d(ATA, w, axis=0, mode="constant", cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

            h = scipy.ndimage.correlate1d(ATb, w, axis=0, mode="constant", cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

            d = (S @ np.linalg.solve(G, h)[..., None])[..., 0]

        # 如果 mu 不为0，则应使用它来正则化最小二乘问题，同时将背景扭曲强加到不确定的像素上
        else:
            # 计算全局参数化扭曲
            G_avg = np.mean(ATA, axis=(0, 1))
            h_avg = np.mean(ATb, axis=(0, 1))
            p_avg = np.linalg.solve(G_avg, h_avg)
            d_avg = (S @ p_avg[..., None])[..., 0]

            # mu 的默认值是将 mu 设置为 G_avg 迹的1/2
            if mu is None:
                mu = 1 / 2 * np.trace(G_avg)

            G = scipy.ndimage.correlate1d(A_T @ A, w, axis=0, mode="constant", cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)

            h = scipy.ndimage.correlate1d((A_T @ delB[..., None])[..., 0], w, axis=0, mode="constant", cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)

            # 改进光流矢量的估计
            d = np.linalg.solve(G + mu * np.eye(2), h + mu * d_avg)

    return d
