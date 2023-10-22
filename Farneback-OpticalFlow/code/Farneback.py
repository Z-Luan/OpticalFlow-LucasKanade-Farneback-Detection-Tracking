import os
from functools import partial
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform

from optical_flow import flow_iterative


# Farneback
def main():
    path1 = 'data/Rapid_Minion1.jpg'
    path2 = 'data/Rapid_Minion2.jpg'

    f1 = cv2.imread(path1, 0).astype(np.double)
    f2 = cv2.imread(path2, 0).astype(np.double)

    # 构造确定性系数
    # [:, None]增加一个新维度
    # np.minimum() 取对应位置上的较小值
    # test = np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1]))
    # test.shape
    # [[  0   0   0 ...   0   0   0]
    #  [  0   1   1 ...   1   1   1]
    #  [  0   1   2 ...   2   2   2]
    #  ...
    #  [  0   1   2 ... 717 717 717]
    #  [  0   1   2 ... 718 718 718]
    #  [  0   1   2 ... 719 719 719]]
    c1 = np.minimum(1, 1 / 5 * np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(c1, 1 / 5 * np.minimum(f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None], f1.shape[1] - 1 - np.arange(f1.shape[1])))
    c2 = c1
    # c1.shape
    # [[0.  0.  0.  ... 0.  0.  0. ]
    #  [0.  0.2 0.2 ... 0.2 0.2 0. ]
    #  [0.  0.2 0.4 ... 0.4 0.2 0. ]
    #  ...
    #  [0.  0.2 0.4 ... 0.4 0.2 0. ]
    #  [0.  0.2 0.2 ... 0.2 0.2 0. ]
    #  [0.  0.  0.  ... 0.  0.  0. ]]

    # 金字层塔
    n_pyr = 4
    # Farneback 参数
    opts = dict(sigma = 4.0, sigma_flow = 4.0, num_iter = 3, model = "constant", mu = 0)
    # 存储位移
    d = None

    # 从最小的金字塔开始计算
    for pyr1, pyr2, c1_, c2_ in reversed(list(zip(*list(map(partial(skimage.transform.pyramid_gaussian, max_layer = n_pyr),[f1, f2, c1, c2]))))):
        if d is not None:
            d = skimage.transform.pyramid_expand(d, multichannel=True)
            d = d[: pyr1.shape[0], : pyr2.shape[1]]

        d = flow_iterative(pyr1, pyr2, c1 = c1_, c2 = c2_, d = d, **opts)

    # d.shape
    # (720, 1280, 2)

    # np.moveaxis()将数组的轴移到新位置

    # [[[   0    0    0 ...    0    0    0]
    #   [   1    1    1 ...    1    1    1]
    #   [   2    2    2 ...    2    2    2]
    #   ...
    #   [ 717  717  717 ...  717  717  717]
    #   [ 718  718  718 ...  718  718  718]
    #   [ 719  719  719 ...  719  719  719]]

    #  [[   0    1    2 ... 1277 1278 1279]
    #   [   0    1    2 ... 1277 1278 1279]
    #   [   0    1    2 ... 1277 1278 1279]
    #   ...
    #   [   0    1    2 ... 1277 1278 1279]
    #   [   0    1    2 ... 1277 1278 1279]
    #   [   0    1    2 ... 1277 1278 1279]]]
    xw = d + np.moveaxis(np.indices(f1.shape), 0, -1)
    # skimage.transform.warp 
    # np.moveaxis(xw, -1, 0) 将输出图像中像素点的坐标转换为输入图像中对应像素点的坐标的函数
    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval = np.nan)

    _, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    # 待截断的直方图边缘阈值
    p = 2.0 
    vmin, vmax = np.nanpercentile(f1 - f2, [p, 100 - p])
    cmap = "gray"

    axes[0, 0].imshow(f1, cmap=cmap)
    axes[0, 0].set_title("f1")
    axes[0, 1].imshow(f2, cmap=cmap)
    axes[0, 1].set_title("f2")
    # vmin , vmax与 norm 结合使用以标准化亮度数据
    axes[1, 1].imshow(f1 - f2_w, cmap = cmap, vmin = vmin, vmax = vmax)
    # axes[1, 1].imshow(f1 - f2_w, cmap = cmap)
    axes[1, 1].set_title("difference")
    plt.show()

if __name__ == "__main__":
    main()
