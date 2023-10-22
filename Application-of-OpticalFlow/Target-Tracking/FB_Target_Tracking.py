import numpy as np
import cv2

# 视频文件路径
vidpath = "FPS.mp4"
# 是否保存视频
savevid = True
# 帧率
fps = 30

# FarneBack 参数
pyr_scale = 0.5 # 高斯金字塔尺度
levels = 5 # 高斯金字塔层数
winsize = 15 # 窗口大小
iterations = 3 # 迭代次数
poly_n = 2 # 像素领域范围大小
poly_sigma = 1.2 # 高斯标准差
flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

# 加载视频
cap = cv2.VideoCapture(vidpath)
_, old_frame = cap.read()
old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# create black result image
# HSV颜色模型:
# 色调H: 用角度度量, 取值范围为0° ~ 360°, 从红色开始按逆时针方向计算, 红色为0°, 绿色为120°, 蓝色为240°, 之间的补色依次为黄色60°, 青色为180°, 紫色为360°
# 饱和度S: 饱和度S表示颜色接近光谱色的程度, 一种颜色可以看成是某种光谱色与白色混合的结果
#         其中光谱色所占的比例越大，颜色接近光谱色的程度就越高，颜色的饱和程度也就越高，饱和度高，颜色则深
#         光谱色的白光成分为0，饱和度达到最高, S 通常取值范围为 0%~100%，值越大，颜色越饱和
# 明度V: 明度表示颜色明亮的程度, 对于光源色, 明度值与发光体的光亮度有关
#        对于物体色，此值和物体的透射比或反射比有关, V 通常取值范围为 0%（黑）到 100%（白）
hsv_img = np.zeros_like(old_frame)
hsv_img[...,1] = 255


if savevid:
    savepath = vidpath.split('.')[0] + '_FB' + '.avi'
    height, width, channels = old_frame.shape
    videoOut = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))


while(True):
    _, new_frame = cap.read()
    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # FarneBack optical flow
    flow = cv2.calcOpticalFlowFarneback(old_frame_gray, new_frame_gray, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

    # cv2.cartToPolar 用于将直角坐标(笛卡尔坐标)转换为极坐标
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # 由光流矢量大小确定色调, 饱和度, 明度
    # cv2.normalizee(array, None, 0, 255, cv2.NORM_MINMAX) 将图片的值放缩到 0-255 之间, 便于OpenCV可视化
    hsv_img[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv_img[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv_img[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 根据光流方向设置图像色调
    # hsv_img[..., 0] = ang * 180 / np.pi / 2
    # 根据光流大小(标准化)设置图像值
    # hsv_img[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    cv2.imshow('FarneBack Optical Flow', bgr_img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    if savevid:
        videoOut.write(bgr_img)

    old_frame_gray = new_frame_gray

videoOut.release()
cv2.destroyAllWindows()