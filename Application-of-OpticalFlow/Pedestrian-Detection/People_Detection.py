import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression


def detect_people(frame):
    # detectMultiScale 函数对输入的图片img进行多尺度行人检测
    rects, weights = hog.detectMultiScale(frame, winStride = (4, 4), padding = (12, 12), scale = 1.05)

    for i in range(len(rects)):
        rects[i][2] += rects[i][0]
        rects[i][3] += rects[i][1]
    
    # 通过非极大值抑制，去除边界框重叠
    # 原理: 删除IoU大于阈值的边界框, IoU两个边界框的交集部分除以它们的并集
    rectsd = non_max_suppression(rects, probs = None, overlapThresh = 0.3)
    for (x, y, w, h) in rectsd:
        center = (x + (w - x) // 2, y + (h - y) // 2)
        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
    return frame, rects


def ShiTomasi_Corner_Detection(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners = 80, 
                          qualityLevel = 0.01,
                          minDistance = 10, 
                          blockSize=7)
    corners = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    return corners


def detect_face(frame):
    # cv2.CascadeClassifier()人脸检测级联分类器
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.samples.findFile("../haarcascades/haarcascade_frontalface_alt.xml"))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.equalizeHist()图像均衡化, 用于增强局部的对比度而不影响整体的对比度
    frame_gray = cv2.equalizeHist(frame_gray)
    # 检测人脸
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        # cv2.ellipse()绘制椭圆
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    return frame


if __name__ == '__main__':
    # 固定输出图像宽度
    output_image_width = 500
    # 视频路径
    video_path = "People.mp4"
    # 存储视频
    savevid = True
    # fps
    fps = 30
    # 提取 HOG 特征
    hog = cv2.HOGDescriptor()
    # setSVMDetector方法给用于对 HOG 特征进行分类的svm模型的系数赋值
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # 是否人脸检测
    face_detec = False
    
    # 读取视频
    camera = cv2.VideoCapture(video_path)
    _, frame = camera.read()
    
    # imutils.resize(img, width or heigth)
    # 按比例修改图像大小
    frame_resized = imutils.resize(frame, width = min(output_image_width, frame.shape[1]))
    if savevid:
        savepath = video_path.split('.')[0] + '_People_Detection' + '.avi'
        height, width, channels = frame_resized.shape
        videoOut = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

    # cv2.COLOR_BGR2GRAY
    # 转化为灰度图像
    previous_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    current_frame = 0

    color = np.random.randint(0, 255, (100, 3))
    
    # Shi-Tomasi 算法是 Harris 算法的改进,角点响应函数为 lambda1 + lambda2
    # minDistance：对于初选出的角点而言，如果在其周围 minDistance 范围内存在其他更强角点，则将此角点删除
    # blockSize：计算协方差矩阵时的窗口大小
    ShiTomasi_corner = ShiTomasi_Corner_Detection(frame_resized)
    
    motion_line = np.zeros_like(frame_resized)

    while True:
        read, frame = camera.read()
        if not read:
            break
        # 按比例修改图像大小
        frame_resized = imutils.resize(frame, width = min(output_image_width, frame.shape[1]))
        # 转化为灰度图像
        frame_resized_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        # 检测行人
        frame_processed, rects = detect_people(frame_resized)

        if face_detec == "True":
            frame_processed = detect_face(frame_processed)

        # lucas_kanade optical flow
        # winSize: 每层金字塔的搜索窗口大小
        # maxLevel: 最大的金字塔层数
        lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        ShiTomasi_corner_new, state, err = cv2.calcOpticalFlowPyrLK(previous_frame, frame_resized_gray, ShiTomasi_corner, None, **lk_params)

        # 删选好的ShiTomasi_corner
        if ShiTomasi_corner_new is not None:
            good_ShiTomasi_corner_new = ShiTomasi_corner_new[state == 1]
            good_ShiTomasi_corner = ShiTomasi_corner[state == 1]

        # 绘制轨迹
        for i, (new, old) in enumerate(zip(good_ShiTomasi_corner_new, good_ShiTomasi_corner)):
            draw = False
            # np.ravel() 将 array 数组降为一维 
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
        
            for (x, y, w, h) in rects:
                if x < x_new < w and y < y_new < h:
                    # 在行人检测框中则执行绘制
                    draw = True
            if draw:
                motion_line = cv2.line(motion_line, (int(x_new), int(y_new)), (int(x_old), int(y_old)), color[i].tolist(), 2)
                
        img = cv2.add(frame_resized, motion_line)
        cv2.imshow('frame', img)

        # 存储视频
        if savevid:
            videoOut.write(img)
   
        # 0xFF <==> 1111 1111
        # 由于同一按键对应的ASCII值不一定相同, 但是后8位一定相同, 因此只取按键对应的ASCII值的后8位来排除不同按键的干扰
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("result_"+str(current_frame)+".png", img)

        # 更新
        ShiTomasi_corner = good_ShiTomasi_corner_new.reshape(-1, 1, 2)
        previous_frame = frame_resized_gray.copy()
        current_frame += 1

    camera.release()
    cv2.destroyAllWindows()