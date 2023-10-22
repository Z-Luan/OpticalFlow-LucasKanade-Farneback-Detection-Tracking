import cv2

video_path = 'Pen.mp4'
IMG_name = video_path.split('.')[0]
cap = cv2.VideoCapture(video_path)
for i in range (300):
    _, frame = cap.read()
    frame_name = IMG_name + str(i) + '.jpg'
    cv2.imwrite('./'+ frame_name, frame)
