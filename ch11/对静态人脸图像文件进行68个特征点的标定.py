#Python 3 利用 Dlib 实现人脸 68个 特征点的标定
# 对静态人脸图像文件进行68个特征点的标定

import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('d:\\pythoncode\\shape_predictor_68_face_landmarks.dat')

# 读取图像文件
filename="d:\\testface\\testface1.jpg"
img_rd = cv2.imread(filename)
img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

# 人脸数
faces = detector(img_gray, 0)

# 待会要写的字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 标 68 个点
if len(faces) != 0:
    # 检测到人脸
    for i in range(len(faces)):
        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68 点的坐标
            pos = (point[0, 0], point[0, 1])

            # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
            cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
            # 利用 cv2.putText 写数字 1-68
            cv2.putText(img_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img_rd, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
else:
    # 没有检测到人脸
    cv2.putText(img_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

# 窗口显示
# 参数取 0 可以拖动缩放窗口，为 1 不可以
# cv2.namedWindow("image", 0)
cv2.namedWindow("image", 1)
cv2.imshow("image", img_rd)
cv2.waitKey(0)
