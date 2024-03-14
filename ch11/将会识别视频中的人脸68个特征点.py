import cv2
import dlib
predictor_path = "d:\\pythoncode\\shape_predictor_68_face_landmarks.dat"
#使用dlib自带的frontal_face_detector作为人脸检测器
detector = dlib.get_frontal_face_detector()
# 使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)
#初始化窗口
win = dlib.image_window()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok,cv_img = cap.read()  
    img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)#转灰  
    # 与人脸检测程序相同,使用detector进行人脸检测,dets为返回的结果
    dets = detector(img, 0)
    shapes =[]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break
    else:
        # 使用enumerate 函数遍历序列中的元素以及它们的下标
        # 下标k即为人脸序号
        for k, d in enumerate(dets):
            # 使用predictor进行人脸关键点识别 shape为返回的结果 
            shape = predictor(img, d)
            #绘制特征点
            for index, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 1, (0,225, 0), 2) #利用cv2.putText输出1-68
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(index+1),pt_pos,font,
                            0.3, (0, 0, 255), 1, cv2.LINE_AA)
        win.clear_overlay()
        win.set_image(img)
        if len(shapes)!= 0 :
            for i in range(len(shapes)):
                win.add_overlay(shapes[i])        
        win.add_overlay(dets)       
cap.release()
cv2.destroyAllWindows()  

