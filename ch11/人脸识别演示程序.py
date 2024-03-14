# -*- coding: UTF-8 -*-
import face_recognition
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw,ImageFont
# 在你的电脑摄像头上实时运行人脸识别
video_capture = cv2.VideoCapture(0)
# 加载示例图片并学习如何识别它。
path ="d:\\facelib\\"#在同级目录下的images文件中放需要被识别出的人物图
total_image=[]
total_image_name=[]
total_face_encoding=[]
for fn in os.listdir(path): #fn 表示的是文件名
  total_face_encoding.append(face_recognition.face_encodings
                             (face_recognition.load_image_file(path+fn))[0])
  fn=fn[:(len(fn)-4)]#截取图片名（这里应该把images文件中的图片名命名为为人物名）
  total_image_name.append(fn)#图片名字列表
while True:
  # 抓取一帧视频
  ret, frame = video_capture.read()			#捕获一帧图片
  small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)#将图片缩小1/4，为人脸识别提速
  rgb_small_frame = small_frame[:, :, ::-1]	#将opencv的BGR格式转为RGB格式
  # 发现在视频帧所有的脸和face_enqcodings
  #face_locations = face_recognition.face_locations(frame)
  #face_encodings = face_recognition.face_encodings(frame, face_locations)
  #face_locations = face_recognition.face_locations(rgb_small_frame)
  face_locations = face_recognition.face_locations(rgb_small_frame)
  face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
  # 在这个视频帧中循环遍历每个人脸
  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      top *= 4				  #还原人脸的原始尺寸
      right *= 4
      bottom *= 4
      left *= 4
    
      # 看看面部是否与已知人脸相匹配。
      for i,v in enumerate(total_face_encoding):
          match = face_recognition.compare_faces([v], face_encoding,tolerance=0.42)
          name = "Unknown"
          if match[0]:
              name = total_image_name[i]
              break
      
      # 画出一个框，框住脸    
      cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)    
      # 画出一个带名字的标签，放在框下
      img_PIL=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))#转换图片格式   						
      position = (left + 6, bottom - 6) 								#指定文字输出位置
      draw = ImageDraw.Draw(img_PIL)
      font1 = ImageFont.truetype('simhei.ttf', 20) 
      draw.text((20,20),'按Q键退出',font=font1,fill=(255,255,255))
      font2 = ImageFont.truetype('simhei.ttf', 40) #加载字体
      draw.text(position, name, font=font2, fill=(255, 255, 255)) #绘制文字
      frame = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR) #将图片转回OpenCV格式     
  # 显示结果图像
  cv2.imshow('Video', frame)
  # 按q退出
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
# 释放摄像头中的流
video_capture.release()
cv2.destroyAllWindows()
