#####MNIST_可视化################################
import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
index=3
#载入MNIST数据集，如果指定地址即当前执行的目录是否有MNIST_data目前以及是否已经有文件，还没有，会下载数据
# 加载MNIST数据集
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
image=np.reshape(mnist.train.images[index],[28,-1])
print(mnist.train.labels[index])  #显示label
plt.imshow(image, cmap=plt.get_cmap('gray_r'))  #画图
plt.show()
