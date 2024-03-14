import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#建立layer函数
def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs
#建立输入层（x），输入的数字图像是784像素
x = tf.placeholder("float", [None, 784])
#建立隐藏层h1,隐藏层神经元个数1000，输入层的神经元个数784
h1=layer(output_dim=1000,input_dim=784,
         inputs=x ,activation=tf.nn.relu)
#建立隐藏层h2,隐藏层神经元个数1000，这层的输入是h1
h2=layer(output_dim=1000,input_dim=1000,
         inputs=h1 ,activation=tf.nn.relu)
#建立输出层，输出层的神经元个数是10，隐藏层h2是它的输入
y_predict=layer(output_dim=10,input_dim=1000,
                inputs=h2,activation=None)
#建立训练数据label真实值的placeholder
y_label = tf.placeholder("float", [None, 10])
#定义损失函数，使用交叉熵
loss_function = tf.reduce_mean(
                   tf.nn.softmax_cross_entropy_with_logits
                       (logits=y_predict ,
                        labels=y_label))
#定义优化器算法，使用AdamOptiomizer设置learning_rate=0.001
optimizer = tf.train.AdamOptimizer(learning_rate=0.001) \
                    .minimize(loss_function)
#定义评估模型准确率的方式
#计算每一项数据是否预测正确
correct_prediction = tf.equal(tf.argmax(y_label  , 1),
                              tf.argmax(y_predict, 1))
#计算预测正确结果的平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#tensorflow必须编写程序代码来控制训练的每一个过程
#执行15个训练周期，每一批次项数位100
trainEpochs = 15
batchSize = 100
#每个训练周期所需要执行批次=训练数据项数/每一批次项数
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];accuracy_list=[];loss_list=[];
from time import time
startTime=time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#进行训练
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y_label: batch_y})
    #使用验证数据计算准确率
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images,
                                    y_label: mnist.validation.labels})

    epoch_list.append(epoch)
    loss_list.append(loss);
    accuracy_list.append(acc)

    print("Train Epoch:", '%02d' % (epoch + 1), \
          "Loss=", "{:.9f}".format(loss), " Accuracy=", acc)
duration = time() - startTime
print("Train Finished takes:", duration)
#画出准确率的执行结果
import matplotlib.pyplot as plt
plt.plot(epoch_list, accuracy_list,label="accuracy" )
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
#评估模型准确率
print("Accuracy:", sess.run(accuracy,
                           feed_dict={x: mnist.test.images,
                                      y_label: mnist.test.labels}))
#进行预测
prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x: mnist.test.images })
print("查看预测结果的前10项数据")
print(prediction_result[:10])
print("查看预测结果的第248项数据")
print(prediction_result[247])
print("查看测试数据的第248项数据")
import numpy as np
print(np.argmax(mnist.test.labels[248]))
#找出预测错误的
print("找出预测错误的")
for i in range(500):
    if prediction_result[i]!=np.argmax(mnist.test.labels[i]):
        print("i="+str(i)+
              "   label=",np.argmax(mnist.test.labels[i]),
              "predict=",prediction_result[i])
#保存模型
saver = tf.train.Saver()
save_path = saver.save(sess, "saveModel/tensorflow_mlp_model1")
print("Model saved in file: %s" % save_path)
#关闭会话
sess.close()