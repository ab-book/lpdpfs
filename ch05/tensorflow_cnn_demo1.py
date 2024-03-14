import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#定义weight函数，用于建立权重张量
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),
                       name ='W')
#定义bias函数，用于建立偏差张量
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape)
                       , name = 'b')
#定义conv2d函数，用于进行卷积运算
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],
                        padding='SAME')
#定义max_pool_2x2函数，用于建立池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')
#建立输入层
with tf.name_scope('Input_Layer'):
    x = tf.placeholder("float",shape=[None, 784]
                       ,name="x")
    #x原本一维，后续要进行卷积与池化运算，必须转换为四维张量
    x_image = tf.reshape(x, [-1, 28, 28, 1])
#建立卷积层1
with tf.name_scope('C1_Conv'):
    W1 = weight([5,5,1,16])
    b1 = bias([16])
    Conv1=conv2d(x_image, W1)+ b1
    C1_Conv = tf.nn.relu(Conv1 )
#建立池化层1
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)
#建立卷积层2
with tf.name_scope('C2_Conv'):
    W2 = weight([5,5,16,36])
    b2 = bias([36])
    Conv2=conv2d(C1_Pool, W2)+ b2
    C2_Conv = tf.nn.relu(Conv2)
#建立池化层2
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)
#建立平坦层
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])
#建立隐藏层，加入Dropout避免过度拟合
with tf.name_scope('D_Hidden_Layer'):
        W3 = weight([1764, 128])
        b3 = bias([128])
        D_Hidden = tf.nn.relu(
            tf.matmul(D_Flat, W3) + b3)
        D_Hidden_Dropout = tf.nn.dropout(D_Hidden,
                                         keep_prob=0.8)
#建立输出层
with tf.name_scope('Output_Layer'):
    W4 = weight([128,10])
    b4 = bias([10])
    y_predict= tf.nn.softmax(
                 tf.matmul(D_Hidden_Dropout,
                           W4)+b4)
#定义模型训练方式
with tf.name_scope("optimizer"):
    y_label = tf.placeholder("float", shape=[None, 10],
                             name="y_label")

    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits
        (logits=y_predict,
         labels=y_label))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) \
        .minimize(loss_function)
#定义评估模型的准确率
with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#定义训练参数
trainEpochs = 30
batchSize = 100
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
#画出准确率执行的结果
import matplotlib.pyplot as plt
plt.plot(epoch_list, accuracy_list,label="accuracy" )
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
#使用test测试数据集评估模型的准确率
print("Accuracy:",
      sess.run(accuracy,feed_dict={x: mnist.test.images,
                                   y_label: mnist.test.labels}))
#进行预测
prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x: mnist.test.images ,
                                      y_label: mnist.test.labels})

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