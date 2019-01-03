import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
batch_size = 8
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None 可以方便使用不同的batch大小
# 在训练的时候需要把数据分成比较小的batch
# 但是在调试的时候可以一次性使用全部的数据
# 当数据集比较小的时候这样做比较方便测试
# 数据集比较大时候将大量数据放入一个batch可能会导致内存溢出
x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_= tf.placeholder(tf.float32, shape=(None,1), name='y-input')

# 定义神经网络向前传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


# 创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(w1))
	print(sess.run(w2))
	STEPS = 5000
	for i in range(STEPS):
		start = (i * batch_size) % 128
		end = (i*batch_size) % 128 + batch_size
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if i % 1000 == 0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
			print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
		
			
	print(sess.run(w1))
	print(sess.run(w2))	