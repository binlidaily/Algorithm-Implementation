# from tensorflow.keras.datasets.fashion_mnist
from tensorflow.examples.tutorials.mnist import input_data
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

class LeNetRaw:
    def __init__(self):
        pass

    def weights_init(self, shape):
        """
        shape size: [height, width, n_channels, n_filters], [卷积核长，卷积核宽，通道数，卷积核个数也就是输出通道数]
        """
        initial = tf.truncated_normal(shape, stddev=0.1)    # 变量的初始值为截断正太分布
        return tf.Variable(initial)
    
    def bias_init(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def train(self):
        # mnist = tf.keras.datasets.fashion_mnist.load_data()
        mnist = input_data.read_data_sets('datasets/MNIST/', one_hot=True)   # 读取数据集

        x = tf.placeholder(tf.float32, [None, 784])
        x_img = tf.reshape(x, [-1, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])

        # 第一个 block
        W_conv1 = self.weights_init([5, 5, 1, 32])
        b_conv1 = self.bias_init([32])
        a_conv1 = tf.nn.elu(self.conv2d(x_img, W_conv1) + b_conv1)
        o_pool1 = self.max_pool(a_conv1)

        # 第二个 block
        W_conv2 = self.weights_init([5, 5, 32, 64])
        b_conv2 = self.bias_init([64])
        a_conv2 = tf.nn.elu(self.conv2d(o_pool1, W_conv2) + b_conv2)
        o_pool2 = self.max_pool(a_conv2)

        # 第三个 block
        W_fc1 = self.weights_init([7 * 7 * 64, 1024])
        b_fc1 = self.bias_init([1024])
        o_pool2_flat = tf.reshape(o_pool2, [-1, 7 * 7 * 64])
        o_fc1 = tf.nn.elu(tf.matmul(o_pool2_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob)

        # 第四个 block
        W_fc2 = self.weights_init([1024, 10])
        b_fc2 = self.bias_init([10])
        y_conv = tf.nn.softmax(tf.matmul(o_fc1_drop, W_fc2) + b_fc2)

        # loss
        corss_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(corss_entropy)
        correct_preds = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        # run
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())

        max_iters = 20000

        for i in range(max_iters):
            batch = mnist.train.next_batch(50)
            if i % 200 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        print('Final test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
    LeNet = LeNetRaw()
    LeNet.train()