
"""
reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None)

第一个参数input_tensor： 输入的待降维的tensor;
第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
第四个参数name： 操作的名称;
第五个参数 reduction_indices：在以前版本中用来指定轴，已弃用;

"""
import tensorflow as tf

x = [[1, 2, 3],
     [1, 2, 3]]

xx = tf.cast(x, tf.float32)

mean_all = tf.reduce_mean(xx, keep_dims=False)
mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)

with tf.Session() as sess:
    m_a, m_0, m_1 = sess.run([mean_all, mean_0, mean_1])

print(m_a)   # output: 2.0

print(m_0 ) # output: [ 1.  2.  3.] ，列的平均值

print(m_1)  # output:  [ 2.  2.]，行的平均值



















