#正则化可以缓解过度拟合，在loss总引入模型复杂度指标，给w加权值，弱化了数据的噪声，一般不正则化b
#loss=loss(y与y_)+REGULARIZER*loss(w)
#loss=模型中所有参数的损失函数，如交叉熵、均方误差 + REGULARIZER给出参数w在总loss中的比例，即正则化的权重 * loss(需要正则化的参数)

#loss(w)有两种计算方法：loss(w)=tf.contrib.layers.l1_regularizer(REGULARIZER)(w)
# loss(w)=tf.contrib.layers.l2_regularizer(REGULARIZER)(w)

# tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
# loss=cem+tf.add_n(tf.get_collection('losses'))

# 直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
# X[0,:]是指第0行所有数据