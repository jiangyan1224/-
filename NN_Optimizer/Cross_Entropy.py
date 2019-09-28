# 交叉熵ce:表征两个概率分布之间的距离：H(y_,y)=-求和(y_*logy)
# 如：已知y_=(1,0) 预测y1=(0.6,0.4) y2=(0.8,0.2)
# H1((1,0).(0.6,0.4))=0.222
# H2((1,0).(0.8,0.2))=0.097 所以H2预测更准
# ce=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-12,1.0)))
# y如果小于1e-12,取值1e-12;如果大于1，取值1.0

# n分类，每次都有n个输出：y1~yn，表示第n种情况出现的可能性大小
# 所以要求y1~yn之和为1，每个yi值大于等于0小于等于1：
# 实际程序中得到的n个输出，通过softmax方法符合上述的概率分布要求
# tensorflow中有:对输出进行softmax处理，满足概率分布后，再与标准答案求交叉熵的方法，也就是可以代替上面那行计算ce的式子
# ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
# cem=tf.reduce_mean(ce)
# 计算结果的含义是当前计算出的预测值与标准答案的差距，也就是损失函数
