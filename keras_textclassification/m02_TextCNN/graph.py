# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/3 10:51
# @author   :Mo
# @function :graph of base


from keras.layers import Reshape, Concatenate, Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

from keras_textclassification.base.graph import graph


class TextCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        embedding = self.word_embedding.output
        embedding_reshape = Reshape((self.len_max, self.embed_size, 1))(embedding)
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        for filter in self.filters:
            conv = Conv2D(filters = self.filters_num,
                          kernel_size = (filter, self.embed_size),
                          padding = 'valid',
                          kernel_initializer = 'normal',
                          activation = 'tanh',
                          )(embedding_reshape)
            pooled = MaxPool2D(pool_size = (self.len_max - filter + 1, 1),
                               strides = (1, 1),
                               padding = 'valid',
                               )(conv)
            conv_pools.append(pooled)
        # 拼接
        x = Concatenate(axis=-1)(conv_pools)
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        x = Dense(units=64, activation='tanh')(x)
        x = Dropout(self.dropout)(x)
        output = Dense(units=self.label, activation=self.activate_classify)(x)
        self.model = Model(inputs=self.word_embedding.input, outputs=output)
        self.model.summary(120)



    # def focal_loss(self, gamma=2, alpha=0.75): # 0.25, 0.5
    def focal_loss(self, gamma=2, alpha=0.75, batch_size=None, label_num=None, epsilon=1.e-7, multi_dim=False, use_softmax=True):
        from tensorflow.python.ops import array_ops
        import keras.backend as K
        import tensorflow as tf
        def focal_loss_fixed(y_true, y_pred):  # with tensorflow
            eps = 1e-12
            y_pred = K.clip(y_pred, eps, 1. - eps)  # improve the stability of the focal loss and see issues 1 for more information
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            return loss

        def focal_loss_all(prediction_tensor, target_tensor):
            r"""Compute focal loss for predictions.
                Multi-labels Focal loss formula:
                    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                         ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
            Args:
             prediction_tensor: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted logits for each class
             target_tensor: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
             weights: A float tensor of shape [batch_size, num_anchors]
             alpha: A scalar tensor for focal loss alpha hyper-parameter
             gamma: A scalar tensor for focal loss gamma hyper-parameter
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
            """
            sigmoid_p = tf.nn.sigmoid(prediction_tensor)
            zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

            # For poitive prediction, only need consider front part loss, back part is 0;
            # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
            pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

            # For negative prediction, only need consider back part loss, front part is 0;
            # target_tensor > zeros <=> z=1, so negative coefficient = 0.
            neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
            per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                                  - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
            return tf.reduce_sum(per_entry_cross_ent)

        def focal_loss_category(logits, labels):
            '''
            :param logits:  [batch_size, n_class]
            :param labels: [batch_size]  not one-hot !!!
            :return: -alpha*(1-y)^r * log(y)
            它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
            logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

            怎么把alpha的权重加上去？
            通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

            是否需要对logits转换后的概率值进行限制？
            需要的，避免极端情况的影响

            针对输入是 (N，P，C )和  (N，P)怎么处理？
            先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

            bug:
            ValueError: Cannot convert an unknown Dimension to a Tensor: ?
            因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

            '''

            if multi_dim:
                logits = tf.reshape(logits, [-1, logits.shape[2]])
                labels = tf.reshape(labels, [-1])

            # (Class ,1)
            alpha = tf.constant([0.5]*batch_size, dtype=tf.float32)

            labels = tf.argmax(labels) #
            labels = tf.cast(labels, dtype=tf.int32)
            logits = tf.cast(logits, tf.float32)
            if use_softmax:
                # (N,Class) > N*Class
                softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
            else:
                softmax = tf.reshape(tf.nn.sigmoid(logits), [-1])  # [batch_size * n_class]
            # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
            # labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
            labels_shift = tf.range(0, label_num) * batch_size + labels
            # (N*Class,) > (N,)
            prob = tf.gather(softmax, labels_shift)
            # 预防预测概率值为0的情况  ; (N,)
            prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
            # (Class ,1) > (N,)
            alpha_choice = tf.gather(alpha, labels)
            # (N,) > (N,)
            weight = tf.pow(tf.subtract(1., prob), gamma)
            weight = tf.multiply(alpha_choice, weight)
            # (N,) > 1
            loss = -tf.reduce_sum(tf.multiply(weight, tf.log(prob)))
            return loss

        return focal_loss_fixed


    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return:
        """
        from keras_textclassification.keras_layers.keras_radam import RAdam
        from keras.optimizers import Adam
        # self.model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
        #                    loss=[self.focal_loss(alpha=.25, gamma=2)],
        #                    metrics=['accuracy'])

        self.model.compile(optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                           loss=[self.focal_loss(alpha=.25, gamma=2)], # self.loss, #
                           # loss_weights=[0.6, 0.5],
                           # loss=[self.focal_loss(gamma=2, alpha=0.25, batch_size=self.batch_size, label_num=self.label, epsilon=1.e-7, multi_dim=False, use_softmax=False)],
                           # loss=[self.focal_loss(gamma=2, alpha=0.75)],
                           metrics=['accuracy'])  # Any optimize


