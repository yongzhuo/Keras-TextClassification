# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/12 16:12
# @author  : Mo
# @function: radam of keras
# @codefrom: https://github.com/bojone/keras_radam


from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K


class RAdam(Optimizer):
    """RAdam optimizer.
    Default parameters follow those provided in the original Adam paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [RAdam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1908.03265)
        - [On The Variance Of The Adaptive Learning Rate And Beyond]
          (https://arxiv.org/abs/1908.03265)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)
        rho = 2 / (1 - self.beta_2) - 1
        rho_t = rho - 2 * t * beta_2_t / (1 - beta_2_t)
        r_t = K.sqrt(
            K.relu(rho_t - 4) * K.relu(rho_t - 2) * rho / ((rho - 4) * (rho - 2) * rho_t)
        )
        flag = K.cast(rho_t > 4, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            mhat_t = m_t / (1 - beta_1_t)
            vhat_t = K.sqrt(v_t / (1 - beta_2_t))
            p_t = p - lr * mhat_t * (flag * r_t / (vhat_t + self.epsilon) + (1 - flag))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))