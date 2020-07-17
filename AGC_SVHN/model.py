import tensorflow as tf
import numpy as np
from functools import partial
from tensorflow.contrib.framework.python.ops import add_arg_scope


class BAE:
    def __init__(self, args):
        self.dims            = args['dims']
        self.shape           = args['shape']
        self.n_label         = args['n_labeled']
        self.random_seed     = args['random_seed']
        self.coef_emb        = args['coef_emb']
        self.logit_dist_cost = args['logit_dist_cost']
        self.margin          = args['margin']

        self.data         = tf.placeholder(dtype=tf.float32, shape=(None, self.shape[0], self.shape[1], self.shape[2]), name='inputs')
        self.data2        = tf.placeholder(dtype=tf.float32, shape=(None, self.shape[0], self.shape[1], self.shape[2]), name='inputs')
        self.labels       = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='labels')
        self.targets      = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='target_cnn')
        self.mask         = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='mask')
        self.is_train     = tf.placeholder(dtype=tf.bool)
        self.ratio        = tf.placeholder(dtype=tf.float32)
        self.hyper        = tf.placeholder(dtype=tf.float32)
        self.is_first     = tf.placeholder(dtype=tf.bool)
        self.lr           = tf.placeholder(dtype=tf.float32)
        self.adam_beta1   = tf.placeholder(dtype=tf.float32)
        self.adam_beta2   = tf.placeholder(dtype=tf.float32)
        self.ema_decay    = tf.placeholder(dtype=tf.float32)

    def build_graph(self):

        self.data_s = tf.cond(self.is_train,
                              lambda: self.data + tf.random.normal(tf.shape(self.data), mean=0.0, stddev=0.15),
                              lambda: self.data)

        self.data_t = self.data2 + tf.random.normal(tf.shape(self.data2), mean=0.0, stddev=0.15)
        self.feats_s, self.outputs_s, self.outputs_s_2 = self.vggnet(self.data_s)

        global_step = tf.Variable(0.0, name='global_step', dtype=tf.float32, trainable=False)
        self.global_step_op = global_step.assign_add(1.0)

        self.ema    = tf.train.ExponentialMovingAverage(self.ema_decay, zero_debias=True, num_updates=global_step)

        self.vars_c = [var for var in tf.trainable_variables() if var.name.startswith("CNN")]

        self.ema_op = self.ema.apply(self.vars_c)



        self.feats_t, self.outputs_t, _ = self.vggnet(self.data_t, reuse=True, ema=self.ema)

        self.preds_labels_s = tf.nn.softmax(self.outputs_s)
        self.preds_labels_t = tf.nn.softmax(self.outputs_t)

        self.feats = tf.concat([self.feats_s, self.feats_t], axis=1)

        self.outputs_labels, self.outputs_data = self.BasisAE(self.feats)
        self.output_BAE = tf.argmax(self.outputs_labels, axis=1)

    def optimization(self):
        total_params = 0
        for var in self.vars_c:
            shape = var.get_shape()
            name  = var.op.name
            print('{} | {}'.format(name, shape))
            total_params += np.prod(shape)
        print("Total # of CNN params: {}\n".format(total_params))

        total_params = 0
        vars_b = [var for var in tf.trainable_variables() if var.name.startswith("BAE")]
        for var in vars_b:
            shape = var.get_shape()
            total_params += np.prod(shape)
        print("Total # of BAE params: {}".format(total_params))

        pseudo = self.output_BAE

        batch_size = tf.shape(self.data)[0]
        self.hardlabel = tf.cast(tf.equal(pseudo[:batch_size//2], pseudo[batch_size//2:]), dtype=tf.float32)

        self.D = tf.reduce_mean(tf.square(self.feats_s[:batch_size//2] - self.feats_s[batch_size//2:]), axis=1)
        self.D_sq = tf.sqrt(self.D)

        self.pos = self.D * self.hardlabel
        self.neg = (1. - self.hardlabel) * tf.square(tf.nn.relu(self.margin - self.D_sq))
        self.semi_loss  = self.ratio * self.coef_emb * tf.reduce_mean(self.pos + self.neg)
        self.entropy_kl = self.ratio * 1.0 * tf.reduce_mean(tf.pow(self.preds_labels_s - self.preds_labels_t, 2))
        self.res_loss = self.logit_dist_cost * tf.reduce_mean(tf.pow(self.outputs_s - self.outputs_s_2, 2))

        self.loss_sup = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=self.targets,
                                            logits=self.outputs_s,
                                            reduction=tf.losses.Reduction.NONE) \
            * self.mask[:, 0])


        self.loss_CNN = self.loss_sup + self.entropy_kl + self.semi_loss + self.res_loss

        CNN_op = self.adam_updates(params=self.vars_c, cost_or_grads=self.loss_CNN, lr=self.lr,
                                   mom1=self.adam_beta1, mom2=self.adam_beta2)

        # BAE Op
        c_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss_BAE = self.hyper * tf.losses.mean_squared_error(labels=self.feats,
                                                     predictions=self.outputs_data) \
                        + tf.losses.mean_squared_error(labels=self.targets,
                                                       predictions=self.outputs_labels,
                                                       weights=self.mask) \
                        + c_reg

        BAE_op = self.adam_updates(params=vars_b, cost_or_grads=self.loss_BAE, lr=1e-4, mom1=self.adam_beta1,
                                   mom2=self.adam_beta2)

        self.CNN_op = tf.group([CNN_op, BAE_op, self.ema_op, self.global_step_op] + self.bn_updates)

    def train(self):
        self.bn_updates = []
        self.init_op = []
        self.build_graph()
        self.optimization()

        self.test_feats, self.test_outputs, _ = self.vggnet(self.data, reuse=True, ema=self.ema)
        self.init_feats, _, _ = self.vggnet(self.data_s, reuse=True, init=True)
        self.test_preds = tf.nn.softmax(self.test_outputs)

        self.test_bae   = tf.identity(self.outputs_labels)
        
    def vggnet(self, images, ema=None, reuse=False, init=False):
        with tf.variable_scope("CNN", reuse=reuse):
            vggnet_conv = partial(self.conv2d,
                                  nonlinearity=tf.nn.leaky_relu,
                                  padding='SAME',
                                  ema=ema)

            x = vggnet_conv(images, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=0, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=1, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=2, reuse=reuse)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
            x = tf.layers.dropout(x, training=self.is_train)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=3, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=4, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=5, reuse=reuse)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
            x = tf.layers.dropout(x, training=self.is_train)
            x = vggnet_conv(x, filters=512, kernel_size=[3, 3], strides=[1, 1],
                            pad='VALID', init=init, indices=6, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[1, 1],
                            strides=[1, 1], init=init, indices=7, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[1, 1],
                            strides=[1, 1], init=init, indices=8, reuse=reuse)

            feats = tf.reduce_mean(x, axis=[1, 2])

            outputs  = self.dense(feats, num_units=10, nonlinearity=None, init=init, ema=ema, indices=0, reuse=reuse)
            outputs2 = self.dense(feats, num_units=10, nonlinearity=None, init=init, ema=ema, indices=1, reuse=reuse)

        return feats, outputs, outputs2

    def BasisAE(self, feature, reuse=False):
        with tf.variable_scope("BAE", reuse=reuse):

            w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.random_seed)
            aug_inputs = tf.concat([tf.zeros_like(self.labels), feature], axis=1)

            # 1st hidden layer
            h = tf.layers.dense(aug_inputs, self.dims[0], activation='linear', kernel_initializer=w_init)
            h = tf.nn.leaky_relu(h, alpha=0.1)

            # 2nd hidden layer
            h = tf.layers.dense(h, self.dims[1], activation='linear', kernel_initializer=w_init)
            h = tf.nn.tanh(h)

            # Output layer
            outputs_labels = tf.layers.dense(h, 10, activation='linear', use_bias=False, kernel_initializer=w_init)

            outputs_data = tf.layers.dense(h, 128*2, activation='linear', use_bias=False, kernel_initializer=w_init)

        return outputs_labels, outputs_data

    @add_arg_scope
    def conv2d(self, x_, filters, kernel_size=[3, 3], strides=[1, 1], pad='SAME',
               nonlinearity=None, init_scale=1., counters={}, init=False, ema=None,
               reuse=False, indices=1, **kwargs):
        ''' convolutional layer '''
        name = 'conv2d_{}'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('V', ema, shape=kernel_size + [int(x_.get_shape()[-1]), filters], dtype=tf.float32,
                                       initializer=tf.initializers.random_normal(stddev=0.05, seed=self.random_seed), trainable=True)
            g = self.get_var_maybe_avg('g', ema, shape=[filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.), trainable=True)
            b = self.get_var_maybe_avg('b', ema, shape=[filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.), trainable=True)
            self.batch_mean = tf.get_variable('batch_mean', shape=[filters], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0),
                                              trainable=False)
            # use weight normalization (Salimans & Kingma, 2016)
            W = V * tf.reshape(g, [1, 1, 1, filters]) / tf.sqrt(tf.reduce_sum(tf.square(V), axis=(0, 1, 2), keepdims=True))

            # calculate convolutional layer output
            x = tf.nn.conv2d(x_, W, [1] + strides + [1], pad)

            # mean-only batch-norm.
            if ema is not None:
                x = x - self.batch_mean
            else:
                m = tf.reduce_mean(x, axis=[0, 1, 2])
                x = x - tf.reshape(m, [1, 1, 1, filters])
                bn_update = self.batch_mean.assign(self.ema_decay*self.batch_mean + (1.0-self.ema_decay)*m)

                # x, bn_update, m_init = mean_only_batch_norm()
                if not reuse:
                    self.bn_updates.append(bn_update)

                if init:
                    stdv = tf.sqrt(tf.reduce_mean(tf.square(x), axis=(0, 1, 2)))
                    scale_init = init_scale / stdv
                    x *= tf.reshape(scale_init, [1, 1, 1, filters])
                    self.init_op.append(g.assign(g * scale_init))

            x = tf.nn.bias_add(x, b)

             # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x, alpha=0.1)

            return x

    @add_arg_scope
    def dense(self, x_, num_units, nonlinearity=None, init_scale=1., counters={},
              init=False, ema=None, reuse=False, indices=0, **kwargs):
        ''' fully connected layer '''
        # name = self.get_name('dense', counters)
        name = 'dense_{}'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('V', ema, shape=[int(x_.get_shape()[1]), num_units], dtype=tf.float32,
                                       initializer=tf.initializers.random_normal(stddev=0.05, seed=self.random_seed),
                                       trainable=True)
            g = self.get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.), trainable=True)
            b = self.get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                                       initializer=tf.initializers.constant(0., dtype=tf.float32), trainable=True)
            self.batch_mean = tf.get_variable('batch_mean', shape=[num_units], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0),
                                              trainable=False)
            # use weight normalization (Salimans & Kingma, 2016)
            W = V * g / tf.sqrt(tf.reduce_sum(tf.square(V), axis=[0]))

            x = tf.matmul(x_, W)

            if ema is not None:
                x = x - self.batch_mean
            else:
                m = tf.reduce_mean(x, axis=[0])
                x = x - m
                bn_update = self.batch_mean.assign(self.ema_decay * self.batch_mean + (1.0 - self.ema_decay) * m)
                # x, bn_update = mean_only_batch_norm()
                if not reuse:
                    self.bn_updates.append(bn_update)

                if init:
                    stdv = tf.sqrt(tf.reduce_mean(tf.square(x), axis=[0]))
                    scale_init = init_scale / stdv
                    x /= stdv
                    self.init_op.append([g.assign(g * scale_init)])

            # apply nonlinearity
            # if nonlinearity is not None:
            #     x = nonlinearity(x)
            return x + tf.reshape(b, [1, num_units])


    def get_name(self,layer_name, counters):
        ''' utlity for keeping track of layer names '''
        if not layer_name in counters:
            counters[layer_name] = 0
        name = layer_name + '_' + str(counters[layer_name])
        counters[layer_name] += 1
        return name

    def get_var_maybe_avg(self,var_name, ema, **kwargs):
        ''' utility for retrieving polyak averaged params '''
        v = tf.get_variable(var_name, **kwargs)
        if ema is not None:
            v = ema.average(v)
        return v
    
    def adam_updates(self, params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
        ''' Adam optimizer '''
        updates = []
        if type(cost_or_grads) is not list:
            grads = tf.gradients(cost_or_grads, params)
        else:
            grads = cost_or_grads

        t = tf.Variable(1., 'adam_t', dtype=tf.float32)
        coef = lr * tf.sqrt(1 - mom2 ** t) / (1 - mom1 ** t)
        for p, g in zip(params, grads):
            mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')

            mg_t = mom1 * mg + (1. - mom1) * g
            v_t = mom2 * v + (1. - mom2) * tf.square(g)
            g_t = mg_t / (tf.sqrt(v_t) + 1e-8)
            p_t = p - coef * g_t

            updates.append(mg.assign(mg_t))
            updates.append(v.assign(v_t))
            updates.append(p.assign(p_t))
        updates.append(t.assign_add(1))
        return tf.group(*updates)