import tensorflow as tf
import numpy as np
from functools import partial
from tensorflow.python import control_flow_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope


class BAE:
    def __init__(self, args):
        self.height = 784
        self.width = 784
        self.l2_lambda   = args['l2_lambda']
        self.dims        = args['dims']
        self.ema_decay   = args['ema_decay']
        self.shape       = args['shape']
        self.n_label     = args['n_labeled']
        self.random_seed = args['random_seed']
        self.coef_emb    = args['coef_emb']
        self.large_net   = True if args['large_net'] == 'True' else False
        self.global_pool = [5, 5] if args['dataset'] == 'mnist' else [6, 6]

        self.data       = tf.placeholder(dtype=tf.float32, shape=(None, self.shape[0], self.shape[1], self.shape[2]), name='inputs')
        self.labels     = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='labels')
        self.targets    = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='target_cnn')
        self.mask       = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='mask')

        self.is_train     = tf.placeholder(dtype=tf.bool)
        self.is_train2    = tf.placeholder(dtype=tf.bool)
        self.ratio        = tf.placeholder(dtype=tf.float32)
        self.hyper        = tf.placeholder(dtype=tf.float32)
        self.size         = tf.placeholder(dtype=tf.float32)
        self.pseudo       = tf.placeholder(dtype=tf.int64, shape=(None))
        self.is_offset    = tf.placeholder(dtype=tf.bool)
        self.use_bae      = tf.placeholder(dtype=tf.bool)
        self.lr_c          = tf.placeholder(dtype=tf.float32)
        self.lr_b          = tf.placeholder(dtype=tf.float32)
        self.adam_beta1   = tf.placeholder(dtype=tf.float32)

    def build_graph(self):
        self.offset = tf.cond(self.is_offset,lambda:0,lambda:self.n_label)
        self.data_1 = tf.cond(self.is_train,
                              lambda: self.data + tf.random.normal(tf.shape(self.data), mean=0.0, stddev=0.15),
                              lambda: self.data)
        self.data_noise = self.data + tf.random.normal(tf.shape(self.data), mean=0.0, stddev=0.15)

        self.feats, self.outputs = self.vggnet(self.data_1[self.offset:])

        self.feats_corrupted, self.outputs_corrupted = self.vggnet(self.data_noise, reuse=True)

        self.preds_labels = tf.nn.softmax(self.outputs)
        self.preds_labels_corrupted = tf.nn.softmax(self.outputs_corrupted)

        self.CNN_PRED = self.preds_labels_corrupted[self.offset:]

        self.ema = tf.train.ExponentialMovingAverage(self.ema_decay, zero_debias=True)

        self.margin = 1.0

        total_params = 0
        vars_c = [var for var in tf.trainable_variables() if var.name.startswith("CNN")]
        for var in vars_c:
            shape = var.get_shape()
            print(shape)
            total_params += np.prod(shape)
        print("Total # of CNN params: {}".format(total_params))

        ema_op = self.ema.apply(vars_c)

        self.train_outputs_labels, self.train_outputs_data = self.BasisAE(self.feats_corrupted)
        self.train_output_BAE = tf.argmax(self.train_outputs_labels, axis=1)

        self.test_feats, self.test_outputs = self.vggnet(self.data_1, reuse=True, ema=self.ema)
        self.init_feats, _ = self.vggnet(self.data_1, reuse=True, init=True)

        self.outputs_labels, self.outputs_data = self.BasisAE(self.test_feats, reuse=True)
        self.output_BAE = tf.argmax(self.outputs_labels, axis=1)


        total_params = 0
        vars_b = [var for var in tf.trainable_variables() if var.name.startswith("BAE")]
        for var in vars_b:
            shape = var.get_shape()
            print(shape)
            total_params += np.prod(shape)
        print("Total # of BAE params: {}".format(total_params))


        pseudo = self.train_output_BAE[self.offset:]

        self.MC_Graph = self.train_output_BAE[self.offset:]
        batch_size = tf.shape(self.feats)[0]
        self.hardlabel = tf.cast(tf.equal(pseudo[:batch_size // 2], pseudo[batch_size // 2:]), dtype=tf.float32)

        self.D = tf.reduce_mean(tf.square(self.feats[:batch_size // 2] - self.feats[batch_size // 2:]), axis=1)
        self.D = tf.clip_by_value(self.D, clip_value_min=1e-16, clip_value_max=10)
        self.D_sq = tf.sqrt(self.D)

        self.pos = self.D * self.hardlabel
        self.neg = (1. - self.hardlabel) * tf.square(tf.nn.relu(self.margin - self.D_sq))
        self.semi_loss = self.ratio * self.coef_emb * tf.reduce_mean(self.pos + self.neg)
        self.entropy_kl = self.ratio * 1.0 * tf.reduce_mean(tf.pow(self.preds_labels - self.CNN_PRED, 2))

        self.loss_sup = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=self.targets[self.offset:],
                                            logits=self.outputs,
                                            reduction=tf.losses.Reduction.NONE) \
            * self.mask[self.offset:, 0])

        self.loss_CNN = self.loss_sup + self.entropy_kl + self.semi_loss

        CNN_op = self.adam_updates(params=vars_c, cost_or_grads=self.loss_CNN, lr=self.lr_c, mom1=self.adam_beta1)
        self.CNN_op = tf.group([CNN_op, ema_op] + self.bn_updates)

        c_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss_BAE = tf.reduce_mean(tf.pow(self.targets - self.train_outputs_labels * self.mask, 2)) \
                        + self.hyper * tf.reduce_mean(tf.pow(self.feats_corrupted - self.train_outputs_data, 2)) \
                        + c_reg

        self.BAE_op = tf.train.AdamOptimizer(self.lr_b).minimize(self.loss_BAE, var_list=[vars_b])
        self.Tot_op = tf.group([self.CNN_op, self.BAE_op])

        self.test_preds = tf.nn.softmax(self.test_outputs)
        self.test_bae = tf.identity(self.outputs_labels)

    def train(self):
        self.bn_updates = []
        self.init_op = []
        self.build_graph()

        
    def vggnet(self, images, ema=None, reuse=False, init=False):
        with tf.variable_scope("CNN", reuse=reuse):
            vggnet_conv = partial(self.conv2d,
                                  nonlinearity=tf.nn.leaky_relu,
                                  padding='SAME',
                                  ema=ema)

            if self.large_net:
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
                outputs = self.dense(feats, num_units=10, nonlinearity=None, init=init, ema=ema, indices=0, reuse=reuse)
            else:
                x = vggnet_conv(images, filters=32, kernel_size=[5, 5], strides=[1, 1],
                                pad='VALID', init=init, indices=0, reuse=reuse)
                x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
                x = tf.layers.dropout(x, training=self.is_train)
                x = vggnet_conv(x, filters=64, kernel_size=[3, 3],
                                strides=[1, 1], init=init, indices=1, reuse=reuse)
                x = vggnet_conv(x, filters=64, kernel_size=[3, 3],
                                strides=[1, 1], init=init, indices=2, reuse=reuse)
                x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
                x = tf.layers.dropout(x, training=self.is_train)
                x = vggnet_conv(x, filters=128, kernel_size=[3, 3], pad='VALID',
                                strides=[1, 1], init=init, indices=3, reuse=reuse)
                x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                                strides=[1, 1], init=init, indices=4, reuse=reuse)
                feats = tf.reduce_mean(x, axis=[1, 2])
                outputs = self.dense(feats, num_units=10, nonlinearity=None, init=init, ema=ema, indices=0, reuse=reuse)

        return feats, outputs

    def BasisAE(self, feature, reuse=False):
        with tf.variable_scope("BAE", reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.random_seed)
            aug_inputs = tf.concat([self.labels, feature], axis=1)

            # 1st hidden layer
            h = tf.layers.dense(aug_inputs, self.dims[0], activation='linear',
                                kernel_initializer=w_init,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda))
            h = tf.nn.leaky_relu(h, alpha=0.1)


            # 2nd hidden layer
            h = tf.layers.dense(h, self.dims[1], activation='linear',
                                kernel_initializer=w_init,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda))
            h = tf.nn.leaky_relu(h, alpha=0.1)


            # 3rd hidden layer
            h = tf.layers.dense(h, self.dims[2], activation='linear',
                                kernel_initializer=w_init,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda))
            h = tf.nn.tanh(h)

            # Output layer
            outputs_labels = tf.layers.dense(h, 10, activation='linear',
                                             use_bias=False,
                                             kernel_initializer=w_init,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda))

            outputs_data = tf.layers.dense(h, 128, activation='linear',
                                                use_bias=False,
                                                kernel_initializer=w_init,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_lambda))

        return  outputs_labels, outputs_data

    @add_arg_scope
    def conv2d(self, x_, filters, kernel_size=[3, 3], strides=[1, 1], pad='SAME',
               nonlinearity=None, init_scale=1., counters={}, init=False, ema=None,
               reuse=False, indices=1, **kwargs):
        ''' convolutional layer '''
        name = 'conv2d_{}'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('V', ema, shape=kernel_size + [int(x_.get_shape()[-1]), filters], dtype=tf.float32,
                                       initializer=tf.initializers.he_normal(seed=self.random_seed), trainable=True)
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
        name = 'dense_'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('V', ema, shape=[int(x_.get_shape()[1]), num_units], dtype=tf.float32,
                                       initializer=tf.initializers.variance_scaling(seed=self.random_seed),
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