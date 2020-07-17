import tensorflow as tf
import numpy as np

from functools import partial
from tensorflow.contrib.framework.python.ops import add_arg_scope

class AGC:
    def __init__(self, args):
        self.dims              = args['dims']
        self.shape             = args['shape']
        self.n_labeled         = args['n_labeled']
        self.random_seed       = args['random_seed']
        self.lb_batch_size     = args['lb_batch_size']
        self.ul_batch_size     = args['ul_batch_size']
        self.mixup_sup_alpha   = args['mixup_sup_alpha']
        self.mixup_usup_alpha  = args['mixup_usup_alpha']
        self.weight_decay_coef = args['weight_decay_coef']
        self.margin            = args['margin']
        self.coeff             = args['coeff']

        self.data           = tf.placeholder(dtype=tf.float32, shape=(None, self.shape[0], self.shape[1], self.shape[2]), name='inputs')
        self.targets        = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='target_cnn')
        self.mask           = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='mask')
        self.features       = tf.placeholder(dtype=tf.float32, shape=(None, 128), name='features')
        self.is_training    = tf.placeholder(dtype=tf.bool, name='is_training')
        self.ratio          = tf.placeholder(dtype=tf.float32)
        self.lr             = tf.placeholder(dtype=tf.float32)
        self.ema_decay      = tf.placeholder(dtype=tf.float32)
        self.hyper          = tf.placeholder(dtype=tf.float32)

    def build_graph(self):
        self.batch_size = tf.shape(self.data)[0]
        self.data_lb = self.data[:self.lb_batch_size] # labeled da
        self.data_ul = self.data[self.lb_batch_size:] # unlabeled data

        # Supervised
        mixed_input, self.target_a_var, self.target_b_var, self.lam = \
            self.mixup_data_sup(self.data_lb, self.targets[:self.lb_batch_size],
                                self.mixup_sup_alpha)

        _, self.mix_lb_logits = self.vggnet(mixed_input)

        # Semi-Supervised
        self.create_ema()

        self.ul_feats, _ = self.vggnet(self.data_ul, reuse=True)
        self.ul_feats = tf.identity(self.ul_feats, name='train_features')

        feats, self.ul_ema_logits = self.vggnet(self.data, ema=self.ema, reuse=True, trainable=True)
        # self.ul_ema_logits = self.ul_ema_logits

        mixedup_x, mixedup_target, lam = \
            self.mixup_data(self.data_ul, self.ul_ema_logits[self.lb_batch_size:], self.mixup_usup_alpha)

        _, self.outputs_t = self.vggnet(mixedup_x, reuse=True)

        # Softmax
        self.preds_labels   = tf.nn.softmax(self.mix_lb_logits)
        self.preds_labels_2 = tf.nn.softmax(self.outputs_t)
        self.preds_mixedup  = tf.nn.softmax(mixedup_target)

        self.merged = tf.nn.softmax(self.ul_ema_logits)

        self.feats = tf.concat([feats, feats], axis=1)

        self.outputs_labels, self.outputs_data = self.BasisAE(self.feats)
        self.output_BAE = tf.argmax(self.outputs_labels, axis=1)

    def optimization(self):
        total_params = 0
        self.weight_decay_b = []
        vars_b = [var for var in tf.trainable_variables() if var.name.startswith("BAE")]
        for var in vars_b:
            shape = var.get_shape()
            total_params += np.prod(shape)
            self.weight_decay_b.append(tf.nn.l2_loss(var))
        print("Total # of BAE params: {}".format(total_params))

        pseudo = self.output_BAE[self.lb_batch_size:]#tf.argmax(self.merged, axis=1)

        self.hardlabel = tf.cast(tf.equal(pseudo[:self.ul_batch_size//2], pseudo[self.ul_batch_size//2:]), dtype=tf.float32)

        self.D      = tf.reduce_mean(tf.square(self.ul_feats[:self.ul_batch_size//2] - self.ul_feats[self.ul_batch_size//2:]), axis=1)#self._pairwise_distances(self.feats[:50], self.feats[50:], squared=True)
        self.D_sq   = tf.sqrt(self.D)
        self.check  = tf.reduce_sum(self.hardlabel)

        self.pos = self.D * self.hardlabel
        self.neg = (1. - self.hardlabel) * tf.square(tf.maximum(self.margin - self.D_sq, 0))
        self.semi_loss  = self.ratio * self.coeff * tf.reduce_mean(self.pos+self.neg)
        self.entropy_kl = self.ratio * 1.0 * tf.reduce_mean(tf.pow(self.preds_mixedup - self.preds_labels_2, 2))

        self.loss_cnn = self.lam * tf.losses.softmax_cross_entropy(onehot_labels=self.target_a_var, logits=self.mix_lb_logits) \
                        + (1-self.lam) * tf.losses.softmax_cross_entropy(onehot_labels=self.target_b_var, logits=self.mix_lb_logits)
        self.reg_loss = self.weight_decay_coef * tf.reduce_sum(self.weight_decay)
        self.loss_CNN = self.loss_cnn + self.entropy_kl + self.reg_loss + self.semi_loss

        CNN_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9, use_nesterov=False).minimize(self.loss_CNN, var_list=self.vars_c)
        #
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

        # ---------------------- BAE optimize ------------------------------
        self.loss_BAE = self.hyper * tf.losses.mean_squared_error(labels=self.feats,
                                                                  predictions=self.outputs_data) \
                        + tf.losses.mean_squared_error(labels=self.targets,
                                                       predictions=self.outputs_labels,
                                                       weights=self.mask)

        BAE_op = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(loss=self.loss_BAE, var_list=vars_b)
        self.train_op = tf.group([CNN_op, BAE_op, self.global_step.assign_add(1)] + update_ops)

        with tf.control_dependencies([self.train_op]):
            self.ema_op = self.ema.apply(self.vars_c)

    def train(self):
        self.bn_updates = []
        self.init_op = []
        self.build_graph()
        self.optimization()

        self.init_feats, _ = self.vggnet(self.data_lb, reuse=True, init=True)

        self.test_feats, self.test_outputs = self.vggnet(self.data, reuse=True, ema=self.ema, bn=True) #TODO: CHECK whether using ema in test mode

        self.test_feats     = tf.identity(self.test_feats, name='test_features')
        self.test_outputs   = tf.identity(self.test_outputs, name='test_outputs')
        self.test_preds     = tf.nn.softmax(self.test_outputs, name='test_preds')

    def vggnet(self, images, ema=None, bn=False, reuse=False, init=False, trainable=True, layers_mix=None):
        x = images

        with tf.variable_scope("CNN", reuse=reuse):
            vggnet_conv = partial(self.conv2d,
                                  nonlinearity=tf.nn.leaky_relu,
                                  pad='SAME',
                                  ema=ema,
                                  bn=bn,
                                  trainable=trainable)


            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=0, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=1, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=2, reuse=reuse)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=3, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=4, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[3, 3],
                            strides=[1, 1], init=init, indices=5, reuse=reuse)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])
            x = vggnet_conv(x, filters=512, kernel_size=[3, 3], strides=[1, 1],
                            pad='VALID', init=init, indices=6, reuse=reuse)
            x = vggnet_conv(x, filters=256, kernel_size=[1, 1],
                            strides=[1, 1], init=init, indices=7, reuse=reuse)
            x = vggnet_conv(x, filters=128, kernel_size=[1, 1],
                            strides=[1, 1], init=init, indices=8, reuse=reuse)

            feats = tf.reduce_mean(x, axis=[1, 2])

            outputs  = self.dense(feats, num_units=10, nonlinearity=None, init=init, ema=ema, indices=0, reuse=reuse)

        return feats, outputs

    @add_arg_scope
    def conv2d(self, x_, filters, kernel_size=[3, 3], strides=[1, 1], pad='SAME',
               nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, bn=False,
               reuse=False, indices=1, trainable=True, **kwargs):
        ''' convolutional layer '''
        name = 'conv2d_{}'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('Vector', ema, shape=kernel_size + [int(x_.get_shape()[-1]), filters], dtype=tf.float32,
                                       initializer=tf.initializers.he_uniform(seed=self.random_seed), trainable=trainable)
            g = self.get_var_maybe_avg('scaler', ema, shape=[filters], dtype=tf.float32,
                                       initializer=tf.initializers.constant(1.0), trainable=trainable)
            b = self.get_var_maybe_avg('bias', ema, shape=[filters], dtype=tf.float32,
                                       initializer=tf.initializers.variance_scaling(seed=self.random_seed, distribution='uniform', scale=0.333), trainable=trainable)
            beta = self.get_var_maybe_avg('beta', ema, shape=[filters], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0),
                                              trainable=trainable)
            gamma= self.get_var_maybe_avg('gamma', ema, shape=[filters], dtype=tf.float32,
                                               initializer=tf.constant_initializer(1.0),
                                               trainable=trainable)
            # use weight normalization (Salimans & Kingma, 2016)
            W = V * tf.reshape(g, [1, 1, 1, filters]) / tf.sqrt(tf.reduce_sum(tf.square(V), axis=(0, 1, 2), keepdims=True))

            # calculate convolutional layer output
            x = tf.nn.conv2d(x_, W, [1] + strides + [1], pad) + b

            # mean-only batch-norm.
            reuse_bn = True if ema is not None else False
            x = tf.compat.v1.layers.batch_normalization(x, training=self.is_training, momentum=0.9,
                                                        trainable=reuse_bn, scale=False, center=False, name='bn')
            x = x * gamma + beta

            if init:
                stdv = tf.sqrt(tf.reduce_sum(tf.square(V), axis=(0, 1, 2)))
                scale_init = init_scale / stdv
                self.init_op.append(g.assign(g * scale_init))

             # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x, alpha=0.1)

            return x

    @add_arg_scope
    def dense(self, x_, num_units, nonlinearity=None, init_scale=1., counters={},
              init=False, ema=None, reuse=False, indices=0, trainable=True, **kwargs):
        ''' fully connected layer '''
        name = 'dense_{}'.format(indices)
        with tf.variable_scope(name):
            V = self.get_var_maybe_avg('Vector', ema, shape=[int(x_.get_shape()[1]), num_units], dtype=tf.float32,
                                       initializer=tf.initializers.variance_scaling(seed=self.random_seed, distribution='uniform', scale=0.333), trainable=trainable)
            g = self.get_var_maybe_avg('scaler', ema, shape=[num_units], dtype=tf.float32,
                                       initializer=tf.initializers.constant(1.0), trainable=trainable)
            b = self.get_var_maybe_avg('bias', ema, shape=[num_units], dtype=tf.float32,
                                       initializer=tf.initializers.variance_scaling(seed=self.random_seed, distribution='uniform', scale=0.333), trainable=trainable)

            # use weight normalization (Salimans & Kingma, 2016)
            W = V * g / tf.sqrt(tf.reduce_sum(tf.square(V), axis=[0]))

            x = tf.matmul(x_, W) + tf.reshape(b, [1, num_units])

            if init:
                stdv = tf.sqrt(tf.reduce_sum(tf.square(V), axis=[0]))
                scale_init = init_scale / stdv
                self.init_op.append([g.assign(g * scale_init)])
            return x

    def BasisAE(self, feature, reuse=False):
        with tf.variable_scope("BAE", reuse=reuse):
            # w_init = tf.initializers.truncated_normal(mean=0, stddev=0.0316)
            w_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.random_seed)
            aug_inputs = tf.concat([tf.zeros((self.batch_size, 10)), feature], axis=1)

            # 1st hidden layer
            h = tf.layers.dense(aug_inputs, self.dims[0], activation='linear', kernel_initializer=w_init,)
            h = tf.nn.leaky_relu(h, alpha=0.1)

            # 2nd hidden layer
            h = tf.layers.dense(h, self.dims[1], activation='linear', kernel_initializer=w_init)
            h = tf.nn.sigmoid(h)

            # Output layer
            outputs_labels = tf.layers.dense(h, 10, activation='linear', use_bias=False, kernel_initializer=w_init)
            outputs_data   = tf.layers.dense(h, 256, activation='linear', use_bias=False, kernel_initializer=w_init)

        return outputs_labels, outputs_data

    def get_name(self, layer_name, counters):
        ''' utlity for keeping track of layer names '''
        if not layer_name in counters:
            counters[layer_name] = 0
        name = layer_name + '_' + str(counters[layer_name])
        counters[layer_name] += 1
        return name

    def get_var_maybe_avg(self, var_name, ema, **kwargs):
        ''' utility for retrieving polyak averaged params '''
        v = tf.get_variable(var_name, **kwargs)
        if ema is not None:
            v = ema.average(v)
        return v

    def mixup_data_sup(self, x, y, alpha):
        dist = tf.distributions.Beta(alpha, alpha)
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = dist.sample(1)#np.random.beta(alpha, alpha)
        else:
            lam = 1.
        self.lam_sup = lam
        x_shape = tf.shape(x)
        x_rev = tf.reshape(x, [x_shape[0], -1])
        index = tf.concat([x_rev, y], axis=1)
        index = tf.random.shuffle(index)
        x_rev = tf.reshape(index[:, :-10], x_shape)
        y_rev = index[:, -10:]
        mixed_x = lam * x + (1 - lam) * x_rev
        y_a, y_b = y, y_rev
        return mixed_x, y_a, y_b, lam

    def mixup_data(self, x, y, alpha):
        dist = tf.distributions.Beta(alpha, alpha)
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = dist.sample(1)#np.random.beta(alpha, alpha)
        else:
            lam = 1.
        x_shape = tf.shape(x)
        x_rev = tf.reshape(x, [x_shape[0], -1])
        index = tf.concat([x_rev, y], axis=1)
        index = index[::-1]#tf.random.shuffle(index)
        x_rev = tf.reshape(index[:, :-10], x_shape)
        y_rev = index[:, -10:]
        mixed_x = lam * x + (1 - lam) * x_rev
        mixed_y = lam * y + (1 - lam) * y_rev
        return mixed_x, mixed_y, lam
    
    def create_ema(self):
        self.global_step = tf.Variable(0.0, trainable=False, name='global_step')
        ema_decay = tf.minimum(self.ema_decay, 1.0 - 1.0 / (tf.cast(self.global_step, tf.float32) + 1.0))
        self.ema = tf.train.ExponentialMovingAverage(ema_decay)

        total_params = 0
        self.weight_decay = []
        self.vars_c = [var for var in tf.trainable_variables() if var.name.startswith("CNN")]
        for var in self.vars_c:
            shape = var.get_shape()
            print(var.op.name)
            total_params += np.prod(shape)
            # if 'Vector' in var.op.name or 'bias' in var.op.name or 'scaler' in var.op.name:
            self.weight_decay.append(tf.nn.l2_loss(var))
        print("Total # of params: {}".format(total_params))

        self.ema.apply(self.vars_c)