import tensorflow as tf
import numpy as np
import time
import argparse
import model
import math
import utils
import data
import csv

from utils import Option
opt = Option('./config.json')

utils.init()

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

args, flags = utils.parse_args(opt, parser)

tf.compat.v1.random.set_random_seed(args['random_seed'])

def rampup(epoch):
    if epoch < args['rampup_length']:
        p = max(0.0, float(epoch)) / float(args['rampup_length'])
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0

def rampup_ratio(epoch):
    if epoch < args['rampup_ratio']:
        p = max(0.0, float(epoch)) / args['rampup_ratio']
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (args['n_epochs'] - args['rampdown_length']):
        ep = (epoch - (args['n_epochs'] - args['rampdown_length'])) * 0.5
        return math.exp(-(ep * ep) / args['rampdown_length'])
    else:
        return 1.0

args['shape'] = (28, 28, 1) if args['dataset'] == 'mnist' else (32, 32, 3)

BAE   = model.BAE(args)
mnist = data.mnist(args)


unlabeled_idx = np.copy(mnist.unlabeled_idx)
labeled_idx = np.copy(mnist.labeled_idx)

sparse_label  = np.copy(mnist.sparse_label)
batch_size    = args['batch_size']

new_label = np.copy(sparse_label)
new_label = np.asarray(new_label)

mask_AE  = np.copy(mnist.train_mask)


drops_label_sup=np.ones([batch_size,10])
target_sup=np.ones([batch_size,10])
mask_sup = np.ones([batch_size,10])
mask2_sup = np.ones([batch_size,batch_size,batch_size])
mask_init_sup = np.ones([batch_size,10])
pseudo_labels_sup = np.ones([batch_size,10])




iter = np.arange(int(args['n_epochs']))
scaled_ratio_max = args['ratio_max']
scaled_ratio_max *= 1.0 * args['n_labeled'] / mnist.n_images

print('\n\n[*] Start optimization\n')
with tf.Session() as sess:
    BAE.train()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tic = time.clock()
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    for epoch in range(int(args['n_epochs'])):
        print("\n\nEpoch {0:03d} / {1:03d}\n".format(epoch, int(args['n_epochs'])))
        rampup_value    = rampup(epoch)
        rampdown_value  = rampdown(epoch)
        learning_rate   = rampup_value * rampdown_value * args['lr_max']
        ratio           = rampup_ratio(epoch) * scaled_ratio_max
        adam_beta1      = rampdown_value * args['adam_beta1'] + (1.0 - rampdown_value) * args['rampdown_beta1']
        max_iter        = np.ceil(rampup_ratio(epoch) * 1)
        hyper           = rampup_value * 1.0


        if epoch < 1:
            for m, batch_idx, images1, images2, target, drops_label, mask in mnist.next_batch(is_training=True):
                ratio = 0.0
                feed_dict = {BAE.data       : images1[mnist.n_labeled:],
                             BAE.labels     : drops_label[mnist.n_labeled:],
                             BAE.targets    : np.zeros_like(target[mnist.n_labeled:]),
                             BAE.mask       : np.zeros_like(mask[mnist.n_labeled:]),
                             BAE.is_train   : True,
                             BAE.is_train2  : False,
                             BAE.ratio      : ratio,
                             BAE.lr_c       : learning_rate,
                             BAE.adam_beta1 : adam_beta1,
                             BAE.pseudo     : np.zeros((args['batch_size'], )),
                             BAE.size       : batch_size,
                             BAE.hyper      : hyper,
                             BAE.is_offset   : True,
                             BAE.use_bae    : True}

                _ = sess.run(BAE.init_op, feed_dict=feed_dict)
                break

        # else:
        total_losses, train_n, sup_losses, graph_losses = 0., 0., 0., 0.
        pos_err, neg_err = 0., 0.
        top1_mc_t        = []
        top1_cnn_t       = []

        use_bae = False
        for m, batch_idx, images1, images2, target, drops_label, mask in mnist.next_batch(is_training=True):

            use_bae = True
            feed_dict = {BAE.data       : images1,
                         BAE.labels     : drops_label,
                         BAE.targets    : target,
                         BAE.mask       : mask,
                         BAE.is_train   : True,
                         BAE.is_train2  : True,
                         BAE.ratio      : ratio,
                         BAE.lr_c       : learning_rate,
                         BAE.lr_b       : 1e-4,
                         BAE.adam_beta1 : adam_beta1,
                         BAE.pseudo     : pseudo_labels_sup,
                         BAE.size       : batch_size,
                         BAE.hyper      : hyper,
                         BAE.is_offset  : False,
                         BAE.use_bae: use_bae}

            _,  loss_BAE,loss_CNN,graph_loss,perturbation,loss_sup,X_filled, check, check2,preds_CNN= sess.run([BAE.Tot_op,BAE.loss_BAE, BAE.loss_CNN,BAE.semi_loss,BAE.entropy_kl,BAE.loss_sup, BAE.MC_Graph,BAE.pos,BAE.neg,BAE.CNN_PRED],
                                          feed_dict=feed_dict)

            arg_pred_mc           = X_filled
            arg_target            = np.argmax(mnist.y_train[batch_idx, :], axis=1)
            top1_mc_t.append(np.mean(arg_target == arg_pred_mc, dtype=np.float32))

            total_losses    += loss_CNN * m
            sup_losses      += loss_sup * m
            graph_losses    += graph_loss * m
            train_n         += m
            pos_err += np.mean(check) * m
            neg_err += np.mean(check2) * m

            arg_pred_cnn = np.argmax(preds_CNN, axis=1)
            top1_cnn_t.append(np.mean(arg_target == arg_pred_cnn, dtype=np.float32))

        toc = time.time()
        print("[*] total loss:%.4e | sup loss: %.4e | perturbation-loss:%.4e | graph-loss:%.4e" % (total_losses / train_n, sup_losses / train_n, perturbation, graph_losses / train_n))
        print("[*] lr: %.7f | beta1: %f | ratio: %f | max_iter: %d" % (learning_rate, adam_beta1, ratio, max_iter))
        print("[*] TRAIN Top-1 CNN Acc: %.4f | BAE Acc: %.4f" % (np.mean(top1_cnn_t), np.mean(top1_mc_t)))
        print("[*] Pos: %f, NEG: %f"%(pos_err, neg_err))

        preds = []
        preds_mc = []
        preds_mc_noise = []
        for m, images1, target in mnist.next_batch(is_training=False):
            feed_dict = {BAE.data       : images1,
                         BAE.labels     : np.zeros([batch_size,10]),
                         BAE.targets    : target,
                         BAE.is_train   : False,
                         BAE.is_train2  : False,
                         BAE.ratio      : ratio,
                         BAE.size       : batch_size,
                         BAE.hyper      : hyper,
                         BAE.is_offset   : True}

            pred, pred_mc = sess.run([BAE.test_preds,
                                      BAE.test_bae],
                                     feed_dict=feed_dict)
            preds.extend(pred)
            preds_mc.extend(pred_mc)


        preds        = np.asarray(preds)
        arg_pred     = np.argmax(preds, axis=1)
        preds_mc     = np.asarray(preds_mc)
        arg_preds_mc = np.argmax(preds_mc, axis=1)


        arg_target  = np.argmax(mnist.y_test[:len(arg_pred)], axis=1)
        top1_acc    = np.mean(arg_target == arg_pred, dtype=np.float32)
        top1_acc_mc = np.mean(arg_target == arg_preds_mc, dtype=np.float32)


        toc = time.clock()
        print("[*] TEST  Top-1 CNN Acc: %.4f | BAE Acc: %.4f | Time: %.4f" % (top1_acc,top1_acc_mc,toc-tic))

        data = [epoch,  np.mean(top1_mc_t), top1_acc]
        csvfile = open('output.csv', 'a')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(data)
        csvfile.close()


