import tensorflow as tf
import numpy as np
import time
import argparse
import model
import math
import utils
import data

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

def rampdown(epoch):
    if epoch >= (args['n_epochs'] - args['rampdown_length']):
        ep = (epoch - (args['n_epochs'] - args['rampdown_length'])) * 0.5
        return math.exp(-(ep * ep) / args['rampdown_length'])
    else:
        return 1.0

args['shape'] = (28, 28, 1) if args['dataset'] == 'mnist' else (32, 32, 3)

BAE   = model.BAE(args)
mnist = data.mnist(args)

def init_params():
    for m, batch_idx, images1, images2, target, mask in mnist.next_batch(mode='train'):
        ratio = 0.0
        feed_dict = {BAE.data       : images1,
                     BAE.data2      : images2,
                     BAE.labels     : np.zeros_like(target),
                     BAE.targets    : np.zeros_like(target),
                     BAE.mask       : np.zeros_like(mask),
                     BAE.is_train   : True,
                     BAE.ratio      : ratio,
                     BAE.lr         : learning_rate,
                     BAE.adam_beta1 : adam_beta1,
                     BAE.adam_beta2 : adam_beta2,
                     BAE.ema_decay  : ema_decay,
                     BAE.hyper      : hyper,
                     BAE.is_first   : True}

        _ = sess.run(BAE.init_op, feed_dict=feed_dict)
        break

def validate_acc():
    preds = []
    preds_mc = []

    for m, images1, target in mnist.next_batch(mode='valid'):
        feed_dict = {BAE.data       : images1,
                     BAE.data2      : images1,
                     BAE.labels     : np.zeros_like(target),
                     BAE.targets    : target,
                     BAE.is_train   : False,
                     BAE.ratio      : ratio,
                     BAE.hyper      : hyper,
                     BAE.is_first   : False}

        pred, pred_mc = sess.run([BAE.test_preds,
                                  BAE.test_bae],
                                 feed_dict=feed_dict)
        preds.extend(pred)
        preds_mc.extend(pred_mc)

    preds        = np.asarray(preds)
    arg_pred     = np.argmax(preds, axis=1)
    preds_mc     = np.asarray(preds_mc)
    arg_preds_mc = np.argmax(preds_mc, axis=1)

    arg_target  = np.argmax(mnist.y_valid[:len(arg_pred)], axis=1)
    top1_acc    = np.mean(arg_target == arg_pred, dtype=np.float32)
    top1_acc_mc = np.mean(arg_target == arg_preds_mc, dtype=np.float32)

    return top1_acc, top1_acc_mc

def test_acc():
    preds = []
    preds_mc = []

    for m, images1, target in mnist.next_batch(mode='test'):
        feed_dict = {BAE.data       : images1,
                     BAE.data2      : images1,
                     BAE.labels     : np.zeros_like(target),
                     BAE.targets    : target,
                     BAE.is_train   : False,
                     BAE.ratio      : ratio,
                     BAE.hyper      : hyper,
                     BAE.is_first   : False}

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

    return top1_acc, top1_acc_mc

iter = np.arange(int(args['n_epochs']))
scaled_ratio_max = args['ratio_max']

print('\n\n[*] Start optimization\n')
with tf.Session() as sess:
    BAE.train()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tic = time.clock()
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    valid_history, test_history = [], []
    for epoch in range(int(args['n_epochs'])):
        print("\n\nEpoch {0:03d} / {1:03d}\n".format(epoch, int(args['n_epochs'])))
        rampup_value    = rampup(epoch)
        rampdown_value  = rampdown(epoch)
        learning_rate   = rampup_value * rampdown_value * args['lr_max']
        ratio           = rampup_value * scaled_ratio_max #TODO: check when the label is 100
        adam_beta1      = rampdown_value * args['adam_beta1'] + (1.0 - rampdown_value) * args['rampdown_beta1']
        hyper           = rampup_value * 1
        adam_beta2      = 0.99 if epoch < args['rampup_length'] else 0.999
        ema_decay       = 0.99 if epoch < args['rampup_length'] else 0.999

        # initialize weight normalization
        if epoch == 0:
            init_params()

        total_losses, train_n, sup_losses, graph_losses = 0., 0., 0., 0.
        pos_err, neg_err = 0., 0.
        top1_mc_t        = []
        top1_cnn_t       = []

        for _ in range(int(733. / mnist.n_labeled * mnist.batch_size)):
            for m, batch_idx, images1, images2, target, mask in mnist.next_batch(mode='train'):
                feed_dict = {BAE.data       : images1,
                             BAE.data2      : images2,
                             BAE.labels     : target,
                             BAE.targets    : target,
                             BAE.mask       : mask,
                             BAE.is_train   : True,
                             BAE.ratio      : ratio,
                             BAE.lr         : learning_rate,
                             BAE.adam_beta1 : adam_beta1,
                             BAE.adam_beta2 : adam_beta2,
                             BAE.ema_decay  : ema_decay,
                             BAE.hyper      : hyper,
                             BAE.is_first   : False,}

                _, loss_CNN, graph_loss, perturbation, loss_sup, preds_CNN, X_filled, check, check2, = sess.run([BAE.CNN_op,
                                                                                                       BAE.loss_CNN,
                                                                                                       BAE.semi_loss,
                                                                                                       BAE.entropy_kl,
                                                                                                       BAE.loss_sup,
                                                                                                       BAE.preds_labels_t,
                                                                                                       BAE.output_BAE,
                                                                                                       BAE.pos,
                                                                                                       BAE.neg],
                                                                                                      feed_dict=feed_dict)

                total_losses    += loss_CNN * m
                sup_losses      += loss_sup * m
                graph_losses    += graph_loss * m
                train_n         += m
                pos_err += np.mean(check) * m
                neg_err += np.mean(check2) * m

                arg_target = np.argmax(mnist.y_train[batch_idx, :], axis=1)
                arg_pred_cnn = np.argmax(preds_CNN[mnist.batch_size:], axis=1)
                top1_cnn_t.append(np.mean(arg_target == arg_pred_cnn, dtype=np.float32))
                top1_mc_t.append(np.mean(arg_target == X_filled[mnist.batch_size:], dtype=np.float32))

        toc = time.clock()
        print("[*] total loss:%.4e | sup loss: %.4e | perturbation-loss:%.4e | graph-loss:%.4e" % (total_losses / train_n, sup_losses / train_n, perturbation, graph_losses / train_n))
        print("[*] lr: %.7f | beta1: %f | ratio: %f" % (learning_rate, adam_beta1, ratio))
        print("[*] TRAIN Top-1 CNN Acc: %.4f | BAE Acc: %.4f" % (np.mean(top1_cnn_t), np.mean(top1_mc_t)))
        print("[*] Pos: %f, NEG: %f"%(pos_err, neg_err))

        # TEST
        top1_acc, top1_acc_mc = test_acc()
        toc = time.clock()
        print("[*] TEST  Top-1 CNN Acc: {0:.4f} | BAE Acc: {1:.4f} | Time: {2:.1f}".format(top1_acc, top1_acc_mc, toc - tic))

