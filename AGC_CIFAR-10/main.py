import tensorflow as tf
import numpy as np
import time
import math
import argparse
import model
import utils
import data
import os

from utils import Option
opt = Option('./config.json')

utils.init()

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

args, flags = utils.parse_args(opt, parser)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.set_random_seed(args['random_seed'])

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

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        lr = 1.0
    else:
        lr = current / rampup_length

    return lr

def adjust_lr(epoch, step_in_epoch, total_steps_in_epoch):
    lr = args['lr_max']
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr = linear_rampup(epoch, 0) * (lr)

    if args['lr_rampdown_epochs']:
        assert args['lr_rampdown_epochs'] >= args['n_epochs']
        lr *= cosine_rampdown(epoch, args['lr_rampdown_epochs'])

    return lr

args['shape'] = (28, 28, 1) if args['dataset'] == 'mnist' else (32, 32, 3)

mnist = data.mnist(args)
AGC   = model.AGC(args)

batch_size       = args['lb_batch_size']
scaled_ratio_max = args['ratio_max']
# scaled_ratio_max *= 1.0 * args['ratio_max']# args['n_labeled'] / mnist.n_images

save_path = os.path.join(args['log_dir'], args['dataset'], 'random_seed_{}'.format(args['random_seed']))
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('\n\n[*] Start optimization\n')
with tf.Session() as sess:
    AGC.train()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    tic = time.time()
    is_first = True
    valids_acc, test_top1_acc, best_acc = [], [], 0.0
    for epoch in range(int(args['n_epochs'])):
        print("\n\nEpoch {0:03d} / {1:03d}\n".format(epoch, int(args['n_epochs'])))
        rampup_value    = rampup(epoch)
        rampdown_value  = rampdown(epoch)
        learning_rate   = rampup_value * rampdown_value * args['lr_max']
        ratio           = rampup_value * scaled_ratio_max
        hyper           = rampup_value * 1.0
        ema_decay       = 0.999

        if epoch < 1:
            for m, batch_idx, images1, target, mask in mnist.next_batch(mode='train'):
                feed_dict = {AGC.data        : images1,
                             AGC.targets     : np.zeros_like(target),
                             AGC.mask        : np.zeros_like(mask),
                             AGC.ratio       : ratio,
                             AGC.lr          : learning_rate,
                             AGC.ema_decay   : ema_decay,
                             AGC.hyper       : hyper,
                             AGC.is_training : False}

                _, _ = sess.run([AGC.init_feats, AGC.init_op], feed_dict=feed_dict)
                is_first = False
                break

        total_losses, train_n, sup_losses, graph_losses, pertb_losses = 0., 0., 0., 0., 0.
        pos_err, neg_err, n_hardlabel = 0., 0., 0.
        top1_accs, top1_bae_accs = [], []
        batch_iter = 0.0

        for _ in range(int(450. / mnist.n_labeled * mnist.batch_size)):
            for m, batch_idx, images1, target, mask in mnist.next_batch(mode='train'):
                ratio = rampup(epoch + batch_iter / 440.0) * scaled_ratio_max
                learning_rate = adjust_lr(epoch, batch_iter, 440)
                feed_dict = {AGC.data        : images1,
                             AGC.targets     : target,
                             AGC.mask        : mask,
                             AGC.ratio       : ratio,
                             AGC.lr          : learning_rate,
                             AGC.ema_decay   : ema_decay,
                             AGC.hyper       : hyper,
                             AGC.is_training : True}

                _, loss_CNN, graph_loss, perturbation, sup_loss, pred_cnn, pos, neg, n_hard, reg_loss, pred_bae\
                    = sess.run([AGC.ema_op, AGC.loss_CNN, AGC.semi_loss, AGC.entropy_kl, AGC.loss_cnn, AGC.merged,
                                AGC.pos, AGC.neg, AGC.check, AGC.reg_loss, AGC.output_BAE], feed_dict=feed_dict)

                arg_target = np.argmax(mnist.y_train[batch_idx, :], axis=1)
                arg_pred   = np.argmax(pred_cnn, axis=1)
                top1_accs.append(np.mean(arg_target == arg_pred[mnist.batch_size:], dtype=np.float32))
                top1_bae_accs.append(np.mean(arg_target == pred_bae[mnist.batch_size:], dtype=np.float32))
                # print(reg_loss * 1e4)
                total_losses += loss_CNN * m
                sup_losses   += sup_loss * m
                graph_losses += graph_loss * m
                pertb_losses += perturbation * m
                train_n      += m
                pos_err      += np.mean(pos) * m
                neg_err      += np.mean(neg) * m
                n_hardlabel  += n_hard
                batch_iter   += 1.0

        toc = time.time()

        # print(check)
        print("[*] total loss:%.4e | sup loss: %.4e | perturbation-loss:%.4e | graph-loss:%.4e" % (total_losses / train_n, sup_losses / train_n, pertb_losses / train_n, graph_losses / train_n))
        print("[*] lr: %.7f | ratio: %f | time: %f" % (learning_rate, ratio, toc-tic))
        print("[*] TRAIN Top-1 CNN Acc: %.4f | BAE Acc: %.4f" % (np.mean(top1_accs), np.mean(top1_bae_accs)))
        print("[*] Pos: %f, NEG: %f, N_hard: %f"%(pos_err, neg_err, n_hardlabel))
        saver.save(sess, save_path + '/cnn_model')

        # VALIDATION
        preds = []
        preds_mc = []

        for m, images1, target in mnist.next_batch(mode='validation'):
            feed_dict = {AGC.data          : images1,
                         AGC.is_training   : False}

            pred, pred_mc = sess.run([AGC.test_preds, AGC.output_BAE], feed_dict=feed_dict)
            preds.extend(pred)
            preds_mc.extend(pred_mc)

        preds        = np.asarray(preds)
        arg_pred     = np.argmax(preds, axis=1)
        arg_preds_mc = preds_mc

        arg_target  = np.argmax(mnist.y_valid[:len(preds)], axis=1)
        top1_acc    = np.mean(arg_target == arg_pred, dtype=np.float32)
        top1_acc_mc = np.mean(arg_target == arg_preds_mc, dtype=np.float32)

        print("[*] VALID  Top-1 CNN Acc: {0:.4f} | BAE Acc: {1:.4f} | Time: {2:.1f}".format(top1_acc, top1_acc_mc, toc - tic))
        valids_acc.append(top1_acc)
        best_valid_idx = np.argmax(valids_acc)

        # ------------------------- TEST -------------------------
        preds = []
        preds_mc = []
        feats = []
        for m, images1, target in mnist.next_batch(mode='test'):
            feed_dict = {AGC.data          : images1,
                         AGC.is_training   : False}

            feat, pred, pred_mc = sess.run([AGC.test_feats, AGC.test_preds, AGC.output_BAE], feed_dict=feed_dict)
            preds.extend(pred)
            preds_mc.extend(pred_mc)
            feats.extend(feat)
        preds       = np.asarray(preds)
        arg_pred    = np.argmax(preds, axis=1)
        arg_pred_mc = preds_mc

        arg_target  = np.argmax(mnist.y_test, axis=1)
        top1_acc    = np.mean(arg_target == arg_pred, dtype=np.float32)
        top1_acc_mc = np.mean(arg_target == arg_pred_mc, dtype=np.float32)
        test_top1_acc.append(top1_acc)
        print("[*] TEST   Top-1 CNN Acc: {0:.4f} | BAE Acc: {1:.4f} | Time: {2:.1f}".format(top1_acc, top1_acc_mc, toc - tic))

        if best_acc < top1_acc:
            best_acc = top1_acc
            best_feats = feats

        print("[*] BEST is {0}-th results, Acc: {1:0.4f}".format(best_valid_idx, test_top1_acc[best_valid_idx]))

    np.savetxt("best_feats.tsv", best_feats, delimiter="\t")