import argparse


class RequireInCmdline(object):
    pass


def _default(vals, key):
    v = vals.get(key)
    return None if v is RequireInCmdline else v


def _required(vals, key):
    return vals.get(key) is RequireInCmdline

def parse_args(init_vals, custom_parser=None):
    if custom_parser is None:
        f = argparse.ArgumentDefaultsHelpFormatter
        p = argparse.ArgumentParser(formatter_class=f)
    else:
        p = custom_parser

    p.add_argument('--data_dir',
                   default=_default(init_vals, 'data_dir'),
                   required=_required(init_vals, 'data_dir'),
                   help='Path to dataset')

    p.add_argument('--log_dir',
                   default=_default(init_vals, 'log_dir'),
                   required=_required(init_vals, 'log_dir'),
                   help="""Directory in which to write training
                   summarizes and checkpoints.""")

    p.add_argument('--dataset',
                   default=_default(init_vals, 'dataset'),
                   required=_required(init_vals, 'dataset'),
                   help='Types of dataset')

    p.add_argument('--whiten_norm', choices=['norm', 'zca'],
                   default=_default(init_vals, 'whiten_norm'),
                   required=_required(init_vals, 'whiten_norm'),
                   help='Methodology for whitening')

    p.add_argument('--augment_mirror', choices=['True', 'False'],
                   default=_default(init_vals, 'augment_mirror'),
                   required=_required(init_vals, 'augment_mirror'),
                   help='Whether or not to augment mirror images')

    p.add_argument('--augment_translation', type=int,
                   default=_default(init_vals, 'augment_translation'),
                   required=_required(init_vals, 'augment_translation'),
                   help="The number of augmentation")

    p.add_argument('-e', '--n_epochs', type=int,
                   default=_default(init_vals, 'n_epochs'),
                   required=_required(init_vals, 'n_epochs'),
                   help="Number of epochs to run")

    p.add_argument('-b', '--batch_size', type=int,
                   default=_default(init_vals, 'batch_size'),
                   required=_required(init_vals, 'batch_size'),
                   help="Size of each minibatch.")

    p.add_argument('--n_labeled', type=int,
                   default=_default(init_vals, 'n_labeled'),
                   required=_required(init_vals, 'n_labeled'),
                   help="The number of labels")

    p.add_argument('--precision', choices=['fp32', 'fp16'],
                   default=_default(init_vals, 'precision'),
                   required=_required(init_vals, 'precision'),
                   help="Select single of half precision arithmetic.")

    p.add_argument('--l2_lambda', type=float,
                   default=_default(init_vals, 'l2_lambda'),
                   required=_required(init_vals, 'l2_lambda'),
                   help='L2 regularization parameters.')

    p.add_argument('--margin', type=float,
                   default=_default(init_vals, 'margin'),
                   required=_required(init_vals, 'margin'),
                   help='Margin for Siamme Net.')

    p.add_argument('--mixup_sup_alpha', type=float,
                   default=_default(init_vals, 'mixup_sup_alpha'),
                   required=_required(init_vals, 'mixup_sup_alpha'),
                   help='mixup_sup_alpha')

    p.add_argument('--mixup_usup_alpha', type=float,
                   default=_default(init_vals, 'mixup_usup_alpha'),
                   required=_required(init_vals, 'mixup_usup_alpha'),
                   help='mixup_usup_alpha')

    p.add_argument('--coeff', type=float,
                   default=_default(init_vals, 'coeff'),
                   required=_required(init_vals, 'coeff'),
                   help='Coefficient for SNTG Loss.')

    p.add_argument('--lr_max', type=float,
                   default=_default(init_vals, 'lr_max'),
                   required=_required(init_vals, 'lr_max'),
                   help='Learning rate')

    p.add_argument('--ratio_max', type=float,
                   default=_default(init_vals, 'ratio_max'),
                   required=_required(init_vals, 'ratio_max'),
                   help='Learning rate')

    p.add_argument('--display_every', type=int,
                   default=_default(init_vals, 'display_every'),
                   required=_required(init_vals, 'display_every'),
                   help='How often to print out information')

    p.add_argument('--random_seed', type=int,
                   default=_default(init_vals, 'random_seed'),
                   required=_required(init_vals, 'random_seed'),
                   help='random seed')

    p.add_argument('-d', '--dims', '--list', nargs='+',
                   default=_default(init_vals, 'dims'),
                   required=True,
                   help='The number of units')

    FLAGS, unknown_args = p.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    vals = init_vals
    vals['data_dir'] = FLAGS.data_dir
    del FLAGS.data_dir
    vals['log_dir'] = FLAGS.log_dir
    del FLAGS.log_dir
    vals['n_epochs'] = FLAGS.n_epochs
    del FLAGS.n_epochs
    vals['batch_size'] = FLAGS.batch_size
    del FLAGS.batch_size
    vals['precision'] = FLAGS.precision
    del FLAGS.precision
    vals['l2_lambda'] = FLAGS.l2_lambda
    del FLAGS.l2_lambda
    vals['lr_max'] = FLAGS.lr_max
    del FLAGS.lr_max
    vals['display_every'] = FLAGS.display_every
    del FLAGS.display_every
    vals['random_seed'] = FLAGS.random_seed
    del FLAGS.random_seed
    vals['dims'] = FLAGS.dims
    del FLAGS.dims
    vals['dataset'] = FLAGS.dataset
    del FLAGS.dataset
    vals['whiten_norm'] = FLAGS.whiten_norm
    del FLAGS.whiten_norm
    vals['augment_mirror'] = FLAGS.augment_mirror
    del FLAGS.augment_mirror
    vals['augment_translation'] = FLAGS.augment_translation
    del FLAGS.augment_translation
    vals['n_labeled'] = FLAGS.n_labeled
    del FLAGS.n_labeled
    vals['ratio_max'] = FLAGS.ratio_max
    del FLAGS.ratio_max
    vals['margin'] = FLAGS.margin
    del FLAGS.margin
    vals['mixup_sup_alpha'] = FLAGS.mixup_sup_alpha
    del FLAGS.mixup_sup_alpha
    vals['mixup_usup_alpha'] = FLAGS.mixup_usup_alpha
    del FLAGS.mixup_usup_alpha
    vals['coeff'] = FLAGS.coeff
    del FLAGS.coeff

    return vals, FLAGS
