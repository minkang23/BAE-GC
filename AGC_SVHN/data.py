import numpy as np
import os
import pickle
import utils
import scipy.io
import collections

class mnist():
    def __init__(self, args):
        self.rand_seed = np.random.RandomState(args['random_seed'])
        print("Random seed: {0}".format(args['random_seed']))

        self.batch_size              = args['batch_size']
        self.ul_batch_size           = args['ul_batch_size']
        self.drop_rate               = args['drop_rate']
        self.n_labeled               = args['n_labeled']
        self.dataset                 = args['dataset']
        self.aug_trans               = args['augment_translation']
        self.augment_mirror          = args['augment_mirror']

        if args['dataset'] == 'cifar-10':
            x_train, y_train, x_test, y_test, x_valid, y_valid \
                = load_cifar_10(args['data_dir'])
        elif args['dataset'] == 'svhn':
            x_train, y_train, x_test, y_test, x_valid, y_valid \
                = load_svhn(args['data_dir'])
        elif args['dataset'] == 'mnist':
            x_train, y_train, x_test, y_test\
                = load_mnist_realval(args['data_dir'])

        if args['whiten_norm'] == 'norm':
            x_train = whiten_norm(x_train)
            x_test  = whiten_norm(x_test)
            x_valid = whiten_norm(x_valid)
        elif args['whiten_norm'] == 'zca':
            whitener = utils.ZCA(x=x_train)
            x_train = whitener.apply(x_train)
            x_test  = whitener.apply(x_test)
            x_valid = whitener.apply(x_valid)
        else:
            print("Unkonwon input whitening mode {}".format(args['whiten_norm']))
            exit()

        p = args['augment_translation']
        if p > 0:
            x_train = np.pad(x_train, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')
            x_test  = np.pad(x_test, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')
            x_valid = np.pad(x_valid, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

        # Random Shuffle.
        indices = np.arange(len(x_train))
        self.rand_seed.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]

        # Corrupt some of labels if needed.
        num_classes = len(set(y_train))

        # Construct mask_train. It has a zero when label is unknown, otherwise one.
        max_count = self.n_labeled // num_classes
        mask_train = np.zeros((len(y_train), num_classes))
        count = [0] * num_classes
        for i in range(len(y_train)):
            label = y_train[i]
            if count[label] < max_count:
                mask_train[i, :] = np.ones(num_classes, dtype=np.float32)
            count[label] += 1

        self.train_mask = mask_train

        self.x_train = x_train
        self.y_train = one_hot(y_train)
        self.x_test  = x_test
        self.y_test  = one_hot(y_test)
        self.x_valid = x_valid
        self.y_valid = one_hot(y_valid)

        self.n_images   = np.shape(self.x_train)[0]
        self.n_t_images = np.shape(self.x_test)[0]

        self.labeled_idx = np.where(self.train_mask[:, 0] == 1)[0]
        self.unlabeled_idx = np.where(self.train_mask[:, 0] == 0)[0]

        self.lx_train = self.x_train[self.labeled_idx, :, :, :]
        self.ly_train = self.y_train[self.labeled_idx, :]
        self.ulx_train = self.x_train[self.unlabeled_idx, :, :, :]
        self.uly_train = self.y_train[self.unlabeled_idx, :]

        self.sparse_label = self.train_mask * self.y_train
        self.sparse_label = np.asarray(self.sparse_label, dtype=np.float32)

    def next_batch(self, mode='train'):
        if mode == 'train':
            crop = self.aug_trans
            n_images = len(self.lx_train)
            indices = np.arange(n_images)

            self.rand_seed.shuffle(indices)

            n_xl = 28 if self.dataset is 'mnist' else 32
            for start_idx in range(0, self.n_labeled, self.batch_size):  # self.num_iter_per_epoch):
                excerpt = indices[start_idx: start_idx + self.batch_size]
                noisy_la, noisy_lb = [], []
                for img in self.lx_train[excerpt]:
                    if self.augment_mirror == "True" and self.rand_seed.uniform() > 0.5:
                        img = img[:, ::-1, :]
                    t = self.aug_trans
                    ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                    ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                    img_a = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                    ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                    ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                    img_b = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                    noisy_la.append(img_a)
                    noisy_lb.append(img_b)

                noisy_ua, noisy_ub = [], []
                idx = np.random.choice(self.n_images, self.ul_batch_size, replace=False)
                for img in self.x_train[idx]:
                    if self.augment_mirror == "True" and self.rand_seed.uniform() > 0.5:
                        img = img[:, ::-1, :]
                    t = self.aug_trans
                    ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                    ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                    img_a = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                    ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                    ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                    img_b = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                    noisy_ua.append(img_a)
                    noisy_ub.append(img_b)

                images_a = np.concatenate([np.asarray(noisy_la), np.asarray(noisy_ua)], axis=0)
                images_b = np.concatenate([np.asarray(noisy_lb), np.asarray(noisy_ub)], axis=0)
                labels   = np.concatenate([self.ly_train[excerpt, :], np.zeros((self.ul_batch_size, 10))], axis=0)
                masks    = np.concatenate([np.ones((self.batch_size, 10)), np.zeros((self.ul_batch_size, 10))], axis=0)
                yield len(excerpt), idx, images_a, images_b, labels, masks

        elif mode == 'test':
            indices = np.arange(self.n_t_images)
            crop = self.aug_trans
            n_xl = 28 if self.dataset is 'mnist' else 32

            for start_idx in range(0, self.n_t_images, self.batch_size):
                if start_idx + self.batch_size <= self.n_t_images:
                    excerpt = indices[start_idx: start_idx + self.batch_size]
                    yield len(excerpt), self.x_test[excerpt, crop: crop + n_xl, crop: crop + n_xl, :], self.y_test[excerpt]

        else:
            n_images = len(self.x_valid)
            indices = np.arange(n_images)
            crop = self.aug_trans
            n_xl = 28 if self.dataset is 'mnist' else 32

            for start_idx in range(0, n_images, self.batch_size):
                if start_idx + self.batch_size <= self.n_t_images:
                    excerpt = indices[start_idx: start_idx + self.batch_size]

                    yield len(excerpt), self.x_valid[excerpt, crop: crop + n_xl, crop: crop + n_xl, :], self.y_valid[excerpt]

def load_mnist_realval(data_path):
    from six.moves import urllib
    import gzip
    path = os.path.join(data_path, 'mnist', 'mnist.pkl.gz')
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)

        def download_dataset(url, path):
            print('Downloading data from %s' % url)
            urllib.request.urlretrieve(url, path)

        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    x_train, y_train = train_set[0], train_set[1]
    x_valid, y_valid = valid_set[0], valid_set[1]
    x_test, y_test = test_set[0], test_set[1]
    x_train = np.vstack([x_train, x_valid]).astype('float32')
    y_train = np.hstack([y_train, y_valid]).astype('int32')
    x_train = x_train.reshape([-1, 28, 28, 1])
    x_test = x_test.reshape([-1, 28, 28, 1]).astype('float32')
    y_test = y_test.astype('int32')
    return x_train, y_train, x_test, y_test

def load_svhn(data_path):
    import cPickle
    import tempfile
    from six.moves import urllib
    URLS = 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat'
    def _load_svhn():
        splits = collections.OrderedDict()
        for split in ['train', 'test']:
            with tempfile.NamedTemporaryFile() as f:
                urllib.request.urlretrieve(URLS.format(split), f.name)
                data_dict = scipy.io.loadmat(f.name)
            dataset = {}
            dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2]) / np.float32(255)
            dataset['labels'] = data_dict['y'].reshape((-1))
            # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
            dataset['labels'] -= 1
            splits[split] = dataset
        return splits

    def load_svhn_files(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            with open(os.path.join(data_path, 'svhn', fn), 'rb') as f:
                X, y = cPickle.load(f)
            images.append(np.asarray(X, dtype='float32') / np.float32(255))
            labels.append(np.asarray(y, dtype='int32'))
        return np.concatenate(images), np.concatenate(labels)

    # X_train, y_train = load_svhn_files(['train_%d.pkl' % i for i in (1, 2, 3)])
    # X_test, y_test = load_svhn_files('test.pkl')
    dataset = _load_svhn()
    X_train, y_train = dataset['train']['images'], dataset['train']['labels']
    X_test, y_test = dataset['test']['images'], dataset['test']['labels']

    # X_train = np.transpose(X_train, axes=(0, 2, 3, 1))
    # X_test  = np.transpose(X_test, axes=(0, 2, 3, 1))

    valid_idx = np.random.choice(len(X_train), 1000, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_train)), valid_idx)
    X_valid = X_train[valid_idx, :, :, :]
    X_train = X_train[train_idx, :, :, :]
    y_valid = y_train[valid_idx]
    y_train = y_train[train_idx]

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def load_cifar_10(data_path):
    import cPickle
    def load_cifar_batches(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            with open(os.path.join(data_path, 'cifar-10', fn), 'rb') as f:
                data = cPickle.load(f)
            images.append(
                np.asarray(data['data'], dtype='float32').reshape(-1, 3, 32,
                                                                  32) / np.float32(
                    255))
            labels.append(np.asarray(data['labels'], dtype='int32'))
        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = load_cifar_batches(
        ['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
    X_test, y_test = load_cifar_batches('test_batch')

    valid_idx = np.random.choice(len(X_train), 1000, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_train)), valid_idx)
    X_valid = X_train[valid_idx, :, :, :]
    X_train = X_train[train_idx, :, :, :]
    y_valid = y_train[valid_idx]
    y_train = y_train[train_idx]

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x

def one_hot(x):
    n_values = np.max(x) + 1

    return np.eye(n_values)[x]