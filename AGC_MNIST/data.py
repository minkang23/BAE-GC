import numpy as np
import os
import pickle
import utils

class mnist():
    def __init__(self, args):
        self.rand_seed = np.random.RandomState(args['random_seed'])
        print("Random seed: {0}".format(args['random_seed']))

        self.batch_size              = args['batch_size']
        self.drop_rate               = args['drop_rate']
        self.n_labeled               = args['n_labeled']
        self.dataset                 = args['dataset']
        self.aug_trans               = args['augment_translation']
        self.max_unlabeled_per_epoch = args['max_unlabeled_per_epoch']
        self.augment_mirror          = args['augment_mirror']

        if args['dataset'] == 'cifar-10':
            x_train, y_train, x_test, y_test \
                = load_cifar_10(args['data_dir'])
        elif args['dataset'] == 'svhn':
            x_train, y_train, x_test, y_test \
                = load_svhn(args['data_dir'])
        elif args['dataset'] == 'mnist':
            x_train, y_train, x_test, y_test\
                = load_mnist_realval(args['data_dir'])

        if args['whiten_norm'] == 'norm':
            x_train = whiten_norm(x_train)
            x_test  = whiten_norm(x_test)
        elif args['whiten_norm'] == 'zca':
            whitener = utils.ZCA(x=x_train)
            x_train = whitener.apply(x_train)
            x_test  = whitener.apply(x_test)
        else:
            print("Unkonwon input whitening mode {}".format(args['whiten_norm']))
            exit()

        p = args['augment_translation']
        if p > 0:
            x_train = np.pad(x_train, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')
            x_test = np.pad(x_test, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

        y_train = y_train
        y_test  = y_test

        # Random Shuffle.
        indices = np.arange(len(x_train))
        self.rand_seed.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Corrupt some of labels if needed.
        num_classes = len(set(y_train))
        if args['corruption_percentage'] > 0:
            corrupt_labels = int(0.01 * self.n_labeled * args['corruption_percentage'])
            corrupt_labels = min(corrupt_labels, self.n_labeled)
            print("Corrupting %d labels." % corrupt_labels)
            for i in range(corrupt_labels):
                self.y_train[i] = self.rand_seed.randint(0, num_classes)


        # Reshuffle
        indices = np.arange(len(x_train))
        self.rand_seed.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        n_xl = 28 if self.dataset is 'mnist' else 32

        # Construct mask_train. It has a zero when label is unknown, otherwise one.
        num_classes = len(set(y_train))
        max_count = self.n_labeled // num_classes
        mask_train = np.zeros((len(y_train), num_classes))
        count = [0] * num_classes
        for i in range(len(y_train)):
            label = y_train[i]
            if count[label] < max_count:
                mask_train[i, :] = np.ones(num_classes, dtype=np.float32)
            count[label] += 1

        if args['aux_tinyimg'] != "None":
            print("Augmenting with unlabeled data from tiny images dataset.")
            with open(os.path.join(args['data_dir'], 'tinyimages', 'tiny_index.pkl'),
                      'rb') as f:
                tinyimg_index = pickle.load(f)

            if args['aux_tinyimg'] == 'c100':
                print("Using all classes common with CIFAR-100.")

                with open(os.path.join(args['data_dir'], 'cifar-100', 'meta'),
                          'rb') as f:
                    cifar_labels = pickle.load(f)['fine_label_names']
                cifar_to_tinyimg = {'maple_tree': 'maple', 'aquarium_fish': 'fish'}
                cifar_labels = [
                    l if l not in cifar_to_tinyimg else cifar_to_tinyimg[l] for l in
                    cifar_labels]

                load_indices = sum(
                    [list(range(*tinyimg_index[label])) for label in cifar_labels],
                    [])
            else:
                print("Using %d random images." % args['aux_tinyimg'])

                num_all_images = max(e for s, e in tinyimg_index.values())
                load_indices = np.arange(num_all_images)
                self.rand_seed.shuffle(load_indices)
                load_indices = load_indices[:args['aux_tinyimg']]
                load_indices.sort()  # Some coherence in seeks.

            # Load the images.

            num_aux_images = len(load_indices)
            print("Loading %d auxiliary unlabeled images." % num_aux_images)
            Z_train = load_tinyimages(args['data_dir'], load_indices)

            # Whiten and pad.

            if args['whiten_inputs'] == 'norm':
                Z_train = whiten_norm(Z_train)
            # elif args['whiten_inputs'] == 'zca':
            #     Z_train = whitener.apply(Z_train)
            Z_train = np.pad(Z_train, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')

            # Concatenate to training data and append zeros to labels and mask.
            x_train = np.concatenate((x_train, Z_train))
            y_train = np.concatenate(
                (y_train, np.zeros(num_aux_images, dtype='int32')))
            mask_train = np.concatenate(
                (mask_train, np.zeros(num_aux_images, dtype='float32')))

        self.train_mask = mask_train

        self.x_train = x_train
        self.y_train = one_hot(y_train)
        self.x_test  = x_test
        self.y_test  = one_hot(y_test)

        # Add noise
        if args['corruption_images'] > 0:
            print("Corrupting %f stddev." % args['corruption_images'])
            self.x_train += self.rand_seed.normal(size=np.shape(x_train), scale=args['corruption_images'])
            self.x_test  += self.rand_seed.normal(size=np.shape(x_test), scale=args['corruption_images'])

        self.n_images   = np.shape(self.x_train)[0]
        self.n_t_images = np.shape(self.x_test)[0]

        self.labeled_idx   = np.where(self.train_mask[:, 0] == 1)[0]#np.asarray(self.labeled_idx)
        self.unlabeled_idx = np.where(self.train_mask[:, 0] == 0)[0]#np.setdiff1d(np.arange(self.n_images), self.labeled_idx)#np.random.choice(self.n_images, args['n_unlabel'], replace=False)


        self.test_mask = np.ones_like(self.y_test)

        self.sparse_label = self.train_mask * self.y_train
        self.sparse_label = np.asarray(self.sparse_label, dtype=np.float32)

        self.pseudo_label  = np.argmax(self.y_train, axis=1)
        self.pseudo_label2 = np.argmax(self.y_test, axis=1)

    def next_batch(self, is_training):
        if is_training:
            crop = self.aug_trans

            if self.max_unlabeled_per_epoch == "None":
                indices = np.arange(self.n_images)

            self.rand_seed.shuffle(indices)
            print(indices)

            n_xl = 28 if self.dataset is 'mnist' else 32
            for start_idx in range(0, self.n_images, self.batch_size):
               if start_idx + self.batch_size <= self.n_images:
                    excerpt = indices[start_idx : start_idx + self.batch_size]
                    noisy_a, noisy_b = [], []
                    for img in self.x_train[excerpt]:
                        if self.augment_mirror == "True" and self.rand_seed.uniform() > 0.5:
                            img = img[:, ::-1, :]
                        t = self.aug_trans
                        ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                        ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                        img_a = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                        ofs0 = self.rand_seed.randint(-t, t + 1) + crop
                        ofs1 = self.rand_seed.randint(-t, t + 1) + crop
                        img_b = img[ofs0:ofs0 + n_xl, ofs1:ofs1 + n_xl, :]
                        noisy_a.append(img_a)
                        noisy_b.append(img_b)

                    noisy_a = np.asarray(noisy_a)
                    noisy_b = np.asarray(noisy_b)

                    labeled_idx = np.random.choice(self.n_labeled, self.n_labeled, replace=False)
                    batch_a = np.concatenate([self.x_train[self.labeled_idx[labeled_idx], crop: crop+n_xl, crop: crop+n_xl, :], noisy_a])

                    batch_b = np.concatenate([self.x_train[self.labeled_idx[labeled_idx], crop: crop+n_xl, crop: crop+n_xl, :], noisy_b])
                    label   = np.concatenate([self.sparse_label[self.labeled_idx[labeled_idx], :], self.sparse_label[excerpt, :]])
                    mask    = np.concatenate([self.train_mask[self.labeled_idx[labeled_idx], :], self.train_mask[excerpt, :]])

                    drops_idx = np.random.choice(self.n_labeled, int(self.n_labeled * self.drop_rate),
                                                      replace=False)
                    drops = np.copy(label)
                    drops[drops_idx, :] = np.zeros(10)

                    yield len(excerpt), excerpt, batch_a, batch_b, label, drops, mask

        else:
            indices = np.arange(self.n_t_images)
            crop = self.aug_trans
            n_xl = 28 if self.dataset is 'mnist' else 32

            for start_idx in range(0, self.n_t_images, self.batch_size):
                if start_idx + self.batch_size <= self.n_t_images:
                    excerpt = indices[start_idx: start_idx + self.batch_size]

                    yield len(excerpt), self.x_test[excerpt, crop: crop+n_xl, crop: crop+n_xl, :], self.y_test[excerpt]

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

    X_train, y_train = load_svhn_files(['train_%d.pkl' % i for i in (1, 2, 3)])
    X_test, y_test = load_svhn_files('test.pkl')

    X_train = np.transpose(X_train, axes=(0, 2, 3, 1))
    X_test  = np.transpose(X_test, axes=(0, 2, 3, 1))

    return X_train, y_train, X_test, y_test

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

    return X_train, y_train, X_test, y_test

def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x

def one_hot(x):
    n_values = np.max(x) + 1

    return np.eye(n_values)[x]

def load_tinyimages(data_dir, indices, output_array=None, output_start_index=0):
    images = output_array
    if images is None:
        images = np.zeros((len(indices), 32, 32, 3), dtype='float32')
    assert (
    images.shape[0] >= len(indices) + output_start_index and images.shape[
                                                             1:] == (32, 32, 3))
    with open(os.path.join(data_dir, 'tinyimages', 'tiny_images.bin'),
              'rb') as f:
        for i, idx in enumerate(indices):
            f.seek(3072 * idx)
            images[output_start_index + i] = np.fromfile(f, dtype='uint8',
                                                         count=3072).reshape(32,
                                                                             32,
                                                                             3).transpose(
                (0, 2, 1)) / np.float32(255)
    return images