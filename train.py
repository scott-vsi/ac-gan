# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

from sugartensor.sg_data import _data_to_tensor


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
cat_dim = 10   # total categorical factor
con_dim = 2    # total continuous factor
rand_dim = 38  # total random latent dimension


class ByYaw(object):
    _data_dir = '/data/by_yaw'

    def __init__(self, batch_size=32):
      is_pan = True
      # load dataset
      (X_train, y_train), (X_test, y_test) = load_data(nb_images_per_label=10000, is_pan=is_pan)
      nb_train = int(len(y_train)*0.9)
      X_valid, y_valid = X_train[nb_train:,...], y_train[nb_train:]
      X_train, y_train = X_train[:nb_train,...], y_train[:nb_train]

      if is_pan:
        X_train = np.expand_dims(X_train, axis=3)
        X_valid = np.expand_dims(X_valid, axis=3)
        X_test = np.expand_dims(X_test, axis=3)

      X_train, y_train = X_train.astype(np.float32) / 255.0, np.array(y_train, dtype=np.int32)
      X_valid, y_valid = X_valid.astype(np.float32) / 255.0, np.array(y_valid, dtype=np.int32)
      X_test,  y_test  = X_test.astype(np.float32)  / 255.0, np.array(y_test,  dtype=np.int32)

      # member initialize
      self.train, self.valid, self.test = tf.sg_opt(), tf.sg_opt, tf.sg_opt()

      # convert to tensor queue
      self.train.image, self.train.label = _data_to_tensor([X_train, y_train], batch_size, name='train')
      self.valid.image, self.valid.label = _data_to_tensor([X_valid, y_valid], batch_size, name='valid')
      self.test.image, self.test.label = _data_to_tensor([X_test, y_test], batch_size, name='test')

      # calc total batch count
      self.train.num_batch = y_train.shape[0] // batch_size
      self.valid.num_batch = y_valid.shape[0] // batch_size
      self.test.num_batch  = y_test.shape[0]  // batch_size


def load_data(nb_images=None, nb_images_per_label=10000, is_pan=False):
    # nb_images : number of images to load
    # nb_images_per_label : number of images per label to load
    # if nb_images is set and nb_images_per_label is None, images are drawn
    # from categories in proportion to their frequency in the dataset.
    # if nb_images_per_label is set and nb_images is None, the categories
    # are re-ballanced
    import os, os.path as path
    from glob import glob
    import scipy.misc as misc

    filenames, labels = [], []
    for root, dirs, files in os.walk('/data/by_yaw'):
        for i,d in enumerate(dirs):
            files = np.random.permutation(glob(path.join(root, d, '*.JPEG')))
            filenames.extend(files[:nb_images_per_label])
            labels.extend([i]*len(files[:nb_images_per_label]))

    inds = np.random.permutation(len(filenames))[:nb_images]
    filenames, labels = [filenames[i] for i in inds], [labels[i] for i in inds]

    images = [misc.imresize(misc.imread(f, mode='P' if is_pan else 'RGB'), size=(28,28), interp='nearest') 
        for f in filenames] # silently requires Pillow...

    def make_band_interleaved(pixel_interleaved_image):
        # nbands, nrows, ncols, nchannels
        return np.transpose(pixel_interleaved_image, (0,3,1,2))

    nb_train = int(0.9*len(filenames))
    # requires numpy > 1.10
    X_train, y_train = np.stack(images[:nb_train]), labels[:nb_train]
    X_test, y_test = np.stack(images[nb_train:]), labels[nb_train:]
    if not is_pan: X_train, X_test = map(make_band_interleaved, [X_train, X_test])

    return (X_train, y_train), (X_test, y_test)


#
# create generator & discriminator function
#

# generator network
def generator(tensor):

    # reuse flag
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0

    with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True, reuse=reuse):
        res = (tensor
               .sg_dense(dim=1024, name='fc1')
               .sg_dense(dim=7*7*128, name='fc2')
               .sg_reshape(shape=(-1, 7, 7, 128))
               .sg_upconv(dim=64, name='conv1')
               .sg_upconv(dim=1, act='sigmoid', bn=False, name='conv2'))
    return res


def discriminator(tensor):

    # reuse flag
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0

    with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu', reuse=reuse):
        # shared part
        shared = (tensor
                  .sg_conv(dim=64, name='conv1')
                  .sg_conv(dim=128, name='conv2')
                  .sg_flatten()
                  .sg_dense(dim=1024, name='fc1'))

        # discriminator end
        disc = shared.sg_dense(dim=1, act='linear', name='disc').sg_squeeze()

        # shared recognizer part
        recog_shared = shared.sg_dense(dim=128, name='recog')

        # categorical auxiliary classifier end
        cat = recog_shared.sg_dense(dim=cat_dim, act='linear', name='cat')

        # continuous auxiliary classifier end
        con = recog_shared.sg_dense(dim=con_dim, act='sigmoid', name='con')

        return disc, cat, con


#
# inputs
#

# MNIST input tensor ( with QueueRunner )
#data = tf.sg_data.Mnist(batch_size=batch_size)
data = ByYaw(batch_size=batch_size)

# input images and label
x = data.train.image
y = data.train.label

# labels for discriminator
y_real = tf.ones(batch_size)
y_fake = tf.zeros(batch_size)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

# categorical latent variable
z_cat = tf.multinomial(tf.ones((batch_size, cat_dim), dtype=tf.sg_floatx) / cat_dim, 1).sg_squeeze().sg_int()
# continuous latent variable
z_con = tf.random_normal((batch_size, con_dim))
# random latent variable dimension
z_rand = tf.random_normal((batch_size, rand_dim))
# latent variable
z = tf.concat(1, [z_cat.sg_one_hot(depth=cat_dim), z_con, z_rand])


#
# Computational graph
#

# generator
gen = generator(z)

# add image summary
tf.sg_summary_image(x, name='real')
tf.sg_summary_image(gen, name='fake')

# discriminator
disc_real, cat_real, _ = discriminator(x)
disc_fake, cat_fake, con_fake = discriminator(gen)


#
# loss
#

# discriminator loss
loss_d_r = disc_real.sg_bce(target=y_real, name='disc_real')
loss_d_f = disc_fake.sg_bce(target=y_fake, name='disc_fake')
loss_d = (loss_d_r + loss_d_f) / 2


# generator loss
loss_g = disc_fake.sg_bce(target=y_real, name='gen')

# categorical factor loss
loss_c_r = cat_real.sg_ce(target=y, name='cat_real')
loss_c_d = cat_fake.sg_ce(target=z_cat, name='cat_fake')
loss_c = (loss_c_r + loss_c_d) / 2

# continuous factor loss
loss_con = con_fake.sg_mse(target=z_con, name='con').sg_mean(dims=1)


#
# train ops
#

# discriminator train ops
train_disc = tf.sg_optim(loss_d + loss_c + loss_con, lr=0.0001, category='discriminator')
# generator train ops
train_gen = tf.sg_optim(loss_g + loss_c + loss_con, lr=0.001, category='generator')


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_d, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_g, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, max_ep=30, ep_size=data.train.num_batch, early_stop=False)

