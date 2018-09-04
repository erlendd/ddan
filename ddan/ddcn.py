import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow.python.framework import ops
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from utils import shuffle_aligned_list, batch_gen, val_batch_gen
from mmd import maximum_mean_discrepancy


class DDCNModel(object):

    def __init__(self, nfeatures=50, arch=[8, 'act'], mmd_layer_idx=[1],
        batch_size=16, supervised=False, confusion=0.0, confusion_incr=1e-3, confusion_max=1,
        val_data=None, validate_every=1, 
        activations='relu', epochs=1000, optimizer=None, noise=0.0, droprate=0.0, verbose=True):

        self.batch_size = batch_size
        self.epochs = epochs
        self.validate_every = validate_every
        self.supervised = supervised
        self.verbose = verbose

        if val_data is None:
            self.validate_every = 0
        else:
            self.Xval = val_data[0]
            self.yval = val_data[1]

        self._build_model(nfeatures, arch, supervised, confusion, confusion_incr, 
            confusion_max, activations, noise, droprate, mmd_layer_idx, optimizer)

        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self, nfeatures, architecture, supervised, confusion, confusion_incr, confusion_max, 
        activations, noise, droprate, mmd_layer_idx, optimizer):

        self.inp_a = tf.placeholder(tf.float32, shape=(None, nfeatures))
        self.inp_b = tf.placeholder(tf.float32, shape=(None, nfeatures))
        self.labels_a = tf.placeholder(tf.float32, shape=(None, 1))

        nlayers = len(architecture)
        layers_a = [self.inp_a]
        layers_b = [self.inp_b]

        for i, nunits in enumerate(architecture):

            print nunits,
            if i in mmd_layer_idx: print '(MMD)'
            else: print

            if isinstance(nunits, int):
                shared_layer = Dense(nunits, activation='linear')
            elif nunits == 'noise':
                shared_layer = GaussianNoise(noise)
            elif nunits == 'bn':
                shared_layer = BatchNormalization()
            elif nunits == 'drop':
                shared_layer = Dropout(droprate)
            elif nunits == 'act':
                if activations == 'prelu':
                    shared_layer = PReLU()
                elif activations == 'elu':
                    shared_layer = ELU()
                elif activations == 'leakyrelu':
                    shared_layer = LeakyReLU()
                else:
                    shared_layer = Activation(activations)

            layers_a += [shared_layer(layers_a[-1])]
            layers_b += [shared_layer(layers_b[-1])]

        y_logits = Dense(1, activation='linear', name='a_output')(layers_a[-1])
        self.y_clf = Activation('sigmoid')(y_logits)

        # Sum the losses from both branches...
        self.xe_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_a, logits=y_logits))

        self.mmd_losses = []
        for idx in mmd_layer_idx:
            self.mmd_losses += [maximum_mean_discrepancy(layers_a[idx], layers_b[idx])]

        self.domain_loss = tf.reduce_sum(self.mmd_losses)

        self.confusion = tf.Variable(float(confusion), trainable=False, dtype=tf.float32)
        conf_incr = tf.cond(self.confusion < confusion_max, lambda: float(confusion_incr), lambda: 0.)
        self.increment_confusion = tf.assign(self.confusion, self.confusion + conf_incr)

        self.total_loss = tf.add(self.confusion*self.domain_loss, self.xe_loss)

        if supervised:
            self.labels_b = tf.placeholder(tf.float32, shape=(None, 1))
            b_logits = Dense(1, activation='linear', name='b_output')(layers_b[-1])
            self.bloss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_b, logits=b_logits))
            self.total_loss = tf.add(self.total_loss, self.bloss)

        if optimizer is None:
            self.train_step = tf.train.MomentumOptimizer(1e-3, 0.9)
        else:
            self.train_step = optimizer
        self.train_step = self.train_step.minimize(self.total_loss)

    def predict_proba(self, X, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        yprobs = np.zeros((X.shape[0]), dtype=float)
        
        idx = np.arange(X.shape[0])
        vbatch = val_batch_gen([idx, X], batch_size)
        for i, (thisidx, thisX) in enumerate(vbatch):
            yprobs[thisidx] = self.sess.run(self.y_clf, 
                feed_dict={self.inp_a: thisX, K.learning_phase(): 0}).flatten()
        return yprobs

    def evaluate(self, X, y, batch_size=None):
        yprobs = self.predict_proba(X, batch_size)
        return log_loss(y, yprobs)

    def fit(self, Xs, ys,  Xt, yt=None, Xval=None, yval=None,
            epochs=None, batch_size=None, verbose=None):

        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size
        if Xval is None: 
            Xval = self.Xval
            yval = self.yval
        if verbose is None: verbose = self.verbose

        S_batches = batch_gen([Xs, ys], batch_size=batch_size)
        if yt is None: yt = np.ones(Xt.shape[0])
        T_batches = batch_gen([Xt, yt], batch_size=batch_size)
       
        self.history = {'source_loss': [], 'target_loss': [], 'val_loss': [], 'domain_loss': []}
        for i in range(epochs):
            
            Xsource, ysource = S_batches.next()
            Xtarget, ytarget = T_batches.next()

            feed_dict = {self.inp_a: Xsource, self.inp_b: Xtarget,
                self.labels_a: ysource.reshape(-1, 1), K.learning_phase(): 1}
            if self.supervised:
                feed_dict[self.labels_b] = ytarget.reshape(-1, 1)

            # train
            _, _, confusion, xeloss, dloss, tloss = self.sess.run([
                                                             self.train_step,
                                                             self.increment_confusion,
                                                             self.confusion, 
                                                             self.xe_loss,
                                                             self.domain_loss, 
                                                             self.total_loss],
                                                             feed_dict=feed_dict)

            if self.validate_every > 0 and i % self.validate_every == 0:
             
                if i == 0:
                    print 'Epoch confusion  dloss  sloss tloss vloss'
                
                self.history['source_loss'] += [self.evaluate(Xs, ys)]
                self.history['target_loss'] += [self.evaluate(Xt, yt)]
                self.history['val_loss'] += [self.evaluate(Xval, yval)]
                self.history['domain_loss'] += [dloss]

                print '{:04d} {:.2f}  {:.4f}  {:.4f}  {:.5f} {:.5f} {:.5f} '.format(i, confusion, dloss, tloss,
                    self.history['source_loss'][-1], self.history['target_loss'][-1], self.history['val_loss'][-1])
 




if __name__ == '__main__':

    from sklearn.datasets import make_blobs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()

    batch_size = 200

    Xs, ys = make_blobs(300, centers=[[0, 0], [0, 1]], cluster_std=0.2)
    Xt, yt = make_blobs(300, centers=[[1, -1], [1, 0]], cluster_std=0.2)
    Xall = np.vstack([Xs, Xt])
    yall = np.hstack( [ys, yt])
    plt.scatter(Xall[:, 0], Xall[:, 1], c=yall)
    plt.savefig('blobs.png')
    plt.close()

    print 'MMD:', compute_mmd_on_samples(Xs, Xt)

