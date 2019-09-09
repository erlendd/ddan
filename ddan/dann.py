import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.framework import ops
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from utils import shuffle_aligned_list, batch_gen, val_batch_gen
from keras.regularizers import l1_l2, l1, l2

''' Domain-adversarial neural network (https://arxiv.org/pdf/1505.07818.pdf)
Some parts of this code were inspired by https://github.com/pumpikano/tf-dann. '''

class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()


class DANNModel(object):

    def __init__(self, nfeatures=50, arch_shared=[32, 'act'], arch_domain=[8, 'act'], arch_clf=[], 
        batch_size=16, supervised=False, val_data=None, validate_every=1, 
        activations='relu', epochs=1000, optimizer=None, noise=0.0, droprate=0.0, stop_at_target_loss=0.0):

        self.batch_size = batch_size
        self.epochs = epochs
        self.validate_every = validate_every
        self.stop_at_target_loss = stop_at_target_loss

        if val_data is None:
            validate_every = 0
        else:
            self.Xval = val_data[0]
            self.yval = val_data[1]

        self._build_model(nfeatures, arch_shared, arch_domain, arch_clf, 
            activations, supervised, noise, droprate, optimizer)

        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build(self, input_layer, arch, activations, noise, droprate, l2reg):
        print 'Building network layers...'
        network = [input_layer]
        for nunits in arch:
            print nunits
            if isinstance(nunits, int):
                network += [Dense(nunits, activation='linear', kernel_regularizer=l1_l2(l1=0.01, l2=l2reg))(network[-1])]

            elif nunits == 'noise':
                network += [GaussianNoise(noise)(network[-1])]

            elif nunits == 'bn':
                network += [BatchNormalization()(network[-1])]

            elif nunits == 'drop':
                network += [Dropout(droprate)(network[-1])]

            elif nunits == 'act':
                if activations == 'prelu':
                    network += [PReLU()(network[-1])]
                elif activations == 'leakyrelu':
                	network += [LeakyReLU()(network[-1])]
                elif activations == 'elu':
                	network += [ELU()(network[-1])]
                else:
                    print 'Activation({})'.format(activations)
                    network += [Activation(activations)(network[-1])]
        return network

    def _build_model(self, nfeatures, arch_shared, arch_domain, arch_clf, 
        activations, supervised, noise, droprate, optimizer):

        self.X = tf.placeholder(tf.float32, [None, nfeatures])
        self.domain = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])

        self.l = tf.placeholder(tf.float32, [], name='l')
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        self.train = tf.placeholder(tf.bool, [])

        full_feat = self._build(self.X, arch_shared, activations, noise, droprate, 0.0001)

        if supervised:
            # we have labels for the target data and the source data
            select_feat = full_feat[-1]
            select_y = self.y
        else:
            # if we are training here we use only the first half of the
            # the batch (corresponding to labelled source data)..
            select_feat = tf.cond(self.train,
                lambda: tf.slice(full_feat[-1], [0, 0], [self.batch_size/2, -1]),
                lambda: full_feat[-1])
    
            select_y = tf.cond(self.train,
                lambda: tf.slice(self.y, [0, 0], [self.batch_size/2, -1]),
                lambda: self.y)

        grl = flip_gradient(full_feat[-1], self.l)

        y_net = self._build(select_feat, arch_clf, activations, noise, droprate, 0.01)
        y_logits = Dense(1, activation='linear')(y_net[-1])
        self.y_clf = Activation('sigmoid')(y_logits)

        domain_net = self._build(grl, arch_domain, activations, noise, droprate, 0.01)
        self.domain_logits = Dense(1, activation='linear')(domain_net[-1])
        self.domain_clf = Activation('sigmoid')(self.domain_logits)

        # Sum the losses from both branches...
        self.xe_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=select_y, logits=y_logits))

        self.domain_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.domain, logits=self.domain_logits))

        self.total_loss = tf.add(self.domain_loss, self.xe_loss)

        if optimizer is None:
            self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9)
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
                feed_dict={self.X: thisX, self.train: False, self.l: 1.0}).flatten()
        return yprobs

    def evaluate(self, X, y, batch_size=None):
        yprobs = self.predict_proba(X, batch_size)
        return log_loss(y, yprobs, labels=[0, 1])

    def domains_predict_proba(self, X, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        dprobs = np.zeros((X.shape[0]), dtype=float)
        
        idx = np.arange(X.shape[0])
        vbatch = val_batch_gen([idx, X], batch_size)
        for i, (thisidx, thisX) in enumerate(vbatch):
            dprobs[thisidx] = self.sess.run(self.domain_clf, feed_dict={self.X: thisX}).flatten()
        return dprobs

    def fit(self, Xs, ys, Xt, yt=None, Xval=None, yval=None,  
        validate_every=None, epochs=None, batch_size=None, l=None):

        print 'Training on X {}, labels {}'.format(Xs.shape, ys.shape)
        print 'Target on X {}'.format(Xt.shape),
        if yt is not None: 
            print ', labels {}'.format(ys.shape)
        else:
            print

        if validate_every is None: validate_every = self.validate_every
        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size
        if Xval is None:
            Xval, yval = self.Xval, self.yval
        if Xval is not None and validate_every > 0:
            print 'Validating on X {}, labels {}, every {} iterations'.format(Xval.shape, yval.shape, validate_every)

        S_batches = batch_gen([Xs, ys], batch_size/2)
        if yt is None: yt = np.ones(Xt.shape[0])
        T_batches = batch_gen([Xt, yt], batch_size/2)

        self.history = {'source_loss': [], 'target_loss': [], 'val_loss': [], 'd_auc': []}

        for i in range(epochs):

            p = i / float(epochs)
            lp = 2. / (1. + np.exp(-10*p)) - 1
            lr = 0.01 / (1 + 10.*p)**0.75

            if l is not None: lp = l

            Xsource, ysource = S_batches.next()
            Xtarget, ytarget = T_batches.next()
            Xbatch = np.vstack([Xsource, Xtarget])
            ybatch = np.hstack([ysource, ytarget])
            # first half of batch is from the source batch (=0)
            # second half is the target batch (=1) 
            Dbatch = np.hstack(
                [np.zeros(batch_size/2, dtype=np.int32),
                 np.ones(batch_size/2, dtype=np.int32)]
                )

            # train step, also get training and domain losses
            _, tloss, dloss, xeloss = self.sess.run(
                [self.train_step, self.total_loss, self.domain_loss, self.xe_loss],
                feed_dict={self.X: Xbatch, self.domain: Dbatch.reshape(-1, 1), self.y: ybatch.reshape(-1, 1),
                    self.train: True, self.l: lp, K.learning_phase(): 1, self.lr: lr}
                )

            if validate_every > 0 and i % validate_every == 0:

                if i == 0:
                    print 'Epoch grl  sloss tloss vloss dauc'
                
                Xall = np.vstack([Xs, Xt])
                dall = np.hstack([np.zeros(Xs.shape[0]), np.ones(Xt.shape[0])])
                dprobs = self.domains_predict_proba(Xall)

                self.history['source_loss'] += [self.evaluate(Xs, ys)]
                self.history['target_loss'] += [self.evaluate(Xt, yt)]
                self.history['val_loss'] += [self.evaluate(Xval, yval)]
                self.history['d_auc'] += [roc_auc_score(dall, dprobs)]

                print '{:04d} {:.2f}  {:.5f} {:.5f} {:.5f}  {:.3f}'.format(i, lp, 
                    self.history['source_loss'][-1], self.history['target_loss'][-1], self.history['val_loss'][-1], self.history['d_auc'][-1])

                if self.history['target_loss'][-1] < self.stop_at_target_loss:
                    print 'Stopping: target loss below stop value.'
                    break


