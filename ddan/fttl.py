import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss
from tensorflow.python.framework import ops
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from utils import shuffle_aligned_list, batch_gen, val_batch_gen



class FineTuningNet:

    def __init__(self, nfeatures=50, arch=[8, 'act', 8, 'act'], fine_tune_layers=[2, 3], batch_size=16, 
        val_data=None, validate_every=1, activations='relu', epochs=5000, epochs_finetune=5000, optimizer=None, optimizer_finetune=None,
        noise=0.0, droprate=0.0, verbose=True, stop_at_target_loss=0):

        self.batch_size = batch_size
        self.validate_every = validate_every
        self.epochs = epochs
        self.epochs_finetune = epochs_finetune
        self.verbose = verbose
        self.stop_at_target_loss = stop_at_target_loss
        if val_data is None:
            self.validate_every = 0
        else:
            self.Xval = val_data[0]
            self.yval = val_data[1]

        self._build_model(nfeatures, arch, activations, noise, droprate, optimizer, optimizer_finetune, fine_tune_layers)

        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self, nfeatures, arch, activations, noise, droprate, optimizer, optimizer_finetune, fine_tune_layers):
        print 'Building network layers...'

        self.inp = tf.placeholder(tf.float32, shape=(None, nfeatures))
        self.labels = tf.placeholder(tf.float32, shape=(None, 1))

        self.layers = [self.inp]

        for i, nunits in enumerate(arch):
            print nunits
            scope = 'feature_extractor'

            if i in fine_tune_layers:
                print 'layer {}: {} is for fine-tuning'.format(i, nunits)
                scope = 'tuning_layers'

            with tf.variable_scope(scope):
                if isinstance(nunits, int):
                    self.layers += [Dense(nunits, activation='linear')(self.layers[-1])]
     
                elif nunits == 'noise':
                    self.layers += [GaussianNoise(noise)(self.layers[-1])]
     
                elif nunits == 'bn':
                    self.layers += [BatchNormalization()(self.layers[-1])]
     
                elif nunits == 'drop':
                    self.layers += [Dropout(droprate)(self.layers[-1])]
     
                elif nunits == 'act':
                    if activations == 'prelu':
                        self.layers += [PReLU()(self.layers[-1])]
                    elif activations == 'leakyrelu':
                        self.layers += [LeakyReLU()(self.layers[-1])]
                    elif activations == 'elu':
                        self.layers += [ELU()(self.layers[-1])]
                    else:
                        print 'Activation({})'.format(activations)
                        self.layers += [Activation(activations)(self.layers[-1])]

        with tf.variable_scope('tuning_layers'):
            y_logits = Dense(1, activation='linear')(self.layers[-1])
        self.y_clf = Activation('sigmoid')(y_logits)

        self.xe_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=y_logits))

        if optimizer is None:
            self.train_step = tf.train.MomentumOptimizer(1e-3, 0.9)
        else:
            self.train_step = optimizer
        self.train_step = self.train_step.minimize(self.xe_loss)

        if optimizer_finetune is None:
            self.train_step_finetune = tf.train.MomentumOptimizer(1e-4, 0.9)
        else:
            self.train_step_finetune = optimizer_finetune

        # use variable_scope to freeze the feature_extractor layeres..
        finetune_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            "tuning_layers") 
        print "Will train {} variables-groups".format(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
        print "Will fine-tune {} variables-groups".format(len(finetune_train_vars))
        if len(finetune_train_vars) == 0:
            print 'WARNING: no fine-tuning layers selected!'
        else:
            self.train_step_finetune = self.train_step_finetune.minimize(self.xe_loss, var_list=finetune_train_vars)

    def predict_proba(self, X, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        yprobs = np.zeros((X.shape[0]), dtype=float)
        
        idx = np.arange(X.shape[0])
        vbatch = val_batch_gen([idx, X], batch_size)
        for i, (thisidx, thisX) in enumerate(vbatch):
            yprobs[thisidx] = self.sess.run(self.y_clf, 
                feed_dict={self.inp: thisX, K.learning_phase(): 0}).flatten()
        return yprobs

    def evaluate(self, X, y, batch_size=None):
        yprobs = self.predict_proba(X, batch_size)
        return log_loss(y, yprobs)

    def _validate(self, i, Xs, ys, Xt, yt, Xval, yval):

        self.history['source_loss'] += [self.evaluate(Xs, ys)]
        self.history['target_loss'] += [self.evaluate(Xt, yt)]
        self.history['val_loss'] += [self.evaluate(Xval, yval)]

        print '{:04d}  {:.5f} {:.5f} {:.5f} '.format(i,
            self.history['source_loss'][-1], self.history['target_loss'][-1], self.history['val_loss'][-1])        

    def _fit(self, batch_generator, train_step, epochs, batch_size, Xs, ys, Xt, yt, Xval, yval):
        for i in xrange(epochs):

            Xsource, ysource = batch_generator.next()

            _, xeloss = self.sess.run([train_step, self.xe_loss], feed_dict={
                self.inp: Xsource, self.labels: ysource.reshape(-1, 1), K.learning_phase(): 1
                })

            if self.validate_every > 0 and i % self.validate_every == 0 and Xval is not None and Xt is not None:
                self._validate(i, Xs, ys, Xt, yt, Xval, yval)

            if self.stop_at_target_loss > 0 and self.history['target_loss'] < self.stop_at_target_loss:
                print 'Stopping: target loss below stop value.'
                break

    def fit(self, Xs, ys,  Xt=None, yt=None, Xval=None, yval=None,
            epochs=None, epochs_finetune=None, batch_size=None, verbose=None):

        if epochs is None: epochs = self.epochs
        if epochs_finetune is None: epochs_finetune = self.epochs_finetune
        if batch_size is None: batch_size = self.batch_size
        if Xval is None: 
            Xval = self.Xval
            yval = self.yval
        if verbose is None: verbose = self.verbose

        S_batches = batch_gen([Xs, ys], batch_size=batch_size)
        T_batches = batch_gen([Xt, yt], batch_size=batch_size)

        self.history = {'source_loss': [], 'target_loss': [], 'val_loss': []}
        print 'Epoch  sloss tloss vloss'

        self._fit(S_batches, self.train_step, epochs, batch_size, Xs, ys, Xt, yt, Xval, yval)

        if Xt is None:
            print 'No data for fine-tuning: stopping here.'
        else:
            print 'Fine tuning on the target data...'
            self._fit(T_batches, self.train_step_finetune, epochs_finetune, batch_size, Xs, ys, Xt, yt, Xval, yval)



