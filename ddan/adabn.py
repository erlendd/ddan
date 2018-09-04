import numpy as np
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, SGD
from sklearn.metrics import log_loss
from abn import AdaptiveBatchNormalization as AdaBN
from keras.callbacks import EarlyStopping
import tensorflow as tf
from utils import domain_batch_gen, domain_val_batch_gen



class AdaBNModel:

    def __init__(self, nfeatures=50, arch=[8, 'abn', 'act'], activations='relu',
                 droprate=0.0, noise=0.0, optimizer=None, val_data=None,
                 validate_every=1, epochs=5000, batch_size=128, verbose=False):

        self.epochs = epochs
        self.batch_size = batch_size
        self.noise = noise
        self.verbose = verbose

        self.validate_every = validate_every
        if val_data is None:
            self.validate_every = 0
            self.Xval = None
            self.yval = None
        else:
            self.Xval = val_data[0]
            self.yval = val_data[1]

        self._build_model(arch, activations, nfeatures, droprate, noise, optimizer) 

    def _build_model(self, arch, activations, nfeatures, droprate, noise, optimizer):

        self.layers = [Input(shape=(nfeatures,))]

        for i, nunits in enumerate(arch):

            if isinstance(nunits, int):
                self.layers += [Dense(nunits, activation='linear')(self.layers[-1])]

            elif nunits == 'noise':
                self.layers += [GaussianNoise(noise)(self.layers[-1])]

            elif nunits == 'bn':
                self.layers += [BatchNormalization()(self.layers[-1])]
            
            elif nunits == 'abn':
                self.layers += [AdaBN()(self.layers[-1])]

            elif nunits == 'drop':
                self.layers += [Dropout(droprate)(self.layers[-1])]

            elif nunits == 'act':
                if activations == 'prelu':
                    self.layers += [PReLU()(self.layers[-1])]
                elif activations == 'elu':
                    self.layers += [ELU()(self.layers[-1])]
                elif activations == 'leakyrelu':
                    self.layers += [LeakyReLU()(self.layers[-1])]
                else:
                    self.layers += [Activation(activations)(self.layers[-1])]

            else:
                print 'Unrecognised layer {}, type: {}'.format(nunits, type(nunits))

        self.layers += [Dense(1, activation='sigmoid')(self.layers[-1])]

        self.model = Model(self.layers[0], self.layers[-1])
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer) 

    def _fit(self, Xs, ys, Xt, yt, domains, Xval=None, yval=None, epochs=None, batch_size=None, verbose=None):

        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size
        if Xval is None: 
            Xval = self.Xval
            yval = self.yval
        if verbose is None: verbose = self.verbose

        # batch generator that ensures that samples are always from the same domain
        S_batches = domain_batch_gen(Xs, ys, domains, batch_size=batch_size)

        self.history = {'source_loss': [], 'target_loss': [], 'val_loss': []}
        for i in xrange(epochs):
            Xsource, ysource = S_batches.next()
            self.model.fit(Xsource, ysource, epochs=1, batch_size=batch_size, verbose=0) 

            if self.validate_every > 0 and i % self.validate_every == 0:
             
                if i == 0:
                    print 'Epoch  sloss tloss vloss'
                
                self.history['source_loss'] += [self.evaluate(Xs, ys, domains)]
                self.history['target_loss'] += [self.evaluate(Xt, yt)]
                self.history['val_loss'] += [self.evaluate(Xval, yval)]

                print '{:04d}  {:.5f} {:.5f} {:.5f} '.format(i,
                    self.history['source_loss'][-1], self.history['target_loss'][-1], self.history['val_loss'][-1])


    def fit(self, Xs, ys, Xt, yt, domains=None, Xval=None, yval=None, epochs=None, batch_size=None, verbose=None):

        if epochs is None: epochs = self.epochs
        if batch_size is None: batch_size = self.batch_size
        if Xval is None: 
            Xval = self.Xval
            yval = self.yval
        if verbose is None: verbose = self.verbose

        if domains is None: domains = np.ones_like(ys, dtype=int)
        self._fit(Xs, ys, Xt, yt, domains, Xval, yval, epochs, batch_size, verbose)         

    def predict_proba(self, X, domains=None, batch_size=None):
        if batch_size is None: batch_size = self.batch_size
        if domains is None:
            return self.model.predict(X)
        else:
            ypreds = np.zeros(X.shape[0])
            udomains = np.unique(domains)
            for i in xrange(udomains.shape[0]):
                idx = domains == udomains[i]
                thisX = X[idx]
                ypreds[idx] = self.model.predict(thisX, batch_size=batch_size).flatten()
    
            return ypreds

    def evaluate(self, X, y, domains=None, batch_size=None):
        yprobs = self.predict_proba(X, domains, batch_size)
        return log_loss(y, yprobs)




if __name__ == "__main__":

    from sklearn.datasets import make_blobs
    Xs, ys = make_blobs(300, centers=[[0, 0], [0, 1]], cluster_std=0.2)
    Xs2, ys2 = make_blobs(300, centers=[[-1, 0], [-1, 2]], cluster_std=0.2)
    Xt, yt = make_blobs(300, centers=[[1, -1], [1, 0]], cluster_std=0.2)
    Xall = np.vstack([Xs, Xs2, Xt])
    yall = np.hstack( [ys, ys2, yt])

    architecture = [8, 'bn', 'act', 8, 'act']

    # Ex.1: Single source domain, single target domain
    model = AdaBNModel(architecture, nfeatures=2)
    model.model.summary()
    print

    model.fit(Xs, ys, np.zeros_like(ys), epochs=1000, batch_size=100, 
        validation_data=(Xt, yt, np.ones_like(yt)), verbose=True)

    # Ex.2: Two source domains, single target domain
    Xall = np.vstack([Xs, Xs2])
    yall = np.hstack( [ys, ys2])

    model = AdaBNModel(architecture, nfeatures=2)
    model.model.summary()
    print

    domains = np.zeros_like(yall)
    domains[300:] = 1

    model.fit(Xall, yall, domains, epochs=1000, batch_size=100, 
        validation_data=(Xt, yt, np.ones_like(yt)), verbose=True)

    #Ex.3: One source domain, two target domains
    Xtall = np.vstack([Xs2, Xt])
    ytall = np.hstack([ys2, yt])

    model = AdaBNModel(architecture, nfeatures=2)
    model.model.summary()
    print

    domains = np.zeros_like(yall)
    domains[300:] = 1

    print Xtall.shape, ytall.shape
    print Xs.shape, ys.shape

    vbatch_gen = val_batch_generator(Xtall, ytall, domains)
    model.fit(Xs, ys, np.zeros_like(ys), epochs=1000, batch_size=100, 
        validation_data=vbatch_gen, validation_steps=len(np.unique(domains)), verbose=True)

