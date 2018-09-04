import numpy as np
from ddan import AdaBNModel
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from data import *

# plot training (source and target) data
plt.scatter(Xall[:, 0], Xall[:, 1], c=yall)
plt.savefig('blobs.png')
plt.close()

opt = tf.train.MomentumOptimizer(1e-3, 0.9)

''' Performance (baseline) no transfer learning... '''
model = AdaBNModel(nfeatures=Xs.shape[1], arch=[8, 'act'],
    val_data=(Xv, yv), epochs=5000, batch_size=128, validate_every=100, 
    optimizer=opt, activations='leakyrelu')

model.fit(Xs, ys, Xt, yt)
vloss_baseline = model.evaluate(Xv, yv)

''' Performance with Adaptive Batch Normalization... '''
model = AdaBNModel(nfeatures=Xs.shape[1], arch=[8, 'abn', 'act'],
    val_data=(Xv, yv), epochs=5000, batch_size=128, validate_every=100, 
    optimizer=opt, activations='leakyrelu')

model.fit(Xs, ys, Xt, yt)
vloss_adabn = model.evaluate(Xv, yv)

''' Performance with Adaptive Batch Normalization (multi-domain)... '''
Xunion = np.vstack([Xs, Xt])
yunion = np.hstack([ys, yt])
domains = np.zeros_like(yunion, dtype=int)
domains[len(ys):] = 1

model = AdaBNModel(nfeatures=Xs.shape[1], arch=[8, 'abn', 'act'],
    val_data=(Xv, yv), epochs=5000, batch_size=128, validate_every=100, 
    optimizer=opt, activations='leakyrelu')

model.fit(Xunion, yunion, Xt, yt, domains)
vloss_adabn_multidomain = model.evaluate(Xv, yv)

print vloss_baseline
print vloss_adabn
print vloss_adabn_multidomain

