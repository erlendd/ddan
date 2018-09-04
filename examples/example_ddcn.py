import numpy as np
from ddan import DDCNModel
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
#opt = tf.train.AdamOptimizer()

''' Performance with unsupervised deep-domain confusion... '''
model = DDCNModel(nfeatures=Xs.shape[1], arch=[16, 'act'], mmd_layer_idx=[1], val_data=(Xv, yv), 
    epochs=20000, batch_size=128, validate_every=100, confusion_incr=5e-4, confusion=1.5, confusion_max=10.,
    optimizer=opt, activations='leakyrelu')

# target labels (yt) aren't used in training, but if you provide them
# then each iteration will report performance on the target data.
model.fit(Xs, ys, Xt, yt)
vloss_mmd = model.evaluate(Xv, yv)

''' Perforformance with supervised deep-domain confusion... '''
model = DDCNModel(nfeatures=Xs.shape[1], arch=[16, 'act'], mmd_layer_idx=[1], val_data=(Xv, yv), 
    epochs=20000, batch_size=128, validate_every=100, confusion_incr=5e-4, confusion=1.5, confusion_max=10.,
    optimizer=opt, activations='leakyrelu', supervised=True)

# target labels (yt) aren't used in training, but if you provide them
# then each iteration will report performance on the target data.
model.fit(Xs, ys, Xt, yt)
vloss_smmd = model.evaluate(Xv, yv)

''' Perforformance (baseline) no adaptation... '''
model = DDCNModel(nfeatures=Xs.shape[1], arch=[16, 'act'], mmd_layer_idx=[], val_data=(Xv, yv), 
    epochs=20000, batch_size=128, validate_every=100, confusion_incr=5e-4, confusion=1.5, confusion_max=10.,
    optimizer=opt, activations='leakyrelu')

# target labels (yt) aren't used in training, but if you provide them
# then each iteration will report performance on the target data.
model.fit(Xs, ys, Xt, yt)
vloss_mlp = model.evaluate(Xv, yv)

''' Perforformance (baseline) no adaptation but with labelled target... '''
model = DDCNModel(nfeatures=Xs.shape[1], arch=[16, 'act'], mmd_layer_idx=[], val_data=(Xv, yv), 
    epochs=20000, batch_size=128, validate_every=100, confusion_incr=5e-4, confusion=1.5, confusion_max=10.,
    optimizer=opt, activations='leakyrelu', supervised=True)

# target labels (yt) aren't used in training, but if you provide them
# then each iteration will report performance on the target data.
model.fit(Xs, ys, Xt, yt)
vloss_mlp_target = model.evaluate(Xv, yv)


print 'Performance without domain-adaptation (baseline): {:.4f}'.format(vloss_mlp)
print 'Performance without domain-adaptation, but with labels in target domain: {:.4f}'.format(vloss_mlp_target)
print 'Performance with unsupervised domain-adaptation: {:.4f}'.format(vloss_mmd)
print 'Performance with supervised domain-adaptation: {:.4f}'.format(vloss_smmd)
