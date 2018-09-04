import numpy as np
from ddan import DANNModel
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

''' Performance without GRL (standard MLP)... '''
model = DANNModel(nfeatures=Xs.shape[1], arch_shared=[16, 'act'], arch_domain=[4, 'act'], 
    arch_clf=[4, 'act'], val_data=(Xv, yv), epochs=10000, batch_size=128, validate_every=100,
    optimizer=opt, activations='leakyrelu')

# target labels (yt) aren't used in training, but if you provide them
# then each iteration will report performance on the target data.
model.fit(Xs, ys, Xt, yt, l=0)
vloss_mlp = model.evaluate(Xv, yv)

''' Performance with GRL (domain-adversarial training)... '''
model = DANNModel(nfeatures=Xs.shape[1], arch_shared=[16, 'act'], arch_domain=[4, 'act'], 
    arch_clf=[4, 'act'], val_data=(Xv, yv), epochs=10000, batch_size=128, validate_every=100,
    optimizer=opt, activations='leakyrelu')

model.fit(Xs, ys, Xt, yt)
vloss_grl = model.evaluate(Xv, yv)

''' Performance with GRL (domain-adversarial training)... '''
model = DANNModel(nfeatures=Xs.shape[1], arch_shared=[16, 'act'], arch_domain=[4, 'act'], 
    arch_clf=[4, 'act'], val_data=(Xv, yv), epochs=10000, batch_size=128, validate_every=100,
    optimizer=opt, activations='leakyrelu', supervised=True)

# supervised domain-adaptation: yt labels are used this time.
model.fit(Xs, ys, Xt, yt)
vloss_sgrl = model.evaluate(Xv, yv)


# Performance comparison...
print 'Performance without domain-adaptation (baseline): {:.4f}'.format(vloss_mlp)
print 'Performance with unsupervised domain-adaptation (DaNN): {:.4f}'.format(vloss_grl)
print 'Performance with supervised domain-adaptation (DaNN): {:.4f}'.format(vloss_sgrl)

