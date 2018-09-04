import numpy as np
from ddan import FineTuningNet
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
opt_finetune = tf.train.MomentumOptimizer(1e-4, 0.9)

''' Performance (baseline) no transfer learning... '''
model = FineTuningNet(nfeatures=Xs.shape[1], arch=[16, 'act', 8, 'act'], fine_tune_layers=[], 
    val_data=(Xv, yv), epochs=5000, epochs_finetune=0, batch_size=128, validate_every=100, 
    optimizer=opt, optimizer_finetune=opt_finetune, activations='leakyrelu')

model.fit(Xs, ys, Xt, yt)
vloss_baseline = model.evaluate(Xv, yv)

''' Performance with supervised fine tuning... '''
model = FineTuningNet(nfeatures=Xs.shape[1], arch=[16, 'act', 8, 'act'], fine_tune_layers=[2, 3], 
    val_data=(Xv, yv), epochs=5000, epochs_finetune=5000, batch_size=128, validate_every=100, 
    optimizer=opt, optimizer_finetune=opt_finetune, activations='leakyrelu')

model.fit(Xs, ys, Xt, yt)
vloss_ft = model.evaluate(Xv, yv)


''' Performance with supervised fine tuning on union on source and target... '''
model = FineTuningNet(nfeatures=Xs.shape[1], arch=[16, 'act', 8, 'act'], fine_tune_layers=[2, 3], 
    val_data=(Xv, yv), epochs=5000, epochs_finetune=5000, batch_size=128, validate_every=100, 
    optimizer=opt, optimizer_finetune=opt_finetune, activations='leakyrelu')

Xunion, yunion = np.vstack([Xs, Xt]), np.hstack([ys, yt])

model.fit(Xs, ys, Xunion, yunion)
vloss_ft_union = model.evaluate(Xv, yv)

print 'Performance without fine-tuning (baseline): {:.4f}'.format(vloss_baseline)
print 'Performance with supervised fine-tuning on target: {:.4f}'.format(vloss_ft)
print 'Performance with supervised fine-tuning on source+target: {:.4f}'.format(vloss_ft_union)

