import numpy as np
from sklearn.datasets import make_blobs

# source and target domains
Xs, ys = make_blobs(1000, centers=[[0, 0], [0, 1]], cluster_std=0.2)
Xt, yt = make_blobs(500, centers=[[1, -1], [1, 0]], cluster_std=0.2)
# some unseen data from the target domain (validation)
Xv, yv = make_blobs(500, centers=[[1, -1], [1, 0]], cluster_std=0.2)

# concat the source and target domains (keep val data for validation)
Xall = np.vstack([Xs, Xt])
yall = np.hstack( [ys, yt])

