import numpy as np

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    idx = data[0].shape[0]
    p = np.random.permutation(idx)
    return [d[p] for d in data]

def batch_gen(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def domain_batch_gen(X, y, domains, batch_size):

    udomains = np.unique(domains)
    while True:
        # first isolate a single domain
        this_domain = np.random.choice(udomains)
        mask = (domains == this_domain)
        thisX = X[mask, :]
        thisy = y[mask]
        # then choose batch_size samples from that domain
        idx = np.random.choice(np.arange(thisX.shape[0]), batch_size, replace=False)
        yield thisX[idx], thisy[idx]

def val_batch_gen(data, batch_size):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    batch_count = 0
    #while True:
    nbatch = len(data[0]) // batch_size
    if nbatch*batch_size < len(data[0]): nbatch += 1

    for i in xrange(nbatch):
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def domain_val_batch_gen(X, y, domains, batch_size=None):
    udomains = np.unique(domains)
    counter = 0
    while True:
        domain = udomains[counter]
        mask = (domains == domain)
        counter += 1
        if counter >= len(udomains): counter = 0
        yield thisX[mask], y[mask]



