# About Deep Domain Adaptation Networks (ddan)
Deep domain adaptation networks (DDAN) is a Python library for domain adapation written in TensorFlow.

## Implemented models

The following neural network models are implemented in ddan:
* Adaptive Batch Normalization (AdaBN, paper here: https://arxiv.org/abs/1603.04779)
* Domain-adversarial Networks (Gradient reversal layer, paper here: https://arxiv.org/abs/1505.07818 and https://arxiv.org/abs/1409.7495)
* Deep-domain Confusion Network (siamese network with MMD loss, paper here: https://arxiv.org/abs/1412.3474)
* Fine-tuning Network

# Installation

    git clone https://github.com/erlendd/ddan.git
    cd ddan
    sudo pip install .
    *OR*
    sudo python setup.py install
    
# Usage

Full working examples can be found in the examples/ directory.

    import ddan
    
    model = ddan.DANNModel(epochs=1000, batch_size=64)
    # Xs, ys are for the source domain, Xt, (yt) are for the target domain
    model.fit(Xs, ys, Xt)
    # Obtain predictions from trained model
    yprobs = model.predict_proba(Xval)

