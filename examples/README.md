# Examples

The following examples are included on a very simple domain adaptation task (blobs).
* example_dann.py: Domain-adversarial neural network [1]
* example_ddcn.py: Deep domain confusion network [2]
* example_adabn.py: Adaptive Batch Normalization [3]
* example_fttl.py: Simple fine-tuning model.

These examples will run in a few minutes on a modern desktop or laptop computer.

## Results

| Model | Log loss on validation set
|-|------------|
|Neural network (no adaption)| 3.46 |
|DaNN| 0.085 |
|DDCN| 0.089 |
|AdaBN| 0.060 |
|FTTL| 0.086 |

### References
[1] https://arxiv.org/abs/1505.07818
[2] https://arxiv.org/abs/1412.3474
[3] https://arxiv.org/abs/1412.3474

