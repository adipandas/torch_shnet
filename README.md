# torch_shnet

Stacked Hourglass Network (shnet) for human pose estimation implemented in PyTorch.

# Training

Recommended to use multi-gpu training (For single GPU, reduce the batch size in ``config.yaml`` to ``4``.).
I haven't tested ``train.py`` which is using ``DistributedDataParallel``.

To start training:
``python train_pl.py``

## References:
1. Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass networks for human pose estimation." European conference on computer vision. Springer, Cham, 2016. [[arxiv](https://arxiv.org/abs/1603.06937)]
2. **Stacked Hourglass Network** model implementation was adopted from **Chris Rockwell**'s implementation available in **[this GitHub repository](https://github.com/princeton-vl/pytorch_stacked_hourglass)**.

