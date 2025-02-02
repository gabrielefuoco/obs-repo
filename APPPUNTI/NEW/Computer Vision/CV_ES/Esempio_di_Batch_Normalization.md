```python
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt

%matplotlib inline
```

# Batch Normalization

Batch normalization was introduced in Sergey Ioffe's and Christian Szegedy's 2015 paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf). The idea is that, instead of just normalizing the inputs to the network, we normalize the inputs to _layers within_ the network. It's called "batch" normalization because during training, we normalize each layer's inputs by using the mean and variance of the values in the current mini-batch.

![image.png](47ca0f0c-ffc0-468c-b01f-5a1f43fa2ab8.png)

```python
# generiamo un batch di immagini A con una distribuzione normale e del rumore casuale
batch_size = 30
A = torch.zeros(batch_size, 32, 32)

for i in range(batch_size):
    # ogni pixel è ottenuto come: 
    # random(0, 1) * x + b
    A[i, :, :] = torch.randn(32 * 32).view(32, 32) * torch.randint(5, size=(1,)) + torch.randint(50, size=(1,))

A.shape, A
```

```python
def describe_batch(image_bn, ax1, titleprefix='Batch'):
    
    title=f'{titleprefix}\nMin {image_bn.min():.4f} - Max {image_bn.max():.4f}' \
          f'\nMean {image_bn.mean():.4f} - Var {image_bn.var():.4f}'
    
    ax1.hist(image_bn.flatten().numpy(), bins='auto')
    ax1.set_title(title)
    
    
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
describe_batch(A, ax)
```

    
![png](Esempio_di_Batch_Normalization_4_0.png)
    

# Stima dei parametri gamma e beta

```python
eps = 1e-16
values_gamma = (1, 10, 100)
values_beta = (0, 10, 100)

fig, axes_plot = plt.subplots(3, 3, figsize=(20, 20))

axes = axes_plot.ravel()

for (gamma, beta), ax in zip(itertools.product(values_gamma, values_beta), axes):
    
    correctedA = gamma * (A - A.mean()) / (torch.sqrt(A.var() + eps)) + beta
    describe_batch(correctedA, ax, f'gamma = {gamma}, beta ${beta}')

fig.tight_layout()    
```

    
![png](Esempio_di_Batch_Normalization_6_0.png)
    

