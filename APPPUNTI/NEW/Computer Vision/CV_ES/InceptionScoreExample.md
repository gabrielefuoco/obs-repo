# Inception Score Example

Esempio di calcolo dell'Inception Score

Credit: <https://github.com/sbarratt/inception-score-pytorch>

```python
import torch
import warnings

warnings.filterwarnings("ignore")

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy

USE_CUDA = True
```

# Caricamento del dataset

Utilizzo solo le immagini ignorando le labal

```python
class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]  # ritorno l'immagine

    def __len__(self):
        return len(self.orig)

cifar = dset.CIFAR10(root='data/', download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
)

dataset = IgnoreLabelDataset(cifar)

```

# Classificazione del dataset

Calcolo le probabilità per ogni classe

```python
N = len(dataset)
batch_size = 32

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

if USE_CUDA:
    dtype = torch.cuda.FloatTensor
else:
    if torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably set cuda=True")
    dtype = torch.FloatTensor

inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
inception_model.eval();
up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
softmax = torch.nn.Softmax(dim=1)
def get_pred(x):
    x = up(x)
    x = inception_model(x)
    return softmax(x).cpu().detach().numpy()

# Get predictions
preds = np.zeros((N, 1000))

for i, batch in enumerate(dataloader):
    batch = batch.type(dtype)
    batch_size_i = batch.size()[0]

    preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

preds.shape
```

    (50000, 1000)

# Calcolo dello score

A partire dalla distribuzione delle probabilità calcolo la KL e l'IS

```python
splits = 10
split_scores = []

for k in range(splits):
    part = preds[k * (N // splits): (k+1) * (N // splits), :]
    py = np.mean(part, axis=0)
    scores = []
    for i in range(part.shape[0]):
        pyx = part[i, :]
        scores.append(entropy(pyx, py))
    split_scores.append(np.exp(np.mean(scores)))

score, devstd = np.mean(split_scores), np.std(split_scores)

print(f'Inception Score is {score} with devstd {devstd}')
```

    Inception Score is 9.672773924506332 with devstd 0.14991434268517465
    
