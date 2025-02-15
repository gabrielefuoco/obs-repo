

Questo documento descrive un esempio di calcolo dell'Inception Score (IS), una metrica utilizzata per valutare la qualità di immagini generate da modelli di deep learning generativi.  L'esempio utilizza il dataset CIFAR-10 e il modello pre-addestrato Inception v3.

## 1. Importazione delle Librerie e Impostazioni

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

Questo blocco importa le librerie necessarie: `torch` per le operazioni tensoriali, `warnings` per gestire gli avvisi, `nn` e `functional` da `torch` per le operazioni di reti neurali, `torch.utils.data` per la gestione dei dataset, `inception_v3` da `torchvision.models` per il modello Inception, `torchvision.datasets` e `torchvision.transforms` per il caricamento e la pre-elaborazione del dataset CIFAR-10, `numpy` per le operazioni numeriche, `scipy.stats` per il calcolo dell'entropia e infine si imposta la variabile `USE_CUDA` per l'utilizzo della GPU se disponibile.


## 2. Caricamento del Dataset

```python
class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig
    def __getitem__(self, index):
        return self.orig[index][0] # ritorno l'immagine
    def __len__(self):
        return len(self.orig)

cifar = dset.CIFAR10(root='data/', download=True, transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
dataset = IgnoreLabelDataset(cifar)
```

Questo blocco definisce una classe `IgnoreLabelDataset` che estende `torch.utils.data.Dataset`.  Questa classe serve per ignorare le etichette del dataset CIFAR-10, restituendo solo le immagini.

- `__init__(self, orig)`: inizializza la classe con il dataset originale.
- `__getitem__(self, index)`: restituisce l'immagine all'indice `index`.
- `__len__(self)`: restituisce la lunghezza del dataset.

Successivamente, viene caricato il dataset CIFAR-10 con una trasformazione che ridimensiona le immagini a 32x32 pixel, le converte in tensori e le normalizza. Infine, viene creato un nuovo dataset `dataset` usando `IgnoreLabelDataset` per ignorare le etichette.


## 3. Classificazione del Dataset

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

preds = np.zeros((N, 1000))
for i, batch in enumerate(dataloader):
    batch = batch.type(dtype)
    batch_size_i = batch.size()[0]
    preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)
preds.shape
```

Questo blocco definisce un `dataloader` per elaborare il dataset a batch.  Viene caricato il modello Inception v3 pre-addestrato e impostato in modalità valutazione (`eval()`).  La funzione `get_pred(x)` riceve un batch di immagini come input, le ridimensiona a 299x299 (dimensione richiesta da Inception v3), le passa attraverso il modello Inception v3, applica una softmax per ottenere le probabilità per ogni classe (1000 classi) e restituisce un array NumPy.  Il codice itera sui batch del `dataloader`, ottiene le predizioni usando `get_pred` e le salva nella matrice `preds`.  `preds.shape` stampa la dimensione della matrice delle predizioni (50000, 1000), indicando 50000 immagini e 1000 probabilità per classe per ogni immagine.


## 4. Calcolo dello Score

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

Questo blocco calcola l'Inception Score. Il dataset viene diviso in 10 parti (`splits`). Per ogni parte:

- `py` rappresenta la distribuzione di probabilità media sulle classi per quella parte.
- Per ogni immagine (`pyx`), viene calcolata la divergenza di Kullback-Leibler (KL) tra la distribuzione di probabilità dell'immagine (`pyx`) e la distribuzione media della parte (`py`).  La funzione `entropy` di `scipy.stats` calcola l'entropia, che è correlata alla KL divergence.
- Il punteggio per la parte è l'esponenziale della media delle divergenze KL.
- Infine, viene calcolata la media e la deviazione standard dei punteggi delle 10 parti, che rappresentano l'Inception Score e la sua deviazione standard.

L'output mostra l'Inception Score e la sua deviazione standard.  Un Inception Score più alto indica una maggiore qualità delle immagini generate, suggerendo una maggiore diversità e coerenza.


In conclusione, questo esempio dimostra come calcolare l'Inception Score utilizzando il modello Inception v3 pre-addestrato e un dataset di immagini.  Il codice è ben strutturato e facile da comprendere, offrendo una solida base per la valutazione della qualità delle immagini generate da modelli generativi.


