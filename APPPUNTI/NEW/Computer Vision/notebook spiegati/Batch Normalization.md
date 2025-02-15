

Questo documento spiega il codice Python fornito, focalizzandosi sulla Batch Normalization e sull'effetto dei parametri γ (gamma) e β (beta).

### Importazione delle librerie

```python
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
```

Questo blocco importa le librerie necessarie: NumPy per operazioni numeriche, PyTorch per la computazione tensoriale, `itertools` per iterare su combinazioni di parametri e `matplotlib.pyplot` per la visualizzazione dei grafici.  `%matplotlib inline` configura Matplotlib per visualizzare i grafici direttamente nell'output Jupyter Notebook.


### Generazione di un batch di immagini

```python
batch_size = 30
A = torch.zeros(batch_size, 32, 32)
for i in range(batch_size):
    A[i, :, :] = torch.randn(32 * 32).view(32, 32) * torch.randint(5, size=(1,)) + torch.randint(50, size=(1,))
A.shape, A
```

Questo codice genera un batch di 30 immagini ( `batch_size = 30` ), ciascuna di dimensione 32x32 pixel.  Ogni pixel è generato casualmente usando una distribuzione normale (`torch.randn`) moltiplicata per un intero casuale tra 0 e 4 (`torch.randint(5, size=(1,))`) e sommata ad un altro intero casuale tra 0 e 49 (`torch.randint(50, size=(1,))`). Questo crea una distribuzione di valori pixel non uniforme.  `A.shape` mostra la dimensione del tensore risultante (30, 32, 32).  `A` contiene il tensore stesso.

![image.png](47ca0f0c-ffc0-468c-b01f-5a1f43fa2ab8.png)  Questa immagine (mancante nel testo originale) dovrebbe mostrare un esempio della distribuzione dei valori dei pixel nel batch A generato.


### Funzione `describe_batch`

```python
def describe_batch(image_bn, ax1, titleprefix='Batch'):
    title = f'{titleprefix}\nMin {image_bn.min():.4f} - Max {image_bn.max():.4f}' \
            f'\nMean {image_bn.mean():.4f} - Var {image_bn.var():.4f}'
    ax1.hist(image_bn.flatten().numpy(), bins='auto')
    ax1.set_title(title)
```

Questa funzione crea un istogramma che visualizza la distribuzione dei valori dei pixel in un batch di immagini.  Prende in ingresso:

* `image_bn`: il tensore contenente il batch di immagini.
* `ax1`: un oggetto Matplotlib Axes su cui disegnare l'istogramma.
* `titleprefix`: un prefisso per il titolo dell'istogramma (default: 'Batch').

La funzione calcola il minimo, il massimo, la media e la varianza dei valori dei pixel e li include nel titolo dell'istogramma.  Poi appiattisce il tensore (`image_bn.flatten().numpy()`) e crea un istogramma usando `ax1.hist()`.

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
describe_batch(A, ax)
```

Questo codice crea una figura e un asse Matplotlib, poi chiama `describe_batch` per visualizzare l'istogramma del batch `A` generato precedentemente.

![png](Esempio_di_Batch_Normalization_4_0.png) Questa immagine mostra l'istogramma della distribuzione dei pixel del batch A prima della normalizzazione.


### Stima dei parametri gamma e beta

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

Questo codice esplora l'effetto dei parametri di scala (`gamma`) e di shift (`beta`) nella Batch Normalization.  `eps` è un piccolo valore aggiunto alla varianza per evitare divisioni per zero.  Il codice itera su diverse combinazioni di `gamma` e `beta` (definite in `values_gamma` e `values_beta`). Per ogni combinazione, applica la formula della Batch Normalization:

`correctedA = gamma * (A - A.mean()) / (torch.sqrt(A.var() + eps)) + beta`

Questa formula normalizza il batch `A` sottraendo la media e dividendo per la deviazione standard, poi scala il risultato con `gamma` e sposta con `beta`.  Infine, chiama `describe_batch` per visualizzare l'istogramma del batch normalizzato per ogni combinazione di `gamma` e `beta`.

![png](Esempio_di_Batch_Normalization_6_0.png) Questa immagine mostra una griglia di istogrammi, ognuno rappresentante la distribuzione dei pixel dopo la Batch Normalization con diverse combinazioni di gamma e beta.  Si può osservare come questi parametri influenzano la forma e la posizione della distribuzione.


In sintesi, il codice dimostra il processo di Batch Normalization e l'influenza dei parametri `gamma` e `beta` sulla distribuzione dei dati.  L'utilizzo di istogrammi permette una visualizzazione chiara dell'effetto della normalizzazione.


