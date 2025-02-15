
## Caricamento e Preparazione delle Immagini

Il codice inizia importando le librerie necessarie:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
```

`os` è usato per la gestione dei file, `numpy` per le operazioni su array, `matplotlib.pyplot` per la visualizzazione, `PIL` (Pillow) per la manipolazione delle immagini e `torchvision.transforms` per le trasformazioni di tensori PyTorch.

La funzione `myResourcePath` gestisce il caricamento delle immagini da una cartella specifica:

```python
def myResourcePath(fname):
    filename = os.path.join(IMGSRC, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename
```

Questa funzione prende il nome del file come input (`fname`) e costruisce il percorso completo (`filename`) concatenandolo con il percorso base `IMGSRC` (definito precedentemente come 'data').  Verifica se il file esiste e, in caso contrario, solleva un'eccezione. Restituisce il percorso completo del file.

Un'immagine viene caricata usando `PIL`:

```python
pil_image = Image.open(myResourcePath('google_android.jpg'))
```

`Image.open()` apre l'immagine specificata dal percorso restituito da `myResourcePath()`. Il risultato è un oggetto `PIL.Image`.

Il tipo dell'oggetto `pil_image` viene stampato per verifica:

```python
type(pil_image)
```

## Visualizzazione dell'Immagine

Per visualizzare l'immagine con `matplotlib.pyplot.imshow()`, è necessario convertirla in un array NumPy o in un tensore PyTorch.  La documentazione di `imshow()` indica che accetta array con shape (M, N), (M, N, 3) o (M, N, 4).

Il codice converte l'immagine `PIL` in un tensore PyTorch e in un array NumPy:

```python
pil2tensor = transforms.ToTensor()
tensor_image = pil2tensor(pil_image)
pil2array = np.array(pil_image)
```

`transforms.ToTensor()` converte l'immagine PIL in un tensore PyTorch, normalizzando i valori dei pixel tra 0 e 1 e spostando la dimensione del canale (RGB) nella prima posizione.  `np.array()` converte l'immagine PIL in un array NumPy.

I tipi e i valori vengono stampati per verifica:

```python
print(f'type of tensor_image is {type(tensor_image)} and toString: {tensor_image}')
print(f'type of pil2array is {type(pil2array)} and toString: {pil2array}')
```

La funzione `plot_image` visualizza l'immagine usando `matplotlib`:

```python
def plot_image(tensor):
    plt.figure()
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()
```

Questa funzione prende un tensore come input.  Prima di passare il tensore a `imshow()`, lo converte in un array NumPy usando `.numpy()` e poi traspone le dimensioni usando `.transpose(1, 2, 0)` per ottenere la shape (H, W, C) richiesta da `imshow()`.  Infine, visualizza l'immagine usando `plt.show()`.

L'immagine viene visualizzata:

```python
plot_image(tensor_image)
```

![png](01_LoadingImage_7_0.png)

Le shape del tensore e dell'array NumPy vengono stampate:

```python
print(f'tensor shape {tensor_image.shape}')
print(f'np.array shape {pil2array.shape}')
```

La shape del tensore è (3, 416, 600) (canali, altezza, larghezza), mentre quella dell'array NumPy è (416, 600, 3) (altezza, larghezza, canali).  La trasposizione è necessaria per il tensore per renderlo compatibile con `imshow()`.

La trasformazione del tensore per renderlo compatibile con `imshow()` è mostrata:

```python
tensor_image.numpy().transpose(1, 2, 0).shape
```

Questo produce una shape (416, 600, 3), adatta per `imshow()`.


## Caricamento e Visualizzazione di Immagini con Python e Torchvision

Questo documento illustra come caricare e visualizzare immagini singolarmente e in dataset, utilizzando librerie Python come `matplotlib`, `numpy` e `torchvision`.

### 1. Visualizzazione di una singola immagine

Il codice seguente mostra come visualizzare un'immagine caricata utilizzando `matplotlib.pyplot` e `numpy`.  Si assume che `pil2array` sia un array NumPy che rappresenta l'immagine.

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure() # Crea una nuova figura
plt.imshow(pil2array) # Mostra l'immagine
plt.show() # Mostra la figura
```

Questo snippet utilizza `matplotlib.pyplot.imshow()` per visualizzare l'immagine.  `imshow()` richiede un array NumPy; se l'immagine è in un altro formato (ad esempio, PIL), deve essere convertita.  La funzione `plt.figure()` crea una nuova finestra per la visualizzazione.  `plt.show()` rende visibile la finestra.

![png](01_LoadingImage_11_0.png)  Questa immagine mostra un esempio di output di questo codice, visualizzando una singola immagine.


### 2. Caricamento di un Dataset di Immagini con `torchvision.datasets.ImageFolder`

In Computer Vision, è comune lavorare con dataset di immagini.  `torchvision.datasets.ImageFolder` semplifica questo processo.

```python
import torchvision.datasets as dataset_util
import torchvision.transforms as transforms

dataset = dataset_util.ImageFolder(myResourcePath('.'), transform=transforms.ToTensor())
```

`ImageFolder` crea un dataset a partire da una directory.  `myResourcePath('.')` indica il percorso della directory contenente le immagini (in questo caso, la directory corrente).  `transform=transforms.ToTensor()` converte ogni immagine in un tensore PyTorch, formato adatto per l'elaborazione con PyTorch.  Il costruttore di `ImageFolder` accetta altri parametri opzionali, come `target_transform` (per trasformare le etichette) e `loader` (per specificare una funzione di caricamento personalizzata).

Iterando sul dataset, ogni elemento è una tupla `(sample, target)`, dove `sample` è l'immagine e `target` è l'indice della classe a cui appartiene.

```python
for i, (item, c_index) in enumerate(dataset):
    print(f'{i} -> {item.shape}')
    plot_image(item) # Funzione non mostrata nel codice fornito, ma presumibilmente visualizza l'immagine 'item'
```

Questo codice itera sul dataset, stampando la forma di ogni immagine (`item.shape`) e visualizzandola tramite una funzione `plot_image` (non definita nel testo fornito, ma presumibilmente simile al codice della sezione 1).

L'output mostra la forma di ogni immagine (es. `torch.Size([3, 266, 400])`, che indica 3 canali, 266 righe e 400 colonne).

0 -> torch.Size([3, 266, 400]) ![png](01_LoadingImage_16_1.png)
1 -> torch.Size([3, 267, 400]) ![png](01_LoadingImage_16_3.png)
2 -> torch.Size([3, 267, 400]) ![png](01_LoadingImage_16_5.png)
3 -> torch.Size([3, 281, 400]) ![png](01_LoadingImage_16_7.png)

Queste immagini mostrano esempi delle immagini caricate dal dataset.


### 3. Visualizzazione di un Dataset di Immagini con Subplot

Il codice seguente mostra come visualizzare più immagini utilizzando un subplot di `matplotlib`.

```python
import matplotlib.pyplot as plt

image_list = [item for item, c_index in dataset] # Crea una lista di immagini dal dataset
fig = plt.figure(figsize=(8, 8)) # Crea una figura di dimensioni 8x8
columns = 2
rows = 2
for i in range(1, columns*rows +1):
    img = image_list[i - 1]
    fig.add_subplot(rows, columns, i) # Aggiunge un subplot alla figura
    plt.imshow(img.numpy().transpose(1, 2, 0)) # Mostra l'immagine, trasposta per matplotlib
    plt.show()
```

Questo codice crea una figura con 4 subplot (2x2) e visualizza le prime 4 immagini del dataset.  `img.numpy().transpose(1, 2, 0)` converte il tensore PyTorch in un array NumPy e lo traspone per renderlo compatibile con `imshow()`.

![png](01_LoadingImage_17_0.png) 
Questa immagine mostra l'output del codice, con le 4 immagini disposte in un subplot 2x2.



