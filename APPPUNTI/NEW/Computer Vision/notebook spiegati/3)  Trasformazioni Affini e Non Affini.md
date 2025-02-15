
## 1. Importazione delle Librerie e Definizione di Funzioni Ausiliarie

Il codice inizia importando le librerie necessarie:

```python
import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO
import IPython.display
%matplotlib inline
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()
IMGSRC = 'data'

def myResourcePath(fname):
    filename = os.path.join(IMGSRC, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename
```

Questo blocco importa librerie per la manipolazione di immagini (`PIL`, `numpy`, `matplotlib`), per la creazione di tensori (`torch`, `torchvision.transforms`), per la gestione dei file (`os`) e per la visualizzazione (`IPython.display`, `matplotlib`).  `matplotlib.use('PS')` specifica il backend per la visualizzazione.  `pil2tensor` e `tensor2pil` sono trasformate predefinite di `torchvision.transforms` per la conversione tra immagini PIL e tensori PyTorch.  La funzione `myResourcePath` costruisce il percorso completo di un file di immagine, sollevando un'eccezione se il file non esiste.


## 2. Visualizzazione dell'Immagine e Analisi delle Dimensioni

La funzione `plot_image` visualizza un array NumPy come immagine:

```python
def plot_image(np_array):
    plt.figure()
    plt.imshow(np_array)
    plt.show()
```

Il codice poi apre un'immagine, la converte in un array NumPy e ne stampa le dimensioni:

```python
img = Image.open(myResourcePath('car.jpg'))
rgb_image = np.array(img)
print(f'image shape is {rgb_image.shape}')
plot_image(rgb_image)
```

L'output mostra le dimensioni dell'immagine (altezza, larghezza, canali colore): `image shape is (133, 200, 3)`.  Segue l'immagine stessa:

![png](02_transformations_2_1.png)


## 3. Confronto tra NumPy e Tensori PyTorch

Il codice evidenzia la differenza nella rappresentazione delle immagini tra NumPy e i tensori PyTorch:

> Numpy -> array 3D: H x W x rgb
>
> Tensor -> tensor 3D: rgb x H x W

Questa sezione spiega che NumPy rappresenta l'immagine come un array 3D con dimensioni Altezza x Larghezza x Canali colore (RGB), mentre PyTorch usa un tensore 3D con dimensioni Canali colore x Altezza x Larghezza.


## 4. Rotazione dell'Immagine

Il codice dimostra la rotazione di un'immagine usando due metodi:

**a) Rotazione con `rotate()` di PIL:**

```python
image2 = img.rotate(45)
plot_image(np.array(image2))
```

Questo codice ruota l'immagine di 45 gradi usando il metodo `rotate()` della libreria PIL.  L'immagine risultante è:

![png](02_transformations_4_0.png)

**b) Rotazione con `transpose()` di PIL:**

```python
image2 = img.transpose(Image.ROTATE_90)
plot_image(np.array(image2))
```

Questo codice ruota l'immagine di 90 gradi usando il metodo `transpose()` di PIL.  Il metodo `transpose()` permette rotazioni di multipli di 90 gradi e ribaltamenti. L'immagine risultante è:

![png](02_transformations_6_0.png)

Il codice poi itera su diverse costanti di `Image.TRANSPOSE` per mostrare tutti i possibili ribaltamenti e rotazioni:

```python
transpose_contants = {Image.FLIP_LEFT_RIGHT: 'FLIP_LEFT_RIGHT', Image.FLIP_TOP_BOTTOM: 'FLIP_TOP_BOTTOM', Image.ROTATE_90: 'ROTATE_90', Image.ROTATE_180: 'ROTATE_180', Image.ROTATE_270: 'ROTATE_270'}
for method, name in transpose_contants.items():
    print(f'try method {name}')
    image2 = img.transpose(method)
    plot_image(np.array(image2))
```

Le immagini risultanti sono:

* `FLIP_LEFT_RIGHT`: ![png](02_transformations_7_1.png)
* `FLIP_TOP_BOTTOM`: ![png](02_transformations_7_3.png)
* `ROTATE_90`: ![png](02_transformations_7_5.png)
* `ROTATE_180`: ![png](02_transformations_7_7.png)
* `ROTATE_270`: ![png](02_transformations_7_9.png)


## 5. Ridimensionamento dell'Immagine (Resizing)

Il codice mostra diversi esempi di ridimensionamento usando il metodo `resize()` di PIL:

```python
w, h = img.size
w2 = int(w / 2)
h2 = int(h / 2)
image2 = img.resize((w2, h2))
plot_image(np.array(image2))
```

Questo ridimensiona l'immagine a metà della sua dimensione originale.  L'immagine risultante è:

![png](02_transformations_8_1.png)

Altri esempi mostrano ridimensionamenti che modificano solo la larghezza o solo l'altezza, o che raddoppiano le dimensioni:

* Ridimensionamento a metà larghezza: ![png](02_transformations_9_1.png)
* Raddoppio delle dimensioni: ![png](02_transformations_10_1.png)


## 6. Ritaglio dell'Immagine (Cropping)

Il codice dimostra il ritaglio di una parte dell'immagine usando il metodo `crop()` di PIL:

```python
(left, upper, right, lower) = (7, 55, 185, 112)
image2 = img.crop((left, upper, right, lower))
plot_image(np.array(img))
plot_image(np.array(image2))
print(f'original size is {img.size}')
print(f'new size is {image2.size}')
```

Questo codice ritaglia una sezione dell'immagine definita dalle coordinate `(left, upper, right, lower)`.  Vengono mostrate sia l'immagine originale che quella ritagliata:

![png](02_transformations_11_0.png)  ![png](02_transformations_11_1.png)

Le dimensioni dell'immagine originale e di quella ritagliata vengono stampate a console.


In conclusione, il codice fornisce una panoramica completa delle trasformazioni geometriche di base su immagini usando la libreria PIL di Python, illustrando rotazioni, ridimensionamenti e ritagli.  L'utilizzo di NumPy e la breve introduzione ai tensori PyTorch forniscono un contesto più ampio per la manipolazione di immagini in ambito di elaborazione di immagini e deep learning.


## Trasformazioni di immagini con torchvision

**1. Caricamento del dataset e visualizzazione:**

Il codice inizia caricando un dataset di immagini utilizzando `torchvision.datasets.ImageFolder`.  `IMAGE_DATASET` indica il percorso alla cartella contenente le immagini.

```python
import torchvision.datasets as dataset_util
IMAGE_DATASET = myResourcePath('.') # load 'data' dir
dataset = dataset_util.ImageFolder(IMAGE_DATASET, transform=transforms.ToTensor())
for i, (item, c_index) in enumerate(dataset):
    print(f'{i} -> {item.shape}')
    show_tensor_image(item)
```

`dataset_util.ImageFolder` carica le immagini organizzate in sottocartelle, dove ogni sottocartella rappresenta una classe.  Il parametro `transform=transforms.ToTensor()` converte ogni immagine in un tensore PyTorch.  Il ciclo `for` itera sul dataset, stampando le dimensioni di ogni immagine (`item.shape`) e visualizzandola tramite la funzione `show_tensor_image`.  Quest'ultima, non mostrata esplicitamente, presumibilmente utilizza `matplotlib.pyplot` per visualizzare il tensore come immagine.

Le immagini seguenti mostrano le immagini originali del dataset prima di qualsiasi trasformazione:

0 -> torch.Size([3, 266, 400]) ![png](02_transformations_13_1.png)
1 -> torch.Size([3, 267, 400]) ![png](02_transformations_13_3.png)
2 -> torch.Size([3, 267, 400]) ![png](02_transformations_13_5.png)
3 -> torch.Size([3, 281, 400]) ![png](02_transformations_13_7.png)


**2. Applicazione di trasformazioni composte:**

Questo esempio dimostra l'applicazione di una sequenza di trasformazioni utilizzando `torchvision.transforms.Compose`.

```python
transformations = T.Compose([
    # T.CenterCrop(100),
    # T.Resize((32,32)),
    T.ToTensor(),
    # T.RandomErasing()
])
dataset = dataset_util.ImageFolder(IMAGE_DATASET, transform=transformations)
for i, (item, c_index) in enumerate(dataset):
    print(f'{i} -> {item.shape}')
    show_tensor_image(item)
```

`T.Compose` crea una pipeline di trasformazioni.  In questo caso, solo `T.ToTensor()` è attivo, convertendo le immagini in tensori.  Le altre trasformazioni (`T.CenterCrop`, `T.Resize`, `T.RandomErasing`) sono commentate.  `T.CenterCrop` ritaglia un'area centrale dell'immagine, `T.Resize` ridimensiona l'immagine, e `T.RandomErasing` cancella casualmente una parte dell'immagine.  Il risultato è la visualizzazione delle immagini ridimensionate a 32x32 pixel:

0 -> torch.Size([3, 32, 32]) ![png](02_transformations_14_1.png)
1 -> torch.Size([3, 32, 32]) ![png](02_transformations_14_3.png)
2 -> torch.Size([3, 32, 32]) ![png](02_transformations_14_5.png)
3 -> torch.Size([3, 32, 32]) ![png](02_transformations_14_7.png)


**3. Applicazione di una trasformazione casuale:**

Questo esempio utilizza `torchvision.transforms.RandomChoice` per applicare una trasformazione casuale ad ogni immagine.

```python
randomTransf = T.RandomChoice([
    T.CenterCrop(100),
    T.ColorJitter(brightness=10, contrast=10),
    T.RandomRotation(90),
    T.RandomVerticalFlip(),
    T.RandomAffine(degrees=45, translate=(0.8, 1)),
    T.RandomAffine(degrees=45, scale=(80, 100), fillcolor=(255,0,0)),
])
transformations = T.Compose([randomTransf, T.ToTensor()])
dataset = dataset_util.ImageFolder(IMAGE_DATASET, transform=transformations)
for i, (item, c_index) in enumerate(dataset):
    print(f'{i} -> {item.shape}')
    show_tensor_image(item)
```

`T.RandomChoice` sceglie casualmente una delle trasformazioni specificate ad ogni iterazione.  Le trasformazioni includono: `T.CenterCrop`, `T.ColorJitter` (modifica luminosità e contrasto), `T.RandomRotation` (rotazione casuale), `T.RandomVerticalFlip` (ribaltamento verticale), e due istanze di `T.RandomAffine` (trasformazioni affini con traslazione e ridimensionamento, rispettivamente).  Il risultato mostra le immagini trasformate casualmente, mantenendo le dimensioni originali:

0 -> torch.Size([3, 266, 400]) ![png](02_transformations_15_1.png)
1 -> torch.Size([3, 267, 400]) ![png](02_transformations_15_3.png)
2 -> torch.Size([3, 267, 400]) ![png](02_transformations_15_5.png)
3 -> torch.Size([3, 281, 400]) ![png](02_transformations_15_7.png)

