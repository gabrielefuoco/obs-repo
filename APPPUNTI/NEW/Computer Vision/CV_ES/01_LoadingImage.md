## Getting Started with Images

In questo notebook vedremo come processare le immagini in python e come visualizzarle tramite la libreria matplotlib

```python
# import delle librerie

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Required magic to display matplotlib plots in notebooks

%matplotlib inline

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

# in questa folder sono memorizzati alcuni file a supporto (path relativo al notebook corrente)

IMGSRC = 'data'

def myResourcePath(fname):
    filename = os.path.join(IMGSRC, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename
```

```python
# loading an image as numpy array

pil_image = Image.open(myResourcePath('google_android.jpg'))
```

```python
type(pil_image)
```

## Visualizzazione di un'immagine

Possiamo visualizzare l'immagine appena caricata con il metodo *imshow* di matplotlib

Dalla documentazione sappiamo che il primo parametro richiede:

array-like or PIL image
The image data. Supported array shapes are:

(M, N): an image with scalar data. The values are mapped to colors using normalization and a colormap. See parameters norm, cmap, vmin, vmax.
(M, N, 3): an image with RGB values (0-1 float or 0-255 int).
(M, N, 4): an image with RGBA values (0-1 float or 0-255 int), i.e. including transparency.
The first two dimensions (M, N) define the rows and columns of the image.

Quindi dobbiamo convertire l'immagine in un oggetto array-like.

Tra le molte opzioni consideriamo un numpy array oppure un tensore pytorch

```python
# as tensor

# pytorch provides a function to convert PIL images to tensors.

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

tensor_image = pil2tensor(pil_image)

# as numpy array

pil2array = np.array(pil_image)
```

```python
print(f'type of tensor_image is {type(tensor_image)} and toString: {tensor_image}')
print(f'type of pil2array is {type(pil2array)} and toString: {pil2array}')
```

```python
# Plot the image here using matplotlib.

def plot_image(tensor):
    plt.figure()
    # imshow needs a numpy array with the channel dimension

    # as the the last dimension so we have to transpose things.

    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()

plot_image(tensor_image)
```

![png](01_LoadingImage_7_0.png)

```python
print(f'tensor shape {tensor_image.shape}')
print(f'np.array shape {pil2array.shape}')
```

tensor shape torch.Size([3, 416, 600])
np.array shape (416, 600, 3)

## Shape del tensore vs shape del numpy array

Per visualizzare con il metodo *imshow* l'array in input deve avere una shape **(H, W, rgb)**.

L'array numpy *pil2array* ha una shape già compatibile, invece il tensore deve essere trasformato.

L'istruzione seguente è un esempio di una trasformazione

tensore (rgb, H, W) -> numpy array (rgb, H, W) -> numpy array (H, W, rgb)

```python
tensor_image.numpy().transpose(1, 2, 0).shape
```

(416, 600, 3)

```python
# plot with numpy array

plt.figure()
# imshow needs a numpy array with the channel dimension

# as the the last dimension so we have to transpose things.

plt.imshow(pil2array)
plt.show()
```

![png](01_LoadingImage_11_0.png)

## Dataset di immagini

Nella CV è molto più frequente avere un dataset di immagini, quindi esitono dei metodo che facilitano il caricamento e la costruzione del dataset. Nel package torchvision è presente la classe ImageFolder che restituisce un oggetto che rappresenta il dataset

torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>, is_valid_file=None)
A generic data loader where the images are arranged in this way:

Iterando sul dataset, ogni elemento è rappresentato dalla tupla (sample, target) where target is class_index of the target class.

Nell'esempio le classi sono *dog* e *cat*

```python
IMAGE_DATASET = myResourcePath('nature')

for i, fname in enumerate(sorted(os.listdir(IMAGE_DATASET))):
    print(f'{i + 1}. {fname}')
```

```python
import torchvision.datasets as dataset_util

dataset = dataset_util.ImageFolder(myResourcePath('.'), transform=transforms.ToTensor())
```

```python
for i, (item, c_index) in enumerate(dataset):
    print(f'{i} -> {item.shape}')
```

0 -> torch.Size([3, 266, 400])
```python
for i, (item, c_index) in enumerate(dataset):
    print(f'{i} -> {item.shape}')
    plot_image(item)
```

0 -> torch.Size([3, 266, 400])
![png](01_LoadingImage_16_1.png)

1 -> torch.Size([3, 267, 400])
![png](01_LoadingImage_16_3.png)

2 -> torch.Size([3, 267, 400])
![png](01_LoadingImage_16_5.png)

3 -> torch.Size([3, 281, 400])
![png](01_LoadingImage_16_7.png)

```python
# visualizzazione con un subplot 2 x 2

image_list = [item for item, c_index in dataset] # len() == 4

fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(1, columns*rows +1):
    img = image_list[i - 1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img.numpy().transpose(1, 2, 0))
plt.show()
```

![png](01_LoadingImage_17_0.png)

