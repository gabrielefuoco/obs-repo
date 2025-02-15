
## 1. Importazione delle librerie e definizione di funzioni ausiliarie

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
IMGSRC = 'data'

def myResourcePath(fname):
    filename = os.path.join(IMGSRC, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename
```

Questo blocco importa le librerie necessarie: `os` per la gestione dei file, `numpy` per le operazioni su array, `matplotlib.pyplot` per la visualizzazione delle immagini, e `PIL` (Pillow) per la manipolazione delle immagini.  `%matplotlib inline` configura Matplotlib per visualizzare i grafici direttamente nel notebook Jupyter.  La variabile `IMGSRC` specifica il percorso della cartella contenente le immagini di esempio. La funzione `myResourcePath` controlla l'esistenza di un file nella cartella `data` e restituisce il percorso completo se il file esiste, altrimenti solleva un'eccezione.


## 2. Immagini di esempio da scikit-image

```python
from skimage import data
img0 = data.chelsea()
img1 = data.rocket()
img2 = data.astronaut()
plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(img0)
plt.subplot(132)
plt.imshow(img1)
plt.subplot(133)
plt.imshow(img2)
```

Questo codice utilizza la funzione `data` di scikit-image per caricare tre immagini di esempio: "chelsea", "rocket" e "astronaut".  `plt.imshow` visualizza ciascuna immagine in un subplot separato.

![png](04_skimage_2_1.png)

L'immagine mostra le tre immagini di esempio caricate da scikit-image.


## 3. Creazione e visualizzazione di array lineari

```python
linear0 = np.linspace(0, 1, 2500).reshape((50, 50))
linear1 = np.linspace(0, 255, 2500).reshape((50, 50)).astype(np.uint8)
print("Linear0:", linear0.dtype, linear0.min(), linear0.max())
print("Linear1:", linear1.dtype, linear1.min(), linear1.max())
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 15))
ax0.imshow(linear0, cmap='gray')
ax1.imshow(linear1, cmap='gray')
```

Questo codice crea due array NumPy `linear0` e `linear1`.  `linear0` contiene valori in virgola mobile tra 0 e 1, mentre `linear1` contiene valori interi senza segno (uint8) tra 0 e 255.  Entrambi gli array vengono rimodellati in una matrice 50x50 e visualizzati come immagini in scala di grigi usando `matplotlib.pyplot.imshow`.  La stampa mostra il tipo di dato, il valore minimo e massimo di ciascun array.

Linear0: float64 0.0 1.0
Linear1: uint8 0 255

![png](04_skimage_3_2.png)

L'immagine mostra i due array lineari visualizzati come immagini in scala di grigi.


## 4. Conversione tra formati di immagine [0, 1] e [0, 255]

```python
from skimage import img_as_float, img_as_ubyte
image = data.chelsea()
image_ubyte = img_as_ubyte(image)
image_float = img_as_float(image)
print("type, min, max:", image_ubyte.dtype, image_ubyte.min(), image_ubyte.max())
print("type, min, max:", image_float.dtype, image_float.min(), image_float.max())
print()
print("231/255 =", 231/255.)
```

Questo codice utilizza le funzioni `img_as_ubyte` e `img_as_float` di scikit-image per convertire un'immagine tra i formati [0, 255] (uint8) e [0, 1] (float).  Vengono stampate le informazioni sul tipo di dato, il valore minimo e massimo dell'immagine convertita.


## 5. Caricamento e conversione di un'immagine da file

```python
from skimage import io, color
image = io.imread(myResourcePath('car.jpg'))
print(type(image), image.dtype, image.shape, image.min(), image.max())
gray = color.rgb2gray(image)
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(gray, cmap='gray')
```

Questo codice carica un'immagine da file usando `io.imread`, stampa le informazioni sull'immagine (tipo, tipo di dato, forma, valori minimi e massimi) e converte l'immagine a scala di grigi usando `color.rgb2gray`.  L'immagine originale e quella in scala di grigi vengono visualizzate.

![png](04_skimage_5_2.png)

L'immagine mostra l'immagine originale a colori e la sua versione in scala di grigi.


## 6. Caricamento di una collezione di immagini

```python
from skimage import io
ic = io.ImageCollection(os.path.join(IMGSRC, '*.jpg'))
print(f'Type: {type(ic)}')
print(ic.files)

f, axes = plt.subplots(nrows=2, ncols=len(ic) // 2 + 1, figsize=(20, 5))
axes = axes.ravel()
for ax in axes:
    ax.axis('off')
for i, image in enumerate(ic):
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(os.path.basename(ic.files[i]))
plt.tight_layout()
```

Questo codice carica una collezione di immagini JPEG dalla cartella `data` usando `io.ImageCollection`.  Viene stampata la lista dei file caricati.  Successivamente, le immagini vengono visualizzate in un subplot multi-immagine.

![png](04_skimage_7_0.png)

L'immagine mostra le immagini caricate dalla cartella `data`.


## 7. Separazione dei canali RGB

```python
import skdemo
image = data.chelsea()
red_image = np.zeros_like(image)
green_image = np.zeros_like(image)
blue_image = np.zeros_like(image)
red_image[:, :, 0] = image[:, :, 0]
green_image[:, :, 1] = image[:, :, 1]
blue_image[:, :, 2] = image[:, :, 2]
plt.figure(figsize=(12, 8))
plt.subplot(141)
plt.imshow(image)
plt.subplot(142)
plt.imshow(red_image)
plt.subplot(143)
plt.imshow(green_image)
plt.subplot(144)
plt.imshow(blue_image)
```

Questo codice separa i canali RGB dell'immagine "chelsea" e visualizza ogni canale separatamente.

![png](04_skimage_8_1.png)

L'immagine mostra l'immagine originale e i suoi canali RGB separati.


## 8. Visualizzazione di un'immagine con il suo istogramma (skdemo)

```python
skdemo.imshow_with_histogram?
skdemo.imshow_with_histogram(data.camera())
skdemo.imshow_with_histogram(data.chelsea())
```

Questo codice utilizza la funzione `skdemo.imshow_with_histogram` per visualizzare un'immagine insieme al suo istogramma.  La documentazione della funzione viene mostrata con il punto interrogativo (`?`).  Vengono mostrati esempi con le immagini "camera" e "chelsea".

![png](04_skimage_10_1.png)
![png](04_skimage_11_1.png)

Le immagini mostrano le immagini "camera" e "chelsea" con i rispettivi istogrammi.


Questo documento fornisce una panoramica completa degli esempi di codice forniti, spiegando il funzionamento di ogni blocco e il significato delle immagini generate.  L'utilizzo di commenti e una struttura chiara rende la spiegazione accessibile anche a lettori con una conoscenza base di Python e delle librerie utilizzate.


## Analisi del codice Python per le trasformazioni e i filtri di immagini con scikit-image

Questo documento analizza il codice Python fornito, che utilizza la libreria scikit-image per effettuare trasformazioni geometriche e applicare filtri su immagini.

### Sezione 1: Trasformazioni di base

Questa sezione illustra l'utilizzo di `skimage.transform` per eseguire roto-traslazioni e scaling di immagini.

**1.1 Rotazione:**

```python
from skimage import transform
image = data.camera()
rotated = transform.rotate(image, 45)
skdemo.imshow_all(image, rotated)
```

Questo codice ruota l'immagine `image` (caricata tramite `data.camera()`) di 45 gradi in senso antiorario usando la funzione `transform.rotate()`.  Il risultato `rotated` viene poi visualizzato insieme all'immagine originale usando una funzione `skdemo.imshow_all()` (non definita nel codice fornito, ma presumibilmente una funzione di visualizzazione).

![png](04_skimage_13_0.png)  *Immagine originale e immagine ruotata di 45 gradi.*


**1.2 Traslazione:**

```python
from skimage.transform import SimilarityTransform
tform = SimilarityTransform(translation=(-50, -100))
warped = transform.warp(image, tform)
skdemo.imshow_all(image, warped)
```

Qui viene creata una trasformazione di similarità (`SimilarityTransform`) che sposta l'immagine di -50 pixel sull'asse x e -100 pixel sull'asse y.  La funzione `transform.warp()` applica questa trasformazione all'immagine `image`, producendo l'immagine `warped`.  Anche in questo caso, `skdemo.imshow_all()` visualizza i risultati.

![png](04_skimage_14_0.png) *Immagine originale e immagine traslata.*


**1.3 Scaling, Rotazione e Traslazione combinate:**

```python
tform = SimilarityTransform(scale=.5, translation=(120, -70), rotation=np.deg2rad(30))
warped = transform.warp(image, tform)
skdemo.imshow_all(image, warped)
```

Questo esempio combina scaling, rotazione e traslazione.  `SimilarityTransform` riceve come parametri: `scale` (fattore di scala), `translation` (traslazione in x e y) e `rotation` (angolo di rotazione in radianti).  `transform.warp()` applica la trasformazione combinata.

![png](04_skimage_15_0.png) *Immagine originale e immagine scalata, ruotata e traslata.*


### Sezione 2: Filtri Skimage

Questa sezione mostra l'utilizzo del package `skimage.filters` per l'applicazione di diversi filtri.

**2.1 Soglia di Otsu:**

```python
from skimage import filters
threshold = filters.threshold_otsu(image)
print(f'suggested threashold is {threshold}')
skdemo.imshow_all(image, image > threshold)
```

Questo codice utilizza il metodo `filters.threshold_otsu()` per calcolare automaticamente una soglia ottimale per la binarizzazione di un'immagine in scala di grigi.  La soglia trovata viene stampata e poi utilizzata per creare un'immagine binaria (`image > threshold`).

suggested threashold is 102

![png](04_skimage_17_1.png) *Immagine originale e immagine binarizzata con la soglia di Otsu.*


**2.2 Filtro Gaussiano:**

```python
image = data.astronaut()
gaussian_result = filters.gaussian(image, sigma=10, multichannel=True)
skdemo.imshow_all(image, gaussian_result)
```

Questo codice applica un filtro gaussiano all'immagine `data.astronaut()` usando `filters.gaussian()`. Il parametro `sigma` controlla l'ampiezza del kernel gaussiano, mentre `multichannel=True` indica che l'immagine è a colori.

![png](04_skimage_18_0.png) *Immagine originale e immagine filtrata con filtro gaussiano.*


### Sezione 3: Rilevamento dei bordi (Edge Detection)

Questa sezione illustra l'utilizzo del filtro Sobel per il rilevamento dei bordi.

**3.1 Filtro Sobel:**

```python
pixelated = data.camera()
pixelated_gradient = filters.sobel(pixelated)
skdemo.imshow_all(pixelated, pixelated_gradient)
```

Il filtro Sobel (`filters.sobel()`) viene applicato all'immagine `pixelated` per calcolare il gradiente dell'immagine, evidenziando i bordi.

![png](04_skimage_20_0.png) *Immagine originale e immagine con il gradiente calcolato con il filtro Sobel.*


**3.2 Filtro Sobel orizzontale e verticale:**

```python
h_result = filters.sobel_h(pixelated)
v_result = filters.sobel_v(pixelated)
skdemo.imshow_all(pixelated, pixelated_gradient, h_result, v_result, titles=('Original', 'SObel', 'HSobel', 'VSobel'))
```

Questo codice applica separatamente i filtri Sobel orizzontale (`filters.sobel_h()`) e verticale (`filters.sobel_v()`), mostrando i risultati individuali e il risultato combinato.

![png](04_skimage_21_0.png) *Immagine originale, gradiente Sobel, gradiente Sobel orizzontale e verticale.*

Ulteriori esempi con immagini diverse vengono mostrati nelle figure  ![png](04_skimage_22_0.png), ![png](04_skimage_23_1.png), ![png](04_skimage_24_0.png) e ![png](04_skimage_25_0.png), dimostrando la robustezza del filtro Sobel anche con immagini ruotate e semplificate.


### Sezione 4: Histogram of Oriented Gradients (HOG)

Questa sezione mostra come calcolare e visualizzare l'istogramma dei gradienti orientati (HOG).

```python
out = skimage.feature.hog(image, feature_vector=False)
print(out.shape)
```

Questo codice calcola l'HOG dell'immagine `image` usando `skimage.feature.hog()`.  `feature_vector=False` restituisce un array multidimensionale invece di un singolo vettore.  La forma dell'array risultante viene stampata.

(35, 54, 3, 3, 9)

Il codice successivo visualizza l'immagine originale e l'immagine HOG:

```python
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
# ... codice per la visualizzazione con matplotlib ...
```

Questo codice calcola l'HOG e visualizza sia l'immagine originale che l'immagine HOG usando `matplotlib`. I parametri di `hog()` specificano il numero di orientazioni, la dimensione delle celle e dei blocchi. `visualize=True` indica che si vuole ottenere anche l'immagine HOG visualizzabile.

![png](04_skimage_27_0.png) *Immagine originale e immagine HOG.*


In sintesi, il codice illustra le funzionalità di scikit-image per la manipolazione e l'analisi di immagini, coprendo trasformazioni geometriche e diverse tecniche di filtraggio e feature extraction.  L'utilizzo di funzioni chiare e ben documentate rende il codice facile da comprendere e riutilizzare.


