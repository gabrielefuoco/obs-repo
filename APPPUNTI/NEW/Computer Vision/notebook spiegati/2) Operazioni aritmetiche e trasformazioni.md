
## 1. Caricamento e visualizzazione dell'immagine

Il codice inizia importando le librerie necessarie:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
```

`os` è usato per la gestione dei file, `numpy` per le operazioni su array, `matplotlib.pyplot` per la visualizzazione delle immagini e `PIL` (Pillow) per la manipolazione delle immagini.  `%matplotlib inline` permette la visualizzazione dei grafici direttamente nel notebook Jupyter.

La funzione `myResourcePath` gestisce il percorso del file immagine:

```python
IMGSRC = 'data'
def myResourcePath(fname):
    filename = os.path.join(IMGSRC, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename
```

Questa funzione prende il nome del file come input (`fname`) e restituisce il percorso completo del file, sollevando un'eccezione se il file non esiste.  `IMGSRC` definisce la cartella contenente le immagini.

L'immagine viene caricata e visualizzata usando `PIL` e `matplotlib`:

```python
image = Image.open(myResourcePath('car.jpg'))
plt.imshow(image, cmap='gray');
```

![png](01_math_operations_2_0.png)

Questo codice apre l'immagine "car.jpg" usando `Image.open()` e la visualizza in scala di grigi usando `plt.imshow()` con il `cmap='gray'`.


## 2. Conversione in scala di grigi e rappresentazione come array NumPy

L'immagine viene convertita in scala di grigi e rappresentata come un array NumPy:

```python
from scipy import misc
image = Image.open(myResourcePath('car.jpg')).convert('L')
plt.imshow(image, cmap='gray')
image_array = np.array(image)
print(f'shape = {image_array.shape}')
```

`image.convert('L')` converte l'immagine in scala di grigi (L sta per Luminance).  `np.array(image)` converte l'immagine in un array NumPy, la cui forma viene stampata a console.

![png](01_math_operations_4_1.png)

La forma dell'array rappresenta le dimensioni dell'immagine (altezza, larghezza).


## 3. Operazioni di manipolazione dell'immagine

Il codice esegue tre operazioni sull'array dell'immagine:

```python
# Slicing
img = image_array.copy()
img[40:50] = 255 # Imposta le righe 40-49 a bianco (255)

# Masks
lx, ly = img.shape
X, Y = np.ogrid[0:lx, 0:ly]
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
img[mask] = 0 # Imposta a nero i pixel fuori da un cerchio

# Fancy indexing
img[range(100), range(100)] = 255 # Imposta a bianco la diagonale principale fino a 100
plt.imshow(img, cmap='gray')
```

* **Slicing:**  `img[40:50] = 255` seleziona le righe da 40 a 49 e imposta i loro valori a 255 (bianco).
* **Maschere:** `np.ogrid` crea una griglia di coordinate. La maschera seleziona i pixel al di fuori di un cerchio e li imposta a 0 (nero).
* **Fancy indexing:** `img[range(100), range(100)] = 255` imposta a bianco i pixel lungo la diagonale principale dell'immagine fino al pixel (100,100).

![png](01_math_operations_6_2.png)

Un altro esempio di utilizzo di una maschera:

```python
# MASK
img = image_array.copy()
lx, ly = img.shape
mask = np.full((lx, ly), False)
mask[range(0, 20), :] = True
mask[range(lx - 20, lx), :] = True
img[mask] = 255
plt.imshow(img, cmap='gray')
```

Questa sezione crea una maschera che seleziona le prime 20 e le ultime 20 righe dell'immagine, impostandole a bianco.

![png](01_math_operations_7_2.png)


## 4. Trasformazioni geometriche

Infine, il codice esegue alcune trasformazioni geometriche usando `scipy.ndimage`:

```python
from scipy import ndimage
img = image_array.copy()
lx, ly = img.shape
# Cropping
crop_face = img[lx // 4: - lx // 4, ly // 4: - ly // 4]
# up <-> down flip
flip_ud_img = np.flipud(img)
# rotation
rotate_img = ndimage.rotate(img, 45)
rotate_img_noreshape = ndimage.rotate(img, 45, reshape=False)
plt.figure(figsize=(20, 12))
# ... (codice per visualizzare le immagini) ...
```

* **Cropping:**  `img[lx // 4: - lx // 4, ly // 4: - ly // 4]` ritaglia l'immagine, rimuovendo un quarto dai bordi.
* **Flip:** `np.flipud(img)` capovolge l'immagine verticalmente.
* **Rotation:** `ndimage.rotate(img, 45)` ruota l'immagine di 45 gradi.  `reshape=False` previene il ridimensionamento dell'immagine dopo la rotazione.

![png](01_math_operations_8_0.png)

Questo codice dimostra diverse tecniche di manipolazione di immagini, combinando slicing, maschere e trasformazioni geometriche per modificare e analizzare le immagini.  L'utilizzo di NumPy permette un'efficiente elaborazione degli array di pixel.


## Blurring/smoothing: Filtro Gaussiano vs Filtro Uniforme

Questa sezione confronta due metodi di blurring (sfocatura) di immagini: il filtro Gaussiano e il filtro Uniforme. Entrambi utilizzano la libreria `scipy.ndimage`.

### Filtro Gaussiano

Il filtro Gaussiano applica una convoluzione con un kernel gaussiano, producendo una sfocatura più naturale rispetto al filtro uniforme.  La sfocatura è controllata dal parametro `sigma`.

```python
blurred_image = ndimage.gaussian_filter(img, sigma=3)
very_blurred = ndimage.gaussian_filter(img, sigma=5)
```

* **`ndimage.gaussian_filter(img, sigma=3)`:** Questo metodo applica un filtro Gaussiano all'immagine `img`.  Il parametro `sigma` controlla la deviazione standard del kernel Gaussiano. Un valore di `sigma` maggiore produce una sfocatura più intensa.  In questo esempio, `sigma=3` produce una sfocatura moderata, mentre `sigma=5` produce una sfocatura più marcata. Il metodo restituisce l'immagine sfocata.


### Filtro Uniforme

Il filtro Uniforme calcola la media dei pixel all'interno di una finestra di dimensioni definite dal parametro `size`.  Produce una sfocatura più uniforme ma meno naturale rispetto al filtro Gaussiano.

```python
local_mean = ndimage.uniform_filter(img, size=11)
```

* **`ndimage.uniform_filter(img, size=11)`:** Questo metodo applica un filtro uniforme all'immagine `img`. Il parametro `size` specifica le dimensioni della finestra di media.  In questo caso, `size=11` indica una finestra 11x11 pixel. Il metodo restituisce l'immagine con la media locale applicata.


Il codice seguente visualizza i risultati:

```python
plt.figure(figsize=(20, 10))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 4, 2)
plt.imshow(blurred_image, cmap='gray')
plt.subplot(1, 4, 3)
plt.imshow(very_blurred, cmap='gray')
plt.subplot(1, 4, 4)
plt.imshow(local_mean, cmap='gray');
```

Questo codice utilizza `matplotlib.pyplot` per visualizzare l'immagine originale e le immagini filtrate.  `figsize` imposta le dimensioni della figura, `subplot` crea una griglia di subplot (4 in questo caso), e `imshow` visualizza le immagini in scala di grigi (`cmap='gray'`).

![png](01_math_operations_10_0.png)  *Immagine che mostra l'immagine originale e le immagini filtrate con filtro Gaussiano (sigma=3 e sigma=5) e filtro uniforme (size=11).*


## Sharpening

Questa sezione descrive un metodo per aumentare la nitidezza di un'immagine, invertendo parzialmente l'effetto di una sfocatura Gaussiana.

```python
img = image_array.copy()
blurred_image = ndimage.gaussian_filter(img, sigma=3)
filter_blurred = ndimage.gaussian_filter(blurred_image, sigma=1)
alpha = 10
sharpened = blurred_image + alpha * (blurred_image - filter_blurred)
```

* **`blurred_image = ndimage.gaussian_filter(img, sigma=3)`:**  L'immagine viene prima sfocata con un filtro Gaussiano (come visto nella sezione precedente).
* **`filter_blurred = ndimage.gaussian_filter(blurred_image, sigma=1)`:** L'immagine già sfocata viene ulteriormente sfocata con un filtro Gaussiano con `sigma=1`, creando una versione leggermente più sfocata.
* **`sharpened = blurred_image + alpha * (blurred_image - filter_blurred)`:** Questa è la chiave del processo di sharpening.  Si sottrae la versione più sfocata da quella meno sfocata, amplificando la differenza con il fattore `alpha`.  Il risultato viene poi aggiunto all'immagine originariamente sfocata, enfatizzando i dettagli e aumentando la nitidezza.


Il codice seguente visualizza i risultati:

```python
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
```

Questo codice, simile a quello precedente, visualizza l'immagine originale, l'immagine sfocata e l'immagine nitida.

![png](01_math_operations_12_1.png) *Immagine che mostra l'immagine originale, l'immagine sfocata e l'immagine nitida dopo l'applicazione del metodo di sharpening.*


