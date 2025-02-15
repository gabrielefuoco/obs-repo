
Questo documento illustra l'applicazione della trasformata di Hough per rilevare linee in un'immagine.  Vengono utilizzati OpenCV e scikit-image per il processing dell'immagine.

## 1. Caricamento e Preprocessing dell'Immagine

```python
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray, gray2rgb
import skimage

building = np.array(imageio.imread('build_001.jpg'))
plt.imshow(building);
```

![png](trasformata_di_hough_2_0.png)

Questo blocco di codice importa le librerie necessarie (OpenCV, Matplotlib, scikit-image, NumPy) e carica un'immagine chiamata `build_001.jpg` utilizzando `imageio.imread()`. L'immagine viene visualizzata usando `matplotlib.pyplot.imshow()`.  L'immagine originale è a colori.

```python
gray_image = rgb2gray(building)
plt.imshow(gray_image, cmap='gray');
```

![png](trasformata_di_hough_3_0.png)

Successivamente, l'immagine viene convertita in scala di grigi usando `skimage.color.rgb2gray()`.  La conversione in scala di grigi è un passaggio di preprocessing comune per la semplificazione dell'elaborazione successiva. L'immagine in scala di grigi viene poi visualizzata.


## 2. Rilevamento dei Contorni con Canny

```python
img = skimage.img_as_ubyte(gray_image)
edges = cv2.Canny(img, 50, 200, None, 3)
plt.imshow(edges, cmap='gray');
```

![png](trasformata_di_hough_5_0.png)

Prima di applicare la trasformata di Hough, è necessario rilevare i bordi nell'immagine.  Questo viene fatto utilizzando l'algoritmo di Canny con `cv2.Canny()`.  `img` viene convertito in un tipo di dato unsigned byte (8-bit) usando `skimage.img_as_ubyte()` per compatibilità con OpenCV.  `cv2.Canny()` prende come input l'immagine in scala di grigi, due soglie (50 e 200 in questo caso) e un parametro per lo spessore del bordo (3).  L'output è un'immagine binaria contenente solo i bordi rilevati.


## 3. Trasformata di Hough Standard

```python
lines = cv2.HoughLines(edges, 1, np.pi / 180, 400, None, 0, 0)
lines.shape
```

Questo codice applica la trasformata di Hough standard utilizzando `cv2.HoughLines()`.  I parametri sono:
* `edges`: l'immagine dei bordi.
* `1`: la risoluzione della distanza ρ.
* `np.pi / 180`: la risoluzione dell'angolo θ (in radianti).
* `400`: la soglia minima per il numero di intersezioni necessarie per considerare una linea.

La funzione restituisce un array di linee, dove ogni linea è rappresentata da (ρ, θ).

```python
def compute_line_parameters(point1, point2):
    # ax + by = c
    # m = -a/b
    # n = c/b
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = a*(point1[0]) + b*(point1[1])
    if a != 0 and b != 0:
        return [-a/b, c/b]

# ... (codice per disegnare le linee sull'immagine) ...
```

La funzione `compute_line_parameters` calcola i parametri `m` (pendenza) e `n` (intercetta) dell'equazione della retta `y = mx + n` a partire da due punti.  Questo è utile per disegnare le linee sull'immagine originale.

Il codice successivo itera sulle linee rilevate, calcola i punti finali di ogni linea e li disegna sull'immagine originale.

![png](trasformata_di_hough_9_1.png)

L'immagine mostra le linee rilevate sovrapposte all'immagine originale usando la trasformata di Hough standard.


## 4. Trasformata di Hough Probabilistica

```python
lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, 400)
lines_p.shape
```

Questo codice applica la trasformata di Hough probabilistica usando `cv2.HoughLinesP()`.  Questa versione è più efficiente per immagini con molte linee.  Restituisce un array di segmenti di linea, dove ogni segmento è rappresentato da due punti (x1, y1, x2, y2).

```python
building_copy = np.copy(building)
for i in range(lines_p.shape[0]):
    pt1 = (lines_p[i, 0, 0], lines_p[i, 0, 1])
    pt2 = (lines_p[i, 0, 2], lines_p[i, 0, 3])
    cv2.line(building_copy, pt1, pt2, (255,0,0), 5)
plt.imshow(building_copy)
plt.show()
```

Questo codice itera sui segmenti di linea rilevati e disegna ogni segmento sull'immagine originale usando `cv2.line()`.

![png](trasformata_di_hough_13_0.png)

L'immagine mostra le linee rilevate sovrapposte all'immagine originale usando la trasformata di Hough probabilistica.  Si noti che questa versione rileva segmenti di linea invece di linee infinite.


In sintesi, questo documento dimostra come utilizzare la trasformata di Hough (sia standard che probabilistica) per rilevare linee in un'immagine, illustrando i passaggi di preprocessing, l'applicazione degli algoritmi e la visualizzazione dei risultati.  La trasformata di Hough probabilistica è generalmente preferita per la sua efficienza, soprattutto in presenza di un elevato numero di linee nell'immagine.


## Spiegazione del codice Python per la Trasformata di Hough

Questo codice dimostra l'applicazione della Trasformata di Hough per la detezione di linee e cerchi in immagini.  È suddiviso in due parti principali: la detezione di linee e la detezione di cerchi.

### Parte 1: Detezione di Linee

Questa sezione del codice identifica le linee presenti in un'immagine (presumibilmente `building`).

**1. Calcolo dei parametri della retta:**

```python
def compute_line_parameters(point1, point2):
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = a * (point1[0]) + b * (point1[1])
    if a != 0 and b != 0:
        return [-a / b, c / b]
```

Questo metodo `compute_line_parameters` calcola i parametri di una retta data due punti.  La retta è rappresentata nella forma implicita `ax + by = c`.  I parametri `a`, `b` e `c` sono calcolati usando la geometria analitica. La funzione restituisce una lista `[-a/b, c/b]` che rappresenta la pendenza (`m = -a/b`) e l'intercetta (`n = c/b`) della retta, solo se entrambi `a` e `b` sono diversi da zero (per evitare divisioni per zero).  Altrimenti, restituisce `None`.

**2. Iterazione sulle linee e disegno:**

Il codice itera su un array `lines_p` (non mostrato nel testo, ma presumibilmente contenente coordinate di punti che definiscono segmenti di linea) e per ogni coppia di punti, chiama `compute_line_parameters` per ottenere i parametri della retta. Se la retta viene calcolata correttamente (`line` non è `None`), viene aggiunta alla lista `plotted_lines`. Infine, le rette vengono disegnate sull'immagine `building`.

```python
for line in plotted_lines:
    f = lambda x: line[0]*x + line[1]  # Funzione lambda per calcolare y dato x
    x = np.linspace(-2500, 7500)       # Range di valori x
    y = f(x)                           # Calcolo dei valori y corrispondenti
    plt.plot(x, y, 'r')                # Disegna la retta in rosso
```

![png](trasformata_di_hough_14_1.png)  Questa immagine mostra il risultato della detezione delle linee sull'immagine `building`.


### Parte 2: Detezione di Cerchi

Questa parte del codice utilizza la Trasformata di Hough per individuare i cerchi in un'immagine di una moneta.

**1. Preprocessing dell'immagine:**

```python
coin_blur = cv2.GaussianBlur(coin, (31, 31), 5)
```

L'immagine della moneta (`coin`) viene sfocata usando un filtro gaussiano (`cv2.GaussianBlur`) per ridurre il rumore prima dell'applicazione della Trasformata di Hough.  Il filtro ha dimensioni (31, 31) e deviazione standard 5.

![png](trasformata_di_hough_16_0.png) Mostra l'immagine originale della moneta.
![png](trasformata_di_hough_18_0.png) Mostra l'immagine della moneta dopo l'applicazione del filtro gaussiano.


**2. Applicazione della Trasformata di Hough:**

```python
circles_float = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, minDist=120, param1=50, param2=30, minRadius=90, maxRadius=220)
circles = np.uint16(np.around(circles_float))
```

La funzione `cv2.HoughCircles` applica la Trasformata di Hough per la detezione di cerchi.  I parametri sono:

* `img`: L'immagine di input (in scala di grigi e di tipo `uint8`).
* `cv2.HOUGH_GRADIENT`: Metodo della Trasformata di Hough utilizzato.
* `0.9`:  Precisione dell'accumulatore.
* `minDist`: Distanza minima tra i centri dei cerchi.
* `param1`: Soglia superiore per il rilevamento dei bordi (Canny).
* `param2`: Soglia inferiore per il rilevamento dei cerchi nell'accumulatore.
* `minRadius`: Raggio minimo dei cerchi.
* `maxRadius`: Raggio massimo dei cerchi.

Il risultato è un array `circles` contenente le coordinate (x, y) e il raggio di ogni cerchio rilevato.

Prima dell'applicazione del filtro gaussiano, vengono rilevati 138 cerchi, mentre dopo l'applicazione del filtro, vengono rilevati 13 cerchi. Questo evidenzia l'importanza della pre-elaborazione per migliorare la robustezza del rilevamento.

```python
print(circles)
```

Questo mostra l'array `circles` contenente le coordinate (x, y) e il raggio di ogni cerchio rilevato, ordinati per coordinata x.

**3. Disegno dei cerchi:**

```python
for i in range(circles.shape[0]):
    c = (circles[i, 0], circles[i, 1])
    r = circles[i, 2]
    cv2.circle(img_coin, c, r, (0,0, 255), 10)
```

Questo codice itera sui cerchi rilevati e disegna un cerchio rosso su ogni posizione rilevata sull'immagine originale a colori `img_coin`.

![png](trasformata_di_hough_23_0.png) Mostra l'immagine della moneta con i cerchi rilevati sovrapposti.


In sintesi, il codice dimostra come utilizzare la Trasformata di Hough per rilevare linee e cerchi in immagini, evidenziando l'importanza della pre-elaborazione e la scelta appropriata dei parametri per ottenere risultati ottimali.  La documentazione di `cv2.HoughCircles` è richiamata per fornire maggiori dettagli sui parametri utilizzati.


