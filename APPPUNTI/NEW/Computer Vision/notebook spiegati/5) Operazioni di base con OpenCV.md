
# Installazione e importazione delle librerie

Il documento inizia spiegando come installare OpenCV-Python.  Si consiglia la versione `headless` per sistemi senza interfaccia grafica utente (GUI):

```bash
pip install opencv-contrib-python-headless
```

Successivamente, vengono importate le librerie necessarie:

```python
import os
import numpy as np
import cv2  # OpenCV-Python
%matplotlib inline
import matplotlib.pyplot as plt
print("OpenCV-Python Version %s" % cv2.__version__)
```

Questo codice importa le librerie `os` (per la gestione dei file), `numpy` (per la manipolazione di array), `cv2` (OpenCV), `matplotlib` (per la visualizzazione di immagini) e stampa la versione di OpenCV installata.  `%matplotlib inline` è un comando specifico per Jupyter Notebook che permette di visualizzare i grafici direttamente nel notebook.

La funzione `myResourcePath` gestisce il percorso dei file immagine:

```python
IMGSRC = 'data'
def myResourcePath(fname):
    filename = os.path.join(IMGSRC, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename
```

Questa funzione prende il nome del file come input (`fname`) e costruisce il percorso completo (`filename`) concatenandolo con la directory `IMGSRC`.  Se il file non esiste, solleva un'eccezione `RuntimeError`.  Restituisce il percorso completo del file.


## Lettura e visualizzazione di un'immagine

Il codice legge un'immagine usando `cv2.imread()`:

```python
img = cv2.imread(myResourcePath('car.jpg'), cv2.IMREAD_COLOR)
if img is None:
    print('Open Error')
else:
    print('Image Loaded')
```

`cv2.imread()` legge l'immagine specificata dal percorso restituito da `myResourcePath('car.jpg')`.  `cv2.IMREAD_COLOR` specifica che l'immagine deve essere letta a colori.  Il codice controlla se la lettura è avvenuta correttamente verificando se `img` è `None`.

Successivamente, vengono stampate informazioni sull'immagine:

```python
print(type(img))
print(img.shape)
print(img.dtype)
print(img[:2, :2, :])
```

Questo codice stampa il tipo dell'immagine (`numpy.ndarray`), le sue dimensioni (`(altezza, larghezza, canali)`), il tipo di dati dei pixel (`uint8`) e i valori dei pixel nell'angolo superiore sinistro dell'immagine.

L'immagine viene visualizzata usando `matplotlib.pyplot.imshow()`:

```python
plt.imshow(cv2.imread(myResourcePath('car.jpg'), cv2.IMREAD_COLOR))
```

![png](03_opencv_5_1.png)

Questo codice visualizza l'immagine.  Si noti che OpenCV utilizza l'ordine BGR (Blu, Verde, Rosso) per i canali di colore, mentre Matplotlib usa RGB.  Quindi, per una corretta visualizzazione con Matplotlib, è necessario convertire l'immagine:

```python
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

![png](03_opencv_6_1.png)

`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` converte l'immagine dallo spazio colore BGR a RGB.


## Operazioni di base sulle immagini: Disegnare oggetti

Questo sezione mostra come disegnare forme geometriche su un'immagine usando OpenCV.  Viene creata un'immagine nera:

```python
img2 = np.zeros((512,512,3), np.uint8)
plt.imshow(img2)
```

![png](03_opencv_8_1.png)

Questo codice crea un array NumPy di dimensioni 512x512x3 (altezza, larghezza, canali) riempito con zeri, rappresentando un'immagine nera.

Vengono poi disegnate una linea, un rettangolo, un cerchio e un'ellisse usando le funzioni di OpenCV:

```python
cv2.line(img2, (0,0), (511,511), (255,0,0), 5) # Linea rossa diagonale
cv2.rectangle(img2, (384,0), (510,128), (0,255,0), 3) # Rettangolo verde
cv2.circle(img2, (447,63), 63, (0,0,255), -1) # Cerchio blu pieno
cv2.ellipse(img2, (256,256), (100,50), -45, 0, 180, (255,0,0), -1) # Ellisse rossa
```

Queste funzioni modificano direttamente l'immagine `img2`.  Ogni funzione prende come parametri l'immagine, i parametri geometrici della forma (punti, centro, raggi, assi, angoli) e il colore (in BGR) e lo spessore.  Uno spessore di -1 indica che la forma deve essere riempita.  Il codice non mostra la visualizzazione del risultato, ma si presume che venga visualizzato successivamente.


In sintesi, il documento fornisce una introduzione pratica all'utilizzo di OpenCV-Python per la lettura, visualizzazione e manipolazione di immagini, mostrando esempi concreti di operazioni di base.


## Analisi del codice Python per la manipolazione di immagini con OpenCV

Questo documento analizza il codice Python fornito, che utilizza la libreria OpenCV per manipolare immagini. Il codice è suddiviso in sezioni che trattano diverse operazioni, come il disegno di forme, la modifica di pixel e ROI (Region of Interest), e le trasformazioni geometriche.

### 1. Disegno di forme e testo

Questa sezione mostra come disegnare forme e testo su un'immagine usando OpenCV.

**1.1 Disegno di una linea poligonale:**

```python
pts = np.array([[10,10],[150,200],[300,150],[200,50]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img2,[pts],True,(0,255,255),3) # => Cyan closed quadrangle
```

Questo codice disegna un quadrilatero chiuso di colore ciano (0, 255, 255 in BGR) su un'immagine `img2`.  `cv2.polylines()` accetta i seguenti parametri:

* `img2`: L'immagine su cui disegnare.
* `[pts]`: Una lista contenente un array NumPy di punti che definiscono la poligonale.  `pts.reshape((-1,1,2))` rimodella l'array per adattarlo al formato richiesto da `cv2.polylines`.
* `True`: Indica che la poligonale deve essere chiusa.
* `(0,255,255)`: Il colore della linea in formato BGR.
* `3`: Lo spessore della linea.

Il risultato è mostrato nell'immagine seguente:

![png](03_opencv_10_2.png)


**1.2 Inserimento di testo:**

```python
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img2, 'OpenCV', (10,500), font, 4, (255,255,255), 10, cv2.LINE_AA)
```

Questo codice inserisce il testo "OpenCV" sull'immagine `img2`. `cv2.putText()` accetta i seguenti parametri:

* `img2`: L'immagine su cui scrivere il testo.
* `'OpenCV'`: Il testo da scrivere.
* `(10,500)`: Le coordinate del punto in basso a sinistra del testo.
* `font`: Il tipo di font.
* `4`: La dimensione del font.
* `(255,255,255)`: Il colore del testo in BGR (bianco).
* `10`: Lo spessore del testo.
* `cv2.LINE_AA`: Un flag per l'anti-aliasing, che rende il testo più liscio.

Il risultato è mostrato nell'immagine seguente:

![png](03_opencv_11_1.png)


### 2. Modifica di pixel e ROI

Questa sezione illustra come accedere e modificare i pixel di un'immagine, sia singolarmente che in regioni di interesse (ROI).

**2.1 Accesso a un singolo pixel:**

```python
img[50, 135]
```

Questo codice accede al valore del pixel nella riga 50 e colonna 135 dell'immagine `img`. Il valore restituito è un array NumPy che rappresenta il colore del pixel in formato BGR.

**2.2 Modifica di pixel:**

```python
for i in range(15):
    for j in range(15):
        img[50+i, 135+j] = (0, 255, 0)
```

Questo codice modifica i valori di pixel in una regione 15x15 attorno al pixel (50, 135), impostandoli tutti a verde (0, 255, 0 in BGR).

Il risultato è mostrato nell'immagine seguente:

![png](03_opencv_14_1.png)


**2.3 Manipolazione di ROI:**

```python
ball = img[80:115, 25:60]
img[80:115, 75:110] = ball
```

Questo codice estrae una ROI (ball) dall'immagine `img` e la copia in un'altra posizione all'interno della stessa immagine.  `img[80:115, 25:60]` seleziona una sotto-matrice (ROI) dell'immagine originale. Questa ROI viene poi copiata in `img[80:115, 75:110]`.

Il risultato è mostrato nell'immagine seguente:

![png](03_opencv_15_1.png)


### 3. Trasformazioni geometriche

Questa sezione mostra come applicare trasformazioni geometriche alle immagini usando OpenCV.

**3.1 Scaling:**

```python
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
```

Questo codice ridimensiona l'immagine `img` al doppio delle sue dimensioni originali usando l'interpolazione cubica (`cv2.INTER_CUBIC`).  `cv2.resize()` accetta come parametri l'immagine di input, le nuove dimensioni e il metodo di interpolazione.

Il risultato è mostrato nell'immagine seguente:

![png](03_opencv_18_2.png)


**3.2 Traslazione:**

```python
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img, M, (cols,rows))
```

Questo codice trasla l'immagine `img` di 100 pixel a destra e 50 pixel in basso.  `M` è la matrice di trasformazione, e `cv2.warpAffine()` applica la trasformazione all'immagine.  `(cols, rows)` specifica le dimensioni dell'immagine di output.

Il risultato è mostrato nell'immagine seguente:

![png](03_opencv_20_2.png)


**3.3 Rotazione:**

```python
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
```

Questo codice calcola la matrice di trasformazione per ruotare l'immagine di 90 gradi attorno al suo centro. `cv2.getRotationMatrix2D()` accetta come parametri il centro di rotazione, l'angolo di rotazione e il fattore di scala.  Il codice non mostra l'applicazione della rotazione tramite `cv2.warpAffine()`, ma il principio è analogo alla traslazione.


## Trasformazioni Affini

Le trasformazioni affini mantengono il parallelismo delle linee.  Per definirle, sono necessari tre punti dell'immagine di input e le loro corrispondenti posizioni nell'immagine di output.  La funzione `cv2.getAffineTransform` crea una matrice di trasformazione 2x3.  `cv2.warpAffine` applica poi questa trasformazione all'immagine.

Il codice seguente mostra un esempio:

```python
pts1 = np.float32([[50,50],[200,50],[50,200]]) # Punti nell'immagine di input
pts2 = np.float32([[10,100],[200,50],[100,250]]) # Punti corrispondenti nell'immagine di output
M = cv2.getAffineTransform(pts1,pts2) # Calcola la matrice di trasformazione
dst = cv2.warpAffine(img,M,(cols,rows)) # Applica la trasformazione
```

`cv2.getAffineTransform(pts1, pts2)` prende come input due array NumPy, `pts1` e `pts2`, contenenti le coordinate dei punti corrispondenti nell'immagine sorgente e di destinazione rispettivamente. Restituisce la matrice di trasformazione affine `M` (2x3).  `cv2.warpAffine(img, M, (cols, rows))` applica la trasformazione `M` all'immagine `img`, producendo un'immagine di output `dst` con dimensioni `(cols, rows)`.  Il codice visualizza poi l'immagine originale e quella trasformata.

![png](03_opencv_24_0.png)  Questa immagine mostra un esempio di trasformazione affine applicata ad un'immagine.


## Trasformazioni Prospettiche

Le trasformazioni prospettiche richiedono una matrice di trasformazione 3x3.  Anche in questo caso, le linee rette rimangono rette dopo la trasformazione.  Sono necessari quattro punti non collineari nell'immagine di input e le loro corrispondenti posizioni nell'immagine di output.  `cv2.getPerspectiveTransform` calcola la matrice di trasformazione 3x3, che viene poi applicata tramite `cv2.warpPerspective`.

Un esempio di codice:

```python
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]]) # Punti nell'immagine di input
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]]) # Punti corrispondenti nell'immagine di output
M = cv2.getPerspectiveTransform(pts1,pts2) # Calcola la matrice di trasformazione
dst = cv2.warpPerspective(img,M,(300,300)) # Applica la trasformazione
```

`cv2.getPerspectiveTransform(pts1, pts2)` è simile a `getAffineTransform`, ma richiede quattro punti e restituisce una matrice 3x3. `cv2.warpPerspective(img, M, (300, 300))` applica la trasformazione prospettica `M` all'immagine `img`, creando un'immagine di output `dst` con dimensioni (300, 300).

![png](03_opencv_26_0.png) Questa immagine mostra un esempio di trasformazione prospettica applicata ad un'immagine.


La funzione `cv2.warpPerspective` è descritta più dettagliatamente:

```python
dst = cv.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
```

* `src`: immagine di input.
* `M`: matrice di trasformazione 3x3.
* `dsize`: dimensioni dell'immagine di output.
* `dst`: immagine di output (opzionale).
* `flags`: metodi di interpolazione (es. `INTER_LINEAR`, `INTER_NEAREST`) e flag opzionale `WARP_INVERSE_MAP`.
* `borderMode`: metodo di estrapolazione dei pixel (es. `BORDER_CONSTANT`, `BORDER_REPLICATE`).
* `borderValue`: valore usato per i bordi costanti (default 0).


### Integrazione di Immagini tramite Trasformazione Prospettica

Il codice seguente mostra come utilizzare la trasformazione prospettica per integrare un logo in un'immagine più grande:

```python
pts1 = np.float32([(0,0),(cols2-1,0),(cols2-1,rows2-1),(0,rows2-1)]) # Vertici del logo
pts2 = np.float32([(671,314),(1084,546),(689,663),(386,361)]) # Posizione desiderata del logo nell'immagine principale
M = cv2.getPerspectiveTransform(pts1,pts2) # Matrice di trasformazione
cv2.warpPerspective(logo,M,(cols1,rows1),imgResult,borderMode=cv2.BORDER_TRANSPARENT) # Applica la trasformazione, usando il borderMode per la trasparenza
```

La matrice di trasformazione `M` viene calcolata e applicata al logo per posizionarlo correttamente nell'immagine principale `imgResult`, utilizzando `cv2.BORDER_TRANSPARENT` per mantenere la trasparenza del logo.

La matrice M calcolata è:

```
[[ 6.87722951e-02 -1.24273581e+00 6.71000000e+02]
 [ 8.66486286e-02 4.83370157e-02 3.14000000e+02]
 [-7.63013127e-04 -3.68781623e-04 1.00000000e+00]]
```

![png](03_opencv_28_2.png) Questa immagine mostra il risultato dell'integrazione del logo nell'immagine principale.



## Analisi del codice Python per l'elaborazione di immagini con OpenCV

Questo documento analizza diversi snippet di codice Python che utilizzano la libreria OpenCV per l'elaborazione di immagini.  Verranno spiegati i metodi utilizzati, i parametri in ingresso e i valori restituiti, illustrando il funzionamento di ogni sezione di codice.

### Estrazione di features da un istogramma

```python
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)
    plt.plot(hist, color = color)
    plt.xlim([0, 256])
print(f'flattened feature vector size: {np.array(features).flatten().shape}')
```

Questo codice itera sui canali di colore di un'immagine (presumibilmente `chans` contiene i singoli canali, e `colors` i colori corrispondenti per la visualizzazione). Per ogni canale:

1. `cv2.calcHist([chan], [0], None, [256], [0, 256])`: Calcola l'istogramma del canale corrente.  I parametri sono:
    - `[chan]`: L'immagine di input (un singolo canale).
    - `[0]`: L'indice del canale da considerare (0 per il primo canale).
    - `None`:  Maschera opzionale (nessuna maschera in questo caso).
    - `[256]`: Il numero di bin dell'istogramma (256 per i valori da 0 a 255).
    - `[0, 256]`: L'intervallo di valori.
2. `features.extend(hist)`: Aggiunge i valori dell'istogramma alla lista `features`.
3. `plt.plot(hist, color = color)`: Traccia l'istogramma.
4. `plt.xlim([0, 256])`: Imposta i limiti dell'asse x.

Infine, `print(f'flattened feature vector size: {np.array(features).flatten().shape}')` stampa la dimensione del vettore di features, che è un concatenazione degli istogrammi di tutti i canali.  L'output mostrato è `(768,)`, indicando un vettore di 768 elementi (256 bin per canale x 3 canali).

![png](03_opencv_30_1.png)  Questa immagine mostra probabilmente il grafico generato dal codice, con gli istogrammi dei tre canali di colore.


### Analisi dell'istogramma HSV

```python
img = cv2.imread(myResourcePath('car.jpg'), cv2.IMREAD_COLOR)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
plt.figure(figsize=(10,8))
plt.subplot(311); plt.title("Hue"); plt.hist(np.ndarray.flatten(hue), bins=180)
plt.subplot(312); plt.title("Saturation"); plt.hist(np.ndarray.flatten(sat), bins=128)
plt.subplot(313); plt.title("Luminosity Value"); plt.hist(np.ndarray.flatten(val), bins=128)
plt.show()
```

Questo codice legge un'immagine, la converte nello spazio colore HSV (`cv2.COLOR_BGR2HSV`) e poi visualizza gli istogrammi separati per i canali Hue, Saturation e Value.  `np.ndarray.flatten()` appiattisce ogni canale in un array 1D prima di creare l'istogramma con `plt.hist()`. Il numero di bin è specificato (180 per Hue, 128 per Saturation e Value).

![png](03_opencv_31_0.png) Questa immagine mostra i tre istogrammi (Hue, Saturation, Value) dell'immagine convertita nello spazio colore HSV.


### Operazioni di Blurring e Sharpening

```python
blur = cv2.GaussianBlur(img,(5,5),0)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
```

Questo codice applica un filtro Gaussian Blur all'immagine usando `cv2.GaussianBlur(img,(5,5),0)`.  `(5,5)` specifica la dimensione del kernel (5x5), e `0` calcola automaticamente la deviazione standard sigma.

![png](03_opencv_33_0.png) Questa immagine mostra una comparazione tra l'immagine originale e quella sfocata (blurred).


```python
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img2 = cv2.filter2D(img, -1, kernel)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
```

Questo codice applica un filtro di sharpening all'immagine usando `cv2.filter2D()`. Il kernel definito enfatizza il contrasto centrale, aumentando la nitidezza.

![png](03_opencv_34_0.png)  Questa immagine mostra una comparazione tra l'immagine originale e quella affinata (sharpened).

![png](03_opencv_35_0.png) Questa immagine mostra una comparazione tra l'immagine sfocata e quella affinata dopo la sfocatura.


### Image Thresholding

```python
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
```

Questo codice applica una soglia binaria all'immagine usando `cv2.threshold()`.  `127` è la soglia, `255` il valore massimo (bianco), e `cv2.THRESH_BINARY` specifica il tipo di soglia (valori sopra la soglia diventano 255, altrimenti 0).  Altri tipi di soglia sono mostrati nel codice successivo (`THRESH_BINARY_INV`, `THRESH_TRUNC`, `THRESH_TOZERO`, `THRESH_TOZERO_INV`).

![png](03_opencv_37_1.png) Questa immagine mostra l'immagine di input per il thresholding, un gradiente di grigio.

![png](03_opencv_38_0.png) Questa immagine mostra i risultati dei diversi tipi di thresholding applicati all'immagine di input.


### Operazioni Aritmetiche

```python
x = np.uint8([250])
y = np.uint8([10])
print( cv2.add(x,y) ) # 250+10 = 260 => 255
print( x+y ) # 250+10 = 260 % 256 = 4
```

Questo codice dimostra la differenza tra l'addizione usando `cv2.add()` (che satura a 255) e l'addizione standard di NumPy (che esegue un modulo 256).

```python
img2 = cv2.resize(img2, (W, H))
```

Questo codice ridimensiona l'immagine `img2` alle dimensioni di `img1` usando `cv2.resize()`.


In sintesi, il codice illustra diverse tecniche di elaborazione di immagini con OpenCV, dalla manipolazione degli istogrammi all'applicazione di filtri e operazioni aritmetiche.  L'utilizzo di Matplotlib permette la visualizzazione dei risultati intermedi e finali.


## Spiegazione del codice OpenCV

Questo documento spiega due blocchi di codice Python che utilizzano la libreria OpenCV per manipolare immagini.  Il primo esempio mostra come combinare due immagini usando `cv2.addWeighted`, mentre il secondo illustra l'utilizzo di operazioni bitwise per sovrapporre un logo su un'immagine.

### Sezione 1: Combinazione di immagini con `cv2.addWeighted`

Questo blocco di codice combina due immagini, `img1` e `img2`, utilizzando una media pesata.

```python
dst = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
```

* **`cv2.addWeighted(img1, alpha, img2, beta, gamma)`:** Questa funzione di OpenCV esegue una combinazione lineare di due immagini.
    * **`img1`:** La prima immagine.
    * **`alpha`:** Il peso applicato alla prima immagine (0.4 in questo caso).
    * **`img2`:** La seconda immagine.
    * **`beta`:** Il peso applicato alla seconda immagine (0.6 in questo caso).
    * **`gamma`:** Un valore costante aggiunto al risultato (0 in questo caso).
    * **Valore restituito:** L'immagine risultante dalla combinazione pesata.  La formula utilizzata è: `dst = src1*alpha + src2*beta + gamma`.

Il codice visualizza poi le tre immagini: l'originale `img1`, l'originale `img2` e l'immagine risultante `dst` utilizzando `matplotlib.pyplot`.

![png](03_opencv_41_1.png)  Questa immagine mostra il risultato della combinazione lineare delle due immagini, con `img2` più prominente a causa del peso maggiore (0.6).


### Sezione 2: Operazioni Bitwise per sovrapporre un logo

Questo blocco di codice sovrappone un logo (`img2`) su un'immagine principale (`img1`) utilizzando operazioni bitwise.

```python
# ... (caricamento e ridimensionamento delle immagini) ...

# Creazione di una Region of Interest (ROI)
roi = img1[0:rows, 0:cols]

# Creazione di una maschera dal logo
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Operazioni bitwise per sovrapporre il logo
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
dst = cv2.add(img1_bg, img2_fg)
img1edit[0:rows, 0:cols] = dst
```

* **`cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)`:** Converte l'immagine del logo in scala di grigi.
* **`cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY)`:** Crea una maschera binaria.  I pixel con valore maggiore di 230 diventano bianchi (255), altrimenti neri (0).
* **`cv2.bitwise_not(mask)`:** Crea l'inverso della maschera.
* **`cv2.bitwise_and(roi, roi, mask=mask_inv)`:** Esegue un'operazione AND bitwise tra la ROI e se stessa, usando la maschera inversa per oscurare la zona dove andrà il logo.
* **`cv2.bitwise_and(img2, img2, mask=mask)`:** Estrae solo la parte del logo definita dalla maschera.
* **`cv2.add(img1_bg, img2_fg)`:** Combina le due immagini risultanti.

Il codice stampa le dimensioni delle immagini:

```
shape img1 = (133, 200, 3)
shape img2 = (260, 462, 3)
shape img2 resized = (52, 92, 3)
```

Infine, il codice visualizza le immagini originale, il logo e l'immagine modificata.

![png](03_opencv_43_1.png) Questa immagine mostra l'immagine originale, il logo e il risultato finale con il logo sovrapposto correttamente.  L'operazione bitwise permette di integrare il logo senza creare artefatti di sovrapposizione.

