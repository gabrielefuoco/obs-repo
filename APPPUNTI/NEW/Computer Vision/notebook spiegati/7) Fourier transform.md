Questo codice Python dimostra l'applicazione di filtri passa-basso e passa-alto nel dominio delle frequenze su un'immagine utilizzando la trasformata di Fourier.  Analizziamo passo passo le diverse funzioni e operazioni.

**1. Caricamento dell'immagine e visualizzazione:**

```python
image = mp_image.imread(os.path.join(img_src,'einstein2.png'))
image = image[:,:,0] # Seleziona solo il canale di luminosità (grayscale)
img_show(image,cmap=cm.gray)
```

Questo codice carica l'immagine "einstein2.png" usando `mp_image.imread()` (presumibilmente una funzione custom o da una libreria non standard, che si occupa della lettura dell'immagine).  Viene poi selezionato solo il canale di luminosità (`[:,:,0]`), trasformando l'immagine a scala di grigi. Infine, `img_show()` (probabilmente una funzione di visualizzazione custom) mostra l'immagine originale.

![png](4.Fourier_transform_2_0.png)  
*Immagine originale di Einstein.*


**2. Trasformata di Fourier e visualizzazione dello spettro di ampiezza:**

```python
im_fft = np.fft.fft2(image)
img_fft_shited = np.fft.fftshift(im_fft)
f_abs = np.abs(img_fft_shited) + 1
f_bounded = np.log(1+f_abs)
f_img = 255 * f_bounded / np.max(f_bounded)
f_img = f_img.astype(np.uint8)
img_show(f_img,cmap=cm.gray)
```

Qui viene calcolata la trasformata di Fourier bidimensionale (`np.fft.fft2()`) dell'immagine.  `np.fft.fftshift()` sposta la componente continua (DC) al centro dello spettro per una migliore visualizzazione.  L'ampiezza dello spettro viene poi normalizzata e visualizzata usando una scala logaritmica (`np.log(1+f_abs)`) per evidenziare le componenti a bassa frequenza.  `astype(np.uint8)` converte l'immagine in un formato adatto alla visualizzazione.

![png](4.Fourier_transform_3_0.png) 
*Spettro di ampiezza della trasformata di Fourier dell'immagine.*


**3. Visualizzazione dello spettro di fase:**

```python
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(f_img,cmap=cm.gray)
plt.axis('off')
fig.add_subplot(1, 2, 2)
plt.imshow(np.angle(img_fft_shited),cmap=cm.gray)
plt.axis('off')
plt.show()
```

Questo codice visualizza sia lo spettro di ampiezza (già calcolato prima) che lo spettro di fase (`np.angle(img_fft_shited)`) della trasformata di Fourier.  La fase contiene informazioni sulla posizione spaziale delle frequenze.

![png](4.Fourier_transform_5_0.png) 
*Spettro di ampiezza (sinistra) e spettro di fase (destra).*


**4. Implementazione del filtro passa-basso (Box Filter):**

```python
def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def BoxFilterLP(fraction,imgShape):
    D0 = int(min(fraction*imgShape[:2]))
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 255
    return base

keep_fraction = 0.1
filter = BoxFilterLP(keep_fraction,np.array([r,c]))
img_show(filter,cmap=cm.gray)
```

`distance()` calcola la distanza euclidea tra due punti. `BoxFilterLP()` crea un filtro passa-basso circolare.  `fraction` determina la frazione di frequenze da mantenere (10% in questo caso). Il filtro è una matrice di zeri, tranne che all'interno di un cerchio centrato, dove i valori sono impostati a 255.

![png](4.Fourier_transform_7_0.png) 
*Filtro passa-basso circolare.*


**5. Applicazione del filtro passa-basso e visualizzazione:**

```python
LowPassCenter = img_fft_shited*filter
LowPass = np.fft.ifftshift(LowPassCenter)
# ... visualizzazione ...
```

Il filtro viene applicato moltiplicando lo spettro di ampiezza (`img_fft_shited`) per il filtro (`filter`). `np.fft.ifftshift()` riporta la componente continua alla sua posizione originale.

![png](4.Fourier_transform_9_0.png) 
*Spettro di ampiezza dopo l'applicazione del filtro passa-basso (centrato e non centrato).*


**6. Trasformata inversa di Fourier e visualizzazione dell'immagine filtrata:**

```python
inverse_LowPass = np.fft.ifft2(LowPass)
img_show(np.abs(inverse_LowPass),cmap=cm.gray)
```

La trasformata di Fourier inversa (`np.fft.ifft2()`) ricostruisce l'immagine filtrata.  Il valore assoluto (`np.abs()`) è necessario perché la trasformata inversa può restituire numeri complessi.

![png](4.Fourier_transform_11_0.png) 
*Immagine filtrata con il filtro passa-basso.*

**7. Filtro Passa-Alto:**

Il codice per il filtro passa-alto è simile, ma il filtro viene creato invertendo il filtro passa-basso (`filter = 255-filter`).

![png](4.Fourier_transform_15_0.png) 
*Filtro passa-alto, spettro dopo applicazione del filtro e spettro non centrato.*

![png](4.Fourier_transform_16_0.png) 
*Immagine filtrata con il filtro passa-alto.*

**8. Teorema di Convoluzione:**

Questo esempio dimostra il teorema di convoluzione, mostrando che la convoluzione nel dominio spaziale equivale alla moltiplicazione nel dominio delle frequenze.  Un kernel viene esteso per adattarsi alle dimensioni dell'immagine tramite padding.

```python
kernel_orig=np.array([ [1,0,-1], [2,0,-2], [1,0,-1]])
# ... padding del kernel ...
fft_k = np.fft.ifftshift(np.fft.fft2(kernel))
fft_im = np.fft.ifftshift(np.fft.fft2(img))
con_in_f=fft_im*fft_k
# ... visualizzazione ...
filtered = abs(np.fft.ifftshift(np.fft.ifft2(con_in_f)))
```

![png](4.Fourier_transform_19_0.png) 
*Kernel originale ed esteso.*

![png](4.Fourier_transform_20_0.png) 
*Trasformata di Fourier del kernel, dell'immagine e del loro prodotto.*

In sintesi, il codice illustra come applicare filtri passa-basso e passa-alto nel dominio delle frequenze usando la trasformata di Fourier, dimostrando concetti fondamentali di elaborazione delle immagini. La decentralizzazione (`fftshift`) è applicata per facilitare la visualizzazione e l'applicazione dei filtri, che sono tipicamente centrati nello spettro delle frequenze.


## Spiegazione del codice Python per la visualizzazione di immagini trasformate di Fourier

Il codice Python mostrato visualizza due immagini affiancate: l'immagine originale e la sua versione filtrata nel dominio della frequenza.  Utilizzando la libreria `matplotlib.pyplot`, crea una figura con due subplot per visualizzare le immagini.  Analizziamo il codice a blocchi:

**1. Visualizzazione delle immagini:**

Il codice principale si concentra sulla visualizzazione delle immagini usando `matplotlib.pyplot`.

```python
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(image, cmap=cm.gray)
plt.title("Original image")
plt.axis('off')
fig.add_subplot(1, 2, 2)
plt.title("Filtered (frequency domain)")
plt.imshow(filtered, cmap=cm.gray)
plt.axis('off')
plt.show()
```

* `fig = plt.figure(figsize=(10, 10))`: Crea una figura di dimensioni 10x10 pollici.
* `fig.add_subplot(1, 2, 1)`: Aggiunge un subplot alla figura.  `1, 2, 1` indica una griglia di 1 riga e 2 colonne, e seleziona il primo subplot (quello di sinistra).
* `plt.imshow(image, cmap=cm.gray)`: Visualizza l'immagine `image` in scala di grigi (`cmap=cm.gray`).  `image` è una matrice NumPy che rappresenta l'immagine.
* `plt.title("Original image")`: Imposta il titolo del subplot.
* `plt.axis('off')`: Nasconde gli assi del subplot.
* Le stesse istruzioni vengono ripetute per il secondo subplot (`fig.add_subplot(1, 2, 2)`), visualizzando l'immagine `filtered`, che presumibilmente rappresenta l'immagine originale dopo l'applicazione di un filtro nel dominio della frequenza.
* `plt.show()`: Mostra la figura con i due subplot.


**2. Immagine di esempio:**

![png](4.Fourier_transform_21_0.png)

L'immagine mostra il risultato dell'esecuzione del codice.  A sinistra è presente l'immagine originale (`image`), mentre a destra è mostrata l'immagine filtrata nel dominio della frequenza (`filtered`).  L'immagine illustra visivamente l'effetto del filtro applicato all'immagine originale.  Si nota una differenza tra le due immagini, suggerendo che il filtro ha modificato il contenuto dell'immagine nel dominio della frequenza, probabilmente rimuovendo o attenuando alcune componenti frequenziali.  Senza conoscere il codice che genera `filtered`, non è possibile specificare la natura del filtro utilizzato.


