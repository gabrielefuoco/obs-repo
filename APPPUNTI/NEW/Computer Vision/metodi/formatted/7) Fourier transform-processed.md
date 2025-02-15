
| Metodo/Funzione | Descrizione | Parametri | Output |
|---|---|---|---|
| `mp_image.imread(path)` | Legge un'immagine dal percorso specificato. (Presumibilmente una funzione custom o da una libreria non standard) | `path` (stringa): percorso dell'immagine. | Array NumPy rappresentante l'immagine. |
| `img_show(image, cmap=cm.gray)` | Mostra un'immagine. (Presumibilmente una funzione custom) | `image` (array NumPy): immagine da visualizzare. <br> `cmap` (opzionale): mappa di colori (default: scala di grigi). | Nessuno (visualizza l'immagine). |
| `np.fft.fft2(image)` | Calcola la trasformata di Fourier bidimensionale di un'immagine. | `image` (array NumPy): immagine di input. | Array NumPy complesso rappresentante la trasformata di Fourier. |
| `np.fft.fftshift(im_fft)` | Sposta la componente continua (DC) al centro dello spettro di Fourier. | `im_fft` (array NumPy): spettro di Fourier. | Array NumPy con la componente DC al centro. |
| `np.abs(array)` | Calcola il valore assoluto di un array (o di un numero complesso). | `array` (array NumPy o numero complesso): array o numero di cui calcolare il valore assoluto. | Array NumPy (o numero) con i valori assoluti. |
| `np.log(array)` | Calcola il logaritmo naturale di un array. | `array` (array NumPy): array di input. | Array NumPy con i logaritmi naturali degli elementi. |
| `np.max(array)` | Trova il valore massimo in un array. | `array` (array NumPy): array di input. | Valore massimo nell'array. |
| `array.astype(dtype)` | Converte un array in un tipo di dato specificato. | `array` (array NumPy): array di input. <br> `dtype` (tipo di dato NumPy): tipo di dato desiderato. | Array NumPy convertito. |
| `distance(point1, point2)` | Calcola la distanza euclidea tra due punti. | `point1` (tupla): coordinate del primo punto (y, x). <br> `point2` (tupla): coordinate del secondo punto (y, x). | Distanza euclidea (float). |
| `BoxFilterLP(fraction, imgShape)` | Crea un filtro passa-basso circolare. | `fraction` (float): frazione di frequenze da mantenere. <br> `imgShape` (array NumPy): dimensioni dell'immagine (altezza, larghezza). | Array NumPy rappresentante il filtro passa-basso. |
| `np.zeros(shape)` | Crea un array di zeri con la forma specificata. | `shape` (tupla): forma dell'array. | Array NumPy di zeri. |
| `np.fft.ifftshift(array)` | Riporta la componente continua (DC) alla sua posizione originale nello spettro di Fourier. | `array` (array NumPy): spettro di Fourier. | Array NumPy con la componente DC nella posizione originale. |
| `np.fft.ifft2(array)` | Calcola la trasformata di Fourier inversa bidimensionale. | `array` (array NumPy): trasformata di Fourier. | Array NumPy complesso rappresentante la trasformata inversa. |
| `plt.figure(figsize=size)` | Crea una nuova figura. (Matplotlib) | `figsize` (tupla): dimensioni della figura. | Oggetto Figure di Matplotlib. |
| `fig.add_subplot(nrows, ncols, index)` | Aggiunge un subplot a una figura. (Matplotlib) | `nrows`, `ncols`: numero di righe e colonne di subplot. <br> `index`: indice del subplot. | Oggetto Axes di Matplotlib. |
| `plt.imshow(image, cmap=cmap)` | Mostra un'immagine. (Matplotlib) | `image` (array NumPy): immagine da visualizzare. <br> `cmap` (opzionale): mappa di colori. | Nessuno (visualizza l'immagine). |
| `plt.title(title)` | Imposta il titolo di un subplot. (Matplotlib) | `title` (stringa): titolo. | Nessuno. |
| `plt.axis('off')` | Nasconde gli assi di un subplot. (Matplotlib) | Nessuno. | Nessuno. |
| `plt.show()` | Mostra la figura. (Matplotlib) | Nessuno. | Nessuno. |
| `np.angle(array)` | Restituisce l'angolo (fase) di numeri complessi in un array. | `array` (array NumPy): array di numeri complessi. | Array NumPy con gli angoli (fasi). |
| `fig.add_subplot(1, 2, 1)`        | Aggiunge un subplot a una figura.               | `nrows`: numero di righe, `ncols`: numero di colonne, `index`: indice del subplot.                           | Oggetto subplot.               |
| `plt.imshow(image, cmap=cm.gray)` | Visualizza un'immagine.                         | `image`: matrice NumPy che rappresenta l'immagine, `cmap`: mappa di colori (in questo caso, scala di grigi). | Visualizzazione dell'immagine. |
| `plt.title("Original image")`     | Imposta il titolo di un subplot.                | `title`: stringa che rappresenta il titolo.                                                                  | Imposta il titolo del subplot. |
| `plt.axis('off')`                 | Nasconde gli assi di un subplot.                | Nessuno.                                                                                                     | Nasconde gli assi del subplot. |
| `fig.add_subplot(1, 2, 2)`        | Aggiunge un secondo subplot alla stessa figura. | `nrows`: numero di righe, `ncols`: numero di colonne, `index`: indice del subplot.                           | Oggetto subplot.               |
| `plt.show()`                      | Mostra la figura con tutti i subplot.           | Nessuno.                                                                                                     | Visualizzazione della figura.  |
