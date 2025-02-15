
| Metodo/Funzione | Descrizione | Parametri | Output |
|---|---|---|---|
| `myResourcePath(fname)` | Controlla l'esistenza di un file nella cartella `IMGSRC` e restituisce il percorso completo. Solleva un'eccezione se il file non esiste. | `fname` (stringa): nome del file | Stringa: percorso completo del file, oppure solleva un'eccezione `RuntimeError` |
| `data.chelsea()` | Carica l'immagine di esempio "chelsea" da scikit-image. | Nessuno | Array NumPy rappresentante l'immagine. |
| `data.rocket()` | Carica l'immagine di esempio "rocket" da scikit-image. | Nessuno | Array NumPy rappresentante l'immagine. |
| `data.astronaut()` | Carica l'immagine di esempio "astronaut" da scikit-image. | Nessuno | Array NumPy rappresentante l'immagine. |
| `plt.imshow()` | Visualizza un'immagine. | Array NumPy rappresentante l'immagine, altri parametri opzionali (es. `cmap`) | Nessuno (visualizza l'immagine) |
| `plt.figure()` | Crea una nuova figura. | Parametri opzionali per la configurazione della figura (es. `figsize`) | Oggetto matplotlib.figure.Figure |
| `plt.subplot()` | Crea un subplot all'interno di una figura. | Indici di riga, colonna e posizione del subplot | Oggetto matplotlib.axes._subplots.AxesSubplot |
| `np.linspace(start, stop, num)` | Crea un array di numeri spaziati uniformemente tra `start` e `stop`. | `start`, `stop`, `num` (numeri) | Array NumPy |
| `np.reshape(a, newshape)` | Rimodellata un array in una nuova forma. | `a` (array), `newshape` (tupla) | Array NumPy rimodellato |
| `astype(dtype)` | Converte il tipo di dato di un array. | `dtype` (tipo di dato) | Array NumPy con il nuovo tipo di dato |
| `plt.subplots()` | Crea una figura e un array di subplot. | Parametri opzionali per la configurazione della figura e dei subplot (es. `nrows`, `ncols`, `figsize`) | Oggetto matplotlib.figure.Figure e array di oggetti matplotlib.axes._subplots.AxesSubplot |
| `img_as_ubyte(image)` | Converte un'immagine nel formato [0, 255] (uint8). | `image` (array NumPy) | Array NumPy con tipo di dato uint8 |
| `img_as_float(image)` | Converte un'immagine nel formato [0, 1] (float). | `image` (array NumPy) | Array NumPy con tipo di dato float |
| `io.imread(fname)` | Legge un'immagine da un file. | `fname` (stringa): percorso del file | Array NumPy rappresentante l'immagine |
| `color.rgb2gray(image)` | Converte un'immagine a colori in scala di grigi. | `image` (array NumPy) | Array NumPy rappresentante l'immagine in scala di grigi |
| `io.ImageCollection(filenames)` | Crea una collezione di immagini da una lista di file. | `filenames` (lista di stringhe): percorsi dei file | Oggetto scikit-image.io.collection.ImageCollection |
| `plt.tight_layout()` | Regola automaticamente i parametri di spaziatura dei subplot per evitare sovrapposizioni. | Nessuno | Nessuno (modifica la figura in place) |
| `np.zeros_like(a)` | Crea un array di zeri con la stessa forma e tipo di dato di `a`. | `a` (array NumPy) | Array NumPy di zeri |
| `skdemo.imshow_with_histogram(image)` | Visualizza un'immagine e il suo istogramma. | `image` (array NumPy) | Nessuno (visualizza l'immagine e l'istogramma) |
| `skimage.transform.rotate()` | Ruota un'immagine di un angolo specificato. | `image` (immagine da ruotare), `angle` (angolo di rotazione in gradi) | Immagine ruotata |
| `skimage.transform.SimilarityTransform()` | Crea una trasformazione di similarità (combinazione di rotazione, traslazione e scaling). | `scale` (fattore di scala, opzionale), `translation` (traslazione in x e y, opzionale), `rotation` (angolo di rotazione in radianti, opzionale) | Oggetto `SimilarityTransform` |
| `skimage.transform.warp()` | Applica una trasformazione geometrica a un'immagine. | `image` (immagine da trasformare), `tform` (oggetto trasformazione) | Immagine trasformata |
| `skimage.filters.threshold_otsu()` | Calcola automaticamente una soglia ottimale per la binarizzazione di un'immagine in scala di grigi usando il metodo di Otsu. | `image` (immagine in scala di grigi) | Soglia ottimale (valore numerico) |
| `skimage.filters.gaussian()` | Applica un filtro gaussiano a un'immagine. | `image` (immagine), `sigma` (ampiezza del kernel gaussiano), `multichannel` (True se l'immagine è a colori) | Immagine filtrata |
| `skimage.filters.sobel()` | Applica il filtro Sobel per il rilevamento dei bordi (calcola il gradiente). | `image` (immagine) | Immagine con il gradiente calcolato |
| `skimage.filters.sobel_h()` | Applica il filtro Sobel orizzontale. | `image` (immagine) | Immagine con il gradiente orizzontale calcolato |
| `skimage.filters.sobel_v()` | Applica il filtro Sobel verticale. | `image` (immagine) | Immagine con il gradiente verticale calcolato |
| `skimage.feature.hog()` | Calcola l'istogramma dei gradienti orientati (HOG). | `image` (immagine), `feature_vector` (booleano, se True restituisce un singolo vettore, altrimenti un array multidimensionale), `orientations`, `pixels_per_cell`, `cells_per_block`, `visualize`, `multichannel` | Array multidimensionale (o vettore se `feature_vector=True`) e, se `visualize=True`, l'immagine HOG visualizzabile. |
| `data.camera()` | Carica un'immagine di esempio (presumibilmente una fotocamera). |  | Immagine |
| `data.astronaut()` | Carica un'immagine di esempio (presumibilmente un astronauta). |  | Immagine |
| `skdemo.imshow_all()` | Funzione di visualizzazione (non definita nel codice fornito). | `image` (una o più immagini), `titles` (titoli opzionali per le immagini) | Visualizza le immagini. |
| `np.deg2rad()` | (Presumibilmente da NumPy) Converte un angolo da gradi a radianti. | `angle` (angolo in gradi) | Angolo in radianti |

