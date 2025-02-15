# Output processing per: Trasformata di Hough

## Metodo di splitting: headers
## Prompt utilizzati (1):
- TMP

---

## Chunk 1/2

### Risultato da: TMP
| Metodo/Funzione | Descrizione | Parametri | Output | Libreria |
|---|---|---|---|---|
| `imageio.imread()` | Carica un'immagine da un file. | Percorso del file immagine (stringa) | Array NumPy rappresentante l'immagine | imageio |
| `matplotlib.pyplot.imshow()` | Visualizza un'immagine. | Array NumPy rappresentante l'immagine, altri parametri opzionali (es. `cmap`) | Nessuno (visualizza l'immagine) | matplotlib.pyplot |
| `skimage.color.rgb2gray()` | Converte un'immagine a colori in scala di grigi. | Array NumPy rappresentante l'immagine a colori | Array NumPy rappresentante l'immagine in scala di grigi | skimage.color |
| `skimage.img_as_ubyte()` | Converte un'immagine in un array di byte unsigned a 8 bit. | Array NumPy rappresentante l'immagine | Array NumPy rappresentante l'immagine come byte unsigned a 8 bit | skimage |
| `cv2.Canny()` | Applica l'algoritmo di Canny per il rilevamento dei bordi. | Immagine in scala di grigi, soglia inferiore, soglia superiore, aperture size | Immagine binaria contenente i bordi rilevati | cv2 |
| `cv2.HoughLines()` | Applica la trasformata di Hough standard per il rilevamento delle linee. | Immagine dei bordi, risoluzione ρ, risoluzione θ, soglia, altri parametri opzionali | Array di linee, dove ogni linea è rappresentata da (ρ, θ) | cv2 |
| `compute_line_parameters(point1, point2)` | Calcola i parametri (pendenza e intercetta) di una retta dati due punti. | Due tuple o liste rappresentanti le coordinate (x, y) di due punti | Lista contenente pendenza e intercetta [m, n]  oppure None se la pendenza è indefinita | Nessuna libreria specifica (funzione definita dall'utente) |
| `cv2.HoughLinesP()` | Applica la trasformata di Hough probabilistica per il rilevamento dei segmenti di linea. | Immagine dei bordi, risoluzione ρ, risoluzione θ, soglia, altri parametri opzionali | Array di segmenti di linea, dove ogni segmento è rappresentato da (x1, y1, x2, y2) | cv2 |
| `cv2.line()` | Disegna una linea su un'immagine. | Immagine, punto iniziale, punto finale, colore, spessore | Immagine modificata con la linea disegnata | cv2 |
| `np.copy()` | Crea una copia di un array NumPy. | Array NumPy | Copia dell'array NumPy | NumPy |


**Nota:**  La tabella non include metodi come `plt.show()` o l'operatore `.` per l'accesso agli attributi, in quanto non sono funzioni o metodi in senso stretto nel contesto della programmazione orientata agli oggetti.  Sono invece funzioni di libreria o operatori del linguaggio Python.


---

## Chunk 2/2

### Risultato da: TMP
| Metodo/Funzione | Descrizione | Parametri | Output |
|---|---|---|---|
| `compute_line_parameters` | Calcola i parametri (pendenza e intercetta) di una retta dati due punti. | Due punti (presumibilmente coordinate x,y) | Lista `[-a/b, c/b]` (pendenza, intercetta) se `a` e `b` sono diversi da zero; altrimenti `None`. |
| `lambda x: line[0]*x + line[1]` | Funzione lambda che calcola il valore di y dato x, usando i parametri della retta. | `x` (valore x) | `y` (valore y corrispondente) |
| `np.linspace(-2500, 7500)` | Crea un array di numeri spaziati uniformemente tra -2500 e 7500. | Valore iniziale, valore finale | Array di numeri |
| `plt.plot(x, y, 'r')` | Disegna una linea rossa su un grafico. | `x` (array di valori x), `y` (array di valori y), 'r' (colore rosso) | Grafico con la linea disegnata |
| `cv2.GaussianBlur(coin, (31, 31), 5)` | Applica un filtro gaussiano all'immagine `coin` per ridurre il rumore. | `coin` (immagine di input), `(31, 31)` (dimensione del kernel), `5` (deviazione standard) | Immagine sfocata |
| `cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.9, minDist=120, param1=50, param2=30, minRadius=90, maxRadius=220)` | Applica la Trasformata di Hough per la detezione di cerchi. | `img` (immagine di input), `cv2.HOUGH_GRADIENT` (metodo), `0.9` (precisione), `minDist`, `param1`, `param2`, `minRadius`, `maxRadius` | Array di cerchi rilevati (coordinate x, y e raggio) |
| `np.uint16(np.around(circles_float))` | Converte un array di numeri a virgola mobile in un array di interi a 16 bit, arrotondando i valori. | `circles_float` (array di numeri a virgola mobile) | Array di interi a 16 bit |
| `cv2.circle(img_coin, c, r, (0,0, 255), 10)` | Disegna un cerchio sull'immagine `img_coin`. | `img_coin` (immagine), `c` (coordinate del centro), `r` (raggio), `(0,0,255)` (colore rosso), `10` (spessore) | Immagine con il cerchio disegnato |
| `print(circles)` | Stampa il contenuto dell'array `circles`. | `circles` (array) | Output sulla console |


Nota:  Alcuni parametri delle funzioni OpenCV (`cv2`) sono descritti nel testo, ma non sono esplicitamente definiti come parametri di funzione nella tabella, per semplicità.  La descrizione dei parametri è comunque inclusa nel testo.


---

