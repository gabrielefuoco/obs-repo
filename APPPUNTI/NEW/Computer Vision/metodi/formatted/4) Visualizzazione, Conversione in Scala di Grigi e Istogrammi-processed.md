
| Metodo/Funzione | Descrizione | Parametri | Output |
|---|---|---|---|
| `myResourcePath(fname)` | Controlla l'esistenza di un file nella cartella `IMGSRC` e restituisce il percorso completo. Solleva un'eccezione se il file non esiste. | `fname` (stringa: nome del file) | Stringa (percorso completo del file) o eccezione `RuntimeError` |
| `plot_image(tensor)` | Visualizza un tensore PyTorch come immagine RGB usando Matplotlib. | `tensor` (tensore PyTorch 3D: Canali, Altezza, Larghezza) | Nessuno (visualizza l'immagine) |
| `show_grayscale_image(tensor)` | Visualizza un tensore PyTorch come immagine in scala di grigi usando IPython.display. | `tensor` (tensore PyTorch 2D o 3D) | Nessuno (visualizza l'immagine) |
| `plot_grayscale_image(tensor)` | Visualizza un tensore PyTorch come immagine in scala di grigi usando Matplotlib. | `tensor` (tensore PyTorch 2D) | Nessuno (visualizza l'immagine) |
| `show_image(tensor)` | Converte un tensore PyTorch in un'immagine PNG e la visualizza usando IPython. | `tensor` (tensore PyTorch) | Nessuno (visualizza l'immagine) |
| `torch.histc(input, bins, min, max)` | Calcola l'istogramma di un tensore PyTorch. | `input` (tensore PyTorch), `bins` (numero di bin), `min` (valore minimo), `max` (valore massimo) | Tensore PyTorch contenente i conteggi per ogni bin |
| `tensor.mul(value)` | Moltiplica ogni elemento di un tensore per un valore scalare. | `value` (scalare) | Tensore PyTorch modificato |
| `tensor.clamp_(min, max)` | Limita i valori di un tensore tra un minimo e un massimo (in-place). | `min`, `max` (scalari) | Tensore PyTorch modificato (in-place) |
| `tensor.div(value)` | Divide ogni elemento di un tensore per un valore scalare. | `value` (scalare) | Tensore PyTorch modificato |
| `tensor.sum()` | Calcola la somma di tutti gli elementi di un tensore. | Nessuno | Scalare (somma degli elementi) |
| `tensor.numpy()` | Converte un tensore PyTorch in un array NumPy. | Nessuno | Array NumPy |
| `np.uint8(array)` | Converte un array in un array di interi senza segno a 8 bit. | `array` (array NumPy) | Array NumPy di interi senza segno a 8 bit |
| `array.transpose(axes)` | Trasforma le dimensioni di un array NumPy. | `axes` (tupla di indici) | Array NumPy trasposto |
| `Image.fromarray(array)` | Crea un'immagine PIL da un array NumPy. | `array` (array NumPy) | Oggetto immagine PIL |
| `Image.save(fp, format)` | Salva un'immagine PIL in un file. | `fp` (oggetto file), `format` (formato dell'immagine) | Nessuno |
| `torch.cat(tensors, dim)` | Concatena tensori lungo una dimensione specificata. | `tensors` (lista di tensori), `dim` (dimensione lungo cui concatenare) | Tensore PyTorch concatenato |
| `get_channels(rgbim, bins=10)` | Calcola gli istogrammi normalizzati dei canali RGB di un'immagine. | `rgbim` (tensore PyTorch RGB), `bins` (numero di bin, default 10) | Tre tensori PyTorch (istogrammi normalizzati per R, G, B) |
| `np.histogram(a, bins, range)` | Calcola l'istogramma di un array NumPy. | `a` (array NumPy), `bins` (numero di bin), `range` (intervallo) | Due array NumPy: `histogram` (conteggi) e `bin_edges` (bordi dei bin) |
| `plt.xlim(lim)` | Imposta i limiti dell'asse x di un grafico. | `lim` (tupla di due valori) | Nessuno |
| `plt.plot(x, y, color)` | Traccia una linea su un grafico. | `x` (array di valori x), `y` (array di valori y), `color` (colore della linea) | Nessuno |
| `plt.xlabel(label)` | Imposta l'etichetta dell'asse x di un grafico. | `label` (stringa) | Nessuno |
| `plt.ylabel(label)` | Imposta l'etichetta dell'asse y di un grafico. | `label` (stringa) | Nessuno |
| `plt.show()` | Mostra un grafico. | Nessuno | Nessuno |
| `mul(255)` | Moltiplica ogni elemento di un tensore per 255.  Si presume sia un metodo di una classe tensoriale (es. PyTorch). | Un tensore numerico. | Un tensore numerico con valori scalati. |
| `.numpy()` | Converte un tensore PyTorch in un array NumPy. | Nessuno (implicito dal contesto). | Un array NumPy. |
| `.transpose(1, 2, 0)` | Trasposta un array NumPy, riordinando le dimensioni. | Tre interi che specificano il nuovo ordine delle dimensioni. | Un array NumPy trasposto. |
| `show_chart()` | Visualizza un istogramma di un array NumPy (probabilmente un'immagine). | Un array NumPy rappresentante l'immagine. | Un grafico visualizzato (nessun valore di ritorno esplicito). |

**Note:**

* `pil2tensor` e `tensor2pil` sono trasformate di torchvision, non funzioni definite esplicitamente nel codice.  Sono utilizzate per la conversione tra immagini PIL e tensori PyTorch.
*  `Image.open()`, `torch.cat()`, `.clone()`, `.mul()`, `.clamp()`, `.div()`, `.transpose()`, `.numpy()` e altre funzioni/metodi di librerie come `numpy`, `PIL`, `torch` e `matplotlib` sono utilizzate nel codice ma non sono definite esplicitamente.  La tabella sopra si concentra sulle funzioni definite dall'utente.
* Le funzioni `print()` sono usate per output di testo, ma non sono considerate funzioni principali del codice di manipolazione immagini.
**Nota:**  `IPython.display.display` e `IPython.display.Image` sono usate all'interno di `show_image`, ma non sono definite come funzioni a sé stanti nel codice fornito.  Sono funzioni di IPython per la visualizzazione in un notebook Jupyter.  Inoltre, le funzioni di `matplotlib.pyplot` sono usate implicitamente nel codice per la visualizzazione degli istogrammi, ma non sono esplicitamente definite.
