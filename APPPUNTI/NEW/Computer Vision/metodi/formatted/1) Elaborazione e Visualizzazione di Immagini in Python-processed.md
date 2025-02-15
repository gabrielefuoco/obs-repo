
| Metodo/Funzione | Descrizione | Parametri | Output |
|---|---|---|---|
| `myResourcePath(fname)` | Costruisce il percorso completo di un file, verifica l'esistenza del file e restituisce il percorso completo. | `fname` (str): nome del file | `filename` (str): percorso completo del file, oppure solleva un'eccezione `RuntimeError` se il file non esiste. |
| `Image.open(filename)` | Apre un'immagine dal percorso specificato. | `filename` (str): percorso del file dell'immagine | Oggetto `PIL.Image` rappresentante l'immagine. |
| `transforms.ToTensor()` | Converte un'immagine PIL in un tensore PyTorch, normalizzando i valori dei pixel tra 0 e 1 e spostando la dimensione del canale (RGB) nella prima posizione. | Oggetto `PIL.Image` | Oggetto `torch.Tensor` rappresentante l'immagine. |
| `np.array(pil_image)` | Converte un'oggetto `PIL.Image` in un array NumPy. | `pil_image` (PIL.Image): immagine PIL | Array NumPy rappresentante l'immagine. |
| `plot_image(tensor)` | Visualizza un'immagine rappresentata da un tensore PyTorch usando matplotlib. | `tensor` (torch.Tensor): tensore PyTorch rappresentante l'immagine. | Nessun valore di ritorno, visualizza l'immagine. |
| `plt.figure()` | Crea una nuova figura matplotlib. | Nessuno | Oggetto `matplotlib.figure.Figure`. |
| `plt.imshow(array)` | Visualizza un'immagine rappresentata da un array NumPy o un tensore PyTorch. | `array` (NumPy array o torch.Tensor): array o tensore rappresentante l'immagine. | Nessun valore di ritorno, visualizza l'immagine. |
| `plt.show()` | Mostra la figura matplotlib. | Nessuno | Nessun valore di ritorno. |
| `type(object)` | Restituisce il tipo di un oggetto. | `object`: oggetto Python | Tipo dell'oggetto. |
| `print(object)` | Stampa un oggetto su console. | `object`: oggetto Python | Nessun valore di ritorno. |
| `tensor.numpy()` | Converte un tensore PyTorch in un array NumPy. | `tensor` (torch.Tensor): tensore PyTorch | Array NumPy. |
| `tensor.transpose(1, 2, 0)` | Traspone le dimensioni di un tensore. | `tensor` (torch.Tensor): tensore PyTorch, `1`, `2`, `0`: indici delle dimensioni da trasporre. | Tensore PyTorch con dimensioni trasposte. |
| `tensor.shape` | Restituisce la forma (shape) di un tensore. | `tensor` (torch.Tensor): tensore PyTorch | Tupla contenente le dimensioni del tensore. |
| `dataset_util.ImageFolder(root, transform=None, target_transform=None, loader=default_loader)` | Crea un dataset di immagini da una directory. | `root` (str): percorso della directory contenente le immagini; `transform`: trasformazione da applicare alle immagini; `target_transform`: trasformazione da applicare alle etichette; `loader`: funzione di caricamento personalizzata. | Oggetto `torchvision.datasets.ImageFolder` rappresentante il dataset. |
| `plt.figure()` | `matplotlib.pyplot` | Crea una nuova figura. | `figsize` (tuple): dimensioni della figura (larghezza, altezza) in pollici. | Oggetto `Figure` di matplotlib. |
| `fig.add_subplot()` | `matplotlib.figure.Figure` | Aggiunge un subplot alla figura. | `rows`, `columns`, `i`: numero di righe, colonne e indice del subplot. | Oggetto `Axes` di matplotlib. |
| `plt.imshow()` | `matplotlib.pyplot` | Mostra un'immagine. | Array NumPy (o tensore PyTorch convertito in NumPy) rappresentante l'immagine. | Nessun output esplicito, ma visualizza l'immagine. |
| `plt.show()` | `matplotlib.pyplot` | Mostra la figura con i subplot. | Nessun parametro. | Nessun output esplicito, ma visualizza la figura. |
| `img.numpy()` | `torch.Tensor` (implicito) | Converte un tensore PyTorch in un array NumPy. | Nessun parametro. | Array NumPy. |
| `.transpose(1, 2, 0)` | `numpy.ndarray` | Traspone un array NumPy. | `1`, `2`, `0`: indici delle dimensioni da permutare. | Array NumPy trasposto. |


**Nota:**  `[item for item, c_index in dataset]` è una list comprehension, non una funzione o un metodo in senso stretto.  Crea una lista a partire da un iterabile (il `dataset`).  La sua descrizione è inclusa nella descrizione del codice.

