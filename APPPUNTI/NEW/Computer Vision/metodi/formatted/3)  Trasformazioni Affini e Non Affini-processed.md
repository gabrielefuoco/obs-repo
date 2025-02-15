
| Metodo/Funzione | Descrizione | Parametri | Output |
|---|---|---|---|
| `myResourcePath(fname)` | Costruisce il percorso completo di un file, sollevando un'eccezione se il file non esiste. | `fname` (stringa: nome del file) | Stringa (percorso completo del file) |
| `plot_image(np_array)` | Visualizza un array NumPy come immagine. | `np_array` (array NumPy) | Nessuno (visualizza un'immagine) |
| `img.rotate(angle)` | Ruota un'immagine di un angolo specificato. | `angle` (float: angolo di rotazione in gradi) | Oggetto Image (immagine ruotata) |
| `img.transpose(method)` | Applica una trasformazione geometrica all'immagine (rotazione o ribaltamento). | `method` (costante di `Image.TRANSPOSE`) | Oggetto Image (immagine trasformata) |
| `img.resize((width, height))` | Ridimensiona un'immagine alle dimensioni specificate. | `(width, height)` (tupla: larghezza e altezza desiderate) | Oggetto Image (immagine ridimensionata) |
| `img.crop((left, upper, right, lower))` | Ritaglia una porzione di un'immagine. | `(left, upper, right, lower)` (tupla: coordinate di inizio e fine del ritaglio) | Oggetto Image (immagine ritagliata) |
| `Image.open(filepath)` | Apre un'immagine dal percorso specificato. | `filepath` (stringa: percorso del file) | Oggetto Image |
| `np.array(img)` | Converte un oggetto Image in un array NumPy. | `img` (oggetto Image) | Array NumPy |
| `transforms.ToTensor()` | Converte un'immagine PIL in un tensore PyTorch. | Nessuno | Tensore PyTorch |
| `transforms.ToPILImage()` | Converte un tensore PyTorch in un'immagine PIL. | Nessuno | Oggetto Image |
| `dataset_util.ImageFolder(IMAGE_DATASET, transform=transforms.ToTensor())` | Carica un dataset di immagini da una cartella. | `IMAGE_DATASET` (stringa: percorso alla cartella), `transform` (trasformazione da applicare alle immagini) | Oggetto ImageFolder (dataset di immagini) |
| `matplotlib.pyplot` (implicita) | Visualizza un tensore come immagine. | Tensore (presumibilmente) | Immagine visualizzata |
| `torchvision.transforms.Compose` | Crea una pipeline di trasformazioni da applicare sequenzialmente. | Lista di trasformazioni | Oggetto Compose che applica le trasformazioni in sequenza |
| `torchvision.transforms.ToTensor` | Converte un'immagine in un tensore PyTorch. | Immagine | Tensore PyTorch rappresentante l'immagine |
| `torchvision.transforms.CenterCrop` | Ritaglia un'area centrale di un'immagine. | Dimensione del ritaglio (larghezza, altezza) | Immagine ritagliata |
| `torchvision.transforms.Resize` | Ridimensiona un'immagine. | Dimensione desiderata (larghezza, altezza) | Immagine ridimensionata |
| `torchvision.transforms.RandomErasing` | Cancella casualmente una parte di un'immagine. | Parametri opzionali per controllare la cancellazione (non specificati nel testo) | Immagine con parte cancellata |
| `torchvision.transforms.RandomChoice` | Sceglie casualmente una trasformazione da una lista. | Lista di trasformazioni | Trasformazione scelta casualmente |
| `torchvision.transforms.ColorJitter` | Modifica casualmente la luminosità e il contrasto di un'immagine. | Parametri opzionali per controllare la variazione di luminosità e contrasto (nel testo: brightness=10, contrast=10) | Immagine con luminosità e contrasto modificati |
| `torchvision.transforms.RandomRotation` | Ruota casualmente un'immagine. | Angolo di rotazione (in gradi) | Immagine ruotata |
| `torchvision.transforms.RandomVerticalFlip` | Ribalta casualmente un'immagine verticalmente. | Nessun parametro | Immagine ribaltata verticalmente |
| `torchvision.transforms.RandomAffine` | Applica una trasformazione affine casuale (rotazione, traslazione, ridimensionamento, shear). | Parametri opzionali per controllare rotazione, traslazione, ridimensionamento (nel testo: degrees, translate, scale, fillcolor) | Immagine trasformata affinemente |
| `dataset_util.ImageFolder` | Carica un dataset di immagini da una cartella. | Percorso della cartella, oggetto `transform` (opzionale) | Dataset di immagini |
| `show_tensor_image` | Visualizza un tensore come immagine (funzione non definita nel testo, ma utilizzata). | Tensore | Immagine visualizzata |
**Nota:**  `plt.figure()`, `plt.imshow()`, `plt.show()` sono metodi della libreria Matplotlib, ma non sono definite esplicitamente nel codice come funzioni o metodi personalizzati.  Similmente, le costanti di `Image.TRANSPOSE` (es. `Image.FLIP_LEFT_RIGHT`) sono attributi della libreria PIL, non funzioni.  La funzione `show_tensor_image` è menzionata ma non definita nel codice fornito.
