
Questo documento illustra il calcolo della FID, una metrica utilizzata per confrontare due distribuzioni di immagini basate sulle loro statistiche di media e covarianza.  Il codice utilizza le librerie `numpy`, `scipy`, e `matplotlib`.

## Calcolo della FID

Il cuore del codice è la funzione `calculate_fid`:

```python
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
```

**Scopo:** Calcola la FID tra due insiemi di attivazioni di una rete neurale (o, in questo caso semplificato, due matrici di dati).

**Funzionamento:**

1. **Calcolo della media e della covarianza:**  La funzione calcola la media (`mu`) e la matrice di covarianza (`sigma`) per entrambi gli insiemi di dati (`act1` e `act2`) lungo l'asse 0 (cioè, per ogni feature). `np.cov(act1, rowvar=False)` calcola la matrice di covarianza, assumendo che le features siano in colonne (`rowvar=False`).

2. **Differenza quadratica tra le medie:** Calcola la somma delle differenze al quadrato tra le medie delle due distribuzioni (`ssdiff`).

3. **Radice quadrata della matrice prodotto delle covarianze:** Calcola la radice quadrata della matrice prodotto delle due matrici di covarianza usando `sqrtm` da `scipy.linalg`. Questa operazione richiede attenzione perché potrebbe restituire numeri complessi; il codice gestisce questo caso prendendo la parte reale (`covmean.real`).

4. **Calcolo della FID:** La FID è calcolata come la somma della differenza quadratica tra le medie e la traccia della somma delle matrici di covarianza meno due volte la radice quadrata della matrice prodotto delle covarianze.

**Parametri in ingresso:**

* `act1`: Matrice NumPy rappresentante il primo insieme di dati.
* `act2`: Matrice NumPy rappresentante il secondo insieme di dati.

**Valore restituito:**

* Un valore scalare rappresentante la FID tra `act1` e `act2`. Un valore basso indica una maggiore similarità tra le due distribuzioni.


## Esempio con vettori random

Questo esempio genera tre vettori random e calcola la FID tra coppie di questi vettori.  I vettori `item1` e `item3` sono molto simili, mentre `item1` e `item2` sono molto diversi.  Il codice normalizza i dati dividendo per 255 per ottenere valori tra 0 e 1.

```python
item1 = np.random.randint(0, 255, size=(2, 32 * 32))
item2 = np.random.randint(0, 255, size=(2, 32 * 32))
item3 = (item1 - 30).clip(0, 255)
FFitem1 = item1 / 255
FFitem2 = item2 / 255
FFitem3 = item3 / 255
fid1 = calculate_fid(FFitem1, FFitem1)
fid2 = calculate_fid(FFitem1, FFitem2)
fid3 = calculate_fid(FFitem1, FFitem3)
```

Il codice poi visualizza i vettori come immagini:

```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
ax1.imshow(item1[0].reshape(32, 32), cmap='gray'); ax1.set_title(f'Fig. 1. FID: {fid1:.5f}')
ax2.imshow(item2[0].reshape(32, 32), cmap='gray'); ax2.set_title(f'Fig. 2. FID: {fid2:.5f}')
ax3.imshow(item3[0].reshape(32, 32), cmap='gray'); ax3.set_title(f'Fig. 3. FID: {fid3:.5f}')
fig.show();
```

![png](FrechetInceptionDistanceExample_8_0.png)


Come previsto, la FID tra `FFitem1` e `FFitem3` è molto più bassa rispetto a quella tra `FFitem1` e `FFitem2`.


## Vettori non random

Questo esempio ripete il processo con vettori non random, creando immagini semplici con diverse tonalità di grigio.  L'interpretazione è analoga al caso precedente: immagini simili hanno FID basse.

```python
example1 = np.zeros((2, 32, 32))
example1[:,:16, :] = 255
# ... (resto del codice per creare example2, example3, example4 e calcolare le FID) ...
```

Il codice visualizza le immagini:

```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 8))
ax1.imshow(example1[0].reshape(32, 32), cmap='gray', vmin=0, vmax=255); ax1.set_title(f'Fig. 1. FID: {fid1:.5f}')
# ... (resto del codice per visualizzare le altre immagini) ...
fig.show();
```

![png](FrechetInceptionDistanceExample_11_0.png)


## Esempio con immagini

Questo esempio utilizza immagini reali (`cat1.jpg`, `cat2.jpg`, `nocat.jpg`) per calcolare la FID.  Il codice carica le immagini, le ridimensiona a 100x100 pixel e le converte in tensori usando `torchvision.transforms.ToTensor()`.  Il calcolo della FID richiederebbe l'estrazione delle attivazioni da una rete pre-addestrata (non incluso in questo snippet).


In sintesi, questo documento fornisce una spiegazione dettagliata del calcolo della FID e mostra diversi esempi di come applicarla a diversi tipi di dati, illustrando il suo utilizzo per quantificare la similarità tra distribuzioni di immagini.


Questo codice Python calcola la Frechet Inception Distance (FID) tra diverse immagini, utilizzando un modello Inception pre-addestrato. La FID è una metrica utilizzata per valutare la qualità di immagini generate da modelli di deep learning, confrontando la distribuzione delle feature estratte da un modello di percezione visiva (in questo caso, Inception).

**Sezione 1: Caricamento del modello e calcolo delle Feature Maps (FM)**

Il codice inizia caricando un modello Inception V3 pre-addestrato da `torchvision.models`:

```python
device = 'cuda:0' # oppure 'cpu'
from torchvision.models import inception_v3
inception_model = inception_v3(pretrained=True)
inception_model.to(device)
inception_model.eval() # reset classification layers
inception_model.fc = torch.nn.Identity()
```

* `device = 'cuda:0'`:  Specifica il dispositivo di elaborazione (GPU o CPU).  `'cuda:0'` indica la prima GPU disponibile.
* `inception_v3(pretrained=True)`: carica il modello Inception V3 con i pesi pre-addestrati su ImageNet.
* `inception_model.to(device)`: sposta il modello sul dispositivo specificato.
* `inception_model.eval()`: imposta il modello in modalità di valutazione (disattiva il dropout e la batch normalization).
* `inception_model.fc = torch.nn.Identity()`: sostituisce lo strato di classificazione finale (`fc`) con un'identità, in modo da ottenere le feature maps invece delle probabilità di classe.

Successivamente, vengono calcolate le feature maps (FM) per diverse immagini (`cat1`, `cat2`, `nocat`), ridimensionate a 100x100 pixel:

```python
cat1FF = inception_model(cat1.view(1, 3, 100, 100).to(device)).cpu().detach()
```

* `cat1.view(1, 3, 100, 100)`: rimodella il tensore dell'immagine `cat1` nella forma richiesta dal modello (batch size, canali, altezza, larghezza).
* `.to(device)`: sposta il tensore sul dispositivo di elaborazione.
* `inception_model(...)`: esegue l'inferenza del modello sull'immagine.
* `.cpu().detach()`: sposta il risultato sulla CPU e lo distacca dal grafo computazionale (non necessario per il calcolo della FID, ma utile per risparmiare memoria).

Lo stesso processo viene ripetuto per le immagini `cat2` e `nocat`.  Il risultato `cat1FF` (e analogamente per le altre immagini) è un tensore di forma `torch.Size([1, 2048])`, rappresentante le 2048 feature maps estratte dal modello.

Per simulare un batch di immagini (necessario per il calcolo della FID), vengono concatenati due esempi della stessa immagine:

```python
cat1FF = torch.cat((cat1FF, cat1FF), dim=0)
```

Questo crea un tensore di forma `torch.Size([2, 2048])`.


**Sezione 2: Calcolo della FID e visualizzazione dei risultati**

La FID viene calcolata utilizzando una funzione `calculate_fid` (non mostrata nel codice fornito, ma presumibilmente una funzione che calcola la distanza di Fréchet tra le distribuzioni delle feature maps):

```python
fid1 = calculate_fid(cat1FF.numpy(), cat1FF.numpy())
fid2 = calculate_fid(cat1FF.numpy(), cat2FF.numpy())
fid3 = calculate_fid(cat1FF.numpy(), nocatFF.numpy())
```

I risultati vengono poi visualizzati insieme alle immagini corrispondenti:

```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
ax1.imshow(cat1.numpy().transpose(1, 2, 0))
ax1.set_title(f'Fig. 1. FID: {fid1:.5f}')
# ... (simile per ax2 e ax3)
fig.show();
```

![png](FrechetInceptionDistanceExample_23_0.png)

Questa immagine mostra tre subplot, ognuno contenente un'immagine e il suo corrispondente valore FID.  Si nota che la FID è 0 quando si confronta un'immagine con se stessa, mentre è maggiore quando si confrontano immagini diverse.


**Sezione 3: Esempio con varianti della stessa immagine**

Questa sezione applica un filtro gaussiano con diversi valori di sigma all'immagine `cat1` e calcola la FID tra le immagini filtrate:

```python
from skimage.filters import gaussian
cat1filteredA = gaussian(cat1, sigma=.1)
cat1filteredB = gaussian(cat1, sigma=1)
cat1filteredC = gaussian(cat1, sigma=1.5)
```

![png](FrechetInceptionDistanceExample_25_0.png)

Questa immagine mostra le tre immagini filtrate con diversi valori di sigma.

Le feature maps vengono estratte dalle immagini filtrate e la FID viene calcolata tra le diverse coppie:

```python
cat1filteredFF_A = inception_model(tensorMapper(cat1filteredA).view(1, 3, 100, 100).to(device)).cpu().detach()
# ... (simile per B e C)
fidAB = calculate_fid(cat1filteredFF_A.numpy(), cat1filteredFF_B.numpy())
fidAC = calculate_fid(cat1filteredFF_A.numpy(), cat1filteredFF_C.numpy())
```

I risultati vengono visualizzati:

```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
ax1.imshow(cat1filteredA.transpose(1, 2, 0))
ax2.imshow(cat1filteredB.transpose(1, 2, 0))
ax2.set_title(f'Fig. FID A-B: {fidAB:.5f}')
ax3.imshow(cat1filteredC.transpose(1, 2, 0))
ax3.set_title(f'Fig. FID A-C: {fidAC:.5f}')
fig.show();
```

![png](FrechetInceptionDistanceExample_27_0.png)

Questa immagine mostra le immagini filtrate e i valori FID tra le coppie A-B e A-C.  Si osserva che la FID aumenta all'aumentare della differenza tra i valori di sigma, riflettendo la maggiore differenza tra le immagini filtrate.  La funzione `tensorMapper` non è definita nel codice fornito, ma si presume che sia una funzione di pre-processing per le immagini.


In sintesi, il codice fornisce un esempio di come calcolare la FID utilizzando un modello Inception pre-addestrato.  L'esempio mostra come la FID può essere utilizzata per confrontare la similarità tra immagini, sia immagini diverse che varianti della stessa immagine.  La bassa qualità del batch simulato (solo due ripetizioni della stessa immagine) limita l'affidabilità dei risultati, ma illustra il concetto di base del calcolo della FID.


