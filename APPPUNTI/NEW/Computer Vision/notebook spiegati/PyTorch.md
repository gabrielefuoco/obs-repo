
Questo documento fornisce una panoramica introduttiva a PyTorch, illustrando la creazione e la manipolazione di tensori, le operazioni di base e l'utilizzo della GPU per accelerare i calcoli.  Verranno analizzati anche gli aspetti legati alla differenziazione automatica.

## Creazione di Tensori

PyTorch offre diverse funzioni per creare tensori.

**1. Tensore con valori casuali:**

```python
x = torch.rand(5, 3)
print(x)
```

Questo codice crea un tensore `x` di dimensioni 5x3, popolato con numeri casuali distribuiti uniformemente tra 0 e 1.  `torch.rand()` è la funzione utilizzata per questa operazione. Il risultato è stampato a console.

**2. Tensore di zeri e uni:**

```python
x = torch.zeros(5, 3, dtype=torch.long) # Tensore di zeri
# ...
x = torch.ones(5,3) # Tensore di uni (dtype di default è float32)
```

`torch.zeros()` crea un tensore riempito con zeri, mentre (non mostrato esplicitamente nel testo ma deducibile dal commento) `torch.ones()` crea un tensore riempito con uni.  Il parametro `dtype` specifica il tipo di dato degli elementi del tensore (in questo caso, `torch.long` per interi a 64 bit).

**3. Tensore da lista multidimensionale:**

```python
x = torch.tensor([[5, 3, 1],[3,5,2], [4,4,4],[1,2,3],[0,0,1]])
print(x)
```

Questo codice crea un tensore a partire da una lista di liste (una matrice). `torch.tensor()` inferisce il tipo di dato dai valori della lista.

**4. Tipo di dato e dimensioni del tensore:**

```python
print(x.size())
print(x.shape)
```

`x.size()` e `x.shape` restituiscono le dimensioni del tensore come un oggetto `torch.Size` e una tupla, rispettivamente.  Entrambe forniscono le stesse informazioni.  L'esempio mostra che il tensore `x` ha dimensioni 5x3.

**5. Da NumPy a PyTorch e viceversa:**

```python
import numpy as np
x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()
print(type(z))
```

Questo snippet dimostra la conversione tra array NumPy e tensori PyTorch. `torch.from_numpy()` converte un array NumPy in un tensore PyTorch, mentre `.numpy()` esegue la conversione inversa.


## Operazioni di base

**1. Addizione di tensori:**

```python
x = torch.rand([5,3])
y = torch.rand([5,3])
z = x + y
print(z)
```

Questo codice esegue l'addizione elemento per elemento tra due tensori `x` e `y` di dimensioni identiche, memorizzando il risultato in `z`.

**2. Reshape con `view()`:**

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.shape, y.shape, z.shape)
```

Il metodo `view()` cambia la forma di un tensore senza copiare i dati.  `y` riorganizza `x` in un vettore di 16 elementi. `z` riorganizza `x` in una matrice 2x8, usando `-1` per dedurre automaticamente una dimensione in base alle altre.


## Utilizzo della GPU

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
x = torch.randn(5, 3)
y = torch.ones([5,3], device=device) # creazione diretta su GPU
x = x.to(device) # spostamento su GPU
z = x + y
print(z)
print(z.to("cpu"))
if torch.cuda.is_available():
    print('move y from cpu to cuda')
    y = torch.ones([5,3])
    y = y.cuda()
```

Questo codice verifica la disponibilità di una GPU. Se disponibile, crea un tensore `y` direttamente sulla GPU usando `device=device`. Altrimenti, usa la CPU.  `x.to(device)` sposta il tensore `x` sulla GPU. L'addizione viene eseguita sulla GPU e il risultato viene spostato sulla CPU per la stampa.  L'ultimo blocco mostra come spostare un tensore dalla CPU alla GPU usando `.cuda()`.


## Differenziazione automatica

```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

Questo codice crea un tensore `x` con `requires_grad=True`. Questo abilita il tracciamento automatico dei gradienti per il calcolo automatico delle derivate durante l'addestramento di modelli di machine learning.  Il tensore `x` è ora pronto per essere utilizzato in operazioni che richiedono il calcolo del gradiente.  (Il codice successivo, mancante nel testo fornito, mostrerebbe come calcolare i gradienti).


In sintesi, questo documento fornisce una solida introduzione alle funzionalità base di PyTorch, mostrando come creare, manipolare e utilizzare i tensori, sfruttando le potenzialità della GPU e preparando il terreno per l'utilizzo della differenziazione automatica.  Non sono presenti immagini nel testo fornito.


## Spiegazione del codice PyTorch

Questo documento spiega il codice PyTorch fornito, analizzando i metodi e illustrando il flusso di calcolo del gradiente e l'addestramento di una rete neurale.

### Calcolo del Gradiente con Autograd

Il codice inizia mostrando come PyTorch calcola automaticamente i gradienti utilizzando la funzione `autograd`.

```python
x = torch.zeros(2, 2, requires_grad=True) # Crea un tensore con requires_grad=True per abilitare il tracciamento del gradiente
y = x + 2
z = torch.pow(y, 2)
t = 3 * z
out = z.mean()
```

Questo snippet crea un tensore `x` e poi esegue una serie di operazioni per creare `y`, `z`, `t` e `out`.  `requires_grad=True` indica a PyTorch di tracciare le operazioni su `x` per il calcolo del gradiente. Ogni operazione crea un nodo nel grafo computazionale, memorizzando la funzione che mappa gli input agli output.  `grad_fn` mostra la funzione utilizzata nell'ultimo step di calcolo.

```python
print(f'grad z = {z.grad_fn}') # Mostra la funzione utilizzata per calcolare z (PowBackward0)
print(f'grad t = {t.grad_fn}') # Mostra la funzione utilizzata per calcolare t (MulBackward0)
print(f'grad out = {out.grad_fn}') # Mostra la funzione utilizzata per calcolare out (MeanBackward0)
```

Il metodo `.backward()` calcola il gradiente di `out` rispetto a `x`.

```python
out.backward()
print(x.grad) # Stampa il gradiente di out rispetto a x
```

![svg](pytorch_basics_24_0.svg)

L'immagine mostra il grafo computazionale generato da `torchviz.make_dot(out)`, visualizzando la dipendenza tra le variabili.  `out` dipende da `z`, che dipende da `y`, che dipende da `x`.  Il metodo `.backward()` percorre questo grafo all'indietro (backpropagation) per calcolare i gradienti.  Prima di chiamare `.backward()`, `x.grad` è `None`; dopo la chiamata, contiene il gradiente calcolato.


### Rete neurale per regressione lineare

La seconda parte del codice mostra come costruire e addestrare una semplice rete neurale per la regressione lineare.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ... (definizione del dataset x_train e y_train) ...

plt.plot(x_train, y_train, 'ro')
plt.show()
```

![png](pytorch_basics_28_0.png)

L'immagine mostra il dataset utilizzato per la regressione lineare.

```python
# Linear regression model
model = nn.Linear(input_size, output_size)
# Loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Questo codice definisce un modello di regressione lineare usando `nn.Linear`, una funzione di perdita MSE (`nn.MSELoss`) e un ottimizzatore SGD (`torch.optim.SGD`).  `nn.Linear` implementa la trasformazione lineare  `y = A^T x + b`, dove `A` e `b` sono i parametri del modello che vengono appresi durante l'addestramento.

```python
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

Questo loop di addestramento esegue il forward pass (`outputs = model(inputs)`), calcola la perdita (`loss = criterion(outputs, targets)`), esegue il backward pass (`loss.backward()`) per calcolare i gradienti e aggiorna i parametri del modello usando l'ottimizzatore (`optimizer.step()`).  `optimizer.zero_grad()` azzera i gradienti accumulati prima di ogni iterazione per evitare l'accumulo di gradienti da iterazioni precedenti.


In sintesi, il codice dimostra il calcolo automatico dei gradienti in PyTorch usando `autograd` e l'addestramento di una semplice rete neurale per la regressione lineare, illustrando i concetti fondamentali del deep learning con PyTorch.


## Analisi del codice Python per regressione e classificazione con PyTorch

Questo documento analizza due esempi di codice Python che utilizzano PyTorch per la regressione lineare e la regressione logistica.

### Sezione 1: Regressione Lineare

Questa sezione mostra un esempio di regressione lineare, dove si cerca di adattare una retta ai dati di training.

**1.1 Visualizzazione dei risultati della regressione:**

Il codice inizia visualizzando graficamente i dati originali e la retta di regressione ottenuta dal modello.

```python
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
```

* `predicted = model(torch.from_numpy(x_train)).detach().numpy()`:  Questo calcola le predizioni del modello (`model`) sui dati di training (`x_train`).  `torch.from_numpy(x_train)` converte l'array NumPy in un tensore PyTorch. `.detach()` separa il grafo computazionale, evitando il calcolo del gradiente su questa parte. `.numpy()` converte il tensore PyTorch risultante in un array NumPy per la compatibilità con Matplotlib.
* `plt.plot(x_train, y_train, 'ro', label='Original data')`:  Crea un grafico a dispersione dei dati originali.
* `plt.plot(x_train, predicted, label='Fitted line')`:  Crea un grafico della retta di regressione.
* `plt.legend()`: Aggiunge una legenda al grafico.
* `plt.show()`: Mostra il grafico.

![png](pytorch_basics_34_0.png)  Questa immagine mostra il grafico generato dal codice sopra, con i punti dati originali e la retta di regressione.


**1.2 Confronto tra valori predetti e valori reali:**

Il codice successivo confronta i valori predetti dal modello con i valori reali dei dati di training.

```python
for x, y, hat_y in zip(x_train, y_train, predicted):
    print(f'input = {x[0]:.3f} \t->\t output = {hat_y[0]:.3f} \t (y = {y[0]:.3f})')
```

Questo codice itera attraverso le tuple di `x_train`, `y_train` e `predicted`, stampando per ogni esempio l'input, l'output predetto e l'output reale, formattati con tre cifre decimali.


### Sezione 2: Regressione Logistica

Questa sezione mostra un esempio di regressione logistica per la classificazione binaria.  L'obiettivo è addestrare un modello che classifichi i dati in base alla regola: se x ≥ 6, allora classe = 1, altrimenti classe = 0.

**2.1 Dati di training:**

```python
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([ [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0] ],dtype=np.int)
```

Questo codice definisce i dati di training: `x_train` contiene i valori di input e `y_train` contiene le corrispondenti classi (0 o 1).


**2.2 Addestramento del modello:**

```python
model = nn.Linear(input_size, output_size) # Definisce un modello lineare
criterion = nn.BCELoss() # Definisce la funzione di loss (Binary Cross Entropy)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Definisce l'ottimizzatore Adam
num_epochs = 20000
log_freq = num_epochs / 10
inputs = torch.from_numpy(x_train).float()
targets = torch.from_numpy(y_train).float()
for epoch in range(num_epochs):
    outputs = torch.sigmoid(model(inputs)) # Passata in avanti, con sigmoid per ottenere probabilità
    loss = criterion(outputs, targets) # Calcolo della loss
    optimizer.zero_grad() # Azzera i gradienti
    loss.backward() # Calcolo dei gradienti
    optimizer.step() # Aggiornamento dei pesi
    if (epoch+1) % log_freq == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
```

Questo codice addestra un modello di regressione logistica.  `nn.Linear` crea un layer lineare, `nn.BCELoss` è la funzione di loss per la classificazione binaria, `torch.optim.Adam` è l'ottimizzatore. Il ciclo `for` itera per un numero specificato di epoche, aggiornando i pesi del modello ad ogni iterazione.  `torch.sigmoid` applica la funzione sigmoid all'output del modello per ottenere probabilità tra 0 e 1.


**2.3 Visualizzazione della linea di decisione:**

```python
line_y = np.linspace(0, 1., 1000)
line_x = [6] * len(line_y)
predicted = torch.sigmoid(model(torch.from_numpy(x_train))).detach().numpy()
plt.plot(x_train, predicted,'o',label='probabilities')
plt.plot(line_x, line_y, 'r')
plt.show()
```

Questo codice visualizza la linea di decisione (x = 6) e le probabilità predette dal modello per ogni punto dati.

![png](pytorch_basics_40_0.png) Questa immagine mostra il grafico generato, con i punti dati e la linea di decisione x=6.  Si nota come il modello approssima la linea di decisione.


In sintesi, il codice dimostra come utilizzare PyTorch per costruire e addestrare modelli di regressione lineare e logistica, visualizzando i risultati e confrontando le predizioni con i valori reali.  L'utilizzo di `torch.sigmoid` per la regressione logistica è fondamentale per ottenere probabilità di appartenenza alle classi.


Questo codice implementa una semplice rete neurale completamente connessa per la classificazione del dataset MNIST.  Analizziamolo passo passo.

**1. Predizioni iniziali:**

Il codice inizia mostrando alcune predizioni di un modello (non definito nel codice fornito):

```python
Predictions for x,y,hat_y in zip(x_train,y_train,predicted): 
    print(f'input = {x[0]:.3f} \t->\t Probability of 1 = {hat_y[0]:.3f} \t (y = {y[0]:.3f})') 
```

Questo snippet itera su tre iterabili (`x_train`, `y_train`, `predicted`), presumibilmente contenenti rispettivamente gli input, i valori target e le predizioni di un modello. Per ogni elemento, stampa l'input (`x[0]`), la probabilità predetta di appartenere alla classe 1 (`hat_y[0]`), e il valore target (`y[0]`).  Il `.3f` formatta i numeri a tre cifre decimali.  L'output mostra esempi di predizioni, indicando una bassa probabilità di appartenere alla classe 1 per tutti gli esempi mostrati.

**2. Caricamento del dataset MNIST:**

```python
import torchvision
import torchvision.transforms as transforms

batch_size = 64

train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

Questo codice carica il dataset MNIST utilizzando la libreria `torchvision`.  `transforms.ToTensor()` converte le immagini in tensori PyTorch.  `train_loader` e `test_loader` creano dei DataLoader per iterare efficientemente sui dati di training e test, con un batch size di 64.  `shuffle=True` mescola i dati di training ad ogni epoca.

**3. Visualizzazione di alcune immagini MNIST:**

```python
image, label = train_dataset[0]
print(image.shape) # torch.Size([1, 28, 28])

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    image, label = train_dataset[i]
    plt.imshow(image[0],cmap='gray', interpolation='none')
    plt.title("Class {}".format(label))
    plt.axis('off')
```

Questo codice accede al primo elemento del `train_dataset` e stampa la forma dell'immagine (1x28x28, un singolo canale, 28x28 pixel).  Poi, visualizza le prime 9 immagini del dataset MNIST usando `matplotlib.pyplot`.  `cmap='gray'` specifica una mappa di colori in scala di grigi.  `interpolation='none'` evita l'interpolazione, mantenendo i pixel netti.

![png](pytorch_basics_45_1.png)  Questa immagine mostra le prime nove immagini del dataset MNIST, come visualizzate dal codice precedente.


**4. Definizione della rete neurale:**

```python
class SimpleFullyConnectedNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(SimpleFullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

Questo definisce una semplice rete neurale completamente connessa con due livelli.  `__init__` inizializza due livelli lineari (`fc1` e `fc2`) e una funzione di attivazione ReLU.  `forward` definisce il flusso di dati attraverso la rete: l'input `x` passa attraverso `fc1`, ReLU, e `fc2`, restituendo l'output finale.

**5. Configurazione del modello e del dispositivo:**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28*28
hidden_size = 300
num_classes = 10
batch_size = 100
full_model = SimpleFullyConnectedNet(input_size, hidden_size, num_classes).to(device)
```

Questo codice configura il dispositivo di calcolo (GPU se disponibile, altrimenti CPU), le dimensioni dell'input (28x28 pixel), la dimensione dello strato nascosto (300 neuroni), il numero di classi (10 cifre), e crea un'istanza del modello `SimpleFullyConnectedNet`, spostandolo sul dispositivo selezionato.

**6. Training:**

```python
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)
num_epochs = 3
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(num_epochs + 1)]

total_step = len(train_loader)

def train(epoch,model,criterion,optimizer,reshape=True):
    # ... (codice per il training di un'epoca) ...
def test(model,criterion,reshape=True):
    # ... (codice per la valutazione del modello) ...

for epoch in range(1,num_epochs+1):
    train(epoch,full_model,criterion,optimizer)
    test(full_model,criterion)
```

Questo codice definisce la funzione di perdita (`CrossEntropyLoss`), l'ottimizzatore (`Adam`), il numero di epoche e le liste per salvare le perdite di training e test.  `train` e `test` sono funzioni che eseguono rispettivamente il training e la valutazione del modello su un'epoca o sull'intero set di test.  Il ciclo `for` esegue il training per 3 epoche.  Il codice delle funzioni `train` e `test` è omesso per brevità, ma esse gestiscono il passaggio in avanti, il calcolo della perdita, il backpropagation e l'aggiornamento dei pesi.  `reshape=True` indica che le immagini vengono rimodellate in vettori prima di essere passate al modello.

**7. Visualizzazione delle perdite:**

```python
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Cross Entropy Loss')
plt.show()
```

Questo codice crea un grafico che mostra le perdite di training e test durante il training.

![png](pytorch_basics_53_0.png) Questa immagine mostra il grafico delle perdite di training (blu) e test (rosso) nel corso delle epoche di training.  Si osserva una diminuzione delle perdite sia di training che di test, indicando un buon apprendimento del modello.


In sintesi, il codice implementa un semplice modello di classificazione di immagini MNIST usando PyTorch, mostrando le fasi di caricamento dati, costruzione del modello, training e valutazione.  L'output numerico mostra le perdite e l'accuratezza del modello ad ogni epoca, mentre il grafico visualizza l'andamento delle perdite nel tempo.


## Analisi del codice Python per la visualizzazione di predizioni di un modello

Il codice Python fornito mostra un esempio di come visualizzare le predizioni di un modello di machine learning, presumibilmente un modello di classificazione di immagini, addestrato con PyTorch.  Analizziamo i frammenti di codice cruciali.

**Sezione 1: Caricamento dei dati e iterazione**

Il codice inizia con l'iterazione su un `test_loader`, che si presume sia un DataLoader di PyTorch contenente dati di test.

```python
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
```

Questo snippet estrae la prima batch di dati dal `test_loader`.  `enumerate(test_loader)` crea un iteratore che restituisce sia l'indice della batch (`batch_idx`) sia i dati stessi (`example_data` e `example_targets`). `next(examples)` estrae la prima batch.  `example_data` contiene le immagini di test e `example_targets` le corrispondenti etichette (ground truth).  Il codice poi itera su un sottoinsieme di 9 immagini per visualizzarle.

**Sezione 2: Predizione e visualizzazione**

Il cuore del codice risiede nel loop `for` che itera sulle prime 9 immagini del dataset di training (`train_dataset`).

```python
for i in range(9):
    plt.subplot(3,3,i+1)
    image, label = train_dataset[i]
    with torch.no_grad():
        output = full_model(image.reshape(-1, 28*28).to(device))
        _, predicted = torch.max(output.data, 1)
    plt.tight_layout()
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.title("Pred: {}".format(predicted.item()))
    plt.xticks([])
    plt.yticks([])
```

Questo loop esegue le seguenti operazioni per ogni immagine:

1. **`plt.subplot(3,3,i+1)`:** Crea un subplot in una griglia 3x3 per visualizzare le 9 immagini.

2. **`image, label = train_dataset[i]`:**  Recupera l'immagine (`image`) e la sua etichetta (`label`) dal `train_dataset`.  Si noti che viene utilizzato il dataset di *training*, non quello di test come suggerito dalla prima parte del codice. Questo potrebbe essere un errore nel codice originale.

3. **`with torch.no_grad(): ...`:** Disabilita il calcolo del gradiente, necessario solo durante l'addestramento.  Questo aumenta l'efficienza durante l'inferenza.

4. **`output = full_model(image.reshape(-1, 28*28).to(device))`:**  Passa l'immagine al modello `full_model`.  `image.reshape(-1, 28*28)` rimodella l'immagine in un vettore di 784 elementi (28x28 pixel), necessario per l'input del modello. `.to(device)` sposta l'immagine sul dispositivo di calcolo (CPU o GPU).

5. **`_, predicted = torch.max(output.data, 1)`:** Ottiene la classe predetta dal modello. `output` contiene le probabilità per ogni classe; `torch.max(output.data, 1)` restituisce il valore massimo e l'indice corrispondente (la classe predetta).  L'indice viene assegnato a `predicted`.

6. **`plt.imshow(...)`:** Visualizza l'immagine in scala di grigi.

7. **`plt.title(...)`:** Imposta il titolo del subplot con la predizione del modello.

8. **`plt.xticks([])` e `plt.yticks([])`:** Rimuovono le etichette degli assi.

9. **`plt.tight_layout()`:**  Adatta automaticamente gli spazi tra i subplot.


**Sezione 3: Immagine di esempio**

![png](pytorch_basics_54_0.png)

Questa immagine mostra il risultato dell'esecuzione del codice: una griglia 3x3 di immagini con le rispettive predizioni del modello.  Ogni immagine è accompagnata dal titolo "Pred: [numero]", dove il numero rappresenta la classe predetta dal modello.


In sintesi, il codice fornisce un esempio semplice ma efficace di come caricare dati, effettuare predizioni con un modello PyTorch e visualizzare i risultati.  Tuttavia, come notato, c'è una discrepanza tra l'utilizzo del `test_loader` all'inizio e l'utilizzo del `train_dataset` nel loop principale, suggerendo un possibile errore nel codice originale.  Il codice dovrebbe essere corretto per utilizzare coerentemente il dataset di test per una valutazione accurata del modello.


