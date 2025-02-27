
Prendiamo come esempio il caso in cui dobbiamo identificare un animale presente in un'immagine. Questo tipo di applicazione può essere risolta tramite un'architettura di rete neurale. Abbiamo già visto esempi di architetture come LeNet, piuttosto semplice, e GoogLeNet (o Inception), più sofisticata e probabilmente più interessante per noi. 

La domanda è: come adatteremmo l'architettura per risolvere questo specifico problema? Qual è la differenza tra l'identificare un solo animale e identificarne diversi nella stessa immagine? 

In entrambi i casi, l'output desiderato è un vettore che associa valori a diverse categorie. Ad esempio, se avessimo solo due categorie, "cane" e "gatto", il vettore di output potrebbe essere [1, 0] per un cane e [0, 1] per un gatto. Ma cosa succede se abbiamo 10 possibili animali? Non possiamo avere 1024 risposte diverse per ogni immagine. Come possiamo quindi risolvere questo problema?

### Soluzione con Rete Fully Connected e Softmax

La soluzione consiste nell'utilizzare una rete neurale con uno strato fully connected e una funzione di attivazione Softmax. Vediamo i passaggi:

1. **Estrazione delle feature:** L'immagine di input viene passata al blocco di estrazione delle feature.
2. **Linearizzazione:** Le feature estratte vengono linearizzate.
3. **Strato Fully Connected:** Viene applicato uno strato fully connected alle feature linearizzate.
4. **Output:** Otteniamo un vettore di output con la stessa dimensione del numero di categorie (ad esempio, 5 categorie: cane, gatto, gorilla, orso, toro). Ogni elemento del vettore di output rappresenta la probabilità che l'immagine appartenga a quella specifica categoria. La somma di tutte le probabilità deve essere uguale a 1.

### Funzione Softmax e Stabilità Numerica

Per ottenere le probabilità, utilizziamo la funzione Softmax. La formula della Softmax è la seguente:

```
P(classe i | x) = exp(f(x)_i) / Σ(j=1 a N) exp(f(x)_j)
```

Dove:

* P(classe i | x) è la probabilità che l'input x appartenga alla classe i.
* f(x)_i è l'output dello strato fully connected per la classe i.
* N è il numero di classi.

Tuttavia, la formula della Softmax può portare a problemi di stabilità numerica a causa degli esponenziali. Per evitarli, si utilizza il logaritmo della Softmax (log-softmax). La formula della log-softmax è:

```
log(P(classe i | x)) = f(x)_i - log(Σ(j=1 a N) exp(f(x)_j))
```

Il secondo termine della formula è il logaritmo della somma degli esponenziali (log-sum-exp). Per calcolare il log-sum-exp in modo stabile, si può utilizzare il seguente trucco matematico:

```
log(Σ(i=1 a N) exp(y_i)) = m + log(Σ(i=1 a N) exp(y_i - m))
```

Dove m è il valore massimo di y. 

### Implementazione in PyTorch

In PyTorch, la funzione log-softmax è già implementata e viene utilizzata per calcolare la perdita durante l'addestramento.

### Considerazioni sul Batch Size

È importante ricordare che le reti neurali elaborano i dati in batch. Quindi, l'input della rete non sarà una singola immagine, ma un batch di immagini. Di conseguenza, la dimensione dell'input dovrà essere modificata per includere la dimensione del batch. Ad esempio, se il batch size è 64, la dimensione dell'input sarà [64, 1, 28, 28] per immagini in scala di grigio di dimensione 28x28.

## Analisi di un Modello di Classificazione

Consideriamo un modello di classificazione che utilizza un layer convoluzionale seguito da un layer fully connected. Il layer convoluzionale estrae feature dall'immagine, mentre il layer fully connected classifica le feature estratte.

**Esempio:**

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Appiattimento
        x = self.fc(x)
        return x

model = Model()
x = torch.randn(1, 1, 28, 28)  # Input di dimensione [batch_size, canali, altezza, larghezza]
y = model(x)
```

In questa linea di codice, stiamo appiattendo tutto tranne la prima dimensione. Il risultato finale sarà una matrice con la dimensione del batch come prima componente e 256 come seconda dimensione, ottenuta appiattendo il blocco.

**Spiegazione:**

* **`x`**: rappresenta le feature dell'immagine.
* **`x.shape`**: restituisce la forma di `x`.
* **`x.view`**: modifica la forma di `x`.
* **`model.fc(x)`**: applica il layer fully connected a `x`.

**Esempio:**

Consideriamo un blocco di un edificio nell'immagine. Il layer convoluzionale estrae un vettore di dimensione 1x16x4x4 da questo blocco. Moltiplicando questi quattro elementi, otteniamo un blocco di 256 elementi. Appiattendo questo blocco, otteniamo una matrice 1x256. Ricordiamo che "1" rappresenta la dimensione del batch.

**Processo di Classificazione:**

1. **Input:** L'immagine di dimensione 1x1x28x28 viene passata al modello.
2. **Estrazione Feature:** Il layer convoluzionale estrae feature, ottenendo un vettore 1x16x4x4.
3. **Appiattimento:** Il blocco di feature viene appiattito, ottenendo una matrice di dimensione "dimensione del batch" x 256.
4. **Classificazione:** Il layer fully connected classifica le feature appiattite, fornendo la risposta sulla quale calcolare la logica.

**Softmax:**

Potremmo anche calcolare direttamente la softmax sul risultato del layer fully connected, ottenendo le probabilità per ogni classe.

## Classificazione Multipla

### Classificazione Mutuamente Esclusiva

Il processo di classificazione in Computer Vision prevede l'utilizzo di una rete neurale per classificare un'immagine in una delle possibili classi. Questo processo può essere suddiviso in due fasi:

1. **Estrazione delle feature:** La rete neurale elabora l'immagine di input ("x") attraverso una serie di layer convoluzionali e pooling, estraendo le feature significative.
2. **Classificazione:** Le feature estratte vengono appiattite e passate a layer fully connected, che producono un vettore di probabilità per ogni classe. La funzione di attivazione softmax viene applicata a questo vettore, normalizzando le probabilità in modo che la loro somma sia pari a 1.

In questo scenario, la rete prevede una sola classe per ogni immagine, ovvero la classe con la probabilità più alta. Questo tipo di classificazione è definita **mutuamente esclusiva**, poiché un'immagine può appartenere solo a una classe alla volta.

### Classificazione Multipla

Nel caso della classificazione multipla, un'immagine può appartenere a più classi contemporaneamente. Ad esempio, un'immagine potrebbe contenere sia un cane che un gatto. Per gestire questo tipo di classificazione, è necessario modificare la rete neurale in due modi:

1. **Funzione di attivazione:** La funzione di attivazione softmax viene sostituita con la funzione sigmoid. La sigmoid opera elemento per elemento, restituendo un vettore con la stessa dimensione dell'input, dove ogni elemento rappresenta la probabilità associata a una specifica classe.
2. **Funzione di loss:** La cross-entropia su tutte le classi viene sostituita con una somma di cross-entropie binarie. Le etichette reali sono rappresentate da un vettore binario con "1" in corrispondenza degli oggetti presenti nell'immagine.

### Architettura della Rete

È importante notare che l'architettura della rete neurale rimane invariata sia per la classificazione mutuamente esclusiva che per la classificazione multipla. La differenza risiede nella funzione di attivazione e nella funzione di loss. La prima parte della rete, che estrae le feature, rimane invariata perché il suo obiettivo è estrarre informazioni significative dall'immagine, indipendentemente dal tipo di classificazione. La seconda parte della rete, che si occupa della classificazione, utilizza le informazioni estratte dalla prima parte per generare le predizioni.

## Architettura di una Rete Neurale

L'architettura di una rete neurale può essere suddivisa in due parti principali:

1. **Estrazione delle Caratteristiche:** Questa parte "vede" l'immagine e ne estrae le caratteristiche salienti.
2. **Interpretazione delle Caratteristiche:** Questa parte "interpreta" le caratteristiche estratte per classificare l'immagine.

Questa architettura è in grado di risolvere sia problemi di classificazione esclusiva che multipla, adattando semplicemente la funzione di attivazione e la funzione di loss.

### Adattamento dell'Output per Classificazione Multipla

Nel caso della classificazione multipla, l'output della rete neurale deve essere adattato per gestire più di una classe. Ad esempio, se l'immagine può appartenere a una delle 1024 classi possibili, l'output dovrebbe essere un vettore di 1024 elementi, dove ogni elemento rappresenta la probabilità che l'immagine appartenga a una specifica classe.

**Esempio:**

Se l'ultimo layer della rete è 84x10, per adattarlo alla classificazione multipla con 1024 classi, dovremmo modificarlo in 84x2^10. Questo significa che il layer dovrebbe avere 2^10 (1024) neuroni in output, uno per ogni classe.

**Soluzione alternativa:**

Una soluzione alternativa, più efficiente, è quella di mantenere l'ultimo layer con 84x10 neuroni e utilizzare una funzione di attivazione che permetta di ottenere un output multi-classe. Ad esempio, si potrebbe utilizzare una funzione di attivazione sigmoidale per ogni neurone in output, ottenendo così una probabilità di appartenenza a ciascuna delle 10 classi.

## Object Detection

### La sfida della complessità

Un approccio diretto all'Object Detection potrebbe prevedere la scansione dell'immagine con finestre di diverse dimensioni e posizioni, cercando di identificare la presenza di oggetti al loro interno. Tuttavia, questo metodo presenta un'elevata **complessità computazionale**, dovendo analizzare un numero enorme di possibili finestre. Come abbiamo visto con le CNN, anche una rete relativamente semplice richiede un certo tempo di elaborazione per ogni classificazione. Analizzare migliaia di sottoimmagini estratte da un'immagine diventerebbe quindi un processo estremamente lento.

### Esigenze di velocità e accuratezza

La velocità di elaborazione è un requisito fondamentale in molte applicazioni di Object Detection, come ad esempio la **guida autonoma**. In questi contesti, il sistema deve essere in grado di analizzare le immagini in tempo reale, con tempi di risposta inferiori al secondo. Un ritardo nell'individuazione di un pedone che attraversa la strada potrebbe avere conseguenze disastrose.

### Piramidi e Regioni di Interesse

Per mitigare il problema della complessità, possiamo utilizzare le **piramidi**, che permettono di analizzare l'immagine a diverse scale di risoluzione. Tuttavia, questa soluzione offre solo un miglioramento parziale. Un approccio più efficace è quello delle **Region Proposals** o **Regioni di Interesse**. Questo metodo prevede l'individuazione preliminare di aree dell'immagine che potrebbero contenere oggetti di interesse.

### Come funzionano le Region Proposals?

Le Region Proposals sono aree predefinite all'interno dell'immagine, di varie dimensioni e posizioni. L'idea è quella di semplificare il processo di Object Detection in due fasi:

1. **Individuazione delle Region Proposals:** si selezionano le aree dell'immagine che potrebbero contenere oggetti.
2. **Predizione e Classificazione:** si analizzano solo le Region Proposals selezionate per identificare la presenza di oggetti e determinarne la posizione precisa tramite il **Bounding Box**.

Questo approccio offre diversi vantaggi:

* **Riduzione della complessità:** non è necessario analizzare ogni possibile finestra dell'immagine.
* **Semplificazione della regressione:** la posizione del Bounding Box viene calcolata rispetto alla Region Proposal, semplificando il problema.
* **Gestione della variabilità di scala:** la dimensione del Bounding Box viene definita in relazione alla Region Proposal, riducendo la variabilità.

### Valutazione dell'Object Detection

Per valutare l'efficacia di un sistema di Object Detection si utilizzano diverse metriche:

* **Precisione:** quanti oggetti individuati sono effettivamente presenti nell'immagine.
* **Recall:** quanti oggetti effettivamente presenti nell'immagine sono stati individuati.
* **Velocità:** quanti fotogrammi al secondo (FPS) il sistema è in grado di analizzare. 

## Valutazione delle prestazioni e architetture

### Il compromesso tra velocità e accuratezza

Oggi esistono sistemi di Object Detection molto accurati ma lenti, e sistemi molto veloci ma meno accurati. La scelta del sistema più adatto dipende dall'applicazione specifica.

### Intersection over Union (IoU)

Per valutare la sovrapposizione tra due rettangoli, ad esempio il Bounding Box reale e quello predetto, si utilizza l'**Intersection over Union (IoU)**. Questa metrica misura la sovrapposizione tra le due aree rispetto all'area totale.

### Valutazione delle prestazioni

Il punto fondamentale è questo: se un elemento rappresenta ciò che vogliamo rilevare e l'altro ciò che effettivamente rileviamo, la loro sovrapposizione ci permette di valutare la bontà del nostro sistema. Ad esempio, in questa immagine, la rilevazione è buona o no? Direi di sì, è buona. Potrebbe essere migliore, l'ideale sarebbe una sovrapposizione totale tra il quadrato rosso e l'area blu. Tuttavia, anche così è abbastanza accettabile. Siete d'accordo?

Ma come si quantifica l'accettabilità? Si misura attraverso l'**Intersection over Union (IoU)**, ovvero il rapporto tra l'area dell'intersezione e l'area dell'unione tra il riquadro rosso (la rilevazione del nostro sistema) e il bounding box (l'area che definisce l'oggetto reale).

Il sistema, in pratica, restituisce una lista di coordinate che identificano i bounding box degli oggetti rilevati. Il bounding box ci dice "in quest'area ho individuato un oggetto". Tuttavia, il concetto di "in quest'area" è flessibile. Un bounding box leggermente disallineato potrebbe essere accettabile, ma se fosse completamente fuori posto, sarebbe un errore.

Possiamo quindi identificare quattro situazioni:

1. **True Positive (TP):** Il bounding box rosso si sovrappone in modo significativo al bounding box reale, indicando una corretta rilevazione.
2. **False Positive (FP):** Il bounding box rosso non si sovrappone in modo significativo al bounding box reale, indicando una rilevazione errata. Questo può accadere se il bounding box è troppo piccolo o posizionato in modo errato.
3. **True Negative (TN):** Non viene rilevato alcun oggetto in un'area dove effettivamente non c'è alcun oggetto.
4. **False Negative (FN):** Non viene rilevato alcun oggetto in un'area dove effettivamente è presente un oggetto.

Le situazioni più interessanti per noi sono i **falsi positivi** e i **falsi negativi**. La **precisione** e la **sensibilità** (recall) vengono misurate in base a queste quattro situazioni.

In pratica, contiamo quanti oggetti sono presenti nell'immagine e quanti ne vengono rilevati correttamente, ovvero con un'IoU superiore ad una soglia predefinita (ad esempio 50%, 70% o 90%).

È importante sottolineare che queste misure servono a valutare le prestazioni del sistema, non ad addestrarlo. Per l'addestramento, utilizzeremo altre metriche, come l'errore assoluto o l'entropia incrociata, che sono derivabili e quindi utilizzabili per la backpropagation.

### Altre metriche di valutazione

Un'altra misura utile è l'**Average Precision (AP)**, che rappresenta la precisione media calcolata su diverse soglie di IoU. L'AP ci fornisce un'indicazione della robustezza del sistema.

Se abbiamo a che fare con più classi di oggetti, possiamo calcolare la **mean Average Precision (mAP)**, ovvero la media dell'AP su tutte le classi. La mAP è la metrica più comunemente utilizzata per valutare le prestazioni dei sistemi di object detection.

## Evoluzione delle Architetture di Object Detection

L'object detection, ovvero la capacità di identificare e localizzare oggetti all'interno di un'immagine, ha visto un'evoluzione significativa negli ultimi anni. Prima del 2012, le prestazioni dei sistemi di object detection erano limitate. L'avvento delle reti neurali convoluzionali (CNN) e di architetture come VGG e AlexNet ha portato a un notevole miglioramento delle prestazioni. Oggi, con l'utilizzo dei transformer, si raggiungono precisioni superiori al 95%.

### Tipi di Architetture

Esistono due tipi principali di architetture di object detection:

**1. Multi-shot:**

* Queste architetture utilizzano due fasi distinte:
 * **Fase 1: Generazione di Region Proposal:** In questa fase, l'algoritmo identifica aree dell'immagine che potrebbero contenere oggetti, chiamate "region proposal".
 * **Fase 2: Classificazione e Localizzazione:** In questa fase, le region proposal vengono classificate e localizzate con precisione.
* Esempi di architetture multi-shot includono:
 * **R-CNN (Regions with CNN features):** Un'architettura pionieristica che ha introdotto l'utilizzo delle CNN per l'object detection.
 * **Fast R-CNN:** Un'evoluzione di R-CNN che ha migliorato la velocità di elaborazione.
 * **Faster R-CNN:** Un'ulteriore evoluzione che ha integrato la generazione di region proposal all'interno della rete neurale, rendendola ancora più efficiente.

**2. Single-shot:**

* Queste architetture combinano le due fasi in un unico passaggio, rendendole più veloci ma leggermente meno accurate rispetto alle architetture multi-shot.
* Esempi di architetture single-shot includono:
 * **SSD (Single Shot MultiBox Detector):** Un'architettura che utilizza una rete neurale per generare direttamente le bounding box e classificare gli oggetti.
 * **YOLO (You Only Look Once):** Un'architettura che elabora l'intera immagine in un'unica volta, rendendola molto veloce.

### Scelta dell'Architettura

La scelta dell'architettura dipende dalle esigenze specifiche dell'applicazione. Le architetture multi-shot sono più accurate ma anche più lente, mentre le architetture single-shot sono più veloci e adatte ad applicazioni real-time.
