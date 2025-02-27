## Metodi per Migliorare l'Efficacia dei Modelli 

Nel campo della Computer Vision, sono stati sviluppati diversi metodi per migliorare l'efficacia dei modelli. Uno dei problemi comuni è il **vanishing gradient**, che si verifica quando i gradienti si riducono drasticamente durante la backpropagation, rendendo difficile l'apprendimento dei pesi della rete.

La loss è definita come $l(\theta)=nll(\theta)+\|\theta\|$
### Tecniche per Contrastare il Vanishing Gradient

**1. Funzioni di Attivazione:**

* L'utilizzo di funzioni di attivazione come **ReLU** (Rectified Linear Unit) può aiutare a risolvere il problema del vanishing gradient. ReLU è una funzione lineare per valori positivi e zero per valori negativi, evitando il problema della saturazione che si verifica con funzioni come la sigmoide.

**2. Regolarizzazione:**

* La regolarizzazione è una tecnica che "disciplina" la rete durante il fitting, evitando che i pesi assumano valori estremi.
* La **regolarizzazione L2** aggiunge un termine alla loss function che penalizza i pesi grandi. La formula della loss con regolarizzazione L2 è:

$$Loss = Loss_{originale} + λ * ||w||^2$$
dove:

* `Loss_originale` è la loss function originale.
* `λ` è un parametro che controlla l'intensità della penalizzazione.
* `w` è il vettore dei pesi della rete.

* La regolarizzazione L2 aiuta a prevenire l'overfitting, ovvero l'adattamento eccessivo della rete al training set, rendendola meno generalizzabile a nuovi dati.

**3. Clipping del Gradiente:**

* Il clipping del gradiente limita l'ampiezza dei gradienti, evitando che esplodano durante la backpropagation.
* Questa tecnica è utile in situazioni in cui i gradienti possono diventare molto grandi, ad esempio quando si utilizzano reti ricorrenti per l'elaborazione di dati sequenziali.

### Gestione dei Dati Sequenziali

* I video sono esempi di dati sequenziali, in cui l'ordine degli elementi è importante.
* Per gestire i dati sequenziali, è necessario utilizzare strutture di rete che siano in grado di tenere conto della sequenza temporale.
* Esempi di queste strutture includono le **reti ricorrenti (RNN)** e le **reti convoluzionali ricorrenti (RCNN)**.

# Tecniche per la Regolazione delle Reti Neurali

Uno dei problemi che si possono incontrare nelle architetture di rete che gestiscono sequenze, come le reti ricorrenti, è l'**esplosione** del gradiente. L'esplosione del gradiente è il contrario dell'**annullamento**: nell'annullamento i valori dei pesi della rete tendono a zero, mentre nell'esplosione i valori dei pesi aumentano a dismisura.

## Clipping

Il **clipping** è una strategia che nasce per mitigare il problema dell'esplosione, ma che si rivela utile anche come metodo di **regolarizzazione**. 

Il clipping si basa sull'idea di "tagliare" i valori dei pesi che superano una certa soglia. In pratica, se un valore di peso supera la soglia, viene "tagliato" e riportato al valore massimo o minimo consentito.

## Dropout

![[1-processed-20241020171249127.png|476]]
Un'altra strategia per gestire le reti neurali e ridurre l'overfitting è il **dropout**. Questa tecnica, proposta da Hinton, consiste nell'annullare in modo casuale i pesi di alcuni neuroni durante la fase di training.

**Esempio:**

Consideriamo una rete a tre livelli fully connected. In una rete fully connected, ogni neurone è connesso a tutti i neuroni del livello precedente e a tutti i neuroni del livello successivo.

Durante il training di una rete neurale, alcuni archi tra i nodi possono diventare molto grandi in valore assoluto, influenzando eccessivamente la propagazione lungo il percorso. Questo può portare all'overfitting, ovvero la rete impara a memoria i dati di training e non generalizza bene su dati nuovi.

**Effetto del Dropout:**

Il dropout forza la rete a non fare affidamento su singoli neuroni, ma a distribuire l'apprendimento su un gruppo più ampio di neuroni. Questo aiuta a prevenire l'overfitting e a migliorare la generalizzazione della rete.

**Come funziona il Dropout:**

Durante il training, il dropout seleziona in modo casuale alcuni neuroni e azzera il loro contributo. Ad esempio, in un livello della rete, alcuni neuroni potrebbero essere "disabilitati" e non partecipare al calcolo del forward pass
In particolare:
1. Durante il forward pass, alcuni neuroni vengono disabilitati in modo casuale.
2. Il calcolo della loss viene effettuato solo sui neuroni attivi.
3. Durante il backward pass, i gradienti vengono aggiornati solo per i neuroni attivi.

**Vantaggi del dropout:**

* **Contrasta l'overfitting:** impedisce alla rete di affidarsi eccessivamente a un singolo percorso.
* **Migliora la generalizzazione:** la rete impara a fare la classificazione utilizzando percorsi diversi, rendendola più robusta.
* **È semplice da implementare:** richiede solo una piccola modifica al codice di training.

**Analogia con le combinazioni di classificatori:**

Il dropout può essere visto come una combinazione di più reti neurali, ciascuna con una diversa configurazione di nodi attivi. Questo è simile al concetto di combinazione di classificatori, dove si addestrano modelli su diversi dataset o con configurazioni differenti per ottenere una migliore performance.

## Reti Neurali Convoluzionali

Questo concetto emerge analizzando una rete neurale fully connected. Prendiamo ad esempio una rete semplice a 4 livelli. L'input ha dimensione 4, e le equazioni che la caratterizzano sono rappresentate di seguito:
$$ \begin{aligned} a_i & = \sum_{j<i} w_{i,j} z_{j} \\ z_i & = f(a_i) \\ \mathbf{a}^{(h+1)} & = \mathbf{W}^{(h)} \mathbf{z}^{(h)} \\ z^{(h+1)} & = f(\mathbf{a}^{(h+1)}) \\ z^{(0)} & = \mathbf{x}\end{aligned} $$
Ogni nuovo livello in Computer Vision dipende dai livelli precedenti, combinati tramite una matrice di pesi (L e H) e una funzione di attivazione.
### Notazione Semplificata

La formula generale per il calcolo dell'attivazione di un nodo è $W^*X + b$, dove $W$ è la matrice dei pesi, $X$ è il vettore di input e $b$ è il vettore dei bias. Spesso si utilizza una notazione semplificata, accodando il vettore dei bias alla matrice dei pesi ($Wb$) e aggiungendo una componente 1 al vettore di input ($Xb$). Questa notazione semplifica il calcolo, riducendolo a un prodotto scalare.

### Analisi dei Nodi

Ogni nodo in una rete *fully-connected* ha un numero di pesi pari al numero di input più il peso del bias. Ad esempio, un nodo con 4 input avrà 5 pesi. In un livello con più nodi, il numero di pesi aumenta in modo esponenziale.

### Complessità Crescente

La complessità delle reti *fully-connected* aumenta drasticamente con l'aumentare della dimensione dell'input. Ad esempio, per un dataset di immagini come MNIST (28x28 pixel), il numero di pesi diventa enorme, rendendo la rete pesante dal punto di vista computazionale e della memoria.

**Esempio:**

In un esempio semplice con 4 input, il primo livello ha una matrice 5x4 (5 pesi per 4 input). Se l'input fosse un'immagine MNIST (784 pixel), il primo livello avrebbe una matrice 785x784, con un numero di pesi molto maggiore.

## Da reti fully-connected a filtri

![[1-processed-20241020171517750.png]]

Consideriamo un'immagine. In questo contesto, la rappresentiamo in due dimensioni, x e y. Ad esempio, prendiamo un'immagine a scala di grigi (un solo canale) di dimensioni 5x5. Questa è la nostra griglia di input.

Immaginiamo di voler costruire una rete neurale *fully-connected* per mappare questa immagine in un'immagine 4x4. In termini di rete *fully-connected*, stiamo creando due layer:

1. **Layer di input:** Composto da 25 elementi, uno per ogni cella dell'immagine di input.
2. **Layer di output:** Composto da 16 elementi, uno per ogni cella dell'immagine finale (4x4).

In sostanza, stiamo linearizzando l'immagine di input 5x5 e trasformandola in un vettore di 25 elementi. Questo vettore viene poi mappato in un vettore di output di 16 elementi, che rappresenta l'immagine 4x4.

**Rappresentazione alternativa:**

Possiamo anche vedere questa trasformazione come una serie di connessioni tra ogni elemento dell'immagine di input e ogni elemento dell'immagine di output. Ogni connessione ha un peso associato (rappresentato dalla matrice dei pesi) e un bias (rappresentato dal vettore dei bias).

Nel nostro esempio, avremmo 25 elementi di input e 16 elementi di output, quindi la matrice dei pesi sarebbe una matrice 25x16. Se consideriamo anche il vettore dei bias, avremmo una matrice 25x16 e un vettore di 16 elementi.

**Introduzione ai filtri:**

Ora, proviamo a pensare all'idea del filtro e a come possiamo costruire questa configurazione utilizzando un filtro.

Un filtro, ad esempio un filtro 3x3, viene applicato all'immagine di input in modo scorrevole. Per ogni posizione del filtro sull'immagine, calcoliamo il prodotto scalare tra il filtro e la porzione corrispondente dell'immagine. Il risultato di questo prodotto scalare viene poi associato alla cella corrispondente nell'immagine di output.

**Possiamo definire il filtro come la matrice dei pesi**: in questo caso, il filtro, applicato all'immagine di input, calcola la componente che poi dovrà essere attivata (chiamiamola A). Questa componente A è in realtà il risultato del prodotto scalare tra il filtro e la porzione corrispondente dell'immagine.
$$a=W \cdot lin(x)$$
- Dove $lin(X)$ è la linearizzazione di x
$$a=conv(x,w)$$
In altre parole, il filtro rappresenta la matrice dei pesi della rete neurale *fully-connected* che stiamo costruendo. L'applicazione del filtro è equivalente alla moltiplicazione della matrice dei pesi per il vettore di input linearizzato.

![[1-processed-20241020173241253.png|370]]
![[1-processed-20241020173250567.png|373]]
## Convoluzione e Feature Map

Invece di utilizzare una rete *fully-connected* per la linearizzazione dell'immagine, proponiamo di applicare la convoluzione.

**Confronto tra *Fully-Connected* e Convoluzione:**

* **Fully-Connected:** La rete *fully-connected* moltiplica un vettore di input (linearizzazione dell'immagine) per una matrice di pesi. Questa matrice ha dimensioni 26x16 nel nostro esempio.
* **Convoluzione:** La convoluzione opera su un sottoinsieme di pixel dell'immagine di input, utilizzando un filtro di dimensioni più piccole (ad esempio, 9 elementi).

**Vantaggi della Convoluzione:**

* **Efficienza:** La convoluzione utilizza un filtro più piccolo rispetto alla matrice di pesi *fully-connected*, riducendo il numero di parametri da apprendere.
* **Mappatura Informativa:** La convoluzione mappa un sottoinsieme di pixel in un altro spazio, estraendo informazioni significative dall'immagine.

**Analogia con i Filtri:**

La convoluzione è simile all'applicazione di filtri alle immagini. Ad esempio:

* **Filtro Gaussiano:** Effettua lo smoothing dell'immagine.
* **Filtro di Sobel:** Calcola il gradiente, evidenziando le variazioni di intensità.

**Feature Map:**

L'output della convoluzione è una *feature map*, che contiene nuove feature informative derivate dall'immagine originale. Ad esempio, se il filtro W è un gradiente, la *feature map* mostrerà i contorni dell'immagine.

**Apprendimento delle Feature:**

Invece di definire a priori i filtri, possiamo farli apprendere alla rete tramite la discesa del gradiente. Questo consente alla rete di scoprire le feature più informative per il compito specifico.

**Definizione della Convoluzione:**

L'operazione di convoluzione è una somma ponderata degli elementi dell'immagine di input, utilizzando un filtro di dimensioni fisse. La formula è la seguente:

$$a_{j,k}^{(h)} = \sum_{l=1}^c \sum_{m=1}^d w_{m,l}z_{j+l,k+m}^{(h-1)}$$
- Dove i pesi rappresentano il kernel di dimensione (c,d)

**Somma degli Indici e Backpropagation**

La derivabilità degli indici nei vettori è fondamentale perché permette di applicare la **backpropagation**, un algoritmo chiave nell'apprendimento automatico. La backpropagation consente di costruire un **grafo di computazione** che rappresenta le operazioni svolte e di calcolare i gradienti sui componenti del grafo.

**Apprendimento dei Filtri**

La derivabilità della somma degli indici apre la possibilità di **apprendere direttamente i filtri** durante il processo di addestramento. Questo è un passo importante perché permette di **ottimizzare la rappresentazione dell'immagine** in base al compito specifico.

## Condivisione dei Pesi e Kernel Multipli

![[1-processed-20241020173436878.png|538]]
Un'ulteriore ottimizzazione si ottiene condividendo i pesi su più kernel. Questo consente di **ridurre il numero di parametri** necessari per l'addestramento. Inoltre, non c'è bisogno di limitarsi a un solo kernel. È possibile utilizzare **più kernel** con diverse caratteristiche, aumentando la capacità di estrarre informazioni dall'immagine.

**Vantaggi dei Kernel Multipli**

L'utilizzo di più kernel offre diversi vantaggi:

* **Rappresentazione più informativa:** Ogni kernel può essere specializzato nell'estrazione di una specifica feature, aumentando la ricchezza della rappresentazione dell'immagine.
* **Addestramento automatico dei filtri:** L'algoritmo di apprendimento automatico determina automaticamente i filtri più efficaci per il compito specifico.
* **Efficienza:** La condivisione dei pesi e l'utilizzo di più kernel riducono il numero di parametri da addestrare, rendendo il processo più efficiente.

## Relazione tra Input, Output e Convoluzione

È la formula per calcolare la dimensione dell'output di una convoluzione su un'immagine, tenendo conto di vari parametri:
### Parametri:

- **Input \(I\)**: Dimensione dell'input (larghezza o altezza).
- **Padding \(P\)**: Quantità di padding applicata sui bordi dell'immagine.
- **Kernel size \(K\)**: Dimensione del kernel (finestra di convoluzione).
- **Stride \(S\)**: Passo con cui si muove il kernel sull'immagine.
- **Dilation \(D\)**: Fattore di dilatazione, che espande lo spazio tra gli elementi del kernel.

### Formula per la dimensione dell'output:

$$\left\lfloor \frac{I - K - (K - 1)(D - 1) + 2P}{S} \right\rfloor + 1$$

### Spiegazione:

1. **\(I - K\)**: Rimuove la parte dell'immagine coperta dalla dimensione del kernel.
2. **\((K - 1)(D - 1)\)**: Aggiusta l'area coperta dal kernel in base alla dilatazione.
3. **\(2P\)**: Aggiunge lo spazio introdotto dal padding, che aumenta la dimensione effettiva dell'input.
4. **Divisione per \(S\)**: Considera il passo con cui si sposta il kernel sull'immagine, riducendo la dimensione dell'output.
5. **Somma di 1**: Aggiunge un'unità per includere il primo elemento convoluto.

Il risultato è la dimensione dell'output dopo l'operazione di convoluzione, tenendo conto del padding, della dilatazione e dello stride.

## Padding

Il padding consiste nell'aggiungere una cornice di zeri attorno all'immagine di input. Questo serve a gestire i bordi dell'immagine, garantendo che il filtro possa essere applicato anche ai pixel che si trovano ai margini.

**Tipi di Padding:**

* **No-padding:** Non viene aggiunto alcun bordo.
* **Dark padding:** Viene aggiunto un bordo di zeri di una certa dimensione.
* **Half padding:** La dimensione del bordo è pari alla metà della dimensione del filtro. Questo tipo di padding è particolarmente utile per centrare il filtro sui pixel di bordo.
* **Full padding:** La dimensione del bordo è pari alla dimensione del filtro.

**Esempio:**

Consideriamo un'immagine di input 4x4 e un filtro 3x3. Applicando il filtro senza padding (no-padding), otterremo una feature map 2x2. Questo perché ci sono 4 sottoinsiemi di dimensione 3x3 nell'immagine di input.

Applicando invece l'half padding, la dimensione della feature map sarà pari alla dimensione del filtro (3x3). Questo perché il padding aggiunge un bordo di 1 pixel attorno all'immagine, consentendo al filtro di essere applicato anche ai pixel di bordo.

### Stride

Lo stride rappresenta il passo con cui il filtro viene spostato sull'immagine di input.

**Esempio:**

Uno stride di 1 significa che il filtro viene spostato di un pixel alla volta. Uno stride di 2 significa che il filtro viene spostato di due pixel alla volta.

**Impatto dello Stride:**

* **Stride 1:** La feature map avrà la stessa dimensione dell'immagine di input (escluso il padding).
* **Stride > 1:** La feature map avrà una dimensione inferiore rispetto all'immagine di input.

### Dilatazione

Oltre al padding e allo stride, è possibile utilizzare anche la **dilatazione** per modificare il comportamento del filtro. La dilatazione consiste nell'inserire degli zeri tra i coefficienti del filtro. Questo ha l'effetto di aumentare il campo visivo del filtro, consentendogli di catturare informazioni da un'area più ampia dell'immagine.

## Dimensione dell'Output nella Convoluzione

**Domanda:** Come possiamo definire una regola per l'evoluzione della dimensione dell'output di una convoluzione?

**Risposta:** La dimensione dell'output dipende da diversi parametri:

* **Dimensione dell'immagine di input:** La dimensione dell'immagine originale.
* **Stride:** Il passo con cui il kernel si sposta sull'immagine di input.
* **Padding:** La quantità di valori aggiuntivi inseriti attorno ai bordi dell'immagine di input.
* **Dimensione del kernel:** La dimensione del filtro utilizzato per la convoluzione.

**Formula:**

La formula per calcolare la dimensione dell'output è la seguente:

$$Dimensione\_Output = (Dimensione\_Input + 2 * Padding - Dimensione\_Kernel) / Stride + 1$$

**Esempio:**

Consideriamo un'immagine di input di dimensione 5x5, un kernel di dimensione 3x3, uno stride di 1 e un padding di 0.

* Dimensione_Input = 5
* Dimensione_Kernel = 3
* Stride = 1
* Padding = 0

Applicando la formula:

```
Dimensione_Output = (5 + 2 * 0 - 3) / 1 + 1 = 3
```

Quindi, la dimensione dell'output sarà 3x3.

**Padding:**

Il padding è un'operazione che aggiunge valori aggiuntivi attorno ai bordi dell'immagine di input. Questo può essere utile per evitare la perdita di informazioni ai bordi dell'immagine.

* **Padding a metà:** In questo caso, il padding è calcolato come la metà della dimensione del kernel. Ad esempio, per un kernel di dimensione 3x3, il padding a metà sarebbe 1.

**Esempio con Padding:**

Consideriamo lo stesso esempio precedente, ma con un padding a metà (1).

* Dimensione_Input = 5
* Dimensione_Kernel = 3
* Stride = 1
* Padding = 1

Applicando la formula:

```
Dimensione_Output = (5 + 2 * 1 - 3) / 1 + 1 = 5
```

Quindi, la dimensione dell'output sarà 5x5, la stessa dimensione dell'immagine di input.

**Interpretazione Geometrica:**

La formula per la dimensione dell'output può essere interpretata geometricamente. La convoluzione può essere vista come una serie di operazioni di scorrimento del kernel sull'immagine di input. La dimensione dell'output rappresenta il numero di posizioni in cui il kernel può essere posizionato sull'immagine di input.

**Struttura dell'Output:**

L'output di una convoluzione è una mappa di attivazione, che rappresenta la risposta del kernel all'immagine di input. Se utilizziamo più kernel, otterremo una struttura di mappe di attivazione, ciascuna corrispondente a un kernel diverso.

## Reti Neurali a Livelli

Le immagini, in Computer Vision, sono rappresentate come tensori. Un'immagine in scala di grigi è un tensore con un canale, altezza e larghezza. Un'immagine a colori ha tre canali (rosso, verde, blu), altezza e larghezza.

Le reti neurali a livelli, come la rete T4 nell'esempio, sono caratterizzate da una struttura più complessa rispetto alle reti semplici viste in precedenza. In queste reti, ogni livello è rappresentato da un tensore tridimensionale, che funge sia da output del livello precedente che da input del livello successivo.

**Differenze rispetto alle reti semplici:**

* **Livelli multipli:** Le reti a livelli hanno più livelli, ognuno con il proprio tensore di pesi.
* **Tensori come input e output:** Ogni livello elabora un tensore, non solo un singolo vettore.

### Evoluzione delle Equazioni

L'equazione fondamentale che descrive l'attivazione di un livello, ovvero il prodotto tra la matrice dei pesi del livello precedente e la feature map, deve essere ripensata in termini di tensori.

**Considerazioni:**

* **Tensori di pesi:** Ogni livello ha il proprio tensore di pesi, non una singola matrice.
* **Trasformazione:** L'applicazione del tensore di pesi trasforma l'immagine di input, modificando il numero di canali e la dimensione.

## Dimensione di un tensore

La dimensione di un tensore di output in una rete neurale convoluzionale è definita da due parametri:

* **Altezza e Ampiezza:** Calcolate attraverso le equazioni per i filtri.
* **Numero di canali:** Definito dal numero di filtri applicati.

**Interpretazione dei canali:**

* **Canali di input:** Rappresentano i tensori tridimensionali delle immagini di input. Ad esempio, un'immagine RGB ha 3 canali.
* **Canali di output:** Rappresentano i tensori tridimensionali delle immagini di output. Ogni canale corrisponde a un filtro applicato all'immagine di input.

**Esempio:**

Se abbiamo un'immagine di input in scala di grigi e 6 filtri 3x3, il tensore dei kernel sarà 6x3x3. Se l'immagine di input fosse RGB, il tensore dei kernel sarebbe 3x6x3x3.

**Generalizzazione della formula:**

Per gestire un numero qualsiasi di canali di input e output, la formula viene generalizzata come segue:

```
attivazione(i, j, h) = Σ(c=1 to C) [Σ(k=1 to K) Σ(l=1 to K) kernel(c, h, k, l) * input(i + k - 1, j + l - 1, c)]
```

Dove:

* `C`: numero di canali di input.
* `h`: numero di canali di output.
* `kernel(c, h, k, l)`: peso del kernel per il canale di input `c`, il canale di output `h` e la posizione `(k, l)` nel kernel.

## I Tensori

I tensori sono quadrimensionali perché devono gestire diversi aspetti dell'input e dell'output:

* **Canali di input:** Rappresentano le diverse informazioni che vengono fornite al modello, come ad esempio i canali RGB di un'immagine.
* **Canali di output:** Rappresentano le diverse informazioni che il modello produce, come ad esempio le probabilità di appartenenza a diverse classi in un problema di classificazione.
* **Dimensione del tensore:** Rappresenta la dimensione spaziale dell'input o dell'output, come ad esempio la larghezza e l'altezza di un'immagine.

**Esempio:**

Un tensore che rappresenta un'immagine RGB con dimensioni 100x100 avrà le seguenti dimensioni:

* **Canali di input:** 3 (RGB)
* **Canali di output:** 1 (l'immagine stessa)
* **Dimensione del tensore:** 100x100

Quindi, la dimensione totale del tensore sarà 3x1x100x100.

