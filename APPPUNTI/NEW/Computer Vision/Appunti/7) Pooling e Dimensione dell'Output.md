### Architettura di una Rete Neurale

Un esempio di rete neurale può avere un'architettura composta da due layer non lineari e tre layer lineari. Le funzioni non lineari sono generalmente note, mentre i layer lineari sono quelli che ci interessano in questa lezione.

### Convoluzioni

Le convoluzioni sono un'operazione fondamentale nella Computer Vision. Un filtro viene applicato all'immagine per estrarre caratteristiche specifiche.
![[cv-20241022115515306.png|477]]

#### Pooling

Il pooling è un'operazione che riduce la dimensione dell'immagine, mantenendo le informazioni più importanti. Un tipo comune di pooling è il **max pooling**, dove viene selezionato il valore massimo all'interno di una finestra di dimensioni fisse.

#### Stride

Lo **stride** determina il passo con cui il filtro si sposta sull'immagine. Uno stride di 2 significa che il filtro si sposta di due pixel alla volta.

## Dimensione dell'Output

La dimensione dell'output del pooling dipende dalla dimensione dell'input, dalla dimensione della finestra di pooling e dallo stride. In generale, l'output è la dimensione dell'input dimezzata.

#### Posizione del Filtro

La posizione del filtro è determinata dallo stride. Un filtro con stride 2 si sposterà di due pixel alla volta, quindi la sua posizione sarà sempre pari a un multiplo di 2.

#### Formula per la Dimensione dell'Output

La formula per calcolare la dimensione dell'output del pooling è:

$$O=\left[ \frac{{I-K+2P}}{S} \right]+1$$

Dove:

* **Input Size:** Dimensione dell'input.
* **Filter Size:** Dimensione del filtro.
* **Padding:** Quantità di padding applicata all'immagine.
* **Stride:** Passo del filtro.
  
La formula $O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$ è una versione semplificata della formula che tiene conto di meno parametri, e rappresenta il calcolo della dimensione dell'output di una convoluzione quando **non c'è dilatazione** del kernel.

**Formula generale (con dilatazione)**:
   $$
   O = \left\lfloor \frac{I - K - (K-1)(D-1) + 2P}{S} \right\rfloor + 1
   $$
- Aggiunge il termine $-(K-1)(D-1)$, che tiene conto dell'effetto della dilatazione. Quando $D = 1$, questo termine diventa zero, riportandoci alla formula semplificata.
- La dilatazione fa sì che gli elementi del kernel si "espandano", separandosi e quindi coprendo un'area maggiore rispetto al caso standard, e questo deve essere tenuto in considerazione nella formula.

## Relazione tra i Parametri nella Convoluzione 2D

I parametri chiave sono:

* **Dimensione del kernel (k):** Determina la dimensione dell'area su cui il kernel opera.
* **Dimensione dell'output (o):** Determina la dimensione della feature map in uscita.
* **Dimensione dell'input (i):** Determina la dimensione della feature map in ingresso.
* **Stride (s):** Determina il passo con cui il kernel si sposta sulla feature map in ingresso.
* **Dilation (d):** Determina la spaziatura tra i pesi del kernel.
* **Padding (p):** Determina la quantità di padding aggiunta ai bordi della feature map in ingresso.

Questa formula deve essere applicata sia per l'altezza che per la larghezza della feature map in uscita.

**Nota:** La parte intera inferiore e superiore (rappresentata dalle parentesi quadre) indica che il risultato della formula viene arrotondato all'intero più vicino.

**Padding:** Il padding viene aggiunto sia a destra che a sinistra, sia sopra che sotto la feature map in ingresso.

**Dilation:** La dilation è il prodotto di k-1. Una dilation pari a 1 indica che non c'è dilation, e lo spostamento del kernel è di 1 pixel.

**Esempio:** Se ignoriamo tutti gli altri parametri, la formula `i-k` ci dice che se l'input è un'immagine di dimensione `i` e il kernel ha dimensione `k`, la dimensione della feature map in uscita sarà `i-k`.

**Pooling:** Il pooling è un'operazione che riduce la dimensione della feature map in ingresso. A differenza della convoluzione, il pooling non ha molti parametri. L'unico parametro principale è la dimensione del pooling. Ad esempio, un pooling 2x2 significa che 4 pixel in ingresso vengono ridotti a 1 pixel in uscita. Il pooling può avere una dilation in alcune implementazioni, ma nella sua versione base non ne ha.

**Relazioni Inverse:** I layer di pooling, tramite lo stride, vengono tipicamente utilizzati per dimezzare la dimensione della feature map. Questo è dovuto alle relazioni inverse tra i parametri, che permettono di controllare la dimensione della feature map in uscita.

**Conclusione:** La formula generale fornisce un modo per legare le dimensioni delle feature map in uscita a partire da un dato input. Comprendere queste relazioni è fondamentale per progettare e ottimizzare le architetture di reti neurali convoluzionali.

#### Esempio

Se l'input è un'immagine di dimensione 10x10, il filtro è di dimensione 3x3 e lo stride è 2, la dimensione dell'output sarà:

$Output Size = \frac{10 - 3 + 2 * 0}{2} + 1 = 4$

Quindi l'output sarà un'immagine di dimensione 4x4.

### GPU

Le GPU (Graphics Processing Units) sono unità di elaborazione specializzate per l'elaborazione grafica. Sono particolarmente adatte per le operazioni di Computer Vision, come le convoluzioni, grazie alla loro capacità di eseguire calcoli paralleli su grandi quantità di dati.


### Calcolo della Dimensione dell'Output

Per calcolare la dimensione dell'output di un layer convoluzionale, è necessario considerare diversi fattori:

* **Dimensione dell'input:** La dimensione dell'immagine di input, in questo caso 28x28.
* **Dimensione del kernel:** La dimensione del filtro convoluzionale, in questo caso 5x5.
* **Padding:** La quantità di padding applicata all'immagine di input, in questo caso 2.
* **Stride:** Il passo con cui il kernel si sposta sull'immagine di input, in questo caso 1.


### Esempio Pratico

Consideriamo un esempio con un input di 28x28, un kernel di 5x5, un padding di 2 e uno stride di 1.

1. **Calcolo della dimensione dell'output dopo il primo layer convoluzionale:**

 $Output_{size} = \frac{28 + 2 * 2 - 5}{1} + 1 = 24$

   Quindi, la dimensione dell'output del primo layer convoluzionale è 24x24.

2. **Calcolo della dimensione dell'output dopo il secondo layer convoluzionale:**

   Assumiamo che il secondo layer convoluzionale abbia un kernel di 3x3, un padding di 1 e uno stride di 2.

   $Output_{size} = \frac{24 + 2 * 1 - 3}{2} + 1 = 12$

   Quindi, la dimensione dell'output del secondo layer convoluzionale è 12x12.

3. **Calcolo della dimensione dell'output dopo il terzo layer convoluzionale:**

   Assumiamo che il terzo layer convoluzionale abbia un kernel di 3x3, un padding di 0 e uno stride di 2.

   $Output_{size} = \frac{12 + 2 * 0 - 3}{2} + 1 = 6$

   Quindi, la dimensione dell'output del terzo layer convoluzionale è 6x6.

### Dimensione del Fully-Connected Layer

La dimensione finale del fully-connected layer dipende dal numero di canali finali e dalla dimensione dell'output del layer convoluzionale finale. In questo caso, abbiamo 16 canali finali e un output di 6x6. Quindi, la dimensione del fully-connected layer sarà 6 * 6 * 16 = 576.

### Considerazioni Importanti

* La dimensione dell'output di ogni layer convoluzionale dipende dai parametri del layer precedente.
* La dimensione del fully-connected layer dipende dalla dimensione dell'output del layer convoluzionale finale e dal numero di canali finali.
* La formula per calcolare la dimensione dell'output è valida solo per layer convoluzionali con kernel quadrati.

![[cv-20241022115657586.png|535]]
![[cv-20241022115719153.png|536]]
## Analisi di un Esempio di Convoluzione

L'obiettivo è comprendere come la dimensione dell'output di una convoluzione è determinata dalla dimensione dell'input, dal numero di filtri e dal passo (stride) utilizzato.

**Esempio:**

Consideriamo un esempio con un input di dimensione 6,5 e un passo (stride) di 2. Il numero di filtri è 3.

1. **Calcolo della dimensione dell'output:**
    - Dividiamo la dimensione dell'input per il passo: 6,5 / 2 = 3,25
    - Arrotondando per difetto, otteniamo 3.
    - Aggiungiamo il numero di filtri: 3 + 3 = 6
    - La dimensione finale dell'output è quindi 6.

2. **Spiegazione:**
    - Il passo (stride) determina quanti pixel vengono saltati durante la convoluzione.
    - Il numero di filtri determina il numero di operazioni di convoluzione eseguite.
    - La dimensione finale dell'output è determinata dal numero di operazioni di convoluzione e dalla dimensione dell'input dopo l'arrotondamento per difetto.

**Applicazione:**

Se utilizziamo la funzione `model.summary()` in Keras, possiamo ottenere la dimensione dell'output per ogni layer. Ad esempio, se l'output è `(16, 4, 4)`, significa che abbiamo 16 filtri e la dimensione dell'output è 4x4.

### Adattamento del Modello

L'adattamento del modello non è un processo complesso. La principale differenza rispetto ad altri modelli, come MNIST, è la gestione dei canali. In questo caso, non si tratta di un singolo canale, ma di più canali.

### GPU e Parallelizzazione

Le operazioni di calcolo matriciale, tipiche del deep learning, possono essere parallelizzate in modo efficiente utilizzando le GPU. Le GPU Nvidia sono storicamente le più utilizzate per la computer vision e il deep learning.

Nvidia ha sviluppato CUDA, una libreria che sfrutta le capacità delle GPU Nvidia per accelerare il calcolo matriciale. 

Se si dispone di un Mac con il framework Metal installato (di solito presente sui Mac di ultima generazione con chip M1, M2, M3), è possibile utilizzare tutti i dispositivi compatibili. In caso contrario, è possibile eseguire il codice in CPU, anche se le prestazioni potrebbero essere inferiori.

### Convoluzione e Calcolo Matriciale

L'obiettivo è comprendere come la convoluzione può essere espressa in termini di calcolo matriciale.

**Passaggi:**

1. **Trasformazione dell'immagine in patch:** L'immagine viene suddivisa in patch di dimensioni definite.
2. **Trasformazione in moltiplicazione di matrici:** L'operazione di convoluzione viene trasformata in una moltiplicazione di matrici.

**Esempio:**

* **Kernel:** Un kernel 3x3 viene utilizzato per la convoluzione.
* **Matrice:** La matrice che rappresenta il kernel viene creata.
* **Patching:** La matrice del kernel viene applicata alla matrice dell'immagine, creando una nuova matrice.

**Esempio pratico:**

* **Matrice immagine:** Una matrice 4x4 con elementi numerati da 1 a 16.
* **Kernel:** Un kernel 3x3.
* **Applicazione del kernel:** Il kernel viene applicato alla matrice immagine, creando una nuova matrice.

**Vettorizzazione:**

La matrice immagine viene trasformata in un vettore di 16 elementi. Il kernel viene applicato a questo vettore, moltiplicando la matrice del kernel per il vettore.

## Applicazione di filtri

Rappresenta il processo di applicazione di un filtro a una matrice di dati in Computer Vision. 
Il filtro viene rappresentato come una matrice, e l'applicazione del filtro equivale a una moltiplicazione matriciale.

**Esempio:**

Consideriamo una matrice di dati e un filtro:

* **Matrice di dati:**
```
[ a b c d
  e f g h
  i j k l
  m n o p ]
```

* **Filtro:**
```
[ 1 2 0 0
  3 4 0 0
  0 0 0 0
  0 0 0 0 ]
```

La moltiplicazione di queste due matrici produce un nuovo vettore:

```
[ a*1 + b*2 + c*0 + d*0
  e*1 + f*2 + g*0 + h*0
  i*1 + j*2 + k*0 + l*0
  m*1 + n*2 + o*0 + p*0
  a*3 + b*4 + c*0 + d*0
  e*3 + f*4 + g*0 + h*0
  i*3 + j*4 + k*0 + l*0
  m*3 + n*4 + o*0 + p*0
  a*0 + b*0 + c*0 + d*0
  e*0 + f*0 + g*0 + h*0
  i*0 + j*0 + k*0 + l*0
  m*0 + n*0 + o*0 + p*0
  a*0 + b*0 + c*0 + d*0
  e*0 + f*0 + g*0 + h*0
  i*0 + j*0 + k*0 + l*0
  m*0 + n*0 + o*0 + p*0 ]
```

**Interpretazione:**

* Ogni riga del vettore risultante rappresenta l'applicazione del filtro su una "patch" della matrice di dati.
* La prima riga corrisponde alla prima patch (a, b, c, d).
* La seconda riga corrisponde alla seconda patch (e, f, g, h), e così via.
* Gli elementi del filtro vengono moltiplicati per i corrispondenti elementi della patch.
* Gli elementi del filtro che sono zero non contribuiscono al risultato.

**Estensione a più canali:**

Se la matrice di dati ha più canali (ad esempio, RGB), la linearizzazione di ogni canale crea nuove colonne nella matrice. La moltiplicazione matriciale viene quindi eseguita su questa matrice più grande.

**Padding e Stride:**

* Il padding aggiunge elementi aggiuntivi alla matrice di dati, consentendo di applicare il filtro anche ai bordi.
* Lo stride determina il passo con cui il filtro viene applicato alla matrice di dati.

### Moltiplicazione di Matrici e Tensori

La moltiplicazione di matrici è un'operazione fondamentale nelle reti neurali convoluzionali. Quando si applica un filtro a un singolo canale di un'immagine, si effettua una moltiplicazione di matrici. Se si utilizzano più filtri, la moltiplicazione diventa una moltiplicazione di tensori, dove ogni filtro corrisponde a un tensore.

### Struttura delle Reti Neurali Convoluzionali

Le reti neurali convoluzionali sono tipicamente strutturate in due parti:

1. **Feature Learning:** Questa parte è composta da layer convoluzionali che estraggono le caratteristiche dell'immagine. I filtri convoluzionali, o kernel, vengono applicati all'immagine per creare mappe di feature, che rappresentano l'immagine in uno spazio di feature alternativo.

2. **Classificazione:** Questa parte utilizza le feature estratte nella prima parte per classificare l'immagine.

### Pattern

Esistono due pattern comuni per le reti neurali convoluzionali:

1. **Due Parti:** La rete è composta da due parti distinte: feature learning e classificazione.
![[cv-20241022120003661.png|582]]

2. **Reticolo Convoluzionale:** Il reticolo convoluzionale è una struttura standard che viene utilizzata per estrarre le feature.
![[cv-20241022115957833.png|597]]

## Feature Learning e Reti Convoluzionali

Il processo di **feature learning** in Computer Vision si divide in due fasi:

1. **Estrazione delle feature:** da un'immagine si estrae un insieme di feature.
2. **Organizzazione delle feature:** le feature estratte vengono organizzate per la classificazione.

La parte di organizzazione delle feature è spesso realizzata tramite tecniche tradizionali come la **regressione logistica**, eventualmente con più layer. Tuttavia, le reti neurali convoluzionali (CNN) offrono un modo più potente per rappresentare l'input in modo significativo.

### Pattern di Design delle CNN

Un pattern chiave nel design delle CNN è la **riduzione della dimensione dell'immagine** e l'**aumento del numero di canali** man mano che si procede in profondità nella rete. Questo pattern è evidente nella figura presentata e rappresenta una rivoluzione nel campo della Computer Vision.

**Principi chiave:**

* **Riduzione della dimensione:** la dimensione dell'immagine viene progressivamente ridotta durante il processo di convoluzione.
* **Aumento dei canali:** il numero di canali, ovvero il numero di feature estratte, aumenta man mano che si procede in profondità nella rete.

**Esempio:** La rete LeNet segue questo pattern. Inizia con un singolo canale, poi passa a 6 canali, quindi a 16 canali. La dimensione dell'immagine parte da 32x32 e si riduce a 28x28, poi a 10x10 e infine a 5x5.
![[cv-20241022115935820.png|666]]

### La rete VGG

![[cv-20241022120041728.png|582]]

La rete **VGG**, sviluppata dal Visual Geometry Group nel 2014, è un esempio di rete convoluzionale che standardizza il processo di feature learning. La versione VGG16, in particolare, si basa su un principio di standardizzazione che può essere meglio compreso analizzando la LeNet.

**Standardizzazione:**

* **LeNet:** utilizza filtri di convoluzione 5x5. Perché non 7x7 o 3x3? La scelta della dimensione del filtro è un iperparametro della rete.
* **VGG:** standardizza la dimensione dei filtri, utilizzando filtri 3x3 in tutti i layer convoluzionali.

La prossima settimana, durante la lezione, si approfondirà l'architettura della VGG e altre reti convoluzionali.

## Standardizzazione dei Filtri nelle Reti Neurali Convoluzionali

La VGG è una rete neurale convoluzionale che si distingue per la sua semplicità e standardizzazione nell'utilizzo dei filtri. A differenza di altre reti, come la AlexNet, che utilizzano filtri di dimensioni diverse (11x11, 5x5, 3x3), la VGG si basa esclusivamente su filtri 3x3 con stride 1 e padding 1.

**Perché la VGG utilizza solo filtri 3x3?**

La scelta di utilizzare solo filtri 3x3 è motivata da due fattori principali:

1. **Simulazione di filtri più grandi:** Un layer 5x5 può essere simulato con due layer 3x3 in cascata. Allo stesso modo, un layer 7x7 può essere simulato con tre layer 3x3 in cascata. Questo significa che la VGG può ottenere la stessa capacità di rappresentazione di reti con filtri più grandi, ma con un numero inferiore di parametri.
2. **Riduzione della complessità computazionale:** Sebbene la simulazione di filtri più grandi con layer 3x3 in cascata non sia gratuita dal punto di vista computazionale, la VGG riesce comunque a ridurre la complessità complessiva della rete. Questo perché il numero di parametri da apprendere è inferiore rispetto a una rete con filtri più grandi.

**Padding 1: Mantenere la dimensione dell'output**

La VGG utilizza un padding 1 per tutti i suoi layer convoluzionali. Questo significa che l'output di ogni layer convoluzionale ha la stessa dimensione dell'input. 

Con un padding 1, la dimensione dell'output è uguale alla dimensione dell'input. Questo è importante per mantenere la risoluzione dell'immagine durante il processo di convoluzione.

**Vantaggi della standardizzazione dei filtri nella VGG:**

* **Semplicità:** La VGG è una rete relativamente semplice da implementare e da comprendere.
* **Efficienza:** La standardizzazione dei filtri consente di ridurre il numero di parametri da apprendere, rendendo la rete più efficiente dal punto di vista computazionale.
* **Prestazioni elevate:** Nonostante la sua semplicità, la VGG ha dimostrato di ottenere prestazioni elevate in diversi compiti di computer vision.

## Architettura VGG 

### Impatto dei filtri 1x1

Quando si applica un filtro 1x1 al centro di un'immagine, l'impatto è minore rispetto a filtri di dimensioni maggiori. Questo perché un filtro 1x1 mantiene la dimensione dell'immagine inalterata. Ad esempio, se si sottrae 3 da 28, si ottiene 25. Dividendo 25 per 1, si ottiene ancora 25. Sommando 1 a 25 e elevando il risultato alla potenza di 20, si ottiene un valore con la stessa dimensione iniziale.

### Pooling e Riduzione Dimensionale

L'utilizzo di un layer di pooling con un kernel 2x2 e stride pari a 2, riduce la dimensione dell'immagine. In questo caso, con stride pari a 2, la dimensione dell'immagine viene dimezzata.

### Struttura della VGG

L'architettura VGG è caratterizzata da una struttura specifica che prevede l'applicazione di filtri convoluzionali in cascata. Un esempio di questa struttura è la configurazione D, che prevede due layer convoluzionali con 64 canali.

### Implementazione della VGG

L'implementazione della VGG è realizzata in modo iterativo, leggendo la configurazione. La configurazione specifica il numero di layer, i canali e l'applicazione del pooling. La funzione "create_features" costruisce la componente di estrazione delle feature della rete.

### Classificatore

La parte successiva della rete, definita dalla configurazione, corrisponde al classificatore. L'immagine di input viene passata alla rete VGG, ottenendo una nuova rappresentazione. Questa rappresentazione viene linearizzata e passata al classificatore.

## Varianti della VGG

Esistono cinque configurazioni della VGG, che si differenziano per il numero di layer e quindi per il numero di composizioni possibili.


Le reti VGG rappresentano un'evoluzione significativa rispetto a reti come LeNet, caratterizzate da un numero di layer convoluzionali molto più elevato. 

**Configurazioni VGG:**

* **VGG 16:** La configurazione più utilizzata, con 16 layer convoluzionali.
* **VGG 19:** Si differenzia dalla VGG 16 per il numero di layer, che in questo caso sono 19.
* **VGG 11:** Contiene 11 layer convoluzionali.
* **VGG 13:** Contiene 13 layer convoluzionali.

**Confronto con LeNet:**

Mentre LeNet presentava solo due layer convoluzionali, le reti VGG dimostrano un aumento significativo nella profondità delle architetture. Questo trend verso reti con un numero crescente di layer è un aspetto interessante e significativo nell'evoluzione delle reti neurali.

**Importanza della profondità:**

L'utilizzo di un numero maggiore di layer convoluzionali consente alle reti VGG di estrarre feature più complesse e di apprendere rappresentazioni più ricche dei dati. Questo si traduce in prestazioni migliori in compiti di classificazione e riconoscimento di immagini.

![[cv-20241022120308545.png]]
## Reti neurali convoluzionali e riconoscimento oggetti

**Dimensione e complessità delle reti:**

- La rete ResNet-44.000, ad esempio, ha 138 milioni di parametri.
- Modelli più recenti, come i Transformers, possono avere miliardi di parametri.
- L'aumento del numero di parametri porta a un'architettura di rete più complessa, ma anche a sfide computazionali e di memoria.

**Estrazione di features:**

- Le CNN elaborano un'immagine di input (tipicamente 224x224 pixel con 3 canali RGB) attraverso una serie di strati convoluzionali.
- Ogni strato estrae informazioni sempre più complesse dall'immagine, riducendone la dimensione spaziale e aumentando la profondità in termini di canali.
- Il processo di estrazione di features trasforma l'immagine originale in un vettore di features (ad esempio 512x7x7), che cattura le informazioni essenziali per la classificazione.

**Miglioramenti nella classificazione di immagini:**

- L'aumento del numero di layer nelle architetture di rete (come VGG e ResNet) ha portato a un notevole miglioramento nella precisione della classificazione delle immagini.
- Si è passati da circa l'80% di accuratezza con VGG a oltre il 93% con VGG e addirittura al 98% con le più recenti configurazioni di ResNet.

**Riconoscimento di oggetti:**

- Il riconoscimento di oggetti si differenzia dalla classificazione di immagini perché mira a identificare **tutti** gli oggetti presenti nell'immagine, localizzarli spazialmente e classificarli.
- Per fare ciò, è necessario disegnare un riquadro (bounding box) attorno a ciascun oggetto e assegnare un'etichetta con la relativa probabilità.

**Architetture di rete per il riconoscimento di oggetti:**

- Architetture come ResNet, grazie alla loro capacità di estrarre features ricche e informative, sono diventate la base per lo sviluppo di algoritmi di Object Detection sempre più accurati ed efficienti.

