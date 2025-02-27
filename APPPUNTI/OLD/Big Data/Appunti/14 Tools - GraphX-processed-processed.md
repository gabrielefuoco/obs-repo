
# Efficient Graph-Parallel Computing with Spark GraphX

## Graph Analysis

Un **grafo** è una struttura dati composta da un insieme di vertici (nodi) connessi da archi. I grafi sono adatti a rappresentare relazioni non lineari tra oggetti, trovando applicazione in diversi ambiti:

* Analisi di social network
* Rappresentazione di dati e conoscenza
* Ottimizzazione e routing
* Sistemi di raccomandazione
* Modellazione della diffusione di malattie
* ...

Modellare queste relazioni permette di ottenere informazioni utili sui pattern sottostanti, creando rappresentazioni più accurate dei fenomeni analizzati.

### La necessità di strumenti grafo-paralleli nell'ambito del Big Data

Con l'aumento delle dimensioni e della complessità dei dataset, gli strumenti tradizionali di elaborazione di grafi diventano inefficienti. I framework di elaborazione del Big Data, come Hadoop o Spark, non sono ottimali per i grafi perché:

* Non considerano la struttura del grafo sottostante ai dati.
* Il calcolo può causare eccessivo movimento di dati e degradazione delle prestazioni.

Questo evidenzia la necessità di soluzioni *ad hoc* per un calcolo grafo-parallelo efficiente. **Pregel**, sviluppato da Google, è un framework per l'elaborazione efficiente di grafi su larga scala tramite cluster distribuiti. È particolarmente adatto per algoritmi iterativi altamente grafo-paralleli e si basa sul modello **Bulk Synchronous Parallel** (BSP).

## Il modello BSP

![[|331](_page_7_Figure_1.jpeg)
### Architettura di un computer BSP

**Bulk Synchronous Parallel** (BSP), introdotto da Leslie Valiant, è un modello di calcolo parallelo per la progettazione di algoritmi paralleli, soprattutto in ambienti distribuiti. È adatto per algoritmi paralleli iterativi su larga scala con significativa comunicazione e sincronizzazione. Un computer BSP è composto da:

* Diversi **elementi di elaborazione** (PE), ognuno per calcoli locali.
* Un **router** per la consegna di messaggi punto-a-punto tra PE.
* Un **sincronizzatore** per la sincronizzazione dei PE a intervalli regolari.

![[|384](_page_7_Figure_9.jpeg)]

## Il superstep

Il calcolo BSP procede in **supersteps**, unità di esecuzione sequenziali composte da tre fasi:

* **Calcolo concorrente**: ogni processore esegue calcoli locali in modo asincrono.
* **Comunicazione globale**: i processi scambiano dati.
* **Sincronizzazione a barriera**: i processi attendono il completamento di tutti gli altri prima di procedere.

![[|468](_page_8_Figure_10.jpeg)] 

### Programmazione vertex-centric e il framework Pregel

Google Pregel, per l'elaborazione distribuita scalabile di grafi, si basa su:

* **BSP**: i vertici eseguono calcoli locali, inviano messaggi e si sincronizzano tra i superstep.
* **Programmazione vertex-centric**: i grafi sono elaborati tramite funzioni che operano su singoli vertici e archi associati.

BSP fornisce un framework strutturato per il calcolo parallelo, la tolleranza ai guasti e la scalabilità. La programmazione vertex-centric semplifica lo sviluppo di algoritmi grafici, aumentando chiarezza ed efficienza. Sebbene l'implementazione di Pregel di Google non sia pubblica, esistono alternative open source come Apache **Giraph**, l'API **Gelly** di Apache Flink e **GraphX** di Apache Spark.
]
## Spark GraphX

### Caratteristiche principali

Apache Spark **GraphX** è una libreria di elaborazione di grafi di Apache Spark, fornendo un framework distribuito per l'elaborazione scalabile ed efficiente di grafi su larga scala.

* **Grafi Distribuiti Resilienti (RDG):** GraphX estende gli RDD di Spark con l'astrazione Grafo RDG, progettata per partizionare efficientemente i dati del grafo su un cluster.
* **Algoritmi Grafo:** Include algoritmi integrati come PageRank, componenti connesse e conteggio dei triangoli.
* **Operatori Grafo:** Fornisce operatori per la trasformazione e manipolazione di grafi (mappatura, creazione di sottografi, inversione di archi, ecc.).
* **Integrazione con Spark:** Si integra perfettamente con altri componenti Spark.

### Grafi Distribuiti Resilienti

GraphX estende gli RDD di Spark introducendo il **Grafo Distribuito Resiliente**, un multigrafo diretto con proprietà associate a vertici e archi. Fornisce un'interfaccia unificata per rappresentare i dati considerando la struttura del grafo, mantenendo l'efficienza degli RDD di Spark. Permette di sfruttare concetti di grafo e primitive efficienti per il calcolo del grafo e operazioni parallele distribuite di dati tipiche di Spark. Un grafo contiene due RDD distinti: uno per gli archi e uno per i vertici.

![[](_page_12_Figure_9.jpeg)
GraphX fornisce un'ulteriore visualizzazione della struttura del grafo tramite **EdgeTriplet**, estendendo le informazioni degli RDD degli archi aggiungendo le proprietà dei vertici sorgente e destinazione.

* RDD degli archi: insieme di tuple (*id sorgente, id destinazione, attributo*) per ogni arco.
* RDD dei vertici: insieme di tuple (*id vertice, attributo*) per ogni vertice.
* La tripla arco fornisce la tupla (*id sorgente, id destinazione, attributo sorgente, attributo arco, attributo destinazione*).

![[|475](_page_13_Figure_4.jpeg)

### Partizionamento Vertex-cut

I grafi in GraphX sono partizionati per applicazioni distribuite grafo-parallele scalabili. Si usa un approccio **basato sui vertici** per ridurre i costi di comunicazione e archiviazione. L'operatore *Graph.partitionBy* permette di scegliere tra diversi algoritmi di partizionamento.

![[|403](_page_14_Picture_7.jpeg)

### Operatori di base

### Operatori di base

Dato un grafo di input, vengono fornite informazioni topologiche e rappresentazioni basate su RDD:

```scala
// Informazioni sul grafo
val numEdges: Long
val numVertices: Long
val inDegrees: VertexRDD[Int]
val outDegrees: VertexRDD[Int]
val degrees: VertexRDD[Int]

// Viste del grafo come collezioni
val vertices: VertexRDD[VD]
val edges: EdgeRDD[ED]
val triplets: EdgeTriplet[VD, ED]
```

È possibile trasformare gli attributi di vertici, archi e triplette tramite funzioni di mappatura definite dall'utente. È inoltre possibile unire dati da RDD esterni con il grafo:

```scala
// Trasforma gli attributi di vertici e archi
def mapVertices[VD2](map: (VertexId, VD) => VD2): Graph[VD2, ED]
def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2]
def mapEdges[ED2](map: (PartitionID, Iterator[Edge[ED]]) => Iterator[ED2]): Graph[VD, ED2]
def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2]
def mapTriplets[ED2](map: (PartitionID, Iterator[EdgeTriplet[VD, ED]]) => Iterator[ED2]): Graph[VD, ED2]

// Unisci RDD con il grafo
def joinVertices[U](table: RDD[(VertexId, U)])(mapFunc: (VertexId, VD, U) => VD): Graph[VD, ED]
def outerJoinVertices[U, VD2](other: RDD[(VertexId, U)])(mapFunc: (VertexId, VD, Option[U]) => VD2): Graph[VD2, ED]
```

## L'API Pregel

GraphX fornisce un'implementazione del modello di calcolo vertex-centrico **Pregel** per applicazioni grafo su larga scala altamente parallele. Durante un superstep Pregel, il framework invoca una funzione definita dall'utente (UDF) per ogni vertice, eseguita in parallelo. Ogni vertice può cambiare il suo stato, leggere messaggi ricevuti nel superstep precedente o inviarne di nuovi per il superstep successivo.

Differenze con il Pregel standard:

* GraphX esegue il calcolo dei messaggi in parallelo come funzione della tripla arco, accedendo alle proprietà del vertice sorgente e di quello di destinazione.
* I vertici possono inviare messaggi solo ai vicini; quelli che non ricevono messaggi in un superstep vengono saltati.

### L'operatore Pregel

L'operatore **Pregel** accetta due insiemi di parametri di input:

* Il primo specifica il messaggio iniziale, il numero di iterazioni e la direzione dell'arco.
* Il secondo aspetta tre funzioni definite dall'utente:
 * **vprog: (VertexId, VD, A) => VD**: codifica il comportamento del vertice. Questa UDF è invocata su ogni vertice che riceve un messaggio e calcola il valore del vertice aggiornato.

## Algoritmi di Elaborazione di Grafi con GraphX

Le User Defined Functions (UDF) utilizzate in GraphX per l'elaborazione di grafi includono:

* **`sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)]`**: Questa UDF viene applicata agli archi in uscita dei vertici che hanno ricevuto messaggi nell'iterazione corrente. Restituisce un iteratore di coppie (VertexId, A), dove `A` è il tipo del messaggio.

* **`mergeMsg: (A, A) => A`**: Specifica come due messaggi ricevuti da un vertice devono essere uniti in un singolo messaggio dello stesso tipo. Questa UDF deve essere commutativa e associativa.

Al termine del calcolo, il grafo risultante viene restituito in output.

## L'Algoritmo PageRank

PageRank è un algoritmo iterativo sviluppato da Larry Page e Sergey Brin, utilizzato da Google Search per classificare le pagine web. Si basa sull'idea che un link da una pagina ad un'altra rappresenta un voto di fiducia: pagine con molti link in entrata da fonti autorevoli sono considerate più importanti.

PageRank modella il processo di navigazione di un utente:

![](_page_20_Figure_5.jpeg)

L'utente può interrompere la navigazione e effettuare un "*random hop*", ovvero passare ad un URL diverso non direttamente raggiungibile dalla pagina corrente.

La probabilità di continuare a cliccare sui link in uscita è il fattore di smorzamento, generalmente impostato a *d* = 0.85; la probabilità di un *random hop* è 1 − *d* = 0.15.

### Approfondimento Matematico

Il PageRank della pagina *p<sub>i</sub>*, *PR(p<sub>i</sub>)*, rappresenta la probabilità che un utente, partendo da una pagina *p<sub>j</sub>*, arrivi su *p<sub>i</sub>*. È dato dalla somma di due probabilità:

* La probabilità di raggiungere *p<sub>i</sub>* tramite un *random hop*. Si assume una probabilità uniforme di atterrare su ciascuna delle *N* pagine disponibili.
* La probabilità di raggiungere *p<sub>i</sub>* seguendo un link esistente in *p<sub>j</sub>*, assumendo una probabilità uniforme di seguire ciascun link in *p<sub>j</sub>*.

La formula è:

$$PR(p_i) = \frac{1 - d}{N} + d \left( \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)} \right)$$

Dove:

* *d*: fattore di smorzamento
* 1-*d*: probabilità di *random hop*
* *N*: numero di pagine disponibili
* *M(p<sub>i</sub>)*: insieme delle pagine che collegano a *p<sub>i</sub>*
* *L(p<sub>j</sub>)*: numero di link nella pagina *p<sub>j</sub>*

### Tecniche di Sintesi di Testo

* **Sintesi Estrattive:** Identificazione ed estrazione delle informazioni più importanti (es. frasi principali) da un documento, presentate in un riassunto conciso.

* **Sintesi Astrattive:** Tecniche più complesse che prevedono la comprensione del significato e del contesto del testo originale e la creazione di un riassunto simile a quello umano, utilizzando tecniche di NLP (es. LLM).

**Applicazioni principali:** riepiloghi di notizie, motori di ricerca, e-learning, ricerca accademica.

### TextRank: Utilizzo di PageRank per la Sintesi Estrattiva

TextRank, proposto da Mihalcea e Tarau nel 2004, è un algoritmo di sintesi estrattiva. Rappresenta il testo di input come un grafo pesato di frasi semanticamente connesse ed estrae un riassunto identificando le *k* frasi più rappresentative tramite PageRank.

Dato un testo T da riassumere, l'algoritmo crea un grafo G = <S,E>:

* S = s<sub>1</sub>, …, s<sub>n</sub> contiene le *n* frasi presenti in T.
* E contiene gli archi del grafo, collegando ogni coppia di frasi in S con un peso *w<sub>i,j</sub>* che rappresenta la similarità testuale tra s<sub>i</sub> e s<sub>j</sub>.

## Implementazione di TextRank usando GraphX

L'implementazione di TextRank con GraphX prevede i seguenti step:

1. **Inizializzazione:** Si inizializza una sessione Spark e si definisce una funzione per calcolare la similarità tra frasi.

2. **Creazione del grafo:** Si legge il testo, si estraggono le frasi, si crea un grafo completamente connesso dove ogni nodo è una frase e il peso di ogni arco è la similarità tra le frasi connesse.

3. **Preparazione del grafo per PageRank:** Si normalizzano i pesi degli archi (somma dei pesi uscenti da ogni nodo = 1) e si assegna un PageRank iniziale a ogni nodo.

4. **Definizione della UDF Pregel:** Si definisce una UDF Pregel con tre funzioni: `vertexProgram` (aggiorna il PageRank di ogni nodo), `sendMessage` (invia il PageRank ai vicini) e `messageCombiner` (aggrega i messaggi ricevuti).

5. **Costruzione del riassunto:** La funzione `buildSummary` seleziona le *k* frasi con il PageRank più alto, mantenendo l'ordine originale.

6. **Esecuzione:** Il metodo principale coordina la costruzione del grafo, la sua preparazione, l'esecuzione di Pregel per 50 iterazioni e la generazione del riassunto finale usando `buildSummary`.
