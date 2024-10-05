
| **Termine**                           | **Definizione**                                                                                                                                                                                                              |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Grafo**                             | Struttura matematica composta da vertici (nodi) collegati da archi, usata per rappresentare relazioni tra oggetti.                                                                                                           |
| **Graph-Parallel**                    | Paradigma di elaborazione per grafi su larga scala, dove i calcoli vengono eseguiti in parallelo sui vertici.                                                                                                                |
| **Big Data**                          | Insiemi di dati estremamente grandi e complessi, difficili da gestire con strumenti tradizionali.                                                                                                                            |
| **Framework**                         | In informatica, una struttura software che fornisce una base per lo sviluppo di applicazioni.                                                                                                                                |
| **Hadoop, Spark**                     | Framework open-source per l'elaborazione distribuita di Big Data.                                                                                                                                                            |
| **Bulk Synchronous Parallel (BSP)**   | Modello di calcolo parallelo dove i processi vengono eseguiti in fasi di calcolo locale, comunicazione globale e sincronizzazione.                                                                                           |
| **Pregel**                            | Framework di Google per l'elaborazione distribuita di grafi basato sul modello BSP e sulla programmazione Vertex-Centric.                                                                                                    |
| **Vertex-Centric**                    | Paradigma di programmazione per grafi dove il codice viene eseguito su ogni vertice e i messaggi vengono scambiati tra vertici adiacenti.                                                                                    |
| **Apache Giraph, Gelly, GraphX**      | Framework open-source per l'elaborazione distribuita di grafi.                                                                                                                                                               |
| **Resilient Distributed Graph (RDG)** | Astrazione di GraphX per rappresentare grafi distribuiti in modo resiliente su un cluster Spark.                                                                                                                             |
| **EdgeTriplet**                       | Struttura dati in GraphX che combina le informazioni di un arco con quelle dei vertici di origine e destinazione.                                                                                                            |
| **Vertex-Cut Partitioning**           | Tecnica di partizionamento dei dati utilizzata in GraphX per distribuire i calcoli del grafo su più macchine.                                                                                                                |
| **PageRank**                          | Algoritmo che assegna un punteggio di importanza a ogni vertice di un grafo, basandosi sul numero e sulla qualità dei collegamenti in entrata. Sviluppato da Google per classificare le pagine web nei risultati di ricerca. |
| **TextRank**                          | Algoritmo di riassunto di testo che utilizza il PageRank per identificare le frasi più importanti in un documento.                                                                                                           |

## Cos'è un Grafo?
- Un grafo è una struttura composta da un insieme di **vertici (nodi)** connessi da **archi**.
- Utilizzato per rappresentare relazioni non lineari tra oggetti, è applicabile in diversi ambiti, tra cui:
  - **Reti sociali**
  - **Rappresentazione di dati** e **conoscenza**
  - **Ottimizzazione** e **routing**
  - **Sistemi di raccomandazione**
  - **Modellazione della diffusione di malattie**
- I grafi permettono di ottenere intuizioni sui pattern sottostanti e rappresentazioni accurate dei fenomeni analizzati.

## Strumenti Graph-Parallel e Big Data
- Con dataset sempre più **grandi** e **complessi**, gli strumenti tradizionali per elaborare grafi sono **inefficienti**.
- Framework come **Hadoop** e **Spark** non sono ottimali per i grafi perché:
  - Non considerano la **struttura del grafo** sottostante.
  - Portano a eccessivi **spostamenti di dati** e **prestazioni degradate**.
#### Soluzione: Framework Graph-Parallel
- Framework come **Pregel** di Google sono progettati per l’elaborazione efficiente di **grafi su larga scala**:
  - Supportano algoritmi **graph-parallel** iterativi.
  - Si basano sul modello **Bulk Synchronous Parallel (BSP)**.
---
## Il Modello Bulk Synchronous Parallel (BSP)

### Architettura di un sistema BSP
- **Bulk Synchronous Parallel (BSP)**, introdotto da Leslie Valiant negli anni '90, è un modello per calcolo parallelo distribuito, particolarmente adatto a problemi iterativi su larga scala.
- Un sistema BSP è composto da:
  - **Elementi di elaborazione (PE)**: eseguono calcoli locali.
  - **Router**: gestisce la comunicazione **point-to-point** tra PE.
  - **Sincronizzatore**: coordina la sincronizzazione tra PE.

### Superstep
- Il calcolo nel BSP avviene in **superstep**, ognuno con tre fasi:
  1. **Calcolo concorrente**: ogni processore esegue calcoli locali in modo asincrono.
  2. **Comunicazione globale**: i processi scambiano dati tra loro.
  3. **Sincronizzazione**: ogni processo attende che tutti raggiungano la stessa barriera prima di procedere.
---
## Framework Pregel e Programmazione Vertex-Centric
- **Pregel** (Google) supporta l'elaborazione distribuita di grafi basandosi su:
  - **BSP**: i vertici eseguono calcoli locali, inviano messaggi e si sincronizzano.
  - **Programmazione vertex-centric**: permette di operare sui singoli vertici e i loro archi.
  
#### Vantaggi
- **BSP** offre un framework strutturato per **calcolo parallelo**, tolleranza ai guasti e **scalabilità**.
- La programmazione **vertex-centric** semplifica lo sviluppo di algoritmi sui grafi, migliorando **chiarezza** ed **efficienza**.
  
### Alternative a Pregel
- Alternative open-source includono:
  - **Apache Giraph**
  - **Gelly (Apache Flink)**
  - **GraphX (Apache Spark)**

---
# Apache Spark GraphX

## Caratteristiche Chiave
- **GraphX** è una libreria di Apache Spark per l'elaborazione distribuita di grafi su larga scala.
- Le caratteristiche principali includono:
  - **Resilient Distributed Graphs (RDG)**: Estende gli RDD di Spark per partizionare in modo efficiente i dati del grafo su un cluster.
  - **Algoritmi per Grafi**: Include algoritmi integrati come **PageRank**, **componenti connesse**, e **conteggio dei triangoli**.
  - **Operatori per Grafi**: Permette operazioni come **mappatura**, creazione di **sottografi**, inversione degli archi e altre manipolazioni dei grafi.
  - **Integrazione con Spark**: Supporta l'integrazione con altri componenti Spark per combinare l'elaborazione di grafi con machine learning e analisi dati.

## Resilient Distributed Graphs (RDG)
- GraphX introduce il **Resilient Distributed Graph (RDG)**, una nuova astrazione basata su un multigrafo diretto con proprietà attaccate ai vertici e agli archi.
- Ogni grafo è composto da:
  - Un **RDD per i vertici** (tupla: ID, attributo)
  - Un **RDD per gli archi** (tupla: ID sorgente, ID destinazione, attributo)
- Questa struttura consente operazioni **data-parallel distribuite** tipiche di Spark.

## EdgeTriplet
- **EdgeTriplet** fornisce una vista estesa del grafo, combinando le informazioni di un arco con quelle dei vertici di origine e destinazione:
  - **Tripla**: (ID sorgente, ID destinazione, attributo sorgente, attributo arco, attributo destinazione)
- Utile per ottenere informazioni complete su archi e vertici interconnessi.

## Partizionamento Vertex-Cut
- GraphX utilizza un **partizionamento vertex-cut** per distribuire il calcolo del grafo su più macchine.
- Il partizionamento assegna gli **archi** alle macchine e distribuisce i **vertici** su più nodi, riducendo il costo di **comunicazione** e **storage**.

## Operatori di Base
- Informazioni sui grafi e statistiche di base:

```scala
val numEdges: Long
val numVertices: Long
val inDegrees: VertexRDD[Int]
val outDegrees: VertexRDD[Int]
val degrees: VertexRDD[Int]
```

- Rappresentazioni topologiche:

```scala
val vertices: VertexRDD[VD]
val edges: EdgeRDD[ED]
val triplets: RDD[EdgeTriplet[VD, ED]]
```

- Trasformazione di attributi su vertici, archi e triplette:

```scala
def mapVertices[VD2](map: (VertexId, VD) => VD2): Graph[VD2, ED]
def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2]
def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2]
```

- **Unione di RDD esterni** con i grafi:

```scala
def joinVertices[U](table: RDD[(VertexId, U)])(mapFunc: (VertexId, VD, U) => VD): Graph[VD, ED]
```

## API Pregel
- GraphX implementa il modello **Pregel** per applicazioni **graph-parallel**:
  - Ogni vertice esegue in parallelo una funzione definita dall'utente (UDF) e può inviare messaggi ai vicini.
  - Il calcolo dei messaggi viene eseguito in base alla **tripla** dell'arco.

#### Operatore Pregel
- Esempio di utilizzo:

```scala
def pregel[A](
  initialMsg: A,
  maxIterations: Int,
  activeDirection: EdgeDirection)(
  vprog: (VertexId, VD, A) => VD,
  sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)],
  mergeMsg: (A, A) => A
): Graph[VD, ED]
```

- **Funzioni UDF**:
  - `vprog`: comportamento del vertice, calcola il valore aggiornato.
  - `sendMsg`: invia messaggi ai vertici adiacenti.
  - `mergeMsg`: unisce i messaggi ricevuti in uno solo.

## Vantaggi di GraphX
- **Scalabilità** e supporto per l'elaborazione di **grafi su larga scala**.
- **Integrazione con Spark** per combinare analisi di dati e machine learning.
- Ampia collezione di **algoritmi per grafi** e operatori per la manipolazione flessibile.
---
## Esempio di Programmazione

### L'algoritmo PageRank
- PageRank è un algoritmo iterativo sviluppato da Larry Page e Sergey Brin, utilizzato da Google Search per classificare le pagine web nei risultati del suo motore di ricerca.
- Si basa sull'idea che un collegamento da una pagina a un'altra possa essere visto come un voto di fiducia: le pagine con più collegamenti in entrata da fonti autorevoli sono considerate più importanti o autorevoli.
- PageRank modella il processo che porta un utente a una data pagina:
  - L'utente generalmente arriva su quella pagina attraverso una sequenza di link casuali, seguendo un percorso attraverso multiple pagine.
  - L'utente può eventualmente smettere di cliccare sui link in uscita e cercare un URL diverso che non è direttamente raggiungibile dalla pagina corrente (cioè, il salto casuale).
- La probabilità che un utente continui a cliccare sui link in uscita è il fattore di smorzamento, generalmente impostato a d = 0,85, mentre la probabilità di salto casuale è 1 - d = 0,15.

### Approfondimento matematico
Il PageRank di una pagina pi rappresenta la probabilità che un utente, situato su una pagina pj, arrivi su pi. È dato dalla somma di due probabilità:

- Il primo termine esprime la probabilità che un utente raggiunga pi attraverso un salto casuale, smettendo di cliccare sui link presenti in pj. Si assume che la probabilità di atterrare su ciascuna delle N pagine disponibili sia equamente distribuita.
- Il secondo termine dà la probabilità che l'utente arrivi sulla pagina pi seguendo un link esistente in pj, assumendo che la probabilità che l'utente segua ogni link in pj sia equamente distribuita.

La formula matematica è:

$$
PR(A) = \frac{(1-d)}{n} + d \sum_{j=1}^{n} \frac{PR(P_j)}{L(P_j)}
$$

Dove:
- $PR(A)$ è il PageRank della pagina $A$
- $d$ è il fattore di smorzamento, solitamente impostato a $0.85$
- $n$ è il numero totale di pagine
- $PR(P_j)$ è il PageRank della pagina $P_j$ che punta alla pagina $A$
- $L(P_j)$ è il numero di link in uscita dalla pagina $P_j$


## Tecniche di Riassunto
- **Riassunto estrattivo**: Estrae le frasi più importanti da un testo, presentandole in modo conciso.
- **Riassunto astrattivo**: Richiede comprensione e rielaborazione del testo per creare un riassunto simile a quello prodotto da un umano.
#### Applicazioni
- **Motori di ricerca**: Riassunti estrattivi per snippet nei risultati di ricerca.
- **E-learning**: Riassunti concisi per materiali educativi.
- **Ricerca accademica**: Riassunti per comprendere rapidamente articoli scientifici

### TextRank: uso di PageRank per il riassunto estrattivo

- **TextRank** è un algoritmo di riassunto estrattivo basato su **PageRank**, proposto da Mihalcea e Tarau nel 2004.
- Modella un testo come un grafo pesato di frasi collegate semanticamente, selezionando le frasi più rappresentative.
- Formalmente, dato un testo T da riassumere, l'algoritmo crea un grafo G = <S,E>:
  - L'insieme S = {s1,...,sn} contiene le n frasi presenti in T.
  - L'insieme E contiene gli archi del grafo ed è creato collegando ogni coppia di frasi in S, dove ogni connessione è associata a un peso wi,j che rappresenta la similarità testuale tra si e sj.

## Messaggi Chiave
- I **grafi** sono strumenti potenti per modellare scenari reali.
- Nei **Big Data**, framework come **GraphX** sono essenziali per l'elaborazione distribuita dei grafi, garantendo **scalabilità** e **tolleranza ai guasti**.
- **GraphX** fornisce un'API ottimizzata per eseguire elaborazioni complesse di grafi, integrata nell'ecosistema Spark.
---
# Domande Frequenti sui Grafi e GraphX

## 1. Cosa sono i grafi e perché sono importanti nell'analisi dei dati?

Un grafo è una struttura matematica composta da un insieme di vertici (nodi) collegati da archi. Sono utilizzati per rappresentare relazioni e connessioni tra oggetti, come reti sociali, relazioni tra dati e conoscenza, o sistemi di raccomandazione. La loro capacità di modellare scenari complessi li rende strumenti potenti per l'analisi dei dati.

## 2. Quali sono le sfide nell'elaborazione di grafi su larga scala e come vengono affrontate?

I grafi su larga scala, con milioni o miliardi di vertici e archi, presentano sfide significative per gli strumenti di elaborazione tradizionali. I framework come Hadoop e Spark non sono ottimizzati per la struttura dei grafi, causando inefficienze e prestazioni ridotte. Per questo motivo, sono stati sviluppati framework specifici per l'elaborazione di grafi, come Pregel e GraphX, che sfruttano il parallelismo e architetture distribuite per gestire efficacemente la mole di dati.

## 3. Cos'è il modello Bulk Synchronous Parallel (BSP) e come si applica all'elaborazione dei grafi?

Il modello BSP è un paradigma di calcolo parallelo in cui l'elaborazione avviene in "superstep". Ogni superstep consiste in tre fasi: calcolo locale sui processori, comunicazione globale tra i processori e sincronizzazione per garantire che tutti i processori siano allineati prima di procedere al superstep successivo. Questo modello si adatta bene all'elaborazione dei grafi perché consente di suddividere il grafo in parti elaborabili in parallelo, riducendo la comunicazione e migliorando le prestazioni.

## 4. Cosa significa programmazione "vertex-centric" e quali sono i suoi vantaggi?

La programmazione "vertex-centric" è un approccio in cui l'elaborazione si concentra sui singoli vertici del grafo. Ogni vertice esegue una funzione definita dall'utente sui propri dati e può inviare messaggi ai vertici adiacenti. Questo approccio semplifica lo sviluppo di algoritmi sui grafi, migliorando la leggibilità del codice e facilitando il ragionamento sulle operazioni da eseguire.

## 5. Cos'è Apache Spark GraphX e quali sono le sue caratteristiche principali?

Apache Spark GraphX è una libreria di Apache Spark progettata per l'elaborazione distribuita di grafi su larga scala. Offre:

- **Resilient Distributed Graphs (RDG):** un'estensione degli RDD di Spark per gestire i dati del grafo in modo distribuito e resiliente ai guasti.
- **Algoritmi per grafi:** algoritmi predefiniti come PageRank, ricerca di componenti connesse e conteggio dei triangoli.
- **Operatori per grafi:** funzioni per manipolare i grafi, come mappatura, filtraggio e aggregazione.
- **Integrazione con Spark:** sinergia con altri componenti di Spark per combinare l'elaborazione dei grafi con machine learning e analisi dei dati.

## 6. Come funziona il partizionamento "vertex-cut" in GraphX e perché è importante?

Il partizionamento "vertex-cut" in GraphX assegna gli archi alle macchine in base ai loro vertici di origine e destinazione. I vertici possono essere replicati su più nodi per ridurre la comunicazione durante l'elaborazione. Questo approccio riduce il traffico di rete e migliora le prestazioni complessive.

## 7. Cosa sono gli EdgeTriplet e come vengono utilizzati in GraphX?

Un EdgeTriplet in GraphX è una struttura dati che rappresenta un arco con informazioni aggiuntive sui vertici di origine e destinazione. Fornisce una vista completa dell'arco e dei suoi vertici adiacenti, facilitando operazioni come l'aggregazione di informazioni dai vicini o l'applicazione di filtri basati sugli attributi dei vertici.

## 8. Puoi fornire un esempio di applicazione pratica di GraphX?

Un esempio pratico è l'utilizzo di GraphX per implementare l'algoritmo PageRank, che classifica l'importanza delle pagine web in base ai collegamenti in entrata. GraphX può elaborare grafi web su larga scala, calcolando il PageRank in modo efficiente e scalabile. Questo risultato può essere utilizzato per migliorare i motori di ricerca o per analizzare l'influenza e la reputazione di siti web e utenti sui social media.

### Quiz


1. **Quali sono i vantaggi dell'utilizzo di un framework Graph-Parallel rispetto a framework tradizionali come Hadoop o Spark per l'elaborazione di grafi su larga scala?**
2. **Descrivere il modello di calcolo Bulk Synchronous Parallel (BSP) e come viene applicato all'elaborazione di grafi.**
3. **Quali sono le caratteristiche principali di Apache Spark GraphX?**
4. **Spiegare il concetto di Resilient Distributed Graph (RDG) in GraphX.**
5. **Qual è lo scopo della struttura dati EdgeTriplet in GraphX?**
6. **Come viene implementato il partizionamento Vertex-Cut in GraphX per la distribuzione dei dati?**
7. **Descrivere l'API Pregel in GraphX e il suo funzionamento.**
8. **Quali sono i vantaggi dell'utilizzo di GraphX per l'elaborazione di grafi?**
9. **Come funziona l'algoritmo PageRank e a cosa serve?**
10. **In che modo TextRank utilizza PageRank per il riassunto di testo?**

### Chiave di Risposta del Quiz

1. I framework Graph-Parallel sono ottimizzati per l'elaborazione di grafi, considerando la loro struttura sottostante e riducendo gli spostamenti di dati, a differenza di Hadoop e Spark che sono più generalisti e meno efficienti per questo tipo di elaborazione.
2. Il modello BSP organizza il calcolo in superstep, ognuno composto da fasi di calcolo locale, comunicazione globale e sincronizzazione, garantendo correttezza e scalabilità per problemi iterativi come quelli sui grafi.
3. GraphX offre RDG per la gestione di grafi distribuiti, algoritmi predefiniti, operatori per manipolazioni flessibili e integrazione con altri componenti di Spark.
4. RDG è un'estensione degli RDD di Spark che partiziona i dati del grafo (vertici e archi) su un cluster, garantendo resilienza e elaborazione distribuita efficiente.
5. EdgeTriplet fornisce una vista completa di un arco, includendo informazioni sui vertici di origine e destinazione, facilitando l'elaborazione basata sul contesto locale.
6. Il partizionamento Vertex-Cut assegna gli archi alle macchine e distribuisce i vertici, riducendo la comunicazione inter-nodo e ottimizzando l'utilizzo della memoria.
7. L'API Pregel permette di implementare algoritmi iterativi vertex-centric, definendo funzioni per il comportamento dei vertici, l'invio di messaggi e la loro unione.
8. GraphX offre scalabilità, integrazione con Spark, un'ampia collezione di algoritmi, operatori flessibili e supporto per l'elaborazione distribuita, semplificando l'analisi di grafi complessi.
9. PageRank assegna un punteggio di importanza ai vertici di un grafo in base al numero e all'importanza dei link in entrata, simulando il comportamento di un utente che naviga casualmente il grafo.
10. TextRank modella un testo come un grafo di frasi, assegnando pesi agli archi in base alla similarità semantica. Applica poi PageRank per identificare le frasi più centrali e informative per il riassunto.

