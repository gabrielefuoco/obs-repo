
## Schema Riassuntivo: Analisi di Grafi e Framework Paralleli

**1. Introduzione all'Analisi di Grafi**
    *   I grafi rappresentano relazioni non lineari tra oggetti.
    *   Applicazioni:
        *   Analisi di social network
        *   Rappresentazione di dati e conoscenza
        *   Ottimizzazione e routing
        *   Sistemi di raccomandazione
        *   Modellazione della diffusione di malattie

**2. Necessità di Strumenti Grafo-Paralleli nel Big Data**
    *   Strumenti tradizionali inefficienti con dataset di grandi dimensioni.
    *   Framework Big Data (Hadoop, Spark) non ottimizzati per grafi:
        *   Non considerano la struttura del grafo.
        *   Causano eccessivo movimento di dati.
    *   Soluzioni *ad hoc* necessarie per il calcolo grafo-parallelo efficiente.
    *   **Pregel** (Google): framework per l'elaborazione efficiente di grafi su larga scala.
        *   Basato sul modello **Bulk Synchronous Parallel (BSP)**.
        *   Adatto per algoritmi iterativi altamente grafo-paralleli.

**3. Modello Bulk Synchronous Parallel (BSP)**
    *   Modello di calcolo parallelo per la progettazione di algoritmi paralleli in ambienti distribuiti.
    *   Adatto per algoritmi paralleli iterativi su larga scala con significativa comunicazione e sincronizzazione.
    *   Architettura di un computer BSP:
        *   **Elementi di elaborazione (PE):** per calcoli locali.
        *   **Router:** per la consegna di messaggi punto-a-punto tra PE.
        *   **Sincronizzatore:** per la sincronizzazione dei PE a intervalli regolari.

**4. Superstep nel Modello BSP**
    *   Unità di esecuzione sequenziali nel calcolo BSP.
    *   Tre fasi:
        *   **Calcolo concorrente:** ogni processore esegue calcoli locali in modo asincrono.
        *   **Comunicazione globale:** i processi scambiano dati.
        *   **Sincronizzazione a barriera:** i processi attendono il completamento di tutti gli altri prima di procedere.

**5. Programmazione Vertex-Centric e Pregel**
    *   Google Pregel si basa su:
        *   **BSP:** vertici eseguono calcoli locali, inviano messaggi e si sincronizzano tra i superstep.
        *   **Programmazione vertex-centric:** i grafi sono elaborati tramite funzioni che operano su singoli vertici e archi associati.
    *   Vantaggi:
        *   BSP fornisce un framework strutturato per il calcolo parallelo, la tolleranza ai guasti e la scalabilità.
        *   La programmazione vertex-centric semplifica lo sviluppo di algoritmi grafici, aumentando chiarezza ed efficienza.
    *   Alternative open source:
        *   Apache **Giraph**
        *   API **Gelly** di Apache Flink
        *   **GraphX** di Apache Spark

**6. Spark GraphX**
    *   Libreria di elaborazione di grafi di Apache Spark.
    *   Framework distribuito per l'elaborazione scalabile ed efficiente di grafi su larga scala.
    *   Caratteristiche principali:
        *   **Grafi Distribuiti Resilienti (RDG):** estensione degli RDD di Spark per partizionare efficientemente i dati del grafo.
        *   **Algoritmi Grafo:** algoritmi integrati (PageRank, componenti connesse, conteggio dei triangoli).
        *   **Operatori Grafo:** trasformazione e manipolazione di grafi (mappatura, creazione di sottografi, inversione di archi, ecc.).
        *   **Integrazione con Spark:** si integra perfettamente con altri componenti Spark.

**7. Grafi Distribuiti Resilienti (RDG) in GraphX**
    *   Estensione degli RDD di Spark.
    *   Multigrafo diretto con proprietà associate a vertici e archi.
    *   Interfaccia unificata per rappresentare i dati considerando la struttura del grafo.
    *   Mantiene l'efficienza degli RDD di Spark.

---

**Schema Riassuntivo GraphX**

**I. Introduzione a GraphX**

*   Sfrutta concetti di grafo e primitive efficienti per calcolo e operazioni parallele distribuite.
*   Contiene due RDD distinti:
    *   RDD degli archi: tuple (*id sorgente, id destinazione, attributo*).
    *   RDD dei vertici: tuple (*id vertice, attributo*).
*   **EdgeTriplet**: Visualizzazione estesa degli archi con proprietà dei vertici sorgente e destinazione.
    *   Tupla: (*id sorgente, id destinazione, attributo sorgente, attributo arco, attributo destinazione*).

**II. Partizionamento Vertex-cut**

*   Grafi partizionati per applicazioni distribuite grafo-parallele scalabili.
*   Approccio **basato sui vertici** per ridurre costi di comunicazione e archiviazione.
*   Operatore *Graph.partitionBy* per scegliere algoritmi di partizionamento.

**III. Operatori di Base**

*   Informazioni topologiche e rappresentazioni basate su RDD:
    *   `val numEdges: Long`
    *   `val numVertices: Long`
    *   `val inDegrees: VertexRDD[Int]`
    *   `val outDegrees: VertexRDD[Int]`
    *   `val degrees: VertexRDD[Int]`
*   Viste del grafo come collezioni:
    *   `val vertices: VertexRDD[VD]`
    *   `val edges: EdgeRDD[ED]`
    *   `val triplets: EdgeTriplet[VD, ED]`
*   Trasformazione attributi tramite funzioni di mappatura definite dall'utente:
    *   `def mapVertices[VD2](map: (VertexId, VD) => VD2): Graph[VD2, ED]`
    *   `def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2]`
    *   `def mapEdges[ED2](map: (PartitionID, Iterator[Edge[ED]]) => Iterator[ED2]): Graph[VD, ED2]`
    *   `def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2]`
    *   `def mapTriplets[ED2](map: (PartitionID, Iterator[EdgeTriplet[VD, ED]]) => Iterator[ED2]): Graph[VD, ED2]`
*   Unione dati da RDD esterni:
    *   `def joinVertices[U](table: RDD[(VertexId, U)])(mapFunc: (VertexId, VD, U) => VD): Graph[VD, ED]`
    *   `def outerJoinVertices[U, VD2](other: RDD[(VertexId, U)])(mapFunc: (VertexId, VD, Option[U]) => VD2): Graph[VD2, ED]`

**IV. API Pregel**

*   Implementazione del modello vertex-centrico **Pregel** per applicazioni grafo su larga scala.
*   Funzione definita dall'utente (UDF) invocata per ogni vertice in parallelo durante un superstep.
*   Differenze con Pregel standard:
    *   Calcolo dei messaggi in parallelo come funzione della tripla arco.
    *   Vertici inviano messaggi solo ai vicini; quelli senza messaggi vengono saltati.

**V. Operatore Pregel**

*   Accetta due insiemi di parametri di input:
    *   Messaggio iniziale, numero di iterazioni, direzione dell'arco.
    *   Tre funzioni definite dall'utente:
        *   `vprog: (VertexId, VD, A) => VD`: Comportamento del vertice, calcola il valore aggiornato.

**VI. Algoritmi di Elaborazione di Grafi con GraphX**

*   User Defined Functions (UDF) includono:
    *   `sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)]`: Applicata agli archi in uscita dei vertici che hanno ricevuto messaggi. Restituisce un iteratore di coppie (VertexId, A).
    *   `mergeMsg: (A, A) => A`: Specifica come unire due messaggi ricevuti da un vertice. Deve essere commutativa e associativa.
*   Il grafo risultante viene restituito in output.

---

**Schema Riassuntivo: Algoritmo PageRank e TextRank**

**1. PageRank: Algoritmo di Classificazione Web**
    *   **1.1. Concetto Fondamentale:** Misura l'importanza di una pagina web basandosi sui link in entrata (voti di fiducia).
    *   **1.2. Modello di Navigazione Utente:** Simula il comportamento di un utente che naviga sul web, con possibilità di *random hop*.
    *   **1.3. Fattore di Smorzamento (d):** Probabilità di continuare a cliccare sui link (tipicamente d = 0.85).
    *   **1.4. Probabilità di Random Hop (1-d):** Probabilità di passare a una pagina casuale (tipicamente 1-d = 0.15).
    *   **1.5. Formula Matematica:**
        $$PR(p_i) = \frac{1 - d}{N} + d \left( \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)} \right)$$
        *   *d*: fattore di smorzamento
        *   *1-d*: probabilità di *random hop*
        *   *N*: numero di pagine disponibili
        *   *M(p<sub>i</sub>)*: insieme delle pagine che collegano a *p<sub>i</sub>*
        *   *L(p<sub>j</sub>)*: numero di link nella pagina *p<sub>j</sub>*

**2. Tecniche di Sintesi di Testo**
    *   **2.1. Sintesi Estrattiva:**
        *   **2.1.1. Definizione:** Estrazione delle informazioni più importanti (es. frasi chiave) dal testo originale.
    *   **2.2. Sintesi Astrattiva:**
        *   **2.2.1. Definizione:** Comprensione del testo e generazione di un riassunto simile a quello umano, usando NLP (es. LLM).
    *   **2.3. Applicazioni:**
        *   **2.3.1.** Riepiloghi di notizie, motori di ricerca, e-learning, ricerca accademica.

**3. TextRank: Sintesi Estrattiva con PageRank**
    *   **3.1. Concetto:** Utilizzo di PageRank per identificare le frasi più importanti in un testo.
    *   **3.2. Rappresentazione del Testo:**
        *   **3.2.1.** Grafo pesato di frasi semanticamente connesse (G = <S,E>).
        *   **3.2.2.** Nodi (S): Frasi del testo (s<sub>1</sub>, …, s<sub>n</sub>).
        *   **3.2.3.** Archi (E): Connessioni tra frasi con peso (w<sub>i,j</sub>) che rappresenta la similarità.

**4. Implementazione di TextRank con GraphX**
    *   **4.1. Inizializzazione:** Sessione Spark e funzione di similarità tra frasi.
    *   **4.2. Creazione del Grafo:**
        *   **4.2.1.** Lettura del testo ed estrazione delle frasi.
        *   **4.2.2.** Creazione di un grafo completamente connesso con pesi basati sulla similarità.
    *   **4.3. Preparazione per PageRank:**
        *   **4.3.1.** Normalizzazione dei pesi degli archi (somma dei pesi uscenti = 1).
        *   **4.3.2.** Assegnazione di un PageRank iniziale a ogni nodo.
    *   **4.4. UDF Pregel:**
        *   **4.4.1.** `vertexProgram`: Aggiorna il PageRank di ogni nodo.
        *   **4.4.2.** `sendMessage`: Invia il PageRank ai vicini.
        *   **4.4.3.** `messageCombiner`: Aggrega i messaggi ricevuti.
    *   **4.5. Costruzione del Riassunto:** Selezione delle *k* frasi con il PageRank più alto, mantenendo l'ordine originale.

---

- **Costruzione del Grafo:** Il grafo viene creato a partire dai dati di input.
- **Preparazione del Grafo:** Il grafo viene preparato per l'esecuzione di Pregel.
- **Esecuzione di Pregel:** L'algoritmo Pregel viene eseguito sul grafo.
    - **Iterazioni:** Pregel viene eseguito per 50 iterazioni.
- **Generazione del Riassunto:** Un riassunto finale dei risultati viene generato utilizzando la funzione `buildSummary`.

---
