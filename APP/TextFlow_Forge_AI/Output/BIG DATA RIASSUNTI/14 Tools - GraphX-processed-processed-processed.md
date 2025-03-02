
# Riassunto: Calcolo Grafo-Parallelo Efficiente con Spark GraphX

L'analisi di grafi, strutture dati che rappresentano relazioni tra oggetti, è fondamentale in diversi ambiti (social network, sistemi di raccomandazione, ecc.).  Con l'aumento dei dati, gli strumenti tradizionali diventano inefficienti, richiedendo soluzioni *ad hoc* per il calcolo grafo-parallelo.

Il modello **Bulk Synchronous Parallel (BSP)**, composto da elementi di elaborazione (PE), un router e un sincronizzatore, offre un framework per algoritmi paralleli iterativi su larga scala. Il calcolo BSP procede tramite **supersteps**, fasi sequenziali di calcolo concorrente, comunicazione globale e sincronizzazione a barriera.  ![[]](_page_7_Figure_1.jpeg) ![[|384](_page_7_Figure_9.jpeg)] ![[|468](_page_8_Figure_10.jpeg)]

**Pregel**, sviluppato da Google, è un esempio di framework basato su BSP e su programmazione *vertex-centric*, dove le funzioni operano su singoli vertici e archi.  Sebbene Pregel non sia open source, esistono alternative come Giraph, Gelly e **GraphX**.

**Apache Spark GraphX** è una libreria per l'elaborazione efficiente di grafi su larga scala.  Le sue caratteristiche principali includono:

* **Grafi Distribuiti Resilienti (RDG):** estensione degli RDD di Spark per la partizione efficiente dei dati del grafo.
* **Algoritmi Grafo:** implementazioni integrate di algoritmi come PageRank e componenti connesse.
* **Operatori Grafo:** operatori per la manipolazione di grafi (mappatura, creazione di sottografi, ecc.).
* **Integrazione con Spark:** perfetta integrazione con altri componenti Spark.

GraphX rappresenta i grafi come **Grafi Distribuiti Resilienti**, multigrafi diretti con proprietà associate a vertici e archi.

---

GraphX fornisce un'interfaccia unificata per l'elaborazione di grafi distribuiti su Spark, sfruttando l'efficienza degli RDD.  Rappresenta un grafo tramite due RDD distinti: uno per i vertici (tuple `(id vertice, attributo)`) e uno per gli archi (tuple `(id sorgente, id destinazione, attributo)`).  L'estensione `EdgeTriplet` arricchisce ulteriormente la rappresentazione includendo gli attributi di entrambi i vertici coinvolti in un arco (`(id sorgente, id destinazione, attributo sorgente, attributo arco, attributo destinazione)`).

I grafi in GraphX sono partizionati con un approccio *vertex-cut* tramite l'operatore `Graph.partitionBy`, ottimizzando la comunicazione e l'archiviazione distribuita.  L'API offre operatori base per accedere a informazioni topologiche (numero di vertici ed archi, gradi in entrata e in uscita) e per manipolare il grafo come collezioni (`vertices`, `edges`, `triplets`).  Sono disponibili funzioni di mappatura (`mapVertices`, `mapEdges`, `mapTriplets`) per trasformare gli attributi e funzioni di join (`joinVertices`, `outerJoinVertices`) per integrare dati esterni.

GraphX implementa il modello di calcolo vertex-centrico Pregel per elaborazioni parallele su larga scala.  Il processo iterativo (superstep) invoca una UDF (`vprog: (VertexId, VD, A) => VD`) su ogni vertice che riceve un messaggio, aggiornando il suo stato.  A differenza del Pregel standard, GraphX calcola i messaggi in parallelo tramite una funzione sulla tripla arco (`sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexId, A)]`) e i vertici inviano messaggi solo ai vicini.  Una funzione di merge (`mergeMsg: (A, A) => A`) gestisce la combinazione di messaggi multipli ricevuti da uno stesso vertice.  L'operatore `Pregel` richiede come input il messaggio iniziale, il numero di iterazioni, la direzione degli archi e le tre UDF: `vprog`, `sendMsg` e `mergeMsg`.

---

Questo documento descrive l'algoritmo TextRank, un metodo di sintesi estrattiva basato sull'algoritmo PageRank.

**PageRank:**  PageRank assegna un punteggio di importanza (*PageRank*) a ciascuna pagina web in base al numero e all'importanza dei link in entrata.  Un utente naviga tra le pagine seguendo i link con probabilità *d* (generalmente 0.85), oppure effettua un "random hop" con probabilità 1-*d*, passando a una pagina casuale. Il PageRank di una pagina *p<sub>i</sub>*,  `PR(p<sub>i</sub>)`, è calcolato con la seguente formula:

$$PR(p_i) = \frac{1 - d}{N} + d \left( \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)} \right)$$

dove *d* è il fattore di smorzamento, *N* il numero totale di pagine, *M(p<sub>i</sub>)* l'insieme delle pagine che linkano a *p<sub>i</sub>*, e *L(p<sub>j</sub>)* il numero di link in uscita da *p<sub>j</sub>*.

**TextRank:**  TextRank applica PageRank alla sintesi estrattiva.  Un testo viene rappresentato come un grafo dove i nodi sono le frasi e il peso degli archi rappresenta la similarità semantica tra le frasi.  L'algoritmo calcola il PageRank di ogni frase e seleziona le *k* frasi con il punteggio più alto per formare il riassunto, mantenendo l'ordine originale.

**Implementazione con GraphX:** L'implementazione con GraphX prevede:

1. **Inizializzazione:** Creazione di una sessione Spark e definizione di una funzione per calcolare la similarità tra frasi.
2. **Creazione del grafo:** Costruzione di un grafo completamente connesso rappresentante il testo.
3. **Preparazione per PageRank:** Normalizzazione dei pesi degli archi e assegnazione di un PageRank iniziale a ogni nodo.
4. **UDF Pregel:** Definizione di una UDF Pregel con le funzioni `vertexProgram`, `sendMessage`, e `messageCombiner` per l'iterazione di PageRank.
5. **Costruzione del riassunto:** Selezione delle *k* frasi con il PageRank più alto.


L'algoritmo TextRank sfrutta la commutatività e l'associatività della UDF utilizzata nel calcolo del PageRank.  Il risultato finale è un riassunto del testo di input.

---

Il processo di esecuzione prevede quattro fasi principali:

1. **Costruzione del grafo:** Viene creato il grafo necessario per l'elaborazione.

2. **Preparazione del grafo:** Il grafo viene preparato per l'esecuzione dell'algoritmo Pregel.  Questa fase potrebbe includere operazioni di pre-processing come ottimizzazione della struttura dati.

3. **Esecuzione di Pregel:** L'algoritmo Pregel viene eseguito per 50 iterazioni sul grafo preparato.  Questa è la fase di elaborazione principale.

4. **Generazione del riassunto:**  Una volta completate le 50 iterazioni di Pregel, la funzione `buildSummary` genera un riassunto finale dei risultati dell'elaborazione.

---
