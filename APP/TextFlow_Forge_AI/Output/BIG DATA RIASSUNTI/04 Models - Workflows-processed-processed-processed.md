
# Modelli di Programmazione: Workflow

Un workflow è una sequenza di attività (task) interconnesse, eseguite per raggiungere un obiettivo.  La Workflow Management Coalition (WMC) lo definisce come l'automazione di un processo aziendale, con passaggio di documenti/informazioni tra partecipanti secondo regole procedurali.  I workflow sono modelli di programmazione utili per gestire processi complessi su piattaforme distribuite, integrando analisi dati, calcolo scientifico e simulazione.  Un processo è un insieme di task correlati per produrre un risultato o servizio, mentre un task è una singola unità di lavoro.  I workflow, con approccio dichiarativo, nascondono dettagli di basso livello, permettendo la memorizzazione, il recupero e il riutilizzo di pattern comuni.  Un Sistema di Gestione dei Workflow (WMS) facilita la definizione, lo sviluppo e l'esecuzione, coordinando le attività.

Un workflow è rappresentato da un grafo: i vertici sono i task, gli archi il flusso di esecuzione tra essi.  L'implementazione avviene tramite software, linguaggi di programmazione e librerie che supportano l'orchestrazione.

## Pattern dei Workflow

I task possono essere combinati in diversi pattern:

* **Sequenza:** Esecuzione di task in ordine specifico (Fig. 9).  ![[]]
* **Ramificazione:**  Il flusso si divide in base a condizioni (Fig. 4). ![[ ]]  Esistono tre varianti:
    * **AND-split:** Esecuzione concorrente di tutte le diramazioni.
    * **XOR-split:** Esecuzione di una sola diramazione, scelta in base a condizioni.
    * **OR-split:** Esecuzione di una o più diramazioni, scelta in base a condizioni.
* **Sincronizzazione:** (non dettagliata nel testo fornito)
* **Ripetizione:** (non dettagliata nel testo fornito)


Le figure (Fig. 4 e Fig. 9) illustrate graficamente i pattern di sequenza e ramificazione.  ![[]] ![[ ]]

---

### Pattern di Sincronizzazione

I pattern di sincronizzazione gestiscono la congiunzione di flussi di controllo multipli in un singolo flusso.  Tre varianti principali sono:

* **AND-join:** tutti i rami devono completarsi prima di procedere.
* **XOR-join:** solo un ramo deve completarsi.
* **OR-join:** almeno un ramo deve completarsi.

![[]]


### Pattern di Ripetizione

I pattern di ripetizione definiscono la ripetizione di task:

* **Ciclo Arbitrario:** ripetizione non strutturata (simile a `goto`).
* **Ciclo Strutturato:** ripetizione con condizione di terminazione (es. `while`, `repeat…until`).
* **Ricorsione:** ripetizione tramite auto-invocazione.


### Grafi Aciclici Diretti (DAG)

Un DAG è un workflow diretto e aciclico (senza cicli).  È una struttura comune per la gestione di workflow, particolarmente utile in ambito big data (es. Apache Spark).  I DAG modellano efficacemente processi complessi, come quelli di data mining, su sistemi distribuiti.  Le dipendenze nei DAG possono essere:

* **Dipendenze dati:** l'output di un task è l'input di un altro.
* **Dipendenze di controllo:** un task deve completarsi prima che un altro inizi.

Le dipendenze possono essere definite esplicitamente (tramite istruzioni) o implicitamente (dedotte dal sistema analizzando le relazioni input-output).  Il modello DAG generalizza MapReduce, offrendo maggiore flessibilità e ottimizzazione.


### Grafi Ciclici Diretti (DCG)

I DCG sono workflow più complessi contenenti cicli, che rappresentano meccanismi di iterazione.  I nodi rappresentano servizi o componenti software, mentre gli archi rappresentano messaggi o flussi di dati tra essi.

---
