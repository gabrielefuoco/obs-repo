
## Il modello SQL-like

I database relazionali non possono scalare orizzontalmente su più computer, rendendo difficile archiviare e gestire l'enorme volume di dati generato quotidianamente da numerose applicazioni. L'approccio **NoSQL** (*not only SQL*) ha guadagnato popolarità come alternativa non relazionale, consentendo la scalabilità orizzontale per le operazioni di base di lettura/scrittura del database su più server.

Contrariamente ai database relazionali che seguono il modello ACID (Atomicità, Coerenza, Isolamento, Durabilità), i database NoSQL generalmente aderiscono al modello **BASE** (Basic Availability, Soft state, Eventual consistency). BASE, a differenza di ACID, priorizza la disponibilità di base, uno stato soft e la coerenza eventuale, eliminando la necessità di coerenza immediata dopo ogni transazione e consentendo l'elaborazione concorrente di più istanze su server diversi.

I database NoSQL sono spesso inadatti all'analisi dei dati. I sistemi SQL-like mirano a combinare l'efficienza della programmazione MapReduce con la semplicità di un linguaggio simile a SQL. MapReduce risolve i problemi di scalabilità e riduce i tempi di query, ma la sua complessità può rappresentare una sfida per gli utenti meno esperti. I sistemi SQL-like superano queste complessità per operazioni semplici (aggregazioni di righe, selezioni o conteggi), mantenendo la velocità e la scalabilità delle query. In molti casi, ottimizzano automaticamente le query su repository di grandi dimensioni utilizzando MapReduce in background.

Soluzioni come **Apache Hive** sono state sviluppate per migliorare le capacità di query dei sistemi MapReduce, semplificando lo sviluppo di applicazioni di base per l'analisi dei dati utilizzando un linguaggio simile a SQL.

![[Pasted image 20250223163240.png|337]]

## Perché utilizzare SQL con i big data?

SQL è lo strumento preferito per sviluppatori, amministratori di database e data scientist, ampiamente utilizzato nei prodotti commerciali per l'interrogazione, la modifica e la visualizzazione dei dati. I suoi principali vantaggi includono:

* **Linguaggio dichiarativo:** SQL è dichiarativo, descrive le trasformazioni e le operazioni sui dati, rendendolo facilmente comprensibile.
* **Interoperabilità:** Essendo standardizzato, consente a diversi sistemi di fornire le proprie implementazioni garantendo compatibilità e una sintassi facilmente comprensibile.
* **Data-driven:** Le operazioni SQL riflettono le trasformazioni e le modifiche dei dataset di input, rendendolo un modello di programmazione conveniente per applicazioni incentrate sui dati in ambienti tradizionali e big data.

I sistemi SQL-like per i big data supportano la tecnica *query-in-place*, eseguendo le query direttamente sui dati nella loro posizione originale senza spostamento o trasformazione in un database analitico separato. La *query-in-place* offre un cambio di paradigma nell'analisi dei big data, fornendo un mezzo potente, efficiente ed economico per estrarre informazioni utili direttamente da dataset massicci. Questa tecnica preserva l'integrità dei dati originali e offre vantaggi chiave come:

* Ottimizzazione dell'utilizzo dei dati, eliminazione di processi ridondanti e riduzione della complessità associata allo spostamento dei dati.
* Minore latenza per l'esecuzione di query SQL ad hoc su dataset massicci, offrendo disponibilità immediata dei dati e riducendo i costi operativi.

### Partizionamento dei dati per l'interrogazione di big data

Il **partizionamento dei dati** è fondamentale per interrogare efficientemente i big data utilizzando SQL. Questo processo divide i dati di una tabella in base a specifici valori di colonna, creando file e/o directory distinti. Una strategia di partizionamento riduce i costi di I/O evitando letture di dati non necessarie e accelerando significativamente l'elaborazione delle query.

## Modelli di Programmazione Parallela: PGAS

## Partizionamento eccessivo e PGAS

Il partizionamento eccessivo può portare a un numero elevato di file e directory, aumentando i costi per il nodo master, che deve mantenere tutti i metadati in memoria.

Il **Partitioned Global Address Space (PGAS)** è un modello di programmazione parallela progettato per aumentare la produttività del programmatore mantenendo elevate prestazioni. L'idea principale è quella di utilizzare uno **spazio di indirizzamento globale condiviso**, migliorando la produttività e implementando contemporaneamente una separazione tra accessi ai dati *locali* e *remoti*. Questa separazione è fondamentale per ottenere miglioramenti delle prestazioni e garantire la scalabilità su architetture parallele su larga scala.

In un modello PGAS, un programma include più processi che eseguono contemporaneamente lo stesso codice su nodi diversi. Ogni processo ha un **rank**, corrispondente all'indice del nodo su cui viene eseguito. Questi processi hanno accesso a uno spazio di indirizzamento globale **partizionato** in spazi di indirizzamento locali. Un indirizzo locale è direttamente accessibile da un processo, mentre gli indirizzi remoti di altri processi richiedono chiamate API per l'accesso.

![[Pasted image 20250223163311.png|480]]

I linguaggi PGAS considerano lo spazio di indirizzamento come un **ambiente globale**. Un thread o un processo può ottenere un puntatore ai dati situati ovunque nel sistema e può leggere o scrivere dati remoti locali ad altri thread. I linguaggi PGAS distinguono tra **memoria condivisa** (accessibile a tutti i thread) e **memoria privata** (accessibile solo al thread proprietario). Ogni thread ha la sua porzione di spazio privato e una sezione di spazio condiviso.

### Parallelismo nel modello PGAS

Il modello PGAS delinea tre principali modelli di esecuzione parallela:

- **Single Program Multiple Data (SPMD):** un numero predeterminato di thread viene avviato all'avvio del programma e ognuno esegue lo stesso programma.
- **Asynchronous PGAS (APGAS):** all'avvio del programma, un singolo thread avvia l'esecuzione. Successivamente, è possibile generare dinamicamente nuovi thread per operare all'interno delle stesse o di diverse partizioni dello spazio di indirizzamento remoto. Ogni thread generato può eseguire un codice diverso.
- **Parallelismo Implicito:** non c'è parallelismo visibile nel codice, e il programma sembra descrivere un singolo thread di controllo. Tuttavia, è possibile generare dinamicamente più thread di controllo durante l'esecuzione per accelerare il calcolo.

### Memoria e funzione di costo nel modello PGAS

Il modello PGAS suddivide lo spazio di memoria in *places*, dove ogni *place* rappresenta un nodo di calcolo associato a un processo o thread specifico. Un thread può accedere alle posizioni di memoria all'interno del suo *place* a un costo basso e uniforme, mentre l'accesso alle memorie in altri *places* comporta costi maggiori. Il modello PGAS utilizza il modello di memoria condivisa **Non-Uniform Memory Access (NUMA)** attraverso la sua funzione di costo, che definisce gli accessi alla memoria. In genere, i linguaggi PGAS incorporano una struttura di costo a due livelli: **economico** e **costoso**. Le posizioni di memoria vicine all'origine della richiesta di accesso sono considerate economiche, mentre le posizioni di memoria distanti sono considerate costose. Due fattori contribuiscono al calcolo del costo: il *place* di origine della richiesta di accesso e il *place* in cui si trovano i dati richiesti.

### Distribuzione dei dati nel modello PGAS

I linguaggi PGAS possono essere classificati in base a come i dati vengono distribuiti tra i *places*. Tre modelli di distribuzione comuni sono:

* **Ciclico:** i dati vengono suddivisi in blocchi consecutivi disposti ciclicamente tra i *places*.

### Metodi di Partizionamento dei Dati

Due metodi principali per il partizionamento dei dati sono:

- **Block:** I dati vengono suddivisi in blocchi di dimensione uguale e consecutiva, distribuiti tra diverse posizioni (o nodi).
- **Block-cyclic:** I dati vengono suddivisi in blocchi di dimensione parametrizzabile, disposti sequenzialmente tra le diverse posizioni in modo ciclico.

