
Apache Storm è un sistema distribuito per l'elaborazione in tempo reale di flussi di dati illimitati, caratterizzato da alta scalabilità e bassa latenza.  A differenza di soluzioni precedenti, complesse e basate su code e worker, Storm semplifica lo sviluppo grazie ad un livello di astrazione intermedio.

Scritto in Clojure, con API in Java e supporto per altri linguaggi tramite protocollo multilingue (es. Python), Storm sfrutta il parallelismo dei dati e dei task.  Il suo paradigma si basa su cinque astrazioni principali:

* **Tuple:** Unità di base dei dati, un elenco di campi di vari tipi.
* **Stream:** Sequenza illimitata di tuple, elaborata in parallelo, con possibilità di usare serializer personalizzati.
* **Spout:** Sorgente dati, che legge da fonti esterne (es. social network, sensori, Kafka) e alimenta gli stream.
* **Bolt:** Unità di elaborazione che esegue operazioni sugli stream (es. pulizia dati, join).
* **Topology:** Grafo aciclico diretto (DAG) che rappresenta il job, con spout e bolt come nodi e stream come archi.  Viene eseguito indefinitamente.

Il raggruppamento degli stream tra i bolt avviene tramite diversi metodi:  `Shuffle grouping` (casuale), `Field grouping` (basato sul valore di un campo) e `Direct grouping` (determinato dal produttore).

Infine, Storm garantisce l'affidabilità dell'elaborazione tramite task di tracciamento che monitorano il flusso delle tuple e ne confermano la ricezione.

---

Storm è un sistema di elaborazione di stream in tempo reale distribuito, progettato per la tolleranza ai guasti e l'elaborazione ad alta velocità.  Utilizza un'architettura distribuita con un nodo Nimbus (master), molti Supervisor (worker manager) e ZooKeeper per la gestione dello stato del cluster.

**Architettura e Tolleranza ai Guasti:**  L'architettura è illustrata in `![|375](_page_7_Figure_2.jpeg)`.  Storm garantisce la tolleranza ai guasti attraverso il riavvio automatico dei worker da parte dei Supervisor, la riallocazione dei worker da parte di Nimbus in caso di fallimento, e l'utilizzo di ZooKeeper per la gestione dello stato senza stato di Nimbus e Supervisor.  Il fallimento di Nimbus o Supervisor non interrompe i worker in esecuzione, sebbene la riallocazione automatica potrebbe essere compromessa senza Nimbus.

**Flusso di Esecuzione:** Il flusso di un'applicazione Storm inizia con Nimbus che riceve la topologia, la distribuisce ai Supervisor, che a loro volta eseguono i task e inviano heartbeat a Nimbus.

**Programmazione:** Le applicazioni Storm sono composte da tre componenti principali:

* **Spout:** Genera tuple (dati) usando il metodo `nextTuple()`. Include metodi per l'inizializzazione (`open()`), la gestione dell'acknowledgement (`ack()`, `fail()`), e la dichiarazione dello schema di output (`declareOutputFields()`).
* **Bolt:** Processa le tuple ricevute usando il metodo `execute()`. Include metodi per la preparazione (`prepare()`), la pulizia (`cleanup()`) e la dichiarazione dello schema di output (`declareOutputFields()`).
* **Main:** Definisce la topologia, collegando spout e bolt usando `setSpout` e `setBolt`.

Un `Collector` ( `SpoutOutputCollector` per spout, `OutputCollector` per bolt) emette le tuple. Gli spout possono usare ID tuple per l'acknowledgement, implementando una semantica "almeno una volta". Trident, un'API di micro-batching su Storm, offre una semantica "exactly once".  Il meccanismo di acknowledgement utilizza un "ack val" (un intero a 64 bit) per tracciare il completamento dell'elaborazione delle tuple.

**Esempio:** Un esempio di applicazione conta le occorrenze di hashtag in un flusso di tweet, usando un `TweetSpout` per leggere i tweet e un `SplitHashtag` bolt per estrarre gli hashtag.

---

Il programma `HashtagCounter` conta le occorrenze di hashtag in un testo di input (non specificato nel testo originale).  Utilizza una struttura dati interna, probabilmente una mappa o un dizionario, per memorizzare gli hashtag e il loro rispettivo conteggio.  Infine, stampa il conteggio di ogni hashtag.  Non sono fornite ulteriori informazioni sull'implementazione o sul metodo di input.

---
