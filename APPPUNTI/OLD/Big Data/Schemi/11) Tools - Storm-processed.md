
**Schema Riassuntivo di Apache Storm**

**I. Introduzione a Apache Storm**
    *   Sistema distribuito per calcolo in tempo reale.
    *   Alta scalabilità e facilità d'uso.
    *   Elaborazione affidabile di flussi di dati illimitati con bassa latenza.
    *   Semplifica lo sviluppo di soluzioni di elaborazione in tempo reale rispetto a soluzioni basate su code e worker.

**II. Livello di Astrazione e Supporto Linguistico**
    *   Fornisce un livello medio di astrazione per la definizione di applicazioni.
    *   Permette il test locale delle applicazioni prima della distribuzione.
    *   Scritto in Clojure, con API in Java e supporto multilingue (Python incluso).
    *   Supporta sia il parallelismo dei dati che il parallelismo dei task.

**III. Astrazioni di Dati e Calcolo**
    *   **Tuple:** Unità di base dei dati (lista di campi di vari tipi).
    *   **Stream:** Sequenza illimitata di tuple, creata ed elaborata in parallelo.
    *   **Spout:** Sorgente dati di uno stream (legge da sorgenti esterne e inserisce dati).
    *   **Bolt:** Entità di elaborazione (esegue task/algoritmi).
    *   **Topology:** Rappresenta un job come un grafo aciclico diretto (DAG), eseguito indefinitamente.

**IV. Stream e Raggruppamento**
    *   Gli stream sono creati ed elaborati in parallelo.
    *   Spout crea stream, Bolt li riceve e può generarne di nuovi.
    *   Raggruppamento degli stream:
        *   **Shuffle grouping:** Tuple divise casualmente tra i bolt.
        *   **Field grouping:** Tuple partizionate in base al valore di un campo specifico.
        *   **Direct grouping:** Il produttore della tuple determina il task del consumatore.

**V. Elaborazione dei Messaggi e Affidabilità**
    *   Garantisce l'elaborazione affidabile dei messaggi.
    *   Utilizza task di tracciamento ("tracking tasks") per tracciare il DAG delle tuple.
    *   Conferma la ricezione delle tuple.

---

**Schema Riassuntivo di Storm**

**1. Architettura e Tolleranza ai Guasti**

*   **1.1. Tolleranza ai Guasti Integrata**
    *   I worker vengono riavviati automaticamente dai Supervisor in caso di blocco.
    *   Se il riavvio fallisce, Nimbus riassegna il worker.
    *   Nimbus e Supervisor sono stateless, con lo stato in ZooKeeper.
    *   Il fallimento di Nimbus/Supervisor non influenza i worker in esecuzione (ma la riallocazione automatica non è garantita senza Nimbus).
*   **1.2. Ruolo di ZooKeeper**
    *   Coordina e condivide le informazioni di configurazione.
    *   Memorizza gli stati del cluster e dei task.

**2. Flusso di Esecuzione**

*   **2.1. Componenti del Cluster**
    *   Un nodo Nimbus.
    *   Molti Supervisor.
    *   Un'istanza di ZooKeeper.
*   **2.2. Fasi dell'Esecuzione**
    *   Nimbus riceve la topologia.
    *   Nimbus distribuisce i task ai Supervisor.
    *   I Supervisor inviano heartbeat a Nimbus.
    *   I Supervisor eseguono i task e attendono nuovi task.
    *   Nimbus attende nuove topologie.

**3. Basi di Programmazione**

*   **3.1. Componenti Principali**
    *   **Spout:** Implementa `IRichSpout`, emette tuple con `nextTuple()`.
    *   **Bolt:** Implementa `IRichBolt`, elabora tuple con `execute()`.
    *   **Main:** Definisce la topologia usando `setSpout` e `setBolt`.
*   **3.2. Spout (IRichSpout)**
    *   `open()`: Inizializzazione del task.
    *   `nextTuple()`: Emissione di tuple (non bloccante).
    *   `ack()` e `fail()`: Gestione del completamento o del fallimento delle tuple.
    *   `declareOutputFields()`: Dichiarazione dello schema di output.
*   **3.3. Bolt (IRichBolt)**
    *   `prepare()`: Riceve un output collector.
    *   `execute()`: Elabora le tuple.
    *   `cleanup()`: Pulizia delle risorse.
    *   `declareOutputFields()`: Dichiarazione dello schema di output.
*   **3.4. Collector**
    *   `SpoutOutputCollector` (per spout), `OutputCollector` (per bolt).
    *   Emette tuple.
    *   Gli spout possono contrassegnare i messaggi con ID per l'acknowledgment o il fallimento.

**4. Meccanismo di Acknowledgment e Trident**

*   **4.1. Acknowledgment**
    *   Ogni tracker usa una mappa che collega un ID tuple spout a un "ack val".
    *   "ack val" è un intero a 64 bit ottenuto dall'XOR degli ID tuple nell'albero.
    *   Quando l'ack val diventa zero, l'albero delle tuple è completo.
    *   Meccanismo "almeno una volta".
*   **4.2. Trident**
    *   API di micro-batching su Storm.
    *   Offre una semantica di elaborazione "exactly once".

**5. Esempio: Conteggio di Hashtag da Tweet**

*   **5.1. Componenti**
    *   **TweetSpout:** Legge i tweet dalle API di Twitter.
    *   **SplitHashtag:** Estrae gli hashtag dai tweet.

---

## Schema Riassuntivo HashtagCounter

**I. Funzionalità Principale:**

*   Conta le occorrenze degli hashtag.

**II. Implementazione:**

*   Utilizza una mappa interna per memorizzare i conteggi.

**III. Output:**

*   Stampa il conteggio degli hashtag.

---
