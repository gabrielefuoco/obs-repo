
Apache Storm è un sistema distribuito di calcolo in tempo reale, altamente scalabile e facile da usare, progettato per l'elaborazione affidabile di flussi di dati illimitati con bassa latenza. Prima di Storm, l'elaborazione in tempo reale richiedeva lo sviluppo di soluzioni complesse basate su code e worker, con una significativa parte di logica dedicata alla gestione dei messaggi e alla garanzia della disponibilità dei componenti. Storm semplifica notevolmente questo processo.

### Livello di Astrazione e Linguaggi Supportati

Storm fornisce un livello medio di astrazione, permettendo ai programmatori di definire applicazioni usando astrazioni di base (spout, stream, bolt e topologie) e testarle localmente prima della distribuzione su un cluster. Scritto in Clojure (un dialetto di Lisp), offre API anche in Java e supporta altri linguaggi tramite un protocollo multilingue, incluso Python. Il suo runtime supporta sia il parallelismo dei dati (molti thread eseguono lo stesso codice su chunk diversi) che il parallelismo dei task (diversi spout e bolt eseguono in parallelo).

### Astrazioni di Dati e Calcolo

Il paradigma di programmazione di Storm si basa su cinque astrazioni principali:

1. **Tuple:** Unità di base dei dati, costituita da un elenco di campi di vari tipi (byte, char, integer, long, ecc.).
2. **Stream:** Sequenza illimitata di tuple, creata ed elaborata in parallelo. Gli stream possono essere creati usando serializer standard o personalizzati.
3. **Spout:** Sorgente dati di uno stream. Legge dati da sorgenti esterne (API di social network, reti di sensori, sistemi di messaggistica come JMS, Kafka, Redis) e li inserisce nell'applicazione.
4. **Bolt:** Entità di elaborazione che esegue task o algoritmi (pulizia dati, join, query, ecc.).
5. **Topology:** Rappresenta un job, configurato come un grafo aciclico diretto (DAG). Spout e bolt sono i vertici, gli stream gli archi. Viene eseguita indefinitamente fino all'arresto.

### Stream e Raggruppamento

Gli stream sono creati ed elaborati in parallelo. Lo spout crea stream, il bolt li riceve in input e può generarne di nuovi in output. Il raggruppamento degli stream definisce come gli stream vengono partizionati tra i bolt:

* **Shuffle grouping:** Le tuple sono divise casualmente tra i bolt.
* **Field grouping:** Le tuple sono partizionate in base al valore di un campo specifico.
* **Direct grouping:** Il produttore della tuple determina il task del consumatore che la riceverà.

### Elaborazione dei Messaggi e Affidabilità

Storm garantisce l'elaborazione affidabile dei messaggi usando task di tracciamento ("tracking tasks") che tracciano il DAG delle tuple generate dagli spout e confermano la ricezione. Ogni tracker usa una mappa che collega un ID tuple spout a un "ack val" (un intero a 64 bit ottenuto dall'XOR degli ID tuple nell'albero). Quando l'ack val diventa zero, l'albero delle tuple è completo. Questo meccanismo ("almeno una volta") garantisce che tutti i messaggi vengano elaborati, anche se alcuni potrebbero essere elaborati più volte in caso di guasti.
- Trident è un'API di micro-batching su Storm che offre una semantica di elaborazione "exactly once".

## Architettura

![|375](_page_7_Figure_2.jpeg)

## Tolleranza ai Guasti

Storm è progettato per la tolleranza ai guasti:

* I worker vengono riavviati automaticamente dai Supervisor in caso di blocco.
* Se il riavvio fallisce, Nimbus riassegna il worker a un'altra macchina.
* Nimbus e Supervisor sono senza stato (stateless), con lo stato conservato in ZooKeeper.
* Il fallimento di Nimbus o Supervisor non influenza i worker in esecuzione (anche se la riallocazione automatica dei worker in caso di necessità non è più garantita senza Nimbus).
* ZooKeeper coordina e condivide le informazioni di configurazione, memorizzando gli stati del cluster e dei task.

## Flusso di Esecuzione

Un cluster Storm ha un nodo Nimbus, molti Supervisor e un'unica istanza di ZooKeeper. Il flusso di un'applicazione Storm è:

1. Nimbus riceve la topologia.
2. Nimbus distribuisce i task tra i Supervisor.
3. I Supervisor inviano heartbeat a Nimbus.
4. I Supervisor eseguono i task e attendono nuovi task da Nimbus.
5. Nimbus attende nuove topologie.

## Basi di Programmazione

Un programma Storm tipico ha tre componenti:

* **Spout:** Implementa `IRichSpout`, con il metodo `nextTuple()` per emettere tuple.
* **Bolt:** Implementa `IRichBolt`, con il metodo `execute()` per elaborare le tuple.
* **Main:** Definisce la topologia usando `setSpout` e `setBolt`.

### Spout

La classe `Spout` include i metodi:

* `open()`: Inizializzazione del task.
* `nextTuple()`: Emissione di tuple (non bloccante).
* `ack()` e `fail()`: Gestione del completamento o del fallimento delle tuple.
* `declareOutputFields()`: Dichiarazione dello schema di output.

### Bolt

La classe `Bolt` include i metodi:

* `prepare()`: Riceve un output collector.
* `execute()`: Elabora le tuple.
* `cleanup()`: Pulizia delle risorse.
* `declareOutputFields()`: Dichiarazione dello schema di output.

### Collector

L'oggetto `collector` ( `SpoutOutputCollector` per spout, `OutputCollector` per bolt) emette tuple. Gli spout possono contrassegnare i messaggi con ID per l'acknowledgment o il fallimento.

### Esempio: Conteggio di Hashtag da Tweet

Un esempio analizza un flusso di tweet per contare le occorrenze di hashtag:

* **TweetSpout:** Legge i tweet dalle API di Twitter.
* **SplitHashtag:** Estrae gli hashtag dai tweet.
* **HashtagCounter:** Conta le occorrenze degli hashtag usando una mappa interna e stampa il conteggio.
