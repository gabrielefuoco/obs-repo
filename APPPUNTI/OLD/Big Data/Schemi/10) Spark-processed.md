
## Schema Riassuntivo: MapReduce, Spark e Confronto

**I. MapReduce: Debolezze e Limitazioni**

    *   **A. Modello di Programmazione Restrittivo:**
        *   Difficoltà nell'implementazione di algoritmi complessi direttamente.
        *   Necessità di passaggi MapReduce multipli anche per operazioni semplici (es. WordCount con ordinamento).
        *   Mancanza di controllo a basso livello su strutture e tipi di dati.

    *   **B. Supporto Inefficiente per l'Iterazione:**
        *   Ogni iterazione implica scrittura/lettura da disco, causando overhead.
        *   Necessità di minimizzare il numero di iterazioni nella progettazione degli algoritmi.

    *   **C. Efficienza Limitata e HDFS:**
        *   Alto costo di comunicazione (map, shuffle, reduce).
        *   Scrittura frequente su disco limita le prestazioni.
        *   Sfruttamento limitato della memoria principale.
        *   Non adatto all'elaborazione di flussi di dati in tempo reale (richiede scansione completa dell'input).

**II. Apache Spark: Panoramica**

    *   **A. Motore Veloce e Versatile per Big Data:**
        *   Piattaforma leader per SQL su larga scala, elaborazione batch, streaming e machine learning.
        *   Motore analitico unificato per l'elaborazione di dati su larga scala.

    *   **B. Caratteristiche Chiave:**
        *   **Archiviazione in memoria:** Elaborazione iterativa rapida (almeno 10 volte più veloce di Hadoop).
        *   **Ottimizzazioni:** Supporta grafi di esecuzione generali e potenti ottimizzazioni.
        *   **Compatibilità Hadoop:** Compatibile con le API di storage di Hadoop (HDFS, HBase).

    *   **C. Condivisione di Dati:**
        *   **MapReduce:** Lenta a causa di replicazione, serializzazione e I/O su disco.
        *   **Spark:** Memoria distribuita: 10-100 volte più veloce rispetto a disco e rete.

**III. Spark vs Hadoop MapReduce: Confronto**

    *   **A. Paradigma di Programmazione Simile ("scatter-gather"):**
        *   Spark offre vantaggi significativi.

    *   **B. Vantaggi di Spark:**
        *   **Modello di dati più generale:** RDD, DataSet e DataFrame.
        *   **Modello di programmazione più user-friendly:** Trasformazioni (simili a map), azioni (simili a reduce).
        *   **Agnosticità rispetto allo storage:** Supporta HDFS, Cassandra, S3, file Parquet, ecc.

**IV. Stack Spark**

    *   **A. Spark Core:**
        *   Funzionalità di base: pianificazione dei task, gestione della memoria, ripristino da guasti, interazione con sistemi di storage.
        *   Concetto chiave: **Resilient Distributed Dataset (RDD)** - raccolta di elementi distribuiti manipolabili in parallelo.
        *   API per Java, Python, R e Scala.

    *   **B. Moduli di Livello Superiore:**
        *   **Spark SQL:** Dati strutturati, interrogazioni SQL, supporta diverse sorgenti dati (tabelle Hive, Parquet, JSON, ecc.), estende l'API Spark RDD.
        *   **Spark Streaming:** Elaborazione di flussi di dati in tempo reale, estende l'API Spark RDD.
        *   **MLlib:** Libreria di machine learning scalabile (estrazione di feature, classificazione, regressione, clustering, raccomandazione).
        *   **GraphX:** API per manipolazione di grafi e calcoli paralleli (PageRank), estende l'API Spark RDD.

**V. Spark su Gestori di Cluster**

    *   **A. Modalità Standalone:**
        *   Utilizza un semplice scheduler FIFO incluso in Spark.

---

**Schema Riassuntivo di Spark**

**I. Architettura Spark**
    *   **A. Componenti Principali:**
        *   1.  **Programma Driver:**
            *   a. Esegue la funzione `main()`.
            *   b. Crea l'oggetto `SparkContext`.
        *   2.  **Executor:**
            *   a. Processi attivi per la durata dell'applicazione.
            *   b. Eseguono task in thread multipli.
            *   c. Garantiscono l'isolamento tra applicazioni.
        *   3.  **SparkContext:**
            *   a. Si connette al gestore di cluster per l'allocazione risorse.
            *   b. Acquisisce gli executor.
            *   c. Invia il codice dell'applicazione agli executor.
            *   d. Invia i task agli executor per l'esecuzione.
    *   **B. Gestori di Cluster (Esempi):**
        *   1.  Hadoop YARN
        *   2.  Mesos (AMP Lab @ UC Berkeley)
        *   3.  Kubernetes

**II. Resilient Distributed Dataset (RDD)**
    *   **A. Definizione:**
        *   1.  Astrazione di programmazione chiave in Spark.
        *   2.  Struttura dati immutabile, distribuita, partizionata e fault-tolerant.
        *   3.  Memorizzata in memoria principale (se disponibile) o su disco locale.
    *   **B. Caratteristiche Principali:**
        *   1.  **Immutabile:** Non modificabile dopo la creazione.
        *   2.  **In Memoria (esplicitamente):** Dati risiedono principalmente in memoria.
        *   3.  **Fault-tolerant:** Tolleranza ai guasti garantita (ricostruzione automatica).
        *   4.  **Struttura dati distribuita:** Dati distribuiti tra i nodi del cluster.
        *   5.  **Partizionamento controllato:** Ottimizza il posizionamento dei dati.
        *   6.  **Ricco set di operatori:** Manipolazione tramite un ricco set di operatori.
    *   **C. Distribuzione e Partizionamento:**
        *   1.  Memorizzati nella memoria principale degli executor (nodi worker).
        *   2.  Esecuzione parallela del codice su ogni partizione.
        *   3.  **Partizione:** Frammento atomico di dati, divisione logica, unità base di parallelismo.
        *   4.  Partizioni possono risiedere su nodi cluster diversi.
    *   **D. Immutabilità e Tolleranza ai Guasti:**
        *   1.  Immutabili: Modifiche richiedono la creazione di un nuovo RDD.
        *   2.  Fault-tolerant: Ricostruiti automaticamente in caso di errore.
        *   3.  **Lineage:** Informazioni che tracciano la storia della creazione dell'RDD.
        *   4.  Spark utilizza la lineage per ricalcolare i dati persi.
    *   **E. Gestione degli RDD:**
        *   1.  Spark gestisce la suddivisione in partizioni e l'allocazione ai nodi.
        *   2.  Gestisce la tolleranza agli errori (ricostruzione automatica).
        *   3.  Sfrutta il DAG (Directed Acyclic Graph) di lineage (piano di esecuzione logica).

**III. API RDD**
    *   **A. Linguaggi Supportati:** Scala, Python, Java, R.
    *   **B. Modalità d'Uso:** Interattiva (console Scala, PySpark).
    *   **C. API di Livello Superiore:** DataFrames, Datasets.

**IV. Idoneità RDD**
    *   **A. Adatto per:** Applicazioni batch (stessa operazione su tutti gli elementi).
    *   **B. Meno Adatto per:** Aggiornamenti asincroni a grana fine su stato condiviso.

**V. Operazioni nell'API RDD**
    *   **A. Creazione:** Da dati esterni o da altri RDD.
    *   **B. Trasformazioni:**
        *   1.  Operazioni a grana grossa che definiscono nuovi dataset.
        *   2.  Esempi: `map`, `filter`, `join`.
        *   3.  *Lazy*: Calcolo eseguito solo quando richiesto da un'azione.

---

**Schema Riassuntivo del Modello di Programmazione Spark**

1.  **Modello di Programmazione Spark**
    *   Si basa su operatori parallelizzabili (funzioni di ordine superiore).
    *   Esegue funzioni definite dall'utente in parallelo.
    *   Flusso di dati rappresentato da un DAG (Directed Acyclic Graph).
        *   Il DAG collega sorgenti di dati, operatori e sink di dati.

2.  **Funzioni di Ordine Superiore e Operatori RDD**
    *   Gli operatori RDD sono funzioni di ordine superiore.
    *   Suddivisi in:
        *   Trasformazioni:
            *   Operazioni *lazy* che creano nuovi RDD.
        *   Azioni:
            *   Operazioni che restituiscono un valore al programma driver dopo aver eseguito un calcolo o scritto dati.

3.  **Creazione di un RDD**
    *   Tre modi principali:
        *   Parallelizzando collezioni:
            *   Utilizzando il metodo `parallelize` dell'API RDD.
            *   Specificando il numero di partizioni.
            *   Esempio:
                ```python
                lines = sc.parallelize(["pandas", "i like pandas"])
                # sc è il contesto Spark
                # sc.parallelize(data, 10) # specifica il numero di partizioni
                ```
        *   Da file:
            *   Utilizzando il metodo `textFile` dell'API RDD.
            *   Leggendo dati da file (HDFS, file system locale, etc.).
            *   Una partizione per blocco HDFS.
            *   Esempio:
                ```python
                lines = sc.textFile("/path/to/README.md")
                ```
            *   Spark tenta di impostare automaticamente il numero di partizioni, ma è consigliabile impostarlo manualmente.
        *   Trasformando un RDD esistente:
            *   Applicando trasformazioni come `map`, `filter`, `flatMap` ad un RDD esistente.
            *   Il numero di partizioni dipende dalla trasformazione.

4.  **Trasformazioni RDD**
    *   `map`:
        *   Applica una funzione a ogni elemento di un RDD.
        *   Trasforma ogni elemento di input in un altro elemento.
        *   Esempio:
            ```python
            nums = sc.parallelize([1, 2, 3, 4])
            squares = nums.map(lambda x: x * x) # squares = [1, 4, 9, 16]
            ```
    *   `filter`:
        *   Crea un nuovo RDD contenente solo gli elementi dell'RDD originale per cui la funzione specificata restituisce `True`.
        *   Esempio:
            ```python
            even = squares.filter(lambda num: num % 2 == 0) # even = [4, 16]
            ```
    *   `flatMap`:
        *   Applica una funzione a ogni elemento dell'RDD.
        *   Mappa ogni elemento di input a zero o più elementi di output.
        *   Esempi:
            ```python
            ranges = nums.flatMap(lambda x: range(0, x)) # ranges = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
            # range(0, x) in Python genera una sequenza di interi da 0 a x-1
            ```
            ```python
            lines = sc.parallelize(["hello world", "hi"])
            words = lines.flatMap(lambda line: line.split(" ")) # words = ["hello", "world", "hi"]
            ```
    *   `join`:
        *   Esegue un equi-join su due RDD basati su una chiave comune.
        *   Solo le chiavi presenti in entrambi gli RDD vengono restituite.
        *   I candidati per il join vengono elaborati in parallelo.
        *   Esempio:
            ```python
            users = sc.parallelize([(0, "Alex"), (1, "Bert"), (2, "Curt"), (3, "Don")])
            hobbies = sc.parallelize([(0, "writing"), (0, "gym"), (1, "swimming")])
            users.join(hobbies).collect() # [(0, ('Alex', 'writing')), (0, ('Alex', 'gym')), (1, ('Bert', 'swimming'))]
            ```
    *   `reduceByKey`:
        *   Aggrega i valori associati alla stessa chiave usando una funzione specificata.
        *   Esegue diverse operazioni di riduzione in parallelo, una per ogni chiave.
        *   Esempio:
            ```python
            x = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("a", 1), ("b", 1), ("b", 1), ("b", 1), ("b", 1)], 3)
            y = x.reduceByKey(lambda accum, n: accum + n) # y = [('b', 5), ('a', 3)]
            ```

5.  **Azioni RDD**
    *   Azioni base:
        *   `collect`:
            *   Restituisce tutti gli elementi dell'RDD come una lista.
            *   Esempio:
                ```python
                nums = sc.parallelize([1, 2, 3, 4])
                nums.collect() # [1, 2, 3, 4]
                ```
        *   `take(n)`:
            *   Restituisce un array con i primi `n` elementi dell'RDD.
            *   Esempio:
                ```python
                nums = sc.parallelize([1, 2, 3, 4])
                nums.take(3)
                ```

---

## Schema Riassuntivo di Spark

### 1. Operazioni RDD
    *   **1.1 Azioni:** Operazioni che restituiscono un valore o scrivono dati su storage.
        *   `count()`: Restituisce il numero di elementi nell'RDD.
            *   Esempio: `nums.count() # 4`
        *   `reduce(func)`: Aggrega gli elementi dell'RDD usando la funzione `func`.
            *   Esempio: `sum = nums.reduce(lambda x, y: x + y)`
        *   `saveAsTextFile(path)`: Salva gli elementi dell'RDD in un file di testo al percorso `path`.
            *   Esempio: `nums.saveAsTextFile("hdfs://file.txt")`
    *   **1.2 Trasformazioni Lazy:** Operazioni che creano un nuovo RDD a partire da uno esistente, eseguite solo quando un'azione richiede il risultato.
        *   Permettono a Spark di ottimizzare l'esecuzione raggruppando le operazioni.
        *   Possono evitare spostamenti di dati in rete se i dati sono già partizionati.

### 2. Esempi di WordCount
    *   **2.1 Scala:**
        ```scala
        val textFile = sc.textFile("hdfs://...")
        val counts = textFile.flatMap(line => line.split(" "))
                         .map(word => (word, 1))
                         .reduceByKey(_ + _)
        counts.saveAsTextFile("hdfs:// ... ")
        ```
    *   **2.2 Python:**
        ```python
        text_file = sc.textFile("hdfs:// ... ")
        counts = text_file.flatMap(lambda line: line.split(" ")) \
                         .map(lambda word: (word, 1)) \
                         .reduceByKey(lambda a, b: a + b)
        counts.saveAsTextFile("hdfs:// ... ")
        ```
    *   **2.3 Java 8:**
        ```java
        JavaRDD<String> textFile = sc.textFile("hdfs://...");
        JavaPairRDD<String, Integer> counts = textFile
                .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey((a, b) -> a + b);
        counts.saveAsTextFile("hdfs://...");
        ```

### 3. Inizializzazione di Spark: `SparkContext`
    *   Necessario per iniziare un programma Spark.
    *   Rappresenta la connessione al cluster Spark.
    *   Disponibile nella shell Spark come `sc`.
    *   Solo un `SparkContext` può essere attivo per JVM.
    *   `SparkConf`: Usato per configurare l'applicazione Spark.
        *   Esempio:
            ```scala
            val conf = new SparkConf().setAppName(appName).setMaster(master)
            new SparkContext(conf)
            ```

### 4. Persistenza RDD
    *   Gli RDD trasformati vengono ricalcolati ad ogni azione per impostazione predefinita.
    *   La persistenza (o memorizzazione nella cache) degli RDD in memoria permette un riutilizzo rapido.
    *   Accelera significativamente le azioni future.
    *   Utilizzare `persist()` o `cache()` per persistere un RDD.
    *   La cache di Spark è tollerante ai guasti.
    *   Utile per algoritmi iterativi e utilizzo interattivo.
    *   **4.1 Livello di Archiviazione:**
        *   `persist()` permette di specificare il livello di archiviazione.
        *   `cache()` equivale a `persist()` con il livello predefinito (MEMORY_ONLY).
        *   Livelli di archiviazione disponibili:
            *   `MEMORY_ONLY`
            *   `MEMORY_AND_DISK`
            *   `MEMORY_ONLY_SER`, `MEMORY_AND_DISK_SER` (Java e Scala)
            *   `DISK_ONLY`
    *   **4.2 Scelta del Livello di Archiviazione:**
        *   Privilegiare la memoria (RAM).
        *   La serializzazione migliora l'efficienza spaziale.
        *   Evitare il disco a meno che le funzioni di calcolo non siano costose.
        *   Utilizzare livelli di archiviazione replicati solo per un rapido ripristino in caso di guasto.

### 5. Come Funziona Spark durante l'Esecuzione
    *   L'applicazione crea RDD, li trasforma ed esegue azioni, generando un DAG (Directed Acyclic Graph) di operatori.
    *   Il DAG viene compilato in fasi, sequenze di RDD senza shuffle intermedi.
    *   Ogni fase è eseguita come una serie di task (uno per ogni partizione).
    *   Le azioni guidano l'esecuzione.
    *   **5.1 Esecuzione degli Stage:**
        *   Spark crea un task per ogni partizione nel nuovo RDD.
        *   Pianifica e assegna i task ai nodi worker.
        *   Questo processo è interno e non richiede intervento dell'utente.

---

**Schema Riassuntivo di Spark**

**1. Componenti di Spark (Granularità)**
    *   **1.1 RDD (Resilient Distributed Dataset):** Dataset parallelo con partizioni.
    *   **1.2 DAG (Directed Acyclic Graph):** Grafo logico delle operazioni RDD.
    *   **1.3 Stage:** Insieme di task eseguiti in parallelo.
    *   **1.4 Task:** Unità fondamentale di esecuzione in Spark.

**2. Tolleranza ai Guasti**
    *   **2.1 Lineage:** Gli RDD tracciano la loro serie di trasformazioni (lineage).
    *   **2.2 Ricalcolo:** La lineage è usata per ricalcolare i dati persi.
    *   **2.3 Memorizzazione:** Gli RDD sono memorizzati come una catena di oggetti che catturano la lineage.
    *   **2.4 Esempio Scala:**
        ```scala
        val file = sc.textFile("hdfs:// ... ")
        val sics = file.filter (_. contains ("SICS"))
        val cachedSics = sics.cache()
        val ones = cachedSics.map(_ => 1)
        val count = ones.reduce(_+_)
        ```

**3. Pianificazione dei Job**
    *   **3.1 Considerazioni:** La pianificazione considera la disponibilità in memoria delle partizioni degli RDD persistenti.
    *   **3.2 DAG di Stage:** Lo scheduler costruisce un DAG di stage dal grafo di lineage dell'RDD.
    *   **3.3 Stage:** Contiene trasformazioni in pipeline con dipendenze strette.
    *   **3.4 Confini Stage:** Definiti dallo shuffle (dipendenze ampie) e dalle partizioni già calcolate.
    *   **3.5 Esecuzione Task:** Lo scheduler lancia i task per calcolare le partizioni mancanti di ogni stage.
    *   **3.6 Località dei Dati:** I task sono assegnati alle macchine in base alla località dei dati.

**4. API DataFrame e Dataset**
    *   **4.1 Natura:** Collezioni di dati immutabili distribuite, valutate in modo pigro.
    *   **4.2 DataFrame (Spark 1.3):**
        *   **4.2.1 Schema:** Introducono uno schema per descrivere i dati, organizzati in colonne denominate.
        *   **4.2.2 Dati:** Funzionano su dati strutturati e semi-strutturati.
        *   **4.2.3 Spark SQL:** Permette query SQL su DataFrame.
        *   **4.2.4 Implementazione:** Da Spark 2.0, i DataFrame sono implementati come un caso speciale di Dataset.
    *   **4.3 Dataset (Spark 1.6):**
        *   **4.3.1 Estensione:** Estendono i DataFrame con un'interfaccia orientata agli oggetti di tipo sicuro.
        *   **4.3.2 Tipizzazione:** Collezioni di oggetti JVM fortemente tipizzati.
        *   **4.3.3 SparkSession:** Punto di ingresso per entrambe le API.
        *   **4.3.4 Vantaggi:** Combinano i vantaggi degli RDD (tipizzazione forte, funzioni lambda) con l'ottimizzatore Catalyst di Spark SQL.
        *   **4.3.5 Linguaggi:** Disponibili in Scala e Java.
        *   **4.3.6 Operazioni:** Manipolabili con trasformazioni funzionali (map, filter, flatMap, ...).
        *   **4.3.7 Valutazione:** Pigri; il calcolo avviene solo con un'azione.
        *   **4.3.8 Ottimizzazione:** Piano logico ottimizzato in un piano fisico.
        *   **4.3.9 Creazione:**
            *   Da un file usando `read`.
            *   Da un RDD esistente, convertendolo.
            *   Tramite trasformazioni su Dataset esistenti.
        *   **4.3.10 Esempio Scala:**
            ```scala
            val names = people.map(_.name) //names è un Dataset[String]
            ```
        *   **4.3.11 Esempio Java:**
            ```java
            Dataset<String> names = people.map((Person p) -> p.name, Encoders.STRING());
            ```
    *   **4.4 DataFrame (Dettagli):**
        *   **4.4.1 Struttura:** Dataset organizzato in colonne denominate.
        *   **4.4.2 Equivalenza:** Equivalente a una tabella in un database relazionale.
        *   **4.4.3 Ottimizzazione:** Sfrutta l'ottimizzatore Catalyst.
        *   **4.4.4 Linguaggi:** Disponibile in Scala, Java, Python e R.
        *   **4.4.5 Rappresentazione:** In Scala e Java, un DataFrame è rappresentato da un Dataset di `Row`.
        *   **4.4.6 Costruzione:**
            *   Da RDD esistenti, inferendo lo schema o specificandolo.
            *   Tabelle in Hive.
            *   File di dati strutturati (JSON, Parquet, CSV, Avro).
        *   **4.4.7 Manipolazione:** Manipolato in modi simili agli RDD.

**5. Spark Streaming**
    *   **5.1 Descrizione:** Estensione per analizzare dati in streaming, ingeriti e analizzati in micro-batch.
    *   **5.2 DStream:** Astrazione di alto livello (discretized stream) che rappresenta un flusso continuo di dati: una sequenza di RDD.

**6. Spark MLlib**
    *   **6.1 Descrizione:** Fornisce molti algoritmi di machine learning distribuiti.
    *   **6.2 Esempi:**
        *   Classificazione (es. regressione logistica).
        *   Regressione.
        *   Clustering.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Funzionalità Principali**

    A. Algoritmi di Machine Learning
        1. K-means
        2. Raccomandazione
        3. Alberi decisionali
        4. Foreste casuali
        5. Altri

    B. Utility per Machine Learning
        1. Trasformazioni di feature
        2. Valutazione del modello
        3. Tuning degli iperparametri

    C. Algebra Lineare Distribuita e Statistica
        1. PCA (esempio di algebra lineare)
        2. Statistiche riassuntive
        3. Test di ipotesi

    D. Supporto dei Dati
        1. Adotta DataFrame
        2. Supporta una varietà di tipi di dati

---
