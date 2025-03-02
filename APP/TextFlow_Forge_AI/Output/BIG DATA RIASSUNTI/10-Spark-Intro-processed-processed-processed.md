
## MapReduce: Limitazioni e Inefficienze

MapReduce, pur essendo un modello di programmazione influente per l'elaborazione distribuita, presenta significative limitazioni.  La sua difficoltà nell'implementare algoritmi complessi, richiedendo spesso molteplici passaggi per operazioni semplici (es. WordCount con ordinamento), e la mancanza di controllo a basso livello su strutture dati e tipi di dati, ne compromettono la flessibilità.  L'assenza di supporto nativo per l'iterazione efficiente, con la conseguente scrittura e lettura su disco ad ogni iterazione, genera un elevato overhead.  Inoltre, l'alto costo di comunicazione (map, shuffle, reduce), la frequente scrittura su disco e lo sfruttamento limitato della memoria principale ne riducono l'efficienza, rendendolo inadatto all'elaborazione di flussi di dati in tempo reale.  La condivisione dei dati è lenta a causa della replicazione, serializzazione e I/O su disco.  ![|456](_page_1_Figure_9.jpeg)  ![|450](_page_4_Figure_2.jpeg)


## Apache Spark: Un'Alternativa Migliorata

Apache Spark offre una soluzione più performante ed efficiente rispetto a MapReduce.  Si distingue per l'archiviazione dei dati in memoria, garantendo un'elaborazione iterativa almeno 10 volte più veloce di Hadoop.  Supporta ottimizzazioni avanzate e grafi di esecuzione generali, offrendo compatibilità con le API di storage di Hadoop (HDFS, HBase, ecc.). La condivisione dei dati avviene tramite memoria distribuita, risultando 10-100 volte più veloce rispetto a disco e rete. ![|450](_page_5_Figure_2.jpeg)


## Spark vs. MapReduce: Confronto Chiave

Spark, pur mantenendo un paradigma di programmazione simile a MapReduce ("scatter-gather"), offre un modello di dati più generale (RDD, DataSet, DataFrame) e un'interfaccia più user-friendly.  La sua agnosticism rispetto allo storage (HDFS, Cassandra, S3, Parquet) lo rende altamente versatile.


## Architettura e Componenti di Spark

Spark si basa su **Spark Core**, che fornisce funzionalità di base come pianificazione dei task, gestione della memoria, ripristino da guasti e interazione con sistemi di storage.  Introduce gli **RDD (Resilient Distributed Dataset)**, collezioni di elementi distribuiti manipolabili in parallelo, con API disponibili per Java, Python, R e Scala.  ![|360](_page_7_Figure_1.jpeg)

Spark integra diversi moduli: **Spark SQL** per l'elaborazione di dati strutturati tramite SQL; **Spark Streaming** per l'elaborazione di flussi in tempo reale; **MLlib** per il machine learning; e **GraphX** per l'analisi di grafi.  Tutti questi moduli estendono l'API Spark RDD e sono interoperabili.

Spark può essere eseguito su diversi gestori di cluster: modalità standalone, Hadoop YARN, Mesos e Kubernetes. Ogni applicazione Spark è composta da un programma driver e da executor nel cluster.

---

## Spark: Architettura e Programmazione

Spark è un framework di elaborazione dati distribuita che si basa su un'architettura master-slave.  Il **programma driver**, che esegue la funzione `main()`, crea un `SparkContext` per connettersi al gestore di cluster, acquisire gli **executor** (processi che eseguono task in parallelo su più thread, garantendo l'isolamento tra applicazioni) e distribuire il codice.  Il `SparkContext` gestisce l'allocazione delle risorse e l'invio dei task agli executor.

Il concetto chiave in Spark è il **Resilient Distributed Dataset (RDD)**: una struttura dati immutabile, distribuita, partizionata e fault-tolerant.  Gli RDD sono memorizzati in memoria (o su disco se la memoria è insufficiente) e replicati per garantire la tolleranza ai guasti.  La loro immutabilità implica che le modifiche creano nuovi RDD, mentre la fault-tolerance è garantita dalla *lineage*, un tracciamento della storia di creazione dell'RDD che permette il ricalcolo efficiente dei dati persi.  Spark gestisce automaticamente la partizione e l'allocazione degli RDD, nascondendo la complessità della tolleranza agli errori tramite un DAG (Directed Acyclic Graph) di lineage.

L'**API RDD**, disponibile in Scala, Python, Java e R, offre un ricco set di **trasformazioni** (operazioni *lazy* che creano nuovi RDD, es: `map`, `filter`, `join`) e **azioni** (operazioni che avviano un job e restituiscono un valore al driver, es: `count`, `collect`, `save`).  Gli RDD sono ideali per applicazioni batch ma meno adatti per aggiornamenti asincroni a grana fine su uno stato condiviso.

Il **modello di programmazione Spark** si basa su operatori parallelizzabili, rappresentati da un DAG che collega sorgenti di dati, operatori e sink di dati.  Spark offre anche API di livello superiore come DataFrames e Datasets.

![|326](_page_13_Figure_3.jpeg)
![[](_page_18_Figure_6.jpeg)
![|313](_page_20_Figure_4.jpeg)
![|408](_page_23_Figure_5.jpeg)

---

## RDD in Spark: Trasformazioni e Azioni

Gli RDD (Resilient Distributed Datasets) in Apache Spark sono collezioni di dati distribuite, immutabili e parallelizzate.  Le operazioni sugli RDD sono divise in due categorie: **trasformazioni** e **azioni**.

**1. Creazione di RDD:**

Un RDD può essere creato in tre modi:

* **Parallelizzando una collezione:**  `sc.parallelize(data, numPartitions)` crea un RDD da una collezione Python, specificando opzionalmente il numero di partizioni.  Esempio: `lines = sc.parallelize(["pandas", "i like pandas"])`
* **Da un file:** `sc.textFile("/path/to/file")` crea un RDD leggendo dati da un file (HDFS, file system locale, etc.).  Il numero di partizioni è spesso determinato automaticamente, ma può essere impostato manualmente per ottimizzare le prestazioni. Esempio: `lines = sc.textFile("/path/to/README.md")`
* **Trasformando un RDD esistente:** Applicando trasformazioni ad un RDD esistente. Il numero di partizioni può variare a seconda della trasformazione.


**2. Trasformazioni RDD:**  Sono operazioni *lazy*, ovvero non vengono eseguite immediatamente ma solo quando viene richiesta un'azione.

* **`map(func)`:** Applica una funzione `func` ad ogni elemento dell'RDD, restituendo un nuovo RDD con gli elementi trasformati. Esempio: `squares = nums.map(lambda x: x * x)`
* **`filter(func)`:** Crea un nuovo RDD contenente solo gli elementi che soddisfano la condizione definita dalla funzione `func`. Esempio: `even = squares.filter(lambda num: num % 2 == 0)`
* **`flatMap(func)`:** Simile a `map`, ma la funzione `func` può restituire zero o più elementi per ogni elemento di input. Esempio: `words = lines.flatMap(lambda line: line.split(" "))`
* **`join()`:** Esegue un equi-join tra due RDD basati su una chiave comune.  Solo le chiavi presenti in entrambi gli RDD vengono restituite.
* **`reduceByKey(func)`:** Aggrega i valori associati alla stessa chiave usando la funzione `func`.


**3. Azioni RDD:** Sono operazioni che restituiscono un risultato al driver dopo aver eseguito un calcolo o scritto dati.

* **`collect()`:** Restituisce tutti gli elementi dell'RDD come una lista.
* **`take(n)`:** Restituisce un array con i primi `n` elementi.
* **`count()`:** Restituisce il numero di elementi.
* **`reduce(func)`:** Aggrega tutti gli elementi usando la funzione `func`.
* **`saveAsTextFile(path)`:** Salva gli elementi in un file di testo.


**4. Lazy Evaluation:** Le trasformazioni RDD sono *lazy*.  L'esecuzione effettiva avviene solo quando viene chiamata un'azione. Questo permette a Spark di ottimizzare l'esecuzione raggruppando le operazioni.

---

## Riassunto di Spark: RDD, Persistenza e Esecuzione

Questo documento descrive i concetti fondamentali di Apache Spark, focalizzandosi su RDD, persistenza e esecuzione.

### RDD e Operazioni di Base

Spark utilizza **RDD (Resilient Distributed Datasets)**, collezioni di dati distribuite e parallelizzate.  Le operazioni su RDD sono di due tipi: **trasformazioni** (creano nuovi RDD, es. `flatMap`, `map`, `reduceByKey`) e **azioni** (avviano il calcolo e restituiscono un risultato, es. `saveAsTextFile`, `reduce`).  Esemplificazioni di `WordCount` in Scala, Python e Java 8 dimostrano come combinare trasformazioni per elaborare dati in parallelo.  Spark ottimizza l'esecuzione combinando operazioni e sfruttando la partizione dei dati per ridurre gli spostamenti in rete.

```scala
val counts = textFile.flatMap(line => line.split(" "))
                     .map(word => (word, 1))
                     .reduceByKey(_ + _)
```

```python
counts = text_file.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a + b)
```

```java
JavaPairRDD<String, Integer> counts = textFile
    .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
    .mapToPair(word -> new Tuple2<>(word, 1))
    .reduceByKey((a, b) -> a + b);
```

### Inizializzazione e Persistenza

Un programma Spark inizia creando un `SparkContext` tramite `SparkConf`, che configura l'applicazione.  Solo un `SparkContext` per JVM è consentito.  La persistenza degli RDD tramite `persist()` o `cache()` (equivalente a `persist()` con livello predefinito `MEMORY_ONLY`) migliora significativamente le prestazioni, riducendo i ricalcoli.  Diversi livelli di archiviazione sono disponibili (`MEMORY_ONLY`, `MEMORY_AND_DISK`, `MEMORY_ONLY_SER`, `MEMORY_AND_DISK_SER`, `DISK_ONLY`, etc.),  la cui scelta dipende da fattori come disponibilità di RAM, necessità di serializzazione e costo computazionale.  ![[_page_46_Figure_2.jpeg]] ![|454](_page_47_Figure_7.jpeg)

```scala
val conf = new SparkConf().setAppName(appName).setMaster(master)
new SparkContext(conf)
```

### Esecuzione e Tolleranza ai Guasti

Spark esegue le operazioni creando un **DAG (Directed Acyclic Graph)** di operatori.  Il DAG viene suddiviso in **stage**, eseguiti come insiemi di **task** (uno per partizione).  Le azioni innescano l'esecuzione.  Spark gestisce la tolleranza ai guasti tramite la **lineage** degli RDD, permettendo il ricalcolo dei dati persi.  ![|281](_page_50_Figure_4.jpeg)  La granularità dei componenti di Spark è: RDD, DAG, Stage, Task.


```scala
val cachedSics = sics.cache()
```

---

## Riassunto di Pianificazione Job, DataFrame/Dataset, Spark Streaming e MLlib in Spark

Questo riassunto descrive i concetti chiave di Spark relativi alla pianificazione dei job, alle API DataFrame e Dataset, a Spark Streaming e a Spark MLlib.

### Pianificazione dei Job in Spark

Lo scheduler di Spark costruisce un Directed Acyclic Graph (DAG) di *stage* dal grafo di lineage degli RDD. Ogni stage contiene trasformazioni in pipeline con dipendenze strette, delimitate da operazioni di *shuffle* (per dipendenze ampie) e dalla disponibilità di partizioni già calcolate.  ![|301](_page_51_Figure_7.jpeg) Lo scheduler assegna i *task* alle macchine, privilegiando la località dei dati: se una partizione è in memoria su un nodo, il task viene inviato a quel nodo.  I task vengono eseguiti fino al completamento dell'RDD di destinazione.

### API DataFrame e Dataset

DataFrame e Dataset sono collezioni di dati immutabili, distribuite e valutate in modo pigro, simili agli RDD. I DataFrame (da Spark 1.3) introducono uno schema con colonne denominate (simile a una tabella di database), supportando dati strutturati e semi-strutturati e query SQL tramite Spark SQL.  Da Spark 2.0, i DataFrame sono un caso speciale di Dataset. I Dataset (da Spark 1.6) estendono i DataFrame con un'interfaccia orientata agli oggetti di tipo sicuro, offrendo collezioni di oggetti JVM fortemente tipizzati (a differenza di `Dataset[Row]` dei DataFrame).  `SparkSession` è il punto di ingresso per entrambe le API.  I Dataset combinano i vantaggi degli RDD (tipizzazione forte, funzioni lambda) con l'ottimizzatore Catalyst di Spark SQL. Sono disponibili in Scala e Java, costruibili da oggetti JVM e manipolabili con trasformazioni funzionali (map, filter, flatMap, ecc.).  Il calcolo avviene solo con un'azione, dopo l'ottimizzazione del piano logico in un piano fisico.  Possono essere creati da file (`read`), da RDD esistenti o tramite trasformazioni su Dataset esistenti.

Esempio Scala: `val names = people.map(_.name) //names è un Dataset[String]`

Esempio Java: `Dataset<String> names = people.map((Person p) -> p.name, Encoders.STRING());`

I DataFrame, disponibili anche in Python e R, sono rappresentati in Scala e Java da un Dataset di `Row`. Possono essere costruiti da RDD (con schema inferito o specificato), tabelle Hive e file strutturati (JSON, Parquet, CSV, Avro).

### Spark Streaming

Spark Streaming analizza dati in streaming in micro-batch, usando l'astrazione DStream (discretized stream), una sequenza di RDD. ![|373](_page_58_Figure_5.jpeg)  ![|380](_page_59_Figure_2.jpeg)

### Spark MLlib

Spark MLlib fornisce algoritmi di machine learning distribuiti (classificazione, regressione, clustering, raccomandazione, alberi decisionali, foreste casuali, ecc.), utility per trasformazioni di feature, valutazione del modello, tuning degli iperparametri, algebra lineare distribuita e statistica. Utilizza i DataFrame per supportare diversi tipi di dati.  Un esempio di regressione logistica è mostrato con codice Scala, che crea un DataFrame da dati con etichette e vettori di feature per predire le etichette dai vettori.  `df = sqlContext.createDataFrame(data, ["label", "features"])` (Codice incompleto, ma illustrativo).

---

Il codice presentato mostra l'utilizzo di `LogisticRegression` in Scala per la modellazione predittiva.  Viene creato un modello di regressione logistica (`lr`) con un massimo di 10 iterazioni (`maxIter=10`).  Successivamente, il modello viene addestrato (`model = lr.fit(df)`) utilizzando un DataFrame (`df`) contenente i dati di training. Infine, il modello addestrato viene utilizzato per predire le etichette di un dataset (probabilmente lo stesso `df`), e i risultati vengono visualizzati utilizzando il metodo `show()` (`model.transform(df).show()`).  In sintesi, il codice illustra un flusso di lavoro completo di machine learning: addestramento e predizione con un modello di regressione logistica.

---

Per favore, forniscimi il testo da riassumere.  Non ho ricevuto alcun testo nell'input precedente.  Una volta che mi fornirai il testo, potrò creare un riassunto secondo le tue istruzioni.

---
