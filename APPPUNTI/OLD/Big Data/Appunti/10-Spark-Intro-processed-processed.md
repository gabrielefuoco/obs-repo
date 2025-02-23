

## MapReduce: Debolezze e Limitazioni

### Modello di Programmazione

Il modello di programmazione MapReduce presenta diverse limitazioni:

* È difficile implementare algoritmi complessi come programmi MapReduce diretti.  Sono spesso necessari più passaggi MapReduce anche per operazioni semplici.  Ad esempio, un WordCount che ordina le parole per frequenza richiede più passaggi.
* Mancanza di controllo a basso livello sulle strutture dati e sui tipi di dati.

### Supporto per l'Iterazione

* MapReduce non supporta nativamente l'iterazione efficiente. Ogni iterazione implica la scrittura e la lettura dei dati dal disco, causando un significativo overhead.  È necessario progettare algoritmi che minimizzino il numero di iterazioni.

![|456](_page_1_Figure_9.jpeg)


### Efficienza e HDFS

* **Alto costo di comunicazione:** Il processo MapReduce comporta tre fasi principali: calcolo (map), comunicazione (shuffle) e calcolo (reduce).  Ogni fase contribuisce al costo complessivo.
* **Scrittura frequente su disco:** La scrittura frequente dell'output su disco limita le prestazioni.
* **Sfruttamento limitato della memoria principale:** MapReduce non sfrutta efficacemente la memoria principale.

MapReduce non è adatto all'elaborazione di flussi di dati in tempo reale, in quanto richiede la scansione dell'intero input prima dell'elaborazione.


## Apache Spark



Apache Spark è un motore veloce e versatile per l'elaborazione di Big Data.  Non è una versione modificata di Hadoop, ma una piattaforma leader per SQL su larga scala, elaborazione batch, streaming e machine learning.  È un motore analitico unificato per l'elaborazione di dati su larga scala.

* **Archiviazione in memoria:** Spark archivia i dati in memoria per una rapida elaborazione iterativa, risultando almeno 10 volte più veloce di Hadoop.
* **Ottimizzazioni:** Supporta grafi di esecuzione generali e potenti ottimizzazioni.
* **Compatibilità Hadoop:** È compatibile con le API di storage di Hadoop, leggendo e scrivendo su sistemi come HDFS e HBase.
**Condivisione di dati in MapReduce:** Lenta a causa della replicazione, serializzazione e I/O col disco
![|450](_page_4_Figure_2.jpeg)
**Condivisione di dati in Spark:** Memoria distribuita: 10-100 volte più veloce rispetto al disco e alla rete
![|450](_page_5_Figure_2.jpeg)


## Spark vs Hadoop MapReduce

Il paradigma di programmazione di Spark è simile a MapReduce ("scatter-gather"), ma offre vantaggi significativi:

* **Modello di dati più generale:** Spark offre RDD, DataSet e DataFrame.
* **Modello di programmazione più user-friendly:** Le trasformazioni in Spark corrispondono alle map di MapReduce, mentre le azioni corrispondono alle reduce.
* **Agnosticità rispetto allo storage:** Spark supporta diversi sistemi di storage, tra cui HDFS, Cassandra, S3 e file Parquet.


## Stack Spark

![[|360](_page_7_Figure_1.jpeg)


### Spark Core

Spark Core fornisce le funzionalità di base, tra cui la pianificazione dei task, la gestione della memoria, il ripristino da guasti e l'interazione con i sistemi di storage.  Introduce il concetto di **Resilient Distributed Dataset (RDD)**, una raccolta di elementi distribuiti che possono essere manipolati in parallelo.  Spark Core offre API per Java, Python, R e Scala.


### Spark come Motore Unificato

Spark include diversi moduli di livello superiore integrati e interoperabili:

* **Spark SQL:** Per lavorare con dati strutturati, interrogandoli tramite SQL. Supporta diverse sorgenti dati (tabelle Hive, Parquet, JSON, ecc.). Estende l'API Spark RDD.
* **Spark Streaming:** Per elaborare flussi di dati in tempo reale. Estende l'API Spark RDD.
* **MLlib:** Libreria di machine learning scalabile con algoritmi distribuiti per estrazione di feature, classificazione, regressione, clustering e raccomandazione.
* **GraphX:** API per la manipolazione di grafi e l'esecuzione di calcoli paralleli sui grafi, inclusi algoritmi come PageRank. Estende l'API Spark RDD.





## Spark su Gestori di Cluster

Spark può essere eseguito su diversi gestori di risorse del cluster:

* **Modalità standalone:** Utilizza un semplice scheduler FIFO incluso in Spark.
* **Hadoop YARN**
* **Mesos:**  Mesos e Spark provengono entrambi da AMPLab @ UC Berkeley.
* **Kubernetes**


### Architettura Spark

Ogni applicazione Spark è composta da un programma driver e da executor nel cluster.

* **Programma driver:** Esegue la funzione `main()` e crea l'oggetto `SparkContext`.
* **Executor:** Processi che rimangono attivi per tutta la durata dell'applicazione ed eseguono task in più thread, garantendo l'isolamento tra applicazioni concorrenti.

**`SparkContext`** si connette a un gestore di cluster per l'allocazione delle risorse, acquisisce gli executor e invia il codice dell'applicazione agli executor.  Infine, invia i task agli executor per l'esecuzione.

![|326](_page_13_Figure_3.jpeg)




## Resilient Distributed Dataset

Gli RDD sono l'astrazione di programmazione chiave in Spark: una struttura dati immutabile, distribuita, partizionata e fault-tolerant.  Sono memorizzati in memoria principale attraverso i nodi del cluster. Ogni nodo contiene almeno una partizione degli RDD definiti nell'applicazione.

* **Immutabile:** La struttura dati non viene modificata dopo la creazione.
* **In memoria (esplicitamente):**  I dati risiedono principalmente in memoria.
* **Fault-tolerant:**  La tolleranza ai guasti è garantita grazie alla replica dei dati.
* **Struttura dati distribuita:** I dati sono distribuiti tra i nodi del cluster.
* **Partizionamento controllato:** Il partizionamento ottimizza il posizionamento dei dati.
* **Ricco set di operatori:** Gli RDD possono essere manipolati usando un ricco set di operatori.

### RDDs: Distribuiti e Partizionati

Gli RDD (Resilient Distributed Datasets) sono memorizzati nella memoria principale degli executor in esecuzione sui nodi worker, se disponibile.  In caso contrario, vengono memorizzati sul disco locale del nodo. Questa caratteristica consente l'esecuzione parallela del codice su di essi.  Ogni executor di un nodo worker esegue il codice specificato sulla propria partizione dell'RDD.  Una *partizione* è un frammento atomico di dati, una divisione logica dei dati e l'unità base di parallelismo. Le partizioni di un RDD possono risiedere su nodi cluster diversi.

![](_page_18_Figure_6.jpeg)

### RDDs: Immutabili e Fault-Tolerant

Gli RDD sono immutabili dopo la loro creazione; il loro contenuto non può essere modificato. Per effettuare modifiche, si crea un nuovo RDD basato su quello esistente.  Sono inoltre *fault-tolerant*: vengono ricostruiti automaticamente in caso di errore, senza richiedere replicazione.  Questo è possibile grazie alla *lineage*, ovvero alle informazioni che tracciano la storia della creazione dell'RDD.  Spark utilizza la lineage per ricalcolare in modo efficiente i dati persi a causa di errori dei nodi.

### Gestione degli RDDs

Spark gestisce la suddivisione degli RDD in partizioni e la loro allocazione ai nodi del cluster.  Nasconde inoltre le complessità della tolleranza agli errori, ricostruendo automaticamente gli RDD in caso di fallimento, sfruttando il DAG (Directed Acyclic Graph) di lineage, che rappresenta il piano di esecuzione logica.

![|313](_page_20_Figure_4.jpeg)

### API RDD

L'API RDD è pulita e integrata nei linguaggi Scala, Python, Java e R. Può essere utilizzata interattivamente dalla console Scala e dalla console PySpark.  Spark offre anche API di livello superiore come DataFrames e Datasets.

### Idoneità RDD

Gli RDD sono più adatti per applicazioni batch che applicano la stessa operazione a tutti gli elementi di un dataset. Sono meno adatti per applicazioni che richiedono aggiornamenti asincroni a grana fine su uno stato condiviso, come ad esempio un sistema di storage per un'applicazione web.


### Operazioni nell'API RDD

I programmi Spark sono basati su operazioni sugli RDD.  Questi vengono creati da dati esterni o da altri RDD tramite:

* **Trasformazioni:** Operazioni a grana grossa che definiscono un nuovo dataset basato su quelli precedenti (es: `map`, `filter`, `join`). Sono *lazy*, ovvero il calcolo viene eseguito solo quando richiesto da un'azione.
* **Azioni:** Operazioni che avviano un job sul cluster e restituiscono un valore al programma driver (es: `count`, `collect`, `save`).


## Modello di Programmazione Spark

Il modello di programmazione Spark si basa su operatori parallelizzabili, ovvero funzioni di ordine superiore che eseguono funzioni definite dall'utente in parallelo. Il flusso di dati è rappresentato da un DAG (Directed Acyclic Graph) che collega sorgenti di dati, operatori e sink di dati.

![|408](_page_23_Figure_5.jpeg)


## Funzioni di Ordine Superiore e Operatori RDD

Le funzioni di ordine superiore sono gli operatori RDD, suddivisi in trasformazioni e azioni.

* **Trasformazioni:** Operazioni *lazy* che creano nuovi RDD.
* **Azioni:** Operazioni che restituiscono un valore al programma driver dopo aver eseguito un calcolo o scritto dati.

![[Pasted image 20250223170527.png]]
## Creazione di un RDD

Un RDD può essere creato in tre modi:

1. **Parallelizzando collezioni:** Utilizzando il metodo `parallelize` dell'API RDD, specificando il numero di partizioni.
2. **Da file:** Utilizzando il metodo `textFile` dell'API RDD, leggendo dati da file (HDFS, file system locale, etc.).  Una partizione per blocco HDFS.
3. **Trasformando un RDD esistente:** Applicando trasformazioni come `map`, `filter`, `flatMap` ad un RDD esistente. Il numero di partizioni dipende dalla trasformazione.


## Esempi di Creazione di RDDs

* **Da una collezione:**

```python
lines = sc.parallelize(["pandas", "i like pandas"])
```

`sc` è il contesto Spark.  Il numero di partizioni può essere specificato come secondo parametro (es: `sc.parallelize(data, 10)`).

* **Da un file:**

```python
lines = sc.textFile("/path/to/README.md")
```

Spark tenta di impostare automaticamente il numero di partizioni, ma è consigliabile impostarlo manualmente per un controllo ottimale.

---

## Trasformazioni RDD


- **`map`**: Applica una funzione a ogni elemento di un RDD, trasformando ogni elemento di input in un altro elemento.

  Esempio:

  ```python
  nums = sc.parallelize([1, 2, 3, 4])
  squares = nums.map(lambda x: x * x)  # squares = [1, 4, 9, 16]
  ```

- **`filter`**: Crea un nuovo RDD contenente solo gli elementi dell'RDD originale per cui la funzione specificata restituisce `True`.

  Esempio:

  ```python
  even = squares.filter(lambda num: num % 2 == 0)  # even = [4, 16]
  ```



- **`flatMap`**: Applica una funzione a ogni elemento dell'RDD.  A differenza di `map`, può mappare ogni elemento di input a zero o più elementi di output.

  Esempi:

  ```python
  ranges = nums.flatMap(lambda x: range(0, x))  # ranges = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
  ```

  `range(0, x)` in Python genera una sequenza di interi da 0 a x-1.

  ```python
  lines = sc.parallelize(["hello world", "hi"])
  words = lines.flatMap(lambda line: line.split(" "))  # words = ["hello", "world", "hi"]
  ```




- **`join`**: Esegue un equi-join su due RDD basati su una chiave comune.
	- Solo le chiavi presenti in entrambi gli RDD vengono restituite.
	- I candidati per il join vengono elaborati in parallelo.
Esempio:
```python
users = sc.parallelize([(0, "Alex"), (1, "Bert"), (2, "Curt"), (3, "Don")])
hobbies = sc.parallelize([(0, "writing"), (0, "gym"), (1, "swimming")])
users.join(hobbies).collect()  # [(0, ('Alex', 'writing')), (0, ('Alex', 'gym')), (1, ('Bert', 'swimming'))]
```


- **`reduceByKey`**: Aggrega i valori associati alla stessa chiave usando una funzione specificata.
	- Esegue diverse operazioni di riduzione in parallelo, una per ogni chiave.


Esempio:

```python
x = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("a", 1), ("b", 1), ("b", 1), ("b", 1), ("b", 1)], 3)
y = x.reduceByKey(lambda accum, n: accum + n)  # y = [('b', 5), ('a', 3)]
```

## Azioni RDD

### Azioni base

- **`collect`**: Restituisce tutti gli elementi dell'RDD come una lista.

  ```python
  nums = sc.parallelize([1, 2, 3, 4])
  nums.collect()  # [1, 2, 3, 4]
  ```

- **`take(n)`**: Restituisce un array con i primi `n` elementi dell'RDD.

  ```python
  nums.take(3)  # [1, 2, 3]
  ```

- **`count`**: Restituisce il numero di elementi nell'RDD.

  ```python
  nums.count()  # 4
  ```

- **`reduce`**: Aggrega tutti gli elementi dell'RDD usando una funzione specificata.

  ```python
  sum = nums.reduce(lambda x, y: x + y)
  ```

- **`saveAsTextFile`**: Salva gli elementi dell'RDD in un file di testo.

  ```python
  nums.saveAsTextFile("hdfs://file.txt")
  ```

# Trasformazioni Lazy

- Le trasformazioni RDD sono pigre: vengono eseguite solo quando un'azione richiede il risultato.
- Questo permette a Spark di ottimizzare l'esecuzione raggruppando le operazioni.  Ad esempio, più operazioni di filtro o map possono essere combinate in una singola passata.  Se i dati sono già partizionati, Spark può evitare spostamenti di dati in rete per operazioni come `groupBy`.

![[|572](_page_35_Figure_2.jpeg)


## Esempi

#### WordCount in Scala

```scala
val textFile = sc.textFile("hdfs://...")
val counts = textFile.flatMap(line => line.split(" "))
           .map(word => (word, 1))
           .reduceByKey(_ + _)
counts.saveAsTextFile("hdfs:// ... ")
```

#### WordCount in Python

```python
text_file = sc.textFile("hdfs:// ... ")
counts = text_file.flatMap(lambda line: line.split(" ")) \
                   .map(lambda word: (word, 1)) \
                   .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("hdfs:// ... ")
```

![|471](_page_37_Figure_5.jpeg)

### WordCount in Java 8

```java
JavaRDD<String> textFile = sc.textFile("hdfs://...");
JavaPairRDD<String, Integer> counts = textFile
    .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
    .mapToPair(word -> new Tuple2<>(word, 1))
    .reduceByKey((a, b) -> a + b);
counts.saveAsTextFile("hdfs://...");
```

## Inizializzazione di Spark: `SparkContext`

- Per iniziare un programma Spark, è necessario creare un oggetto `SparkContext`. Questo rappresenta la connessione al cluster Spark e permette di creare RDD.  È disponibile anche nella shell Spark come `sc`.
- Solo un `SparkContext` può essere attivo per JVM. Chiamare `stop()` sul `SparkContext` esistente prima di crearne uno nuovo.
- `SparkConf` è usato per configurare l'applicazione Spark, impostando parametri chiave-valore.

```scala
val conf = new SparkConf().setAppName(appName).setMaster(master)
new SparkContext(conf)
```

## Persistenza RDD

Per impostazione predefinita, ogni RDD trasformato viene ricalcolato ad ogni azione. Spark supporta la persistenza (o memorizzazione nella cache) degli RDD in memoria per un riutilizzo rapido.  Quando un RDD è persistente, ogni nodo memorizza le sue partizioni, riutilizzandole in azioni successive. Questo accelera significativamente le azioni future (anche di 100 volte).  Per persistere un RDD, utilizzare `persist()` o `cache()`. La cache di Spark è tollerante ai guasti: le partizioni perse vengono ricalcolate automaticamente. La persistenza è uno strumento chiave per algoritmi iterativi e utilizzo interattivo.


### Livello di Archiviazione

`persist()` permette di specificare il livello di archiviazione. `cache()` equivale a `persist()` con il livello predefinito (MEMORY_ONLY). I livelli di archiviazione disponibili per `persist()` includono:

* `MEMORY_ONLY`
* `MEMORY_AND_DISK`
* `MEMORY_ONLY_SER`, `MEMORY_AND_DISK_SER` (Java e Scala). In Python, gli oggetti sono sempre serializzati con Pickle.
* `DISK_ONLY`, ...


### Scelta del Livello di Archiviazione

La scelta del livello di archiviazione ottimale dipende da diversi fattori:

* Privilegiare la memoria (RAM).
* La serializzazione migliora l'efficienza spaziale, ma richiede una libreria di serializzazione veloce (es. Kryo).
* Evitare il disco a meno che le funzioni di calcolo non siano costose (es. filtraggio di grandi quantità di dati).
* Utilizzare livelli di archiviazione replicati solo per un rapido ripristino in caso di guasto.




## Come Funziona Spark durante l'Esecuzione

L'applicazione crea RDD, li trasforma ed esegue azioni, generando un DAG (Directed Acyclic Graph) di operatori. Il DAG viene compilato in fasi, sequenze di RDD senza shuffle intermedi. Ogni fase è eseguita come una serie di task (uno per ogni partizione). Le azioni guidano l'esecuzione.

![[_page_46_Figure_2.jpeg]]


![|454](_page_47_Figure_7.jpeg)
### Esecuzione degli Stage

Spark crea un task per ogni partizione nel nuovo RDD, pianificando e assegnando i task ai nodi worker. Questo processo è interno e non richiede intervento dell'utente.


### Riepilogo dei Componenti di Spark in ordine di Granularità

* **RDD:** Dataset parallelo con partizioni.
* **DAG:** Grafo logico delle operazioni RDD.
* **Stage:** Insieme di task eseguiti in parallelo.
* **Task:** Unità fondamentale di esecuzione in Spark.




## Tolleranza ai Guasti in Spark

Gli RDD tracciano la loro lineage (serie di trasformazioni).  Questa informazione viene usata per ricalcolare i dati persi. Gli RDD sono memorizzati come una catena di oggetti che catturano la lineage.

![|281](_page_50_Figure_4.jpeg)

```scala
val file = sc.textFile("hdfs:// ... ")
val sics = file.filter (_. contains ("SICS"))
val cachedSics = sics.cache()
val ones = cachedSics.map(_ => 1)
val count = ones.reduce(_+_)
```

## Pianificazione dei Job in Spark

La pianificazione considera la disponibilità in memoria delle partizioni degli RDD persistenti.  Lo scheduler costruisce un DAG di stage dal grafo di lineage dell'RDD. Uno stage contiene trasformazioni in pipeline con dipendenze strette. Il confine di uno stage è definito dallo shuffle (per dipendenze ampie) e dalle partizioni già calcolate.

![|301](_page_51_Figure_7.jpeg)


Lo scheduler lancia i task per calcolare le partizioni mancanti di ogni stage fino a completare l'RDD di destinazione. I task sono assegnati alle macchine in base alla località dei dati; se una partizione è in memoria su un nodo, il task viene inviato a quel nodo.


## API DataFrame e Dataset

DataFrame e Dataset sono collezioni di dati immutabili distribuite, valutate in modo pigro, come gli RDD. I DataFrame (da Spark 1.3) introducono uno schema per descrivere i dati, organizzati in colonne denominate (simile a una tabella di database).  Funzionano su dati strutturati e semi-strutturati. Spark SQL permette query SQL su DataFrame. Da Spark 2.0, i DataFrame sono implementati come un caso speciale di Dataset.



I Dataset (da Spark 1.6) estendono i DataFrame con un'interfaccia orientata agli oggetti di tipo sicuro. Sono collezioni di oggetti JVM fortemente tipizzati, a differenza dei DataFrame che usano `Dataset[Row]` (Row è un oggetto generico).  `SparkSession` è il punto di ingresso per entrambe le API.


I Dataset combinano i vantaggi degli RDD (tipizzazione forte, funzioni lambda) con l'ottimizzatore Catalyst di Spark SQL. Sono disponibili in Scala e Java (non in Python e R), costruibili da oggetti JVM e manipolabili con trasformazioni funzionali (map, filter, flatMap, ...). Sono pigri; il calcolo avviene solo con un'azione. Internamente, un piano logico viene ottimizzato in un piano fisico per un'esecuzione efficiente.

Un Dataset può essere creato:

* Da un file usando `read`.
* Da un RDD esistente, convertendolo.
* Tramite trasformazioni su Dataset esistenti.

Esempio in Scala:

```scala
val names = people.map(_.name) //names è un Dataset[String]
```

Esempio in Java:

```java
Dataset<String> names = people.map((Person p) -> p.name, Encoders.STRING());
```


### DataFrame

Un DataFrame è un dataset organizzato in colonne denominate. Concettualmente è equivalente a una tabella in un database relazionale, ma con ottimizzazioni più ricche.  Come i Dataset, sfrutta l'ottimizzatore Catalyst. È disponibile in Scala, Java, Python e R. In Scala e Java, un DataFrame è rappresentato da un Dataset di `Row`.

Può essere costruito da:

* RDD esistenti, inferendo lo schema usando la riflessione o specificando programmaticamente lo schema.
* Tabelle in Hive.
* File di dati strutturati (JSON, Parquet, CSV, Avro).

Può essere manipolato in modi simili agli RDD.


## Spark Streaming

Spark Streaming è un'estensione che permette di analizzare dati in streaming, ingeriti e analizzati in micro-batch. Utilizza un'astrazione di alto livello chiamata DStream (discretized stream), che rappresenta un flusso continuo di dati: una sequenza di RDD.

![|373](_page_58_Figure_5.jpeg)
**Rappresentazione interna:;**
![|380](_page_59_Figure_2.jpeg)


### Spark MLlib

Spark MLlib fornisce molti algoritmi di machine learning distribuiti, tra cui: classificazione (es. regressione logistica), regressione, clustering (es. K-means), raccomandazione, alberi decisionali, foreste casuali e altri.

Fornisce inoltre utility per:

* **Machine Learning:** trasformazioni di feature, valutazione del modello e tuning degli iperparametri.
* **Algebra lineare distribuita:** (es. PCA) e statistica (es. statistiche riassuntive, test di ipotesi).

Adotta i DataFrame per supportare una varietà di tipi di dati.


####  Esempio

Questo esempio mostra come utilizzare Spark MLlib per la regressione logistica.  Il dataset contiene etichette e vettori di feature. L'obiettivo è imparare a predire le etichette dai vettori di feature.

```scala
// Ogni record di questo DataFrame contiene l'etichetta e
// le feature rappresentate da un vettore.
df = sqlContext.createDataFrame(data, ["label", "features"])
```

```scala
// Imposta i parametri per l'algoritmo.
// Qui, limitiamo il numero di iterazioni a 10.
lr = LogisticRegression(maxIter=10)
```

```scala
// Adatta il modello ai dati.
model = lr.fit(df)
```

```scala
// Dato un dataset, predici l'etichetta di ogni punto e mostra i risultati.
model.transform(df).show()
```
