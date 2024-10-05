# APACHE SPARK vs HADOOP MapReduce

**MapReduce**
- Facilita l'elaborazione di grandi quantità di dati in parallelo su cluster, garantendo affidabilità e tolleranza ai guasti.
- Gestisce la schedulazione dei task, il loro monitoraggio e la riesecuzione di quelli falliti.

**HDFS & MapReduce:**
- Esecuzione congiunta su nodi di calcolo e storage, mantenendo i dati vicini al calcolo -> throughput elevato.

**YARN & MapReduce:**
- Architettura con un resource manager master, node manager per nodo e AppMaster per applicazione.
---
## Debolezze e limitazioni di MapReduce

**Modello di programmazione**
- Difficoltà nell'implementare tutto come programma MapReduce.
- Operazioni semplici richiedono più passaggi MapReduce (es. WordCount con ordinamento per frequenza).
- Mancanza di strutture dati e tipi avanzati.
- Nessun supporto nativo per iterazioni, ogni iterazione legge/scrive su disco -> overhead elevato.
**Efficienza**
- Alto costo di comunicazione e frequente scrittura dell'output su disco.
- Utilizzo limitato della memoria.
**Non adatto per l'elaborazione in tempo reale**
- Necessità di scansionare l'intero input prima dell'elaborazione.
---
## Apache Spark

- Motore veloce e general-purpose per Big Data, non una modifica di Hadoop.
- Piattaforma leader per SQL su larga scala, elaborazione batch, stream e machine learning.
#### Caratteristiche principali:
- Elaborazione iterativa veloce grazie all'archiviazione dei dati in memoria -> almeno 10 volte più veloce di Hadoop.
- Adatto per grafi di esecuzione generali e ottimizzazioni avanzate.
- Compatibile con le API di storage di Hadoop, inclusi HDFS e HBase.

**Condivisione dei dati: MapReduce vs Spark**
- **MapReduce:** Lento per via di replicazione, serializzazione e I/O su disco.
- **Spark:** Distribuzione in memoria, 10-100 volte più veloce rispetto a disco e rete.

**Hadoop vs Spark: Flusso dei dati**
- **ETL (Extract, Transform, Load):** Processo di copia dei dati da diverse fonti a un sistema di destinazione che rappresenta i dati in modo diverso.

**Paradigma di programmazione**
- Sia Hadoop che Spark usano un paradigma simile a MapReduce: "scatter-gather", che distribuisce dati e calcolo su più nodi e raccoglie i risultati finali.

**Spark offre**:
- Modello dati più generale: RDDs, DataSets, DataFrames.
- Programmazione più user-friendly: Transformations e Actions.
- Agnostico rispetto allo storage (es. HDFS, Cassandra, S3, Parquet).

---
## Stack di Spark: Utilizzo dei componenti Hadoop

### Componenti di Elaborazione
- **Spark Streaming**: Elaborazione in tempo reale di flussi di dati.
- **Spark SQL**: Gestione ed esecuzione di query SQL su Big Data.
- **Spark ML**: Libreria per il machine learning.
- **Altre Applicazioni**: Ulteriori applicazioni basate su Spark.

### Spark Core
- Il cuore di Spark che gestisce le funzionalità fondamentali.

### Gestore delle risorse
- **Scheduler Standalone di Spark**: Gestore delle risorse nativo di Spark.
- **Mesos**: Gestore di risorse distribuito per ambienti multi-cluster.
- **YARN (Yet Another Resource Negotiator)**: Sistema di gestione delle risorse di Hadoop.
- **Altri gestori**: Possibilità di integrazione con altri cluster manager.

### Archiviazione dei Dati
- **Hadoop Distributed File System (HDFS)**: File system distribuito per l'archiviazione su larga scala.
- **Hadoop NoSQL Database (HBase)**: Database NoSQL ottimizzato per operazioni in tempo reale.

### Sistemi di Ingestion dei Dati
- **Apache Kafka**: Piattaforma di streaming per flussi di dati distribuiti.
- **Flume**: Servizio distribuito per la raccolta e l'aggregazione di grandi quantità di dati log.

---
## Integrazione di Spark nell'ecosistema Hadoop
Spark sfrutta componenti chiave di Hadoop e di altri progetti Apache per creare un ambiente di Big Data flessibile e scalabile. Nello stack Spark:

- Le librerie di elaborazione come **Spark Streaming**, **Spark SQL** e **Spark ML** sono costruite sopra **Spark Core**.
- **Spark Core** funge da motore principale per tutte le operazioni di elaborazione.
- Per la gestione delle risorse, Spark può utilizzare:
  - Il suo **scheduler standalone**.
  - Altri gestori di risorse come **Mesos** o **YARN**.
- Spark può accedere ai dati archiviati su **HDFS** o **HBase**.
- Per l'ingestion dei dati, Spark si integra con sistemi come **Kafka** e **Flume** per raccogliere e processare flussi di dati in tempo reale. 
---
### Data locality: Principio

- **Località dei dati**: Questo principio consiste nello spostare il calcolo vicino ai dati, piuttosto che trasferire grandi quantità di dati verso il calcolo. Ciò minimizza la congestione della rete e aumenta la produttività complessiva del sistema.
  
- Tuttavia, se il codice e i dati sono separati, uno dei due deve essere spostato. In genere, è più veloce trasferire il codice serializzato che un grande volume di dati, poiché il codice è solitamente di dimensioni molto minori rispetto ai dati. Spark basa la sua pianificazione su questo principio di località dei dati.

- **Eccezioni**: Non sempre è possibile rispettare il principio di località dei dati. In situazioni dove non ci sono dati non processati su nessun esecutore libero, bisogna prendere una decisione:
  1. Aspettare che un processore occupato si liberi per avviare il task sullo stesso server dove risiedono i dati, oppure
  2. Avviare subito un task su un altro nodo, spostando i dati necessari.

### Performance: Spark vs MapReduce

- **Algoritmi iterativi**:
  - Spark è più veloce di MapReduce, grazie a un flusso di dati semplificato.
  - Spark evita la materializzazione dei dati su HDFS dopo ogni iterazione.
  
- **Esempio: Algoritmo k-means, 1 iterazione**:
  - Lettura da HDFS
  - **Map**: Assegna i campioni al centroide più vicino.
  - **GroupBy(Centroid_ID)**: Raggruppa per ID del centroide.
  - Shuffle di rete.
  - **Reduce**: Calcola i nuovi centroidi.
  - Scrittura su HDFS.

### Codice: Hadoop vs Spark (esempio Word Count)

- **Hadoop (Word Count)**:
  ```java
  public class WordCount {
      public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
          private final static IntWritable one = new IntWritable(1);
          private Text word = new Text();
          
          public void map(LongWritable key, Text value, Context context) throws IOException {
              String line = value.toString();
              StringTokenizer tokenizer = new StringTokenizer(line);
              
              while (tokenizer.hasMoreTokens()) {
                  word.set(tokenizer.nextToken());
                  context.write(word, one);
              }
          }
      }
      
      public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
          public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException {
              int sum = 0;
              for (IntWritable val : values) {
                  sum += val.get();
              }
              context.write(key, new IntWritable(sum));
          }
      }
      
      public static void main(String[] args) throws Exception {
          Configuration conf = new Configuration();
          Job job = new Job(conf, "wordcount");
          job.setOutputKeyClass(Text.class);
          job.setOutputValueClass(IntWritable.class);
          job.setMapperClass(Map.class);
          job.setReducerClass(Reduce.class);
          job.setInputFormatClass(TextInputFormat.class);
          job.setOutputFormatClass(TextOutputFormat.class);
          
          FileInputFormat.addInputPath(job, new Path(args[0]));
          FileOutputFormat.setOutputPath(job, new Path(args[1]));
          job.waitForCompletion(true);
      }
  }
  ```

- **Spark (Word Count)**:
  ```scala
  val file = sc.textFile("hdfs://...")
  val counts = file.flatMap(line => line.split(" "))
                   .map(word => (word, 1))
                   .reduceByKey(_ + _)
  counts.saveAsTextFile("hdfs://...")
  ```
  - Codice semplice e conciso.
  - Pipeline in più fasi.
  - Operazioni:
    - **Transformations**: Applicano il codice dell'utente per distribuire i dati in parallelo.
    - **Actions**: Assemblano l'output finale dai dati distribuiti.

### Motivazione
- **MapReduce**: Il motore di elaborazione scalabile originale dell'ecosistema Hadoop.
  - Framework di elaborazione basato su disco (file HDFS).
  - I risultati intermedi vengono persistiti su disco.
  - I dati vengono ricaricati dal disco a ogni query, causando costose operazioni I/O.
  - Ideale per carichi di lavoro ETL (elaborazione batch).
  - Non adatto per algoritmi iterativi o elaborazione in streaming a causa delle I/O costose.

- **Spark**: Framework di elaborazione generale che migliora significativamente le prestazioni di MapReduce, mantenendo il modello di base.
  - Framework di elaborazione basato sulla memoria, evita I/O costose mantenendo i risultati intermedi in memoria.
  - Sfrutta la memoria distribuita.
  - Memorizza le operazioni applicate al dataset.
  - Calcolo basato sulla località dei dati, garantendo alte prestazioni.
  - Ottimo sia per carichi iterativi (o in streaming) che batch.
---
### Spark: Fondamenti

- **Spark Stack**:
  - Composto da **Spark Core** e varie librerie integrate.
  - Include:
    - L'**architettura di Spark**.
    - Il **modello di programmazione** basato su Resilient Distributed Datasets (RDD).
    - Il **flusso di dati** in Spark, che sfrutta il parallelismo distribuito.
    - La **Shell di Spark** per l'interazione diretta con l'engine.

- **Obiettivi di Spark**:
  - **Semplicità**: API intuitive e ricche per diversi linguaggi (Scala, Java, Python).
  - **Generalità**: API adatte a vari carichi di lavoro (batch, streaming, machine learning, grafi).
  - **Bassa latenza**: Elaborazione e caching in memoria per prestazioni elevate.
  - **Tolleranza ai guasti**: Il sistema gestisce i guasti in modo trasparente, senza interruzioni per l'utente.

---

### Spark Stack

- **Spark SQL**: Gestisce dati strutturati e supporta query SQL.
- **Spark Streaming**: Elaborazione in tempo reale di flussi di dati.
- **MLlib**: Libreria per machine learning distribuito.
- **GraphX**: API per la gestione e l'elaborazione di grafi distribuiti.
- **Spark Core**:
  - Cuore del sistema, gestisce task, memoria, tolleranza ai guasti e interazione con lo storage.
  - Introdotto il concetto di **RDD** (Resilient Distributed Dataset), che permette di distribuire e manipolare i dati su più nodi.
  - Scritto in **Scala**, con API anche per **Java**, **Python** e **R**.

---

### SPARK come motore unificato

- **Moduli ad alto livello**:
  - Spark integra vari moduli di alto livello, che possono essere combinati all'interno di una singola applicazione, garantendo flessibilità e scalabilità.

- **Spark SQL**:
  - Lavora con dati strutturati e supporta interrogazioni SQL.
  - Compatibile con varie fonti di dati come tabelle Hive, Parquet, JSON, ecc.
  - Estende l'API di RDD per includere funzionalità SQL.

- **Spark Streaming**:
  - Permette l'elaborazione in tempo reale dei flussi di dati.
  - Si basa sull'API RDD per gestire i dati provenienti da fonti di stream.

---

### Spark e le sue librerie

- **MLlib**:
  - Libreria scalabile per il machine learning.
  - Offre algoritmi distribuiti per estrazione di feature, classificazione, regressione, clustering e sistemi di raccomandazione.

- **GraphX**:
  - API per manipolare grafi distribuiti e calcolare metriche parallele su di essi.
  - Contiene algoritmi di grafi comuni come PageRank.
  - Basato e integrato con l'API RDD.

---

### Spark sopra il cluster manager

- Spark può essere eseguito su diversi **gestori di risorse cluster**, tra cui:
  - **Modalità standalone di Spark**: Con un semplice scheduler FIFO.
  - **Hadoop YARN**: Utilizza il gestore delle risorse di Hadoop per distribuire i task.
  - **Mesos**: Originato dallo stesso laboratorio di Spark (AMPLab @ UC Berkeley), fornisce gestione avanzata dei cluster.
  - **Kubernetes**: Supporta l'esecuzione di Spark in ambienti containerizzati.



---

### Architettura di SPARK

- Ogni applicazione consiste in un **programma driver** e degli **executor** sul cluster.
  - **Driver program**: Il processo che esegue la funzione `main()` dell'applicazione e crea l'oggetto `SparkContext`.
  
- Ogni applicazione ha i propri **executor**, che sono processi che rimangono attivi per tutta la durata dell'applicazione e eseguono task in più thread, garantendo l'isolamento delle applicazioni concorrenti.

- Per eseguire un'applicazione su un cluster, il `SparkContext` si collega a un cluster manager, che alloca le risorse necessarie.

- Una volta connesso, Spark acquisisce gli executor sui nodi del cluster e invia loro il codice dell'applicazione (ad es. il file `.jar`).

- Infine, il `SparkContext` invia i task agli executor per essere eseguiti.

---

### Modello di programmazione SPARK

- **Driver program**: Comunica con il gestore del cluster.
- **Nodo di lavoro**: Dove gli executor eseguono i task.

- **RDD (Resilient Distributed Dataset)**: 
  - Struttura dati immutabile, distribuita e tollerante ai guasti.
  - In memoria (esplicitamente), con partizionamento controllato per ottimizzare la collocazione dei dati.
  - Può essere manipolata usando un ricco set di operatori.

---

### Programmazione in SPARK

- Ci sono due modalità principali per manipolare i dati in Spark:
  1. **Spark Shell**:
     - Interattivo, utile per apprendere o esplorare i dati.
     - Supporta Python o Scala.
  
  2. **Spark Applications**:
     - Per l'elaborazione di grandi quantità di dati.
     - Supporta Python, Scala o Java.

---

### Spark Shell

- La shell Spark offre un'esplorazione interattiva dei dati (REPL: Repeat, Evaluate, Print, Loop).

- **Shell Python**: `pyspark`
  ```bash
  $ pyspark
  ```

- **Shell Scala**: `spark-shell`
  ```bash
  $ spark-shell
  ```

---

### Spark Context

- Ogni applicazione Spark richiede un **Spark Context**, che è il punto di accesso principale all'API di Spark.
- La Spark Shell fornisce un **Spark Context** preconfigurato, chiamato `sc`.
  
- **Standalone applications**: Il codice driver utilizza lo Spark Context.
- Lo **Spark Context** funziona come un client e rappresenta la connessione a un cluster Spark.

---
### RDD (Resilient Distributed Dataset)  

L'**RDD (Resilient Distributed Dataset)** è l'unità fondamentale dei dati in Spark. È una **collezione immutabile** di oggetti (o record, o elementi) che possono essere elaborati "in parallelo" (distribuiti su un cluster).

- **Resiliente**: se i dati in memoria vengono persi, possono essere ricreati, ovvero Spark è tollerante ai guasti.
  - Recupera da guasti ai nodi.
  - Un RDD conserva le informazioni di lineage, per cui può essere ricostruito a partire dagli RDD genitori.
  
- **Distribuito**: gli RDD vengono elaborati in parallelo nel cluster.
  - Ogni RDD è composto da una o più partizioni (più partizioni -> maggiore parallelismo).

- **Dataset**: i dati iniziali possono provenire da un file o essere creati programmaticamente.

---

### RDD: Concetti principali

- Scrivere le applicazioni in termini di **trasformazioni sui dataset distribuiti**.
  - Le collezioni di oggetti sono distribuite in un layer di cache in memoria tollerante ai guasti.
  - Possono usare il disco se i dataset non entrano in memoria.
  - Creati tramite trasformazioni parallele (es. `map`, `filter`, `group-by`, `join`, ecc.).
  - Ricostruiti automaticamente in caso di guasto.
  - Possibilità di controllare la persistenza (ad es. caching in RAM).

---

### Immutabilità degli RDD

- Gli RDD sono **immutabili** una volta creati, perciò non possono essere modificati.
  - Un nuovo RDD può essere creato a partire da uno esistente.
  
- Sono ricostruiti automaticamente in caso di guasto, senza replicazione, tracciando le informazioni di **lineage**.
  - Spark conosce come un RDD è stato costruito e può ricostruirlo in caso di guasto.
  - Queste informazioni sono rappresentate tramite un **DAG** di lineage che collega i dati di input e gli RDD.

---

### Distribuzione degli RDD

- Spark gestisce la suddivisione degli RDD in **partizioni** e assegna queste partizioni ai nodi del cluster.
  - Le partizioni degli RDD possono essere distribuite su diversi nodi del cluster.

- In caso di guasto, Spark ricostruisce automaticamente gli RDD utilizzando il **DAG di lineage**.

---

### API degli RDD

- Spark offre una API pulita e integrata con linguaggi come **Scala, Python, Java** e **R**.
- Gli RDD possono essere creati e manipolati attraverso:
  - **Trasformazioni coarse-grained**: definiscono un nuovo dataset basato su quelli precedenti (es. `map`, `filter`, `join`).
  - **Azioni**: avviano un job da eseguire sul cluster (es. `count`, `collect`, `save`).

---

### Modello di programmazione con RDD

- Basato su **operatori parallelizzabili**, cioè funzioni di ordine superiore che eseguono funzioni definite dall'utente in parallelo.
- Il flusso di dati è composto da **sorgenti di dati, operatori e sink** collegati tra loro.
- La descrizione di un job è basata su un **grafo aciclico diretto (DAG)**.

---

### Creazione degli RDD

Gli RDD possono essere creati in diversi modi:
  
1. **Parallelizzando collezioni esistenti** nel linguaggio di programmazione ospitante (ad es. collezioni e liste in Scala, Java, Python, o R).
   - L'utente può specificare il numero di partizioni.

2. **Da file (grandi) archiviati** in HDFS o in altri file system.
   - Una partizione per ogni blocco HDFS.

3. **Trasformando un RDD esistente**.
   - Il numero di partizioni dipende dal tipo di trasformazione.

---

### Esempi di utilizzo degli RDD

- **Parallelizzare una collezione**:
  ```python
  lines = sc.parallelize(["pandas", "i like pandas"])
  ```
  - `sc` è la variabile Spark context.
  - Parametro importante: numero di partizioni.
  - Spark eseguirà un task per ogni partizione nel cluster.
  
- **Caricare dati da un file**:
  ```python
  lines = sc.textFile("/path/to/README.md")
  ```
---
### Trasformazioni RDD

Esistono due tipi di operazioni su un RDD:
1. **Trasformazioni**: definiscono un nuovo RDD basato sugli RDD esistenti.
2. **Azioni**: restituiscono dei valori.

#### Esempio di codice:
```scala
val sc = new SparkContext("spark", "MyJob", home, jars)
val file = sc.textFile("hdfs://...")  // Questo è un RDD
val errors = file.filter(_.contains("ERROR"))  // Questo è un RDD
errors.cache()
errors.count()  // Questa è un'azione
```

---

### Trasformazioni RDD
- Insieme di operazioni su un RDD che ne definiscono la trasformazione.
- Come nell'algebra relazionale, l'applicazione di una trasformazione ad un RDD genera un nuovo RDD (poiché gli RDD sono immutabili).
- Le trasformazioni sono valutate in modo **pigro** (*lazy evaluation*), consentendo ottimizzazioni prima dell'esecuzione.
- Esempi di trasformazioni: `map()`, `filter()`, `groupByKey()`, `sortByKey()`, ecc.

---

### Esempio di trasformazioni: `map` e `filter`

#### Azioni sugli RDD
- Si applicano catene di trasformazioni su RDDs, eseguendo poi azioni aggiuntive (es. conteggio).
- Alcune azioni memorizzano i dati in una fonte esterna (es. HDFS), altre estraggono i dati dall'RDD e li restituiscono al driver.
- **Azioni comuni**:
  - `count()`: restituisce il numero di elementi.
  - `take(n)`: restituisce un array con i primi n elementi.
  - `collect()`: restituisce un array con tutti gli elementi.
  - `saveAsTextFile(file)`: salva gli RDD in file di testo.

---

### Esecuzione lazy degli RDD
- I dati contenuti negli RDD non vengono elaborati fino a quando non viene eseguita un'azione.

#### Esempio:
```python
lines = sc.textFile("purplecow.txt")
errors = lines.filter(lambda line: "ERROR" in line)
errors.count()  # L'elaborazione viene avviata solo quando si chiama un'azione
```

---

### Creazione di RDD
- **Convertire una collezione Python in un RDD**:
  ```python
  sc.parallelize([1, 2, 3])
  ```
- **Caricare file di testo** dal filesystem locale, HDFS o S3:
  ```python
  sc.textFile("file.txt")
  sc.textFile("directory/*.txt")
  sc.textFile("hdfs://namenode:9000/path/file")
  ```

---

### Funzioni di ordine superiore
- Le trasformazioni e le azioni disponibili sugli RDD in Spark sono implementate come **funzioni di ordine superiore**.
  - **Seq[T]**: sequenza di elementi di tipo T.

#### Trasformazioni:
- `map(f: T => U)`: applica una funzione a ogni elemento.
- `filter(f: T => Bool)`: restituisce solo gli elementi che soddisfano una condizione.
- `flatMap(f: T => Seq[U])`: simile a `map`, ma può restituire 0 o più elementi.
- `groupByKey()`: raggruppa i dati per chiave.
- `reduceByKey(f: (V, V) => V)`: riduce gli elementi con la stessa chiave.
  
#### Azioni:
- `count()`: conta gli elementi.
- `collect()`: restituisce tutti gli elementi.
- `reduce(f: (T, T) => T)`: riduce l'RDD a un singolo valore.
- `lookup(k: K)`: cerca il valore associato a una chiave.

---
### Esempi di trasformazioni comuni

| **Trasformazione**                   | **Significato**                                                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `map(func)`                          | Restituisce un nuovo RDD applicando `func` ad ogni elemento dell'RDD sorgente.                                                            |
| `filter(func)`                       | Restituisce un nuovo RDD con gli elementi che soddisfano la condizione di `func`.                                                         |
| `flatMap(func)`                      | Simile a `map`, ma `func` può restituire 0 o più elementi (una sequenza).                                                                 |
| `mapPartitions(func)`                | Simile a `map`, ma opera separatamente su ogni partizione dell'RDD.                                                                       |
| `union(otherDataset)`                | Restituisce un nuovo dataset con l'unione degli elementi dell'RDD sorgente e del dataset fornito.                                         |
| `distinct([numTasks])`               | Restituisce un nuovo RDD con gli elementi distinti dell'RDD sorgente.                                                                     |
| `groupByKey([numTasks])`             | Quando chiamato su un dataset di coppie (K, V), restituisce un dataset di coppie (K, Seq[V]).                                             |
| `reduceByKey(func, [numTasks])`      | Quando chiamato su un dataset di coppie (K, V), restituisce un dataset di coppie (K, V) aggregato usando la funzione di riduzione `func`. |
| `sortByKey([ascending], [numTasks])` | Quando chiamato su un dataset di coppie (K, V), restituisce un dataset ordinato per chiave in ordine ascendente o discendente.            |
| `join(otherDataset, [numTasks])`     | Quando chiamato su dataset di tipo (K, V) e (K, W), restituisce un dataset di coppie (K, (V, W)) per ogni chiave.                         |
| `cogroup(otherDataset, [numTasks])`  | Quando chiamato su dataset di tipo (K, V) e (K, W), restituisce un dataset di tuple (K, Seq[V], Seq[W]).                                  |
| `cartesian(otherDataset)`            | Quando chiamato su dataset di tipi T e U, restituisce un dataset di coppie (T, U) (tutte le possibili combinazioni tra gli elementi).     |

## Tipi di Trasformazioni

Ci sono due tipi di trasformazioni:

1. **Trasformazione Stretta (Narrow transformation):** tutti gli elementi necessari per calcolare i record in una singola partizione vivono nella singola partizione del RDD genitore. Un sottoinsieme limitato di partizioni viene utilizzato per calcolare il risultato.

2. **Trasformazione Ampia (Wide transformation):** tutti gli elementi necessari per calcolare i record in una singola partizione possono vivere in molte partizioni del RDD genitore. La partizione può vivere in molte partizioni del RDD genitore.

### Esempi di Trasformazioni Ampie
- Intersection
- Distinct
- ReduceByKey
- GroupByKey
- Join
- Cartesian
- Repartition
- Coalesce

# Azioni RDD

| **Operazione**                           | **Significato**                                                                                                                                                                     |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reduce(func)`                           | Aggrega gli elementi del dataset usando una funzione `func` (commutativa e associativa) per il calcolo parallelo.                                                                   |
| `collect()`                              | Restituisce tutti gli elementi del dataset come un array al programma driver, utile per dataset di piccole dimensioni dopo operazioni come `filter`.                                |
| `count()`                                | Restituisce il numero di elementi nel dataset.                                                                                                                                      |
| `first()`                                | Restituisce il primo elemento del dataset (simile a `take(1)`).                                                                                                                     |
| `take(n)`                                | Restituisce un array con i primi `n` elementi del dataset. Questo non viene eseguito in parallelo; il driver calcola tutti gli elementi.                                            |
| `takeSample(withReplacement, num, seed)` | Restituisce un campione casuale di `num` elementi dal dataset, con o senza sostituzione, usando il seme del generatore di numeri casuali fornito.                                   |
| `saveAsTextFile(path)`                   | Scrive gli elementi del dataset come file di testo (o set di file) in una directory nel filesystem locale, HDFS o altri filesystem supportati da Hadoop.                            |
| `saveAsSequenceFile(path)`               | Scrive gli elementi del dataset come Hadoop SequenceFile in un percorso specificato. Disponibile solo su RDD di coppie chiave-valore che implementano Writable o tipi convertibili. |
| `countByKey()`                           | Disponibile su RDD di tipo (K, V). Restituisce una mappa con il conteggio di ogni chiave.                                                                                           |
| `foreach(func)`                          | Esegue una funzione `func` su ogni elemento del dataset, spesso usata per effetti collaterali come aggiornamenti a variabili accumulatore o interazioni con sistemi esterni.        |
# Trasformazioni di Base

```python
nums = sc.parallelize([1, 2, 3])

# Passa ogni elemento attraverso una funzione
squares = nums.map(lambda x: x*x) # {1, 4, 9}

# Mantiene gli elementi che passano un predicato
even = squares.filter(lambda x: x % 2 == 0) # {4}

# Mappa ogni elemento a zero o più altri
nums.flatMap(lambda x: range(x))
# => {0, 0, 1, 0, 1, 2}
```

## Trasformazioni RDD: map e filter

```python
# Trasformazione di ogni elemento attraverso una funzione
nums = sc.parallelize([1, 2, 3, 4])
squares = nums.map(lambda x: x * x) # [1, 4, 9, 16]

# Selezione degli elementi per cui func restituisce true
even = squares.filter(lambda num: num % 2 == 0) # [4, 16]
```

## Trasformazioni RDD: flatMap

```python
# Mappatura di ogni elemento a zero o più altri
ranges = nums.flatMap(lambda x: range(0, x, 1))
# [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

# Divisione delle linee di input in parole
lines = sc.parallelize(["hello world", "hi"])
words = lines.flatMap(lambda line: line.split(" "))
# ['hello', 'world', 'hi']
```

# Lavorare con Coppie Chiave-Valore

Spark utilizza trasformazioni "distributed reduce" che operano su RDD di coppie chiave-valore.

```python
# Python
pair = (a, b)
pair[0] # => a
pair[1] # => b

# Scala
val pair = (a, b)
pair._1 // => a
pair._2 // => b

# Java
Tuple2 pair = new Tuple2(a, b);
pair._1 // => a
pair._2 // => b
```

## Alcune Operazioni Chiave-Valore

```python
pets = sc.parallelize([("cat", 1), ("dog", 1), ("cat", 2)])

pets.reduceByKey(lambda x, y: x + y)
# => {(cat, 3), (dog, 1)}

pets.groupByKey() # => {(cat, [1, 2]), (dog, [1])}

pets.sortByKey() # => {(cat, 1), (cat, 2), (dog, 1)}
```

reduceByKey implementa anche automaticamente i combinatori sul lato map.

## Trasformazioni RDD: join

```python
users = sc.parallelize([(0, "Alex"), (1, "Bert"), (2, "Curt"), (3, "Don")])
hobbies = sc.parallelize([(0, "writing"), (0, "gym"), (1, "swimming")])
users.join(hobbies).collect()
# [(0, ('Alex', 'writing')), (0, ('Alex', 'gym')), (1, ('Bert', 'swimming'))]
```

## Trasformazioni RDD: reduceByKey

```python
x = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("a", 1),
                    ("b", 1), ("b", 1), ("b", 1), ("b", 1)], 3)
# Applicazione dell'operazione reduceByKey
y = x.reduceByKey(lambda accum, n: accum + n)
# [('b', 5), ('a', 3)]
```


# Impostazione del Livello di Parallelismo

Tutte le operazioni su RDD di coppie accettano un secondo parametro opzionale per il numero di task:

```python
words.reduceByKey(lambda x, y: x + y, 5)
words.groupByKey(5)
visits.join(pageviews, 5)
```

# Alcune Azioni RDD

- **collect**: restituisce tutti gli elementi dell'RDD come una lista

```python
nums = sc.parallelize([1, 2, 3, 4])
nums.collect() # [1, 2, 3, 4]
```

- **take**: restituisce un array con i primi n elementi nell'RDD

```python
nums.take(3) # [1, 2, 3]
```

- **count**: restituisce il numero di elementi nell'RDD

```python
nums.count() # 4
```

- **reduce**: aggrega gli elementi nell'RDD usando la funzione specificata

```python
sum = nums.reduce(lambda x, y: x + y)
```

- **saveAsTextFile**: scrive gli elementi dell'RDD come file di testo nel file system locale o HDFS

```python
nums.saveAsTextFile("hdfs://file.txt")
```

# Trasformazioni Lazy

- Le trasformazioni sono lazy: non vengono calcolate finché un'azione non richiede che un risultato sia restituito al programma driver
- Questo design permette a Spark di eseguire operazioni più efficientemente, poiché le operazioni possono essere raggruppate
  - Es. se ci fossero più operazioni di filter o map, Spark può fonderle in un unico passaggio
  - Es. se Spark sa che i dati sono partizionati, può evitare di spostarli sulla rete per groupBy

# Esempio: WordCount in Scala

```scala
val textFile = sc.textFile("hdfs://...")
val words = textFile.flatMap(line => line.split(" "))
val ones = words.map(word => (word, 1))
val counts = ones.reduceByKey(_ + _) // Equivalente a ones.reduceByKey((a, b) => a + b)
counts.saveAsTextFile("hdfs://...")
```

[Immagine del flusso di dati]

# Esempio: WordCount in Scala con concatenazione

```scala
val textFile = sc.textFile("hdfs://...")
val counts = textFile.flatMap(line => line.split(" "))
                     .map(word => (word, 1))
                     .reduceByKey(_ + _)
counts.saveAsTextFile("hdfs://...")
```

# Esempio: WordCount in Python

```python
text_file = sc.textFile("hdfs://...")
counts = text_file.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a + b)
output = counts.collect()
output.saveAsTextFile("hdfs://...")
```

# Esempio: WordCount in Java 7

```java
JavaRDD<String> textFile = sc.textFile("hdfs://...");
JavaRDD<String> words = textFile.flatMap(new FlatMapFunction<String, String>() {
    public Iterable<String> call(String s) { return Arrays.asList(s.split(" ")); }
});
JavaPairRDD<String, Integer> pairs = words.mapToPair(new PairFunction<String, String, Integer>() {
    public Tuple2<String, Integer> call(String s) { return new Tuple2<String, Integer>(s, 1); }
});
JavaPairRDD<String, Integer> counts = pairs.reduceByKey(new Function2<Integer, Integer, Integer>() {
    public Integer call(Integer a, Integer b) { return a + b; }
});
counts.saveAsTextFile("hdfs://...");
```

Nota: PairRDD sono RDD contenenti coppie chiave/valore. L'API Java di Spark permette di creare tuple usando la classe scala.Tuple2.

# Esempio: WordCount in Java 8

```java
JavaRDD<String> textFile = sc.textFile("hdfs://...");
JavaPairRDD<String, Integer> counts = textFile
    .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
    .mapToPair(word -> new Tuple2<>(word, 1))
    .reduceByKey((a, b) -> a + b);
counts.saveAsTextFile("hdfs://...");
```

# Inizializzazione di Spark: SparkContext

- Il primo passo in un programma Spark è creare un oggetto SparkContext, che è il punto di ingresso principale per le funzionalità Spark
  - Rappresenta la connessione al cluster Spark, può essere usato per creare RDD su quel cluster
  - Disponibile anche nella shell, nella variabile chiamata sc
- Solo un SparkContext può essere attivo per JVM
  - Fermare (stop()) lo SparkContext attivo prima di crearne uno nuovo
- L'oggetto SparkConf: configurazione per un'applicazione Spark
  - Usato per impostare vari parametri Spark come coppie chiave-valore

```scala
val conf = new SparkConf().setAppName(appName).setMaster(master)
new SparkContext(conf)
```

# Esempio: WordCount in Java (completo)

```java
package org.apache.spark.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public final class WordCount {
    private static final Pattern SPACE = Pattern.compile(" ");

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: WordCount <file>");
            System.exit(1);
        }

        final SparkConf sparkConf = new SparkConf().setAppName("WordCount");
        final JavaSparkContext ctx = new JavaSparkContext(sparkConf);
        final JavaRDD<String> lines = ctx.textFile(args[0], 1);

        final JavaRDD<String> words = lines.flatMap(s -> Arrays.asList(SPACE.split(s)));
        final JavaPairRDD<String, Integer> ones = words.mapToPair(s -> new Tuple2<>(s, 1));
        final JavaPairRDD<String, Integer> counts = ones.reduceByKey((i1, i2) -> i1 + i2);

        final List<Tuple2<String, Integer>> output = counts.collect();
        for (Tuple2<?, ?> tuple : output) {
            System.out.println(tuple._1() + ": " + tuple._2());
        }
        ctx.stop();
    }
}
```

# Esempio: Numero Primo

```scala
nums = sc.parallelize(range(1000000))
// Calcola il numero di primi nell'RDD
print(nums.filter(isPrime).count())

def isPrime(i: Int): Boolean =
    if (i <= 1)
        false
    else if (i == 2)
        true
    else
        !(2 until i).exists(n => i % n == 0)
```

# Stima di Pi

```scala
val count = sc.parallelize(1 to NUM_SAMPLES)
    .filter { _ =>
        val x = math.random
        val y = math.random
        x*x + y*y < 1
    }.count()
println(s"Pi è approssimativamente ${4.0 * count / NUM_SAMPLES}")
```

```java
List<Integer> l = new ArrayList<>(NUM_SAMPLES);
for (int i = 0; i < NUM_SAMPLES; i++) {
    l.add(i);
}
long count = sc.parallelize(l).filter(i -> {
    double x = Math.random();
    double y = Math.random();
    return x*x + y*y < 1;
}).count();
System.out.println("Pi è approssimativamente " + 4.0 * count / NUM_SAMPLES);
```

Nota: Questo metodo utilizza il metodo Monte Carlo per stimare il valore di Pi. Grazie al numero molto grande e alla distribuzione casuale, possiamo approssimare la misura delle aree con il numero di punti contenuti in esse.

### Persistenza RDD
- Una delle capacità più importanti in Spark è la persistenza (o caching) di un dataset in memoria durante le operazioni.
- Quando si persiste un RDD, ogni nodo memorizza in memoria le sue fette (slices) calcolate e le riutilizza in altre azioni sullo stesso dataset (o su dataset derivati).
- Questo consente che le azioni future siano notevolmente più veloci (spesso di oltre 10 volte). Il caching è uno strumento chiave per costruire algoritmi iterativi con Spark e per l'uso interattivo dall'interprete.
- Puoi contrassegnare un RDD da persistere usando i metodi `persist()` o `cache()`. La prima volta che viene calcolato in un'azione, verrà mantenuto in memoria sui nodi. La cache è tollerante ai guasti: se una partizione di un RDD viene persa, verrà automaticamente ricalcolata utilizzando le trasformazioni che l'hanno originariamente creata.

---

### Quale Livello di Memorizzazione Scegliere?
- Se i tuoi RDD si adattano comodamente con il livello di memorizzazione predefinito (**MEMORY_ONLY**), lasciali così. Questa è l'opzione più efficiente in termini di CPU, consentendo alle operazioni sugli RDD di essere eseguite il più velocemente possibile.
- Se non si adattano, prova a usare **MEMORY_ONLY_SER** e seleziona una libreria di serializzazione veloce per rendere gli oggetti molto più efficienti in termini di spazio, ma comunque ragionevolmente veloci da accedere.
- Non scrivere su disco a meno che le funzioni che hanno calcolato i tuoi dataset non siano costose, o se filtrano una grande quantità di dati. Altrimenti, il ricalcolo di una partizione è pressoché veloce quanto la lettura dal disco.
- Usa i livelli di memorizzazione replicati se desideri un rapido recupero da guasti (ad es. se utilizzi Spark per servire richieste da un'applicazione web). Tutti i livelli di memorizzazione forniscono tolleranza ai guasti ricompilando i dati persi, ma quelli replicati ti permettono di continuare a eseguire compiti sull'RDD senza attendere il ricalcolo di una partizione persa.

---

### Esempio: Estrazione di Log (uso della cache)
Carica i messaggi di errore da un log in memoria, quindi cerca interattivamente vari modelli:
```python
lines = spark.textFile("hdfs://HadoopRDD")
errors = lines.filter(lambda s: s.startswith("ERROR"))  # RDD filtrato
messages = errors.map(lambda s: s.split("\t")[2])
messages.cache()
messages.filter(lambda s: "foo" in s).count()
```

#### Diagramma di Flusso
- **RDD**
  - **Block 1**
  - **Driver**
    - `lines = spark.textFile("hdfs://...")`
    - `errors = lines.filter(_.startswith("ERROR"))`
    - `messages = errors.map(_.split('\t')(2))`
    - `cachedMsgs = messages.cache()`
- **Worker**
  - `cachedMsgs.filter(_.contains("foo")).count()`
  - `cachedMsgs.filter(_.contains("bar")).count()`

---

### Comportamento senza Cache
- Se non si utilizza la cache, ogni operazione deve rielaborare i dati.
  
### Comportamento con Cache
- (*) Il comportamento della cache dipende dalla memoria disponibile. Nel nostro esempio, se il file non si adatta alla memoria, l'operazione `lines.count` seguirà il comportamento usuale e rileggerà il file.

---

### Requisiti
- **Java 8+**: utilizzeremo l'interfaccia Function di Java 8 e le espressioni lambda.
- **Eclipse**: un IDE per lo sviluppo Java/JavaEE.

---

### Requisiti (Java/Spark)
- **Java 8 + IntelliJ (o Eclipse)**
- Scarica l'eseguibile **winutils** e **hadoop.dll** dal repository Hortonworks:
  - [winutils](https://github.com/cdarlint/winutils)
- Crea una directory dove posizionare l'eseguibile scaricato **winutils.exe**. Ad esempio: `C:\Hadoop\bin`.
- Aggiungi la variabile d'ambiente **HADOOP_HOME** che punta a `C:\Hadoop`. Aggiungi `C:\Hadoop\bin` a **PATH**. Maggiori informazioni su [Install Spark on Windows 10](https://phoenixnap.com/kb/install-spark-on-windows-10).

---
### Requisiti (Java/Spark)
- **Pom.xml**
```xml
<dependency>
  <groupId>org.apache.spark</groupId>
  <artifactId>spark-core_2.12</artifactId>
  <version>2.4.4</version>
</dependency>
```
- **HDFS (opzionale)**
```xml
<dependency>
  <groupId>org.apache.hadoop</groupId>
  <artifactId>hadoop-client</artifactId>
  <version><your-hdfs-version></version>
</dependency>
```
---
# FAQ su Apache Spark

## 1. Qual è la differenza tra Apache Spark e Hadoop MapReduce?

Sebbene sia Apache Spark che Hadoop MapReduce siano framework di elaborazione distribuita progettati per gestire grandi set di dati, differiscono in termini di prestazioni, modello di programmazione e casi d'uso ideali.

**Hadoop MapReduce**:
- Framework di elaborazione basato su disco che archivia i risultati intermedi su disco (HDFS) dopo ogni attività Map o Reduce.
- Adatto per carichi di lavoro batch (ETL) su larga scala in cui la latenza non è un problema critico.
- Il modello di programmazione è meno flessibile e può essere prolisso per operazioni complesse.

**Apache Spark**:
- Framework di elaborazione in memoria che archivia i risultati intermedi in memoria, con conseguenti guadagni di prestazioni significativi, in particolare per attività iterative o in tempo reale.
- Adatto per un'ampia gamma di casi d'uso, tra cui elaborazione batch, elaborazione di flussi, query interattive, machine learning e analisi di grafi.
- Offre un modello di programmazione più ricco e intuitivo con API di livello superiore.

## 2. In che modo Spark raggiunge velocità superiori rispetto a MapReduce?

Spark ottiene velocità superiori rispetto a MapReduce principalmente grazie alla sua architettura in memoria e all'esecuzione ottimizzata.

- **Elaborazione in memoria**: Spark elabora i dati in memoria, mentre MapReduce si basa su operazioni su disco. L'accesso alla memoria è significativamente più veloce dell'I/O su disco, con conseguenti tempi di elaborazione ridotti.
- **Esecuzione DAG (Directed Acyclic Graph)**: Spark utilizza un DAG per rappresentare il flusso di lavoro dell'applicazione, consentendo ottimizzazioni come il concatenamento delle attività e l'esecuzione parallela, riducendo al minimo il sovraccarico e la latenza.
- **Caching dei dati**: Spark può memorizzare nella cache i set di dati utilizzati frequentemente in memoria, consentendo un accesso più rapido nelle successive iterazioni o operazioni.

## 3. Cosa sono gli RDD (Resilient Distributed Datasets) in Spark?

Un RDD è una struttura dati immutabile, distribuita e tollerante ai guasti che costituisce l'elemento fondamentale dei dati in Spark.

- **Immutabile**: una volta creato un RDD, non può essere modificato. Le trasformazioni creano nuovi RDD basati su quelli esistenti.
- **Distribuito**: gli RDD vengono partizionati e distribuiti su più nodi in un cluster, consentendo l'elaborazione parallela.
- **Tollerante ai guasti**: Spark può ricostruire gli RDD persi o danneggiati grazie al suo DAG di derivazione, garantendo la resilienza ai guasti hardware o software.

## 4. Quali linguaggi di programmazione sono supportati da Spark?

Spark supporta diversi linguaggi di programmazione, tra cui:

- **Scala**: il linguaggio principale in cui è scritto Spark e offre l'integrazione più nativa.
- **Java**: un'API Java completa e popolare per gli sviluppatori Java.
- **Python**: PySpark, l'API Python, è ampiamente utilizzata per la sua semplicità e le sue ampie librerie.
- **R**: un'API R è disponibile per gli utenti R per sfruttare le capacità di Spark.

## 5. Quali sono i componenti principali dello stack Spark?

Lo stack Spark comprende diversi componenti che lavorano insieme per fornire analisi Big Data complete:

- **Spark Core**: il motore di esecuzione fondamentale che gestisce la gestione della memoria, la pianificazione delle attività, la tolleranza ai guasti e le interazioni con i sistemi di archiviazione.
- **Spark SQL**: consente l'elaborazione di dati strutturati utilizzando query simili a SQL, fornendo un'interfaccia familiare agli utenti di database.
- **Spark Streaming**: consente l'elaborazione di flussi di dati in tempo reale da varie origini, come Kafka e Flume.
- **MLlib**: una libreria scalabile di machine learning con algoritmi e utilità per attività di apprendimento supervisionato e non supervisionato.
- **GraphX**: un'API per l'elaborazione di grafi e l'esecuzione di analisi di grafi su larga scala.

## 6. In che modo Spark gestisce la tolleranza ai guasti?

Spark raggiunge la tolleranza ai guasti attraverso diverse caratteristiche:

- **RDD immutabili**: gli RDD non possono essere modificati, garantendo che un'operazione non riuscita non corrompa i dati originali.
- **DAG di derivazione**: Spark tiene traccia delle trasformazioni utilizzate per creare un RDD, consentendo la ricostruzione di partizioni perse o danneggiate.
- **Scrittura ridondante di dati in linea**: Spark può replicare i dati in linea, garantendo che anche se un nodo si guasta, i dati siano ancora disponibili su altri nodi.

## 7. Quali sono i diversi livelli di persistenza RDD in Spark?

Spark fornisce diversi livelli di persistenza per controllare come gli RDD vengono memorizzati nella cache:

- **MEMORY_ONLY**: archivia gli RDD solo in memoria, offrendo le prestazioni più veloci ma potenzialmente soggetto a perdita di dati in caso di guasti ai nodi.
- **MEMORY_AND_DISK**: archivia gli RDD in memoria e su disco, fornendo un compromesso tra prestazioni e tolleranza ai guasti.
- **DISK_ONLY**: archivia gli RDD solo su disco, adatto per set di dati molto grandi che non possono essere contenuti in memoria.

## 8. Spark può essere integrato con Hadoop?

Sì, Spark può essere strettamente integrato con l'ecosistema Hadoop.

- **Utilizzo di HDFS**: Spark può leggere e scrivere dati da Hadoop Distributed File System (HDFS), consentendo l'elaborazione di dati archiviati in cluster Hadoop.
- **Utilizzo di YARN**: Spark può essere eseguito su Hadoop YARN (Yet Another Resource Negotiator), consentendogli di condividere risorse con altre applicazioni Hadoop.
- **Utilizzo di librerie Hadoop**: Spark può utilizzare librerie Hadoop esistenti per l'accesso e l'elaborazione di vari formati di dati, come file di testo, file di sequenza e dati Avro.
---
# Riassunto

Apache Spark si è affermato come una valida alternativa a Hadoop MapReduce, offrendo prestazioni superiori, semplicità e una gamma di casi d'uso più ampia, inclusi batch, stream e machine learning. Spark è particolarmente adatto per carichi di lavoro iterativi e in tempo reale, grazie alla sua capacità di elaborare in memoria e scalare facilmente su cluster di grandi dimensioni.

---
## Hadoop MapReduce

### Punti di forza:
- **Elaborazione parallela**: gestisce grandi volumi di dati distribuendoli in parallelo su cluster.
- **Gestione dei task**: schedulazione, monitoraggio e riesecuzione dei task falliti.
- **Località dei dati**: HDFS e MapReduce eseguono operazioni sui nodi che contengono i dati, migliorando il throughput.
- **Architettura YARN**: introduce il **resource manager master**, il **node manager** per nodo e l'**AppMaster** per applicazione, migliorando la gestione delle risorse.

### Debolezze:
- **Complessità di programmazione**: il modello MapReduce è complesso e richiede più passaggi per operazioni semplici.
- **Efficienza**: 
  - Elevato overhead di comunicazione e frequenti operazioni di I/O su disco.
  - Utilizzo limitato della memoria.
- **Elaborazione in tempo reale**: non è adatto per flussi di dati in tempo reale.

---

## Apache Spark

### Caratteristiche principali:
- **Elaborazione iterativa veloce**: Spark utilizza la memoria, risultando 10 volte più veloce di Hadoop per l'elaborazione iterativa.
- **Adatto per grafi di esecuzione generali**: consente ottimizzazioni avanzate.
- **Compatibilità**: supporta API di storage di Hadoop (HDFS, HBase).
- **Condivisione dati efficiente**: riduce il ricorso a disco e rete, sfruttando la memoria per velocizzare le operazioni.

---

## Confronto Spark vs MapReduce

| **Caratteristica**     | **MapReduce**            | **Spark**                    |
|------------------------|--------------------------|------------------------------|
| **Elaborazione**        | Basata su disco           | Basata su memoria             |
| **Velocità**            | Più lento, iterazioni lente| Significativamente più veloce |
| **Adatto per**          | Elaborazione batch        | Batch, iterativo, real-time   |
| **Utilizzo memoria**    | Limitato                  | Ottimizzato                   |
| **Complessità**         | Complesso                 | Più semplice e conciso        |

---

## Stack di Spark

- **Spark Streaming**: tempo reale.
- **Spark SQL**: elaborazione dati strutturati e SQL.
- **Spark MLlib**: machine learning.
- **Spark Core**: gestione delle funzionalità principali e RDD.
- **Gestione risorse**: scheduler standalone, Mesos, YARN.
- **Archiviazione dati**: HDFS, HBase.
- **Ingestione dati**: Kafka, Flume.

### Principio di Località dei dati

- Spark esegue il calcolo vicino ai dati per ridurre il traffico di rete e migliorare le performance.
- In alcuni casi può essere necessario spostare i dati, quando non ci sono executor liberi nei nodi contenenti i dati non elaborati.

---

## Performance: Spark vs MapReduce

### Algoritmi iterativi:
Spark è superiore per operazioni iterative, evitando la materializzazione dei dati su HDFS dopo ogni iterazione.

### Esempio: Algoritmo **k-means** (1 iterazione):

| **Fase**           | **MapReduce**         | **Spark**              |
|--------------------|-----------------------|------------------------|
| 1. Lettura         | Da HDFS                | Da HDFS                |
| 2. Mappa           | Assegnazione centroide | Assegnazione centroide |
| 3. GroupBy         | Per ID centroide       | Per ID centroide       |
| 4. Shuffle         | Sulla rete             | In memoria             |
| 5. Riduzione       | Calcolo centroide      | Calcolo centroide      |
| 6. Scrittura       | Su HDFS                | In memoria             |

---

## Esempi di codice: Word Count

### Hadoop (Java):
```java
public class WordCount {
  // Codice MapReduce
}
```

### Spark (Scala):
```scala
val file = sc.textFile("hdfs://...")
val counts = file.flatMap(line => line.split(" "))
                 .map(word => (word, 1))
                 .reduceByKey(_ + _)
counts.saveAsTextFile("hdfs://...")
```
Il codice Spark è più conciso, sfruttando trasformazioni e azioni su RDD.

---

## Motivazione per Spark

- **Miglioramento delle prestazioni**: particolarmente per iterazioni e stream grazie all'uso della memoria e del caching.
- **Semplicità**: API intuitive e ricche in vari linguaggi.
- **Generalità**: supporto a elaborazione batch, streaming, machine learning e grafi.

---

## Fondamenti di Spark

### Spark Stack:

- **Spark Core**: task management, memoria, tolleranza ai guasti, storage. Introduce **RDD**.
- **Moduli di alto livello**:
  - **Spark SQL**: dati strutturati e query SQL.
  - **Spark Streaming**: elaborazione in tempo reale.
  - **MLlib**: machine learning.
  - **GraphX**: elaborazione grafi.

### Obiettivi di progettazione:
- **Semplicità**
- **Generalità**
- **Bassa latenza**
- **Tolleranza ai guasti**

---

## Architettura di Spark

- **Driver program**: gestisce l'oggetto SparkContext e definisce trasformazioni e azioni.
- **Executor**: processi che eseguono i task in parallelo.
- **Cluster manager**: gestisce le risorse del cluster (Standalone, YARN, Mesos).

---

## Modello di programmazione di Spark

- **RDD**: unità di base dei dati, tolleranti ai guasti, distribuiti e immutabili.
- **Lazy evaluation**: le trasformazioni vengono eseguite solo quando si richiede un'azione.
- **Trasformazioni**: creano nuovi RDD (es. map, filter).
- **Azioni**: restituiscono un valore o salvano i dati su storage (es. count, collect).

### Tipi di Trasformazioni:
- **Narrow transformation**: ogni partizione del genitore è usata da una sola partizione del figlio.
- **Wide transformation**: più partizioni del figlio dipendono da una singola partizione del genitore (richiede shuffle).

### Livelli di Persistenza RDD:
- **MEMORY_ONLY**: solo in memoria.
- **MEMORY_AND_DISK**: in memoria e su disco.
- **MEMORY_ONLY_SER**: in memoria serializzata.
- **MEMORY_AND_DISK_SER**: in memoria/disco serializzata.
- **DISK_ONLY**: solo su disco.

---

## Spark Shell

Interfaccia interattiva per lavorare con Spark:

- **Shell Python**: `pyspark`
- **Shell Scala**: `spark-shell`

### SparkContext:
```scala
val conf = new SparkConf().setAppName(appName).setMaster(master)
new SparkContext(conf)
```

