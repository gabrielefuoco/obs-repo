| **Termine**                             | **Definizione**                                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Apache Spark**                        | Un framework open-source per l'elaborazione distribuita di Big Data.                                              |
| **RDD (Resilient Distributed Dataset)** | Struttura dati immutabile, distribuita e tollerante ai guasti in Spark.                                           |
| **Elaborazione in-memory**              | Elaborazione dei dati principalmente in memoria (RAM) per migliorare le prestazioni.                              |
| **Trasformazione**                      | Operazione lazy su un RDD che definisce un nuovo RDD senza eseguire immediatamente il calcolo.                    |
| **Azione**                              | Operazione su un RDD che attiva l'elaborazione e restituisce risultati o li scrive su storage.                    |
| **Funzione di ordine superiore**        | Funzione che accetta o restituisce altre funzioni come argomenti.                                                 |
| **Lineage**                             | Grafico aciclico diretto (DAG) che tiene traccia delle trasformazioni che hanno portato alla creazione di un RDD. |
| **Persistenza**                         | Memorizzazione di RDD in memoria o su disco per un accesso più rapido.                                            |
| **Spark Streaming**                     | Estensione di Spark per l'elaborazione di flussi di dati in tempo reale.                                          |
| **DStream**                             | Astrazione di alto livello che rappresenta un flusso continuo di dati come una sequenza di RDD.                   |
| **Spark MLlib**                         | Libreria per il Machine Learning distribuito in Spark.                                                            |
| **DataFrame**                           | Collezione distribuita di dati organizzati in colonne, simile a una tabella di database.                          |
| **Dataset**                             | Collezione distribuita e tipizzata di oggetti JVM.                                                                |
| **Spark SQL**                           | Modulo Spark per lavorare con dati strutturati utilizzando query simili a SQL.                                    |
| **Catalyst**                            | Ottimizzatore di query in Spark SQL e Dataset.                                                                    |

## MapReduce: Debolezze e Limitazioni

**Modello di Programmazione**:
- MapReduce è difficile da usare per certi problemi; anche operazioni semplici possono richiedere più passaggi.
  - Esempio: il semplice conteggio delle parole richiede diversi passaggi per ordinare i risultati per frequenza.
- Manca di strutture dati avanzate, controllo e supporto per iterazioni:
  - Le iterazioni richiedono il salvataggio continuo su disco, il che aumenta l'**overhead** e rende il processo lento.
  
**Efficienza**:
- L'alto costo della comunicazione e la frequente scrittura su disco penalizzano l'efficienza.
- MapReduce non sfrutta al meglio la memoria principale e opera spesso con I/O su disco.

**Limitazioni nell'elaborazione in tempo reale**:
- Non è adatto per l'elaborazione **streaming** o in tempo reale perché scansiona l'intero input prima di procedere con l'elaborazione.


## Apache Spark: Vantaggi

**Prestazioni**:
- Spark è un motore general-purpose per il Big Data, significativamente più veloce di Hadoop grazie all'elaborazione **in-memory**. Questo riduce la necessità di scrivere su disco e aumenta le prestazioni, specialmente per i calcoli iterativi.
  
**Modello di Programmazione**:
- Il modello di Spark è più flessibile rispetto a MapReduce:
  - Offre strutture come **RDD** (Resilient Distributed Datasets), **DataSet** e **DataFrame** che rendono più facile lavorare con dati strutturati e semistrutturati.
  - Le trasformazioni (map) e le azioni (reduce) sono integrate in un'API più amichevole per lo sviluppatore.

**Efficienza nell'uso della memoria**:
- Spark permette la **condivisione dei dati in-memory**, riducendo drasticamente i tempi di accesso e di elaborazione, superando i limiti di MapReduce che si basa su disco.

**Stack Unificato**:
- Spark offre diversi moduli che possono essere integrati in un'unica applicazione:
  - **Spark SQL** per i dati strutturati
  - **Spark Streaming** per i flussi di dati in tempo reale
  - **MLlib** per il machine learning distribuito
  - **GraphX** per il calcolo su grafi

### Confronto Spark vs Hadoop MapReduce
- Entrambi distribuiscono i calcoli su più nodi, ma **Spark** gestisce meglio la memoria e offre un modello più generalizzato.
- **Compatibilità**: Spark può leggere/scrivere da sistemi come HDFS, Cassandra e S3, garantendo flessibilità.

### Architettura di Spark
- **Spark Core**: La base del motore Spark, gestisce task, memoria, e interazione con lo storage.
- Moduli di alto livello (SQL, Streaming, MLlib, GraphX) permettono di lavorare su diversi tipi di dati senza dover cambiare il motore sottostante.

### Gestione del Cluster
- Spark supporta diversi gestori di cluster come **YARN**, **Mesos** e **Kubernetes**, oltre a una modalità standalone per piccole configurazioni.

## Architettura Spark

- **Driver ed Esecutori**: Ogni applicazione Spark si compone di un **programma driver** e **esecutori** distribuiti su un cluster.
  - **Driver**: Processo che esegue la funzione `main()` e crea l'oggetto **SparkContext**.
  - **Esecutori**: Processi che eseguono task su più thread e rimangono attivi per tutta la durata dell'applicazione. Ogni applicazione ha i propri esecutori per garantire l'**isolamento** tra applicazioni concorrenti.
- **SparkContext**: Si connette a un **gestore del cluster** per allocare risorse e distribuire i task.
- Una volta connesso al cluster, Spark invia il codice dell'applicazione (come file jar) agli esecutori, che eseguono i task richiesti.


## Modello di Programmazione Spark: RDD (Resilient Distributed Dataset)
- Gli **RDD** sono strutture dati immutabili, distribuite, in-memory e fault-tolerant.
- Gli RDD vengono partizionati per ottimizzare la distribuzione dei dati e sono manipolabili tramite una varietà di operatori.
#### Caratteristiche degli RDD:
1. **Immutabilità**: Una volta creati, gli RDD non possono essere modificati. Ogni trasformazione su un RDD crea un nuovo RDD.
2. **Distribuzione**: Gli RDD vengono suddivisi in **partizioni**, distribuite tra i nodi del cluster. Ogni partizione rappresenta un’unità di parallelismo.
3. **Tolleranza ai guasti**: In caso di guasto, gli RDD possono essere ricostruiti senza replicazione, utilizzando il **lineage** (una traccia della loro creazione).
#### Esecuzione Parallela:
- Gli **esecutori** sui nodi worker eseguono operazioni sui loro pezzi di RDD in parallelo, garantendo efficienza.
- Le **partizioni** sono frammenti logici dei dati che consentono un’elaborazione parallela.

### RDD: Distribuiti, Immutabili e Fault-Tolerant
- Gli RDD sono **distribuiti** nei nodi del cluster. Quando c'è sufficiente memoria, sono conservati in-memory, altrimenti su disco.
- La loro **immutabilità** garantisce che qualsiasi operazione su di essi crei nuovi RDD, preservando lo stato originale.
- **Fault tolerance**: Spark ricostruisce automaticamente gli RDD persi grazie al lineage, che tiene traccia di come sono stati generati.

### API RDD e Idoneità
- **API RDD**: Disponibile per vari linguaggi (Scala, Python, Java, R), l'API consente di interagire con RDD in modo facile e intuitivo. Le operazioni possono essere eseguite sia da interfacce interattive che da script.
- **Idoneità**: Gli RDD sono ideali per operazioni batch che devono essere applicate a un intero dataset, ma sono meno adatti per applicazioni che richiedono aggiornamenti asincroni e a grana fine.

### Operazioni sugli RDD: Trasformazioni e Azioni
- Le applicazioni Spark si scrivono in termini di **trasformazioni** e **azioni** sugli RDD:
  - **Trasformazioni**: Operazioni "lazy" che definiscono un nuovo RDD da un dataset esistente, come `map`, `filter`, `join`. Poiché sono lazy, non vengono eseguite fino a quando non viene richiesta un'azione.
  - **Azioni**: Eseguono effettivamente i calcoli sui dati, restituendo risultati al driver o salvando su storage esterno (es. `count`, `collect`, `save`).

### Modello di Programmazione Spark: DAG
- Spark utilizza un **grafo aciclico diretto (DAG)** per descrivere l'esecuzione di job. Ogni job Spark è scomposto in una serie di trasformazioni (nodi del grafo) e azioni che Spark esegue in modo parallelo e ottimizzato.

### Funzioni di Ordine Superiore
- Spark sfrutta le **funzioni di ordine superiore** per operare in parallelo su RDD, distribuendo i task tra i nodi del cluster.
  - **Trasformazioni**: Sono lazy e creano nuovi RDD senza eseguirli immediatamente.
  - **Azioni**: Avviano effettivamente il calcolo e restituiscono il risultato al driver o lo scrivono su storage.

### Trasformazioni e azioni disponibili su RDD in Spark

- `Seq[T]`: sequenza di elementi di tipo T

#### Trasformazioni

```
map(f : T => U)                 RDD[T] => RDD[U]
filter(f : T => Bool)           RDD[T] => RDD[T]
flatMap(f : T => Seq[U])        RDD[T] => RDD[U]
sample(fraction : Float)        RDD[T] => RDD[T] (Campionamento deterministico)
groupByKey()                    RDD[(K, V)] => RDD[(K, Seq[V])]
reduceByKey(f : (V, V) => V)    RDD[(K, V)] => RDD[(K, V)]
union()                         (RDD[T], RDD[T]) => RDD[T]
join()                          (RDD[(K, V)], RDD[(K, W)]) => RDD[(K, (V, W))]
cogroup()                       (RDD[(K, V)], RDD[(K, W)]) => RDD[(K, (Seq[V], Seq[W]))]
crossProduct()                  (RDD[T], RDD[U]) => RDD[(T, U)]
mapValues(f : V => W)           RDD[(K, V)] => RDD[(K, W)] (Preserva il partizionamento)
sort(c : Comparator[K])         RDD[(K, V)] => RDD[(K, V)]
partitionBy(p : Partitioner[K]) RDD[(K, V)] => RDD[(K, V)]
```
#### Azioni
```
count()                         RDD[T] => Long
collect()                       RDD[T] => Seq[T]
reduce(f : (T, T) => T)         RDD[T] => T
lookup(k : K)                   RDD[(K, V)] => Seq[V] (Su RDD partizionati per hash/range)
save(path : String)             Salva l'RDD su un sistema di storage, es. HDFS
```

## Come creare RDD
Gli **RDD** possono essere creati in tre modi principali:
1. **Parallelizzando collezioni esistenti** del linguaggio host (come le collezioni o le liste in Scala, Java, Python o R):
   - L'utente può specificare il numero di partizioni.
   - Nell'API RDD: utilizzando il metodo `parallelize`.
2. **Da file di grandi dimensioni** memorizzati in HDFS o in altri file system compatibili:
   - Viene creata una partizione per ogni blocco HDFS.
   - Nell'API RDD: utilizzando il metodo `textFile`.
3. **Trasformando un RDD esistente**:
   - Il numero di partizioni dipende dal tipo di trasformazione applicata.
   - Nell'API RDD: attraverso operazioni di trasformazione come `map`, `filter`, `flatMap`.
### Esempi in Python
```python
# Trasformare una collezione esistente in un RDD
lines = sc.parallelize(["pandas", "i like pandas"])
# sc è la variabile di contesto Spark
# Parametro importante: numero di partizioni in cui dividere il dataset
# Spark eseguirà un task per ogni partizione del cluster
# (impostazione tipica: 2-4 partizioni per ogni CPU nel cluster)
# Spark cerca di impostare automaticamente il numero di partizioni
# Puoi anche impostarlo manualmente passandolo come secondo parametro a parallelize, es. sc.parallelize(data, 10)

# Caricare dati dallo storage (file system locale, HDFS o S3)
lines = sc.textFile("/path/to/README.md")
```
## Trasformazioni RDD: map e filter
```python
# map: trasforma ogni elemento attraverso una funzione
nums = sc.parallelize([1, 2, 3, 4])
squares = nums.map(lambda x: x * x) # [1, 4, 9, 16]

# filter: seleziona gli elementi per cui la funzione restituisce true
even = squares.filter(lambda num: num % 2 == 0) # [4, 16]
```
## Trasformazione RDD: flatMap
```python
# flatMap: mappa ogni elemento a zero o più altri
ranges = nums.flatMap(lambda x: range(0, x, 1))
# [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

# dividere le righe di input in parole
lines = sc.parallelize(["hello world", "hi"])
words = lines.flatMap(lambda line: line.split(" "))
# ['hello', 'world', 'hi']
```
La funzione range in Python: sequenza ordinata di valori interi nell'intervallo [start, end) con passo non zero
## Trasformazione RDD: join
```python
# join: esegue un equi-join sulle chiavi di due RDD
# Solo le chiavi presenti in entrambi gli RDD vengono emesse
# I candidati join vengono elaborati indipendentemente

users = sc.parallelize([(0, "Alex"), (1, "Bert"), (2, "Curt"), (3, "Don")])
hobbies = sc.parallelize([(0, "writing"), (0, "gym"), (1, "swimming")])
users.join(hobbies).collect()
# [(0, ('Alex', 'writing')), (0, ('Alex', 'gym')), (1, ('Bert', 'swimming'))]
```
## Trasformazione RDD: reduceByKey
```python
# reduceByKey: aggrega valori con chiave identica usando la funzione specificata
# Esegue diverse operazioni di reduce parallele, una per ogni chiave nel dataset,
# dove ogni operazione combina valori che hanno la stessa chiave

x = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("a", 1),
    ("b", 1), ("b", 1), ("b", 1), ("b", 1)], 3)
# Applicando l'operazione reduceByKey
y = x.reduceByKey(lambda accum, n: accum + n)
# [('b', 5), ('a', 3)]
```
## Alcune azioni RDD

```python
# collect: restituisce tutti gli elementi dell'RDD come una lista
nums = sc.parallelize([1, 2, 3, 4])
nums.collect() # [1, 2, 3, 4]

# take: restituisce un array con i primi n elementi nell'RDD
nums.take(3) # [1, 2, 3]

# count: restituisce il numero di elementi nell'RDD
nums.count() # 4

# reduce: aggrega gli elementi nell'RDD usando la funzione specificata
sum = nums.reduce(lambda x, y: x + y)

# saveAsTextFile: scrive gli elementi dell'RDD come file di testo
# sul file system locale o HDFS
nums.saveAsTextFile("hdfs://file.txt")
```
## Trasformazioni lazy
- Le trasformazioni sono **lazy**: non vengono calcolate finché un'azione non richiede che un risultato venga restituito al programma driver
- Questo design permette a Spark di eseguire le operazioni in modo più efficiente poiché le operazioni possono essere raggruppate insieme
- **Esempi:**
  - Se ci fossero più operazioni di filter o map, Spark può fonderle in un unico passaggio
  - Se Spark sa che i dati sono partizionati, può evitare di spostarli sulla rete per un groupBy

### Esempio: **WordCount in Scala**

1. **Lettura del file** da HDFS come RDD:
   ```scala
   val textFile = sc.textFile("hdfs://...")
   ```
   
2. **Estrazione delle parole** usando `flatMap`:
   ```scala
   val words = textFile.flatMap(line => line.split(" "))
   ```

3. **Assegnazione di un conteggio iniziale** di 1 a ogni parola:
   ```scala
   val ones = words.map(word => (word, 1))
   ```

4. **Somma delle occorrenze di ogni parola** con `reduceByKey`:
   ```scala
   val counts = ones.reduceByKey(_ + _)
   ```

5. **Salvataggio del risultato** su HDFS:
   ```scala
   counts.saveAsTextFile("hdfs://...")
   ```

### Flusso di Dati
1. **Input (testo)**:  
   `Hello World Bye World Hello Hadoop Goodbye Hadoop`

2. **Parole estratte (RDD)**:  
   `Hello, World, Bye, World, Hello, Hadoop, Goodbye, Hadoop`

3. **Mappatura a (parola, 1)**:  
   `(Hello, 1), (World, 1), (Bye, 1), (World, 1), (Hello, 1), (Hadoop, 1), (Goodbye, 1), (Hadoop, 1)`

4. **Somma delle occorrenze (RDD)**:  
   `(Hello, 2), (World, 2), (Bye, 1), (Hadoop, 2), (Goodbye, 1)`

### Versione Concatenata in **Scala**

Le operazioni di trasformazione possono essere concatenate:
```scala
val counts = sc.textFile("hdfs://...")
               .flatMap(line => line.split(" "))
               .map(word => (word, 1))
               .reduceByKey(_ + _)
counts.saveAsTextFile("hdfs://...")
```

### Esempio in **Python**
```python
text_file = sc.textFile("hdfs://...")
counts = text_file.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("hdfs://...")
```

### Esempio in **Java 7**
```java
JavaRDD<String> textFile = sc.textFile("hdfs://...");
JavaRDD<String> words = textFile.flatMap(new FlatMapFunction<String, String>() {
    public Iterable<String> call(String s) { return Arrays.asList(s.split(" ")); }
});
JavaPairRDD<String, Integer> pairs = words.mapToPair(new PairFunction<String, String, Integer>() {
    public Tuple2<String, Integer> call(String s) {
        return new Tuple2<>(s, 1);
    }
});
JavaPairRDD<String, Integer> counts = pairs.reduceByKey(new Function2<Integer, Integer, Integer>() {
    public Integer call(Integer a, Integer b) { return a + b; }
});
counts.saveAsTextFile("hdfs://...");
```

### Esempio in **Java 8**
Con Java 8, grazie alle espressioni lambda, il codice diventa più conciso:
```java
JavaRDD<String> textFile = sc.textFile("hdfs://...");
JavaPairRDD<String, Integer> counts = textFile
    .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
    .mapToPair(word -> new Tuple2<>(word, 1))
    .reduceByKey((a, b) -> a + b);
counts.saveAsTextFile("hdfs://...");
```

##### Nota su `PairRDD`
In Java, Spark utilizza `Tuple2` per rappresentare le coppie chiave-valore nelle operazioni su RDD.

---
### Inizializzazione di Spark: SparkContext

1. **SparkContext**:
   - Punto di ingresso principale per tutte le operazioni Spark.
   - Gestisce la connessione al cluster Spark e permette di creare RDD.
   - Solo un'istanza di `SparkContext` può essere attiva per JVM.
   - Deve essere fermato (`stop()`) prima di crearne uno nuovo.

2. **SparkConf**:
   - Configura l'applicazione Spark con parametri come `AppName` e `Master`.
   - Esempio di creazione di uno `SparkContext`:
     ```scala
     val conf = new SparkConf().setAppName(appName).setMaster(master)
     new SparkContext(conf)
     ```

## Esempio: WordCount in Java

Parte 1: Importazione e configurazione Spark
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
```

Parte 2: Elaborazione dei dati
```java
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

## Persistenza RDD

- **RDD di default**: Viene ricalcolato ogni volta che un'azione viene eseguita.
- **Persistenza**: Consente di memorizzare in memoria le partizioni di un RDD per il riutilizzo rapido, migliorando le performance.
  - Metodi: `persist()` e `cache()`.
  - Fault-tolerant: Se una partizione viene persa, Spark la ricalcola automaticamente.
  
#### Livelli di Storage:
- **`MEMORY_ONLY`**: Salva solo in memoria (default di `cache()`).
- **`MEMORY_AND_DISK`**: Memorizza in memoria e, se necessario, riversa su disco.
- **`MEMORY_ONLY_SER`**: Salva oggetti serializzati in memoria (riduce l'uso di memoria).
- **`DISK_ONLY`**: Salva solo su disco.
- La serializzazione rende gli oggetti più efficienti in termini di spazio.
- Gli storage replicati andrebbero utilizzati solo se si desidera un rapido ripristino dai guasti

## Runtime di Spark

1. **Driver Program**:
   - Esegue il codice principale e invia task ai worker.
   
2. **Architettura di esecuzione**:
   ```
   Driver Program
   SparkContext
            ↓
        Cluster Manager
            ↓        ↓
    Worker Node   Worker Node
     Executor      Executor
       Task          Task
       Task          Task
   ```

### Esecuzione degli Stage
- **RDD** vengono trasformati in un **DAG** di operazioni.
- Il **DAG** è diviso in **stage** (insieme di operazioni senza shuffle).
- Ogni stage è suddiviso in **task** (uno per partizione).
- Le **azioni** guidano l'esecuzione e scatenano il calcolo.
### Riepilogo Componenti Spark
1. **RDD**: Dataset distribuito e partizionato.
2. **DAG**: Grafo logico delle operazioni su RDD.
3. **Stage**: Insieme di task eseguiti in parallelo.
4. **Task**: Unità fondamentale di esecuzione in Spark.

## Fault Tolerance in Spark

- **RDD lineage**: Gli RDD tengono traccia delle trasformazioni che li hanno creati, registrando un "lineage".
- **Ricalcolo dei dati**: Se un dato viene perso, Spark può ricalcolarlo seguendo il lineage.
- **Catena di oggetti lineage**: Ogni RDD è memorizzato come una catena di oggetti che catturano il suo lineage.

Esempio di lineage e caching in Scala:
```scala
val file = sc.textFile("hdfs://...")
val sics = file.filter(_.contains("SICS"))
val cachedSics = sics.cache()
val ones = cachedSics.map(_ => 1)
val count = ones.reduce(_+_)
```

### Scheduling dei Job in Spark
- Lo **scheduler** costruisce un **DAG di stage** a partire dal lineage di un RDD quando viene eseguita un'azione.
- Uno **stage** contiene tutte le trasformazioni con dipendenze strette (piplined).
- Il confine di uno stage è determinato da:
  - **Shuffle** per dipendenze ampie.
  - **Partizioni già calcolate**.
  
- **Task Scheduling**:
  - Spark assegna i task in base alla località dei dati (data locality).
  - Se una partizione è già in memoria su un nodo, il task viene inviato a quel nodo.

### API DataFrame e Dataset
1. **DataFrame** (da Spark 1.3):
   - Collezione distribuita di dati organizzati in colonne, come una tabella.
   - Supporta dati strutturati e semi-strutturati.
   - Permette query con una sintassi simile a SQL tramite **Spark SQL**.
   - Implementato come caso speciale di Dataset da Spark 2.0.
   
2. **Dataset** (da Spark 1.6):
   - Collezione distribuita e **tipizzata** di oggetti JVM.
   - Fornisce un'interfaccia OO **type-safe**.
   - **DataFrame** è un caso particolare di Dataset[Row], dove `Row` è un oggetto non tipizzato.
   - Disponibile in **Scala** e **Java**, ma non in Python o R.
   - Usa l'ottimizzatore **Catalyst** per ottimizzare le query e il piano di esecuzione.

- **SparkSession**: Punto di ingresso per entrambe le API.
  
### Dataset
- Combina i vantaggi degli **RDD** (strong typing, uso di lambda) con il motore di esecuzione ottimizzato di **Spark SQL**.
- Manipolabile tramite trasformazioni come `map`, `filter`, `flatMap`.
- **Lazy evaluation**: Il calcolo avviene solo quando viene eseguita un'azione.
  - Spark genera prima un **piano logico**, lo ottimizza, e poi lo traduce in un piano fisico eseguibile.

#### Come creare un Dataset:
- Da un file tramite la funzione `read`.
- Convertendo un RDD.
- Applicando trasformazioni su un Dataset esistente.

Esempio in Scala:
```scala
val names = people.map(_.name) // names è un Dataset[String]
Dataset<String> names = people.map((Person p) -> p.name, Encoders.STRING());
```

## DataFrame in Spark

- **DataFrame**: Dataset organizzato in colonne con nome.
  - Simile a una tabella di un database relazionale, ma con ottimizzazioni più avanzate.
  - Sfrutta l'ottimizzatore Catalyst, come i Dataset.
- Disponibile in **Scala**, **Java**, **Python** e **R**.
  - In **Scala** e **Java**, un DataFrame è un `Dataset` di `Row`.
  
#### Costruzione di un DataFrame:
  - Da **RDD** esistenti (inferendo o specificando lo schema).
  - Da **tabelle Hive**.
  - Da **file strutturati**: JSON, Parquet, CSV, Avro.
  
#### Manipolazione:
  - Simile a come si manipolano gli RDD, con operazioni come `select`, `filter`, `groupBy`, ecc.

### Spark Streaming
- Estensione di Spark per l'analisi di **dati in streaming**.
  - Dati ingeriti in **micro-batch** e analizzati in tempo reale.

#### DStream:
  - Un'astrazione di alto livello che rappresenta un flusso continuo di dati.
  - Internamente rappresentato come una sequenza di **RDD**.
  
#### Flusso di lavoro:
  ```
  input data stream -> Streaming Engine -> batches of input data -> Spark -> batches of processed data
  ```
  - Il flusso di dati viene segmentato in micro-batch, elaborato e restituito come output.

### Spark MLlib
- Libreria per **Machine Learning distribuito**.
  - Supporta algoritmi per:
    - **Classificazione** (es. Regressione Logistica).
    - **Regressione**.
    - **Clustering** (es. K-means).
    - **Raccomandazione**.
    - **Alberi decisionali**, **foreste casuali**, ecc.

#### Strumenti addizionali:
  - **ML**: trasformazioni delle caratteristiche, valutazione di modelli, ottimizzazione iper-parametri.
  - **Algebra lineare distribuita** (es. PCA).
  - **Statistica**: statistiche descrittive, test d'ipotesi.

- Utilizza **DataFrame** per supportare vari tipi di dati e algoritmi.

## Esempio Spark MLlib: Regressione Logistica

- Dataset contenente **etichette** e **vettori di caratteristiche**.
- Obiettivo: predire etichette dai vettori usando la **Regressione Logistica**.

```python
# Creazione di un DataFrame con etichette e caratteristiche
df = sqlContext.createDataFrame(data, ["label", "features"])

# Impostazione dell'algoritmo di regressione logistica
lr = LogisticRegression(maxIter=10)

# Addestramento del modello
model = lr.fit(df)

# Predizione delle etichette e visualizzazione dei risultati
model.transform(df).show()
```

In questo esempio, il modello viene addestrato sui dati e utilizzato per predire le etichette del dataset.

