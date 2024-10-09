
### Componenti Software

* **Libreria:** Spark funziona come una libreria all'interno del tuo programma, con un'istanza per ogni applicazione.
* **Esecuzione:** Può eseguire task sia localmente che su cluster.
* **Gestione Cluster:** Supporta diverse modalità di gestione del cluster: Mesos, YARN e standalone.
* **Accesso Storage:** Si interfaccia con sistemi di storage tramite l'API Hadoop InputFormat.
* **Sistemi Supportati:** Può utilizzare HBase, HDFS, S3 e altri sistemi di storage.

### Task Scheduler

* **Grafi di Task:** Gestisce grafi di task generali.
* **Pipeline Automatiche:** Supporta funzioni di pipeline automatiche.
* **Località Dati:** È consapevole della località dei dati per ottimizzare l'esecuzione.
* **Partizionamento:** È consapevole del partizionamento per evitare shuffle inutili.
* **groupBy:** Supporta l'operazione groupBy.

**Esempio:**

```
= RDD
= partizione memorizzata nella cache
```

### Funzionalità Avanzate

* **Partizionamento Controllabile:** Permette di controllare il partizionamento dei dati.
* **Join Veloci:** Velocizza le operazioni di join contro un dataset.
* **Formati di Storage:** Supporta diversi formati di storage controllabili.
* **Efficienza:** Mantiene i dati serializzati per efficienza, replica su più nodi e memorizza nella cache su disco.
* **Variabili Condivisi:** Supporta variabili condivise come broadcast e accumulatori.

### Esecuzione Locale

* **URL Master:** Per l'esecuzione locale, basta passare `local` o `local[k]` come URL master.
* **Debug:** Supporta il debug tramite debugger locali.
* **Java e Scala:** Per Java e Scala, è possibile eseguire il programma direttamente in un debugger.
* **Python:** Per Python, è necessario utilizzare un debugger attaccabile (es. PyDev).
* **Sviluppo e Test:** L'esecuzione locale è ideale per lo sviluppo e i test unitari.

### Esecuzione su Cluster

* **EC2:** Il modo più semplice per lanciare Spark su un cluster è tramite EC2:

```bash
./spark-ec2 -k keypair-i id_rsa.pem -s slaves \
[launch|stop|start|destroy] clusterName
```

* **Cluster Privati:** Sono disponibili diverse opzioni per cluster privati:
    * **Modalità Standalone:** Simile agli script di deployment di Hadoop.
    * **Mesos:** Un framework di orchestrazione dei cluster.
    * **Hadoop YARN:** Il framework di gestione dei cluster di Hadoop.
    * **Amazon EMR:** Un servizio di cluster gestito da Amazon (tinyurl.com/spark-emr). 
## Persistenza RDD

**Definizione:** La persistenza (o memorizzazione nella cache) di un RDD in memoria è una funzionalità fondamentale di Spark che consente di migliorare le prestazioni delle operazioni successive.

**Come Funziona:**

* Quando si persiste un RDD, ogni nodo del cluster memorizza in memoria le sue fette (partizioni) del dataset.
* Queste fette vengono riutilizzate in azioni successive sullo stesso dataset o su dataset derivati da esso.
* Questo approccio porta a un'accelerazione significativa delle operazioni future, spesso di oltre 10 volte.

**Utilizzo:**

* La persistenza è uno strumento chiave per la costruzione di algoritmi iterativi con Spark e per l'utilizzo interattivo dall'interprete.
* Per persistere un RDD, si utilizzano i metodi `persist()` o `cache()`.
* La prima volta che un RDD viene calcolato in un'azione, le sue partizioni vengono memorizzate in memoria.
* La cache è *fault-tolerant*: se una partizione viene persa, verrà automaticamente ricalcolata utilizzando le trasformazioni originali.

**Esempio: Estrazione di Log**

**Scenario:** Caricare messaggi di errore da un log in memoria e cercare interattivamente schemi specifici.

**Passaggi:**

1. **Caricamento:** Caricare il file di log in un RDD: `lines = spark.textFile("hdfs://...")`
2. **Filtraggio:** Filtrare le righe che iniziano con "ERROR": `errors = lines.filter(lambda s: s.startswith("ERROR"))`
3. **Mappatura:** Estrarre i messaggi di errore: `messages = errors.map(lambda s: s.spli[2])`
4. **Memorizzazione nella Cache:** Memorizzare l'RDD `messages` nella cache: `messages.cache()`
5. **Ricerca:** Eseguire ricerche interattive sui messaggi:
    * `messages.filter(lambda s: "mysql" in s).count()`
    * `messages.filter(lambda s: "php" in s).count()`

**Risultati:**

* **Wikipedia:** Un esempio di ricerca full-text su un dataset di 60GB su 20 macchine EC2.
    * Tempo di esecuzione con cache: 0.5 secondi
    * Tempo di esecuzione senza cache: 20 secondi

**Tabella di Confronto:**

| % del set di lavoro nella cache | Tempo di esecuzione (s) |
|---|---|
| 25% |  |
| 50% |  |
| 75% |  |
| Completamente disabilitato |  |

**Livelli di Storage:**

* **`MEMORY_ONLY`:** Memorizza l'RDD come oggetti Java deserializzati nella JVM. Se l'RDD non si adatta in memoria, alcune partizioni non verranno memorizzate nella cache e verranno ricalcolate al volo. Questo è il livello predefinito.
* **`MEMORY_AND_DISK`:** Memorizza l'RDD come oggetti Java deserializzati nella JVM. Se l'RDD non si adatta in memoria, memorizza le partizioni che non si adattano su disco e leggile da lì quando sono necessarie.
* **`MEMORY_ONLY_SER`:** Memorizza l'RDD come oggetti Java serializzati (un array di byte per partizione). Questo è generalmente più efficiente in termini di spazio rispetto agli oggetti deserializzati, soprattutto quando si utilizza un serializzatore veloce, ma più intensivo in termini di CPU da leggere.
* **`MEMORY_AND_DISK_SER`:** Simile a `MEMORY_ONLY_SER`, ma trasferisce le partizioni che non si adattano in memoria su disco invece di ricalcolarle al volo.
* **`DISK_ONLY`:** Memorizza le partizioni RDD solo su disco.
* **`MEMORY_ONLY_2`, `MEMORY_AND_DISK_2`, ecc.:** Stesso dei livelli precedenti, ma replica ogni partizione su due nodi del cluster.

**Selezione del Livello:**

* Il livello di storage viene scelto passando un oggetto `org.apache.spark.storage.StorageLevel` al metodo `persist()`.
* Il metodo `cache()` utilizza il livello predefinito `StorageLevel.MEMORY_ONLY`.


## Persistenza RDD: Scegliere il Livello di Storage

**Livello Predefinito:** Se i tuoi RDD si adattano comodamente al livello di storage predefinito `MEMORY_ONLY`, lascialo così. Questo è il livello più efficiente in termini di CPU, garantendo la massima velocità di esecuzione delle operazioni sugli RDD.

**Livello `MEMORY_ONLY_SER`:** Se i tuoi RDD sono troppo grandi per la memoria, prova a utilizzare `MEMORY_ONLY_SER`. Questo livello memorizza gli oggetti serializzati, risparmiando spazio. Scegli una libreria di serializzazione veloce per mantenere un buon livello di prestazioni.

**Livello Disco:** Evita di trasferire i dati su disco a meno che le funzioni che hanno calcolato i tuoi dataset siano costose o filtrino una grande quantità di dati. In generale, ricalcolare una partizione è veloce quanto leggerla da disco.

**Livelli Replicati:** Se hai bisogno di un rapido recupero da errori (ad esempio, in un'applicazione web), utilizza i livelli di storage replicati. Questi livelli replicano le partizioni su più nodi, garantendo una continuità di esecuzione anche in caso di perdita di dati.

## Come Funziona Spark a Tempo di Esecuzione

**Architettura:** Un'applicazione Spark è composta da un programma driver che esegue la funzione principale dell'utente e gestisce l'esecuzione di operazioni parallele su un cluster.

**Componenti:**

* **Driver Program:** Esegue la funzione principale dell'utente e gestisce il contesto di esecuzione (SparkContext).
* **Cluster Manager:** Gestisce i nodi del cluster e assegna i task ai worker.

**Esecuzione:**

* **Task:** L'applicazione crea RDD, li trasforma ed esegue azioni.
* **DAG (Directed Acyclic Graph):** Le trasformazioni e le azioni vengono rappresentate come un grafo di operatori.
* **Stage:** Il DAG viene diviso in stage, sequenze di RDD senza shuffle tra loro.
* **Task:** Ogni stage viene eseguito come una serie di task, una per ogni partizione.
* **Azioni:** Le azioni guidano l'esecuzione del DAG.

### Esecuzione di uno Stage

* **Spark:** Crea un task per ogni partizione del nuovo RDD, pianifica e assegna i task ai nodi worker.
* **Esecuzione Interna:** Tutto questo processo avviene internamente, senza intervento dell'utente.


## Riepilogo dei Componenti di Spark

**Livello Grossolano:**

* **RDD:** Dataset parallelo con partizioni.
* **DAG:** Grafo logico delle operazioni RDD.
* **Stage:** Insieme di task che vengono eseguite in parallelo.
* **Task:** Unità fondamentale di esecuzione in Spark.

**Livello Fine:**

* **Partizioni:** Suddivisioni del dataset RDD.
* **Operatori:** Funzioni che trasformano o agiscono sugli RDD.
* **Shuffle:** Operazione che ridistribuisce i dati tra le partizioni.


## Vista a Livello di Partizione di un RDD

**Vista a Livello di Dataset:**

```
log:
    HadoopRDD
        path = hdfs://...
    errors:
        FilteredRDD
            func = contains(„.)
            shouldCache = true
```


**Spiegazione:**

* **Vista a Livello di Dataset:** Mostra la struttura gerarchica degli RDD, partendo dal dataset originale (in questo caso, `log`) e mostrando le trasformazioni applicate (ad esempio, `errors` è un RDD filtrato da `log`).
* **Vista a Livello di Partizione:** Mostra come le partizioni dell'RDD vengono distribuite tra i task. Ogni riga rappresenta una partizione, e le lettere all'interno rappresentano i dati contenuti in quella partizione.
* **Task:** Ogni task elabora una o più partizioni dell'RDD.

**Nota:** Il codice `' f` e `„.` nel testo originale sono probabilmente errori di battitura. Ho corretto il codice per renderlo più leggibile.

## Durata di un Job in Spark (Flusso dei Task)

| Fase                                                  | **Descrizione**                                                                                                                                                                          |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Oggetti RDD**                                        | * `rddl.join(rdd2)`: Unione di due RDD. <br> * `.groupBy(...)`: Raggruppamento di elementi in base a una chiave. <br> * `.filter(...)`: Filtraggio di elementi in base a una condizione. |
| **2. Costruzione del DAG di operatori**                   | Il DAG (Directed Acyclic Graph) rappresenta le operazioni sugli RDD come un grafo.                                                                                                       |
| **3. DAG Scheduler**                                      | Il DAG Scheduler divide il DAG in stage, sequenze di operazioni senza shuffle.                                                                                                           |
| **4. Task Scheduler**                                     | Il Task Scheduler assegna i task ai worker del cluster.                                                                                                                                  |
| **5. Worker**                                             | I worker eseguono i task assegnati.                                                                                                                                                      |
| **6. Cluster Manager**                                    | Gestisce i nodi del cluster e assegna i task ai worker.                                                                                                                                  |
| **7. Threads**                                            | I thread eseguono i task all'interno dei worker.                                                                                                                                         |
| **8. Block Manager**                                      | Memorizza e serve i blocchi di dati.                                                                                                                                                     |
| **9. Dividi il DAG in stage di task**                     | Il DAG Scheduler divide il DAG in stage, sequenze di operazioni senza shuffle.                                                                                                           |
| **10. Invia ogni stage e i suoi task quando sono pronti** | Il Task Scheduler invia gli stage e i task ai worker quando sono pronti per l'esecuzione.                                                                                                |
| **11. Lancia i task tramite Master**                      | Il Master (o Cluster Manager) lancia i task sui worker.                                                                                                                                  |
| **12. Riprova i task falliti e strag­gler**               | Il Task Scheduler riprova i task falliti o che impiegano troppo tempo (stragglers).                                                                                                      |
| **13. Esegui i task**                                     | I worker eseguono i task assegnati.                                                                                                                                                      |
| **14. Memorizza e servi i blocchi**                       | Il Block Manager memorizza e serve i blocchi di dati utilizzati dai task.                                                                                                                |

**Spiegazione:**
Il processo inizia con la creazione di un DAG di operatori che rappresenta le operazioni sugli RDD. Il DAG Scheduler divide il DAG in stage, che vengono poi assegnati ai worker del cluster. I worker eseguono i task assegnati, memorizzando i dati nel Block Manager. Il Task Scheduler gestisce l'esecuzione dei task, riprovando quelli falliti o che impiegano troppo tempo.

## Architettura di un'Applicazione Spark

| Componente                       | Descrizione                                                                                                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Applicazione**                 | Un programma che viene eseguito su un cluster di nodi, composto da diversi componenti che lavorano insieme per elaborare i dati.                                              |
| **Client**                       | Il punto di ingresso per l'applicazione Spark. <br>Invia l'applicazione al Cluster Manager e alloca le risorse necessarie per l'esecuzione dell'applicazione (core, memoria). |
| **Cluster Manager (YARN/Mesos)** | Gestisce i nodi del cluster e assegna le risorse ai worker (Spark Worker) per l'esecuzione dei task.                                                                          |
| **Nodi Dati**                    | I nodi del cluster che memorizzano i dati. <br>Esempio: Nodo 1, Nodo 2, Nodo 3, Nodo 4.                                                                                       |
| **Spark Worker**                 | I processi che eseguono i task assegnati dal Cluster Manager. Memorizzano i dati in memoria per un accesso rapido.                                                            |
| **Driver**                       | Il processo che esegue la funzione principale dell'applicazione. Crea il SparkContext, che è il punto di accesso all'API Spark.                                               |
| **Executors**                    | I processi che eseguono i task assegnati dal Driver. Sono eseguiti sui nodi dati.                                                                                             |

## Tolleranza agli Errori in Spark

* Gli RDD (Resilient Distributed Datasets) tracciano la serie di trasformazioni utilizzate per la loro costruzione (la loro lineage).
* Le informazioni sulla lineage vengono utilizzate per ricalcolare i dati persi in caso di errore.
* Gli RDD sono memorizzati come una catena di oggetti che catturano la lineage di ogni RDD.

**Esempio:**

```scala
val file = sc.textFile("hdfs://...") // Carica un file da HDFS
val sics = file.filter(_.contains("SICS")) // Filtra le righe che contengono "SICS"
val cachedSics = sics.cache() // Memorizza in cache l'RDD per un accesso rapido
val ones = cachedSics.map(_ => 1) // Crea un RDD con il valore 1 per ogni riga
val count = ones.reduce(_+_) // Calcola la somma degli elementi dell'RDD
```

**Spiegazione:**

* Se un nodo dati fallisce durante l'esecuzione di un task, Spark può ricalcolare i dati persi utilizzando la lineage dell'RDD.
* Ad esempio, se il nodo che memorizza `cachedSics` fallisce, Spark può ricalcolare `cachedSics` a partire da `file` e `sics`.
* Questo meccanismo di tolleranza agli errori rende Spark molto robusto e affidabile.


## Pianificazione dei Job in Spark

* La pianificazione dei job in Spark tiene conto delle partizioni degli RDD persistenti disponibili in memoria.
* Quando un utente esegue un'azione su un RDD, lo scheduler costruisce un DAG (Directed Acyclic Graph) di stage dal grafo di lineage dell'RDD.
* Uno stage contiene il maggior numero possibile di trasformazioni in pipeline con dipendenze strette.
* Il confine di uno stage è definito da:
    * **Shuffle:** per dipendenze ampie (ad esempio, operazioni di join o group by).
    * **Partizioni già calcolate:** se una partizione è già stata calcolata e memorizzata in memoria, non è necessario ricalcolarla.

## Pianificazione dei Job in Spark (continua)

* Lo scheduler lancia i task per calcolare le partizioni mancanti da ogni stage fino a quando non viene calcolato l'RDD target.
* I task vengono assegnati alle macchine in base alla località dei dati.
* Se un task necessita di una partizione disponibile in memoria su un nodo, il task viene inviato a quel nodo.

## Località dei Dati

* **Principio di Località dei Dati:** il processo di spostare il calcolo vicino a dove i dati risiedono effettivamente sul nodo, invece di spostare grandi quantità di dati al calcolo. Questo minimizza la congestione della rete e aumenta la produttività complessiva del sistema.
* **Efficienza:** Se i dati e il codice che opera su di essi sono insieme, il calcolo tende ad essere veloce. Ma se codice e dati sono separati, uno deve spostarsi verso l'altro. In genere è più veloce spedire codice serializzato da un posto all'altro rispetto a un blocco di dati perché la dimensione del codice è molto più piccola dei dati. Spark costruisce la sua pianificazione attorno a questo principio generale di località dei dati.

## Località dei Dati (continua)

* **Livelli di Località:** La località dei dati indica quanto sono vicini i dati al codice che li elabora. Esistono diversi livelli di località in base alla posizione corrente dei dati. In ordine dal più vicino al più lontano:
    * **PROCESS_LOCAL:** i dati e il task si trovano sullo stesso executor.
    * **NODE_LOCAL:** i dati e il task si trovano sullo stesso nodo, ma su executor diversi.
    * **RACK_LOCAL:** i dati e il task si trovano sullo stesso rack, ma su nodi diversi.
    * **ANY:** i dati e il task possono trovarsi ovunque nel cluster.
* **Strategia di Pianificazione:** Spark preferisce pianificare tutti i task al livello di località migliore, ma questo non è sempre possibile. In situazioni in cui non ci sono dati non elaborati su alcun executor inattivo, Spark passa a livelli di località inferiori. Ci sono due opzioni:
    * **Aspettare:** Aspettare che un CPU occupato si liberi per avviare un task sui dati sullo stesso server.
    * **Avviare Immediatamente:** Avviare immediatamente un nuovo task in un luogo più lontano che richiede lo spostamento dei dati.
* **Comportamento Tipico:** Spark in genere aspetta un po' nella speranza che un CPU occupato si liberi. Una volta scaduto il timeout, inizia a spostare i dati da lontano al CPU libero.

## Spark: Impostazione del Livello di Parallelismo

Tutte le operazioni sugli RDD di coppie accettano un secondo parametro opzionale per il numero di task:

```scala
words.reduceByKey(lambda x, y: x + y, 5)
words.groupByKey(5)
visits.join(pageViews, 5)
```

## Variabili condivise

* Normalmente, quando una funzione passata a un'operazione Spark (come `map` o `reduce`) viene eseguita su un nodo remoto del cluster, funziona su copie separate di tutte le variabili utilizzate nella funzione.
* Queste variabili vengono copiate su ogni macchina e gli aggiornamenti alle variabili sulla macchina remota non vengono propagati al programma driver.
* Supportare variabili condivise generali di lettura-scrittura tra i task sarebbe inefficiente.
* Tuttavia, Spark fornisce due tipi limitati di variabili condivise per due schemi di utilizzo comuni: **_variabili broadcast_** e **_accumulatori_**.

## Variabili condivise

* **_Variabili broadcast_** consentono al programmatore di mantenere una variabile di sola lettura memorizzata nella cache su ogni macchina anziché spedire una copia con i task.

```scala
val broadcastVar = sc.broadcast(Array(l, 2, 3))
```

* **_Accumulatori_** sono variabili a cui si "aggiunge" solo tramite un'operazione associativa e possono quindi essere supportate in modo efficiente in parallelo. Possono essere utilizzate per implementare contatori (come in MapReduce) o somme.

```scala
val accum = sc.accumulator(0)
```

## Esempio: Regressione Logistica

Obiettivo: trovare la linea migliore che separa due insiemi di punti.

```scala
val data = spark.textFile(...).map(readPoint).cache()
var w = vector.random(D)

for (i <- 1 to ITERATIONS) {
  val gradient = data.map(p =>
    (1 / (1 + exp(-p.y*(w dot p.x))) - 1) * p.y * p.x
  ).reduce(_ + _)
  w -= gradient
}

println("Final w: " + w)
```

## Regressione Logistica (Python)

```python
# Regressione Logistica - algoritmo di apprendimento automatico iterativo
# Trova l'iperpiano migliore che separa due insiemi di punti in uno spazio di caratteristiche multidimensionale.
# Applica l'operazione MapReduce ripetutamente allo stesso set di dati, quindi beneficia notevolmente
# dalla memorizzazione nella cache dell'input in RAM

points = spark.textFile(...).map(parsePoint).cache()
w = numpy.random.ranf(size = D) # piano di separazione corrente

for i in range(ITERATIONS):
  gradient = points.map(
    lambda p: (1 / (1 + exp(-p.y*(w.dot(p.x)))) -1) * p.y * p.x
  ).reduce(lambda a, b: a + b)
  w -= gradient

print "Final separating plane: %s" % w
```



## Regressione Logistica (Scala)

```scala
// La stessa cosa in Scala

val points = spark.textFile(...).map(parsePoint).cache()
var w = Vector.random(D) // piano di separazione corrente

for (i <- 1 to ITERATIONS) {
  val gradient = points.map(p =>
    (1 / (1 + exp(-p.y*(w dot p.x))) -1) * p.y * p.x
  ).reduce(_ + _)
  w -= gradient
}

println("Final separating plane: " + w)
```


## Regressione Logistica (Java)

```java
// La stessa cosa in Java

class ComputeGradient extends Function<DataPoint, Vector> {
  private Vector w;

  ComputeGradient(Vector w) {this.w = w;}

  public Vector call(DataPoint p) {
    return p.x.times(p.y * (1 / (1 + Math.exp(w.dot(p.x))) -1));
  }
}

JavaRDD<DataPoint> points = spark.textFile(...).map(new ParsePoint()).cache();
Vector w = Vector.random(D); // piano di separazione corrente

for (int i = 0; i < ITERATIONS; i++) {
  Vector gradient = points.map(new ComputeGradient(w)).reduce(new AddVectors());
  w = w.subtract(gradient);
}

System.out.println("Final separating plane: " + w);
```

## DataFrames e Datasets

* **Evoluzione delle API Spark:** DataFrames e Datasets rappresentano un'evoluzione delle API di Spark, offrendo nuove funzionalità e miglioramenti rispetto agli RDD.
* **Collezioni di dati distribuite immutabili:** Come gli RDD, DataFrames e Datasets sono collezioni di dati distribuite e immutabili. Ciò significa che i dati sono suddivisi tra i nodi del cluster e non possono essere modificati direttamente.
* **Valutazione pigra:**  Anche DataFrames e Datasets vengono valutati in modo pigro, come gli RDD. Le operazioni vengono eseguite solo quando viene invocata un'azione (ad esempio, `collect`, `count`, `take`).
* **DataFrames:** Introdotti in Spark 1.3, i DataFrames introducono il concetto di schema per descrivere i dati. Lo schema definisce il tipo di dati di ogni colonna e il nome della colonna.
    * **Organizzazione dei dati:** I dati in un DataFrame sono organizzati in colonne denominate, come una tabella in un database relazionale. Questo rende i DataFrames più facili da usare e da comprendere rispetto agli RDD.
    * **Dati strutturati e semi-strutturati:** I DataFrames sono progettati per lavorare con dati strutturati e semi-strutturati, come file JSON, CSV, Parquet e Avro.
    * **Spark SQL:** Spark SQL fornisce API per eseguire query SQL su DataFrames con una semplice sintassi simile a SQL.
    * **Implementazione come Datasets:** Da Spark 2.0, i DataFrames sono implementati come un caso speciale di Datasets.

## Datasets

* **Estensione dei DataFrames:** Introdotti in Spark 1.6, i Datasets estendono i DataFrames fornendo un'interfaccia di programmazione tipizzata e sicura.
    * **Collezione di dati tipizzata:** Un Dataset è una collezione di dati strutturata ma tipizzata. Ciò significa che ogni elemento del Dataset ha un tipo di dati definito.
    * **Tipizzazione forte:** A differenza dei DataFrames, che sono generici e non tipizzati, i Datasets sono fortemente tipizzati. Questo rende il codice più sicuro e più facile da leggere.
    * **Classe SparkSession:** La classe `SparkSession` è il punto di ingresso per entrambe le API, DataFrames e Datasets.

## Datasets

* **Vantaggi:** I Datasets offrono i vantaggi degli RDD (tipizzazione forte, capacità di utilizzare funzioni lambda) con quelli del motore di esecuzione ottimizzato di Spark SQL (ottimizzatore Catalyst).
    * **Tipizzazione forte e funzioni lambda:** I Datasets consentono di utilizzare la tipizzazione forte e le funzioni lambda, come gli RDD.
    * **Ottimizzazione Catalyst:** I Datasets sfruttano l'ottimizzatore Catalyst di Spark SQL, che ottimizza il piano di esecuzione delle query per migliorare le prestazioni.
    * **Disponibilità:** I Datasets sono disponibili in Scala e Java, ma non in Python e R.
    * **Costruzione:** I Datasets possono essere costruiti da oggetti JVM.
    * **Trasformazioni funzionali:** I Datasets possono essere manipolati utilizzando trasformazioni funzionali come `map`, `filter`, `flatMap`, ecc.
    * **Valutazione pigra:** I Datasets sono pigri, ovvero il calcolo viene attivato solo quando viene invocata un'azione.
    * **Piano logico e piano fisico:** Internamente, un piano logico descrive il calcolo necessario per produrre i dati. Quando viene invocata un'azione, l'ottimizzatore di query di Spark ottimizza il piano logico e genera un piano fisico per un'esecuzione efficiente in modo parallelo e distribuito.

## Datasets

* **Creazione di un Dataset:**
    * **Da un file:** Utilizzando la funzione `read`.
    * **Da un RDD:** Convertendo un RDD esistente.
    * **Tramite trasformazioni:** Applicando trasformazioni su Datasets esistenti.

* **Esempio in Scala:**

```scala
val names = people.map(_.name) // names è un Dataset[String]
```

```java
Dataset<String> names = people.map((Person p) -> p.name, Encoders.STRING);
```

## DataFrames

* **Dataset organizzato in colonne:** Un DataFrame è un _Dataset_ organizzato in colonne denominate.
* **Equivalente a una tabella:** Concettualmente equivalente a una tabella in un database relazionale, ma con ottimizzazioni più ricche.
* **Ottimizzazione Catalyst:** Come i Datasets, i DataFrames sfruttano l'ottimizzatore Catalyst.
* **Disponibilità:** I DataFrames sono disponibili in Scala, Java, Python e R.
* **Rappresentazione:** In Scala e Java, un DataFrame è rappresentato da un Dataset di righe.
* **Costruzione:** I DataFrames possono essere costruiti da:
    * **RDD esistenti:** Sia inferendo lo schema usando la riflessione sia specificando lo schema in modo programmatico.
    * **Tabelle in Hive:**
    * **File di dati strutturati:** JSON, Parquet, CSV, Avro.
* **Manipolazione:** I DataFrames possono essere manipolati in modo simile agli RDD.

## _MLlib_

_MLlib_, la libreria di Machine Learning (ML) di Spark, fornisce molti algoritmi ML _distribuiti_. Questi algoritmi coprono compiti come l'estrazione di feature, la classificazione, la regressione, il clustering, la raccomandazione e altro ancora. MLlib fornisce anche strumenti come ML Pipelines per la costruzione di flussi di lavoro, CrossValidator per la messa a punto dei parametri e la persistenza del modello per il salvataggio e il caricamento dei modelli.

**Esempio di utilizzo di MLlib:**

```scala
// Ogni record di questo DataFrame contiene l'etichetta e
// le feature rappresentate da un vettore.
val df = sqlContext.createDataFrame(data).toDF("label", "features")
// Imposta i parametri per l'algoritmo.
// Qui, limitiamo il numero di iterazioni a 10.
val lr = new LogisticRegression().setMaxIter(10)
// Adatta il modello ai dati.
val model = lr.fit(df)
// Ispeziona il modello: ottieni i pesi delle feature.
val weights = model.weights
// Dato un set di dati, predici l'etichetta di ogni punto e mostra i risultati.
model.transform(df).show()
```

**Funzionalità di MLlib:**

* **Algoritmi ML distribuiti:**
    * Classificazione (ad esempio, regressione logistica)
    * Regressione
    * Clustering (ad esempio, K-mean)
    * Raccomandazione
    * Alberi decisionali
    * Foreste casuali
    * E altro ancora.
* **Utility:**
    * Trasformazioni di feature
    * Valutazione del modello
    * Messa a punto degli iperparametri
* **Algebra lineare distribuita:**
    * PCA (Principal Component Analysis)
* **Statistica:**
    * Statistiche riassuntive
    * Test di ipotesi
* **Supporto per DataFrames:** MLlib adotta DataFrames per supportare una varietà di tipi di dati.
## Mllib: Esempio

* **Set di dati:** Il set di dati contiene etichette e vettori di feature.
* **Obiettivo:** Imparare a prevedere le etichette dai vettori di feature usando la regressione logistica.

**Passaggi:**

1. **Crea un DataFrame:** Ogni record del DataFrame contiene l'etichetta e le feature rappresentate da un vettore.

   ```python
   df = sqlContext.createDataFrame(data, ["label", "features"])
   ```

2. **Imposta i parametri dell'algoritmo:** In questo caso, limitiamo il numero di iterazioni a 10.

   ```python
   lr = LogisticRegression(maxIter=10)
   ```

3. **Adatta il modello ai dati:**

   ```python
   model = lr.fit(df)
   ```

4. **Predici le etichette e mostra i risultati:** Dato un set di dati, predici l'etichetta di ogni punto e mostra i risultati.

   ```python
   model.transform(df).show()
   ```

## Spark Streaming

Spark Streaming è un'estensione di Spark che consente di analizzare i dati in streaming. I dati vengono ingeriti e analizzati in micro-batch, ovvero in piccoli blocchi di dati elaborati in modo sequenziale.

Spark Streaming utilizza un'astrazione di alto livello chiamata **DStream** (discretized stream) che rappresenta un flusso continuo di dati. Un DStream è essenzialmente una sequenza di RDD, dove ogni RDD rappresenta un micro-batch di dati.

**Funzionamento interno:**

Spark Streaming funziona internamente come segue:

1. **Dati di input:** I dati in streaming vengono ricevuti da una o più fonti.
2. **Flusso:** I dati vengono elaborati in modo continuo come un flusso.
3. **Micro-batch:** I dati vengono suddivisi in micro-batch.
4. **Motore Spark:** Ogni micro-batch viene elaborato dal motore Spark.
5. **Dati elaborati:** I risultati dell'elaborazione vengono restituiti come un nuovo flusso di dati.


## Spark Streaming: Esempio

Spark Streaming rappresenta i flussi di dati come una serie di RDD nel tempo. Ogni RDD rappresenta un micro-batch di dati, elaborato in un intervallo di tempo configurabile (tipicamente sub-secondi).

**Esempio di codice:**

```scala
val spammers = sc.sequenceFile("hdfs://spammers.seq")

sc.twitterStream(...)
  .filter(t => t.text.contains("Santa Clara University"))
  .transform(tweets => tweets.map(t => (t.user, t)).join(spammers))
  .print()
```

In questo esempio:

1. Viene caricato un file di spammers da HDFS.
2. Viene creato un flusso di tweet da Twitter.
3. I tweet che contengono "Santa Clara University" vengono filtrati.
4. Ogni tweet viene associato all'utente corrispondente e viene verificato se l'utente è presente nel file di spammers.
5. I risultati vengono stampati.

## Operazioni su finestre temporali

Spark Streaming supporta le operazioni su finestre temporali, che consentono di elaborare i dati in base a intervalli di tempo specifici.

**Operazioni di finestra:**

Le operazioni di finestra accettano due parametri: `windowLength` e `slideInterval`.

| Trasformazione | Significato |
|---|---|
| `window(windowLength, slideInterval)` | Restituisce un nuovo DStream calcolato in base a batch con finestra del DStream sorgente. |
| `countByWindow(windowLength, slideInterval)` | Restituisce un conteggio della finestra scorrevole degli elementi nel flusso. |
| `reduceByWindow(func, windowLength, slideInterval)` | Restituisce un nuovo flusso di un singolo elemento, creato aggregando gli elementi nel flusso su un intervallo scorrevole usando la funzione `func`. La funzione dovrebbe essere associativa e commutativa in modo che possa essere calcolata correttamente in parallelo. |
| `reduceByKeyAndWindow(func, windowLength, slideInterval, [numTasks])` | Quando viene chiamato su un DStream di coppie (K, V), restituisce un nuovo DStream di coppie (K, V) in cui i valori per ogni chiave sono aggregati usando la funzione di riduzione `func` data su batch in una finestra scorrevole. Nota: Per impostazione predefinita, questo utilizza il numero predefinito di task paralleli di Spark (2 per la modalità locale e in modalità cluster il numero è determinato dalla proprietà di configurazione `spark.default.parallelism`) per eseguire il raggruppamento. È possibile passare un argomento `numTasks` opzionale per impostare un numero diverso di task. |


## Spark: Combinazione di librerie (pipeline unificata)

Spark offre la possibilità di combinare diverse librerie per creare pipeline di elaborazione dati complete. Un esempio di questo è l'utilizzo di Spark SQL, MLlib e Spark Streaming per analizzare i dati in streaming e applicare modelli di machine learning.

**Esempio:**

1. **Carica i dati usando Spark SQL:**

   ```python
   points = spark.sql("select latitude, longitude from tweets")
   ```

   Questo codice carica i dati di latitudine e longitudine dai tweet in un DataFrame usando Spark SQL.

2. **Addestra un modello di machine learning:**

   ```python
   model = KMeans.train(points, 10)
   ```

   Questo codice addestra un modello di clustering K-means usando MLlib, utilizzando i dati di latitudine e longitudine caricati nel DataFrame.

3. **Applica il modello a un flusso:**

   ```python
   sc.twitterStream(...)
     .map(lambda t: (model.predict(t.location), 1))
     .reduceByWindow("5s", lambda a, b: a + b)
   ```

   Questo codice crea un flusso di tweet da Twitter, predice il cluster di appartenenza per ogni tweet usando il modello K-means addestrato e calcola il numero di tweet per cluster ogni 5 secondi.

## Spark GraphX

Spark GraphX è una libreria che estende Spark per l'elaborazione di grafi. Offre un'astrazione di grafo che rappresenta un multigrafo diretto con proprietà associate a ciascun vertice e bordo.

**Caratteristiche principali:**

* **Calcolo parallelo sui grafi:** GraphX consente di eseguire calcoli paralleli su grafi di grandi dimensioni.
* **Operatori fondamentali:** GraphX fornisce un set di operatori fondamentali per la manipolazione dei grafi, come `subgraph`, `joinVertices` e `aggregateMessages`.
* **Algoritmi e costruttori:** GraphX include una collezione di algoritmi e costruttori di grafi per semplificare i compiti di analisi dei grafi.



```scala
// Supponiamo che lo SparkContext sia già stato costruito
val sc: SparkContext
// Crea un RDD per i vertici
val users: RDD[(VertexId, (String, String))] =
  sc.parallelize(Seq((3L, ("rxin", "student")), (7L, ("jgonzal", "postdoc")),
    (5L, ("franklin", "prof")), (2L, ("istoica", "prof"))))
// Crea un RDD per i bordi
val relationships: RDD[Edge[String]] =
  sc.parallelize(Seq(Edge(3L, 7L, "collab"), Edge(5L, 3L, "advisor"),
    Edge(2L, 5L, "colleague"), Edge(5L, 7L, "pi")))
// Definisci un utente predefinito nel caso in cui ci siano relazioni con utenti mancanti
val defaultUser = ("John Doe", "Missing")
// Costruisci il grafo iniziale
val graph = Graph(users, relationships, defaultUser)
```

Query

```scala
val graph: Graph[(String, String), String] // Costruito da sopra
// Conta tutti gli utenti che sono postdoc
graph.vertices.filter { case (id, (name, pos)) => pos == "postdoc" }.count
// Conta tutti i bordi in cui src > dst
graph.edges.filter(e => e.srcId > e.dstId).count
```

## Esempio: PageRank

L'algoritmo PageRank è un esempio di elaborazione dati più complessa con Spark, che illustra:

* **Algoritmo multi-fase:** Utilizza più fasi di map & reduce.
* **Cache in memoria:** Beneficia della cache in memoria di Spark per migliorare le prestazioni.
* **Iterazioni multiple:** Richiede più iterazioni sugli stessi dati per convergere al risultato.

**Scopo:**

PageRank assegna un rango (punteggio) ai nodi (pagine web) in base ai link che puntano a loro. L'idea di base è che:

* **Più link in entrata = Rango più alto:** Pagine con molti link in entrata sono considerate più importanti.
* **Link da pagine importanti = Rango più alto:** Un link da una pagina con un rango alto conferisce maggiore importanza alla pagina di destinazione.

**Algoritmo:**

1. **Inizializzazione:** Inizializza il rango di ogni pagina a 1.
2. **Iterazione:** Per un numero fisso di iterazioni o fino alla convergenza:
    * **Calcolo dei contributi:** Ogni pagina `p` contribuisce con `rank(p) / |neighbors(p)|` a tutte le pagine a cui punta (i suoi vicini).
    * **Aggiornamento dei ranghi:** Il rango di ogni pagina viene aggiornato a `0.15 + 0.85 * somma_contributi`. La costante 0.15 rappresenta un fattore di smorzamento che tiene conto dei "salti casuali" tra le pagine.

**Implementazione in Scala:**

```scala
// Carica le coppie (URL, vicini) in un RDD
val links = sc.parallelize(Array(
  ("N1", Array("N2")),
  ("N2", Array("N1", "N3")),
  ("N3", Array("N1", "N4")),
  ("N4", Array("N1"))
))

// Inizializza i ranghi a 1.0 per ogni URL
var ranks = links.map(e => (e._1, 1.0))

// Esegui un numero fisso di iterazioni
for (i <- 1 to ITERATIONS) {
  // Calcola i contributi di ogni pagina ai suoi vicini
  val contribs = links.join(ranks).flatMap {
    case (url, (links, rank)) =>
      links.map(dest => (dest, rank / links.size))
  }

  // Aggiorna i ranghi sommando i contributi e applicando il fattore di smorzamento
  ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
}

// Salva i ranghi calcolati
ranks.saveAsTextFile(...)
```

**Prestazioni:**

Spark offre prestazioni significativamente migliori rispetto a Hadoop per l'esecuzione di PageRank, grazie alla sua capacità di mantenere i dati in memoria durante le iterazioni.

**Grafico delle prestazioni:**

(Immagine del grafico che mostra il tempo di iterazione in secondi per Hadoop e Spark al variare del numero di macchine)

**Legenda:**

* Asse X: Numero di macchine
* Asse Y: Tempo di iterazione (s)
* Barra blu: Hadoop
* Barra arancione: Spark 
