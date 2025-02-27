
## Caratteristiche principali degli strumenti di programmazione

#### Livello di astrazione

* **Basso livello:** permette agli sviluppatori di sfruttare API, meccanismi e istruzioni di basso livello, potenti ma complessi da utilizzare.
* **Medio livello:** permette agli sviluppatori di definire applicazioni usando un insieme limitato di costrutti di programmazione, nascondendo i dettagli di basso livello.
* **Alto livello:** permette agli sviluppatori di costruire applicazioni usando interfacce di alto livello, come IDE visivi, con costrutti di alto livello non correlati all'architettura sottostante.

#### Tipo di parallelismo

* **Parallelismo dati:** lo stesso codice viene eseguito in parallelo su diversi elementi dati.
* **Parallelismo task:** diversi task che compongono un'applicazione vengono eseguiti in parallelo.

## Apache Hadoop

**Apache Hadoop** è il framework open-source più popolare per implementare il modello di programmazione MapReduce. È progettato per sviluppare applicazioni data-intensive scalabili in vari linguaggi di programmazione (es. Java, Python) da eseguire su sistemi paralleli e distribuiti. L'approccio di programmazione in Hadoop permette l'astrazione dai classici problemi del calcolo distribuito, inclusi la località dei dati, il bilanciamento del carico di lavoro, la tolleranza ai guasti e il risparmio di larghezza di banda di rete.

### Altri framework basati su MapReduce

Esistono anche alcune implementazioni minori del modello MapReduce:

* **Phoenix++:** basato su C++, utilizza chip multi-core e multi-processori a memoria condivisa. Il suo runtime gestisce la creazione di thread, la partizione dei dati, la pianificazione dinamica dei task e la tolleranza ai guasti.
* **Sailfish:** un framework MapReduce che sfrutta la trasmissione batch dai mapper ai reducer. Utilizza un'astrazione chiamata *I-files* per supportare l'aggregazione dei dati, raggruppando in modo efficiente i dati scritti e letti da più nodi.

## Caratteristiche di Apache Hadoop

* **Elaborazione batch:** Apache Hadoop è un framework comunemente usato per l'elaborazione batch, ma è inefficiente per applicazioni altamente iterative a causa dell'elaborazione basata su disco nel file system distribuito.
* **Community open-source:** Il progetto Hadoop è supportato da una grande community open-source, che fornisce aggiornamenti e correzioni di bug costanti.
* **Basso livello di astrazione:** Hadoop fornisce un basso livello di astrazione: gli sviluppatori definiscono le applicazioni usando API potenti ma non user-friendly, che richiedono una comprensione di basso livello del sistema. Lo sviluppo in Hadoop richiede più impegno e codice rispetto ai sistemi di astrazione di livello superiore (es. Pig o Hive), ma il codice è generalmente più efficiente in quanto può essere completamente ottimizzato.
* **Parallelismo dati:** Hadoop è progettato per sfruttare il parallelismo dei dati, poiché i dati di input vengono partizionati in chunk ed elaborati in parallelo da macchine diverse.
* **Tolleranza ai guasti:** Il framework garantisce un'elevata tolleranza ai guasti insieme a meccanismi di checkpoint e ripristino.

## Moduli di Hadoop

Il progetto Hadoop include molti altri moduli, come:

* **Hadoop Distributed File System (HDFS):** un file system distribuito che offre tolleranza ai guasti con ripristino automatico, portabilità su hardware e sistemi operativi commodity eterogenei ed economici.
* **Yet Another Resource Negotiator (YARN):** un framework per la gestione delle risorse del cluster e la pianificazione dei job.
* **Hadoop Common:** utility e librerie che supportano gli altri moduli Hadoop.

Nel corso degli anni, Hadoop si è evoluto in una piattaforma versatile che supporta molti sistemi di programmazione, come:

* **Storm** per l'analisi di dati in streaming
* **Hive** per l'interrogazione di grandi dataset
* **Giraph** per l'elaborazione iterativa di grafi
* **Ambari** per il provisioning e il monitoraggio del cluster

## Stack software Hadoop

![[Pasted image 20250223163540.png|414]]
## HDFS (Hadoop Distributed File System)

L'**Hadoop Distributed File System (HDFS)** è stato progettato per archiviare grandi volumi di dati garantendo una lettura veloce e tolleranza ai guasti. I file in HDFS sono distribuiti e replicati su diversi nodi di storage, supportando un'organizzazione gerarchica dei file come i file system tradizionali.

Un cluster HDFS ha un'architettura *master-workers* e consiste di:

* Un **namenode** (master) che gestisce il file system distribuito, mantenendo l'albero del file system e memorizzando nomi e metadati;
* Un certo numero di **datanode** (workers) che memorizzano e recuperano blocchi di dati, comunicando periodicamente con il namenode;
* Un **secondary namenode** opzionale per la tolleranza ai guasti, che memorizza lo stato del file system in caso di errori del namenode.

### Caratteristiche di HDFS

HDFS memorizza i file come una sequenza di *blocchi di dati*, ognuno dei quali rappresenta la quantità minima di dati per la lettura o la scrittura. La dimensione del blocco predefinita è di 128 MB, ma può essere configurata. Per la tolleranza ai guasti, ogni blocco viene replicato tra i datanode con un *fattore di replicazione* dato, configurabile alla creazione e alla modifica del file. Hadoop sfrutta la *data locality* per l'efficienza nella distribuzione dei job di calcolo tra i worker, minimizzando il trasferimento di dati sulla rete, riducendo la congestione e aumentando la produttività complessiva del sistema.

### Architettura HDFS

![[|363](_page_9_Figure_2.jpeg)]

## Flusso di Esecuzione (MapReduce)

Il flusso di esecuzione in Hadoop coinvolge diversi componenti:

* **Dati di Input:** i file di input per un job MapReduce, tipicamente memorizzati su HDFS.
* **InputFormat:** definisce come i dati di input vengono suddivisi e letti per creare *input split*.
* **InputSplit:** rappresenta la porzione di dati che verrà elaborata da una singola istanza di un mapper. Ogni split è diviso in *record* prima di essere elaborato.
* **RecordReader:** converte uno split di input in coppie chiave-valore adatte per essere lette ed elaborate dal mapper.

### Fasi

Il flusso di esecuzione di un'applicazione MapReduce in Apache Hadoop comprende diverse fasi:

1. **Mapper:** Applica una funzione di mapping a ciascuna coppia chiave-valore di input, producendo un elenco di coppie chiave-valore come output. 

2. **Combiner:** Esegue un'aggregazione locale dell'output del mapper per ridurre il trasferimento di dati intermedi tra mapper e reducer.

3. **Partitioner:** Suddivide l'output del combiner (o del mapper se non è presente un combiner) usando una funzione di hashing, assicurando che le tuple con la stessa chiave finiscano nella stessa partizione.

4. **Shuffle e Ordinamento:** Ogni partizione viene trasferita in rete ai nodi reducer (*shuffling*). Prima del trasferimento, il framework Hadoop esegue l'ordinamento per chiave.

5. **Reducer:** Esegue l'aggregazione finale applicando una funzione di riduzione ai dati ricevuti.

6. **RecordWriter:** Scrive le coppie chiave-valore di output dalla fase di riduzione nei file di output. L' `OutputFormat` definisce il formato di scrittura.

![[|431](_page_12_Figure_2.jpeg)

## Basi di Programmazione MapReduce

Un programma MapReduce in Hadoop consiste di tre parti principali:

1. **Mapper:** Estende la classe `Mapper` per fornire un'implementazione personalizzata del metodo `map`.

2. **Reducer:** Estende la classe `Reducer` per fornire un'implementazione personalizzata del metodo `reduce`.

3. **Driver:** Configura il job MapReduce e contiene il `main` del programma. 

### Classe Mapper

La classe `Mapper` è definita come:

```java
class Mapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT>
```

dove:

* `KEYIN`: Chiave di input.
* `VALUEIN`: Valore di input.
* `KEYOUT`: Chiave di output.
* `VALUEOUT`: Valore di output.

Metodi sovrascrivibili:

* `void setup(Context context)`: Chiamato una volta all'inizio del task.
* `void map(KEYIN key, VALUEIN value, Context context)`: Chiamato per ogni coppia chiave-valore nello split di input.
* `void cleanup(Context context)`: Chiamato alla fine del task map. 

### Classe Reducer

La classe `Reducer` è definita come:

```java
class Reducer<KEYIN, VALUEIN, KEYOUT, VALUEOUT>
```

dove:

* `KEYIN`: Chiave di input.
* `VALUEIN`: Valore di input.
* `KEYOUT`: Chiave di output.
* `VALUEOUT`: Valore di output.

Metodi sovrascrivibili:

* `void setup(Context context)`: Chiamato una sola volta all'inizio del task.
* `void reduce(KEYIN key, Iterable<VALUEIN> values, Context context)`: Chiamato per ogni chiave per elaborare tutti i valori associati.
* `void cleanup(Context context)`: Chiamato alla fine del task di riduzione. 

### Classe Driver

La classe `Driver` configura il job MapReduce, specificando:

* Nome del job.
* Tipi di dati di input/output.
* Classi mapper e reducer.
* Altri parametri.

L'oggetto `Context` permette al mapper e al reducer di interagire con Hadoop, accedendo a dati di configurazione ed emettendo output. 

## Ordinamento Secondario

Hadoop ordina le tuple intermedie per chiave prima di inviarle al reducer. L'ordinamento secondario permette un controllo più fine usando una chiave composita `<chiave_primaria, chiave_secondaria>`:

1. Un *partitioner* personalizzato assegna tuple con la stessa chiave primaria allo stesso nodo reducer.
2. Un *comparator* personalizzato ordina le tuple usando l'intera chiave composita.
3. Un *group comparator* personalizzato raggruppa le tuple per chiave primaria prima del metodo `reduce`.

Esempio: usando `<user_id, timestamp>`, le tuple vengono partizionate per `user_id`, ordinate per la chiave composita e raggruppate per `user_id` prima dell'elaborazione del reducer. 

## Creazione di un Indice Inverso con Hadoop

Hadoop può essere usato per creare un indice inverso per un vasto insieme di documenti web. Un indice inverso mappa parole agli ID dei documenti che le contengono e al numero di occorrenze. ![[](_page_18_Figure_4.jpeg)

### Esempio di programmazione

![[|432](_page_19_Figure_3.jpeg)

La classe `MapTask` implementa il mapper, che riceve un elenco di documenti. Il metodo `map`:

1. Ottiene il `documentID`.
2. Analizza le righe di testo ed emette coppie `<word, documentID:numberOfOccurrences>` (dove `numberOfOccurrences = 1`).
3. Può preelaborare le parole (rimozione punteggiatura, lemmatizzazione, stemming).
4. Utilizza `Text` e `IntWritable` per una migliore serializzazione. 

### CombineTask e ReduceTask

La classe `CombineTask` implementa un combiner che aggrega i dati intermedi sommando le occorrenze di ogni parola in un documento, emettendo coppie `<word, documentID:sumNumberOfOccurrences>`.

La classe `ReduceTask` implementa il reducer che, per ogni parola, produce l'elenco dei documenti che la contengono e il numero di occorrenze: `<word, List(documentID:numberOfOccurrences)>`. Questo insieme di coppie forma l'indice inverso. 

## Configurazione del Job in MapReduce

La configurazione di un job in MapReduce richiede la specifica di tre elementi principali:

1. **Classi:** Si devono indicare le classi Java da utilizzare come `mapper`, `combiner` e `reducer`. Queste classi implementano la logica di elaborazione dei dati.

2. **Formati chiave/valore:** È necessario definire i formati chiave/valore sia per l'input che per l'output. Questi formati specificano il tipo di dati che vengono elaborati in ogni fase del processo MapReduce.

3. **Percorsi di input/output:** Infine, si devono specificare i percorsi del file system distribuito (ad esempio, HDFS) che contengono i dati di input e dove verranno scritti i dati di output.
