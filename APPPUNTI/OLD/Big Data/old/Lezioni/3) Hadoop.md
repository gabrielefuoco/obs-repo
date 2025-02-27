
| **Termine** | **Spiegazione** |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Apache Hadoop** | Un framework software open-source per l'elaborazione distribuita di grandi set di dati su cluster di hardware commodity. |
| **HDFS (Hadoop Distributed File System)** | Un sistema di file distribuito progettato per gestire grandi set di dati e garantire tolleranza ai guasti tramite la replica dei dati. |
| **Hadoop YARN (Yet Another Resource Negotiator)** | Un gestore di risorse che si occupa di allocare le risorse del cluster alle applicazioni, garantendo un'esecuzione efficiente. |
| **Hadoop MapReduce** | Un modello di programmazione per l'elaborazione distribuita dei dati, che suddivide il lavoro in task Map (filtraggio e classificazione) e task Reduce (aggregazione dei risultati). |
| **Scalabilità Orizzontale** | La capacità di un sistema di aumentare le prestazioni aggiungendo più macchine al cluster. |
| **Server Commodity** | Server economici e facilmente reperibili. |
| **Mapper** | Una funzione che elabora una coppia chiave-valore di input e produce un insieme di coppie chiave-valore intermedie. |
| **Reducer** | Una funzione che elabora un insieme di coppie chiave-valore intermedie con la stessa chiave e produce un insieme di coppie chiave-valore di output. |
| **Shuffle and Sort** | La fase di MapReduce che raggruppa le coppie chiave-valore intermedie per chiave e le ordina. |
| **Combiner** | Una funzione opzionale che può essere utilizzata per aggregare i dati intermedi prima che vengano inviati ai reducer. |
| **Partizionamento** | Il processo di divisione dei dati in sottoinsiemi che vengono elaborati dai reducer. |
| **NameNode** | Il server master in HDFS che gestisce lo spazio dei nomi del file system e controlla l'accesso ai file. |
| **DataNode** | I server che archiviano i blocchi di dati, replicati per garantire la tolleranza ai guasti. |
| **EditLog** | Un registro delle transazioni che registra le modifiche ai metadati in HDFS. |
| **FsImage** | Un file che contiene la mappatura dei blocchi e altre proprietà del file system in HDFS. |
| **ResourceManager (RM)** | Gestisce globalmente le risorse del cluster, pianifica e alloca risorse per i job in YARN. |
| **NodeManager (NM)** | Agente distribuito su ogni nodo del cluster, monitorizza e riporta lo stato delle risorse al ResourceManager in YARN. |
| **ApplicationMaster (AM)** | Creato per ogni applicazione, negozia le risorse con il ResourceManager e coordina l'esecuzione dei task con i NodeManager in YARN. |
| **Località dei Dati** | Il principio di eseguire i task vicino ai dati per evitare l'uso eccessivo della larghezza di banda. |
| **Amazon Elastic MapReduce (EMR)** | Un servizio AWS che distribuisce il lavoro su server virtuali in AWS (EC2), supporta anche Hive, Pig, HBase, Spark. |
| **Google Cloud Dataproc** | Un servizio Google Cloud che esegue su Google Cloud, supporta anche Hadoop e Spark. |
| **Hadoop Streaming** | Un'API che permette di scrivere funzioni Map e Reduce in linguaggi di programmazione diversi da Java. |
| **Cloudera CDH** | Una distribuzione integrata basata su Hadoop che contiene tutti i componenti necessari per la produzione. |
| **MapR** | Una distribuzione integrata basata su Hadoop che utilizza un file system proprietario (MapR-FS) invece di HDFS. |

## Cos'è Apache Hadoop?

Apache Hadoop è un framework software open-source progettato per il **calcolo distribuito** affidabile e scalabile su set di dati massivi, utilizzando **cluster di hardware commodity**. È stato originariamente sviluppato da Yahoo! per gestire grandi quantità di dati su infrastrutture distribuite.

#### Obiettivi principali:

- **Archiviazione ed elaborazione massiva** di set di dati su cluster di server economici.
- **Affidabilità e scalabilità** su hardware con una tendenza al guasto.
- Include componenti fondamentali:
 - **HDFS**: Hadoop Distributed File System
 - **Hadoop YARN**: Gestione delle risorse del cluster
 - **Hadoop MapReduce**: Sistema per calcoli paralleli su grandi dataset

#### HDFS (Hadoop Distributed File System):

- Sistema di file distribuito progettato per gestire **grandi set di dati** (da GB a TB) e garantire **tolleranza ai guasti** attraverso la replicazione dei dati.
- Ottimizzato per l'accesso **in streaming** ai dati, con un'architettura pensata per l'elaborazione batch.
#### Hadoop YARN:

- Gestisce le risorse del cluster, garantendo che i **calcoli siano distribuiti** in modo efficiente tra i nodi del cluster.
#### Hadoop MapReduce:

- Paradigma di programmazione che consente l'**elaborazione distribuita** di grandi dataset in modo parallelo.
- Il sistema suddivide il lavoro in **task Map** (dove i dati vengono filtrati e classificati) e **task Reduce** (dove i risultati parziali vengono combinati).
## Architettura del cluster Hadoop:

- I nodi di calcolo sono organizzati in **rack**, ognuno contenente da 8 a 64 nodi.
- I rack sono collegati tramite reti ad alta velocità (Ethernet Gigabit), ma la **comunicazione intra-rack** è più rapida rispetto alla comunicazione tra rack.
- Poiché i nodi possono guastarsi, Hadoop si basa sulla **replicazione** dei dati e sulla ridondanza per garantire l'affidabilità.

## Hadoop su hardware commodity:

- **Vantaggi**:
 - Utilizzo di **hardware standardizzato** e accessibile, senza dover dipendere da costosi fornitori proprietari.
 - Possibilità di scalare orizzontalmente (aggiungendo più nodi) piuttosto che verticalmente (aggiornando l'hardware esistente).

- **Svantaggi**:
 - **Tassi di guasto** più elevati rispetto a soluzioni proprietarie, mitigati dalla replicazione e dalla gestione distribuita dei task.
#### Esempi di utilizzo:

- **Yahoo!** ha utilizzato Hadoop per elaborare enormi dataset:
 - Nel 2014, Hadoop era installato su oltre **100.000 CPU** distribuite su più di **40.000 macchine**.
 - Nel 2017, Hadoop gestiva **36 cluster distinti** per vari progetti come Apache HBase, YARN e Storm.

### Ottimizzazioni con Hadoop MapReduce:

- MapReduce consente la programmazione di calcoli distribuiti, e può essere migliorato con strumenti come:
 - **Apache Pig** e **Apache Hive** per una programmazione più accessibile.
 - **Apache HBase** per un accesso più efficiente ai dati distribuiti.

## HDFS

- Progettato per hardware a basso costo, HDFS è tollerante ai guasti e fornisce **accesso veloce ai dati** per applicazioni con grandi set di dati.
- Presupposti:
 - **Guasto hardware**: La rilevazione e il recupero dai guasti sono fondamentali.
 - **Accesso in streaming**: Ottimizzato per l'elaborazione batch, con enfasi sulla velocità di trasmissione.
 - **Grandi set di dati**: Scalabile a centinaia di nodi per set di dati di dimensioni da GB a TB.
 - **Modello di coerenza semplice**: Modello di accesso "write-once-read-many".
 - **Efficienza del calcolo**: Spostare il calcolo vicino ai dati è più efficiente che spostare i dati.
 - **Portabilità**: Progettato per essere eseguibile su diverse piattaforme hardware e software.

### Architettura di HDFS

- **Master/Slave**: L'architettura di HDFS segue un modello **master/slave**.
- Il cluster HDFS è composto da:
 - **NameNode**: È il server master che gestisce lo **spazio dei nomi** del file system e controlla l'accesso ai file.
 - **DataNode**: Un DataNode per ogni nodo del cluster, gestisce lo **storage** effettivo. 
- I file vengono divisi in **blocchi** e distribuiti sui DataNode.

#### Namespace del File System

- HDFS espone un namespace che consente agli utenti di memorizzare dati in file e directory.
- **NameNode**:
 - Esegue operazioni sui file e directory (apertura, chiusura, ridenominazione).
 - Mappa i blocchi ai DataNode.
- **DataNode**:
 - Gestisce le richieste di lettura e scrittura dai client.
 - Crea, elimina e replica i blocchi su istruzione del NameNode.
- La struttura dei file segue un'**organizzazione gerarchica** (simile ad altri file system).

#### Metadati del file system

- Il NameNode registra le modifiche ai **metadati** nel registro delle transazioni chiamato **EditLog**.
- L'intero spazio dei nomi è memorizzato in un file chiamato **FsImage**, che contiene la mappatura dei blocchi e altre proprietà del file system.

#### Lettura e scrittura in HDFS

- **Lettura**: Il client chiede al NameNode la posizione dei blocchi per accedere ai dati.
- **Scrittura**:
 - Il client richiede al NameNode una lista di DataNode disponibili.
 - Viene formata una pipeline in cui il primo DataNode memorizza il blocco e lo inoltra agli altri DataNode.

#### Portabilità di HDFS

- HDFS è scritto in **Java**, il che lo rende eseguibile su qualsiasi macchina che supporti Java. Questo lo rende portabile su diverse piattaforme hardware.
#### Accessibilità di HDFS

- HDFS è accessibile tramite:
 - **API Java**: per le applicazioni che interagiscono direttamente con HDFS.
 - **FS Shell**: un'interfaccia a riga di comando per interagire con HDFS.

## Esempi di comandi FS Shell

- **Creare una directory**:
  ```bash
  bin/hadoop dfs -mkdir /foodir
  ```
- **Rimuovere una directory**:
  ```bash
  bin/hadoop dfs -rmr /foodir
  ```
- **Visualizzare il contenuto di un file**:
  ```bash
  bin/hadoop dfs -cat /foodir/myfile.txt
  ```

- Quando un file viene eliminato, viene spostato nella directory **/trash**.
- I file possono essere recuperati fino a quando rimangono in /trash (durata predefinita: 6 ore).
- Dopo questo periodo, il file viene rimosso permanentemente.

## YARN

- **YARN** (Yet Another Resource Negotiator) è il gestore di risorse e pianificatore di job di Hadoop.
- Si occupa di allocare risorse di sistema alle applicazioni in esecuzione su un cluster e di pianificare i task sui nodi del cluster.
- **Introdotto con Hadoop 2.0**, ha separato HDFS dal motore di elaborazione MapReduce, permettendo di eseguire altre applicazioni oltre a MapReduce.
- Supera i limiti di Hadoop 1.0, che supportava solo applicazioni MapReduce.
- YARN opera tra **HDFS** e i motori di elaborazione, gestendo dinamicamente l'allocazione delle risorse per migliorare le prestazioni e l'uso delle risorse rispetto all'allocazione statica di MapReduce.

- Supporta diversi **scheduler**, tra cui:
 - **FIFO Scheduler**: esegue job in ordine di arrivo.
 - **Fair Scheduler**: distribuisce equamente le risorse tra i job in esecuzione.
#### Confronto con MapReduce

- In Hadoop 1.0, il **JobTracker** centralizzava la gestione delle risorse e la pianificazione, causando colli di bottiglia.
- YARN distribuisce meglio queste funzioni, riducendo i limiti legati alla scalabilità del cluster.

#### Componenti di YARN

YARN suddivide la gestione dei job di elaborazione in tre principali componenti:
- **ResourceManager (RM)**: gestisce globalmente le risorse del cluster, pianifica e alloca risorse per i job.
- **NodeManager (NM)**: Agente distribuito su ogni nodo del cluster, monitorizza e riporta lo stato delle risorse al ResourceManager.
- **ApplicationMaster (AM)**: creato per ogni applicazione, negozia le risorse con il ResourceManager e coordina l'esecuzione dei task con i NodeManager.

#### Ottimizzazione della Località dei Dati

YARN cerca di eseguire i task vicino ai dati per evitare l'uso eccessivo della larghezza di banda:
- **Nodi locali ai dati**: Prima scelta per l'esecuzione dei task.
- **Rack locale**: Se non disponibile un nodo locale.
- **Fuori dal rack**: Come ultima opzione.

## Flusso dei dati

#### Flusso dei Dati MapReduce con un Reducer

1. **Phase Map**:
 - I task Map elaborano i dati di input e producono coppie chiave-valore intermedie.
2. **Shuffling & Sorting**:
 - Le coppie intermedie vengono raggruppate e ordinate per chiave.
3. **Phase Reduce**:
 - Il singolo task Reduce riceve tutte le coppie chiave-valore ordinate e le aggrega per produrre l'output finale.
#### Flusso dei Dati MapReduce con più Reducer

1. **Phase Map**:
 - I task Map elaborano i dati di input e producono coppie chiave-valore intermedie.
2. **Shuffling & Partitioning**:
 - L'output dei task Map viene suddiviso in partizioni (una per ciascun Reducer) in base a una funzione di partizionamento.
 - Le coppie chiave-valore intermedie vengono ordinate e inviate ai rispettivi Reducer.
3. **Phase Reduce**:
 - Ogni task Reduce riceve un sottoinsieme delle chiavi e aggrega i valori associati, producendo una parte dell'output finale.
#### Flusso dei Dati MapReduce senza Reducer

1. **Phase Map**:
 - I task Map elaborano i dati di input e producono coppie chiave-valore intermedie.
2. **Shuffling & Sorting**:
 - Nessuna aggregazione viene effettuata poiché non è presente un task Reduce.
 - L'output delle fasi Map diventa l'output finale direttamente.

#### Differenze tra i tre scenari

1. **Con un Reducer**:
 - Shuffling & Sorting delle chiavi.
 - Un singolo task Reduce per l'aggregazione delle chiavi.
2. **Senza Reducer**:
 - Nessun Reduce viene eseguito.
 - L'output delle Map diventa l'output finale.
3. **Con più Reducer**:
 - Shuffling & Sorting delle chiavi.
 - Output di ogni task Map partizionato tra diversi Reducer.
 - Ogni reducer riceve un sottoinsieme delle chiavi da aggregare.

#### Combiner

- Il **Combiner** è un'ottimizzazione locale per ridurre il traffico di rete tra i task Map e Reduce.
- Aggrega localmente i dati intermedi con la stessa chiave prima di trasmetterli ai Reducer.
- Anche con un combiner, il task Reduce finale è comunque necessario per aggregare i risultati da più nodi.

## Distribuzioni di Hadoop

- **Cloudera CDH e MapR**: 
 - Sono distribuzioni integrate basate su Hadoop che contengono tutti i componenti necessari per la produzione.
 - **Cloudera**: Si è fusa con Hortonworks nel 2019.
 - **MapR**: Utilizza un file system proprietario (MapR-FS) invece di HDFS.
Queste distribuzioni includono framework di sicurezza (es. Sentri, Ranger) per gestire l'accesso e le politiche di sicurezza.
Cloudera CDH può essere installato su un **cluster a singolo nodo** usando hypervisor come VMware, KVM, VirtualBox o Docker. Le versioni QuickStart non sono adatte per l'uso in produzione.
**I/O del disco** è spesso il principale collo di bottiglia delle prestazioni.
**Fattori chiave di ottimizzazione**:

 - Numero di dischi per massimizzare la larghezza di banda I/O.
 - Configurazioni del BIOS.
 - Tipo di file system (es. EXT4 su Linux è preferito).
 - Gestione dei file aperti.
 - Dimensione ottimale del blocco HDFS.
 - Ottimizzazione dell'heap Java e delle garbage collection per le impostazioni JVM.
### Configurazione di Hadoop: Ottimizzazione dei Parametri

- **Mapper**: Determinato dal numero di blocchi dei file di input. La dimensione del blocco HDFS può essere regolata per influenzare il numero di mapper. Idealmente, si dovrebbero avere tra 10 e 100 mapper per nodo, con mapper che durano almeno un minuto.
- **Reducer**: Il numero può essere impostato dall'utente. Si consiglia un valore tra 0,95 e 1,75 moltiplicato per il numero di nodi e container massimi per nodo. Ridurre a zero se non è necessaria alcuna fase di riduzione.

### Consigli per le Prestazioni

- **Compressione**: Comprimere dati di input, output dei map e output dei reduce per risparmiare larghezza di banda.
- **Gestione del Buffer di Spill**: Ottimizzare il buffer di ordinamento e i record di spill per evitare eccessivi scritture su disco.
- **Traffico di rete**: Comprimere l'output dei map e usare un combiner per ridurre i dati trasmessi durante lo shuffle.

## Hadoop nel Cloud

#### Vantaggi

- **Scalabilità ed Elasticità**: Beneficia della flessibilità del Cloud.
- **Nessuna Gestione dell'Infrastruttura**: Non serve gestire direttamente l'infrastruttura.

#### Sfide Principali

- **Spostamento dei Dati nel Cloud**: Problemi di latenza e larghezza di banda.
- **Sicurezza e Privacy dei Dati**: Protezione dei dati trasferiti e archiviati nel Cloud.

### Amazon Elastic MapReduce (EMR)

- **Funzionamento**: Distribuisce il lavoro su server virtuali in AWS (EC2), supporta anche Hive, Pig, HBase, Spark.
- **Input/Output**: Interagisce con Amazon S3, HDFS, DynamoDB.
- **Creazione di un Cluster EMR**: Selezione del nome, release, applicazioni, istanze, e configurazione della chiave EC2.

### Google Cloud Dataproc

- **Funzionamento**: Esegue su Google Cloud, supporta anche Hadoop e Spark.
- **Input/Output**: Utilizza Cloud Storage e Bigtable, accesso tramite API REST e Cloud SDK.

**Java** è il principale linguaggio di programmazione per Hadoop. Composto da:
 - **Main**: Configura il job (mapper, reducer, partitioner).
 - **Mapper**: Elabora coppie (k,v).
 - **Reducer**: Aggrega i risultati da diverse chiavi.
---
### WordCount in Java

```java
import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {
    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```
- Il metodo `map` elabora una riga alla volta, divide la riga in token separati da spazi bianchi, tramite `StringTokenizer`, ed emette una coppia chiave-valore `<word, 1>`.
- Il metodo `reduce` somma i valori, che sono i conteggi delle occorrenze per ogni chiave.
- Il metodo `main` specifica vari aspetti del job.
- L'output di ogni map viene passato attraverso un combiner locale (uguale al Reducer) per l'aggregazione locale, dopo essere stato ordinato sulle chiavi.

**Esempio di input e output:**

Input:
```
Hello World Bye World
Hello Hadoop Goodbye Hadoop
```

Output:
```
Bye 1
Goodbye 1
Hadoop 2
Hello 2
World 2
```

**Esempio di flusso dei dati:**

1° output mapper:
```
< Hello, 1>
< World, 1>
< Bye, 1 >
< World, 1>
```

Dopo il combiner locale:
```
< Bye, 1 >
< Hello, 1>
< World, 2>
```

2° output mapper:
```
< Hello, 1>
< Hadoop, 1>
< Goodbye, 1>
< Hadoop, 1>
```

Dopo il combiner locale:
```
< Goodbye, 1>
< Hadoop, 2>
< Hello, 1>
```

Output del Reducer:
```
< Bye, 1>
< Goodbye, 1>
< Hadoop, 2>
< Hello, 2>
< World, 2>
```

Per quanto riguarda i **linguaggi di programmazione per Hadoop**:
- L'API Hadoop Streaming permette di scrivere funzioni Map e Reduce in linguaggi di programmazione diversi da Java.
- Utilizza gli stream standard Unix come interfaccia tra Hadoop e il programma MapReduce.
- Consente di utilizzare qualsiasi linguaggio (ad esempio Python) che possa leggere l'input standard (stdin) e scrivere sull'output standard (stdout) per scrivere il programma MapReduce.
- L'interfaccia del Reducer per lo streaming è diversa da Java: invece di ricevere `reduce(k, Iterator[v])`, lo script riceve una riga per valore, inclusa la chiave.
