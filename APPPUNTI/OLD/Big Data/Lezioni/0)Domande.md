**Java e Programmazione**
1. Lambda expression
2. Benefici Java stream
3. Java stream lazy

**Modelli di Programmazione Parallela**
1. Parametri MPI
2. Speedup, tempo esecuzione parallelo e sequenziale
3. Legge di Amdahl
4. BSP in generale
5. Costo del calcolo BSP
6. Send receive non bloccanti e bloccanti
7. Comunicazione in MPI sincrona e asincrona e meccanismi
8. Caratteristiche di un programma in parallelo
9. Superlinear speedup

**Hadoop e MapReduce**
1. Differenze Spark e Hadoop
2. WordCount 
3. Mapper e Reducer
4. Spark e Hadoop convenienza
5. WordCount reverse codice(chiave lunghezza parole)
6. Combiner in MapReduce 
7. Numero di reducer e mapper
8. WordLengthCount
9. Pseudocodice funzioni map e reduce
10. Architettura HDFS e file di configurazione delle risorse

**Apache Spark**
1. RDD
2. Spark lazy execution

**Apache Storm**
1. Che tipologia di programmi esegue Storm
2. Possono esserci più spout?
3. Quali metodi deve implementare spout e quali bolt
4. Topologia Storm

**Altri Concetti e Tecnologie**
1. Hama
2. ZooKeeper
3. Trajectory discovery
4. Logica di Hive
5. Watermark

**Nota:** Da quest'anno il programma è cambiato, non si fa più Hama e si studiano GraphX e Apache Airflow (slide su Teams)

### Obiettivi
• Conoscenza delle problematiche relative all’analisi dei Big Data
• Conoscenza della programmazione funzionale e delle JAVA Stream API
	- Possono essere definiti come una sequenza di elementi da una sorgente su cui è possibile eseguire operazioni aggregate.
• Conoscenza del paradigma MapReduce e del framework Apache Hadoop
	- MapReduce: Modello di programmazione funzionale
	- Apace Hadoop: principale implementazione open source del modello
• Conoscenza di Apache HIVE
	- Apache Hive è un'infrastruttura costruita su Apache Hadoop per fornire riepilogo dei dati, interrogazione e analisi. Usa un linguaggio di interrogazione SQL-like
• Conoscenza di Apache SPARK
	- Framework open source per l’analisi distribuita di dati in-memory
• Conoscenza dell’analisi in streaming e di Apache STORM
	- framework open source per analizzare flussi di dati (stream) in tempo reale
• Conoscenza del paradigma BSP e del framework Apache HAMA
	- Bulk Synchronous Parallelism (BSP) è un modello di calcolo parallelo che divide le computazioni in fasi. 
	-Apache Hama è un sistema di data analysis basato sul modello BSP.
• Conoscenza di sistemi emergenti per Big data processing
