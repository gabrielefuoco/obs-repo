
# Scegliere il Framework Giusto per Gestire i Big Data 

## Fattori Principali

I principali fattori da considerare quando si seleziona il framework appropriato per implementare un'applicazione big data includono:

* **Dati di input:**  Volume, velocità e varietà dei dati.
* **Classe dell'applicazione:** Tipo di applicazione (batch, streaming, basate su grafi, basate su query).
* **Infrastruttura:** Infrastruttura di storage e di calcolo (on-premise, cloud, ibrida).

### Dati di Input: Volume

Il *volume* dei dati influenza i requisiti di storage e di elaborazione.  L'archiviazione di grandi quantità di dati richiede soluzioni distribuite come Hadoop Distributed File System (**HDFS**), che fornisce replica, tolleranza ai guasti e scalabilità.  L'elaborazione richiede sistemi distribuiti in grado di scalare orizzontalmente, come **Hadoop** (per elaborazione parallela) e **Spark** (per elaborazione in-memory, adatta ad algoritmi iterativi e analisi interattive).  Dati ad alta dimensionalità possono richiedere tecniche di riduzione della dimensionalità (PCA, SVD), supportate da **Spark** tramite la libreria MLlib.


### Dati di Input: Velocità

La *velocità* di generazione dei dati richiede modelli e sistemi in grado di acquisire, elaborare e analizzare i dati in tempo reale.  Sono necessarie tecniche come il *windowing* e l'aggregazione basata sul tempo. L'elaborazione in tempo reale necessita di bassa latenza, come l'elaborazione in-memory.  Sistemi micro-batch come **Spark Streaming** possono essere utilizzati in alcuni scenari, mentre framework di elaborazione stream come **Storm** sono più adatti per flussi di dati in tempo reale.


### Dati di Input: Varietà

La *varietà* dei dati (eterogeneità di tipi, formati e sorgenti) richiede gestione flessibile dell'integrazione, trasformazione e analisi.  Sistemi ETL come **Hive** e **Pig** preelaborano i dati.  **Hive** e **Pig** analizzano dati strutturati.  Dati non strutturati (es. testo) richiedono tecniche speciali (NLP), offerte da pochi sistemi, tra cui **Spark**.  **Spark** è versatile per dati eterogenei, offrendo API per elaborazione batch, stream, machine learning, elaborazione di grafi e DataFrames per diversi tipi di dati (CSV, JSON, tabelle di database).


## Classe dell'Applicazione: Batch

Le applicazioni *batch* elaborano grandi quantità di dati raccolti e analizzati insieme, tipicamente durante periodi di bassa domanda. Sono utili per analizzare dati storici, generare report ed eseguire analisi complesse. **Spark** e **Hadoop** sono ampiamente utilizzati: Hadoop offre storage tollerante ai guasti e un framework di elaborazione distribuita; Spark offre un motore di elaborazione veloce e flessibile con API di alto livello e funzionalità di machine learning. **Airflow** può essere utilizzato per sviluppare e monitorare workflow orientati al batch, orchestrando e automatizzando i processi di elaborazione.


## Classe dell'Applicazione: Stream

Le applicazioni *stream* elaborano e analizzano i dati in tempo reale, senza archiviazione centralizzata. Sono utili in settori che richiedono analisi in tempo reale (finanza, telecomunicazioni, trasporti).

---

# Classi di Applicazioni Big Data e Framework Adatti

## Applicazioni basate su Stream

Le applicazioni big data basate su stream includono l'analisi di dati in tempo reale, la rilevazione di frodi e il monitoraggio in tempo reale di sensori e dispositivi IoT.  `Storm` e `Spark` sono entrambi utilizzati per l'elaborazione di stream di dati:

* **Storm:** offre elaborazione in tempo reale a bassa latenza, scalabile e fault-tolerant.
* **Spark:** fornisce elaborazione di stream micro-batch attraverso API specializzate.



## Applicazioni basate su Grafi

Le applicazioni basate su grafi sono progettate per elaborare e analizzare dati interconnessi in reti complesse o strutture grafiche. Ciò implica l'analisi delle relazioni tra i nodi dati in un grafo per scoprire schemi che potrebbero non essere evidenti utilizzando metodi di analisi tradizionali. Esempi di applicazioni big data basate su grafi includono l'analisi di social network, i motori di raccomandazione e la rilevazione di frodi.  `MPI` e `Spark` sono entrambi adatti all'elaborazione di grafi di big data:

* **MPI:** offre un controllo a basso livello sul parallelismo e la comunicazione.
* **Spark:** fornisce API di alto livello specializzate (ad esempio, GraphX) per l'elaborazione di grafi efficiente e scalabile.



## Applicazioni basate su Query

Le applicazioni basate su query sono progettate per fornire un accesso rapido ed efficiente a grandi volumi di dati tramite linguaggi di query e strumenti di ricerca. Ciò implica l'archiviazione dei dati in un sistema distribuito e l'utilizzo di linguaggi di query, come SQL, per recuperare i dati dal sistema. Esempi di applicazioni big data basate su query includono business intelligence, esplorazione dei dati e analisi di dati *ad hoc*. `Hive`, `Pig` e `Spark` sono adatti all'elaborazione di query su grandi dataset:

* **Hive:** fornisce un'interfaccia simile a SQL.
* **Pig:** fornisce un semplice linguaggio di scripting.
* **Spark SQL:** consente di eseguire query e analizzare i dati utilizzando una sintassi SQL.



## Infrastruttura: On-Premise

L'infrastruttura on-premise si riferisce alla distribuzione di hardware e software all'interno dei locali di un'organizzazione, senza richiedere il trasferimento di grandi quantità di dati a una posizione remota. In questo scenario, i dati vengono elaborati e archiviati in un data center proprietario, consentendo una maggiore sicurezza e una maggiore facilità di conformità alle rigorose normative in materia di accessibilità e privacy. Le infrastrutture on-premise, soprattutto per le organizzazioni con budget IT limitati, sono spesso costituite da macchine interconnesse dotate di hardware commodity:

* **Hadoop:** può essere utilizzato efficacemente per elaborare grandi dataset a costi inferiori su hardware commodity eterogeneo, basandosi su qualsiasi tipo di storage su disco per l'elaborazione dei dati. HDFS è in grado di distribuire i dati su macchine diverse che eseguono sistemi operativi diversi senza richiedere driver speciali.
* **Spark:** per le aziende IT con budget maggiori, è una soluzione efficace per l'elaborazione rapida in memoria di grandi quantità di dati. Tuttavia, opera a un costo maggiore perché richiede grandi quantità di RAM per avviare i nodi.



## Infrastruttura: Cloud-Based

L'infrastruttura cloud-based si riferisce all'utilizzo di risorse cloud per archiviare ed elaborare i dati. I servizi cloud sono generalmente adottati per la loro scalabilità e flessibilità, consentendo di aggiungere e rimuovere risorse in base alle esigenze dell'applicazione. Includono servizi specifici per l'elaborazione di big data, come `Amazon EMR`, `Azure HDInsight` e `Google Cloud Dataproc`, e framework big data completamente gestiti, come `Hadoop` e `Spark`, ottimizzati per il cloud. L'utilizzo di infrastrutture cloud pone molti problemi di privacy e gestione dei dati, inclusi quelli relativi alla sicurezza, alla conformità normativa, ai vincoli giurisdizionali e al controllo dell'accesso ai dati. Per evitare problemi legali, è fondamentale che un'infrastruttura cloud pubblica soddisfi le normative sui dati pertinenti, come il Regolamento generale sulla protezione dei dati (GDPR) e il California Consumer Privacy Act (CCPA).


## Infrastruttura: Ibrida

L'infrastruttura ibrida è una combinazione di elaborazione, storage e servizi in ambienti diversi, inclusi quelli on-premise e cloud-based. Diversi framework, a supporto dell'elaborazione dati ibrida, possono essere una scelta adatta in queste infrastrutture:

* **Spark:** supporta la lettura e la scrittura di dati da diverse sorgenti, tra cui HDFS, Network File Systems (NFS) e molti servizi di storage cloud object (ad esempio, Amazon S3, Azure Blob Storage e Google Cloud Storage).
* **Kafka:** è una piattaforma di streaming distribuita che consente di creare pipeline di dati in tempo reale tra ambienti on-premise e cloud-based e viene spesso utilizzata come livello di integrazione dei dati tra questi ambienti.
* **Airflow:** può essere utilizzato per l'ingestione e l'elaborazione dei dati, consentendo ai dati di fluire senza problemi tra ambienti on-premise e cloud-based.



## Altri fattori

Altri fattori che generalmente influenzano la scelta di designer e sviluppatori includono:

* Le competenze di programmazione dei designer e degli sviluppatori.
* Le caratteristiche dell'ecosistema del framework di programmazione scelto.
* Le dimensioni e il grado di attività della community di sviluppatori.
* I requisiti di privacy dei dati che definiscono l'accesso e l'elaborazione dei dati.
* I costi dell'infrastruttura hardware/software da utilizzare.



# Aspetti da considerare nella scelta di un framework di programmazione per Data Science e Machine Learning

Due aspetti cruciali da valutare nella scelta di un framework di programmazione per progetti di data science e machine learning sono:

**1. Disponibilità di librerie:**  La ricchezza e la qualità delle librerie di analisi dati e machine learning disponibili all'interno del framework sono fondamentali.  Una vasta scelta di librerie ben documentate e mantenute permette di accelerare lo sviluppo e di accedere a funzionalità avanzate senza dover implementare algoritmi da zero.  La presenza di librerie specializzate per specifiche attività (es. elaborazione del linguaggio naturale, visione artificiale) è un ulteriore vantaggio.

**2. Livello di astrazione del modello di programmazione:** Il livello di astrazione offerto dal framework influenza la produttività e la complessità dello sviluppo. Un alto livello di astrazione semplifica la scrittura del codice, rendendolo più leggibile e manutenibile.  Framework con un'elevata astrazione permettono di concentrarsi sulla logica del problema piuttosto che sui dettagli implementativi.  Al contrario, un basso livello di astrazione offre maggiore controllo ma richiede una maggiore competenza di programmazione e un tempo di sviluppo più lungo.


