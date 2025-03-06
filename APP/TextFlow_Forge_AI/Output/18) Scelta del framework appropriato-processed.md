
**Schema Riassuntivo: Selezione del Framework Big Data**

**I. Fattori Principali per la Selezione del Framework**

    *   **A. Dati di Input:**
        *   1.  Volume
        *   2.  Velocità
        *   3.  Varietà
    *   **B. Classe dell'Applicazione:**
        *   1.  Batch
        *   2.  Streaming
        *   3.  Basate su Grafi (implicito)
        *   4.  Basate su Query (implicito)
    *   **C. Infrastruttura:**
        *   1.  On-premise
        *   2.  Cloud
        *   3.  Ibrida

**II. Dati di Input: Dettagli**

    *   **A. Volume:**
        *   1.  Influenza storage ed elaborazione.
        *   2.  Soluzioni distribuite necessarie (es. HDFS).
            *   a. HDFS: replica, tolleranza ai guasti, scalabilità.
        *   3.  Sistemi distribuiti per scalabilità orizzontale (es. Hadoop, Spark).
            *   a. Hadoop: elaborazione parallela.
            *   b. Spark: elaborazione in-memory (iterativa, interattiva).
        *   4.  Dati ad alta dimensionalità: riduzione della dimensionalità (PCA, SVD).
            *   a. Supportata da Spark (MLlib).
    *   **B. Velocità:**
        *   1.  Richiede elaborazione in tempo reale.
        *   2.  Tecniche: windowing, aggregazione basata sul tempo.
        *   3.  Bassa latenza (elaborazione in-memory).
        *   4.  Framework:
            *   a. Spark Streaming: micro-batch (scenari specifici).
            *   b. Storm: elaborazione stream (flussi in tempo reale).
    *   **C. Varietà:**
        *   1.  Richiede gestione flessibile di integrazione, trasformazione, analisi.
        *   2.  Sistemi ETL (es. Hive, Pig): pre-elaborazione.
        *   3.  Hive e Pig: analisi dati strutturati.
        *   4.  Dati non strutturati (es. testo): tecniche speciali (NLP).
        *   5.  Spark: versatile per dati eterogenei.
            *   a. API per batch, stream, machine learning, grafi.
            *   b. DataFrames per diversi tipi di dati (CSV, JSON, database).

**III. Classe dell'Applicazione: Dettagli**

    *   **A. Batch:**
        *   1.  Elaborazione di grandi quantità di dati raccolti.
        *   2.  Analisi dati storici, report, analisi complesse.
        *   3.  Framework:
            *   a. Spark: motore veloce e flessibile, API di alto livello, machine learning.
            *   b. Hadoop: storage tollerante ai guasti, elaborazione distribuita.
            *   c. Airflow: workflow orientati al batch, orchestrazione, automazione.
    *   **B. Stream:**
        *   1.  Elaborazione e analisi dati in tempo reale.
        *   2.  Senza archiviazione centralizzata.
        *   3.  Settori: finanza, telecomunicazioni, trasporti.
        *   4.  Esempi: analisi dati in tempo reale, rilevazione frodi, monitoraggio IoT.

---

**I. Elaborazione di Stream di Dati**

*   **A. Storm:**
    *   Elaborazione in tempo reale a bassa latenza.
    *   Scalabile e fault-tolerant.
*   **B. Spark:**
    *   Elaborazione di stream micro-batch tramite API specializzate.

**II. Applicazioni Basate su Grafi**

*   **A. Definizione:** Elaborazione e analisi di dati interconnessi in reti complesse.
    *   Analisi delle relazioni tra nodi per scoprire schemi.
*   **B. Esempi:**
    *   Analisi di social network.
    *   Motori di raccomandazione.
    *   Rilevazione di frodi.
*   **C. Framework:**
    *   **1. MPI:** Controllo a basso livello sul parallelismo e la comunicazione.
    *   **2. Spark:** API di alto livello (es. GraphX) per elaborazione efficiente e scalabile.

**III. Applicazioni Basate su Query**

*   **A. Definizione:** Accesso rapido ed efficiente a grandi volumi di dati tramite query.
    *   Archiviazione dei dati in un sistema distribuito.
    *   Utilizzo di linguaggi di query (es. SQL).
*   **B. Esempi:**
    *   Business intelligence.
    *   Esplorazione dei dati.
    *   Analisi di dati *ad hoc*.
*   **C. Framework:**
    *   **1. Hive:** Interfaccia simile a SQL.
    *   **2. Pig:** Linguaggio di scripting semplice.
    *   **3. Spark SQL:** Esecuzione di query e analisi dati con sintassi SQL.

**IV. Infrastruttura: On-Premise**

*   **A. Definizione:** Distribuzione di hardware e software all'interno dei locali dell'organizzazione.
    *   Elaborazione e archiviazione dati in un data center proprietario.
    *   Maggiore sicurezza e conformità normativa.
*   **B. Caratteristiche:** Macchine interconnesse con hardware commodity.
*   **C. Framework:**
    *   **1. Hadoop:**
        *   Elaborazione di grandi dataset a costi inferiori su hardware commodity eterogeneo.
        *   Si basa su qualsiasi tipo di storage su disco.
        *   HDFS distribuisce i dati su macchine diverse con sistemi operativi diversi senza driver speciali.
    *   **2. Spark:**
        *   Elaborazione rapida in memoria di grandi quantità di dati.
        *   Costo maggiore (richiede grandi quantità di RAM).

**V. Infrastruttura: Cloud-Based**

*   **A. Definizione:** Utilizzo di risorse cloud per archiviare ed elaborare i dati.

---

**I. Servizi Cloud e Big Data**

*   **A. Vantaggi:**
    *   Scalabilità e flessibilità (aggiunta/rimozione risorse in base alle esigenze)
*   **B. Esempi di servizi:**
    *   `Amazon EMR`
    *   `Azure HDInsight`
    *   `Google Cloud Dataproc`
*   **C. Framework gestiti:**
    *   `Hadoop`
    *   `Spark` (ottimizzati per il cloud)
*   **D. Problematiche:**
    *   Privacy e gestione dei dati
        *   Sicurezza
        *   Conformità normativa (es. GDPR, CCPA)
        *   Vincoli giurisdizionali
        *   Controllo dell'accesso ai dati

**II. Infrastruttura Ibrida**

*   **A. Definizione:** Combinazione di risorse on-premise e cloud-based (elaborazione, storage, servizi)
*   **B. Framework per elaborazione dati ibrida:**
    *   **Spark:**
        *   Supporta lettura/scrittura da diverse sorgenti (HDFS, NFS, Amazon S3, Azure Blob Storage, Google Cloud Storage)
    *   **Kafka:**
        *   Piattaforma di streaming distribuita per pipeline di dati in tempo reale tra ambienti on-premise e cloud
        *   Livello di integrazione dati
    *   **Airflow:**
        *   Ingestione ed elaborazione dati tra ambienti on-premise e cloud

**III. Altri Fattori Influenzanti la Scelta**

*   **A. Competenze:** Competenze di programmazione di designer e sviluppatori
*   **B. Ecosistema:** Caratteristiche dell'ecosistema del framework
*   **C. Community:** Dimensione e attività della community di sviluppatori
*   **D. Privacy:** Requisiti di privacy dei dati (accesso ed elaborazione)
*   **E. Costi:** Costi dell'infrastruttura hardware/software

**IV. Aspetti Cruciali nella Scelta del Framework (Data Science/Machine Learning)**

*   **A. Disponibilità di librerie:**
    *   Ricchezza e qualità delle librerie di analisi dati e machine learning
    *   Accelerazione dello sviluppo e accesso a funzionalità avanzate
    *   Librerie specializzate (es. NLP, Visione Artificiale)
*   **B. Livello di astrazione del modello di programmazione:**
    *   Influenza la produttività e la complessità dello sviluppo
    *   Alto livello di astrazione semplifica il codice (leggibilità e manutenibilità)
    *   Concentrazione sulla logica del problema

---

**Schema Riassuntivo: Livelli di Astrazione nella Programmazione**

*   **Alto Livello di Astrazione:**

    *   Maggiore semplicità e velocità di sviluppo.
    *   Minore controllo diretto sull'hardware.
    *   Richiede minore competenza di programmazione.

*   **Basso Livello di Astrazione:**

    *   Maggiore controllo sull'hardware.
    *   Richiede maggiore competenza di programmazione.
    *   Tempo di sviluppo più lungo.

---
