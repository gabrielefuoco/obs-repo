
**Schema Riassuntivo: Strumenti di Programmazione e Apache Hadoop**

**1. Caratteristiche degli Strumenti di Programmazione**

   *   **1.1 Livello di Astrazione**
        *   1.1.1 **Basso Livello:** API potenti ma complesse, sfruttamento di meccanismi e istruzioni di basso livello.
        *   1.1.2 **Medio Livello:** Costrutti di programmazione limitati, dettagli di basso livello nascosti.
        *   1.1.3 **Alto Livello:** Interfacce di alto livello (IDE visivi), costrutti non correlati all'architettura sottostante.

   *   **1.2 Tipo di Parallelismo**
        *   1.2.1 **Parallelismo Dati:** Stesso codice eseguito in parallelo su diversi elementi dati.
        *   1.2.2 **Parallelismo Task:** Diversi task eseguiti in parallelo.

**2. Apache Hadoop**

   *   **2.1 Definizione:** Framework open-source per implementare il modello MapReduce.
        *   2.1.1 Progettato per applicazioni data-intensive scalabili.
        *   2.1.2 Supporta vari linguaggi (Java, Python).
        *   2.1.3 Esecuzione su sistemi paralleli e distribuiti.
   *   **2.2 Astrazione dai Problemi del Calcolo Distribuito:**
        *   2.2.1 Località dei dati.
        *   2.2.2 Bilanciamento del carico di lavoro.
        *   2.2.3 Tolleranza ai guasti.
        *   2.2.4 Risparmio di larghezza di banda di rete.

**3. Altri Framework basati su MapReduce**

   *   **3.1 Phoenix++:**
        *   3.1.1 Basato su C++.
        *   3.1.2 Utilizza chip multi-core e multi-processori a memoria condivisa.
        *   3.1.3 Gestisce creazione thread, partizione dati, pianificazione dinamica task e tolleranza ai guasti.
   *   **3.2 Sailfish:**
        *   3.2.1 Framework MapReduce con trasmissione batch dai mapper ai reducer.
        *   3.2.2 Utilizza *I-files* per l'aggregazione dei dati.

**4. Caratteristiche di Apache Hadoop**

   *   **4.1 Elaborazione Batch:**
        *   4.1.1 Efficiente per l'elaborazione batch.
        *   4.1.2 Inefficiente per applicazioni iterative (elaborazione basata su disco).
   *   **4.2 Community Open-Source:**
        *   4.2.1 Supporto da una vasta community.
        *   4.2.2 Aggiornamenti e correzioni di bug costanti.
   *   **4.3 Basso Livello di Astrazione:**
        *   4.3.1 API potenti ma non user-friendly.
        *   4.3.2 Richiede comprensione di basso livello del sistema.
        *   4.3.3 Sviluppo più complesso ma codice più efficiente.
   *   **4.4 Parallelismo Dati:**
        *   4.4.1 Dati di input partizionati in chunk.
        *   4.4.2 Elaborazione parallela da macchine diverse.
   *   **4.5 Tolleranza ai Guasti:**
        *   4.5.1 Elevata tolleranza ai guasti.
        *   4.5.2 Meccanismi di checkpoint e ripristino.

**5. Moduli di Hadoop**

   *   **5.1 Hadoop Distributed File System (HDFS):**
        *   5.1.1 File system distribuito.
        *   5.1.2 Tolleranza ai guasti con ripristino automatico.
        *   5.1.3 Portabilità su hardware e sistemi operativi eterogenei.
   *   **5.2 Yet Another Resource Negotiator (YARN):**
        *   5.2.1 Framework per la gestione delle risorse del cluster.
        *   5.2.2 Pianificazione dei job.
   *   **5.3 Hadoop Common:**
        *   5.3.1 Utility e librerie.
        *   5.3.2 Supporta gli altri moduli Hadoop.

---

**Schema Riassuntivo di Hadoop**

**1. Evoluzione di Hadoop**
    *   Piattaforma versatile che supporta diversi sistemi di programmazione.
    *   Esempi di sistemi supportati:
        *   Storm: Analisi di dati in streaming.
        *   Hive: Interrogazione di grandi dataset.
        *   Giraph: Elaborazione iterativa di grafi.
        *   Ambari: Provisioning e monitoraggio del cluster.

**2. Hadoop Distributed File System (HDFS)**
    *   Progettato per archiviare grandi volumi di dati con lettura veloce e tolleranza ai guasti.
    *   Distribuzione e replicazione dei file su diversi nodi.
    *   Organizzazione gerarchica dei file.
    *   Architettura Master-Workers:
        *   Namenode (Master): Gestisce il file system distribuito, mantenendo l'albero del file system e i metadati.
        *   Datanode (Workers): Memorizzano e recuperano blocchi di dati, comunicando con il namenode.
        *   Secondary Namenode (Opzionale): Tolleranza ai guasti, memorizza lo stato del file system.

**3. Caratteristiche di HDFS**
    *   Memorizzazione dei file come sequenza di blocchi di dati.
        *   Dimensione blocco predefinita: 128 MB (configurabile).
    *   Tolleranza ai guasti tramite replicazione dei blocchi.
        *   Fattore di replicazione: Configurabile alla creazione e modifica del file.
    *   Data Locality: Minimizza il trasferimento di dati sulla rete per efficienza.

**4. Flusso di Esecuzione (MapReduce)**
    *   Componenti:
        *   Dati di Input: File di input su HDFS.
        *   InputFormat: Definisce come i dati vengono suddivisi e letti per creare input split.
        *   InputSplit: Porzione di dati elaborata da un mapper.
        *   RecordReader: Converte uno split in coppie chiave-valore.

**5. Fasi del Flusso MapReduce**
    *   Mapper: Applica una funzione di mapping a coppie chiave-valore, producendo coppie chiave-valore.
    *   Combiner: Aggregazione locale dell'output del mapper (riduce il trasferimento dati).
    *   Partitioner: Suddivide l'output usando una funzione di hashing (stessa chiave nella stessa partizione).
    *   Shuffle e Ordinamento: Trasferimento in rete ai reducer (shuffling), ordinamento per chiave.
    *   Reducer: Aggregazione finale dei dati ricevuti.
    *   RecordWriter: Scrive le coppie chiave-valore di output, `OutputFormat` definisce il formato.

---

## Schema Riassuntivo MapReduce

**I. Basi di Programmazione MapReduce**

    *   **A. Componenti Principali:**
        *   1.  **Mapper:** Implementa la logica di mappatura.
            *   a. Estende la classe `Mapper`.
            *   b. Implementa il metodo `map`.
        *   2.  **Reducer:** Implementa la logica di riduzione.
            *   a. Estende la classe `Reducer`.
            *   b. Implementa il metodo `reduce`.
        *   3.  **Driver:** Configura il job e contiene il `main`.

    *   **B. Classe Mapper:**
        *   1.  Definizione: `class Mapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT>`
            *   a. `KEYIN`: Chiave di input.
            *   b. `VALUEIN`: Valore di input.
            *   c. `KEYOUT`: Chiave di output.
            *   d. `VALUEOUT`: Valore di output.
        *   2.  Metodi Sovrascrivibili:
            *   a. `void setup(Context context)`: Inizializzazione.
            *   b. `void map(KEYIN key, VALUEIN value, Context context)`: Elaborazione di ogni coppia chiave-valore.
            *   c. `void cleanup(Context context)`: Pulizia finale.

    *   **C. Classe Reducer:**
        *   1.  Definizione: `class Reducer<KEYIN, VALUEIN, KEYOUT, VALUEOUT>`
            *   a. `KEYIN`: Chiave di input.
            *   b. `VALUEIN`: Valore di input.
            *   c. `KEYOUT`: Chiave di output.
            *   d. `VALUEOUT`: Valore di output.
        *   2.  Metodi Sovrascrivibili:
            *   a. `void setup(Context context)`: Inizializzazione.
            *   b. `void reduce(KEYIN key, Iterable<VALUEIN> values, Context context)`: Elaborazione dei valori associati a una chiave.
            *   c. `void cleanup(Context context)`: Pulizia finale.

    *   **D. Classe Driver:**
        *   1.  Configurazione del job:
            *   a. Nome del job.
            *   b. Tipi di dati di input/output.
            *   c. Classi mapper e reducer.
            *   d. Altri parametri.
        *   2.  `Context`: Interazione con Hadoop, accesso a configurazione ed emissione output.

**II. Ordinamento Secondario**

    *   **A. Concetto:** Controllo più fine dell'ordinamento usando una chiave composita `<chiave_primaria, chiave_secondaria>`.
    *   **B. Componenti:**
        *   1.  *Partitioner* personalizzato: Assegna tuple con la stessa chiave primaria allo stesso reducer.
        *   2.  *Comparator* personalizzato: Ordina le tuple usando l'intera chiave composita.
        *   3.  *Group comparator* personalizzato: Raggruppa le tuple per chiave primaria prima del metodo `reduce`.
    *   **C. Esempio:** `<user_id, timestamp>`: partizionamento per `user_id`, ordinamento per chiave composita, raggruppamento per `user_id`.

**III. Creazione di un Indice Inverso con Hadoop**

    *   **A. Obiettivo:** Mappare parole agli ID dei documenti e al numero di occorrenze.
    *   **B. MapTask (Mapper):**
        *   1.  Riceve un elenco di documenti.
        *   2.  Ottiene il `documentID`.
        *   3.  Analizza le righe di testo ed emette coppie `<word, documentID:numberOfOccurrences>` (`numberOfOccurrences = 1`).
        *   4.  Preelaborazione delle parole (rimozione punteggiatura, lemmatizzazione, stemming).
        *   5.  Utilizzo di `Text` e `IntWritable` per serializzazione.
    *   **C. CombineTask (Combiner):**
        *   1.  Aggrega i dati intermedi sommando le occorrenze di ogni parola in un documento.
        *   2.  Emette coppie `<word, documentID:sumNumberOfOccurrences>`.
    *   **D. ReduceTask (Reducer):**
        *   1.  Per ogni parola, produce l'elenco dei documenti e il numero di occorrenze.
        *   2.  Output: `<word, List(documentID:numberOfOccurrences)>`.

**IV. Configurazione del Job in MapReduce**

    *   **A. Elementi Principali:**
        *   1.  Classi Java per `mapper`.
        *   2.  Classi Java per `combiner`.
        *   3.  Classi Java per `reducer`.

---

**Schema Riassuntivo: Logica di Elaborazione Dati**

1.  **Implementazione Logica di Elaborazione Dati**

    *   Definisce la logica per l'elaborazione dei dati in un sistema (es. MapReduce).

2.  **Definizione Formati Chiave/Valore**

    *   Necessità di definire formati chiave/valore sia per l'input che per l'output.
    *   I formati specificano il tipo di dati elaborati in ogni fase.

3.  **Specificare Percorsi Input/Output**

    *   Definizione dei percorsi nel file system distribuito (es. HDFS).
    *   Percorsi per i dati di input e per la scrittura dei dati di output.

---
