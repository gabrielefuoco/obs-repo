
**I. Sistemi Simili a SQL**

*   **A. Obiettivo:** Combinare l'efficacia di Hadoop con la facilità d'uso di linguaggi simili a SQL per analisi dati efficienti.
*   **B. Esempi:**
    *   Apache Hive: Data warehouse su Hadoop per gestione dati su larga scala.
    *   Apache Pig: Framework Hadoop con linguaggio simile a SQL per flussi di dati su larga scala.
    *   Apache Impala: Motore di query parallelo per bassa latenza e alta concorrenza su Hadoop, simile a RDBMS.

**II. Apache Hive**

*   **A. Definizione:** Sistema di data warehouse basato su Hadoop.
*   **B. Funzionamento:**
    *   Utilizza HiveQL (linguaggio simile a SQL) per scrivere query.
    *   HiveQL compilato in job MapReduce eseguiti su Hadoop.
    *   Motore SQL che compila automaticamente query in job MapReduce.
*   **C. Motivazione:** MapReduce è troppo di basso livello per analisi dati di routine.

**III. Caratteristiche di Apache Hive**

*   **A. Supporto SQL:** Supporta gran parte dello standard SQL e estensioni per Hadoop.
*   **B. Schema-on-Read:** Non richiede definizione della struttura della tabella prima dell'importazione; la struttura è proiettata durante l'esecuzione della query.
*   **C. Parallelismo dei Dati:** Esecuzione della stessa query su diverse porzioni di dati.
*   **D. Elevato Livello di Astrazione:** Utilizzo di HiveQL, basato su concetti di database relazionali.
*   **E. Utilizzo:** Interrogazioni e report su grandi set di dati.
*   **F. Supporto:** Vasta community e aziende (Facebook, Netflix, Yahoo!, Airbnb).

**IV. Concetti Principali di Hive**

*   **A. DDL e DML:**
    *   Operazioni DDL: Creazione, modifica, esplorazione ed eliminazione di tabelle.
    *   Operazioni DML: Caricamento, inserimento, aggiornamento, eliminazione e unione di dati.
    *   Astrazioni HiveQL: Basate su concetti di database relazionali (tabella, riga, colonna).
    *   Funzione: Adattatore tra Hadoop e strumenti di analisi dati basati su database relazionali (ETL e BI).
*   **B. OLAP e Funzioni Utente:**
    *   Progettato per OLAP (elaborazione analitica online), non OLTP (elaborazione delle transazioni online).
    *   Non fornisce accesso in tempo reale ai dati.
    *   Supporta tre tipi di funzioni:
        *   Funzioni definite dall'utente (UDF)
        *   Funzioni aggregate definite dall'utente (UDAF)
        *   Funzioni generatrici di tabelle definite dall'utente (UDTF)
    *   Semplificano la scrittura di funzioni personalizzate (Java, Python).

**V. Architettura di Hive**

*   **A. Componenti:**
    *   Interfaccia utente (UI): Punto di ingresso (web o CLI).
    *   Driver: Riceve query, gestisce sessioni, fornisce API di esecuzione e recupero (JDBC/ODBC).
    *   Compilatore: Analizza query, esegue analisi semantica, genera piano di esecuzione.
        *   Converte la query in un albero di sintassi astratta (AST) e poi in un DAG.

---

**Schema Riassuntivo di Hive**

**1. Componenti Chiave di Hive**

*   **1.1 Ottimizzatore:**
    *   Migliora il piano di esecuzione per ridurre i tempi di elaborazione.
*   **1.2 Metastore:**
    *   Utilizza un RDBMS (default: Apache Derby) per memorizzare metadati su tabelle e partizioni.
    *   Informazioni:
        *   Metadati delle entità relazionali persistenti e la loro mappatura su HDFS.
        *   Dettagli sulle colonne e i tipi di colonna.
        *   Serializer/deserializer per lettura/scrittura dati.
        *   Posizioni dei file HDFS.
    *   `SCHEMA` (o `DATABASE`): Specifica un metastore diverso da quello predefinito.
*   **1.3 Motore di Esecuzione:**
    *   Esegue il piano di esecuzione (DAG) generato dal compilatore.
    *   Gestisce le dipendenze tra le fasi del DAG.
    *   Esegue le fasi sui componenti di sistema appropriati.
*   **1.4 HDFS:**
    *   File system distribuito sottostante per l'archiviazione dei dati.

**2. Flusso di Esecuzione di una Query HiveQL**

*   **2.1 Avvio della Query:**
    *   L'interfaccia utente avvia la query nel Driver.
    *   Il Driver crea un handle di sessione e lo invia al Compiler.
*   **2.2 Compilazione e Ottimizzazione:**
    *   Il Compiler ottiene i metadati dal Metastore.
    *   Controllo dei tipi delle espressioni.
    *   Ottimizzazioni all'AST (es. potatura delle partizioni).
*   **2.3 Esecuzione MapReduce:**
    *   L'albero degli operatori viene convertito in un DAG di job MapReduce.
    *   I job MapReduce vengono sottomessi al motore MapReduce sottostante.
*   **2.4 Serializzazione/Deserializzazione:**
    *   SerDe serializza e deserializza i dati per un formato di file specifico durante la scrittura su HDFS.
*   **2.5 Restituzione dei Risultati:**
    *   Il Driver restituisce i risultati all'utente dopo il completamento dei job MapReduce.

**3. Basi di Programmazione: DDL (Data Definition Language)**

*   **3.1 Operazioni DDL:**
    *   Creano e modificano oggetti del database (tabelle, indici, ecc.).
*   **3.2 Creazione di una Tabella:**
    *   Sintassi:
        ```sql
        CREATE [REMOTE]{SCHEMA|DATABASE} [IF NOT EXIST] db_name [LOCATION hdfs_path][ROW FORMAT row_format] [FIELDS TERMINATE BY char];
        ```
    *   Hive mantiene lo schema delle tabelle nel Metastore.
*   **3.3 Altre Operazioni DDL:**
    *   `SHOW`, `ALTER`, `DESCRIBE`, `TRUNCATE`, `DELETE` per modificare e rimuovere tabelle e righe.

**4. Basi di Programmazione: DML (Data Manipulation Language)**

*   **4.1 Operazioni DML:**
    *   Inseriscono, eliminano e aggiornano dati.
*   **4.2 Caricamento Dati:**
    *   Sintassi:
        ```sql
        LOAD DATA[LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename [PARTITION {partcol1=val1, partcol2=val2, "..."};]
        ```
    *   Le operazioni di caricamento sono copie/spostamenti di dati.
    *   Hive non esegue trasformazioni durante il caricamento.
    *   `OVERWRITE` sovrascrive i dati esistenti.
*   **4.3 Inserimento Dati da Query:**
    *   Sintassi:
        ```sql
        INSERT OVERWRITE TABLE tabname [IF NOT EXISTS] select_statement FORM from_statement;
        ```
*   **4.4 Altre Operazioni DML:**
    *   `UPDATE` e `DELETE` per ulteriori modifiche ai dati.

**5. Esempio: Analisi delle Valutazioni dei Film**

*   **5.1 Creazione della Tabella:**
    ```sql
    CREATE TABLE data { userid INT, movieid INT, rating INT, timestamp DATE }
    ```
*   **5.2 Caricamento dei Dati:**
    ```sql
    LOAD DATA LOCAL INPATH '<path>/data' OVERWRITE INTO TABLE data;
    ```
*   **5.3 Analisi Statistiche:**
    *   Conteggio totale delle valutazioni:
        ```sql
        SELECT COUNT (*) FORM data;
        ```
    *   Trovare i film più apprezzati:
        ```sql
        SELECT movieid, COUNT (rating) AS num_ratings FROM data GROUP BY movieid ORDER BY num_rating DESC
        ```
*   **5.4 Integrazione con Altri Linguaggi (es. Python):**
    *   Utilizzo della clausola `TRANSFORM` per integrare script esterni.
    *   Esempio: Mappare il timestamp alla settimana dell'anno con `isocalendar()`.
    *   Le colonne del dataset vengono trasformate in stringhe per impostazione predefinita.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Elaborazione e Trasmissione Dati**

    A. Formattazione Dati:
        1. Stringhe di dati create.
    B. Delimitazione:
        1. Stringhe delimitate da tabulazioni.
    C. Trasmissione:
        1. Stringhe inviate allo script utente.

**II. Aggregazione Dati**

    A. Query:
        1. Esecuzione query `COUNT`.
    B. Raggruppamento:
        1. Dati raggruppati per settimana dell'anno.

---
