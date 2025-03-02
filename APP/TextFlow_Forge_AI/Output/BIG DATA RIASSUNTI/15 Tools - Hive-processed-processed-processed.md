
## Sistemi Simili a SQL per l'Analisi di Big Data

I sistemi simili a SQL, come Apache Hive, Pig e Impala, semplificano l'analisi di grandi dataset su piattaforme Hadoop, combinando l'efficienza di Hadoop con la familiarità del linguaggio SQL.  Questi sistemi permettono lo sviluppo di applicazioni di analisi dati più accessibili rispetto all'utilizzo diretto di MapReduce.

## Apache Hive: Data Warehouse su Hadoop

Apache Hive è un data warehouse costruito su Hadoop che utilizza **HiveQL**, un linguaggio simile a SQL, per interrogare i dati.  HiveQL viene tradotto in job MapReduce, sfruttando il parallelismo di Hadoop per l'elaborazione di grandi volumi di dati.  La sua progettazione risolve la complessità di MapReduce per le attività analitiche di routine.

## Caratteristiche Chiave di Apache Hive

Hive presenta diverse caratteristiche importanti:

* **Schema-on-read:** La struttura dei dati (schema) viene definita solo al momento dell'esecuzione della query, non richiedendo una definizione preventiva.
* **Parallelismo dei dati:**  Le query vengono eseguite in parallelo su diverse partizioni dei dati, accelerando l'elaborazione.
* **Alto livello di astrazione:** HiveQL offre un'interfaccia utente familiare agli utenti di database relazionali, semplificando l'interazione con Hadoop.
* **Supporto DDL e DML:**  Fornisce funzionalità complete per la definizione (DDL) e manipolazione (DML) dei dati, inclusi creazione, modifica ed eliminazione di tabelle, e operazioni di caricamento, inserimento, aggiornamento ed eliminazione di dati.
* **OLAP focalizzato:** Hive è ottimizzato per l'elaborazione analitica online (OLAP), non per le transazioni online (OLTP).
* **Funzioni Utente:** Supporta UDF (User Defined Functions), UDAF (User Defined Aggregate Functions) e UDTF (User Defined Table Generating Functions) in linguaggi come Java e Python per estendere le funzionalità.
* **Ampia adozione:** È utilizzato da numerose aziende, tra cui Facebook, Netflix, Yahoo! e Airbnb.


## Architettura di Apache Hive

![[]] L'architettura di Hive comprende un'interfaccia utente (UI), un driver che gestisce le sessioni e fornisce API (JDBC/ODBC), e un compilatore che analizza le query, genera un piano di esecuzione e lo invia ad Hadoop per l'elaborazione.

---

Hive è un data warehouse costruito su Hadoop, che utilizza HiveQL per interagire con i dati memorizzati in HDFS.  Il processo di elaborazione di una query HiveQL prevede diverse fasi chiave:

1. **Compilazione:** La query viene convertita in un albero di sintassi astratto (AST), sottoposta a controlli di errore e quindi in un Directed Acyclic Graph (DAG). Il compilatore utilizza i metadati delle tabelle e delle partizioni, ottenuti dal Metastore (un RDBMS, di default Apache Derby), per ottimizzare l'AST (ad esempio, tramite potatura delle partizioni).

2. **Ottimizzazione:** Il piano di esecuzione viene ottimizzato per ridurre i tempi di elaborazione.

3. **Esecuzione:** Il DAG viene eseguito dal motore di esecuzione su HDFS, utilizzando un sistema di serializzazione/deserializzazione (SerDe) per gestire i dati in diversi formati di file.  I job MapReduce vengono sottomessi e gestiti dal motore di esecuzione.

4. **Metastore:**  Memorizza informazioni strutturali sulle tabelle e le partizioni in HDFS, inclusi metadati, tipi di colonna e posizione dei file.  È possibile specificare un metastore alternativo tramite l'opzione `SCHEMA` o `DATABASE`.

5. **HDFS:** Il file system distribuito che archivia i dati.


Hive supporta operazioni DDL (Data Definition Language) per creare e gestire oggetti del database (es. `CREATE TABLE`, `SHOW TABLES`, `ALTER TABLE`, `DESCRIBE TABLE`, `TRUNCATE TABLE`, `DROP TABLE`) e DML (Data Manipulation Language) per manipolare i dati.  Le istruzioni DML includono `LOAD DATA` per caricare dati (copie/spostamenti, senza trasformazioni), `INSERT INTO` per inserire risultati di query, `UPDATE` e `DELETE` per modificare i dati.  `OVERWRITE` permette di sovrascrivere dati esistenti durante il caricamento.

Un esempio di analisi di dati su valutazioni di film illustra l'utilizzo di Hive:  viene creata una tabella, i dati vengono caricati da un file e vengono eseguite query HiveQL per analisi statistiche (es. trovare i film più apprezzati).  Hive permette anche integrazioni con altri linguaggi, come Python, tramite la clausola `TRANSFORM` per elaborazioni più complesse (es. mappatura del timestamp alla settimana dell'anno).  In questo caso, i dati vengono trasformati in stringhe delimitate da tabulazioni prima di essere passate allo script esterno.

---

I dati vengono aggregati settimanalmente, considerando le settimane dell'anno come unità di raggruppamento.

---
