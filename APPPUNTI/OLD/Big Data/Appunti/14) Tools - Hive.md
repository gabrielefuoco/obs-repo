
## Sistemi simili a SQL

I sistemi simili a SQL cercano di combinare l'efficacia di Hadoop con la facilità d'uso dei linguaggi simili a SQL, per consentire lo sviluppo di applicazioni di analisi dati semplici ed efficienti. Esempi di questi sistemi includono:

* **Apache Hive:** Un data warehouse costruito su Hadoop per la lettura, la scrittura e la gestione di dati su larga scala.
* **Apache Pig:** Un framework basato su Hadoop che utilizza un linguaggio simile a SQL per eseguire applicazioni di flusso di dati su larga scala.
* **Apache Impala:** Un motore di query massivamente parallelo che fornisce bassa latenza e alta concorrenza per le query analitiche su Hadoop, offrendo un'esperienza simile a quella di un RDBMS.

## Apache Hive

Apache Hive è un sistema di data warehouse basato su Hadoop. Permette agli utenti di scrivere query usando un linguaggio dichiarativo simile a SQL, chiamato **HiveQL**, che viene poi compilato in job MapReduce ed eseguito su Hadoop. Può essere visto come un motore SQL che compila automaticamente una query simile a SQL in un insieme di job MapReduce eseguiti su un cluster Hadoop, con funzionalità aggiuntive per la gestione di dati e metadati. Lo sviluppo di Hive è motivato dal fatto che, sebbene MapReduce sia flessibile, è troppo di basso livello per le attività di analisi dati di routine.

## Caratteristiche di Apache Hive

Hive supporta gran parte dello standard SQL e diverse estensioni per semplificare le interazioni con Hadoop. A differenza dei database tradizionali, non richiede la definizione della struttura della tabella prima dell'importazione dei dati; la struttura tabulare viene proiettata sui dati sottostanti durante l'esecuzione della query. Questa funzionalità è nota come **schema-on-read**.

Hive supporta il **parallelismo dei dati**, permettendo l'esecuzione della stessa query su diverse porzioni di dati. Offre anche un **elevato livello di astrazione**, consentendo agli sviluppatori di utilizzare HiveQL, basato sui concetti tradizionali dei database relazionali. È comunemente usato dagli analisti di dati per interrogazioni e report su grandi set di dati ed è supportato da una vasta community e da aziende come Facebook, Netflix, Yahoo! e Airbnb.

### Concetti principali: DDL e DML

Hive offre operazioni **Data Definition Language (DDL)** e **Data Manipulation Language (DML)** complete:

* Creazione, modifica, esplorazione ed eliminazione di tabelle.
* Caricamento, inserimento, aggiornamento, eliminazione e unione di dati sul file system.

Le astrazioni di HiveQL si basano sui concetti tradizionali dei database relazionali (tabella, riga, colonna), funzionando come adattatore tra Hadoop e gli strumenti di analisi dei dati basati su database relazionali (es. applicazioni ETL e BI).

### Concetti principali: OLAP, Funzioni Utente

Hive è progettato per l'**elaborazione analitica online (OLAP)**, non per l'elaborazione delle transazioni online (OLTP). A differenza di SQL Server, non fornisce accesso in tempo reale ai dati. Supporta tre tipi di funzioni per la manipolazione dei dati:

* Funzioni definite dall'utente (UDF)
* Funzioni aggregate definite dall'utente (UDAF)
* Funzioni generatrici di tabelle definite dall'utente (UDTF)

Queste funzioni semplificano la scrittura di funzioni personalizzate in linguaggi come Java o Python.

### Architettura

![[_page_7_Figure_2.jpeg|350]]

L'architettura di Hive comprende:

- **Interfaccia utente (UI):** Punto di ingresso per gli utenti tramite interfaccia web o CLI.
- **Driver:** Riceve le query, implementa i gestori di sessione e fornisce API di esecuzione e recupero (JDBC/ODBC).
- **Compilatore:** Analizza le query, esegue l'analisi semantica e genera un piano di esecuzione. Converte la query in un **albero di sintassi astratto (AST)** e poi, dopo i controlli di errore, in un DAG. Utilizza i metadati di tabelle e partizioni dal Metastore…
- **Ottimizzatore:** Migliora il piano di esecuzione per ridurre i tempi di elaborazione.
- **Metastore:** Utilizza un RDBMS (di default Apache Derby) per memorizzare informazioni strutturali su varie tabelle e partizioni nel warehouse. Queste informazioni includono metadati delle entità relazionali persistenti e la loro mappatura su HDFS, dettagli sulle colonne, i tipi di colonna, i serializer/deserializer per la lettura/scrittura dei dati e le corrispondenti posizioni dei file HDFS. L'opzione `SCHEMA` (o `DATABASE`) permette di specificare un metastore diverso da quello predefinito.
- **Motore di esecuzione:** Esegue il piano di esecuzione generato dal compilatore sotto forma di DAG. Gestisce le dipendenze tra le fasi del DAG e le esegue sui componenti di sistema appropriati.
- **HDFS:** È il file system distribuito sottostante utilizzato per l'archiviazione dei dati in Hadoop.

## Flusso di esecuzione di una query HiveQL

- L'interfaccia utente avvia la query nel Driver, che crea un handle di sessione e lo invia al Compiler per la generazione del piano di esecuzione.
- Il Compiler ottiene i metadati necessari dal Metastore, utilizzandolo per il controllo dei tipi delle espressioni e applicando ottimizzazioni all'AST (es. potatura delle partizioni basata sui predicati della query).
- L'albero degli operatori viene convertito in un DAG di job MapReduce, sottomessi al motore MapReduce sottostante per la valutazione.
- Una libreria di serializzazione-deserializzazione, chiamata **SerDe**, serializza e deserializza i dati per un formato di file specifico durante la scrittura su HDFS.
- Dopo il completamento dei job MapReduce, il Driver restituisce i risultati all'utente.

## Basi di programmazione: DDL

Hive supporta operazioni DDL e DML usando HiveQL. Le istruzioni DDL creano e modificano oggetti del database (tabelle, indici, ecc.). Per creare una tabella, si usa un'istruzione simile a:
```
CREATE [REMOTE]{SCHEMA|DATABASE} [IF NOT EXIST] db_name
[LOCATION hdfs_path][ROW FORMAT row_format] [FIELDS TERMINATE BY char];
```

Hive mantiene lo schema delle tabelle nel Metastore. Operazioni come `SHOW`, `ALTER`, `DESCRIBE`, `TRUNCATE`, `DELETE` permettono di modificare e rimuovere tabelle e righe.

## Basi di programmazione: DML

Le istruzioni DML inseriscono, eliminano e aggiornano dati. Per caricare dati in una tabella:
```
LOAD DATA[LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename
[PARTITION {partcol1=val1, partcol2=val2, "..."};]
```

Le operazioni di caricamento sono copie/spostamenti di dati; Hive non esegue trasformazioni. I risultati di una query possono essere inseriti nelle tabelle usando una sintassi simile. L'opzione `OVERWRITE` sovrascrive i dati esistenti. `UPDATE` e `DELETE` permettono ulteriori modifiche ai dati.
```
INSERT OVERWRITE TABLE tabname [IF NOT EXISTS] select_statement FORM from_statement;
```

## Esempio: Analisi delle valutazioni dei film

Questo esempio mostra come Hive può essere usato per memorizzare e analizzare dati sulle valutazioni di film.

Si crea una tabella con colonne `userid`, `movieid`, `rating`, `timestamp`:
```
CREATE TABLE data {
	userid INT,
	movieid INT,
	rating INT,
	timestamp DATE
}
```

I dati vengono caricati da un file di testo (locale o HDFS), sovrascrivendo il contenuto esistente.
```
LOAD DATA LOCAL INPATH '<path>/data'
OVERWRITE INTO TABLE data;
```

Analisi statistiche possono essere eseguite con query HiveQL:
```
SELECT COUNT (*)
FORM data;
```

Questa query trova i film più apprezzati.
```
SELECT movieid, COUNT (rating) AS num_ratings
FROM data
GROUP BY movieid
ORDER BY num_rating DESC
```

Hive permette analisi più complesse usando altri linguaggi. Ad esempio, uno script Python (`week_mapper.py`) mappa il timestamp alla settimana dell'anno e aggrega i risultati usando la funzione `isocalendar()`. Questo script può essere integrato nelle istruzioni HiveQL tramite la clausola `TRANSFORM`.

Per impostazione predefinita, le colonne del dataset vengono trasformate in stringhe. Queste stringhe sono poi delimitate da tabulazioni prima di essere inviate allo script utente.

Successivamente, viene eseguita una query `COUNT` per aggregare i dati. L'aggregazione avviene raggruppando i dati per settimana dell'anno.

