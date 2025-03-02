
Apache Pig è un framework di alto livello per l'elaborazione di dati su Hadoop, che utilizza il linguaggio Pig Latin, simile a SQL, per semplificare la programmazione MapReduce.  Pig Latin viene tradotto in job MapReduce, permettendo agli sviluppatori di concentrarsi sulla logica di manipolazione dati piuttosto che sui dettagli di implementazione.

Pig supporta operazioni di alto livello come `FOREACH`, `FLATTEN` e `COGROUP`, offrendo sia parallelismo dei dati (elaborazione parallela di chunk di dati) che parallelismo dei task (esecuzione parallela di più query).  È ampiamente utilizzato per query, analisi dati semplici e ETL (estrazione, trasformazione e caricamento) da diverse fonti, come HDFS, file e stream.  Aziende come LinkedIn, PayPal e Mendeley lo utilizzano in produzione.

Il modello dati di Pig è nidificato e supporta tipi scalari (`int`, `long`, `double`, `chararray`, `bytearray`) e tipi complessi:

* **map:** matrice associativa (chiave-valore).
* **tuple:** lista ordinata di elementi (campi) di qualsiasi tipo.
* **bag:** collezione di tuple, simile a una relazione in un database relazionale, ma senza vincoli sul numero di campi per tupla.

Pig offre un livello di astrazione medio, semplificando lo sviluppo rispetto a programmare direttamente in MapReduce.  Il motore Pig ottimizza automaticamente gli script traducendoli in job MapReduce efficienti, ad esempio rimuovendo istruzioni ridondanti o applicando filtri durante il caricamento dei dati.

---

## Riassunto di Pig: Ottimizzazione, Architettura e Programmazione

Pig è un sistema di elaborazione dati ad alto livello che semplifica l'analisi di grandi dataset su Hadoop.  Opera attraverso due tipi di ottimizzazione: **logica**, che riorganizza il grafo di flusso dati, e **fisica**, che ottimizza la traduzione del grafo in un piano di esecuzione MapReduce.  L'elaborazione è pigra: un piano logico viene creato per ogni *bag* (una collezione di tuple) solo quando viene invocato il comando `STORE`, permettendo il pipelining e altre ottimizzazioni.

### Architettura di Pig

L'architettura di Pig è composta da quattro componenti principali:

1. **Parser:** Analizza le istruzioni Pig Latin, verificando errori di sintassi e tipo, generando un grafo aciclico diretto (DAG) che rappresenta gli operatori e il flusso dati.
2. **Optimizer:** Applica ottimizzazioni al DAG per migliorare le prestazioni, come *split*, *merge*, *projection*, *pushdown*, *transform* e *reorder*.  Esempi includono l'eliminazione di dati o colonne non necessarie tramite *pushdown* e *projection*.
3. **Compiler:** Genera una sequenza di job MapReduce dall'output dell'optimizer, includendo ulteriori ottimizzazioni sull'ordine di esecuzione.
4. **Execution engine:** Esegue i job MapReduce su Hadoop. I risultati possono essere visualizzati con `DUMP` o salvati su HDFS con `STORE`.  ![[]]

### Programmazione in Pig Latin

Pig Latin utilizza i *bag* per rappresentare collezioni di tuple.  I tipi di dati supportati includono tipi semplici ( `int`, `long`, `float`, `double`, `chararray`, `boolean`) e tipi complessi (tuple, mappe, bag annidati). I dati possono essere caricati dal file system.

Le istruzioni Pig Latin comuni includono:

* **`LOAD`:** Carica dati (es. `A = LOAD 'studenti.txt' AS (nome:chararray, eta:int);`).
* **`FILTER`:** Filtra le tuple basate su una condizione.
* **`JOIN`:** Esegue join interni o esterni su relazioni.
* **`FOREACH`:** Applica trasformazioni di dati, spesso in combinazione con `GENERATE`.
* **`STORE`:** Salva i risultati nel file system.

Pig supporta anche le **UDF (User Defined Functions)**, permettendo agli utenti di definire e utilizzare funzioni personalizzate.

---

Questo documento descrive un'implementazione in Apache Pig per l'analisi del sentiment di un testo.  Il sistema utilizza dizionari esterni per assegnare un punteggio di sentiment (positivo o negativo) a ciascuna parola.  

L'elaborazione inizia caricando da HDFS i dati (recensioni di testo in formato CSV) e il dizionario dei sentiment.  Una UDF (User Defined Function) in Java, chiamata `PROCESS`, preelabora il testo rimuovendo la punteggiatura.  Successivamente, tramite gli operatori `FOREACH` e `FLATTEN`, il testo viene tokenizzato, generando un bag di parole (`words`) nella forma `<id recensione, testo, parola>`.

Un join tra il bag `words` e il dizionario dei sentiment crea il bag `matches`, contenente le corrispondenze tra le parole del testo e i loro punteggi.  Questo bag viene poi trasformato in `matches_rating`, con triple `<id recensione, testo, valutazione>`.

Un'operazione di `group by` su `<id recensione, testo>` crea il bag `group_rating`, raggruppando le valutazioni per ogni recensione.  Infine, l'operatore `AVG` calcola la media delle valutazioni per ogni recensione, generando il bag finale `avg_ratings`, che viene salvato su HDFS.  L'accesso agli elementi del risultato del `group by` avviene tramite accesso posizionale (`$0`, `$1`, ecc.).  Ad esempio, `AVG($1.$2)` calcola la media delle valutazioni (`$1.$2`).

![[]]

---

Il testo fornisce due esempi di valutazioni, rappresentate da coppie di numeri:  `(3, 4)` e `(-3, -2)`.  Non indica come calcolare la media di queste valutazioni né se si debba calcolare una media per ogni coppia o una media complessiva di tutte le valutazioni.  Per ottenere una media, è necessario specificare il metodo di calcolo (es. media aritmetica di ogni componente della coppia, media aritmetica delle medie delle coppie, etc.).

---
