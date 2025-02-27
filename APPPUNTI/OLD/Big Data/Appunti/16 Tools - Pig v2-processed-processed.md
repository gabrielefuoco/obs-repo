
## Apache Pig

* **Apache Pig** è un framework di flusso dati di alto livello per l'esecuzione di programmi MapReduce su Hadoop utilizzando un linguaggio simile a SQL.
* Pig è stato proposto per colmare il divario tra l'interrogazione dichiarativa di alto livello di SQL e lo stile procedurale di basso livello del modello di programmazione MapReduce.
* Le query sono scritte usando un linguaggio personalizzato, chiamato **Pig Latin**, e vengono quindi convertite in piani di esecuzione che vengono eseguiti come job MapReduce su Hadoop.

* Il sistema di programmazione Pig consente di comporre operazioni di manipolazione dati di alto livello utilizzando uno stile simile a SQL (ad esempio, operazioni di programmazione parallela, come FOREACH, FLATTEN e COGROUP), mantenendo le caratteristiche principali, i tipi di dati e i carichi di lavoro di MapReduce.
* Pig sfrutta un sistema di esecuzione multi-query per elaborare un intero script o un batch di istruzioni contemporaneamente. Pertanto, supporta sia il **parallelismo dei dati**, sfruttato suddividendo i dati in chunk ed elaborandoli in parallelo, sia il **parallelismo dei task**, quando più query vengono eseguite in parallelo sugli stessi dati.

* Pig è comunemente usato per lo sviluppo di query sui dati, analisi dati semplici e applicazioni di estrazione, trasformazione e caricamento (**ETL**), raccogliendo dati da diverse fonti, come stream, HDFS o file.
* Aziende e organizzazioni che utilizzano Pig in produzione includono LinkedIn, PayPal e Mendeley.
* Grazie al linguaggio di scripting Pig Latin, Pig fornisce un **livello di astrazione medio**, il che significa che, rispetto ad altri sistemi come Hadoop, gli sviluppatori Pig non sono tenuti a scrivere codici complessi e lunghi.

### Concetti principali

* **Modello dati**: Pig fornisce un modello dati nidificato, che consente di gestire dati complessi e non normalizzati. Supporta tipi scalari, come `int`, `long`, `double`, `chararray` (cioè stringa) e tipi `bytearray`.
* Inoltre, fornisce tre modelli di dati complessi:
 * **map**: una matrice associativa, dove una stringa è la chiave e il valore può essere di qualsiasi tipo.
 * **tuple**: un elenco ordinato di elementi dati, anche chiamati *campi*, dove ogni campo è un dato. Gli elementi di una tupla possono essere di qualsiasi tipo, consentendo tipi complessi nidificati.
 * **bag**: una raccolta di tuple, simile a un database relazionale. Le tuple in un bag corrispondono alle righe di una tabella, sebbene, a differenza di una tabella relazionale, i bag di Pig non richiedano che ogni tupla contenga lo stesso numero di campi. Un bag è anche identificato come una **relazione**.

* **Ottimizzazione delle query**: ogni script Pig viene tradotto in un insieme di job MapReduce che vengono automaticamente ottimizzati dal motore Pig utilizzando diverse regole di ottimizzazione, come la riduzione di istruzioni inutilizzate o l'applicazione di filtri durante il caricamento dei dati. Questa ottimizzazione può essere logica o fisica:
 * Le ottimizzazioni **logiche** riorganizzano il grafo di flusso dati logico inviato dall'utente, generando un nuovo grafo che è semanticamente equivalente all'originale ma può essere valutato in modo più efficiente.
 * Le ottimizzazioni **fisiche** riguardano il modo in cui il grafo di flusso dati logico viene tradotto in un piano di esecuzione fisico, come una serie di processi MapReduce.
* Un piano logico viene creato per ogni bag definito dall'utente. Quando il piano logico è costruito, non si verifica alcuna elaborazione, ma inizia solo quando l'utente invoca un comando `STORE` su un bag. A quel punto, il piano logico per quel bag viene trasformato in un piano fisico, che viene poi eseguito.
* Questa esecuzione pigra è vantaggiosa poiché consente il pipelining in memoria e altre ottimizzazioni.

### Architettura

![[|444](_page_6_Figure_2.jpeg)

* Pig è costituito da quattro componenti principali:
 * **Parser**, che gestisce tutte le istruzioni Pig Latin che controllano gli errori di sintassi e di tipo di dati. Produce in output un DAG, che rappresenta gli operatori logici degli script come nodi e il flusso di dati come archi.
 * **Optimizer**, che applica operazioni di ottimizzazione sul DAG prodotto dal parser per migliorare la velocità delle query, come split, merge, projection, pushdown, transform e reorder. Ad esempio, pushdown e projection omettono dati o colonne non necessari, riducendo la quantità di dati da elaborare.
 * **Compiler**, che genera una sequenza di job MapReduce, a partire dall'output dell'optimizer. Questo processo include altre ottimizzazioni, come la riorganizzazione dell'ordine di esecuzione.
 * **Execution engine**, che esegue i job MapReduce generati dal compilatore sul runtime Hadoop. L'output può essere visualizzato sullo schermo usando il comando `DUMP` o salvato in HDFS usando la funzione `STORE`.

### Fondamenti di programmazione

Le istruzioni in Pig Latin vengono espresse usando *bag*; un bag è una collezione di tuple che possono essere create:

* Utilizzando tipi di dati nativi supportati da Pig, sia semplici (es. `int`, `long`, `float`, `double`, `chararray`, e `boolean`) sia complessi (tuple, mappe, o bag annidati).
* Caricando dati dal file system. Esempio di istruzione Pig Latin che carica dati su alcuni studenti (nome ed età):

Altre comuni istruzioni Pig Latin sono:

* **FILTER**, che seleziona tuple da una relazione basata su una condizione.
* **JOIN** (interno o esterno), che esegue un join interno/esterno di due o più relazioni basate su valori di campo comuni.
* **FOREACH**, che genera trasformazioni di dati basate sulle colonne di dati. L'utilizzo di FOREACH è solitamente accoppiato all'operazione **GENERATE**, che permette di lavorare con le colonne di dati.

* **STORE**, che salva i risultati nel file system. Pig supporta la definizione di **UDF** (User Defined Functions) da parte del programmatore, permettendo di registrare un file JAR da utilizzare nello script e di assegnare alias alle UDF.

L'esempio seguente implementa un analizzatore di sentiment basato su dizionario. Dato un dizionario di parole associate a sentiment positivo o negativo, il sentiment di un testo (es. una frase, una recensione, un tweet o un commento) viene calcolato sommando i punteggi delle parole positive e negative nel testo e calcolando la valutazione media. Poiché Pig non fornisce una libreria integrata per l'analisi del sentiment, il sistema sfrutta dizionari esterni per associare le parole ai loro sentiment e determinare l'orientamento semantico delle parole di opinione.

![[|529](_page_11_Figure_5.jpeg)

Gli sviluppatori possono includere analisi avanzate in uno script definendo UDF. Ad esempio, la UDF **PROCESS** ha lo scopo di elaborare una tupla rimuovendo la punteggiatura come fase di pre-elaborazione. Altre funzionalità possono essere aggiunte al metodo *exec*, implementato in Java. La UDF definita in Java è registrata con l'alias PROCESS.

Successivamente, i dati relativi alle recensioni di testo vengono caricati da HDFS come file CSV delimitato da tabulazione, e similmente viene caricato il dizionario dei sentiment delle parole. Durante il caricamento dei dati da un file, l'utente può specificare lo schema tramite colonne denominate.

Una volta caricati i dati, ogni riga viene tokenizzata ed elaborata. Tramite l'operatore **FOREACH**, ogni riga dei dati di input viene prima elaborata usando la UDF precedentemente registrata e poi tokenizzata, producendo come output un array di token. Questo array viene successivamente appiattito tramite l'operatore integrato **FLATTEN**. L'operatore **GENERATE** viene utilizzato in collaborazione con FOREACH per produrre come output triple nella forma `<id recensione, testo, parola>`, che vengono memorizzate in un bag chiamato *words*.

### Strumenti di programmazione simili a SQL: Apache Pig

Esempio di programmazione (le parole vengono tokenizzate e appiattite)

Il codice identifica prima tutte le corrispondenze tra le parole di una recensione e le parole del dizionario unendo il bag intermedio creato sopra e le parole nel dizionario. I risultati vengono memorizzati in un bag chiamato *matches*, che viene poi iterato per assegnare il punteggio a ciascuna parola. Il risultato di questa operazione, memorizzato in un bag chiamato *matches_rating*, è una tripla nella forma `<id recensione, testo, valutazione>`.

(una valutazione dal dizionario è associata a ciascuna parola)
(vengono selezionati solo i campi (id, testo, valutazione))

La coppia `<id recensione, testo>` viene utilizzata per eseguire un'operazione di group by e raccogliere tutte le valutazioni trovate nel dizionario (ovvero, il bag chiamato *group_rating*). Dopo il raggruppamento, per ogni recensione, l'operatore integrato **AVG** viene utilizzato per aggregare tutte le valutazioni delle parole, e la valutazione finale di una recensione viene calcolata come media dei punteggi dei suoi token. Infine, il bag di output *avg_ratings* viene memorizzato in un file sul file system HDFS.

L'output del group by è nella forma `((id, testo), {(id, testo, valutazione)})`. La chiave `((id, testo))` è `$0` (accesso posizionale), mentre le triple associate a quella chiave sono accessibili tramite `$1`. Pertanto, `$1.$0` è l'id, `$1.$1` è il campo testo e `$1.$2` è il terzo campo, valutazione. `AVG($1.$2)` calcola la media di tutte le valutazioni, come `(3, 4)` per la prima recensione e `(-3, -2)` per la seconda recensione.
