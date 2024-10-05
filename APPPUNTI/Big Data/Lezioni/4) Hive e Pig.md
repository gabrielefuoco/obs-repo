| Termine                         | **Definizione**                                                                                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Hadoop                          | Un framework open source per l'elaborazione distribuita di grandi set di dati.                                                               |
| Hive                            | Un sistema di data warehousing basato su Hadoop che fornisce un'interfaccia simile a SQL (HiveQL) per interrogare e analizzare i dati.       |
| HiveQL                          | Un linguaggio di query simile a SQL utilizzato in Hive per interagire con i dati in HDFS.                                                    |
| Pig                             | Un motore di elaborazione dati di alto livello che opera su Hadoop e utilizza Pig Latin come linguaggio di scripting.                        |
| Pig Latin                       | Un linguaggio procedurale e orientato agli insiemi utilizzato in Pig per esprimere trasformazioni di dati.                                   |
| MapReduce                       | Un modello di programmazione per l'elaborazione parallela di grandi set di dati su un cluster di computer.                                   |
| HDFS                            | Hadoop Distributed File System, un sistema di file distribuito progettato per l'archiviazione di dati su larga scala.                        |
| UDF (User-Defined Function)     | Funzioni personalizzate scritte dagli utenti per estendere le funzionalità di Hive o Pig.                                                    |
| SerDe (Serializer/Deserializer) | Componenti utilizzati in Hive per serializzare e deserializzare dati in diversi formati.                                                     |
| Schema                          | Una definizione della struttura di una tabella in Hive, che specifica i nomi e i tipi di dati delle colonne.                                 |
| Partizione                      | Una suddivisione logica di una tabella in Hive, basata sui valori di una o più colonne.                                                      |
| Bag                             | Un tipo di dati in Pig Latin che rappresenta una collezione non ordinata di tuple, che possono avere lunghezze e tipi di campo diversi.      |
| Tuple                           | Un tipo di dati in Pig Latin che rappresenta una sequenza ordinata di campi, che possono avere tipi diversi.                                 |
| Canonicalizzazione              | Il processo di standardizzazione dei dati in un formato comune, come la normalizzazione degli URL.                                           |
| Join                            | Un'operazione di database che combina righe di due o più tabelle in base a una condizione di join.                                           |
| PageRank                        | Un algoritmo che assegna un punteggio di importanza alle pagine web, basato sul numero e sulla qualità dei link che puntano a quella pagina. |

---
### Hive
Hive è un'applicazione di data warehousing in Hadoop.
- Il linguaggio di query è HQL, variante di SQL.
- Le tabelle sono memorizzate su HDFS come file piatti.
- Sviluppato da Facebook, ora open source.
### Pig 
Ѐ un sistema di elaborazione dati su larga scala.
- Gli script sono scritti in Pig Latin, un linguaggio di flusso dati
- Sviluppato da Yahoo!, ora open source
- Circa 1/3 di tutti i job interni di Yahoo!
### Idea comune:
- Fornire un linguaggio di alto livello per facilitare l'elaborazione di grandi dati
- Il linguaggio di alto livello viene "compilato" in job Hadoop

### Hive: Motivazione
- L'analisi dei dati è richiesta sia da ingegneri che da utenti non tecnici.
- Il volume dei dati cresce rapidamente, superando le capacità dei tradizionali DBMS relazionali, che presentano limiti nella dimensione delle tabelle e nei file a causa delle restrizioni imposte dai sistemi operativi.
- Le soluzioni tradizionali tendono ad essere non scalabili, costose e spesso proprietarie.
- Hadoop supporta applicazioni distribuite per l'elaborazione di grandi volumi di dati, ma richiede l'uso del modello MapReduce, che presenta diversi svantaggi:
	  - È complesso da programmare.
	  - Ha un basso riutilizzo del codice.
	  - È soggetto a errori.
	  - Spesso richiede l'esecuzione di più fasi di job MapReduce.
	  - La maggior parte degli utenti ha familiarità con SQL, non con MapReduce.
#### Soluzione
- Hive rende i dati non strutturati accessibili come tabelle, indipendentemente dal loro formato fisico.
- Consente l'esecuzione di query SQL su queste tabelle.
- Genera automaticamente un piano di esecuzione specifico per ogni query.
- **Hive** è un sistema per la gestione di big data che:
  - Memorizza dati strutturati su HDFS.
  - Fornisce un'interfaccia di query semplice, utilizzando il modello MapReduce di Hadoop.
  - Supporta anche l'esecuzione su Spark tramite Hive on Spark.

### Cos'è Hive?
- Hive è un data warehouse basato su Hadoop, progettato per fornire funzionalità di riepilogo, query e analisi dei dati.
**Struttura**
- **Accesso a diversi storage**: consente l'integrazione con vari sistemi di archiviazione dati.
- **HiveQL**: un linguaggio di query simile a SQL, facile da usare per chi conosce già SQL.
- **Esecuzione delle query**: le query HiveQL vengono trasformate in job MapReduce per l'elaborazione distribuita.
**Principi chiave**:
- **Familiarità con SQL**: rende l'accesso ai big data più semplice per chi ha esperienza con SQL.
- **Estensibilità**: supporta l'estensione con tipi di dati personalizzati, funzioni, formati di file e script.
- **Prestazioni**: progettato per garantire efficienza nell'elaborazione di grandi volumi di dati.

### Scenari di Applicazione
- **Nessuna query in tempo reale**: a causa dell'elevata latenza, Hive non è adatto per risposte istantanee.
- **Nessun supporto per aggiornamenti a livello di riga**: non consente modifiche su singole righe, limitando l'uso in applicazioni transazionali.
- **Non adatto per OLTP (Online Transaction Processing)**: manca di operazioni di inserimento e aggiornamento a livello di riga.
- **Utilizzo ottimale**: Hive eccelle nell'elaborazione batch di grandi set di dati immutabili, come:
  - Elaborazione di log
  - Data mining e text mining
  - Business intelligence

### Layout Fisico
- **Directory del warehouse in HDFS**: i dati sono memorizzati in una directory principale, ad esempio `/user/hive/warehouse`.
- **Tabelle e partizioni**: le tabelle sono organizzate in sottodirectory del warehouse, con le partizioni che creano ulteriori sottodirectory per la gestione dei dati.
- **Dati effettivi**: archiviati in file piatti, come file di testo delimitati o SequenceFiles. Con un SerDe (serializer/deserializer) personalizzato, Hive può gestire anche formati di dati arbitrari.

### Hive: Panoramica della sintassi
- Hive assomiglia a un database SQL
- Join relazionale su due tabelle:
  - Tabella dei conteggi delle parole dalla collezione di Shakespeare
  - Tabella dei conteggi delle parole dalla Bibbia

```sql
SELECT s.word, s.freq, k.freq 
FROM Shakespeare s
JOIN bible k ON (s.word = k.word) 
WHERE s.freq >= 1 AND k.freq >= 1
ORDER BY s.freq DESC 
LIMIT 10;
```

Risultato:
```
the     62394   25848
and     19671   38985
to      18038   13526
of      16700   34654
a       14170   8057
you     12702   2720
my      11297   4135
in      10797   12445
is      8882    6884
```
---
## Hive: Dietro le Quinte (Approfondimento non richiesto)
Quando si esegue una query SQL in Hive, il sistema la trasforma in una serie di operazioni MapReduce, adattando il linguaggio SQL alle capacità di elaborazione distribuita di Hadoop.
## Fasi della trasformazione
#### 1. Abstract Syntax Tree (AST)
Il primo passo è trasformare la query SQL in un albero di sintassi astratta (AST). Questo è una rappresentazione strutturata che cattura la logica della query in termini di componenti linguistici, come le operazioni e le condizioni.
#### 2. Job MapReduce
L'AST viene successivamente trasformato in uno o più job MapReduce. Questi job gestiscono l'elaborazione dei dati distribuiti, sfruttando il cluster Hadoop.

### Piano di Esecuzione
Il piano di esecuzione risultante dalla trasformazione della query è suddiviso in più fasi (o stadi). Ogni fase rappresenta una parte del processo di elaborazione e viene eseguita attraverso MapReduce. Le fasi chiave sono:

#### Stage 1: MapReduce
Nella prima fase, la query è gestita da due strutture operative:
- **Map Operator Tree**: Questo rappresenta la parte Map del job MapReduce.
  - **TableScan**: Esegue una scansione della tabella, identificando i dati necessari per la query.
  - **Filter Operator**: Applica i filtri alle righe selezionate per ridurre il set di dati.
  - **Reduce Output Operator**: Prepara i dati filtrati per essere inviati al passo successivo, che è la parte Reduce.

- **Reduce Operator Tree**: Rappresenta la fase Reduce, che esegue ulteriori elaborazioni sui dati.
  - **Join Operator**: Unisce i dati provenienti da più fonti, secondo le condizioni di join specificate nella query.
  - **Filter Operator**: Applica ulteriori filtri ai dati dopo il join.
  - **Select Operator**: Seleziona le colonne specificate nella query.
  - **File Output Operator**: Scrive i risultati in un file.

#### Stage 2: MapReduce
Questa fase continua l'elaborazione dei dati, eseguendo operazioni aggiuntive sui risultati intermedi.
- **Map Operator Tree**: La parte Map prepara nuovamente i dati per il passo Reduce.
  - **Reduce Output Operator**: Prepara l'output da inviare alla fase Reduce.

- **Reduce Operator Tree**: Qui si eseguono le ultime operazioni.
  - **Extract**: Estrae i risultati finali.
  - **Limit**: Limita il numero di risultati se la query ha un'operazione `LIMIT`.
  - **File Output Operator**: Scrive i risultati finali su HDFS o su un altro storage.

#### Stage 0: Fetch
Infine, la fase di fetch recupera i risultati finali dall'output per restituirli all'utente.

### Dettagli del Piano di Esecuzione
Ogni fase del piano di esecuzione include informazioni specifiche come:
- **Espressioni chiave**: Come i dati vengono mappati e ridotti.
- **Ordine di ordinamento**: Specifica come i dati devono essere ordinati durante l'elaborazione.
- **Colonne di partizione MapReduce**: Definisce le colonne che determinano come i dati sono distribuiti tra le diverse istanze Map e Reduce.
- **Espressioni di valore**: Specifica come i dati devono essere trasformati.
- **Formati di input/output**: Definisce i formati dei file di input e output.
- **Predicati**: Le condizioni per filtrare i dati.
- **Alias delle tabelle**: Gli alias usati per riferirsi alle tabelle nel piano di esecuzione.

---
# Apache Pig: Motivazione

## Big Data
- I dati dei Big Data presentano le caratteristiche delle "3 V": **varietà** (provengono da molteplici fonti e formati) e **volume** (insiemi di dati molto grandi).
- Non è necessario modificare i dati originali, l'analisi richiede solo operazioni di lettura.
- I dati analizzati possono essere temporanei e spesso vengono eliminati dopo l'analisi.

## Obiettivi dell'Analisi dei Dati
- **Velocità**: Sfruttare la potenza dell'elaborazione parallela su sistemi distribuiti.
- **Facilità**: Consentire la scrittura di programmi o query senza una curva di apprendimento complessa, offrendo compiti di analisi predefiniti.
- **Flessibilità**: Permettere la trasformazione dei dati in strutture utili senza un overhead eccessivo, supportando elaborazioni personalizzate.
- **Trasparenza**: L'analisi dovrebbe essere chiara e comprensibile.

## Apache Pig: Soluzione
- **Apache Pig** fornisce un sistema di elaborazione dati di alto livello basato su MapReduce, semplificando la scrittura di script per l'analisi dei dati.
  - Inizialmente sviluppato da Yahoo!
- Gli script scritti in **Pig Latin**, il linguaggio di alto livello di Pig, vengono automaticamente tradotti in job MapReduce dal compilatore di Pig.
- Utilizza il framework MapReduce per eseguire tutte le operazioni di elaborazione dati.
  - Gli script Pig Latin vengono convertiti in uno o più job MapReduce, che poi vengono eseguiti su Hadoop.
- **Apache Pig** è disponibile anche su **Spark** come motore di esecuzione, consentendo la conversione dei comandi Pig Latin in trasformazioni e azioni Spark per un'elaborazione più rapida e flessibile.
---
# Pig Latin

- **Linguaggio di trasformazione dati** orientato agli insiemi e procedurale
  - Offre primitive per operazioni come filtrare, combinare, suddividere e ordinare i dati.
  - Si concentra sul flusso di dati, senza strutture di controllo come cicli `for` o condizioni `if`.
  - Gli utenti descrivono le trasformazioni come una serie di passi.
  - Ogni trasformazione su insiemi di dati è senza stato (*stateless*).
- **Modello di dati flessibile**
  - Supporta bag annidate di tuple.
  - Gestisce tipi di dati semi-strutturati.
- **Esecuzione in Hadoop**
  - Un compilatore converte gli script Pig Latin in flussi di lavoro MapReduce.

## Compilazione ed esecuzione degli script Pig
1. **Analisi sintattica**: Il programma in Pig Latin viene analizzato per verificare la correttezza della sintassi e delle istanze.
   - L'output è un **piano logico** rappresentato come un DAG (Directed Acyclic Graph), che permette ottimizzazioni logiche.
2. **Compilazione**: Il piano logico viene trasformato in una serie di istruzioni MapReduce.
3. **Ottimizzazione**: Un ottimizzatore MR esegue ottimizzazioni come l'aggregazione anticipata, utilizzando la funzione `combiner` di MapReduce.
4. **Esecuzione**: Il programma MR finale viene inviato al gestore dei job Hadoop per l'esecuzione.

## Vantaggi di Pig
- **Facilità di programmazione**
  - Compiti complessi con molte trasformazioni di dati correlate possono essere codificati come sequenze di flusso di dati, semplificando la scrittura, la comprensione e la manutenzione.
  - Riduce il tempo di sviluppo.
- **Ottimizzazione automatica**
  - Il sistema è in grado di ottimizzare l'esecuzione dei compiti, permettendo agli sviluppatori di concentrarsi sulla semantica piuttosto che sull'efficienza.
- **Estensibilità**
  - Supporta UDF (funzioni definite dall'utente) in linguaggi come Java, Python e JavaScript per esigenze di elaborazione specifiche.

## Svantaggi di Pig
- **Lentezza di avvio** dei job MapReduce
  - L'inizializzazione e la pulizia dei job MapReduce richiedono tempo, poiché Hadoop deve pianificare i task.
- **Non adatto per analisi OLAP interattive**
  - Non è pensato per fornire risultati in tempi brevi (<1 secondo).
- **Uso intensivo di UDF** in applicazioni complesse
  - L'uso massiccio di UDF può compromettere la semplicità rispetto a MapReduce.
- **Debugging complicato**
  - Gli errori generati dalle UDF possono essere difficili da interpretare.

## Pig Latin: Modello di Dati
- **Atom**: Valore atomico semplice (es. numero o stringa).
- **Tuple**: Sequenza di campi, ciascuno dei quali può essere di qualsiasi tipo.
- **Bag**: Collezione di tuple, con possibilità di duplicati.
  - Le tuple in una bag possono avere lunghezze e tipi di campo diversi.
- **Map**: Collezione di coppie chiave-valore.
  - La chiave è un atom, mentre il valore può essere di qualsiasi tipo.
---
# Comandi Pig Latin

## `LOAD`
- Il comando `LOAD` serve per caricare i dati, che sono trattati come una **bag** (sequenza di tuple).
- Puoi specificare un **serializer** con l'opzione `USING`.
- È possibile definire uno **schema** con l'opzione `AS` per indicare i nomi e i tipi di campo.

```pig
newBag = LOAD 'filename'
USING functionName()
AS (fieldName1, fieldName2, ...);
```

## `FOREACH ... GENERATE`
- Il comando `FOREACH ... GENERATE` applica trasformazioni sui campi di una bag.
- Ogni campo trasformato può essere:
  - Il nome di un campo della bag.
  - Una costante.
  - Un'espressione semplice (es. `f1 + f2`).
  - Una funzione predefinita (es. `SUM`, `AVG`, `COUNT`, `FLATTEN`).
  - Una funzione definita dall'utente (UDF), ad esempio: `tax(gross, percentage)`.

```pig
newBag = FOREACH bagName GENERATE field1, field2, ...;
```
- `GENERATE` definisce i campi trasformati e genera una nuova riga basata su quelli originali.

## `FILTER ... BY`
- Il comando `FILTER ... BY` seleziona un sottoinsieme delle tuple di una bag che soddisfano una determinata condizione.

```pig
newBag = FILTER bagName BY expression;
```
- L'espressione può includere operatori di confronto (`==`, `!=`, `<`, `>`, ...) e connettori logici (`AND`, `NOT`, `OR`).

Esempi:
```pig
some_apples = FILTER apples BY colour != 'red';
some_apples = FILTER apples BY NOT isRed(colour);
```

## `GROUP ... BY`
- Il comando `GROUP ... BY` raggruppa le tuple in base a una chiave comune (il campo di raggruppamento).

```pig
newBag = GROUP bagName BY expression;
```

- Di solito, l'espressione è un campo, ma può includere anche operatori o funzioni UDF.

Esempi:
```pig
stat1 = GROUP students BY age;
stat2 = GROUP employees BY salary + bonus;
stat3 = GROUP employees BY netsal(salary, taxes);
```

## `JOIN`
- Il comando `JOIN` unisce due dataset su un campo comune.

```pig
joined_data = JOIN results BY querystring, revenue BY querystring;
```

### Esempio di dati in `results`:
```text
(queryString, url, rank)
(lakers, nba.com, 1)
(lakers, espn.com, 2)
(kings, nhl.com, 1)
(kings, nba.com, 2)
```

### Esempio di dati in `revenue`:
```text
(queryString, adSlot, amount)
(lakers, top, 50)
(lakers, side, 20)
(kings, top, 30)
(kings, side, 10)
```

### Risultato del `JOIN`:
```text
(lakers, nba.com, 1, top, 50)
(lakers, nba.com, 1, side, 20)
(lakers, espn.com, 2, top, 50)
(lakers, espn.com, 2, side, 20)
```
---
# Esempio di Analisi Dati: Trovare utenti che tendono a visitare pagine "buone"

## Dati di Esempio
### Visite
```
user | url           | time
-----|---------------|-----
Amy  | www.cnn.com   | 8:00
Amy  | www.crap.com  | 8:05
Amy  | www.myblog.com| 10:00
Amy  | www.flickr.com| 10:05
Fred | cnn.com/index | 12:00
```
### Pagine
```
url            | pagerank
---------------|----------
www.cnn.com    | 0.9
www.flickr.com | 0.9
www.myblog.com | 0.7
www.crap.com   | 0.2
```

Questa analisi mira a identificare gli utenti che tendono a visitare pagine web di alta qualità, basandosi sul PageRank delle pagine visitate. Il processo si divide in due fasi principali: il flusso dati concettuale e il flusso dati a livello di sistema.

## Flusso Dati Concettuale

1. **Caricamento dei Dati**:
   - Caricamento delle visite (user, url, time)
   - Caricamento delle pagine (url, pagerank)

2. **Canonicalizzazione degli URL**: 
   Standardizzazione degli URL per garantire coerenza (es. rimozione di "www." o normalizzazione dei protocolli).

3. **Join**:
   Unione dei dati delle visite con i dati delle pagine basandosi sull'URL.

4. **Raggruppamento per Utente**:
   Aggregazione dei dati per ciascun utente.

5. **Calcolo del PageRank Medio**:
   Per ogni utente, si calcola la media dei PageRank delle pagine visitate.

6. **Filtraggio**:
   Selezione degli utenti con PageRank medio superiore a 0.5.

## Flusso Dati a Livello di Sistema

1. **Caricamento**:
   - Caricamento parallelo dei dati delle visite e delle pagine in più nodi.

2. **Canonicalizzazione**:
   Applicazione della funzione di canonicalizzazione agli URL delle visite.

3. **Join Distribuito**:
   Unione dei dati delle visite e delle pagine attraverso i nodi del cluster.

4. **Raggruppamento e Calcolo**:
   - Raggruppamento dei dati per utente.
   - Calcolo del PageRank medio per ogni utente.

5. **Filtraggio Finale**:
   Applicazione del filtro per selezionare gli utenti con PageRank medio > 0.5.

6. **Risultato**:
   Produzione del set finale di "buoni utenti".

## Esempio Pratico

Considerando i dati di esempio:

1. **Dopo il Join**:
   ```
   Amy, www.cnn.com, 8:00, 0.9
   Amy, www.crap.com, 8:05, 0.2
   Amy, www.myblog.com, 10:00, 0.7
   Amy, www.flickr.com, 10:05, 0.9
   Fred, cnn.com/index, 12:00, 0.9 (assumendo che sia lo stesso di www.cnn.com)
   ```

2. **Dopo il Raggruppamento e Calcolo della Media**:
   ```
   Amy, 0.675 (media di 0.9, 0.2, 0.7, 0.9)
   Fred, 0.9
   ```

3. **Dopo il Filtraggio (avgPR > 0.5)**:
   ```
   Amy, 0.675
   Fred, 0.9
   ```

Entrambi gli utenti sono considerati "buoni utenti" secondo questo criterio.

Questo approccio permette di processare grandi volumi di dati in modo distribuito, sfruttando la potenza di calcolo di un cluster per analizzare efficacemente i pattern di navigazione degli utenti e la qualità delle pagine visitate.

## Script Pig Latin

```pig
Visits = load '/data/visits' as (user, url, time);
Visits = foreach Visits generate user, Canonicalize(url), time;
Pages = load '/data/pages' as (url, pagerank);
VP = join Visits by url, Pages by url;
UserVisits = group VP by user;
UserPageranks = foreach UserVisits generate user, AVG(VP.pagerank) as avgpr;
GoodUsers = filter UserPageranks by avgpr > '0.5';
store GoodUsers into '/data/good_users';
```

## Java vs. Pig Latin
- Pig Latin richiede circa 1/20 delle righe di codice rispetto a Java
- Il tempo di sviluppo con Pig è circa 1/16 rispetto a Hadoop puro
- Le prestazioni sono paragonabili a quelle di Hadoop puro
---
## Come Pig viene utilizzato in pratica
- Utile per calcoli su grandi dataset distribuiti
- Astrae i dettagli del framework di esecuzione
- Gli utenti possono modificare l'ordine dei passaggi per migliorare le prestazioni
- Usato in tandem con Hadoop e HDFS:
  - Le trasformazioni sono convertite in flussi di dati MapReduce
  - HDFS tiene traccia di dove sono memorizzati i dati
- Le operazioni sono pianificate vicino ai loro dati
---

# FAQ su Hive e Pig

## Hive

### 1. Qual è lo scopo di Hive?

Hive è un sistema di data warehousing basato su Hadoop che semplifica l'elaborazione di grandi set di dati. Fornisce un'interfaccia simile a SQL chiamata HiveQL per eseguire query su dati strutturati memorizzati in HDFS (Hadoop Distributed File System).

### 2. Quali sono i vantaggi di usare Hive?

- **Familiarità con SQL**: Gli utenti che hanno familiarità con SQL possono facilmente imparare e utilizzare HiveQL.
- **Scalabilità**: Hive è progettato per gestire enormi quantità di dati sfruttando la potenza di elaborazione distribuita di Hadoop.
- **Estensibilità**: Hive supporta tipi di dati, funzioni e formati di file personalizzati.
- **Astrazione**: Hive nasconde la complessità di MapReduce, rendendo più semplice l'analisi dei dati.

### 3. In quali scenari Hive non è adatto?

- **Query in tempo reale**: Hive non è adatto per query a bassa latenza o applicazioni in tempo reale a causa del suo tempo di elaborazione.
- **Aggiornamenti a livello di riga**: Hive non supporta gli aggiornamenti a livello di riga, il che lo rende inadatto per carichi di lavoro transazionali.

## Pig

### 4. Qual è lo scopo di Pig?

Pig è un linguaggio di alto livello e una piattaforma per l'analisi di grandi set di dati. Utilizza un linguaggio procedurale chiamato Pig Latin, che viene poi compilato in job MapReduce eseguibili su Hadoop.

### 5. Quali sono i vantaggi di utilizzare Pig?

- **Facilità di programmazione**: Pig Latin è più facile da imparare e utilizzare rispetto a Java per scrivere programmi MapReduce.
- **Ottimizzazione automatica**: Il compilatore Pig ottimizza i programmi Pig Latin per migliorare le prestazioni.
- **Flessibilità**: Pig supporta diversi formati di dati e consente agli utenti di definire le proprie funzioni.

### 6. Quali sono gli svantaggi di Pig?

- **Avvio lento**: I job Pig possono richiedere tempo per l'inizializzazione e la pulizia, il che li rende meno adatti per analisi interattive.
- **Debugging**: Il debug di programmi Pig può essere complicato, specialmente quando si utilizzano funzioni definite dall'utente (UDF).

## Hive e Pig

### 7. Quali sono le differenze principali tra Hive e Pig?

- **Linguaggio**: Hive utilizza HiveQL, un linguaggio simile a SQL, mentre Pig utilizza Pig Latin, un linguaggio procedurale.
- **Modello di esecuzione**: Hive traduce le query in job MapReduce, mentre Pig può utilizzare diversi motori di esecuzione, tra cui MapReduce e Spark.
- **Casi d'uso**: Hive è più adatto per attività di data warehousing e analisi ad hoc, mentre Pig è più versatile e può essere utilizzato per una gamma più ampia di attività di elaborazione dati.

### 8. In che modo sia Hive che Pig interagiscono con Hadoop?

Sia Hive che Pig si basano su Hadoop per l'archiviazione e l'elaborazione dei dati. Traducono i propri linguaggi di alto livello in job MapReduce eseguibili sul cluster Hadoop, sfruttando la potenza di elaborazione distribuita di Hadoop per gestire grandi set di dati.

---
## Quiz
1. Quali sono le motivazioni principali per l'utilizzo di Hive in un ambiente Hadoop?
2. In che modo Hive affronta le sfide poste dall'analisi di grandi volumi di dati per utenti con diverse competenze tecniche?
3. Descrivi la struttura di base di Hive e come interagisce con HDFS.
4. Quali sono le principali differenze tra HiveQL e SQL standard?
5. In quali scenari applicativi Hive eccelle e in quali mostra i suoi limiti?
6. Quali sono le motivazioni che hanno portato allo sviluppo di Pig?
7. Descrivi i vantaggi di utilizzare Pig Latin rispetto alla programmazione diretta in MapReduce.
8. Quali sono le fasi principali della compilazione ed esecuzione di uno script Pig?
9. Spiega il concetto di "modello di dati flessibile" in Pig Latin.
10. Fai un esempio pratico di come Pig può essere utilizzato per analizzare dati reali e ottenere informazioni significative.

## Risposte al Quiz
1. Hive è stato sviluppato per facilitare l'analisi di grandi dati in Hadoop da parte di utenti con diverse competenze tecniche, in particolare coloro che hanno familiarità con SQL. Offre un'interfaccia simile a SQL (HiveQL) per interagire con i dati in HDFS, semplificando query e analisi.
2. Hive affronta le sfide dei big data fornendo un'interfaccia SQL astratta (HiveQL) che traduce le query in job MapReduce eseguibili su Hadoop. Questo consente agli utenti di concentrarsi sulla logica dell'analisi anziché sulla complessità della programmazione MapReduce.
3. Hive utilizza un metastore per memorizzare informazioni sulle tabelle, inclusi schemi, posizioni e altre proprietà. I dati effettivi risiedono in HDFS come file. Le query HiveQL vengono elaborate da un driver che le traduce in job MapReduce eseguiti su Hadoop, accedendo ai dati in HDFS.
4. HiveQL è un linguaggio simile a SQL, ma con alcune differenze. Non supporta completamente tutte le funzionalità SQL, come gli aggiornamenti a livello di riga e le transazioni. Inoltre, l'esecuzione delle query in HiveQL può essere più lenta rispetto a SQL standard a causa della sua natura batch-oriented.
5. Hive eccelle nell'elaborazione batch di grandi set di dati immutabili, come l'elaborazione di log, il data mining e la business intelligence. Non è adatto per query in tempo reale, aggiornamenti a livello di riga o elaborazione di transazioni online (OLTP).
6. Pig è stato sviluppato per semplificare la scrittura di script per l'analisi di dati su larga scala in Hadoop. Il suo linguaggio, Pig Latin, offre un livello di astrazione più elevato rispetto a MapReduce, rendendo più facile esprimere trasformazioni di dati complesse.
7. Pig Latin offre una sintassi più concisa e leggibile rispetto a Java per MapReduce. Pig si occupa automaticamente dell'ottimizzazione e della parallelizza dei job, riducendo il tempo di sviluppo e semplificando la manutenzione del codice.
8. Le fasi principali sono: analisi sintattica dello script Pig Latin, compilazione in un piano logico (DAG), ottimizzazione del piano logico, conversione in job MapReduce ed esecuzione su Hadoop.
9. Il modello di dati di Pig Latin è flessibile perché supporta tipi di dati annidati e semi-strutturati come bag (collezioni di tuple) e map (coppie chiave-valore). Questo lo rende adatto a gestire dati provenienti da diverse fonti, inclusi dati non strutturati.
10. Un esempio pratico di utilizzo di Pig è l'analisi dei log web per identificare le pagine più popolari. Pig può caricare i dati dei log, raggruppare le visite per URL, contare le visite per ogni gruppo e ordinare i risultati per ottenere le pagine più visitate.