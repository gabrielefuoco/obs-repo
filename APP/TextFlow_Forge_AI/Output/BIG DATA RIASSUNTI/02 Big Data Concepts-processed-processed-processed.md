
## Big Data: Definizioni e Caratteristiche

Le definizioni di Big Data convergono su tre o quattro caratteristiche principali, spesso indicate come "V":

* **Volume:** La grande quantità di dati.
* **Velocità:** La rapidità di generazione e elaborazione dei dati.
* **Varietà:** La diversità dei tipi di dati (strutturati, semi-strutturati, non strutturati).
* **Valore/Variabilità:** Alcune definizioni aggiungono il valore estratto dai dati o la loro variabilità nel tempo.  Altre caratteristiche includono la viralità, la visualizzazione, la viscosità e la provenienza dei dati (luogo).


## Data Science: Disciplina e Processi

La Data Science è una disciplina interdisciplinare che combina informatica, matematica applicata e tecniche di analisi dei dati per estrarre informazioni da grandi quantità di dati.  Il processo di data science comprende:

1. Formulazione del problema.
2. Raccolta dei dati.
3. Elaborazione dei dati.
4. Esplorazione dei dati.
5. Analisi approfondita.
6. Comunicazione dei risultati.

Le competenze chiave includono la conoscenza del dominio applicativo, la comunicazione efficace, la gestione della qualità dei dati, la rappresentazione e la visualizzazione dei dati, e la consapevolezza delle implicazioni etiche.


## Archiviazione Big Data: Scaling

L'archiviazione di Big Data richiede approcci di scaling per gestire l'enorme volume di dati:

* **Scaling Verticale:** Aumentare le risorse di un singolo server.
* **Scaling Orizzontale:** Aggiungere più nodi al sistema per distribuire il carico.

I database NoSQL sono particolarmente adatti allo scaling orizzontale, offrendo una soluzione scalabile e distribuita per la gestione di dati di varia natura (scalari, binari, oggetti complessi), garantendo tolleranza ai guasti e distribuzione automatica dei dati.

---

# Tipi di Database NoSQL

Questo documento descrive diversi tipi di database NoSQL, classificati in base al loro modello di dati.

## Modelli di Dati NoSQL

* **Archiviazione chiave-valore:**  Archivia dati come coppie `⟨chiave, valore⟩` distribuite su più server, spesso utilizzando una tabella hash distribuita (DHT) per l'indicizzazione.  Esempi: DynamoDB, Redis.

* **Negozi di documenti:** Supportano indici secondari e diversi tipi di documenti per database, permettendo query basate su più attributi.  I documenti sono spesso in formato JSON-simile (es. BSON in MongoDB). Esempio: MongoDB.

* **Negozi basati su colonne:** Archiviano record estensibili, partizionabili orizzontalmente (record su nodi diversi) e verticalmente (parti di un record su server diversi). Le colonne possono essere distribuite su più server usando gruppi di colonne. Esempi: Cassandra, HBase, BigTable.

* **Negozi basati su grafi:** Archiviano e interrogano informazioni rappresentate come grafi (nodi, archi e proprietà), ottimizzati per query grafiche complesse. Esempio: Neo4j.


## Esempi di Database NoSQL

### MongoDB

* **Tipo:** Negozio basato su documenti.
* **Descrizione:** Utilizza il formato BSON (simile a JSON) per i documenti, supportando query complesse, indici, sharding automatico, replica e gestione dell'archiviazione.

### Google Bigtable

* **Tipo:** Negozio basato su colonne.
* **Descrizione:** Basato su Google File System (GFS), archivia grandi quantità di dati in tabelle multidimensionali sparse.  I dati sono organizzati in righe, colonne (raggruppate in famiglie di colonne) e sono partizionati in "tablet" distribuiti su un cluster.

### HBase

* **Tipo:** Negozio basato su colonne.
* **Descrizione:** Simile a BigTable, ma utilizza Hadoop e HDFS.  Supporta tabelle senza schema con una chiave primaria definita per ogni tabella. Si integra bene con Hive per l'elaborazione batch di big data.

### Redis

* **Tipo:** Key-value store.

---

Il testo descrive quattro diversi tipi di database: un data store in-memory open-source, DynamoDB, Cassandra e Neo4j.

**Data store in-memory open-source:**  Questo database in-memory, open-source, è utilizzato come database, cache, message broker e motore di streaming. Supporta diverse operazioni atomiche su dati in memoria (es. aggiunta di stringhe, incremento di valori in hash, gestione di liste e insiemi).  Permette la persistenza dei dati su disco, a seconda delle necessità.

**DynamoDB:** Un servizio key-value store NoSQL completamente gestito da AWS.  È ottimizzato per applicazioni che richiedono bassa latenza, come app mobili e giochi. Offre funzionalità di sicurezza integrate (crittografia, controllo accessi), backup gestiti, ripristino e alta disponibilità tramite replica multi-regione e multi-master.  Si integra con altri servizi AWS.

**Cassandra:** Un column-based store con architettura ad anello senza master.  Ogni nodo è identico, permettendo alta disponibilità e scalabilità tramite l'aggiunta di nodi senza downtime. La replica dati è personalizzabile, garantendo la disponibilità anche in caso di guasto di un nodo.  La distribuzione dei dati è automatica.

**Neo4j:** Un graph-based store dove i nodi contengono relazioni ad altri nodi e attributi.  Le relazioni hanno nome, direzione, nodo iniziale e finale, e possono avere proprietà aggiuntive.  Nodi e relazioni possono avere etichette per la categorizzazione. I cluster Neo4j offrono alta disponibilità e scalabilità di lettura orizzontale tramite replica master-slave.

---

## Confronto di Sistemi NoSQL

Questa tabella confronta diversi sistemi NoSQL, evidenziando tipologia, modalità di archiviazione, supporto per MapReduce, persistenza, replica, scalabilità, prestazioni, alta disponibilità, linguaggio di programmazione e licenza.

| Sistema     | Tipo | Archiviazione Dati | MapReduce | Persistenza | Replica | Scalabilità | Prestazioni | Alta Disponibilità | Linguaggio      | Licenza      |
|-------------|------|--------------------|-----------|-------------|---------|-------------|--------------------|-------------|-----------------|---------------|
| DynamoDB    | KV   | MEM<br>FS          | Sì        | Sì          | Sì       | Alta         | Alta             | Sì            | Java            | Proprietaria   |
| Cassandra   | Col  | HDFS<br>CFS        | Sì        | Sì          | Sì       | Alta         | Alta             | Sì            | Java            | Apache2       |
| HBase       | Col  | HDFS               | Sì        | Sì          | Sì       | Alta         | Alta             | Sì            | Java            | Apache2       |
| Redis       | KV   | MEM<br>FS          | No        | Sì          | Sì       | Alta         | Alta             | Sì            | Ansi-C          | BSD            |
| BigTable    | Col  | GFS                | Sì        | Sì          | Sì       | Alta         | Alta             | Sì            | Java<br>Python<br>Go<br>Ruby | Proprietaria   |
| MongoDB     | Doc  | MEM<br>FS          | Sì        | Sì          | Sì       | Alta         | Alta             | Sì            | C++            | AGPL3         |
| Neo4j      | Graph| MEM<br>FS          | No        | Sì          | Sì       | Alta,<br>variabile | Alta             | Sì            | Java            | GPL3           |


**Teorema CAP:** Un sistema distribuito non può garantire simultaneamente coerenza (tutti i nodi vedono gli stessi dati nello stesso momento), disponibilità (ogni richiesta riceve risposta in tempo ragionevole) e tolleranza alle partizioni (funzionamento nonostante partizioni di rete).  La maggior parte dei sistemi NoSQL privilegia o la coerenza o la disponibilità.


## Concetti Big Data: Tipi di Database NoSQL

I database NoSQL si suddividono in diverse categorie, ognuna con punti di forza e debolezza in termini di scalabilità e compromessi CAP:

**1. Database Chiave-Valore (KV):**  Offrono alta scalabilità orizzontale tramite sharding, ideali per dati semplici e velocità estrema.  Generalmente privilegiano la coerenza.  Pro: semplicità, alta scalabilità; Contro:  limitazioni per dati complessi.

**2. Database Shardati:**  Hanno capacità di scaling orizzontale limitate, adatti per grandi quantità di dati semplici con requisiti di consistenza non stringenti.  Privilegiano solitamente la disponibilità. Pro: semplicità di implementazione; Contro: inefficienza di alcune query, mancanza di standardizzazione API.

**3. Database a Colonne (Col):**  Offrono alta scalabilità orizzontale, ideali per consistenza ed elevata scalabilità senza caching.  Privilegiano generalmente la consistenza. Pro: alto throughput, query multi-attributo; Contro: maggiore complessità, non adatti per dati interconnessi.

**4. Database basati su Grafo (Graph):**  Hanno scarso scaling orizzontale, ideali per dati relazionali (es. social network).  Di solito privilegiano la disponibilità. Pro: gestione di relazioni; Contro: limitata scalabilità orizzontale.

---

# Riassunto del Testo: Analisi Dati, Big Data Analytics e Calcolo Parallelo

Questo testo tratta l'analisi dei dati, la big data analytics e il calcolo parallelo, evidenziando le loro interconnessioni.

## Analisi vs. Analitica dei Dati

Il testo distingue tra **analisi dei dati**, che si concentra sul processo di esplorazione, interrogazione e visualizzazione di dataset, e **analitica dei dati**, un concetto più ampio che include l'analisi dei dati, ma anche gli strumenti e le tecniche utilizzate (data warehouse, visualizzazione dati).

## Big Data Analytics

La **big data analytics** applica tecniche avanzate (data mining, statistica, AI, machine learning, NLP) a grandi dataset.  Le sue applicazioni includono analisi del testo, analisi predittiva, analisi di grafi e analisi prescrittiva.  Questi processi sono computazionalmente intensivi e richiedono approcci paralleli e distribuiti, spesso sfruttando HPC e cloud computing.

## Calcolo Parallelo

Il **calcolo parallelo** affronta problemi di dimensione *n* suddividendoli in *k ≥ 2* parti risolte simultaneamente da *p* processori.  Questo paradigma "divide et impera" è applicabile solo a problemi parallelizzabili.  La concorrenza (task in corso simultaneamente) è una condizione necessaria ma non sufficiente per il parallelismo (task in esecuzione simultaneamente).

## Divide et Impera: Esempio

Il testo illustra il "divide et impera" con il calcolo del prodotto scalare di due vettori:

$$a = [a_1, a_2, \dots, a_n] \qquad \qquad b = [b_1, b_2, \dots, b_n]$$

Il prodotto scalare:

$$a \cdot b = \sum_{i=1}^n a_ib_i$$

può essere parallelizzato suddividendo la somma in *k* somme parziali (divide) calcolate simultaneamente (impera) su processori distinti.  Il risultato finale è ottenuto combinando le somme parziali (che richiede sincronizzazione).

$$\sum_{i=1}^{n} a_i b_i = \sum_{i=1}^{j_1} a_i b_i + \sum_{i=j_1+1}^{j_2} a_i b_i + \dots + \sum_{i=j_{k-1}+1}^{n} a_i b_i,$$

dove  $$1 < j_1 < j_2 < \cdots < j_{k-1} < n$$

---

## Riassunto del testo su Parallelismo, Architetture Parallele e Cloud Computing

Questo testo introduce i concetti di parallelismo, architetture parallele e cloud computing, focalizzandosi sulle loro caratteristiche principali e sulle metriche di performance.

### Parallelismo

Un problema con dominio *D* può essere **data-parallel**, dove una funzione *f* viene applicata a sottoinsiemi di *D* ($f(D) = f(d_1) + f(d_2) + \dots + f(d_k)$), o **task-parallel**, dove un insieme di funzioni *F* viene applicato a *D* ($F(D) = f_1(D) + f_2(D) + \dots + f_k(D)$).

### Architetture Parallele

La tassonomia di Flynn classifica le architetture parallele in base al flusso di istruzioni e dati: SISD (Single Instruction, Single Data), SIMD (Single Instruction, Multiple Data), MISD (Multiple Instruction, Single Data) e MIMD (Multiple Instruction, Multiple Data).  ![[]](_page_36_Figure_3.jpeg)

### Metriche di Performance

Le metriche principali per valutare le prestazioni di un sistema parallelo sono:

* **Tempo di esecuzione sequenziale ($T_s$)**: tempo di esecuzione su un singolo processore.
* **Tempo di esecuzione parallelo ($T_n$)**: tempo di esecuzione su *n* processori.
* **Speed-up ($S_n = T_s / T_n$)**: rapporto tra $T_s$ e $T_n$.
* **Efficienza ($E_n = S_n/n = T_s/(n \cdot T_n$)**: utilizzo effettivo di ciascun processore.

### Legge di Amdahl

La legge di Amdahl definisce lo speed-up teorico ($\hat{S}_n$) in funzione del numero di processori (*n*) e della frazione parallelizzabile (*F*) del programma:

$$ \hat{S}_n = \frac{1}{(1 - F) + \frac{F}{n}} $$

Lo speed-up massimo è limitato da $\frac{1}{(1 - F)}$, indipendentemente dal numero di processori. ![[|450](_page_41_Figure_2.jpeg)]

### Cloud Computing

Il cloud computing è un modello che permette l'accesso on-demand a risorse condivise (reti, server, storage, applicazioni e servizi).  Le sue caratteristiche principali sono: self-service on-demand, ampio accesso alla rete, pool di risorse, elasticità rapida e servizio misurato.  Esistono tre modelli di servizio (SaaS, PaaS, IaaS) e tre modelli di deployment (cloud pubblico, privato e ibrido).

### Servizi Cloud per Big Data e Piattaforme Cloud

I principali provider di servizi cloud per big data sono AWS, Google Cloud Platform e Microsoft Azure, ognuno con le proprie offerte di data management, compute e storage.  Il testo fornisce una breve panoramica delle offerte di ciascuna piattaforma.

---

Il documento tratta di sistemi exascale e del loro utilizzo nell'apprendimento automatico parallelo e distribuito.  Inizia descrivendo OpenStack, una piattaforma cloud open source che offre servizi di compute, storage (Blob, Table, Queue), e networking, oltre a servizi condivisi come gestione utenti e immagini server.

Successivamente, introduce il calcolo exascale, definito come la capacità di raggiungere almeno un exaFLOP (10<sup>18</sup> operazioni in virgola mobile al secondo).  Descrive gli attributi chiave di un sistema exascale: attributi fisici (consumo energetico, dimensioni), tasso di calcolo, capacità di storage e tasso di larghezza di banda.  Le principali sfide dei sistemi exascale includono gestione dell'energia, concorrenza, e località dei dati a diversi livelli (intra-rack, inter-rack, nodi).

Il documento si concentra poi sull'apprendimento automatico parallelo e distribuito, necessario per gestire grandi dataset. Vengono presentate tre strategie di apprendimento parallelo: parallelismo indipendente, SPMD (Single Program Multiple Data) e parallelismo delle attività.

Infine, vengono illustrate le strategie di apprendimento distribuito, dove modelli locali vengono aggregati per creare un modello globale.  Questo approccio è utilizzato in tecniche come il meta-apprendimento (dove *N* classificatori di base su *N* nodi creano un classificatore globale tramite un insieme di addestramento di meta-livello), l'apprendimento di ensemble, l'apprendimento federato e il data mining collettivo.  Il meta-apprendimento, in particolare, viene spiegato con un esempio di classificazione a due fasi: creazione di modelli locali e successiva creazione di un modello globale basato sulle previsioni dei modelli locali.

---

### Apprendimento di Ensemble

L'apprendimento di ensemble migliora la precisione predittiva combinando le previsioni di più modelli.  Due tecniche principali sono il *bagging*, che aggrega modelli (uguali o diversi) addestrati su sottoinsiemi di dati, e il *boosting*, che pondera i modelli in base alla loro accuratezza, dando maggiore peso a quelli più performanti. Il risultato è un modello più accurato rispetto ai singoli modelli componenti.

### Apprendimento Federato

L'apprendimento federato permette l'addestramento di modelli su dati distribuiti senza centralizzarli. Un server invia un modello iniziale ai nodi, che lo addestrano localmente sui propri dati, aggiornando solo i parametri del modello e non i dati stessi.  Questo approccio preserva la privacy e la sicurezza dei dati, consentendo una collaborazione distribuita per l'apprendimento di un modello condiviso.

### Data Mining Collettivo

Il data mining collettivo crea un modello globale combinando modelli *parziali* calcolati su diversi siti, a differenza di tecniche che combinano modelli *completi*.  Questo approccio si basa sulla rappresentazione distribuita di funzioni tramite *funzioni di base*, potenzialmente non lineari.  Se le funzioni di base sono ortonormali, l'analisi locale produce componenti direttamente utilizzabili nel modello globale.  La presenza di termini non lineari richiede la considerazione di interazioni tra caratteristiche di nodi diversi, rendendo il modello globalmente meno scomponibile.

---
