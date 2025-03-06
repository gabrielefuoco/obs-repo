
## Big data: Alcune definizioni

Diverse definizioni caratterizzano i Big Data, focalizzandosi su differenti aspetti:

**Definizione 1 (Gartner, Inc.):** "I big data sono asset informativi ad **alto volume**, **alta velocità** e/o **alta varietà** che richiedono forme economicamente convenienti e innovative di elaborazione delle informazioni per consentire una maggiore comprensione, un processo decisionale migliorato e l'automazione dei processi". Questa definizione si basa sul modello "3V":

* **Volume:** Quantità di dati.
* **Velocità:** Velocità di generazione e elaborazione dei dati.
* **Varietà:** Diversità dei tipi di dati.

**Definizione 2 (Gantz e Reinsel, 2011):** "Le tecnologie Big Data descrivono una nuova generazione di tecnologie e architetture, progettate per estrarre economicamente **valore** da volumi molto grandi di un'ampia **varietà** di dati, consentendo l'acquisizione, la scoperta e/o l'analisi ad **alta velocità**". Questa definizione introduce un quarto "V":

* *Volume*
* *Velocità*
* *Varietà*
* **Valore:** Utilità delle informazioni estratte dai dati.

**Definizione 3 (Chang e Grady, 2015):** "I big data consistono in ampi set di dati caratterizzati principalmente da **volume**, **varietà**, **velocità** e/o **variabilità**, che richiedono un'architettura scalabile per un'archiviazione, una manipolazione e un'analisi efficienti". Anche questa definizione utilizza quattro "V", ma sostituisce il "Valore" con la "Variabilità":

* *Volume*
* *Velocità*
* *Varietà*
* **Variabilità:** Cambiamenti nel tempo dei dati.

**Altre V:** Oltre ai tre o quattro "V" principali, altre caratteristiche possono essere considerate:

* **Viralità:** Capacità dei dati di diffondersi rapidamente.
* **Visualizzazione:** Possibilità di rappresentare graficamente i dati.
* **Viscosità:** Capacità delle informazioni estratte di mantenere l'interesse.
* **Luogo (Venue):** Origine dei dati, spesso da sorgenti distribuite ed eterogenee.

## Data science

La data science è una disciplina interdisciplinare che combina **informatica**, **matematica applicata** e tecniche di **analisi dei dati** per estrarre informazioni da grandi quantità di dati. Migliora le scoperte basando le decisioni su informazioni estratte da grandi set di dati attraverso l'utilizzo di algoritmi per:

* Raccolta
* Pulizia
* Trasformazione
* Analisi (di Big data)

### Processi di data science

I passaggi principali di un processo di data science sono:

- Formulazione del problema
- Raccolta dei dati necessari
- Elaborazione dei dati per l'analisi
- Esplorazione dei dati
- Esecuzione di analisi approfondite
- Comunicazione dei risultati dell'analisi

### Competenze in data science

Le competenze principali necessarie nella data science includono:

* Apprendimento del dominio di applicazione
* Comunicazione con i proprietari/utenti dei dati
* Attenzione alla qualità dei dati
* Conoscenza della rappresentazione dei dati
* Gestione della trasformazione e dell'analisi dei dati
* Conoscenza della visualizzazione e della presentazione dei dati
* Considerazione delle questioni etiche

## Archiviazione Big Data: approcci di scaling

Esistono due principali approcci per scalare l'archiviazione Big Data:

* **Scaling verticale:** Aumento delle risorse (CPU, RAM, disco, I/O di rete) di un singolo server.
* **Scaling orizzontale:** Aggiunta di più nodi di archiviazione/elaborazione al sistema per distribuire il carico di lavoro.

### Database NoSQL

La maggior parte dei database relazionali ha poca capacità di scaling orizzontale. I database NoSQL offrono un'alternativa per garantire la scalabilità orizzontale delle operazioni di lettura/scrittura, distribuite su molti server. I database NoSQL sfruttano nuovi nodi in modo trasparente, senza richiedere la distribuzione manuale delle informazioni o gestione aggiuntiva. Sono progettati per garantire la distribuzione automatica dei dati e la tolleranza ai guasti.

I database NoSQL forniscono modi per archiviare valori scalari (es. numeri, stringhe), oggetti binari (es. immagini, video) o strutture più complesse (es. grafi). In base al loro modello di dati, possono essere classificati come:

* Archiviazione chiave-valore
* Archiviazione basata su documenti
* Archiviazione basata su colonne
* Archiviazione basata su grafi

### Archiviazione chiave-valore

Gli archivi chiave-valore archiviano i dati come coppie ⟨chiave, valore⟩ su più server. Una tabella hash distribuita (DHT) può essere utilizzata per implementare una struttura di indicizzazione scalabile, in cui il recupero dei dati avviene usando una chiave per trovare un valore. Esempi: DynamoDB, Redis.

## Tipi di Database NoSQL

### Archivi di documenti

A differenza degli archivi chiave-valore, i negozi di documenti supportano indici secondari e più tipi di documenti per database, fornendo meccanismi per interrogare le collezioni in base a più vincoli di valore degli attributi. Esempio: MongoDB.

### Archivi basati su colonne

I data store basati su colonne (negozi di record estensibili) archiviano record estensibili partizionabili su più server. I record sono estensibili perché è possibile aggiungere nuovi attributi. I negozi di record estensibili forniscono partizione orizzontale (record su nodi diversi) e verticale (parti di un singolo record su server diversi). Le colonne possono essere distribuite su più server usando gruppi di colonne. Esempi: Cassandra, HBase, BigTable.

### Archivi basati su grafi

I data store basati su grafi archiviano e interrogano informazioni rappresentate come grafi (nodi, archi e proprietà), difficili da gestire con database relazionali. Consentono query efficienti sui grafi, accelerando algoritmi grafici (comunità, gradi, centralità, distanze, percorsi). Esempio: Neo4j.

## MongoDB

* **Tipo:** Archivio basato su documenti.
* **Descrizione:** Progettato per supportare applicazioni internet e basate sul web. Rappresenta i documenti in un formato simile a JSON, chiamato BSON, che funge da formato di trasferimento di rete. MongoDB comprende la struttura interna degli oggetti BSON e può accedere a oggetti nidificati usando la notazione a punto. Questa capacità permette la costruzione di indici e il confronto di oggetti con espressioni di query, includendo chiavi BSON di livello superiore e nidificate. Fornisce supporto completo per query complesse e indici completi, includendo funzionalità come lo sharding automatico, la replica e la gestione semplificata dell'archiviazione.

## Google Bigtable

* **Tipo:** Archivio basato su colonne.
* **Descrizione:** Costruito su Google File System (GFS), può archiviare fino a petabyte di dati. I dati sono archiviati in tabelle multidimensionali sparse, distribuite e persistenti, composte da righe e colonne. Ogni riga è indicizzata da una singola chiave di riga, e le colonne correlate sono raggruppate in insiemi chiamati famiglie di colonne. Una colonna generica è identificata da una famiglia di colonne e un qualificatore di colonna (nome univoco all'interno della famiglia). Per migliorare la scalabilità, i dati sono ordinati per chiave di riga e l'intervallo di righe per una tabella è partizionato dinamicamente in blocchi contigui, chiamati tablet. Questi tablet sono distribuiti tra diversi nodi di un cluster Bigtable (server tablet).

## HBase

* **Tipo:** Archivio basato su colonne.
* **Descrizione:** Similmente a Bigtable che usa GFS, HBase sfrutta Hadoop e Hadoop Distributed File System (HDFS) per fornire funzionalità simili. Il design di HBase è progettato per scalare linearmente, costituito da tabelle standard con righe e colonne simili ai database tradizionali. Le tabelle sono senza schema e le famiglie di colonne sono definite durante la creazione di una tabella, non le singole colonne. Ogni tabella deve avere una chiave primaria definita, usata per tutti gli accessi. HBase si integra bene con Hive, che funge da motore di query per l'elaborazione batch di big data, abilitando applicazioni big data fault-tolerant.

## Redis

* **Tipo:** Archivio Key-value.
* **Descrizione:** Un popolare data store in-memory open-source usato come database, cache, message broker e motore di streaming. Fornisce supporto a diversi tipi di operazioni atomiche, come: aggiunta di una stringa, incremento del valore in un hash, inserimento di un elemento in una lista, calcolo dell'intersezione, unione e/o differenza di insiemi, estrazione dell'elemento con il punteggio massimo da un insieme ordinato. Sebbene funzioni con set di dati in memoria per migliorare le prestazioni, può persistere i dati scaricandoli periodicamente su disco, a seconda del caso d'uso.

## DynamoDB

* **Tipo:** Key-value store.
* **Descrizione:** Servizio di database NoSQL completamente gestito fornito da Amazon Web Services (AWS). Ideale per applicazioni che richiedono accesso ai dati a bassa latenza, come app di gioco, applicazioni mobili e piattaforme di e-commerce. Fornisce funzionalità di sicurezza integrate, come la crittografia a riposo e in transito, il controllo degli accessi a grana fine e l'integrazione con AWS Identity and Access Management (IAM). Offre backup completamente gestiti, funzionalità di ripristino e sincronizzazione multi-regione, multi-master, garantendo alta disponibilità e durata dei dati. DynamoDB si integra anche con altri servizi AWS (es. Amazon S3), permettendo agli sviluppatori di creare architetture serverless potenti.

## Cassandra

* **Tipo:** Column-based store.
* **Descrizione:** Utilizza un'architettura ad anello senza master, dove tutti i nodi svolgono un ruolo identico, permettendo a qualsiasi utente autorizzato di connettersi a qualsiasi nodo in qualsiasi data center. Questa architettura semplice e flessibile permette l'aggiunta di nodi senza tempi di inattività. La distribuzione dei dati tra i nodi non richiede operazioni programmatiche da parte degli sviluppatori. Non presenta un singolo punto di errore, garantendo la continua disponibilità dei dati e il tempo di attività del servizio. Fornisce un servizio di replica dati personalizzabile che consente la replica dei dati tra nodi organizzati in un anello. In caso di guasto di un nodo, una o più copie dei dati necessari sono disponibili su altri nodi.

## Neo4j

* **Tipo:** Graph-based store.
* **Descrizione:** Ogni nodo contiene un elenco di record di relazioni che fanno riferimento ad altri nodi e attributi aggiuntivi (es. timestamp, metadati, coppie chiave-valore). Ogni record di relazione deve avere un nome, una direzione, un nodo iniziale e un nodo finale e può contenere proprietà aggiuntive. È possibile assegnare una o più etichette sia ai nodi che alle relazioni. Queste etichette possono rappresentare i ruoli di un nodo nel grafo (es. utente, indirizzo, azienda) o per associare indici e vincoli a gruppi di nodi. I cluster Neo4j sono progettati per un'elevata disponibilità e una scalabilità di lettura orizzontale usando la replica master-slave.

## Confronto dei Sistemi NoSQL

| Sistema | Tipo | Archiviazione Dati | MapReduce | Persistenza | Replica | Scalabilità | Prestazioni | Alta Disponibilità | Linguaggio | Licenza |
|--------------|----------|--------------------|-----------|-------------|---------|-------------|----------------------|--------------------|-------------|---------------|
| DynamoDB | KV | MEM<br>FS | Sì | Sì | Sì | Alta | Alta | Sì | Java | Proprietaria |
| Cassandra | Col | HDFS<br>CFS | Sì | Sì | Sì | Alta | Alta | Sì | Java | Apache2 |
| HBase | Col | HDFS | Sì | Sì | Sì | Alta | Alta | Sì | Java | Apache2 |
| Redis | KV | MEM<br>FS | No | Sì | Sì | Alta | Alta | Sì | Ansi-C | BSD |
| BigTable | Col | GFS | Sì | Sì | Sì | Alta | Alta | Sì | Java<br>Python<br>Go<br>Ruby | Proprietaria |
| MongoDB | Doc | MEM<br>FS | Sì | Sì | Sì | Alta | Alta | Sì | C++ | AGPL3 |
| Neo4j | Graph | MEM<br>FS | No | Sì | Sì | Alta | Alta,<br>variabile | Sì | Java | GPL3 |

**Teorema CAP (Gilbert e Lynch, 2002):** Un sistema distribuito non può garantire simultaneamente tutte e tre le seguenti proprietà:

* **Coerenza (C):** Tutti i nodi vedono gli stessi dati nello stesso momento.
* **Disponibilità (A):** Ogni richiesta riceverà una risposta entro un tempo ragionevole.
* **Tolleranza alle partizioni (P):** Il sistema continua a funzionare anche se si verificano partizioni di rete arbitrarie a causa di guasti.

La maggior parte dei sistemi NoSQL preferisce la coerenza alla disponibilità; altri preferiscono la disponibilità alla coerenza.

## Concetti Big Data

### Database Chiave-Valore

* **Scalabilità orizzontale:** Scalabilità molto elevata fornita tramite sharding.
* **Quando utilizzare:** Schema dati semplice o scenario di velocità estrema (ad esempio, in tempo reale).
* **Compromesso CAP:** La maggior parte delle soluzioni preferisce la coerenza alla disponibilità.
* **Pro:** Modello di dati semplice; scalabilità molto elevata; i dati possono essere accessibili utilizzando un linguaggio di query, come SQL.

### Database basati su documenti

* **Scaling orizzontale:** Capacità di scaling orizzontale limitate.
* **Quando usarli:** Quando è necessaria una semplice gestione di grandi quantità di dati, con requisiti di consistenza non stringenti.
* **Compromesso CAP:** La maggior parte delle soluzioni privilegia la disponibilità sulla consistenza.
* **Pro:** Semplicità di implementazione e manutenzione; adatto per grandi quantità di dati semplici.
* **Contro:** Alcune query potrebbero essere inefficienti o limitate a causa dello sharding (ad esempio, operazioni di join tra shard); nessuna standardizzazione dell'API; la manutenzione è difficile; non adatto per dati complessi.

### Database a Colonne

* **Scaling orizzontale:** Capacità di scaling orizzontale molto elevate.
* **Quando usarli:** Quando sono necessarie consistenza ed elevata scalabilità, senza utilizzare un front-end di caching indicizzato.
* **Compromesso CAP:** La maggior parte delle soluzioni privilegia la consistenza sulla disponibilità.
* **Pro:** Maggiore throughput e maggiore concorrenza quando è possibile partizionare i dati; query multi-attributo; i dati sono naturalmente indicizzati per colonne; supporto per dati semi-strutturati.
* **Contro:** Maggiore complessità; non adatto per dati interconnessi.

### Database basati su Grafo

* **Scaling orizzontale:** Scarso scaling orizzontale.
* **Quando usarli:** Per l'archiviazione e l'interrogazione di entità collegate tra loro da relazioni; i casi d'uso sono i social network e i motori di raccomandazione.
* **Compromesso CAP:** Di solito si preferisce la disponibilità alla consistenza.
* **Pro:** Potente modellazione dei dati e rappresentazione delle relazioni; dati connessi indicizzati localmente; facile ed efficiente da interrogare.
* **Contro:** Non adatto per dati non grafici.

## Analisi dei Dati vs. Analitica dei Dati

* **Analisi dei dati:** Le applicazioni esplorano, interrogano, analizzano, visualizzano e, in generale, elaborano set di dati su larga scala.
* **Analitica dei dati:** È la scienza di raccogliere ed esaminare dati grezzi allo scopo di trarre conclusioni significative su tali informazioni.

*Analitica dei dati* è un termine più ampio, che include *analisi dei dati*:

* *Analisi dei dati* si riferisce al "processo" di preparazione e analisi dei dati per estrarre conoscenze significative.
* *Analitica dei dati* include anche gli strumenti e le tecniche utilizzati a questo scopo, come strumenti di visualizzazione dei dati o data warehouse.

### Big Data Analytics

* **Big data analytics:** Si riferisce alle tecniche avanzate di analisi dei dati applicate a set di dati di grandi dimensioni: data mining, statistica, visualizzazione dei dati, intelligenza artificiale, machine learning, elaborazione del linguaggio naturale, ecc.

Alcuni campi di applicazione della big data analytics sono:

* Analisi del testo
* Analisi predittiva
* Analisi di grafi (o analisi di rete)
* Analisi prescrittiva

I processi di big data analytics sono computazionalmente intensivi, collaborativi e distribuiti per natura. I sistemi HPC e i cloud offrono strumenti e ambienti per supportare strategie di esecuzione parallela per l'analisi, l'inferenza e la scoperta su dati distribuiti disponibili in molti settori scientifici e aziendali. La creazione di framework basati sul **calcolo parallelo** e sulle **tecnologie cloud** è una condizione abilitante per lo sviluppo di attività di analisi dei dati ad alte prestazioni e tecniche di machine learning.

## Calcolo Parallelo

* **Concorrenza:** Due o più task possono essere *in corso* simultaneamente.
* **Parallelismo:** Due o più task vengono *eseguiti* simultaneamente.

Essere in corso non implica necessariamente essere in esecuzione, quindi il parallelismo richiede concorrenza, ma non viceversa.

Il **calcolo parallelo** può essere definito come la pratica di affrontare un problema di dimensione *n* suddividendolo in *k ≥ 2* parti che vengono risolte sfruttando *p* processori fisici, simultaneamente. Questo paradigma di risoluzione dei problemi, chiamato anche *divide et impera*, è applicabile solo se il problema è parallelizzabile, ovvero può essere espresso come una decomposizione in *k* sottoproblemi distinti.

## Divide et Impera

Il risultato finale si ottiene combinando le somme parziali (il che implica la necessità di sincronizzazione).

Formalmente:

$$\sum_{i=1}^{n} a_i b_i = \sum_{i=1}^{j_1} a_i b_i + \sum_{i=j_1+1}^{j_2} a_i b_i + \dots + \sum_{i=j_{k-1}+1}^{n} a_i b_i,$$

dove

$$1 < j_1 < j_2 < \cdots < j_{k-1} < n$$

### Divide et Impera: Un esempio

Prodotto scalare tra due vettori:

$$a = [a_1, a_2, \dots, a_n] \qquad \qquad b = [b_1, b_2, \dots, b_n]$$

definito come:

$$a \cdot b = a_1b_1 + a_2b_2 + \dots + a_nb_n = \sum_{i=1}^n a_ib_i$$

Può essere parallelizzato decomponendo il problema in *k* somme parziali (*divide* step) da calcolare su processori distinti simultaneamente (*impera* step).

## Natura parallela dei problemi

Un problema con dominio *D* può essere di due tipi:

* **Data-parallel:** *D* è un insieme di elementi dati e la soluzione del problema può essere espressa come l'applicazione di una funzione *f* a ciascun sottoinsieme di *D*:

$$f(D) = f(d_1) + f(d_2) + \dots + f(d_k)$$

* **Task-parallel:** *F* è un insieme di funzioni e la soluzione del problema può essere espressa come l'applicazione di ciascuna funzione in *F* a *D*:

$$F(D) = f_1(D) + f_2(D) + \dots + f_k(D)$$

## Architetture parallele

* **Tassonomia di Flynn** (Flynn e Rudd, 1996) classifica i diversi modelli di sistemi paralleli in base alla molteplicità del flusso di istruzioni e dati:
* Single instruction stream, single data stream (SISD)
* Single instruction stream, multiple data stream (SIMD)
* Multiple instruction stream, single data stream (MISD)
* Multiple instruction stream, multiple data stream (MIMD)

![[_page_36_Figure_3.jpeg|460]]]

## Metriche di performance

Dati:

* **Tempo di esecuzione sequenziale** (su 1 processore) $T_s$
* **Tempo di esecuzione parallelo** (su *n* processori) $T_n$

* **Speed-up:** rapporto tra il tempo di esecuzione sequenziale e quello parallelo:

$$S_n = T_s / T_n$$

* **Efficienza:** utilizzo effettivo di ciascun processore:

$$E_n = S_n/n = T_s/(n \cdot T_n)$$

## Legge di Amdahl

Dati:

* *n*: **numero di processori**
* *F*: **frazione parallelizzabile** del programma

Secondo la **legge di Amdahl**, lo speed-up teorico è:

$$ \hat{S}_n = \frac{1}{(1 - F) + \frac{F}{n}} $$

Casi limite:

* F = 0 (nessuna parte parallelizzabile) → $\hat{S}_n = 1$
* F = 1 (tutto parallelizzabile) → $\hat{S}_n = n$

Lo speed-up massimo ottenibile è:

$$S_{max} = \lim_{n \to \infty} \hat{S}_n = \frac{1}{(1 - F)}$$

Anche aumentando il parallelismo, lo speed-up è limitato dall'inverso della frazione non parallelizzabile.

![[_page_41_Figure_2.jpeg|450]]]

## Cloud computing

"Un modello per abilitare un accesso di rete conveniente, on-demand, a un pool condiviso di risorse di elaborazione configurabili (ad esempio, reti, server, storage, applicazioni e servizi) che possono essere rapidamente fornite e rilasciate con il minimo sforzo di gestione o interazione con il fornitore di servizi" (Mell *et al.*, 2011)

Cinque caratteristiche essenziali:

* Self-service on-demand
* Ampio accesso alla rete
* Pool di risorse
* Elasticità rapida
* Servizio misurato

### Modelli di servizio cloud

* **Software as a Service (SaaS):** software e dati forniti tramite Internet.
* **Platform as a Service (PaaS):** ambiente con database, server applicativi e ambiente di sviluppo.
* **Infrastructure as a Service (IaaS):** outsourcing di risorse come CPU, dischi o server virtualizzati.

### Modelli di deployment cloud

* **Cloud pubblico:** servizi al pubblico generale tramite Internet.
* **Cloud privato:** servizi su una intranet aziendale o data center privato.
* **Cloud ibrido:** composizione di due o più cloud (privati o pubblici).

### Servizi cloud per big data

I servizi per i big data basati su cloud sono forniti da:

* Amazon Web Services
* Google Cloud Platform
* Microsoft Azure
* OpenStack

## Piattaforme Cloud

### Amazon Web Services (AWS)

* **Data management:** database relazionali (Amazon RDS) e NoSQL (Amazon DynamoDB).
* **Compute:** Elastic Compute Cloud (EC2) e Amazon Elastic MapReduce.
* **Storage:** opzioni flessibili per dati permanenti e transitori.

### Google Cloud Platform

* **Compute:** IaaS (Google Compute Engine) e PaaS (Google App Engine).
* **Storage:** Google Cloud Storage, Datastore, Cloud SQL e Bigtable.
* **Networking:** Google Cloud DNS, CDN e servizi di sicurezza.

### Microsoft Azure

* **Compute:** ambiente computazionale con ruoli Web, Worker e Macchine Virtuali.
* **Storage:** storage scalabile per dati binari e testuali (Blob), tabelle non relazionali (Table) e code (Queue).
* **Fabric controller:** per costruire una rete di nodi interconnessi.

## OpenStack

OpenStack è una piattaforma di cloud computing open source che fornisce diversi servizi:

* **Compute:** Gestisce server virtuali su richiesta, utilizzando un pool di risorse di elaborazione disponibili nel data center. Supporta diverse tecnologie di virtualizzazione, come VMware e KVM.
* **Storage:** Fornisce un sistema di storage scalabile e ridondante, supportando Object Storage e Block Storage per l'archiviazione e il recupero di oggetti e file.
* **Networking:** Gestisce le reti e gli indirizzi IP all'interno dell'ambiente OpenStack.
* **Shared Services:** Offre servizi aggiuntivi come Identity Service (per la mappatura di utenti e servizi), Image Service (per la gestione delle immagini del server) e Database Service (per i database relazionali).

## Calcolo Exascale

* **High-performance computing (HPC):** Utilizza l'elaborazione parallela per gestire grandi quantità di dati e calcoli complessi.
* **High-performance data analytics (HPDA):** Applica le soluzioni HPC all'analisi dei dati.
* **Exascale:** Rappresenta la nuova frontiera dell'HPC, riferendosi a sistemi in grado di raggiungere almeno un exaFLOP (10<sup>18</sup> operazioni in virgola mobile al secondo).

### Attributi di un Sistema Exascale

Un sistema exascale è caratterizzato da diversi attributi:

* **Attributi fisici:** Consumo energetico, dimensioni (area e volume).
* **Tasso di calcolo:** Velocità di esecuzione delle operazioni (misurata in FLOPS, istruzioni al secondo e accessi alla memoria al secondo).
* **Capacità di storage:** Quantità di memoria disponibile nella gerarchia di storage (memoria principale, scratch e storage persistente).
* **Tasso di larghezza di banda:** Velocità di spostamento dei dati nel sistema (larghezza di banda della memoria locale, larghezza di banda del checkpoint, larghezza di banda I/O e larghezza di banda on-chip).

## Sistemi Exascale e Apprendimento Automatico Parallelo e Distribuito

### Principali Sfide dei Sistemi Exascale

* Energia
* Concorrenza
* Località dei dati
* Località dei nodi
* Località intra-rack
* Località inter-rack
* Memoria
* Resilienza

### Apprendimento Automatico Parallelo e Distribuito

Gli approcci centralizzati all'apprendimento automatico non sono adatti per grandi dataset distribuiti su molti dispositivi di storage. Per ridurre i tempi di esecuzione, si utilizzano modelli e infrastrutture di calcolo **parallelo** e **distribuito**.

### Strategie di Apprendimento Parallelo

* **Parallelismo indipendente:** Ogni processo accede all'intero dataset o alla propria partizione senza comunicazione o sincronizzazione.
* **Parallelismo Single Program Multiple Data (SPMD):** Insieme di processi che eseguono lo stesso algoritmo su diverse partizioni del dataset, cooperando tramite scambio di risultati parziali.
* **Parallelismo delle attività:** Ogni processo esegue algoritmi diversi su diverse partizioni del dataset, comunicando secondo le modalità richieste dall'algoritmo.

## Strategie di Apprendimento Distribuito

Nella maggior parte degli algoritmi distribuiti, un modello locale viene calcolato su ciascun sito e poi aggregato/combinato in un sito centrale o condiviso tra i nodi per produrre il modello globale. Questo schema è comune a diverse tecniche:

* Meta-apprendimento
* Apprendimento di ensemble
* Apprendimento federato
* Data mining collettivo

### Meta-apprendimento

Il meta-apprendimento crea un modello globale analizzando un insieme di dataset distribuiti. In uno scenario di classificazione:

- *N* algoritmi di apprendimento su *N* nodi creano *N* modelli di classificazione (*classificatori di base*).
- Un insieme di addestramento di meta-livello viene creato combinando le previsioni dei classificatori di base su un insieme di validazione comune.
- Un *classificatore globale* viene addestrato dall'insieme di addestramento di meta-livello da un algoritmo di meta-apprendimento.

### Apprendimento di Ensemble

L'apprendimento di ensemble migliora l'accuratezza del modello aggregando le previsioni di diversi learner. Due strategie principali sono:

* **Bagging:** Combina le previsioni di un insieme di modelli (dello stesso tipo o diversi) addestrati su diversi dataset.
* **Boosting:** Combina le decisioni di diversi modelli, dando più peso a quelli di maggior successo.

Il risultato è un classificatore di ensemble che mostra una maggiore accuratezza di classificazione rispetto a ciascun classificatore di base utilizzato per comporlo.

### Apprendimento Federato

L'apprendimento federato è progettato per analizzare dati grezzi distribuiti senza la necessità di spostarli su un singolo server o data center. Questa strategia seleziona un insieme di nodi e invia una prima versione, contenente i parametri del modello, a tutti i nodi. Ogni nodo esegue quindi il modello, lo addestra sui propri dati locali e mantiene una versione locale del modello. L'apprendimento federato consente ai dispositivi mobili di apprendere collaborativamente un modello di apprendimento condiviso, mantenendo tutti i dati di addestramento sui dispositivi stessi, migliorando così la sicurezza e la privacy.

### Data Mining Collettivo

Il data mining collettivo costruisce un modello globale combinando modelli *parziali* calcolati in diversi siti. Questo si differenzia da altre tecniche che combinano un insieme di modelli *completi* generati in ogni sito. La classificazione globale si basa sul principio che qualsiasi funzione può essere espressa in modo distribuito utilizzando un insieme di opportune *funzioni di base*, che possono includere termini non lineari. Se le funzioni di base sono ortonormali, un'analisi locale genera risultati che possono essere efficacemente utilizzati come componenti del modello globale. Se un termine non lineare è presente nella funzione di sommazione, il modello globale non è completamente scomponibile tra i siti locali e devono essere considerati termini crociati che coinvolgono caratteristiche da nodi diversi.
