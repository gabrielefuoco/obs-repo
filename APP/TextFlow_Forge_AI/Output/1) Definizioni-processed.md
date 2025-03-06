
**I. Big Data: Definizioni e Caratteristiche**

*   **A. Definizioni:**
    *   1.  **Definizione 1 (Gartner, Inc.):** Asset informativi ad alto volume, alta velocità e/o alta varietà che richiedono elaborazione innovativa.
        *   a.  Modello "3V":
            *   i.  Volume: Quantità di dati.
            *   ii. Velocità: Velocità di generazione e elaborazione.
            *   iii. Varietà: Diversità dei tipi di dati.
    *   2.  **Definizione 2 (Gantz e Reinsel, 2011):** Tecnologie per estrarre valore da grandi volumi di dati ad alta velocità.
        *   a.  Modello "4V":
            *   i.  Volume
            *   ii. Velocità
            *   iii. Varietà
            *   iv. Valore: Utilità delle informazioni estratte.
    *   3.  **Definizione 3 (Chang e Grady, 2015):** Ampi set di dati caratterizzati da volume, varietà, velocità e/o variabilità.
        *   a.  Modello "4V":
            *   i.  Volume
            *   ii. Velocità
            *   iii. Varietà
            *   iv. Variabilità: Cambiamenti nel tempo dei dati.
*   **B. Altre "V":**
    *   1.  Viralità: Capacità di diffondersi rapidamente.
    *   2.  Visualizzazione: Possibilità di rappresentare graficamente i dati.
    *   3.  Viscosità: Capacità delle informazioni estratte di mantenere l'interesse.
    *   4.  Luogo (Venue): Origine dei dati.

**II. Data Science**

*   **A. Definizione:** Disciplina interdisciplinare che combina informatica, matematica applicata e analisi dei dati per estrarre informazioni.
*   **B. Processi di Data Science:**
    *   1.  Formulazione del problema.
    *   2.  Raccolta dei dati necessari.
    *   3.  Elaborazione dei dati per l'analisi.
    *   4.  Esplorazione dei dati.
    *   5.  Esecuzione di analisi approfondite.
    *   6.  Comunicazione dei risultati dell'analisi.
*   **C. Competenze in Data Science:**
    *   1.  Apprendimento del dominio di applicazione.
    *   2.  Comunicazione con i proprietari/utenti dei dati.
    *   3.  Attenzione alla qualità dei dati.
    *   4.  Conoscenza della rappresentazione dei dati.
    *   5.  Gestione della trasformazione e dell'analisi dei dati.
    *   6.  Conoscenza della visualizzazione e della presentazione dei dati.
    *   7.  Considerazione delle questioni etiche.

**III. Archiviazione Big Data: Approcci di Scaling**

*   **A. Approcci Principali:**
    *   1.  Scaling Verticale: Aumento delle risorse di un singolo server (CPU, RAM, disco, I/O di rete).
    *   2.  Scaling Orizzontale: Aggiunta di più nodi di archiviazione/elaborazione al sistema.
*   **B. Database NoSQL:**
    *   1.  Alternativa per la scalabilità orizzontale delle operazioni di lettura/scrittura.
    *   2.  Sfruttano nuovi nodi in modo trasparente.
    *   3.  Garantiscono la distribuzione automatica dei dati e la tolleranza ai guasti.
    *   4.  Supportano l'archiviazione di valori scalari, oggetti binari e strutture complesse.

---

**I. Tipi di Database NoSQL**

    A. Archiviazione Chiave-Valore
        1.  Archivia i dati come coppie ⟨chiave, valore⟩ su più server.
        2.  Utilizza una tabella hash distribuita (DHT) per l'indicizzazione scalabile.
        3.  Recupero dati tramite chiave per trovare il valore.
        4.  Esempi: DynamoDB, Redis.

    B. Archivi di Documenti
        1.  Supportano indici secondari e più tipi di documenti per database.
        2.  Forniscono meccanismi per interrogare le collezioni in base a più vincoli di valore degli attributi.
        3.  Esempio: MongoDB.

    C. Archivi Basati su Colonne
        1.  Archiviano record estensibili partizionabili su più server.
        2.  Record estensibili: è possibile aggiungere nuovi attributi.
        3.  Forniscono partizione orizzontale (record su nodi diversi) e verticale (parti di un singolo record su server diversi).
        4.  Le colonne possono essere distribuite su più server usando gruppi di colonne.
        5.  Esempi: Cassandra, HBase, BigTable.

    D. Archivi Basati su Grafi
        1.  Archiviano e interrogano informazioni rappresentate come grafi (nodi, archi e proprietà).
        2.  Adatti per dati complessi da gestire con database relazionali.
        3.  Consentono query efficienti sui grafi, accelerando algoritmi grafici (comunità, gradi, centralità, distanze, percorsi).
        4.  Esempio: Neo4j.

**II. Esempi di Database NoSQL**

    A. MongoDB
        1.  Tipo: Archivio basato su documenti.
        2.  Descrizione:
            a.  Progettato per supportare applicazioni internet e basate sul web.
            b.  Rappresenta i documenti in formato BSON (simile a JSON).
            c.  Accede a oggetti nidificati usando la notazione a punto.
            d.  Supporta query complesse e indici completi.
            e.  Include funzionalità come lo sharding automatico, la replica e la gestione semplificata dell'archiviazione.

    B. Google Bigtable
        1.  Tipo: Archivio basato su colonne.
        2.  Descrizione:
            a.  Costruito su Google File System (GFS).
            b.  Può archiviare fino a petabyte di dati.
            c.  Archivia i dati in tabelle multidimensionali sparse, distribuite e persistenti, composte da righe e colonne.
            d.  Ogni riga è indicizzata da una singola chiave di riga.
            e.  Le colonne correlate sono raggruppate in insiemi chiamati famiglie di colonne.
            f.  Una colonna generica è identificata da una famiglia di colonne e un qualificatore di colonna.
            g.  I dati sono ordinati per chiave di riga e l'intervallo di righe è partizionato in tablet.
            h.  I tablet sono distribuiti tra diversi nodi di un cluster Bigtable.

    C. HBase
        1.  Tipo: Archivio basato su colonne.
        2.  Descrizione:
            a.  Sfrutta Hadoop e Hadoop Distributed File System (HDFS).
            b.  Progettato per scalare linearmente.
            c.  Costituito da tabelle standard con righe e colonne.
            d.  Tabelle senza schema.
            e.  Le famiglie di colonne sono definite durante la creazione di una tabella.
            f.  Ogni tabella deve avere una chiave primaria definita.
            g.  Si integra bene con Hive per l'elaborazione batch di big data.

    D. Redis
        1. Tipo: Archivio Key-value.

---

**Schema Riassuntivo dei Data Store NoSQL**

**1. Redis**
    *   **Descrizione:** Data store in-memory open-source versatile.
    *   **Funzionalità:**
        *   Database, cache, message broker, motore di streaming.
        *   Supporto operazioni atomiche (stringhe, hash, liste, insiemi, insiemi ordinati).
    *   **Persistenza:** Dati persistiti su disco periodicamente (a seconda del caso d'uso).

**2. DynamoDB**
    *   **Tipo:** Key-value store.
    *   **Descrizione:** Servizio NoSQL completamente gestito su AWS.
    *   **Caratteristiche:**
        *   Bassa latenza.
        *   Sicurezza integrata (crittografia, controllo accessi, IAM).
        *   Backup e ripristino gestiti.
        *   Sincronizzazione multi-regione, multi-master (alta disponibilità e durata).
        *   Integrazione con altri servizi AWS (es. Amazon S3).

**3. Cassandra**
    *   **Tipo:** Column-based store.
    *   **Descrizione:** Architettura ad anello senza master.
    *   **Caratteristiche:**
        *   Nodi con ruolo identico.
        *   Aggiunta di nodi senza downtime.
        *   Distribuzione dati automatica.
        *   Nessun singolo punto di errore (alta disponibilità).
        *   Replica dati personalizzabile.

**4. Neo4j**
    *   **Tipo:** Graph-based store.
    *   **Descrizione:** Data store basato su grafi.
    *   **Struttura:**
        *   Nodi con lista di relazioni e attributi.
        *   Relazioni con nome, direzione, nodo iniziale e finale, proprietà.
        *   Etichette per nodi e relazioni (ruoli, indici, vincoli).
    *   **Scalabilità:** Cluster per alta disponibilità e scalabilità di lettura (replica master-slave).

---

## Schema Riassuntivo Sistemi NoSQL e Big Data

### 1. Confronto Sistemi NoSQL

*   **1.1 Tabella Comparativa:** (Rielaborazione della tabella originale)

    | Sistema     | Tipo    | Archiviazione Dati | MapReduce | Persistenza | Replica | Scalabilità | Prestazioni | Alta Disponibilità | Linguaggio                       | Licenza       |
    | ----------- | ------- | -------------------- | --------- | ------------- | ------- | ------------- | ------------- | -------------------- | -------------------------------- | ------------- |
    | DynamoDB    | KV      | MEM/FS               | Sì        | Sì            | Sì      | Alta          | Alta          | Sì                   | Java                             | Proprietaria  |
    | Cassandra   | Colonna | HDFS/CFS             | Sì        | Sì            | Sì      | Alta          | Alta          | Sì                   | Java                             | Apache2       |
    | HBase       | Colonna | HDFS                 | Sì        | Sì            | Sì      | Alta          | Alta          | Sì                   | Java                             | Apache2       |
    | Redis       | KV      | MEM/FS               | No        | Sì            | Sì      | Alta          | Alta          | Sì                   | Ansi-C                           | BSD           |
    | BigTable    | Colonna | GFS                  | Sì        | Sì            | Sì      | Alta          | Alta          | Sì                   | Java/Python/Go/Ruby              | Proprietaria  |
    | MongoDB     | Documento | MEM/FS               | Sì        | Sì            | Sì      | Alta          | Alta          | Sì                   | C++                              | AGPL3         |
    | Neo4j       | Grafo   | MEM/FS               | No        | Sì            | Sì      | Alta          | Alta (variabile) | Sì                   | Java                             | GPL3          |

*   **1.2 Teorema CAP (Gilbert e Lynch, 2002):**

    *   **1.2.1 Definizione:** Un sistema distribuito non può garantire simultaneamente:
        *   **Coerenza (C):** Tutti i nodi vedono gli stessi dati nello stesso momento.
        *   **Disponibilità (A):** Ogni richiesta riceve una risposta entro un tempo ragionevole.
        *   **Tolleranza alle partizioni (P):** Il sistema continua a funzionare anche con partizioni di rete.
    *   **1.2.2 Compromessi:** Sistemi NoSQL spesso privilegiano C o A.

### 2. Concetti Big Data

*   **2.1 Database Chiave-Valore (KV):**

    *   **2.1.1 Scalabilità:** Scalabilità orizzontale elevata (sharding).
    *   **2.1.2 Quando Usare:** Schema dati semplice, velocità estrema (es. real-time).
    *   **2.1.3 Compromesso CAP:** Preferenza per la Coerenza (C).
    *   **2.1.4 Pro:** Modello dati semplice, alta scalabilità, accesso dati tramite linguaggi di query (es. SQL).

*   **2.2 Database basati su Documenti:**

    *   **2.2.1 Scalabilità:** Scalabilità orizzontale limitata.
    *   **2.2.2 Quando Usare:** Gestione semplice di grandi quantità di dati, requisiti di consistenza non stringenti.
    *   **2.2.3 Compromesso CAP:** Preferenza per la Disponibilità (A).
    *   **2.2.4 Pro:** Semplicità di implementazione e manutenzione, adatto per grandi quantità di dati semplici.
    *   **2.2.5 Contro:** Query inefficienti/limitate (sharding, join), mancanza di standardizzazione API, manutenzione difficile, inadatto per dati complessi.

*   **2.3 Database a Colonne:**

    *   **2.3.1 Scalabilità:** Scalabilità orizzontale molto elevata.
    *   **2.3.2 Quando Usare:** Consistenza ed elevata scalabilità, senza caching indicizzato front-end.
    *   **2.3.3 Compromesso CAP:** Preferenza per la Coerenza (C).
    *   **2.3.4 Pro:** Maggiore throughput e concorrenza (partizionamento), query multi-attributo, indicizzazione naturale per colonne, supporto dati semi-strutturati.
    *   **2.3.5 Contro:** Maggiore complessità, inadatto per dati interconnessi.

*   **2.4 Database basati su Grafo:**

    *   **2.4.1 Scalabilità:** Scarso scaling orizzontale.
    *   **2.4.2 Quando Usare:** Archiviazione e interrogazione di entità collegate da relazioni (social network, motori di raccomandazione).
    *   **2.4.3 Compromesso CAP:** Preferenza per la Disponibilità (A).

---

**I. Grafi:**

*   **A.** Pro:
    *   1.  Potente modellazione dei dati e rappresentazione delle relazioni.
    *   2.  Dati connessi indicizzati localmente.
    *   3.  Facile ed efficiente da interrogare.
*   **B.** Contro:
    *   1.  Non adatto per dati non grafici.

**II. Analisi dei Dati vs. Analitica dei Dati:**

*   **A.** Analisi dei dati:
    *   1.  Esplorazione, interrogazione, analisi, visualizzazione ed elaborazione di set di dati su larga scala.
*   **B.** Analitica dei dati:
    *   1.  Scienza di raccogliere ed esaminare dati grezzi per trarre conclusioni significative.
    *   2.  Termine più ampio che include l'analisi dei dati.
        *   a. Analisi dei dati: processo di preparazione e analisi dei dati per estrarre conoscenze significative.
        *   b. Analitica dei dati: include strumenti e tecniche (visualizzazione, data warehouse).

**III. Big Data Analytics:**

*   **A.** Definizione:
    *   1.  Tecniche avanzate di analisi dei dati applicate a set di dati di grandi dimensioni.
        *   a. Data mining, statistica, visualizzazione dei dati, intelligenza artificiale, machine learning, elaborazione del linguaggio naturale, ecc.
*   **B.** Campi di applicazione:
    *   1.  Analisi del testo.
    *   2.  Analisi predittiva.
    *   3.  Analisi di grafi (o analisi di rete).
    *   4.  Analisi prescrittiva.
*   **C.** Caratteristiche:
    *   1.  Computazionalmente intensivi, collaborativi e distribuiti.
*   **D.** Supporto:
    *   1.  Sistemi HPC e cloud offrono strumenti e ambienti per l'esecuzione parallela.
    *   2.  Calcolo parallelo e tecnologie cloud abilitano analisi dei dati ad alte prestazioni e machine learning.

**IV. Calcolo Parallelo:**

*   **A.** Concorrenza:
    *   1.  Due o più task possono essere *in corso* simultaneamente.
*   **B.** Parallelismo:
    *   1.  Due o più task vengono *eseguiti* simultaneamente.
    *   2.  Il parallelismo richiede concorrenza, ma non viceversa.
*   **C.** Definizione:
    *   1.  Suddivisione di un problema di dimensione *n* in *k ≥ 2* parti risolte simultaneamente su *p* processori fisici.
*   **D.** Condizione:
    *   1.  Il problema deve essere parallelizzabile (decomponibile in *k* sottoproblemi distinti).

**V. Divide et Impera:**

*   **A.** Descrizione:
    *   1.  Il risultato finale si ottiene combinando le somme parziali (richiede sincronizzazione).
*   **B.** Formula:
    *   1.  $$\sum_{i=1}^{n} a_i b_i = \sum_{i=1}^{j_1} a_i b_i + \sum_{i=j_1+1}^{j_2} a_i b_i + \dots + \sum_{i=j_{k-1}+1}^{n} a_i b_i,$$ dove $$1 < j_1 < j_2 < \cdots < j_{k-1} < n$$

**VI. Divide et Impera: Esempio (Prodotto Scalare):**

*   **A.** Vettori:
    *   1.  $$a = [a_1, a_2, \dots, a_n] \qquad \qquad b = [b_1, b_2, \dots, b_n]$$
*   **B.** Prodotto scalare:
    *   1.  $$a \cdot b = a_1b_1 + a_2b_2 + \dots + a_nb_n = \sum_{i=1}^n a_ib_i$$
*   **C.** Parallelizzazione:
    *   1.  Decomposizione in *k* somme parziali (divide) calcolate simultaneamente (impera).

---

Ecco lo schema riassuntivo del testo fornito:

**1. Natura Parallela dei Problemi**

   *   **1.1 Data-parallel:**
        *   Dominio *D* è un insieme di elementi dati.
        *   Soluzione: applicazione di *f* a sottoinsiemi di *D*.
        *   Formula:  `f(D) = f(d_1) + f(d_2) + ... + f(d_k)`

   *   **1.2 Task-parallel:**
        *   *F* è un insieme di funzioni.
        *   Soluzione: applicazione di ogni funzione in *F* a *D*.
        *   Formula: `F(D) = f_1(D) + f_2(D) + ... + f_k(D)`

**2. Architetture Parallele**

   *   **2.1 Tassonomia di Flynn:**
        *   Classificazione basata su flusso di istruzioni e dati.
        *   **2.1.1 SISD:** Single instruction stream, single data stream.
        *   **2.1.2 SIMD:** Single instruction stream, multiple data stream.
        *   **2.1.3 MISD:** Multiple instruction stream, single data stream.
        *   **2.1.4 MIMD:** Multiple instruction stream, multiple data stream.

**3. Metriche di Performance**

   *   **3.1 Tempo di esecuzione sequenziale:** *T<sub>s</sub>* (su 1 processore)
   *   **3.2 Tempo di esecuzione parallelo:** *T<sub>n</sub>* (su *n* processori)
   *   **3.3 Speed-up:**
        *   Definizione: Rapporto tra tempo sequenziale e parallelo.
        *   Formula: `S_n = T_s / T_n`
   *   **3.4 Efficienza:**
        *   Definizione: Utilizzo effettivo di ciascun processore.
        *   Formula: `E_n = S_n/n = T_s/(n * T_n)`

**4. Legge di Amdahl**

   *   **4.1 Definizioni:**
        *   *n*: Numero di processori.
        *   *F*: Frazione parallelizzabile del programma.
   *   **4.2 Speed-up teorico:**
        *   Formula: `Ŝ_n = 1 / ((1 - F) + F/n)`
   *   **4.3 Casi Limite:**
        *   F = 0: `Ŝ_n = 1`
        *   F = 1: `Ŝ_n = n`
   *   **4.4 Speed-up massimo:**
        *   Formula: `S_max = lim (n→∞) Ŝ_n = 1 / (1 - F)`
        *   Limitato dalla frazione non parallelizzabile.

**5. Cloud Computing**

   *   **5.1 Definizione:** Accesso on-demand a risorse di calcolo condivise.
   *   **5.2 Caratteristiche Essenziali:**
        *   Self-service on-demand
        *   Ampio accesso alla rete
        *   Pool di risorse
        *   Elasticità rapida
        *   Servizio misurato
   *   **5.3 Modelli di Servizio Cloud:**
        *   **5.3.1 SaaS:** Software as a Service (software e dati via Internet).
        *   **5.3.2 PaaS:** Platform as a Service (ambiente di sviluppo, database, server).
        *   **5.3.3 IaaS:** Infrastructure as a Service (risorse come CPU, dischi, server virtualizzati).
   *   **5.4 Modelli di Deployment Cloud:**
        *   **5.4.1 Cloud Pubblico:** Servizi al pubblico generale via Internet.
        *   **5.4.2 Cloud Privato:** Servizi su intranet aziendale o data center privato.
        *   **5.4.3 Cloud Ibrido:** Composizione di cloud privati e pubblici.
   *   **5.5 Servizi Cloud per Big Data:**
        *   Amazon Web Services
        *   Google Cloud Platform
        *   Microsoft Azure
        *   OpenStack

**6. Piattaforme Cloud**

   *   **6.1 Amazon Web Services (AWS):**
        *   **6.1.1 Data Management:** Amazon RDS (relazionale), Amazon DynamoDB (NoSQL).
        *   **6.1.2 Compute:** EC2, Amazon Elastic MapReduce.
        *   **6.1.3 Storage:** Opzioni flessibili per dati permanenti e transitori.
   *   **6.2 Google Cloud Platform:**
        *   **6.2.1 Compute:** Google Compute Engine (IaaS), Google App Engine (PaaS).
        *   **6.2.2 Storage:** Google Cloud Storage, Datastore, Cloud SQL, Bigtable.
        *   **6.2.3 Networking:** Google Cloud DNS, CDN, servizi di sicurezza.
   *   **6.3 Microsoft Azure:**
        *   **6.3.1 Compute:** Ruoli Web, Worker, Macchine Virtuali.

---

**Schema Riassuntivo**

**1. Piattaforme Cloud**

    *   **1.1 Azure**
        *   1.1.1 Storage: Blob (dati binari e testuali), Table (tabelle non relazionali), Queue (code).
        *   1.1.2 Fabric controller: Rete di nodi interconnessi.

    *   **1.2 OpenStack**
        *   1.2.1 Compute: Gestione server virtuali su richiesta (VMware, KVM).
        *   1.2.2 Storage: Object Storage, Block Storage (scalabile e ridondante).
        *   1.2.3 Networking: Gestione reti e indirizzi IP.
        *   1.2.4 Shared Services:
            *   Identity Service: Mappatura utenti e servizi.
            *   Image Service: Gestione immagini server.
            *   Database Service: Database relazionali.

**2. Calcolo Exascale**

    *   **2.1 Concetti Fondamentali**
        *   2.1.1 High-performance computing (HPC): Elaborazione parallela per grandi dati e calcoli complessi.
        *   2.1.2 High-performance data analytics (HPDA): Applicazione HPC all'analisi dei dati.
        *   2.1.3 Exascale: Sistemi con almeno 1 exaFLOP (10<sup>18</sup> operazioni in virgola mobile al secondo).

    *   **2.2 Attributi di un Sistema Exascale**
        *   2.2.1 Attributi fisici: Consumo energetico, dimensioni (area e volume).
        *   2.2.2 Tasso di calcolo: FLOPS, istruzioni al secondo, accessi alla memoria al secondo.
        *   2.2.3 Capacità di storage: Memoria principale, scratch, storage persistente.
        *   2.2.4 Tasso di larghezza di banda: Larghezza di banda della memoria locale, larghezza di banda del checkpoint, larghezza di banda I/O, larghezza di banda on-chip.

**3. Sistemi Exascale e Apprendimento Automatico Parallelo e Distribuito**

    *   **3.1 Principali Sfide dei Sistemi Exascale**
        *   Energia
        *   Concorrenza
        *   Località dei dati
        *   Località dei nodi
        *   Località intra-rack
        *   Località inter-rack
        *   Memoria
        *   Resilienza

    *   **3.2 Apprendimento Automatico Parallelo e Distribuito**
        *   3.2.1 Necessità: Gestione di grandi dataset distribuiti.
        *   3.2.2 Approccio: Modelli e infrastrutture di calcolo parallelo e distribuito.

    *   **3.3 Strategie di Apprendimento Parallelo**
        *   3.3.1 Parallelismo indipendente: Accesso indipendente ai dati.
        *   3.3.2 Parallelismo Single Program Multiple Data (SPMD): Stesso algoritmo su diverse partizioni, cooperazione tramite scambio risultati.
        *   3.3.3 Parallelismo delle attività: Algoritmi diversi su diverse partizioni, comunicazione specifica.

    *   **3.4 Strategie di Apprendimento Distribuito**
        *   3.4.1 Schema Generale: Modello locale calcolato su ciascun sito, aggregato/combinato per modello globale.
        *   3.4.2 Tecniche:
            *   Meta-apprendimento
            *   Apprendimento di ensemble
            *   Apprendimento federato
            *   Data mining collettivo

    *   **3.5 Meta-apprendimento**
        *   3.5.1 Creazione modello globale analizzando dataset distribuiti.
        *   3.5.2 Scenario di classificazione:
            *   *N* algoritmi su *N* nodi creano *N* classificatori di base.
            *   Insieme di addestramento di meta-livello creato combinando previsioni dei classificatori di base su un insieme di validazione.
            *   Classificatore globale addestrato dall'insieme di addestramento di meta-livello.

---

**Schema Riassuntivo:**

**I. Apprendimento di Ensemble**

    *   Migliora l'accuratezza del modello aggregando le previsioni di diversi learner.
    *   Strategie principali:
        *   **Bagging:** Combina le previsioni di modelli addestrati su diversi dataset.
        *   **Boosting:** Combina le decisioni di modelli, dando più peso a quelli di maggior successo.
    *   Risultato: Classificatore di ensemble con maggiore accuratezza rispetto ai classificatori di base.

**II. Apprendimento Federato**

    *   Analizza dati distribuiti senza spostarli su un singolo server.
    *   Processo:
        *   Selezione di un insieme di nodi.
        *   Invio di una versione iniziale del modello (parametri) ai nodi.
        *   Ogni nodo addestra il modello sui propri dati locali.
        *   Mantenimento di una versione locale del modello su ogni nodo.
    *   Benefici: Permette ai dispositivi mobili di apprendere collaborativamente mantenendo i dati di addestramento sui dispositivi, migliorando sicurezza e privacy.

**III. Data Mining Collettivo**

    *   Costruisce un modello globale combinando modelli *parziali* calcolati in diversi siti.
    *   Differenza chiave: Combina modelli *parziali* (vs. modelli *completi* in altre tecniche).
    *   Principio: Qualsiasi funzione può essere espressa distribuita usando funzioni di base.
    *   Funzioni di base ortonormali: Analisi locale genera risultati utilizzabili come componenti del modello globale.
    *   Termini non lineari:
        *   Modello globale non completamente scomponibile.
        *   Necessità di considerare termini crociati tra caratteristiche di nodi diversi.

---
