
| **Termine** | **Definizione** |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Apache Maven** | Uno strumento per la gestione e l'automazione della build di progetti software, particolarmente utilizzato in ambito Java. Gestisce le dipendenze del progetto e semplifica il processo di compilazione e packaging. |
| **POM (Project Object Model)** | Un file XML che descrive un progetto Maven, incluse le sue dipendenze, i plugin e altre informazioni di configurazione. |
| **Cluster Storm** | Un sistema distribuito per l'elaborazione di flussi di dati in tempo reale, simile a Hadoop ma progettato per l'elaborazione continua. |
| **Nimbus** | Il nodo master in un cluster Storm, responsabile della distribuzione del codice, dell'assegnazione dei task e del monitoraggio dello stato dei worker. |
| **Supervisor** | Un demone in esecuzione su ogni nodo worker in un cluster Storm, responsabile dell'avvio e della gestione dei processi worker. |
| **ZooKeeper** | Un servizio di coordinamento distribuito utilizzato da Storm per gestire lo stato del cluster e garantire la tolleranza ai guasti. |
| **Topologia** | Un programma Storm che definisce un grafo di elaborazione per un flusso di dati continuo. |
| **Spout** | Una sorgente di dati in una topologia Storm, responsabile della lettura dei dati da fonti esterne e dell'emissione di tuple nello stream. |
| **Bolt** | Un'unità di elaborazione in una topologia Storm, responsabile dell'esecuzione di operazioni sui dati ricevuti dagli spout o da altri bolt. |
| **Tuple** | L'unità di dati di base in Storm, rappresentata come una lista ordinata di campi. |
| **Stream** | Una sequenza illimitata di tuple, elaborate in parallelo dai bolt in una topologia. |
| **Tick Tuple** | Una tupla speciale emessa a intervalli regolari, utile per attivare operazioni periodiche come l'aggregazione o la scrittura su database. |
| **Raggruppamento degli stream** | Meccanismo che determina come le tuple vengono distribuite tra i bolt in una topologia. Esempi includono: shuffle, fields, direct, all, global, none e local grouping. |
## Apache Maven

- Apache Maven è uno strumento di gestione di progetti software (Java, Scala, C#) e build automation.
- Maven usa un costrutto conosciuto come Project Object Model (POM); un file XML che descrive le dipendenze fra il progetto e le varie versioni di librerie necessarie nonché le dipendenze fra di esse.
- Maven effettua automaticamente il download di librerie Java e plug-in dai vari repository definiti scaricandoli in locale, permettendo di recuperare in modo uniforme i vari file JAR e di poter spostare il progetto indipendentemente da un ambiente all'altro avendo la sicurezza di utilizzare sempre le stesse versioni delle librerie.

### File di configurazione pom.xml (obbligatorio)

```xml
<dependency>
    <groupId>org.apache.storm</groupId>
    <artifactId>storm-client</artifactId>
    <version>${project.version}</version>
</dependency>
<dependency>
    <groupId>org.apache.storm</groupId>
    <artifactId>storm-server</artifactId>
    <version>${project.version}</version>
</dependency>
```

Il lato client contiene le funzionalità core necessarie per definire, configurare e sottomettere topologie Storm. Include classi e utilità per definire la logica di elaborazione dati (spout e bolt), serializzazione/deserializzazione, configurazione e interazione con un cluster Storm. Questa dipendenza è tipicamente usata in applicazioni client che definiscono e sottomettono topologie a un cluster Storm.

Il componente lato server che gira sul nodo master di un cluster Storm. Permette di eseguire una topologia in modalità locale automatizzata senza effettivamente avviare un cluster Storm. È utile per sviluppare e testare topologie.
## Cluster Storm

Un **cluster Storm** è simile a un cluster Hadoop, ma con alcune differenze chiave:
- **Hadoop** esegue **job MapReduce**, che hanno una durata definita e terminano una volta completato il lavoro.
- **Storm**, invece, esegue **topologie**, che sono processi continui per l'elaborazione di dati e non terminano mai, a meno che non vengano esplicitamente interrotti.

#### Componenti di un Cluster Storm:

- **Nodo master (Nimbus)**:
- Esegue il demone Nimbus, che ha le seguenti responsabilità:
- Distribuisce il codice dell'applicazione nel cluster.
- Assegna i task (compiti) ai nodi worker.
- Monitora lo stato dei task per rilevare eventuali fallimenti.
- Riavvia i task o li riassegna ad altre macchine in caso di errore.
- Nimbus è progettato per essere **stateless**: non conserva direttamente alcuno stato. Utilizza **ZooKeeper** per memorizzare tutte le informazioni sullo stato e sulle configurazioni.

- **Nodi worker (Supervisor)**:
- Ogni nodo worker esegue un demone chiamato **Supervisor**.
- Il Supervisor avvia e ferma i processi worker in base ai task assegnati da Nimbus.
- Ogni worker esegue un sottoinsieme di una topologia.

- **ZooKeeper**:
- Coordina e condivide informazioni di configurazione tra Nimbus e i Supervisor.
- Memorizza tutti gli stati associati al cluster e ai task.
- Garantisce che i componenti Nimbus e Supervisor possano essere riavviati senza perdere lo stato.

#### Tolleranza ai guasti (Fault tolerance):

- Se un worker smette di funzionare, il **Supervisor** lo riavvia.
- Se il worker non può essere riavviato (perché, ad esempio, la connessione con Nimbus fallisce), **Nimbus** riassegna il task a un'altra macchina.
- **Nimbus e i Supervisor** sono progettati per essere **fail-fast** (ovvero, smettono immediatamente di funzionare in caso di errore) e **stateless** (non mantengono dati interni, affidandosi a ZooKeeper).

#### Processo di esecuzione:

- **Sottomissione della topologia**: L'utente sottomette una topologia a Nimbus.
- **Distribuzione dei task**: Nimbus distribuisce i task della topologia tra i Supervisor.
- **Heartbeat**: I Supervisor inviano regolarmente segnali di attività (heartbeat) a Nimbus per confermare che stanno ancora funzionando.
- **Esecuzione dei task**: I worker eseguono i task assegnati. Se un task fallisce, viene riavviato o riassegnato.
- **Attesa di nuovi task**: Una volta completati i task, i Supervisor attendono nuovi task da Nimbus.

#### Supporto per linguaggi non JVM:

Storm supporta l'integrazione con linguaggi diversi da Java (non JVM) tramite il concetto di **componenti multilang** o "shelling". Questo permette di implementare spout e bolt in linguaggi come:
- **Python**
- **JavaScript**
- **Ruby**

L'integrazione avviene tramite uno script che esegue la logica dell'applicazione.

## Astrazioni di dati e calcolo

Il paradigma di programmazione offerto da Storm si basa su cinque astrazioni per dati e calcolo:
- **Tuple**: è l'unità base di dati che può essere elaborata da un'applicazione Storm. Una tupla consiste in una lista di campi (es. byte, char, integer, long e altri);
- **Stream**: rappresenta una sequenza illimitata di tuple, che è creata o elaborata in parallelo. Gli stream possono essere creati usando serializzatori standard (es. integer, double) o con serializzatori personalizzati;
- **Spout**: è la sorgente dati di uno stream. I dati sono letti da diverse fonti esterne, come API di social network, reti di sensori, sistemi di code (es. Java Message Service, Kafka, Redis), e poi sono alimentati nell'applicazione;
- **Bolt**: rappresenta l'entità di elaborazione. Specificamente, può eseguire qualsiasi tipo di task o algoritmo (es. pulizia dati, funzioni, join, query);
- **Topology**: rappresenta un job. Una topologia generica è configurata come un DAG, dove spout e bolt rappresentano i vertici del grafo e gli stream agiscono come i loro archi. Può girare per sempre fino a quando non viene fermata.

## Dettagli implementativi: Spout

Uno Spout è una sorgente di stream in una topologia. Generalmente gli spout leggono tuple da una fonte esterna e le emettono nella topologia (es. l'API di Twitter).

Per implementare uno Spout è necessario estendere una classe astratta che implementa l'interfaccia IRichSpout. I principali metodi sugli spout sono:

- `open()`, che viene chiamato quando un task viene inizializzato all'interno di un worker sul cluster.
- `nextTuple()`, che è usato per emettere tuple nella topologia attraverso un collettore. Questo metodo dovrebbe essere non bloccante, quindi se lo Spout non ha tuple da emettere dovrebbe ritornare.
- `ack()` e `fail()`, che sono chiamati quando Storm rileva che una tupla emessa dallo spout è stata completata con successo o ha fallito.
- `declareOutputFields()`, che dichiara lo schema di output per tutti gli stream della topologia.

## Dettagli implementativi: Bolt

I Bolt possono fare qualsiasi cosa, dal filtraggio, funzioni, aggregazioni, join, parlare con database, trasformazioni di stream semplici e complesse, che spesso richiedono molteplici passi e quindi molteplici bolt.

Per implementare un Bolt è necessario estendere una classe astratta che implementa l'interfaccia IRichBolt (interfaccia generale per i bolt) o IBasicBolt (interfaccia di convenienza per definire bolt che fanno filtraggio o funzioni semplici). I principali metodi sui bolt sono:

- `prepare()`, che fornisce al bolt un output collector che è usato per emettere tuple.
- `execute()`, che riceve una tupla da uno degli input del bolt, applica la logica di elaborazione e opzionalmente emette nuove tuple basate sulla tupla di input.
- `cleanup()`, che viene chiamato quando un bolt viene spento e dovrebbe pulire tutte le risorse che sono state aperte.
- `declareOutputFields()`, che dichiara lo schema di output per tutti gli stream della topologia.

## Raggruppamento degli stream

Parte della creazione di una topologia è definire quali stream ogni bolt dovrebbe ricevere come input tramite un raggruppamento di stream che determina come lo stream dovrebbe essere partizionato tra i bolt. Shuffle, fields e direct grouping sono esempi di raggruppamenti di stream ingenui:

- In shuffle grouping, le tuple sono divise casualmente tra i bolt in modo che ogni bolt riceva una quantità uguale di tuple.
- In field grouping, le tuple sono partizionate basandosi sul campo fornito nel raggruppamento (cioè, tuple con lo stesso campo sono assegnate allo stesso task, ma tuple con campi diversi possono essere elaborate da task diversi).
- In direct grouping il produttore della tupla determina quale task del consumatore la riceverà.
- In all grouping lo stream è replicato su tutti i task del bolt.
- In global grouping l'intero stream va a un singolo task del bolt. Specificamente, va al task con l'id più basso.
- In none grouping il programmatore non si preoccupa di come lo stream è raggruppato. Attualmente, i raggruppamenti none sono equivalenti ai raggruppamenti shuffle.
- In local grouping se il bolt di destinazione ha uno o più task nello stesso processo worker, le tuple saranno miscelate solo a quei task in-process. Altrimenti, questo agisce come un normale raggruppamento shuffle.

## Problema 1: Aggiungere punti esclamativi a parole casuali

### Descrizione:

L'obiettivo è creare un'applicazione Storm che prenda parole da una sorgente casuale (spout), e le trasformi aggiungendo tre punti esclamativi due volte (es. da "nathan" a "nathan!!!!!!").

### Componenti dell'applicazione:

#### Spout:

- **Funzione**: Il compito dello spout è emettere le parole casuali nello stream di output.
- Utilizza lo **SpoutOutputCollector** per emettere tuple (gruppi di dati) usando il metodo `emit()`.
- **Values** è una classe che rappresenta la tupla da emettere. In questo caso, la tupla è una singola parola.
- `declarer.declare(new Fields("word"))` viene utilizzato per dichiarare che ogni tupla contiene un campo chiamato "word", ovvero la parola casuale emessa.

#### Bolt:

- **Funzione**: Il bolt elabora la parola emessa dallo spout, aggiungendo tre punti esclamativi due volte alla parola.
- L'**OutputCollector** serve per emettere nuove tuple elaborate.
- La tupla (una lista di valori) contiene la parola originale a cui vengono aggiunti gli esclamativi.
- `collector.emit(tuple, new Values(val))` emette la nuova tupla, ancorata alla tupla originale per garantire la corretta gestione dei fallimenti.
- Anche in questo caso, si dichiara che si sta emettendo una tupla contenente un campo chiamato "word".

#### Topologia:

- La topologia definisce il flusso dell'applicazione, indicando come gli spout e i bolt sono collegati tra loro.
- **Config**: Consente di configurare la topologia, come il numero di worker per gestire i task.
- **TopologyBuilder**: Permette di costruire la topologia specificando spout e bolt.
- **Shuffle grouping**: Le tuple sono distribuite casualmente tra i bolt per bilanciare il carico.
- **Parallelism hint**: Definisce quanti thread devono eseguire un particolare bolt o spout.
- **LocalCluster**: Simula un cluster locale per testare la topologia senza distribuire l'applicazione su un vero cluster.

### Tick tuple:

- La **tick tuple** è una funzionalità che genera una tupla ad intervalli regolari. Viene usata per task che richiedono operazioni temporizzate, come:
- Pulire la cache ogni tot secondi.
- Inserire batch di dati in un database.

Per abilitare la tick tuple, si sovrascrive il metodo `getComponentConfiguration()`, specificando la frequenza con cui la tick tuple deve essere emessa (es. ogni 5 secondi).

### Soluzione complessiva per il Problema 1:

- Lo **spout** emette una parola casuale.
- Il **bolt** aggiunge sei punti esclamativi a ogni parola.
- La **topologia** definisce come i vari componenti (spout e bolt) interagiscono.

## Problema 2: Conteggio delle occorrenze delle parole

### Descrizione:

L'applicazione Storm prende frasi casuali, le divide in parole e conta quante volte ogni parola appare. La **tick tuple** viene usata per gestire il conteggio ed emissione dei risultati in modo efficiente.

### Punti principali:

- **getComponentConfiguration**:
- Imposta la frequenza della tick tuple, che viene emessa ogni 10 secondi.
- **execute**:
- Il bolt raccoglie i conteggi in una mappa interna e li emette solo quando riceve una tick tuple, evitando emissioni continue.
- **shuffleGrouping**:
- Distribuisce le frasi in modo casuale tra le istanze del bolt che divide le frasi in parole.
- **fieldsGrouping**:
- Assicura che la stessa parola sia inviata alla stessa istanza del bolt per un conteggio corretto.
### Flusso della topologia:

- Lo **spout** emette frasi casuali.
- Il bolt **SplitSentence** divide le frasi in parole.
- Il bolt **WordCount** conta le parole e le emette solo quando riceve una tick tuple.
- La **tick tuple** attiva l'emissione dei conteggi ogni 10 secondi.

