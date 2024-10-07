
| **Termine**                     | **Definizione**                                                                                                                                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Apache Maven**                | Uno strumento per la gestione e l'automazione della build di progetti software, particolarmente utilizzato in ambito Java. Gestisce le dipendenze del progetto e semplifica il processo di compilazione e packaging. |
| **POM (Project Object Model)**  | Un file XML che descrive un progetto Maven, incluse le sue dipendenze, i plugin e altre informazioni di configurazione.                                                                                              |
| **Cluster Storm**               | Un sistema distribuito per l'elaborazione di flussi di dati in tempo reale, simile a Hadoop ma progettato per l'elaborazione continua.                                                                               |
| **Nimbus**                      | Il nodo master in un cluster Storm, responsabile della distribuzione del codice, dell'assegnazione dei task e del monitoraggio dello stato dei worker.                                                               |
| **Supervisor**                  | Un demone in esecuzione su ogni nodo worker in un cluster Storm, responsabile dell'avvio e della gestione dei processi worker.                                                                                       |
| **ZooKeeper**                   | Un servizio di coordinamento distribuito utilizzato da Storm per gestire lo stato del cluster e garantire la tolleranza ai guasti.                                                                                   |
| **Topologia**                   | Un programma Storm che definisce un grafo di elaborazione per un flusso di dati continuo.                                                                                                                            |
| **Spout**                       | Una sorgente di dati in una topologia Storm, responsabile della lettura dei dati da fonti esterne e dell'emissione di tuple nello stream.                                                                            |
| **Bolt**                        | Un'unità di elaborazione in una topologia Storm, responsabile dell'esecuzione di operazioni sui dati ricevuti dagli spout o da altri bolt.                                                                           |
| **Tuple**                       | L'unità di dati di base in Storm, rappresentata come una lista ordinata di campi.                                                                                                                                    |
| **Stream**                      | Una sequenza illimitata di tuple, elaborate in parallelo dai bolt in una topologia.                                                                                                                                  |
| **Tick Tuple**                  | Una tupla speciale emessa a intervalli regolari, utile per attivare operazioni periodiche come l'aggregazione o la scrittura su database.                                                                            |
| **Raggruppamento degli stream** | Meccanismo che determina come le tuple vengono distribuite tra i bolt in una topologia. Esempi includono: shuffle, fields, direct, all, global, none e local grouping.                                               |
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
1. **Nodo master (Nimbus)**:
   - Esegue il demone Nimbus, che ha le seguenti responsabilità:
     - Distribuisce il codice dell'applicazione nel cluster.
     - Assegna i task (compiti) ai nodi worker.
     - Monitora lo stato dei task per rilevare eventuali fallimenti.
     - Riavvia i task o li riassegna ad altre macchine in caso di errore.
   - Nimbus è progettato per essere **stateless**: non conserva direttamente alcuno stato. Utilizza **ZooKeeper** per memorizzare tutte le informazioni sullo stato e sulle configurazioni.

2. **Nodi worker (Supervisor)**:
   - Ogni nodo worker esegue un demone chiamato **Supervisor**.
   - Il Supervisor avvia e ferma i processi worker in base ai task assegnati da Nimbus.
   - Ogni worker esegue un sottoinsieme di una topologia.

3. **ZooKeeper**:
   - Coordina e condivide informazioni di configurazione tra Nimbus e i Supervisor.
   - Memorizza tutti gli stati associati al cluster e ai task.
   - Garantisce che i componenti Nimbus e Supervisor possano essere riavviati senza perdere lo stato.

#### Tolleranza ai guasti (Fault tolerance):
- Se un worker smette di funzionare, il **Supervisor** lo riavvia.
- Se il worker non può essere riavviato (perché, ad esempio, la connessione con Nimbus fallisce), **Nimbus** riassegna il task a un'altra macchina.
- **Nimbus e i Supervisor** sono progettati per essere **fail-fast** (ovvero, smettono immediatamente di funzionare in caso di errore) e **stateless** (non mantengono dati interni, affidandosi a ZooKeeper).

#### Processo di esecuzione:
1. **Sottomissione della topologia**: L'utente sottomette una topologia a Nimbus.
2. **Distribuzione dei task**: Nimbus distribuisce i task della topologia tra i Supervisor.
3. **Heartbeat**: I Supervisor inviano regolarmente segnali di attività (heartbeat) a Nimbus per confermare che stanno ancora funzionando.
4. **Esecuzione dei task**: I worker eseguono i task assegnati. Se un task fallisce, viene riavviato o riassegnato.
5. **Attesa di nuovi task**: Una volta completati i task, i Supervisor attendono nuovi task da Nimbus.

#### Supporto per linguaggi non JVM:
Storm supporta l'integrazione con linguaggi diversi da Java (non JVM) tramite il concetto di **componenti multilang** o "shelling". Questo permette di implementare spout e bolt in linguaggi come:
- **Python**
- **JavaScript**
- **Ruby**

L'integrazione avviene tramite uno script che esegue la logica dell'applicazione.


## Astrazioni di dati e calcolo
Il paradigma di programmazione offerto da Storm si basa su cinque astrazioni per dati e calcolo:
- Tuple: è l'unità base di dati che può essere elaborata da un'applicazione Storm. Una tupla consiste in una lista di campi (es. byte, char, integer, long e altri);
- Stream: rappresenta una sequenza illimitata di tuple, che è creata o elaborata in parallelo. Gli stream possono essere creati usando serializzatori standard (es. integer, double) o con serializzatori personalizzati;
- Spout: è la sorgente dati di uno stream. I dati sono letti da diverse fonti esterne, come API di social network, reti di sensori, sistemi di code (es. Java Message Service, Kafka, Redis), e poi sono alimentati nell'applicazione;
- Bolt: rappresenta l'entità di elaborazione. Specificamente, può eseguire qualsiasi tipo di task o algoritmo (es. pulizia dati, funzioni, join, query);
- Topology: rappresenta un job. Una topologia generica è configurata come un DAG, dove spout e bolt rappresentano i vertici del grafo e gli stream agiscono come i loro archi. Può girare per sempre fino a quando non viene fermata.

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
1. **getComponentConfiguration**:
   - Imposta la frequenza della tick tuple, che viene emessa ogni 10 secondi.
2. **execute**:
   - Il bolt raccoglie i conteggi in una mappa interna e li emette solo quando riceve una tick tuple, evitando emissioni continue.
3. **shuffleGrouping**:
   - Distribuisce le frasi in modo casuale tra le istanze del bolt che divide le frasi in parole.
4. **fieldsGrouping**:
   - Assicura che la stessa parola sia inviata alla stessa istanza del bolt per un conteggio corretto.
### Flusso della topologia:
1. Lo **spout** emette frasi casuali.
2. Il bolt **SplitSentence** divide le frasi in parole.
3. Il bolt **WordCount** conta le parole e le emette solo quando riceve una tick tuple.
4. La **tick tuple** attiva l'emissione dei conteggi ogni 10 secondi.

# Domande frequenti su Apache Storm

## 1. Cos'è Apache Storm e in cosa si differenzia da Hadoop?

Apache Storm è un sistema di elaborazione di flussi di dati distribuito, open-source e fault-tolerant. A differenza di Hadoop, che esegue job MapReduce con una durata definita, Storm esegue "topologie" che elaborano flussi di dati continui senza mai terminare (a meno che non vengano interrotte).

## 2. Quali sono i componenti principali di un cluster Storm?

Un cluster Storm è composto da tre tipi di nodi:

- **Nodo Master (Nimbus):** Distribuisce il codice, assegna i task ai worker e monitora lo stato del cluster. È stateless e utilizza ZooKeeper per la persistenza dei dati.
- **Nodi Worker (Supervisor):** Eseguono i processi worker che gestiscono i task assegnati da Nimbus. Ogni worker esegue un sottoinsieme di una topologia.
- **ZooKeeper:** Fornisce servizi di coordinamento e gestione dello stato tra Nimbus e i Supervisor.

## 3. Come gestisce Storm la tolleranza ai guasti?

Storm è progettato per essere fault-tolerant. Se un worker si arresta in modo anomalo, il Supervisor lo riavvia. Se il riavvio fallisce, Nimbus riassegna il task a un altro worker. Sia Nimbus che i Supervisor sono "fail-fast" e "stateless", garantendo che il sistema possa riprendersi dagli errori senza perdere dati.

## 4. Quali sono le cinque astrazioni chiave di dati e calcolo in Storm?

Storm utilizza cinque astrazioni per l'elaborazione dei dati:

- **Tuple:** Unità base di dati, rappresentata come un elenco di campi.
- **Stream:** Sequenza illimitata di tuple, elaborate in parallelo.
- **Spout:** Sorgente dei dati, legge da fonti esterne e genera stream di tuple.
- **Bolt:** Unità di elaborazione, esegue operazioni sulle tuple ricevute e può emetterne di nuove.
- **Topology:** Grafo diretto aciclico (DAG) che definisce il flusso di dati tra spout e bolt.

## 5. Cosa sono le Tick Tuple e come vengono utilizzate?

Le Tick Tuple sono tuple speciali emesse a intervalli regolari, utili per attività temporizzate all'interno di una topologia, come l'emissione periodica di risultati aggregati o la pulizia della cache.

## 6. Quali sono i tipi di raggruppamento di stream disponibili in Storm?

I raggruppamenti di stream definiscono come le tuple vengono distribuite tra i bolt. Alcuni tipi comuni sono:

- **Shuffle Grouping:** Distribuzione casuale delle tuple tra i bolt.
- **Fields Grouping:** Le tuple con lo stesso valore per i campi specificati vengono inviate allo stesso bolt.
- **Direct Grouping:** Il bolt emittente decide a quale bolt ricevente inviare la tupla.
- **All Grouping:** Lo stream viene replicato su tutti i bolt.

## 7. Come posso testare una topologia Storm localmente?

Storm fornisce una classe LocalCluster che consente di simulare un cluster Storm sul proprio computer, utile per lo sviluppo e il debug delle topologie prima di distribuirle su un cluster reale.

## 8. Storm supporta linguaggi diversi da Java?

Sì, Storm supporta linguaggi non JVM tramite il concetto di "componenti multilingua". Ciò significa che è possibile scrivere spout e bolt in linguaggi come Python, Ruby o JavaScript, che interagiscono con il resto della topologia tramite protocolli di comunicazione standard.

---
### Quiz a Risposta Breve

1. Descrivi brevemente Apache Maven e il suo ruolo nello sviluppo software.
2. Quali sono le differenze principali tra un cluster Hadoop e un cluster Storm?
3. Quali sono i tre componenti principali di un cluster Storm e le loro funzioni?
4. Cosa sono le tick tuple in Storm e come vengono utilizzate?
5. Spiega la differenza tra uno Spout e un Bolt in una topologia Storm.
6. Quali sono i metodi principali che devi implementare quando crei uno Spout personalizzato?
7. Descrivi il concetto di "raggruppamento degli stream" in Storm e fornisci un esempio di un tipo di raggruppamento.
8. Nel Problema 1, come viene modificata la parola di input dal bolt?
9. Nel Problema 2, qual è lo scopo di fieldsGrouping?
10. Nel contesto di Storm, cosa si intende per elaborazione "fault-tolerant"?

### Risposte al Quiz

1. Apache Maven è uno strumento di gestione e automazione della build di progetti software. Semplifica la gestione delle dipendenze, la compilazione del codice, l'esecuzione dei test e il packaging dell'applicazione.
2. Hadoop è progettato per l'elaborazione batch di grandi set di dati, mentre Storm è progettato per l'elaborazione di flussi di dati in tempo reale. Hadoop esegue job MapReduce con una durata definita, mentre Storm esegue topologie che elaborano dati continuamente.
3. I tre componenti principali di un cluster Storm sono: Nimbus (il nodo master), i Supervisor (che eseguono i processi worker) e ZooKeeper (per la gestione dello stato e la tolleranza ai guasti).
4. Le tick tuple sono tuple speciali emesse a intervalli regolari in una topologia. Sono utili per eseguire operazioni periodiche, come l'aggregazione dei dati o la scrittura su database, senza dover gestire manualmente il tempo.
5. Uno Spout è la sorgente dei dati in una topologia, leggendo dati da fonti esterne ed emettendoli come tuple. Un Bolt elabora le tuple ricevute dagli spout o da altri bolt, eseguendo operazioni come il filtraggio, l'aggregazione o la scrittura su database.
6. I metodi principali da implementare in uno Spout personalizzato sono: open() (per l'inizializzazione), nextTuple() (per l'emissione di tuple), ack() e fail() (per la gestione degli esiti delle tuple), e declareOutputFields() (per dichiarare lo schema delle tuple emesse).
7. Il "raggruppamento degli stream" determina come le tuple vengono distribuite tra le istanze dei bolt in una topologia. Ad esempio, fieldsGrouping invia tuple con lo stesso valore in un campo specifico alla stessa istanza del bolt.
8. Nel Problema 1, il bolt aggiunge tre punti esclamativi due volte alla parola di input, trasformando ad esempio "ciao" in "ciao!!!!!!".
9. Nel Problema 2, fieldsGrouping assicura che tutte le occorrenze della stessa parola vengano inviate alla stessa istanza del bolt WordCount, consentendo un conteggio accurato.
10. L'elaborazione "fault-tolerant" in Storm si riferisce alla capacità del sistema di continuare a funzionare correttamente anche in caso di guasti hardware o software. Ciò viene ottenuto tramite la replica dei dati, il monitoraggio dello stato dei worker e la riassegnazione automatica dei task in caso di errori.

