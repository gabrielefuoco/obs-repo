

### Modelli per sistemi exascale

I **sistemi exascale** rappresentano un'opportunità promettente, ma la loro progettazione e implementazione sono complesse a causa di sfide come scalabilità, latenza di rete, affidabilità e robustezza delle operazioni sui dati. La gestione efficiente di enormi volumi di dati richiede algoritmi scalabili in grado di partizionare e analizzare i dati attraverso milioni di operazioni parallele. I moderni sistemi HPC richiedono modelli di programmazione scalabili per prestazioni ottimali, supportando i programmatori nell'affrontare la complessità della gestione di milioni o miliardi di thread concorrenti.


## Requisiti dei Modelli Exascale

Un modello di programmazione exascale scalabile dovrebbe incorporare i seguenti meccanismi:

* **Accesso parallelo ai dati:** per migliorare la larghezza di banda di accesso ai dati accedendo contemporaneamente a elementi diversi.
* **Resilienza ai guasti:** per gestire i guasti che si verificano durante la comunicazione non locale.
* **Comunicazione locale guidata dai dati:** per limitare lo scambio di dati.
* **Elaborazione dei dati su gruppi limitati di core:** su specifiche macchine exascale.
* **Sincronizzazione near-data:** riducendo l'overhead generato dalla sincronizzazione tra molti core distanti.
* **Analisi in-memory:** per ridurre i tempi di reazione memorizzando nella cache i dati nelle RAM dei nodi di elaborazione.
* **Selezione dei dati basata sulla località:** per ridurre la latenza mantenendo localmente disponibile un sottoinsieme di dati.



Le soluzioni tradizionalmente utilizzate nei sistemi HPC (ad esempio, MPI, OpenMP e Map-Reduce) non sono sufficienti/appropriate per la programmazione di software progettato per essere eseguito su sistemi exascale. Le cinque proprietà più **essenziali** dei modelli di programmazione influenzate dalla transizione exascale sono:

* Scheduling dei thread
* Comunicazione
* Sincronizzazione
* Distribuzione dei dati
* Viste di controllo



I sistemi exascale mirano attualmente al **parallelismo a memoria distribuita** e, quindi, si prevede un'adozione parziale dell'architettura **message-passing**.  Mentre l'**MPI (Message Passing Interface)** si è dimostrata efficace con milioni di core in scenari specifici, presenta alcune sfide:

* Richiede agli utenti di affrontare vari aspetti della parallelizzazione, tra cui la distribuzione di dati e lavoro, la comunicazione e la sincronizzazione.
* Principalmente progettata per la distribuzione statica dei dati, non è adatta per il bilanciamento del carico dinamico.
* I problemi di scalabilità derivano dalla comunicazione many-to-many nel message passing, assumendo una rete completamente connessa con pattern di comunicazione densi.
* L'I/O diventa un collo di bottiglia nei sistemi basati su MPI, suggerendo la necessità di rivedere il modello attuale.



I sistemi exascale dovrebbero supportare centinaia di core su una singola CPU o GPU. L'utilizzo di sistemi a memoria condivisa parallela di medie dimensioni rappresenta un'alternativa valida al message passing, in quanto trasferisce la responsabilità della parallelizzazione dal programmatore al compilatore. I modelli di programmazione a **memoria condivisa** spesso adottano un modello di controllo del parallelismo privo di controllo della distribuzione dei dati e impiegano meccanismi di sincronizzazione non scalabili, come lock o sezioni atomiche. La visione globale dei dati promuove la sincronizzazione congiunta degli accessi ai dati remoti di tutti i thread, paragonabile agli accessi locali, con conseguente programmazione inefficiente.



I cluster che comprendono nodi eterogenei, che combinano CPU multi-core e GPU, sono sempre più utilizzati per l'HPC a causa dei vantaggi che offrono in termini di prestazioni di picco ed efficienza energetica. Per sfruttare appieno queste piattaforme, gli sviluppatori spesso impiegano una combinazione di paradigmi di programmazione parallela. Tuttavia, questa **programmazione eterogenea** presenta una nuova sfida nella gestione di ambienti di esecuzione e modelli di programmazione diversi. Data la natura soggetta a errori della programmazione su queste piattaforme, è necessario disporre di nuove astrazioni, modelli di programmazione e strumenti per affrontare queste sfide.



### Modelli per la programmazione exascala

Le applicazioni parallele su sistemi exascala devono gestire in modo efficiente milioni di thread in esecuzione su un array molto ampio di core, rendendo necessarie strategie per minimizzare la sincronizzazione, ridurre la comunicazione e l'utilizzo della memoria remota e affrontare potenziali guasti software e hardware. Diversi modelli di programmazione sono stati proposti per far fronte alle esigenze degli ambienti exascala, come:


### Legion

Legion è un modello di programmazione a memoria distribuita progettato per elevate prestazioni su diverse architetture parallele. L'organizzazione dei dati si basa sull'utilizzo di *regioni logiche*, che possono essere allocate dinamicamente, rimosse e utilizzate per memorizzare gruppi di oggetti in strutture dati. Le regioni possono anche essere fornite come input a funzioni distinte, chiamate *task*, che leggono i dati in regioni specifiche e forniscono informazioni sulla località. Le regioni logiche possono essere suddivise in sottoregioni *disgiunte* o *aliasate*, offrendo informazioni cruciali per valutare l'indipendenza del calcolo.


### Charm++

Charm++ è un modello di programmazione a memoria distribuita in cui un programma definisce collezioni di oggetti interagenti mappati dinamicamente ai processori dal sistema di runtime. Impiega un approccio asincrono, basato su messaggi e task, con oggetti mobili. Gli oggetti possono essere migrati tra i processori, consentendo alle operazioni di inviare dati a oggetti logici piuttosto che a processori fisici. Charm++ utilizza l'*overdecomposition* per dividere le applicazioni in molti oggetti piccoli che rappresentano unità di lavoro e/o dati grossolani, che possono superare di gran lunga il numero di processori.



### DCEx

DCEx è un modello di programmazione basato su PGAS per l'implementazione di applicazioni parallele su larga scala e datacentriche su sistemi exascala. È costruito su operazioni di base consapevoli dei dati per applicazioni data-intensive, consentendo l'utilizzo scalabile di un numero massiccio di elementi di elaborazione. Utilizzando strutture dati private e minimizzando lo scambio di dati tra thread concorrenti, DCEx impiega la *sincronizzazione near-data* per consentire ai thread di calcolo di operare in stretta vicinanza con i dati. Un programma DCEx è strutturato in blocchi data-paralleli, che fungono da unità di memoria/archiviazione per il calcolo, la comunicazione e la migrazione paralleli a memoria condivisa e distribuita.

### X10

X10 è un modello di programmazione basato su APGAS, che introduce le locazioni come astrazione del contesto computazionale. Le locazioni forniscono una vista localmente sincrona della memoria condivisa. In un calcolo X10, vengono distribuiti più luoghi, ognuno dei quali memorizza dati ed esegue una o più attività (thread leggeri) che possono essere create dinamicamente. Le attività possono utilizzare sincronicamente una o più regioni di memoria all'interno del luogo in cui risiedono.



### Chapel

Chapel è un modello di programmazione basato su APGAS che utilizza astrazioni di linguaggio di alto livello per la programmazione parallela generale, fornendo *strutture dati a vista globale* e una *vista globale del controllo* per migliorare il livello di astrazione sia per i dati che per il flusso di controllo. Le *strutture dati a vista globale* includono array e altri dati aggregati con dimensioni e indici rappresentati globalmente, anche se le loro implementazioni sono distribuite tra i *locale* del sistema parallelo. Un *locale* in Chapel è un'astrazione dell'unità di accesso uniforme alla memoria dell'architettura di destinazione, garantendo che tutti i thread all'interno di un locale abbiano tempi di accesso simili a qualsiasi singolo indirizzo di memoria. La **vista globale del controllo** significa che un'applicazione inizia con un singolo thread logico di controllo e introduce il parallelismo attraverso concetti specifici del linguaggio.



### UPC++

UPC++ è una libreria C++ progettata per la programmazione PGAS, che include strumenti per descrivere le dipendenze tra calcoli asincroni e trasferimento di dati. La libreria facilita la comunicazione unidirezionale efficiente e consente di spostare il calcolo sui dati tramite chiamate a procedure remote, facilitando l'implementazione di complesse strutture dati distribuite.  La libreria presenta tre concetti di programmazione principali:

* **Puntatori globali**, che supportano un'efficace sfruttamento della località dei dati.
* **Programmazione asincrona basata su RPC**, che consente lo sviluppo efficiente di programmi asincroni.
* **Futures**, per gestire la disponibilità dei dati provenienti da computazioni.

