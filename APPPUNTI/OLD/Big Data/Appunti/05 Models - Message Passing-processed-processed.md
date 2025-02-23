## Il Modello di Passaggio di Messaggi

Il **modello di passaggio di messaggi** è un paradigma per la comunicazione inter-processo (IPC) nell'informatica distribuita, dove ogni elemento di elaborazione ha la propria memoria privata. I meccanismi IPC, forniti dal sistema operativo, includono la *memoria condivisa* e la *memoria distribuita* o il *passaggio di messaggi*. I modelli di programmazione parallela sono generalmente categorizzati in base all'utilizzo della memoria.



## Modello a Memoria Condivisa vs. Modello a Passaggio di Messaggi

Nel **modello a memoria condivisa**, più processi accedono a uno spazio di indirizzi condiviso. Tali processi possono comunicare condividendo direttamente le variabili, e la comunicazione è tipicamente più veloce ma richiede meccanismi di sincronizzazione.

Nel **modello a passaggio di messaggi**, un'applicazione opera come un insieme di processi indipendenti, ognuno con la propria memoria locale e che comunica con gli altri tramite **scambio di messaggi**. I processi di invio e ricezione devono trasferire i dati dalla memoria locale di uno alla memoria locale dell'altro. Questo modello può essere più flessibile nei sistemi distribuiti, ma può comportare un maggiore overhead di comunicazione.

La distinzione chiave risiede nel modo in cui i processi interagiscono e condividono i dati: la memoria condivisa si basa su uno spazio di indirizzi comune, mentre il passaggio di messaggi si basa sulla comunicazione tramite scambio esplicito di messaggi.

![[|469](_page_3_Figure_3.jpeg)]

## Primitive di Passaggio di Messaggi

Il modello di passaggio di messaggi consiste principalmente di due primitive principali:

*   `Send(destinazione, messaggio)`: un processo invia un *messaggio* a un altro processo identificato come *destinazione*.
*   `Receive(sorgente, messaggio)`: un processo riceve un *messaggio* da un altro processo identificato come *sorgente*.

Il processo di invio crea il messaggio contenente i dati da condividere con il processo ricevente e lo trasmette alla rete eseguendo un'operazione di *invio*. Il processo ricevente deve essere consapevole di aspettarsi dati e indica la sua disponibilità a ricevere un messaggio eseguendo un'operazione di *ricezione*.

### Implementazioni del Passaggio di Messaggi

L'implementazione pratica delle operazioni di invio e ricezione determina varie implementazioni del passaggio di messaggi, che possono essere categorizzate come:

*   *dirette* o *indirette*
*   *bufferizzate* o *non bufferizzate*
*   *bloccanti* o *non bloccanti*

![[|335](_page_5_Figure_6.jpeg)]

## Passaggio di Messaggi Diretto e Indiretto

Nel **passaggio di messaggi diretto**, esiste un collegamento diretto tra due processi per lo scambio di dati, dove l'identità del ricevente è nota e i messaggi vengono inviati direttamente. Questo approccio manca di modularità, poiché la modifica dell'identità di un processo richiede l'aggiornamento di tutti i mittenti e i riceventi ad esso collegati.

Nel **passaggio di messaggi indiretto**, vengono utilizzate mailbox o porte per la consegna dei messaggi e possono essere associate a un processo ricevente. A differenza del passaggio di messaggi diretto, la stessa porta può essere riassegnata a un altro processo in seguito. In questo approccio, il mittente non è a conoscenza di quale processo riceverà il suo messaggio. Inoltre, più processi possono inviare messaggi alla stessa porta, abilitando collegamenti multi-processo e maggiore flessibilità.

## Passaggio di Messaggi Bloccante e Non Bloccante

Esiste una distinzione significativa tra il passaggio di messaggi **bloccante** (o *sincrono*) e **non bloccante** (o *asincrono*).

### Messaggi di Blocco e Non di Blocco

In un'operazione di *invio bloccante* (blocking send), il mittente deve attendere che il ricevente confermi la ricezione del messaggio. In un'operazione di *ricezione bloccante* (blocking receive), il ricevente attende di ricevere un messaggio prima di procedere con le sue attività. Le operazioni bloccanti sono spesso definite **sincronizzate** (synchronous) perché sia il mittente che il ricevente sono sincronizzati durante la comunicazione.


### Invio e Ricezione di Messaggi Bloccanti e Non Bloccanti

In un'operazione di *invio non bloccante* (non-blocking send), il mittente continua le sue operazioni senza attendere l'avviso di ricezione. Tuttavia, il mittente si aspetta una conferma dal ricevente nel caso in cui l'invio fallisca.

In un'operazione di *ricezione non bloccante* (non-blocking receive), il ricevente può accettare sia un messaggio valido che un messaggio nullo, ponendo la sfida critica di come il processo di ricezione determina che un messaggio è arrivato in una primitiva di ricezione non bloccante. Se la trasmissione continua a fallire, il ricevente potrebbe attendere indefinitamente.

Sono raccomandate tre combinazioni fondamentali:

*   Invio bloccante e ricezione bloccante, chiamata comunicazione *rendez-vous*.
*   Invio non bloccante e ricezione non bloccante.
*   Invio non bloccante e ricezione bloccante, che è la più utilizzata.


## Buffering 

Un modo per distinguere tra i modelli di passaggio dei messaggi è considerare la dimensione della coda del ricevitore. Esistono tre alternative:

*   *Coda a capacità zero* (o *nessuna coda*): richiede un rendez-vous poiché il mittente deve attendere che il ricevente sia pronto a ricevere il messaggio.
*   *Coda limitata*: la coda è limitata a *n* messaggi o byte, causando il blocco del mittente quando è piena.
*   *Coda illimitata*: i mittenti procedono senza attendere, ponendo potenziali rischi a causa delle risorse fisiche limitate.


## Comunicazione di Gruppo

Nelle applicazioni distribuite parallele, un sistema di passaggio dei messaggi potrebbe aver bisogno di primitive di comunicazione di gruppo per prestazioni migliorate e semplicità di sviluppo. La comunicazione di gruppo può assumere tre forme:

### Comunicazione Uno-a-Molti (Multicast)

Nella comunicazione **uno-a-molti**, un singolo mittente trasmette un messaggio a più ricevitori, chiamato anche comunicazione *multicast*. I processi di ricezione dei messaggi stabiliscono un gruppo, che può essere *chiuso* o *aperto*. In un **gruppo chiuso**, solo i membri possono inviare messaggi internamente. In un **gruppo aperto**, qualsiasi processo nel sistema può inviare messaggi all'intero gruppo. Un caso speciale della comunicazione uno-a-molti è la comunicazione **broadcast**, in cui un messaggio viene inviato a tutti i processori connessi a una rete.


### Comunicazione Molti-a-Uno

Nella comunicazione **molti-a-uno**, più mittenti trasmettono messaggi a un singolo ricevitore. Il singolo ricevitore può essere:

*   *selettivo*, identificando un mittente specifico per lo scambio di messaggi.
*   o *non selettivo*, rispondendo a qualsiasi mittente da un set predefinito.

Il non determinismo pone una sfida significativa nella comunicazione molti-a-uno, poiché rimane incerto quale/i membro/i del gruppo avrà/avranno per prima/e la propria informazione disponibile.



### Comunicazione Molti-a-Molti

Nella comunicazione **molti-a-molti**, più mittenti possono trasmettere messaggi a più ricevitori. Questo schema di comunicazione è flessibile e consente interazioni complesse nei sistemi distribuiti, risultando particolarmente utile negli scenari in cui è necessaria una comunicazione e un coordinamento decentralizzati tra più entità. La consegna ordinata dei messaggi è cruciale nelle comunicazioni molti-a-molti, garantendo che tutti i messaggi raggiungano i destinatari in un ordine accettabile per le applicazioni coinvolte.

