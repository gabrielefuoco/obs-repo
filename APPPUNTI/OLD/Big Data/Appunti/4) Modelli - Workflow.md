
Un **workflow** è una serie di attività, eventi o task che devono essere completati per raggiungere un obiettivo e/o un risultato. La **Workflow Management Coalition** (WMC) definisce un workflow come *"l'automazione di un processo aziendale, in tutto o in parte, durante il quale documenti, informazioni o task vengono passati da un partecipante all'altro per l'azione, secondo un insieme di regole procedurali"*. I workflow sono diventati un prezioso modello di programmazione, consentendo a scienziati e ingegneri di creare programmi complessi per l'elaborazione di grandi repository di dati su piattaforme di calcolo distribuite, combinando analisi dati, calcolo scientifico e metodi di simulazione complessi.

Un **processo** rappresenta un insieme di task connessi tra loro con l'obiettivo di produrre un prodotto, calcolare un risultato o fornire un servizio. Un **task** (o *attività*) è un'unità di lavoro che rappresenta un singolo passo logico nell'intero processo. I workflow, come modello di programmazione, rappresentano pattern ben definiti e possibilmente ripetibili o raggruppamenti sistematici di attività finalizzati al raggiungimento di una certa trasformazione dei dati.

I workflow adottano un approccio dichiarativo per esprimere la logica di alto livello di molti tipi di applicazioni, occultando i dettagli di basso livello non essenziali per la progettazione dell'applicazione. Un vantaggio significativo dei workflow è la possibilità di essere memorizzati e recuperati, facilitando la modifica e/o la riesecuzione. Ciò consente agli utenti di progettare e riutilizzare pattern comuni in contesti multipli. Un **sistema di gestione dei workflow** (WMS) facilita la definizione, lo sviluppo e l'esecuzione dei processi, con il coordinamento delle attività (o attuazione) che gioca un ruolo fondamentale durante l'esecuzione del workflow.

Un **workflow** è strutturato come un grafo costituito da un insieme finito di archi e vertici:

* I **vertici** rappresentano task, attività o fasi specifiche all'interno del processo complessivo.
* Gli **archi** rappresentano il flusso o la sequenza dei task, indicando l'ordine in cui i task devono essere eseguiti. Ogni arco è diretto da un vertice a un altro.

I workflow possono essere implementati come programmi software, utilizzando linguaggi di programmazione, librerie o sistemi che consentono l'espressione dei passaggi fondamentali del workflow. Questi strumenti includono anche meccanismi per l'orchestrazione dell'esecuzione del workflow.

## Pattern dei workflow

I task dei workflow possono essere combinati in vari modi, consentendo ai progettisti di soddisfare le esigenze di una vasta gamma di scenari applicativi attraverso l'utilizzo di costrutti ricorrenti e riutilizzabili (sequenziali e paralleli). I **pattern dei workflow** forniscono un modo standardizzato di organizzare e orchestrare i task all'interno di un processo. I principali pattern dei workflow sono:

* **Sequenza**
* **Ramificazione**
* **Sincronizzazione**
* **Ripetizione**

![[_page_5_Figure_9.jpeg|299]]]

### Pattern di sequenza

Il **pattern di sequenza** indica una sequenza di task che devono essere completati in un ordine specifico. Questi task sono collegati da archi diretti che indicano la direzione del flusso di controllo, specificando l'ordine sequenziale in cui i task vengono eseguiti.

![[_page_6_Figure_4.jpeg|312]]]

### Pattern di ramificazione

Il **pattern di ramificazione** descrive situazioni in cui una diramazione in un workflow è suddivisa in due o più diramazioni diverse in base a determinate condizioni, come il risultato di task precedenti, valori di dati, input dell'utente o qualsiasi altro criterio rilevante per il workflow. Questo pattern consente al workflow di adattarsi dinamicamente a diversi scenari applicativi.

![[_page_7_Figure_4.jpeg|317]]

#### Varianti di ramificazione

Si possono identificare tre varianti di ramificazione:

* **AND-split:** una diramazione si suddivide in flussi di esecuzione concorrenti in ogni diramazione successiva.
* **XOR-split:** una diramazione è diretta in una sola delle diramazioni generate dal costrutto di split. La diramazione successiva da perseguire si basa sulla valutazione delle condizioni associate a ciascuno degli archi in uscita.
* **OR-split:** una diramazione si suddivide in flussi di esecuzione concorrenti in una o più diramazioni successive in base alla valutazione delle condizioni associate a ciascuno degli archi in uscita. È possibile scegliere più rami.

### Pattern di sincronizzazione

Il **pattern di sincronizzazione** descrive situazioni in cui flussi di controllo multipli su uno o più rami devono essere uniti in un singolo ramo. Tali scenari sono comuni nei flussi di lavoro del mondo reale, dove l'esecuzione di un compito specifico deve attendere il completamento di uno o più compiti precedenti.

![[_page_9_Figure_4.jpeg|313]]]

#### Varianti di sincronizzazione

In pratica possono verificarsi tre varianti del pattern di sincronizzazione:

* **AND-join**, dove tutti i rami in ingresso devono essere completati prima di procedere al compito successivo.
* **XOR-join**, dove solo uno dei rami in ingresso deve essere completato prima di procedere.
* **OR-join**, dove almeno uno dei rami in ingresso deve essere completato prima che il controllo venga passato al compito successivo.

### Pattern di ripetizione

I **pattern di ripetizione** descrivono vari modi per specificare la ripetizione:

* **Ciclo Arbitrario:** indica uno o più task ripetuti all'interno di un workflow, equivalente all'istruzione *goto*.
* **Ciclo Strutturato:** descrive un insieme specifico di task ripetuti con una condizione di terminazione specifica, valutata prima (*while..do*) o dopo ogni iterazione (*repeat…until*).
* **Ricorsione:** utilizzata per descrivere situazioni in cui un task specifico viene ripetuto tramite auto-invocazione.

## Grafi Aciclici Diretti (DAG)

Un **Grafo Aciclico Diretto (DAG)** è un workflow che è sia:

* **Diretto:** se esistono più task, ognuno deve avere almeno un task precedente o successivo, o entrambi. Tuttavia, alcuni DAG hanno più task paralleli, il che implica che non ci sono dipendenze.
* **Aciclico:** i task non possono generare dati che si riferiscono a se stessi, potenzialmente causando un loop infinito. Ciò significa che i DAG non hanno cicli.

I DAG sono la struttura di programmazione più comunemente utilizzata nella gestione dei workflow e si sono dimostrati estremamente utili in una varietà di framework big data, incluso Apache Spark.

Il paradigma DAG è efficace per modellare processi complessi di analisi dei dati, come le applicazioni di data mining, che possono essere eseguite in modo efficiente su sistemi di calcolo distribuiti, come una piattaforma cloud. I DAG possono facilmente modellare molti tipi diversi di applicazioni, in cui l'input, l'output e i task di un'applicazione dipendono dai task di un'altra.
![[Pasted image 20250223162720.png|438]]

I DAG possono avere due tipi di dipendenze:

* **Dipendenze dati:** l'output di un task funge da input per i task successivi.
* **Dipendenze di controllo:** determinati task devono essere completati prima di iniziare un altro task o un insieme di task.

I task di un'applicazione DAG e le loro dipendenze possono essere definite:

* **Esplicitamente**, quando le dipendenze tra i task sono definite tramite istruzioni esplicite (es., T2 dipende da T1);
* **Implicitamente**, quando il sistema deduce automaticamente le dipendenze tra i task (es., T2 legge l'input O1, che è un output di T1) analizzando le loro relazioni input-output.

Il modello DAG è una generalizzazione rigorosa del modello MapReduce, poiché i calcoli complessi richiedono più fasi *map* e *reduce*, che si traducono in un DAG di operazioni. Il modello DAG offre maggiore flessibilità rispetto a MapReduce e consente una migliore ottimizzazione globale abilitando la riorganizzazione e la combinazione di operatori ovunque possibile. Ad esempio, se consideriamo due operazioni, come *map* e *filter*, un'ottimizzazione potrebbe essere quella di eseguirle in ordine inverso perché il filtro potrebbe ridurre il numero di record che subiscono operazioni di mappatura.

## Grafi Ciclici Diretti (DCG)

I **Grafi Ciclici Diretti (DCG)** sono un modello di workflow più complesso, dove i cicli rappresentano una qualche forma di meccanismo di controllo del loop o dell'iterazione implicito o esplicito. In questo caso, il grafo del workflow descrive frequentemente una rete di task, dove:

* i *nodi* rappresentano servizi, istanze di componenti software o oggetti di controllo più astratti.
* gli *edge* del grafo rappresentano messaggi, flussi di dati o pipe che permettono ai servizi e ai componenti di scambiare lavoro o informazioni.

