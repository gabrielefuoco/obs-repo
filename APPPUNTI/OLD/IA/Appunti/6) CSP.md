
Quando si risolvono problemi di **soddisfacimento di vincoli (CSP)**, i vincoli possono essere rappresentati in modo **esplicito** o **implicito**. 

- Con una **rappresentazione esplicita**, i vincoli sono trattati come un database, dove per ogni vincolo si definisce un insieme di variabili e un insieme di valori validi. Ad esempio, se abbiamo tre vincoli $C1(WA,NT), C2(WA,SA), C3(NT,SA)$, questi rappresentano restrizioni sui valori ammissibili tra le variabili $WA, NT, SA$.
- Una soluzione al CSP è una mappatura delle variabili ai loro domini, tale che i valori assegnati rispettino tutti i vincoli definiti nel DB.

Nel mondo dei database, risolvere un CSP equivale a risolvere **query congiuntive** (o **select-project-join** queries). 
##### La complessità può variare:

1. **Data complexity**: Si fissa la query e si assume che il database sia molto grande.
2. **Query complexity**: Il database è piccolo, ma il numero di vincoli o strutture può variare.
3. **Combined complexity**: Si analizzano entrambi i casi senza fare assunzioni, ed è il più complesso da gestire.

### Constraint Graph 

Un **Constraint Graph** è una rappresentazione grafica dei vincoli di un problema di soddisfacimento di vincoli (CSP). In questo grafo:

- I nodi (**V**) rappresentano le variabili.
- Gli archi (**E**) collegano le variabili che condividono un vincolo.

Quindi, se due variabili $V_i$ e $V_j$ appaiono nello stesso vincolo $C_i$, c'è un arco tra di loro. Questo tipo di grafo funziona bene quando i vincoli coinvolgono solo **due variabili** (vincoli binari).

Ad esempio, se abbiamo vincoli come:

- $C1(V1, V2, V3)$
- $C2(V1, V3, V4)$
- $C3(V4, V2)$

il grafo collegherà tutte le variabili coinvolte in ciascun vincolo, ma con archi solo tra le coppie di variabili.

Tuttavia, se i vincoli coinvolgono **più di due variabili**, il **Constraint Graph** diventa meno efficiente. In questi casi, si utilizza un **ipergrafo**, dove un **iper-arco** collega **più di due nodi**. L'ipergrafo è più informativo perché mostra meglio come le variabili sono collegate dai vincoli complessi.

## Problema dell' Omomorfismo

Il **Problema dell'Omomorfismo** (Hom) consiste nel decidere se esiste una corrispondenza tra due strutture relazionali. Una **struttura relazionale** è costituita da:

1. **Simboli di relazione**: $r_1, r_2, \dots, r_n$, ciascuno con una propria **arità** (cioè il numero di elementi che partecipano alla relazione).
2. **Database (DB)**: un insieme di tuple che rappresentano i fatti o i dati per ogni relazione.

Per ogni simbolo di relazione $r_i$, esiste una relazione associata $r_i^{DB}$ che contiene le tuple nel database. Ad esempio, se abbiamo lo schema di relazione $r_i(A, B, C)$ ciò indica che la relazione $r_i$ coinvolge tre variabili. Le tuple corrispondenti possono essere rappresentate come $<1, 2, 3>$, $<1, 4, 5>$, e così via.

Un **omomorfismo** tra due strutture relazionali è una **mappatura** che preserva le relazioni: se esiste una certa relazione tra un insieme di elementi nella prima struttura, la stessa relazione deve esistere tra gli elementi corrispondenti nella seconda struttura.

Il problema consiste nel verificare se è possibile "trasferire" la struttura dei dati di un database a un altro in modo che le relazioni tra gli elementi siano mantenute.

## Core

Il **core** di un CSP (Constraint Satisfaction Problem) è la *versione più semplice* e ridotta del problema originale, ottenuta attraverso un processo di semplificazione. 

Quando troviamo una struttura ridondante o relazioni tra variabili che possono essere "compresse", possiamo ridurre il numero di vincoli senza perdere informazioni importanti.

Per esempio, se in un database ho le relazioni *a -> b* e *b -> d*, posso ridurre queste a *a -> d*, semplificando il problema. Se applico una mappatura (*endomorfismo*) che collega due variabili, posso ottenere un sottoproblema più piccolo e più semplice da risolvere.

Il **core** è quindi questa struttura ridotta: il più piccolo sottoproblema che mantiene la stessa essenza del problema originale. Un core esiste quando possiamo trovare un **endomorfismo** (una funzione che mappa gli elementi di una struttura su se stessa) che semplifica il problema senza perdere informazioni essenziali

In altre parole:
- Il **core** è la versione più piccola di un CSP che contiene tutte le informazioni necessarie per risolvere il problema originale.
- È unico a meno di **isomorfismi** (cioè, possono esistere più core equivalenti, ma sono tutti strutturalmente uguali a meno delle etichette delle variabili). 

La soluzione del problema semplificato può essere usata per risolvere il problema originale, semplificando il processo di risoluzione.

## Backtracking Search

Ѐ l'algoritmo di ricerca non informata usato solitamente per risolvere problemi di CSP.
- **Algoritmo Base**: 
 - Sceglie una variabile non assegnata.
 - Prova tutti i valori del dominio di quella variabile.
 - Per ogni valore, tenta di estendere l'assegnamento con una chiamata ricorsiva.
 - Se la chiamata ha successo, restituisce la soluzione; altrimenti, riporta l'assegnamento allo stato precedente e prova un nuovo valore.
 - Se nessun valore funziona, restituisce fallimento.

- **Miglioramenti**:
 - **Forward Checking**: 
 - Filtra i valori dei domini delle variabili non assegnate, eliminando i valori che violano i vincoli con gli assegnamenti futuri.
 - Garantisce che i valori assegnati siano consistenti con i vincoli.

 - **Arc Consistency**: 
 - Verifica la consistenza di tutti i vincoli tra coppie di variabili.
 - Un arco $x \to y$ è consistente se, per ogni valore di $x$, esiste un valore di $y$ che soddisfa il vincolo.
 - Se un valore $x$ non ha un corrispondente $y$ valido, viene eliminato dal dominio di $x$.

- **Propagazione**:
 - Ottimizza il backtracking forzando l'arc consistency: quando un vincolo viene verificato, la propagazione si estende agli altri vincoli collegati, migliorando l'efficienza della ricerca.

## Tipi di consistenza

Un CSP è detto **globalmente consistente** se, per ogni sottoinsieme di variabili del problema, qualsiasi assegnazione parziale coerente di valori a queste variabili può essere estesa a un'assegnazione coerente dell'intero insieme di variabili.

In altre parole:

- Non ci sono conflitti locali che potrebbero impedire un'estensione valida delle assegnazioni parziali.
- Ogni assegnazione parziale che soddisfa i vincoli può essere completata.
### Consistenza locale

- **1-Consistency (Node-Consistency)**:
 - Consistenza a livello di singola variabile.
 - Ogni valore nel dominio di una variabile soddisfa i vincoli unari associati a quella variabile.

- **2-Consistency (Arc-Consistency)**:
 - Ogni assegnamento consistente a una variabile può essere esteso all'altra variabile, sullo stesso arco.
	 - Per due variabili legate da un vincolo (un "arco" tra di loro), se assegni un valore a una variabile, esiste almeno un valore nell'altra variabile che soddisfa il vincolo. In altre parole, un assegnamento è consistente se, dato un valore per la prima variabile, puoi sempre trovare un valore per la seconda variabile che rispetti il vincolo che le collega.
 - Se la struttura del problema è **aciclica**, forzare l'arc-consistency garantisce l'esistenza di una soluzione.
 - **Costo computazionale:** O(n²), dove *n* è il numero di variabili.

- **K-Consistency**:
 - Estensione della 2-consistency a k variabili.
 - Ogni assegnamento consistente a *k - 1* variabili può essere esteso alla *k-esima* variabile.
 - Utilizzato per risolvere problemi in maniera efficiente.
 - **Costo computazionale:** $O(n · d^k)$, dove *n* è il numero di variabili e *d* è la dimensione massima dei domini.
 - Se *k* è fissato, il costo è polinomiale.

## Euristiche per la Risoluzione dei CSP

Le due principali euristiche per l'ordinamento delle variabili sono Minimum Remaining Values e Least Constraining Value:

- **Minimum Remaining Values (MRV)**:
- Si sceglie la variabile con il minor numero di valori validi rimasti nel dominio.
- Questo approccio permette di fallire velocemente (fail-fast) se non ci sono valori consistenti, accelerando la ricerca.

- **Least Constraining Value**:
- Si seleziona il valore che restringe meno i domini delle variabili rimanenti, riducendo il numero di vincoli che devono essere soddisfatti.
#### Tecniche di Risoluzione con Assegnamento Completo

- Invece di partire da un assegnamento vuoto, si parte da un **assegnamento completo** (anche se non consistente) e si cercano correzioni:
 - Seleziona una variabile in conflitto.
	- Modifica il valore della variabile cercando di minimizzare i conflitti.

- **Euristiche per la correzione**:
 - **Minimo Conflitto**: seleziona il valore che viola il minor numero di vincoli.
 - **Hill Climbing** o **Simulated Annealing** possono essere utilizzati per trovare la soluzione ottimale partendo da assegnamenti completi e migliorando progressivamente.

##### Problema del Punto Critico

- In problemi bilanciati (stesso numero di variabili e vincoli), può verificarsi una **transizione di fase** chiamata **punto critico**, in cui gli algoritmi euristici potrebbero fallire.
- Questo è un problema generale per molti algoritmi euristici, non solo per quelli basati su minimo conflitto.

## Omomorfismo su strutture acicliche

Il problema dell'omomorfismo su strutture acicliche si riferisce al decidere se esiste un **omomorfismo**, ovvero una mappatura che preserva la struttura di due strutture relazionali, chiamate $A$ e $B$, che rappresentano vincoli e relazioni di un problema CSP.

La complessità di risolvere questo problema su strutture **acicliche** è $O(||A|| · ||B|| · log ||B||)$, dove $\|A\|$ e $\|B\|$ rappresentano la dimensione delle strutture.

#### Procedura(algoritmo di Yannakakis)

Per risolvere il problema su strutture acicliche, si utilizza un **algoritmo di programmazione dinamica**, in particolare l'algoritmo di **Yannakakis**. 
L'algoritmo di Yannakakis trasforma il problema su strutture acicliche in una serie di operazioni efficienti che permettono di risolvere il problema senza tornare indietro e di rappresentare le soluzioni in modo compatto.

1. **Filtraggio verso l'alto**: 
 - Si parte da una relazione inferiore (chiamiamola **t**) e si verifica che tutte le tuple di una relazione superiore (chiamiamola **r**) siano compatibili con quelle di **t**.
 - Se alcune tuple di **r** non sono compatibili con le tuple di **t**, vengono eliminate.
 - Una volta fatto questo, si ripete lo stesso processo con la relazione successiva in basso, procedendo verso l’alto fino a completare il filtraggio.

2. **Verifica dell'esistenza di una soluzione**: 
 - Se dopo questo filtraggio nessuna relazione si svuota (cioè non restano vuote), allora **esiste una soluzione** al problema.
 - A questo punto, il problema decisionale è già risolto, ossia possiamo dire se una soluzione esiste o meno.

3. **Filtraggio verso il basso**: 
 - Si passa ora a un filtraggio discendente, partendo dall'ultima relazione filtrata e assicurandosi che tutte le tuple rimaste appartengano effettivamente a una soluzione possibile. Questo garantisce che le soluzioni siano valide.

4. **Costo del filtraggio**: 
 - Il filtraggio viene eseguito ordinando i valori delle relazioni e poi effettuando una "merge" tra le relazioni. Una singola operazione costa **n · log n**, dove $n$ è il numero di elementi nelle relazioni.
 - Il numero totale di operazioni è proporzionale al numero dei vincoli.

### Soluzione Backtrack-free

- Una volta effettuato il filtraggio, si ottiene una **struttura filtrata** che consente di trovare una soluzione senza la necessità di eseguire il backtracking (ovvero, senza dover tornare indietro per provare nuove soluzioni). Questo riduce enormemente la complessità della ricerca.
- Anche se il numero di soluzioni può essere esponenziale, l'algoritmo offre un modo polinomiale per:
	- Decidere se esiste una soluzione.
	- Calcolare una soluzione.
	- Trovare tutte le soluzioni.

## Strutture quasi ad Albero

Il concetto di **Strutture quasi ad Albero di CSP** riguarda la trasformazione di un grafo che rappresenta un CSP in una forma *aciclica*, per sfruttare algoritmi efficienti applicabili ai grafi senza cicli. Questo avviene attraverso la rimozione di nodi dal grafo, rendendolo aciclico.

### Procedura e Concetti chiave

1. **Cut Set (Feedback Vertex Number)**: 
 - Il **Cut Set** è il numero minimo di nodi da rimuovere per rendere un grafo aciclico.

2. **Risoluzione del problema aciclico**:
 - Una volta eliminato il nodo, si sceglie un valore specifico per quella variabile dal suo dominio.
	 - Fissare un valore riduce la complessità perché il problema diventa meno "flessibile".
 - Il problema aciclico risultante può essere risolto utilizzando metodi già noti per problemi aciclici, come quelli basati sulla propagazione dei vincoli (arc consistency).
 - Se si trova una soluzione compatibile con la variabile fissata, il problema è risolto.
 - Se invece non si trova una soluzione, si prova a cambiare il valore della variabile e ripetere il processo.

3. **Costo computazionale**:
 - Nel caso peggiore, il costo di questa procedura dipende dal **dominio** della variabile rimossa, moltiplicato per il costo della risoluzione del problema aciclico risultante. 
 - Il costo è approssimato come: 
 **dominio(var) × costo del problema aciclico**.

 - Se il numero minimo di nodi da rimuovere è $c$, allora il problema può essere risolto con complessità:
 $$ 
 O(d^c \cdot (n - c) \cdot d^2)
 $$
 Dove:
 - $d$ è la dimensione massima del dominio delle variabili.
 - $n$ è il numero di variabili.

4. **Ipergrafi e iperarchi**:
 - Un'ulteriore **ottimizzazione** è possibile utilizzando un **approccio basato sugli ipergrafi**, che permette di fissare interi **vincoli** (un **gruppo** di variabili) anziché singole variabili. 
 - Questo può ridurre ulteriormente la complessità in certi casi, permettendo una risoluzione più efficiente del problema.

## Join Tree 

 Un **Join-Tree** è una struttura che organizza gli iperarchi in modo tale che ogni variabile si propaghi correttamente lungo l'albero e non venga "persa" o riutilizzata in modo errato. Questo permette una corretta gestione dei vincoli tra le variabili.

* Sia l'**ipergrafo H** una struttura che rappresenta vincoli complessi tra gruppi di variabili, con ogni iper-arco che collega più di due variabili.
	- **H** è un **ipergrafo** (con variabili $\{A, B, C, D, E\}$ e iperarchi $\{A, B, C\}, \{A, B, D\}, \{D, E\})$.Gli **iperarchi** rappresentano insiemi di variabili che sono collegate tra loro da vincoli.
	- **V'** è l'insieme degli **iperarchi** di $H$ (i gruppi di variabili collegati dai vincoli).

### Regole chiave:

Un **Join-Tree T**, è una struttura che connette degli iperarchi in modo che la propagazione delle variabili segua determinate regole:

1. Se due iperarchi **p** e **q** condividono delle variabili comuni, quelle variabili devono apparire in tutti i vertici lungo il percorso che connette **p** e **q** nell'albero **T**. Questo garantisce che le informazioni sulle variabili comuni si propaghino lungo l'albero in modo corretto.
2. Una variabile che scompare in un certo punto dell'albero non può più riapparire successivamente nel percorso: una volta che la sua informazione è stata utilizzata o propagata, non la si ritrova in altre parti.

## Ipergrafi Aciclici: 

$$\text{H è un ipergrafo aciclico}\iff\text{H ha un join-tree}$$
La definizione di un ipergrafo aciclico è più potente di quella di un grafo aciclico. Mentre un grafo aciclico è un grafo senza cicli, in un ipergrafo avere dei cicli potrebbe non apportare problemi, quindi in alcuni casi si può considerare aciclico. 
Decidere se un ipergrafo è aciclico è un problema log-space-completo ed è trattabile in tempo lineare. La stessa complessità per i grafi

## Tree Decomposition 

La **tree decomposition** è un metodo che permette di semplificare un problema complesso rappresentato da un grafo, suddividendolo in sottoproblemi aciclici più facili da risolvere.

- **Obiettivo**: Prendere un grafo con cicli e trasformarlo in una struttura che può essere trattata come un **albero**, eliminando i cicli.
- **Metodo**: Raggruppiamo le variabili del problema in insiemi. Ogni insieme (o "nodo") contiene **k + 1** variabili, e questi insiemi formano gli **iperarchi** di un nuovo ipergrafo aciclico.
- **Tree decomposition**: Viene costruito un **albero** T dove ogni nodo contiene un insieme di variabili del grafo originale. Ciascun nodo è etichettato con un insieme di variabili tramite una funzione $χ$.

### Proprietà:

1. **Copertura degli archi**: Per ogni arco del grafo originale, esiste almeno un nodo nell'albero $T$ che contiene entrambe le variabili dell'arco.
2. **Proprietà di connessione**: Se una variabile $x$ appare in più nodi dell'albero, allora deve comparire in tutti i nodi lungo il percorso che li collega.

### Tree-width:

- La **width** (larghezza) di una decomposizione è la dimensione massima dei nodi nell'albero meno uno.
- La **tree-width** del grafo è la larghezza minima tra tutte le possibili decomposizioni.
---
- **Teorema**
	- Sia k una costante. Calcolare una tree-decomposition di width minore o uguale di k richiede tempo lineare. Se k non è fissato (è parte dell’input) il problema è NP-Hard.

- **Tree Decomposition e CSP**:
 - La tree decomposition è particolarmente utile per risolvere problemi di Constraint Satisfaction (CSP), poichè una struttura aciclica facilita la risoluzione dei CSP.
- **Processo di Risoluzione**:
 - I nodi della tree decomposition possono essere considerati come vincoli unici.
 - Risolvendo i sottoproblemi per ciascun nodi e combinando i risultati, si ottiene un problema aciclico.
- **Risoluzione tramite Tree Decomposition**:
 - Aggiungere vincoli non restrittivi non altera il problema originale.
 - Con una tree decomposition, si ottiene la local consistency (equivalente ad arc consistency).
 - La local consistency implica la global consistency su istanze acicliche.
- **Complessità**:
 - Costruzione dei nuovi vincoli: $O(n · D^{(k+1)}),$ dove D è la dimensione del dominio e k è la width.
 - Risoluzione del problema con nuova struttura:
 - Costo per ogni vincolo: $O(D^{(k+1)} \cdot log D^{(k+1)})$
 - Costo totale: $O((D^{(k+1)} + log D^{(k+1)}) · (n + m))$
- **Feasibility**:
 - Ogni problema CSP con tree-width ≤ k è risolvibile in tempo polinomiale, dove k è fissato.

## Teorema di Grohe

Il **Teorema di Grohe** afferma che risolvere problemi CSP appartenenti a S è **fattibile in tempo polinomiale** (P-TIME) **se e solo se** il **core** delle strutture in S ha una **treewidth (tw)** fissata (limitata da un valore $k$).

#### Contesto del Teorema:

- **Strutture relazionali**: Abbiamo una classe $S$ di strutture relazionali (insiemi di vincoli) con arità fissata (cioè ogni vincolo ha un numero fisso di variabili).
- **CSP (S, -)**: Si riferisce a problemi CSP dove i vincoli appartengono a $S$, ma non c'è alcuna restrizione sul database (DB) che contiene le possibili assegnazioni dei valori.

1. **Treewidth (tw)**: Misura la "complessità" ciclica di un grafo. Più piccola è la treewidth, più facile è risolvere il problema CSP associato a quel grafo. Se la treewidth è fissata, il problema si può risolvere in modo efficiente($≤ k \to \text{ tempo polinomiale}$ ).
2. **Core**: È la parte "essenziale" di una struttura CSP, ovvero la versione semplificata del problema che mantiene la stessa soluzione ma con meno ridondanze.
3. **Se la treewidth del core è limitata (≤ k)**: Vuol dire che la struttura è abbastanza semplice da essere risolta in tempo polinomiale.

#### Algoritmo risolutivo:

1. Per ogni gruppo di $k+1$ variabili, crea nuovi vincoli sulle loro possibili combinazioni di valori.
2. Filtra i valori non compatibili tra i vincoli, eliminandoli man mano, fino a quando non è possibile fare ulteriori riduzioni.
3. Se nessuna relazione è vuota, significa che abbiamo trovato una soluzione.

## Dimostrazione Teorema 

La dimostrazione prova che i problemi CSP con core di treewidth limitata sono risolvibili in tempo polinomiale. Data una istanza CSP:

1. **Riduzione al Core:** Si trova il core dell'istanza, preservando l'equivalenza.

2. **Vincoli Aggiuntivi:** Poiché la treewidth del core è limitata da *k*, si aggiungono nuovi vincoli che impongono la consistenza su ogni sottoinsieme di *k*+1 variabili. Il numero di questi vincoli è polinomiale nel numero di variabili. Ogni nuovo vincolo ha dimensione polinomiale nel dominio.

3. **Consistenza Locale:** Si applica un algoritmo di consistenza locale (es. arc consistency, k-consistency) per rimuovere assegnazioni inconsistenti. Questo passo è polinomiale nel numero di vincoli e nella dimensione dei vincoli.

4. **Verifica:** Se dopo l'applicazione della consistenza locale rimangono assegnazioni, si ha una soluzione; altrimenti no.

La complessità complessiva è polinomiale perché ogni passo è polinomiale e il numero di nuovi vincoli è polinomiale grazie alla limitazione della treewidth. La dimensione dei nuovi vincoli è anch'essa polinomiale.

