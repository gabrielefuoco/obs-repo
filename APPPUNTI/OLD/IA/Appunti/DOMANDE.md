## Ricerca:

### Definizione del problema di ricerca:

- Quali sono i quattro elementi che definiscono un problema di ricerca?
- Cosa si intende per "spazio degli stati"?
- Come si definisce una soluzione in un problema di ricerca?
- Qual è la differenza tra "search state" e "world state"?

### Rappresentazione dello spazio degli stati:

- Quali sono le due principali rappresentazioni grafiche dello spazio degli stati?
- Cosa rappresentano i nodi e gli archi in un "state space graph"?
- Cosa rappresenta la radice in un "search tree"?

### Algoritmo di ricerca:

- Qual è l'obiettivo principale della ricerca?
- Perché è importante evitare i loop durante la ricerca?

### Ricerca ad albero:

- Quali sono i cinque passaggi principali della ricerca ad albero?
- Cosa si intende per "frontiera" nella ricerca ad albero?
- Quali sono i tre modi principali per implementare la frontiera?
- Qual è la differenza tra una coda FIFO e una coda LIFO?

### Strategie di ricerca non informata:

- Cosa caratterizza le strategie di ricerca non informata?
- Quali sono i tre criteri per valutare le prestazioni delle strategie di ricerca non informata?
- Cosa si intende per "completezza" di un algoritmo di ricerca?
- Cosa si intende per "ottimalità" di un algoritmo di ricerca?
- Qual è la complessità spaziale e temporale della ricerca in ampiezza?
- Qual è la complessità spaziale e temporale della ricerca in profondità?
- Come funziona l'algoritmo di Iterative Deepening?
- Qual è la complessità spaziale e temporale di Bidirectional Best-First?

### Strategie di ricerca informata o euristica:

- Cosa caratterizza le strategie di ricerca informata?
- Cosa rappresenta la funzione h(n)?
- Come si combina il costo del passato con il costo stimato del futuro nelle strategie di ricerca informata?
- Qual è la funzione di valutazione utilizzata nelle strategie di ricerca informata?
- Qual è l'obiettivo principale delle strategie di ricerca informata?
- Cosa si intende per "rilassamento" del problema?
- Come funziona l'algoritmo di Uniform Cost Search?
- Qual è la funzione di costo utilizzata in Uniform Cost Search?
- Quali sono le caratteristiche di Uniform Cost Search?
- Come funziona l'algoritmo di Greedy Best-First Search?
- Qual è la funzione di costo utilizzata in Greedy Best-First Search?
- Quali sono le caratteristiche di Greedy Best-First Search?

### Proprietà delle euristiche:

- Quali sono le due principali proprietà delle euristiche?
- Cosa si intende per "ammissibilità" di un'euristica?
- Cosa si intende per "consistenza" di un'euristica?
- Qual è la relazione tra ammissibilità e consistenza?
- Perché è importante avere valutazioni quantitative delle euristiche?

### A* Tree Search:

- Qual è la funzione di costo utilizzata in A* Tree Search?
- Qual è la strategia utilizzata in A* Tree Search?
- Cosa succede se non si utilizza un'euristica in A* Tree Search?
- Qual è la caratteristica chiave di A* Tree Search?
- Quali sono le condizioni per la completezza di A* Tree Search?
- Qual è la condizione per l'ottimalità di A* Tree Search?
- Come si dimostra l'ottimalità di A* Tree Search?

### A* Graph Search:

- Qual è la differenza tra A* Tree Search e A* Graph Search?
- Cosa si intende per "closed list" in A* Graph Search?
- Qual è il problema che si presenta in A* Graph Search?
- Perché la consistenza è importante per l'ottimalità di A* Graph Search?
- Come si dimostra l'ottimalità di A* Graph Search?

### Teorema: Consistenza implica Ammissibilità:

- Come si dimostra che la consistenza implica l'ammissibilità?
- Qual è il caso base della dimostrazione?
- Qual è l'ipotesi induttiva della dimostrazione?
- Qual è il passo induttivo della dimostrazione?

### Weighted A* Search:

- Qual è l'obiettivo di Weighted A* Search?
- Come si modifica la funzione di costo in Weighted A* Search?
- Cosa succede quando w = 0, w = 1 e w tende all'infinito?
- Quali sono gli effetti di aumentare il valore di w?

### Recursive Best-First Search (RBFS):

- Qual è l'obiettivo di RBFS?
- Come funziona RBFS?
- Qual è il problema principale di RBFS?

## Ricerca in ambienti complessi

### Ricerca di Stati Finali

- Qual è la differenza principale tra le strategie di ricerca precedenti e la ricerca di stati finali?
- In quali situazioni la ricerca di stati finali è più adatta rispetto alla ricerca di cammini?
- Cosa si intende per "buon stato" nel contesto della ricerca di stati finali?
- Quali sono le caratteristiche di un ambiente non deterministico?
- Quali sono le sfide che un agente deve affrontare in un ambiente non osservabile?

### Ricerca Hill Climbing

- Come funziona l'algoritmo di ricerca Hill Climbing?
- Qual è il principale svantaggio dell'Hill Climbing?
- Perché l'Hill Climbing è chiamato anche ricerca locale greedy?

### Simulated Annealing

- In che modo il Simulated Annealing evita di rimanere intrappolato in ottimi locali?
- Come funziona il processo di "scuotimento" nello stato corrente?
- Qual è il ruolo della temperatura nell'algoritmo Simulated Annealing?
- Come si comporta la probabilità di accettare una mossa cattiva in funzione della temperatura e della qualità della mossa?
- Cosa garantisce la proprietà della distribuzione di Boltzmann riguardo al Simulated Annealing?

### Ricerca con Azioni non Deterministiche

- Perché è necessario utilizzare piani condizionali in un ambiente non deterministico?
- Come si può rappresentare un piano condizionale?
- Qual è il ruolo dell'istruzione if in un piano condizionale?

### Ricerca AND-OR

- Qual è la differenza principale tra gli alberi AND-OR e gli alberi di ricerca tradizionali?
- Cosa rappresentano i nodi OR e i nodi AND in un albero AND-OR?
- Quali sono le caratteristiche di una soluzione per un problema di ricerca AND-OR?
- Come si assicura l'algoritmo di ricerca AND-OR di terminare in uno spazio degli stati finito?
- Cosa si intende per piano ciclico?
- Quali sono le condizioni per cui un piano ciclico è considerato una soluzione?

### Ricerca con Osservazioni Parziali

- Cosa sono i belief states?
- Quali sono le tre fasi di un piano in un ambiente con osservazioni parziali?
- In che modo la ricerca AND-OR si adatta al contesto delle osservazioni parziali?

## Ricerca con avversari e giochi

### Approcci alla Teoria dei Giochi

* Quali sono i tre approcci principali per affrontare gli ambienti multi-agente nella teoria dei giochi?
* In che modo l'approccio dell'economia aggregata differisce dagli altri due approcci?
* Qual è la differenza principale tra l'approccio dell'ambiente non deterministico e l'approccio degli alberi di gioco?

### Tipi di Giochi

* Quali sono le due principali categorie di giochi in base al determinismo?
* Come si classificano i giochi in base al numero di giocatori?
* Cosa caratterizza un gioco a somma zero?
* Cosa significa che un gioco ha informazione perfetta?

### Formalizzazione di un Gioco Deterministico

* Quali sono gli elementi che compongono la formalizzazione di un gioco deterministico?
* Cosa rappresenta l'insieme di stati S?
* Cosa rappresenta l'insieme di mosse/azioni A?
* Come funziona il modello di transizione?
* Cosa rappresenta il test di terminazione?
* Cosa rappresenta la funzione di utilità?
* Cosa si intende per soluzione in un gioco deterministico?

### Albero di Gioco

* Cosa si intende per Albero di Gioco?
* In quali casi l'albero di gioco potrebbe essere infinito?

### Giochi a Somma Zero

* Quali sono le caratteristiche principali dei giochi a somma zero?
* In che modo i giochi a somma zero si differenziano dai giochi generali?

### Ricerca MiniMax

* Cosa si intende per mossa e ply in un albero di gioco?
* Come si calcola il valore MiniMax di uno stato?
* Cosa si assume riguardo al comportamento dei giocatori quando si calcola il valore MiniMax?
* Qual è la complessità spaziale e temporale della ricerca MiniMax nel caso peggiore?

### Alpha-Beta Pruning

* Qual è lo scopo dell'Alpha-Beta Pruning?
* Come funzionano i valori α e β nell'algoritmo Alpha-Beta Pruning?
* In che modo l'Alpha-Beta Pruning riduce il numero di nodi da esplorare?
* Come influisce l'ordine di esplorazione dei successori sull'efficacia del pruning?

### Ricerca a Profondità Limitata

* Qual è il problema che risolve la ricerca a profondità limitata?
* Come funziona la ricerca a profondità limitata?
* Quali sono le garanzie di ottimalità nella ricerca a profondità limitata?

### Ricerca Euristica Alpha-Beta

* Qual è il problema che risolve la ricerca euristica Alpha-Beta?
* Come funziona il test di taglio nella ricerca euristica Alpha-Beta?
* Quali sono le proprietà che deve soddisfare l'euristica?
* Qual è l'importanza dell'euristica nella ricerca euristica Alpha-Beta?

### Ricerca ad Albero Monte Carlo

* Qual è lo scopo della ricerca ad Albero Monte Carlo?
* Quali sono le quattro fasi della ricerca ad Albero Monte Carlo?
* Cosa determina la politica di selezione?
* Cosa determina la politica di simulazione?
* Come funziona la politica di selezione UCB1?

### Pacman Veloce

* Come viene modellato il labirinto di Pacman?
* Come viene definito il grafo di gioco?
* Cosa rappresenta uno stato del gioco?
* Cosa si intende per strategia vincente per i fantasmi?
* Come può essere rappresentata una strategia vincente per i fantasmi?
* Qual è la complessità dell'algoritmo per trovare una strategia vincente per i fantasmi?

## Incertezza e Utilità

### Agenti non Razionali

* Cosa rende un agente non razionale?
* Quali sono le due cause principali di comportamento casuale negli agenti non razionali?
* Perché il concetto di "caso peggiore" non è più rilevante in presenza di agenti non razionali?
* Cosa rappresenta il valore atteso e come viene calcolato?

### Expectimax

* Come si differenzia Expectimax da MiniMax?
* Cosa rappresentano i nodi chance in Expectimax?
* Come viene calcolato il valore di un nodo chance?
* Fornisci un esempio di come viene calcolato il valore atteso di un nodo chance.

### Probabilità e Simulazione

* Da dove possono derivare le probabilità utilizzate in Expectimax?
* Quali sono i due metodi principali per ottenere le probabilità?
* Qual è lo scopo della simulazione in Expectimax?
* Quali sono i potenziali svantaggi della simulazione?

### Incertezza, Pessimismo e Ottimismo

* In che modo l'introduzione di incertezza può migliorare la strategia?
* Come può un atteggiamento pessimista influenzare la scelta di una strategia?
* Come può un atteggiamento ottimistico influenzare la scelta di una strategia?

### ExpectiMiniMax

* Come si differenzia ExpectiMiniMax da Expectimax?
* Quali tipi di nodi sono inclusi in ExpectiMiniMax?
* Come viene calcolato il valore di un nodo chance in ExpectiMiniMax?

### Utilità nei Giochi con Più Agenti

* Come viene rappresentata l'utilità in giochi con più agenti?
* Quali sono le possibili conseguenze dell'interazione tra più agenti in un gioco?
* Cosa misura l'utilità?
* Qual è l'obiettivo di un agente razionale in relazione all'utilità?

### Importanza della Scala

* In quali contesti la scala della funzione di utilità non è importante?
* In quali contesti la scala della funzione di utilità è importante?
* Perché è necessario utilizzare le lotterie quando si ha a che fare con l'incertezza?

#### Lotterie e Preferenze

* Cosa rappresenta una lotteria?
* Come vengono espresse le preferenze tra due lotterie?
* Spiega la notazione utilizzata per rappresentare le lotterie.

### Assiomi di Razionalità

* Qual è lo scopo degli assiomi di razionalità?
* Spiega l'assioma di transitività e perché è importante.
* Quali sono le conseguenze di non avere preferenze transitive?

### Assiomi di Razionalità per le Preferenze

* Quali sono i quattro assiomi di razionalità per le preferenze?
* Spiega la definizione e la formulazione di ogni assioma.

### Teorema di Ramsey, von Neumann & Morgenstern

* Cosa dimostra il teorema di Ramsey, von Neumann & Morgenstern?
* Qual è la formulazione del teorema?
* Come viene calcolata l'utilità di una lotteria secondo il teorema?

### Principio della Massima Utilità Attesa (MEU)

* Cosa afferma il principio MEU?
* Dove viene applicato il principio MEU?
* Quali sono i limiti del principio MEU?

### Razionalità Umana

* Perché gli esseri umani non sempre si comportano in modo razionale?
* Spiega il concetto di avversione al rischio.
* Come influenza la ricchezza le scelte degli individui?
* Cosa rappresenta il premio di assicurazione?
* Fornisci un esempio di irrazionalità nelle scelte umane.

## CSP

### Soddisfacimento di Vincoli (CSP)

* Quali sono i due modi principali per rappresentare i vincoli in un CSP?
* Cosa rappresenta una soluzione a un CSP?
* Come si relaziona la risoluzione di un CSP con le query congiuntive nei database?
* Quali sono i tre tipi di complessità che possono essere considerati in un CSP?

### Constraint Graph

* Cosa rappresenta un Constraint Graph?
* Cosa rappresentano i nodi e gli archi in un Constraint Graph?
* Quali sono i limiti di un Constraint Graph per vincoli che coinvolgono più di due variabili?
* Cosa è un ipergrafo e come viene utilizzato per rappresentare vincoli complessi?

### Problema dell'Omomorfismo

* Qual è l'obiettivo del Problema dell'Omomorfismo?
* Cosa costituisce una struttura relazionale?
* Spiega il concetto di arità in relazione ai simboli di relazione.
* Cosa rappresenta un omomorfismo tra due strutture relazionali?
* In che modo il problema dell'omomorfismo può essere interpretato in termini di "trasferimento" di struttura dei dati?

### Core

* Cosa rappresenta il core di un CSP?
* Come viene ottenuto il core di un CSP?
* Fornisci un esempio di come la ridondanza può essere eliminata per ottenere il core.
* Cosa è un endomorfismo e come si relaziona al core?
* Quali sono le proprietà chiave del core di un CSP?

### Backtracking Search

* Qual è l'algoritmo di base per la ricerca backtracking in un CSP?
* Spiega come funziona l'algoritmo di base con un esempio.
* Quali sono i due principali miglioramenti all'algoritmo di base?
* Spiega il concetto di forward checking e come funziona.
* Spiega il concetto di arc consistency e come funziona.
* Come funziona la propagazione dei vincoli nel backtracking?

### Tipi di consistenza

* Spiega il concetto di 1-consistency (node-consistency).
* Spiega il concetto di 2-consistency (arc-consistency).
* Quali sono le conseguenze di forzare l'arc-consistency in una struttura aciclica?
* Spiega il concetto di k-consistency.
* Qual è il costo computazionale della k-consistency?

### Euristiche per la Risoluzione dei CSP

* Quali sono le due principali euristiche per l'ordinamento delle variabili?
* Spiega l'euristica Minimum Remaining Values (MRV).
* Spiega l'euristica Least Constraining Value.
* Quali sono le tecniche di risoluzione con assegnamento completo?
* Spiega l'euristica Minimo Conflitto.
* Come possono essere utilizzate le tecniche di Hill Climbing o Simulated Annealing per risolvere i CSP?
* Cosa rappresenta il problema del punto critico?

### Omomorfismo su strutture acicliche

* Cosa rappresenta il problema dell'omomorfismo su strutture acicliche?
* Qual è la complessità di risolvere il problema dell'omomorfismo su strutture acicliche?
* Qual è l'algoritmo di Yannakakis e come funziona?
* Spiega le fasi di filtraggio verso l'alto e verso il basso nell'algoritmo di Yannakakis.
* Qual è il costo computazionale del filtraggio nell'algoritmo di Yannakakis?

### Soluzione Backtrack-free

* Cosa si intende per soluzione backtrack-free?
* Come si ottiene una soluzione backtrack-free utilizzando l'algoritmo di Yannakakis?
* Quali sono i vantaggi di una soluzione backtrack-free?

### Strutture quasi ad Albero

* Qual è l'obiettivo delle strutture quasi ad albero?
* Cosa rappresenta il cut set (feedback vertex number)?
* Come viene risolto un problema aciclico dopo la rimozione di un nodo?
* Qual è il costo computazionale della risoluzione di un problema quasi ad albero?
* Come possono essere utilizzati gli ipergrafi per ottimizzare la risoluzione di problemi quasi ad albero?

### Join Tree

* Cosa rappresenta un Join-Tree?
* Quali sono le regole chiave per la costruzione di un Join-Tree?
* Cosa rappresenta l'ipergrafo H?
* Cosa rappresenta l'insieme V'?
* Cosa rappresenta l'albero T?

### Ipergrafi Aciclici

* Quando possiamo dire che un ipergrafo è aciclico?
* Qual è la complessità di decidere se un ipergrafo è aciclico?

### Tree Decomposition

* Qual è l'obiettivo della tree decomposition?
* Come viene costruita una tree decomposition?
* Quali sono le proprietà chiave di una tree decomposition?
* Cosa rappresenta la width di una decomposizione?
* Cosa rappresenta la tree-width di un grafo?
* Qual è la complessità di calcolare una tree decomposition di width ≤ k?
* Come viene utilizzata la tree decomposition per risolvere i CSP?
* Qual è il processo di risoluzione di un CSP tramite tree decomposition?
* Qual è la complessità della risoluzione di un CSP tramite tree decomposition?
* Cosa si intende per local consistency e global consistency?

### Teorema di Grohe

* Cosa afferma il Teorema di Grohe?
* Qual è il contesto del Teorema di Grohe?
* Cosa rappresenta la treewidth (tw) in relazione al Teorema di Grohe?
* Cosa rappresenta il core in relazione al Teorema di Grohe?
* Qual è l'algoritmo risolutivo per i problemi CSP appartenenti a S secondo il Teorema di Grohe?
* Qual è la complessità dell'algoritmo risolutivo?

## Teoria dei Giochi

### Tipi di giochi

- Quali sono le caratteristiche principali dei giochi strategici e quale esempio viene fornito?
- Cosa distingue i giochi estensivi dagli altri tipi di gioco e in quali contesti si applicano?
- Come influisce la ripetitività di una situazione sulle strategie dei giocatori nei giochi ripetitivi?
- Qual è l'obiettivo dei giochi cooperativi e quale esempio viene proposto?
- Cosa si intende per informazione perfetta, imperfetta, completa e incompleta in un gioco?
- Cos'è il Mechanism design e in quale contesto viene applicato?

### Giochi Coalizionali

- Cosa sono i giochi coalizionali e qual è il loro obiettivo principale?
- Cosa rappresenta la "worth" in un gioco coalizionale?
- Cosa si intende per "solution concept" in un gioco coalizionale e quali proprietà dovrebbe soddisfare?
- Quali sono le componenti principali della struttura di un gioco coalizionale?
- Quali sono le domande fondamentali a cui un gioco coalizionale cerca di rispondere?
- Qual è la differenza tra Transferable Utility (TU) e Non-Transferable Utility (NTU)?
- Come viene formalmente definito un gioco coalizionale e cosa rappresenta l'outcome?
- Quali proprietà deve soddisfare un'imputazione per essere considerata ammissibile?
- Cosa rappresentano i concetti di Fairness e Stability in un gioco coalizionale?

### Caratteristiche dei Giochi Coalizionali

- Cosa caratterizza un gioco superadditivo e qual è la sua formula?
- Cosa distingue un gioco additivo da un gioco superadditivo?
- Cosa si intende per gioco a somma costante e qual è la sua formula?
- Cosa caratterizza un gioco convesso e in che modo incentiva la collaborazione?
- Cosa sono i giochi semplici e quale esempio viene fornito?
- Cosa sono i "proper simple games"?

### Core di un Gioco Coalizionale

- Cos'è il core di un gioco coalizionale e cosa rappresenta?
- Come viene definito formalmente il core?
- Perché il core è importante per la stabilità della coalizione?
- È possibile che il core sia vuoto?
- Cosa implica l'assioma di simmetria in un gioco coalizionale?
- Chi sono i giocatori nulli e come vengono definiti?
- Cosa significa che una soluzione è additiva in un gioco coalizionale?

### Shapley Value

- Cos'è lo Shapley Value e su cosa si basa?
- Quali sono le caratteristiche chiave dello Shapley Value?
- Come si calcola lo Shapley Value per un giocatore i?
- Come si interpreta la formula dello Shapley Value?
- Puoi fornire un esempio di come si applica lo Shapley Value in un gioco a tre giocatori?

### Nucleolo in un gioco coalizionale

- Cos'è il nucleolo in un gioco coalizionale e qual è l'idea alla base del suo funzionamento?
- Come si definisce l'eccesso di una coalizione rispetto a una imputazione?
- Cos'è il vettore degli eccessi e a cosa serve?
- Come viene definito formalmente il nucleolo di un gioco?
- Quali sono le proprietà del nucleolo?
- Qual è la relazione tra il nucleolo e l'ε-core?
- Come si trova il nucleolo attraverso una procedura iterativa?
- Qual è il problema lineare che si risolve al primo passo dell'iterazione?
- Cosa sono le coalizioni critiche e come vengono identificate?
- Come si procede nell'iterazione successiva per restringere l'insieme delle coalizioni critiche?

### Contested Garment Rule

- Cos'è la Contested Garment Rule e in quale contesto si applica?
- Quali sono i passaggi per applicare la Contested Garment Rule?
- Qual è la relazione tra la Contested Garment Rule e il nucleolo del gioco associato?

### Aste e Mechanism Design

- Di cosa si occupano le aste e il Mechanism Design?
- Quali sono le due principali categorie di aste?
- Quali sono le differenze tra asta inglese, giapponese e olandese?
- Cosa si intende per "mechanism design truthful"?

### Sealed-Bid Auctions (Aste a busta chiusa)

- Cosa sono le aste a busta chiusa e su cosa si basano gli agenti per fare le offerte?
- Cosa sono le distribuzioni IPV (Independent Private Values)?

### Second-Price Auctions (Aste al secondo prezzo)

- Come funzionano le aste al secondo prezzo?
- Perché le aste al secondo prezzo sono considerate "mechanism design truthful"?
- Qual è la strategia ottimale in un'asta al secondo prezzo e perché?
- In che modo l'asta giapponese si comporta in modo simile all'asta al secondo prezzo?

### First-Price Auctions: Aste al Primo Prezzo

- Come funzionano le aste al primo prezzo?
- Qual è la strategia ottimale in un'asta al primo prezzo con due giocatori e distribuzioni uniformi?
- Come si calcola il valore atteso dell'utilità in un'asta al primo prezzo?
- Cos'è un equilibrio di Nash in questo contesto?
- Qual è la strategia ottimale in un'asta al primo prezzo con n giocatori?

### Giochi Strategici

- Cosa sono i giochi strategici e qual è il loro obiettivo?
- In che modo il dilemma del prigioniero illustra i principi dei giochi strategici?
- Quali sono i payoff nel dilemma del prigioniero e cosa rappresentano?
- Cos'è l'equilibrio di Nash nel dilemma del prigioniero e perché si verifica?

### Equilibrio di Nash

- Come viene definito formalmente un gioco strategico?
- Cos'è un profilo di azione e cosa rappresenta?
- Come viene definito l'equilibrio di Nash?
- Puoi fornire un esempio di equilibrio di Nash nel gioco dei Bach e Stravinsky?
- Cos'è la funzione di best response e come si relaziona all'equilibrio di Nash?

### Strategie Miste

- Cosa sono le strategie miste e perché vengono utilizzate?
- Come si calcola l'utilità attesa di una strategia mista?
- Cos'è un equilibrio di Nash con strategie miste?
- Puoi fornire un esempio di utilità attesa in una lotteria?
- Cos'è la funzione di utilità di Von Neumann-Morgenstern e a cosa serve?

### Strategie Miste nei Giochi Strategici

- Come si calcola l'utilità attesa di una strategia mista in un gioco strategico?
- Come viene definito l'equilibrio di Nash con strategie miste in un gioco strategico?

### Teorema di Nash

- Cosa afferma il Teorema di Nash?
- Quali sono le due proprietà fondamentali dell'equilibrio di Nash in strategie miste?
- Cos'è il supporto di una strategia mista?
- Come si enuncia il Teorema di Nash in termini di best response e supporto?

### Dimostrazione del Teorema di Nash

- Come si dimostra la prima parte del Teorema di Nash (implicazione $\Rightarrow$)?
- Come si dimostra la seconda parte del Teorema di Nash (implicazione $\Leftarrow$)?
- Cosa afferma il Corollario 1 della dimostrazione del Teorema di Nash?
- Cosa afferma il Corollario 2 della dimostrazione del Teorema di Nash?

### Esempio di Equilibrio di Nash Misto: Bach e Stravinsky

- Come si calcola l'equilibrio di Nash misto nel gioco di Bach e Stravinsky?
- Quali sono le strategie dei giocatori e come si arriva all'equilibrio?

### Giochi Strategici (Ripresa)

- Quali sono le modalità di gioco e le caratteristiche dell'informazione nei giochi strategici?

### Giochi in forma estesa

- Cosa sono i giochi in forma estesa e come vengono rappresentati?
- Quali sono le caratteristiche dell'informazione e della memoria nei giochi in forma estesa?
- Quali sono le componenti principali della struttura di un gioco in forma estesa?
- Cosa si intende per informazione perfetta e imperfetta in un gioco in forma estesa?
- Cos'è un information set?
- Cos'è il Subgame Perfect Equilibrium (SPE) e come si differenzia dall'equilibrio di Nash?
- Cos'è l'algoritmo Minimax e come si relaziona alla dimostrazione dell'esistenza di un SPE?

## Planning

### Domande sul paragrafo "Modello di Pianificazione"

- Cosa rappresenta uno stato nel contesto della pianificazione?
- In che modo le azioni influenzano lo stato del sistema?
- Qual è la differenza tra eventi e azioni?
- Cosa fa la funzione di transizione di stato?
- Cosa significa che un sistema è non deterministico?
- Qual è l'obiettivo del processo di pianificazione?
- Qual è la differenza tra un planner e un controller?
- Quali sono i compiti principali di un planner?
- Quali sono i compiti principali di un controller?
- Qual è la differenza tra pianificazione offline e pianificazione dinamica?
- Qual è la differenza tra un planner domain-specific e un planner domain-independent?
- Cosa significa che un planner è configurabile?
- Qual è la differenza tra scheduling e planning?
- Qual è la complessità computazionale del problema di scheduling?
- Qual è la complessità computazionale del problema di planning?

### Domande sul paragrafo "Classical Planning"

- Quali sono le cinque caratteristiche principali del Classical Planning?
- Cosa significa che il sistema è completo e deterministico?
- Cosa sono gli stati goal?
- Cosa significa che i piani sono sequenziali?
- Cosa significa che il tempo è implicito nel Classical Planning?
- Qual è il problema di pianificazione nel Classical Planning?
- Come viene rappresentato il problema di pianificazione come un grafo?
- Qual è il problema principale quando il numero di stati è molto grande?
- Perché i planner configurabili sono spesso usati nel Classical Planning?
- Quali sono i quattro passaggi principali del processo di planning?
- Perché è importante creare piani di riserva?

### Domande sul paragrafo "Classical Representation"

- Qual è la differenza tra predicati e costanti?
- Cosa sono gli atomi nella Classical Representation?
- Qual è la differenza tra ground expressions e unground expressions?
- Cosa significa grounding?
- Cosa fa una sostituzione?
- Cosa rappresenta uno stato nella Classical Representation?
- Spiega l'esempio del predicato `top(pallet, shelf)` e delle sue ground expressions.

### Domande sul paragrafo "Operatore"

- Quali sono le tre componenti di un operatore?
- Cosa rappresenta `name(o)`?
- Cosa rappresentano `preconditions(o)`?
- Cosa rappresentano `effects(o)`?

### Domande sul paragrafo "Azione e Ground Instance"

- Cosa è un'azione?
- Come si ottiene una azione da un operatore?
- Cosa rappresentano `precond+(a)`, `precond−(a)`, `effects+(a)` e `effects−(a)`?

### Domande sul paragrafo "Applicabilità delle Azioni"

- Quando un'azione è applicabile in uno stato?
- Cosa sono i letterali positivi e negativi?

### Domande sul paragrafo "Dominio di Planning"

- Cosa è un dominio di planning?

### Domande sul paragrafo "Piano e Soluzione"

- Cosa è un piano?
- Quando un piano è una soluzione per un problema di planning?
- Cosa significa che un piano è eseguibile?
- Cosa significa che uno stato finale soddisfa l'insieme degli obiettivi?
- Spiega la formulazione formale di una soluzione.

### Domande sul paragrafo "Soluzioni Ridondanti"

- Perché si possono trovare più soluzioni per un problema di planning?
- Qual è il problema delle soluzioni ridondanti?
- Cosa significa trovare una soluzione minima o più breve?
- Cosa fa la rappresentazione set-theoretic?
- Qual è il problema principale della rappresentazione set-theoretic?

### Domande sul paragrafo "Rappresentazione State-Variable"

- Cosa sono le proprietà statiche e dinamiche?
- Come si rappresentano le proprietà statiche e dinamiche nella rappresentazione state-variable?
- Spiega l'esempio di `top(p1) = c3`.

## State-Space Planning

### Domande sul paragrafo "State-Space Planning"

- Come viene rappresentato il problema di pianificazione nello state-space planning?
- Cosa rappresenta ogni nodo nello state-space planning?
- Cosa rappresenta un piano nello state-space planning?
- Come funziona il plan-space planning?
- Cosa rappresenta ogni nodo nel plan-space planning?

### Domande sul paragrafo "Approcci e Tecniche"

#### Domande sul paragrafo "Forward Search"

- Qual è l'idea principale del forward search?
- Quali sono i passaggi principali dell'algoritmo di forward search?
- Cosa significa che un algoritmo è sound?
- Cosa significa che un algoritmo è complete?
- Quali tecniche di ricerca possono essere utilizzate per implementare il forward search?
- Quali sono i vantaggi e gli svantaggi di BFS, DFS, Best-First Search, Greedy e A* Search?
- Quali sono i problemi principali del forward search?

#### Domande sul paragrafo "Backward Search"

- Qual è l'idea principale del backward search?
- Come si definiscono le azioni rilevanti per un goal?
- Cosa fa la funzione inversa della transizione?
- Quali sono i passaggi principali dell'algoritmo di backward search?
- Quali sono i problemi principali del backward search?

### Domande sul paragrafo "Lifting"

- Qual è l'obiettivo del lifting?
- Come si utilizzano le variabili nel lifting?
- Come il lifting riduce lo spazio di ricerca?
- Cosa è il maximum general unifier?
- Qual è il problema principale del lifting?

### Domande sul paragrafo "STRIPS"

- Su cosa si basa l'algoritmo STRIPS?
- Quali sono le caratteristiche principali di STRIPS?
- Come risolve STRIPS i goal?
- Perché STRIPS non è completo?
- Cosa è l'anomalia di Sussman?
- Perché l'anomalia di Sussman dimostra il fallimento di STRIPS?
- Quali sono gli approcci alternativi per gestire i problemi come l'anomalia di Sussman?

### Domande sul paragrafo "Tecniche del Planning-Graph"

- Qual è l'obiettivo del Planning-Graph?
- Come funziona il Planning-Graph?
- Cosa sono le azioni di maintenance/frame?
- Cosa significa mutex?
- Quali sono i tipi di mutex?
- Quali sono i vantaggi del Planning-Graph?
- Come si costruisce il Planning-Graph?
- Come si esegue la backward search nel Planning-Graph?
- Quali sono i vantaggi e gli svantaggi del Planning-Graph?
- Come si utilizzano le azioni di maintenance/frame nel Planning-Graph?
- Come si definiscono i mutex tra azioni e stati nel Planning-Graph?
- Quali sono i vantaggi e gli svantaggi del Planning-Graph rispetto ad altri approcci?

