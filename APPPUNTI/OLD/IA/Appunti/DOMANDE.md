# Ricerca:

### Definizione del problema di ricerca:

1. Quali sono i quattro elementi che definiscono un problema di ricerca?
2. Cosa si intende per "spazio degli stati"?
3. Come si definisce una soluzione in un problema di ricerca?
4. Qual è la differenza tra "search state" e "world state"?

### Rappresentazione dello spazio degli stati:

5. Quali sono le due principali rappresentazioni grafiche dello spazio degli stati?
6. Cosa rappresentano i nodi e gli archi in un "state space graph"?
7. Cosa rappresenta la radice in un "search tree"?

### Algoritmo di ricerca:

8. Qual è l'obiettivo principale della ricerca?
9. Perché è importante evitare i loop durante la ricerca?

### Ricerca ad albero:

10. Quali sono i cinque passaggi principali della ricerca ad albero?
11. Cosa si intende per "frontiera" nella ricerca ad albero?
12. Quali sono i tre modi principali per implementare la frontiera?
13. Qual è la differenza tra una coda FIFO e una coda LIFO?

### Strategie di ricerca non informata:

14. Cosa caratterizza le strategie di ricerca non informata?
15. Quali sono i tre criteri per valutare le prestazioni delle strategie di ricerca non informata?
16. Cosa si intende per "completezza" di un algoritmo di ricerca?
17. Cosa si intende per "ottimalità" di un algoritmo di ricerca?
18. Qual è la complessità spaziale e temporale della ricerca in ampiezza?
19. Qual è la complessità spaziale e temporale della ricerca in profondità?
20. Come funziona l'algoritmo di Iterative Deepening?
21. Qual è la complessità spaziale e temporale di Bidirectional Best-First?

### Strategie di ricerca informata o euristica:

22. Cosa caratterizza le strategie di ricerca informata?
23. Cosa rappresenta la funzione h(n)?
24. Come si combina il costo del passato con il costo stimato del futuro nelle strategie di ricerca informata?
25. Qual è la funzione di valutazione utilizzata nelle strategie di ricerca informata?
26. Qual è l'obiettivo principale delle strategie di ricerca informata?
27. Cosa si intende per "rilassamento" del problema?
28. Come funziona l'algoritmo di Uniform Cost Search?
29. Qual è la funzione di costo utilizzata in Uniform Cost Search?
30. Quali sono le caratteristiche di Uniform Cost Search?
31. Come funziona l'algoritmo di Greedy Best-First Search?
32. Qual è la funzione di costo utilizzata in Greedy Best-First Search?
33. Quali sono le caratteristiche di Greedy Best-First Search?

### Proprietà delle euristiche:

34. Quali sono le due principali proprietà delle euristiche?
35. Cosa si intende per "ammissibilità" di un'euristica?
36. Cosa si intende per "consistenza" di un'euristica?
37. Qual è la relazione tra ammissibilità e consistenza?
38. Perché è importante avere valutazioni quantitative delle euristiche?

### A* Tree Search:

39. Qual è la funzione di costo utilizzata in A* Tree Search?
40. Qual è la strategia utilizzata in A* Tree Search?
41. Cosa succede se non si utilizza un'euristica in A* Tree Search?
42. Qual è la caratteristica chiave di A* Tree Search?
43. Quali sono le condizioni per la completezza di A* Tree Search?
44. Qual è la condizione per l'ottimalità di A* Tree Search?
45. Come si dimostra l'ottimalità di A* Tree Search?

### A* Graph Search:

46. Qual è la differenza tra A* Tree Search e A* Graph Search?
47. Cosa si intende per "closed list" in A* Graph Search?
48. Qual è il problema che si presenta in A* Graph Search?
49. Perché la consistenza è importante per l'ottimalità di A* Graph Search?
50. Come si dimostra l'ottimalità di A* Graph Search?

### Teorema: Consistenza implica Ammissibilità:

51. Come si dimostra che la consistenza implica l'ammissibilità?
52. Qual è il caso base della dimostrazione?
53. Qual è l'ipotesi induttiva della dimostrazione?
54. Qual è il passo induttivo della dimostrazione?

### Weighted A* Search:

55. Qual è l'obiettivo di Weighted A* Search?
56. Come si modifica la funzione di costo in Weighted A* Search?
57. Cosa succede quando w = 0, w = 1 e w tende all'infinito?
58. Quali sono gli effetti di aumentare il valore di w?

### Recursive Best-First Search (RBFS):

59. Qual è l'obiettivo di RBFS?
60. Come funziona RBFS?
61. Qual è il problema principale di RBFS?

# Ricerca in ambienti complessi

### Ricerca di Stati Finali

1. Qual è la differenza principale tra le strategie di ricerca precedenti e la ricerca di stati finali?
2. In quali situazioni la ricerca di stati finali è più adatta rispetto alla ricerca di cammini?
3. Cosa si intende per "buon stato" nel contesto della ricerca di stati finali?
4. Quali sono le caratteristiche di un ambiente non deterministico?
5. Quali sono le sfide che un agente deve affrontare in un ambiente non osservabile?

### Ricerca Hill Climbing

1. Come funziona l'algoritmo di ricerca Hill Climbing?
2. Qual è il principale svantaggio dell'Hill Climbing?
3. Perché l'Hill Climbing è chiamato anche ricerca locale greedy?

### Simulated Annealing

1. In che modo il Simulated Annealing evita di rimanere intrappolato in ottimi locali?
2. Come funziona il processo di "scuotimento" nello stato corrente?
3. Qual è il ruolo della temperatura nell'algoritmo Simulated Annealing?
4. Come si comporta la probabilità di accettare una mossa cattiva in funzione della temperatura e della qualità della mossa?
5. Cosa garantisce la proprietà della distribuzione di Boltzmann riguardo al Simulated Annealing?

### Ricerca con Azioni non Deterministiche

1. Perché è necessario utilizzare piani condizionali in un ambiente non deterministico?
2. Come si può rappresentare un piano condizionale?
3. Qual è il ruolo dell'istruzione if in un piano condizionale?

### Ricerca AND-OR

1. Qual è la differenza principale tra gli alberi AND-OR e gli alberi di ricerca tradizionali?
2. Cosa rappresentano i nodi OR e i nodi AND in un albero AND-OR?
3. Quali sono le caratteristiche di una soluzione per un problema di ricerca AND-OR?
4. Come si assicura l'algoritmo di ricerca AND-OR di terminare in uno spazio degli stati finito?
5. Cosa si intende per piano ciclico?
6. Quali sono le condizioni per cui un piano ciclico è considerato una soluzione?

### Ricerca con Osservazioni Parziali

1. Cosa sono i belief states?
2. Quali sono le tre fasi di un piano in un ambiente con osservazioni parziali?
3. In che modo la ricerca AND-OR si adatta al contesto delle osservazioni parziali?

# Ricerca con avversari e giochi

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

# Incertezza e Utilità

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

# CSP

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

# Teoria dei Giochi

### Tipi di giochi

1. Quali sono le caratteristiche principali dei giochi strategici e quale esempio viene fornito?
2. Cosa distingue i giochi estensivi dagli altri tipi di gioco e in quali contesti si applicano?
3. Come influisce la ripetitività di una situazione sulle strategie dei giocatori nei giochi ripetitivi?
4. Qual è l'obiettivo dei giochi cooperativi e quale esempio viene proposto?
5. Cosa si intende per informazione perfetta, imperfetta, completa e incompleta in un gioco?
6. Cos'è il Mechanism design e in quale contesto viene applicato?

### Giochi Coalizionali

1. Cosa sono i giochi coalizionali e qual è il loro obiettivo principale?
2. Cosa rappresenta la "worth" in un gioco coalizionale?
3. Cosa si intende per "solution concept" in un gioco coalizionale e quali proprietà dovrebbe soddisfare?
4. Quali sono le componenti principali della struttura di un gioco coalizionale?
5. Quali sono le domande fondamentali a cui un gioco coalizionale cerca di rispondere?
6. Qual è la differenza tra Transferable Utility (TU) e Non-Transferable Utility (NTU)?
7. Come viene formalmente definito un gioco coalizionale e cosa rappresenta l'outcome?
8. Quali proprietà deve soddisfare un'imputazione per essere considerata ammissibile?
9. Cosa rappresentano i concetti di Fairness e Stability in un gioco coalizionale?

### Caratteristiche dei Giochi Coalizionali

1. Cosa caratterizza un gioco superadditivo e qual è la sua formula?
2. Cosa distingue un gioco additivo da un gioco superadditivo?
3. Cosa si intende per gioco a somma costante e qual è la sua formula?
4. Cosa caratterizza un gioco convesso e in che modo incentiva la collaborazione?
5. Cosa sono i giochi semplici e quale esempio viene fornito?
6. Cosa sono i "proper simple games"?

### Core di un Gioco Coalizionale

1. Cos'è il core di un gioco coalizionale e cosa rappresenta?
2. Come viene definito formalmente il core?
3. Perché il core è importante per la stabilità della coalizione?
4. È possibile che il core sia vuoto?
5. Cosa implica l'assioma di simmetria in un gioco coalizionale?
6. Chi sono i giocatori nulli e come vengono definiti?
7. Cosa significa che una soluzione è additiva in un gioco coalizionale?

### Shapley Value

1. Cos'è lo Shapley Value e su cosa si basa?
2. Quali sono le caratteristiche chiave dello Shapley Value?
3. Come si calcola lo Shapley Value per un giocatore i?
4. Come si interpreta la formula dello Shapley Value?
5. Puoi fornire un esempio di come si applica lo Shapley Value in un gioco a tre giocatori?

### Nucleolo in un gioco coalizionale

1. Cos'è il nucleolo in un gioco coalizionale e qual è l'idea alla base del suo funzionamento?
2. Come si definisce l'eccesso di una coalizione rispetto a una imputazione?
3. Cos'è il vettore degli eccessi e a cosa serve?
4. Come viene definito formalmente il nucleolo di un gioco?
5. Quali sono le proprietà del nucleolo?
6. Qual è la relazione tra il nucleolo e l'ε-core?
7. Come si trova il nucleolo attraverso una procedura iterativa?
8. Qual è il problema lineare che si risolve al primo passo dell'iterazione?
9. Cosa sono le coalizioni critiche e come vengono identificate?
10. Come si procede nell'iterazione successiva per restringere l'insieme delle coalizioni critiche?

### Contested Garment Rule

1. Cos'è la Contested Garment Rule e in quale contesto si applica?
2. Quali sono i passaggi per applicare la Contested Garment Rule?
3. Qual è la relazione tra la Contested Garment Rule e il nucleolo del gioco associato?

### Aste e Mechanism Design

1. Di cosa si occupano le aste e il Mechanism Design?
2. Quali sono le due principali categorie di aste?
3. Quali sono le differenze tra asta inglese, giapponese e olandese?
4. Cosa si intende per "mechanism design truthful"?

### Sealed-Bid Auctions (Aste a busta chiusa)

1. Cosa sono le aste a busta chiusa e su cosa si basano gli agenti per fare le offerte?
2. Cosa sono le distribuzioni IPV (Independent Private Values)?

### Second-Price Auctions (Aste al secondo prezzo)

1. Come funzionano le aste al secondo prezzo?
2. Perché le aste al secondo prezzo sono considerate "mechanism design truthful"?
3. Qual è la strategia ottimale in un'asta al secondo prezzo e perché?
4. In che modo l'asta giapponese si comporta in modo simile all'asta al secondo prezzo?

### First-Price Auctions: Aste al Primo Prezzo

1. Come funzionano le aste al primo prezzo?
2. Qual è la strategia ottimale in un'asta al primo prezzo con due giocatori e distribuzioni uniformi?
3. Come si calcola il valore atteso dell'utilità in un'asta al primo prezzo?
4. Cos'è un equilibrio di Nash in questo contesto?
5. Qual è la strategia ottimale in un'asta al primo prezzo con n giocatori?

### Giochi Strategici

1. Cosa sono i giochi strategici e qual è il loro obiettivo?
2. In che modo il dilemma del prigioniero illustra i principi dei giochi strategici?
3. Quali sono i payoff nel dilemma del prigioniero e cosa rappresentano?
4. Cos'è l'equilibrio di Nash nel dilemma del prigioniero e perché si verifica?

### Equilibrio di Nash

1. Come viene definito formalmente un gioco strategico?
2. Cos'è un profilo di azione e cosa rappresenta?
3. Come viene definito l'equilibrio di Nash?
4. Puoi fornire un esempio di equilibrio di Nash nel gioco dei Bach e Stravinsky?
5. Cos'è la funzione di best response e come si relaziona all'equilibrio di Nash?

### Strategie Miste

1. Cosa sono le strategie miste e perché vengono utilizzate?
2. Come si calcola l'utilità attesa di una strategia mista?
3. Cos'è un equilibrio di Nash con strategie miste?
4. Puoi fornire un esempio di utilità attesa in una lotteria?
5. Cos'è la funzione di utilità di Von Neumann-Morgenstern e a cosa serve?

### Strategie Miste nei Giochi Strategici

1. Come si calcola l'utilità attesa di una strategia mista in un gioco strategico?
2. Come viene definito l'equilibrio di Nash con strategie miste in un gioco strategico?

### Teorema di Nash

1. Cosa afferma il Teorema di Nash?
2. Quali sono le due proprietà fondamentali dell'equilibrio di Nash in strategie miste?
3. Cos'è il supporto di una strategia mista?
4. Come si enuncia il Teorema di Nash in termini di best response e supporto?

### Dimostrazione del Teorema di Nash

1. Come si dimostra la prima parte del Teorema di Nash (implicazione $\Rightarrow$)?
2. Come si dimostra la seconda parte del Teorema di Nash (implicazione $\Leftarrow$)?
3. Cosa afferma il Corollario 1 della dimostrazione del Teorema di Nash?
4. Cosa afferma il Corollario 2 della dimostrazione del Teorema di Nash?

### Esempio di Equilibrio di Nash Misto: Bach e Stravinsky

1. Come si calcola l'equilibrio di Nash misto nel gioco di Bach e Stravinsky?
2. Quali sono le strategie dei giocatori e come si arriva all'equilibrio?

### Giochi Strategici (Ripresa)

1. Quali sono le modalità di gioco e le caratteristiche dell'informazione nei giochi strategici?

### Giochi in forma estesa

1. Cosa sono i giochi in forma estesa e come vengono rappresentati?
2. Quali sono le caratteristiche dell'informazione e della memoria nei giochi in forma estesa?
3. Quali sono le componenti principali della struttura di un gioco in forma estesa?
4. Cosa si intende per informazione perfetta e imperfetta in un gioco in forma estesa?
5. Cos'è un information set?
6. Cos'è il Subgame Perfect Equilibrium (SPE) e come si differenzia dall'equilibrio di Nash?
7. Cos'è l'algoritmo Minimax e come si relaziona alla dimostrazione dell'esistenza di un SPE?

# Planning 

### Domande sul paragrafo "Modello di Pianificazione"

1. Cosa rappresenta uno stato nel contesto della pianificazione?
2. In che modo le azioni influenzano lo stato del sistema?
3. Qual è la differenza tra eventi e azioni?
4. Cosa fa la funzione di transizione di stato?
5. Cosa significa che un sistema è non deterministico?
6. Qual è l'obiettivo del processo di pianificazione?
7. Qual è la differenza tra un planner e un controller?
8. Quali sono i compiti principali di un planner?
9. Quali sono i compiti principali di un controller?
10. Qual è la differenza tra pianificazione offline e pianificazione dinamica?
11. Qual è la differenza tra un planner domain-specific e un planner domain-independent?
12. Cosa significa che un planner è configurabile?
13. Qual è la differenza tra scheduling e planning?
14. Qual è la complessità computazionale del problema di scheduling?
15. Qual è la complessità computazionale del problema di planning?

### Domande sul paragrafo "Classical Planning"

1. Quali sono le cinque caratteristiche principali del Classical Planning?
2. Cosa significa che il sistema è completo e deterministico?
3. Cosa sono gli stati goal?
4. Cosa significa che i piani sono sequenziali?
5. Cosa significa che il tempo è implicito nel Classical Planning?
6. Qual è il problema di pianificazione nel Classical Planning?
7. Come viene rappresentato il problema di pianificazione come un grafo?
8. Qual è il problema principale quando il numero di stati è molto grande?
9. Perché i planner configurabili sono spesso usati nel Classical Planning?
10. Quali sono i quattro passaggi principali del processo di planning?
11. Perché è importante creare piani di riserva?

### Domande sul paragrafo "Classical Representation"

1. Qual è la differenza tra predicati e costanti?
2. Cosa sono gli atomi nella Classical Representation?
3. Qual è la differenza tra ground expressions e unground expressions?
4. Cosa significa grounding?
5. Cosa fa una sostituzione?
6. Cosa rappresenta uno stato nella Classical Representation?
7. Spiega l'esempio del predicato `top(pallet, shelf)` e delle sue ground expressions.

### Domande sul paragrafo "Operatore"

1. Quali sono le tre componenti di un operatore?
2. Cosa rappresenta `name(o)`?
3. Cosa rappresentano `preconditions(o)`?
4. Cosa rappresentano `effects(o)`?

### Domande sul paragrafo "Azione e Ground Instance"

1. Cosa è un'azione?
2. Come si ottiene una azione da un operatore?
3. Cosa rappresentano `precond+(a)`, `precond−(a)`, `effects+(a)` e `effects−(a)`?

### Domande sul paragrafo "Applicabilità delle Azioni"

1. Quando un'azione è applicabile in uno stato?
2. Cosa sono i letterali positivi e negativi?

### Domande sul paragrafo "Dominio di Planning"

1. Cosa è un dominio di planning?

### Domande sul paragrafo "Piano e Soluzione"

1. Cosa è un piano?
2. Quando un piano è una soluzione per un problema di planning?
3. Cosa significa che un piano è eseguibile?
4. Cosa significa che uno stato finale soddisfa l'insieme degli obiettivi?
5. Spiega la formulazione formale di una soluzione.

### Domande sul paragrafo "Soluzioni Ridondanti"

1. Perché si possono trovare più soluzioni per un problema di planning?
2. Qual è il problema delle soluzioni ridondanti?
3. Cosa significa trovare una soluzione minima o più breve?
4. Cosa fa la rappresentazione set-theoretic?
5. Qual è il problema principale della rappresentazione set-theoretic?

### Domande sul paragrafo "Rappresentazione State-Variable"

1. Cosa sono le proprietà statiche e dinamiche?
2. Come si rappresentano le proprietà statiche e dinamiche nella rappresentazione state-variable?
3. Spiega l'esempio di `top(p1) = c3`.

# State-Space Planning

### Domande sul paragrafo "State-Space Planning"

1. Come viene rappresentato il problema di pianificazione nello state-space planning?
2. Cosa rappresenta ogni nodo nello state-space planning?
3. Cosa rappresenta un piano nello state-space planning?
4. Come funziona il plan-space planning?
5. Cosa rappresenta ogni nodo nel plan-space planning?

### Domande sul paragrafo "Approcci e Tecniche"

#### Domande sul paragrafo "Forward Search"

1. Qual è l'idea principale del forward search?
2. Quali sono i passaggi principali dell'algoritmo di forward search?
3. Cosa significa che un algoritmo è sound?
4. Cosa significa che un algoritmo è complete?
5. Quali tecniche di ricerca possono essere utilizzate per implementare il forward search?
6. Quali sono i vantaggi e gli svantaggi di BFS, DFS, Best-First Search, Greedy e A* Search?
7. Quali sono i problemi principali del forward search?

#### Domande sul paragrafo "Backward Search"

1. Qual è l'idea principale del backward search?
2. Come si definiscono le azioni rilevanti per un goal?
3. Cosa fa la funzione inversa della transizione?
4. Quali sono i passaggi principali dell'algoritmo di backward search?
5. Quali sono i problemi principali del backward search?

### Domande sul paragrafo "Lifting"

1. Qual è l'obiettivo del lifting?
2. Come si utilizzano le variabili nel lifting?
3. Come il lifting riduce lo spazio di ricerca?
4. Cosa è il maximum general unifier?
5. Qual è il problema principale del lifting?

### Domande sul paragrafo "STRIPS"

1. Su cosa si basa l'algoritmo STRIPS?
2. Quali sono le caratteristiche principali di STRIPS?
3. Come risolve STRIPS i goal?
4. Perché STRIPS non è completo?
5. Cosa è l'anomalia di Sussman?
6. Perché l'anomalia di Sussman dimostra il fallimento di STRIPS?
7. Quali sono gli approcci alternativi per gestire i problemi come l'anomalia di Sussman?

### Domande sul paragrafo "Tecniche del Planning-Graph"

1. Qual è l'obiettivo del Planning-Graph?
2. Come funziona il Planning-Graph?
3. Cosa sono le azioni di maintenance/frame?
4. Cosa significa mutex?
5. Quali sono i tipi di mutex?
6. Quali sono i vantaggi del Planning-Graph?
7. Come si costruisce il Planning-Graph?
8. Come si esegue la backward search nel Planning-Graph?
9. Quali sono i vantaggi e gli svantaggi del Planning-Graph?
10. Come si utilizzano le azioni di maintenance/frame nel Planning-Graph?
11. Come si definiscono i mutex tra azioni e stati nel Planning-Graph?
12. Quali sono i vantaggi e gli svantaggi del Planning-Graph rispetto ad altri approcci?

