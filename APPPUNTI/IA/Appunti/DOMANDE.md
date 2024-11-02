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
