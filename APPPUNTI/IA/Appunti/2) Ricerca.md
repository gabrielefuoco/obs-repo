
| Termine                                | **Spiegazione**                                                                                                                                |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Spazio degli stati**                 | L'insieme di tutti gli stati possibili che un problema può assumere.                                                                           |
| **Funzione di successione**            | Definisce le azioni possibili e i costi per passare da uno stato all'altro.                                                                    |
| **Stato iniziale**                     | Lo stato di partenza del problema.                                                                                                             |
| **Soluzione**                          | Una sequenza di azioni che porta dallo stato iniziale a uno stato che soddisfa il goal.                                                        |
| **Search state**                       | Uno stato del problema che contiene solo le informazioni rilevanti per la pianificazione.                                                      |
| **World state**                        | Uno stato del problema che contiene tutti i dettagli del mondo.                                                                                |
| **State space graph**                  | Rappresentazione grafica dello spazio degli stati, con nodi che rappresentano gli stati e archi che rappresentano le azioni.                   |
| **Search tree**                        | Rappresentazione ad albero dello spazio degli stati, con la radice che rappresenta lo stato iniziale e i rami che rappresentano le azioni.     |
| **Frontiera**                          | L'insieme dei nodi che sono stati generati ma non ancora esplorati.                                                                            |
| **Coda di priorità**                   | Una struttura dati che ordina gli elementi in base a una funzione di priorità.                                                                 |
| **Completezza**                        | La capacità di un algoritmo di trovare una soluzione se esiste.                                                                                |
| **Ottimalità**                         | La capacità di un algoritmo di trovare la soluzione con il costo minimo.                                                                       |
| **Branching factor**                   | Il numero medio di successori di un nodo.                                                                                                      |
| **Euristica**                          | Una funzione che fornisce una stima del costo per raggiungere il goal.                                                                         |
| **Ammissibilità**                      | Una proprietà di un'euristica che garantisce che non sovrastimi mai il costo reale per raggiungere il goal.                                    |
| **Consistenza**                        | Una proprietà di un'euristica che garantisce che il costo stimato non diminuisca mai quando si passa da un nodo a uno dei suoi successori.     |
| **Uniform Cost Search (UCS)**          | Una strategia di ricerca che esplora i nodi in base al costo accumulato fino a quel punto.                                                     |
| **Greedy Best-First Search**           | Una strategia di ricerca che esplora i nodi in base alla loro prossimità al goal, in termini di euristica.                                     |
| **A* Search**                          | Una strategia di ricerca che combina il costo accumulato con una stima del costo futuro, utilizzando un'euristica.                             |
| **Weighted A* Search**                 | Una variante di A* Search che assegna un peso all'euristica per controllare l'importanza della stima del costo futuro.                         |
| **Recursive Best-First Search (RBFS)** | Una strategia di ricerca che riduce il numero di nodi memorizzati nella frontiera, utilizzando una tecnica ricorsiva.                          |
| **Tree Search**                        | Una strategia di ricerca che esplora lo spazio degli stati come un albero, senza memorizzare gli stati già visitati.                           |
| **Graph Search**                       | Una strategia di ricerca che esplora lo spazio degli stati come un grafo, memorizzando gli stati già visitati per evitare di rivisitarli.      |
| **Closed list**                        | Un insieme che contiene gli stati già visitati durante una ricerca Graph Search.                                                               |
| **Iterative Deepening**                | Una strategia di ricerca che risolve il problema della ricerca in profondità evitando che si perda in cammini infiniti.                        |
| **Bidirectional Best-First**           | Una strategia di ricerca che esegue due ricerche simultaneamente: una in avanti dallo stato iniziale e una all'indietro dagli stati obiettivo. |

## Problema di Ricerca: Definizione
Un problema di ricerca è definito da:
* **Spazio degli stati:** Insieme di tutti gli stati possibili.
* **Funzione di successione:** Definisce le azioni possibili e i costi per passare da uno stato all'altro.
* **Stato iniziale:** Stato di partenza.
* **Goal:** Funzione che determina se uno stato soddisfa la condizione di obiettivo.

Una **soluzione** è una sequenza di azioni che porta dallo stato iniziale a uno stato che soddisfa il goal.

### Differenza tra Search State e World State
* **Search state:** Contiene solo le informazioni rilevanti per la pianificazione.
* **World state:** Contiene tutti i dettagli dello stato del mondo.

### Rappresentazione dello Spazio degli Stati
* **State space graph:** Rappresentazione grafica dello spazio degli stati, con nodi che rappresentano gli stati e archi che rappresentano le azioni.
* **Search tree:** Rappresentazione ad albero dello spazio degli stati, con la radice che rappresenta lo stato iniziale e i rami che rappresentano le azioni.

### Algoritmo di Ricerca
La ricerca consiste nell'esplorare lo spazio degli stati a partire dallo stato iniziale, generando stati successivi mediante la funzione di successione.

È importante evitare i loop memorizzando i nodi già visitati. 

## Ricerca ad Albero: Algoritmo Generale

La ricerca ad albero è un processo iterativo che esplora lo spazio degli stati di un problema per trovare una soluzione. Funziona in questo modo:

1. **Espansione dello stato corrente:** Si generano tutti i possibili stati successivi allo stato corrente e si inseriscono in un insieme chiamato **frontiera**.
2. **Selezione di un nodo:** Si sceglie un nodo dalla frontiera in base a una strategia specifica.
3. **Verifica del goal:** Se il nodo selezionato soddisfa la condizione di obiettivo, la ricerca ha successo e termina.
4. **Espansione del nodo:** Se il nodo selezionato non soddisfa il goal, si espande il nodo, ovvero si generano i suoi stati successivi e si aggiungono alla frontiera.
5. **Fine della ricerca:** Se la frontiera è vuota, ovvero non ci sono più nodi da espandere, la ricerca fallisce.

### Frontiera
La frontiera è l'insieme dei nodi che sono stati generati ma non ancora esplorati. Può essere implementata in diversi modi:

* **Coda di priorità:** I nodi vengono ordinati in base a una funzione di costo, e il nodo con il costo minore viene selezionato per l'espansione.
* **Coda FIFO (First In First Out):** I nodi vengono inseriti e estratti dalla frontiera in ordine di arrivo.
* **Coda LIFO (Last In First Out):** I nodi vengono inseriti e estratti dalla frontiera in ordine inverso di arrivo.

### Strategie di Ricerca Non Informata
Le strategie di ricerca non informata non utilizzano informazioni sulla distanza dal goal per guidare la ricerca. Le prestazioni di queste strategie possono essere valutate in base a:

* **Completezza:** Se esiste una soluzione, l'algoritmo la trova.
* **Ottimalità:** L'algoritmo trova la soluzione con il costo minimo.
* **Complessità temporale e spaziale:** Le risorse computazionali necessarie per eseguire l'algoritmo. 

### Ricerca Best-First

La ricerca Best-First è una strategia generale che utilizza una funzione di costo f(n) per valutare i nodi. La frontiera è una coda con priorità, dove il nodo con il valore f(n) minimo viene selezionato per l'espansione. 

La scelta della funzione f(n) determina il comportamento dell'algoritmo.

### Ricerca in Ampiezza

La ricerca in ampiezza è una strategia che esplora lo spazio degli stati in modo sistematico, espandendo prima la radice, poi i suoi successori, e così via. 

È utile quando tutte le azioni hanno lo stesso costo. Può essere implementata come una ricerca Best-First con f(n) pari alla profondità del nodo. Un'implementazione più efficiente utilizza una coda FIFO.

La ricerca in ampiezza è completa anche su spazi infiniti e, se le azioni hanno lo stesso costo, è anche ottimale. La complessità spaziale e temporale è $O(bd),$ dove d è la profondità e b è il numero di successori di ogni nodo.

### Ricerca in Profondità

La ricerca in profondità esplora lo spazio degli stati espandendo sempre il nodo con la maggiore profondità. Può essere implementata come una ricerca Best-First con f(n) pari all'opposto della profondità. Un'implementazione più efficiente utilizza una coda LIFO.

La ricerca in profondità è completa solo se la struttura è aciclica e non è ottimale. La complessità spaziale è $O(b · m) $e quella temporale è $O(bm)$, dove b è il branching factor e m è la massima profondità dell'albero. 

## Strategie di Ricerca Non Informata: Sintesi

### Iterative Deepening

L'Iterative Deepening è una strategia che risolve il problema della ricerca in profondità evitando che si perda in cammini infiniti. Funziona eseguendo una ricerca in profondità (DFS) con un limite di profondità crescente. Se la ricerca fallisce, il limite di profondità viene incrementato e la ricerca viene ripetuta.

L'algoritmo è completo ma non ottimale. La complessità spaziale è $O(b · d)$ e quella temporale è $O(b^d)$, dove b è il branching factor e d è la profondità massima.

### Bidirectional Best-First

La Bidirectional Best-First è una strategia che esegue due ricerche simultaneamente: una in avanti dallo stato iniziale e una all'indietro dagli stati obiettivo. Le due ricerche utilizzano una strategia Best-First e si incontrano quando trovano un nodo comune.

L'algoritmo è completo e può essere ottimale a seconda della strategia utilizzata. La complessità spaziale e temporale è $O(b^\frac{d}{2})$.

---
### Strategie di Ricerca Informata o Euristica
- **Caratteristiche**:
  - Utilizzano **informazioni aggiuntive** (euristiche) per stimare la distanza o il costo per raggiungere l'obiettivo.
  - L'euristica è rappresentata dalla funzione $h(n)$, che fornisce una stima del costo dal nodo corrente all'obiettivo.
  - Si combinano il costo del passato (percorso già effettuato) con il costo stimato del futuro, bilanciando efficienza e qualità della soluzione.
  - La funzione di valutazione utilizzata è generalmente: $f(n) = g(n) + h(n)$, dove:
    - $g(n)$ è il costo accumulato per arrivare al nodo $n$.
    - $h(n)$ è una stima del costo per arrivare all’obiettivo.
  - Queste strategie cercano di **ridurre il tempo di ricerca**, a scapito talvolta della qualità della soluzione, attraverso un **rilassamento** del problema.

### Uniform Cost Search (UCS)
- **Descrizione**:
  - È una variante della **Breadth-First Search** (ricerca in ampiezza) che esplora i nodi in base al **costo accumulato** fino a quel punto.
  - Utilizza la funzione di costo: $f(n) = g(n)$, dove $g(n)$ è il costo del percorso dal nodo iniziale a $n.$
- **Caratteristiche**:
  - L'algoritmo espande sempre il nodo con il costo minore rispetto al nodo iniziale.
  - **Ottimale**: garantisce di trovare la soluzione con il costo minimo, a condizione che i costi degli archi siano positivi.
  - **Completo** se lo spazio di ricerca è finito e tutti i costi degli archi sono superiori a zero.
  - L'algoritmo diventa **inefficiente** su spazi di ricerca con costi piccoli o nulli, poiché potrebbe esplorare molte alternative con costi bassi prima di trovare la soluzione ottimale.

### Greedy Best-First Search
  - Una strategia di ricerca che esplora i nodi in base alla loro **prossimità** all’obiettivo, in termini di euristica.
  - Usa una funzione di costo che considera **solo l'euristica**: $f(n) = h(n)$.
- **Caratteristiche**:
  - Si basa esclusivamente sull’**euristica**, cercando di **avvicinarsi il più possibile** all'obiettivo a ogni passo.
  - **Non garantisce l'ottimalità** della soluzione poiché ignora il costo accumulato (ossia $g(n)$).
  - Completo solo in spazi di ricerca finiti.
  - Complessità spaziale e temporale: $O(|v|)$, dove $|v|$ è il numero di nodi, ma può migliorare a $O(b \cdot m)$, con $b$ il branching factor e $m$ la profondità massima dell'albero.
---
### Riepilogo delle Funzioni di Valutazione
- **Greedy Best-First Search**: $f(n) = h(n)$ (solo euristica, non ottimale).
- **Uniform Cost Search (UCS)**: $f(n) = g(n)$ (solo costo passato, ottimale).
- **A\***: $f(n) = g(n) + h(n)$ (costo accumulato + stima futura, ottimale se $h(n)$ è ammissibile).
#### Confronto delle Strategie:
- **Greedy Best-First**: Veloce ma non garantisce la soluzione ottima.
- **UCS**: Lenta ma garantisce l'ottimalità.
- **A***: Unisce i vantaggi di entrambe, garantendo l'ottimalità in modo più efficiente (se l'euristica è ben scelta).
--- 
### A* Tree Search
- **Funzione di costo**: $f(n) = g(n) + h(n)$  (combina UCS e Greedy Search).
  - $g(n)$: costo del percorso dal nodo iniziale a $n$ (guarda al passato).
  - $h(n)$: euristica, stima del costo dal nodo $n$ all'obiettivo (guarda al futuro).
  - - **Strategia**: best-first.
- **Caratteristica**: L'euristica introduce un rilassamento del problema, migliorando i tempi di risoluzione ma riducendo la bontà della soluzione (trade-off).
- **Caso senza euristica**: $f(n) = g(n)$, e l'algoritmo diventa una *Uniform Cost Search (UCS)*.
- **Caratteristica chiave**: come nella UCS, un nodo viene espanso prima di verificare se soddisfa il goal, per gestire potenziali cammini con costo minore.
---
### Proprietà delle Euristiche
Esistono due proprietà principali per le euristiche: **ammissibilità** e **consistenza**.

#### 1. Ammissibilità
- **Definizione**: Una euristica $h(n)$ è **ammissibile** se **non sovrastima mai** il costo per raggiungere un obiettivo.
- **Condizioni**:
  - Per ogni nodo $n$, la condizione deve essere $h(n) \leq h^*(n)$, dove:
    - $h^*(n)$ è il costo reale del cammino ottimo da $n$ al goal.
    - $h(goal) = 0$, cioè il costo per raggiungere l'obiettivo dal goal stesso è nullo.
- **Conseguenza**: Se $h(n)$ è ammissibile, l'algoritmo A\* trova sempre una soluzione ottima.

#### 2. Consistenza
- **Definizione**: Una euristica $h(n)$ è **consistente** (o **monotona**) se, per ogni nodo $n$ e suo successore $n'$ generato da un'azione $a$, la seguente condizione è soddisfatta:
  $$
  h(n) \leq cost(n, a, n') + h(n')
 $$
  Dove:
  - $cost(n, a, n')$ è il costo per passare da $n$ a $n'$ tramite l'azione $a$.
- **Relazione tra ammissibilità e consistenza**: 
  - Ogni euristica **consistente** è anche **ammissibile**.
  - Il contrario **non** è vero: un'euristica ammissibile non è necessariamente consistente.
---
### Completezza di A* Tree Search
- **Completezza**: L'algoritmo A\* è completo, ovvero **trova sempre una soluzione** se esiste, a condizione che:
  - Lo spazio degli stati abbia una soluzione o sia finito.
  - I costi delle azioni siano **positivi** e maggiori di una soglia $\epsilon > 0$.

---
### Ottimalità di A* Tree Search
- **Premessa**: Se l'euristica $h(n)$ è **ammissibile**, l'algoritmo A\* è **ottimale** in termini di costo. Vediamo come dimostrarlo.

#### Passi per dimostrare l'ottimalità:

1. **Supposizione iniziale**:
   - Consideriamo un albero in cui:
     - $A$ è il nodo **ottimo**.
     - $B$ è un nodo obiettivo **non ottimo**.
   - Sappiamo che:
     $$
     f(A) = g(A) + h(A) < f(B) = g(B) + h(B)
     $$
     Dove $f(n) = g(n) + h(n)$ è la funzione di costo valutata per ogni nodo.

2. **Nodo antenato**:
   - Consideriamo un nodo **antenato** di $A$, indicato con $n$, che era presente nella frontiera contemporaneamente a $B$.
   - Dobbiamo dimostrare che $n$ (che si trova sul cammino ottimo) venga sempre **estratto prima** di $B$.

3. **Definizione di $f(n)$**:
   - Per il nodo $n$, dal momento che $f(n) < f(A)$, abbiamo:
     $$
     f(n) = g^*(n) + h(n)
     $$
     Dove $g^*(n)$ è il costo effettivo per raggiungere $n$ e $h(n)$ è l'euristica ammissibile.

4. **Ammissibilità dell'euristica**:
   - Siccome $h(n)$ è un'euristica ammissibile, vale che:
     $$
     \forall n, h(n) \leq h^*(n)
     $$
     Ciò significa che $h(n)$ è un **lower bound** sul costo effettivo per raggiungere l'obiettivo.

5. **Conseguenza**:
   - Quindi possiamo scrivere:
     $$
     f(n) = g^*(n) + h(n) \leq g^*(n) + h^*(n) = f(A)
     $$
	Il nodo $A$ è il **nodo ottimale**, quindi $f(A)$ rappresenta il valore $f$ per il cammino ottimale. In particolare, sappiamo che:
   $$
   f(A) = g^*(A) + h^*(A)
   $$
   - $g^*(A)$ rappresenta il **costo ottimale** per raggiungere il nodo $n$ dal nodo iniziale (in altre parole, il costo minimo possibile).
   - $h^*(A)$ rappresenta il **costo esatto** per raggiungere il goal dal nodo $n$.
   
   Siccome il nodo $A$ è ottimo, abbiamo:
   $$
   f(A) < f(B)
   $$
   Dove $B$ è un nodo obiettivo non ottimale.
   Dalla relazione precedente possiamo dedurre che:
   $$
   f(n) \leq f(A)
   $$
   E, poiché $f(A) < f(B)$, otteniamo:
   $$
   f(n) \leq f(A) < f(B)
   $$

6. **Conclusione**:
   - Questo implica che **$n$ viene sempre estratto prima di $B$** dalla frontiera.
   - Dopo l'espansione di $n$, i suoi figli verranno inseriti nella frontiera. Tra questi figli ci sarà un nodo $n'$ per il quale si applica lo stesso ragionamento.
   - Di conseguenza, **tutti gli antenati di $A$** (incluso $A$) verranno sempre estratti **prima di $B.$**
**Tutti i nodi** sul percorso ottimo (compreso $A$) vengono espansi prima di $B$, garantendo che l'algoritmo A\* trovi la soluzione ottima prima di espandere nodi subottimali come $B$.
### Interpretazione

In pratica, questo passaggio della dimostrazione mostra che A* espanderà sempre i nodi che si trovano lungo il cammino ottimale prima di espandere i nodi che appartengono a cammini sub-ottimali. Questo è garantito dall'ammissibilità dell'euristica, che assicura che il valore $f(n)$ per i nodi sul cammino ottimale sarà sempre minore o uguale a $f(B)$, dove $B$ è un nodo sub-ottimale. 

In altre parole, l'algoritmo A* non sceglierà mai un cammino peggiore finché esiste un cammino migliore, garantendo così l'ottimalità della soluzione.

---
### A\* Graph Search
- Nella **Tree Search**, gli stati possono essere generati più volte, esplorando ripetutamente lo stesso stato attraverso percorsi diversi. Questo peggiora le prestazioni ma riduce la memoria utilizzata (poiché non vengono memorizzati gli stati già visitati).
- La **Graph Search** migliora le prestazioni memorizzando gli **stati visitati** in un set chiamato **closed list**. Prima di espandere un nodo, si verifica se il suo stato è già presente nel set. Se uno stato è già stato visitato, non viene riesplorato.
  
- **Problema**:
  - Potrebbe esserci un nodo che presenta un cammino meno costoso rispetto a uno stato già visitato. Per garantire l'**ottimalità** in questo caso, è necessario che l'euristica sia **ammissibile** e **consistente**.
  
- **Relazione dalla consistenza**:
  - Consideriamo due nodi $a$ e $b$ su un certo cammino. 
  - Ricordiamo che l'euristica $h(n)$ è detta **consistente** se soddisfa la seguente proprietà per ogni coppia di nodi $a$ e $b$, dove $b$ è un successore di $a$:
$$
h(a) \leq g(b) - g(a) + h(b)
$$
Da cui segue che:
    $$
    h(a) + g(a) \leq g(b) + h(b) \quad \text{ovvero} \quad f(a) \leq f(b)
    $$
  - Questa relazione è fondamentale per dimostrare l'**ottimalità** di A\* Graph Search
  - La consistenza garantisce che il valore f(n) non diminuisca mai quando si passa da un nodo a uno dei suoi successori. Questo significa che, se lo stato di un nodo è stato già visitato e salvato nella "closed list", è impossibile che in futuro venga trovato un cammino più economico verso quello stesso stato.

### Ottimalità di A\* Graph Search
- **Ipotesi**:
  - Sia $n$ un nodo sul **cammino ottimo**(percorso che porta al goal con costo minimo) e $n'$ un nodo su un **cammino non ottimo** che si riferiscono allo stesso stato.
  - L'euristica $h(n)$ è **ammissibile** e **consistente**.
  - L'obiettivo è dimostrare che, se A* segue queste condizioni, espanderà sempre i nodi sul **cammino ottimo** prima dei nodi su cammini subottimali.
  
- **Assurdo**:
  - Supponiamo per assurdo che $n'$ venga estratto prima di $n$, e che lo stato di $n'$ venga inserito nei **closed states**.
	  - (Significa che il costo complessivo stimato per $n'$ è stato giudicato minore rispetto a quello di $n$)
  - Supponiamo che $p$ sia un antenato (non necessariamente padre) di $n$ che era nella frontiera quando è stato scelto $n'$.
  
- **Passaggi logici**:
  1. Dalla consistenza sappiamo che:
     $$
     f(p) \leq f(n)
     $$
  2. Inoltre, poiché $n$ e $n'$ si riferiscono allo **stesso stato**, l'euristica deve essere identica per entrambi:
     $$
     h(n) = h(n')
     $$
  3. Poiché $n$ è sul cammino ottimo e $n'$ è su un cammino non ottimo, sappiamo che $g(n') > g(n)$, ma abbiamo anche che $h(n) = h(n')$. Ne consegue che $f(n) < f(n')$, e dunque:
     $$
     f(p) \leq f(n) < f(n')
     $$

- **Contraddizione**
	Se il nodo $p$ era nella frontiera al momento dell'espansione di $n'$, allora avrebbe dovuto essere espanso **prima** di $n'$, poiché il suo valore $f(p)$ è minore di $f(n')$. L'espansione di $p$ avrebbe generato i suoi successori, inclusi i nodi antenati di $n$, e questo processo sarebbe continuato fino a espandere $n$ prima di $n'$, il che contraddice l'ipotesi che $n'$ sia stato espanso prima di $n$, dimostrando che A* Graph Search espande sempre i nodi sul cammino ottimo prima di quelli subottimali, garantendo così l'**ottimalità**.

### Riassunto:
- La **Graph Search** salva memoria ma può ignorare stati con cammini meno costosi se non correttamente gestita.
- La **consistenza** dell'euristica garantisce che l'algoritmo esplori i nodi sul cammino ottimo prima di quelli subottimali, mantenendo l'**ottimalità** anche in A* Graph Search.
- ---
### Teorema: Consistenza implica Ammissibilità
- **Obiettivo**: Dimostrare che se un'euristica $h(n)$ è **consistente**, allora è anche **ammissibile**.
- **Metodo**: La dimostrazione viene condotta per **induzione**.
#### Caso base: stato $n = 1$ (dista 1 dal goal):
1. Dall'ipotesi di **Consistenza** sappiamo che: 
   $$
   \forall \text{azione } a, \quad h(n) \leq costo(n, a, goal) + h(goal)
   $$
2. **Poiché** per definizione $h(goal) = 0$, segue:
   $$
   h(n) \leq costo(n, a, goal)
   $$
3. Consideriamo ora l'azione ottima $a*$ che porta al goal con il costo minimo $h(n)$. Allora possiamo scrivere:
   $$
   h(n) \leq costo(n, a^*, goal) = h^*(n)
   $$
	Ovvero, l'euristica $h(n)$ non sovrastima il costo reale per raggiungere il goal, confermando la **ammissibilità** dell'euristica per lo stato $n$ che dista 1 dal goal.
4. **Conclusione del caso base**:
   $$
   h(n) \leq h^*(n)
   $$
   - Abbiamo dimostrato che l'euristica è **ammissibile** per lo stato $n$ a distanza 1 dal goal.

#### Ipotesi induttiva:
- Supponiamo che il teorema valga per tutti gli stati $n^*$ che distano $k$ dal goal: 
  $$
  h(n^*) \leq h^*(n^*)
  $$
  - In altre parole, l'euristica è ammissibile per gli stati a distanza $k$ dal goal.

#### Passo induttivo (stato $n$ a distanza $k+1$ dal goal):
1. Dall'ipotesi di**Consistenza** deriva che: 
   $$
   \forall a, \quad h(n) \leq costo(n, a, n') + h(n')
   $$
   - Dove $n'$ è un nodo che dista 1 da $n$.
   
2. **Per la migliore azione $a^*$ e il miglior nodo $n'^*$**:
   $$
   h(n) \leq costo(n, a^*, n'^*) + h(n'^*)
   $$
   
3. **Applicando l'ipotesi induttiva**: Poiché l'euristica è ammissibile per gli stati che distano k passi dal goal, come $n'^*$, possiamo utilizzare l'ipotesi induttiva per sostituire $h(n'*)$ con $h^*(n'*)$, ovvero il costo effettivo minimo per raggiungere il goal partendo da $n'^*$:
   $$
   h(n'^*) \leq h^*(n'^*)
   $$

4. **Sostituendo**:
   $$
   h(n) \leq costo(n, a^*, n'^*) + h^*(n'^*) = h^*(n)
   $$
   
5. **Conclusione del passo induttivo**:
   $$
   h(n) \leq h^*(n)
   $$
   - L'euristica è ammissibile per lo stato $n$ a distanza $k+1$ dal goal.

---
### Weighted A∗ Search
- **Obiettivo**: Ridurre il numero di nodi espansi accettando soluzioni subottime ma soddisfacenti.
- **Modifica della funzione di costo**:
  $$
  f(n) = g(n) + w \cdot h(n)
  $$
  Dove $w \in (1, \infty)$, controlla l'importanza dell'euristica.
  
- **Interpretazione**:
  - **$w = 0$** → si ottiene **Uniform Cost Search** (UCS).
  - **$w = 1$** → si ottiene **A∗ Search**.
  - **$w \to \infty$** → si ottiene **Greedy Search**.
  
- **Effetti**:
  - **Più euristica**: Aumentando $w$, si esplorano meno nodi, ma si potrebbe ottenere una soluzione **non ottima**.
  - **Trade-off**: Minor tempo e spazio, ma soluzioni meno accurate.

---
### Recursive Best-First Search (RBFS)
- **Obiettivo**: Ridurre il numero di nodi memorizzati nella frontiera, come alternativa alla ricerca A∗.
  
- **Funzionamento**:
  - Simula una **Best-First Search** e si comporta simile a una **DFS**.
  - Tiene traccia del valore **$f$** del miglior cammino alternativo dai nodi antenati.
  - Se il nodo corrente supera un certo limite (ossia tutti i suoi figli hanno un valore $f$ maggiore), la ricerca torna indietro al cammino alternativo.
  - Durante il ritorno, aggiorna il valore $f$ di ogni nodo con il miglior valore $f$ dei suoi figli.
  
- **Problema**: **Rigenerazione di nodi**: i nodi vengono spesso ricalcolati.

---
### Considerazioni sulle Euristiche
- **Necessità di valutazioni quantitative**:
  - Le euristiche devono fornire una stima **numerica** e non solo qualitativa.
  
- **Motivo**:
  - **Ricerca Best-First**: La valutazione qualitativa può essere sufficiente (confronto tra stati).
  - **Ricerca A∗**: Necessita di una combinazione di criteri numerici per quantificare quanto uno stato sia preferibile rispetto a un altro.
