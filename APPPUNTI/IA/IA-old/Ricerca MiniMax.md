La **Ricerca MiniMax** è un ==algoritmo utilizzato in intelligenza artificiale per prendere decisioni ottimali in giochi a due giocatori a somma zero==, come scacchi, dama, e tris. L'obiettivo dell'algoritmo è determinare la mossa migliore per un giocatore assumendo che l'avversario giochi in modo ottimale.

### Principio di Base

L'algoritmo MiniMax si basa su un albero di decisioni, dove:

- **Nodi MAX:** Rappresentano il turno del giocatore che sta cercando di ==massimizzare il proprio punteggio.==
- **Nodi MIN:** Rappresentano il turno dell'avversario, che cerca di ==minimizzare il punteggio del giocatore MAX.==

L'algoritmo esplora l'albero di gioco per determinare la mossa ottimale, valutando i possibili risultati di ciascuna mossa, ==assumendo che entrambi i giocatori giochino al meglio delle loro capacità.==

### Come Funziona l'Algoritmo MiniMax

1. **Costruzione dell'Albero di Gioco:**
   - L'albero di gioco viene costruito a partire dalla situazione attuale (nodo radice), con ==ogni livello dell'albero che rappresenta una possibile mossa.== I livelli alternano tra il giocatore MAX e il giocatore MIN.

2. **Valutazione delle Foglie:**
   - Alla base dell'albero ci sono i ==nodi foglia, che rappresentano le possibili situazioni finali== del gioco (o una situazione valutabile se l'albero non viene esplorato completamente). ==Ogni nodo foglia viene valutato con una funzione di valutazione che stima il vantaggio per il giocatore MAX.==

3. **Propagazione dei Valori:**
   - Partendo dalle foglie, i valori vengono propagati verso l'alto:
     - **Nodi MAX:** Ogni nodo MAX sceglie il valore massimo tra i suoi figli, poiché MAX cerca di ottenere il massimo punteggio.
     - **Nodi MIN:** Ogni nodo MIN sceglie il valore minimo tra i suoi figli, poiché MIN cerca di minimizzare il punteggio di MAX.
     
1. **Scelta della Mossa Ottimale:**
   - Una volta propagati i valori fino alla radice, il giocatore MAX sceglie la mossa che corrisponde al valore massimo alla radice dell'albero. Questa mossa è considerata la migliore scelta possibile, assumendo che l'avversario giochi in modo ottimale.

### Esempio di Ricerca MiniMax

Supponiamo di essere al turno di MAX in un gioco di tris, con la seguente configurazione:

```
 X | O | X
---+---+---
 O | X |  
---+---+---
   |   | O
```

Il giocatore MAX (X) deve decidere dove piazzare la sua prossima X. Il MiniMax esplorerà le possibili mosse:

1. **Possibili Mosse di MAX:** (Mettere X in 6, 7, 8 o 9).
2. **Risposte di MIN (O):** Per ogni mossa di MAX, MIN sceglierà la sua mossa ottimale.
3. **Valutazione delle Situazioni:** Alla fine di ogni possibile sequenza di mosse, si assegna un punteggio, ad esempio +1 per una vittoria di MAX, -1 per una vittoria di MIN, e 0 per un pareggio.
4. **Propagazione:** MAX sceglie la mossa che minimizza le possibilità di perdere e massimizza le possibilità di vincere.

### Complessità e Limitazioni

- **Complessità Temporale:** Nel caso peggiore, la complessità temporale di MiniMax è $$(O(b^d)), $$dove ==(b) è il branching factor (numero medio di mosse per stato) e (d) è la profondità dell'albero. ==Questo può diventare proibitivo per giochi complessi come gli scacchi.

- **Complessità Spaziale:** L'algoritmo richiede anche una memoria pari a$$ (O(b * d)) $$per mantenere l'albero di gioco.

- **Limitazioni:** ==La Ricerca MiniMax diventa inefficiente per giochi con alberi di ricerca molto profondi. In questi casi, è comune usare potature come l'**Alpha-Beta Pruning** per ridurre il numero di nodi esplorati.
==
### Alpha-Beta Pruning

L'**Alpha-Beta Pruning** è un'ottimizzazione dell'algoritmo MiniMax che riduce il numero di nodi da esaminare nell'albero di gioco, senza influire sulla decisione finale. ==Funziona eliminando rami dell'albero che non possono influenzare la decisione finale==, permettendo così di esplorare solo le mosse più promettenti.

### Mossa

La **mossa** è semplicemente una singola azione che un giocatore può fare in un gioco. In giochi come scacchi o dama, una mossa consiste nello spostamento di un pezzo da una posizione a un'altra sulla scacchiera.

- **Esempio:** Negli scacchi, spostare un cavallo da `g1` a `f3` è una mossa.
- In un gioco a due giocatori, ogni giocatore alterna le mosse; nel contesto del MiniMax, le mosse determinano la costruzione dell'albero di gioco.

### Ply

Un **ply** rappresenta una singola mossa di uno dei due giocatori. Nella ricerca MiniMax, il termine **ply** viene utilizzato per riferirsi alla profondità di un albero di gioco, con ogni ply corrispondente a una mossa di un singolo giocatore.

- **Un ply** corrisponde a una singola mossa di un giocatore.
- **Due ply** corrispondono a una mossa del giocatore seguito da una mossa dell'avversario (quindi un turno completo di gioco).

#### Differenza tra Mossa e Ply

- **Mossa:** Rappresenta un'azione di un giocatore, indipendentemente dall'altro.
- **Ply:** Indica una mossa di un singolo giocatore in termini di profondità nell'albero MiniMax. Ogni livello di profondità dell'albero è un ply, e due ply rappresentano una mossa completa da parte di entrambi i giocatori.

### Esempio in MiniMax

Se stai simulando una partita di scacchi:

- Quando il giocatore "bianco" fa una mossa, l'albero MiniMax scende di **1 ply**.
- Quando l'avversario (giocatore "nero") risponde, scende di **un altro ply**.

Quindi, ogni ciclo completo di "bianco" e "nero" corrisponde a **2 ply**, ma rappresenta **1 turno** del gioco. Nell'implementazione di MiniMax, è comune esplorare l'albero fino a una certa profondità in ply (ad esempio, 6 ply) per decidere la migliore mossa da fare.

### Conclusione

- **Mossa**: Un'azione effettuata da uno dei giocatori.
- **Ply**: Un singolo passo di profondità nell'albero di gioco, che rappresenta una singola mossa di uno dei giocatori.

