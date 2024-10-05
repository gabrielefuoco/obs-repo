## Join Tree & Tree Decomposition
Sia l'ipergrafo **H** una struttura che rappresenta vincoli complessi tra gruppi di variabili, con ogni iper-arco collegando più di due variabili.

- **H** è un ipergrafo con variabili \( \{A, B, C, D, E\} \) e iperarchi \( \{A, B, C\}, \{A, B, D\}, \{D, E\} \). Gli **iperarchi** rappresentano insiemi di variabili che sono collegate tra loro da vincoli.

- V'  è l'insieme degli iperarchi di \( H \) (i gruppi di variabili collegati dai vincoli).

- **T** è un **Join-Tree**, una struttura che connette questi iperarchi in modo che la propagazione delle variabili segua determinate regole.

### Regole chiave:

1. Se due iperarchi \( p \) e \( q \) condividono delle variabili comuni, quelle variabili devono apparire in tutti i vertici lungo il percorso che connette \( p \) e \( q \) nell'albero \( T \). Questo garantisce che le informazioni sulle variabili comuni si propaghino lungo l'albero in modo corretto.
   
2. Una variabile che scompare in un certo punto dell'albero non può più riapparire successivamente nel percorso: una volta che la sua informazione è stata utilizzata o propagata, non la si ritrova in altre parti.

#### In breve:
Un **Join-Tree** è una struttura che organizza gli iperarchi in modo tale che ogni variabile si propaghi correttamente lungo l'albero e non venga "persa" o riutilizzata in modo errato. Questo permette una corretta gestione dei vincoli tra le variabili.

## Ipergrafi Aciclici: 
Possiamo dire che H è un ipergrafo aciclico ⇐⇒ esso ha un join-tree. 
La definizione di un ipergrafo aciclico è più potente di quella di un grafo aciclico. Mentre un grafo aciclico è un grafo senza cicli, in un ipergrafo avere dei cicli potrebbe non apportare problemi, quindi in alcuni casi si può considerare aciclico. 
Decidere se un ipergrafo è aciclico è un problema log-space-completo ed è trattabile in tempo lineare. La stessa complessità per i grafi

## Tree Decomposition 
La **tree decomposition** è un metodo che p==ermette di semplificare un problema complesso rappresentato da un grafo, suddividendolo in sottoproblemi aciclici più facili da risolvere.==

- **Obiettivo**: Prendere un grafo con cicli e trasformarlo in una struttura che può essere trattata come un **albero**, eliminando i cicli.

- **Metodo**: Raggruppiamo le variabili del problema in insiemi. Ogni insieme (o "nodo") contiene **k + 1** variabili, e questi insiemi formano gli **iperarchi** di un nuovo ipergrafo aciclico.

- **Tree decomposition**: Viene costruito un **albero** \( T \) dove ogni nodo contiene un insieme di variabili del grafo originale. Ciascun nodo è etichettato con un insieme di variabili tramite una funzione \( χ \).

### Proprietà:

1. **Copertura degli archi**: Per ogni arco del grafo originale, esiste almeno un nodo nell'albero \( T \) che contiene entrambe le variabili dell'arco.
   
2. **Proprietà di connessione**: Se una variabile \( x \) appare in più nodi dell'albero, allora deve comparire in tutti i nodi lungo il percorso che li collega.

### Tree-width:

- La **width** (larghezza) di una decomposizione è la dimensione massima dei nodi nell'albero meno uno.
- La **tree-width** del grafo è la larghezza minima tra tutte le possibili decomposizioni.

### Teorema
Sia k una costante. Calcolare una tree-decomposition di width minore o uguale di k
richiede tempo lineare. Se k non è fissato (è parte dell’input) il problema è NP-Hard.

### In breve:
La tree decomposition permette di trasformare un problema con cicli in un albero aciclico più gestibile, raggruppando le variabili. La **tree-width** misura quanto è complicato effettuare questa trasformazione.