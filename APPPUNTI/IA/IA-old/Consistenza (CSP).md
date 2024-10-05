### **1. K-Consistency**

- **1-Consistency (Node-Consistency)**:
  - Consistenza a livello di singola variabile.
  - ==Ogni valore nel dominio di una variabile soddisfa i vincoli unari associati a quella variabile==.

- **2-Consistency (Arc-Consistency)**:
  - ==Ogni assegnamento consistente a una variabile può essere esteso all'altra variabile in una coppia di variabili (arco).==
	  - Per due variabili legate da un vincolo (un "arco" tra di loro), se assegni un valore a una variabile, esiste almeno un valore nell'altra variabile che soddisfa il vincolo. In altre parole, ==un assegnamento è consistente se, dato un valore per la prima variabile, puoi sempre trovare un valore per la seconda variabile che rispetti il vincolo che le collega.==
  - Se la struttura del problema è **aciclica**, forzare l'arc-consistency garantisce l'esistenza di una soluzione.
  - **Costo computazionale:** O(n²), dove *n* è il numero di variabili.

- **K-Consistency**:
  - Estensione della 2-consistency a k variabili.
  - ==Ogni assegnamento consistente a *k - 1* variabili può essere esteso alla *k-esima* variabile.==
  - Utilizzato per risolvere problemi in maniera efficiente.
  - **Costo computazionale:** O(n · d^k), dove *n* è il numero di variabili e *d* è la dimensione massima dei domini.
  - Se *k* è fissato, il costo è polinomiale.

---

### **2. Euristiche per la Risoluzione dei CSP**

- **Variable Ordering**:
  - **Minimum Remaining Values (MRV)**:
    - ==Si sceglie la variabile con il minor numero di valori legali rimasti nel dominio.==
    - Questo approccio permette di fallire velocemente (fail-fast) se non ci sono valori consistenti, accelerando la ricerca.
  
- **Value Ordering**:
  - **Least Constraining Value**:
    - ==Si seleziona il valore che restringe meno i domini delle variabili rimanenti, riducendo il numero di vincoli che devono essere soddisfatti.==

---

### **3. Tecniche di Risoluzione con Assegnamento Completo**

- ==Invece di partire da un assegnamento vuoto, si parte da un **assegnamento completo** (anche se non consistente) e si cercano correzioni:==
  
  - Seleziona una variabile in conflitto.
  - Modifica il valore della variabile cercando di minimizzare i conflitti.
  
- **Heuristiche per la correzione**:
  - **Minimo Conflitto**: seleziona il valore che viola il minor numero di vincoli.
  - **Hill Climbing** o **Simulated Annealing** possono essere utilizzati per trovare la soluzione ottimale partendo da assegnamenti completi e migliorando progressivamente.

---

### **4. Problema del Punto Critico**

- ==In problemi bilanciati (stesso numero di variabili e vincoli), può verificarsi una **transizione di fase** chiamata **punto critico**, in cui gli algoritmi euristici potrebbero fallire.==
- Questo è un problema generale per molti algoritmi euristici, non solo per quelli basati su minimo conflitto.
