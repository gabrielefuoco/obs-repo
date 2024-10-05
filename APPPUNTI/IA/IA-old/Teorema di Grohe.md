### Teorema di Grohe
Il **Teorema di Grohe** riguarda la risoluzione di problemi di **CSP** (Constraint Satisfaction Problem), ovvero problemi in cui bisogna assegnare valori a variabili in modo da rispettare determinati vincoli.

### Contesto del Teorema:

- **Strutture relazionali**: Abbiamo una classe \( S \) di strutture relazionali (insiemi di vincoli) con arità fissata (cioè ogni vincolo ha un numero fisso di variabili).
- **CSP (S, -)**: Si riferisce a problemi CSP dove i vincoli appartengono a \( S \), ma non c'è alcuna restrizione sul database (DB) che contiene le possibili assegnazioni dei valori.

Il teorema dice che risolvere problemi CSP appartenenti a \( S \) è **fattibile in tempo polinomiale (P-TIME)** **se e solo se** il **core** delle strutture in \( S \) ha una **treewidth (tw)** fissata (limitata da un valore \( k \)).

### Cosa significa:

1. **Treewidth (tw)**: Misura la "complessità" ciclica di un grafo. Più piccola è la treewidth, più facile è risolvere il problema CSP associato a quel grafo. Se la treewidth è fissata, il problema si può risolvere in modo efficiente.
  
2. **Core**: È la parte "essenziale" di una struttura CSP, ovvero la versione semplificata del problema che mantiene la stessa soluzione ma con meno ridondanze.

3. **Se la treewidth del core è limitata (≤ k)**: Vuol dire che la struttura è abbastanza semplice da essere risolta in tempo polinomiale.

### Algoritmo per Risolvere:

L'algoritmo procede così:
1. Per ogni gruppo di \( k+1 \) variabili, crea nuovi vincoli sulle loro possibili combinazioni di valori.
2. Filtra i valori non compatibili tra i vincoli, eliminandoli man mano, fino a quando non è possibile fare ulteriori riduzioni.
3. Se nessuna relazione è vuota, significa che abbiamo trovato una soluzione.

### Complessità:
Il costo computazionale dipende dal numero di variabili \( n \), dal numero di vincoli \( m \) e dal numero di possibili valori nel DB \( d \). Nel caso peggiore, il costo può essere esponenziale in base al valore di \( k \), ma grazie alla limitazione sulla **treewidth** possiamo esprimere questo costo come una funzione polinomiale rispetto ai parametri \( d \), \( n \), e \( k \).

### In sintesi:
Il teorema afferma che un problema di **CSP** appartenente a una certa classe \( S \) è risolvibile in tempo polinomiale **se e solo se** la complessità strutturale del problema, misurata dalla treewidth del core, è limitata.