- **Algoritmo Base**: 
  - Sceglie una variabile non assegnata.
  - Prova tutti i valori del dominio di quella variabile.
  - Per ogni valore, tenta di estendere l'assegnamento con una chiamata ricorsiva.
  - Se la chiamata ha successo, restituisce la soluzione; altrimenti, riporta l'assegnamento allo stato precedente e prova un nuovo valore.
  - Se nessun valore funziona, restituisce fallimento.

- **Miglioramenti**:
  - **Forward Checking**: 
    - ==Filtra i valori dei domini delle variabili non assegnate, eliminando i valori che violano i vincoli con gli assegnamenti futuri.==
    - Garantisce che i valori assegnati siano consistenti con i vincoli.
  
  - **Arc Consistency**: 
    - ==Verifica la consistenza di tutti i vincoli tra coppie di variabili.==
    - Un arco \( x -> y \) è consistente se, per ogni valore di \( x \), esiste un valore di \( y \) che soddisfa il vincolo.
    - Se un valore \( x \) non ha un corrispondente \( y \) valido, viene eliminato dal dominio di \( x \).

- **Propagazione**:
  - Ottimizza il backtracking forzando l'arc consistency: ==quando un vincolo viene verificato, la propagazione si estende agli altri vincoli collegati, migliorando l'efficienza della ricerca.==