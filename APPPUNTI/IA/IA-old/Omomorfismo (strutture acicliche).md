Il problema dell'omomorfismo su strutture acicliche ==si riferisce alla decisione se esiste un **omomorfismo** (una mappa che preserva la struttura) tra due strutture relazionali, chiamate \(A\) e \(B\), che rappresentano vincoli e relazioni di un problema CSP== (Constraint Satisfaction Problem). La complessità di risolvere questo problema su strutture **acicliche** (che non contengono cicli) è **O(||A|| · ||B|| · log ||B||)**, dove \(\|A\|\) e \(\|B\|\) rappresentano la dimensione delle strutture.

### Procedura

Per risolvere il problema su strutture acicliche, si utilizza un **algoritmo di programmazione dinamica**, in particolare l'algoritmo di **Yannakakis**. 
==L'algoritmo di Yannakakis trasforma il problema su strutture acicliche in una serie di operazioni efficienti che permettono di risolvere il problema senza tornare indietro e di rappresentare le soluzioni in modo compatto.==

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
   - Il filtraggio viene eseguito ordinando i valori delle relazioni e poi effettuando una "merge" tra le relazioni. Una singola operazione costa **n · log n**, dove \(n\) è il numero di elementi nelle relazioni.
   - Il numero totale di operazioni è proporzionale al numero dei vincoli.

### Soluzione Backtrack-free

- ==Una volta effettuato il filtraggio, si ottiene una **struttura filtrata** che consente di trovare una soluzione senza la necessità di eseguire il backtracking== (ovvero, senza dover tornare indietro per provare nuove soluzioni). Questo riduce enormemente la complessità della ricerca.
- Anche se il numero di soluzioni può essere esponenziale, l'algoritmo offre un modo polinomiale per:
  - Decidere se esiste una soluzione.
  - Calcolare una soluzione.
  - Trovare tutte le soluzioni.
