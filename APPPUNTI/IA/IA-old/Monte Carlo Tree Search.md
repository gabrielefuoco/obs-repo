La **Monte Carlo Tree Search** (MCTS) è un algoritmo usato in giochi complessi dove è difficile definire buone euristiche o quando il **branching factor** (numero di mosse possibili) è troppo grande. Si usa quando non si può esplorare a fondo l'intero albero di gioco.

### Come funziona MCTS:
1. **Selezione**: Si parte dalla radice (stato attuale del gioco) e, seguendo una politica di selezione, si sceglie un figlio fino a raggiungere un nodo foglia (un nodo non ancora esplorato).
   
2. **Espansione**: Si crea un nuovo nodo figlio a partire dalla foglia, che rappresenta un possibile nuovo stato del gioco.

3. **Simulazione**: Da questo nuovo nodo, si gioca una partita simulata fino a un risultato finale (vittoria, sconfitta o pareggio) scegliendo mosse casuali o basate su una politica di simulazione.

4. **Retropropagazione**: Il risultato della simulazione viene usato per aggiornare tutti i nodi visitati fino alla radice, migliorando la stima della qualità di ogni mossa.

### Politiche
- **Politica di simulazione**: Sceglie le mosse da simulare, possibilmente privilegiando mosse più promettenti.
- **Politica di selezione**: Decide quali nodi esplorare bilanciando tra l’esplorare nuove mosse o sfruttare quelle che sembrano già vincenti.

Alla fine, la mossa che ha il maggior numero di simulazioni viene scelta come la migliore.