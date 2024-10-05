L'**Alpha-Beta Pruning** (potatura alfa-beta) è una tecnica di ottimizzazione dell'algoritmo **MiniMax**, che ==consente di ridurre drasticamente il numero di nodi da esplorare nell'albero di gioco. L'obiettivo principale dell'Alpha-Beta Pruning è evitare di esplorare rami che non influenzeranno il risultato finale della decisione, migliorando così l'efficienza computazionale.==

### Principio di Base

L'idea chiave dell'Alpha-Beta Pruning è basata sul concetto che, ==se si conosce abbastanza del sottoproblema corrente, è possibile interrompere l'esplorazione di certi rami dell'albero senza compromettere la decisione finale. ==Questo permette di ridurre il numero di valutazioni dei nodi e quindi il tempo di esecuzione, mantenendo però lo stesso risultato finale dell'algoritmo MiniMax.

In MiniMax, ci sono due giocatori: il giocatore **MAX** che cerca di massimizzare il punteggio e il giocatore **MIN** che cerca di minimizzarlo. Durante l'esplorazione dell'albero di gioco, Alpha-Beta Pruning mantiene due valori:

- **Alpha (α)**: rappresenta il **miglior punteggio** che il giocatore MAX può ottenere finora. Inizia da \(-\infty\) e viene aggiornato man mano che si trova una mossa migliore.
- **Beta (β)**: rappresenta il **miglior punteggio** che il giocatore MIN può ottenere finora. Inizia da \(+\infty\) e viene aggiornato man mano che si trova una mossa peggiore per MAX.

### Come Funziona l'Alpha-Beta Pruning

Quando MiniMax esplora l'albero di gioco:

1. **Alpha (α) e Beta (β) vengono aggiornati** durante l'esplorazione dei nodi:
   - Se si trova un punteggio migliore per MAX, si aggiorna α.
   - Se si trova un punteggio migliore per MIN (ossia, peggiore per MAX), si aggiorna β.
   
2. **Potatura dei rami:** Se in qualsiasi momento si verifica che:
   - Il valore corrente di β è minore o uguale ad α (\(β \leq α\)), significa che ==il giocatore MIN non sceglierà mai quel percorso, perché ha già trovato un'opzione migliore.== In questo caso, si interrompe l'esplorazione di quel ramo dell'albero.

In altre parole, ==Alpha-Beta Pruning evita di esplorare i nodi quando si può già determinare che il percorso non sarà scelto da uno dei due giocatori.==

### Esempio di Alpha-Beta Pruning

Supponiamo di avere un albero MiniMax che stiamo esplorando per determinare la miglior mossa in un gioco a due giocatori. 

1. Il giocatore MAX cerca di massimizzare il proprio punteggio e inizia ad esplorare i rami. Supponiamo che il giocatore MIN stia cercando di minimizzare il punteggio.

2. Durante l'esplorazione di un certo ramo, MiniMax determina che MAX può ottenere un punteggio di 5 seguendo quel percorso. Questo aggiorna il valore di α a 5.

3. Più avanti, durante l'esplorazione di un altro ramo, MIN trova un percorso che darebbe un punteggio di 3 per MAX. A questo punto, MIN non è più interessato a esplorare ulteriori rami da quel punto in avanti, perché può già ottenere un punteggio migliore scegliendo un percorso diverso. Quindi, MIN decide di interrompere l'esplorazione (potatura) di quel ramo.

4. L'algoritmo continua a esplorare solo i rami che possono effettivamente influenzare la decisione, migliorando così l'efficienza.

### Vantaggi dell'Alpha-Beta Pruning

1. **Riduzione del numero di nodi esplorati:** L'algoritmo può ridurre significativamente il numero di nodi da esplorare rispetto al MiniMax standard. Nel migliore dei casi, può ridurre la complessità da $$(O(b^d))$$a
$$ (O(b^{d/2}))$$ dove \(b\) è il fattore di ramificazione e \(d\) è la profondità dell'albero.

2. **Migliore gestione del tempo:** Poiché si esplorano meno nodi, l'Alpha-Beta Pruning consente all'algoritmo di MiniMax di esplorare alberi più profondi, migliorando così la qualità delle decisioni prese in un tempo limitato.

3. **Stesso risultato di MiniMax:** Nonostante l'ottimizzazione, Alpha-Beta Pruning non modifica il risultato finale rispetto al MiniMax tradizionale. Troverà sempre la mossa ottimale se i rami vengono esplorati correttamente.

### Limiti dell'Alpha-Beta Pruning

- **Ordine di esplorazione:**  ==L'efficacia dell'Alpha-Beta Pruning dipende dall'ordine con cui i nodi vengono esplorati==. Se l'algoritmo esplora prima i rami meno promettenti, potrà effettuare meno potature. Se invece esplora prima i rami migliori, l'efficienza migliora significativamente.

- **Applicabile solo a giochi con due giocatori e perfetta informazione:** L'Alpha-Beta Pruning funziona bene in giochi come scacchi e tris, dove entrambi i giocatori hanno accesso a tutte le informazioni sullo stato del gioco. Non è applicabile a giochi con più di due giocatori o con informazioni incomplete (come il poker).

### Conclusione

L'**Alpha-Beta Pruning** è un'ottimizzazione dell'algoritmo MiniMax che riduce il numero di nodi da esplorare, migliorando l'efficienza del processo decisionale senza alterare la qualità del risultato. Grazie a questa tecnica, MiniMax può essere utilizzato per esplorare alberi di gioco più profondi e complessi, rendendolo un algoritmo molto pratico per giochi come gli scacchi e la dama.