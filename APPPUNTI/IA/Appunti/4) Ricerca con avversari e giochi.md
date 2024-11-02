
| Termine                              | Spiegazione                                                                                                                                                |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ambienti Competitivi**             | Ambienti in cui due o più agenti hanno obiettivi in conflitto tra loro, generando problemi di ricerca con avversari.                                       |
| **Giochi Deterministici**            | Giochi in cui l'esito di ogni mossa è completamente determinato dalle regole del gioco e dalle azioni dei giocatori.                                       |
| **Giochi Stocastici**                | Giochi in cui l'esito di alcune mosse è influenzato da elementi casuali, come il lancio di un dado o l'estrazione di una carta.                            |
| **Giochi a Somma Zero**              | Giochi in cui la somma dei guadagni di tutti i giocatori è sempre zero. Un giocatore può vincere solo se un altro giocatore perde.                         |
| **Giochi con Informazione Perfetta** | Giochi in cui tutti i giocatori hanno accesso a tutte le informazioni sullo stato del gioco in ogni momento.                                               |
| **Albero di Gioco**                  | Struttura ad albero che rappresenta tutte le possibili sequenze di mosse in un gioco, fino a raggiungere uno stato terminale.                              |
| **Ricerca MiniMax**                  | Algoritmo di ricerca che determina la strategia ottima in un gioco a somma zero, assumendo che entrambi i giocatori giochino in modo ottimale.             |
| **Valore MiniMax**                   | L'utilità di trovarsi in uno stato specifico, calcolata ricorsivamente considerando le mosse ottimali di entrambi i giocatori.                             |
| **Alpha-Beta Pruning**               | Tecnica di potatura dell'albero di gioco che elimina rami irrilevanti per la ricerca MiniMax, migliorando l'efficienza.                                    |
| **Test di Taglio**                   | Test utilizzato nella ricerca a profondità limitata per decidere quando interrompere la ricerca, in base alla profondità dello stato o ad altre proprietà. |
| **Valutazione Euristica**            | Stima della bontà di uno stato in un gioco, utilizzata quando la ricerca a profondità limitata non raggiunge uno stato terminale.                          |
| **Ricerca ad Albero Monte Carlo**    | Tecnica di ricerca che utilizza simulazioni casuali per costruire un albero di gioco e stimare il valore di ogni stato.                                    |
| **Politica di Selezione**            | Strategia utilizzata per scegliere il nodo da espandere nell'albero di gioco durante la ricerca ad Albero Monte Carlo.                                     |
| **Politica di Simulazione**          | Strategia utilizzata per scegliere le mosse durante le simulazioni nella ricerca ad Albero Monte Carlo.                                                    |
| **UCB1**                             | Una tipica politica di selezione che bilancia l'esplorazione di nuovi nodi con lo sfruttamento di nodi già esplorati.                                      |
| **Grafo di Gioco**                   | Rappresentazione grafica del gioco, dove i nodi rappresentano gli stati e gli archi rappresentano le mosse possibili.                                      |
| **Componente Connessa**              | Insieme di nodi in un grafo che sono collegati tra loro da un cammino.                                                                                     |

### Ambienti Competitivi
Gli ambienti competitivi sono caratterizzati dalla presenza di due o più agenti con obiettivi in conflitto. Questo genera problemi di ricerca con avversari.

## Approcci alla Teoria dei Giochi
Esistono tre approcci principali per affrontare gli ambienti multi-agente nella teoria dei giochi:

1. **Economia aggregata:** Si considera un aggregato di agenti come un'economia, senza predire l'azione del singolo agente.
2. **Ambiente non deterministico:** Gli agenti avversari sono considerati un elemento che rende l'ambiente non deterministico, senza considerare che gli avversari tentano attivamente di vincere.
3. **Alberi di gioco:** Gli avversari sono modellati esplicitamente con tecniche di ricerca basate su alberi di giochi.

### Tipi di Giochi
I giochi possono essere classificati in base a diversi criteri:
* **Deterministici vs Stocastici:** I giochi deterministici sono completamente prevedibili, mentre i giochi stocastici hanno una componente casuale.
* **Numero di giocatori:** I giochi possono essere a uno, due o più giocatori.
* **Somma zero:** In un gioco a somma zero, la vincita di un giocatore corrisponde alla perdita di un altro giocatore.
* **Informazione perfetta:** Un gioco con informazione perfetta è completamente osservabile da tutti i giocatori.

### Formalizzazione di un Gioco Deterministico
Un gioco deterministico può essere formalizzato nel seguente modo:

* **Insieme di stati S:** dove S0 è lo stato iniziale.
* **Insieme di player P = {1, ..., N}:** che, solitamente, giocano a turni.
* **Insieme di mosse/azioni A:** legali nello stato s ∈ S.
* **Modello di transizione (s, a) → S:** che definisce lo stato risultante dall'esecuzione dell'azione a ∈ A nello stato s ∈ S.
* **Test di terminazione s → {true, false}:** che restituisce true se la partita è finita altrimenti restituisce falso. Gli stati che fanno terminare la partita sono detti stati terminali.
* **Funzione di utilità (s, p) → r:** detta anche funzione obiettivo/di payoff che definisce il valore numerico finale (risultato) per il giocatore p ∈ P quando il gioco termina nello stato s ∈ S.

Una soluzione è una strategia s → A che da uno stato s ∈ S decide quale azione a ∈ A conviene compiere per raggiungere l'obiettivo, ovvero uno stato terminale.

### Albero di Gioco
Definiamo Albero di Gioco un albero di ricerca che segue ogni sequenza di mosse fino a raggiungere uno stato terminale. L'albero di gioco potrebbe essere infinito se lo spazio degli stati è illimitato o se le regole del gioco consentono di ripetere le posizioni (stati) all'infinito.

## Giochi a Somma Zero
Vi sono due player che giocano a turni, dove le azioni che avvantaggiano un player danneggiano l'altro, dunque non è possibile un risultato dove vincono entrambi. Vi è una competizione pura: gli agenti hanno funzioni di utilità opposte e si può pensare che uno cerchi di massimizzare un certo valore mentre l'altro cerca di minimizzarlo. Si differenziano dalla più ampia categoria dei giochi generali, dove gli agenti hanno funzioni di utilità indipendenti e, di conseguenza, possono cooperare, ignorarsi o anche essere in competizione (magari in una forma minore rispetto a quella dei giochi a somma zero).

### Ricerca MiniMax
Chiamiamo i due player MIN e MAX. La strategia di MAX deve essere un piano condizionale, una strategia che specifica ogni reazione a ogni possibile mossa di MIN, e viceversa. Dunque, nei casi in cui il risultato è binario (vittoria/sconfitta), per generare il piano condizionale potremmo usare una ricerca AND-OR ma se sono possibili più risultati è necessario utilizzare una ricerca più generale, ovvero una ricerca MiniMax.

Anzitutto, con mossa si intende che entrambi i giocatori abbiano compiuto un'azione mentre con ply (oppure strato) si indica un'azione compiuta da un singolo giocatore che ci fa scendere di un livello più in basso nell'albero di gioco. Dato un albero di gioco, è possibile determinare la strategia ottima calcolando il valore MiniMax di ogni stato. Si assume che entrambi i giocatori giochino in maniera ottima, altrimenti la strategia potrebbe essere non ottima.

Il valore MiniMax di uno stato è l'utilità (per MAX) di trovarsi nello stato corrispondente, è calcolato ricorsivamente ed è definito come:

```
MiniMax(s) = 
    se s è uno stato terminale allora utilità(s, MAX)
    altrimenti max_{a ∈ A(s)} MiniMax(Successore(s, a)) se MAX deve muovere
    altrimenti min_{a ∈ A(s)} MiniMax(Successore(s, a)) se MIN deve muovere
```

Ad esempio, consideriamo il seguente albero di gioco:

| Nodo | Successori | Valore MiniMax |
|---|---|---|
| A | B, C, D | 3 |
| B | 3, 12, 8 | 3 |
| C | 2, 14, 6 | 2 |
| D | 10, 4, 2 | 2 |

Ad ogni turno, MAX preferirà muoversi verso uno stato di valore massimo mentre MIN preferirà muoversi verso uno stato di valore minimo. Nell'esempio, i nodi terminali prendono i loro valori dalla funzione di utilità; il primo nodo MIN, ovvero B, ha 3 successori con valori {3,12,8} quindi il suo valore MiniMax è pari a 3... per lo stesso motivo, il valore MiniMax dei nodi C,D è pari a 2; il nodo MAX ha 3 successori con valori {3,2,2}, quindi il suo valore MiniMax è 3.
#### Efficienza
Quanto è efficiente questa ricerca? Nel caso peggiore bisogna esplorare tutto lo spazio di ricerca, come una DFS. Quindi, la complessità spaziale è O(b · m) mentre quella temporale è O(b^m) dove m è la profondità e b è il fattore di branching.


## Alpha-Beta Pruning: 
La ricerca MiniMax, utilizzata per determinare la strategia ottima in un gioco a somma zero, può diventare computazionalmente costosa a causa della crescita esponenziale del numero di stati con l'aumentare della profondità dell'albero di gioco. L'Alpha-Beta Pruning è una tecnica di ottimizzazione che permette di potare porzioni dell'albero di gioco irrilevanti per la ricerca, riducendo il numero di nodi da esplorare.

### Come Funziona
L'algoritmo Alpha-Beta Pruning utilizza due valori:

* **α:** Il miglior valore per MAX nel cammino dalla radice.
* **β:** Il miglior valore per MIN nel cammino dalla radice.

Questi valori vengono aggiornati durante la ricorsione e utilizzati per fare pruning. Se, ad esempio, il valore di un nodo MIN è inferiore a α, allora MAX non sceglierebbe mai quel nodo, quindi non ha senso esplorare i suoi successori.

### Efficacia
L'efficacia del pruning dipende dall'ordine in cui vengono esplorati i successori. Se l'ordine è perfetto, la ricerca Alpha-Beta esaminerebbe solo $O(b^\frac{m}{2}))$ nodi, mentre con un ordine casuale ne esaminerebbe $O(b^\frac{3}{4m})).$

### Ricerca a Profondità Limitata
Nei giochi reali, gli alberi di gioco sono spesso troppo grandi per essere esplorati completamente. La ricerca a profondità limitata risolve questo problema interrompendo la ricerca ad una certa profondità e utilizzando euristiche per stimare il valore dei nodi non esplorati. Tuttavia, in questo caso non si hanno garanzie di ottimalità.


## Ricerca Euristica Alpha-Beta
Nei giochi reali, il tempo di calcolo è spesso limitato, rendendo impossibile esplorare completamente l'albero di gioco. La ricerca euristica Alpha-Beta risolve questo problema interrompendo la ricerca ad una certa profondità e utilizzando una valutazione euristica per stimare il valore dei nodi non esplorati.

### Test di Taglio e Valutazione Euristica
Il test di taglio sostituisce il test di terminazione tradizionale, interrompendo la ricerca in base alla profondità dello stato o ad altre proprietà. Se il test di taglio ha successo, la funzione di utilità viene sostituita da una valutazione euristica, EVAL, che dipende dallo stato e dalla profondità.

### Proprietà dell'Euristica
L'euristica deve soddisfare le seguenti proprietà:

* **Stima corretta:** La valutazione euristica deve essere compresa tra il valore di sconfitta e il valore di vittoria per il giocatore.
* **Efficienza:** Il calcolo dell'euristica deve essere veloce per non rallentare la ricerca.
* **Correlazione con la vittoria:** L'euristica deve essere strettamente correlata alle possibilità di vincere per il giocatore.

### Importanza dell'Euristica
Un'euristica ben definita può migliorare l'efficienza del pruning Alpha-Beta, permettendo di ordinare le mosse in base al loro valore e di tagliare nodi meno promettenti. Tuttavia, è importante ricordare che l'euristica fornisce solo una stima e potrebbe portare a tagliare nodi importanti.

## Ricerca ad Albero Monte Carlo
La ricerca ad Albero Monte Carlo è una tecnica utilizzata per la ricerca in giochi complessi, in particolare quando è difficile definire buone euristiche o il fattore di branching è troppo elevato. Questa tecnica si basa su simulazioni casuali per costruire un albero di gioco e stimare il valore di ogni stato.

### Funzionamento
La ricerca ad Albero Monte Carlo procede in quattro fasi:
1. **Selezione:** Partendo dalla radice dell'albero di gioco, si sceglie un nodo figlio e si prosegue fino a raggiungere una foglia (un nodo mai esplorato). La scelta del nodo figlio è guidata da una politica di selezione.
2. **Espansione:** Si genera un nuovo figlio (o più di uno) al nodo foglia e gli si assegna un termine di sfruttamento, inizialmente pari a 0.
3. **Simulazione:** Partendo dal nodo appena generato, si simulano le mosse per entrambi i giocatori fino a raggiungere uno stato terminale. La scelta delle mosse durante la simulazione è guidata da una politica di simulazione.
4. **Retropropagazione:** Il risultato della simulazione viene utilizzato per aggiornare il termine di sfruttamento di tutti i nodi esplorati durante la simulazione, fino alla radice del nodo espanso.

### Politiche di Selezione e Simulazione
La politica di simulazione determina come vengono scelte le mosse durante le simulazioni. Idealmente, questa politica dovrebbe favorire le mosse più vincenti. La politica di selezione, invece, deve bilanciare l'esplorazione di nuovi nodi con lo sfruttamento di nodi già esplorati.

### UCB1
Una politica di selezione comune è l'Upper Confidence Bounds applied to Trees (UCB1). Questa politica assegna un valore a ogni nodo, che è la somma del fattore di sfruttamento $\left( \frac{U(n)}{N(n)} \right)$ e del fattore di esplorazione $\left(c \cdot \sqrt{(log N\frac{(padre(n))}{N(n))}}\right)$. Il fattore di sfruttamento rappresenta il valore medio delle simulazioni passate per il nodo, mentre il fattore di esplorazione incoraggia l'esplorazione di nodi meno esplorati.


## Pacman Veloce
Il gioco Pacman vs Fantasmi è un esempio di gioco su grafo, dove i fantasmi cercano di catturare Pacman.
### Modellazione del Gioco
Il labirinto di Pacman viene modellato come un grafo, dove i nodi rappresentano le posizioni e gli archi rappresentano i possibili movimenti. Lo spazio di movimento di Pacman è dato dalla componente connessa del grafo, escludendo i nodi occupati dai fantasmi.

### Definizione del Grafo di Gioco
Il grafo di gioco è definito come G = (V, E), dove V è l'insieme dei nodi e E è l'insieme degli archi. I fantasmi sono rappresentati da F1, ..., Fk, mentre Pacman è rappresentato da P. Uno stato del gioco è definito dalla posizione di tutti i fantasmi e di Pacman.

### Strategia Vincente per i Fantasmi
Una strategia vincente per i fantasmi esiste se, per ogni possibile posizione di Pacman, i fantasmi possono posizionarsi in modo da bloccarlo. Questo tipo di strategia può essere rappresentata da un albero AND-OR, dove i nodi AND rappresentano le possibili posizioni di Pacman e i nodi OR rappresentano le possibili posizioni dei fantasmi.

### Complessità
La complessità dell'algoritmo per trovare una strategia vincente per i fantasmi è polinomiale se lo stato del gioco viene rappresentato in modo logaritmico. La dimensione dello stato è data da $(k + 1) · log_2 |V|$, 
dove $k$ è il numero di fantasmi, $|V|$ è il numero di nodi nel grafo.
