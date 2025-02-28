
| Termine | Spiegazione |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agente non razionale** | Un agente che non si comporta in modo perfettamente razionale, ma presenta una componente casuale nel suo comportamento. |
| **Aspetti realmente casuali** | Eventi imprevedibili che influenzano il comportamento di un agente. |
| **Modellazione dell'incertezza** | Utilizzo della teoria della probabilità per rappresentare l'incertezza nel comportamento di un agente. |
| **Valore atteso** | Il valore medio di un risultato, ponderato in base alla probabilità di ogni possibile evento. |
| **Expectimax** | Un algoritmo di ricerca che estende MiniMax per gestire agenti non razionali, considerando i risultati casuali. |
| **Nodo chance** | Un nodo in un albero di ricerca che rappresenta un risultato casuale. |
| **Simulazione** | Un metodo per stimare le probabilità di comportamento di un avversario o dell'ambiente. |
| **Strategia ottima** | La strategia che massimizza il risultato atteso in un gioco. |
| **Pessimismo e ottimismo** | Atteggiamenti che possono influenzare la scelta di una strategia, portando a sottovalutare o sovrastimare l'incertezza. |
| **ExpectiMiniMax** | Un algoritmo che estende MiniMax per gestire l'incertezza, calcolando il valore atteso per i nodi Chance. |
| **Utilità** | Una funzione che misura la soddisfazione di un agente rispetto a un determinato stato o risultato. |
| **Insensibilità alle trasformazioni monotone** | La proprietà di una funzione di utilità di non essere influenzata da trasformazioni monotone della scala. |
| **Assiomi di razionalità** | Proprietà che devono essere soddisfatte dalle preferenze di un agente razionale per essere considerate coerenti. |
| **Transitività** | Un assioma di razionalità che garantisce che le preferenze di un agente siano coerenti. |
| **Orderability** | Un assioma di razionalità che afferma che un agente deve avere una preferenza definita tra due premi, o essere indifferente tra essi. |
| **Continuity** | Un assioma di razionalità che afferma che se un agente preferisce A a B e B a C, allora esiste un livello di probabilità p per cui la lotteria che offre A con probabilità p e C con probabilità (1-p) è indifferente rispetto a B. |
| **Substitutability** | Un assioma di razionalità che afferma che se un agente è indifferente tra A e B, allora è anche indifferente tra due lotterie che offrono A e B con la stessa probabilità. |
| **Monotonicity** | Un assioma di razionalità che afferma che se A è preferito a B, allora una lotteria che offre A con una probabilità maggiore è preferita a una lotteria che offre A con una probabilità minore. |
| **Teorema di Ramsey, von Neumann & Morgenstern** | Un teorema che dimostra che se le preferenze di un agente soddisfano gli assiomi di razionalità, allora esiste una funzione di utilità a valori reali che può essere utilizzata per rappresentare tali preferenze. |
| **Principio della Massima Utilità Attesa (MEU)** | Un principio che afferma che un agente razionale dovrebbe scegliere l'azione che massimizza la sua utilità attesa. |
| **Avversione al rischio** | La tendenza degli esseri umani a preferire un guadagno certo a un guadagno incerto con un valore atteso maggiore. |
| **Premio di assicurazione** | La differenza tra il valore atteso di un evento e il valore monetario che un individuo accetterebbe con certezza. |

## Agenti non Razionali

Gli agenti non razionali sono caratterizzati da una componente casuale nel loro comportamento, dovuta a:
* **Aspetti realmente casuali:** Eventi imprevedibili che influenzano il comportamento dell'agente.
* **Modellazione dell'incertezza:** Utilizzo della teoria della probabilità per rappresentare l'incertezza nel comportamento dell'agente.

In presenza di agenti non razionali, il concetto di "caso peggiore" non è più rilevante. Si introduce il concetto di **valore atteso**, che rappresenta il valore medio di un risultato, ponderato in base alla probabilità di ogni possibile evento.

### Expectimax

Expectimax è un algoritmo di ricerca che estende MiniMax per gestire agenti non razionali. I nodi di tipo max si comportano come in MiniMax, mentre i nodi chance rappresentano i risultati casuali. Il valore di un nodo chance è il valore atteso dei suoi figli, calcolato come la somma dei valori dei figli ponderati in base alla probabilità di ogni azione.

##### Esempio:

Se un nodo chance ha due figli con valori 12 e -12, e la probabilità di raggiungere il primo figlio è 0.8 e la probabilità di raggiungere il secondo figlio è 0.2, il valore atteso del nodo chance è:

```
0.8 * 12 + 0.2 * (-12) = 10
```

### Probabilità in Expectimax

Per utilizzare Expectimax, è necessario avere un modello probabilistico del comportamento dell'avversario o dell'ambiente. Le probabilità possono essere note a priori o stimate tramite simulazioni.

### Simulazione

La simulazione è un metodo per stimare le probabilità di comportamento dell'avversario. Tuttavia, la simulazione può essere computazionalmente costosa, soprattutto se l'avversario sta anche simulando le nostre mosse.

### Incertezza e Strategia Ottima

In alcuni giochi, non esiste una strategia ottima deterministica. L'introduzione di incertezza, come scelte casuali, può migliorare il valore atteso della strategia, rendendola più efficace.

### Pessimismo e Ottimismo

La scelta di una strategia può essere influenzata da un atteggiamento eccessivamente pessimista o ottimistico. Un atteggiamento pessimista può portare a considerare situazioni pericolose che in realtà non lo sono, mentre un atteggiamento ottimistico può portare a sottovalutare l'incertezza.

### Backgammon: Un Esempio di Gioco con Incertezza

Il Backgammon è un gioco che presenta sia un avversario che incertezza. Il giocatore deve compiere una mossa, poi si lancia il dado e l'avversario sceglie la sua mossa in base al risultato del dado.

### ExpectiMiniMax

ExpectiMiniMax è un algoritmo che estende MiniMax per gestire l'incertezza. L'algoritmo include nodi Min, Max e Chance. Per i nodi Chance, si calcola il valore atteso, ovvero la somma dei valori dei figli ponderati in base alla probabilità di ogni azione.

### Utilità nei Giochi con Più Agenti

In giochi con più agenti, ogni nodo ha associata una tripla di valori che rappresenta l'utilità/stima per ogni giocatore in quello stato. Gli algoritmi possono far emergere forme di coalizione tra i giocatori, con ogni giocatore che cerca di massimizzare il proprio valore.

L'utilità è una funzione che misura la soddisfazione di un agente rispetto a un determinato stato o risultato. Un agente razionale dovrebbe massimizzare la propria utilità. Tuttavia, non sempre è facile definire una funzione di utilità che rifletta accuratamente le preferenze di un agente.

### Importanza della Scala

In alcuni contesti, la scala della funzione di utilità non è importante. Ad esempio, nel ragionamento MiniMax, la funzione di utilità può essere trasformata in modo monotono senza alterare il risultato della ricerca.

In altri contesti, la scala della funzione di utilità è importante. Ad esempio, quando si ha a che fare con l'incertezza, è necessario utilizzare le lotterie per modellare le preferenze dell'agente rispetto a risultati incerti.

### Lotterie e Preferenze

Una lotteria è un insieme di premi associati a delle probabilità. Un agente deve esprimere una preferenza tra due lotterie.

* **$L = [p, A; (1 − p), B]$:** Lotteria con premio A con probabilità p e premio B con probabilità $(1 − p)$.
* **A ≻ B:** L'agente preferisce A a B.
* **A ∼ B:** L'agente è indifferente tra A e B.

### Assiomi di Razionalità

Per poter parlare di preferenze razionali, è necessario introdurre alcuni assiomi, come l'assioma di transitività:

##### (A ≻ B) ∧ (B ≻ C) =⇒ (A ≻ C)

L'assioma di transitività garantisce che le preferenze dell'agente siano coerenti. Se un agente preferisce A a B e B a C, allora deve preferire A a C.

La transitività è importante perché evita cicli infiniti nelle preferenze dell'agente. Se un agente non avesse preferenze transitive, potrebbe essere coinvolto in un ciclo infinito di scambi, pagando sempre un centesimo per ottenere un premio che considera migliore, senza mai raggiungere un risultato finale.

## Assiomi di Razionalità per le Preferenze

Questi assiomi definiscono le proprietà che devono essere soddisfatte dalle preferenze di un agente razionale per poter essere considerate coerenti e razionali.

### Orderability (Ordinabilità)

* **Definizione:** Un agente razionale deve avere una preferenza definita tra due premi, o essere indifferente tra essi.
* **Formulazione:** $(A ≻ B) ∨ (B ≻ A) ∨ (A ∼ B)$
 * A ≻ B: A è preferito a B
 * B ≻ A: B è preferito ad A
 * A ∼ B: A e B sono indifferenti

### Continuity (Continuità)

* **Definizione:** Se un agente preferisce A a B e B a C, allora esiste un livello di probabilità p per cui la lotteria che offre A con probabilità p e C con probabilità (1-p) è indifferente rispetto a B.
* **Formulazione:** $A ≻ B ≻ C =⇒ ∃p : [p, A; (1 − p), C] ∼ B$

### Substitutability (Sostituibilità)

* **Definizione:** Se un agente è indifferente tra A e B, allora è anche indifferente tra due lotterie che offrono A e B con la stessa probabilità.
* **Formulazione:** $A ∼ B =⇒ [p, A; (1 − p), C] ∼ [p, B; (1 − p), C]$

### Monotonicity (Monotonia)

* **Definizione:** Se A è preferito a B, allora una lotteria che offre A con una probabilità maggiore è preferita a una lotteria che offre A con una probabilità minore.
* **Formulazione:** $A ≻ B =⇒ (p ≥ q ⇐⇒ [p, A; (1 − p), B] ⪰ [q, A; (1 − q), B])$
 * ⪰ indica una preferenza debole (può essere indifferente o preferita)

##### Notazione:

* **A, B, C:** Premi
* **p, q:** Probabilità
* **$[p, A; (1 − p), B]$:** Lotteria che offre A con probabilità p e B con probabilità (1-p)
* **≻:** Preferenza
* **∼:** Indifferenza
* **⪰:** Preferenza debole

## Teorema di Ramsey, von Neumann & Morgenstern

Se le preferenze di un agente soddisfano gli assiomi di razionalità, allora esiste una funzione di utilità a valori reali che può essere utilizzata per rappresentare tali preferenze.

* **Formulazione:** $U(A) ≥ U(B) ⇐⇒ A ⪰ B$
 * U(A): Utilità del premio A
 * U(B): Utilità del premio B
 * A ⪰ B: A è preferito o indifferente a B
Questa formulazione implica che le preferenze possono essere ordinate in base al valore di utilità, permettendo così all'agente di fare scelte razionali.

* **Utilità di una lotteria:** $U([p_1, S_1; ... ; p_n, S_n]) = \sum_i \ p_i · U(S_i)$
 * Si: i-esimo stato
 * pi_: Probabilità dello stato Si
 * U(Si): Utilità dello stato Si
Questa formula consente di calcolare l'utilità attesa di una lotteria, sommando le utilità pesate dalle rispettive probabilità.

In sintesi, il teorema afferma che se le preferenze di un agente sono razionali, allora esiste una funzione di utilità che può essere utilizzata per ordinare le preferenze tra premi e lotterie.

## Principio della Massima Utilità Attesa (MEU)

Il principio MEU afferma che un agente razionale dovrebbe scegliere l'azione che massimizza la sua utilità attesa.

* **Applicazione:** Questo principio è alla base di algoritmi come ExpectiMax, che cercano di massimizzare l'utilità attesa in situazioni di incertezza.

* **Limiti:** Il principio MEU non è sempre applicabile agli esseri umani, che spesso non si comportano in modo razionale e possono avere preferenze che non sono facilmente quantificabili con una funzione di utilità.

## Razionalità Umana

Gli esseri umani non sempre si comportano in modo razionale e possono avere preferenze che non sono in linea con il principio MEU.

* **Avversione al rischio:** Gli umani tendono ad essere avversi al rischio, preferendo un guadagno certo a un guadagno incerto con un valore atteso maggiore.

* **Influenza della ricchezza:** La ricchezza di un individuo influenza le sue scelte, poiché il valore del denaro non è sempre equivalente all'utilità.

* **Premio di assicurazione:** La differenza tra il valore atteso di un evento e il valore monetario che un individuo accetterebbe con certezza è definito premio di assicurazione.

* **Irrazionalità:** Le scelte degli esseri umani possono essere irrazionali, come dimostrato da esempi in cui le preferenze non sono coerenti con una funzione di utilità.
