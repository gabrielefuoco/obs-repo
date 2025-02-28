
Un sistema di retrival, data una query da una sorgente informativa (repository), è in grado di identificare una porzione di risorse ritenute rilevanti alla query. 

##### Sottoinsiemi Rilevanti e Recuperati

Si definiscono due sottoinsiemi:

* **Relevant:** L'insieme di tutte le risorse realmente rilevanti per la query.
* **Retrieved:** L'insieme di tutte le risorse recuperate dal sistema in risposta alla query.

L'obiettivo è ottenere un **overlap** (intersezione) il più grande possibile tra questi due sottoinsiemi.

##### Misurazione degli Errori

L'intersezione rappresenta la porzione di documenti recuperati dal sistema che sono realmente rilevanti. Questa intersezione rappresenta un livello di contingenza statistica che corrisponde ai **true positive**.

* **Falsi positivi (false positive):** Documenti recuperati dal sistema che non appartengono all'insieme dei documenti realmente rilevanti.
* **Falsi negativi (false negative):** Documenti realmente rilevanti che non sono stati recuperati dal sistema.
* **Veri negativi (true negative):** Documenti non rilevanti che non sono stati recuperati dal sistema.

##### Elementi di Valutazione

Per valutare un sistema di recupero informazioni, sono necessari tre elementi:

1. **Collezione di documenti di riferimento:** Un insieme di documenti utilizzati per la valutazione.
2. **Set di query di riferimento:** Un insieme di query utilizzate per la valutazione.
3. **Valutazione di ogni documento:** Per ogni query, ogni documento viene valutato come rilevante o non rilevante.

#### Valutazione del recupero non ordinato: Precisione e Richiamo

Le valutazioni binarie si basano sulla classificazione dei documenti come rilevanti o non rilevanti. Le misure più comuni sono:

* **Precision:** La frazione di documenti recuperati che sono rilevanti. In termini di probabilità, è la probabilità di un documento essere rilevante dato che è stato recuperato: P(rilevante|recuperato).

* **Recall:** La frazione di documenti rilevanti che sono recuperati. In termini di probabilità, è la probabilità di un documento essere recuperato dato che è rilevante: P(recuperato|rilevante). Il richiamo è anche noto come *true positive rate* o *sensitivity*. Indica quanto il sistema è stato in grado di trovare informazioni rilevanti nella repository.

Un sistema che è più selettivo e meno abile a massimizzare il proprio spazio di ricerca avrà una precisione migliore del richiamo.

| | Relevant | Nonrelevant |
| ----------------- | -------- | ----------- |
| **Retrieved** | tp | fp |
| **Not Retrieved** | fn | tn |

- **Precision (P)** = $\frac{tp}{tp + fp}$
- **Recall (R)** = $\frac{tp}{tp + fn}$

##### Conflitto tra Precisione e Richiamo:

Le due misure sono in conflitto tra loro, poiché un sistema che cura l'aspetto della selezione tenderà ad essere più accurato di uno che tende ad esplorare lo spazio di ricerca, che sarà più esposto ad errori. Questo conflitto è dovuto al fatto che queste due misure sono correlate negativamente. 

## Trade-off tra Precisione e Richiamo

Esiste un trade-off intrinseco tra precisione e richiamo:

* **Aumentare il richiamo:** Restituendo più documenti, si aumenta il richiamo. Il richiamo è una funzione non decrescente del numero di documenti recuperati. Un sistema che restituisce tutti i documenti ha un richiamo del 100%!
* **Aumentare la precisione:** È facile ottenere un'alta precisione per un richiamo molto basso. Ad esempio, se il documento con il punteggio più alto è rilevante, possiamo massimizzare la precisione restituendo solo quel documento.

##### Misure di Combinazione:

La media aritmetica di precisione e richiamo rischia di fornire una media troppo generosa. Per questo motivo, si utilizzano altre misure di combinazione:

* **Media geometrica:** Considera la radice quadrata del prodotto di precisione e richiamo.
* **Media armonica:** È preferita perché può assegnare un peso diverso a precisione e richiamo. Un'altra proprietà è che la media armonica è penalizzata dalle misurazioni più basse e gestisce bene gli score vicini allo zero.

##### F-measure:

La F-measure è una misura di combinazione che permette un trade-off tra precisione e richiamo:

$$F=\frac{1}{\alpha \frac{1}{P}+(1-\alpha)\frac{1}{R}}=\frac{(\beta^2+1)PR}{\beta^2P+R}$$

Dove:

* $\beta^2=\frac{1-\alpha}{\alpha}$
* $\alpha$ è il parametro di smoothing che controlla la combinazione lineare tra il reciproco della precisione e il reciproco del richiamo.
* $\beta$ è un numero che possiamo variare < o > di 1.

##### Impostazione di Beta:

* **Beta > 1:** Pesa maggiormente la recall.
* **Beta < 1:** Pesa maggiormente la precisione.

![[Senza nome-20241104120509218.png]]
Questo plot ha sull'asse x la precisione a una recall fissata (asse y). Confronta la media armonica e la media geometrica: hanno lo stesso andamento, ma la media armonica penalizza di più rispetto alla geometrica in regimi bassi di uno dei due valori (precisione o richiamo). 

## Misure Basate sul Ranking

Le misure basate sul ranking valutano la qualità di un sistema di ranking, ovvero la capacità di ordinare i risultati in base alla loro rilevanza. Queste misure possono essere calcolate per valori fissati di un'altra misura, come la precisione o il richiamo.

### Rilevanza Binaria

Nel caso di rilevanza binaria (un documento è rilevante o non rilevante), le misure più comuni sono:

* **Precisione@K (P@K):** Calcola la percentuale di documenti rilevanti tra i primi K risultati.
* **Media della Precisione Media (MAP):** Calcola la media della precisione per ogni query, considerando tutti i valori di K.
* **Preferenza Binaria:** Misura la probabilità che un documento rilevante sia classificato più in alto rispetto a un documento non rilevante.
* **Media del Rango Reciproco (MRR):** Calcola il reciproco del rango del primo documento rilevante per ogni query.

### Livelli di Rilevanza Multipli

Quando si hanno livelli di rilevanza multipli (ad esempio, molto rilevante, rilevante, non rilevante), si può utilizzare:

* **Guadagno Cumulativo Scontato Normalizzato (NDCG):** Considera l'ordine dei documenti e assegna un punteggio più alto ai documenti più rilevanti e classificati più in alto.

### Curva Precision-Richiamo

L'idea è di adattare le misure di set (come precisione e richiamo) per diventare misure per liste ordinate, ovvero per il ranking.

* **Precisione/Richiamo/F:** Sono misure per set non ordinati.
* **Trasformazione in misure di liste ordinate:** Si calcola la misura di set per ogni "prefisso" della lista ordinata: i primi 1, i primi 2, i primi 3, i primi 4 risultati, ecc.
* **Curva Precision-Richiamo:** Si ottiene tracciando la precisione in funzione del richiamo per ogni prefisso.

### Precision@K

* **Soglia di rango K:** Si imposta una soglia di rango K (parametro specificato).
* **Determinazione della porzione di documenti rilevanti:** Si determina la porzione dei documenti rilevanti tra i primi K risultati.
* **Valutazione della precisione per diversi valori di K:** Si valuta la precisione per diversi valori di K.
* **Calcolo della % di rilevanza nei primi K:** Si calcola la percentuale di documenti rilevanti nei primi K risultati.
* **Ignorare i documenti classificati al di sotto di K:** Si ignorano i documenti classificati al di sotto di K.

##### Esempio:

* **Prec@3 di 2/3:** Se tra i primi 3 risultati, 2 sono rilevanti, la Precision@3 è 2/3.
	![[Senza nome-20241104121345108.png]]
* **Prec@4 di 2/4:** Se tra i primi 4 risultati, 2 sono rilevanti, la Precision@4 è 2/4.
* **Prec@5 di 3/5:** Se tra i primi 5 risultati, 3 sono rilevanti, la Precision@5 è 3/5.

Allo stesso modo, si può calcolare la **Recall@K**, che misura la percentuale di documenti rilevanti recuperati tra i primi K risultati.

### Curva Precision-Recall

La curva Precision-Richiamo fornisce una visione d'insieme globale delle prestazioni di un sistema di ranking. L'obiettivo è individuare i valori di precisione per ogni query e interpolarli a un livello fissato di richiamo.

* **Calcolo delle prestazioni medie su un set di argomenti:** Si calcola la precisione per ogni argomento, considerando un numero diverso di documenti rilevanti per ogni argomento.
* **Interpolazione a un set di livelli di richiamo standard:** I valori di precisione individuali per argomento vengono interpolati a un set di livelli di richiamo standard (da 0 a 1 con incrementi di 0,1).
* **Regola di interpolazione:** Per interpolare la precisione al livello di richiamo standard *i*, si utilizza la massima precisione ottenuta per l'argomento per qualsiasi livello di richiamo effettivo maggiore o uguale a *i*.
* **Razionale per l'interpolazione:** L'utente è disposto a guardare più cose se sia la precisione che il richiamo migliorano.

##### Esempio:

![[Senza nome-20241104121409721.png]]

Per valori bassi di richiamo, si hanno valori alti di precisione. Quando ci si sposta dall'origine, si ha un valore basso di richiamo ($r_1=0.5$), che poi sale a circa $r_2=0.8$. In questo punto, si ha la prima interpolazione. 
Si interpola in corrispondenza di $r_2$, che è il valore massimo di precisione ottenuto fino a quel momento. 
Non si utilizza la coppia richiamo = $r_1$ e precisione = 0.5, ma si interpola dicendo che per ogni valore di richiamo maggiore o uguale a $r_2$, si ha il valore massimo di precisione (da quel punto in poi). 

Dunque, l'interpolazione avviene quando vi è una risalita nel grafico. Il tratto in rosso rappresenta l'interpolazione.

## Mean Average Precision (MAP)

La Mean Average Precision (MAP) è una misura utilizzata per valutare la performance di un sistema di recupero informazioni. 

##### Calcolo della MAP:

1. **Posizioni dei documenti rilevanti:** Si identificano le posizioni nel ranking di ogni documento rilevante, indicate con $(K_1, K_2, … K_R)$.
2. **Precision@K:** Si calcola la Precision@K per ogni posizione dei documenti rilevanti. La Precision@K è il rapporto tra il numero di documenti rilevanti recuperati fino alla posizione K e il numero totale di documenti recuperati fino alla posizione K.
3. **Average Precision:** Si calcola la media delle Precision@K per tutti i documenti rilevanti.
4. **MAP:** Si calcola la media aritmetica delle Average Precision ottenute per diverse query/ranking.

##### Esempio

![[Senza nome-20241104121732053.png]]

In questo esempio, l'Average Precision è: $\frac{1}{3}\cdot\left( \frac{1}{1}+\frac{2}{3}+\frac{3}{5} \right)=0.76$ 

La MAP è la media delle Average Precision calcolate per diverse query.

##### Vantaggi della MAP

La MAP è una misura solida per valutare la performance dei sistemi di recupero informazioni perché:

* Tiene conto della posizione dei documenti rilevanti nel ranking.
* È sensibile alla qualità del ranking, premiando i sistemi che posizionano i documenti rilevanti in posizioni più alte.
* È una misura aggregata che considera la performance su diverse query.

## Media della Precisione Media (MAP)

La Media della Precisione Media (MAP) è una misura di performance per i sistemi di recupero dell'informazione. Essa assume che l'utente sia interessato a sapere quale sia la **precisione** del sistema per ogni documento rilevante.

* Se un documento rilevante non viene mai recuperato, si assume che la precisione corrispondente a quel documento rilevante sia zero.
* MAP è una **media macro**: ogni query conta allo stesso modo.
* MAP presuppone che l'utente sia interessato a trovare molti documenti rilevanti per ogni query.
* MAP richiede molti giudizi di rilevanza nella collezione di testo.
* Non ci piace che richieda la conoscenza globale di dove stanno tutti i documenti rilevanti. Si introduce dunque un'altra misura.

## Preferenza Binaria

La Preferenza Binaria è una misura di performance progettata per situazioni in cui i giudizi di rilevanza sono noti per essere incompleti (ne abbiamo solo una conoscenza parziale).

Calcola una relazione di preferenza su se i documenti giudicati rilevanti vengono recuperati prima dei documenti giudicati irrilevanti. 
* **Non vogliamo conoscere globalmente la posizione dei documenti realmente rilevanti.**
* È irrazionale pensare che l'insieme dei retrieval coincida con quello dei rilevanti.

La misura è basata solo sui ranghi relativi dei documenti giudicati:

 $$\text{bpref}=\frac{1}{R}\sum_{r}\left( 1-\frac{|\text{n ranked higer than r}|}{min(R,N)} \right)$$

Dove:

* R è il numero di documenti giudicati rilevanti
* N è il numero di documenti giudicati irrilevanti
* r è un documento rilevante recuperato
* n è un membro dei primi R documenti irrilevanti recuperati

Misuriamo il numero di documenti per ogni valore di rank e il numero di documenti irrilevanti che si presentano prima del rank.

La **preferenza binaria** (*Bpref*) è una misura di rilevanza utile quando si ha una visione parziale dei documenti o quando si è interessati a valutare l'abilità del sistema di identificare ciò che è realmente irrilevante.

*Bpref* si concentra su come i documenti noti come rilevanti e non rilevanti vengono classificati, senza richiedere la conoscenza di tutti i documenti rilevanti nella collezione.

*Bpref* può essere interpretato come l'inverso della frazione di documenti giudicati irrilevanti che vengono recuperati prima di quelli rilevanti.

*Bpref* e *MAP* (**Mean Average Precision**) sono altamente correlate quando si utilizzano giudizi completi. Tuttavia, quando i giudizi sono incompleti, le classifiche dei sistemi per Bpref rimangono altamente correlate alla classificazione originale, mentre le classifiche dei sistemi per MAP non lo sono.

## Mean Reciprocal Rank (MRR)

Il Mean Reciprocal Rank (MRR) è una misura di rilevanza che si basa sulla posizione del primo documento rilevante nella lista dei risultati di una query.

* **Posizione del primo documento rilevante:** Si considera il rango (K) del primo documento rilevante nella lista.  Questo potrebbe corrispondere al primo documento cliccato dall'utente.
* **Punteggio del Rango Reciproco (RR):** Il punteggio RR è calcolato come $\frac{1}{K}$.
* **MRR:** Il MRR è la media dei punteggi RR calcolati su più query.
* **Vantaggi:** Il MRR è semplice da calcolare.
* **Svantaggi:**  Il MRR penalizza pesantemente i sistemi che non posizionano il primo documento rilevante nelle prime posizioni della lista.  Un documento rilevante in posizione 2 avrà un punteggio RR di 0.5, mentre uno in posizione 10 avrà un punteggio di 0.1.


### Esempio di Calcolo del MRR

Consideriamo due liste di ranking con i rispettivi valori di Average Precision (AP): AP₁ = 0.622 e AP₂ = 0.520.

Il Mean Average Precision (MAP) si calcola come media delle AP:

$MAP = \frac{0.622 + 0.520}{2} = 0.571$

Assumendo che i ranghi dei primi documenti rilevanti nelle due liste siano 1 e 2 rispettivamente, il Mean Reciprocal Rank (MRR) si calcola come:

$MRR = \frac{\frac{1}{1} + \frac{1}{2}}{2} = \frac{1.0 + 0.5}{2} = 0.75$


### Esempio di Calcolo di $b_{pref}$

Il calcolo di $b_{pref}$ (una metrica di preferenza) per due liste di ranking è mostrato di seguito.  La formula sembra calcolare una sorta di media pesata, dove il peso è dato dalla differenza tra 1 e il rapporto tra il rank e il numero totale di elementi. Senza ulteriori informazioni sul contesto, è difficile fornire una interpretazione più precisa.

$$
b_{pref_{f1}} = \frac{1}{5} \left( (1 - \frac{0}{5}) + (1 - \frac{1}{5}) + (1 - \frac{3}{5}) + (1 - \frac{5}{5}) + (1 - \frac{5}{5}) \right) = 0.44
$$

$$b_{pref_{f2}} = \frac{1}{5} \left( (1 - \frac{1}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) \right) = 0.48
$$


### Esempio con Documenti Rilevanti

Consideriamo un esempio con 5 documenti rilevanti (verdi) e due ranking (due query).

##### Ranking 1:

* **Recall@k (0.2):** 1 documento rilevante su 5 (0.2 = 1/5).
* **Precision@3:** 0.67 (2 documenti rilevanti su 3 recuperati fino alla terza posizione).

**MAP:** La MAP è la media della precisione media su tutte le query.

**MRR:** La MRR è la media dei reciproci dei ranghi dei primi documenti rilevanti (verdi). Nel primo ranking, il primo documento rilevante è in posizione 1; nel secondo ranking è in posizione 2 (questo dato manca nel testo originale, ma è necessario per il calcolo del MRR).

##### Bpref:

* **Bpref per il Ranking 1:** $\frac{1}{5} \times (1 - \frac{1}{5})$  (poiché il secondo documento rilevante è in posizione 3 e prima c'è solo un documento non rilevante).
* **Bpref per il Ranking 2:** $\frac{1}{5} \times (1 - \frac{3}{5})$ (poiché ci sono 3 documenti non rilevanti prima del secondo documento rilevante).


## Top Weightedness

La **Top Weightedness** si riferisce alla capacità di un sistema di assegnare maggiore importanza ai risultati posizionati ai ranghi migliori.  Si vuole che il sistema attribuisca un peso maggiore ai risultati in cima alla lista, poiché sono considerati più rilevanti.

La misura **Discounted Cumulative Gain (DCG)** è un esempio di metrica che tiene conto della Top Weightedness.

## Guadagno Cumulativo Scontato (DCG)

La MRR (Mean Reciprocal Rank) presenta limitazioni, soprattutto per la sua sensibilità alla posizione del solo primo documento rilevante.  Il DCG (Discounted Cumulative Gain) offre una soluzione più sofisticata.

**Due assunzioni fondamentali del DCG:**

1. I documenti altamente rilevanti sono più utili dei documenti marginalmente rilevanti.
2. Più alta è la posizione di un documento rilevante nella classifica (rank bassi = posizioni migliori), meno utile è per l'utente, poiché è meno probabile che venga esaminato.

Un aspetto cruciale nella valutazione dei sistemi di Information Retrieval è la **top-weightedness**, ovvero la capacità di dare maggiore peso ai risultati posizionati più in alto nella lista.  Il DCG è una metrica progettata per valutare proprio questo aspetto.

Il DCG considera il guadagno (o utilità) ottenuto da ciascun documento recuperato, applicando un fattore di sconto che diminuisce all'aumentare del rango del documento.  A differenza di metriche più semplici, il DCG introduce il concetto di **rilevanza graduata**, ovvero un punteggio che quantifica il grado di rilevanza di un documento, invece di una semplice distinzione binaria (rilevante/irrilevante).

**Caratteristiche principali del DCG:**

* Utilizza la rilevanza graduata come misura di utilità, o guadagno, derivante dall'esame di un documento.
* Il guadagno viene accumulato a partire dalla cima della classifica e viene scontato (ridotto) per i ranghi maggiori.
* Lo sconto tipico è $\frac{1}{log_2(rango)}$, dove il logaritmo è solitamente in base 2.  Con base 2, lo sconto al rango 4 è 1/2, e al rango 8 è 1/3.


**Calcolo del DCG:**

Cosa succede se i giudizi di rilevanza sono in una scala di $[0, r]$? ($r > 2$)

* **Guadagno Cumulativo (CG) al rango n:**
    * Siano i **rating** (punteggi di rilevanza) degli n documenti $r_1, r_2, \dots, r_n$ (in ordine di classifica).
    * $CG = r_1 + r_2 + \dots + r_n$

* **Guadagno Cumulativo Scontato (DCG) al rango n:**
    * $DCG = r_1 + \frac{r_2}{\log_2(2)} + \frac{r_3}{\log_2(3)} + \dots + \frac{r_n}{\log_2(n)}$
    * Il logaritmo rappresenta il fattore di sconto applicato al rating.
* È possibile utilizzare basi diverse da 2 per il logaritmo.

##### Guadagno Cumulativo Scontato

* Il DCG (Discounted Cumulative Gain) è il guadagno totale accumulato ad un particolare rango p: 

$$DCG_{p}=rel_{1}+\sum_{i=2}^p \frac{rel_{i}}{\log_{2}i}$$

dove:

* $rel_1$ = il giudizio di rilevanza dato
* La sommatoria indica la posizione dei giudizi di rilevanza

##### Formulazione alternativa (per il recupero di documenti più rilevanti):

* Esiste una formulazione alternativa del DCG che enfatizza ulteriormente il recupero dei documenti più rilevanti, applicando uno scaling diverso ai guadagni cumulati.
$$DCG_{p}=\sum_{i=1}^p \frac{2^{rel_{i}}-1}{\log(1+i)}$$
* Utilizzata da alcune aziende di ricerca web.
* Enfasi sul recupero di documenti altamente rilevanti.

## NDCG per riassumere le classifiche


Il Discounted Cumulative Gain (DCG) è spesso normalizzato per ottenere il Normalized Discounted Cumulative Gain (NDCG).  Questa normalizzazione è supervisionata, richiedendo la conoscenza del ranking ideale per la query in questione.  L'NDCG si calcola dividendo il DCG ottenuto dal sistema per il DCG del ranking ideale, calcolato posizione per posizione.

* **NDCG (Normalized Discounted Cumulative Gain) al rango *n*:**  Normalizza il DCG al rango *n* con il valore DCG al rango *n* della classifica ideale. La classifica ideale ordina i documenti prima in base al livello di rilevanza più alto, poi al successivo più alto e così via.
* La normalizzazione è fondamentale per confrontare query con un numero variabile di risultati rilevanti.
* L'NDCG è attualmente un metodo molto popolare per la valutazione dei sistemi di ricerca sul Web ed è preferibile al DCG quando la rilevanza della query è particolarmente importante.  Il calcolo prevede la divisione del DCG ottenuto dal sistema per il DCG del ranking ideale, calcolato posizione per posizione.

Assumendo di conoscere il ranking ideale, si calcola il DCG sia per il ranking del sistema che per il ranking ideale.  Successivamente, si divide il DCG del ranking del sistema per il DCG del ranking ideale.

## Confronto di liste di classifiche: Tau di Kendall

La statistica di Tau di Kendall è una misura generale per confrontare i ranking. È una variante del coefficiente di correlazione di Pearson, ma funziona su ranking invece che su insiemi di dati. In sostanza, è un coefficiente di correlazione di Pearson calcolato su una coppia di variabili di lunghezza 1/2 n(n-1).
Si basa sul conteggio delle coppie concordanti (coppie di elementi che appaiono nello stesso ordine in entrambi i ranking) e discordanti (coppie di elementi che appaiono in ordine inverso).

La formula per il Tau di Kendall è:

$$\tau=\frac{n_{c}-n_{d}}{\frac{1}{2}n(n-1)}$$

Dove:

* **n:** numero totale di elementi
* **n(n -1)/2:** numero di coppie possibili
* **nc:** numero di coppie in concordanza (con lo stesso ordine nei due ranking)
* **nd:** numero di coppie in discordanza (con ordine diverso nei due ranking)

In pratica, il Tau di Kendall è calcolato come un coefficiente di correlazione tra due insiemi di N(N−1) valori binari. Ogni insieme rappresenta tutte le possibili coppie per N oggetti, e viene assegnato un valore di 1 quando una coppia è presente nello stesso ordine in entrambi i ranking e 0 se non lo è.

Un aspetto importante da considerare quando si calcolano statistiche di correlazione è di non valutare il significato di un singolo dato, ma l'insieme dei dati. Anche quando si annotano dati in liste, teoricamente non dovrebbero influenzare lo stesso punto.

##### Esempio:

Consideriamo due ranking di quattro elementi:

* **S1:** [0.4, 0.3, 0.2, 0.1] ⇒ [1, 2, 3, 4]
* **S2:** [0.4, 0.1, 0.25, 0.05] ⇒ [1, 3, 2, 4]

Le coppie possibili sono:

* **Pairs1:** {(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)}
* **Pairs2:** {(1,3), (1,2), (1,4), (3,2), (3,4), (2,4)}

In questo caso, nc=5 e nd=1, quindi τ = 0.67.

##### Significatività statistica:

Come per qualsiasi statistica di correlazione, è importante valutare la significatività statistica del Tau di Kendall. Questo significa determinare se l'accordo osservato tra i ranking è significativo o se potrebbe essere dovuto al caso.

##### Kappa measure:

Quando si confrontano più ranking, è utile utilizzare una misura statistica che valuti l'accordo tra i diversi predittori. Una misura comunemente utilizzata è la **Kappa measure**.

La Kappa measure è una misura di quanto i giudici (o i predittori) sono d'accordo o in disaccordo. È progettata per giudizi categorici e corregge per l'accordo casuale.

La formula per la *Kappa measure* è:

$$k=\frac{P(A)-P(E)}{1-P(E)}$$

Dove:

* **P(A):** proporzione di volte in cui i giudici sono d'accordo
* **P(E):** proporzione di accordo che ci aspetteremmo per caso

I valori di κ nell'intervallo $\left[ \frac{2}{3}, 1 \right]$ sono considerati accettabili. Valori più piccoli indicano che è necessario riprogettare la metodologia di valutazione della rilevanza utilizzata.

In sostanza, la Kappa measure confronta la probabilità di accordo osservata (P(A)) con la probabilità di accordo attesa per caso (P(E)).

## Esempio di Matrice di Accordo tra Giudici

Questo esempio illustra il calcolo della statistica Kappa per valutare l'accordo tra due giudici sulla rilevanza di elementi.

**Dati:**

Si consideri una matrice di accordo tra due giudici, dove:

* 300 volte entrambi i giudici hanno concordato sulla rilevanza di un elemento.
* 70 volte entrambi i giudici hanno concordato sulla non rilevanza di un elemento.
* Il numero totale di valutazioni è 400 per ogni giudice.


**Calcolo della proporzione osservata di accordo:**

La proporzione osservata di accordo tra i giudici,  `P(A)`, è calcolata come segue:

$$P(A) = \frac{300 + 70}{400} = \frac{370}{400} = 0.925$$


**Calcolo delle proporzioni marginali raggruppate:**

Le proporzioni marginali raggruppate rappresentano la probabilità di classificare un elemento come rilevante o non rilevante, considerando le valutazioni di entrambi i giudici:

$$P(nonrelevant) = \frac{(80 + 90)}{(400 + 400)} = \frac{170}{800} = 0.2125$$

$$P(relevant) = \frac{(320 + 310)}{(400 + 400)} = \frac{630}{800} = 0.7875$$


**Calcolo della probabilità di accordo casuale:**

La probabilità che i due giudici siano d'accordo per caso, `P(E)`, è calcolata come:

$$P(E) = P(nonrelevant)^2 + P(relevant)^2 = 0.2125^2 + 0.7875^2 = 0.665$$


**Calcolo della statistica Kappa:**

La statistica Kappa,  `κ`, misura l'accordo tra i giudici al di là di quello che ci si aspetterebbe per caso:

$$\kappa = \frac{P(A) - P(E)}{1 - P(E)} = \frac{0.925 - 0.665}{1 - 0.665} = 0.776$$

Questo valore di Kappa (0.776) indica un accordo accettabile tra i due giudici.


**Interpretazione:**

La matrice di accordo fornisce informazioni sulla concordanza tra i due giudici nella valutazione della rilevanza degli elementi.  In questo caso specifico, l'alto valore di Kappa suggerisce un elevato livello di accordo tra i giudici, andando oltre quanto ci si aspetterebbe per puro caso.


##### Probabilità di Accordo (p(A))

La probabilità di accordo (p(A)) rappresenta la frazione di volte in cui i due giudici sono d'accordo. Si calcola come:

```
p(A) = (Numero di accordi) / (Numero totale di valutazioni)
```

##### Probabilità Marginali

Le probabilità marginali servono per stimare la possibilità che i due giudici siano d'accordo per casualità. Si calcolano come segue:

```
p(E) = P(non rilevante)^2 + p(rilevante)^2
```

Dove:

* **p(non rilevante)** è la probabilità che un singolo giudice consideri l'elemento non rilevante.
* **p(rilevante)** è la probabilità che un singolo giudice consideri l'elemento rilevante.

##### Interpretazione

Confrontare la probabilità di accordo (p(A)) con la probabilità marginale (p(E)) ci permette di valutare se l'accordo tra i giudici è dovuto a una reale concordanza di opinioni o se è semplicemente frutto del caso. Se p(A) è significativamente maggiore di p(E), allora possiamo concludere che i giudici sono effettivamente d'accordo.
