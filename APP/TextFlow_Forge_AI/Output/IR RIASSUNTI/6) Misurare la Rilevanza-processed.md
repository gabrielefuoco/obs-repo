
Un sistema di retrieval identifica risorse rilevanti a una query da un repository.  Due insiemi chiave sono:  `Relevant` (risorse realmente rilevanti) e `Retrieved` (risorse recuperate dal sistema). L'obiettivo è massimizzare l'intersezione tra questi due insiemi.

La valutazione del sistema si basa su una matrice di contingenza:

|             | Relevant | Nonrelevant |
|-------------|----------|-------------|
| **Retrieved** | tp       | fp           |
| **Not Retrieved** | fn       | tn           |

dove:

* `tp` (true positive): documenti rilevanti recuperati.
* `fp` (false positive): documenti non rilevanti recuperati.
* `fn` (false negative): documenti rilevanti non recuperati.
* `tn` (true negative): documenti non rilevanti non recuperati.

La valutazione richiede: una collezione di documenti, un set di query e una valutazione di rilevanza per ogni documento/query.

Le metriche principali per la valutazione del recupero non ordinato sono:

* **Precisione (P):** $\frac{tp}{tp + fp}$  (probabilità che un documento recuperato sia rilevante).
* **Richiamo (R):** $\frac{tp}{tp + fn}$ (probabilità che un documento rilevante sia recuperato; anche detto *true positive rate* o *sensitivity*).

Precisione e richiamo sono in conflitto: aumentare il richiamo (recuperando più documenti) generalmente diminuisce la precisione, e viceversa.  Un sistema molto selettivo avrà alta precisione e basso richiamo, mentre un sistema che esplora ampiamente avrà basso precisione e alto richiamo.

Per combinare precisione e richiamo, si usano misure come la media geometrica o, preferibilmente, la media armonica, che assegna pesi diversi a precisione e richiamo.

---

Il testo descrive diverse metriche per valutare le performance di sistemi di informazione e recupero di informazioni, focalizzandosi su misure basate su set e su ranking.

**Misure basate su set:**

* **F-measure:** Combinazione di precisione e richiamo, ponderata dal parametro β.  β > 1 privilegia il richiamo, β < 1 la precisione. La formula è:  `F = ((β²+1)PR) / (β²P + R)`, dove P è la precisione e R il richiamo.  La media armonica, una variante dell'F-measure, penalizza fortemente valori bassi di precisione o richiamo, più della media geometrica.  `![[]]` mostra graficamente questa differenza.

**Misure basate sul ranking:**  Queste metriche valutano l'ordinamento dei risultati.

* **Rilevanza binaria:**
    * **Precision@K (P@K):** Percentuale di documenti rilevanti tra i primi K risultati.  Esempio: `Prec@3 di 2/3` significa che 2 su 3 dei primi risultati sono rilevanti. `![[]]` illustra graficamente esempi di P@K.
    * **Media della Precisione Media (MAP):** Media della precisione per ogni query, considerando tutti i valori di K.
    * **Media del Rango Reciproco (MRR):** Reciproco del rango del primo documento rilevante per ogni query.
    * **Preferenza Binaria:** Probabilità che un documento rilevante sia classificato più in alto di uno non rilevante.

* **Livelli di rilevanza multipli:**
    * **NDCG (Normalized Discounted Cumulative Gain):** Considera l'ordine e il livello di rilevanza dei documenti, assegnando punteggi maggiori a documenti più rilevanti in posizioni alte.

* **Curva Precision-Richiamo:** Rappresenta graficamente la precisione in funzione del richiamo per ogni prefisso della lista ordinata (i primi 1, 2, 3, ecc. risultati).  Precisione e richiamo, inizialmente misure per set non ordinati, vengono adattate per il ranking calcolandole per ogni prefisso.  Recall@K è l'analogo di Precision@K per il richiamo.

---

## Curva Precision-Recall

La curva Precision-Recall visualizza le prestazioni di un sistema di ranking.  Si calcola la precisione per ogni query, considerando un numero variabile di documenti rilevanti.  Questi valori vengono poi interpolati a livelli di richiamo standard (0-1, incrementi di 0.1). L'interpolazione utilizza la massima precisione ottenuta per un livello di richiamo effettivo maggiore o uguale al livello standard.  Questo perché si assume che l'utente sia disposto a esaminare più risultati se sia precisione che richiamo migliorano.  L'interpolazione avviene quando si osserva una risalita nel grafico Precision-Recall.  ![[]]

## Mean Average Precision (MAP)

La Mean Average Precision (MAP) valuta le prestazioni di un sistema di recupero informazioni. Il calcolo avviene in quattro fasi:

1. **Identificazione delle posizioni:** Si individuano le posizioni nel ranking di ogni documento rilevante ($K_1, K_2, … K_R$).
2. **Precision@K:** Si calcola la Precision@K per ogni posizione rilevante (documenti rilevanti recuperati fino a K / totale documenti recuperati fino a K).
3. **Average Precision:** Si calcola la media delle Precision@K per tutti i documenti rilevanti di una singola query.
4. **MAP:** Si calcola la media aritmetica delle Average Precision ottenute per tutte le query.

Esempio:  ![[]]  In questo caso, Average Precision = $\frac{1}{3}\cdot\left( \frac{1}{1}+\frac{2}{3}+\frac{3}{5} \right)=0.76$. La MAP è la media di queste Average Precision su diverse query.

La MAP è una misura robusta perché considera la posizione dei documenti rilevanti nel ranking, premia i sistemi che posizionano i documenti rilevanti in alto e aggrega le prestazioni su diverse query.


## Media della Precisione Media (MAP)

La Media della Precisione Media (MAP) è una misura di performance che considera la precisione per ogni documento rilevante.  Se un documento rilevante non viene recuperato, la sua precisione è zero.  È una media macro (ogni query ha lo stesso peso) che presuppone l'interesse dell'utente a trovare molti documenti rilevanti per query. Richiede molti giudizi di rilevanza e la conoscenza della posizione di tutti i documenti rilevanti, aspetto considerato un limite.

---

## Riassunto delle Metriche di Performance per Sistemi di Recupero dell'Informazione

Questo documento introduce due metriche per valutare le performance dei sistemi di recupero dell'informazione: la Preferenza Binaria (*Bpref*) e il Mean Reciprocal Rank (*MRR*).

### Preferenza Binaria (*Bpref*)

*Bpref* è una metrica progettata per situazioni con giudizi di rilevanza incompleti.  A differenza di altre metriche, non richiede la conoscenza della posizione di *tutti* i documenti rilevanti.  Si concentra sulla relazione di preferenza tra documenti rilevanti e irrilevanti, valutando se i primi sono posizionati prima dei secondi nel ranking.  La formula è:

$$\text{bpref}=\frac{1}{R}\sum_{r}\left( 1-\frac{|\text{n ranked higer than r}|}{min(R,N)} \right)$$

dove:

* R: numero di documenti rilevanti giudicati.
* N: numero di documenti irrilevanti giudicati.
* r: documento rilevante recuperato.
* n: documento irrilevante recuperato prima di r.

*Bpref* è particolarmente utile quando si ha una conoscenza parziale dei documenti rilevanti e si vuole valutare la capacità del sistema di distinguere tra rilevante e irrilevante.  Mentre è altamente correlata a *MAP* (Mean Average Precision) con giudizi completi, mantiene una maggiore correlazione con la classificazione originale rispetto a *MAP* quando i giudizi sono incompleti.


### Mean Reciprocal Rank (*MRR*)

*MRR* è una metrica che si concentra sulla posizione del primo documento rilevante nel ranking.  Il punteggio Reciprocal Rank (*RR*) per una singola query è calcolato come $\frac{1}{K}$, dove K è il rango del primo documento rilevante.  L'*MRR* è la media dei punteggi *RR* su tutte le query.

**Vantaggi:** Semplicità di calcolo.

**Svantaggi:** Penalizza pesantemente i sistemi che non posizionano il primo documento rilevante nelle prime posizioni.

**Esempio:**  Se il primo documento rilevante è al rango 1 in una query e al rango 2 in un'altra, l'*MRR* è $\frac{\frac{1}{1} + \frac{1}{2}}{2} = 0.75$.

L'esempio fornito di calcolo di *bpref* mostra un'applicazione pratica della formula, ma senza ulteriori dettagli contestuali, non è possibile fornire un'interpretazione più approfondita.  L'esempio del calcolo del *MAP* è incluso ma non strettamente necessario per la comprensione delle due metriche principali.

---

## Riassunto del Testo sulle Metriche di Valutazione dell'Information Retrieval

Questo testo descrive diverse metriche per valutare l'efficacia dei sistemi di Information Retrieval, focalizzandosi sulla **Top Weightedness**, ovvero l'importanza di posizionare i risultati più rilevanti ai primi posti del ranking.

### Metriche di base: Precision, Recall, MAP, MRR e Bpref

Il testo introduce brevemente metriche come *Precision@k*, *Recall@k*, *Mean Average Precision (MAP)* e *Mean Reciprocal Rank (MRR)*,  illustrandone il calcolo con un esempio di 5 documenti rilevanti e due ranking.  Viene inoltre introdotto il *Bpref*, una metrica che considera la posizione dei documenti rilevanti nel ranking, calcolata come media pesata della proporzione di documenti non rilevanti prima di ogni documento rilevante.  Un esempio di calcolo di $b_{pref_{f2}}$ è fornito:

$$b_{pref_{f2}} = \frac{1}{5} \left( (1 - \frac{1}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) \right) = 0.48$$

### Discounted Cumulative Gain (DCG): una metrica per la Top Weightedness

Il testo evidenzia le limitazioni della MRR, sensibile solo al rango del primo documento rilevante, introducendo il **Discounted Cumulative Gain (DCG)** come soluzione più sofisticata.  Il DCG si basa su due assunzioni: 1) documenti altamente rilevanti sono più utili di quelli marginalmente rilevanti; 2) la posizione di un documento rilevante influenza la sua utilità per l'utente (i documenti in posizioni alte sono più utili).

Il DCG considera la **rilevanza graduata**, assegnando un punteggio di rilevanza a ciascun documento, anziché una semplice classificazione binaria (rilevante/irrilevante).  Il calcolo del DCG prevede l'accumulo dei punteggi di rilevanza ($r_i$) di ciascun documento, scontati in base alla loro posizione ($i$) nel ranking:

* **Guadagno Cumulativo (CG) al rango n:** $CG = r_1 + r_2 + \dots + r_n$
* **Guadagno Cumulativo Scontato (DCG) al rango n:** $DCG = r_1 + \frac{r_2}{\log_2(2)} + \frac{r_3}{\log_2(3)} + \dots + \frac{r_n}{\log_2(n)}$

Il fattore di sconto $\frac{1}{\log_2(i)}$ diminuisce all'aumentare del rango, dando maggiore peso ai documenti posizionati in alto nella classifica.  Il testo specifica che se i giudizi di rilevanza sono in una scala $[0, r]$ con $r > 2$, la formula del DCG rimane la stessa.

---

Il documento descrive metriche per la valutazione di sistemi di ranking, in particolare di sistemi di recupero informazioni.

### Discounted Cumulative Gain (DCG)

Il DCG misura il guadagno cumulativo scontato di un ranking.  Esistono due formulazioni principali:

* **Formulazione standard:**  $DCG_{p}=rel_{1}+\sum_{i=2}^p \frac{rel_{i}}{\log_{2}i}$, dove  `relᵢ` è il giudizio di rilevanza del documento in posizione `i`.  Questa formula sconta il contributo dei documenti in posizioni inferiori.

* **Formulazione alternativa:** $DCG_{p}=\sum_{i=1}^p \frac{2^{rel_{i}}-1}{\log(1+i)}$. Questa versione enfatizza maggiormente i documenti altamente rilevanti, applicando uno scaling diverso ai guadagni. È utilizzata da alcune aziende di ricerca web.


### Normalized Discounted Cumulative Gain (NDCG)

L'NDCG normalizza il DCG rispetto al DCG ideale (ottenuto con il ranking perfetto).  Questo permette di confrontare i ranking di query con un numero diverso di risultati rilevanti.  Si calcola dividendo il DCG del sistema per il DCG del ranking ideale:  `NDCG = DCG<sub>sistema</sub> / DCG<sub>ideale</sub>`.  L'NDCG è una metrica popolare per la valutazione dei sistemi di ricerca web, preferibile al DCG quando la rilevanza della query è cruciale.


### Tau di Kendall

Il Tau di Kendall è una misura per confrontare due ranking, basata sul conteggio delle coppie di elementi concordanti e discordanti tra i due ranking. La formula è:  $$\tau=\frac{n_{c}-n_{d}}{\frac{1}{2}n(n-1)}$$, dove `n` è il numero di elementi, `n<sub>c</sub>` il numero di coppie concordanti e `n<sub>d</sub>` il numero di coppie discordanti.  In sostanza, misura la correlazione tra due ranking, trattandoli come insiemi di coppie ordinate.  È importante considerare l'insieme dei dati e non singoli valori quando si interpretano le statistiche di correlazione.

---

Il testo descrive metodi per valutare l'accordo tra diversi ranking o giudizi, focalizzandosi sul Tau di Kendall e sulla Kappa measure.

Il Tau di Kendall misura l'accordo tra due ranking, considerando le coppie concordanti e discordanti.  Un esempio mostra il calcolo di τ a partire da due set di ranking.  La significatività statistica del Tau di Kendall, come per altre misure di correlazione, deve essere valutata.

La Kappa measure è invece più adatta per valutare l'accordo tra più di due giudici su giudizi categorici (es. rilevante/non rilevante), correggendo per l'accordo casuale.  La formula è:

$$k=\frac{P(A)-P(E)}{1-P(E)}$$

dove P(A) è la proporzione di accordo osservato e P(E) è la proporzione di accordo attesa per caso.  Valori di κ nell'intervallo $\left[ \frac{2}{3}, 1 \right]$ sono considerati accettabili.

Un esempio illustra il calcolo della Kappa measure a partire da una matrice di accordo tra due giudici che valutano la rilevanza di 400 elementi: 300 accordi su elementi rilevanti e 70 accordi su elementi non rilevanti.  Si calcola P(A) = 370/400 = 0.925.  Le probabilità marginali raggruppate (P(nonrelevant) e P(relevant)) vengono calcolate considerando le valutazioni di entrambi i giudici per poi determinare P(E) = 0.2125² + 0.7875² = 0.665.  Infine, κ = (0.925 - 0.665) / (1 - 0.665) = 0.776, indicando un accordo accettabile.

Il testo definisce anche la probabilità di accordo P(A) come il rapporto tra il numero di accordi e il numero totale di valutazioni, e spiega come calcolare le probabilità marginali P(non rilevante) e P(rilevante) per stimare P(E), la probabilità di accordo casuale.

---

Il confronto tra la probabilità di accordo tra giudici (p(A)) e la probabilità marginale di accordo casuale (p(E)) permette di determinare se l'accordo osservato è significativo o dovuto al caso.  Un valore di p(A) significativamente superiore a p(E) indica una reale concordanza di opinioni tra i giudici.

---
