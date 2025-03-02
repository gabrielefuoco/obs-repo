
## Riassunto: Topic Modeling, Basi di Conoscenza e Incertezza nell'Information Retrieval

Questo documento tratta tre concetti chiave nell'elaborazione del linguaggio naturale e nell'Information Retrieval: il Topic Modeling, le Basi di Conoscenza Lessicali e la gestione dell'incertezza.

**Topic Modeling:**  Il Topic Modeling, in particolare la sua variante stocastica, rappresenta un documento come una miscela di topic, dove ogni topic è definito come una distribuzione di probabilità sui termini.  Questo approccio, divenuto influente negli anni 2000, differisce dai modelli precedenti rappresentando i topic in modo probabilistico.

**Basi di Conoscenza Lessicali:**  Il documento evidenzia l'importanza delle Basi di Conoscenza Lessicali, come WordNet, nel Knowledge Management.  Queste risorse forniscono informazioni semantiche sulle parole e sono utilizzate in ricerche recenti per valutare la comprensione dei Language Models in specifici task.

**Information Retrieval e Incertezza:**  Il Vector Space Model, un approccio tradizionale all'Information Retrieval, viene rivisitato, includendo una versione probabilistica della funzione TF per il ranking dei risultati.  Si sottolinea l'importanza di considerare l'incertezza nelle query degli utenti, dovuta all'imprecisione della traduzione in linguaggio naturale delle esigenze informative.  Questa incertezza è affrontata introducendo modelli probabilistici, analogamente a quanto avviene nell' "uncertainty data mining",  dove l'incertezza nei dati, ad esempio quelli sensoriali, viene gestita tramite approcci probabilistici.


## Riassunto: Incertezza nelle Misurazioni e Modelli Probabilistici di Retrieval

Questo testo tratta l'incertezza nei dati e la sua gestione nei sistemi di retrieval delle informazioni.  L'incertezza è introdotta inizialmente con un esempio di sensori ottici per il particolato atmosferico, la cui misurazione è influenzata dall'umidità, portando a sovrastima.  Questo evidenzia la difficoltà di ottenere valori precisi nelle misurazioni sensoriali.

Nell'analisi dei dati, l'incertezza viene gestita associando a ciascun valore numerico una distribuzione di probabilità, tipicamente gaussiana.  Questo trasforma i vettori multidimensionali in insiemi di distribuzioni, aumentando la complessità ma migliorando il realismo.  Strumenti come la divergenza di Shannon-Jensen permettono di confrontare queste distribuzioni.

L'incertezza è particolarmente rilevante nel data mining, knowledge management e information retrieval, permettendo la creazione di sistemi di retrieval più sofisticati.  Diversi modelli probabilistici di retrieval vengono presentati:

* **Probability Ranking Principle:** Classifica i documenti in base alla probabilità di pertinenza.
* **Binary Independence Model:**  Modello bayesiano che assume l'indipendenza tra i termini.
* **Okapi BM25:**  Versione probabilistica del TF-IDF, considerando frequenza dei termini e lunghezza del documento.
* **Reti bayesiane:**  Modellano le dipendenze tra i termini, applicando l'inferenza bayesiana per gestire l'incertezza.

L'utilizzo di modelli probabilistici per il ranking dei documenti, sebbene matematicamente eleganti già negli anni 2000, è diventato fondamentale oggi grazie a tecnologie più efficienti.  Il principio di base è determinare il miglior *result set*, ordinando i documenti per rilevanza basandosi sulla probabilità di pertinenza.

Questo approccio probabilistico si formalizza come un problema di classificazione, dove si vuole determinare  `P(R=1|X)`, la probabilità che un documento `X` sia rilevante (`R=1`).  Utilizzando la formula di Bayes:

$$P(R=1|X) = \frac{{P(X|R=1) \cdot P(R=1)}}{P(X)}$$

dove `P(X|R=1)` è la likelihood, `P(R=1)` la probabilità a priori di rilevanza e `P(X)` la probabilità di osservare il documento `X`.

---

## Riassunto: Classificazione, Bias e Probability Ranking

Questo testo tratta la classificazione di documenti, focalizzandosi sul problema del *bias* e sull'applicazione del *Probability Ranking Principle* (PRP).

La classificazione, basata sulla formula di Price, utilizza la stima di massima verosimiglianza (*maximum likelihood estimate*) per il termine di *likelihood*, assumendo l'indipendenza condizionata tra le dimensioni (features) dato una classe.  Questa assunzione, sebbene semplifichi il calcolo, è spesso problematica, soprattutto nei dati testuali dove le dimensioni (es. parole) sono interdipendenti.  La stima delle probabilità *a priori* delle classi si ottiene da un *training set*. Il *bias*, derivante dalla violazione dell'indipendenza condizionata, è un problema significativo nella classificazione.  La validità dell'assunzione di indipendenza dipende dalla fase di pre-processamento dei dati; nei dati testuali, questa assunzione è spesso irrealistica a causa delle relazioni grammaticali e semantiche tra le parole.

Il PRP ordina i documenti in base alla probabilità di rilevanza  `P(R=1|D,Q)`, dove `R=1` indica rilevanza e `D` e `Q` rappresentano rispettivamente il documento e la query.  Questo approccio si basa sulle probabilità *a priori* `p(R=1), p(R=0)` e sulle probabilità condizionate `p(x|R=1), p(x|R=0)`, dove `x` rappresenta un documento specifico.  Un documento è considerato rilevante se `P(R=1|D,Q) > P(R=0|D,Q)`.  L'utilizzo del PRP minimizza il rischio Bayesiano, corrispondente all'errore di classificazione.  Il termine `P(X)` è costante rispetto alle classi e non influenza il ranking, che dipende quindi da `P(X|R=1)` e `P(R=1)`.  Anche con un modello semplificato come quello Naive Bayes, si possono ottenere buone prestazioni se le features rilevanti sono ben descritte.

---

## Riassunto del Modello BIM e Applicazione del Teorema di Bayes per il Ranking di Documenti

Il Binary Independence Model (BIM) rappresenta i documenti come vettori binari, dove `x<sub>i</sub> = 1` indica la presenza del termine *i*-esimo e `x<sub>i</sub> = 0` la sua assenza.  Il modello assume l'indipendenza tra i termini.  L'obiettivo è il ranking dei documenti in base alla rilevanza ad una query, calcolando  `P(R=1|D,Q)` (probabilità di rilevanza) e `P(R=0|D,Q)` (probabilità di non rilevanza).

Per il ranking, si utilizza il rapporto di probabilità (odds) `O(R|Q\vec{X}) = \frac{P(R=1|Q\vec{X})}{P(R=0|Q\vec{X})}`,  dove `R` rappresenta la rilevanza, `Q` la query e `X` il documento.  Questo rapporto, ottenuto applicando il teorema di Bayes a ciascun termine, permette di ordinare i documenti senza necessità di calcolare probabilità assolute.

Gli Odds Ratio (OR) sono una misura di associazione tra due variabili binarie, calcolata come  `OR = \frac{n_{11} \cdot n_{00}}{n_{10} \cdot n_{01}}`  da una tabella di contingenza 2x2.  Sono applicati in diversi campi, tra cui epidemiologia, ricerca sociale e marketing.

Il Rischio Relativo (RR) e l'Odds Ratio (OR) sono misure di associazione tra un evento e una variabile.  Il RR è il rapporto tra la probabilità di un evento in presenza e in assenza della variabile. L'OR è il rapporto tra le odds di un evento in due gruppi diversi.  Mentre RR = 1 indica assenza di associazione,  OR = 1 indica assenza di differenza nelle odds.  In caso di bassa prevalenza dell'evento, l'OR può approssimare il RR.

---

## Riassunto: Odds Ratio, Rischio Relativo e OTS-I

Questo documento tratta due misure di associazione in epidemiologia, l'**Odds Ratio (OR)** e il **Rischio Relativo (RR)**, e introduce l'**OTS-I (Odd of Term Significance - I)**.

### Odds Ratio (OR) e Rischio Relativo (RR)

L'OR è il rapporto tra le odds di un evento nel gruppo esposto a un fattore di rischio e le odds nello stesso evento nel gruppo non esposto.  È definita come:

OR = (Odds di Y in presenza di X) / (Odds di Y in assenza di X)

dove X è l'esposizione al fattore e Y l'incidenza della malattia.  L'OR è non simmetrica e robusta, stimabile anche in studi retrospettivi.  Per eventi rari, approssima il RR.

Il RR è il rapporto tra il rischio di un evento nel gruppo esposto e il rischio nel gruppo non esposto.  In studi retrospettivi si preferisce l'OR, mentre in quelli prospettici il RR, che fornisce una misura diretta del rischio.  Per eventi rari, l'OR approssima bene il RR; per eventi frequenti, può sovrastimarlo.  Un esempio mostra il calcolo dell'OR per l'associazione tra fumo e malattia, interpretando il risultato come un aumento del rischio nei fumatori.

### OTS-I (Odd of Term Significance - I)

L'OTS-I è una query che valuta la rilevanza di un termine (R=1 per rilevante, R=0 per non rilevante).  La sua formula chiave è il rapporto di likelihood:

$$\frac{PR(Q | R=1)}{PR (Q | R=0)}$$

dove PR(Q | R=1) è la probabilità di osservare il dato Q dato che l'ipotesi di rilevanza è vera, e PR(Q | R=0) è la probabilità di osservare Q dato che è falsa.  Sotto l'ipotesi di indipendenza tra i termini, questo rapporto diventa una produttoria:

$$OTS-i(Q)=
\frac{p(\vec{x} \mid R = 1, q)}{p(\vec{x} \mid R = 0, q)} = \prod_{i = 1}^{n} \frac{p(x_i \mid R = 1, q)}{p(x_i \mid R = 0, q)}
$$

dove X<sub>i</sub> indica la presenza o assenza del termine i-esimo nel documento.  Il focus principale è sulla stima del rapporto tra le probabilità di osservare un dato Q, data la verità o falsità dell'ipotesi di rilevanza.

---

Questo testo descrive un metodo per stimare la produttoria di likelihood nella valutazione della rilevanza di un documento.  La produttoria viene suddivisa in due parti: una per i termini presenti nel documento  ($\prod_{x_{i}=1} \frac{p\left(x_{i}=1 \mid R=1,q\right)}{p\left(x_{i}=1 \mid R=0,q\right)}$) e una per i termini assenti ($\prod_{x_{i}=0} \frac{p\left(x_{i}=0 \mid R=1,q\right)}{p\left(x_{i}=0 \mid R=0,q\right)}$).  Per semplificare, si usa la notazione  $\prod_{i=1}^{n} \left( \frac{P_i}{R_i} \right)$, dove $P_i$ è la likelihood di presenza del termine *i*-esimo dato che il documento è rilevante, e $R_i$ la likelihood di presenza dello stesso termine dato che il documento non è rilevante.

Il testo introduce poi il concetto di probabilità di osservare un termine in un documento, considerando sia la rilevanza (R) che la novità (t) del termine.  Una tabella di contingenza illustra la relazione tra presenza/assenza del termine ($x_i = 1$ o $x_i = 0$) e rilevanza/non rilevanza (r=1 o r=0) del documento:

| | Rilevanza (r=1) | Non Rilevanza (r=0) |
| ------------------ | --------------- | ------------------- |
| Presenza $(x_i=1)$ | $p_i$ | $r_i$ |
| Assenza $(x_i=0)$ | $(1-p_i)$ | $(1-r_i)$ |

dove $p_i$ e $r_i$ rappresentano rispettivamente la probabilità di osservare il termine *i*-esimo dato che il documento è rilevante o non rilevante.  Infine, il testo accenna al calcolo degli odds, distinguendo tra termini presenti ("matching") e assenti ("non matching") sia nella query che nel documento.

---

## Riassunto della Derivazione della Formula Finale e del Retrieval Status Value (RSV)

Questo documento descrive la derivazione di una formula per il calcolo della rilevanza di un documento rispetto ad una query, culminando nel calcolo del *Retrieval Status Value* (RSV).  Il modello assume che l'assenza di termini nella query non venga considerata e che la probabilità di occorrenza di un termine non presente nella query non venga valutata.

La formula finale per gli *odds* di rilevanza (R) dato un documento  `x` e una query `q` è:

$$
\begin{aligned}
O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i=q_i=1}} \frac{p_i(1-r_i)}{r_i(1-p_i)} \cdot \prod_{\substack{q_i=1 }} \frac{1-p_i}{1-r_i}
\end{aligned}
$$

dove:

* `O(R|q)` rappresenta gli *odds* a priori di rilevanza dato `q`.
* La prima produttoria itera sui termini presenti sia nella query (`q_i = 1`) che nel documento (`x_i = 1`), calcolando un *odds ratio* per ogni termine "match".  `p_i` rappresenta la probabilità di osservare il termine `i` in un documento rilevante, e `r_i` la probabilità in un documento non rilevante.
* La seconda produttoria itera su tutti i termini presenti nella query (`q_i = 1`), considerando i termini "non-match" (presenti nella query ma non nel documento).

Il *Retrieval Status Value* (RSV) è un punteggio di rilevanza derivato da questa formula.  Il suo calcolo prevede:

1. Il calcolo dell' *odds ratio* per ogni termine della query presente nel documento. Questo *odds ratio* rappresenta il rapporto tra la probabilità che il termine appaia nel documento rilevante e la probabilità che appaia in un documento non rilevante.
2. La somma dei logaritmi di questi *odds ratio*.
3. Il risultato di questa somma rappresenta l'RSV.  Un RSV più alto indica una maggiore rilevanza.

Un esempio semplice con una query di due termini ("X" e "Q") entrambi presenti nel documento, con *odds ratio* pari a 1 per entrambi, porta ad un RSV di 0 (log(1) + log(1) = 0).

---

## Riassunto: Stima delle Probabilità e Analisi dei Dati di Addestramento nei Modelli Linguistici

Questo documento tratta la stima delle probabilità e l'analisi dei dati di addestramento nei modelli linguistici, focalizzandosi sul calcolo dell'Odds Ratio (Oz Ratio).

### Stima delle Probabilità e Odds Ratio

Per calcolare l'Odds Ratio, si stimano le probabilità di occorrenza di un termine in documenti rilevanti e non rilevanti, utilizzando tecniche come lo smoothing di Laplace.  L'Odds Ratio ($c_i$) quantifica la forza dell'associazione tra un termine e la rilevanza di un documento, calcolato come:

$$c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$$

dove $p_i = \frac{s}{S}$ è la probabilità che il termine sia presente in un documento rilevante (s documenti rilevanti contengono il termine, S documenti totali rilevanti), e $r_i = \frac{n-s}{N-S}$ è la probabilità che il termine sia presente in un documento non rilevante (n documenti contengono il termine, N documenti totali).  L'Odds Ratio può essere espresso anche in termini di conteggi:

$$c_i = K(N,n,S,s) = \log \frac{s/(S-s)}{(n-s)/(N-n-S+s)}$$

Una tabella di contingenza riassume le relazioni tra documenti rilevanti/non rilevanti e presenza/assenza del termine:

| Documents | Relevant | Non-Relevant | Total |
| --------- | -------- | ------------ | ----- |
| $x_i=1$ | s | n-s | n |
| $x_i=0$ | S-s | N-n-S+s | N-n |
| **Total** | **S** | **N-S** | **N** |


### Analisi dei Dati di Addestramento

L'analisi dei dati di addestramento dei modelli linguistici è complessa.  Una sfida principale è l'"inversione del modello", ovvero risalire ai dati originali dal modello e dai suoi output.  La trasparenza riguardo ai dati di addestramento è attualmente limitata, soprattutto per i modelli open source.  Un'analisi efficace richiederebbe l'addestramento di un modello specifico per riconoscere il contesto dei dati originali, un processo altrettanto complesso.

### Simboli e Definizioni

* **RSV:** Logaritmo del produttore di all special:
 $$
RSV = \log \prod_{x_i=q_i=1} \frac{p_i(1-r_i)}{r_i(1-p_i)} = \sum_{x_i=q_i=1} \log \frac{p_i(1-r_i)}{r_i(1-p_i)}
$$
* **i:** Numero di termini (dimensione del vocabolario).
* **n, s, N, S:** Simboli che indicano la dipendenza a livello di rappresentazione di testi e binariet.
$$
RSV = \sum_{x_i=q_i=1} c_i; \quad c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}
$$

### Approssimazione dei Documenti Non Rilevanti

La probabilità $r_i$ può essere approssimata considerando l'insieme dei documenti non rilevanti come l'intera collezione di documenti.  In questo caso, $r_i$ diventa la frazione di documenti nella collezione totale che contengono il termine di ricerca (N piccolo / N grande).

---

# Riassunto del testo: Oz di competenza, Probabilità, Smoothing e Relevance Feedback

Questo testo tratta la stima della rilevanza dei termini nella ricerca di informazioni, introducendo concetti di probabilità e smoothing per migliorare l'accuratezza dei modelli.

## Oz di competenza e IDF

L' *Oz di competenza* è approssimato tramite il logaritmo inverso della frequenza del documento (IDF):  $$\log \frac{1-r_i}{r_i} \approx \log \frac{N}{n} = IDF$$, dove  `N` è il numero totale di documenti e `n` il numero di documenti contenenti il termine `i`.  Questa approssimazione è valida soprattutto in grandi corpus come il web, ma meno accurata in corpus tematici a causa della legge di Zipf.  Il logaritmo del rapporto tra il numero di documenti totali e quelli contenenti un termine rappresenta la norma inversa del termine.

## Probabilità, Smoothing e TFPF

L'approccio probabilistico introduce incertezza nella stima della rilevanza. La *Term Frequency-Probability Factor* (TFPF) è una misura robusta, coerente con la legge di Zipf. Lo *smoothing* è una tecnica per evitare risposte troppo generose del modello, penalizzando eventi frequenti e facendo emergere eventi rari.  Una formula di smoothing bayesiana è: $$ P_{i}^{(h+1)} = \frac{\left|V_{i}\right|+\kappa p_{i}^{(h)}}{\left|V\right|+K} $$, dove  `K` è il parametro di smoothing. Lo smoothing introduce fattori nascosti che rappresentano la correlazione tra i termini.

## Stima di  `pᵢ` e Relevance Feedback

Il testo prosegue con la stima di  `pᵢ`, la probabilità di incidenza del termine nei documenti rilevanti.  Vengono presentati tre approcci:

1. **Stima da un sottoinsieme di documenti etichettati:** Stima  `pᵢ` da documenti etichettati come rilevanti, migliorando la stima tramite feedback dell'utente.
2. **Utilizzo di teorie e leggi:** Approssima  `pᵢ` usando teorie e leggi note.
3. **Probabilistic Relevance Feedback:** Utilizza il feedback dell'utente per raffinare i risultati della ricerca, migliorando l'accuratezza della query spesso tramite espansione della query stessa. L'obiettivo è migliorare la risposta alla query, rendendola più accurata.

---

# Riassunto: Relevance Feedback e Miglioramento del Recupero Informazioni

Questo documento descrive tecniche per migliorare la pertinenza dei risultati di ricerca, focalizzandosi sul *relevance feedback* e su metodi per approssimare la probabilità di rilevanza di un termine.

## Relevance Feedback

Il relevance feedback è un processo iterativo che migliora la qualità dei risultati di ricerca.  L'utente indica quali risultati sono rilevanti, e questo feedback viene usato per raffinare la query iniziale, ottenendo risultati più pertinenti.  Il **probabilistic relevance feedback** utilizza un modello probabilistico per stimare la probabilità che un documento sia rilevante, basandosi sulla presenza/assenza di termini specifici nei documenti e sulla loro rilevanza per la query.  Il processo prevede: stima iniziale delle probabilità, identificazione dei documenti rilevanti dall'utente, raffinamento delle stime basate sul feedback, e ricerca iterativa con le nuove stime.  Un concetto chiave è l'**inverso documento**, che indica la probabilità di un termine in un documento rilevante.

## Approssimazione della Probabilità di Rilevanza

La probabilità di rilevanza di un termine in un documento è approssimata dal rapporto tra la cardinalità dell'insieme dei documenti rilevanti (V) e quella di tutti i documenti (I):  `$$P(rilevanza | termine) ≈ \frac{|V|}{|I|} $$`.  Per evitare costi computazionali elevati con grandi insiemi V, si usa un meccanismo iterativo che raffina progressivamente l'insieme dei documenti candidati.  Lo **smoothing**, con parametro K (tipicamente 5 o 10), introduce una probabilità basata su eventi non osservati, evitando probabilità zero per termini assenti nell'insieme corrente.  In assenza di feedback esplicito, si può usare il **pseudo relevance feedback**, considerando i documenti più in alto nel ranking come altamente rilevanti.

## Miglioramento della Stima e Modelli di Recupero

La stima della rilevanza dei documenti può essere migliorata usando un parametro K che rappresenta la cardinalità dell'insieme iniziale di documenti. Un K maggiore porta a stime più accurate.  Modelli come BM25, pur assegnando punteggi basati su frequenze di parole chiave, possono non catturare completamente la rilevanza contestuale a causa della loro parametrizzazione e normalizzazione della richiesta.  L'esempio del modello vettoriale con due documenti "terri" a grana grossa dimostra la capacità di distinguere documenti anche con informazioni limitate.

---
