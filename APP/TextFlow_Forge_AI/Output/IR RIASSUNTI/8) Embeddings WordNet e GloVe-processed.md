
## Riassunto: Stochastic Topic Modeling, NLP e Rappresentazione Semantica delle Parole

Questo testo tratta tre concetti principali: lo *Stochastic Topic Modeling*, l'Elaborazione del Linguaggio Naturale (NLP) e la rappresentazione semantica delle parole.

**1. Stochastic Topic Modeling:**  Non è un metodo di generazione di testo nel senso tradizionale dei *language model*, ma un processo generativo che caratterizza un documento come una composizione di distribuzioni di probabilità sui topic.  Il processo avviene tramite un doppio campionamento:  (1) campionamento di un topic da una distribuzione di probabilità sui topic; (2) campionamento di parole da una distribuzione di probabilità specifica per il topic selezionato; (3) campionamento dei topic per ogni documento da una distribuzione di probabilità sui topic.

**2. Elaborazione del Linguaggio Naturale (NLP):** L'NLP si concentra sullo sviluppo di sistemi automatici per comprendere e generare linguaggio naturale. Un problema fondamentale è la rappresentazione del linguaggio nel computer, in particolare la rappresentazione del significato delle parole.  La semantica denotativa associa un significante (parola) a un significato (concetto), ma cattura solo parzialmente la complessità del linguaggio.

**3. Limiti di WordNet e Rappresentazione Semantica:** WordNet, pur essendo una risorsa utile, presenta limiti: mancanza di sfumature di significato, incapacità di considerare il contesto, assenza di informazioni quantitative sulla appropriatezza delle parole, necessità di continuo aggiornamento, soggettività e difficoltà nel calcolare accuratamente la similarità tra parole.  Per superare questi limiti, si utilizzano tecniche di *embedding* per rappresentare le parole in uno spazio multidimensionale. Parole con significati simili sono rappresentate da punti vicini nello spazio. Un metodo comune per ottenere queste rappresentazioni si basa sulle co-occorrenze di parole in un corpus di testo.  Maggiore è la frequenza di co-occorrenza, maggiore è la similarità semantica.

---

## Rappresentazione delle Parole in NLP

Il testo descrive diverse tecniche per rappresentare le parole nel Natural Language Processing (NLP).  Inizialmente, vengono introdotte le rappresentazioni localiste, dove ogni parola è un simbolo discreto rappresentato da un vettore *one-hot*.  Questo metodo, pur semplice, presenta due importanti svantaggi: la mancanza di capacità di rappresentare similarità semantiche tra parole e la forte dipendenza dal vocabolario utilizzato.

Successivamente, viene introdotto il concetto di *Word Embedding*,  che si basa sull'idea di rappresentare le parole in uno spazio vettoriale a bassa dimensionalità, dove parole con significati simili sono rappresentate da vettori vicini.  Questo approccio è ispirato alla Latent Semantic Analysis (LSA) e si fonda sulla **semantica distribuzionale**.

La semantica distribuzionale afferma che il significato di una parola è determinato dalle parole che la circondano nel testo.  Questa informazione viene catturata tramite **pattern di co-occorrenza**:  il contesto di una parola *w* è definito come l'insieme delle parole che appaiono entro una finestra di dimensione fissa attorno ad essa.

Un metodo per costruire queste rappresentazioni è la **matrice di co-occorrenza a livello di documento**.  Questo approccio prevede:

1. La definizione di un vocabolario *V*.
2. La creazione di una matrice quadrata di dimensione |V| × |V|, inizialmente popolata di zeri.
3. Il conteggio delle co-occorrenze di parole all'interno di ogni documento: per ogni parola *w*, si incrementa il valore nella cella corrispondente alla riga di *w* e alla colonna di ogni altra parola *w'* presente nel documento.
4. La normalizzazione delle righe della matrice per renderla indipendente dalla lunghezza dei documenti.

L'immagine `![Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241111154837159.png|532]` illustra i problemi legati a questo metodo, che, nonostante la sua intuitività, risulta computazionalmente costoso e genera una rappresentazione sparsa.  Il testo non approfondisce soluzioni alternative a questo problema.

---

# Riassunto: Vettori di Parole e Word2Vec

Questo testo descrive metodi per creare rappresentazioni vettoriali di parole (word embeddings), focalizzandosi su Word2Vec.  La rappresentazione di parole tramite vettori densi permette di catturare similarità semantiche tramite il prodotto scalare tra i vettori.

## Problematiche della Co-occorrenza a Livello di Documento

L'approccio basato sulla co-occorrenza di parole presenta sfide:

* **Dimensione della finestra di contesto:** finestre troppo ampie catturano informazioni semantiche a discapito di quelle sintattiche, mentre finestre troppo piccole hanno l'effetto opposto.
* **Conteggi grezzi:** sovrastimano l'importanza delle parole frequenti.  L'utilizzo del logaritmo della frequenza aiuta a mitigare questo problema.
* **GloVe:** offre un approccio più sofisticato rispetto ai semplici conteggi, apprendendo rappresentazioni basate sulle co-occorrenze in un corpus.

## Word2Vec: Apprendimento di Vettori di Parole

Word2Vec (Mikolov et al., 2013) è un framework per apprendere vettori di parole.  L'idea principale è quella di:

1. Rappresentare ogni parola nel vocabolario con un vettore.
2. Per ogni posizione *t* nel testo, con parola centrale *c* e parole di contesto *o*, calcolare la probabilità delle parole di contesto dato la parola centrale (o viceversa).
3. Aggiustare iterativamente i vettori di parole per massimizzare questa probabilità.  Questo processo sfrutta il principio di località, considerando le parole vicine come contesto.

## Funzione Obiettivo di Word2Vec

La funzione obiettivo di Word2Vec mira a massimizzare la verosimiglianza di osservare le parole di contesto data la parola centrale.  Formalmente, si vuole massimizzare la likelihood:

$$\text{Likelihood}=L_{0}=\prod_{t=1}^T \prod_{-m\leq j\leq m}P(w_{t+j|w_{t};\theta})$$

dove `m` è la dimensione della finestra di contesto e  `θ` rappresenta i parametri del modello (i vettori delle parole).  In pratica, si minimizza la log-verosimiglianza negativa:

$$J(θ) = -\frac{1}{T} \sum_{t=1}^T\sum_{j≠0}\log P(W_{t+j} | W_{t}; θ)$$

Minimizzare questa funzione equivale a massimizzare l'accuratezza nella predizione delle parole di contesto.  Ogni parola ha due vettori distinti, uno per quando è parola centrale e uno per quando è parola di contesto.

---

### Word2Vec: Apprendimento di Rappresentazioni di Parole

Word2Vec apprende rappresentazioni vettoriali di parole tramite l'addestramento di una rete neurale su un corpus di testo.  Per ogni parola, vengono creati due vettori:  `Vw` (per la parola come parola target) e `Uw` (per la parola come parola di contesto).

La probabilità di una parola di contesto `o` data una parola target `c` è calcolata usando una softmax:

$$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w∈V} \exp(u_w^T v_c)}$$

L'algoritmo seleziona iterativamente una parola centrale e la sua finestra di contesto di dimensione *m* da un documento.  Il compito di apprendimento consiste nel predire, data una parola centrale, la probabilità di ogni parola nel vocabolario di essere una parola di contesto nella finestra.  La rete neurale utilizzata ha uno strato di input (one-hot encoding), uno strato nascosto di dimensione *N* (*N* << dimensione del vocabolario), e uno strato di output (one-hot encoding).  L'ottimizzazione dei parametri avviene tramite la discesa del gradiente stocastica (SGD). La funzione di costo è:

$$j(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m}\log P(w_{t+j}|w_{t};\theta)$$

dove  $P(w_{t+j}|w_{t};\theta) = P(o|c)$.  Il gradiente della funzione di costo rispetto al vettore del contesto $V_c$ è:

$$\frac{\delta}{\delta V_{c}}\log p(o|c)=u_{o}-\sum_{x\in V}p(x|c)v_{x}$$


### Derivazione delle Regole di Aggiornamento per la Regressione Logistica

La regressione logistica utilizza la cross-entropy come funzione di perdita:

$$L = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$

dove *y* è il valore reale (0 o 1) e $\hat{y}$ è la probabilità prevista. Il gradiente rispetto a *z* (l'output prima della sigmoide) è:

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

Il gradiente rispetto a *w* (i pesi) è:

$$\frac{\partial L}{\partial w} = (\hat{y}-y)x$$

dove *x* è il valore dell'input.  Questi gradienti vengono utilizzati per aggiornare i parametri del modello tramite la discesa del gradiente.

---

Questo documento descrive l'addestramento di modelli di word embedding, in particolare le architetture Skip-gram e CBOW di Word2Vec.

**Addestramento del modello:** L'addestramento si basa sulla minimizzazione di una funzione di perdita tramite la discesa del gradiente.  I parametri del modello ($w$ e $b$, rappresentati nel diagramma come un lungo vettore $O$) vengono aggiornati iterativamente usando le seguenti regole:

* $w^{(k+1)} = w^{(k)} - \lambda (\hat{y} - y)x$
* $b^{(k+1)} = b^{(k)} - \lambda (\hat{y} - y)$

dove:

* $k$ è l'iterazione corrente;
* $\lambda$ è il learning rate;
* $\hat{y}$ è la predizione;
* $y$ è il valore reale;
* $x$ è l'input.

Il gradiente rispetto a $b$ è $\frac{\partial L}{\partial b} = \hat{y} - y$.  ![[]] mostra la struttura dei parametri del modello, dove ogni parola ha un vettore di parole ($w_i$) e un vettore di contesto ($b_i$).

**Word2Vec: Skip-gram e CBOW:** Word2Vec utilizza due approcci principali:

* **Skip-gram:** Prende una parola target e cerca di predire le parole di contesto entro una finestra di dimensione definita.  L'associazione è uno-a-molti (una parola centrale, M parole di contesto).

* **CBOW:** Prende le parole di contesto e cerca di predire la parola target. L'associazione è uno-a-uno (M parole di contesto, una parola centrale).

Entrambe le varianti utilizzano una matrice di embedding (tra input e hidden layer) e una matrice di contesto (tra hidden layer e output layer) per mappare le parole in uno spazio vettoriale di dimensione D (o N).  In Skip-gram, spesso si usa la media dei vettori delle parole di contesto per la predizione.  CBOW media i vettori delle parole di contesto prima di passare all'output layer.

---

## Riassunto del Modello Word2Vec (CBOW e Skip-gram)

Questo documento descrive il modello Word2Vec, focalizzandosi su CBOW (Continuous Bag-of-Words) e Skip-gram, e sull'estrazione degli embeddings.

### CBOW

In CBOW, il modello prevede la parola centrale dato il suo contesto.  I vettori delle parole di contesto vengono mediati ($\hat{v}$) prima di essere moltiplicati per la matrice dei pesi tra lo strato nascosto e quello di output ($W'$).  Il risultato viene poi passato alla funzione softmax per ottenere la probabilità di ogni parola nel vocabolario.  La dimensione della rappresentazione delle parole è pari alla dimensione del vocabolario. La matrice dei pesi tra lo strato di input e lo strato nascosto è $W$, di dimensione $d \cdot n$, dove *n* è la dimensionalità dello strato nascosto.  L'embedding di una parola può essere estratto da una riga di $W$ (rappresentazione come parola target in Skip-gram) o da $W'$ (rappresentazione come parola di contesto in CBOW). ![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112101025711.png]] ![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112101035268.png|696]] ![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112101140496.png]]


### Skip-gram

In Skip-gram, il modello prevede le parole di contesto data una parola centrale. L'embedding della parola centrale è estratto direttamente dalla matrice dei pesi $W$.


### Estrazione degli Embeddings

In entrambi i modelli, gli embeddings delle parole sono ricavabili dalle matrici di pesi.  In CBOW, si possono ottenere gli embeddings sia dalla matrice $W$ (rappresentando le parole di contesto come parole target in Skip-gram) che dalla matrice $W'$.  In Skip-gram, gli embeddings sono estratti da $W$, rappresentando la parola di input come target.


### Softmax e Funzione di Costo

La probabilità di una parola data un'altra è calcolata usando la funzione softmax.  Tuttavia, la normalizzazione della softmax è computazionalmente costosa.


### Skip-gram con Negative Sampling

Per mitigare il costo computazionale della softmax, il *negative sampling* viene utilizzato.  La funzione di costo viene modificata come segue:

$$J_{t}(\theta)=\log\sigma(u_{o}^Tv_{c})+\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$$

dove:

* $\log\sigma(u_{o}^Tv_{c})$ rappresenta la probabilità della parola osservata.
* $\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$ approssima il denominatore della softmax, campionando *k* parole negative.

Questo approccio rende l'addestramento più efficiente.

---

Word2Vec addestra embedding di parole utilizzando un approccio discriminativo (o contrastivo).  Invece di confrontare una parola centrale con l'intero vocabolario, confronta con *k* parole negative campionate da una distribuzione $P(W)$, che è una distribuzione unigramma elevata alla potenza $\alpha$ (tipicamente 0.75). Questo riduce l'influenza delle parole più frequenti nel campionamento. L'obiettivo è massimizzare la similarità tra la parola centrale e le sue parole di contesto, e massimizzare la distanza tra la parola centrale e le parole negative.

La softmax gerarchica migliora l'efficienza di Word2Vec usando un albero di Huffman per organizzare le parole del vocabolario.  Questo riduce il numero di calcoli necessari per la softmax standard, accelerando l'addestramento.  L'albero è bilanciato, con parole più frequenti più vicine alla radice. La probabilità di una parola è calcolata percorrendo l'albero dalla radice alla foglia corrispondente.

Word2Vec offre due architetture principali:

* **Skip-gram:** Prevede le parole di contesto data una parola centrale. È più preciso per parole rare e finestre di contesto ampie, ma più lento nell'addestramento e meno efficiente per task *document-oriented*.

* **CBOW:** Prevede la parola centrale date le parole di contesto. È più preciso per parole frequenti, più veloce nell'addestramento e adatto a task *document-oriented*, ma meno preciso per parole rare.

La tabella seguente riassume le differenze tra Skip-gram e CBOW:

| Caratteristica | Skip-gram | CBOW |
|---|---|---|
| **Obiettivo** | Predire parole di contesto | Predire parola centrale |
| **Parole frequenti** | Meno preciso | Più preciso |
| **Parole poco frequenti** | Più preciso | Meno preciso |
| **Finestra di contesto** | Più grande | Più piccola |
| **Velocità di addestramento** | Più lento | Più veloce |
| **Task** | Similarità, *relatedness*, analogia | Classificazione, task document-oriented |

Skip-gram è generalmente preferibile per vocabolari ampi, corpus specialistici, parole rare, finestre di contesto ampie e task *word-oriented*.

---

# Word Embeddings: CBOW, Negative Sampling, e GloVe

Questo documento riassume le tecniche CBOW (Continuous Bag-of-Words) e GloVe per la creazione di *word embeddings*.

## CBOW

CBOW è un algoritmo veloce per l'addestramento di *word embeddings*, particolarmente adatto a vocabolari di dimensioni moderate, corpus generici e task *document-oriented*.  L'addestramento può utilizzare *Hierarchical Softmax* (ottimale per parole rare) o *Negative Sampling* (ottimale per parole frequenti e vettori a bassa dimensionalità). Il *sottocampionamento* delle parole frequenti migliora accuratezza e velocità, specialmente con dataset grandi (con probabilità tra 1e-3 e 1e-5). La dimensionalità dei vettori influenza la performance, generalmente valori maggiori sono preferibili, ma non sempre. La dimensione del contesto è tipicamente ~5 per CBOW e ~10 per Skip-gram.

L'aggiornamento del gradiente durante l'addestramento con *negative sampling* è iterativo per ogni finestra di parole ed è molto sparso:  $\nabla_{\theta} J_t(\theta) = \begin{bmatrix} 0 \\ \vdots \\ \nabla_{\theta_{target\_word}} \\ \vdots \\ \nabla_{\theta_{context\_word}} \\ \vdots \\ 0 \end{bmatrix} \in \mathbb{R}^{2dV}$.  Si aggiornano solo i vettori delle parole nella finestra, ottimizzando l'aggiornamento tramite operazioni sparse o hashing.  Questo è cruciale per l'efficienza con milioni di vettori e calcolo distribuito.

Le matrici di co-occorrenza possono essere costruite usando una finestra di parole (cattura informazioni sintattiche e semantiche) o l'intero documento (cattura temi generali, utile per l'Analisi Semantica Latente).

Le rappresentazioni basate direttamente sulle co-occorrenze (come in LSA) presentano limitazioni: sparsità della matrice e cattura principalmente di relazioni lineari, limitando la gestione della polisemia.


## GloVe

GloVe è una tecnica alternativa che combina approcci neurali e statistici.  A differenza di Word2Vec, GloVe considera il contesto delle parole, migliorando la gestione di parole poco frequenti e polisemiche.  Si basa sull'analisi delle co-occorrenze tra parole per costruire una matrice, sfruttando la probabilità di co-occorrenza per creare le rappresentazioni.  I vantaggi principali sono una migliore gestione della polisemia e una rappresentazione più accurata delle parole poco frequenti.

---

GloVe (Global Vectors for Word Representation) è un modello neurale per la creazione di word embedding, più efficiente di altri come Word2Vec.  A differenza di alcuni modelli, GloVe non genera un singolo embedding globale per ogni parola; invece, ogni parola è rappresentata da un unico embedding derivato dall'addestramento su un corpus specifico.  Questi embedding sono considerati rappresentazioni globali e context-free: ogni parola ha una rappresentazione statica, indipendente dal contesto.  Questo approccio, però, presenta limiti: la polisemia (parole con significati multipli) e la scarsità di dati per parole poco frequenti possono compromettere l'accuratezza delle rappresentazioni.

La funzione obiettivo di GloVe minimizza la differenza tra il prodotto scalare degli embedding di due parole e il logaritmo della loro probabilità di co-occorrenza.  La formula della loss function è:

$$ f(x)= \begin{cases} \left( \frac{x}{x_{max}} \right)^\alpha, & if \ x<x_{max} \\ 1, & otherwise \end{cases} $$

$$\text{Loss } J=\sum_{i,j=1}^Vf(X_{ij})(w_{i}T \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij})^2$$

Dove:

* $w_i$ e $w_j$ sono gli embedding delle parole $i$ e $j$.
* $b_i$ e $b_j$ sono i bias per le parole $i$ e $j$.
* $X_{ij}$ è il numero di co-occorrenze delle parole $i$ e $j$ in una finestra di contesto.
* $f(x)$ è una funzione di ponderazione che smorza l'influenza delle parole frequenti, utilizzando una distribuzione unigram con esponente $\frac{3}{4}$.
* $\log X_{ij}$ approssima $P(i|j)$, la probabilità di $w_i$ dato $w_j$.

L'algoritmo si concentra sui rapporti di co-occorrenza tra parole.  Considerando tre parole (i, j, k), il rapporto $P(k|i)/P(k|j)$ indica la correlazione relativa di k con i e j: un rapporto > 1 indica una maggiore correlazione di k con i, mentre un rapporto < 1 indica una maggiore correlazione con j.  La tabella seguente illustra un esempio:

| Probability and Ratio | $k = \text{solid}$ | $k = \text{gas}$ | $k = \text{water}$ | $k = \text{fashion}$ |
| ------------------------------------------------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| $P(k \| \text{ice})$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(k \| \text{steam})$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| $P(k \| \text{ice})/P(k \| \text{steam})$ | $8.9$ | $8.5 \times 10^{-2}$ | $1.36$ | $0.96$ |

---

Questo testo descrive un metodo per rappresentare le relazioni tra parole in uno spazio vettoriale, sfruttando gli *embeddings*.  L'obiettivo è catturare relazioni analogiche come "man" : "woman" :: "king" : "queen".  Ciò viene ottenuto tramite una funzione di confronto, $F$, che quantifica la relazione tra tre parole ($w_i$, $w_j$, $w_k$).

$F$ deve soddisfare due condizioni:

1. **Combinazione Lineare:**  $F((w_{i}-w_{j})^Tw_{k})=\frac{P(k|i)}{P(k|j)}$, dove $P(k|i)$ rappresenta la probabilità di osservare la parola $w_k$ dato $w_i$.  Questa condizione lega la funzione di confronto alla probabilità condizionale delle co-occorrenze.

2. **Simmetria:** $F((w_{i}-w_{j})^Tw_{k})=\frac{F(w_{i}^Tw_{k})}{F(w_{j}^Tw_{k})}$. Questa condizione garantisce che la funzione sia simmetrica rispetto alle parole coinvolte.

La funzione $F$ è definita come: $F(w_{i}^Tw_{k})=e^{w_{i}^Tw_{k}}=P(k|i) = \frac{x_{ik}}{x_{i}}$, dove $x_{ik}$ rappresenta il numero di co-occorrenze di $w_i$ e $w_k$, e $x_i$ il numero totale di occorrenze di $w_i$.

Infine, attraverso manipolazioni algebriche, si ottiene l'equazione: $w_{i}^Tw_{k}+b_{i}+b_{j}=\log X_{jk}$, dove $b_i$ e $b_j$ sono bias (inizializzati a 0) che incorporano il termine $\log x_i$, garantendo la simmetria.  Questa equazione confronta le co-occorrenze attese con quelle effettive delle parole.

---
