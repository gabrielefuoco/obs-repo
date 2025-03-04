
# Modelli Stocastici di Topic e Rappresentazione Semantica delle Parole

## I. Modelli Stocastici di Topic

Il processo generativo dei modelli stocastici di topic non prevede la generazione parola per parola del testo.  Invece, caratterizza un documento come una composizione di distribuzioni di probabilità. Ogni distribuzione rappresenta la probabilità di osservare una parola specifica all'interno di un determinato *topic*. Il processo di generazione del testo avviene attraverso un doppio campionamento:

1. Campionamento del *topic* da una distribuzione di probabilità sui *topic*.
2. Campionamento delle parole da una distribuzione di probabilità sulle parole, specifica per il *topic* campionato al punto precedente.
3. Campionamento dei *topic* per il documento da una distribuzione di probabilità sui *topic*.


## II. Elaborazione del Linguaggio Naturale (NLP)

L'obiettivo dell'NLP è sviluppare sistemi di intelligenza artificiale capaci di comprendere e generare linguaggio naturale, simulando il processo di apprendimento linguistico di un bambino.  Una sfida fondamentale è la **rappresentazione del linguaggio**, ovvero trovare un modo per rappresentare il linguaggio in modo che possa essere elaborato e generato in modo robusto.  Un aspetto cruciale di questa rappresentazione è l'associazione di unità lessicali (parole) a concetti, ovvero la **semantica denotativa**.  Ad esempio, la parola "albero" può essere associata a diverse rappresentazioni visive: 🌲, 🌳, 🪴.


## III. Limiti di WordNet

WordNet, pur essendo una risorsa utile, presenta diverse limitazioni:

* **Mancanza di sfumature e contesto:** Non riesce a catturare tutte le sfumature di significato e il contesto d'uso delle parole. Ad esempio, "proficient" è approssimativamente equivalente a "good" solo in alcuni contesti specifici.
* **Informazione quantitativa:** Non fornisce informazioni quantitative sull'appropriatezza di una parola in un dato contesto.
* **Mantenimento:** Richiede aggiornamenti continui e costosi a causa della costante evoluzione del linguaggio.
* **Soggettività:** È soggetto a bias culturali (es. prospettiva britannica).
* **Similarità:** Non permette il calcolo accurato della similarità semantica tra parole.


## IV. Rappresentazione Semantica delle Parole

Gli *embedding* rappresentano le parole in uno spazio multidimensionale, permettendo il calcolo della similarità semantica.  Il principio chiave è che parole sinonime sono rappresentate da punti vicini nello spazio. Un metodo comune per ottenere questi *embedding* è l'analisi delle co-occorrenze di parole in un ampio corpus di testo: maggiore è la frequenza di co-occorrenza, maggiore è la similarità di significato.


## Rappresentazione delle Parole in NLP

### I. Rappresentazioni Tradizionali

* **A. Simboli Discreti:**  Le parole possono essere rappresentate come simboli discreti, distinguendo tra il *tipo di parola* (entry nel vocabolario) e il *token di parola* (occorrenza nel contesto).

* **B. Vettori One-hot:** Questa rappresentazione localista assegna a ogni parola un vettore di dimensione pari alla cardinalità del vocabolario.  Solo una posizione del vettore ha valore "1", mentre tutte le altre sono "0".  Esempio: *motel* = $[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]$.

    * **Svantaggi:**  Mancanza di capacità di catturare la similarità semantica tra parole e forte dipendenza dal vocabolario utilizzato.


### II. Word Embedding

* **A. Evoluzione da LSA:** Gli *embedding* rappresentano un'evoluzione delle tecniche come l'Analisi Semantica Latente (LSA), che utilizza trasformazioni lineari per catturare relazioni semantiche e ridurre la dimensionalità dello spazio vettoriale.

* **B. Proprietà di Prossimità:**  La proprietà fondamentale degli *embedding* è che parole simili sono rappresentate da vettori vicini nello spazio vettoriale.


### III. Semantica Distribuzionale

* **A. Principio Fondamentale:** Il significato di una parola è determinato dalle parole che la circondano (contesto distribuzionale).

* **B. Pattern di Co-occorrenza:** L'analisi dei *pattern* di co-occorrenza delle parole permette di inferire la similarità semantica.

* **C. Contesto:** Il contesto di una parola *w* è definito come l'insieme delle parole vicine a *w* entro una finestra di dimensione fissa.


### IV. Matrice di Co-occorrenza a Livello di Documento

* **A. Costruzione:**

    1. Definire un vocabolario *V*.
    2. Creare una matrice |V| × |V| (inizialmente a zero).
    3. Contare le co-occorrenze di parole all'interno di ogni documento.
    4. Normalizzare le righe (dividere per la somma dei valori della riga).

* **B. Risultato:** La matrice di co-occorrenza fornisce una rappresentazione sparsa delle relazioni semantiche tra le parole, ma rimane computazionalmente costosa.

* **C. Problemi:** ![[ ]]

---

# Co-occorrenza a livello di documento: Problematiche e Soluzioni

## Dimensioni della finestra di co-occorrenza

* **Finestre grandi:** Catturano informazioni semantiche e di argomento, ma perdono dettagli sintattici.
* **Finestre piccole:** Catturano informazioni sintattiche, ma perdono dettagli semantici e di argomento.
* **Conteggi grezzi:** Sovrastimano l'importanza delle parole frequenti.

### Miglioramento dei conteggi

* **Logaritmo della frequenza:** Mitiga l'effetto delle parole comuni.
* **GloVe:** Apprende rappresentazioni di parole basate su co-occorrenze in un corpus (approccio più avanzato).


# Vettori di Parole (Word Embeddings)

* **Obiettivo:** Creare vettori densi per ogni parola, simili per parole in contesti simili.
* **Similarità:** Misurata tramite prodotto scalare dei vettori.
* **Rappresentazione:** Distribuita (spesso neurale).


# Word2vec: Apprendimento di Vettori di Parole

* **Idea:** Apprendere vettori di parole da un grande corpus di testo, massimizzando la probabilità di co-occorrenza tra parole centrali e di contesto.
* **Processo:**
    * Ogni parola ha un vettore.
    * Si itera su ogni posizione nel testo (parola centrale e parole di contesto).
    * Si usa la similarità dei vettori per calcolare la probabilità di una parola di contesto data la parola centrale (o viceversa).
    * Si aggiornano i vettori per massimizzare questa probabilità.

* **Funzione obiettivo:** Minimizzare la log-verosimiglianza:

   $$ \text{Likelihood}=L_{0}=\prod_{t=1}^T \prod_{-m\leq j\leq m}P(w_{t+j}|w_{t};\theta) $$

   dove  $P(w_{t+j}|w_{t};\theta)$ è la probabilità della parola di contesto shiftata di $j$ rispetto a $t$, data $w_{t}$.

* **Minimizzazione:** Minimizzare la seguente funzione obiettivo:

   $$J(θ) = -\frac{1}{T} \sum_{t=1}^T\sum_{j≠0}\log P(W_{t+j} | W_{t}; θ)$$

   Questo implica massimizzare l'accuratezza di predizione delle parole di contesto.


# I. Calcolo della Probabilità Condizionata

* **Utilizzo di due vettori per parola:**
    * `Vw`: vettore per parola centrale (target word).
    * `Uw`: vettore per parola di contesto (context word).

* **Formula per la probabilità condizionata:**

   * $P(o|c) = \frac{\exp(u_o^T v_c)}{Σ_{w∈V} \exp(u_w^T v_c)}$ (softmax)

* **Distinzione tra vettori:** Necessaria per gestire parole che appaiono in entrambi i ruoli (target e contesto).


# II. Word2vec: Apprendimento delle Rappresentazioni delle Parole

* **Metodo di addestramento:**
    * Selezione iterativa di una parola centrale e della sua finestra di contesto (dimensione *m*) da un documento di lunghezza *T*.
    * Creazione di esempi di addestramento.

* **Rete neurale:**
    * Un singolo strato nascosto di dimensione *N* (*N* << *V*, dove *V* è la dimensione del vocabolario).
    * Strati di input/output: vettori one-hot di dimensione *V*.

* **Compito di apprendimento:** Predire la probabilità di una parola di contesto data una parola centrale.
* **Ottimizzazione:** Discesa del gradiente stocastica (SGD).
* **Funzione di costo:**

   * $j(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m}\log P(w_{t+j}|w_{t};\theta)$

   dove $P(w_{t+j}|w_{t};\theta) = P(o|c)$ (probabilità calcolata tramite softmax).

* **Calcolo del gradiente (esempio per Vc):**

   * $\frac{\delta J}{\delta V_{c}} = u_{o}-\sum_{x\in V}p(x|c)v_{x}$


# III. Derivazione delle Regole di Aggiornamento per la Regressione Logistica

* **Funzione di perdita:** Cross-entropy

   * $L = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$

   * *y*: valore reale (0 o 1)
   * $\hat{y}$: probabilità prevista (tra 0 e 1)

* **Calcolo del gradiente:**
    * **Rispetto a *z*:**
        * $\frac{\partial L}{\partial z} = \hat{y} - y$
    * **Rispetto a *w*:**
        * $\frac{\partial L}{\partial w} = (\hat{y}-y)x$ (dove *x* è il valore dell'input)


# I. Calcolo del Gradiente e Aggiornamento dei Parametri

* **1.** (Il testo termina bruscamente qui. Manca la parte finale della sezione I.)

---

# Word2Vec: Algoritmi e Implementazione

## I. Fondamenti di Rete Neuronale e Backpropagation

**1. Gradiente rispetto a *w* e *b***:

Il gradiente della funzione di perdita L rispetto ai parametri *w* e *b* è calcolato come segue:

* Gradiente rispetto a *w*: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial w} = (\hat{y} - y)x$
* Gradiente rispetto a *b*: $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} = \hat{y} - y$

**2. Regole di Aggiornamento (Discesa del Gradiente):**

Le regole di aggiornamento dei parametri *w* e *b* durante la discesa del gradiente sono:

* $w^{(k+1)} = w^{(k)} - \lambda \frac{\partial L}{\partial w^{(k)}}$
* $b^{(k+1)} = b^{(k)} - \lambda \frac{\partial L}{\partial b^{(k)}}$

Sostituendo i gradienti calcolati precedentemente:

* $w^{(k+1)} = w^{(k)} - \lambda (\hat{y} - y)x$
* $b^{(k+1)} = b^{(k)} - \lambda (\hat{y} - y)$

**3. Addestramento del Modello:**

L'addestramento del modello consiste nell'ottimizzazione dei parametri ($O$) per minimizzare la funzione di perdita.  Nel caso di Word2Vec, $O$ include i vettori $w_i$ (parole) e $b_i$ (contesto) per ogni parola.  ![[]]


## II. Word2Vec: Algoritmi e Implementazione

**1. Famiglia di Algoritmi:**

Word2Vec utilizza due algoritmi principali: Skip-gram e Continuous Bag-of-Words (CBOW). Entrambi utilizzano due vettori per parola ($w_i$, $b_i$) per semplificare l'ottimizzazione (sebbene sia possibile utilizzare un solo vettore).

**2. Skip-gram (SG):**

* Predizione delle parole di contesto (entro una *window size* = n) data una parola target.
* Rappresenta un'associazione uno-a-molti (una parola centrale, M parole di contesto).
* Spesso si usa la media dei vettori di contesto.

**3. Continuous Bag-of-Words (CBOW):**

* Predizione della parola target date le parole di contesto.
* Rappresenta un'associazione molti-a-uno (M parole di contesto, una parola centrale).

**4. Implementazione (generale):**

* **Skip-gram:** Utilizza un input one-hot (parola target), uno strato nascosto (dimensione D), una matrice di embedding (input-hidden) e una matrice di contesto (hidden-output). Calcola la probabilità per ogni parola di contesto.
* **CBOW:** Utilizza un input one-hot (per ogni parola di contesto), uno strato nascosto e uno strato di output. Calcola la media dei vettori di contesto, con la dimensione dello strato nascosto pari a N o D. Codifica le parole di contesto in uno spazio N o D dimensionale.


## III. Word2Vec: CBOW e Skip-gram - Dettagli

### I. CBOW (Continuous Bag-of-Words)

**A. Processo di apprendimento:** Determina i pesi tra input (parole di contesto) e output (parola centrale) tramite uno strato nascosto.

1. One-hot encoding per le parole.
2. Media dei vettori delle parole di contesto.
3. Decodifica del vettore medio con la matrice dei pesi tra hidden layer e output layer (collegato al softmax).

**B. Rappresentazione:**

1. Dimensione della rappresentazione: dimensione del vocabolario.
2. Matrice dei pesi **w** (input-hidden layer): dimensione $d \cdot n$ (d = dimensione vocabolario, n = dimensionalità hidden layer).
3. Codifica delle parole di contesto: moltiplicazione del vettore one-hot per **w**.

**C. Codifica del contesto:**

1. Media delle codifiche delle parole di contesto ($\hat{v}$).
2. Decodifica di $\hat{v}$ con la matrice dei pesi (hidden-output layer) e input al softmax.

**D. Estrazione di Embeddings:**

1. Da matrice **W**: rappresentazione delle parole di contesto come parole target in Skip-gram.
2. Da matrice **W'**: rappresentazione delle parole di contesto come parole di contesto in CBOW.


### II. Skip-gram

**A. Processo di apprendimento:** Predizione delle parole di contesto data una parola centrale.

**B. Estrazione di Embeddings:**

1. Input: singola parola.
2. Embedding estratto da matrice **W**: rappresentazione della parola target.


### III. Estrazione di Embeddings (generale)

**A.** Rappresentazione di una parola come funzione del suo ruolo nella predizione (estraendo embeddings dalla matrice di destra).


### IV. Softmax e Funzione di Costo in Word2Vec

**A.** Probabilità di una parola data un'altra tramite softmax.

**B.** Calcolo computazionalmente costoso con vocabolari grandi.


### V. Skip-gram con Negative Sampling

**A. Problema:** Normalizzazione computazionalmente costosa nella softmax.

**B.** (Segue...)

---

# Word Embedding: Tecniche di Addestramento e Modelli

## Negative Sampling

**Soluzione:** Negative Sampling.

**Funzione di costo modificata:**

$$J_{t}(\theta)=\log\sigma(u_{o}^Tv_{c})+\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$$

1. $\log\sigma(u_{o}^Tv_{c})$: logaritmo della sigmoide del prodotto scalare tra il vettore della parola centrale ($v_c$) e il vettore della parola output/contesto osservata ($u_o$).

2. $\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$: approssimazione del denominatore della softmax.  Questa parte somma le log-probabilità di *k* parole negative campionate, contribuendo a spingere le parole negative lontano dalla parola centrale.


**Obiettivo:** Apprendimento discriminativo per avvicinare le parole di contesto alla parola centrale e allontanare le parole negative (considerate rumore).  Ciò implica:

* Massimizzare la similarità tra parola centrale e parole di contesto.
* Massimizzare la distanza tra parola centrale e parole negative.

**Campionamento delle parole negative:**  Avviene da una distribuzione unigramma elevata a potenza α (≈ 0.75). Questo riduce l'enfasi sulle parole frequenti.

* **Distribuzione unigramma:** frequenza di occorrenza normalizzata.
* **α < 1:** riduce il peso delle parole molto frequenti, dando più importanza alle parole meno frequenti.


## Softmax Gerarchica

**Obiettivo:** Migliorare l'efficienza dell'addestramento rispetto alla softmax standard, particolarmente utile per vocabolari di grandi dimensioni.

**Meccanismo:** Utilizza un albero di Huffman per organizzare le parole del vocabolario.

* **Foglie:** parole del vocabolario.
* **Parole frequenti:** più vicine alla radice dell'albero.

Il calcolo della probabilità di una parola implica il percorso dalla radice alla foglia corrispondente all'interno dell'albero. I pesi lungo questo percorso vengono aggiornati durante l'addestramento.

**Vantaggi:**

* Maggiore efficienza computazionale rispetto alla softmax standard.
* Struttura gerarchica che cattura implicitamente relazioni semantiche tra le parole.


## Word2vec: Skip-gram vs. CBOW

| Caratteristica          | Skip-gram                     | CBOW                           |
|--------------------------|---------------------------------|---------------------------------|
| **Obiettivo**            | Predire contesto da parola centrale | Predire parola centrale da contesto |
| **Parole frequenti**     | Meno preciso                    | Più preciso                     |
| **Parole poco frequenti** | Più preciso                     | Meno preciso                    |
| **Finestra di contesto** | Ampia                           | Piccola                         |
| **Velocità**             | Lenta                           | Veloce                          |
| **Task**                 | Similarità, analogia            | Classificazione, task document-oriented |


**Skip-gram:**

* **Obiettivo:** predire le parole di contesto data la parola centrale.
* **Vantaggi:** migliore per parole rare e finestre di contesto ampie.
* **Svantaggi:** più lento in addestramento, meno efficiente per task *document-oriented*.


**CBOW:**

* **Obiettivo:** predire la parola centrale date le parole di contesto.
* **Vantaggi:** migliore per parole frequenti, più veloce in addestramento, adatto a task *document-oriented*.
* **Svantaggi:** meno preciso per parole rare.


**Skip-gram preferibile per:** vocabolari ampi, corpus specialistici, parole rare, finestre di contesto ampie e task *word-oriented*.


## Word Embeddings: CBOW, GloVe e considerazioni generali

### CBOW (Continuous Bag-of-Words)

**A. Vantaggi:**

1. Addestramento più veloce rispetto a Skip-gram.
2. Adatto a vocabolari moderati e corpus generici.
3. Ideale per task *document-oriented*.

**B. Tecniche di Addestramento:**

1. Hierarchical Softmax: Ottimale per parole rare.
2. Negative Sampling: Ottimale per parole frequenti e vettori a bassa dimensionalità.

**C. Ottimizzazione:**

1. **Sottocampionamento:** Migliora accuratezza e velocità con dataset grandi (1e-3 a 1e-5).  Riduce l'influenza delle parole molto frequenti.
2. **Dimensionalità Vettori:** Generalmente, maggiore è meglio (ma non sempre; dipende dal dataset e dal task).
3. **Dimensione Contesto:** ~5 (numero di parole di contesto considerate a sinistra e a destra della parola target).

**D. Aggiornamento Gradiente con Negative Sampling:**

1. Aggiornamento iterativo per ogni finestra (`window`).
2. **Sparsità del gradiente:** al massimo $2m + 1$ parole (più $2km$ parole per il negative sampling), dove *m* è la dimensione della finestra di contesto e *k* è il numero di parole negative campionate.  Il gradiente ha la forma:

   $\nabla_{\theta} J_t(\theta) = \begin{bmatrix} 0 \\ \vdots \\ \nabla_{\theta_{target\_word}} \\ \vdots \\ \nabla_{\theta_{context\_word}} \\ \vdots \\ 0 \end{bmatrix} \in \mathbb{R}^{2dV}$

   dove *d* è la dimensione del vettore word embedding e *V* è la dimensione del vocabolario.

3. Aggiornamento selettivo dei vettori (solo quelli nella finestra).
4. **Soluzioni per ottimizzare l'aggiornamento:** operazioni sparse, tecniche di hashing.
5. Evitare aggiornamenti ingombranti (importante per calcolo distribuito).

**E. Matrice di Co-occorrenza:**  (La sezione è incompleta nel testo originale e non può essere completata senza ulteriori informazioni).

---

# GloVe: Word Embeddings

## I. Finestra di Contesto e Documento Completo

La rappresentazione di parole tramite GloVe considera due livelli di contesto:

* **Finestra (window):** Cattura informazioni sintattiche e semantiche all'interno di una finestra di parole circostanti (*spazio delle parole*).
* **Documento completo:** Permette di ottenere temi generali e relazioni a lungo raggio (*spazio dei documenti*).


## II. Limitazioni delle Rappresentazioni Basate su Co-occorrenze

Le rappresentazioni basate su co-occorrenze, come quelle utilizzate da tecniche come LSA (Latent Semantic Analysis), presentano alcune limitazioni:

* **A. Rappresentazione sparsa:** La matrice di co-occorrenza è tipicamente molto sparsa, rendendo difficile l'utilizzo efficiente di tecniche come LSA.
* **B. Relazioni lineari:** LSA cattura principalmente relazioni di sinonimia, trascurando la polisemia (molteplicità di significati).  È inefficiente per task *word-oriented* complessi.


## III. GloVe (Global Vectors): Un Approccio Ibrido

GloVe (Global Vectors for Word Representation) è un modello ibrido che combina approcci neurali e statistici per generare word embeddings.

* **A. Approccio ibrido (neurale e statistico):**  Combina la potenza dei modelli neurali con l'efficienza delle tecniche statistiche basate su co-occorrenze.
* **B. Contesto delle parole:** Considera il contesto delle parole per generare rappresentazioni più ricche e informative.
* **C. Funzionamento:** Analizza le co-occorrenze di parole per costruire una matrice di co-occorrenza.  Questa matrice viene poi utilizzata per addestrare un modello che impara le rappresentazioni vettoriali delle parole.
* **D. Vantaggi:**
    * 1. **Gestione della polisemia:**  Migliore capacità di rappresentare i diversi significati di una parola a seconda del contesto.
    * 2. **Migliore rappresentazione di parole poco frequenti:**  Rappresentazioni più accurate anche per parole che appaiono raramente nel corpus.


## IV. Efficienza Computazionale e Natura del Modello GloVe

* GloVe è un modello neurale efficiente per la creazione di word embeddings.
* Utilizza le co-occorrenze di parole per apprendere rappresentazioni dense.
* Non genera un singolo embedding globale per parola; l'embedding dipende dal corpus di addestramento.
* Ogni parola è rappresentata da un unico embedding all'interno di un dato corpus.


## V. Rappresentazioni Globali e Context-Free

* **Rappresentazioni Globali:** Ogni parola ha un solo vettore, indipendentemente dal contesto. Questa rappresentazione è statica.
* **Rappresentazioni Context-Free:** La rappresentazione di una parola ignora il contesto d'uso.
* **Limiti:**
    * **Polisemia:** Non cattura i diversi significati di una parola a seconda del contesto.
    * **Parole poco frequenti:** Rappresentazioni meno accurate per mancanza di dati di addestramento.


## VI. Funzione Obiettivo di GloVe

GloVe minimizza la differenza tra il prodotto scalare degli embedding di due parole e il logaritmo della loro probabilità di co-occorrenza.

* **Funzione obiettivo:**

$$ f(x)= \begin{cases} \left( \frac{x}{x_{max}} \right)^\alpha, & if \ x<x_{max} \\ 1, & otherwise \end{cases} $$

$$ \text{Loss } J=\sum_{i,j=1}^Vf(X_{ij})(w_{i}T \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij})^2 $$

Dove:

* $w_i$, $w_j$: embedding delle parole $i$ e $j$.
* $b_i$, $b_j$: bias per le parole $i$ e $j$.
* $X_{ij}$: numero di co-occorrenze delle parole $i$ e $j$ in una finestra di contesto.
* $f(x)$ include una distribuzione unigram con esponente $\frac{3}{4}$ per smorzare l'effetto delle parole frequenti.
* $\log X_{ij}$ approssima $P(i|j)= \frac{X_{ij}}{X_{i}}$.
* L'errore è la differenza tra co-occorrenza reale e attesa.
* L'obiettivo è catturare le proprietà dei rapporti di co-occorrenza.


## VII. Interpretazione dei Rapporti di Probabilità

Considerando tre parole (i, j, k), dove k è una parola di confronto:

* **Rapporto $P(k|i)/P(k|j) > 1$**: k è più correlata a i che a j.
* **Rapporto $P(k|i)/P(k|j) < 1$**: k è più correlata a j che a i.

*Esempio: Tabella con probabilità e rapporti per parole come "ice", "steam" e diverse parole k ("solid", "gas", "water", "fashion").*  ![[]]


## VIII. Relazioni Geometriche tra Parole

* **Rappresentazione Vettoriale:** L'obiettivo è rappresentare le parole in uno spazio vettoriale per visualizzare le relazioni geometricamente.  Ad esempio, l'analogia "man" : "woman" :: "king" : "queen" può essere rappresentata geometricamente. Gli embeddings permettono di definire queste relazioni come combinazioni lineari.


## IX. Funzione di Confronto *F*

* **Scopo:** Catturare le relazioni tra parole $w_i$, $w_j$, $w_k$.
* **Proprietà:**
    * **Condizione 1 (Combinazione Lineare):** $F((w_{i}-w_{j})^Tw_{k})=\frac{P(k|i)}{P(k|j)}$
    * **Condizione 2 (Simmetria):** $F((w_{i}-w_{j})^Tw_{k})=\frac{F(w_{i}^Tw_{k})}{F(w_{j}^Tw_{k})}$
* **Definizione di *F*:** $F(w_{i}^Tw_{k})=e^{w_{i}^Tw_{k}}=P(k|i) = \frac{x_{ik}}{x_{i}}$ (dove $x_{ik}$ è la co-occorrenza di $w_i$ e $w_k$, e $x_i$ è il conteggio di $w_i$)


---

# Derivazione e Interpretazione del Modello

## Equazione Derivata

L'equazione derivata del modello è:

$w_{i}^Tw_{k}=\log x_{ik}-\log x_{i}$

dove:

* $w_i$ e $w_k$ sono i vettori di parole.
* $x_{ik}$ rappresenta le co-occorrenze osservate tra le parole *i* e *k*.
* $x_i$ rappresenta la frequenza della parola *i*.


## Semplificazione con Bias

Introducendo i bias $b_i$ e $b_j$, inizializzati a 0, l'equazione viene semplificata a:

$w_{i}^Tw_{k}+b_{i}+b_{j}=\log X_{jk}$

dove $X_{jk}$ rappresenta le co-occorrenze osservate tra le parole *j* e *k*.  L'aggiunta dei bias migliora la capacità del modello di gestire le asimmetrie nelle co-occorrenze.


## Interpretazione

L'equazione finale, $w_{i}^Tw_{k}+b_{i}+b_{j}=\log X_{jk}$, confronta le co-occorrenze attese delle parole (il termine a sinistra dell'uguale) con le co-occorrenze effettive (il termine a destra). Il prodotto scalare $w_{i}^Tw_{k}$ rappresenta la similarità tra i vettori delle parole *i* e *k*, mentre i bias $b_i$ e $b_j$ correggono per le asimmetrie nelle co-occorrenze.  La simmetria è gestita tramite l'inclusione del bias $b_j$.

---

Per favore, forniscimi il testo da formattare.  Ho bisogno del testo che desideri che io organizzi e formati secondo le tue istruzioni per poterti aiutare.

---
