
##### Librerie NLP pre-LLM

* **NLTK:**
	* Strumenti per analisi del testo: tokenizzazione, stemming, lemmatizzazione, analisi morfologica, sintattica e classificazione.
* **SpaCy:**
	* Velocità ed efficienza.
	* Funzionalità avanzate: NER, estrazione di entità, analisi del sentimento, classificazione del testo.
	* Utilizzo commerciale.
* **Gensim:**
	* Analisi di argomenti e modellazione di argomenti.
	* Implementazioni di LDA (*Latent Dirichlet Allocation*).

##### Latent Dirichlet Allocation (LDA)

* **Descrizione:** Modello di topic modeling di riferimento. Estensioni per considerare autori o categorie speciali.
* **Proprietà:** Coniugate prior tra le distribuzioni (Dirichlet e polinomiale).
* **Parametri:**
	* $\alpha$: dispersione della distribuzione di probabilità sui topic.
	* $\eta$: dispersione della distribuzione di probabilità sulle parole.
* **Pseudocodice:**
	* For each topic, generate a Dirichlet distribution over terms: $\beta_k \sim \text{Dir}_M(\eta), k \in \{1, \ldots, K\}$
	* For each document $d_i, i \in \{1, \ldots, N\}$:
	* Generate a Dirichlet distribution over topics: $\theta_i \sim \text{Dir}_K(\alpha)$
	* For each word position $j$ in document $d_i$:
	* Choose a topic $z_{ij} \sim \text{Multi}(\theta_i)$
	* Choose word $w_{ij} \sim \text{Multi}(\beta_{z_{ij}})$

##### Modellazione di Documenti Segmentati per Argomento (Ponti, Tagarelli, Karypis, 2011)

* **Approccio:** Modellazione più granulare basata sulla segmentazione del testo.
* **Idea chiave:** Traduzione della relazione tra argomenti, documenti e segmenti in un modello probabilistico congiunto. Considera $P(\text{documento}|\text{argomento}, \text{segmento})$ e viceversa.
* **Motivazione:** Gestione di documenti lunghi e strutture semantiche complesse. Assegnare un solo argomento a un documento lungo è riduttivo.
* **Approcci Alternativi:**
	* Soft clustering: un documento può appartenere a più cluster con una certa probabilità. Tuttavia, questo approccio non considera la struttura logica dei documenti.

##### Topic Drift e Segmentazione del Testo

* **Topic Drift:** Deriva tematica all'interno di un documento, problema da mitigare.
* **Segmentazione:** Soluzione per affrontare il topic drift.
* Suddivisione del documento in unità più piccole (segmenti) per catturare la struttura semantica fine.
* Associazione di diversi argomenti a diversi segmenti.
* Utilizzo dell'overclustering (k >> numero reale di argomenti) per identificare micro-topic, precedentemente applicato con modelli vector space e bag of words.

##### Integrazione della Segmentazione nei Topic Model

* **Obiettivo:** Creazione di un topic model a grana più fine, considerando la variabile "segmento".
* **Modello Probabilistico:** Rappresentato da una rete bayesiana con variabili:
* **D:** Documenti
* **Z:** Topic
* **S:** Segmenti
* **V:** Vocabolario
* **Relazioni Probabilistiche:**
	* $\text{Pr}(z|d)$: Probabilità di scegliere un topic *z* dato un documento *d*.
	* $\text{Pr}(s|z)$: Probabilità di selezionare un segmento *s* dato un topic *z*.
	* $\text{Pr}(w|z)$: Probabilità di scegliere una parola *w* dato un topic *z*.
	* $\text{Pr}(w|s)$: Probabilità di scegliere una parola *w* dato un segmento *s*.
	* **Metodologia:** Estensione di PLSA, evitando la complessità di RDA, migliorando la modellazione dei topic con maggiore dettaglio. Non modella esplicitamente la relazione tra argomenti, documenti e segmenti.

##### Metodi di Segmentazione del Testo

* **Soluzione Naive:** Segmentazione basata sui paragrafi (semplice ma imprecisa).
* **Soluzioni Intelligenti (non supervisionate):**
	* **Text Tiling:** Metodo basato su modello vector space e tf-idf.
	* Rappresentazione vettoriale del testo.
	* Calcolo della similarità coseno tra segmenti consecutivi.
	* Creazione di una curva di similarità.
	* Identificazione dei minimi locali (cambiamenti di topic).
	* Segmentazione del testo nei minimi locali. Massimi locali indicano continuità di topic.

##### Modello di Argomenti Latenti (LDA) per il Text Tiling

##### Modello LDA per Text Tiling:

* Formula di probabilità: $\operatorname{Pr}(d , s, w) = \operatorname{Pr}(d) \sum_{z \in Z} \operatorname{Pr}(z \mid d) \operatorname{Pr}(s \mid z) \operatorname{Pr}(w \mid z, s)$
* Processo generativo:
* Selezione del documento *d*.
* Per ogni segmento *s*:
* Selezione di un argomento *z* per *d*.
* Associazione della probabilità argomento-segmento a *s* per *z*.
* Per ogni parola *w* in *s*:
* Selezione di *w* dall'argomento e segmento correnti.

##### Algoritmo di Inferenza EM per LDA:

* **E-step:** Calcolo della probabilità a posteriori dell'argomento: $\Pr(z|d, s, w) = \frac{\Pr(z|d)\Pr(s|z)\Pr(w|z, s)}{\sum_{z \in Z} \Pr(z|d)\Pr(s|z)\Pr(w|z, s)}$
* **M-step:** Aggiornamento delle probabilità:
* Funzione di verosimiglianza:
$$\mathbf{E}[l] = \sum_{d \in D} \sum_{s \in S} \sum_{w \in V} n(d, s, w) \times \sum_{z \in Z} \Pr(z|d, s, w) \log(\Pr(d, s, w))$$
* Formule di aggiornamento:
* $\Pr(z|d) \propto \sum_{s \in S} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w)$
* $\Pr(s|z) \propto \sum_{d \in D} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w)$
* $\Pr(w|z, s) \propto \sum_{d \in D} n(d, s, w) \Pr(z|d, s, w)$
* Iterazione fino a convergenza.

##### Applicazione del Text Tiling con LDA:

* Segmentazione del documento in base ai cambiamenti di argomento.
* Identificazione dei segmenti tramite minimi locali nella curva delle similarità.

##### Valutazione dei Modelli di Argomenti:

* Metriche tradizionali: Coerenza e Perplessità.
* Coerenza: misura la co-occorrenza tra parole e argomenti (punteggio maggiore = coerenza migliore). Gli incrementi diminuiscono all'aumentare del numero di argomenti.

##### Metodi di Valutazione della Coerenza dei Topic

* **Punteggio di coerenza UMass:**
	* Formula: $C_{\text{UMass}}(w_{i},w_{j})=\log \frac{D(w_{i},w_{j})+1}{D(w_{i})}$
	* $D(w_{i}, w_{j})$: co-occorrenze di $w_{i}$ e $w_{j}$ nei documenti.
	* $D(w_{i})$: occorrenze di $w_{i}$.
* **Punteggio di coerenza UCI:**
	* Formula: $C_{\text{UCI}}(w_{i},w_{j})=\log \frac{P(w_{i},w_{j})+1}{P(w_{i})\cdot P(w_{j})}$
	* $P(w_{i}, w_{j})$: probabilità di co-occorrenza in una finestra scorrevole.
	* $P(w_{i})$: probabilità di occorrenza in una finestra scorrevole.
	* Basato su PMI (Probabilità di Informazione Mutua) e finestre scorrevoli.
	* Approccio robusto ma parametrico (stride e overlap).
* **Coerenza basata sulla similarità intra/inter-argomento:**
	* Similarità intra-argomento: media della similarità tra coppie di parole principali dello stesso argomento.
	* Similarità inter-argomento: media della similarità tra parole principali di argomenti diversi.
	* Utilizza il coseno come misura di similarità.
	* Richiede embedding delle parole.

##### Oltre la Coerenza: Separazione dei Topic

* Valutazione della separazione tra topic oltre alla coerenza interna (compattezza).
* Topic con distribuzioni di probabilità simili sono meno separati.

##### Misure di Incertezza e Distanza tra Distribuzioni

* **Entropia e Cross-entropia:**
	* Entropia: misura l'incertezza di una distribuzione di probabilità di un topic.
	* Cross-entropia: misura l'incertezza di una distribuzione rispetto a un'altra.
* **Divergenza di Kullback-Leibler (KL):**
	* Formula: $D_{KL}(P\|Q) = -\sum_{x\in X}P(x)\log\left( \frac{Q(x)}{P(x)} \right)$
	* Non negativa, asimmetrica, non metrica.
* **Divergenza di Jensen-Shannon (JS):**
	* Formula: $D_{JS} = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$, dove $M = \frac{1}{2}(P + Q)$.
	* **Cross-entropia:** misura la differenza tra distribuzione prevista ed effettiva.

##### Perplessità come Misura di Performance dei Modelli Linguistici

* Misura l'incertezza di un modello linguistico nel generare un nuovo token.
* Minore perplessità indica un modello migliore.
* Dipende dalla tokenizzazione utilizzata.
* Funzione esponenziale dell'entropia.

### Entropia, Perplessità e Modelli Linguistici

##### Concetti Fondamentali:

* **Variabile x:** Sorgente informativa (es. testo); xᵢ rappresenta i token (parole).
* **Entropia di x (H[X]):** Misura il tasso di informazione; massima con distribuzione uniforme dei token.
* **Token Frequenti:** Minimizzano la lunghezza di codifica (-log P(x)).
* **Componente Informativa:** -log P(x), dove P(x) è la probabilità del token x.
* **Perplessità (PP[X]):** $PP[X] := 2^{H[X]}$, misura la difficoltà di predire la sorgente.
* **Teorema di Codifica Senza Rumore di Shannon:** Limite inferiore per la lunghezza attesa del codice: -log(p(x)) per ogni token.

##### Perplessità di un Processo Stocastico (χ): Sequenze Non i.i.d.

* **Ipotesi di Stazionarietà:** La probabilità di una parola è indipendente dalla posizione nel testo (ipotesi semplificativa, non realistica). $\lim_{ n \to \infty } \frac{1}{n}H[X_{1},\dots,X_{n}] = \lim_{ n \to \infty } \frac{1}{n}H[X_{n}|X_{1},\dots,X_{n-1}]$
* **Ipotesi di Ergodicità:** La media temporale su una lunga sequenza approssima il valore atteso. $\frac{1}{n}\sum_{i=1}^n X_{i}\to_{n \to \infty} E_{p}[X_{1}]$ con probabilità 1. Questo, con il Teorema di Shannon-McMillan-Breiman, implica: $-\frac{1}{n}\log p(X_{1},\dots,X_{n})\to_{n\to \infty }H[X]$ con probabilità 1.
* **Applicazione all'Entropia:** Usando un modello linguistico q(x₁, x₂, …) come approssimazione di p(x₁, x₂, …), la cross-entropia fornisce un limite superiore al tasso di entropia: $CE[P,Q]:=\lim_{ n \to \infty }-E_{p}\log q(X_{n}|X_{<n})=\lim_{ n \to \infty } -\frac{1}{n}E_{p}\log q(X_{1},\dots,X_{n})$

##### Perplessità di un Modello Linguistico (Q):

* **Definizione:** La perplexity di Q rispetto al linguaggio naturale P è $2^{CE[P,Q]}$.
* **Calcolo Pratico:** $2$ elevato alla media, per ogni parola in ogni documento, della distribuzione di probabilità delle parole (w,d), normalizzata per il numero di parole nel documento (N,d), sommato su tutti i documenti.

### Perplessità (PP) in Modelli di Topic Modeling

##### Definizione Formale di Perplessità:

$$PP[P,Q]:=2^{CE[P,Q]}$$
$$-\frac{1}{n}\log q(X_{1},\dots,X_{n})\to_{n\to \infty}CE[P,Q]$$ (dove CE è la Cross-Entropy)

##### Perplessità su Campione di Holdout:

* Formula: $$ -\frac{\sum_{d=1}^M\log p(w_{d})}{\sum_{d=1}^MN_{d}} $$
* M: numero di documenti nel campione di test.
* $w_{d}$: parole nel documento d.
* $N_{d}$: numero di parole nel documento d.

##### Perplessità in LDA (Latent Dirichlet Allocation):

* Formula: $$\log p(w|\alpha,\beta)=E[\log p(\theta,z,w|\alpha,\beta)]-E[\log q(\theta,z)]$$
* Valutazione dei parametri di dispersione α e β.
* Utilizzo della *variational inference* (più accurata del Gibbs sampling).

##### Perplessità in SGM (Streaming Generalized Model):

* Formula: $$Pr(d,S_{d},V)=Pr(d)\prod_{S\in S_{d}}\sum_{z\in Z}Pr(z|d)Pr(s|z)\prod_{w\in V}Pr(w|z,s)$$
* Calcolo di probabilità simili a LDA, valutate per ogni documento durante la fase di test.
* Adattamento del modello LDA a stream di dati.

