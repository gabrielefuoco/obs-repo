
## Schema Riassuntivo: Librerie NLP e Modelli di Topic Modeling

**I. Librerie NLP pre-LLM**

* **A. NLTK:**
    * Strumenti per analisi del testo: tokenizzazione, stemming, lemmatizzazione, analisi morfologica, sintattica e classificazione.
* **B. SpaCy:**
    * Velocità ed efficienza.
    * Funzionalità avanzate: NER, estrazione di entità, analisi del sentimento, classificazione del testo.
    * Utilizzo commerciale.
* **C. Gensim:**
    * Analisi di argomenti e modellazione di argomenti.
    * Implementazioni di LDA (*Latent Dirichlet Allocation*).


**II. Latent Dirichlet Allocation (LDA)**

* **A. Descrizione:** Modello di topic modeling di riferimento.  Estensioni per considerare autori o categorie speciali.
* **B. Proprietà:** Coniugate prior tra le distribuzioni (Dirichlet e polinomiale).
* **C. Parametri:**
    * $\alpha$: dispersione della distribuzione di probabilità sui topic.
    * $\eta$: dispersione della distribuzione di probabilità sulle parole.
* **D. Pseudocodice:**
    * For each topic, generate a Dirichlet distribution over terms:  $\beta_k \sim \text{Dir}_M(\eta), k \in \{1, \ldots, K\}$
    * For each document $d_i, i \in \{1, \ldots, N\}$:
        * Generate a Dirichlet distribution over topics: $\theta_i \sim \text{Dir}_K(\alpha)$
        * For each word position $j$ in document $d_i$:
            * Choose a topic $z_{ij} \sim \text{Multi}(\theta_i)$
            * Choose word $w_{ij} \sim \text{Multi}(\beta_{z_{ij}})$


**III. Modellazione di Documenti Segmentati per Argomento (Ponti, Tagarelli, Karypis, 2011)**

* **A. Approccio:** Modellazione più granulare basata sulla segmentazione del testo.
* **B. Idea chiave:** Traduzione della relazione tra argomenti, documenti e segmenti in un modello probabilistico congiunto.  Considera $P(\text{documento}|\text{argomento}, \text{segmento})$ e viceversa.
* **C. Motivazione:**  Gestione di documenti lunghi e strutture semantiche complesse. Assegnare un solo argomento a un documento lungo è riduttivo.
* **D. Approcci Alternativi:**
    * Soft clustering: un documento può appartenere a più cluster con una certa probabilità.  Tuttavia, questo approccio non considera la struttura logica dei documenti.


---

**I. Topic Drift e Segmentazione del Testo**

* **A. Topic Drift:** Deriva tematica all'interno di un documento, problema da mitigare.
* **B. Segmentazione:** Soluzione per affrontare il topic drift.
    * 1. Suddivisione del documento in unità più piccole (segmenti) per catturare la struttura semantica fine.
    * 2. Associazione di diversi argomenti a diversi segmenti.
    * 3. Utilizzo dell'overclustering (k >> numero reale di argomenti) per identificare micro-topic, precedentemente applicato con modelli vector space e bag of words.

**II. Integrazione della Segmentazione nei Topic Model**

* **A. Obiettivo:** Creazione di un topic model a grana più fine, considerando la variabile "segmento".
* **B. Modello Probabilistico:** Rappresentato da una rete bayesiana con variabili:
    * 1. **D:** Documenti
    * 2. **Z:** Topic
    * 3. **S:** Segmenti
    * 4. **V:** Vocabolario
* **C. Relazioni Probabilistiche:**
    * 1. $\text{Pr}(z|d)$: Probabilità di scegliere un topic *z* dato un documento *d*.
    * 2. $\text{Pr}(s|z)$: Probabilità di selezionare un segmento *s* dato un topic *z*.
    * 3. $\text{Pr}(w|z)$: Probabilità di scegliere una parola *w* dato un topic *z*.
    * 4. $\text{Pr}(w|s)$: Probabilità di scegliere una parola *w* dato un segmento *s*.
* **D. Metodologia:** Estensione di PLSA, evitando la complessità di RDA, migliorando la modellazione dei topic con maggiore dettaglio.  Non modella esplicitamente la relazione tra argomenti, documenti e segmenti.

**III. Metodi di Segmentazione del Testo**

* **A. Soluzione Naive:** Segmentazione basata sui paragrafi (semplice ma imprecisa).
* **B. Soluzioni Intelligenti (non supervisionate):**
    * 1. **Text Tiling:** Metodo basato su modello vector space e tf-idf.
        * a. Rappresentazione vettoriale del testo.
        * b. Calcolo della similarità coseno tra segmenti consecutivi.
        * c. Creazione di una curva di similarità.
        * d. Identificazione dei minimi locali (cambiamenti di topic).
        * e. Segmentazione del testo nei minimi locali.  Massimi locali indicano continuità di topic.


---

**Modello di Argomenti Latenti (LDA) per il Text Tiling**

I. **Modello LDA per Text Tiling:**

   * Formula di probabilità:  $\operatorname{Pr}(d , s, w) = \operatorname{Pr}(d) \sum_{z \in Z} \operatorname{Pr}(z \mid d) \operatorname{Pr}(s \mid z) \operatorname{Pr}(w \mid z, s)$
   * Processo generativo:
      * Selezione del documento *d*.
      * Per ogni segmento *s*:
         * Selezione di un argomento *z* per *d*.
         * Associazione della probabilità argomento-segmento a *s* per *z*.
         * Per ogni parola *w* in *s*:
            * Selezione di *w* dall'argomento e segmento correnti.

II. **Algoritmo di Inferenza EM per LDA:**

   * **E-step:** Calcolo della probabilità a posteriori dell'argomento:  $\Pr(z|d, s, w) = \frac{\Pr(z|d)\Pr(s|z)\Pr(w|z, s)}{\sum_{z \in Z} \Pr(z|d)\Pr(s|z)\Pr(w|z, s)}$
   * **M-step:** Aggiornamento delle probabilità:
      * Funzione di verosimiglianza: $\mathbf{E}[l] = \sum_{d \in D} \sum_{s \in S} \sum_{w \in V} n(d, s, w) \times \sum_{z \in Z} \Pr(z|d, s, w) \log(\Pr(d, s, w))$
      * Formule di aggiornamento:
         * $\Pr(z|d) \propto \sum_{s \in S} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w)$
         * $\Pr(s|z) \propto \sum_{d \in D} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w)$
         * $\Pr(w|z, s) \propto \sum_{d \in D} n(d, s, w) \Pr(z|d, s, w)$
   * Iterazione fino a convergenza.

III. **Applicazione del Text Tiling con LDA:**

   * Segmentazione del documento in base ai cambiamenti di argomento.
   * Identificazione dei segmenti tramite minimi locali nella curva delle similarità.

IV. **Valutazione dei Modelli di Argomenti:**

   * Metriche tradizionali: Coerenza e Perplessità.
   * Coerenza: misura la co-occorrenza tra parole e argomenti (punteggio maggiore = coerenza migliore).  Gli incrementi diminuiscono all'aumentare del numero di argomenti.


---

**I. Metodi di Valutazione della Coerenza dei Topic**

* **A. Punteggio di coerenza UMass:**
    * Formula:  $C_{\text{UMass}}(w_{i},w_{j})=\log \frac{D(w_{i},w_{j})+1}{D(w_{i})}$
    * $D(w_{i}, w_{j})$: co-occorrenze di $w_{i}$ e $w_{j}$ nei documenti.
    * $D(w_{i})$: occorrenze di $w_{i}$.
* **B. Punteggio di coerenza UCI:**
    * Formula: $C_{\text{UCI}}(w_{i},w_{j})=\log \frac{P(w_{i},w_{j})+1}{P(w_{i})\cdot P(w_{j})}$
    * $P(w_{i}, w_{j})$: probabilità di co-occorrenza in una finestra scorrevole.
    * $P(w_{i})$: probabilità di occorrenza in una finestra scorrevole.
    * Basato su PMI (Probabilità di Informazione Mutua) e finestre scorrevoli.
    * Approccio robusto ma parametrico (stride e overlap).
* **C. Coerenza basata sulla similarità intra/inter-argomento:**
    * Similarità intra-argomento: media della similarità tra coppie di parole principali dello stesso argomento.
    * Similarità inter-argomento: media della similarità tra parole principali di argomenti diversi.
    * Utilizza il coseno come misura di similarità.
    * Richiede embedding delle parole.

**II. Oltre la Coerenza: Separazione dei Topic**

* Valutazione della separazione tra topic oltre alla coerenza interna (compattezza).
* Topic con distribuzioni di probabilità simili sono meno separati.

**III. Misure di Incertezza e Distanza tra Distribuzioni**

* **A. Entropia e Cross-entropia:**
    * Entropia: misura l'incertezza di una distribuzione di probabilità di un topic.
    * Cross-entropia: misura l'incertezza di una distribuzione rispetto a un'altra.
* **B. Divergenza di Kullback-Leibler (KL):**
    * Formula: $D_{KL}(P\|Q) = -\sum_{x\in X}P(x)\log\left( \frac{Q(x)}{P(x)} \right)$
    * Non negativa, asimmetrica, non metrica.
* **C. Divergenza di Jensen-Shannon (JS):**
    * Formula: $D_{JS} = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$, dove $M = \frac{1}{2}(P + Q)$.
* **D. Cross-entropia:** misura la differenza tra distribuzione prevista ed effettiva.

**IV. Perplessità come Misura di Performance dei Modelli Linguistici**

* Misura l'incertezza di un modello linguistico nel generare un nuovo token.
* Minore perplessità indica un modello migliore.
* Dipende dalla tokenizzazione utilizzata.
* Funzione esponenziale dell'entropia.

---

**Schema Riassuntivo: Entropia, Perplessità e Modelli Linguistici**

I. **Concetti Fondamentali:**

   * A. **Variabile x:** Sorgente informativa (es. testo); xᵢ rappresenta i token (parole).
   * B. **Entropia di x (H[X]):** Misura il tasso di informazione; massima con distribuzione uniforme dei token.
   * C. **Token Frequenti:** Minimizzano la lunghezza di codifica (-log P(x)).
   * D. **Componente Informativa:** -log P(x), dove P(x) è la probabilità del token x.
   * E. **Perplessità (PP[X]):**  $PP[X] := 2^{H[X]}$, misura la difficoltà di predire la sorgente.
   * F. **Teorema di Codifica Senza Rumore di Shannon:** Limite inferiore per la lunghezza attesa del codice: -log(p(x)) per ogni token.


II. **Perplessità di un Processo Stocastico (χ): Sequenze Non i.i.d.**

   * A. **Ipotesi di Stazionarietà:** La probabilità di una parola è indipendente dalla posizione nel testo (ipotesi semplificativa, non realistica).  $\lim_{ n \to \infty } \frac{1}{n}H[X_{1},\dots,X_{n}] = \lim_{ n \to \infty } \frac{1}{n}H[X_{n}|X_{1},\dots,X_{n-1}]$
   * B. **Ipotesi di Ergodicità:**  La media temporale su una lunga sequenza approssima il valore atteso.  $\frac{1}{n}\sum_{i=1}^n X_{i}\to_{n \to \infty} E_{p}[X_{1}]$ con probabilità 1.  Questo, con il Teorema di Shannon-McMillan-Breiman, implica: $-\frac{1}{n}\log p(X_{1},\dots,X_{n})\to_{n\to \infty }H[X]$ con probabilità 1.
   * C. **Applicazione all'Entropia:**  Usando un modello linguistico q(x₁, x₂, …) come approssimazione di p(x₁, x₂, …), la cross-entropia fornisce un limite superiore al tasso di entropia: $CE[P,Q]:=\lim_{ n \to \infty }-E_{p}\log q(X_{n}|X_{<n})=\lim_{ n \to \infty } -\frac{1}{n}E_{p}\log q(X_{1},\dots,X_{n})$


III. **Perplessità di un Modello Linguistico (Q):**

   * A. **Definizione:**  La perplexity di Q rispetto al linguaggio naturale P è $2^{CE[P,Q]}$.
   * B. **Calcolo Pratico:**  $2$ elevato alla media, per ogni parola in ogni documento, della distribuzione di probabilità delle parole (w,d), normalizzata per il numero di parole nel documento (N,d), sommato su tutti i documenti.



---

**Perplessità (PP) in Modelli di Topic Modeling**

I. **Definizione Formale di Perplessità:**

   *  $$PP[P,Q]:=2^{CE[P,Q]}$$
   *  $$-\frac{1}{n}\log q(X_{1},\dots,X_{n})\to_{n\to \infty}CE[P,Q]$$  (dove CE è la Cross-Entropy)

II. **Perplessità su Campione di Holdout:**

   * Formula: $$ -\frac{\sum_{d=1}^M\log p(w_{d})}{\sum_{d=1}^MN_{d}} $$
   *  M: numero di documenti nel campione di test.
   *  $w_{d}$: parole nel documento d.
   *  $N_{d}$: numero di parole nel documento d.

III. **Perplessità in LDA (Latent Dirichlet Allocation):**

   * Formula: $$\log p(w|\alpha,\beta)=E[\log p(\theta,z,w|\alpha,\beta)]-E[\log q(\theta,z)]$$
   *  Valutazione dei parametri di dispersione α e β.
   *  Utilizzo della *variational inference* (più accurata del Gibbs sampling).

IV. **Perplessità in SGM (Streaming Generalized Model):**

   * Formula: $$Pr(d,S_{d},V)=Pr(d)\prod_{S\in S_{d}}\sum_{z\in Z}Pr(z|d)Pr(s|z)\prod_{w\in V}Pr(w|z,s)$$
   *  Calcolo di probabilità simili a LDA, valutate per ogni documento durante la fase di test.
   *  Adattamento del modello LDA a stream di dati.


---
