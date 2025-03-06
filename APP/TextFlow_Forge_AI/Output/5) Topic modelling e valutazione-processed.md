
## Schema Riassuntivo: Librerie NLP e Topic Modeling

**1. Librerie di Riferimento Pre-LLM per NLP:**

*   **1.1 NLTK (Natural Language Toolkit):**
    *   Libreria Python completa per NLP.
    *   Strumenti per: tokenizzazione, stemming, lemmatizzazione, analisi morfologica, analisi sintattica, classificazione del testo.
*   **1.2 SpaCy:**
    *   Libreria Python per NLP focalizzata su velocità ed efficienza.
    *   Funzionalità avanzate: NER, estrazione di entità, analisi del sentimento, classificazione del testo.
    *   Utilizzata commercialmente.
*   **1.3 Gensim:**
    *   Libreria Python per topic modeling e analisi di argomenti.
    *   Implementa modelli stocastici come LDA.
    *   Utilizzata per scoprire argomenti latenti in grandi dataset di testo.

**2. Latent Dirichlet Allocation (LDA):**

*   **2.1 Definizione:** Modello di topic modeling di riferimento.
*   **2.2 Estensioni:** Tecniche basate su LDA per includere il ruolo di autori o categorie.
    *   Modellazione supervisionata/semi-supervisionata del profilo autore (identificazione autori fake).
    *   Complessità computazionale aumentata.
*   **2.3 Proprietà Chiave:** Coniugate prior tra le distribuzioni (Dirichlet e polinomiale).
*   **2.4 Parametri:**
    *   $\alpha$: Parametro di dispersione per la distribuzione di probabilità sui topic.
    *   $\eta$: Parametro di dispersione per la distribuzione di probabilità sulle parole.
    *   Importanza della differenza in ordine di grandezza tra $\alpha$ e $\eta$.
*   **2.5 Pseudo-codice:**
    *   $\text{For each topic, generate a Dirichlet distribution over terms:}$
        $$\beta_k \sim \text{Dir}_M(\eta), k \in \{1, \ldots, K\}$$
    *   $\text{For each document } d_i, i \in \{1, \ldots, N\}$
        *   $\text{Generate a Dirichlet distribution over topics: } \theta_i \sim \text{Dir}_K(\alpha)$
        *   $\text{For each word position } j \text{ in document } d_i:$
            *   $\text{Choose a topic } z_{ij} \text{ from the distribution in step a., i.e., } z_{ij} \sim \text{Multi}(\theta_i)$
            *   $\text{Choose word } w_{ij} \text{ from topic } z_{ij} \text{, i.e., } w_{ij} \sim \text{Multi}(\beta_{z_{ij}})$

**3. Modellazione di Documenti Segmentati per Argomento:**

*   **3.1 Modello Generativo Basato su Segmenti [Ponti, Tagarelli, Karypis, 2011]:**
    *   Approccio granulare alla modellazione argomento-documento.
    *   Basato sulla segmentazione del testo.
*   **3.2 Traduzione in Modello Probabilistico Congiunto:**
    *   Relazione tra argomenti, documenti e segmenti.
    *   Considera la probabilità di un documento dato un argomento e un segmento.
*   **3.3 Motivazione:**
    *   Assegnare un solo argomento a documenti lunghi è riduttivo.
    *   Necessità di rappresentazione più fine per la struttura semantica complessa.
*   **3.4 Approcci Alternativi:**
    *   Soft clustering: Documenti appartengono a più cluster con probabilità.
    *   Struttura logica dei documenti: Segnali informativi in introduzione e conclusione.

---

**Schema Riassuntivo: Segmentazione del Testo e Topic Modeling**

**1. Introduzione: Topic Drift e Segmentazione**
    *   **1.1 Topic Drift:** Deriva tematica dovuta alla difficoltà di catturare la struttura semantica fine dei documenti.
    *   **1.2 Segmentazione:** Suddivisione del documento in unità più piccole (segmenti) per associare diversi argomenti a diversi segmenti.
    *   **1.3 Overclustering:** Stima di un numero di cluster (k) molto più alto del numero effettivo di argomenti per catturare la struttura dei micro-topic.

**2. Integrazione della Segmentazione nei Topic Model**
    *   **2.1 Obiettivo:** Creare un modello a grana più fine che tenga conto della variabile dei segmenti.
    *   **2.2 Modello Probabilistico (Rete Bayesiana):**
        *   **Variabili:**
            *   D: Documenti
            *   Z: Topic
            *   S: Segmenti
            *   V: Vocabolario (parole)
        *   **Processo:** Dato un documento, si sceglie un topic e, condizionatamente a questo e al segmento, si campiona una parola.
        *   **Relazioni Probabilistiche:**
            *   $\text{Pr}(z|d)$: Probabilità di scegliere un topic *z* dato un documento *d*.
            *   $\text{Pr}(s|z)$: Probabilità di selezionare un segmento *s* dato un topic *z*.
            *   $\text{Pr}(w|z)$: Probabilità di scegliere una parola *w* dato un topic *z*.
            *   $\text{Pr}(w|s)$: Probabilità di scegliere una parola *w* dato un segmento *s*.
    *   **2.3 Caratteristiche del Modello:**
        *   Joint model triadico con topic come variabili latenti.
        *   Estensione di PLSA.
        *   Migliora la modellazione dei topic con una grana più fine.

**3. Segmentazione del Testo: Approcci**
    *   **3.1 Obiettivo:** Suddividere il testo in unità significative (segmenti) che rappresentano concetti o temi distinti.
    *   **3.2 Soluzione Naive:**
        *   Segmentare il testo in base ai paragrafi.
        *   Semplice ma spesso inaccurato.
    *   **3.3 Soluzione Intelligente (Non Supervisionata):**
        *   Utilizzo di tecniche di segmentazione non supervisionata in assenza di markup.
        *   Esempio: TextTiling.

**4. Text Tiling: Metodo di Segmentazione Non Supervisionato**
    *   **4.1 Passaggi Principali:**
        *   **Rappresentazione Vettoriale:** Il testo viene rappresentato come un vettore nello spazio vettoriale.
        *   **Calcolo della Similarità:** La similarità tra segmenti consecutivi viene calcolata utilizzando la similarità coseno.
        *   **Curva delle Similarità:** Viene tracciata una curva che rappresenta la similarità tra i segmenti.
        *   **Identificazione dei Punti di Discontinuità:** I minimi locali della curva indicano un cambiamento di topic.
        *   **Segmentazione:** Il testo viene segmentato in corrispondenza dei minimi locali.
    *   **4.2 Interpretazione della Curva:**
        *   Massimi locali: Continuità di topic.
        *   Minimi locali: Cambiamento di topic (punti di segmentazione).

---

## Schema Riassuntivo: LDA per Text Tiling e Valutazione Modelli di Topic

**I. Modello di Argomenti Latenti (LDA) per Text Tiling**

*   **A. Probabilità Congiunta:**
    *   Definizione: Probabilità di un documento, segmento e parola.
    *   Formula: $\operatorname{Pr}(d , s, w) = \operatorname{Pr}(d) \sum_{z \in Z} \operatorname{Pr}(z \mid d) \operatorname{Pr}(s \mid z) \operatorname{Pr}(w \mid z, s)$
    *   Variabili:
        *   $d$: Documento
        *   $s$: Segmento
        *   $w$: Parola
        *   $z$: Argomento
        *   $Z$: Insieme di tutti gli argomenti

*   **B. Processo di Generazione:**
    1.  Seleziona un documento $d$ da $\mathcal{D}$  => $\operatorname{Pr}(d)$
    2.  Per ogni segmento $s \in S_d$:
        *   Scegli un argomento $z$ per il documento $d$ => $\operatorname{Pr}(z \mid d)$
        *   Associa la probabilità argomento-segmento al segmento $s$ per l'argomento $z$ => $\operatorname{Pr}(s \mid z)$
        *   Per ogni parola $w$ nel segmento $s$:
            *   Scegli una parola $w$ dall'argomento e segmento corrente => $\operatorname{Pr}(w \mid z, s)$

**II. Algoritmo di Inferenza EM per LDA**

*   **A. E-step (Expectation):**
    *   Calcola la probabilità a posteriori dell'argomento dato il documento, il segmento e la parola.
    *   Formula: $\Pr(z|d, s, w) = \frac{\Pr(z, d, s, w)}{\Pr(d, s, w)} = \frac{\Pr(z|d)\Pr(s|z)\Pr(w|z, s)}{\sum_{z \in Z} \Pr(z|d)\Pr(s|z)\Pr(w|z, s)}$

*   **B. M-step (Maximization):**
    *   Aggiorna le probabilità di argomento, segmento e parola.
    *   Log-verosimiglianza attesa: $\mathbf{E}[l] = \sum_{d \in D} \sum_{s \in S} \sum_{w \in V} n(d, s, w) \times \sum_{z \in Z} \Pr(z|d, s, w) \log(\Pr(d, s, w))$
    *   Formule di aggiornamento:
        *   $\Pr(z|d) \propto \sum_{s \in S} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w)$
        *   $\Pr(s|z) \propto \sum_{d \in D} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w)$
        *   $\Pr(w|z, s) \propto \sum_{d \in D} n(d, s, w) \Pr(z|d, s, w)$
    *   Variabili:
        *   $n(d, s, w)$: Numero di volte in cui la parola $w$ appare nel segmento $s$ del documento $d$.
        *   $V$: Vocabolario.
    *   Iterazione: Ripetere E-step e M-step fino a convergenza.

**III. Applicazione del Text Tiling con LDA**

*   **A. Segmentazione:** Segmentare un documento in base ai suoi argomenti.
*   **B. Inferenza:** Utilizzare l'algoritmo EM per inferire le probabilità di argomento, segmento e parola.
*   **C. Identificazione:** Identificare i segmenti in base ai cambiamenti di argomento (minimi locali nella curva delle similarità).

**IV. Valutazione dei Modelli di Argomenti**

*   **A. Obiettivo:** Scoprire temi latenti all'interno di un corpus di documenti.
*   **B. Approccio:** Ogni documento è una miscela di topic, e ogni topic è una distribuzione di parole.
*   **C. Metriche di Valutazione Tradizionali:**
    *   **Coerenza:** Misura la co-occorrenza tra parole e argomenti. Punteggi più alti indicano una migliore coerenza. Gli incrementi diminuiscono all'aumentare del numero di argomenti.
    *   **Perplessità:** (Non descritta nel dettaglio nel testo, ma menzionata come metrica tradizionale)

---

**Schema Riassuntivo sulla Valutazione di Topic Model**

**1. Misure di Coerenza dei Topic**

    *   **1.1 Punteggio di Coerenza UMass:**
        *   Calcola la media a coppie sulle prime N parole di un topic.
        *   Formula: $C_{\text{UMass}}(w_{i},w_{j})=\log \frac{D(w_{i},w_{j})+1}{D(w_{i})}$
            *   $D(w_{i}, w_{j})$: numero di co-occorrenze di $w_{i}$ e $w_{j}$ nei documenti.
            *   $D(w_{i})$: numero di occorrenze di $w_{i}$ nei documenti.

    *   **1.2 Punteggio di Coerenza UCI:**
        *   Si basa su finestre scorrevoli e PMI (Pointwise Mutual Information).
        *   Formula: $C_{\text{UCI}}(w_{i},w_{j})=\log \frac{P(w_{i},w_{j})+1}{P(w_{i})\cdot P(w_{j})}$
            *   $P(w_{i}, w_{j})$: probabilità di co-occorrenza di $w_{i}$ e $w_{j}$ in una finestra scorrevole.
            *   $P(w_{i})$: probabilità di $w_{i}$ in una finestra scorrevole.
        *   Utilizza un corpus di Wikipedia in inglese con una finestra di 10 parole.
        *   Parametri: stride e overlap

    *   **1.3 Coerenza basata sulla Similarità Intra/Inter-Argomento:**
        *   Calcola la media della similarità intra-argomento divisa per la similarità inter-argomento.
        *   Similarità intra-argomento: media della similarità tra coppie di parole nello stesso topic.
        *   Similarità inter-argomento: media della similarità tra parole di topic diversi.
        *   Utilizza word embeddings e coseno come misura di similarità.

**2. Separazione dei Topic**

    *   La valutazione dei topic deve considerare sia la coerenza interna (compattezza) che la separazione tra i topic.
    *   Topic con distribuzioni di probabilità simili indicano una scarsa separazione.

**3. Entropia e Cross-Entropia**

    *   Un topic è una classe di attributi.
    *   L'entropia misura l'incertezza associata a un topic.
    *   La cross-entropia misura l'incertezza di una distribuzione rispetto a un'altra.

**4. Distanza tra Distribuzioni**

    *   **4.1 Divergenza di Kullback-Leibler (KL):**
        *   Entropia relativa di P rispetto a Q.
        *   Formula: $D_{KL}(P\|Q) = -\sum_{x\in X}P(x)\log\left( \frac{Q(x)}{P(x)} \right)$
        *   Proprietà:
            *   Non negatività: $D_{KL}(P\|Q) \ge 0$
            *   Asimmetria: $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$
            *   Non metrica.

    *   **4.2 Divergenza di Jensen-Shannon (JS):**
        *   Formula: $D_{JS} = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$
        *   Dove: $M = \frac{1}{2}(P + Q)$

    *   **4.3 Cross-Entropia:**
        *   Misura la differenza tra la distribuzione di probabilità prevista e quella effettiva.
        *   Può essere relativizzata o normalizzata.

**5. Perplessità**

    *   Misura l'incertezza di un modello linguistico (LM) quando genera un nuovo token.
    *   Minore è la perplessità, migliore è il modello.
    *   Dipende dalla tokenizzazione utilizzata.
    *   È la prima misura da utilizzare per valutare le prestazioni di un modello linguistico.
    *   È una funzione esponenziale dell'entropia.

---

**Schema Riassuntivo sulla Perplessità**

**1. Concetti Fondamentali**

*   **1.1 Variabile 'x' (Sorgente Informativa):**
    *   Rappresenta la sorgente di informazione (es. registrazione universitaria).
    *   Valori di 'x' (xᵢ) sono le parole/token.
*   **1.2 Entropia di 'x' (H[X]):**
    *   Misura il tasso di informazione trasmesso dalla sorgente.
    *   Limite superiore per la codifica dei token.
    *   Massima quando la distribuzione degli eventi è uniforme.
*   **1.3 Token Frequenti:**
    *   Eventi più probabili.
    *   Minimizzano la lunghezza della descrizione.
*   **1.4 Componente Informativa:**
    *   Della variabile che modella la frequenza dei token, P(x):  -log P(x).
*   **1.5 Perplessità (PP[X]):**
    *   Variabile aleatoria uguale a 2 elevato all'entropia di x.
    *   Formula:  $PP[X]:=2^{H[X]}$

**2. Teorema di Codifica Senza Rumore di Shannon**

*   Limite inferiore per la lunghezza attesa del codice di codifica dei token.
*   Lunghezza di codifica per ogni token: $-log(p(x))$.
*   Token frequenti assegnati a codici più brevi.

**3. Perplessità di un Processo Stocastico (χ)**

*   Sequenze **non i.i.d.** di variabili casuali (X₁, X₂, …).
*   Occorrenze di parole in un testo non sono indipendenti.

    *   **3.1 Prima Ipotesi: Stazionarietà**
        *   La probabilità di osservare una parola non cambia a seconda della finestra di testo.
        *   Non vera per documenti di testo (distribuzione diversa all'inizio e alla fine).
        *   Implica:
            *   $\lim_{ n \to \infty } \frac{1}{n}H[X_{1},\dots,X_{n}]$
            *   $\lim_{ n \to \infty } \frac{1}{n}H[X_{n}|X_{1},\dots,X_{n-1}]$

    *   **3.2 Seconda Ipotesi: Ergodicità**
        *   Valore atteso coincide con la media delle misurazioni per un numero elevato di osservazioni.
        *   La media dei logaritmi negativi delle probabilità approssima l'entropia per sequenze lunghe.
        *   Teorema Ergodico di Birkhoff:
            *   $\frac{1}{n}\sum_{i=1}^n X_{i}\to_{n \to \infty} E_{p}[X_{1}]\text{ con probabilità 1}$
        *   Implica (con Teorema di Shannon-McMillan-Breiman):
            *   $-\frac{1}{n}\log p(X_{1},\dots,X_{n})\to_{n\to \infty }H[X]\text{ con probabilità 1}$

    *   **3.3 Applicazione all'Entropia**
        *   Si usa un modello linguistico q(x₁, x₂, …) come approssimazione di p(x₁, x₂, …).
        *   Cross-entropia del modello Q rispetto alla sorgente P:
            *   $CE[P,Q]:=\lim_{ n \to \infty }-E_{p}\log q(X_{n}|X_{<n})=\lim_{ n \to \infty } -\frac{1}{n}E_{p}\log q(X_{1},\dots,X_{n})$

**4. Perplessità di un Modello Q per un Linguaggio P**

*   2 elevato alla cross-entropia.
*   In pratica: 2 elevato alla media della distribuzione di probabilità delle parole (w,d) in ogni documento.
*   Normalizzazione con N,d (numero di parole nel documento d).
*   Somma su tutti i documenti e elevazione a 2.

---

Ecco uno schema riassuntivo del testo fornito, organizzato gerarchicamente:

**I. Perplessità (PP): Definizione Formale**

*   Definizione: $$PP[P,Q]:=2^{CE[P,Q]}$$
*   Cross-Entropy (CE): $$-\frac{1}{n}\log q(X_{1},\dots,X_{n})\to_{n\to \infty}CE[P,Q]$$

**II. Perplessità di un Campione di Holdout**

*   Formula: $$ -\frac{\sum_{d=1}^M\log p(w_{d})}{\sum_{d=1}^MN_{d}} $$
*   Variabili:
    *   M: Numero di documenti nel campione di test.
    *   $w_{d}$: Parole nel documento d.
    *   $N_{d}$: Numero di parole nel documento d.

**III. Perplessità in Latent Dirichlet Allocation (LDA)**

*   Formula: $$\log p(w|\alpha,\beta)=E[\log p(\theta,z,w|\alpha,\beta)]-E[\log q(\theta,z)]$$
*   Valutazione dei parametri: $\alpha$ e $\beta$ (parametri di dispersione).
*   Metodo di inferenza: Variational Inference (più accurata del Gibbs sampling).

**IV. Perplessità in Streaming Gibbs Model (SGM)**

*   Formula: $$Pr(d,S_{d},V)=Pr(d)\prod_{S\in S_{d}}\sum_{z\in Z}Pr(z|d)Pr(s|z)\prod_{w\in V}Pr(w|z,s)$$
*   Descrizione: Modello LDA adattato a stream di dati.
*   Calcolo: Basato su probabilità simili a LDA, valutate per ogni documento durante la fase di test.

---
