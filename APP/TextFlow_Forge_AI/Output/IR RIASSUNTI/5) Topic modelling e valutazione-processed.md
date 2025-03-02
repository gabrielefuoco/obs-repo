
## Riassunto delle Librerie NLP e Modelli di Topic Modeling

Questo documento tratta le librerie Python per l'elaborazione del linguaggio naturale (NLP) e i modelli di topic modeling.

### Librerie NLP

* **NLTK:** Libreria completa per l'analisi del testo, offrendo strumenti per tokenizzazione, stemming, lemmatizzazione, analisi morfologica e sintattica, e classificazione del testo.
* **SpaCy:** Libreria focalizzata su velocità ed efficienza, con funzionalità avanzate come NER, estrazione di entità, analisi del sentimento e classificazione del testo, adatta anche per applicazioni commerciali.
* **Gensim:** Libreria per l'analisi di argomenti e modellazione di argomenti, implementando modelli come LDA.


### Latent Dirichlet Allocation (LDA)

LDA è un modello di topic modeling di riferimento.  Utilizza distribuzioni Dirichlet e multinomiali con parametri di dispersione $\alpha$ (per i topic) e $\eta$ (per le parole).  $\alpha$ controlla la dispersione della distribuzione di probabilità sui topic per documento, mentre $\eta$ controlla la dispersione della distribuzione di probabilità sulle parole per topic.  Si cerca di avere una $\alpha$ con grana fine (più inclusivo) e una $\eta$ meno dispersiva.  Il processo generativo è descritto come segue:

```
For each topic, generate a Dirichlet distribution over terms:
βₖ ~ Dirₘ(η), k ∈ {1, …, K}

For each document dᵢ, i ∈ {1, …, N}
    - Generate a Dirichlet distribution over topics: θᵢ ~ Dirₖ(α)
    - For each word position j in document dᵢ:
        - Choose a topic zᵢⱼ from the distribution in step a., i.e., zᵢⱼ ~ Multi(θᵢ)
        - Choose word wᵢⱼ from topic zᵢⱼ, i.e., wᵢⱼ ~ Multi(βzᵢⱼ)
```

![Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/5)-20241031095017801.png]

LDA può essere esteso per includere informazioni su autori o categorie, ma ciò aumenta la complessità computazionale.


### Modellazione di Documenti Segmentati per Argomento

Il modello di Ponti, Tagarelli, e Karypis (2011) propone un approccio più granulare alla modellazione argomento-documento, segmentando il testo in unità più piccole.  Questo modello traduce la relazione tra argomenti, documenti e segmenti in un modello probabilistico congiunto, considerando la probabilità di un documento dato un argomento e un segmento.  Questo approccio è motivato dalla necessità di rappresentare la complessità semantica di documenti lunghi, dove un singolo argomento per l'intero documento potrebbe essere riduttivo.  Approcci alternativi, come il *soft clustering*, permettono ad un documento di appartenere a più cluster, ma non considerano la struttura logica intrinseca del documento.

---

Questo documento descrive un metodo per migliorare la modellazione dei topic nei testi, affrontando il problema del *topic drift* tramite l'integrazione della segmentazione del testo nei topic model.  La segmentazione suddivide il documento in unità più piccole (segmenti) che rappresentano concetti o temi distinti, permettendo una cattura più fine della struttura semantica.  Una tecnica utile è l' *overclustering*, che stima un numero di cluster maggiore del numero effettivo di topic per identificare i *micro-topic*.

![[]]  Questa immagine illustra un modello probabilistico triadico che integra i segmenti (S) nel processo di generazione di parole (V) a partire da un documento (D) e un topic (Z).  Il modello, estensione di PLSA,  modella le probabilità condizionate:  `Pr(z|d)` (topic dato un documento), `Pr(s|z)` (segmento dato un topic), `Pr(w|z)` (parola dato un topic), e `Pr(w|s)` (parola dato un segmento).  Questo approccio migliora la granularità della modellazione dei topic rispetto ai modelli tradizionali.

La segmentazione del testo può essere effettuata con diversi approcci.  Un metodo semplice, ma spesso impreciso, è la segmentazione per paragrafi.  Approcci più sofisticati, come il *Text Tiling*, utilizzano tecniche non supervisionate.

Il *Text Tiling* rappresenta il testo come vettori nello spazio vettoriale (es. usando tf-idf), calcola la similarità coseno tra segmenti consecutivi, e identifica i punti di discontinuità (minimi locali) nella curva di similarità per segmentare il testo in base ai cambiamenti di topic. I minimi locali indicano cambiamenti di topic, mentre i massimi indicano continuità.

---

Il modello di argomenti latenti (LDA) può essere applicato al Text Tiling per segmentare un documento in base ai suoi argomenti.  La probabilità di un documento *d*, un segmento *s*, e una parola *w* è data da:

$$ \operatorname{Pr}(d , s, w) = \operatorname{Pr}(d) \sum_{z \in Z} \operatorname{Pr}(z \mid d) \operatorname{Pr}(s \mid z) \operatorname{Pr}(w \mid z, s) $$

dove *z* rappresenta un argomento e *Z* l'insieme di tutti gli argomenti.  Il processo generativo inizia selezionando un documento, poi, per ogni segmento, scegliendo un argomento e, infine, selezionando le parole per quel segmento e argomento.

L'inferenza delle probabilità si effettua tramite l'algoritmo EM.  L'E-step calcola la probabilità a posteriori dell'argomento dato il documento, il segmento e la parola:

$$ \begin{aligned} \text{E-step} \quad \Pr(z|d, s, w) &= \frac{\Pr(z, d, s, w)}{\Pr(d, s, w)} = \frac{\Pr(z|d)\Pr(s|z)\Pr(w|z, s)}{\sum_{z \in Z} \Pr(z|d)\Pr(s|z)\Pr(w|z, s)} \\ \end{aligned} $$

L'M-step aggiorna le probabilità  usando le formule seguenti:

$$ \begin{aligned} \text{M-step} \quad \mathbf{E}[l] &= \sum_{d \in D} \sum_{s \in S} \sum_{w \in V} n(d, s, w) \times \sum_{z \in Z} \Pr(z|d, s, w) \log(\Pr(d, s, w)) \ \\ \text{Update formulas} \quad \Pr(z|d) &\propto \sum_{s \in S} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w) \ \\ \Pr(s|z) &\propto \sum_{d \in D} \sum_{w \in V} n(d, s, w) \Pr(z|d, s, w) \ \\ \Pr(w|z, s) &\propto \sum_{d \in D} n(d, s, w) \Pr(z|d, s, w) \end{aligned} $$

dove  *n(d, s, w)* è il numero di occorrenze della parola *w* nel segmento *s* del documento *d*, e *V* è il vocabolario. L'algoritmo itera fino alla convergenza.  I segmenti vengono identificati tramite l'analisi dei minimi locali nella curva di similarità tra argomenti.

La valutazione dei modelli di topic, come LDA, si basa tradizionalmente su **coerenza** e **perplessità**. La coerenza misura la co-occorrenza tra parole all'interno di un argomento; punteggi più alti indicano una migliore coerenza.

---

Il testo descrive diverse metriche per valutare la coerenza e la separazione dei topic, nonché la performance dei modelli linguistici.  Sono presentate tre principali metodologie per la valutazione della coerenza dei topic:

1. **Punteggio di coerenza UMass:** Calcola la coerenza tra coppie di parole basandosi sulla loro co-occorrenza nei documenti.  La formula è:  `C<sub>UMass</sub>(w<sub>i</sub>,w<sub>j</sub>) = log [(D(w<sub>i</sub>, w<sub>j</sub>) + 1) / D(w<sub>i</sub>)]`, dove `D(w<sub>i</sub>, w<sub>j</sub>)` è il numero di co-occorrenze e `D(w<sub>i</sub>)` è il numero di occorrenze di `w<sub>i</sub>`.

2. **Punteggio di coerenza UCI:** Utilizza una finestra scorrevole e il PMI (Probabilità di Informazione Mutua) per valutare la coerenza. La formula è: `C<sub>UCI</sub>(w<sub>i</sub>,w<sub>j</sub>) = log [(P(w<sub>i</sub>, w<sub>j</sub>) + 1) / (P(w<sub>i</sub>) * P(w<sub>j</sub>))]`, dove `P(w<sub>i</sub>, w<sub>j</sub>)` e `P(w<sub>i</sub>)` sono le probabilità stimate da un corpus di Wikipedia.  È un approccio parametrico, dipendente da *stride* e *overlap*.

3. **Coerenza basata sulla similarità intra/inter-argomento:** Calcola la coerenza come rapporto tra la similarità media intra-argomento (similarità tra parole dello stesso argomento) e la similarità media inter-argomento (similarità tra parole di argomenti diversi), utilizzando il coseno come misura di similarità.

Oltre alla coerenza interna, è importante considerare la **separazione** dei topic, valutando la similarità delle loro distribuzioni di probabilità.  A tal fine, vengono introdotte l'**entropia** e la **cross-entropia** per misurare l'incertezza di una distribuzione di probabilità.  Per confrontare distribuzioni, si utilizzano la **divergenza di Kullback-Leibler (KL)**: `D<sub>KL</sub>(P||Q) = -Σ<sub>x∈X</sub> P(x)log(Q(x)/P(x))`, e la **divergenza di Jensen-Shannon (JS)**: `D<sub>JS</sub> = 1/2 * D<sub>KL</sub>(P||M) + 1/2 * D<sub>KL</sub>(Q||M)`, dove M è la media di P e Q.  La cross-entropia misura la differenza tra distribuzione prevista ed effettiva.

Infine, la **perplessità** è presentata come misura della performance dei modelli linguistici.  **Minore è la perplessità, migliore è il modello.**  È una funzione esponenziale dell'entropia, dipendente dalla tokenizzazione utilizzata.

---

## Riassunto del concetto di Perplessità

Questo testo tratta il concetto di perplessità, una metrica utilizzata per valutare la performance di un modello linguistico.  La perplessità si basa sul concetto di entropia, che misura l'incertezza o il tasso di informazione di una sorgente.

**Concetti Fondamentali:**

* **Variabile X:** Rappresenta la sorgente di informazioni testuali (es. un corpus), dove i valori xᵢ sono i singoli token (parole).
* **Entropia H[X]:** Misura l'incertezza della sorgente. È massima quando tutti i token hanno la stessa probabilità di apparire.  Il Teorema di Codifica Senza Rumore di Shannon stabilisce che la lunghezza attesa del codice per un token è -log(p(x)), con i token frequenti che hanno codici più brevi.
* **Perplessità di una singola variabile casuale:**  `PP[X] := 2^{H[X]}`.  È una misura dell'incertezza espressa in base 2.

**Perplessità in un processo stocastico (testo):**

Le parole in un testo non sono indipendenti e identicamente distribuite (i.i.d.).  Per calcolare la perplessità di un modello linguistico su un testo, si fanno due ipotesi semplificative:

* **Stazionarietà:** La probabilità di una parola non cambia a seconda della posizione nel testo (ipotesi non perfettamente realistica).
* **Ergodicità:** Per sequenze molto lunghe, la media temporale di una proprietà coincide con la sua aspettativa.  Il Teorema Ergodico di Birkhoff e il Teorema di Shannon-McMillan-Breiman permettono di approssimare l'entropia tramite la media dei logaritmi negativi delle probabilità dei token in una lunga sequenza.

**Perplessità di un modello linguistico:**

Poiché la vera distribuzione di probabilità del linguaggio (P) è sconosciuta, si usa un modello linguistico (Q) come approssimazione.  La perplessità del modello Q rispetto al linguaggio P è data da 2 elevato alla cross-entropia tra P e Q:

`CE[P,Q] := lim_{n→∞} -Eₚlog q(Xₙ|X_<ₙ)`

In pratica, la perplessità si calcola approssimando la cross-entropia come la media, su tutti i documenti e le parole in essi contenute, del logaritmo negativo della probabilità assegnata dal modello a ciascuna parola, elevata al quadrato:  `2^(media(-log(q(w,d))))`, dove `w,d` indica una parola in un documento `d`, e la media è normalizzata per il numero di parole in ogni documento.  Una perplessità minore indica un modello migliore, in quanto riflette una minore incertezza nella predizione delle parole.

---

La perplessità (PP) è una metrica per valutare la performance di un modello probabilistico, definita formalmente come  `$$PP[P,Q]:=2^{CE[P,Q]}$$` dove `$$-\frac{1}{n}\log q(X_{1},\dots,X_{n})\to_{n\to \infty}CE[P,Q]$$` rappresenta la convergenza dell'entropia incrociata (CE) tra la distribuzione vera *P* e quella stimata dal modello *Q*.

Per un campione di holdout, la perplessità si calcola come `$$ -\frac{\sum_{d=1}^M\log p(w_{d})}{\sum_{d=1}^MN_{d}} $$`, dove *M* è il numero di documenti,  `$w_{d}$` le parole nel documento *d*, e `$N_{d}$` il numero di parole in *d*.

Nell'ambito di LDA (Latent Dirichlet Allocation), la perplessità si ottiene tramite la *variational inference*: `$$\log p(w|\alpha,\beta)=E[\log p(\theta,z,w|\alpha,\beta)]-E[\log q(\theta,z)]$$`, dove α e β sono i parametri di dispersione.  Questo metodo è considerato più accurato del Gibbs sampling.

Infine, in SGM (Stochastic Generative Model), un'estensione di LDA per flussi di dati, la perplessità si basa su un calcolo probabilistico simile: `$$Pr(d,S_{d},V)=Pr(d)\prod_{S\in S_{d}}\sum_{z\in Z}Pr(z|d)Pr(s|z)\prod_{w\in V}Pr(w|z,s)$$`, valutato per ogni documento durante la fase di test.  In tutti i casi, la perplessità fornisce una misura della capacità del modello di predire nuovi dati.

---
