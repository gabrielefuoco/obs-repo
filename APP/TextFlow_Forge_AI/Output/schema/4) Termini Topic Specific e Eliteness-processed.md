
**I. Evoluzione del Concetto di Rilevanza nella Ricerca di Informazioni**

* **A. Da binario a probabilistico:**  Il concetto di rilevanza si è evoluto da un approccio semplice (rilevante/non rilevante) a uno probabilistico, a causa dei fattori latenti che introducono incertezza.
    * **1. Fattori latenti:**  Incertezza nelle query utente e nella rappresentazione dei documenti.
* **B. Il Binary Independence Model (BIM):**  Modello che stima la probabilità di rilevanza di un documento basandosi sulla presenza/assenza di termini.
    * **1. Limiti del BIM:** Efficace per query semplici (poche parole chiave), ma inadeguato per query complesse (es. ricerca web).
    * **2. Estensione del BIM:** Integrazione della frequenza dei termini (TF) per una stima più accurata della probabilità di rilevanza.
    * **3. Ipotesi di indipendenza:** Il BIM assume l'indipendenza tra i termini, un'ipotesi semplificativa.
* **C. Pseudo-rilevanza e Smoothing Bayesiano:**
    * **1. Pseudo-rilevanza:** Approssimazione semi-automatica della rilevanza, utile ma non perfetta.
    * **2. Prior Bayesiano:**  Fattore di smoothing per gestire l'incertezza e i fattori latenti, mitigando l'ipotesi di indipendenza dei termini.


**II. Termini Topic-Specific e Eliteness**

* **A. Problema dell'indipendenza dei termini:** L'ipotesi di indipendenza è problematica per i termini meno frequenti ("coda" della distribuzione).
    * **1. Termini "elite":** Termini specifici di un argomento (topic-specific), importanti nel contesto di un documento, ma non necessariamente Named Entities.
    * **2. Named Entities:**  Entità specifiche (persone, luoghi, organizzazioni), importanti ma non sempre i termini chiave per il tema principale.


**III. Approssimazione di Poisson**

* **A. Applicazione:** Utile per analizzare eventi rari in un intervallo di tempo o spazio (sequenza lunga, bassa probabilità di successo).
* **B. Generazione di documenti:**  Modella la probabilità di osservare un evento (parola) per posizione, campionando da una distribuzione multinomiale.
* **C. Regole empiriche:**  Per applicare l'approssimazione di Poisson, si necessitano almeno 30 misurazioni, idealmente 50-100.

---

**Schema Riassuntivo: Approssimazione di Poisson e Distribuzioni Alternative**

I. **Approssimazione di Poisson per la Distribuzione Binomiale**

   A. Regola Empirica:  Se K > 20 o 30 e la probabilità di successo ≈ 1/K, si può usare l'approssimazione di Poisson.
   B. Utilità: Particolarmente utile per la coda della distribuzione (probabilità di pochi eventi).
   C. Derivazione:  Approssimazione della distribuzione binomiale:
      1.  $$B_{T,P}(k)=\begin{pmatrix}T \\ K \end{pmatrix}p^k(1-p)^{t-k}$$  approssimata da:
      2.  $$p_{\lambda}(k)=\frac{\lambda^k}{k!}e^{-\lambda}$$  dove λ = Tp, k è il numero di occorrenze.
   D. Condizioni: T molto grande, p molto piccolo (coda della curva di Zipf).
   E. Parametri: Media = Varianza = λ = cf/T (cf = frequenza cumulativa, T = lunghezza del documento).
   F. Assunzione: "Intervallo fisso" implica lunghezza di documento fissa (es. abstract di dimensioni costanti).
   G. Legge di Zipf: La frequenza globale dei termini segue una legge di Zipf, indipendentemente dal documento.


II. **Limiti dell'Approssimazione di Poisson**

   A. Termini "Topic Specific": L'approssimazione fallisce per termini legati a un argomento specifico.
   B. Caratteristiche dei Termini "Topic Specific":
      1. Mancanza di indipendenza (co-occorrenza significativa).
      2. Contestualità ("contextual bound").
      3. Occorrenza a gruppi.
   C. Implicazioni per la Modellazione: Un modello Poissoniano non cattura la complessità dei termini "topic specific".


III. **Soluzioni Alternative**

   A. Okapi BM25: Approssimazione alternativa vantaggiosa.
   B. Distribuzione Binomiale Negativa:  Migliore adattamento per termini "topic specific".


IV. **Distribuzione Binomiale Negativa**

   A. Definizione: Modella la probabilità di osservare k insuccessi prima di r successi (stopping condition).
   B. Formula: $$NB_{\ r,p}(k)= \begin{pmatrix} k+r-1 \\ k \end{pmatrix}p^r(1-p)^k$$
   C. Parametri: r (numero di successi/dispersione), k (numero di insuccessi), p (probabilità di successo).
   D. Interpretazione:  k e r possono essere scambiati, invertendo il significato di successi e insuccessi.


---

## Schema Riassuntivo: Parametrizzazione Binomiale Negativa e Termini Elite nel Retrieval

**I. Parametrizzazione della Binomiale Negativa**

*   **Formula della media:**  $μ = \frac{rp}{1-p}$  (dove *r* è la dispersione e *p* la probabilità di successo)
*   **Espressioni di *p* e (1-*p*) in funzione di *r* e *μ*:**
    *   $p = \frac{μ}{μ+r}$
    *   $1-p = \frac{r}{μ+r}$
*   **Formula della binomiale negativa riparametrizzata:**
    $NB_{r,p}(k) = \begin{pmatrix} k+r-1 \\ k \end{pmatrix} \left( \frac{r}{μ+r} \right)^r \left( \frac{μ}{μ+r} \right)^k$
    *   *k+r* rappresenta il numero di trial (*T*), con l'assunzione che l'ultimo evento sia un successo.

**II. Termini Elite**

*   **Definizione:** Termini (generalmente sostantivi) che descrivono concetti/topic nel corpus.  L'eliteness è binaria (elite/non elite).
*   **Dipendenza dalla rilevanza del documento:** L'eliteness dipende dalla *term frequency* solo se il documento è rilevante per il topic.
*   **Modellazione dell'eliteness:**
    *   **Variabile nascosta:** $E_i$ (per il termine *i*) indica se il documento tratta il concetto del termine.
*   **Pattern distribuzionali:** Termini elite, essendo topic-specific, mostrano pattern distribuzionali simili (frequenze simili).
*   **Problema del "topic drift":** Un documento può contenere termini elite di concetti diversi, portando a classificazioni errate se non gestito (es. documento su football americano con termini elite sul baseball).

**III. Retrieval Status Value (RSV) con Termini Elite**

*   **Formula RSV con termini elite:** $RSV^{elite} = \sum_{i∈q} c_{i^{elite}}(tf_i)$
*   **Costo del termine i-esimo ($c_{i}^{elite}(tf_i)$):**
    $c_{i}^{elite}(tf_{i}) = \log \frac{p(TF_{i}=tf_{i}|R=1)p(TF_{i}=0|R=0)}{p(TF_{i}=0|R=1)p(TF_{i}=tf_{i}|R=0)}$
*   **Incorporazione dell'eliteness:**
    $p(TF_{i}=tf_{i}|R) = p(TF_{i}=tf_{i}|E=elite)p(E=elite|R) + p(TF_{i}=tf_{i}|E=non\ elite)(1-p(E_{i}=elite|R))$
    *   La probabilità di osservare una specifica *term frequency* ($tf_i$) data la rilevanza (*R*) è modellata come combinazione di probabilità condizionate all'eliteness ($E_i$).


---

**Modello di Recupero dell'Informazione: Evoluzione di BM25**

I. **Modelli di Poisson per la Term Frequency (TF)**

   A. **Modello a Due Poisson:**
      1. Utilizza due distribuzioni di Poisson per modellare la TF: una per termini "elite" (λ) e una per termini non "elite" (μ).
      2. Probabilità di osservare TF=k dato lo stato di rilevanza R:  $p(TF_{i}=k|R)=\pi \frac{\lambda^k}{k!}e^{-\lambda}+(1-\pi) \frac{\mu^k}{k! }e^{-\mu}$  dove π è la probabilità che un termine sia "elite".
      3. Stima dei parametri (π, λ, μ) complessa.

   B. **Modello di Poisson Semplificato:**
      1. Aumenta monotonicamente con la TF.
      2. Asintoticamente si avvicina a un valore massimo che rappresenta il peso dell' "eliteness".
      3. Approssimazione tramite curva parametrica con le stesse proprietà qualitative.

II. **Costo per Termini "Elite"**

   A.  Costo $c_i^{elite}$:
      1. $c_i^{elite}(0) = 0$
      2. Cresce monotonicamente con la TF, saturando per alti valori di λ.
      3. Stima dei parametri tramite funzione con caratteristiche qualitative simili.

III. **Approssimazione della Distribuzione di Poisson**

   A.  Funzione di approssimazione: $\frac{tf}{k_1 + tf}$
   B.  Comportamento:
      1.  Per alti $k_1$, gli incrementi in $tf_i$ contribuiscono significativamente.
      2.  Per bassi $k_1$, i contributi diminuiscono rapidamente.  L'approssimazione peggiora per alti $k_1$.

IV. **BM25: Versioni e Estensioni**

   A. **Versione 1:**
      1. Utilizza la funzione di saturazione: $c_{i}^{BM25v_{1}}(tf_{i})=c_{i}^{BIM} \frac{tf_{i}}{k_{1}+tf_{i}}$
      2. $c_i^{BIM}$: costo del termine i-esimo calcolato con il modello BIM.

   B. **Versione 2:**
      1. Semplifica BIM usando solo IDF: $c_{i}^{BM25v_{2}}(tf_{i})=\log \frac{N}{df_{i}}\times \frac{(k_{1}+1)tf_{i}}{k_{1}+tf_{i}}$
      2. N: numero totale di documenti.
      3. $df_i$: frequenza del termine i-esimo.

   C. **Estensioni:**
      1. **Prima estensione:** Aggiunge un fattore di smoothing basato sulla funzione di saturazione al costo $c_i$ di BM25.
      2. **Seconda estensione:** Utilizza solo la stima di $r_i$ (probabilità di rilevanza dato il termine) e $df_i$, senza il costo $c_i$ completo.

V. **Normalizzazione della Lunghezza del Documento**

   A. Miglioramento di BM25 per compensare le variazioni di lunghezza dei documenti, influenzando il valore di $tf_i$.

---

**Okapi BM25: Schema Riassuntivo**

I. **Lunghezza del Documento e Normalizzazione:**

   * A.  Definizioni:
      *  `dl = Σᵢ∈V tfᵢ`: Lunghezza del documento (somma delle frequenze dei termini).
      *  `avdl`: Lunghezza media dei documenti nella collezione.
   * B.  Motivi per la variabilità della lunghezza dei documenti:
      *  Verbosità (alta `tfᵢ` erronea).
      *  Ampio ambito (alta `tfᵢ` corretta).
   * C.  Componente di normalizzazione della lunghezza:
      *  `B = ((1-b) + b * (dl/avdl))`,  `0 ≤ b ≤ 1`
      *  `b = 1`: Normalizzazione completa.
      *  `b = 0`: Nessuna normalizzazione.


II. **Modello Okapi BM25:**

   * A.  Estensione del modello BIM, considera la lunghezza del documento.
   * B.  Normalizzazione della frequenza del termine (`tfᵢ`):
      *  `tfᵢ' = tfᵢ / B`
   * C.  Formula del punteggio BM25 per il termine `i`:
      *  `cᵢᴮᴹ²⁵(tfᵢ) = log(N/dfᵢ) * ((k₁+1)tfᵢ')/(k₁+tfᵢ') = log(N/dfᵢ) * ((k₁+1)tfᵢ)/(k₁((1-b)+b(dl/avdl))+tfᵢ)`
      *  `N`: Numero totale di documenti.
      *  `dfᵢ`: Numero di documenti contenenti il termine `i`.
      *  `k₁` e `b`: Parametri del modello.
      *  `dl`: Lunghezza del documento.
      *  `avdl`: Lunghezza media dei documenti.
   * D.  Funzione di ranking BM25:
      *  `RSVᴮᴹ²⁵ = Σᵢ∈q cᵢᴮᴹ²⁵(tfᵢ)`  (`q`: termini nella query).


III. **Parametri del Modello BM25:**

   * A.  `k₁`: Gestisce la pendenza della funzione di saturazione (scaling della `tf`).
      *  `k₁ = 0`: Modello binario.
      *  `k₁` grande: `tf` grezza.
   * B.  `b`: Controlla la normalizzazione della lunghezza del documento.
      *  `b = 0`: Nessuna normalizzazione.
      *  `b = 1`: Frequenza relativa (normalizzazione completa).
   * C.  Valori tipici: `k₁ ≈ 1.2-2`, `b ≈ 0.75`.


IV. **Esempio di Applicazione:**

   * A.  Query: "machine learning"
   * B.  Documenti con conteggi di termini:
      *  `doc1`: learning 1024, machine 1
      *  `doc2`: learning 16, machine 8
   * C.  Confronto tf-idf e BM25 (con `k₁ = 2`).


V. **Ranking con Zone:**

   * A.  Le zone sono sezioni specifiche di un documento (es. titolo, abstract, ecc.).



---

**I. Approcci al Ranking con Zone di Documento**

* **A. Idea Semplice: Combinazione Lineare Ponderata**
    * Applicare BM25 a ciascuna zona separatamente.
    * Combinare i punteggi con una combinazione lineare ponderata ($v_z$).
    * Limite: Assume indipendenza irragionevole tra le proprietà di eliteness delle zone.

* **B. Idea Alternativa: Eliteness Condivisa**
    * Assumere che l'eliteness sia una proprietà termine/documento condivisa tra le zone.
    * La relazione tra eliteness e frequenza dei termini dipende dalla zona (es. maggiore densità di termini elite nel titolo).
    * Metodo: Combinare prima le prove tra le zone per ciascun termine, poi tra i termini.


**II. Calcolo delle Varianti Pesate di Frequenza dei Termini e Lunghezza del Documento**

* Formule:
    * $\tilde{t}f_{i} = \sum_{z=1}^{Z} v_{z} tf_{zi}$
    * $\tilde{dl} = \sum_{z=1}^{Z} v_{z} len_{z}$
    * $avdl = \frac{\text{average } \tilde{dl}}{\text{across all docs}}$
    * $v_z$: peso della zona; $tf_{zi}$: frequenza del termine nella zona z; $len_z$: lunghezza della zona z; Z: numero di zone.

* Metodo:
    1. Calcolo della TF per zona.
    2. Normalizzazione per zona.
    3. Assegnazione del peso della zona ($v_z$) - parametro predefinito, non apprendibile.


**III. Modelli BM25F con Zone**

* **A. Simple BM25F:**
    * Interpretazione: zona *z* "replicata" *y* volte.
    * Formula RSV:  $RSV^{SimpleBM25F} = \sum_{i \in q} \log \frac{N}{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b \frac{dl}{avdl}) + tf_i}$

* **B. BM25F con Normalizzazione Specifica per Zona:**
    * Normalizzazione della lunghezza specifica per zona (b_z) migliora le prestazioni.
    * $\tilde{tf}_i = \sum_{z=1}^Z v_z \frac{f_{zi}}{B_z}$  dove $B_z = \left( (1-b_z) + b_z \frac{\text{len}_z}{\text{avlen}_z} \right)$
    * Formula RSV: $RSV^{BM25F} = \sum_{i \in q} \log \frac{N }{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_{1}+tf_{i}}$ (denominatore semplificato rispetto a Simple BM25F).


**IV. Classifica con Caratteristiche Non Testuali**

* **A. Assunzioni:**
    * **Indipendenza Usuale:** Le caratteristiche non testuali sono indipendenti tra loro e dalle caratteristiche testuali.  Formula: $\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$
    * **Indipendenza dalla Query:** Le informazioni di rilevanza sono indipendenti dalla query (es. PageRank, età, tipo).  Consente di mantenere tutte le caratteristiche non testuali nella derivazione in stile BIM.


---

**RSV (Ranking Score Value)**

* **Calcolo del RSV:**
    * Formula:  $$RSV=\sum_{i\in q}c_{i}(tf_{i})+\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$$
    *  $c_i(tf_i)$:  Componente basata sulla frequenza dei termini (tf) nel documento.
    *  $\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$: Componente basata su features aggiuntive ($f_j$).
        * $V_{j}(f_{j})=\log\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$:  Valore della feature $f_j$, basato sulla probabilità condizionata di $f_j$ dato che il documento è rilevante (R=1) o irrilevante (R=0).
        * $\lambda_j$: Parametro di scala per le features, per compensare le approssimazioni.

* **Selezione di $V_j$:**
    * Importanza della scelta di $V_j$ in base alla feature $f_j$.
    * Esempio: La scelta appropriata di $V_j$ spiega l'efficacia di $Rsv_{bm25} + log(\text{pagerank})$.

---
