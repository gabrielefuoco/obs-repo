
# Evoluzione del Concetto di Rilevanza nella Ricerca di Informazioni

## Da binario a probabilistico

Il concetto di rilevanza si è evoluto da un approccio semplice (rilevante/non rilevante) a uno probabilistico, a causa dei fattori latenti che introducono incertezza nella valutazione.

### Fattori latenti

L'incertezza risiede sia nelle query degli utenti (ambiguità, incompletezza) sia nella rappresentazione dei documenti (sinonimi, polisemia).


## Il Binary Independence Model (BIM)

Il BIM stima la probabilità di rilevanza di un documento basandosi sulla presenza/assenza di termini nella query e nel documento.

### Limiti del BIM

Il BIM è efficace per query semplici (poche parole chiave), ma inadeguato per query complesse come quelle tipiche della ricerca web.

### Estensione del BIM

L'integrazione della frequenza dei termini (TF) permette una stima più accurata della probabilità di rilevanza.

### Ipotesi di indipendenza

Il BIM assume l'indipendenza tra i termini, un'ipotesi semplificativa che limita la sua accuratezza.


## Pseudo-rilevanza e Smoothing Bayesiano

### Pseudo-rilevanza

L'utilizzo di un insieme di documenti considerati (semi-automaticamente) rilevanti per una data query, per migliorare la stima della probabilità.  Questa approssimazione è utile ma non perfetta.

### Prior Bayesiano

L'introduzione di un prior Bayesiano agisce come fattore di smoothing, gestendo l'incertezza e mitigando l'ipotesi di indipendenza dei termini.


# Termini Topic-Specific e Eliteness

## Problema dell'indipendenza dei termini

L'ipotesi di indipendenza è particolarmente problematica per i termini meno frequenti ("coda" della distribuzione).

### Termini "elite"

I termini "elite" sono specifici di un argomento (topic-specific), importanti nel contesto di un documento, ma non necessariamente Named Entities.

### Named Entities

Le Named Entities (persone, luoghi, organizzazioni) sono importanti, ma non sempre rappresentano i termini chiave per il tema principale di un documento.


# Approssimazione di Poisson

## Applicazione

L'approssimazione di Poisson è utile per analizzare eventi rari in un intervallo di tempo o spazio (es.  sequenza lunga, bassa probabilità di successo).

## Generazione di documenti

Nel contesto della generazione di documenti, modella la probabilità di osservare una parola in una determinata posizione, campionando da una distribuzione multinomiale.

## Regole empiriche

Per applicare l'approssimazione di Poisson, sono necessarie almeno 30 misurazioni, idealmente 50-100.


# Schema Riassuntivo: Approssimazione di Poisson e Distribuzioni Alternative

## Approssimazione di Poisson per la Distribuzione Binomiale

* **Regola Empirica:** Se K > 20 o 30 e la probabilità di successo ≈ 1/K, si può usare l'approssimazione di Poisson.
* **Utilità:** Particolarmente utile per la coda della distribuzione (probabilità di pochi eventi).
* **Derivazione:** Approssimazione della distribuzione binomiale:
    1. $$B_{T,P}(k)=\begin{pmatrix}T \\ K \end{pmatrix}p^k(1-p)^{t-k}$$ approssimata da:
    2. $$p_{\lambda}(k)=\frac{\lambda^k}{k!}e^{-\lambda}$$ dove λ = Tp, k è il numero di occorrenze.
* **Condizioni:** T molto grande, p molto piccolo (coda della curva di Zipf).
* **Parametri:** Media = Varianza = λ = cf/T (cf = frequenza cumulativa, T = lunghezza del documento).
* **Assunzione:** "Intervallo fisso" implica lunghezza di documento fissa (es. abstract di dimensioni costanti).
* **Legge di Zipf:** La frequenza globale dei termini segue una legge di Zipf, indipendentemente dal documento.


## Limiti dell'Approssimazione di Poisson

* **Termini "Topic Specific":** L'approssimazione fallisce per termini legati a un argomento specifico.
* **Caratteristiche dei Termini "Topic Specific":**
    1. Mancanza di indipendenza (co-occorrenza significativa).
    2. Contestualità ("contextual bound").
    3. Occorrenza a gruppi.
* **Implicazioni per la Modellazione:** Un modello Poissoniano non cattura la complessità dei termini "topic specific".


## Soluzioni Alternative

* **Okapi BM25:** Approssimazione alternativa vantaggiosa.
* **Distribuzione Binomiale Negativa:** Migliore adattamento per termini "topic specific".


## Distribuzione Binomiale Negativa

* **Definizione:** Modella la probabilità di osservare k insuccessi prima di r successi (stopping condition).
* **Formula:** $$NB_{\ r,p}(k)= \begin{pmatrix} k+r-1 \\ k \end{pmatrix}p^r(1-p)^k$$
* **Parametri:** r (numero di successi/dispersione), k (numero di insuccessi), p (probabilità di successo).
* **Interpretazione:** k e r possono essere scambiati, invertendo il significato di successi e insuccessi.


# Schema Riassuntivo: Parametrizzazione Binomiale Negativa e Termini Elite nel Retrieval

**(Questo titolo è incompleto nel testo originale e richiede ulteriori informazioni per una corretta formattazione.)**

---

# Parametrizzazione della Binomiale Negativa

* **Formula della media:**  μ = \frac{rp}{1-p} (dove *r* è la dispersione e *p* la probabilità di successo)

* **Espressioni di *p* e (1-*p*) in funzione di *r* e *μ*:**
    * p = \frac{μ}{μ+r}
    * 1-p = \frac{r}{μ+r}

* **Formula della binomiale negativa riparametrizzata:**
NB<sub>r,p</sub>(k) = \begin{pmatrix} k+r-1 \\ k \end{pmatrix} \left( \frac{r}{μ+r} \right)^r \left( \frac{μ}{μ+r} \right)^k

* *k+r* rappresenta il numero di trial (*T*), con l'assunzione che l'ultimo evento sia un successo.


# II. Termini Elite

* **Definizione:** Termini (generalmente sostantivi) che descrivono concetti/topic nel corpus. L'eliteness è binaria (elite/non elite).

* **Dipendenza dalla rilevanza del documento:** L'eliteness dipende dalla *term frequency* solo se il documento è rilevante per il topic.

* **Modellazione dell'eliteness:**
    * **Variabile nascosta:**  E<sub>i</sub> (per il termine *i*) indica se il documento tratta il concetto del termine.
    * **Pattern distribuzionali:** Termini elite, essendo topic-specific, mostrano pattern distribuzionali simili (frequenze simili).
    * **Problema del "topic drift":** Un documento può contenere termini elite di concetti diversi, portando a classificazioni errate se non gestito (es. documento su football americano con termini elite sul baseball).


# III. Retrieval Status Value (RSV) con Termini Elite

* **Formula RSV con termini elite:** RSV<sup>elite</sup> = \sum_{i∈q} c<sub>i<sup>elite</sup></sub>(tf<sub>i</sub>)

* **Costo del termine i-esimo ($c_{i}^{elite}(tf_i)$):**
c<sub>i<sup>elite</sup></sub>(tf<sub>i</sub>) = \log \frac{p(TF<sub>i</sub>=tf<sub>i</sub>|R=1)p(TF<sub>i</sub>=0|R=0)}{p(TF<sub>i</sub>=0|R=1)p(TF<sub>i</sub>=tf<sub>i</sub>|R=0)}

* **Incorporazione dell'eliteness:**
p(TF<sub>i</sub>=tf<sub>i</sub>|R) = p(TF<sub>i</sub>=tf<sub>i</sub>|E=elite)p(E=elite|R) + p(TF<sub>i</sub>=tf<sub>i</sub>|E=non\ elite)(1-p(E<sub>i</sub>=elite|R))

La probabilità di osservare una specifica *term frequency* (tf<sub>i</sub>) data la rilevanza (*R*) è modellata come combinazione di probabilità condizionate all'eliteness (E<sub>i</sub>).


# Modello di Recupero dell'Informazione: Evoluzione di BM25

## I. Modelli di Poisson per la Term Frequency (TF)

### A. Modello a Due Poisson:

1. Utilizza due distribuzioni di Poisson per modellare la TF: una per termini "elite" (λ) e una per termini non "elite" (μ).
2. Probabilità di osservare TF=k dato lo stato di rilevanza R:
p(TF<sub>i</sub>=k|R) = π \frac{λ<sup>k</sup>}{k!}e<sup>-λ</sup> + (1-π) \frac{μ<sup>k</sup>}{k!}e<sup>-μ</sup>
dove π è la probabilità che un termine sia "elite".
3. Stima dei parametri (π, λ, μ) complessa.

### B. Modello di Poisson Semplificato:

1. Aumenta monotonicamente con la TF.
2. Asintoticamente si avvicina a un valore massimo che rappresenta il peso dell' "eliteness".
3. Approssimazione tramite curva parametrica con le stesse proprietà qualitative.


## II. Costo per Termini "Elite"

### A. Costo c<sub>i</sub><sup>elite</sup>:

1. c<sub>i</sub><sup>elite</sup>(0) = 0
2. Cresce monotonicamente con la TF, saturando per alti valori di λ.
3. Stima dei parametri tramite funzione con caratteristiche qualitative simili.


## III. Approssimazione della Distribuzione di Poisson

### A. Funzione di approssimazione:

\frac{tf}{k<sub>1</sub> + tf}

### B. Comportamento:

1. Per alti k<sub>1</sub>, gli incrementi in tf<sub>i</sub> contribuiscono significativamente.
2. Per bassi k<sub>1</sub>, i contributi diminuiscono rapidamente. L'approssimazione peggiora per alti k<sub>1</sub>.


## IV. BM25: Versioni e Estensioni

### A. Versione 1:

1. Utilizza la funzione di saturazione: c<sub>i</sub><sup>BM25v<sub>1</sub></sup>(tf<sub>i</sub>) = c<sub>i</sub><sup>BIM</sup> \frac{tf<sub>i</sub>}{k<sub>1</sub>+tf<sub>i</sub>}
2. c<sub>i</sub><sup>BIM</sup>: costo del termine i-esimo calcolato con il modello BIM.

### B. Versione 2:

1. Semplifica BIM usando solo IDF: c<sub>i</sub><sup>BM25v<sub>2</sub></sup>(tf<sub>i</sub>) = \log \frac{N}{df<sub>i</sub>} \times \frac{(k<sub>1</sub>+1)tf<sub>i</sub>}{k<sub>1</sub>+tf<sub>i</sub>}
2. N: numero totale di documenti.
3. df<sub>i</sub>: frequenza del termine i-esimo.

### C. Estensioni:

1. **Prima estensione:** Aggiunge un fattore di smoothing basato sulla funzione di saturazione al costo c<sub>i</sub> di BM25.
2. **Seconda estensione:** Utilizza solo la stima di r<sub>i</sub> (probabilità di rilevanza dato il termine) e df<sub>i</sub>, senza il costo c<sub>i</sub> completo.


## V. Normalizzazione della Lunghezza del Documento

Miglioramento di BM25 per compensare le variazioni di lunghezza dei documenti, influenzando il valore di tf<sub>i</sub>.


# Okapi BM25: Schema Riassuntivo

## I. Lunghezza del Documento e Normalizzazione:

### A. Definizioni:

* dl = Σᵢ∈V tfᵢ: Lunghezza del documento (somma delle frequenze dei termini).
* avdl: Lunghezza media dei documenti nella collezione.

### B. Motivi per la variabilità della lunghezza dei documenti:

* Verbosità (alta tfᵢ erronea).
* Ampio ambito (alta tfᵢ corretta).

### C.  ![[]]

---

# Ranking di Documenti: Modelli e Approcci

## I. Normalizzazione della Lunghezza del Documento

Il componente di normalizzazione della lunghezza è definito da:

`B = ((1-b) + b * (dl/avdl))`,  `0 ≤ b ≤ 1`

* `b = 1`: Normalizzazione completa.
* `b = 0`: Nessuna normalizzazione.


## II. Modello Okapi BM25

Il modello Okapi BM25 è un'estensione del modello BIM che considera la lunghezza del documento.

**A. Normalizzazione della frequenza del termine (`tfᵢ`):**

`tfᵢ' = tfᵢ / B`

**B. Formula del punteggio BM25 per il termine `i`:**

`cᵢᴮᴹ²⁵(tfᵢ) = log(N/dfᵢ) * ((k₁+1)tfᵢ')/(k₁+tfᵢ') = log(N/dfᵢ) * ((k₁+1)tfᵢ)/(k₁((1-b)+b(dl/avdl))+tfᵢ)`

dove:

* `N`: Numero totale di documenti.
* `dfᵢ`: Numero di documenti contenenti il termine `i`.
* `k₁` e `b`: Parametri del modello.
* `dl`: Lunghezza del documento.
* `avdl`: Lunghezza media dei documenti.

**C. Funzione di ranking BM25:**

`RSVᴮᴹ²⁵ = Σᵢ∈q cᵢᴮᴹ²⁵(tfᵢ)`  (`q`: termini nella query).


## III. Parametri del Modello BM25

* **A. `k₁`:** Gestisce la pendenza della funzione di saturazione (scaling della `tf`).
    * `k₁ = 0`: Modello binario.
    * `k₁` grande: `tf` grezza.
* **B. `b`:** Controlla la normalizzazione della lunghezza del documento.
    * `b = 0`: Nessuna normalizzazione.
    * `b = 1`: Frequenza relativa (normalizzazione completa).
* **C. Valori tipici:** `k₁ ≈ 1.2-2`, `b ≈ 0.75`.


## IV. Esempio di Applicazione

* **A. Query:** "machine learning"
* **B. Documenti con conteggi di termini:**
    * `doc1`: learning 1024, machine 1
    * `doc2`: learning 16, machine 8
* **C. Confronto tf-idf e BM25 (con `k₁ = 2`):**  (Questo punto richiede ulteriori dettagli per essere completato.)


## V. Ranking con Zone

Le zone sono sezioni specifiche di un documento (es. titolo, abstract, ecc.).

### I. Approcci al Ranking con Zone di Documento

**A. Idea Semplice: Combinazione Lineare Ponderata**

* Applicare BM25 a ciascuna zona separatamente.
* Combinare i punteggi con una combinazione lineare ponderata ($v_z$).
* Limite: Assume indipendenza irragionevole tra le proprietà di eliteness delle zone.

**B. Idea Alternativa: Eliteness Condivisa**

* Assumere che l'eliteness sia una proprietà termine/documento condivisa tra le zone.
* La relazione tra eliteness e frequenza dei termini dipende dalla zona (es. maggiore densità di termini elite nel titolo).
* Metodo: Combinare prima le prove tra le zone per ciascun termine, poi tra i termini.


### II. Calcolo delle Varianti Pesate di Frequenza dei Termini e Lunghezza del Documento

Formule:

* $\tilde{t}f_{i} = \sum_{z=1}^{Z} v_{z} tf_{zi}$
* $\tilde{dl} = \sum_{z=1}^{Z} v_{z} len_{z}$
* $avdl = \frac{\text{average } \tilde{dl}}{\text{across all docs}}$

dove:

* $v_z$: peso della zona;
* $tf_{zi}$: frequenza del termine nella zona z;
* $len_z$: lunghezza della zona z;
* Z: numero di zone.

Metodo:

1. Calcolo della TF per zona.
2. Normalizzazione per zona.
3. Assegnazione del peso della zona ($v_z$) - parametro predefinito, non apprendibile.


### III. Modelli BM25F con Zone

**A. Simple BM25F:**

Interpretazione: zona *z* "replicata" *y* volte.

Formula RSV:  $RSV^{SimpleBM25F} = \sum_{i \in q} \log \frac{N}{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b \frac{dl}{avdl}) + tf_i}$

**B. BM25F con Normalizzazione Specifica per Zona:**

Normalizzazione della lunghezza specifica per zona (b_z) migliora le prestazioni.

$\tilde{tf}_i = \sum_{z=1}^Z v_z \frac{f_{zi}}{B_z}$ dove $B_z = \left( (1-b_z) + b_z \frac{\text{len}_z}{\text{avlen}_z} \right)$

Formula RSV: $RSV^{BM25F} = \sum_{i \in q} \log \frac{N }{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_{1}+tf_{i}}$ (denominatore semplificato rispetto a Simple BM25F).


## IV. Classifica con Caratteristiche Non Testuali

**A. Assunzioni:**

* **Indipendenza Usuale:** Le caratteristiche non testuali sono indipendenti tra loro e dalle caratteristiche testuali. Formula: $\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$
* **Indipendenza dalla Query:** Le informazioni di rilevanza sono indipendenti dalla query (es. PageRank, età, tipo). Consente di mantenere tutte le caratteristiche non testuali nella derivazione in stile BIM.


## RSV (Ranking Score Value)

**Calcolo del RSV:**

Formula: $$RSV=\sum_{i\in q}c_{i}(tf_{i})+\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$$

* $c_i(tf_i)$: Componente basata sulla frequenza dei termini (tf) nel documento.
* $\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$: Componente basata su features aggiuntive ($f_j$).


---

# Valore delle Feature e Selezione di Vj

La formula seguente definisce il valore di una feature  `fj`:

$V_{j}(f_{j})=\log\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$

Questa formula rappresenta il valore della feature `fj` basato sulla probabilità condizionata che la feature assuma il valore `fj`, dato che il documento è rilevante (R=1) o irrilevante (R=0).  Il logaritmo del rapporto di queste probabilità fornisce una misura dell'importanza della feature nel discriminare tra documenti rilevanti e irrilevanti.


Un parametro di scala, λj, viene utilizzato per compensare le approssimazioni nel calcolo del valore della feature:

* $\lambda_j$: Parametro di scala per le features.


## Selezione di Vj

La scelta di una funzione `Vj` appropriata per rappresentare il valore di una feature `fj` è cruciale per l'efficacia del modello di recupero dell'informazione.  La scelta di `Vj` influenza direttamente la capacità del modello di discriminare tra documenti rilevanti e irrilevanti.

Un esempio di una scelta efficace di `Vj` è illustrato dalla combinazione  `Rsv_{bm25} + log(\text{pagerank})`.  L'efficacia di questa combinazione suggerisce che una scelta appropriata di `Vj` può migliorare significativamente le prestazioni del sistema di recupero dell'informazione.  L'esempio evidenzia l'importanza di una attenta selezione della funzione `Vj` per massimizzare l'utilità delle features nel processo di recupero.

---

Per favore, forniscimi il testo da formattare.  Ho bisogno del testo che desideri che io organizzi e formati secondo le tue istruzioni per poterti aiutare.

---
