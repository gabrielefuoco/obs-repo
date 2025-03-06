
**I. Rilevanza e Incertezza nella Ricerca di Informazioni**

*   **A. Evoluzione del concetto di rilevanza:**
    *   Da criterio binario (rilevante/non rilevante) a probabilistico.
    *   Causa: Presenza di fattori latenti che introducono incertezza.

*   **B. Binary Independence Model (BIM):**
    *   Modello probabilistico che considera la probabilità di rilevanza di un documento.
    *   Indicatore di rilevanza: Presenza/assenza di un termine.
    *   Efficace per: Ricerca in archivi o messaggi con query brevi.

*   **C. Estensione del BIM:**
    *   Integrazione della frequenza dei termini (TF).
    *   Migliore stima della probabilità di rilevanza per query complesse.

*   **D. Limitazioni del BIM:**
    *   Assunzione di indipendenza tra i termini (semplificazione).

*   **E. Pseudo-rilevanza:**
    *   Approssimazione semi-automatica della rilevanza.
    *   Utilizzata in ottimizzazione SEO.
    *   Importanza di una stima accurata della rilevanza.

*   **F. Prior Bayesiano:**
    *   Fattore di smoothing per gestire l'incertezza.
    *   Cattura fattori latenti nella rappresentazione dei documenti.
    *   Mitiga l'impatto dell'assunzione di indipendenza tra i termini.

**II. Termini Topic Specific e Eliteness**

*   **A. Problema dell'indipendenza dei termini:**
    *   Criticità per i termini meno frequenti (coda della distribuzione).
    *   Trattamento come indipendenti non corrispondente alla realtà.

*   **B. Termini "Elite":**
    *   Termini specifici di un determinato argomento (topic-specific).
    *   Importanza particolare nel contesto specifico di un documento.
    *   Esempio: In un articolo su Donald Trump, termini come "presidente", "Stati Uniti" o "elezioni".

*   **C. Termini Elite vs. Named Entities:**
    *   Non necessariamente coincidenti.
    *   Named Entities importanti, ma non sempre termini chiave per il tema principale.

**III. L'approssimazione di Poisson**

*   **A. Applicazione:**
    *   Analisi di eventi rari in un intervallo di tempo o spazio.

*   **B. Condizioni:**
    *   Sequenza di eventi molto lunga.
    *   Probabilità di successo per ogni evento molto bassa.

*   **C. Generazione di documenti:**
    *   Modellazione della probabilità di osservare un evento posizione per posizione.
    *   Occorrenza di parola campionata da una distribuzione multinomiale.

*   **D. Regole empiriche:**
    *   Almeno 30 misurazioni, idealmente 50-100, per l'approssimazione di Poisson.

---

**Schema Riassuntivo: Approssimazione di Poisson e Distribuzione Binomiale Negativa**

**1. Approssimazione di Poisson**

   *   **1.1. Condizioni di Applicabilità:**
        *   K > 20 o 30 (K = numero di occorrenze).
        *   Probabilità di successo dell'ordine di 1/K.
   *   **1.2. Utilità:**
        *   Valutazione della probabilità di non occorrenza di eventi (coda della distribuzione).
   *   **1.3. Derivazione:**
        *   Approssimazione della distribuzione binomiale:
            *   $$B_{T,P}(k)=\begin{pmatrix}T \\ K \end{pmatrix}p^k(1-p)^{t-k}$$
        *   Approssimazione di $p^k(1-p)^{t-k}$ con $e^{-Tp}$, ponendo $\lambda=Tp$
        *   Distribuzione di Poisson:
            *   $$p_{\lambda}(k)=\frac{\lambda^k}{k!}e^{-\lambda}$$
            *   k = numero di occorrenze
        *   Per k grande, la probabilità di k eventi è approssimativamente proporzionale a una funzione gaussiana con media t e deviazione standard √t.
   *   **1.4. Condizioni per l'approssimazione:**
        *   T (numero totale di eventi) molto grande.
        *   p (probabilità di successo) molto bassa (coda della curva di Zipf).
   *   **1.5. Parametri:**
        *   Media = Varianza = λ = cf/T
        *   λ = Tp (T grande, p piccolo)
   *   **1.6. Assunzioni:**
        *   "Intervallo fisso" implica una lunghezza di documento fissa (es. abstract di documenti di dimensioni costanti).
        *   Frequenza globale dei termini segue una legge di Zipf, indipendentemente dal documento.

**2. Limiti dell'Approssimazione di Poisson per Termini "Topic Specific"**

   *   **2.1. Definizione:**
        *   Termini legati a un determinato argomento.
   *   **2.2. Caratteristiche:**
        *   Mancanza di indipendenza (co-occorrenza significativa).
        *   Contestualità (forte legame con il contesto del documento).
        *   Occorrenza a gruppi (coerenza e condivisione di pattern).
   *   **2.3. Implicazioni:**
        *   Inadatta a modellare la probabilità di occorrenza a causa di dipendenza e contestualità.
   *   **2.4. Soluzioni Alternative:**
        *   Approssimazione Okapi BM25.
        *   Distribuzione Binomiale Negativa.

**3. Distribuzione Binomiale Negativa**

   *   **3.1. Obiettivo:**
        *   Modellare la probabilità di osservare *k* insuccessi in una sequenza di trial Bernoulliani *i.i.d.* fino a osservare *r* successi (*r* = stopping condition).
   *   **3.2. Formula:**
        *   $$NB_{\ r,p}(k)= \begin{pmatrix} k+r-1 \\ k \end{pmatrix}p^r(1-p)^k$$
   *   **3.3. Parametri:**
        *   *r* = parametro di dispersione (numero di successi).
        *   *k* = numero di failures da osservare.
        *   *p* = probabilità di successo.
   *   **3.4. Interpretazione:**
        *   Osservazione della probabilità di *k* insuccessi.
        *   Switch tra *k* e *r*: *r* diventa numero di insuccessi e *k* di successi.

---

## Schema Riassuntivo: Termini Elite e Binomiale Negativa

**1. Parametrizzazione della Binomiale Negativa**

*   **1.1. Media:**
    *   Definizione: $$\mu=\frac{rp}{1-p}$$
    *   Dove:
        *   *r* = parametro di dispersione (successi)
        *   *p* = probabilità di successo
*   **1.2. Derivazione di *p* e (1-*p*)**
    *   $$p=\frac{\mu}{\mu+r}$$
    *   $$(1-p)=\frac{r}{\mu+r}$$
*   **1.3. Formula Binomiale Negativa Parametrizzata**
    *   $$NB_{\ r,p}(k)= \begin{pmatrix} k+r-1 \\ k \end{pmatrix}\left( \frac{r}{\mu+r} \right)^r\left( \frac{\mu}{\mu+r} \right)^k$$
    *   Dove:
        *   *k+r* = numero di trial (*T*)
        *   -1: l'ultimo evento è un successo

**2. Termini Elite**

*   **2.1. Definizione:** Termine (generalmente sostantivo) che descrive un concetto/tema/topic nel corpus.
*   **2.2. Valutazione:** Binaria (elite o non elite).
*   **2.3. Dipendenza:** La "eliteness" dipende dalla rilevanza del documento al topic.
    *   Determinata dalla *term frequency* nel documento, se il documento è rilevante.

**3. Eliteness e Modellazione**

*   **3.1. Legame con la Rilevanza:** L'eliteness è legata alla rilevanza del documento al topic.
*   **3.2. Modellazione:**
    *   **3.2.1. Variabile Nascosta:** $E_i$ per il termine *i*, rappresenta l'argomento.
        *   Un termine è elite se il documento tratta il concetto denotato dal termine.
    *   **3.2.2. Eliteness Binaria:** Un termine è o non è elite.
*   **3.3. Pattern Distribuzionali:** Termini elite spesso raggruppati con frequenze simili.
*   **3.4. Problema del "Topic Drift":**
    *   Documenti possono contenere termini elite relativi a concetti diversi.
    *   Considerare tutti i termini elite come appartenenti allo stesso concetto può portare a errata classificazione.
    *   **Esempio:** Documento su football americano con termini elite di baseball.

**4. Retrieval Status Value (RSV) con Termini Elite**

*   **4.1. Formula Generale:**
    *   $$RSV^{elite}=\sum_{i\in q}c_{i^{elite}}(tf_{i})$$
    *   Dove:
        *   $$c_{i}^{elite}(tf_{i})=\log \frac{p(TF_{i}=tf_{i}|R=1)p(TF_{i}=0|R=0)}{p(TF_{i}=0|R=1)p(TF_{i}=tf_{i}|R=0)}$$
*   **4.2. Incorporazione dell'Eliteness:**
    *   $$p(TF_{i}=tf_{i}|R)=p(TF_{i}=tf_{i}|E=elite)p(E=elite|R)+p(TF_{i}=tf_{i}|E=elite)(1-p(E_{i}=elite|R))$$
*   **4.3. Obiettivo:** Personalizzare l'RSV tenendo conto dei termini elite.
*   **4.4. Metodo:** Stimare l'odds ratio considerando la term frequency e la presenza/assenza di "eliteness".
*   **4.5. Spiegazione:** La probabilità di osservare TF=tf è espressa come unione di eventi congiunti, considerando la probabilità che il termine sia elite dato R, e la probabilità che non lo sia.

---

## Schema Riassuntivo del Testo

**1. Modello a Due Poisson**

*   **1.1. Motivazione:** Superare i limiti del modello a 1-Poisson considerando termini "elite" e non.
*   **1.2. Formula:** Probabilità di osservare TF=k dato rilevanza R:
    $$p(TF_{i}=k|R)=\pi \frac{\lambda^k}{k!}e^{-\lambda}+(1-\pi) \frac{\mu^k}{k!}e^{-\mu}$$
    *   **π:** Probabilità che un documento sia "elite" per un termine.
    *   **λ:** Tasso per termini "elite".
    *   **μ:** Tasso per altri termini.
*   **1.3. Descrizione:** Combinazione lineare di due distribuzioni di Poisson per modellare il termine *i*-esimo con $TF=k_i$.
*   **1.4. Complessità:** La stima dei parametri è complessa.

**2. Modello di Poisson (Semplificato)**

*   **2.1. Proprietà:**
    *   Aumenta monotonicamente con $tf_i$.
    *   Asintoticamente si avvicina a un valore massimo.
    *   Il limite asintotico rappresenta il peso della caratteristica di "eliteness".
*   **2.2. Approssimazione:** Possibile con una curva parametrica che mantiene le proprietà qualitative.

**3. Modello di Costo per Termini "Elite"**

*   **3.1. Definizione:** Costo $c_i^{elite}$ per il termine *i*-esimo in funzione di $tf_i$.
    *   $c_i^{elite}(0) = 0$
    *   $c_i^{elite}(tf_i)$ cresce monotonicamente, saturando per alti valori di λ.
*   **3.2. Stima dei Parametri:** Preferibile tramite funzione con caratteristiche qualitative simili.

**4. Approssimazione della Poisson**

*   **4.1. Funzione:**
    $$\frac{tf}{k_1 + tf}$$
*   **4.2. Effetti di $k_1$:**
    *   Alti $k_1$: Incrementi in $tf_i$ contribuiscono significativamente al punteggio.
    *   Bassi $k_1$: Contributi diminuiscono rapidamente.
    *   Alti $k_1$: Peggiore approssimazione.

**5. Prime Versioni di BM25**

*   **5.1. Versione 1: Funzione di Saturazione**
    *   **Formula:**
        $$c_{i}^{BM25v_{1}}(tf_{i})=c_{i}^{BIM} \frac{tf_{i}}{k_{1}+tf_{i}}$$
    *   **Descrizione:** Utilizza la funzione di saturazione per calcolare il costo del termine i-esimo.
*   **5.2. Versione 2: Semplificazione BIM a IDF**
    *   **Formula:**
        $$c_{i}^{BM25v_{2}}(tf_{i})=\log \frac{N}{df_{i}}\times \frac{(k_{1}+1)tf_{i}}{k_{1}+tf_{i}}$$
    *   **Variabili:**
        *   $N$: Numero totale di documenti.
        *   $df_i$: Numero di documenti contenenti il termine i-esimo.
    *   **Descrizione:** Semplifica BIM usando solo IDF.

**6. Estensioni del Modello BM25**

*   **6.1. Prima Estensione: Funzione di Saturazione**
    *   **Descrizione:** Introduce un fattore di smoothing basato sulla funzione di saturazione al costo $c_i$ del modello BM25.
*   **6.2. Seconda Estensione: Stima di $r_i$ e Utilizzo di $df$**
    *   **Descrizione:** Utilizza solo la stima di $r_i$ e la $df$, senza il costo $c_i$ completo.

**7. Normalizzazione della Lunghezza del Documento**

*   **7.1. Motivazione:** Migliorare BM25 considerando la variabilità della lunghezza dei documenti.
*   **7.2. Effetto:** La lunghezza influenza il valore di $tf_i$.

---

## Schema Riassuntivo:

**1. Lunghezza del Documento e Normalizzazione**

*   **1.1 Lunghezza del Documento (dl):**
    *   Somma delle term frequency (tf) per tutti i termini nel documento.
    *   Formula:  $$dl=\sum_{i\in V}tf_{i}$$
*   **1.2 Lunghezza Media dei Documenti (avdl):**
    *   Lunghezza media dei documenti nella collezione.
*   **1.3 Motivi della Variabilità della Lunghezza:**
    *   **Verbosità:**  $tf_i$ osservato potrebbe essere troppo alto.
    *   **Ambito Più Ampio:** $tf_i$ osservato potrebbe essere corretto.
*   **1.4 Normalizzazione della Lunghezza:**
    *   **Componente di Normalizzazione (B):** $$B=\left( (1-b)+b \frac{dl}{avdl} \right), 0\leq b\leq1$$
    *   **b = 1:** Normalizzazione completa della lunghezza.
    *   **b = 0:** Nessuna normalizzazione della lunghezza.

**2. Okapi BM25**

*   **2.1 Panoramica:**
    *   Estensione del modello BIM (Best Match) che considera la lunghezza dei documenti.
    *   Normalizza la term frequency (tf) in base alla lunghezza.
*   **2.2 Normalizzazione della Term Frequency:**
    *   Formula: $$tf_i' = \frac{tf_i}{B}$$
*   **2.3 Formula Completa del Punteggio BM25:**
    *   Formula: $$c_i^{BM25}(tf_i) = \log \frac{N}{df_i} \times \frac{(k_1+1)tf_i'}{k_1+tf_i'} = \log \frac{N}{df_i} \times \frac{(k_1+1)tf_i}{k_1((1-b)+b\frac{dl}{avdl})+tf_i}$$
    *   **N:** Numero totale di documenti.
    *   **df<sub>i</sub>:** Numero di documenti contenenti il termine *i*.
    *   **k<sub>1</sub>, b:** Parametri.
    *   **dl:** Lunghezza del documento.
    *   **avdl:** Lunghezza media dei documenti.
*   **2.4 Funzione di Ranking BM25:**
    *   Formula: $$RSV^{BM25} = \sum_{i \in q} c_i^{BM25}(tf_i)$$
    *   **q:** Insieme dei termini nella query.
*   **2.5 Miglioramento rispetto a BIM:**
    *   Normalizza la term frequency in base alla lunghezza dei documenti.

**3. Parametri del Modello BM25**

*   **3.1 Formula Completa (ripetuta per chiarezza):**
    *   $$RSV^{BM25} = \sum_{i \in q} \log \frac{N}{df_i} \cdot \frac{(k_1+1)tf_i}{k_1((1-b)+b\frac{dl}{avdl})+tf_i}$$
*   **3.2 Parametri Principali:**
    *   **k1:** Gestisce la pendenza della funzione di saturazione (scaling della tf).
        *   **k1 = 0:** Modello binario.
        *   **k1 grande:** Term frequency grezza.
    *   **b:** Controlla la normalizzazione della lunghezza.
        *   **b = 0:** Nessuna normalizzazione della lunghezza.
        *   **b = 1:** Frequenza relativa (scala completamente in base alla lunghezza).
    *   **Valori Tipici:** k1 ≈ 1.2-2, b ≈ 0.75.
*   **3.3 Possibili Estensioni:**
    *   Ponderazione dei termini di query.
    *   Feedback di rilevanza (pseudo).

**4. Esempio di Applicazione**

*   **4.1 Query:** "machine learning"
*   **4.2 Documenti:**
    *   **doc1:** learning 1024; machine 1
    *   **doc2:** learning 16; machine 8
*   **4.3 Calcoli:**
    *   **tf-idf:** $log_2 (tf) \cdot \ log_{2} \left( \frac{N}{df} \right)$
        *   **doc1:** $11 * 7 + 1 * 10 = 87$
        *   **doc2:** $5 \cdot 7 + 4 \cdot 10 = 75$
    *   **BM25:** $k_{1} = 2$
        *   **doc1:** $7 \cdot 3 + 10 \cdot 1 = 31$
        *   **doc2:** $7 \cdot 2.67 + 10 \cdot 2.4 = 42.7$

**5. Ranking con Zone**

*   **5.1 Definizione di Zona:**
    *   Sezione specifica di un documento (e.g., titolo, abstract, introduzione, conclusioni, keyword).

---

**Schema Riassuntivo BM25F e Classifica con Caratteristiche Non Testuali**

1.  **Introduzione al BM25F e Approcci Iniziali**

    *   Approccio Semplice: Applicare BM25 a ogni zona e combinare i punteggi linearmente.
        *   Limitazione: Assume indipendenza e proprietà di eliteness diverse tra le zone.
    *   Approccio Alternativo: Eliteness come proprietà condivisa termine/documento, ma dipendenza dalla zona.
        *   Esempio: Uso più denso di parole chiave nel titolo.
    *   Conseguenza: Combinare prima le prove tra le zone per ogni termine, poi tra i termini.

2.  **Calcolo di Varianti Pesate di Frequenza dei Termini e Lunghezza del Documento**

    *   Formule:
        *   Frequenza dei termini totale pesata: $\tilde{t} f_{i} = \sum_{z=1}^{Z} v_{z} t f_{z i}$
        *   Lunghezza del documento pesata: $\tilde{dl} = \sum_{z=1}^{Z} v_{z} l e n_{z}$
        *   Lunghezza media del documento pesata: $avdl = \frac{\text{average } d\tilde{l}}{\text{across all docs}}$
    *   Definizioni:
        *   $v_z$: Peso della zona.
        *   $tf_{zi}$: Frequenza del termine nella zona $z$.
        *   $len_z$: Lunghezza della zona $z$.
        *   $Z$: Numero di zone.

3.  **Metodo per il Calcolo delle Varianti Pesate**

    *   Fasi:
        *   Calcolo della TF per zona: Calcolo della frequenza dei termini separatamente per ogni zona.
        *   Normalizzazione per zona: Normalizzazione della TF in base alla lunghezza della zona.
        *   Peso della zona: Assegnazione di un peso ($v_z$) a ciascuna zona (parametro predefinito).

4.  **Simple BM25F con Zone**

    *   Interpretazione: Zona *z* "replicata" *y* volte.
    *   Formula RSV: $RSV^{SimpleBM25F} = \sum_{i \in q} \log \frac{N}{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b \frac{dl}{avdl}) + tf_i}$
    *   Possibilità di usare parametri specifici per zona (k, b, IDF).

5.  **Normalizzazione della Lunghezza Specifica per Zona**

    *   Utilità: Migliora empiricamente le prestazioni.
    *   Formula della frequenza del termine modificata: $\tilde{tf}_i = \sum_{z=1}^Z v_z \frac{f_{z i}}{B_z}$
        *   Dove: $B_z = \left( (1-b_z) + b_z \frac{\text{len}_z}{\text{avlen}_z} \right), \quad 0 \leq b_z \leq 1$
        *   `len_z`: Lunghezza della zona z.
        *   `avlen_z`: Lunghezza media delle zone z.
    *   Formula RSV con normalizzazione specifica per zona: $\text{RSV}^{BM25F} = \sum_{i \in q} \log \frac{N }{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_{1}+tf_{i}}$
    *   Differenza principale: Utilizzo di $B_z$ per la normalizzazione della lunghezza specifica per zona.

6.  **Classifica con Caratteristiche Non Testuali**

    *   Assunzioni:
        *   Indipendenza usuale: Le caratteristiche non testuali sono indipendenti tra loro e dalle caratteristiche testuali.
            *   Implicazione: $\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$
        *   Informazioni di rilevanza indipendenti dalla query: Vera per PageRank, età, tipo, ecc.
            *   Implicazione: Mantenimento delle caratteristiche non testuali nella derivazione in stile BIM.

---

**Schema Riassuntivo del Ranking Score Value (RSV)**

**1. Definizione del Ranking Score Value (RSV)**

   *  Formula generale:  $$RSV=\sum_{i\in q}c_{i}(tf_{i})+\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$$
   *  Componenti:
      *  $\sum_{i\in q}c_{i}(tf_{i})$:  Sommatoria dei termini di query pesati per la loro frequenza (tf).
      *  $\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$:  Sommatoria di funzioni di features ($V_j$) pesate da un parametro ($\lambda$).

**2. Funzione di Feature V_j**

   *  Definizione: $V_{f}(f_{j})=\log\frac{p(F_{j}=f_{j}|R=1)}{p(F_{j}=f_{j}|R=0)}$
   *  Significato: Logaritmo del rapporto tra la probabilità della feature $f_j$ dato che il documento è rilevante (R=1) e la probabilità della feature $f_j$ dato che il documento non è rilevante (R=0).

**3. Parametro di Ridimensionamento λ**

   *  Ruolo: $\lambda$ è un parametro libero aggiunto per compensare le approssimazioni nel calcolo dell'RSV.

**4. Importanza della Selezione di V_j**

   *  Considerazioni: La scelta appropriata di $V_j$ è cruciale e dipende dalla feature $f_j$ considerata.
   *  Esempio: La performance di $Rsv_{bm25} + log(\text{pagerank})$ suggerisce l'importanza di una selezione oculata di $V_j$.

---
