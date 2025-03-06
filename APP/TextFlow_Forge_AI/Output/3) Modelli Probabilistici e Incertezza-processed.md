
**Schema Riassuntivo**

**I. Topic Modeling**
    *   Definizione: Modellazione e rappresentazione di dati testuali.
    *   Caratteristica Principale:
        *   Topic: Distribuzione di probabilità sullo spazio dei termini.
        *   Documento: Miscela di distribuzioni di probabilità dei topic.

**II. Basi di Conoscenza Lessicali**
    *   Ruolo: Elemento importante del Knowledge Management.
    *   Esempio: WordNet (risorsa lessicale con informazioni semantiche).
    *   Applicazione: Testare le capacità di comprensione dei Language Models.

**III. Ritorno al Vector Space Model**
    *   Contesto: Information Retrieval.
    *   Focus: Modello, proprietà e generalizzazione probabilistica della funzione TF.
    *   Premessa: Necessità di un meccanismo di ranking.

**IV. Incertezza nella Ricerca di Informazioni**
    *   Problema: Imprecisione nella traduzione dell'esigenza informativa in query.
    *   Soluzione: Introduzione di una nozione di incertezza nei sistemi di recupero.

**V. Gestione dell'Incertezza**
    *   Approccio: Modelli probabilistici (analogia con "uncertainty data mining").
    *   Esempio: Incertezza nelle misurazioni sensoriali (es. sensori ottici vs. pneumatici).

**VI. Incertezza nei Dati (Analisi Dati)**
    *   Problema: Valori numerici affetti da incertezza.
    *   Soluzione: Associare a ciascun valore una distribuzione di probabilità univariata (es. gaussiana con media pari al valore).
    *   Trasformazione: Da vettori multidimensionali a insiemi di distribuzioni di probabilità.

---

## Schema Riassuntivo: Gestione dell'Incertezza e Modelli Probabilistici per il Ranking di Documenti

**1. Introduzione all'Incertezza e Strumenti:**
    *   L'incertezza è importante in data mining, knowledge management e information retrieval.
    *   Permette sistemi di retrieval più raffinati e l'identificazione del miglior result set.
    *   Strumenti: Divergenza di Shannon-Jensen (misura di distanza tra distribuzioni).

**2. Modelli di Retrieval Probabilistici:**
    *   **2.1 Probability Ranking Principle:**
        *   Classifica i documenti in base alla probabilità di pertinenza.
    *   **2.2 Binary Independence Model:**
        *   Affinità con la teoria bayesiana e la probabilità a priori.
        *   Assume che i termini siano indipendenti l'uno dall'altro.
    *   **2.3 Modello Okapi BM25:**
        *   Versione probabilistica della TF-IDF.
        *   Tiene conto della frequenza dei termini e della lunghezza del documento.
    *   **2.4 Reti Bayesiane:**
        *   Approccio generale per modellare le dipendenze tra i termini.
        *   L'inferenza bayesiana è fondamentale per la gestione dell'incertezza.

**3. Evoluzione e Importanza dei Modelli Probabilistici:**
    *   Considerati "matematicamente eleganti" negli anni 2000, ma con applicazione pratica limitata.
    *   Oggi, grazie a tecnologie più efficienti, sono uno strumento fondamentale per il ranking.

**4. Principio di Ranking Probabilistico:**
    *   **4.1 Obiettivo:** Determinare il miglior result set per una query.
    *   **4.2 Approccio:** Misurare la probabilità di rilevanza di un documento rispetto a una query.
    *   **4.3 Formalizzazione:** Problema di classificazione (Rilevante vs. Non Rilevante).

**5. Formalizzazione Matematica:**
    *   **5.1 Notazione:**
        *   X: Rappresentazione vettoriale del documento.
        *   R: Classe "rilevante".
        *   R=1: "pertinente", R=0: "non pertinente".
        *   P(R=1|X): Probabilità che il documento X appartenga alla classe R.
    *   **5.2 Formula di Bayes:**
        $$P(R=1|X) = \frac{{P(X|R=1) \cdot P(R=1)}}{P(X)}$$
        *   P(X|R=1): Likelihood (probabilità di osservare X dato che è rilevante).
        *   P(R=1): Probabilità a priori che un documento sia rilevante.
        *   P(X): Probabilità di osservare il documento X.
    *   **5.3 Osservazioni:**
        *   P(X) è costante e non influenza il ranking.
        *   Il ranking dipende da P(X|R=1) e P(R=1).

**6. Classificazione, Bias e Stima delle Probabilità:**
    *   **6.1 Problema del Bias:** L'indipendenza condizionata può essere problematica nei dati testuali.
    *   **6.2 Formula di Price:** Il denominatore rimane costante indipendentemente dalla classe o dalla distanza.
    *   **6.3 Likelihood:**
        *   Termine computazionalmente complesso nel DICE.
        *   Stima tramite Maximum Likelihood Estimate (frequenza relativa).
        *   Assunzione di indipendenza condizionata tra le dimensioni, data la classe J.
    *   **6.4 Probabilità a Priori:**
        *   Stimate osservando un training set.

---

**Schema Riassuntivo: Ranking Probabilistico e Binary Independence Model**

**1. Problema del Bias e Assunzione di Indipendenza**

   *   **1.1. Bias nella Classificazione:**
        *   Importanza del problema del bias in classificazione (testo e altri dati).
   *   **1.2. Validità dell'Assunzione di Indipendenza:**
        *   Valida se il pre-processing riduce la dimensionalità valorizzando l'indipendenza.
        *   Difficile da accettare nei dati testuali.
   *   **1.3. Dipendenza nei Dati Testuali:**
        *   Dimensioni (parole) spesso dello stesso tipo e dipendenti.
        *   Esempio: articoli dipendenti dai sostantivi, grammatica influenza le relazioni.

**2. Probability Ranking Principle (PRP)**

   *   **2.1. Obiettivo:**
        *   Ordinare i documenti in base alla probabilità di rilevanza.
        *   Allineato con la regola di decisione ottimale dell'inferenza bayesiana.
   *   **2.2. Condizione di Rilevanza:**
        *   Documento D rilevante rispetto a query Q se:
            $$P(R=1|D,Q) > P(R=0|D,Q)$$
            *   R=1: documento rilevante.
            *   R=0: documento non rilevante.
   *   **2.3. Probabilità a Priori e Condizionate:**
        *   p(R=1|x): probabilità che un documento x sia pertinente.
        *   p(R=1), p(R=0): probabilità a priori di recuperare un documento pertinente/non pertinente.
        *   p(x|R=1), p(x|R=0): probabilità che un documento pertinente/non pertinente sia x.
   *   **2.4. Minimizzazione del Rischio Bayesiano:**
        *   Adottare il ranking probabilistico minimizza il rischio bayesiano (loss 1-0/0-1 = error rate).

**3. Binary Independence Model (BIM)**

   *   **3.1. Rappresentazione dei Documenti:**
        *   Vettori binari di incidenza dei termini: **x**.
        *   **x<sub>i</sub> = 1** se il termine i-esimo è presente, **x<sub>i</sub> = 0** altrimenti.
   *   **3.2. Assunzione di Indipendenza:**
        *   Indipendenza tra i termini.
   *   **3.3. Calcolo delle Probabilità:**
        *   Calcola $P(R=1|D,Q)$ e $P(R=0|D,Q)$ per l'ordinamento.
        *   Obiettivo: solo il ranking.

**4. Applicazione del Teorema di Bayes per il Ranking**

   *   **4.1. Focus sul Ranking:**
        *   Interesse nel ranking dei documenti, non nel punteggio assoluto.
   *   **4.2. Probabilità Condizionata:**
        *   Utilizzo di **O(R|QX)**: probabilità di osservare la classe R dato il documento X e la query Q.
        $$O(R|Q\vec{X}) = \frac{P(R=1|Q\vec{X})}{P(R=0|Q\vec{X})}$$
        *   Applicazione del teorema di Bayes per calcolare i termini.

**5. Odds e Odds Ratio (OR)**

   *   **5.1. Odds:**
        *   Misura di probabilità: rapporto tra la probabilità di un evento e il suo complemento.
        *   Aiutano a stabilire un ordinamento tra i documenti.
   *   **5.2. Odds Ratio (OR):**
        *   Misura di associazione relativa tra due variabili binarie.
        *   Calcolo:
            $$OR = \frac{n_{11} \cdot n_{00}}{n_{10} \cdot n_{01}}$$
            *   Dove n<sub>ij</sub> sono le frequenze nella tabella di contingenza 2x2.

---

**Schema Riassuntivo su Odds Ratio e Rischio Relativo**

**1. Applicazioni degli Odds Ratio**

*   **1.1 Epidemiologia:** Valutazione del rischio di eventi in relazione a fattori di rischio in diverse popolazioni.
*   **1.2 Ricerca Sociale:** Studio dell'associazione tra variabili sociali.
*   **1.3 Marketing:** Analisi dell'efficacia di campagne pubblicitarie.

**2. Rischio Relativo (RR)**

*   **2.1 Definizione:** Misura la probabilità di un evento in presenza di una variabile rispetto all'assenza.
*   **2.2 Formula:**
    *   RR = (Probabilità di osservare X in presenza di Y) / (Probabilità di osservare X in assenza di Y)
*   **2.3 Interpretazione:**
    *   RR = 1: Nessuna differenza nel rischio.
    *   RR < 1: L'evento Y è meno probabile in presenza di X.
    *   RR > 1: L'evento Y è più probabile in presenza di X.

**3. Odds Ratio (OR)**

*   **3.1 Definizione:** Misura di associazione che confronta le probabilità di un evento in due gruppi diversi.
*   **3.2 Formula:**
    *   OR = (Odds di Y in presenza di X) / (Odds di Y in assenza di X)
*   **3.3 Interpretazione:**
    *   OR = 1: Nessuna differenza nelle odds tra i due gruppi.
    *   OR < 1: Le odds di Y sono più basse nel gruppo con X.
    *   OR > 1: Le odds di Y sono più alte nel gruppo con X.

**4. Relazione tra Rischio Relativo e Odds Ratio**

*   **4.1 Correlazione:** Misure correlate, OR può approssimare RR quando la prevalenza dell'evento è bassa.

**5. Confronto tra OR e RR**

*   **5.1 Proprietà dell'OR:**
    *   Non simmetrica: Scambiando X e Y, il valore dell'OR cambia.
    *   Robusta: Stimabile anche senza incidenze.
    *   Approssimazione del RR: Per eventi rari, OR ≈ RR.
*   **5.2 Scelta tra OR e RR:**
    *   Studi retrospettivi: OR preferibile (stimabile senza incidenze).
    *   Studi prospettivi: RR preferibile (misura diretta del rischio).
    *   Eventi rari: OR approssima bene RR.

**6. Esempio di Calcolo dell'OR**

*   **6.1 Scenario:** Studio epidemiologico su fumo e rischio di malattia.
    *   Gruppo esposto: Fumatori
    *   Gruppo non esposto: Non fumatori
    *   Evento: Sviluppo della malattia
    *   Incidenza: 30% (fumatori), 10% (non fumatori)
*   **6.2 Calcolo:**
    *   Odds di malattia (fumatori): 30/70 = 0.43
    *   Odds di malattia (non fumatori): 10/90 = 0.11
    *   OR: 0.43 / 0.11 = 3.91

---

## Schema Riassuntivo

**1. Odd Ratio (OR) e Applicazioni Epidemiologiche**

*   **1.1 Interpretazione dell'OR:**
    *   L'OR indica il rischio relativo di sviluppare una malattia tra gruppi esposti e non esposti.
    *   Esempio: OR di 3.91 significa che i fumatori hanno un rischio 3.91 volte maggiore di sviluppare la malattia rispetto ai non fumatori.
*   **1.2 Applicazioni Epidemiologiche:**
    *   Utile per studiare l'impatto di fattori di rischio su eventi rari.
    *   Esempio: Valutare l'associazione tra esposizione ambientale e rischio di malattia.
*   **1.3 Nota sull'Approssimazione OR al RR:**
    *   L'approssimazione è valida solo con l'assunzione di sparsità (rarità degli eventi).
    *   In caso di eventi frequenti, l'OR può sovrastimare il RR.

**2. OTS-I (Odd of Term Significance - I) e Likelihood Ratio**

*   **2.1 Definizione di OTS-I:**
    *   Equivalente a TOS (Odd of Term Significance) rispetto alla rilevanza dell'ipotesi.
*   **2.2 Likelihood Ratio:**
    *   Rapporto tra la probabilità di osservare un dato Q data l'ipotesi di rilevanza e la probabilità di osservare Q data l'ipotesi di non rilevanza.
    *   Formula: $\frac{PR(Q | R=1)}{PR (Q | R=0)}$
        *   PR (Q | R=1): Probabilità di osservare Q dato che l'ipotesi di rilevanza è vera (R=1).
        *   PR (Q | R=0): Probabilità di osservare Q dato che l'ipotesi di rilevanza è falsa (R=0).
    *   La likelihood è costante per ogni termine.
*   **2.3 Ipotesi di Indipendenza:**
    *   Sotto l'ipotesi di indipendenza tra i termini, il rapporto di likelihood si trasforma in una produttoria.
    *   Formula: $OTS-i(Q)= \frac{p(\vec{x} \mid R = 1, q)}{p(\vec{x} \mid R = 0, q)} = \prod_{i = 1}^{n} \frac{p(x_i \mid R = 1, q)}{p(x_i \mid R = 0, q)}$
        *   X<sub>i</sub>: Presenza o assenza del termine i-esimo nel documento.
*   **2.4 Stima della Produttoria di Likelihood:**
    *   Divisa in due parti:
        *   Termini presenti nel documento: $\prod_{x_{i}=1} \frac{p\left(x_{i}=1 \mid R=1,q\right)}{p\left(x_{i}=1 \mid R=0,q\right)}$
        *   Termini assenti nel documento: $\prod_{x_{i}=0} \frac{p\left(x_{i}=0 \mid R=1,q\right)}{p\left(x_{i}=0 \mid R=0,q\right)}$
*   **2.5 Notazione Semplificata:**
    *   P<sub>i</sub>: Likelihood rispetto alla presenza del termine i-esimo, data l'ipotesi di rilevanza.
    *   R<sub>i</sub>: Likelihood rispetto alla presenza del termine i-esimo, data l'ipotesi di non rilevanza.
    *   Produttoria riscritta: $\prod_{i=1}^{n} \left( \frac{P_i}{R_i} \right)$

**3. Probabilità di Osservazione dei Termini in un Documento**

*   **3.1 Definizioni:**
    *   R<sub>i</sub>: Probabilità di osservare il termine i-esimo nel documento, assumendo la rilevanza del documento.
    *   t<sub>i</sub>: Probabilità di osservare il termine i-esimo nel documento, assumendo la novità del termine.
*   **3.2 Tabella di Contingenza:**

    | | Rilevanza (r=1) | Non Rilevanza (r=0) |
    | ------------------ | --------------- | ------------------- |
    | Presenza $(x_i=1)$ | $p_i$ | $r_i$ |
    | Assenza $(x_i=0)$ | $(1-p_i)$ | $(1-r_i)$ |
*   **3.3 Interpretazione della Tabella:**
    *   $p_i$: Probabilità di osservare il termine i-esimo nel documento, dato che il documento è rilevante.
    *   $r_i$: Probabilità di osservare il termine i-esimo nel documento, dato che il documento non è rilevante.
    *   $(1-p_i)$: Probabilità di non osservare il termine i-esimo nel documento, dato che il documento è rilevante.
    *   $(1-r_i)$: Probabilità di non osservare il termine i-esimo nel documento, dato che il documento non è rilevante.
*   **3.4 Odds:**
    *   Rapporto tra la probabilità di un evento e la probabilità del suo complemento.
    *   Calcolati per termini matching e non matching.
        *   Matching: Termine presente sia nella query che nel documento (x<sub>i</sub> = 1, q<sub>i</sub> = 1).
        *   Non Matching: Termine presente nella query ma non nel documento (x<sub>i</sub> = 0, q<sub>i</sub> = 1).

---

## Schema Riassuntivo

**I. Assunzioni del Modello**
    * A. Ignora l'assenza di termini nella query (q<sub>i</sub> = 0).
    * B. Non valuta la probabilità di occorrenza per termini non presenti nella query.

**II. Derivazione della Formula Finale**
    * A. Formula di partenza:
        $$
        \begin{aligned}
        O(R \mid q, \tilde{x}) = O(R \mid q) \cdot \prod_{x_{i}=1} \frac{p(x_{i}=1 \mid R=1,q)}{p(x_{i}=1 \mid R=0,q)} \cdot \prod_{x_{i}=0} \frac{p(x_{i}=0 \mid R=1,q)}{p(x_{i}=0 \mid R=0,q)}
        \end{aligned}
        $$
    * B. Distinzione delle Produttorie:
        * 1. Produttoria dei termini "match" (x<sub>i</sub> = 1): Rilevanza del documento per la query.
        * 2. Produttoria dei termini "non-match" (x<sub>i</sub> = 0): Non rilevanza del documento per la query.
    * C. Introduzione e Scomposizione della Produttoria Fittizia:
        $$
        \begin{aligned}
        O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i = 1 \\ q_i = 1}} \frac{p_i}{r_i} \cdot \prod_{\substack{x_i = 0 \\ q_i = 1}} \frac{1 - p_i}{1 - r_i}
        \end{aligned}
        $$
    * D. Formazione dell'Odds Ratio:
        * 1. P / (1 - P): Probabilità di osservare il termine dato che il documento è rilevante, diviso la probabilità di osservare il termine dato che il documento non è rilevante.
        * 2. R / (1 - R): Probabilità di osservare il termine dato che il documento non è rilevante, diviso la probabilità di non osservare il termine dato che il documento non è rilevante.
    * E. Formula Finale:
        $$
        \begin{aligned}
        O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i=q_i=1}} \frac{p_i(1-r_i)}{r_i(1-p_i)} \cdot \prod_{\substack{q_i=1 }} \frac{1-p_i}{1-r_i}
        \end{aligned}
        $$
    * F. Componenti della Formula Finale:
        * 1. Odds ratio di R dato il primo termine: Probabilità di rilevanza del documento dato il primo termine.
        * 2. Produttoria sui termini indipendenti: Influenza dei termini che non dipendono dalla rilevanza.

**III. Retrieval Status Value (RSV)**
    * A. Definizione: Punteggio per determinare la rilevanza di un documento rispetto a una query.
    * B. Calcolo dell'RSV:
        * 1. Produttoria degli odds ratio: Calcolo dell'odds ratio per ogni termine della query presente nel documento.
        * 2. Sommatoria dei logaritmi: Somma dei logaritmi degli odds ratio.
        * 3. RSV: Risultato della sommatoria dei logaritmi.
    * C. Esempio:
        * 1. Query: "X" e "Q".
        * 2. Documento: Contiene "X" e "Q".
        * 3. Odds ratio per "X": 1
        * 4. Odds ratio per "Q": 1
        * 5. RSV: log(1) + log(1) = 0
    * D. Interpretazione:
        * 1. RSV alto: Maggiore rilevanza.
        * 2. RSV basso: Minore rilevanza.

---

## Schema Riassuntivo: Analisi e Calcolo dell'Oz Ratio in NLP

**1. Stima delle Probabilità e Analisi dei Dati di Addestramento**

*   **1.1 Stima delle Probabilità:**
    *   Necessaria per calcolare gli odds ratio.
    *   Utilizzo di tecniche come la smoothing di Laplace.
*   **1.2 Analisi dei Dati di Addestramento:**
    *   Campo di ricerca attuale e complesso.
    *   Sfida principale: "inversione del modello" (risalire ai dati di addestramento).
    *   Importanza della trasparenza sui dati di addestramento (dettaglio oltre "web, prosa e poesia").
    *   Necessità di addestrare modelli specifici per riconoscere il contesto dei dati di addestramento.

**2. Simboli e Definizioni**

*   **2.1 RSV:** Logaritmo del produttore di all special.
    *   Formula: $RSV = \log \prod_{x_i=q_i=1} \frac{p_i(1-r_i)}{r_i(1-p_i)} = \sum_{x_i=q_i=1} \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$
    *   Formula semplificata: $RSV = \sum_{x_i=q_i=1} c_i; \quad c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$
*   **2.2 i:** Numero di termini (dimensione del vocabolario).
*   **2.3 n s N S:** Dipendenza a livello di rappresentazione di testi e binariet.

**3. Calcolo dell'Oz Ratio**

*   **3.1 Definizioni:**
    *   **N:** Numero totale di documenti.
    *   **S:** Numero totale di documenti rilevanti.
    *   **n:** Numero di documenti in cui un termine specifico è presente.
    *   **s:** Numero di documenti rilevanti che contengono il termine specifico.
*   **3.2 Tabella di Contingenza:**

    | Documents | Relevant | Non-Relevant | Total |
    | --------- | -------- | ------------ | ----- |
    | $x_i=1$ | s | n-s | n |
    | $x_i=0$ | S-s | N-n-S+s | N-n |
    | **Total** | **S** | **N-S** | **N** |
*   **3.3 Calcolo delle Probabilità:**
    *   $p_i = \frac{s}{S}$ (Probabilità che il termine sia presente dato che il documento è rilevante).
    *   $r_i = \frac{(n-s)}{(N-S)}$ (Probabilità che il termine sia presente dato che il documento non è rilevante).
*   **3.4 Oz Ratio:**
    *   Misura la forza dell'associazione tra un termine e la rilevanza di un documento.
    *   Formula: $c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$
*   **3.5 Espressione in termini di Count:**
    *   $c_i = K(N,n,S,s) = \log \frac{s/(S-s)}{(n-s)/(N-n-S+s)}$

**4. Approssimazione di r_i**

*   **4.1 Ipotesi:**
    *   L'insieme dei documenti non rilevanti può essere approssimato dall'intera collezione.
*   **4.2 Conseguenze:**
    *   $r_i$ è approssimabile a N piccolo / N grande (frazione dei documenti nella collezione totale che contengono il termine di ricerca).

---

## Schema Riassuntivo: Oz di Competenza, Probabilità e Smoothing

**1. Oz di Competenza e IDF**

*   **1.1. Definizione:** Misura della rarità di un termine in un corpus.
*   **1.2. Formula:** $\log \frac{1-r_i}{r_i} = \log \frac{N-n-S+s}{n-s} \approx \log \frac{N-n}{n} \approx \log \frac{N}{n} = IDF$
*   **1.3. Approssimazione:** $N - n \approx N$ quando N è molto grande rispetto a n.
*   **1.4. Validità:** Più accurata in sistemi aperti (es. web) rispetto a corpus tematici.

**2. Probabilità e Smoothing**

*   **2.1. Incertezza:** L'informazione è trattata in modo probabilistico.
*   **2.2. Norma Inversa del Termine:** $\log(\frac{N}{n})$ rappresenta la norma inversa del termine.
*   **2.3. TFPF (Term Frequency-Probability Factor):** Misura robusta coerente con la legge di Zipf.

**3. Smoothing**

*   **3.1. Scopo:** Evitare risposte troppo generose del modello.
*   **3.2. Effetti:**
    *   **3.2.1.** Penalizza eventi frequenti.
    *   **3.2.2.** Fa emergere eventi rari.
*   **3.3. Applicazione:** Evitare che il modello si affidi eccessivamente ai dati osservati (inferenza bayesiana).
*   **3.4. Formula:** $ P_{i}^{(h+1)} = \frac{\left|V_{i}\right|+\kappa p_{i}^{(h)}}{\left|V\right|+K} $
    *   **3.4.1.** $P_i^{(h+1)}$: Probabilità di osservare l'evento $i$ all'iterazione $h+1$.
    *   **3.4.2.** $|V_i|$: Cardinalità dell'insieme $B$ contenente l'evento $i$.
    *   **3.4.3.** $|V|$: Cardinalità dell'insieme $B$.
    *   **3.4.4.** $K$: Parametro di smoothing.
    *   **3.4.5.** $p_i^{(h)}$: Probabilità di osservare l'evento $i$ all'iterazione $h$.
*   **3.5. Fattori Nascosti:** Lo smoothing introduce correlazioni/dipendenze tra i termini.

**4. Stima di $p_{i}$ e Relevance Feedback**

*   **4.1. Introduzione:** Focus sulla stima di $p_{i}$ (likelihood di incidenza del termine nei documenti rilevanti).
*   **4.2. Approcci per la Stima di $p_{i}$:**
    *   **4.2.1.** Stima da un sottoinsieme di documenti etichettati (con feedback).
    *   **4.2.2.** Utilizzo di teorie e leggi per approssimare $p_{i}$.
    *   **4.2.3.** Probabilistic Relevance Feedback.

**5. Probabilistic Relevance Feedback**

*   **5.1. Definizione:** Applicazione del feedback dell'utente per migliorare la ricerca di informazioni.
*   **5.2. Meccanismo:** L'utente indica documenti rilevanti/non rilevanti.
*   **5.3. Obiettivo:** Migliorare la risposta alla query (espansione della query).

---

## Schema Riassuntivo: Espansione della Query e Relevance Feedback

**1. Espansione della Query**

*   Aggiunta di termini a una query originale.
*   I termini sono identificati nei documenti rilevanti.
*   Processo iterativo:
    *   Identificazione documenti rilevanti.
    *   Estrazione keyword dai documenti rilevanti.
    *   Aggiunta keyword alla query.

**2. Relevance Feedback**

*   Processo iterativo per migliorare i risultati di ricerca.
*   L'utente fornisce feedback sulla rilevanza dei risultati.
*   **Fine ultimo:** Adattare il sistema alle preferenze dell'utente.

**3. Probabilistic Relevance Feedback**

*   Utilizza un modello probabilistico per stimare la probabilità di rilevanza.
*   Basato sulla presenza/assenza di termini e sulla loro rilevanza.
*   **Processo:**
    *   Stima iniziale: P(R|D), P(I|R).
    *   Identificazione documenti rilevanti (utente).
    *   Raffinamento della stima (sistema).
    *   Ricerca iterativa con nuove stime.
*   Concetto chiave: **Inverso documento** (probabilità che un termine sia presente in un documento rilevante).
*   **Esempio:**
    *   5 documenti rilevanti su 10: frazione = 0.5.
    *   Termine "informatica" in 3/5 documenti rilevanti: probabilità = 0.6.

**4. Approssimazione della Probabilità di Rilevanza**

*   Approssimazione della probabilità di rilevanza di un termine:

    $$P(rilevanza | termine) ≈ \frac{|V|}{|I|} $$

    *   Dove:
        *   |V| = Cardinalità dell'insieme dei documenti rilevanti.
        *   |I| = Cardinalità dell'insieme di tutti i documenti.
*   **Problema:** Calcolo costoso se |V| è grande.
*   **Soluzione:**
    *   Meccanismo iterativo per raffinare l'insieme dei documenti candidati.
    *   Smoothing per eventi non osservati.
*   **Smoothing:**
    *   Introduce probabilità dipendente da eventi non osservati.
    *   Evita probabilità di rilevanza zero.
*   **Parametro K:**
    *   Valore piccolo (5 o 10).
    *   Fattore di proporzionalità rispetto alla cardinalità dell'insieme dei documenti candidati.
*   **Pseudo Relevance Feedback:**
    *   Feedback implicito basato sui documenti in cima al ranking.

**5. Miglioramento della Stima dei Documenti Rilevanti**

*   Utilizzo di un valore **K** (cardinalità dell'insieme iniziale di documenti) per una stima più accurata.
*   Modello **BM25:**
    *   Assegna punteggi di rilevanza.
    *   Tiene conto della normalizzazione della richiesta.
    *   La parametrizzazione può influenzare il punteggio rispetto alla rilevanza contestuale.

**6. Esempio di Applicazione del Modello Vettoriale**

*   **Scenario:** Due documenti ("terri") con grana molto grossa (poche informazioni in comune).
*   **Osservazioni:**
    *   Il modello vettoriale riesce a distinguere i due documenti nonostante la scarsità di informazioni.
    *   Dimostra la capacità del modello di gestire dati limitati.

---
