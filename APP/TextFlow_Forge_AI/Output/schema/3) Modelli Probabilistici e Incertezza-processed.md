
**I. Topic Modeling**

*   Definizione: Modellazione e rappresentazione di dati testuali.
*   Differenza chiave da modelli precedenti: Rappresenta un *topic* come una distribuzione di probabilità sullo spazio dei termini; un documento è visto come una miscela di distribuzioni di probabilità dei *topic*.
*   Importanza: Grande impatto nella prima decade degli anni 2000 (Stochastic Topic Modeling).

**II. Basi di Conoscenza Lessicali (BCL)**

*   Ruolo: Elemento importante nel Knowledge Management.
*   Esempio: WordNet (fornisce informazioni semantiche sulle parole).
*   Applicazioni recenti: Utilizzo in ricerche per testare la comprensione dei Language Models in specifici task.

**III. Vector Space Model (VSM)**

*   Ritorno nell'Information Retrieval.
*   Modello e proprietà:  Presentazione di un modello e delle sue caratteristiche.
*   Generalizzazione probabilistica: Versione probabilistica della funzione TF per task di retrieval.
*   Premessa: I sistemi tradizionali di Information Retrieval richiedono un meccanismo di ranking.

**IV. Incertezza nella Ricerca di Informazioni**

*   Problema: Le query degli utenti non sempre sono precise e chiare a causa dell'imprecisa traduzione in linguaggio naturale delle esigenze informative.
*   Soluzione: Introduzione di una nozione di incertezza nei sistemi di recupero delle informazioni.
*   Esempio (Data Mining): Utilizzo di modelli probabilistici per gestire l'incertezza, analogamente all'"uncertainty data mining".

    *   Sotto-esempio (Misurazioni Sensoriali): Sensori ottici per il particolato atmosferico (più economici ma sensibili all'umidità, causando sovrastima).  Evidenzia la difficoltà di ottenere valori precisi nelle misurazioni.

**V. Introduzione all'Incertezza nei Dati**

*   Problema:  In analisi dati (classificazione, retrieval), si assume spesso che i dati siano valori numerici precisi.
*   Gestione dell'incertezza: Associazione di una distribuzione di probabilità univariata (es. gaussiana) a ciascun valore numerico.
*   Conseguenza: Trasformazione da vettori multidimensionali a insiemi di distribuzioni di probabilità (analisi più complessa ma realistica).

---

**Strumenti per la Gestione dell'Incertezza e Modelli di Retrieval Probabilistici**

I. **Gestione dell'Incertezza:**
    *   Utilizzo di strumenti come la divergenza di Shannon-Jensen per misurare la distanza tra distribuzioni di probabilità.
    *   Importanza dell'incertezza nel data mining, knowledge management e information retrieval per sistemi di retrieval più raffinati e complessi.

II. **Modelli di Retrieval Probabilistici:**
    *   **Probability Ranking Principle:** Classifica i documenti in base alla probabilità di pertinenza.
    *   **Binary Independence Model:**  Modello bayesiano che assume l'indipendenza tra i termini.
    *   **Modello Okapi BM25:** Versione probabilistica di TF-IDF, considera frequenza dei termini e lunghezza del documento.
    *   **Reti Bayesiane:** Modellano le dipendenze tra i termini, applicando l'inferenza bayesiana per la gestione dell'incertezza.

III. **Modelli Probabilistici per il Ranking di Documenti:**
    *   Evoluzione significativa negli ultimi anni, grazie a tecnologie più efficienti.
    *   Obiettivo: Determinare il miglior *result set* di documenti altamente rilevanti, ordinati per rilevanza.

IV. **Principio di Ranking Probabilistico:**
    *   Approccio basato sulla probabilità di rilevanza di un documento rispetto a una query.
    *   Formalizzazione come problema di classificazione:  determinare P(R=1|X), ovvero la probabilità che un documento X sia rilevante (R=1) o non rilevante (R=0).
    *   **Notazione:**
        *   X: Rappresentazione vettoriale del documento.
        *   R: Classe "rilevante".
        *   P(R=1|X): Probabilità che il documento X appartenga alla classe R.
    *   **Formula di Bayes:**
        $$P(R=1|X) = \frac{{P(X|R=1) \cdot P(R=1)}}{P(X)}$$
        *   P(X|R=1): Likelihood (probabilità di osservare X dato che è rilevante).
        *   P(R=1): Probabilità a priori che un documento sia rilevante.
        *   P(X): Probabilità di osservare il documento X (costante, non influenza il ranking).
    *   Il ranking dipende da P(X|R=1) e P(R=1).

V. **Classificazione e Problema del Bias:**
    *   L'assunzione di indipendenza condizionata può essere problematica nei dati testuali a causa delle dipendenze tra le dimensioni.
    *   Il problema della classificazione si basa sulla formula di *price*, con denominatore costante.
    *   **Il termine di *likelihood*:**
        *   Computazionalmente complesso nel *DICE*.
        *   Stima tramite *maximum likelihood estimate* (frequenza relativa).
        *   Si basa sull'assunzione di indipendenza condizionata tra le dimensioni, data la classe.
    *   **Stima delle probabilità *a priori*:**
        *   Stimate osservando un *training set*.


---

**I. Problema del Bias e Dipendenza nei Dati**

* **A. Bias nella Classificazione:** Il bias è un problema significativo nella classificazione di testo e altri tipi di dati.
* **B. Validità dell'Indipendenza:** L'assunzione di indipendenza tra dimensioni è spesso valida dopo la riduzione della dimensionalità nella fase di pre-processamento. Tuttavia, nei dati testuali, questa assunzione è spesso irrealistica.
* **C. Dipendenza nei Dati Testuali:** Le dimensioni nei dati testuali (es. parole) sono spesso dipendenti (es. articoli e sostantivi, influenza grammaticale).


**II. Probability Ranking e Binary Independence Model (BIM)**

* **A. Probability Ranking (PRP):** Ordina i documenti in base alla probabilità di rilevanza  $P(R=1|x) > P(R=0|x)$, dove:
    * $P(R=1)$, $P(R=0)$: probabilità a priori di rilevanza/non rilevanza.
    * $P(x|R=1)$, $P(x|R=0)$: probabilità di osservare il documento x dato che è rilevante/non rilevante.
    * L'obiettivo è minimizzare il rischio Bayesiano (error rate).
* **B. Binary Independence Model (BIM):** Rappresenta i documenti con vettori binari di incidenza dei termini ($x_i = 1$ se il termine i-esimo è presente, 0 altrimenti). Assume l'indipendenza tra i termini. Calcola $P(R=1|D,Q)$ e $P(R=0|D,Q)$ per il ranking dei documenti.  L'obiettivo è solo il ranking.


**III. Applicazione del Teorema di Bayes per il Ranking**

* **A. Focus sul Ranking:** L'interesse non è sul punteggio probabilistico assoluto, ma sul ranking dei documenti.
* **B. Odds Ratio (O(R|QX)):** Si utilizza il rapporto di probabilità:
    $$O(R|Q\vec{X}) = \frac{P(R=1|Q\vec{X})}{P(R=0|Q\vec{X})}$$
    dove si applica il teorema di Bayes a ciascun termine.
* **C. Odds:** Misura di probabilità che rappresenta il rapporto tra la probabilità di un evento e la probabilità del suo complemento.  Aiuta a stabilire l'ordinamento dei documenti.
* **D. Odds Ratio (OR):** Misura di associazione relativa tra due variabili binarie.  Calcolato come:
    $$OR = \frac{n_{11} \cdot n_{00}}{n_{10} \cdot n_{01}}$$
    dove $n_{ij}$ sono le frequenze di una tabella di contingenza 2x2.

---

**Applicazioni e Confronto tra Odds Ratio (OR) e Rischio Relativo (RR)**

I. **Applicazioni degli Odds Ratio (OR)**
    * Epidemiologia: Valutazione del rischio di eventi in diverse popolazioni (es. incidenza di malattie in relazione a fattori di rischio).
    * Ricerca sociale: Studio dell'associazione tra variabili sociali.
    * Marketing: Analisi dell'efficacia delle campagne pubblicitarie.

II. **Odds Ratio (OR)**
    * Definizione: Misura di associazione che confronta le probabilità di un evento in due gruppi.
    * Formula:  OR = (Odds di Y in presenza di X) / (Odds di Y in assenza di X)
    * Interpretazione:
        * OR = 1: Nessuna differenza nelle odds tra i gruppi.
        * OR < 1: Odds di Y più basse nel gruppo con X.
        * OR > 1: Odds di Y più alte nel gruppo con X.
    * Proprietà:
        * Non simmetrica (lo scambio di X e Y altera il valore).
        * Robusta (stimabile anche senza incidenze, utile per studi retrospettivi).
        * Approssima il RR per eventi rari.

III. **Rischio Relativo (RR)**
    * Definizione: Misura la probabilità di un evento in presenza di una variabile rispetto alla sua probabilità in assenza.
    * Formula: RR = (Probabilità di osservare X in presenza di Y) / (Probabilità di osservare X in assenza di Y)
    * Interpretazione:
        * RR = 1: Nessuna differenza di rischio.
        * RR < 1: Evento Y meno probabile in presenza di X.
        * RR > 1: Evento Y più probabile in presenza di X.

IV. **Confronto tra OR e RR**
    * Studi retrospettivi: OR preferibile (stimabile senza incidenze).
    * Studi prospettici: RR preferibile (misura diretta del rischio).
    * Eventi rari: OR approssima bene il RR.


V. **Esempio:**
    * Studio sull'associazione tra fumo (X) e malattia (Y).
    * Incidenza della malattia: 30% nei fumatori, 10% nei non fumatori.
    * Calcolo OR:
        * Odds malattia fumatori: 30/70 = 0.43
        * Odds malattia non fumatori: 10/90 = 0.11
        * OR = 0.43 / 0.11 = 3.91


---

**I. Odds Ratio (OR)**

* **Interpretazione:**  L'OR misura l'associazione tra un fattore di rischio e un evento.  Un OR di 3.91 indica un rischio 3.91 volte maggiore per i fumatori rispetto ai non fumatori.
* **Applicazione:** Utile per studi epidemiologici, specialmente per eventi rari, ad esempio, l'associazione tra esposizione ambientale e malattie.
* **Nota:** L'OR approssima il Risk Ratio (RR) solo con eventi rari.  Con eventi frequenti, l'OR sovrastima il RR.


**II. OTS-I e Likelihood Ratio**

* **Definizione OTS-I:** Query equivalente a TOS, che considera la rilevanza (R=1) o non rilevanza (R=0) di un documento.
* **Likelihood Ratio:**  $$\frac{PR(Q | R=1)}{PR (Q | R=0)}$$
    * PR(Q | R=1): Probabilità di osservare il dato Q dato che l'ipotesi di rilevanza è vera.
    * PR(Q | R=0): Probabilità di osservare il dato Q dato che l'ipotesi di rilevanza è falsa.
* **Likelihood:** Probabilità di osservare Q, data la rilevanza o non rilevanza (costante per ogni termine).


**III. Applicazione dell'ipotesi di indipendenza**

* **Formula OTS-I con indipendenza:** $$OTS-i(Q)= \prod_{i = 1}^{n} \frac{p(x_i \mid R = 1, q)}{p(x_i \mid R = 0, q)}$$
    * x<sub>i</sub>: Presenza/assenza del termine i-esimo.


**IV. Stima della produttoria di likelihood**

* **Suddivisione:** La produttoria viene divisa in due parti:
    * Termini presenti nel documento: $$\prod_{x_{i}=1} \frac{p\left(x_{i}=1 \mid R=1,q\right)}{p\left(x_{i}=1 \mid R=0,q\right)}$$
    * Termini assenti nel documento: $$\prod_{x_{i}=0} \frac{p\left(x_{i}=0 \mid R=1,q\right)}{p\left(x_{i}=0 \mid R=0,q\right)}$$


**V. Notazione semplificata**

* **P<sub>i</sub>:** Likelihood di presenza del termine i-esimo, data la rilevanza.
* **R<sub>i</sub>:** Likelihood di presenza del termine i-esimo, data la non rilevanza.
* **Produttoria semplificata:** $\prod_{i=1}^{n} \left( \frac{P_i}{R_i} \right)$


**VI. Probabilità di Osservazione dei Termini in un Documento**

* **Definizioni:**
    * **r<sub>i</sub>:** Probabilità di osservare il termine i-esimo, assumendo la rilevanza.
    * **t<sub>i</sub>:** Probabilità di osservare il termine i-esimo, assumendo la novità.
* **Tabella di contingenza:** Mostra la relazione tra presenza/assenza del termine e rilevanza/non rilevanza del documento.  (vedi tabella nel testo originale)
* **Interpretazione della tabella:** Definisce p<sub>i</sub>, r<sub>i</sub> e le loro probabilità complementari.
* **Odds:** Rapporto tra la probabilità di un evento e la sua probabilità complementare, calcolato per termini "matching" e "non matching" nella query.


---

**Schema Riassuntivo: Modello di Rilevanza dei Documenti**

I. **Assunzioni del Modello:**
    * Ignora l'assenza di termini nella query.
    * Non valuta la probabilità di occorrenza di termini non presenti nella query.

II. **Derivazione della Formula Finale:**
    * **Formula iniziale:**
        $$
        O(R \mid q, \tilde{x}) = O(R \mid q) \cdot \prod_{x_{i}=1} \frac{p(x_{i}=1 \mid R=1,q)}{p(x_{i}=1 \mid R=0,q)} \cdot \prod_{x_{i}=0} \frac{p(x_{i}=0 \mid R=1,q)}{p(x_{i}=0 \mid R=0,q)}
        $$
    * **Scomposizione in due produttorie:**
        1. Produttoria termini "match" (xᵢ = 1).
        2. Produttoria termini "non-match" (xᵢ = 0).
    * **Introduzione di una produttoria fittizia:**
        $$
        O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i = 1 \\ q_i = 1}} \frac{p_i}{r_i} \cdot \prod_{\substack{x_i = 0 \\ q_i = 1}} \frac{1 - p_i}{1 - r_i}
        $$
    * **Formula finale (dopo ridistribuzione):**
        $$
        O(R \mid q, \vec{x}) = O(R \mid q) \cdot \prod_{\substack{x_i=q_i=1}} \frac{p_i(1-r_i)}{r_i(1-p_i)} \cdot \prod_{\substack{q_i=1 }} \frac{1-p_i}{1-r_i}
        $$
    * **Interpretazione della formula finale:** Considera sia termini "match" che "non-match".

III. **Retrieval Status Value (RSV):**
    * **Definizione:** Punteggio di rilevanza di un documento rispetto a una query.
    * **Calcolo:**
        1. Calcolo dell'odds ratio per ogni termine "match".
        2. Somma dei logaritmi degli odds ratio.
        3. RSV = risultato della sommatoria.
    * **Interpretazione:** RSV alto indica maggiore rilevanza; RSV basso indica minore rilevanza.
    * **Esempio:** Query con "X" e "Q"; documento contiene entrambi. RSV = log(1) + log(1) = 0.


---

**Stima delle Probabilità e Analisi dei Modelli Linguistici**

* **Stima delle Probabilità:**
    * Calcolo degli Odds Ratio richiede stima della probabilità di occorrenza dei termini.
    * Tecniche come lo smoothing di Laplace vengono utilizzate.

* **Analisi Dati Addestramento Modelli Linguistici:**
    * Sfida: "inversione del modello" (risalire ai dati di addestramento dal modello).
    * Impatto: scoperta rivoluzionaria, investimenti significativi.
    * Trasparenza limitata, soprattutto nei modelli open source.  Dettagli sui dati di addestramento insufficienti.
    * Soluzione: addestramento di un modello specifico per riconoscere il contesto dei dati originali (processo complesso).

* **Simboli e Definizioni:**
    * **RSV:**  $RSV = \log \prod_{x_i=q_i=1} \frac{p_i(1-r_i)}{r_i(1-p_i)} = \sum_{x_i=q_i=1} \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$ (logaritmo del produttore di all special)
    * **i:** Numero di termini (dimensione del vocabolario).
    * **n, s, N, S:** Simboli per indicare la dipendenza a livello di rappresentazione di testi e binariet.  $RSV = \sum_{x_i=q_i=1} c_i; \quad c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$

* **Calcolo dell'Odds Ratio:**
    * **Contesto:** Analisi di un insieme di N documenti per determinare la rilevanza di un termine.
    * **Definizioni:**
        * S: Numero totale di documenti rilevanti.
        * n: Numero di documenti con il termine specifico.
        * s: Numero di documenti rilevanti con il termine specifico.
    * **Tabellina di Contingenza:** (vedi tabella nel testo originale)
    * **Calcolo Probabilità:** $p_i = \frac{s}{S}$  e  $r_i = \frac{(n-s)}{(N-S)}$
        * $p_i$: Probabilità che il termine sia presente dato documento rilevante.
        * $r_i$: Probabilità che il termine sia presente dato documento non rilevante.
    * **Odds Ratio ($c_i$):** $c_i = \log \frac{p_i(1-r_i)}{r_i(1-p_i)}$  Misura la forza dell'associazione tra termine e rilevanza.
    * **Odds Ratio in termini di count:** $c_i = K(N,n,S,s) = \log \frac{s/(S-s)}{(n-s)/(N-n-S+s)}$
    * **$r_i$ e approssimazione dei non rilevanti:**
        * Ipotesi: L'insieme dei documenti non rilevanti può essere approssimato dall'intera collezione.
        * Conseguenze: $r_i$ è approssimabile a N piccolo / N grande (N piccolo: documenti con il termine; N grande: totale documenti).


---

**Oz di competenza e Informazione Probabilistica**

I. **IDF (Inverse Document Frequency):**

   * Formula:  $$\log \frac{1-r_i}{r_i} = \log \frac{N-n-S+s}{n-s} \approx \log \frac{N-n}{n} \approx \log \frac{N}{n} = IDF$$
   * Approssimazione:  $N >> n$, valida soprattutto in sistemi aperti (es. web). Meno accurata in corpus tematici a causa della legge di Zipf.
   * Interpretazione: Logaritmo del rapporto tra il numero totale di documenti (N) e il numero di documenti in cui appare il termine (n). Rappresenta la norma inversa del termine.

II. **Probabilità e Smoothing:**

   * Approccio probabilistico: Introduzione di incertezza nella stima, coerente con la legge di Zipf.  TFPF (Term Frequency-Probability Factor) come misura robusta.
   * Smoothing: Tecnica per evitare risposte troppo generose, penalizzando eventi frequenti e facendo emergere eventi rari.

     * Effetti: Riduzione della probabilità di eventi frequenti e aumento della probabilità di eventi rari.
     * Formula di probabilità condizionale (Bayesiana): $$ P_{i}^{(h+1)} = \frac{\left|V_{i}\right|+\kappa p_{i}^{(h)}}{\left|V\right|+K} $$
       * $P_i^{(h+1)}$: Probabilità all'iterazione h+1.
       * $|V_i|$: Numero di occorrenze dell'evento i.
       * $|V|$: Numero totale di elementi.
       * K: Parametro di smoothing.
       * $p_i^{(h)}$: Probabilità all'iterazione h.
     * Fattori nascosti: Lo smoothing introduce fattori nascosti che rappresentano la correlazione tra termini.

III. **Stima di  $p_i$ e Relevance Feedback:**

   * Obiettivo: Stimare la likelihood di incidenza del termine nei documenti rilevanti ($p_i$).
   * Approcci:
     * Stima da un sottoinsieme di documenti etichettati (con feedback utente/oracolo).
     * Utilizzo di teorie e leggi per approssimare $p_i$.
     * Probabilistic Relevance Feedback: Rafinamento del result set tramite feedback utente (indicazione di rilevanza/irrilevanza).
   * Obiettivo del Relevance Feedback: Migliorare l'accuratezza della risposta alla query, spesso tramite espansione della query.


---

## Schema Riassuntivo: Espansione Query e Relevance Feedback

**I. Espansione della Query:**

* Aggiunta di nuovi termini alla query originale.
    * Termini selezionati in base ai documenti rilevanti.
    * Processo iterativo: identificazione documenti rilevanti → estrazione keyword → aggiunta alla query.
    * Miglioramento dell'accuratezza della query.

**II. Relevance Feedback:**

* Processo iterativo per migliorare la qualità dei risultati di ricerca.
    * Query iniziale → risultati → feedback utente (rilevanti/non rilevanti) → raffinamento query.
* **Fine ultimo:** Adattare il sistema alle preferenze dell'utente.

**III. Probabilistic Relevance Feedback:**

* Utilizza un modello probabilistico per stimare la probabilità di rilevanza di un documento.
    * Basato sulla presenza/assenza di termini e sulla loro rilevanza per la query.
* **Processo:**
    * 1. Stima iniziale P(R|D) e P(I|R).
    * 2. Identificazione documenti rilevanti (feedback utente).
    * 3. Raffinamento della stima di probabilità.
    * 4. Ricerca iterativa con nuove stime.
* Si basa sul concetto di inverso documento:  P(termine | documento rilevante).
* **Esempio:** 5/10 documenti rilevanti; "informatica" in 3/5 documenti rilevanti → P(informatica | documento rilevante) = 0.6.

**IV. Approssimazione della Probabilità di Rilevanza:**

* Approssimazione:  $$P(rilevanza | termine) ≈ \frac{|V|}{|I|} $$
    * V: insieme documenti rilevanti; I: insieme di tutti i documenti.
* **Problema:** Calcolo computazionalmente costoso se V è grande.
* **Soluzione:** Meccanismo iterativo per raffinare l'insieme di documenti candidati + smoothing.
* **Smoothing:** Introduce probabilità basata su eventi non osservati (presenza/assenza termine in altri documenti).
* **Parametro K:** Fattore di proporzionalità nello smoothing (tipicamente 5 o 10).
* **Pseudo Relevance Feedback:** Feedback implicito basato sui documenti con ranking più alto.

**V. Miglioramento della Stima dei Documenti Rilevanti:**

* Utilizzo del parametro K (cardinalità insieme iniziale documenti) per una stima più accurata.
* **BM25:** Modello di recupero informazioni che assegna punteggi di rilevanza.
    * Considera la frequenza delle parole chiave e la normalizzazione della richiesta.
    * Può assegnare punteggi diversi rispetto alla rilevanza contestuale.
* **Esempio Modello Vettoriale:**  Capacità di distinguere documenti con poche informazioni in comune.



---
