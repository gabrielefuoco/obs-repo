
**I. Sistema di Retrieval**

* **Obiettivo:** Identificare risorse rilevanti da un repository in risposta a una query.
* **Sottoinsiemi:**
    * **Relevant:** Documenti realmente rilevanti alla query.
    * **Retrieved:** Documenti recuperati dal sistema.
    * **Obiettivo:** Massimizzare l'intersezione (overlap) tra *Relevant* e *Retrieved*.
* **Tipi di Errori:**
    * **True Positive (tp):** Documenti rilevanti e recuperati.
    * **False Positive (fp):** Documenti non rilevanti ma recuperati.
    * **False Negative (fn):** Documenti rilevanti ma non recuperati.
    * **True Negative (tn):** Documenti non rilevanti e non recuperati.

**II. Valutazione del Sistema**

* **Elementi Necessari:**
    * Collezione di documenti di riferimento.
    * Set di query di riferimento.
    * Valutazione di rilevanza per ogni documento/query (rilevante/non rilevante).
* **Valutazione del Recupero Non Ordinato:**
    * **Precision (P):** Frazione di documenti recuperati che sono rilevanti.  $P = \frac{tp}{tp + fp}$
    * **Recall (R):** Frazione di documenti rilevanti che sono recuperati. $R = \frac{tp}{tp + fn}$  (anche *true positive rate* o *sensitivity*).
    * **Conflitto Precisione/Recall:**  Relazione negativa; un aumento di uno spesso comporta una diminuzione dell'altro.

**III. Trade-off Precisione/Recall**

* **Aumentare il Recall:** Restituire più documenti (Recall del 100% restituendo tutti i documenti).
* **Aumentare la Precision:** Restituire meno documenti (alta precisione con basso recall).

**IV. Misure di Combinazione Precisione/Recall**

* **Media Aritmetica:**  Troppo generosa.
* **Media Geometrica:** $\sqrt{P \times R}$
* **Media Armonica:** Preferita per la possibilità di assegnare pesi diversi a precisione e richiamo. (Formula non fornita nel testo).


---

**Schema Riassuntivo: Misure di Valutazione dell'Informazione**

I. **Misure di Combinazione:**

   A. **Media Armonica:** Penalizza fortemente le misurazioni basse; gestisce bene gli score vicini a zero.

II. **F-measure:**  Combinazione di precisione (P) e richiamo (R).

   A. Formula:  $F=\frac{1}{\alpha \frac{1}{P}+(1-\alpha)\frac{1}{R}}=\frac{(\beta^2+1)PR}{\beta^2P+R}$
   B.  $\beta^2=\frac{1-\alpha}{\alpha}$
   C.  α: parametro di smoothing (controlla la combinazione lineare di 1/P e 1/R).
   D.  β: peso relativo di P e R (β > 1: peso maggiore a R; β < 1: peso maggiore a P).
   E. Confronto con media geometrica: andamento simile, ma penalizza di più a bassi valori di P o R.

III. **Misure Basate sul Ranking:** Valutano la qualità dell'ordinamento dei risultati.

   A. **Rilevanza Binaria (rilevante/non rilevante):**

      1. **Precision@K (P@K):** Percentuale di documenti rilevanti tra i primi K risultati.  Esempio: Prec@3 di 2/3 significa 2 documenti rilevanti su 3.
      2. **Media della Precisione Media (MAP):** Media della precisione per ogni query, considerando tutti i K.
      3. **Preferenza Binaria:** Probabilità che un documento rilevante sia classificato più in alto di uno non rilevante.
      4. **Media del Rango Reciproco (MRR):** Reciproco del rango del primo documento rilevante per ogni query.

   B. **Livelli di Rilevanza Multipli (es. molto rilevante, rilevante, non rilevante):**

      1. **NDCG (Guadagno Cumulativo Scontato Normalizzato):** Considera l'ordine e la rilevanza dei documenti.

   C. **Curva Precision-Richiamo:** Adattamento delle misure di set (P, R, F) alle liste ordinate.

      1. Calcolo per ogni prefisso della lista (primi 1, 2, 3, ... risultati).
      2. Grafico di precisione in funzione del richiamo.  Precisione/Richiamo/F sono misure per set non ordinati.


IV. **Precision@K (dettagli):**

   A. Soglia di rango K (parametro).
   B. Determinazione della porzione di documenti rilevanti tra i primi K.
   C. Valutazione per diversi K.
   D. Calcolo della % di rilevanza nei primi K.
   E. Ignora i documenti oltre K.  Analogamente si definisce Recall@K.


---

## Schema Riassuntivo: Curva Precision-Recall e Mean Average Precision (MAP)

**I. Curva Precision-Recall**

* **Obiettivo:** Visualizzare le prestazioni di un sistema di ranking, mostrando la precisione a diversi livelli di richiamo.
    * **Calcolo:** Precisione calcolata per ogni query, considerando un numero variabile di documenti rilevanti.
    * **Interpolazione:** Valori di precisione interpolati a livelli di richiamo standard (0-1, incrementi di 0.1).
        * **Regola:** Per il livello *i*, si usa la massima precisione ottenuta per livelli di richiamo ≥ *i*.
        * **Razionale:** L'interpolazione riflette la disponibilità dell'utente a esaminare più risultati se precisione e richiamo migliorano.  L'interpolazione avviene quando si osserva una risalita nel grafico.

**II. Mean Average Precision (MAP)**

* **Obiettivo:** Valutare le prestazioni di un sistema di recupero informazioni.
    * **Calcolo:**
        * **1. Posizioni:** Identificazione delle posizioni dei documenti rilevanti nel ranking ($K_1, K_2, … K_R$).
        * **2. Precision@K:** Calcolo di Precision@K = (documenti rilevanti fino a K) / (documenti totali fino a K) per ogni posizione rilevante.
        * **3. Average Precision:** Media delle Precision@K per tutti i documenti rilevanti di una query.
        * **4. MAP:** Media aritmetica delle Average Precision su diverse query.  Esempio: $\frac{1}{3}\cdot\left( \frac{1}{1}+\frac{2}{3}+\frac{3}{5} \right)=0.76$
    * **Vantaggi:**
        * Considera la posizione dei documenti rilevanti nel ranking.
        * Sensibile alla qualità del ranking (documenti rilevanti in posizioni alte sono premiati).
        * Misura aggregata che considera le prestazioni su diverse query.

**III. Media della Precisione Media (MAP) - Considerazioni Aggiuntive**

* **Assunzioni:**
    * L'utente è interessato alla precisione per ogni documento rilevante (precisione 0 se un documento rilevante non viene recuperato).
    * È una media macro (ogni query ha lo stesso peso).
    * L'utente vuole trovare molti documenti rilevanti per query.
* **Limiti:**
    * Richiede molti giudizi di rilevanza.
    * Richiede la conoscenza globale della posizione di tutti i documenti rilevanti (aspetto considerato negativo).


---

**Misure di Performance per Sistemi di Recupero dell'Informazione**

I. **Preferenza Binaria (Bpref)**

   * **Scopo:** Misurare la performance in situazioni di giudizi di rilevanza incompleti, focalizzandosi sulla relazione di preferenza tra documenti rilevanti e irrilevanti.
   * **Formula:**  $bpref=\frac{1}{R}\sum_{r}\left( 1-\frac{|\text{n ranked higer than r}|}{min(R,N)} \right)$
      * R: numero di documenti rilevanti giudicati.
      * N: numero di documenti irrilevanti giudicati.
      * r: documento rilevante recuperato.
      * n: documento irrilevante tra i primi R recuperati.
   * **Interpretazione:** Inverso della frazione di documenti irrilevanti recuperati prima di quelli rilevanti.
   * **Vantaggi:** Robusta a giudizi incompleti; mantiene correlazione con classifiche originali anche con giudizi incompleti (a differenza di MAP).
   * **Esempio di calcolo:**  $b_{pref_{f1}} = \frac{1}{5} \left( (1 - \frac{0}{5}) + (1 - \frac{1}{5}) + (1 - \frac{3}{5}) + (1 - \frac{5}{5}) + (1 - \frac{5}{5}) \right) = 0.44$


II. **Mean Reciprocal Rank (MRR)**

   * **Scopo:** Misurare la performance basandosi sulla posizione del primo documento rilevante.
   * **Calcolo:**
      * **Rango Reciproco (RR):** $\frac{1}{K}$, dove K è il rango del primo documento rilevante.
      * **MRR:** Media dei RR su più query.
   * **Vantaggi:** Semplice da calcolare.
   * **Svantaggi:** Penalizza pesantemente posizioni non-ottimali del primo documento rilevante.
   * **Esempio di calcolo:**  $MRR = \frac{\frac{1}{1} + \frac{1}{2}}{2} = 0.75$ (con ranghi 1 e 2 per il primo documento rilevante in due query).

III. **Mean Average Precision (MAP) (Menzionato per confronto)**

   * **Scopo:** Misurare la performance considerando la precisione media su tutte le posizioni dei documenti rilevanti.
   * **Calcolo:** Media delle Average Precision (AP) su più query.
   * **Esempio di calcolo:** $MAP = \frac{0.622 + 0.520}{2} = 0.571$
   * **Nota:** Alta correlazione con Bpref quando i giudizi sono completi, ma minore correlazione con giudizi incompleti.


---

**Schema Riassuntivo: Valutazione di Sistemi di Information Retrieval**

I. **Metriche di Valutazione:**

   A. **Bpref:**  Misura la bontà di un ranking considerando solo i documenti rilevanti.
      * Calcolo:  Somma pesata delle differenze tra 1 e la proporzione di documenti irrilevanti prima di ogni documento rilevante.  Esempio:  $b_{pref_{f2}} = \frac{1}{5} \left( (1 - \frac{1}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) + (1 - \frac{3}{5}) \right) = 0.48$
      * Applicazione: Esempio con 5 documenti rilevanti e due ranking, con calcolo di Recall@k, Precision@k e Bpref per ogni ranking.

   B. **Recall@k & Precision@k:** Misure classiche di accuratezza.
      * Recall@k: Proporzione di documenti rilevanti recuperati tra i primi k.
      * Precision@k: Proporzione di documenti rilevanti tra i primi k recuperati.

   C. **MAP (Mean Average Precision):** Media delle precisioni medie su tutte le query.

   D. **MRR (Mean Reciprocal Rank):** Media dei reciproci dei ranghi del primo documento rilevante per ogni query.  Limitata perché considera solo il primo documento rilevante.

II. **Top Weightedness:**

   A. **Definizione:** Capacità di un sistema di assegnare maggiore importanza ai risultati in cima al ranking.

   B. **Metriche:** DCG (Discounted Cumulative Gain) è una metrica che considera la Top Weightedness.

III. **DCG (Discounted Cumulative Gain):**

   A. **Vantaggi rispetto a MRR:** Considera la rilevanza di tutti i documenti recuperati, non solo del primo.

   B. **Assunzioni Fondamentali:**
      * Documenti altamente rilevanti sono più utili di quelli marginalmente rilevanti.
      * L'utilità di un documento diminuisce con l'aumentare del suo rango.

   C. **Caratteristiche Principali:**
      * Utilizza la rilevanza graduata (punteggio di rilevanza).
      * Il guadagno è scontato per i ranghi maggiori, tipicamente con $\frac{1}{log_2(rango)}$.

   D. **Calcolo:**
      * **CG (Cumulative Gain) al rango n:** $CG = r_1 + r_2 + \dots + r_n$  (dove $r_i$ è il punteggio di rilevanza del documento i-esimo).
      * **DCG al rango n:** $DCG = r_1 + \frac{r_2}{\log_2(2)} + \frac{r_3}{\log_2(3)} + \dots + \frac{r_n}{\log_2(n)}$


IV. **Rilevanza Graduata:**

   A. **Definizione:** Punteggio che quantifica il grado di rilevanza di un documento (invece di una semplice distinzione binaria).  Scala di $[0, r]$ con $r > 2$.

---

**Valutazione di Sistemi di Ranking**

I. **Discounted Cumulative Gain (DCG)**

   A. Formulazione standard:  $DCG_{p}=rel_{1}+\sum_{i=2}^p \frac{rel_{i}}{\log_{2}i}$
      *  `relᵢ`: giudizio di rilevanza dell'elemento in posizione *i*.
      *  Enfasi minore sul recupero dei documenti più rilevanti.

   B. Formulazione alternativa (per recupero di documenti più rilevanti): $DCG_{p}=\sum_{i=1}^p \frac{2^{rel_{i}}-1}{\log(1+i)}$
      *  Utilizzata da alcune aziende di ricerca web.
      *  Maggiore enfasi sul recupero di documenti altamente rilevanti.


II. **Normalized Discounted Cumulative Gain (NDCG)**

   A. Calcolo:  NDCG = DCG<sub>sistema</sub> / DCG<sub>ideale</sub>
      *  Normalizza il DCG rispetto al DCG del ranking ideale.
      *  Richiede la conoscenza del ranking ideale.

   B. Applicazione:
      *  Confronto di query con un numero variabile di risultati rilevanti.
      *  Metodo popolare per la valutazione dei sistemi di ricerca web.
      *  Preferibile al DCG quando la rilevanza della query è particolarmente importante.


III. **Tau di Kendall**

   A. Descrizione: Misura per confrontare ranking. Variante del coefficiente di correlazione di Pearson applicata a ranking.

   B. Formula: $\tau=\frac{n_{c}-n_{d}}{\frac{1}{2}n(n-1)}$
      *  `n`: numero totale di elementi.
      *  `n<sub>c</sub>`: numero di coppie concordanti.
      *  `n<sub>d</sub>`: numero di coppie discordanti.

   C. Interpretazione:
      *  Calcolato come coefficiente di correlazione tra due insiemi di N(N−1) valori binari (1 per coppie concordanti, 0 per discordanti).
      *  Considerare l'insieme dei dati, non singoli dati, per una valutazione significativa.



---

**Schema Riassuntivo: Misure di Accordo tra Ranking e Giudici**

I. **Tau di Kendall (τ)**
    * Misura l'accordo tra due ranking.
    * Esempio:  [1, 3, 2, 4] genera coppie di confronti.  nc e nd rappresentano il numero di coppie concordanti e discordanti, rispettivamente.
    * Formula implicita:  τ = funzione di nc e nd (esempio fornito: τ = 0.67).
    * Significatività statistica: Necessaria valutazione per escludere l'accordo casuale.

II. **Kappa Measure (κ)**
    * Misura l'accordo tra più giudici (o predittori) su giudizi categorici.
    * Corregge per l'accordo casuale.
    * Formula:  $$k=\frac{P(A)-P(E)}{1-P(E)}$$
        * P(A): Proporzione di accordo osservato tra i giudici.
        * P(E): Proporzione di accordo atteso per caso.
    * Interpretazione:  Valori in $\left[ \frac{2}{3}, 1 \right]$ sono accettabili; valori inferiori indicano problemi metodologici.

III. **Esempio di Calcolo Kappa**
    * **Dati:** Matrice di accordo tra due giudici (400 valutazioni totali per giudice).
        * 300 accordi su rilevanza.
        * 70 accordi su non-rilevanza.
        * 80 disaccordi su non-rilevanza (80+90=170 totali)
        * 320+310=630 disaccordi su rilevanza
    * **Calcolo P(A):** $$P(A) = \frac{300 + 70}{400} = 0.925$$
    * **Calcolo Probabilità Marginali:**
        * $$P(nonrelevant) = \frac{170}{800} = 0.2125$$
        * $$P(relevant) = \frac{630}{800} = 0.7875$$
    * **Calcolo P(E):** $$P(E) = 0.2125^2 + 0.7875^2 = 0.665$$
    * **Calcolo κ:** $$\kappa = \frac{0.925 - 0.665}{1 - 0.665} = 0.776$$  (Accordo accettabile)

IV. **Elementi del Calcolo Kappa**
    * **Probabilità di Accordo (P(A))**:  `(Numero di accordi) / (Numero totale di valutazioni)`
    * **Probabilità Marginali (P(E))**:  `P(non rilevante)^2 + P(rilevante)^2`  (stima l'accordo casuale)


---

**Interpretazione dell'Accordo tra Giudici**

* **Confronto Probabilità:**  Valutare se l'accordo è reale o casuale.
    * **Probabilità di Accordo (p(A)):** Probabilità che i giudici siano d'accordo.
    * **Probabilità Marginale (p(E)):** Probabilità di accordo per caso.
* **Conclusione:**  Se p(A) >> p(E), allora l'accordo è significativo e indica una reale concordanza di opinioni tra i giudici.

---
