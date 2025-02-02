**Modelli di Classificazione Probabilistici**

I. **Introduzione:**
* Relazione non deterministica tra attributi di input e classe target.
* Incertezza nelle previsioni.
* Utilizzo della teoria della probabilità per quantificare e gestire l'incertezza.

II. **Teoria della Probabilità:**
* Variabile casuale: Variabile con probabilità associate a ciascun risultato.
* Probabilità di un evento: Misura della probabilità di occorrenza.
* Visioni: Frequentista (frequenza relativa) e Bayesiana (più flessibile).
* Probabilità congiunta: $P(X=x_i, Y=y_j) = \frac{n_{ij}}{N}$
* Marginalizzazione: $P(X=x_i) = \sum_{j=1}^K P(X=x_i, Y=y_j)$
* Probabilità marginale: $P(X=x_i)$

III. **Probabilità Condizionata:**
* Probabilità condizionata: $P(Y|X) = \frac{P(X,Y)}{P(X)}$  e  $P(X|Y) = \frac{P(X,Y)}{P(Y)}$
* Relazione tra probabilità congiunta e condizionata: $P(X, Y)= P(Y|X) \times P(X) = P(X|Y) \times P(Y)$

IV. **Teorema di Bayes:**
* Formula: $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$
* Relazione tra probabilità condizionali $P(Y|X)$ e $P(X|Y)$.

V. **Obiettivo della Classificazione:**
* Prevedere la classe Y dati gli attributi $X_1, X_2, ..., X_d$.
* Massimizzare la probabilità a posteriori: $P(Y|X_1, X_2, ..., X_d)$.

VI. **Teorema di Bayes applicato alla Classificazione:**
* Formula: $P(Y|X_1, X_2, ..., X_d) = \frac{P(X_1, X_2, ..., X_d|Y)P(Y)}{P(X_1, X_2, ..., X_d)}$
* Massimizzare $P(Y|X_1, X_2, ..., X_d)$ equivale a massimizzare $P(X_1, X_2, ..., X_d|Y)$.
* $P(X_1, X_2, ..., X_d|Y)$: Probabilità condizionata di classe.

VII. **Termini del Teorema di Bayes:**
* $P(Y)$: Probabilità a priori (convinzioni preliminari sulla distribuzione delle etichette di classe).
* $P(X_1, ..., X_d)$: Probabilità of evidence (costante di normalizzazione).

VIII. **Stima delle Probabilità:**
* $P(Y)$: stimata dalla frazione di record per classe nel training set.
* $P(X_1, X_2, ..., X_d|Y)$: stimata dalla frazione di record di una data classe per ogni combinazione di valori di attributo (problematica per alta dimensionalità e piccoli dataset).


**Classificatore Naïve Bayes**

* **Principio:**  Classificazione basata sull'assunzione di indipendenza condizionale degli attributi data la classe.

    * **Assunzione di indipendenza condizionale:**  $P(X_1, ..., X_d|Y ) = \prod_{i=1}^d P(X_i|Y )$
    * **Classificazione:** Assegna un nuovo oggetto alla classe $Y_j$ che massimizza $P(Y_j) \prod_{i=1}^d P(X_i|Y_j)$.
    * **Calcolo delle probabilità (attributi categorici):**
        * $P(Y_j) = \frac{n_j}{N}$
        * $P(X_i|Y_j) = \frac{|X_{ij}|}{n_j}$
    * **Calcolo delle probabilità (attributi continui):**
        * **Discretizzazione:** suddivisione in intervalli.
        * **Distribuzione di probabilità (es. Normale):** $P(X_{i}=x_{i}|Y=y_{j})=\frac{1}{\sqrt{ 2 \pi\sigma_{ij} }}e^-\frac{(x_{i}-\mu_{ij})^2}{2\sigma^2_{ij}}$  (con $\mu_{ij}$ e $\sigma_{ij}^2$ stimati dai dati).

* **Indipendenza Condizionale:**

    * Definizione: $P(X|Y, Z) = P(X|Z)$  => $P(X, Y |Z) = P(X|Z)P(Y |Z)$

* **Gestione Probabilità Zero:**

    * **Problema:** $P(X_i|Y) = 0$ rende $P(X_1, ..., X_d|Y) = 0$.
    * **Soluzioni:**
        * **Stima di Laplace:** $P(A_i|C) = \frac{N_{ic} + 1}{N_c + c}$
        * **Stima m-estimate:** $P(A_i|C) = \frac{N_{ic} + mp}{N_c + m}$ 


**Reti Bayesiane**

I. **Rappresentazione:**
* Grafo Diretto Aciclico (DAG)
	* Nodi: Variabili casuali
	* Archi: Dipendenze tra variabili

II. **Indipendenza Condizionale:**
* Condizione di Markov: Un nodo è condizionatamente indipendente dai suoi non-discendenti, dati i suoi genitori.
* Implicazioni:
	* Interpretazione delle relazioni genitore-figlio come probabilità condizionate.
	* Struttura di dipendenza spesso sparsa.
	* Maggiore espressività rispetto a Naïve Bayes (classe target non solo alla radice).

III. **Distribuzione di Probabilità Congiunta:**
* Calcolo tramite la condizione di Markov:
	* $P(X_1, ..., X_n) = \prod_{i=1}^{n} P(X_i|X_1, ..., X_{i-1}) = \prod_{i=1}^{n} P(X_i|Parents(X_i))$
* Spiegazione:
	* Scomposizione in probabilità condizionali.
	* Dipendenza solo dai genitori diretti, non da tutti gli antenati.
	* Semplificazione del calcolo probabilistico.

IV. **Tabelle di Probabilità Condizionata (CPT):**
* Rappresentazione della probabilità congiunta.
* Contenuto:
	* Se nessun genitore: $P(X_i)$
	* Altrimenti: $P(X_i|Parents(X_i))$ per ogni combinazione di valori.


**Reti Bayesiane: Inferenza e Apprendimento**

I. **Rappresentazione della Probabilità:**

*   **Tavole di Probabilità Condizionata (CPT):**  Una variabile booleana con *k* genitori richiede una CPT con 2<sup>(k+1)</sup> voci.
*   **Funzione delle CPT:** Quantificano l'influenza dei genitori sul nodo figlio.

II. **Inferenza:**

*   **Calcolo della Probabilità Condizionale:**
	*   $P(Y = y|x) = \frac{P(y, x)}{P(x)} = \frac{P(y, x)}{\sum_{y}P(y,x)}$
	*   Calcola la probabilità che la classe Y assuma il valore y, dati gli attributi osservati x.

*   **Calcolo della Probabilità Marginale:**
	*   $P(x, y) = \sum_{H} P(y, x, H)$
	*   Calcola la probabilità congiunta di x e y marginalizzando sulle variabili nascoste H.

*   **Fattorizzazione:**
	*   Sfrutta la fattorizzazione della probabilità congiunta (derivante dalla struttura della rete Bayesiana) per calcolare $P(y, x, H)$.
	*   Permette il calcolo di probabilità condizionali anche in presenza di variabili nascoste.


