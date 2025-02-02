**Modelli di Classificazione**

* **Obiettivo:** Assegnare record non noti a una classe con la massima accuratezza.
    * Utilizza un *training set* per costruire il modello e un *test set* per la validazione.
    * Tipi di classificatori:
        * Classificatori di base
        * Classificatori Ensemble (Boosting, Bagging, Random Forest)

* **Alberi Decisionali:** Tecnica di classificazione che rappresenta regole tramite una struttura gerarchica.
    * **Componenti:**
        * Nodi interni (o di partizionamento): Attributi di splitting.
        * Nodi foglia (o terminali): Valore dell'attributo di classe (classificazione finale).
    * **Proprietà:**
        * Ricerca dell'albero ottimo: problema NP-Completo.
        * Classificazione veloce: O(ω) nel caso peggiore 
        * Robusti rispetto ad attributi fortemente correlati.
    * **Applicazione del modello:**
        * Si parte dal nodo radice e si segue il percorso basato sulle condizioni di test degli attributi.
        * Si assegna la classe del nodo foglia raggiunto.
    * **Tree Induction Algorithm:** Tecniche greedy (dall'alto verso il basso) per costruire l'albero.
        * Problematiche: scelta del criterio di split e di stop, underfitting, overfitting.
        * **Algoritmo di Hunt:** Approccio ricorsivo per suddividere i record in sottoinsiemi più puri.
            * **Procedura di costruzione:**
                1. Se tutti i record hanno la stessa classe, il nodo diventa una foglia con quella classe.
                2. Altrimenti, si sceglie un attributo e un criterio per suddividere i record.
                3. Si applica ricorsivamente la procedura sui sottoinsiemi.


**Scelta del Criterio di Split negli Alberi Decisionali**

I. **Tipi di Attributi e Split**

   * A. **Attributi Binari:** Due possibili risultati.
   * B. **Attributi Nominali:**
      * 1. Split a più vie: Una partizione per ogni valore distinto.
      * 2. Split a due vie: Suddivisione ottimale in due insiemi.
   * C. **Attributi Ordinali:** Simili ai nominali, ma con ordine preservato.
      * 1. **Split a due vie:**  Considera tutti i possibili valori *v* per il test $A < v$ (computazionalmente costoso).
   * D. **Attributi Continui:**
      * 1. Split a due vie (binario): Test del tipo  $A < v$.
      * 2. Split a più vie: Suddivisione in intervalli  $v_{i} \leq A \leq v_{i+1}$.
	      * Discretizzazione in intervalli disgiunti.



II. **Criterio di Ottimizzazione dello Split**

   * A. **Obiettivo:** Creare nodi figli il più puri possibile (istanze della stessa classe nello stesso nodo).
   * B. **Nodi Impuri:** Aumentano la profondità dell'albero, causando overfitting, minore interpretabilità e maggiore costo computazionale.
   * C. **Misure di Impurità:** Bilanciano purezza dei nodi e complessità dell'albero.


III. **Misure di Impurità dei Nodi**

   * A. **Valutazione dell'impurità di un nodo *t* (con *k* classi e *n* nodi figli):**
      * 1. **Gini Index:** $GINI(t) = 1 - \sum_{j=1}^{k} [p(j|t)]^2$
      * 2. **Entropy:** $Entropy(t) = -\sum_{j=1}^{k} p(j|t) \log_2 p(j|t)$
      * 3. **Misclassification Error:** $Error(t) = 1 - \max p(i|t)$
      * 4. **Impurità complessiva:** $Impurity_{split} = \sum_{i=1}^{n} \frac{m_i}{m} meas(i)$  (*p(j|t)* = frequenza della classe *j* nel nodo *t*)
   * B. **Determinare il Miglior Partizionamento:** Calcolare l'impurità *P* del nodo genitore prima dello splitting.

**Alberi Decisionali:
I. **Criteri di Splitting:**

  A. **Indice di GINI:**
    1. Definizione: $GINI(t)=1-\sum_{j=1}^k[p(j|t)]^2$  
    2. Massimo: $\left( 1-\frac{1}{nc} \right)$ (record equamente distribuiti)
    3. Minimo: 0 (record di una sola classe)
    4. Calcolo per split a due vie: $1-(P_{1})^2-(P_{2})^2$
    5. Calcolo per più nodi: $GINI_{split}=\sum_{i=1}^k \frac{n_{i}}{n} GINI(i)$
    6. Guadagno per attributi binari: $GAIN=GINI(P)-GINI_{split}$
    7. Guadagno per attributi continui: ordinamento delle tuple e scelta del valore intermedio per la partizione.

  B. **Entropia:**
    1. Definizione: $Entropy(t)=- \sum_{j=1}^k p(j|t)\log_{2}(p(j|t))$
    2. Massimo: (log2(nc)) (record equamente distribuiti)
    3. Minimo: 0 (record della stessa classe)
    4. Guadagno: $GAIN_{split}=Entropy(p)-\sum_{i=1}^n \frac{m_{i}}{m}Entropy(i)$

  C. **Errore di Classificazione:**
    1. Definizione: $Error(t)=1-max_{i}p(i|t)$
    2. Massimo: $\left( 1-\frac{1}{n_{c}} \right)$ (record equamente distribuiti)
    3. Minimo: 0 (record di una sola classe)

  D. **Massimizzazione del Guadagno:**
    1. Gain Ratio: $GainRatio_{split}=\frac{Gain_{split}}{SplitInfo}$ con $SplitInfo=-\sum_{i=1}^n {\frac{m_{i}}{m}}\log_{2}\left( \frac{m_{i}}{m} \right)$

II. **Condizioni di Stop:**

  A. Record della stessa classe.
  B. Record con valori simili su tutti gli attributi.
  C. Numero di record inferiore a una soglia.

III. **Pro e Contro:**

  A. **Pro:** Basso costo, velocità, interpretabilità, robustezza al rumore.
  B. **Contro:** Spazio dell'albero esponenziale, mancata considerazione delle interazioni tra attributi.

IV. **Overfitting:**

  A. Definizione: modello eccessivamente complesso che non generalizza bene a dati sconosciuti.
  B. Cause: errore di classificazione sul training set non rappresentativo, aumento del *test error* e diminuzione del *training error* con l'aumentare dei nodi.
  C. Soluzioni:
    1. Validation set.
    2. Occam's Razor (semplicità del modello).
    3. Stima di bound statistici sull'errore di generalizzazione.
    4. Riduzione della complessità del modello.

**Stima dell'Errore di Generalizzazione**

* **Approcci alla stima:**
    * **Ottimistico:**  $e'(t) = e(t)$ (errore sul training set uguale all'errore reale)
    * **Pessimistico:** $err_{gen}(t) = err(t) + \Omega \cdot \frac{K}{N_{train}}$  dove:
* **Minimum Description Length (MDL):**
    * Minimizza il costo totale: $\text{Costo(Modello,Dati)} = \text{Costo(Dati|Modello)} + \alpha \cdot \text{Costo(Modello)}$
        * Primo termine: costo degli errori di classificazione.
        * Secondo termine: costo di codifica della complessità dell'albero (pesato da α).


**Strategie di Pruning**

* **Pre-pruning:**
    * Arresta la crescita dell'albero prima del completamento.
    * Condizione di arresto: guadagno nella stima dell'errore di generalizzazione inferiore a una soglia.
    * Altre condizioni di arresto: numero di istanze sotto soglia, indipendenza tra classi e attributi, mancato miglioramento dell'impurità.
* **Post-pruning:**
    * Sviluppa l'albero completamente, poi pota *bottom-up*.
    * Pota il sottoalbero che riduce maggiormente l'errore di generalizzazione stimato.
    * Etichettamento delle foglie: classe più frequente nel sottoalbero potato o nel training set del sottoalbero.


**Costruzione dei Dataset**

* **Holdout:**
    * Partizione in training set (2/3) e test set (1/3).
    * Svantaggio: training set potenzialmente troppo piccolo.
* **Random Subsampling:**
    * Ripetute iterazioni di Holdout con training set casuali.
* **Cross Validation:**
    * Partizione in *k* sottoinsiemi di uguale dimensione.
    * Ogni sottoinsieme usato una volta come test set, gli altri come training set.
    * Calcolo della performance media su *k* modelli.
    * Esempio: *k* alberi decisionali diversi con attributi e split differenti.
* **Bootstrap:**
    * Ricampionamento *con* reinserimento.
    * Probabilità di inclusione di un record ≈ 63.2% (per N grande).  Probabilità di esclusione ≈ $\frac{1}{e} \approx 0,368$.
    * Utile per dataset piccoli, stabilizza i risultati.


**Class Imbalance Problem**

* Classi distorte (numero di record molto diverso tra le classi).
* Sfide:
    * Molti metodi funzionano meglio con classi bilanciate.
    * L'accuratezza è una metrica inadeguata.


**Metriche Alternative per la Valutazione del Modello**

* **Matrice di Confusione:**
    * TP (True Positive): record positivi correttamente classificati.  (e altri 3 indicatori non specificati nel testo)

   B. **Accuratezza:**
      1. Formula: $Accuracy = \frac{TP + TN}{TP + FN + FP + TN}$
      2. Limite: Non adatta a classi sbilanciate.

   C. **Error Rate:**
      1. Formula: $Error \ rate = \frac{FN + FP}{TP + FN + FP + TN}$
      2. Limite: Non adatta a classi sbilanciate.

   D. **Precision:**
      1. Formula: $Precision(p) = \frac{TP}{TP + FP}$
      2. Descrizione: Quanti positivi classificati sono corretti.

   E. **Recall:**
      1. Formula: $Recall(r) = \frac{TP}{TP + FN}$
      2. Descrizione: Quanti positivi sono stati correttamente classificati sul totale dei positivi.

   F. **F-measure:**
      1. Formula: $F-measure = \frac{2pr}{p + r}$
      2. Descrizione: Media armonica di Precision e Recall.  Alta F-measure indica poche FP e FN.


II. **Tecniche per Dataset Sbilanciati**

   A. **Classificazione Cost-Sensitive:**
      1. Approccio: Assegna costi diversi all'errata classificazione di diverse classi.
      2. Obiettivo: Minimizzare il costo complessivo di errata classificazione.
      3. Matrice dei Costi: Introduce parametri di penalità per errori di classificazione.  
         Formula: $C(M) = \sum C(i,j) \times f(i,j)$ dove  *C(i,j)* è il costo di classificare *i* come *j*, e *f(i,j)* è il numero di elementi classificati erroneamente.
      4. Regola di classificazione: Classifica il nodo *t* con la classe *i* che minimizza $C(i|t) = \sum_{j} p(j|t) \times C(j,i)$, dove *p(j|t)* è la frequenza relativa alla classe *j* al nodo *t*.
      5. Applicazione a classi sbilanciate: Maggiore costo per classificare erroneamente la classe minoritaria (positiva).

   B. **Approccio basato sul Campionamento:**
      1. Obiettivo: Bilanciare la distribuzione delle classi nel training set.
      2. Undersampling: tecnica di bilanciamento delle classi tramite eliminazione di record.
        Potenziale perdita di dati utili.
      3. Oversampling: Aggiunge record alla classe minoritaria.



II. **Classificatori basati su Regole:**
* Utilizzo di regole if-then ( _Condizione_ -> _y_ ).
	* Condizione (antecedente): congiunzione di predicati sugli attributi.
	* y (conseguente): etichetta della classe.
* Costruzione del modello: identificazione di un insieme di regole.
* Copertura di un'istanza x: soddisfazione dell'antecedente della regola r.

III. **Copertura e Accuratezza delle Regole:**
* **Copertura (Coverage(r))**:  $\frac{|A|}{|D|}$ (frazione di record in D che soddisfano l'antecedente di r).
* **Accuratezza (Accuracy(r))**: $\frac{|A \cap y|}{|D|}$ (frazione di istanze che soddisfano sia l'antecedente che il conseguente di r).


IV. **Mutua Esclusività ed Esaustività delle Regole:**
* **Mutua Esclusività:** ogni record coperto al più da una regola.
* **Esaustività:** ogni record coperto almeno da una regola.
* Insieme garantiscono che ogni istanza sia coperta da esattamente una regola.

V. **Gestione della Mancanza di Mutua Esclusività ed Esaustività:**
* **Mancanza di Mutua Esclusività:**
	* Soluzione 1: ordine di attivazione delle regole.
	* Soluzione 2: assegnazione alla classe con più regole attivate.
* **Mancanza di Esaustività:** assegnazione a una classe di default.

VI. **Regole Linearmente Ordinate (Liste di Decisione):**
* Ordinamento delle regole in base a una priorità.
* **Ordinamento rule-based:** in base alle qualità delle singole regole.
* **Ordinamento class-based:** gruppi di regole per classe, con ordinamento tra le classi.
	* Rischio: regole di buona qualità superate da altre di qualità inferiore ma appartenenti a classi più importanti.


**Costruzione di un Classificatore Basato su Regole**

I. **Metodi di Costruzione**
* A. Metodi Diretti: Estraggono regole direttamente dai dati.  Esempio: Sequential Covering.
* B. Metodi Indiretti: Estraggono regole dai risultati di altri metodi di classificazione.

II. **Metodo Diretto: Sequential Covering**
* A. Estrazione regole direttamente dai dati (ordinamento class-based).
* B. Algoritmo:
	1. Lista di decisioni R inizialmente vuota.
	2. Estrazione regola per classe y usando `Learn-one-rule`.
	3. Rimozione record coperti dalla regola dal training set.
	4. Aggiunta della regola a R.
	5. Ripetizione dal punto 2 fino al soddisfacimento del criterio di arresto (estensione regola).

III. **Learn-one-rule**
* A. Obiettivo: Trovare regola che massimizzi esempi positivi e minimizzi quelli negativi.
* B. Approccio Greedy:
	1. Regola iniziale $r:\{\} \to y$.
	2. Raffinazione della regola fino al criterio di arresto.
	3. Miglioramento dell'accuratezza aggiungendo coppie (Attributo, Valore) all'antecedente.
	4. $Accuracy(r)=\frac{nr}{n}$ (nr: istanze correttamente classificate; n: istanze che soddisfano l'antecedente).

IV. **Criteri di Valutazione delle Regole**
* $LikelihoodRatio(r) = 2\sum_{i=1}^k f_i \log_2(\frac{f_i}{e_i})$ 
	* Pota regole con scarsa copertura.
* $Laplace(r)=\frac{f_{+}+1}{n+k}$
	* Pesa l'accuracy in base alla copertura. 
	* Copertura zero: probabilità a priori; copertura alta: asintoticamente all'accuracy; copertura bassa: indice diminuisce.
* $m\text{-}estimate(r) = \frac{f_+ + k p_+}{n + k}$ 
	* Pesa l'accuracy in base alla copertura 
	* Caso speciale: per $p_+ = \frac{1}{k}$, coincide con la stima di Laplace.


II. **FOIL (First Order Inductive Learner): Valutazione del Guadagno di Informazioni**

   * Misura la variazione di informazione aggiungendo un atomo ad una regola.
   * Regola iniziale ($r_0$): $A \to +$ ( $p_0$ positivi, $n_0$ negativi)
   * Regola estesa ($r_1$): $A, B \to +$ ( $p_1$ positivi, $n_1$ negativi)
   * Guadagno di informazione:
     $$ FoilGain(r_0, r_1) = p_1 \left( \log_2 \left( \frac{p_1}{p_1 + n_1} \right) - \log_2 \left( \frac{p_0}{p_0 + n_0} \right) \right) $$
   * Misura alternativa:
     $$ v(r_0, r_1) = \frac{p_1 - n_1}{p_1 + n_1} - \frac{p_0 - n_0}{p_0 + n_0} $$
   * Favorisce regole con alta copertura e accuratezza.


III. **Potatura delle Regole (Rule Pruning)**

   * Semplifica le regole per migliorare la generalizzazione (utile per approcci "greedy").
   * **Reduced Error Pruning:**
      * Iterativamente rimuove l'atomo che migliora maggiormente l'errore sul validation set.
      * Termina quando nessuna rimozione migliora l'errore.


IV. **Metodi Diretti: RIPPER**

   * Algoritmo di classificazione basato su regole.
   * Deriva un insieme di regole dal training set.
   * Scala quasi linearmente con il numero di istanze.
   * Robusto al rumore grazie all'utilizzo del validation set per evitare overfitting.

I. **Classificazione:**

   A. **Problemi a 2 classi:**
      1. Una classe è positiva, l'altra di default.
      2. Apprendimento di regole per la classe positiva.

   B. **Problemi multi-classe:**
      1. Classi ordinate per rilevanza (da *y<sub>1</sub>* meno rilevante a *y<sub>c</sub>* più diffusa).
      2. Regole costruite iterativamente, partendo dalla classe più piccola (*y<sub>1</sub>*).
      3. Esempi delle altre classi considerati negativi ad ogni iterazione.
      4. *y<sub>c</sub>* diventa classe di default.

II. **Costruzione del Set di Regole (Sequential Covering):**

   A. **Learn-one-rule:**
      1. Inizia con una regola vuota.
      2. Aggiunge atomi che migliorano il *FOIL's Information Gain*.
      3. Iterazione fino a quando non vengono coperti più esempi negativi.
      4. **Pruning:** rimozione di atomi che massimizzano  $v=\frac{p-n}{p+n}$ 

   B. **Ottimizzazione delle Regole:**
      1. Per ogni regola *r*, si considerano due alternative: *r*<sup>*</sup> (regola nuova) e *r'* (regola estesa).
      2. Si sceglie l'alternativa che minimizza il *Minimum Description Length*.
      3. Iterazione per gli esempi positivi rimanenti.

   C. **Condizione di Stop:**  *Minimum Description Length principle*.

III. **Metodi Indiretti: C4.5 rules**

   A. Trasformazione di un albero decisionale in un insieme di regole.
   B. Ogni percorso radice-foglia diventa una regola.
   C. Condizioni di test come predicati nell'antecedente.

* **Generazione Regole:**
    * Estrazione regole da albero decisionale (radice → foglie).
    * Potatura regole:
        * Rimozione atomi da antecedente (`A` → `A'`).
        * Mantenimento se errore pessimistico (`r'`) < errore (`r`).
        * Iterazione fino a errore minimo.
        * Rimozione regole duplicate.
* **Ordinamento Class-Based:**
    * Raggruppamento regole per classe (conseguente).
    * Calcolo *description length* per ogni classe:  `L(error) + gL(model)`
        * `L(error)`: bit per errori di classificazione.
        * `L(model)`: bit per rappresentare il modello.
        * `g`: parametro dipendente da attributi ridondanti (default 0.5).
    * Ordinamento classi per *description length* crescente (priorità a minimo).


**Tecniche di Classificazione - Nearest Neighbor**

**Eager Learners vs. Lazy Learners:**
    
**Algoritmi Lazy Learners**

* **Rote Classifier:**
    * Memorizza il training set.
    * Classifica solo istanze identiche a quelle nel training set.
    * Problema: non classifica istanze non presenti nel training set.

* **Nearest Neighbor:**
    * **Approccio:** Classifica basandosi sulla similarità con i vicini nel training set.
    * **Rappresentazione:** Istanze come punti in uno spazio d-dimensionale.
    * **Classificazione:**
        1. Calcola la distanza tra l'istanza di test e quelle del training set.
        2. Identifica i *k* vicini più prossimi.
        3. Assegna la classe più frequente tra i *k* vicini (opzionale: pesi basati sulla distanza).
    * **Parametri:**
        * *k* (numero di vicini):
            * *k* piccolo: sensibile al rumore.
            * *k* grande: include esempi di altre classi.
    * **Pre-processing:**
        * Normalizzazione degli attributi.
        * Attenzione alle misure di similarità per diverse distribuzioni dei dati.

* **Ottimizzazione del Costo di Calcolo:**
    * Tecniche di indicizzazione per ridurre il numero di calcoli di distanza.
    * Condensazione: riduzione del training set mantenendo le prestazioni.


**Strutture a Indice per Ricerca del Vicino Più Prossimo**

I. **Tipi di Query:**

   * **Near neighbor range search:** Trova tutti i punti entro un raggio *r* da un punto *q*.  Esempio: ristoranti entro 400m da un albergo.
   * **Approximate Near neighbor:** Trova punti con distanza massima da *q* pari a (1 + ε) volte la distanza di *q* dal suo vicino più prossimo. Esempio: ristoranti più vicini ad un albergo.
   * **K-Nearest-Neighbor:** Trova i *K* punti più vicini a *q*. Esempio: 4 ristoranti più vicini ad un albergo.
   * **Spatial join:** Trova coppie di punti (p, q) con distanza ≤ *r*, dove *p* ∈ P e *q* ∈ Q. Esempio: coppie (albergo, ristorante) entro 200m.


II. **Approcci:**

A. **Linear scan (Approccio Naïf):**

  * Calcola la distanza tra il punto di query e ogni punto nel database.
  * Tempo di esecuzione O(dN), dove N è la cardinalità del database e d la dimensionalità.
  * Non richiede strutture dati aggiuntive.
  * Inefficiente per grandi dataset o alta dimensionalità.

B. **Indici multi-dimensionali (spaziali):**

  1. **B+tree multi-attributo:**
	 * Organizza tuple in base ai valori degli attributi.
	 * Non adatto a query che non specificano il valore di un attributo.

  2. **Quad-tree:**
	 * Divide ricorsivamente lo spazio in sottoquadri.
	 * Ricerca tramite esplorazione ricorsiva dei sottoquadri.
	 * Svantaggi:
		* Punti vicini possono essere in celle diverse.
		* Complessità temporale e spaziale esponenziale $O(n\cdot 2^d)$ rispetto alla dimensionalità *d*
		* Non scala bene ad alta dimensionalità.

  3. **Kd-trees (k-dimensional tree):**
	 * Struttura dati ad albero per organizzare punti in uno spazio k-dimensionale.



**Kd-tree: Struttura dati per la ricerca in spazi k-dimensionali**

I. **Struttura del Kd-tree:**
* Ogni nodo: iper-rettangolo k-D con un "punto di separazione".
* Funzionamento: suddivide ricorsivamente lo spazio basandosi sulla posizione dei punti rispetto ai piani di separazione definiti dai punti dei nodi.
* Utilità: ottimizza ricerche di punti vicini (k-NN) e ricerche per intervallo.

II. **Costruzione del Kd-tree:**
* 1. Selezione della dimensione: scelta ciclica o strategica dell'asse di partizione.
* 2. Calcolo del valore mediano: determinazione del valore mediano lungo l'asse scelto per creare il piano di separazione.
* 3. Divisione dei punti: suddivisione dei punti in sottoinsiemi (sinistra/destra) rispetto al piano.
* 4. Ricorsione: ripetizione del processo sui sottoinsiemi fino a nodi con al massimo un punto o profondità massima raggiunta.

III. **Ricerca in un Kd-tree:**
* 1. Partenza dalla radice: inizio dalla radice dell'albero.
* 2. Discesa ricorsiva: percorso verso il basso seguendo il ramo appropriato in base alla posizione del punto di query rispetto al piano di separazione.
* 3. Nodo foglia: raggiungimento di un nodo foglia (punto candidato).
* 4. Risalita ricorsiva: controllo dei sottoalberi non esplorati per punti più vicini.
* 5. Controllo della distanza: esplorazione dell'altro sottoalbero se la distanza dal piano di separazione è minore della distanza dal miglior punto trovato finora.
* 6. Continuazione della risalita: altrimenti, si prosegue la risalita senza esplorare l'altro sottoalbero.

IV. **Pro e Contro dei Kd-trees:**
* **Pro:**
	* Partizione efficiente dello spazio k-D.
	* Query k-NN efficienti.
	* Query di range/intervallo.
* **Contro:**
	* Inefficienza ad alte dimensionalità ("maledizione della dimensionalità").
	* Prestazioni dipendenti dalla scelta dell'asse di partizione e del punto di separazione.
	* Scarsa performance su dati distorti o con distribuzioni complesse.


**Calcolo Approssimato per Ricerca di Vicini**

* **Algoritmi di Ricerca Approssimativa:**
    * **Near Neighbor Approssimativo:** Restituisce punti con distanza ≤ c * distanza punti più vicini (c > 1).  Vantaggioso quando l'approssimazione è sufficientemente accurata e la metrica di distanza riflette la qualità percepita dall'utente.
    * **Locality-Sensitive Hashing (LSH):**
        * Tecnica probabilistica per dataset ad alta dimensionalità.
        * Crea funzioni hash che mappano oggetti simili in hash simili, massimizzando le collisioni (al contrario delle HashMap).
        * Ricerca solo tra elementi con lo stesso hash.

* **Gestione Memoria Secondaria per Dataset Grandi:**
    * **Strutture dati disk-based:**
        * **R-tree:** Approccio ottimistico (tempo logaritmico).
        * **Vector Approximation File:** Approccio pessimistico (scansione veloce dell'intero dataset).

**R-tree**

* **Struttura dati:** Estensione dei B+tree per spazi multidimensionali, organizza oggetti in iperrettangoli sovrapposti.
* **Costruzione (bottom-up):**
    1. Raggruppamento oggetti (2-3 elementi).
    2. Calcolo rettangolo minimo per ogni gruppo.
    3. Unione ricorsiva dei rettangoli in nodi intermedi fino alla radice.
* **Pro:**
    * Ricerca del vicino più vicino.
    * Funziona per punti e rettangoli.
    * Evita spazi vuoti.
    * Varianti (X-tree, SS-tree, SR-tree).
    * Efficiente per basse dimensioni.
* **Contro:**
    * Inefficiente per alte dimensioni.


