##### Modelli di Classificazione

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
- Se tutti i record hanno la stessa classe, il nodo diventa una foglia con quella classe.
- Altrimenti, si sceglie un attributo e un criterio per suddividere i record.
- Si applica ricorsivamente la procedura sui sottoinsiemi.

##### Scelta del Criterio di Split negli Alberi Decisionali

##### Tipi di Attributi e Split

* **Attributi Binari:** Due possibili risultati.
* **Attributi Nominali:**
	* Split a più vie: Una partizione per ogni valore distinto.
	* Split a due vie: Suddivisione ottimale in due insiemi.
	* **Attributi Ordinali:** Simili ai nominali, ma con ordine preservato.
	* **Split a due vie:** Considera tutti i possibili valori *v* per il test $A < v$ (computazionalmente costoso).
* **Attributi Continui:**
	* Split a due vie (binario): Test del tipo $A < v$.
	* Split a più vie: Suddivisione in intervalli $v_{i} \leq A \leq v_{i+1}$.
	* Discretizzazione in intervalli disgiunti.

##### Criterio di Ottimizzazione dello Split

* **Obiettivo:** Creare nodi figli il più puri possibile (istanze della stessa classe nello stesso nodo).
* **Nodi Impuri:** Aumentano la profondità dell'albero, causando overfitting, minore interpretabilità e maggiore costo computazionale.
* **Misure di Impurità:** Bilanciano purezza dei nodi e complessità dell'albero.

##### Misure di Impurità dei Nodi

* **Valutazione dell'impurità di un nodo *t* (con *k* classi e *n* nodi figli):**
	* **Gini Index:** $GINI(t) = 1 - \sum_{j=1}^{k} [p(j|t)]^2$
	* **Entropy:** $Entropy(t) = -\sum_{j=1}^{k} p(j|t) \log_2 p(j|t)$
	* **Misclassification Error:** $Error(t) = 1 - \max p(i|t)$
	* **Impurità complessiva:** $Impurity_{split} = \sum_{i=1}^{n} \frac{m_i}{m} meas(i)$ (*p(j|t)* = frequenza della classe *j* nel nodo *t*)
	* **Determinare il Miglior Partizionamento:** Calcolare l'impurità *P* del nodo genitore prima dello splitting.

**Alberi Decisionali:
##### Criteri di Splitting:

##### Indice di GINI:

- Definizione: $GINI(t)=1-\sum_{j=1}^k[p(j|t)]^2$
- Massimo: $\left( 1-\frac{1}{nc} \right)$ (record equamente distribuiti)
- Minimo: 0 (record di una sola classe)
- Calcolo per split a due vie: $1-(P_{1})^2-(P_{2})^2$
- Calcolo per più nodi: $GINI_{split}=\sum_{i=1}^k \frac{n_{i}}{n} GINI(i)$
- Guadagno per attributi binari: $GAIN=GINI(P)-GINI_{split}$
- Guadagno per attributi continui: ordinamento delle tuple e scelta del valore intermedio per la partizione.

##### Entropia:

- Definizione: $Entropy(t)=- \sum_{j=1}^k p(j|t)\log_{2}(p(j|t))$
- Massimo: (log2(nc)) (record equamente distribuiti)
- Minimo: 0 (record della stessa classe)
- Guadagno: $GAIN_{split}=Entropy(p)-\sum_{i=1}^n \frac{m_{i}}{m}Entropy(i)$

##### Errore di Classificazione:

- Definizione: $Error(t)=1-max_{i}p(i|t)$
- Massimo: $\left( 1-\frac{1}{n_{c}} \right)$ (record equamente distribuiti)
- Minimo: 0 (record di una sola classe)

##### Massimizzazione del Guadagno:

- Gain Ratio: $GainRatio_{split}=\frac{Gain_{split}}{SplitInfo}$ con $SplitInfo=-\sum_{i=1}^n {\frac{m_{i}}{m}}\log_{2}\left( \frac{m_{i}}{m} \right)$

##### Condizioni di Stop:

Record della stessa classe.
Record con valori simili su tutti gli attributi.
Numero di record inferiore a una soglia.

##### Pro e Contro:

**Pro:** Basso costo, velocità, interpretabilità, robustezza al rumore.
**Contro:** Spazio dell'albero esponenziale, mancata considerazione delle interazioni tra attributi.

##### Overfitting:

Definizione: modello eccessivamente complesso che non generalizza bene a dati sconosciuti.
Cause: errore di classificazione sul training set non rappresentativo, aumento del *test error* e diminuzione del *training error* con l'aumentare dei nodi.
Soluzioni:
- Validation set.
- Occam's Razor (semplicità del modello).
- Stima di bound statistici sull'errore di generalizzazione.
- Riduzione della complessità del modello.

##### Stima dell'Errore di Generalizzazione

* **Approcci alla stima:**
	* **Ottimistico:** $e'(t) = e(t)$ (errore sul training set uguale all'errore reale)
	* **Pessimistico:** $err_{gen}(t) = err(t) + \Omega \cdot \frac{K}{N_{train}}$ dove:
* **Minimum Description Length (MDL):**
	* Minimizza il costo totale: $\text{Costo(Modello,Dati)} = \text{Costo(Dati|Modello)} + \alpha \cdot \text{Costo(Modello)}$
	* Primo termine: costo degli errori di classificazione.
	* Secondo termine: costo di codifica della complessità dell'albero (pesato da α).

##### Strategie di Pruning

* **Pre-pruning:**
	* Arresta la crescita dell'albero prima del completamento.
	* Condizione di arresto: guadagno nella stima dell'errore di generalizzazione inferiore a una soglia.
	* Altre condizioni di arresto: numero di istanze sotto soglia, indipendenza tra classi e attributi, mancato miglioramento dell'impurità.
* **Post-pruning:**
	* Sviluppa l'albero completamente, poi pota *bottom-up*.
	* Pota il sottoalbero che riduce maggiormente l'errore di generalizzazione stimato.
	* Etichettamento delle foglie: classe più frequente nel sottoalbero potato o nel training set del sottoalbero.

##### Costruzione dei Dataset

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
	* Probabilità di inclusione di un record ≈ 63.2% (per N grande). Probabilità di esclusione ≈ $\frac{1}{e} \approx 0,368$.
	* Utile per dataset piccoli, stabilizza i risultati.

##### Class Imbalance Problem

* Classi distorte (numero di record molto diverso tra le classi).
* Sfide:
* Molti metodi funzionano meglio con classi bilanciate.
* L'accuratezza è una metrica inadeguata.

##### Metriche Alternative per la Valutazione del Modello

* **Matrice di Confusione:**
	* TP (True Positive): record positivi correttamente classificati. (e altri 3 indicatori non specificati nel testo)

##### Accuratezza:

- Formula: $Accuracy = \frac{TP + TN}{TP + FN + FP + TN}$
- Limite: Non adatta a classi sbilanciate.

##### Error Rate:

- Formula: $Error \ rate = \frac{FN + FP}{TP + FN + FP + TN}$
- Limite: Non adatta a classi sbilanciate.

##### Precision:

- Formula: $Precision(p) = \frac{TP}{TP + FP}$
- Descrizione: Quanti positivi classificati sono corretti.

##### Recall:

- Formula: $Recall(r) = \frac{TP}{TP + FN}$
- Descrizione: Quanti positivi sono stati correttamente classificati sul totale dei positivi.

##### F-measure:

- Formula: $F-measure = \frac{2pr}{p + r}$
- Descrizione: Media armonica di Precision e Recall. Alta F-measure indica poche FP e FN.

##### Tecniche per Dataset Sbilanciati

##### Classificazione Cost-Sensitive:

- Approccio: Assegna costi diversi all'errata classificazione di diverse classi.
- Obiettivo: Minimizzare il costo complessivo di errata classificazione.
- Matrice dei Costi: Introduce parametri di penalità per errori di classificazione.
Formula: $C(M) = \sum C(i,j) \times f(i,j)$ dove *C(i,j)* è il costo di classificare *i* come *j*, e *f(i,j)* è il numero di elementi classificati erroneamente.
- Regola di classificazione: Classifica il nodo *t* con la classe *i* che minimizza $C(i|t) = \sum_{j} p(j|t) \times C(j,i)$, dove *p(j|t)* è la frequenza relativa alla classe *j* al nodo *t*.
- Applicazione a classi sbilanciate: Maggiore costo per classificare erroneamente la classe minoritaria (positiva).

##### Approccio basato sul Campionamento:

- Obiettivo: Bilanciare la distribuzione delle classi nel training set.
- Undersampling: tecnica di bilanciamento delle classi tramite eliminazione di record.
Potenziale perdita di dati utili.
- Oversampling: Aggiunge record alla classe minoritaria.

##### Classificatori basati su Regole:

* Utilizzo di regole if-then ( *Condizione* -> *y* ).
* Condizione (antecedente): congiunzione di predicati sugli attributi.
* y (conseguente): etichetta della classe.
* Costruzione del modello: identificazione di un insieme di regole.
* Copertura di un'istanza x: soddisfazione dell'antecedente della regola r.

##### Copertura e Accuratezza delle Regole:

* **Copertura (Coverage(r))**: $\frac{|A|}{|D|}$ (frazione di record in D che soddisfano l'antecedente di r).
* **Accuratezza (Accuracy(r))**: $\frac{|A \cap y|}{|D|}$ (frazione di istanze che soddisfano sia l'antecedente che il conseguente di r).

##### Mutua Esclusività ed Esaustività delle Regole:

* **Mutua Esclusività:** ogni record coperto al più da una regola.
* **Esaustività:** ogni record coperto almeno da una regola.
* Insieme garantiscono che ogni istanza sia coperta da esattamente una regola.

##### Gestione della Mancanza di Mutua Esclusività ed Esaustività:

* **Mancanza di Mutua Esclusività:**
	* Soluzione 1: ordine di attivazione delle regole.
	* Soluzione 2: assegnazione alla classe con più regole attivate.
	* **Mancanza di Esaustività:** assegnazione a una classe di default.

##### Regole Linearmente Ordinate (Liste di Decisione):

* Ordinamento delle regole in base a una priorità.
* **Ordinamento rule-based:** in base alle qualità delle singole regole.
* **Ordinamento class-based:** gruppi di regole per classe, con ordinamento tra le classi.
* Rischio: regole di buona qualità superate da altre di qualità inferiore ma appartenenti a classi più importanti.

##### Costruzione di un Classificatore Basato su Regole

##### Metodi di Costruzione

* Metodi Diretti: Estraggono regole direttamente dai dati. Esempio: Sequential Covering.
* Metodi Indiretti: Estraggono regole dai risultati di altri metodi di classificazione.

##### Metodo Diretto: Sequential Covering

* Estrazione regole direttamente dai dati (ordinamento class-based).
* Algoritmo:
- Lista di decisioni R inizialmente vuota.
- Estrazione regola per classe y usando `Learn-one-rule`.
- Rimozione record coperti dalla regola dal training set.
- Aggiunta della regola a R.
- Ripetizione dal punto 2 fino al soddisfacimento del criterio di arresto (estensione regola).

##### Learn-one-rule

* Obiettivo: Trovare regola che massimizzi esempi positivi e minimizzi quelli negativi.
* Approccio Greedy:
- Regola iniziale $r:\{\} \to y$.
- Raffinazione della regola fino al criterio di arresto.
- Miglioramento dell'accuratezza aggiungendo coppie (Attributo, Valore) all'antecedente.
- $Accuracy(r)=\frac{nr}{n}$ (nr: istanze correttamente classificate; n: istanze che soddisfano l'antecedente).

##### Criteri di Valutazione delle Regole

* $LikelihoodRatio(r) = 2\sum_{i=1}^k f_i \log_2(\frac{f_i}{e_i})$
* Pota regole con scarsa copertura.
* $Laplace(r)=\frac{f_{+}+1}{n+k}$
* Pesa l'accuracy in base alla copertura.
* Copertura zero: probabilità a priori; copertura alta: asintoticamente all'accuracy; copertura bassa: indice diminuisce.
* $m\text{-}estimate(r) = \frac{f_+ + k p_+}{n + k}$
* Pesa l'accuracy in base alla copertura
* Caso speciale: per $p_+ = \frac{1}{k}$, coincide con la stima di Laplace.

##### FOIL (First Order Inductive Learner): Valutazione del Guadagno di Informazioni

* Misura la variazione di informazione aggiungendo un atomo ad una regola.
* Regola iniziale ($r_0$): $A \to +$ ( $p_0$ positivi, $n_0$ negativi)
* Regola estesa ($r_1$): $A, B \to +$ ( $p_1$ positivi, $n_1$ negativi)
* Guadagno di informazione:
$$ FoilGain(r_0, r_1) = p_1 \left( \log_2 \left( \frac{p_1}{p_1 + n_1} \right) - \log_2 \left( \frac{p_0}{p_0 + n_0} \right) \right) $$
* Misura alternativa:
$$ v(r_0, r_1) = \frac{p_1 - n_1}{p_1 + n_1} - \frac{p_0 - n_0}{p_0 + n_0} $$
* Favorisce regole con alta copertura e accuratezza.

##### Potatura delle Regole (Rule Pruning)

* Semplifica le regole per migliorare la generalizzazione (utile per approcci "greedy").
* **Reduced Error Pruning:**
	* Iterativamente rimuove l'atomo che migliora maggiormente l'errore sul validation set.
	* Termina quando nessuna rimozione migliora l'errore.

##### Metodi Diretti: RIPPER

* Algoritmo di classificazione basato su regole.
* Deriva un insieme di regole dal training set.
* Scala quasi linearmente con il numero di istanze.
* Robusto al rumore grazie all'utilizzo del validation set per evitare overfitting.

##### Classificazione:

##### Problemi a 2 classi:

- Una classe è positiva, l'altra di default.
- Apprendimento di regole per la classe positiva.

##### Problemi multi-classe:

- Classi ordinate per rilevanza (da *y<sub>1</sub>* meno rilevante a *y<sub>c</sub>* più diffusa).
- Regole costruite iterativamente, partendo dalla classe più piccola (*y<sub>1</sub>*).
- Esempi delle altre classi considerati negativi ad ogni iterazione.
- *y<sub>c</sub>* diventa classe di default.

##### Costruzione del Set di Regole (Sequential Covering):

##### Learn-one-rule:

- Inizia con una regola vuota.
- Aggiunge atomi che migliorano il *FOIL's Information Gain*.
- Iterazione fino a quando non vengono coperti più esempi negativi.
- **Pruning:** rimozione di atomi che massimizzano $v=\frac{p-n}{p+n}$

##### Ottimizzazione delle Regole:

- Per ogni regola *r*, si considerano due alternative: *r*<sup>*</sup> (regola nuova) e *r'* (regola estesa).
- Si sceglie l'alternativa che minimizza il *Minimum Description Length*.
- Iterazione per gli esempi positivi rimanenti.

**Condizione di Stop:** *Minimum Description Length principle*.

##### Metodi Indiretti: C4.5 rules

Trasformazione di un albero decisionale in un insieme di regole.
Ogni percorso radice-foglia diventa una regola.
Condizioni di test come predicati nell'antecedente.

* **Generazione Regole:**
	* Estrazione regole da albero decisionale (radice → foglie).
	* Potatura regole:
	* Rimozione atomi da antecedente (`A` → `A'`).
	* Mantenimento se errore pessimistico (`r'`) < errore (`r`).
	* Iterazione fino a errore minimo.
	* Rimozione regole duplicate.
* **Ordinamento Class-Based:**
	* Raggruppamento regole per classe (conseguente).
	* Calcolo *description length* per ogni classe: `L(error) + gL(model)`
	* `L(error)`: bit per errori di classificazione.
	* `L(model)`: bit per rappresentare il modello.
	* `g`: parametro dipendente da attributi ridondanti (default 0.5).
	* Ordinamento classi per *description length* crescente (priorità a minimo).

##### Tecniche di Classificazione - Nearest Neighbor

##### Eager Learners vs. Lazy Learners:

##### Algoritmi Lazy Learners

* **Rote Classifier:**
	* Memorizza il training set.
	* Classifica solo istanze identiche a quelle nel training set.
	* Problema: non classifica istanze non presenti nel training set.

* **Nearest Neighbor:**
	* **Approccio:** Classifica basandosi sulla similarità con i vicini nel training set.
	* **Rappresentazione:** Istanze come punti in uno spazio d-dimensionale.
* **Classificazione:**
- Calcola la distanza tra l'istanza di test e quelle del training set.
- Identifica i *k* vicini più prossimi.
- Assegna la classe più frequente tra i *k* vicini (opzionale: pesi basati sulla distanza).
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

##### Strutture a Indice per Ricerca del Vicino Più Prossimo

##### Tipi di Query:

* **Near neighbor range search:** Trova tutti i punti entro un raggio *r* da un punto *q*. Esempio: ristoranti entro 400m da un albergo.
* **Approximate Near neighbor:** Trova punti con distanza massima da *q* pari a (1 + ε) volte la distanza di *q* dal suo vicino più prossimo. Esempio: ristoranti più vicini ad un albergo.
* **K-Nearest-Neighbor:** Trova i *K* punti più vicini a *q*. Esempio: 4 ristoranti più vicini ad un albergo.
* **Spatial join:** Trova coppie di punti (p, q) con distanza ≤ *r*, dove *p* ∈ P e *q* ∈ Q. Esempio: coppie (albergo, ristorante) entro 200m.

##### Approcci:

##### Linear scan (Approccio Naïf):

* Calcola la distanza tra il punto di query e ogni punto nel database.
* Tempo di esecuzione O(dN), dove N è la cardinalità del database e d la dimensionalità.
* Non richiede strutture dati aggiuntive.
* Inefficiente per grandi dataset o alta dimensionalità.

##### Indici multi-dimensionali (spaziali):

- **B+tree multi-attributo:**
* Organizza tuple in base ai valori degli attributi.
* Non adatto a query che non specificano il valore di un attributo.

- **Quad-tree:**
* Divide ricorsivamente lo spazio in sottoquadri.
* Ricerca tramite esplorazione ricorsiva dei sottoquadri.
* Svantaggi:
* Punti vicini possono essere in celle diverse.
* Complessità temporale e spaziale esponenziale $O(n\cdot 2^d)$ rispetto alla dimensionalità *d*
* Non scala bene ad alta dimensionalità.

- **Kd-trees (k-dimensional tree):**
* Struttura dati ad albero per organizzare punti in uno spazio k-dimensionale.

##### Kd-tree: Struttura dati per la ricerca in spazi k-dimensionali

##### Struttura del Kd-tree:

* Ogni nodo: iper-rettangolo k-D con un "punto di separazione".
* Funzionamento: suddivide ricorsivamente lo spazio basandosi sulla posizione dei punti rispetto ai piani di separazione definiti dai punti dei nodi.
* Utilità: ottimizza ricerche di punti vicini (k-NN) e ricerche per intervallo.

##### Costruzione del Kd-tree:

* Selezione della dimensione: scelta ciclica o strategica dell'asse di partizione.
* Calcolo del valore mediano: determinazione del valore mediano lungo l'asse scelto per creare il piano di separazione.
* Divisione dei punti: suddivisione dei punti in sottoinsiemi (sinistra/destra) rispetto al piano.
* Ricorsione: ripetizione del processo sui sottoinsiemi fino a nodi con al massimo un punto o profondità massima raggiunta.

##### Ricerca in un Kd-tree:

* Partenza dalla radice: inizio dalla radice dell'albero.
* Discesa ricorsiva: percorso verso il basso seguendo il ramo appropriato in base alla posizione del punto di query rispetto al piano di separazione.
* Nodo foglia: raggiungimento di un nodo foglia (punto candidato).
* Risalita ricorsiva: controllo dei sottoalberi non esplorati per punti più vicini.
* Controllo della distanza: esplorazione dell'altro sottoalbero se la distanza dal piano di separazione è minore della distanza dal miglior punto trovato finora.
* Continuazione della risalita: altrimenti, si prosegue la risalita senza esplorare l'altro sottoalbero.

##### Pro e Contro dei Kd-trees:

* **Pro:**
	* Partizione efficiente dello spazio k-D.
	* Query k-NN efficienti.
	* Query di range/intervallo.
* **Contro:**
	* Inefficienza ad alte dimensionalità ("maledizione della dimensionalità").
	* Prestazioni dipendenti dalla scelta dell'asse di partizione e del punto di separazione.
	* Scarsa performance su dati distorti o con distribuzioni complesse.

##### Calcolo Approssimato per Ricerca di Vicini

* **Algoritmi di Ricerca Approssimativa:**
	* **Near Neighbor Approssimativo:** Restituisce punti con distanza ≤ c * distanza punti più vicini (c > 1). Vantaggioso quando l'approssimazione è sufficientemente accurata e la metrica di distanza riflette la qualità percepita dall'utente.
* **Locality-Sensitive Hashing (LSH):**
	* Tecnica probabilistica per dataset ad alta dimensionalità.
	* Crea funzioni hash che mappano oggetti simili in hash simili, massimizzando le collisioni (al contrario delle HashMap).
	* Ricerca solo tra elementi con lo stesso hash.

* **Gestione Memoria Secondaria per Dataset Grandi:**
* **Strutture dati disk-based:**
	* **R-tree:** Approccio ottimistico (tempo logaritmico).
	* **Vector Approximation File:** Approccio pessimistico (scansione veloce dell'intero dataset).

##### R-tree

* **Struttura dati:** Estensione dei B+tree per spazi multidimensionali, organizza oggetti in iperrettangoli sovrapposti.
* **Costruzione (bottom-up):**
- Raggruppamento oggetti (2-3 elementi).
- Calcolo rettangolo minimo per ogni gruppo.
- Unione ricorsiva dei rettangoli in nodi intermedi fino alla radice.
* **Pro:**
	* Ricerca del vicino più vicino.
	* Funziona per punti e rettangoli.
	* Evita spazi vuoti.
	* Varianti (X-tree, SS-tree, SR-tree).
	* Efficiente per basse dimensioni.
* **Contro:**
	* Inefficiente per alte dimensioni.

