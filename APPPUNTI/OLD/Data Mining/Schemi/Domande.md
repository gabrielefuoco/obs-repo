#### Domande Generali sulla Classificazione
* Descrivi l'obiettivo, il training set e le principali tipologie di un modello di classificazione.

#### Alberi Decisionali
* Spiega come un albero decisionale rappresenta le regole, il ruolo dei nodi e la sua complessità computazionale.  Dettaglia il processo di applicazione a nuovi dati, le tecniche *greedy* e l'algoritmo di Hunt (inclusa la ricorsione).

#### Scelta del Criterio di Split negli Alberi Decisionali
* Descrivi i criteri di split negli alberi decisionali e come vengono gestiti gli attributi (binari, nominali, ordinali e continui), inclusi split a due e più vie, la partizione binaria e la discretizzazione. Spiega l'obiettivo del criterio di ottimizzazione, i rischi degli alberi profondi e il ruolo delle misure di impurità.

#### Misure di Impurità dei Nodi
* Cosa misurano le misure di impurità e come si calcolano? Descrivi Gini Index, Entropy e Misclassification Error (formula, valori massimo/minimo, calcolo per singolo e multipli nodi, applicazione ad attributi binari/continui, guadagno, *SplitInfo*).  Quali sono i criteri di stop e i pro/contro della classificazione con alberi decisionali?

#### Model Overfitting
* Cos'è il *model overfitting*? Come influisce la complessità del modello? Descrivi il comportamento di *training error* e *test error*,  il principio di *Occam's Razor* e tre metriche per stimare l'errore di generalizzazione.

#### Generalization Error
* Cos'è il *generalization error*? Descrivi gli approcci ottimistico e pessimistico (formula e parametri) e il principio del *Minimum Description Length*.

#### Pruning: Strategie di Selezione dei Modelli
* Descrivi *prepruning* e *postpruning* (vantaggi/svantaggi e potatura *bottom-up*).

#### Costruzione dei Dataset tramite Partizionamento
* Descrivi *Holdout*, *Random Subsampling*, *Cross Validation* (vantaggi/svantaggi, funzionamento, calcolo performance media) e *Bootstrap* (probabilità di inclusione nel training set).

#### Class Imbalance Problem
* Cos'è il *Class Imbalance Problem*? Quali sfide pone e perché l'accuratezza non è una metrica adatta?

#### Metriche Alternative per la Valutazione del Modello
* Descrivi la matrice di confusione, accuratezza, *error rate*, *precision*, *recall* e F-measure.

#### Tecniche per il Trattamento di Dataset Sbilanciati
* Descrivi la classificazione *Cost Sensitive* (formula costo), e l'approccio basato sul campionamento (*oversampling* e *undersampling*).

#### Tecniche di Classificazione: Regole di Decisione
* Come si esprime una regola di classificazione (antecedente, conseguente, copertura)?

#### Copertura e Accuratezza
* Definisci copertura e accuratezza di una regola (significato di |A|, |A ∩ y| e |D|).

#### Regole Mutuamente Esclusive e Esaustive
* Cosa sono le regole mutuamente esclusive ed esaustive?  Quali sono le conseguenze della loro mancanza e le possibili soluzioni?

#### Regole Linearmente Ordinate
* Cosa sono le regole linearmente ordinate e le liste di decisione (*rule-based* e *class-based*)?

#### Costruzione di un Classificatore Basato su Regole
* Descrivi i metodi diretti (es. *Sequential Covering*, *Learn-one-rule*) e indiretti.

#### Criteri di Valutazione delle Regole
* Descrivi i criteri *Likelihood Ratio*, *Laplace*, *M-Estimate*, *FOIL* (*FOIL Gain* e *v(r0, r1)*).

#### Rule Pruning
* Descrivi il *Reduced error pruning*.

#### Metodi Diretti: RIPPER
* Descrivi RIPPER (2 classi, multi-classe, costruzione set di regole, estensione, *pruning*, ottimizzazioni).

#### Metodi Indiretti: C4.5 rules
* Descrivi C4.5 rules (estrazione regole da albero, *pruning*, ordinamento *Class-Based*).

#### Vantaggi e Svantaggi dei Classificatori Rule-Based
* Elenca vantaggi e svantaggi dei classificatori *Rule-Based*.

#### Differenze tra Alberi Decisionali e Regole
* Quali sono le principali differenze?

#### Tecniche di Classificazione - Nearest Neighbor
* Spiega *Eager/Lazy Learners*, *Rote Classifier*, *Nearest Neighbor* (processo, parametri, scelta di k, *pre-processing*, pro/contro, ottimizzazione costo).

#### Strutture a Indice
* Descrivi i tipi di query, *Linear scan* (complessità), indici multi-dimensionali (*B+tree*, *Quad-tree*, svantaggi).

#### Kd-trees (k-dimensional tree)
* Descrivi struttura, costruzione e ricerca nei *Kd-trees* (pro/contro).

#### Calcolo Approssimato
* Descrivi la ricerca approssimata del vicino e *Locality-Sensitive Hashing*.

#### Utilizzo Memoria Secondaria
* Descrivi *R-tree* e *Vector Approximation File* (*disk-based*, costruzione, pro/contro).

#### Classificatori Bayesiani - Teoria della Probabilità
* Perché la probabilità è utile nella classificazione? Definisci variabile casuale, probabilità (due visioni), probabilità congiunta, marginalizzazione e probabilità marginale.

#### Probabilità Condizionata e Teorema di Bayes
* Definisci la probabilità condizionata (formule, relazione con la congiunta) e il Teorema di Bayes (formula, significato, proprietà della probabilità).

#### Obiettivo della Classificazione e Teorema di Bayes (Classificazione)
* Qual è l'obiettivo della classificazione in termini probabilistici (probabilità a posteriori)? Descrivi il Teorema di Bayes applicato alla classificazione (formula, significato di P(Y|X), P(X|Y), P(Y), P(X)).

#### Stima delle Probabilità e Assunzione di Naïve Bayes
* Come si stima P(Y)?  Perché calcolare P(X|Y) direttamente è problematico? Cos'è l'indipendenza condizionale e come semplifica il calcolo di P(X|Y)?

#### Classificatore Naïve Bayes e Formule per il Calcolo delle Probabilità
* Descrivi il funzionamento del classificatore Naïve Bayes. Fornisci le formule per P(Yj) e P(Xi|Yj) per attributi categorici (significato di N, nj, |Xij|).

#### Indipendenza Condizionale e Applicazione al Naive Bayes
* Definisci l'indipendenza condizionale tra X, Y, Z (relazione tra probabilità congiunta e condizionata). Come si applica al Naïve Bayes?

#### Stima della Probabilità per Attributi Continui e Problemi con Probabilità Zero
* Come gestire gli attributi continui (due strategie, distribuzione normale)?  Cos'è il problema delle probabilità zero e quando si verifica?  Descrivi le soluzioni (Laplace, *m-estimate*).

#### Reti Bayesiane
* Cosa sono le reti bayesiane (nodi, archi)?  Cos'è la condizione di Markov (locale, vantaggi rispetto a Naïve Bayes)? Come si calcola la distribuzione di probabilità congiunta (formula)? Cosa sono le CPT (utilizzo, numero di voci)?  Come si calcola P(Y|x) (formula per P(x, y))?

#### Regressione Lineare
* Descrivi l'obiettivo della regressione (*y*, *X*, *b*). Scrivi l'equazione e il problema chiave.  Come si stimano i parametri (funzione di costo)? Descrivi la discesa del gradiente (passi, formule di aggiornamento, α, convergenza, requisiti). Cos'è il *feature scaling* (*mean normalization*)?

#### Regressione Logistica
* Cos'è e a cosa serve la regressione logistica? Descrivi la funzione logistica (formula, utilizzo). Perché la funzione di costo quadratica non è adatta? Qual è la formula corretta?  Descrivi le formule di aggiornamento dei parametri (univariata e multivariata, similitudini/differenze con regressione lineare).  Scrivi la formula della logistica multivariata (significato dei parametri).

#### Reti Neurali Artificiali (ANN)
* Descrivi il funzionamento (pesi, attivazione), le componenti, le decisioni di progettazione e le caratteristiche principali.

#### Perceptron
* Descrivi architettura (formula, forma vettoriale, pesi, *bias*), addestramento (passi, γ, λ, xj, aggiornamento pesi basato sull'errore), obiettivo, iperpiano di separazione e convergenza (condizioni, limitazioni, teorema).

#### Reti Neurali Multilivello
* Perché sono più complesse dei perceptron? Descrivi gli strati (nascosti, *feedforward*), il problema XOR, l'apprendimento di caratteristiche e le funzioni di attivazione (gradino, sigmoide, tangente iperbolica, RELU, formule, caratteristiche, vantaggi, *Vanishing Gradient Problem*). Descrivi la funzione Soft Max (formula, scopo).

#### Backpropagation
* Cosa misura la funzione di perdita (formula errore quadratico medio)? Qual è l'obiettivo dell'addestramento? Perché la discesa del gradiente può convergere a minimi locali? Descrivi le formule di aggiornamento e il processo di *backpropagation* (formula derivate parziali).  Elenca le caratteristiche e le limitazioni delle ANN.

#### Embedding
* Definisci l'*embedding* (vantaggi). Descrivi *1-Hot Encoding* (limiti), n-gram (similarità), *Word Embedding* (algoritmi) e *Sentence Embedding*.

#### Support Vector Machine (SVM)
* Descrivi le SVM (obiettivo, vettori di supporto, iperpiano, problemi non lineari, vantaggi).

#### Tecniche di Ensemble
* Descrivi l'obiettivo, il funzionamento, le condizioni per migliorare le prestazioni e il ruolo della combinazione dei voti. Perché l'indipendenza assoluta è difficile?  Descrivi il metodo "Manipulate data distribution".

#### Bagging (Bootstrap Aggregating) e Boosting
* Descrivi *Bagging* (passi, percentuale dati, miglioramento generalizzazione, limiti) e *Boosting* (aggiornamento pesi, combinazione predizioni, differenze con *Bagging*).

#### AdaBoost e Gradient Boosting
* Descrivi *AdaBoost* (passi, αi, aggiornamento wi, previsione finale) e *Gradient Boosting* (addestramento modelli deboli, combinazione, iperparametri).

#### Costruzione classificatori ensemble e Random Forest
* Descrivi *Random Forest* (passi, parametro p, randomizzazione) ed *Error-Correcting Output Codes* (passi, classificazione).

#### Analisi delle Regole Associative
* Descrivi l'obiettivo, le "regole che indicano la presenza di un elemento", gli "itemset frequenti", le "regole di associazione" e le "variabili binarie asimmetriche".

#### Itemset e Support Count e Regole di Associazione
* Descrivi l'obiettivo dell'individuazione degli itemset frequenti. Definisci "insieme di elementi", "insieme di transazioni", "itemset", "support count", "support" e "frequent itemset" (*minsup*). Definisci "regola di associazione" (X, Y), "supporto" e "confidenza" (formule). Qual è l'obiettivo? Descrivi l'approccio *brute-force* (limiti) e un approccio efficiente.

#### Scoperta delle Regole di Associazione e Frequente Itemset Generation
* Descrivi il problema (*minsup*, *minconf*), l'importanza dell'efficienza e il significato delle formule. Descrivi l'approccio *brute-force* (complessità, strategie di riduzione).

#### Misure di Validità delle Regole e Il Principio Apriori
* Quali sono le due misure di validità? Enuncia e formalizza il principio Apriori (antimonotonia, miglioramento efficienza, esempio, dimostrazione efficacia *pruning*).

#### Algoritmo Apriori e Generazione di Candidati
* Descrivi l'algoritmo Apriori (fasi, pseudocodice) e la generazione di candidati (*brute-force*, *pruning*, metodi $F_{k−1}×F_{k−2}$ e $F_{k−1}×F_{k−1}$, esempio, alternative, principio generale per l'unione, vantaggi).

#### Generazione di Regole Associative e Complessità degli Algoritmi
* Descrivi il processo di partizionamento (corpo, testa, confidenza, anti-monotonicità, principio Apriori, complessità). Quali sono le due sfide principali e i fattori che influenzano la complessità? Descrivi la complessità del calcolo del supporto, l'ottimizzazione con *hash* (struttura *hash tree*, *matching*, vantaggi), la formula per il numero di itemset totali.

#### Rappresentazione Compatta degli Itemset Frequenti e Valutazione delle Regole
* Spiega il problema della complessità. Definisci "itemset massimale" (vantaggi, principio Apriori) e "itemset chiuso". Quali misure si usano per valutare le regole (problemi)?

#### Tabella di Contingenza e Indipendenza Statistica
* Descrivi la tabella di contingenza (calcolo supporto). Qual è il criterio importante basato su confidenza e supporto?  Scrivi la condizione per una regola interessante. Cosa significa correlazione positiva, negativa e indipendenza? Descrivi *Lift*, *Interest* e ϕ-*coefficient* (formule, *pruning*).

#### Proprietà delle Metriche e Paradosso di Simpson
* Descrivi le proprietà delle metriche (simmetria, variazione di scala, correlazione, casi nulli, esempi). Descrivi il paradosso di Simpson (esempio, stratificazione).

#### Effetti della Distribuzione del Supporto, Cross-Support e H-confidence
* Spiega l'influenza della distribuzione del supporto (esempio dataset non omogenei, *cross-support*, soluzione, formula, interpretazione). Descrivi *H-confidence* (formula, calcolo, vantaggi), *h-confidence* (formula, relazione con *cross-support*, *hyperclique* chiusi/massimali, utilizzo per trovare itemset con supporti comparati).

#### Attributi Categorici e Continui e Discretizzazione
* Come gestire gli attributi categorici (trasformazione in binari, problemi, mitigazione)? Come gestire gli attributi continui? Descrivi le tecniche di discretizzazione (*equal-width*, *equal-depth*, *clustering*, problemi intervalli ampi/stretti, regole ridondanti, evitamento) e l'approccio statistico.  Descrivi l'estrazione di regole con attributo target (binarizzazione, *Z-test* per la significatività) e *minApriori* (calcolo supporto).

#### Regole Associative Multi-Livello e Conseguenze
* Spiega il concetto di multi-livello (conseguenze, soluzione alternativa).

#### Analisi dei Cluster
* Definisci l'analisi dei cluster (gruppi coesi/separati, differenza con classificazione). Spiega clustering, divisione, segmentazione, partizionamento (sovrapposizioni/differenze). Descrivi gli approcci (*Partitional, Hierarchical, ecc.* - caratteristiche, vantaggi/svantaggi, esempi). Spiega clustering esclusivo/non esclusivo (esempi), *fuzzy* (esempio), parziale/completo (esempi), eterogeneo/omogeneo (esempi). Descrivi i tipi di cluster (*Well Separated, ecc.* - caratteristiche, vantaggi/svantaggi, esempi dataset). Spiega la funzione di ottimizzazione (NP-hard, SSE, interpretazione). Come si valuta il clustering partizionale (distanza euclidea, limiti SSE)?

#### Metodi Basati sul Partizionamento e K-means
* Descrivi le differenze tra *K-means* e *K-medoid* (situazioni di utilizzo). Descrivi *K-means* (parole tue, pseudocodice, complessità, misure di vicinanza, determinazione centroide in *K-medoid*, limitazioni, mitigazione debolezze, importanza/conseguenze scelta centroidi iniziali). Descrivi le varianti di *K-means*. Descrivi le euristiche per la scelta dei centroidi (*furthest centre* - vantaggi/svantaggi, mitigazione outlier, altre soluzioni). Descrivi *K-means++* (vantaggi). Come gestire i cluster vuoti (due approcci)? Descrivi l'aggiornamento incrementale dei centroidi (vantaggi/svantaggi). Descrivi tecniche di *pre/post-processing*. Descrivi *Bisecting K-means* (differenze con *K-means*). Descrivi *K-modes* (differenze con *K-means*).

#### Clustering Gerarchico
* Spiega agglomerativo/divisivo (vantaggi, visualizzazione risultati). Descrivi l'algoritmo base (aggiornamento matrice prossimità). Descrivi le definizioni di distanza inter-cluster (MIN, MAX, ecc. - differenze grafi/prototipi). Descrivi le limitazioni dell'agglomerativo, il divisivo (efficienza).

#### DBSCAN
* Descrivi il principio, la classificazione dei punti (core, border, noise) e l'algoritmo (formale, complessità, vantaggi/svantaggi, selezione parametri - metodo per *Eps*).

#### Valutazione dei Cluster
* Spiega indici esterni/interni (esempi). Descrivi la valutazione tramite correlazione (limiti, interpretazione grafici), la curva SSE. Definisci coesione/separazione (tre modi per misurarle). Descrivi il *Silhouette Coefficient* (calcolo, interpretazione).

#### Altre tecniche di clustering
* Descrivi *fuzzy clustering* (differenza con *hard*), *K-Means Fuzzy* (SSE), *grid-based* (algoritmo, vantaggi/svantaggi), clusterizzazione in sottospazi. Descrivi il clustering gerarchico divisivo basato su grafi, *Shared Nearest Neighbour (SNN)* e *SNN Density Based Clustering*.
