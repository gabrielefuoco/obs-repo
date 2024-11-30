**Un modello di classificazione**

* Qual è l'obiettivo principale di un modello di classificazione?
* Cosa si intende per training set e test set nel contesto dei modelli di classificazione?
* Quali sono le due principali categorie di classificatori?  Fornire esempi.


**Alberi decisionali**

* Come rappresenta un albero decisionale un insieme di regole?
* Qual è la differenza tra nodi interni e nodi foglia in un albero decisionale?
* Perché la ricerca di un albero ottimo è un problema NP-Completo?
* Qual è la complessità computazionale della classificazione usando un albero decisionale nel caso peggiore?
* Qual è il vantaggio degli alberi decisionali rispetto alla presenza di attributi fortemente correlati?
* Descrivere il processo di applicazione di un modello di albero decisionale a un nuovo dataset.


**Tree Induction Algorithm**

* Cosa si intende per tecniche greedy nella costruzione di un albero decisionale?
* Quali sono i problemi che gli algoritmi di tree induction devono affrontare?
* Descrivere l'algoritmo di Hunt per la costruzione di un albero decisionale.  Quali sono i suoi passi principali?


**Scelta del criterio di split negli alberi decisionali**

* Come vengono gestiti gli attributi binari, nominali, ordinali e continui nella scelta del criterio di split?
* Descrivere la partizione binaria per attributi ordinali. Quali sono i suoi vantaggi e svantaggi?
* Come vengono gestiti gli attributi continui negli split a più vie?
* Cosa si intende per discretizzazione nel contesto degli attributi continui?


**Criterio di ottimizzazione dello split**

* Qual è l'obiettivo principale nella scelta del criterio di split?
* Cosa si intende per nodi puri e nodi impuri?
* Perché alberi più profondi sono più soggetti a overfitting?
* Qual è il ruolo delle misure di impurità nella costruzione di un albero decisionale?


**Misure di impurità dei nodi**

* Descrivere il Gini Index, l'Entropy, e il Misclassification Error.  Quali sono le loro formule?
* Cosa rappresenta p(j|t) nelle formule delle misure di impurità?
* Come si calcola l'impurità complessiva di uno split?
* Come si determina il partizionamento migliore tra diversi possibili split?  Cosa si intende per guadagno?


**GINI Index**

* Qual è la formula del Gini Index?
* Quali sono i valori massimo e minimo del Gini Index?  In quali situazioni si verificano?
* Come si calcola l'indice di Gini per un nodo in uno split a due vie?
* Come si calcola l'indice di Gini per più nodi in uno split a più vie?
* Come si calcola il GAIN per attributi binari e continui usando il Gini Index?


**Entropia**

* Qual è la formula dell'Entropia?
* Quali sono i valori massimo e minimo dell'Entropia?  In quali situazioni si verificano?
* Come si calcola il guadagno di informazione usando l'Entropia?
* Cosa rappresenta il GainRatio e come viene calcolato?  A cosa serve?


**Classification Error**

* Qual è la formula del Classification Error?
* Quali sono i valori massimo e minimo del Classification Error?  In quali situazioni si verificano?


**Criteri per interrompere lo split**

* Elencare tre criteri per interrompere lo split durante la costruzione di un albero decisionale.


**Pro della classificazione con gli alberi decisionali**

* Elencare tre vantaggi della classificazione con gli alberi decisionali.


**Contro**

* Elencare due svantaggi della classificazione con gli alberi decisionali.


**Model Overfitting**

* Cosa si intende per overfitting in un modello di classificazione?
* Perché l'errore di classificazione sul training set non fornisce stime accurate sul comportamento dell'albero su record sconosciuti?
* Come si relazionano il training error e il test error in caso di overfitting?
* Cosa si intende per alta complessità del modello?


**Alta complessità del modello**

* Elencare tre metriche di stima dell'errore di generalizzazione.


**Generalization error**

* Cosa rappresenta e'(t)?
* Descrivere l'approccio ottimistico e l'approccio pessimistico per stimare gli errori di generalizzazione.  Quali sono le formule?
* Cosa rappresenta la Minimum Description Lenght?  Come viene calcolato il costo totale?


**Pruning: strategie di selezione dei modelli**

* Descrivere le tecniche di prepruning e postpruning.  Quali sono i vantaggi e gli svantaggi di ciascuna?


**Costruzione dei dataset tramite partizionamento**

* Descrivere le tecniche Holdout, Random Subsampling, Cross Validation e Bootstrap.  Quali sono i vantaggi e gli svantaggi di ciascuna?


**Class Imbalance Problem**

* Cosa si intende per Class Imbalance Problem?
* Quali sono le due sfide principali poste dal Class Imbalance Problem?
* Perché l'accuratezza non è adatta per valutare i modelli in presenza di squilibrio?


**Metriche alternative per la valutazione del modello**

* Descrivere la matrice di confusione e i suoi quattro indicatori.
* Qual è la formula dell'accuratezza e dell'error rate?
* Qual è la formula della Precision e del Recall?
* Qual è la formula della F-measure?


**Tecniche per il trattamento di dataset sbilanciati**

* Descrivere le tecniche di classificazione Cost Sensitive.  Qual è la formula del costo complessivo?
* Descrivere le tecniche basate sul campionamento (oversampling e undersampling).  Quali sono i vantaggi e gli svantaggi di ciascuna?


**Tecniche di classificazione: Regole di decisione**

* Come si rappresenta una regola di classificazione?
* Cosa si intende per antecedente e conseguente di una regola?
* Cosa significa che una regola copre un'istanza?


**Copertura e accuratezza**

* Quali sono le formule per la copertura e l'accuratezza di una regola?
* Cosa rappresentano |A|, |A ∩ y|, e |D| nelle formule?


**Regole mutuamente esclusive e esaustive**

* Cosa si intende per regole mutuamente esclusive e regole esaustive?


**Mancanza di mutua esclusività e esaustività**

* Descrivere due soluzioni per la mancanza di mutua esclusività.
* Come si gestisce la mancanza di esaustività?


**Regole linearmente ordinate**

* Cosa si intende per regole linearmente ordinate e lista di decisione?
* Descrivere l'ordinamento rule-based e l'ordinamento class-based.


**Costruzione di un Classificatore Basato su Regole**

* Descrivere i metodi diretti e indiretti per la costruzione di un classificatore basato su regole.


**Metodo diretto: Sequential Covering**

* Descrivere l'algoritmo Sequential Covering.


**Learn-one-rule**

* Descrivere l'algoritmo Learn-one-rule.  Qual è il suo obiettivo?


**Criteri di valutazione delle regole**

* Descrivere il Likelihood Ratio, Laplace, M-Estimate e FOIL.  Quali sono le loro formule?


**Rule pruning**

* Descrivere la tecnica del Reduced error pruning.


**Metodi diretti: RIPPER**

* Descrivere l'algoritmo RIPPER.  Come gestisce i problemi a due classi e multi-classe?
* Come costruisce il set di regole?
* Come estende una regola?
* Come esegue il pruning?


**Metodi indiretti: C4.5 rules**

* Descrivere l'algoritmo C4.5 rules.  Quali sono i suoi passi principali?
* Come vengono ordinate le classi in C4.5rules?  Cosa si intende per description length?


**Vantaggi e svantaggi dei classificatori Rule-Based**

* Elencare i vantaggi e gli svantaggi dei classificatori Rule-Based.


**Differenze tra alberi decisionali e regole**

* Elencare due differenze principali tra alberi decisionali e regole.


**Tecniche di Classificazione - Nearest Neighbor**

* Qual è la differenza tra Eager Learners e Lazy Learners?  Fornire esempi.
* Descrivere il Rote Classifier.  Qual è il suo problema principale?
* Descrivere l'approccio Nearest Neighbor.  Quali sono i suoi passi principali?
* Quali sono i parametri del Nearest Neighbor?
* Come si sceglie il valore di k?
* Quali sono le fasi di pre-processing necessarie?


**Pro e Contro del Nearest Neighbor**

* Elencare i pro e i contro del Nearest Neighbor.


**Ottimizzazione del Costo di Calcolo**

* Descrivere due tecniche per ottimizzare il costo di calcolo del Nearest Neighbor.


**Strutture a Indice**

* Descrivere i diversi tipi di query per la ricerca dei vicini più vicini.
* Descrivere l'approccio Linear scan.  Qual è la sua complessità temporale?


**Indici multi-dimensionali (spaziali)**

* Descrivere il B+tree multi-attributo e il Quad-tree.  Quali sono i vantaggi e gli svantaggi del Quad-tree?


**Kd-trees (k-dimensional tree)**

* Descrivere la struttura dati di un Kd-tree.
* Descrivere il processo di costruzione di un Kd-tree.
* Descrivere il processo di ricerca in un Kd-tree.
* Qual è la metrica di distanza comunemente utilizzata con i Kd-tree?


**Pro e Contro dei Kd-trees**

* Elencare i pro e i contro dei Kd-trees.


**Calcolo Approssimato**

* Cosa si intende per algoritmo di ricerca del vicino più vicino approssimativo?


**Locality-Sensitive Hashing (LSH)**

* Descrivere la tecnica LSH.  Qual è il suo obiettivo principale?


**Utilizzo Memoria Secondaria**

* Descrivere le strutture dati disk-based R-tree e Vector Approximation File.


**R-tree**

* Descrivere la struttura dati R-tree e il suo processo di costruzione.
* Elencare i pro e i contro dell'R-tree.

