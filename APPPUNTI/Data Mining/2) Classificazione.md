
Un modello di classificazione costruisce una mappatura dagli attributi di un record alla sua classe, utilizzando una collezione di record etichettati come training set.

_L'obiettivo è di assegnare i record non noti a una classe nel modo più accurato possibile._ Per farlo, viene utilizzato il training set per costruire il modello e il test set per validarlo.

I classificatori si dividono in classificatori di base e classificatori Ensemble _(Boosting, bagging, random forest)._

## Alberi decisionali

Un albero decisionale è una tecnica di classificazione che rappresenta un insieme di regole attraverso una struttura gerarchica di domande sugli attributi del test da classificare, consentendo di determinarne la classe di appartenenza tramite un percorso di risposte.

Un albero decisionale è composto da:

* **Nodi interni o di partizionamento:** rappresentano gli _attributi di splitting_ sui quali vengono poste le domande.
* **Nodi foglia o terminali:** associati a un valore dell'attributo di classe, che determinano la classificazione finale.

* La ricerca di un albero ottimo è un problema NP-Completo.
* La classificazione usando un albero decisionale è estremamente veloce e il caso peggiore è O(ω), dove ω è la profondità dell'albero.
* Gli alberi di decisione sono robusti rispetto la presenza di attributi fortemente correlati.

### Applicare il modello al data set
Partendo dal nodo radice, applichiamo la condizione di test dell'attributo associato e seguiamo il percorso appropriato. Raggiunto un nodo foglia, assegniamo il valore dell'attributo di classe associato all'istanza del test.

### Tree Induction Algorithm
A partire da un solo dataset è possibile costruire una moltitudine di alberi decisionali, ma alcuni saranno migliori di altri.  
La strategia impiegata si basa su **tecniche greedy,** ovvero la costruzione dell'albero avviene dall'alto verso il basso, prendendo una serie di decisioni ottimali a livello locale. Questi algoritmi devono tenere conti di problemi come la scelta dei criteri di split e di stop, l'underfitting e l'overfitting.

#### Algoritmo di Hunt
Attua un approccio ricorsivo che suddivide progressivamente un insieme di record Dt in insiemi di record via via più puri.

* La _procedura di costruzione_ di un albero decisionale a partire da un training set Dt con possibili etichette di classe yt = y1, ..., yk è la seguente:
    1. Se Dt contiene record di una sola classe yj, il nodo t diventa una foglia con etichetta yj.
    2. Altrimenti, si sceglie un attributo e un criterio per suddividere Dt in sottoinsiemi non vuoti.
    3. Si applica ricorsivamente la stessa procedura sui sottoinsiemi ottenuti.

Questa procedura ricorsiva costruisce l'albero decidendo per ogni nodo interno l'attributo discriminante per la suddivisione, fino a quando non si ottengono nodi foglia con un'unica classe assegnata.

## Scelta del criterio di split negli alberi decisionali

La costruzione di un albero decisionale prevede la scelta di un criterio di split, ovvero la definizione di come le istanze vengono distribuite sui nodi foglia. Questo processo dipende dal tipo di attributo utilizzato per la divisione:

* **Attributi binari:** Generano due possibili risultati.
* **Attributi nominali:** Possono essere divisi in due modi:
    * **Split a più vie:** Crea una partizione per ogni valore distinto dell'attributo.
    * **Split a due vie:** Suddivide in modo ottimale i valori dell'attributo in due insiemi.
* **Attributi ordinali:** Simili agli attributi nominali, ma preservano l'ordinamento dei valori.
* **Attributi continui:** Possono essere divisi in modo binario o a più vie, a seconda del test di comparazione utilizzato:
    * **Binario (split a due vie):** Test del tipo $A< v$.
    * **A più vie:** Suddivisione in intervalli di valori del tipo $v_{i}\leq A\leq v_{i}+1$.

Per gli **split a due vie sugli attributi ordinali**, si considerano tutti i possibili valori v tra il minimo e il massimo dell'attributo nel training set per costruire il test $A< v$. Questo approccio, chiamato partizione binaria, può essere computazionalmente costoso.

Per gli **split a più vie**, si discretizzano i valori continui in intervalli disgiunti che coprono l'intera gamma di valori. La discretizzazione richiede di definire il numero di intervalli e la posizione dei punti di divisione. I valori di uno stesso intervallo vengono mappati alla stessa categoria ordinale.

##@ Criterio di ottimizzazione dello split

La scelta del criterio di split migliore si basa su misure di bontà che mirano a creare partizioni (nodi figli) il più pure possibile, associando istanze della stessa classe allo stesso nodo. Nodi impuri, con istanze di classi diverse, tendono ad aumentare la profondità dell'albero, richiedendo ulteriori partizionamenti.

Alberi più profondi sono più soggetti a overfitting, meno interpretabili e computazionalmente più costosi. Per evitare eccessive suddivisioni, si introducono misure di impurità dei nodi per guidare la costruzione dell'albero verso partizioni ottimali.

In sintesi, le misure di impurità permettono di bilanciare la purezza dei nodi con la complessità dell'albero, conducendo ad alberi decisionali ottimali.


## Misure di impurità dei nodi
* Misurano quanto sono diversi i valori di classe contenuti in un solo nodo.
* Per valutare l'impurità di un nodo *t*, con record appartenenti a *k classi* e con *n nodi figli*, si usano:

    * **Gini Index:**  $GINI(t) = 1 - \sum_{j=1}^{k} [p(j|t)]^2$
    * **Entropy:** $Entropy(t) = -\sum_{j=1}^{k} p(j|t) \log_2 p(j|t)$
    * **Misclassification Error:** $Error(t) = 1 - \max p(i|t)$
    * **Impurità complessiva:** $Impurity_{split} = \sum_{i=1}^{n} \frac{m_i}{m} meas(i)$

    * _p(j|t) rappresenta la frequenza delle istanze del training set della classe j nel nodo t_

### Determinare il partizionamento migliore

* Calcolare il grado di impurità *P* del nodo genitore prima dello splitting.
* Calcolare l'impurità pesata *M* dei nodi figli dopo lo splitting.
* Scegliere la condizione di split che massimizza il guadagno, definito come $Gain=P-M$.

_Il guadagno è una misura non negativa, in quanto P>M._

### GINI Index

$GINI(t)=1-\sum_{j=1}^k[p(j|t)]^2$

*dove p(j|t) è la frequenza relativa della classe j al nodo t.*

* **Massimo:** $\left( 1-\frac{1}{nc} \right)$ ottenuto quando i record sono distribuiti equamente.
* **Minimo:** (0) ottenuto quando i record appartengono a una sola classe.

_dove nc rappresenta il numero di classi_

#### Calcolare indice di GINI per un nodo

Quando è necessario effettuare lo split a due vie, l'indice di GINI è definito come:  
$1-(P_{1})^-(P_{2}^2)$

#### Calcolare indice di GINI per più nodi
$$GINI_{split}=\sum_{i=1}^k k \frac{n_{i}}{n} GINI(i)$$
$$\frac{\text{numero di oggetti presenti nel nodo corrente}}{\text{numero di oggetti totali}}$$

#### Attributi binari:

$GAIN=GINI(P)-GINI_{split}$

#### Attributi continui:

Ordino le tuple in base all'attributo che voglio partizionare.

_Per ogni coppia di valori contigui ne scelgo uno intermedio. In base ai valori ottenuti provo i partizionamenti e scelgo quello con l'indice di gini minore._

### Entropia

$Entropy(t)=- \sum_{j=1}^k p(j|t)\log_{2}(p(j|t))$

_dove t rappresenta il nodo e p(j|t) è la frequenza relativa alla classe j nel nodo t_

Dato nc=numero di classi, allora possiamo definire:

* **Massimo = (log2(nc))** quando i record sono equamente distribuiti
$$=n_{c} \frac{1}{n_{c}}\log_{2}\left( \frac{1}{n_{c}} \right)=-\log_{2}\left( \frac{1}{n_{c}} \right)=\log_{2}(n_{c})$$
* **Minimo = (0)** quando i record appartengono alla stessa classe.

#### Guadagno

$GAIN_{split}=Entropy(p)-\sum_{i=1}^n \frac{m_{i}}{m}Entropy(i)$

dove p rappresenta un nodo contenente m record in n nodi figli, ognuno con mi record.

#### Massimizzare il guadagno

$GainRatio_{split}=\frac{Gain_{split}}{SplitInfo}$
$SplitInfo=-\sum_{i=1}^n {\frac{m_{i}}{m}}\log_{2}\left( \frac{m_{i}}{m} \right)$

Maggiore è il numero dei figli, maggiore è il valore assunto da splitinfo.

### Classification Error

$Error(t)=1-max_{i}p(i|t)$

* **Max = $\left( 1-\frac{1}{n_{c}} \right)$**
* **Min = (0)**

### Criteri per interrompere lo split
* Quando tutti i record appartengono alla stessa classe.
* Quando tutti i record hanno valori simili su tutti gli attributi.
* Quando il numero dei record è inferiore a una certa soglia.

### Pro della classificazione con gli alberi decisionali
* Basso costo.
* Veloci e facili da interpretare.
* Robusti rispetto al rumore.

### Contro
* Lo spazio dell'albero può crescere esponenzialmente.
* Non considerano le interazioni tra attributi. 

## Model Overfitting
* Si verifica quando, nel tentativo di minimizzare l'errore sul training set, viene selezionato un modello eccessivamente complesso che non riesce ad apprendere la vera natura delle relazioni tra gli attributi.
* Questo accade perché l'errore di classificazione sul training set non fornisce stime accurate circa il comportamento dell'albero decisionale su record sconosciuti.
* Se il training set non è sufficientemente rappresentativo, il *test error* cresce e il *training error* decresce con l'aumento del numero dei nodi.
* _L'overfitting determina alberi decisionali più complessi del necessario._

### Alta complessità del modello
Modelli più complessi hanno una migliore capacità di rappresentare dati complessi, ma un modello eccessivamente complesso tende a specializzarsi sui soli dati di training, portando a scarse prestazioni su nuovi dati (overfitting). Per stimare quando un modello diventa troppo complesso, si utilizzano metriche di stima dell'errore di generalizzazione come:

* **Validation set:** dividere i dati in training e validation set (problema: diminuisce troppo la dimensione del training set).
* Incorporare la complessità del modello nell'errore _(Occam's Razor: preferire il modello più semplice)._ 
* Stimare bound statistici sull'errore di generalizzazione a partire dall'errore di training e dalla complessità del modello (approccio ottimistico).

## Generalization error

* Numero di errori commessi sul dataset reale e'(t).

### Stimare gli errori di generalizzazione

* **Approccio ottimistico:** Il training set è perfettamente rappresentativo di tutte le relazioni che caratterizzano il dataset.    e'(t)=e(t)
* **Approccio pessimistico:**
$err_{gen}(t)=err(t)+\Omega \cdot \frac{K}{N_{train}}$

dove:

  - err(T): errore complessivo del training set.
  -  Ω: parametro che stima la penalità dell’aggiunta di un nodo foglia _(modula la penalizzazione)._
  * k: numero di nodi foglia.
   * Ntrain: numero totale di train.
   * $\Omega \cdot \frac{K}{N_{train}}$ è una quantità positiva e più cresce più è penalizzante la complessità.
   * Per gli alberi binari, una penalità di 0.5 implica che un nodo debba essere sempre espanso nei due nodi figli se migliora la classificazione di almeno un record.

* **Minium Description Lenght**

    * Minimizza il costo per descrivere una classificazione. Il costo totale è dato dalla _somma del costo per codificare gli errori di classificazione del modello sui dati e del costo per codificare la complessità stessa del modello._
    * Per alberi decisionali, il costo del modello dipende dal numero di bit necessari per codificare i nodi interni (attributi) e le foglie (valori delle classi).
    $\text{Costo(Modello,Dati)=Costo(Dati|Modello)}+\alpha \cdot Costo(Modello)$

    _Dove il primo termine è il costo degli errori di classificazione e il secondo è il costo di codifica della complessità dell'albero, pesato da un fattore α._

## Pruning: strategie di selezione dei modelli

* **Prepruning**

    * Arresta prematuramente la costruzione dell'albero decisionale prima che diventi completamente sviluppato, implementando una condizione di arresto più restrittiva rispetto alle condizioni standard (tutte le istanze della stessa classe o tutti gli attributi uguali).
    * La nuova condizione ferma l'espansione di un nodo foglia quando il guadagno osservato nella stima dell'errore di generalizzazione scende al di sotto di una certa soglia.
    * **Vantaggio:** _evita di generare sottoalberi eccessivamente complessi che potrebbero portare a overfitting._
    * **Svantaggio:** talvolta _espansioni successive potrebbero portare a migliori sottoalberi, che però non vengono raggiunti a causa dell'arresto prematuro._
    * Altre possibili condizioni di arresto restrittive sono: fermarsi se il numero di istanze è sotto una soglia, se la distribuzione delle classi è indipendente dagli attributi disponibili, o se l'espansione non migliora le misure di impurità.

* **Postpruning**

    * L'albero decisionale viene inizialmente sviluppato completamente fino alla sua massima dimensione. Successivamente, viene effettuata una fase di **potatura bottom-up** (dal basso verso l'alto) in cui si collassano i sottoalberi in nodi foglia, scegliendo di potare il sottoalbero che determina la massima riduzione dell'errore di generalizzazione stimato, se esiste.
    * Le istanze della nuova foglia possono essere etichettate con la classe più frequente nel sottoalbero potato, oppure con la classe più frequente tra le istanze di training appartenenti a quel sottoalbero.
    * **Vantaggio:** _le decisioni di potatura si basano su un albero inizialmente completo, tendendo a restituire risultati migliori._
    * **Svantaggio:** _maggiore costo computazionale dovuto alla necessità di sviluppare inizialmente l'albero completo._

## Costruzione dei dataset tramite partizionamento

* **Holdout:** Il set viene partizionato in due set disgiunti: test set (1/3) e training set (2/3). Lo svantaggio è che il training set potrebbe non essere sufficientemente grande.
* **Random Subsampling:** Variante di Holdout, che consiste in un'esecuzione ripetuta di Holdout in cui il training set è scelto casualmente.
* **Cross Validation:**

    * Metodo di valutazione dei modelli che mira a sfruttare in modo efficiente tutte le istanze etichettate del dataset, sia come training che come test set.
    * _Il dataset di dimensione N viene partizionato in k sottoinsiemi distinti di dimensioni uguali. Un sottoinsieme viene usato come test set, mentre i rimanenti k-1 sottoinsiemi formano il training set per addestrare il modello di classificazione. Questo processo viene ripetuto k volte, con k modelli diversi addestrati e validati._
    * Alla fine, viene calcolata una misura di performance media (ad es. l'accuratezza media) sui k modelli ottenuti. Questo indica quanto quel tipo di modello e le sue caratteristiche si adattano bene al problema specifico.
    * Nel caso di alberi decisionali, la cross validation genererebbe k alberi diversi, con attributi e condizioni di split differenti, in base alle caratteristiche dei rispettivi k training set.

* **Bootstrap:**

    * Si effettua un ricampionamento con reinserimento (reimbussolamento) dei record già selezionati per costruire il training set. _Ogni record ha la stessa probabilità di essere estratto nuovamente._
    * Dato un dataset di dimensione N, Bootstrap crea un training set di N record dove ogni record ha circa il 63,2% di probabilità di essere incluso (con N sufficientemente grande). Infatti, la probabilità che un elemento non venga scelto tende a $\frac{1}{e} \approx 0,368$ per N molto grande, quindi la probabilità che venga scelto tende a $1-\frac{1}{e}\approx 0,632$.
    * Bootstrap non crea un nuovo dataset con più informazioni, ma permette di stabilizzare i risultati ottenibili dal dataset disponibile, risultando particolarmente utile per dataset di piccole dimensioni.

### Class Imbalance Problem

In molti dataset, le classi sono _distorte,_ ossia vi sono molti più record di una classe rispetto alle altre. Ciò pone due sfide per la classificazione:

* Molti dei metodi di classificazione funzionano bene solo quando il training set ha una rappresentazione equilibrata di tutte le classi.
* L'accuratezza non è adatta per valutare i modelli in presenza di squilibrio.

## Metriche alternative per la valutazione del modello

* **Matrice di confusione:** valuta la capacità di un classificatore sulla base di 4 indicatori:

    * TP (true positive): record positivi correttamente classificati.
    * FN (false negative): record positivi erroneamente classificati.
    * FP (false positive): record negativi erroneamente classificati.
    * TN (true negative): record negativi correttamente classificati.

* **Accuratezza:**

    $Accuracy = \frac{TP + TN}{TP + FN + FP + TN}=\frac{\text{record correttamente classificati}}{\text{numero totale di record}}$
    $Error \ rateֱ\frac{FN+FP}{TP+FN+FP+TN}=\frac{\text{record non correttamente classificati}}{\text{numero totale di record}}$

    * Da non usare in caso di classi sbilanciate.
    * _Nel caso di classificazione binaria, la classe rara è chiamata anche classe positiva, mentre quella che include la maggior parte dei record è chiamata classe negativa._

* **Precision:** usata per la corretta classificazione dei record della classe positiva/rara.

    * Misura quanti dei record positivi che ho classificato sono corretti.
    $Precion(p)=\frac{TP}{TP+FP}$

* **Recall:** misura quanti record positivi ho correttamente classificato sul totale delle supposizioni del modello.

    $recall(r)=\frac{TP}{TP+FN}$

* **F-measure:** Combina precision e recall, rappresenta la media armonica tra questi due valori.

    * Se la media armonica è elevata, significa che sia precision che recall lo sono, dunque sono stati commessi pochi errori (pochi falsi, sia positivi che negativi).
    $F-measure=\frac{2pr}{p+r}$



### Tecniche per il trattamento di dataset sbilanciati

* **Classificazione Cost Sensitive:** Le tecniche di Cost-Sensitive lavorano a livello di algoritmo assegnando un costo elevato all'errata classificazione della classe di minoranza.

    * Lo scopo è minimizzare il costo complessivo di errata classificazione.
    * Queste tecniche utilizzano una Matrice dei Costi che, a differenza della Confusion Matrix, introduce dei _parametri di penalità_ per indicare il costo di classificare erroneamente un record in una classe sbagliata.
	    $C(M)=\sum C(i,j) \times f(i,j)$ 
	    dove 
	    *C(i,j)*= Costo della classificazione dell'elemento della classe i nella classe j
	    *f(i,j)*= num di elementi della classe i classificati nella classe j

    * Una penalità negativa indica un premio ottenuto per una corretta classificazione.
    * In caso di problema di _classi sbilanciate_: l'importanza di riconoscere correttamente le osservazioni positive è maggiore di quello delle osservazioni negative; in pratica, _mi costa di più classificare il + come - che viceversa_.
    * L'obiettivo è di minimizzare i costi di errata classificazione e si basa sulla seguente regola:
    Classifica il nodo t con la classe i se il valore i minimizza $C(i|t)=\sum_{j} p(j|t) \times C(j,i)$ 
    * *p(j|t) rappresenta la frequenza relativa alla classe j al nodo t.*

* **Approccio basato sul campionamento:** Eseguono un lavoro di pre-processing sui dati in modo da fornire una distribuzione bilanciata tra le classi.

    * L'obiettivo è far sì che la classe rara sia ben rappresentata all'interno del training-set.
    * **Oversampling:** Caso in cui si aggiungono record.
        * Se si replicano esattamente le osservazioni della classe rara si può incorrere in overfitting.
    * **Undersampling** (sottocampionamento): Caso in cui si eliminano record per bilanciare la distribuzione tra classi.
        * Può comportare lo scarto di dati potenzialmente utili al processo di apprendimento.

### Tecniche di classificazione: Regole di decisione

* Un classificatore basato su regole utilizza un insieme di regole if-then.
* Una regola di classificazione appartenente al set può essere espressa come:

    * _(Condizione)_ -> _y_
    * La condizione è anche chiamato antecedente della regola e contiene una congiunzione di predicati di test sugli attributi.
    * y è chiamato conseguente della regola e contiene l'etichetta della classe y.

* Costruire un modello significa identificare un insieme di regole.
* _Una regola r copre un'istanza x se i valori di x soddisfano l'antecedente di r._

### Copertura e accuratezza

Dato un insieme di dati D e una regola di classificazione $r: A \to y$ 

* **Copertura della regola:** frazione dei record in D che soddisfano l'antecedente di r
$Coverage(r)=\frac{|A|}{|D|}$

* **Accuratezza:** frazione di istanze che, oltre a soddisfare l'antecedente, soddisfano anche il conseguente della regola
$Accuracy(r)=\frac{|A \cap y|}{|D|}$ 

* |A| è il numero delle istanze che soddisfa l’antecedente di r,
* |A ∩ y| rappresenta il numero di istanze che soddisfa contemporaneamente l’antecedente e il conseguente di r.
* |D| indica il numero totale di record.

### Regole mutuamente esclusive e esaustive

* **Regole mutuamente esclusive:** se nessuna coppia di regole può essere attivata dallo stesso record, ovvero, _ogni record è coperto al più da una regola_.
* **Regole esaustive:** Se esiste una regola per ogni combinazione di valori degli attributi, ovvero, _ogni record è coperto almeno una regola_.

Insieme, queste due proprietà assicurano che ogni istanza sia coperta da esattamente una regola.

### Mancanza di mutua esclusività e esaustività

* **Mancanza di mutua esclusività:** Un record può attivare più record dando vita a classificazioni alternative:

    * _Soluzione 1:_ Definire un ordine di attivazione delle regole.
    * _Soluzione 2:_ Assegnare il record alla classe per la quale vengono attivate più regole.

* **Mancanza di esaustività:** Il record viene associato a una classe di default in caso di mancata attivazione delle regole.

### Regole linearmente ordinate

* Le regole appartenenti a un insieme ordinato di regole R sono ordinate in modo decrescente secondo una data priorità.
* Un insieme ordinato di regole è anche chiamato _lista di decisione._

* **Ordinamento rule-based:** Le singole regole sono ordinate in base alle loro qualità.
* **Ordinamento class-based:**

    * Gruppi di regole che determinano la stessa classe compaiono di seguito nella lista.
    * L'ordinamento diventa quello tra le classi.
    * Il rischio è che una regola qualitativamente buona sia superata da una peggiore ma appartenente ad una classe più importante.

## Costruzione di un Classificatore Basato su Regole

### Metodi di costruzione

* **Metodi diretti:** Estraggono le regole di classificazione direttamente dai dati.
* **Metodi indiretti:** Estraggono le regole dal risultato di altri metodi di classificazione.

### Metodo diretto: Sequential Covering

* Permette di estrarre direttamente le regole dai dati (ordinamento class based).
* L'algoritmo, partendo da una _lista di decisione vuota_, R, estrae le regole per ciascuna classe seguendo l'ordinamento specificato su di esse.
* Estrae la regola per una classe y utilizzando _Learn-one-rule_.
* Elimina tutti i record appartenenti al training set coperti dalla regola (evita problemi di stima dell'accuracy).
* La nuova regola viene aggiunta in fondo alla list R.
* Se il criterio di arresto (riguarda l'estensione di una regola) non è soddisfatto, ritorna al punto 2, altrimenti STOP.

### Learn-one-rule

* L'obiettivo dell'algoritmo è di trovare una regola che copra più esempi positivi (tuple del training set che corrispondono alla classe in questione) possibili, e che minimizzi il numero di quelli negativi.
* Trovare una regola ottimale è costoso: l'algoritmo risolve questo problema costruendo progressivamente la regola con un approccio Greedy:

    * Genera una regola iniziale $r:\{\} \to y$.
    * Affina questa regola fino a soddisfare il criterio di arresto.

* Inizialmente, l'accuratezza della regola potrebbe risultare scarsa perché alcuni record del training set potrebbero appartenere alla classe negativa.
* Una coppia (Attributo, Valore) deve essere aggiunta all'antecedente della regola per migliorare l'accuratezza.
$Accuracy(r)=\frac{nr}{n}$

* Dove nr è il numero di istanze correttamente classificate da r.
* n è il numero di istanze che soddisfano l'antecedente di r.

## Criteri di valutazione delle regole

* **Likelihood Ratio:** Usato per potare le regole che hanno una copertura scarsa.
$LikelihoodRatio(r) = 2\sum_{i=1}^k f_i \log_2(\frac{f_i}{e_i})$ 

Dove k è il numero di classi, f_i è la frequenza osservata degli esempi di classe e che sono coperti dalla regola, e_i è la frequenza prevista di una regola che effettua previsioni a caso.

* **Laplace:** Pesa l'accuracy in base alla Coverage.
$Laplace(r)=\frac{f_{+}+1}{n+k}$
Dove k è il numero di classi, $f_{+}$ è il numero di esempi positivi coperti dalla regola r ed n è il numero di esempi coperti dalla regola r.

* _Copertura pari a zero e si assume una distribuzione uniforme dei dati ->_ l'indice si riduce alla probabilità a priori della classe.
* _Copertura alta ->_ l'indice tende asintoticamente al valore dell'accuratezza.
* _Copertura bassa ->_ l'indice tende a diminuire.

* **M-Estimate:** Pesa l'accuracy in base alla Coverage.
$$
m\text{-}estimate(r) = \frac{f_+ + k p_+}{n + k}
$$

Dove $k$ è il numero di classi, $f_+$ è il numero di esempi positivi coperti dalla regola $r$, $n$ è il numero di esempi coperti dalla regola $r$ e $p_+$ è la probabilità a priori della classe $+$. 

Si osservi che per $p_+ = \frac{1}{k}$, $m$-estimate coincide con Laplace.


* **FOIL:** _First order inductive learner_, misura la variazione dovuta all'incremento della regola con l'aggiunta di un nuovo atomo.

	Supponiamo una regola $r_0$:  
	$A \to +$ copre $p_0$ esempi positivi e $n_0$ esempi negativi.
	
	Dopo aver aggiunto un nuovo atomo $B$, la regola estesa $r_1$ risulta così definita:  
	$A, B \to +$ copre $p_1$ esempi positivi e $n_1$ esempi negativi.
	
	Il guadagno di informazioni di FOIL della regola estesa è definito come segue:
	
	$$
	FoilGain(r_0, r_1) = p_1 \left( \log_2 \left( \frac{p_1}{p_1 + n_1} \right) - \log_2 \left( \frac{p_0}{p_0 + n_0} \right) \right)
	$$
	
	L'indice è proporzionale a $p_1$ e a $\frac{p_1}{p_1 + n_1}$, quindi tende a favorire regole che hanno elevata *coverage* e *accuracy*. Misura alternativa:
	
	$$
	v(r_0, r_1) = \frac{p_1 - n_1}{p_1 + n_1} - \frac{p_0 - n_0}{p_0 + n_0}
	$$
	
	* Supponendo una regola iniziale r0 che copre un certo numero di esempi positivi e negativi, dopo l'aggiunta di un nuovo atomo B, la regola estesa r1 copre un diverso numero di esempi positivi e negativi.
	* Il guadagno di informazioni di FOIL della regola estesa è calcolato in base _alla differenza tra i logaritmi delle proporzioni di esempi positivi nella regola estesa e nella regola iniziale._
	* L'indice è proporzionale alla differenza tra il numero di esempi positivi e negativi nella regola estesa, normalizzata rispetto al numero totale di esempi.
	* L'indice tende a favorire regole che hanno una copertura elevata e un'accuratezza elevata.

### Rule pruning

Semplifica le regole di _learn one rule_ per migliorare l'errore di generalizzazione delle regole stesse. Ѐ utile perché l'approccio di costruzione è greedy.

* **Reduced error pruning:** rimuove a turno un atomo dalla regola:

    * Determina l'atomo la cui rimozione comporta il massimo miglioramento dell'error rate sul validation set, altrimenti STOP.
    * Elimina l'atomo e riparti da (1).

### Metodi diretti: RIPPER

* Algoritmo di classificazione basato su regole che permette di derivare un insieme di regole a partire dal training set.
* Scala quasi linearmente con il numero di istanze di addestramento.
* Funziona bene anche con dati rumorosi poiché utilizza il validation set per evitare l'overfitting.

* **Caso 1: problemi a 2 classi**

    * Sceglie una delle classi come classe positiva e apprende delle regole per determinare le istanze afferenti a tale classe.
    * L'altra classe sarà la classe di default.

* **Caso 2: problemi multi-classe**

    * Le classi sono ordinate in base a un criterio di rilevanza delle stesse.
    * Si suppone un insieme di classi $\{y_1, y_2, ...y_c\}$, dove y1 è la classe meno rilevante e yc è la classe più diffusa.
    * Le regole vengono costruite partendo dalla classe più piccola e considerando gli esempi delle altre classi come negativi.
    * Questo processo viene ripetuto fino a quando rimane solo una classe yc, che viene considerata come classe di default secondo il criterio di stop.

* **Costruzione del set di regole:** viene utilizzato l'algoritmo _sequential covering_:

    * Trovata la regola migliore, vengono eliminati tutti i record coperti dalla regola.
    * Se non viola la _Minimum description length principle_ (condizione di stop), la regola viene aggiunta al set di regole.

* **Estensione di una regola:** Ripper utilizza _Learn-one-rule_:

    * Inizia con una regola vuota e aggiunge l'atomo che determina un miglioramento del _FOIL's Information Gain_.
    * L'operazione di aggiunta viene ripetuta finché la regola non copre più esempi negativi.
    * La nuova regola ottenuta è sottoposta a _pruning_: saranno rimossi dalla regola gli atomi che massimizzano la seguente metrica:
    $v=\frac{p-n}{p+n}$
    Ove p è il numero di esempi positivi coperti dalla regola nel validation set e n è il numero di esempi negativi coperti dalla regola nel validation set.
    Si osservi che il pruning è effettuato partendo dall'ultimo atomo aggiunto alla regola.

    * Ripper esegue anche passaggi di ottimizzazione aggiuntivi per determinare se alcune regole possono essere sostituite da un'alternativa migliore. In particolare:

        * Per ogni regola r nel set di regole R, vengono considerate due alternative: la regola di sostituzione r∗ (nuova regola da zero) e la revised rule r′ (regola estesa).
        * Si confronta il set di regole contenente r con i set di regole contenenti r∗ e r′ e si sceglie l'insieme di regole che minimizza il criterio di _Minimum Description Length_.

* Ripetere il processo di generazione e ottimizzazione delle regole per gli esempi positivi rimanenti.

## Metodi indiretti: C4.5 rules

È possibile trasformare un albero decisionale in un insieme di regole: ogni percorso dal nodo radice al nodo foglia di un albero può essere espresso come una regola di classificazione.

* Le condizioni di test costituiscono i predicati dell'antecedente della regola.
* L'etichetta di classe delle foglie costituisce il conseguente della regola.

L'algoritmo su cui ci focalizzeremo sarà il C4.5Rules, che è così costituito:

1. Le regole di classificazione vengono estratte per ogni percorso dalla radice ai nodi foglia dell'albero decisionale.
2. Per ogni regola `r : A → y`:
    * (a) Considera una regola alternativa `r′ : A′ → y` dove `A′` è ottenuto rimuovendo un atomo da `A`.
    * (b) La regola semplificata viene mantenuta a condizione che il suo tasso di errore pessimistico sia inferiore a quello della regola originale `r`.
    * (c) Riparti da `A` fino a che l'errore pessimistico della regola non può essere migliorato ulteriormente.
    * (d) Poiché alcune delle regole possono diventare identiche dopo la potatura, le regole duplicate vengono scartate.

Dopo aver generato il set di regole, C4.5rules effettua l'ordinamento Class-Based. In particolare:

1. Le regole caratterizzate dal medesimo conseguente (classe) vengono raggruppate all'interno dello stesso sottoinsieme.
2. Viene calcolato per ciascun sottoinsieme il valore di *description length* così definito:

    * *Description length* = `L(error)` + `gL(model)`

    * `L(error)` = numero di bit necessari a codificare la classificazione errata
    * `L(model)` = numero di bit necessari a rappresentare il modello
    * `g` = parametro che dipende dal numero di attributi ridondanti (0,5 di default). Assume un valore più piccolo all'aumentare degli attributi ridondanti

* Le classi sono disposte in base al valore di *description length*. Viene data massima priorità alla classe caratterizzata dal minimo valore di *description length* poiché ci si aspetta che contenga il miglior set di regole. 

### Vantaggi e svantaggi dei classificatori Rule-Based

**Vantaggi:**

- Espressivi e di facile interpretazione come gli alberi decisionali.
- Buone prestazioni paragonabili agli alberi decisionali.
- Gestione efficace degli attributi ridondanti.
- Adatti per gestire classi squilibrate.

**Svantaggi:**

- Difficoltà nella gestione di test set con dati incompleti.
- Costo di costruzione che non scala con l'aumento del training set.
- Sensibili al rumore nei dati.

### Differenze tra alberi decisionali e regole

- Espressività simile, ma l'insieme di regole prodotte è di solito diverso.
- La costruzione degli alberi decisionali tiene conto della qualità di tutti i figli generati, mentre l'aggiunta di un atomo alle regole valuta solo la bontà della classe determinata.


## Tecniche di Classificazione - Nearest Neighbor

### Eager Learners vs. Lazy Learners

* **Eager Learners:**
    * Costruiscono un modello di classificazione indipendente dall'input, basandosi su un training set utilizzato durante la fase di apprendimento.
    * Esempi: alberi di decisione, classificatori basati su regole.

* **Lazy Learners:**
    * Posticipano l'apprendimento fino a quando non è necessario classificare i record del test set.
    * **Rote Classifier:** Memorizza tutte le istanze del training set e classifica un'istanza del test set solo se gli attributi di quest'ultima corrispondono esattamente a un record del training set.
    * **Problema:** Alcune istanze potrebbero non essere classificate perché non corrispondono a nessun esempio del training set.

### Nearest Neighbor

* **Approccio:** Trova i record del training set relativamente simili agli attributi delle istanze del test set.
* **Rappresentazione:** Ogni record del training set è rappresentato come un punto in uno spazio d-dimensionale, dove d è il numero di attributi.
* **Classificazione:**
    1. Calcola la vicinanza di un'istanza di test alle istanze del training set utilizzando una misura di prossimità.
    2. Identifica i k nearest neighbors (i k record più vicini).
    3. Determina la classe del record sconosciuto scegliendo la classe più frequente tra i k vicini.
    * **Opzionale:** I voti possono essere pesati in base alla distanza.

* **Parametri:**
    * **Training set:** necessario per la classificazione.
    * **Metrica di distanza:** utilizzata per calcolare la distanza tra due record (es. distanza euclidea).
    * **k:** numero di vicini da utilizzare.

* **Scelta di k:**
    * **k troppo piccolo:** l'approccio è sensibile al rumore.
    * **k troppo grande:** possono essere inclusi molti esempi di altre classi nell'intorno dei vicini.

* **Pre-processing:**
    * **Normalizzazione:** necessaria per attributi con scale di valori diverse.
    * **Attenzione alle misure di somiglianza:** per evitare previsioni errate in presenza di diverse distribuzioni dei dati.

### Pro e Contro del Nearest Neighbor

**Pro:**

* Non richiedono la costruzione di un modello.
* Permettono di costruire contorni delle classi in maniera non lineare, offrendo maggiore flessibilità.

**Contro:**

* Richiedono una misura di distanza per valutare la vicinanza.
* Richiedono una fase di pre-processing per normalizzare il range di variazione degli attributi.
* La classe è determinata localmente, rendendo l'approccio suscettibile al rumore.
* Sono molto sensibili alla presenza di attributi irrilevanti o correlati.
* Il costo di classificazione può essere elevato e dipende linearmente dalla dimensione del training set.

### Ottimizzazione del Costo di Calcolo

* **Evitare il calcolo della distanza da tutti gli oggetti:**
    * Utilizzare tecniche di indicizzazione per ridurre il numero di oggetti da considerare.
* **Condensazione:**
    * Determinare un set di oggetti più piccolo che offre le medesime prestazioni del set originale.
    * Rimuovere oggetti dal training set per migliorare l'efficienza.

## Strutture a Indice

**Problema:** Trovare il punto più vicino a un dato punto in un insieme di punti.

**Tipi di Query:**

* **Near neighbor range search:** Trova tutti i punti entro un raggio r da un punto q.
    * Esempio: Trova i ristoranti nel raggio di 400m dal mio albergo.
* **Approximate Near neighbor:** Trova tutti i punti con distanza massima da q pari a (1 + e) volte la distanza di q dal suo punto più vicino.
    * Esempio: Trova i ristoranti più vicini al mio albergo.
* **K-Nearest-Neighbor:** Trova i K punti più vicini a q.
    * Esempio: Trova i 4 ristoranti più vicini al mio albergo.
* **Spatial join:** Trova tutte le coppie di punti (p, q) con distanza ≤ r, dove p appartiene a un insieme P e q appartiene a un insieme Q.
    * Esempio: Coppie (albergo, ristorante) che distano al massimo 200 m.

**Approcci:**

* **Linear scan (Approccio Naïf):**
    * Calcola la distanza tra il punto di query e ogni altro punto nel database, tenendo traccia del "miglior vicino finora".
    * Tempo di esecuzione O(dN), dove N è la cardinalità del database e d è la dimensionalità dei punti.
    * Non richiede strutture dati di ricerca aggiuntive.
    * Diventa rapidamente inefficiente con l'aumento della dimensione o della dimensionalità del problema.

* **Indici multi-dimensionali (spaziali):**
    * **B+tree multi-attributo:**
        * Organizza le tuple in base ai valori degli attributi.
        * Non è adatto per query che non specificano il valore dell'attributo favorito.
    * **Quad-tree:**
        * Divide ricorsivamente lo spazio d-dimensionale in 2*d sottoquadri uguali.
        * Ogni suddivisione genera un nodo con al più 2*d nodi figli.
        * Per trovare i punti entro un raggio r da un punto q:
            1. Inserisci la radice in uno stack.
            2. Estrai un nodo T dallo stack.
            3. Per ogni figlio C di T:
                * Se C è foglia, esamina i suoi punti.
                * Se C interseca la sfera di raggio r centrata in q, inserisci C nello stack.
            4. Torna al passo 2.

**Svantaggi del Quad-tree:**

* Punti vicini potrebbero richiedere molti livelli per essere separati in celle diverse.
* Ha complessità temporale e spaziale esponenziale $O(n\cdot 2^d)$ rispetto alla dimensionalità d dei dati.
* Non scala bene ad alte dimensionalità a causa della sua natura esponenziale.

## Kd-trees (k-dimensional tree)

* **Struttura dati:** Albero utilizzato per organizzare un insieme di punti in uno spazio k-dimensionale.
* **Ogni nodo:** Rappresenta un iper-rettangolo nello spazio k-D e contiene un punto chiamato "punto di separazione".
* **Funzionamento:** L'iper-rettangolo di un nodo divide lo spazio in due parti, una a sinistra e una a destra del piano di separazione definito dal punto del nodo. I punti del dataset vengono distribuiti nell'albero a seconda della loro posizione rispetto ai piani di separazione.
* **Utilità:** Accelerano operazioni come la ricerca del punto più vicino o di punti all'interno di un certo intervallo, suddividendo efficacemente lo spazio k-dimensionale.

### Processo di Costruzione di un Kd-tree

1. **Selezione della dimensione:** Si sceglie una dimensione (asse) lungo la quale partizionare l'insieme dei punti. La scelta può essere fatta ciclicamente tra le dimensioni.
2. **Calcolo del valore mediano:** Si calcola il valore mediano dei punti lungo la dimensione scelta e si crea un nodo corrispondente a quell'iper-piano di separazione.
3. **Divisione dei punti:** Si dividono i punti in due sottoinsiemi: quelli a sinistra (minori) e quelli a destra (maggiori) del piano di separazione lungo la dimensione scelta.
4. **Ricorsione:** Il processo viene ripetuto ricorsivamente sui due sottoinsiemi, selezionando una nuova dimensione di separazione, fino a che ogni nodo contiene al massimo un punto oppure si raggiunge una dimensione massima dell'albero.

### Ricerca in un Kd-tree

1. **Partenza dalla radice:** Si parte dalla radice dell'albero.
2. **Discesa ricorsiva:** Si scende ricorsivamente verso il basso seguendo il ramo sinistro o destro in base al lato del piano di separazione del nodo corrente in cui cade il punto di query.
3. **Nodo foglia:** Si arriva ad un nodo foglia contenente un singolo punto, che diventa il "miglior punto trovato finora".
4. **Risalita ricorsiva:** Risalendo ricorsivamente, per ogni nodo si controlla se il suo sottoalbero opposto a quello appena esplorato potrebbe contenere un punto più vicino al punto di query.
5. **Controllo della distanza:** Se la distanza del punto di query dal piano di separazione è minore della distanza dal miglior punto trovato, si esplora anche l'altro sottoalbero.
6. **Continuazione della risalita:** Altrimenti si continua a risalire senza esplorare ulteriormente quella porzione.

**Nota:** Il pensiero alla base di questo passaggio è che potrebbe esserci un punto sull'altro lato del piano di divisione più vicino al punto di ricerca rispetto al migliore corrente.

**Metrica di distanza:** La metrica comunemente utilizzata per calcolare la distanza tra punti nello spazio k-dimensionale è la distanza euclidea.

### Pro e Contro dei Kd-trees

**Pro:**

* Permettono una partizione efficiente dello spazio k-dimensionale minimizzando gli spazi vuoti.
* Consentono query efficienti di ricerca dei k-nearest neighbor.
* Possono essere usati anche per query di range/intervallo.

**Contro:**

* Possono diventare inefficienti ad alte dimensionalità a causa della "maledizione della dimensionalità".
* Le prestazioni dipendono molto dalla scelta dell'asse di partizione e della strategia di selezione del punto di separazione.
* Possono non funzionare bene su dati molto distorti o con distribuzioni complesse.

### Calcolo Approssimato

* **Algoritmo di ricerca del vicino più vicino approssimativo:** Restituisce punti con una distanza dal punto dato q non superiore a c volte la distanza ai punti effettivamente più vicini.
* **Vantaggio:** Può essere vantaggioso quando i risultati approssimati sono quasi buoni quanto quelli esatti, specialmente se la misura di distanza riflette bene la nozione di qualità dell'utente.

### Locality-Sensitive Hashing (LSH)

* **Tecnica di hashing probabilistica:** Utilizzata per la ricerca di vicini approssimati in grandi dataset di dati ad alta dimensionalità.
* **Funzionamento:** Crea una famiglia di funzioni hash che mappano i dati in uno spazio di hash a bassa dimensionalità in modo tale che oggetti simili nello spazio di origine abbiano una probabilità più alta di essere mappati in hash simili.
* **Ricerca di vicini approssimati:** Viene effettuata esaminando solo gli elementi che corrispondono allo stesso valore hash, piuttosto che confrontare ogni elemento del dataset.
* **Obiettivo:** Massimizzare le collisioni di hash (approccio opposto rispetto quello delle hasmap).

### Utilizzo Memoria Secondaria

* **Dataset di grandi dimensioni:** Richiedono l'uso della memoria secondaria.
* **Strutture dati disk-based:**
    * **R-tree:** Approccio ottimistico, tende a rispondere alle query in tempo logaritmico.
    * **Vector Approximation File:** Approccio pessimistico, analizza l'intero dataset ma in modo molto veloce.

### R-tree

* **Struttura dati:** Simile ai B+tree, ma estesa per spazi multidimensionali.
* **Organizzazione:** Organizza gli oggetti in iperrettangoli multidimensionali che possono sovrapporsi.
* **Costruzione:** Avviene con un approccio bottom-up:
    1. Partizionare gli oggetti in gruppi di piccola cardinalità (2-3 elementi).
    2. Per ogni gruppo, calcolare il rettangolo minimo che racchiude tutti i suoi oggetti.
    3. Unire ricorsivamente i rettangoli minimi in nodi intermedi, fino ad ottenere un singolo nodo radice che racchiude tutti gli oggetti.

**Pro:**

* Supportano nearest neighbor search.
* Funzionano per punti e per rettangoli.
* Evitano gli spazi vuoti.
* Molte varianti tra le quali X-tree, SS-tree, SR-tree.
* Funzionano bene per le dimensioni ridotte.

**Contro:**

* Non funzionano molto bene per dimensioni elevate.
