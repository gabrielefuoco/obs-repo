## Embedding

L'embedding è una tecnica di rappresentazione dei dati testuali in uno spazio vettoriale, in cui parole o frasi simili vengono mappate in punti vicini nello spazio. Questo consente di catturare le relazioni semantiche tra le parole e di utilizzare tecniche di apprendimento automatico e modelli di deep learning per svolgere varie attività sul testo, come la classificazione, la traduzione e l'analisi del sentimento.

### Tecniche di Embedding:

* **1-Hot Encoding:** È una rappresentazione semplice in cui ogni parola è codificata come un vettore di dimensione pari alla dimensione del vocabolario, con un 1 nella posizione corrispondente alla parola e 0 altrimenti. Questa tecnica non cattura alcuna relazione semantica tra le parole.

* **N-Gram:** Un n-gram è una sequenza contigua di n parole estratta da un testo. In genere, le stop-words (parole comuni come articoli, preposizioni, ecc.) vengono rimosse. Ogni n-gram può essere rappresentato come un nodo in ingresso a un modello, con un vettore di dimensione pari al vocabolario di riferimento. La similarità tra n-gram può essere calcolata utilizzando la similarità del coseno tra i loro vettori.

* **Word Embedding**: È una rappresentazione distribuita delle parole nello spazio vettoriale, in cui le parole simili vengono mappate in punti vicini. Esistono diversi algoritmi per imparare le rappresentazioni di embedding, come Word2Vec e GloVe. Questi algoritmi catturano le relazioni semantiche e sintattiche tra le parole, consentendo un'elaborazione del linguaggio naturale più efficace.

* **Sentence Embedding**: È una rappresentazione vettoriale di un'intera frase o documento, ottenuta combinando le rappresentazioni di embedding delle singole parole che lo compongono. Esistono diversi approcci per ottenere sentence embedding, come l'utilizzo di reti neurali ricorrenti o trasformatori. 

## Support Vector Machine (SVM)

* Le Support Vector Machine (SVM) sono un potente modello di classificazione che impara confini di decisione lineari o non lineari per separare le classi nello spazio degli attributi. L'obiettivo dell'SVM è trovare l'iperpiano di separazione che massimizza il margine, ovvero la distanza minima dalle istanze di training più vicine di entrambe le classi.

* L'SVM rappresenta il confine di decisione utilizzando solo un sottinsieme delle istanze di training più difficili da classificare, chiamati vettori di supporto. Ciò permette di controllare la complessità del modello evitando overfitting.

* Per dataset linearmente separabili, esistono infiniti iperpiani che separano le classi. L'SVM sceglie quello con il margine massimo, chiamato iperpiano di massimo margine, in quanto risulta più robusto e generalizza meglio sui nuovi dati.

* Per problemi non linearmente separabili, l'SVM può apprendere confini di decisione non lineari utilizzando funzioni kernel, estendendo la metodologia del massimo margine.

* Grazie alla capacità di regolarizzazione e all'iperpiano di massimo margine, l'SVM riesce ad apprendere modelli espressivi evitando overfitting e generalizzando bene sui nuovi dati.

## Tecniche di Ensemble

* Le tecniche di ensemble mirano a migliorare l'accuratezza di classificazione combinando le predizioni di più classificatori base.

    * Un classificatore ensemble costruisce un insieme di classificatori base dal training set e classifica un nuovo esempio effettuando un voto sulle predizioni dei singoli classificatori.

* Affinché un ensemble funzioni meglio di un singolo classificatore, è necessario che i classificatori base soddisfino due condizioni:

    * Devono essere tra loro indipendenti, ovvero con errori non correlati;
    * Devono performare meglio di un semplice classificatore casuale.

* Quando queste condizioni sono soddisfatte, la combinazione dei voti dei classificatori base permette all'ensemble di ridurre l'errore complessivo rispetto ai singoli componenti.

    * Ad esempio, con 25 classificatori binari aventi errore 0.35 ciascuno, un ensemble a maggioranza commette un errore di solo 0.06, decisamente inferiore.
    $\\text{ensemble}=P(X \geq 13)=\sum_{i=13}^25 \binom{25}{i} \epsilon^i(1-\epsilon)^{25-i}=0.06$ 

* Tuttavia, nella pratica è difficile garantire l'indipendenza assoluta dei classificatori base. Ciononostante, tecniche ensemble in cui i componenti sono parzialmente correlati hanno generalmente mostrato miglioramenti nelle prestazioni rispetto ai singoli classificatori.

## Metodi per costruire un classificatore Ensemble

L'idea di base è quella di costruire più classificatori a partire dai dati originali e poi aggregare le loro previsioni (tramite un meccanismo di voto) durante la classificazione di esempi sconosciuti.

* **Manipulate data distribution:** consiste nella creazione di più insiemi di dati di addestramento, ottenuti mediante il campionamento casuale del set di dati originale secondo una specifica distribuzione di probabilità. Successivamente avviene la costruzione di un classificatore da ciascun set di addestramento. La distribuzione di probabilità di campionamento determina la probabilità che un record venga selezionato per l’addestramento e può variare da un tentativo all’altro.
## Bagging (Bootstrap Aggregating)

* Tecnica di ensemble che genera molteplici versioni di uno stesso modello di classificazione addestrandolo su diversi sottoinsiemi di dati, ottenuti campionando con ricampionamento i dati originali.

* La procedura prevede di:

    1. Generare k sottoinsiemi $D_i$ di dimensione N campionando con ricampionamento i dati originali.
    2. Addestrare k modelli $C_i$ su ciascun $D_i$.
    3. Per classificare una nuova istanza, assegnarla alla classe che riceve più voti tra i k modelli.

* Ogni insieme $D_i$ contiene circa il 63% dei dati originali, con alcune istanze duplicate e altre omesse. Ciò genera diversità tra i modelli $C_i$, che imparano aspetti leggermente diversi dei dati.

* Il Bagging migliora la generalizzazione riducendo la varianza dei modelli base, risultando efficace quando questi sono instabili e ad alta varianza. Combinando i voti, si riducono gli errori dovuti a fluttuazioni casuali nei dati di addestramento.

* Per modelli stabili a bassa varianza, il Bagging potrebbe non migliorare o addirittura peggiorare le prestazioni, poiché il bias del modello base rimane inalterato e la dimensione dei sottoinsiemi $D_i$ è inferiore ai dati originali.

## Boosting

* Tecnica di ensemble iterativa che adatta i pesi degli esempi di training per forzare il modello a concentrarsi sugli esempi più difficili da classificare ad ogni iterazione.

* Inizialmente, tutti gli esempi hanno lo stesso peso. Ad ogni iterazione:

    * Si addestra un nuovo classificatore su un campione estratto in base ai pesi correnti.
    * Si utilizzano le predizioni di tale classificatore per aggiornare i pesi degli esempi: quelli classificati erroneamente vedono aumentare il loro peso, quelli classificati correttamente lo vedono diminuire.

* In questo modo, gli esempi più difficili acquisiscono sempre più importanza nelle iterazioni successive, permettendo al modello di imparare a classificarli meglio.

* Alla fine, le predizioni di tutti i classificatori addestrati vengono combinate in una previsione finale dell'ensemble.

* Rispetto al Bagging, il Boosting non utilizza campioni bootstrap, ma sfrutta l'adattamento dei pesi per focalizzarsi sugli esempi problematici, risultando efficace per modelli stabili a basso bias.

* Esistono diverse implementazioni di Boosting, che variano per il modo di aggiornare i pesi e combinare le predizioni dei singoli classificatori.

## AdaBoost

* AdaBoost è un potente algoritmo di boosting per costruire un classificatore ensemble forte combinando una sequenza di classificatori base (o deboli) in modo iterativo. L'idea centrale è addestrare nuovi classificatori ad ogni iterazione concentrandosi sugli esempi di training che risultano più difficili da classificare per i modelli precedenti.

* Inizialmente, tutti gli esempi di training hanno lo stesso peso 1/N. Poi, ad ogni iterazione i:

    1. Si campiona un sottoinsieme $D_i$ dal dataset originale utilizzando la distribuzione di pesi corrente $w_i$.
    2. Si addestra un nuovo classificatore base $C_i$ sul sottoinsieme $D_i$.
    3. Si calcola l'errore pesato $\epsilon_i$ di $C_i$, basato sui pesi $w_i$ degli esempi classificati erroneamente.
    4. Si calcola l'importanza $\alpha_i$ di $C_i$ in base a $\epsilon_i$, con $\alpha_i$ alto se $\epsilon_i$ è basso (buon classificatore).
    5. Si aggiornano i pesi $w_i$ per la prossima iterazione: aumentati per gli esempi che $C_i$ ha classificato erroneamente, diminuiti per quelli corretti. Ciò aumenta l'importanza degli esempi difficili.
    6. Se $\epsilon_i > 0.5$, i pesi vengono reinizializzati uniformemente e si ricampiona $D_i$.

* Alla fine delle k iterazioni, la previsione finale dell'ensemble è una combinazione pesata delle previsioni dei k classificatori base, utilizzando i rispettivi $\alpha_i$ come pesi. Ciò permette ai migliori classificatori di influenzare maggiormente la risposta.

* Questo processo iterativo di ri-pesatura degli esempi consente a AdaBoost di concentrarsi sugli esempi più problematici, correggendo gli errori dei classificatori precedenti e producendo un ensemble finale molto più accurato rispetto ai singoli componenti.

## Gradient Boosting

* Costruisce un modello predittivo come somma di molteplici modelli più semplici, addestrati in modo iterativo e sequenziale.

* L'idea di base è utilizzare un algoritmo di ottimizzazione del gradiente (tipicamente gradient descent) per minimizzare iterativamente una funzione di perdita, definita come misura dell'errore di predizione sui dati di training.

* Ad ogni iterazione, l'algoritmo addestra un nuovo modello semplice (chiamato debole) sui residui del modello complessivo fino a quel punto, ovvero sugli esempi che sono stati predetti in modo errato. Il nuovo modello debole punta quindi a correggere gli errori del modello precedente.

* Tutti i modelli deboli addestrati vengono poi combinati in un modello finale tramite una somma pesata, dove il peso di ciascun modello debole è determinato dalla sua capacità di ridurre la funzione di perdita.

* L'algoritmo permette di scegliere diverse funzioni di perdita a seconda del task (regressione, classificazione, ecc). Inoltre, richiede l'ottimizzazione di iperparametri come il numero di iterazioni e la complessità dei modelli deboli.

* In sintesi, il Gradient Boosting costruisce iterativamente un modello predittivo accurato combinando tanti modelli deboli, ognuno mirato a ridurre l'errore residuo del modello corrente. Ciò lo rende molto flessibile ed efficace in numerosi problemi di apprendimento supervisionato.

## Costruire classificatori ensemble

### Manipolando le features di input

* **Random Forest:** costruisce un insieme di alberi decisionali decorrelati, ognuno addestrato su:

    1. Un campione bootstrap dei dati di training (come nel bagging).
    2. Un sottoinsieme casuale di features selezionate ad ogni nodo dell'albero.

* Nello specifico, per costruire una Random Forest di T alberi su un dataset con N istanze e d features:

    1. Si estrae un campione bootstrap $D_i$ di N istanze.
    2. Si addestra un albero decisionale $T_i$ su $D_i$, selezionando casualmente $p \le d$ features ad ogni nodo e scegliendo quella che massimizza la riduzione di impurità.
    3. Si ripetono i passi 1-2 per T volte.

* Il valore di p è un iperparametro chiave: valori piccoli riducono la correlazione tra alberi ma ne limitano la potenza predittiva, mentre valori grandi possono portare ad alberi molto correlati.

* Questo approccio di randomizzazione sia sui dati che sulle features permette di aumentare la diversità tra i singoli classificatori dell'ensemble, migliorando la capacità di generalizzazione complessiva.

* La Random Forest risulta efficace soprattutto con datasets di grandi dimensioni e tante features ridondanti.

### Manipolando le etichette di classe

* **Error-Correcting Output Codes (ECOC):** affronta problemi di classificazione multi-classe quando il numero di classi è elevato. L'idea chiave è trasformare il problema multi-classe in una serie di problemi di classificazione binaria.

* La procedura è la seguente:

    1. Le classi vengono divise casualmente in due sottoinsiemi disgiunti $A_0$ e $A_1$.
    2. I dati di training vengono ricodificati in un problema binario, assegnando classe 0 agli esempi in $A_0$ e classe 1 a quelli in $A_1$.
    3. Si addestra un classificatore binario su questi dati ricodificati.
    4. Si ripetono i passi 1-3 più volte, ottenendo diversi classificatori binari con diversa codifica classe.

* Per classificare un nuovo esempio:

    * Ogni classificatore binario assegna un voto alle classi di $A_0$ o $A_1$ a seconda della sua predizione.
    * Si conta il numero di voti per ogni classe originale.
    * Si assegna l'esempio alla classe con più voti.

* Questa tecnica scompone il problema multi-classe in molteplici problemi binari più semplici, permettendo di sfruttare algoritmi efficienti di classificazione binaria. La diversità nei sottoinsiemi di codifica classe generati casualmente permette di ottenere un ensemble accurato.

