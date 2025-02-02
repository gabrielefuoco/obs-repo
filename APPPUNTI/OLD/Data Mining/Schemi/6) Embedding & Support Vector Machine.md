**Embedding**

* **Definizione:** Rappresentazione di dati testuali in uno spazio vettoriale, dove parole/frasi simili sono vicine. Cattura relazioni semantiche per attività NLP (classificazione, traduzione, analisi del sentimento).

    * **Tecniche:**
        * **1-Hot Encoding:** Vettore con 1 nella posizione della parola, 0 altrimenti.  Non cattura relazioni semantiche.
        * **N-Gram:** Sequenza di *n* parole (senza stop words). Similarità calcolata con similarità del coseno.
        * **Word Embedding (Word2Vec, GloVe):** Rappresentazione distribuita; parole simili sono vicine. Cattura relazioni semantiche e sintattiche.
        * **Sentence Embedding:** Rappresentazione vettoriale di frasi/documenti, combinando word embedding (RNN o Trasformatori).


**Support Vector Machine (SVM)**

* **Definizione:** Modello di classificazione che trova iperpiani di separazione (lineari o non lineari) tra classi.  Massimizza il margine (distanza minima dalle istanze più vicine).

    * **Caratteristiche:**
        * Usa solo i *vettori di supporto* (istanze più difficili da classificare) per definire il confine di decisione, evitando overfitting.
        * Per dati linearmente separabili, sceglie l'iperpiano di massimo margine per maggiore robustezza e generalizzazione.
        * Per dati non linearmente separabili, usa funzioni kernel per apprendere confini non lineari.
        * Regolarizzazione e iperpiano di massimo margine prevengono overfitting e migliorano la generalizzazione.


**Tecniche di Ensemble**

* **Obiettivo:** Migliorare l'accuratezza di classificazione combinando predizioni di più classificatori base.
    * **Meccanismo:** Un ensemble costruisce e combina diversi classificatori base, votando le loro predizioni su un nuovo esempio.
    * **Condizioni per un buon ensemble:**
        * Indipendenza dei classificatori base (errori non correlati).
        * Prestazioni dei classificatori base superiori a un classificatore casuale.
    * **Vantaggio:** Riduzione dell'errore complessivo rispetto ai singoli classificatori.
    * **Sfida:** Difficoltà nel garantire l'indipendenza assoluta dei classificatori.  Miglioramenti osservati anche con correlazione parziale.


**Metodi per costruire un classificatore Ensemble**

* **Manipolazione della distribuzione dei dati:**
    * Creazione di più insiemi di addestramento tramite campionamento casuale del dataset originale, secondo una specifica distribuzione di probabilità.
    * Ogni insieme genera un classificatore.


**Bagging (Bootstrap Aggregating)**

* **Tecnica:** Crea molteplici versioni dello stesso modello, addestrandole su sottoinsiemi di dati campionati con ricampionamento.
    * **Procedura:**
        1. Generare *k* sottoinsiemi  $D_i$ (dimensione N) campionando con rimpiazzo i dati originali.
        2. Addestrare *k* modelli $C_i$ su ciascun $D_i$.
        3. Classificare tramite voto a maggioranza tra i *k* modelli.
* **Effetto del ricampionamento:** Ogni $D_i$ contiene circa il 63% dei dati originali, con duplicati e omissioni, generando diversità tra i modelli $C_i$.
* **Vantaggi:** Migliora la generalizzazione riducendo la varianza dei modelli base (efficace per modelli instabili ad alta varianza).
* **Limiti:** Potrebbe non migliorare (o addirittura peggiorare) le prestazioni per modelli stabili a bassa varianza, a causa del bias inalterato e della dimensione ridotta dei $D_i$.


**Boosting**

* **Tecnica:** Ensemble iterativo che adatta i pesi degli esempi di training.
    * Inizialmente, pesi uguali per tutti gli esempi.
    * Iterazione:
        * Addestramento di un nuovo classificatore su un campione pesato.
        * Aggiornamento dei pesi: aumento per errori, diminuzione per correttezza.
        * Enfasi crescente sugli esempi difficili.
    * Predizione finale: combinazione delle predizioni di tutti i classificatori.
    * Differenza da Bagging: usa pesi anziché bootstrap; efficace per modelli stabili a basso bias.
    * Diverse implementazioni con variazioni nell'aggiornamento dei pesi e nella combinazione delle predizioni.

**AdaBoost**

* **Algoritmo di Boosting:** Combina iterativamente classificatori base (deboli) in un classificatore forte.
    * Focus sugli esempi difficili da classificare.
    * Iterazione *i*:
        1. Campionamento di $D_i$ con pesi correnti $w_i$.
        2. Addestramento di $C_i$ su $D_i$.
        3. Calcolo dell'errore pesato $\epsilon_i$ di $C_i$.
        4. Calcolo dell'importanza $\alpha_i$ di $C_i$ (alto se $\epsilon_i$ basso).
        5. Aggiornamento dei pesi $w_i$: aumento per errori, diminuzione per correttezza.
        6. Reinizializzazione dei pesi se $\epsilon_i > 0.5$.
    * Predizione finale: combinazione pesata delle predizioni dei $k$ classificatori base, usando $\alpha_i$ come pesi.
    * Risultato: accuratezza superiore rispetto ai singoli componenti grazie alla correzione degli errori iterativa.



**Gradient Boosting**

* **Principio Fondamentale:** Costruzione iterativa di un modello predittivo tramite somma pesata di modelli deboli.
    * **Iterazione:** Ad ogni iterazione, un nuovo modello debole viene addestrato sui residui del modello precedente, correggendo gli errori.
    * **Ottimizzazione:** Utilizzo del gradient descent per minimizzare una funzione di perdita (dipendente dal task).
    * **Combinazione:** I modelli deboli sono combinati tramite una somma pesata, i cui pesi dipendono dalla riduzione della funzione di perdita.
    * **Iperparametri:** Numero di iterazioni e complessità dei modelli deboli richiedono ottimizzazione.

**Costruire Classificatori Ensemble**

* **Manipolando le Features di Input:**
    * **Random Forest:** Ensemble di alberi decisionali decorrelati.
        * **Addestramento:** Ogni albero è addestrato su un campione bootstrap dei dati e un sottoinsieme casuale delle features ($p \le d$).
        * **Processo:**
            1. Estrazione di un campione bootstrap $D_i$ di N istanze.
            2. Addestramento di un albero $T_i$ su $D_i$, selezionando casualmente $p$ features ad ogni nodo.
            3. Ripetizione per T volte.
        * **Iperparametro `p`:** Influenza la correlazione tra alberi e la potenza predittiva.
        * **Vantaggi:** Migliore generalizzazione grazie alla diversità tra alberi, efficace con dataset grandi e features ridondanti.

* **Manipolando le Etichette di Classe:**
    * **Error-Correcting Output Codes (ECOC):**  Affronta problemi di classificazione multi-classe con elevato numero di classi.


**Trasformazione del problema multi-classe in problemi binari:**

* **Idea principale:**  Risolvere un problema di classificazione multi-classe tramite un ensemble di classificatori binari.

    * **Procedura:**
        * **1. Suddivisione delle classi:**  Le classi vengono divise casualmente in due sottoinsiemi disgiunti, $A_0$ e $A_1$.
        * **2. Ricodica dei dati:**  Gli esempi in $A_0$ sono assegnati alla classe 0, quelli in $A_1$ alla classe 1.
        * **3. Addestramento del classificatore:** Si addestra un classificatore binario sui dati ricodificati.
        * **4. Iterazione:** Si ripetono i passi 1-3 più volte, creando diversi classificatori binari con diverse suddivisioni delle classi.

    * **Classificazione di un nuovo esempio:**
        * **1. Voti dei classificatori:** Ogni classificatore assegna un voto (predizione) alle classi di $A_0$ o $A_1$.
        * **2. Conteggio dei voti:** Si conta il numero di voti per ogni classe originale.
        * **3. Assegnazione della classe:** L'esempio viene assegnato alla classe con il maggior numero di voti.

    * **Vantaggi:**
        * Semplifica il problema multi-classe in problemi binari più semplici.
        * Permette l'utilizzo di algoritmi di classificazione binaria efficienti.
        * La diversità delle suddivisioni casuali crea un ensemble accurato.


