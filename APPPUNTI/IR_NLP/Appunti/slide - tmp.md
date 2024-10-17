## Perché le probabilità nell'IR?

I sistemi IR tradizionali si basano sull'abbinamento tra query e documenti in uno spazio di termini di indicizzazione "semanticamente impreciso". Le probabilità forniscono una base solida per il ragionamento incerto. Possiamo usare le probabilità per quantificare le nostre incertezze di ricerca?

**Informazioni utente:**

* **Bisogno:** Cosa sta cercando l'utente?
* **Documenti:** Quali documenti sono disponibili?
* **Rappresentazione del documento:** Come vengono rappresentati i documenti?
* **Rappresentazione della query:** Come viene rappresentata la query?

**Come abbinare?**

* **Ipotesi incerta:** il documento contiene contenuti pertinenti?
* **Comprensione incerta:** del bisogno dell'utente.

## IR Probabilistico

### Modello di recupero probabilistico classico

* **Principio di classificazione probabilistica:** classifica i documenti in base alla probabilità di pertinenza.
* **Modello di indipendenza binaria (BIM):** un modello semplificato che assume che i termini siano indipendenti l'uno dall'altro.
* **Categorizzazione di testo Naïve Bayes:** un'applicazione del BIM alla categorizzazione di testo.
* **(Okapi) BM25:** un modello più sofisticato che tiene conto della frequenza dei termini e della lunghezza del documento.
* **Reti bayesiane per il recupero di testo:** un approccio più generale che consente di modellare le dipendenze tra i termini.

### Approccio del modello linguistico all'IR

* **Uno sviluppo importante nell'IR degli anni 2000.**

### Metodi probabilistici: i più antichi ma attualmente i più attuali nell'IR

* **Originariamente, grandi idee, ma non hanno vinto in termini di prestazioni.**
* **Ora è diverso.**

**Personaggi chiave:**

* **Stephen Robertson**
* **Keith van Rijsbergen**
* **Karen Spärck Jones**

## Il principio di classificazione probabilistica (PRP)

**Il metodo di classificazione è il nucleo dei sistemi IR moderni:**

* **In quale ordine presentiamo i documenti all'utente?**
* **Vogliamo che il documento "migliore" sia il primo, il secondo migliore il secondo, ecc.**

**Idea:** Classifica in base alla probabilità di pertinenza del documento rispetto al bisogno di informazioni.

* **p(R=1|doci, query):** "Se la risposta di un sistema di recupero di riferimento a ciascuna richiesta è una classificazione dei documenti nella collezione in ordine di probabilità decrescente di pertinenza per l'utente che ha inviato la richiesta, dove le probabilità sono stimate nel modo più accurato possibile sulla base dei dati che sono stati resi disponibili al sistema per questo scopo, l'efficacia complessiva del sistema per il suo utente sarà la migliore che si possa ottenere sulla base di quei dati."

## Il principio di classificazione probabilistica (PRP)

* **x:** un documento nella collezione
* **R:** pertinenza di un documento rispetto a una query data (fissata)
* **R=1** significa "pertinente" e **R=0** "non pertinente"

**Bisogna trovare p(R=1|x), ovvero la probabilità che un documento x sia pertinente.**

* **p(R=1), p(R=0):** probabilità a priori di recuperare un documento pertinente (non pertinente) a caso.
* **p(x|R=1), p(x|R=0):** probabilità che, se viene recuperato un documento pertinente (non pertinente), sia x.

## Il principio di classificazione probabilistica (PRP)

**Obiettivo del PRP:** classificare tutti i documenti in ordine decrescente di p(R=1|d, q).

**Regola di decisione bayesiana ottimale:** d è pertinente se e solo se P(R = 1|d, q) > P(R = 0|d, q).

**Teorema:** Usare il PRP è ottimale, in quanto minimizza la perdita (rischio bayesiano) con una perdita di 1/0.

* **Dimostrabile se tutte le probabilità sono note correttamente [ad esempio, Ripley 1996].**

## Strategia di recupero probabilistico

**Innanzitutto, stimare come ciascun termine contribuisce alla pertinenza:**

* **Come la frequenza dei termini e la lunghezza del documento influenzano i giudizi sulla pertinenza del documento?**
    * **Non per niente nel BIM.**
    * **BM25 fornisce una risposta più sfumata.**

**Combinare per trovare la probabilità di pertinenza del documento.**

**Classificare i documenti in ordine di probabilità decrescente.** 


## Binary Independence Model

Tradizionalmente utilizzato in combinazione con PRP, il Binary Independence Model rappresenta i documenti come vettori di incidenza binaria dei termini:

* **"Binario"** = Booleano
* **"Indipendenza"**: i termini si verificano nei documenti in modo indipendente.

**Rappresentazione dei Documenti:**

* **Presenza del termine:**  `1` se il termine *i* è presente nel documento *x*.
* **Assenza del termine:** `0` se il termine *i* non è presente nel documento *x*.

**Modellazione delle Query:**

* Le query sono rappresentate come vettori di incidenza binaria dei termini.

**Calcolo della Probabilità di Rilevanza:**

Dato una query *q*, per ogni documento *d*:

* Calcola la probabilità di rilevanza *p(R|q,d)*.
* Sostituisci *p(R|q,d)* con *p(R|q,x)*, dove *x* è il vettore di incidenza binaria dei termini che rappresenta *d*.
* L'obiettivo è solo il ranking.

**Utilizzo delle Odds e della Regola di Bayes:**

* **Odds:** Rapporto tra la probabilità di un evento e la probabilità del suo complemento.
* **Regola di Bayes:** Teorema che descrive la probabilità condizionata di un evento.

**Assunzione di Indipendenza:**

* **p(R|q,x) = p(R|q) * p(x|R,q) / p(x|q)**
* **p(x|R,q) = p(x1|R,q) * p(x2|R,q) * ... * p(xn|R,q)**

**Stima dei Parametri:**

* **p(R|q)**: Probabilità di rilevanza data la query.
* **p(x|R,q)**: Probabilità di osservare il vettore di incidenza *x* dato che il documento è rilevante.
* **p(x|q)**: Probabilità di osservare il vettore di incidenza *x* data la query.

**Stima dei Termini Non Corrispondenti:**

* Per tutti i termini non presenti nella query (qi=0), si assume che:
    * **R=1 (rilevante):** xi = 0 (termine assente) con probabilità (1 – pi).
    * **R=0 (non rilevante):** xi = 0 (termine assente) con probabilità (1 – ri).

**Formulazione del Retrieval Status Value (RSV):**

* **RSV = p(R|q,x) / p(¬R|q,x)**
* **RSV = p(R|q) * p(x|R,q) / p(¬R|q) * p(x|¬R,q)**

**Stima dei Coefficienti RSV:**

* **ci = log(pi / ri)**
* **ci:** Log odds ratio (rapporto di probabilità logaritmico).
* **pi:** Probabilità di presenza del termine *i* nei documenti rilevanti.
* **ri:** Probabilità di presenza del termine *i* nei documenti non rilevanti.

**Stima dei Coefficienti RSV in Teoria:**

* Per ogni termine *i*, si crea una tabella di conteggi dei documenti (n=dfi).
* **Stime:**
    * **pi = dfi / N** (probabilità di presenza del termine *i* nei documenti rilevanti).
    * **ri = n / N** (probabilità di presenza del termine *i* nei documenti non rilevanti).

**Stima di ri:**

* Se i documenti non rilevanti sono approssimati dall'intera collezione, allora **ri = n/N**.
* **Inverse Document Frequency (IDF):** Un concetto chiave di ponderazione dei termini.

**Stima di pi:**

* **pi** non può essere facilmente approssimata.
* **Metodi di stima di pi:**
    * Utilizzo di documenti rilevanti noti.
    * Costante (Croft e Harper combination match).
    * Proporzionale alla probabilità di occorrenza nella collezione (Greiff, SIGIR 1998).

**Probabilistic Relevance Feedback:**

* Processo iterativo di stima per ottenere una stima più accurata di **pi**.
* **Passaggi:**
    1. Sulla base delle stime attuali di **pi** e **ri**, si ipotizza una descrizione preliminare dei documenti **R=1**.
    2. Si utilizza questa descrizione per recuperare un set di documenti candidati rilevanti da fornire all'utente.
    3. Si utilizza il feedback dell'utente per migliorare le stime di **pi** e **ri**.

**Conclusione:**

Il Binary Independence Model è un modello probabilistico che utilizza l'assunzione di indipendenza dei termini per calcolare la probabilità di rilevanza di un documento rispetto a una query. Il modello utilizza le odds e la regola di Bayes per calcolare il Retrieval Status Value (RSV), che rappresenta il punteggio di un documento per una query. La stima dei parametri del modello è fondamentale per ottenere risultati accurati. Il Probabilistic Relevance Feedback è un processo iterativo che consente di migliorare le stime dei parametri del modello utilizzando il feedback dell'utente.


## Pseudo-relevance Feedback

Pseudo-relevance feedback is an iterative process used to improve the accuracy of document retrieval by automatically estimating the relevance of documents based on user feedback. This process involves repeatedly refining the estimates of document relevance and query term weights.

**Steps:**

1. **Initial Assumptions:**
    - **pi:** The probability of a document being relevant is assumed to be constant for all terms in the query and is initially set to 0.5 (even odds).
    - **ri:** The probability of a term being relevant is calculated as before, based on the term frequency and document frequency.

2. **Initial Ranking:**
    - A fixed-size set of documents (V) is retrieved based on the initial ranking model. This set typically consists of the highest-ranked documents.

3. **Re-estimation of pi and ri:**
    - **pi:** The probability of a document being relevant (pi) is re-estimated based on the distribution of query terms in the retrieved set (V). 
        - **If a term (xi) appears in a document (Vi) within the set V:**
            - pi = |Vi| / |V| (where |Vi| is the number of documents containing xi and |V| is the total number of documents in the set).
    - **ri:** The probability of a term being relevant (ri) is re-estimated based on the assumption that documents not retrieved are not relevant.
        - **ri = (ni - |Vi|) / (N - |V|)** (where ni is the total number of documents containing xi, and N is the total number of documents in the collection).

4. **Iteration:**
    - Steps 2 and 3 are repeated until the ranking converges, meaning that the estimated relevance scores for documents stabilize.

**Convergence Criterion:**

The ranking is considered to have converged when the difference between the estimated relevance scores in consecutive iterations falls below a predefined threshold.

**Ranking Formula:**

The final ranking score (ci) for each document is calculated using the following formula:

**ci = log (|Vi| + ½) / (|V| - |Vi| + 1) + log (N / ni)**

**Benefits of Pseudo-relevance Feedback:**

- **Improved Retrieval Accuracy:** By iteratively refining the estimates of document relevance and query term weights, pseudo-relevance feedback can significantly improve the accuracy of document retrieval.
- **Automatic Relevance Estimation:** It automates the process of estimating document relevance, reducing the need for manual user feedback.
- **Adaptive Ranking:** The ranking model adapts to the specific query and document collection, providing more relevant results.

**Limitations:**

- **Initial Assumptions:** The initial assumptions about pi and ri can influence the final ranking.
- **Convergence Issues:** The ranking may not always converge, especially for complex queries or small document collections.
- **Overfitting:** The model may overfit to the retrieved set, leading to poor performance on unseen documents.

**Conclusion:**

Pseudo-relevance feedback is a valuable technique for improving document retrieval accuracy by automatically estimating document relevance. It offers a balance between user effort and retrieval effectiveness, making it a widely used approach in information retrieval systems.


