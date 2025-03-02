
## Weka: Un Toolkit per il Machine Learning

Weka è un toolkit Java per la *knowledge discovery*, scalabile e versatile.  Si concentra sull'implementazione efficiente di algoritmi di machine learning, supportando sia la prototipazione che l'implementazione in applicazioni reali.  Offre un'interfaccia grafica intuitiva, algoritmi di apprendimento supervisionato e non supervisionato (classificazione, regressione, clustering, riduzione della dimensionalità, estrazione di regole associative), strumenti per la pre-elaborazione dei dati e l'integrazione con DBMS tramite JDBC.  Sebbene popolare, Weka compete con altri toolkit come R e Python (con librerie come scikit-learn e TensorFlow).


## Modelli di Rappresentazione del Testo: Vector Space Model (VSM)

Il VSM rappresenta i documenti come vettori in uno spazio vettoriale, dove ogni dimensione corrisponde a un termine del vocabolario e il suo valore rappresenta la rilevanza del termine nel documento, calcolata tramite la funzione TF-IDF (Term Frequency-Inverse Document Frequency).  Spesso associato al Bag of Words Model (BoW), il VSM ignora l'ordine e la struttura sintattica del testo, considerando i documenti come insiemi di parole.


## Limiti del VSM e del BoW

Il VSM e il BoW, pur essendo semplici ed efficaci, presentano limiti significativi: la mancanza di struttura sintattica porta a perdita di informazioni semantiche, e la sensibilità alla frequenza dei termini può causare sovra-rappresentazione di termini comuni e sottorappresentazione di termini rari.  L'esempio delle frasi "Il gatto insegue il topo" e "Il topo insegue il gatto" evidenzia come il BoW, ignorando l'ordine delle parole, fallisca nel distinguere significati diversi.  Per superare questi limiti, si possono utilizzare modelli alternativi come gli N-grammi, che considerano sequenze di N parole consecutive.

---

# Riassunto del testo sui Modelli di Linguaggio e Clustering

Questo testo tratta due argomenti principali: l'utilizzo dei modelli di linguaggio e le tecniche di clustering.

## Modelli di Linguaggio e Posizione delle Parole

I modelli di linguaggio prevedono la probabilità di una parola dato il contesto precedente, considerando la struttura sintattica.  L'importanza della posizione di una parola all'interno di una frase è sottolineata: la stessa parola può assumere significati diversi a seconda della sua collocazione.  Etichettare la posizione delle parole può migliorare la classificazione dei documenti e la ricerca di informazioni specifiche, ma aumenta la complessità del modello e non è sempre un indicatore affidabile del significato, soprattutto in lingue con ordine delle parole meno rigido.


## Cluto: Un Algoritmo di Clustering

Cluto è un algoritmo progettato per il clustering di grandi dataset sparsi e ad alta dimensionalità.  Si basa su un'euristica per ottimizzare una funzione obiettivo non esplicitata.  Cluto permette l'identificazione delle *feature* che caratterizzano i cluster, la visualizzazione delle relazioni tra cluster, oggetti e *feature*, e supporta diversi tipi di input (matrice dati o matrice di similarità) e tipi di cluster (b-cluster e s-cluster). Può essere utilizzato come programma standalone o tramite API.


## Metodi di Clustering in Cluto

Il testo descrive tre metodi di clustering implementati in Cluto:

* **Direct k-way Clustering:** Un algoritmo k-means che assegna direttamente ogni punto dati a uno dei k cluster.

* **Bisetting k-way Clustering:**  A differenza del direct k-way clustering, inizia con un unico cluster e lo suddivide iterativamente in due sotto-cluster (k-1 volte) fino a raggiungere k cluster.  La scelta del cluster da suddividere può seguire diverse strategie (es. il cluster più grande o quello con maggiore varianza).

* **Graph Partitioning Based Clustering:**  Si basa sul concetto di *min cut*, rompendo iterativamente i legami con i valori di similarità più bassi fino a ottenere k cluster (o più, a seconda della connettività del grafo).

---

## Riassunto dell'Algoritmo di Clustering

Questo documento descrive un algoritmo di clustering ibrido che combina il *k-means* con il clustering gerarchico agglomerativo,  specificando il formato dei file di input e le metriche di valutazione.

### Clustering Ibrido k-means/Gerarchico Agglomerativo

L'algoritmo utilizza il *k-means* come pre-processing per il clustering gerarchico agglomerativo.  Il numero di cluster (*k*) nel *k-means* è tipicamente impostato a $\sqrt{n}$, dove *n* è il numero di punti dati. I *k* cluster ottenuti dal *k-means* diventano le foglie dell'albero gerarchico. Questo approccio offre due vantaggi principali:

* **Accuratezza migliorata:** Il *k-means*, essendo dinamico, minimizza l'errore quadratico riallocando gli oggetti tra i cluster, a differenza del clustering gerarchico agglomerativo che è statico dopo il merge.
* **Efficienza:** Partendo da $\sqrt{n}$ foglie, la complessità computazionale del clustering gerarchico diventa $O(n \cdot \log \sqrt{n})$,  beneficiando dell'efficienza del *k-means* nelle fasi iniziali.


### Formato dei File di Input

I dati possono essere forniti in due formati:

* **Formato Sparso:** La prima riga contiene il numero di righe (oggetti), colonne (feature) e entry non nulle. Le righe successive contengono l'indice della feature e il suo valore.
* **Formato Denso:** La prima riga contiene il numero di righe e colonne. Le righe successive contengono i valori di tutte le feature per ogni oggetto.


### File di Etichette (Opzionali)

Possono essere forniti tre file di etichette opzionali:

* `rlabelfile`: Etichette per ogni riga (oggetto).
* `clabelfile`: Etichette per ogni colonna (feature).
* `rclassfile`: Etichette di classe per ogni oggetto, cruciali per la valutazione.


### Valutazione del Clustering

Utilizzando `rclassfile`, si possono calcolare le seguenti metriche:

* **Precision:** $\frac{tp}{tp+fp}$
* **Recall:** $\frac{tp}{tp+fn}$
* **F-measure:** $\frac{2pr}{p+r}$


### Entropia

L'entropia misura la qualità del clustering.  Data la partizione dei cluster $C = \{C_1, ..., C_k\}$ e la classificazione di riferimento $C^* = \{C_1^*, ..., C_h^*\}$, l'entropia di un cluster $j$ è:

$$E_j = -\sum_{i=1}^h Pr(C_i^* | C_j) \log(Pr(C_i^* | C_j))$$

dove $Pr(C_i^* | C_j)$ è la probabilità che un oggetto appartenga alla classe $C_i^*$ dato che appartiene al cluster $C_j$.

---

## Riassunto di Pluto: Tool per Clustering Partizionale

Pluto è un tool per clustering partizionale che assegna ogni oggetto ad un singolo cluster (da 0 a `num cluster - 1`).  Il risultato è un file contenente l'ID del cluster per ogni oggetto.  Opzionalmente, possono essere aggiunti lo *z-score interno* (prossimità al centroide del cluster assegnato) e lo *z-score esterno* (prossimità ai centroidi degli altri cluster).  La probabilità di appartenenza di un oggetto ad una classe, data l'assegnazione a un cluster, è stimata come frequenza relativa:

$$Pr(C_i^* | C_j) = \frac{|C_i^* \cap C_j|}{|C_j|}$$

dove $|C_i^* \cap C_j|$ è il numero di oggetti nella classe $C_i^*$ e nel cluster $C_j$, e $|C_j|$ è il numero di oggetti nel cluster $C_j$.  Questa probabilità corrisponde alla *precision* del cluster per la classe.  L'entropia di un cluster indica la sua omogeneità: bassa entropia significa alta omogeneità.  Pluto non include funzionalità di sicurezza.  Inoltre, Pluto genera un *triflie* tramite clustering gerarchico agglomerativo a partire da un clustering partizionale k-means.

La funzione obiettivo di Pluto, $\Ɛ_1$ (già normalizzata), mira a minimizzare la distanza tra gli oggetti dello stesso cluster e massimizzare la distanza tra i cluster, considerando sia compattezza che separabilità.

Il log di Pluto riporta:

* **Riepilogo dei parametri:**  parametri utilizzati nell'esecuzione.
* **Dettaglio sulla soluzione di cluster:** informazioni su ogni cluster.
* **Entropia e purezza (opzionale):** se fornito un file di etichette di classe (`rclassfile`).
* **Tempistica:** durata del processo di clustering.

Nel caso di *document clustering*, con l'opzione `-showfeatures`, il log mostra le *feature descrittive* (caratterizzano il cluster) e *discriminative* (distinguono il cluster dagli altri).  L'ideale è che i due insiemi coincidano, indicando cluster ben caratterizzati e distinti.  L'analisi dei cluster può essere effettuata identificando le **L** caratteristiche che contribuiscono maggiormente alla similarità interna al cluster, calcolando la percentuale di similarità per ciascuna.

---

L'analisi si concentra sull'identificazione delle caratteristiche più discriminative per distinguere i cluster.  Questo processo individua un sottoinsieme di *L* caratteristiche che massimizzano la dissimilarità tra gli oggetti all'interno di un cluster e quelli appartenenti ad altri cluster.  Per ciascuna di queste *L* caratteristiche, viene calcolata la percentuale di dissimilarità rispetto al resto dei dati, quantificando così il suo contributo alla separazione tra i cluster.

---
