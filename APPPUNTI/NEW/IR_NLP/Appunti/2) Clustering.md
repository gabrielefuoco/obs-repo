## Weka

Weka è un toolkit per la knowledge discovery scritto in Java, progettato per essere scalabile e versatile. Offre soluzioni per la visualizzazione dei dati, ma il suo focus principale è quello di fornire implementazioni efficienti di algoritmi di machine learning.

### Utilizzo di Weka

Weka può essere utilizzato in diverse fasi del processo di knowledge discovery. Può essere utilizzato per:

* **Prototipazione:** Weka offre un ambiente di sviluppo rapido per sperimentare diversi algoritmi di machine learning.
* **Implementazione:** Weka può essere integrato in applicazioni reali per eseguire analisi predittive e di classificazione.

### Caratteristiche di Weka

Weka offre una vasta gamma di funzionalità, tra cui:

* **Algoritmi di apprendimento supervisionato e non supervisionato:** Weka include algoritmi per la classificazione, la regressione, il clustering, la riduzione della dimensionalità e l'estrazione di regole associative.
* **Pre-processing dei dati:** Weka offre una serie di tecniche per la pulizia, la trasformazione e la preparazione dei dati per l'analisi.
* **Integrazione con DBMS:** Weka può essere utilizzato per accedere ai dati direttamente da database relazionali tramite driver JDBC.
* **Interfaccia grafica:** Weka offre un'interfaccia grafica intuitiva per la costruzione di pipeline di knowledge discovery e la visualizzazione dei risultati.

## Modelli di Rappresentazione del Testo: Vector Space Model

Il VSM rappresenta i documenti come vettori in uno spazio vettoriale. Ogni dimensione di questo spazio corrisponde a un termine del vocabolario. Il valore di ogni dimensione rappresenta la rilevanza del termine corrispondente nel documento.

### La Funzione TF-IDF

La rilevanza dei termini viene calcolata utilizzando la funzione **TF-IDF (Term Frequency-Inverse Document Frequency)**. Questa funzione tiene conto della frequenza del termine nel documento (TF) e della sua rarità nell'intero corpus (IDF).

### Bag of Words Model

Il VSM è spesso associato al **Bag of Words Model (BoW)**. Il BoW considera i documenti come insiemi di parole, ignorando l'ordine e la struttura sintattica. In altre parole, un documento è visto come un insieme di parole, senza tenere conto della loro posizione o delle relazioni grammaticali tra loro.

### Limiti del VSM e del BoW

Il VSM e il BoW, pur essendo semplici ed efficaci, presentano limiti significativi: la *mancanza di struttura sintattica* porta a perdita di informazioni semantiche, e la *sensibilità alla frequenza dei termini* può causare sovra-rappresentazione di termini comuni e sottorappresentazione di termini rari. L'esempio delle frasi "Il gatto insegue il topo" e "Il topo insegue il gatto" evidenzia come il BoW, ignorando l'ordine delle parole, fallisca nel distinguere significati diversi

#### Soluzioni Alternative

Per ovviare a questo limite, si possono utilizzare modelli di rappresentazione del testo che tengono conto della struttura sintattica, come ad esempio:

* **N-grammi:** Gli N-grammi sono sequenze di N parole consecutive. Ad esempio, la frase "Il gatto insegue il topo" contiene i seguenti bigrammi (N=2): "Il gatto", "gatto insegue", "insegue il", "il topo".
* **Modelli di linguaggio:** I modelli di linguaggio sono in grado di prevedere la probabilità di una parola data la sequenza di parole precedenti. Questo permette di tenere conto del contesto e della struttura sintattica del testo.

## L'Importanza della Posizione delle Parole

Consideriamo l'idea di associare la posizione di una parola all'interno di una frase al suo significato. Ad esempio, la parola "gatto" potrebbe avere un significato diverso se si trova all'inizio o alla fine di una frase.

### Benefici dell'Etichettatura della Posizione

L'etichettatura della posizione delle parole può portare a diversi vantaggi:

* **Miglioramento della classificazione:** La posizione delle parole può fornire informazioni utili per la classificazione dei documenti. Ad esempio, la presenza di una parola chiave all'inizio di una frase potrebbe indicare un argomento principale.
* **Miglioramento della ricerca:** La posizione delle parole può essere utile per la ricerca di informazioni specifiche. Ad esempio, una query che richiede una frase esatta potrebbe beneficiare dell'indicizzazione della posizione delle parole.

### Limiti dell'Etichettatura della Posizione

Tuttavia, l'etichettatura della posizione delle parole presenta anche alcuni limiti:

* **Complessità:** L'aggiunta di informazioni sulla posizione delle parole aumenta la complessità del modello di rappresentazione del testo.
* **Rilevanza:** La posizione delle parole non è sempre un indicatore affidabile del significato. Ad esempio, in alcune lingue, l'ordine delle parole è meno importante rispetto ad altre.

## Cluto

Cluto è un algoritmo progettato per il clustering di grandi dataset sparsi e ad alta dimensionalità. Si basa su un'euristica per ottimizzare una funzione obiettivo non esplicitata

##### Caratteristiche:

* **Funzione obiettivo:** Un metodo per risolvere un problema di clustering è un metodo che deve ottimizzare una determinata funzione obiettivo. La funzione obiettivo non è esplicitata: si utilizza un'euristica.
* **Euristica, metodo di ottimizzazione e learner:** I termini euristica, metodo di ottimizzazione e learner vengono spesso usati in modo intercambiabile, ma esistono delle differenze. Ad esempio, un algoritmo che richiede una fase di apprendimento (learner) o che raggiunge il risultato tramite una funzione di ottimizzazione viene erroneamente chiamato euristica. Nonostante ciò, l'euristica può essere anche un metodo iterativo.
* **Identificazione delle feature:** Cluto consente di identificare, nei vari cluster, le dimensioni (feature) che meglio caratterizzano e discriminano il cluster.
* **Visualizzazione delle relazioni:** Fornisce strumenti modulari per visualizzare le relazioni tra cluster, oggetti e feature.
* **Utilizzo:** Può essere utilizzato sia come programma stand-alone sia tramite API.
* **Tipi di cluster:** Supporta i b-cluster e gli s-cluster.
* **Input:** L'input può essere una matrice di dati (matrice documenti-termini, input classico di k-means) o una matrice di similarità (input classico per l'agglomerativa).

## Clustering

##### Direct k-way Clustering

- Il direct k-way clustering è un tipo di k-means, un algoritmo di clustering partizionale che assegna ogni punto dati a uno dei k cluster.

##### Bisetting k-way Clustering

- A differenza del direct k-way clustering, inizia con un unico cluster e lo suddivide iterativamente in due sotto-cluster (k-1 volte) fino a raggiungere k cluster.
- La scelta del cluster da suddividere può seguire diverse strategie (es. il cluster più grande o quello con maggiore varianza).
##### Graph Partitioning Based

- Questo metodo si basa sul concetto di min cut.
- Ad ogni iterazione, si rompono i legami con i valori di similarità più bassi.
- Il numero di cluster desiderato (k) rimane come input.
- Si ottengono m componenti connesse, che portano a un totale di k+m cluster.

## Clustering Partizionale a k-vie e Clustering Gerarchico Agglomerativo

Una soluzione di clustering partizionale a k-vie può essere utilizzata come input per un metodo di clustering gerarchico agglomerativo. Il k-means definisce quindi un bias per questo tipo di clustering.

##### Impostazione di k:

Il valore di k viene solitamente impostato a $\sqrt{n}$, dove n è il numero di punti dati.

##### K-means come input per il clustering gerarchico:

Il k-means trova k cluster che diventano le foglie per un clustering gerarchico agglomerativo.

##### Benefici di questo approccio:

- **Accuratezza:** Il k-means è un algoritmo dinamico che cerca di minimizzare l'errore quadratico, incentivando la riallocazione degli oggetti tra i cluster. Questa proprietà non è presente nel clustering gerarchico agglomerativo, dove una volta effettuato il merge non è possibile riconsiderare la scelta. Questo può avere un impatto sull'accuratezza del clustering.
- **Efficienza:** Partendo da $\sqrt{n}$ foglie, il costo del clustering è $O(n \cdot \log \sqrt{n})$. Questo impatta positivamente sull'efficienza del processo.

Il costo quadratico rimane lineare al numero di oggetti iniziali.

Traiamo beneficio dalla dinamicità del k-means, che ha risolto il problema degli errori nelle fasi iniziali. Ad esempio, nel dendrogramma, i branch sono molto bassi all'inizio, e poi aumentano in altezza di volta in volta.

La distanza tra l'ottimo globale e quello locale è determinata nei primi passi dell'algoritmo.

Con questa combinazione in cascata, otteniamo un algoritmo che fa da trade-off tra i due e fa leva sui punti di forza di entrambi.

## Formato dei File di Input

Il formato dei file di input per l'algoritmo di clustering può essere denso o sparso.

##### Formato Sparso

* **Numero di righe e colonne:** La prima riga del file contiene il numero di righe (oggetti), il numero di colonne (feature) e il numero di entry non nulle. Questo serve per inizializzare la struttura dati.
* **Formato delle righe successive:** Ogni riga successiva rappresenta un oggetto e contiene l'indice della colonna (feature) e il valore corrispondente.

##### Formato Denso

* **Numero di righe e colonne:** La prima riga del file contiene il numero di righe (oggetti) e il numero di colonne (feature).
* **Formato delle righe successive:** Ogni riga successiva rappresenta un oggetto e contiene i valori per tutte le feature.

## File di Etichette

Oltre al file di input della matrice, possono essere forniti file di etichette opzionali:

* **`rlabelfile`:** Memorizza l'etichetta per ogni riga della matrice. Utile per la visualizzazione.
* **`clabelfile`:** Analogo a `rlabelfile`, memorizza le etichette per ogni colonna della matrice.
* **`rclassfile`:** Memorizza l'etichetta di classe per ogni oggetto. Questo file è importante per la valutazione della soluzione di clustering, in quanto fornisce una classificazione di riferimento.

## Valutazione del Clustering

Utilizzando il file `rclassfile`, possiamo confrontare la soluzione di clustering ottenuta con una classificazione di riferimento. Questo permette di calcolare metriche come:

* **Precision:** Rappresenta la proporzione di oggetti correttamente classificati in un cluster rispetto al numero totale di oggetti nel cluster.
* $\frac{tp}{tp+fp}$
* **Recall:** Rappresenta la proporzione di oggetti correttamente classificati in un cluster rispetto al numero totale di oggetti della classe.
* $\frac{tp}{tp+fn}$
* **F-measure:** È la media armonica di precision e recall.
* $\frac{2pr}{p+r}$
## Entropia

L'entropia può essere utilizzata per valutare la qualità del clustering.

##### Definizione:

Sia $C = \{C_1, ..., C_k\}$ l'insieme dei cluster e $C^* = \{C_1^*, ..., C_h^*\}$ la classificazione di riferimento. L'entropia di un cluster $j \in \{1, ..., k\}$ è definita come:

$$E_j = -\sum_{i=1}^h \ Pr(C_i^* | C_j) \ \log(Pr(C_i^* | C_j))$$

dove:

* $Pr(C_i^* | C_j)$ è la probabilità che un oggetto appartenga alla classe $C_i^*$ dato che appartiene al cluster $C_j$.

##### Stima della Probabilità:

La probabilità $Pr(C_i^* | C_j)$ può essere stimata come la frequenza relativa:

$$Pr(C_i^* | C_j) = \frac{|C_i^* \cap C_j|}{|C_j|}$$

dove:

* $|C_i^* \cap C_j|$ è il numero di oggetti che appartengono sia alla classe $C_i^*$ che al cluster $C_j$.
* $|C_j|$ è il numero di oggetti nel cluster $C_j$.

##### Interpretazione:

L'entropia di un cluster misura la sua omogeneità rispetto alla classificazione di riferimento. Un'entropia bassa indica che il cluster è composto principalmente da oggetti della stessa classe.

##### Relazione con Precision e Recall:

L'entropia è strettamente correlata a precision e recall. La probabilità $Pr(C_i^* | C_j)$ corrisponde alla precision del cluster $j$ per la classe $i$. La recall sarebbe data dal rapporto tra l'intersezione tra il cluster $j$ e la classe $i$ e la cardinalità della classe $i$.

#### Formato del File di Output

Il formato del file di output è un unico file con tante righe quanti sono gli oggetti. Ogni riga contiene un valore intero che corrisponde all'ID di classe, ovvero l'ID della riga associato al cluster.

#### Tool per Clustering Partizionale

Questo tool è progettato per eseguire il clustering partizionale, assegnando ogni oggetto a un solo cluster. L'assegnazione va da 0 a `num cluster - 1`.

#### Z-Score

Se specificato, ogni riga del file di output conterrà due numeri in più: *internal z-score* ed *external z-score*.

* **Internal z-score:** Indica quanto l'oggetto è vicino al centroide del cluster a cui è stato assegnato.
* **External z-score:** Indica quanto l'oggetto è vicino ai centroidi degli altri cluster.

###### Sicurezza

La sicurezza non è implementata in questo tool.

###### Triflie

Pluto fornisce anche un *triflie*, ottenuto eseguendo un clustering gerarchico agglomerativo partendo da un clustering partizionale a k-vie.

## Funzione Obiettivo

Tutto è in funzione della similarità. $Ɛ_1$ è una versione già normalizzata della funzione obiettivo, ed è la funzione di default.

Se si tiene conto sia della compattezza che della separabilità dei cluster, l'obiettivo è minimizzare la funzione obiettivo.

## Log di Pluto

Il log di Pluto fornisce informazioni dettagliate sul processo di clustering.
In particolare, riporta:

* **Riepilogo dei parametri:** parametri utilizzati nell'esecuzione.
* **Dettaglio sulla soluzione di cluster:** informazioni su ogni cluster.
* **Entropia e purezza (opzionale):** se fornito un file di etichette di classe (`rclassfile`).
* **Tempistica:** durata del processo di clustering.

##### Document Clustering

Nel caso di *document clustering*, con l'opzione `-showfeatures`, il log mostra le *feature descrittive* (caratterizzano il cluster) e *discriminative* (distinguono il cluster dagli altri).
L'ideale è che i due insiemi coincidano, indicando cluster ben caratterizzati e distinti

### Analisi dei Cluster

Per ogni cluster, è possibile analizzare le caratteristiche più significative in due modi:

##### Caratteristiche più descrittive:

* Si identificano le **L** caratteristiche che contribuiscono maggiormente alla similarità tra gli oggetti all'interno del cluster.
* Per ogni caratteristica, si calcola la **percentuale di similarità** rispetto al cluster.

##### Caratteristiche più discriminative:

* Si identificano le **L** caratteristiche che contribuiscono maggiormente alla dissimilarità tra gli oggetti del cluster e gli oggetti appartenenti agli altri cluster.
* Per ogni caratteristica, si calcola la **percentuale di dissimilarità** rispetto al resto degli oggetti.

