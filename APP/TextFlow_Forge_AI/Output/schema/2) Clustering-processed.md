
**I. Weka: Toolkit per la Knowledge Discovery**

* **A. Utilizzo:**
    * 1. Prototipazione di algoritmi di machine learning.
    * 2. Implementazione in applicazioni reali per analisi predittive e di classificazione.
* **B. Caratteristiche:**
    * 1. Algoritmi di apprendimento supervisionato e non supervisionato (classificazione, regressione, clustering, riduzione dimensionalità, estrazione regole associative).
    * 2. Pre-processing dei dati (pulizia, trasformazione, preparazione).
    * 3. Integrazione con DBMS tramite driver JDBC.
    * 4. Interfaccia grafica intuitiva per la costruzione di pipeline e visualizzazione dei risultati.


**II. Modelli di Rappresentazione del Testo: Vector Space Model (VSM)**

* **A. VSM:** Rappresenta i documenti come vettori in uno spazio vettoriale, dove ogni dimensione corrisponde a un termine del vocabolario e il valore rappresenta la rilevanza del termine.
* **B. TF-IDF:** La rilevanza dei termini è calcolata con la funzione  **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* **C. Bag of Words (BoW):** Il VSM è spesso associato al BoW, che considera i documenti come insiemi di parole, ignorando l'ordine e la struttura sintattica.
* **D. Limiti VSM e BoW:**
    * 1. Mancanza di struttura sintattica, portando a perdita di informazioni semantiche.
    * 2. Sensibilità alla frequenza dei termini, causando sovra-rappresentazione di termini comuni e sottorappresentazione di termini rari.  Esempio: "Il gatto insegue il topo" vs "Il topo insegue il gatto".
* **E. Soluzioni Alternative:**
    * 1. **N-grammi:** Sequenze di N parole consecutive (es. bigrammi: "Il gatto", "gatto insegue", etc.).
    * 2. **Modelli di linguaggio:** Prevedono la probabilità di una parola dato il contesto precedente.


**III. Importanza della Posizione delle Parole**

* Associazione del significato di una parola alla sua posizione all'interno di una frase.

---

**I. Etichettatura della Posizione delle Parole**

* **Benefici:**
    * Miglioramento della classificazione (es. parole chiave all'inizio di una frase indicano l'argomento principale).
    * Miglioramento della ricerca (es. query di frasi esatte).
* **Limiti:**
    * Aumenta la complessità del modello di rappresentazione del testo.
    * La posizione non è sempre un indicatore affidabile del significato (dipende dalla lingua).


**II. Algoritmo Cluto**

* **Scopo:** Clustering di grandi dataset sparsi e ad alta dimensionalità.
* **Caratteristiche:**
    * Funzione obiettivo non esplicitata; si usa un'euristica (l'euristica, il metodo di ottimizzazione e il learner sono termini spesso usati in modo intercambiabile).
    * Identifica le feature che caratterizzano e discriminano i cluster.
    * Visualizza le relazioni tra cluster, oggetti e feature.
    * Utilizzo stand-alone o tramite API.
    * Supporta b-cluster e s-cluster.
    * Input: matrice di dati (documenti-termini) o matrice di similarità.


**III. Metodi di Clustering**

* **Clustering Partizionale:**
    * **Direct k-way Clustering:** Tipo di k-means; assegna direttamente ogni punto dati a uno dei k cluster.
    * **Bisetting k-way Clustering:** Inizia con un cluster e lo suddivide iterativamente in due (k-1 volte) fino a raggiungere k cluster.  La scelta del cluster da suddividere può variare (es. cluster più grande o con maggiore varianza).
    * **Graph Partitioning Based:** Basato sul concetto di *min cut*; rompe i legami con i valori di similarità più bassi ad ogni iterazione; richiede k come input; genera k+m cluster (m = componenti connesse).

* **Relazione tra Clustering Partizionale e Gerarchico:** Una soluzione di clustering partizionale a k-vie può essere usata come input per un metodo di clustering gerarchico agglomerativo.

---

**I. K-means e Clustering Gerarchico**

* **A. Impostazione di k:**  $k = \sqrt{n}$ (n = numero di punti dati)
* **B. K-means come input per il clustering gerarchico:**
    * Il k-means genera k cluster, che diventano le foglie di un clustering gerarchico agglomerativo.
* **C. Benefici dell'approccio combinato:**
    * **1. Accuratezza:** Il k-means minimizza l'errore quadratico, permettendo riallocazioni; il clustering gerarchico no.
    * **2. Efficienza:** Costo computazionale $O(n \cdot \log \sqrt{n})$, lineare rispetto al numero di oggetti.  La dinamicità del k-means mitiga gli errori iniziali.

**II. Formato dei File di Input**

* **A. Formato Sparso:**
    * Prima riga: numero righe, colonne, entry non nulle.
    * Righe successive: indice colonna, valore.
* **B. Formato Denso:**
    * Prima riga: numero righe, colonne.
    * Righe successive: valori per tutte le feature.

**III. File di Etichette (Opzionali)**

* **A. `rlabelfile`:** Etichette per ogni riga (visualizzazione).
* **B. `clabelfile`:** Etichette per ogni colonna (visualizzazione).
* **C. `rclassfile`:** Etichette di classe per ogni oggetto (valutazione).

**IV. Valutazione del Clustering (usando `rclassfile`)**

* **A. Precision:** $\frac{tp}{tp+fp}$
* **B. Recall:** $\frac{tp}{tp+fn}$
* **C. F-measure:** $\frac{2pr}{p+r}$

**V. Entropia (Valutazione della Qualità del Clustering)**

* **A. Definizione:**  Data la partizione in cluster $C = \{C_1, ..., C_k\}$ e la classificazione di riferimento $C^* = \{C_1^*, ..., C_h^*\}$, l'entropia misura la qualità del clustering (definizione incompleta nel testo originale).


---

**Pluto Clustering Tool: Schema Riassuntivo**

I. **Clustering Partizionale:**

   * A. **Funzione Obiettivo:** Minimizzare $Ɛ_1$ (funzione obiettivo normalizzata, di default), considerando compattezza e separabilità dei cluster.  Basata sulla similarità.
   * B. **Assegnazione Oggetti:** Ogni oggetto assegnato a un solo cluster (da 0 a `num cluster - 1`).
   * C. **Output:** Un file con una riga per oggetto, contenente l'ID del cluster assegnato.  Opzionalmente, include *internal z-score* ed *external z-score*.
      * 1. *Internal z-score*: Distanza dal centroide del cluster assegnato.
      * 2. *External z-score*: Distanza dai centroidi degli altri cluster.
   * D. **Stima dell'Entropia:**
      * 1. Formula:  $E_j = -\sum_{i=1}^h \ Pr(C_i^* | C_j) \ \log(Pr(C_i^* | C_j))$
      * 2. Probabilità: $Pr(C_i^* | C_j) = \frac{|C_i^* \cap C_j|}{|C_j|}$
      * 3. Interpretazione: Misura l'omogeneità del cluster rispetto alla classificazione di riferimento. Entropia bassa indica alta omogeneità.
      * 4. Relazione con Precision e Recall: $Pr(C_i^* | C_j)$ corrisponde alla precision; la recall è $\frac{|C_i^* \cap C_j|}{|C_i^*|}$.


II. **Clustering Gerarchico Agglomerativo (Triflie):**

   * A.  Pluto genera un clustering gerarchico agglomerativo a partire dal clustering partizionale k-means.


III. **Log di Pluto:**

   * A. **Informazioni:** Parametri utilizzati, dettaglio sulla soluzione di cluster, tempi di esecuzione.
   * B. **Opzionale (con `rclassfile`):** Entropia e purezza.
   * C. **Document Clustering (con `-showfeatures`):** Feature descrittive (similarità interna) e discriminative (differenze tra cluster).  L'ideale è che coincidano.


IV. **Analisi dei Cluster:**

   * A. **Caratteristiche Descrittive:** Le **L** caratteristiche che contribuiscono maggiormente alla similarità interna al cluster, con la relativa percentuale di similarità.
   * B. **Caratteristiche Discriminative:** (Non completamente descritto nel testo).


V. **Sicurezza:** Non implementata.

VI. **Formato File Output:** Un singolo file, una riga per oggetto, con l'ID del cluster assegnato.

---

**Identificazione delle Caratteristiche Discriminative**

* **Scopo:** Identificare le caratteristiche che meglio distinguono i cluster.

    * **Metodo:**  Si individuano le *L* caratteristiche più discriminative.
    * **Misurazione:** Per ogni caratteristica (*L* caratteristiche totali), si calcola la percentuale di dissimilarità rispetto agli oggetti appartenenti ad altri cluster.

---
