
**Schema Riassuntivo**

**I. Weka: Toolkit per Knowledge Discovery**

*   **A.** Definizione: Toolkit in Java per knowledge discovery, scalabile e versatile.
*   **B.** Focus: Implementazioni efficienti di algoritmi di machine learning.
*   **C.** Utilizzo:
    *   **1.** Prototipazione: Sviluppo rapido per sperimentare algoritmi.
    *   **2.** Implementazione: Integrazione in applicazioni per analisi predittive.
*   **D.** Caratteristiche:
    *   **1.** Algoritmi: Apprendimento supervisionato e non supervisionato (classificazione, regressione, clustering, riduzione dimensionalità, regole associative).
    *   **2.** Pre-processing: Tecniche per pulizia, trasformazione e preparazione dati.
    *   **3.** Integrazione DBMS: Accesso a database tramite JDBC.
    *   **4.** Interfaccia Grafica: Costruzione pipeline e visualizzazione risultati.

**II. Modelli di Rappresentazione del Testo: Vector Space Model (VSM)**

*   **A.** Definizione: Documenti rappresentati come vettori in uno spazio vettoriale.
    *   **1.** Dimensioni: Ogni dimensione corrisponde a un termine del vocabolario.
    *   **2.** Valore: Rilevanza del termine nel documento.
*   **B.** Funzione TF-IDF: Calcolo della rilevanza dei termini.
    *   **1.** TF (Term Frequency): Frequenza del termine nel documento.
    *   **2.** IDF (Inverse Document Frequency): Rarità del termine nel corpus.
*   **C.** Bag of Words Model (BoW):
    *   **1.** Definizione: Documenti come insiemi di parole, ignorando ordine e struttura sintattica.
*   **D.** Limiti di VSM e BoW:
    *   **1.** Mancanza di struttura sintattica: Perdita di informazioni semantiche.
    *   **2.** Sensibilità alla frequenza: Sovra-rappresentazione di termini comuni, sottorappresentazione di termini rari.
    *   **3.** Esempio: "Il gatto insegue il topo" vs. "Il topo insegue il gatto".
*   **E.** Soluzioni Alternative:
    *   **1.** N-grammi: Sequenze di N parole consecutive (es. bigrammi: "Il gatto", "gatto insegue").
    *   **2.** Modelli di linguaggio: Previsione della probabilità di una parola data la sequenza precedente.

**III. Importanza della Posizione delle Parole**

*   **A.** Idea: Associare la posizione di una parola al suo significato.

---

**I. Etichettatura della Posizione delle Parole**

   *   **A. Benefici:**
        *   1.  Miglioramento della classificazione: La posizione indica l'argomento principale (es. parola chiave all'inizio).
        *   2.  Miglioramento della ricerca: Utile per la ricerca di frasi esatte.
   *   **B. Limiti:**
        *   1.  Complessità: Aumenta la complessità del modello di rappresentazione del testo.
        *   2.  Rilevanza: La posizione non è sempre un indicatore affidabile del significato (varia tra le lingue).

**II. Cluto: Algoritmo di Clustering per Dataset Sparsi e ad Alta Dimensionalità**

   *   **A. Caratteristiche Principali:**
        *   1.  Funzione Obiettivo: Ottimizzata tramite un'euristica (non esplicitata).
        *   2.  Euristica vs. Metodo di Ottimizzazione vs. Learner: Termini spesso usati in modo intercambiabile, ma con differenze.
        *   3.  Identificazione delle Feature: Individua le dimensioni che meglio caratterizzano e discriminano i cluster.
        *   4.  Visualizzazione: Fornisce strumenti modulari per visualizzare le relazioni tra cluster, oggetti e feature.
        *   5.  Utilizzo: Stand-alone o tramite API.
        *   6.  Tipi di Cluster: Supporta b-cluster e s-cluster.
        *   7.  Input: Matrice di dati (documenti-termini) o matrice di similarità.

**III. Clustering**

   *   **A. Direct k-way Clustering:**
        *   1.  Tipo di k-means: Algoritmo partizionale che assegna ogni punto dati a uno dei k cluster.
   *   **B. Bisecting k-way Clustering:**
        *   1.  Approccio Iterativo: Inizia con un cluster e lo suddivide iterativamente in due sotto-cluster (k-1 volte) fino a raggiungere k cluster.
        *   2.  Strategie di Scelta: Il cluster da suddividere può essere il più grande o quello con maggiore varianza.
   *   **C. Graph Partitioning Based:**
        *   1.  Concetto Chiave: Min cut.
        *   2.  Processo: Rompe i legami con i valori di similarità più bassi ad ogni iterazione.
        *   3.  Input: Numero di cluster desiderato (k).
        *   4.  Output: m componenti connesse, che portano a un totale di k+m cluster.

**IV. Combinazione di Metodi di Clustering**

   *   **A. Integrazione:** Una soluzione di clustering partizionale a k-vie può essere utilizzata come input per un metodo di clustering gerarchico agglomerativo.

---

**Schema Riassuntivo del Testo sul Clustering**

**1. K-means e Clustering Gerarchico**

   *   **1.1 Bias del K-means:** Definisce un bias specifico per il clustering.
   *   **1.2 Impostazione di k:**
        *   $k = \sqrt{n}$, dove n è il numero di punti dati.
   *   **1.3 K-means come Input:** Utilizzato per trovare k cluster che diventano le foglie per il clustering gerarchico agglomerativo.
   *   **1.4 Vantaggi dell'Approccio Combinato:**
        *   **1.4.1 Accuratezza:**
            *   K-means minimizza l'errore quadratico e rialloca gli oggetti.
            *   Il clustering gerarchico agglomerativo non riconsidera i merge.
        *   **1.4.2 Efficienza:**
            *   Costo del clustering: $O(n \cdot \log \sqrt{n})$.
            *   Trade-off tra i punti di forza di entrambi gli algoritmi.

**2. Formato dei File di Input**

   *   **2.1 Formato Sparso:**
        *   **2.1.1 Prima Riga:** Numero di righe, colonne e entry non nulle.
        *   **2.1.2 Righe Successive:** Indice della colonna (feature) e valore corrispondente per ogni oggetto.
   *   **2.2 Formato Denso:**
        *   **2.2.1 Prima Riga:** Numero di righe e colonne.
        *   **2.2.2 Righe Successive:** Valori per tutte le feature per ogni oggetto.

**3. File di Etichette (Opzionali)**

   *   **3.1 `rlabelfile`:** Etichetta per ogni riga (oggetto). Utile per la visualizzazione.
   *   **3.2 `clabelfile`:** Etichetta per ogni colonna (feature).
   *   **3.3 `rclassfile`:** Etichetta di classe per ogni oggetto. Usato per la valutazione del clustering.

**4. Valutazione del Clustering**

   *   **4.1 Confronto con `rclassfile`:** Confronta la soluzione di clustering con una classificazione di riferimento.
   *   **4.2 Metriche di Valutazione:**
        *   **4.2.1 Precision:** $\frac{tp}{tp+fp}$
        *   **4.2.2 Recall:** $\frac{tp}{tp+fn}$
        *   **4.2.3 F-measure:** $\frac{2pr}{p+r}$

**5. Entropia**

   *   **5.1 Definizione:**
        *   $C = \{C_1, ..., C_k\}$: Insieme dei cluster.
        *   $C^* = \{C_1^*, ..., C_h^*\}$: Classificazione di riferimento.
        *   Entropia di un cluster $j \in \{1, ..., k\}$:  (Il testo è incompleto, quindi non posso aggiungere la formula completa dell'entropia)

---

## Schema Riassuntivo del Testo

**1. Entropia del Cluster**

   *   Definizione: Misura l'omogeneità di un cluster rispetto alla classificazione di riferimento.
   *   Formula:  $$E_j = -\sum_{i=1}^h \ Pr(C_i^* | C_j) \ \log(Pr(C_i^* | C_j))$$
   *   Interpretazione:
        *   Entropia bassa: Cluster composto principalmente da oggetti della stessa classe.
   *   Relazione con Precision e Recall:
        *   $Pr(C_i^* | C_j)$ corrisponde alla precision del cluster $j$ per la classe $i$.
        *   Recall: Rapporto tra l'intersezione tra il cluster $j$ e la classe $i$ e la cardinalità della classe $i$.

**2. Stima della Probabilità**

   *   Metodo: Frequenza relativa.
   *   Formula: $$Pr(C_i^* | C_j) = \frac{|C_i^* \cap C_j|}{|C_j|}$$
   *   Definizioni:
        *   $|C_i^* \cap C_j|$: Numero di oggetti che appartengono sia alla classe $C_i^*$ che al cluster $C_j$.
        *   $|C_j|$: Numero di oggetti nel cluster $C_j$.

**3. Tool per Clustering Partizionale**

   *   Tipo di Clustering: Partizionale (ogni oggetto assegnato a un solo cluster).
   *   Assegnazione: Da 0 a `num cluster - 1`.
   *   Z-Score (opzionale):
        *   Internal z-score: Vicinanza dell'oggetto al centroide del cluster assegnato.
        *   External z-score: Vicinanza dell'oggetto ai centroidi degli altri cluster.
   *   Sicurezza: Non implementata.
   *   Triflie Pluto: Clustering gerarchico agglomerativo a partire da un clustering partizionale a k-vie.

**4. Funzione Obiettivo**

   *   Basata sulla similarità.
   *   $Ɛ_1$: Versione normalizzata della funzione obiettivo (default).
   *   Obiettivo: Minimizzare la funzione obiettivo considerando compattezza e separabilità dei cluster.

**5. Log di Pluto**

   *   Informazioni dettagliate sul processo di clustering.
   *   Contenuto:
        *   Riepilogo dei parametri.
        *   Dettaglio sulla soluzione di cluster.
        *   Entropia e purezza (se fornito `rclassfile`).
        *   Tempistica.
   *   Document Clustering (con `-showfeatures`):
        *   Feature descrittive: Caratterizzano il cluster.
        *   Feature discriminative: Distinguono il cluster dagli altri.
        *   Ideale: Feature descrittive e discriminative coincidono.

**6. Analisi dei Cluster**

   *   Obiettivo: Analizzare le caratteristiche più significative di ogni cluster.
   *   Metodi:
        *   Caratteristiche più descrittive:
            *   Identificare le **L** caratteristiche che contribuiscono maggiormente alla similarità tra gli oggetti all'interno del cluster.
            *   Calcolare la **percentuale di similarità** per ogni caratteristica rispetto al cluster.

**7. Formato del File di Output**

   *   Struttura: Un file con tante righe quanti sono gli oggetti.
   *   Contenuto: Ogni riga contiene l'ID di classe (ID della riga associato al cluster).

---

Ecco uno schema riassuntivo del testo fornito:

**I. Identificazione Caratteristiche Discriminative**

   *  A. Obiettivo: Individuare le **L** caratteristiche che massimizzano la dissimilarità tra:
        *   1. Oggetti all'interno di un cluster.
        *   2. Oggetti appartenenti ad altri cluster.

**II. Calcolo Percentuale di Dissimilarità**

   *  A. Metodo: Per ogni caratteristica identificata al punto I.A, calcolare:
        *   1. La **percentuale di dissimilarità** rispetto al resto degli oggetti.

---
