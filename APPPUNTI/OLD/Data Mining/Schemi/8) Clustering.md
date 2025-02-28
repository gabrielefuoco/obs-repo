##### Analisi dei Cluster

I. **Definizione e Contesto:**
* Tecnica di apprendimento non supervisionato.
* Raggruppa oggetti simili in cluster coesi e ben separati.
* Differisce dalla classificazione supervisionata (nessuna etichetta predefinita).
* Distinta da divisione in gruppi e partizionamento (spesso riferito a sottografi).
* Può essere pre-processing per KNN.

II. **Approcci al Clustering:**
* **Partitional:** Assegna ogni oggetto a un solo cluster (non sovrapposti).
* **Hierarchical:** Crea una struttura gerarchica di cluster (albero).
* **Partitioning:** Genera e valuta diverse partizioni (es. minimizzando SSE).
* **Density Based:** Identifica cluster come regioni dense.
* **Grid Based:** Valuta la distribuzione degli oggetti in una griglia.
* **Link Based:** Crea cluster basati su collegamenti tra oggetti.

III. **Caratteristiche del Clustering:**
* **Esclusivo vs Non-Esclusivo:** Un oggetto appartiene a uno o più cluster.
* **Fuzzy:** Ogni oggetto appartiene a tutti i cluster con un peso (0-1).
* **Parziale vs Completo:** Applicato a tutto o parte del dataset.
* **Eterogeneo vs Omogeneo:** Cluster di forme, dimensioni e densità diverse.

IV. **Tipi di Cluster:**
* **Well Separated:** Cluster ben identificabili e separati.
* **Center Based:** Cluster definiti dal loro centroide (forma convessa, non necessariamente ben separati).
* **Contiguous Cluster:** Cluster basati sulla vicinanza degli oggetti.
* **Density Based:** Cluster come regioni dense circondate da regioni meno dense.
* **Shared-Property (Conceptual Clusters):** Cluster basati su proprietà condivise.

##### Clustering Partizionale: Ottimizzazione e Metodi

I. **Funzione di Ottimizzazione:**
* Obiettivo: Minimizzare l'errore di clustering.
* Problema: NP-hard (complessità computazionale elevata).
* Funzione obiettivo: Somma dell'Errore Quadratico Medio (SSE)
	$$SSE = \sum_{i=1}^k \sum_{x \in C_i} dist^2(m_i, x)$$
	* Dove: `Ci` = insieme di cluster; `mi` = centroide; `k` = numero di cluster.
* Interpretazione SSE: minore SSE indica migliore clustering.

II. **Valutazione del Clustering:**
* Misura di qualità per clustering partizionale (dati con distanza euclidea): SSE.
$$SSE = \sum_{i=1}^k \sum_{x \in C_i} dist^2(c_i, x)$$
* Limite: non confrontabile tra soluzioni con diverso numero di cluster (SSE diminuisce all'aumentare di k).

III. **Metodi Basati sul Partizionamento:**
* Obiettivo: trovare `k` cluster che minimizzano l'SSE.
* Metodi principali:
	* **K-means:**
		* Utilizza il centroide (media dei punti) come prototipo.
		* Adatto a dati continui in spazio n-dimensionale.
		* Algoritmo iterativo (pseudocodice: 1. Inizializza centroidi; 2. Assegna punti a centroidi più vicini; 3. Ricalcola centroidi; 4. Ripeti fino a convergenza).
		* Complessità: $O(n \times k \times d \times i)$ (n=punti, k=cluster, d=attributi, i=iterazioni).
		* Misura di vicinanza: qualsiasi misura di distanza.
	* **K-medoid:**
		* Variante di K-means per dati non numerici.
		* Centroide determinato dall'elemento più frequente.

IV. **Limitazioni di K-means:**
* Non gestisce bene cluster:
	* Non convessi.
	* Con densità diverse.
	* Con grandezze diverse (soluzione: aumentare k).

V. **Importanza della Scelta del Punto Iniziale (K-means):**
* Risultato dipendente dalla posizione iniziale dei centroidi.
* Alta probabilità di risultati subottimali, soprattutto con molti cluster.

##### Varianti del K-means

I. **Selezione dei *k* punti iniziali:** Influenza significativamente il risultato.

A. Metodi:
	1. Scelta random all'interno dello spazio: Risultati potenzialmente non ottimali.
	2. Scelta random di un esempio: Utilizza un oggetto come centroide.
	3. Centroidi molto dissimili ("further centre"): Massimizza la distanza tra centroidi.
	4. Scelta multipla dei centroidi: Esegue più iterazioni con diverse inizializzazioni.
	5. Utilizzo di altre tecniche di clustering: Come pre-processing per l'inizializzazione.

II. **Euristica dei centri più lontani ("further centre"):**

A. **Passaggi:**
	1. Selezione casuale di µ₁.
	2. Iterazione (i = 2 a k): Selezione di µᵢ come il punto più distante da tutti i centroidi precedenti.

B. **Definizione matematica:** µᵢ = arg maxₓ min<sub>µⱼ:1<j<i</sub> d(x, µⱼ)

C. **Svantaggi:** Sensibilità agli outliers.

D. **Riepilogo:** Centroidi iniziali ben separati, ma computazionalmente costoso. Spesso applicato a un campione dei dati.

III. **Altre soluzioni per la scelta dei punti iniziali:**

A. Eseguire il clustering più volte con diverse inizializzazioni.
B. Campionamento dei dati e clustering gerarchico.
C. Selezione di un k maggiore e scelta dei centroidi più distanti.
D. Generazione di molti cluster e successivo clustering gerarchico.
E. Utilizzo di Bisecting K-means.

IV. **K-means++:** Garanzia di una soluzione ottimale entro un fattore di O(log(k)) con SSE inferiore.

A. **Passaggi:**
	1. Selezione casuale di un centroide.
	2. Calcolo della distanza D(x) di ogni punto x dal centroide più vicino.
	3. Selezione di un nuovo centroide con probabilità proporzionale a D(x)².
	4. Ripetizione dei passi 2 e 3 fino a k cluster.
	5. Esecuzione del K-means standard.

V. **Calcolo della similarità dei punti e Strategie per calcolare i cluster:**

##### Gestione del Clustering K-means

I. **Gestione Cluster Vuoti**
* Approcci per sostituire centroidi in cluster vuoti:
	* Punto più lontano da qualsiasi centroide esistente.
	* Centroide casuale dal cluster con SSE più alta (ripetibile per più cluster vuoti).

II. **Aggiornamento Incrementale dei Centroidi**
* **Vantaggi:**
	* Previene cluster vuoti.
	* Permette di regolare il peso dei punti durante il processo.
* **Svantaggi:**
	* Dipendenza dall'ordine di elaborazione dei punti.
	* Maggior costo computazionale.

III. **Pre e Post-processing**
* **Pre-processing:**
	* Eliminazione degli outlier.
	* Normalizzazione dei dati.
* **Post-processing:**
	* Eliminazione di cluster di piccole dimensioni.
	* Divisione di cluster con SSE elevato.
	* Unione di cluster vicini con SSE basso.

IV. **Bisecting K-means**
* Algoritmo iterativo che divide ripetutamente i cluster in due.
* **Passaggi:**
	1. Calcola il centroide *w*.
	2. Seleziona un punto casuale *cL*.
	3. Seleziona un punto *cR* simmetrico a *cL* rispetto a *w*.
	4. Suddivide i punti in *R* (più vicini a *cR*) e *L* (più vicini a *cL*).
	5. Ripeti per *R* e *L*.
* Il cluster da dividere è scelto in base a dimensioni, SSE o entrambi.
* Può produrre clustering gerarchico o partizionale.

V. **K-modes**
* Variante del k-means per dati categorici.
* Usa la moda invece della media.
* Richiede nuove misure di distanza per dati categorici.

VI. **Clustering Gerarchico**
* Genera una gerarchia di cluster.
* **Tipi:**
	* Agglomerativo (bottom-up): inizia con punti singoli e unisce i più vicini.
	* Divisivo (top-down): inizia con un cluster unico e lo divide.
* **Visualizzazione:**
	* Dendrogramma.
	* Diagramma di cluster nidificato (per dati bidimensionali).
* **Vantaggi:**
	* Non richiede il numero di cluster iniziale.
	* Si possono ottenere quanti cluster si vogliono.
* **Note:**
	* Usa una matrice di similarità/dissimilarità.
	* *k* non è definito a priori.

##### Algoritmo Gerarchico Agglomerativo

I. **Algoritmo Base:**

* Inizia con ogni punto come cluster separato.
* Iterativamente:
	* Unisci i due cluster più vicini.
	* Aggiorna la matrice di prossimità.
* Termina quando rimane un solo cluster.
* Richiede aggiornamento continuo della matrice di prossimità dopo ogni fusione.

II. **Definizione della Prossimità tra Cluster:**

A. **Approcci basati su grafici:**

* **MIN (Single Link):** Distanza tra i due punti più vicini in cluster diversi. Sensibile a rumore e outliers. 
* **MAX (Complete Link):** Distanza tra i due punti più lontani in cluster diversi. Meno sensibile a rumore, ma può dividere cluster grandi. 
* **Media di Gruppo (Group Average):** Media delle distanze tra tutte le coppie di punti in cluster diversi. Meno sensibile a rumore, tende a cluster globulari.
$$proximity(C_i, C_j) = \frac{{\sum_{p_i \in C_i, p_j \in C_j} proximity(p_i, p_j) }}{|C_i| \times |C_j|} $$

B. **Approcci basati su prototipi:**

* **Vicinanza tra Centroidi:** Distanza tra i centroidi dei cluster. 
* **Metodo di Ward(K-means gerarchico.):** Aumento della SSE (Sum of Squared Errors) risultante dalla fusione di due cluster. Crea cluster globulari, poco sensibile al rumore.
	* Formula: $$∆(A, B) = \sum_{x \in A \cup B} ||x - m_{A \cup B}||^2 - \sum_{x \in A} ||x - m_A||^2 - \sum_{x \in B} ||x - m_B||^2$$

- **Tecniche di Clustering**

- I. **Clustering Gerarchico**
 - A. **Agglomerativo:**
 - Principio: Unione iterativa dei cluster più simili fino a un unico cluster.
 - Limitazioni: Criteri locali, fusioni irreversibili, sensibilità a rumore e outlier, difficoltà con cluster non globulari o grandi.
 - B. **Divisivo:**
 - Principio: Divisione iterativa di un unico cluster fino a cluster singoli.
 - Metodo: Costruzione e divisione di un albero ricoprente minimo.
 - Efficienza: Complessità O(n²) o O(m log(n)), dove m è il numero di archi.

- II. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
 - A. **Principio:** Identificazione di regioni ad alta densità separate da regioni a bassa densità.
 - B. **Classificazione dei punti:**
 - Core point: ≥ MinPts entro distanza Eps.
 - Border point: Vicino a un core point, ma non core point.
 - Noise point: Né core né border point.
 - C. **Connettività:**
 - Directly density-reachable: Punto p direttamente raggiungibile da core point q se p è nelle vicinanze di q.
 - Density reachable: Punto p raggiungibile da core point q tramite catena di punti.
 - Density connected: Due punti raggiungibili dallo stesso punto.
 - D. **Algoritmo:**
 - Etichettare i punti (core, border, noise).
 - Eliminare i noise point.
 - Collegare i core point vicini (< Eps).
 - Creare cluster da gruppi di core point connessi.
 - Assegnare i border point ai cluster dei loro core point associati.
 - E. **Complessità:**
 - Temporale: O(m × tempo per trovare vicini entro Eps), nel peggiore dei casi O(m²), migliorabile a O(m log m) con strutture dati come KD-tree.
 - Spaziale: O(m).
 - F. **Vantaggi:**
 - Resistente al rumore.
 - Gestisce cluster di forme e dimensioni arbitrarie.
 - G. **Svantaggi:**
 - Problemi con cluster a densità diverse.
 - Problemi con dati ad alta dimensione.
 - Costo computazionale elevato.
 - H. **Selezione dei parametri:**
 - Eps (lunghezza minima del raggio).
 - MinPts (numero minimo di oggetti).
 - Analisi della curva dei k-nearest neighbors per determinare Eps.

##### Valutazione dei Cluster

I. **Indici di Valutazione:**
* **Indici Esterni:** Confronto con un risultato ideale (etichette di classe).
* **Indici Interni:** Valutazione basata su caratteristiche interne (es. SSE per K-means).
* **Indici Relativi:** Confronto tra due risultati di clustering.

II. **Misura della Validità tramite Correlazione:**
* Utilizza una matrice di prossimità (distanze tra punti) e una matrice ideale di similarità (1 se stesso cluster, 0 altrimenti).
* Alta correlazione indica buona qualità del clustering.
* **Limitazione:** Non adatta per cluster basati su contiguità o densità.

III. **Valutazione tramite Grafici di Correlazione:**
* **Interpretazione:**
	* Cluster ben definiti: Blocchi distinti sulla diagonale.
	* Cluster poco definiti: Struttura meno definita, blocchi poco distinti o assenti.
* **Utilizzo:**
	* Valutare la qualità del clustering.
	* Confrontare algoritmi di clustering.
	* Ottimizzare parametri dell'algoritmo.

IV. **Misure Interne: SSE (Somma degli Errori Quadratici)**
* **Interpretazione della Curva SSE:**
	* Errore elevato con pochi cluster.
	* Diminuzione dell'errore con più cluster.
	* Punti di flessione significativi indicano potenziali valori ottimali di *k* (numero di cluster).
* **Utilizzo:** I punti di flessione nella curva SSE suggeriscono il numero ottimale di cluster.

##### Valutazione della Clusterizzazione

* **Coesione e Separazione:**
 * **Coesione:** Misura la similarità degli oggetti all'interno di uno stesso cluster. Alta coesione indica oggetti vicini.
 * **Separazione:** Misura la dissimilarità degli oggetti tra cluster diversi. Alta separazione indica cluster ben distinti.
 * **Metodi di misurazione:**
 * **Somma delle distanze:**
 * Coesione: $\sum_{x \in c_i, y \in c_i} proximity(x, y)$
 * Separazione: $\sum_{x \in c_i, y \in c_j} proximity(x, y)$
 * **Distanza dal centroide:**
 * Coesione: $\sum_{x \in c_i} proximity(x, m_i)$
 * Separazione: $proximity(m_i, m_j)$
 * **SSE e BSS:**
 * Coesione (SSE/WSS): $\sum_{i} \sum_{x \in C_i} (x - m_i)^2$
 * Separazione (BSS): $\sum_{i} |C_i|(m - m_i)^2$
 * **Obiettivo:** Bassa coesione e alta separazione indicano una buona clusterizzazione.

* **Silhouette Coefficient:**
 * **Scopo:** Valuta la qualità della clusterizzazione per ogni punto.
 * **Calcolo:** $s = \frac{(b - a)}{max(a, b)}$ dove:
 * *a*: Distanza media tra un punto e gli oggetti del suo stesso cluster.
 * *b*: Distanza media tra un punto e gli oggetti di altri cluster.
 * **Interpretazione:**
 * s ≈ 1: Buon clustering.
 * s ≈ 0: Punto vicino al confine tra cluster.
 * s ≈ -1: Probabile errata classificazione.
 * **Valore alto (≈1):** Indica buona separazione tra cluster.
 * **Valore basso (≈0 o negativo):** Indica scarsa separazione o errata classificazione.

##### Tecniche di Clustering

I. **Fuzzy Clustering**
 A. **Concetto:** Consente l'appartenenza parziale di un punto a più cluster, rappresentata da pesi di appartenenza. La somma dei pesi per ogni punto è 1.
 B. **SSE (Somma dei Quadrati degli Errori):** $$SSE = \sum_{j=1}^k \sum_{i=1}^m w_{ij} \cdot dist(x_i, c_j)²$$
 C. **K-Means Fuzzy:** Variante del K-Means che minimizza una SSE modificata: $$SSE = \sum_{j=1}^k \sum_{i=1}^m w_{ij}^p \cdot dist(x_i, c_j)^2$$dove *p* > 1 controlla l'influenza dei pesi.
 D. **Algoritmo K-Means Fuzzy:**
 1. Inizializzazione casuale dei pesi *w<sub>ij</sub>*.
 2. Aggiornamento centroidi: $$c_{ij} = \frac{{\sum_{i=1}^m w_{ij} \cdot x_i}}{\sum_{i=1}^m w_{ij}} $$
 3. Aggiornamento pesi: $$w_{ij} =\frac{\left( \frac{1}{dist(x_{i}c_{j})}^2 \right)^{\frac{1}{p-1}}}{ \sum_{j=1}^k\left( \frac{1}{dist(x_{i},c_{j})}^2 \right)^{\frac{1}{p-1}}}$$
 4. Iterazione dei passi 2 e 3 fino alla convergenza.

II. **Clustering Grid-Based**
 A. **Concetto:** Divide lo spazio dei dati in celle e considera cluster le celle con un numero di punti superiore a una soglia.
 B. **Algoritmo:**
 1. Suddivisione dello spazio in celle.
 2. Conteggio punti per cella.
 3. Identificazione celle con punti > soglia.
 4. Raggruppamento celle adiacenti con punti > soglia.
 C. **Vantaggi:** Efficiente per grandi dataset, semplice implementazione.
 D. **Svantaggi:** Sensibile alla dimensione delle celle, non adatto a densità variabili.

III. **Clusterizzazione in Sottospazi**
 A. **Concetto:** La clusterizzazione può essere inefficiente con alta dimensionalità. Ridurre la dimensionalità migliora l'identificazione dei cluster. L'esempio mostra come diversi sottoinsiemi di attributi possono rivelare cluster differenti.
 B. **Problema:** L'aumento degli attributi aumenta la distanza tra gli oggetti, rendendo difficile l'identificazione dei cluster.

##### Clusterizzazione basata su Grafi

I. **Tecniche principali:**

 A. **Clustering Gerarchico Divisivo:**
 1. Costruzione di un grafo completo non orientato.
 2. Costruzione dell'albero ricoprente minimo.
 3. Separazione iterativa dell'albero rompendo l'arco con distanza maggiore. Si ripete fino a cluster singoli.
 4. Complessità: $O(n^3)$ (algoritmo $O(n^2)$ per albero ricoprente); $O(n^2\log n)$ (con struttura ad heap).

 B. **Shared Nearest Neighbour (SNN):**
 1. Assegnazione di pesi agli archi basata sul numero di vicini in comune.
 2. Algoritmo di Jarvis-Patrick:
 a. Costruzione di un grafo basato su una soglia di distanza.
 b. Assegnazione dei pesi agli archi.
 c. Eliminazione di archi con peso inferiore a una soglia.
 d. Identificazione dei cluster come componenti connesse.
 3. Vantaggi: Funziona bene con cluster di diverse dimensioni e densità.
 4. Svantaggi: Inefficace con cluster interconnessi.

 C. **SNN Density Based Clustering:**
 1. Combinazione di SNN e DBSCAN.
 2. Fasi:
 a. Calcolo della matrice di similarità.
 b. Costruzione di un grafo sparso.
 c. Assegnazione dei pesi agli archi (SNN).
 d. Applicazione di DBSCAN per identificazione dei cluster basata sulla densità.
 3. Vantaggi: Più efficace dell'algoritmo SNN tradizionale.

