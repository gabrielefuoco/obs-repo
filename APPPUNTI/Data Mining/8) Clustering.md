
L'analisi dei cluster è una tecnica di apprendimento non supervisionato che raggruppa oggetti dati in base alla loro similarità. L'obiettivo è creare gruppi **coesi**, ovvero con oggetti simili tra loro, e **ben separati**, ovvero con oggetti diversi da quelli di altri gruppi.

La clusterizzazione può essere vista come una forma di classificazione, ma a differenza della classificazione supervisionata, non utilizza etichette di classe predefinite. Le etichette dei cluster vengono determinate direttamente dai dati.

Sebbene la divisione in gruppi possa sembrare simile alla clusterizzazione, non sempre è così. Inoltre, termini come **segmentazione** e **partizionamento**, a volte usati come sinonimi di clustering, spesso si riferiscono ad approcci diversi. Il termine partizionamento è spesso associato a tecniche che dividono i grafici in sottografi, non strettamente correlate al clustering.

La clusterizzazione può essere utilizzata anche come pre-processing per migliorare l'attività di classificazione del *KNN*. 
## Approcci al Clustering

Esistono diversi approcci al clustering, ognuno con le sue caratteristiche e applicazioni:

* **Partitional:** Questo approccio divide i dati in insiemi non sovrapposti, assegnando ogni oggetto a un solo cluster.
	* ![[8) Clustering-20241009095438105.png|249]]
* **Hierarchical:** Questo approccio crea una struttura gerarchica di cluster, dove i cluster possono avere sotto-cluster. La struttura è rappresentata da un albero, con la radice che rappresenta l'intero dataset e le foglie che rappresentano cluster individuali.
* **Partitioning:** Questo approccio genera diverse partizioni dei dati e le valuta in base a un criterio, come la somma degli errori al quadrato (SSE).
* **Density Based:** Questo approccio identifica i cluster come regioni dense di oggetti, circondate da regioni meno dense.
* **Grid Based:** Questo approccio, simile a quello basato sulla densità, valuta la distribuzione degli oggetti nello spazio.
* **Link Based:** Questo approccio crea cluster basati sui collegamenti tra gli oggetti, come ad esempio le connessioni in una rete.

### Caratteristiche del Clustering

Il clustering può essere caratterizzato da diverse proprietà:

* **Exclusive vs Non-Exclusive:** In un clustering esclusivo, ogni oggetto appartiene a un solo cluster. In un clustering non esclusivo, un oggetto può appartenere a più cluster.
* **Fuzzy:** In un clustering fuzzy, ogni oggetto appartiene a tutti i cluster con un peso diverso, che varia da 0 a 1.
* **Parziale vs Completo:** In alcuni casi, il clustering viene applicato solo a un sottoinsieme dei dati.
* **Eterogeneo vs Omogeneo:** I cluster possono avere forme, dimensioni e densità diverse.

### Tipi di Cluster

Esistono diversi tipi di cluster, ognuno con una definizione specifica:

* **Well Separated:** I cluster sono ben identificabili e separati tra loro, con oggetti coesi all'interno e ben distinti all'esterno.
* **Center Based:** I cluster sono identificati dal loro centroide, che è la media dei valori degli oggetti che appartengono al cluster. I cluster hanno una forma convessa e possono non essere ben separati tra loro.
	* ![[8) Clustering-20241009095503490.png|97]]
* **Contiguous Cluster:** I cluster sono definiti in base alla distanza tra gli oggetti. Oggetti vicini tra loro, anche se appartenenti a regioni diverse, possono essere considerati parte dello stesso cluster.
	* ![[8) Clustering-20241009095511754.png|170]]
* **Density Based:** I cluster sono regioni dense di oggetti, circondate da regioni meno dense.
	* ![[8) Clustering-20241009095522163.png|181]]
* **Shared-Property (Conceptual Clusters):** I cluster sono identificati in base a proprietà condivise dagli oggetti.

## Funzione di Ottimizzazione

L'obiettivo del clustering è trovare una soluzione che ottimizzi una funzione, ovvero minimizzi l'errore. Questo problema è complesso (*NP-hard*) perché richiede di valutare tutte le possibili combinazioni di valori.

La funzione di ottimizzazione utilizzata è la **somma dell'errore quadratico medio (SSE)**:

$$SSE = \sum_{i=1}^k \sum_{x \in C_i} dist^2(m_i, x) $$

Dove:

* *Ci*: insieme di cluster
* *mi*: centroide
* *k*: numero di cluster

L'SSE misura la distanza tra ogni punto e il centroide del suo cluster, sommando gli errori quadratici di tutti i punti. Un SSE minore indica una migliore clusterizzazione.

### Valutazione del Clustering Partizionale

Il clustering partizionale divide un dataset in *k* cluster. Per dati con distanza euclidea, la qualità del clustering è misurata dall'SSE.

$$SSE = \sum_{i=1}^k \sum_{x \in C_i} dist^2(c_i, x) $$

L'SSE calcola la distanza euclidea di ogni punto dal centroide più vicino e somma gli errori quadratici. Un SSE minore indica una migliore rappresentazione dei punti nei cluster.

È importante notare che non è possibile confrontare due soluzioni con un numero diverso di cluster, poiché l'SSE diminuisce all'aumentare del numero di cluster.

# Metodi Basati sul Partizionamento

L'obiettivo è trovare *k* cluster che minimizzano l'SSE, esaminando tutte le possibili partizioni.

Due metodi importanti sono **K-means** e **K-medoid**:

* **K-means**: utilizza il **centroide**, la media dei punti di un cluster, come prototipo. È adatto per dati continui in uno spazio *n-dimensionale*.
* **K-medoid**: è una variante di K-means che può essere applicata a dati non numerici.

## K-means

L'algoritmo K-means è un metodo iterativo che cerca di minimizzare l'SSE. 

##### Pseudocodice
1. Seleziona K punti come centroidi iniziali.
**Ripeti:**
2. Forma K cluster assegnando ogni punto al suo centroide più vicino.
3. Ricalcola il centroide di ogni cluster.
**Fino a quando:** 
4. I centroidi non cambiano. 

###  Dettagli K-means

L'algoritmo K-means converge in poche iterazioni, con una complessità di:

$O(n \times k \times d \times i)$

dove:

* *n*: numero di punti
* *d*: numero di attributi
* *k*: numero di cluster
* *i*: numero di iterazioni

L'algoritmo è efficiente, ma la sua efficienza può essere influenzata da un numero elevato di punti.

La misura di vicinanza tra gli oggetti può essere qualsiasi misura di distanza. Nel K-medoid, simile al K-means, il centroide è determinato dall'elemento più frequente.

### Limitazioni e Debolezze del K-means

* **Forma convessa:** K-means non riesce a identificare cluster con forme non convesse.
* **Densità diversa:** K-means non riesce a identificare cluster con densità diverse.
* **Grandezza diversa:** K-means non riesce a identificare cluster con grandezze diverse.

Una soluzione potrebbe essere l'utilizzo di un numero di cluster più elevato.

### Importanza nella Scelta del Punto Iniziale

Il risultato del K-means dipende dalla posizione iniziale dei centroidi.

La probabilità di selezionare un centroide da ogni cluster reale è bassa, soprattutto con un numero elevato di cluster.

La scelta dei centroidi iniziali può influenzare significativamente il risultato finale.

## Varianti del K-means

Esistono diverse varianti del K-means che differiscono per:

* Selezione dei *k* punti iniziali
* Calcolo della similarità dei punti
* Strategie per calcolare i cluster

####  Selezione dei *k* Punti Iniziali

La scelta dei centroidi iniziali può influenzare il risultato del K-means. Esistono diverse euristiche per migliorare la scelta dei centroidi iniziali:

* **Scelta random dei centroidi all'interno dello spazio:** può portare a risultati non ottimali.
* **Scelta random di un esempio dal dominio:** utilizza un oggetto come rappresentante invece della media.
* **Scegliere centroidi molto dissimili tra loro (further centre).**
* **Scelta multipla dei centroidi.**
* **Scelta dei centroidi iniziali mediante altre tecniche di clusterizzazione.** 

## Euristica dei centri più lontani (furthest centre)

Questa euristica seleziona i centroidi iniziali per K-means in modo da massimizzare la distanza tra loro.
![[8) Clustering-20241009100711064.png|311]]
**Passaggi:**
1. Si sceglie un punto *µ*1 a caso.
2. Per *i* da 2 a *k*, si sceglie il punto *µi* più distante da qualsiasi centroide precedente.

**Definizione matematica:**
$$µ_{i} = \text{arg max }_x \min_{\mu_{j}:1<j<i} \ d(x,_{\mu_{j}})$$
dove:
• arg max
x : indichiamo il punto che ha la maggior distanza dal centro precedente
•
min
µj:1<j<i d(x, µj): la distanza minima da x a qualsiasi centro precedente

**Svantaggi:**
* Sensibile agli outliers.
![[8) Clustering-20241009100723154.png|356]]
**Riepilogo:**
* Il primo punto è scelto a caso o come centroide di tutti i punti.
* I centroidi successivi sono scelti come i punti più lontani da quelli già selezionati.

Questo metodo garantisce centroidi iniziali ben separati, ma è costoso calcolare il punto più lontano. Per ovviare a questo problema, l'approccio viene spesso applicato a un campione dei punti.

## Altre soluzioni per la scelta dei punti iniziali

* Eseguire il clustering più volte.
* Effettuare un campionamento dei dati e poi eseguire un clustering gerarchico.
* Selezionare un numero *k* maggiore e poi selezionare i centroidi più distanti.
* Generare un numero elevato di cluster e poi effettuare un clustering gerarchico.
* Usare la variante Bisecting K-means.

## K-means ++

K-means++ è un approccio per l'inizializzazione di K-means che garantisce una soluzione ottimale all'interno di un fattore di *O*(*log*(*k*)), con un SSE inferiore.

**Passaggi:**

1. Selezionare un centroide a caso.
2. Calcolare la distanza *D*(*x*) di ogni punto *x* dal centroide più vicino.
3. Scegliere un nuovo centroide con una probabilità proporzionale a *D*(*x*)2.
4. Ripetere i passi 2 e 3 fino a *k* cluster.
5. Eseguire il clustering con l'algoritmo k-means standard.

## Gestire cluster vuoti

Se durante l'assegnazione dei punti si ottengono cluster vuoti, è necessario scegliere un centroide sostitutivo.

**Approcci:**

* **Punto più lontano:** Scegliere il punto più lontano da qualsiasi centroide attuale.
* **Cluster con SSE più alta:** Scegliere un centroide a caso dal cluster con l'SSE più alta.

Se ci sono diversi cluster vuoti, questo processo può essere ripetuto più volte.

## Aggiornare i centroidi in maniera incrementale

Invece di aggiornare i centroidi dopo che tutti i punti sono stati assegnati, è possibile aggiornarli in modo incrementale dopo ogni assegnazione.

**Vantaggi:**

* **Previene cluster vuoti:** Tutti i cluster iniziano con un punto e non possono rimanere vuoti.
* **Peso relativo dei punti:** Il peso dei punti può essere regolato durante il processo di clustering.

**Svantaggi:**

* **Dipendenza dall'ordine:** L'ordine di elaborazione dei punti influenza il risultato.
* **Costo computazionale:** Gli aggiornamenti incrementali sono leggermente più costosi.

## Pre e post processing

**Pre processing:**

* Eliminazione degli outlier.
* Normalizzazione dei dati.

**Post processing:**

* Eliminazione di cluster di piccole dimensioni.
* Divisione di cluster con SSE elevato.
* Unione di cluster vicini con SSE basso.

## 17.8.3 Bisecting K-means

L'algoritmo Bisecting K-means divide l'insieme di punti in due cluster, seleziona uno di questi cluster da dividere e ripete il processo fino a ottenere *k* cluster.

**Passaggi:**

1. Calcolare il centroide *w*.
2. Selezionare un punto casuale *cL*.
3. Selezionare un punto *cR* simmetrico a *cL* rispetto a *w*.
4. Suddividere i punti in due sotto-insiemi: *R* (più vicini a *cR*) e *L* (più vicini a *cL*).
5. Reiterare la procedura per *R* e *L*.

**Note:**

* Il cluster da dividere può essere scelto in base alle dimensioni, all'SSE o a entrambi.
* L'algoritmo può produrre un clustering gerarchico o partizionale.

## K-modes

Variante del k-means per dati categorici. Utilizza la **moda** invece della media e richiede nuove misure di distanza per dati categorici.

## Clustering Gerarchico

Le tecniche di clustering gerarchico generano una gerarchia di cluster.

* **Agglomerative (bottom up):** Inizia con punti singoli e unisce i cluster più vicini.
* **Divisive (top down):** Inizia con un cluster unico e divide i cluster fino a ottenere punti singoli.

**Visualizzazione:**

* **Dendrogramma:** Visualizza le relazioni cluster-subcluster e l'ordine di unione/divisione.
* **Diagramma di cluster nidificato:** Per insiemi di punti bidimensionali.



**Vantaggi:**

* Non richiede di specificare il numero di cluster iniziale.
* Si possono ottenere quanti cluster si vogliono.

**Note:**

* Gli algoritmi gerarchici utilizzano una matrice di similarità/dissimilarità.
* Il valore di *k* non è definito a priori.

## Algoritmo gerarchico di base

Molte tecniche di clustering gerarchico agglomerativo sono variazioni del seguente approccio: a partire dai singoli punti come cluster, si uniscono successivamente i due cluster più vicini fino a quando rimane un solo cluster. Questo approccio è espresso in modo più formale dal seguente algoritmo:

**Passaggi:**
1. **Calcola la matrice di prossimità:** Misura la distanza tra tutti i punti.
2. **Ripeti:**
    - **Unisci i cluster più vicini:** Basati sulla matrice di prossimità.
    - **Aggiorna la matrice di prossimità:** Riflette la nuova prossimità tra il cluster appena formato e gli altri.
3. **Fino a quando:** Rimane un solo cluster

Dobbiamo aggiornare la matrice di prossimità poiché se uniamo due cluster, ci sarà un cluster in meno e di conseguenza una riga e una colonna in meno della matrice. L'operazione chiave è il calcolo della prossimità tra due cluster. A seconda della metodologia utilizzata per il calcolo della prossimità, si definiscono diversi approcci. 

## Definizioni di distanze inter-cluster

La vicinanza tra cluster è definita in base al tipo di cluster considerato.

**Approcci basati su grafici:**

* **MIN (single link):** La vicinanza è data dalla distanza tra i due punti più vicini in cluster diversi. Produce cluster basati sulla contiguità, ma è sensibile al rumore e agli outliers.
    ![[8) Clustering-20241006115321816.png|320]]
* **MAX (complete link):** La vicinanza è data dalla distanza tra i due punti più lontani in cluster diversi. Meno sensibile al rumore, ma può dividere grandi cluster e favorisce forme globulari.
    ![[8) Clustering-20241006115337067.png|323]]
* **Media di gruppo (group average):** La vicinanza è data dalla media delle distanze tra tutte le coppie di punti in cluster diversi. Meno sensibile al rumore, ma tende a creare cluster globulari.
    ![[8) Clustering-20241006115347420.png|319]]
    $$proximity(C_i, C_j) = \frac{{\sum_{p_i \in C_i, p_j \in C_j} proximity(p_i, p_j) }}{|C_i| \times |C_j|} $$

**Approcci basati su prototipi:**

* **Vicinanza tra centroidi:** La vicinanza è data dalla distanza tra i centroidi dei cluster.
    ![[8) Clustering-20241006115356644.png|312]]
* **Metodo di Ward:** Misura la vicinanza in termini di aumento della SSE che deriva dalla fusione dei due cluster. Tende a creare cluster globulari e è poco sensibile al rumore.
    $$∆(A, B) = \sum_{x \in A \cup B} ||x - m_{A \cup B}||^2 - \sum_{x \in A} ||x - m_A||^2 - \sum_{x \in B} ||x - m_B||^2$$

_m_ rappresenta il centroide del cluster.

**Note:**

* MIN e MAX sono chiamati anche single link e complete link.
* Il metodo di Ward può essere visto come un K-means gerarchico.

## Tecniche di Clustering

### Clustering Gerarchico Agglomerativo

* **Principio:** Unisce iterativamente i cluster più simili fino a formare un unico cluster.
* **Limitazioni:**
    * Non ottimizza una funzione obiettivo globale, ma si basa su criteri locali.
    * Le fusioni sono irreversibili.
    * Sensibile a rumore e outlier, difficoltà con cluster non globulari o di grandi dimensioni.

### Clustering Gerarchico Divisivo

* **Principio:** Inizia con un unico cluster e lo divide iterativamente fino a quando ogni punto è in un cluster separato.
* **Metodo:** Costruisce un albero ricoprente minimo del grafo dei punti e lo divide eliminando l'arco massimo.
* **Efficienza:** L'algoritmo ha complessità *O*(*n*2) o *O*(*m* log(*n*)), dove *m* è il numero di archi.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

* **Principio:** Identifica regioni ad alta densità separate da regioni a bassa densità.
* **Classificazione dei punti:**
    * **Core point:** Punto con almeno *MinPts* punti entro una distanza *Eps*.
    * **Border point:** Non è un core point, ma è nelle vicinanze di un core point.
    * **Noise point:** Non è né un core point né un border point.
    * ![[8) Clustering-20241009101935275.png|391]]
* **Suddivisione dei punti per densità:**
    * **Directly density-reachable:** Un punto *p* è direttamente raggiungibile da un core point *q* se *p* è nelle vicinanze di *q*.
    * **Density reachable:** Un punto *p* è raggiungibile da un core point *q* se esiste una catena di punti che lo collegano a *q*.
    * **Density connected:** Due punti sono density-connected se entrambi sono raggiungibili da uno stesso punto.
* **Algoritmo:** Unisce i core point vicini e i border point associati, scartando i noise point.

##  L'algoritmo DBSCAN

Date le precedenti definizioni di punti core, punti border e punti noise, l'algoritmo DBSCAN può essere descritto in modo informale come segue:

1. **Etichettare i punti:** Classificare tutti i punti come core, border o noise.
2. **Eliminare i punti noise:** Rimuovere i punti noise dall'insieme di dati.
3. **Creare connessioni tra i core point:** Inserire un collegamento tra tutti i core point che si trovano a una distanza inferiore a *Eps* l'uno dall'altro.
4. **Creare i cluster:** Ogni gruppo di core point connessi forma un cluster separato.
5. **Assegnare i border point:** Assegnare ogni border point al cluster del suo core point associato.

In sintesi, DBSCAN identifica i cluster come gruppi di core point connessi, includendo i border point associati a questi core point. I punti noise vengono scartati. 

### Complessità Temporale e Spaziale

* La complessità temporale di DBSCAN è *O*(*m×* tempo per trovare punti nelle vicinanze di Eps), dove *m* è il numero di punti.
* Nel peggiore dei casi, la complessità è *O*(m^2).
* Strutture dati come i KD-tree possono ridurre la complessità a *O*(*mlogm*) nel caso medio.
* Il requisito di spazio è *O*(*m*), poiché è necessario conservare solo una piccola quantità di dati per ogni punto.

### Vantaggi e Svantaggi

**Vantaggi:**

* **Resistente al rumore:** DBSCAN è relativamente resistente al rumore grazie alla sua definizione di cluster basata sulla densità.
* **Gestione di cluster di forme e dimensioni arbitrarie:** Può trovare cluster di forme e dimensioni diverse, a differenza di algoritmi come K-means.

**Svantaggi:**

* **Problemi con cluster a densità diverse:** Ha difficoltà con cluster che hanno densità molto diverse.
* **Problemi con dati ad alta dimensione:** La densità è più difficile da definire per dati ad alta dimensione, rendendo DBSCAN meno efficace.
* **Costo computazionale:** Il calcolo dei nearest neighbor può essere costoso, soprattutto per dati ad alta dimensione.

### Selezione dei Parametri

DBSCAN richiede la definizione di due parametri:

* **Lunghezza minima del raggio (Eps):** Determina la distanza massima per considerare i punti come vicini.
* **Numero minimo di oggetti (MinPts):** Definisce il numero minimo di punti necessari per formare un core point.

Un metodo per aiutare a fissare il parametro *Eps* è l'analisi della curva dei *k*-nearest neighbors. Si ordina i punti in base alla distanza dal loro *k*-esimo nearest neighbor e si osserva il "ginocchio" della curva, che può indicare un valore appropriato per *Eps*.

## Valutazione dei Cluster

### Indici di Valutazione

Per valutare la qualità di un clustering, si utilizzano diversi tipi di indici:

* **Indici esterni:** Confrontano il risultato con un risultato ideale, come le etichette di classe in un processo supervisionato.
* **Indici interni:** Valutano la qualità del clustering in base a caratteristiche interne, come la somma dell'errore quadratico (SSE) nel caso di K-means.
* **Indici relativi:** Confrontano due risultati di clustering.

### Misura della Validità tramite la Correlazione
Si possono utilizzare due matrici per valutare la validità del clustering:

* **Matrice di prossimità:** Contiene le distanze tra tutti i punti.
* **Matrice ideale di similarità:** Indica se due punti appartengono allo stesso cluster (1) o a cluster diversi (0).

La correlazione tra queste due matrici può essere utilizzata per valutare la qualità del clustering. Una correlazione elevata indica un buon risultato.

**Limitazioni:** La correlazione non è una buona misura per i cluster basati su contiguità o densità. 

## Valutare il Clustering tramite Grafici di Correlazione

La visualizzazione dei dati tramite grafici di correlazione può essere un metodo efficace per valutare la qualità del clustering. 
![[8) Clustering-20241009102033827.png|254]]
![[8) Clustering-20241009102042362.png|282]]
**Interpretazione dei Grafici:**

* **Cluster ben definiti:** Se i dati sono ben raggruppati in cluster, il grafico di correlazione mostrerà blocchi distinti sulla diagonale, come nel primo esempio. Questi blocchi rappresentano i cluster identificati dall'algoritmo.
* **Cluster poco definiti:** Se il clustering non è di alta qualità, il grafico di correlazione mostrerà una struttura meno definita, con blocchi meno distinti o assenti. Questo indica che l'algoritmo non ha catturato correttamente la struttura dei dati, come negli esempi di K-means e complete-link.

**Utilizzo dei Grafici:**

I grafici di correlazione possono essere utilizzati per:

* **Valutare la qualità del clustering:** Identificare se l'algoritmo ha catturato correttamente la struttura dei dati.
* **Confrontare diversi algoritmi di clustering:** Determinare quale algoritmo produce i cluster più definiti.
* **Ottimizzare i parametri dell'algoritmo:** Adattare i parametri dell'algoritmo per ottenere un clustering migliore.

![[8) Clustering-20241009102107326.png|443]]
![[8) Clustering-20241009102116704.png|447]]
![[8) Clustering-20241009102125172.png|444]]

## Misure Interne: SSE

Un'altra possibilità per valutare la qualità del clustering è quella di misurare la **Somma degli Errori Quadratici (SSE)**.

**Esempio:**

Consideriamo la seguente distribuzione di dati:

![[8) Clustering-20241009102138687.png|322]]

La curva seguente mostra l'SSE in funzione del numero di cluster:

![[8) Clustering-20241009102146917.png|319]]

**Interpretazione della Curva SSE:**

* **Errore elevato con pochi cluster:** Partendo da 1 o 2 cluster, l'errore (SSE) è molto elevato.
* **Diminuzione dell'errore con più cluster:** Man mano che il numero di cluster aumenta, l'SSE diminuisce.
* **Punti di flessione significativi:** La curva presenta due punti di flessione significativi:
    * **Numero di cluster pari a 5:** L'errore scende drasticamente.
    * **Numero di cluster pari a 10:** L'errore diminuisce ancora in modo significativo.
* **Miglioramento minimo con molti cluster:** Aumentando ulteriormente il numero di cluster, il miglioramento dell'SSE diventa minimo.

**Utilizzo dell'SSE per determinare il numero di cluster (k):**

I punti di flessione nella curva SSE possono indicare il numero ottimale di cluster. In questo esempio, i punti di flessione a 5 e 10 cluster suggeriscono che questi potrebbero essere buoni valori per *k*.

## Coesione e Separazione nella Clusterizzazione

La valutazione della validità di un risultato di clusterizzazione si basa su due concetti chiave: **coesione** e **separazione**.

**Coesione:** misura quanto gli oggetti all'interno di uno stesso cluster sono simili tra loro. Un cluster con alta coesione presenta oggetti vicini tra loro.

**Separazione:** misura quanto gli oggetti di cluster diversi sono dissimili tra loro. Cluster con alta separazione sono ben distinti e separati.

Esistono diversi modi per misurare coesione e separazione:

**1. Somma delle distanze:**

* **Coesione:** somma delle distanze tra tutte le coppie di oggetti all'interno di un cluster.
$$cohesion(c_i) = \sum_{x \in c_i, y \in c_i} proximity(x, y) )$$ 
* **Separazione:** somma delle distanze tra tutte le coppie di oggetti appartenenti a due cluster diversi.
$$separation(c_i, c_j) = \sum_{x \in c_i, y \in c_j} proximity(x, y) $$ 

**2. Distanza dal centroide:**

* **Coesione:** somma delle distanze tra ogni punto del cluster e il suo centroide.
$$cohesion(C_i) = \sum_{x \in c_i} proximity(x, m_i) $$ 
* **Separazione:** distanza tra i centroidi di due cluster diversi.
$$separation(C_i, C_j) = proximity(m_i, m_j)$$ 

**3. SSE e BSS:**

* **Coesione:** misurata tramite la somma dei quadrati delle distanze tra ogni punto e il centroide del suo cluster (SSE o WSS).
$$SSE = WSS = \sum_{i} \sum_{x \in C_i} (x - m_i)^2$$ 
* **Separazione:** misurata tramite la somma dei quadrati delle distanze tra i centroidi dei cluster e il centroide complessivo, pesata per la grandezza dei cluster (BSS).
$$BSS = \sum_{i} |C_i|(m - m_i)^2$$ 

In generale, si desidera ottenere una **bassa coesione** (oggetti simili all'interno del cluster) e un'**alta separazione** (oggetti diversi tra cluster differenti). Questo indica una buona clusterizzazione, dove i cluster sono ben definiti e distinti.

## Silhouette Coefficient

Un altro parametro utile per valutare la qualità della clusterizzazione è il **silhouette coefficient**. Questo parametro viene calcolato per ogni punto e fornisce un'indicazione di quanto il punto sia ben classificato nel suo cluster.

Per ogni punto *i*, il silhouette coefficient è calcolato come segue:

* **a**: distanza media tra *i* e gli oggetti dello stesso cluster.
* **b**: distanza media tra *i* e gli oggetti che stanno al di fuori del cluster.

$$s = \frac{(b - a)}{max(a, b)}$$

Il valore del silhouette coefficient varia tra -1 e 1:

* **s ≈ 1:** indica che il punto è ben classificato nel suo cluster, poiché è molto più vicino agli altri punti del suo cluster rispetto ai punti di altri cluster.
* **s ≈ 0:** indica che il punto è vicino al confine tra due cluster, e potrebbe essere classificato in entrambi.
* **s ≈ -1:** indica che il punto è probabilmente classificato nel cluster sbagliato, poiché è più vicino ai punti di un altro cluster rispetto ai punti del suo cluster.

**Interpretazione:**
Un valore di silhouette coefficient alto (vicino a 1) indica una buona separazione tra i cluster, mentre un valore basso (vicino a 0 o negativo) indica una scarsa separazione o una possibile errata classificazione dei punti.

## Altre tecniche di clustering

### Fuzzy clustering

La clusterizzazione fuzzy, o soft clustering, si differenzia dalla clusterizzazione hard (crisp) in quanto consente ai punti di appartenere a più cluster contemporaneamente. In altre parole, un punto può essere membro di più cluster con un diverso grado di appartenenza.

**Appartenenza Fuzzy:**

In un contesto fuzzy, ogni punto *i* è associato a ciascun cluster *j* con un **peso** *wij*, che rappresenta il grado di appartenenza del punto *i* al cluster *j*. La somma dei pesi per ogni punto deve essere uguale a 1, ovvero la somma delle appartenenze di un punto a tutti i cluster è sempre 1.

**SSE nel Fuzzy Clustering:**

L'SSE (Somma dei Quadrati degli Errori) viene adattata per tenere conto dei pesi di appartenenza:

$$SSE = \sum_{j=1}^k \sum_{i=1}^m w_{ij} \cdot dist(x_i, c_j)²$$

dove:

* $x_i$ è il punto che stiamo considerando
* $c_j$ è il centroide del cluster *j*

**K-Means Fuzzy:**

Il K-Means fuzzy è una variante del K-Means che tiene conto dei pesi di appartenenza. L'obiettivo è minimizzare l'SSE, che viene modificata come segue:

$$SSE = \sum_{j=1}^k \sum_{i=1}^m w_{ij}^p \cdot dist(x_i, c_j)^2$$

Il parametro *p* controlla l'influenza dei pesi. Più *p* è elevato, meno peso viene dato ai pesi di appartenenza. In generale, *p >* 1.
![[8) Clustering-20241009102336916.png|439]]

**Algoritmo K-Means Fuzzy:**

1. Inizializzazione dei pesi *wij* in modo casuale.
2. Aggiornamento dei centroidi:

$$c_{ij} = \frac{{\sum_{i=1}^m w_{ij} \cdot x_i}}{\sum_{i=1}^m w_{ij}} $$

3. Aggiornamento dei pesi:

$$w_{ij} =\frac{\left( \frac{1}{dist(x_{i}c_{j})}^2 \right)^{\frac{1}{p-1}}}{ \sum_{j=1}^k\left( \frac{1}{dist(x_{i},c_{j})}^2 \right)^{\frac{1}{p-1}}}$$

4. Ripetizione dei passi 2 e 3 fino alla convergenza.

**Rappresentazione Grafica:**

### Clustering Grid-Based

Il clustering grid-based divide lo spazio dei dati in celle (rettangoli o iper-rettangoli) e conta il numero di punti in ogni cella. Le celle con un numero di punti superiore a una soglia vengono considerate come cluster.

**Algoritmo:**

1. Dividere lo spazio dei dati in celle.
2. Contare il numero di punti in ogni cella.
3. Identificare le celle con un numero di punti superiore a una soglia.
4. Raggruppare le celle adiacenti con un numero di punti superiore alla soglia.

**Rappresentazione Grafica:**
![[8) Clustering-20241009093514470.png|485]]
**Pseudocodice:**
1. Definisci un insieme di celle di griglia.
2. Assegna gli oggetti alle celle appropriate e calcola la densità di ciascuna cella.
3. Elimina le celle con una densità inferiore a una soglia specificata, r.
4. Forma cluster da gruppi contigui (adiacenti) di celle dense. 



**Vantaggi:**
* Efficiente per grandi dataset.
* Semplice da implementare.

**Svantaggi:**
* Sensibile alla dimensione delle celle.
* Non adatto per dati con densità variabile.

### Clusterizzazione in Sottospazi

Fino ad ora non abbiamo mai discusso della dimensionalità degli oggetti. Quando i nostri oggetti hanno molte dimensioni, la clusterizzazione può essere poco efficace. In questi casi, è necessario ridurre la dimensione degli oggetti. Nel contesto della clusterizzazione, questo problema è ancora più rilevante.

Ad esempio, consideriamo la seguente distribuzione di punti in tre dimensioni:

![[8) Clustering-20241009093529428.png|306]]
Con 3 attributi, non è detto che riusciamo ad identificare dei cluster. Tuttavia, se consideriamo gli stessi oggetti in due dimensioni (*x, y*), otteniamo la seguente distribuzione:

![[8) Clustering-20241009093536222.png|346]]

Se diminuiamo ulteriormente il numero di attributi, possiamo identificare 3 cluster (considerando come attributo *x*, possiamo vederli considerando solo l'asse *x* della figura precedente). Anche con *y* identifichiamo tre cluster, ma sono diversi rispetto a quelli individuati con *x*.

Il processo di clusterizzazione è quindi molto influenzato dal numero di attributi, perché all'aumentare degli attributi aumenta la distanza tra gli oggetti. 

## Clusterizzazione basata su Grafi

La clusterizzazione basata su grafi sfrutta la rappresentazione dei dati come grafi per identificare i cluster. Due tecniche principali sono:

**1. Clustering Gerarchico Divisivo:**
![[8) Clustering-20241009102407253.png|538]]
* **Costruzione del Grafo:** Si crea un grafo completo non orientato, dove ogni nodo è collegato a tutti gli altri.
* **Albero Ricoprente Minimo:** Si costruisce l'albero ricoprente minimo del grafo.
* **Separazione dell'Albero:** Si separano i cluster rompendo l'arco con la distanza maggiore nell'albero ricoprente. Questo processo si ripete fino a ottenere solo cluster singole.

**Complessità:** L'algoritmo ha una complessità di $O(n^3)$ se si utilizza un algoritmo di costruzione dell'albero ricoprente con complessità $O(n^2)$. Utilizzando una struttura ad heap, la complessità si riduce a $O(n^2\log n$).

**2. Shared Nearest Neighbour (SNN):**

* **Assegnazione dei Pesi:** Si assegna un peso a ciascun arco del grafo, che corrisponde al numero di vicini in comune tra i due nodi.
* **Algoritmo di Jarvis-Patrick:**
    1. Si costruisce un grafo dove un arco esiste tra due nodi se uno è nel vicinato dell'altro (definito da una soglia di distanza).
    2. Si assegnano i pesi agli archi come descritto sopra.
    3. Si eliminano gli archi con un peso inferiore a una soglia.
    4. Le componenti connesse del grafo risultante rappresentano i cluster.

**Vantaggi:** L'algoritmo SNN funziona bene con cluster di diverse dimensioni e densità.

**Svantaggi:** L'algoritmo non è efficace quando i cluster presentano collegamenti tra loro.

**3. SNN Density Based Clustering:**

* **Combinazione di SNN e DBSCAN:** Questo algoritmo combina l'algoritmo SNN con il DBSCAN.
* **Fasi:**
    1. Si calcola la matrice di similarità.
    2. Si costruisce un grafo sparso.
    3. Si assegnano i pesi agli archi utilizzando la tecnica SNN.
    4. Si applica il DBSCAN per identificare i cluster in base alla densità.

**Vantaggi:** Questo algoritmo è più efficace rispetto all'algoritmo SNN tradizionale.

**In sintesi:** La clusterizzazione basata su grafi offre un approccio alternativo alla clusterizzazione tradizionale, sfruttando la struttura dei dati come grafi per identificare i cluster. Le tecniche SNN e SNN Density Based Clustering sono particolarmente utili per gestire cluster di diverse dimensioni e densità.
