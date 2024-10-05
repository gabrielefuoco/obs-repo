
### 17.1 Cos'è la Cluster Analysis

L'analisi dei cluster raggruppa gli oggetti dati in base alle informazioni che si trovano solo nei dati che descrivono gli oggetti e le loro relazioni. L'obiettivo è che gli oggetti all'interno di un gruppo siano **simili** (o correlati) l'uno all'altro, in altre parole i gruppi devono essere **coesi**, e **diversi** (o non correlati) dagli oggetti di altri gruppi, i gruppi devono essere **ben separati**.

Il clustering può essere considerato come una forma di classificazione in quanto crea un'etichettatura di oggetti con etichette di classe (cluster). Tuttavia, deriva queste etichette solo dai dati e l'analisi che si esegue **non è supervisionata**. Al contrario, la classificazione studiata fino ad ora è classificazione **supervisionata**.

La divisione in gruppi di per sé non è sempre clusterizzazione. Inoltre, mentre i termini **segmentazione** e **partizionamento** sono talvolta usati come sinonimi per il clustering, questi termini sono spesso usati per approcci al di fuori dei limiti tradizionali dell'analisi dei cluster. Il termine partizionamento è spesso usato in relazione a tecniche che dividono i grafici in sottografi e che **non sono** fortemente collegate al clustering.

La clusterizzazione può essere utilizzata anche per effettuare un pre-processing dei dati; può migliorare l'attività di classificazione del *KNN*.

### 17.2 Differenti tipi di clustering

Andiamo ora a vedere i metodi di clusterizzazione sui quali ci concentreremo:

* **Partitional**: semplicemente si dividono i dati in insiemi non sovrapposti tali per cui ogni oggetto del dataset è esattamente in un solo insieme.

![image](image53.png)

* **Hierarchical approach**: se consentiamo ai cluster di avere sotto-cluster allora otteniamo una struttura gerarchica. Il clustering gerarchico può essere schematizzato tramite una struttura ad albero in cui ogni nodo (cluster) dell'albero, eccetto i nodi foglia, sono l'unione dei nodi figli. La radice dell'albero è un unico cluster che contiene tutti i dati. Le foglie in alcuni casi sono cluster che contengono un solo oggetto.

* **Partitioning approach**: si costruiscono diverse partizioni dei dati e poi le si valutano attraverso un criterio, per esempio la somma degli errori al quadrato (*SSE*).

* **Density Based approach**: si genera un cluster nel momento in cui il numero di oggetti per unità di volume (se si considera la definizione euclidea) è molto alto.

Accenneremo anche al metodo **Grid Based** simile all'approccio basato su densità in quanto valuta la distribuzione nello spazio degli oggetti. Un altro esempio è andare a definire i cluster sulla base dei **link**, per esempio, nel caso di una rete internet si vanno a creare delle comunità (cluster) guardando la topologia del grafo.

Altre caratteristiche del clustering sono:

* **Exclusive vs non-exclusive**: un oggetto viene assegnato esattamente ad un solo cluster nel caso di separazione esclusiva. In alcuni casi uno stesso oggetto può essere assegnato a più cluster.
* **Fuzzy**: in questo tipo di clustering ogni oggetto appartiene a tutti i cluster però con un peso diverso. Il peso di appartenenza può andare da 0 a 1.
* **Parziale vs Completo**: in alcuni casi ci interessa effettuare clustering solamente su un sottoinsieme dei dati.
* **Eterogeneo vs omogeneo**: in alcuni casi i cluster possono essere di diversa forma, dimensione, densità.

### 17.3 Differenti tipi di cluster

Andiamo ora ad analizzare i diversi tipi di cluster:

* **Well Separated**: quando parliamo di separazione quello che noi vorremmo è che i cluster siano ben identificabili e ben separati tra di loro. Abbiamo quindi dei gruppi di oggetti coesi all'interno ma ben separati all'esterno.
* **Center based**: in questo caso i cluster sono identificati dal **centroide**, che è il centro del nostro cluster. Se le feature sono attributi numerici non è altro che la media dei valori degli oggetti che appartengono al cluster. In questo caso, i nostri cluster hanno una forma **convessa**. Possiamo avere quindi dei cluster che non sono ben separati tra di loro.

![image](image54.png)

* **Contiguous cluster**: in questo caso si va a misurare la **distanza che vi è tra gli oggetti**. Se si considera l'immagine riportata di seguito i due insiemi circolari collegati dal segmento in realtà vengono considerati come un unico cluster perché gli oggetti dei due insiemi in realtà potrebbero essere ad una distanza molto bassa tra di loro.

![image](image55.png)

* **Density based**: in questa definizione un cluster è una **regione densa** di oggetti circondata da una regione con una più bassa densità.

![image](image56.png)

* **Shared-Property (Conceptual Clusters)**: alcuni algoritmi potrebbero identificare dei cluster a seconda di alcune proprietà che condividono gli oggetti.

### 17.4 Funzione di ottimizzazione

Nel processo di cluster quello che vogliamo è **ottimizzare una funzione**, ovvero, vogliamo trovare una soluzione che risolve un problema di ottimizzazione. Ovviamente trovare la soluzione migliore, come sempre, è un problema difficile (*NP-hard*) perché per trovare tale soluzione significa andare a valutare tutte le possibili combinazioni di valori in grado di risolvere il problema.

La funzione di ottimizzazione che noi vorremmo ottimizzare è la seguente:

$$SSE = \sum_{i=1}^k \sum_{x \in C_i} dist^2(m_i, x) \qquad (17.1)$$ 
| *SSE* = | *i*=1 | *x∈Ci* | > *dist*2(*mi, x*) | (17.1) |


ovvero la funzione data dalla somma dell'errore quadratico medio. Dove:

* *Ci*: insieme di cluster
* *mi*: centroide
* *k*: numero di cluster

Se noi abbiamo due clusterizzazioni alternative riusciamo a misurare, tramite l'SSE, quale delle due è la migliore. Tale distanza non è altro che la somma degli errori quadratici. L'errore è dato dalla somma delle distanze degli oggetti dal centroide. 

## Appunti di Studio: Clustering Partizionale

### 17.5 Valutazione del Clustering Partizionale

Dato un database *D* di *n* oggetti, i metodi di partizionamento suddividono questi *n* oggetti in *k* cluster.

Se consideriamo dati la cui misura di prossimità è la **distanza euclidea**, la funzione obiettivo che misura la qualità di un clustering è la **somma dell'errore quadratico (SSE)**, anche conosciuta come *scatter*.

$$SSE = \sum_{i=1}^k \sum_{x \in C_i} dist^2(c_i, x) \qquad (17.2)$$ 
*k*

> *SSE* = Σ Σ *dist*(*ci, x*)² (17.2)

*i*=1 *x∈Ci*

In altre parole, calcoliamo l'errore di ogni punto dati, ovvero la sua distanza euclidea dal centroide più vicino, e poi calcoliamo la somma totale degli errori quadratici.

Dati due diversi insiemi di cluster, preferiamo quello con l'SSE più piccolo, poiché ciò significa che i prototipi (centroidi) di questo clustering sono una migliore rappresentazione dei punti nel loro cluster.

**Non è possibile confrontare due soluzioni che hanno un numero diverso di cluster**, perché all'aumentare del numero di cluster l'SSE diminuisce.

### 17.6 Metodi Basati sul Partizionamento

L'obiettivo è, dato il numero *k*, trovare un insieme di *k* cluster che enumera esaustivamente tutte le partizioni (obiettivo globale). Tale partizione deve minimizzare l'SSE.

Esistono diverse tecniche per effettuare questo tipo di partizionamento, ma due delle più importanti sono **K-means** e **K-medoid**.

* **K-means** definisce un prototipo in termini di **centroide**, che di solito è la media di un gruppo di punti, ed è tipicamente applicato agli oggetti in uno spazio *n-dimensionale* continuo.
* **K-medoid** è una variante del K-means dove gli attributi possono non essere numerici.

### 17.7 K-means

La tecnica di clustering K-means è semplice:

1. Scegliamo i centroidi iniziali *k*, dove *k* è un parametro specificato dall'utente, ovvero il numero di cluster desiderati.
2. Ogni punto viene assegnato al centroide più vicino.
3. Ogni insieme di punti assegnati a un centroide forma un cluster.
4. Il centroide di ciascun cluster viene aggiornato in base ai punti assegnati al cluster.
5. Ripetiamo i passaggi 2-4 fino a quando nessun punto cambia cluster, o in modo equivalente, fino a quando i centroidi rimangono gli stessi.

![image](image57.png)

#### 17.7.1 Dettagli K-means

Il risultato è un insieme di *k* cluster calcolati sulla base di una scelta iniziale randomica. L'algoritmo converge in poche iterazioni. La complessità dell'algoritmo è:

$O(n \times k \times d \times i)$ 

dove:

* *n*: numero di punti
* *d*: numero di attributi
* *k*: numero di cluster
* *i*: numero di iterazioni

L'algoritmo è molto efficiente, poiché converge in poche iterazioni. Ciò che potrebbe renderlo meno efficiente è un numero troppo elevato di punti.

La vicinanza tra gli oggetti può essere misurata con una qualunque misura di distanza studiata a inizio corso. Nel K-medoid, praticamente identico al K-means, il centroide è determinato dall'elemento più frequente.

### Limitazioni e Debolezze del K-means

* **Forma convessa:** I cluster devono avere una forma convessa, altrimenti non riusciamo a catturarli con l'algoritmo K-means.
* **Densità diversa:** K-means non riesce ad identificare i cluster che hanno densità diversa.
* **Grandezza diversa:** L'algoritmo non riesce a identificare cluster con grandezza diversa.

Una soluzione è usare un numero di cluster più elevato.

### Importanza nella Scelta del Punto Iniziale

Il risultato dipende da dove fissiamo il centroide inizialmente.

Se ci sono *k* cluster "reali", la probabilità di selezionare un centroide da ogni cluster è piccola. La probabilità si abbassa all'aumentare di *k*.

A volte i centroidi iniziali si riaggiustano nel modo "giusto", e a volte non lo fanno. Non abbiamo margine di errore.

**Esempio:**

Consideriamo un rettangolo con vertici:


$A(1, 1), B(3, 1), C(1, 2), D(3, 2)$


e due cluster.

![image](image58.png)



Scegliendo come centroide iniziale A e C, i centroidi finali sono (2,1) e (2,2) (coordinate segnate con la x) con un errore uguale a 4. La soluzione ottimale, ottenuta selezionando come centroidi iniziali A e B, ha un errore uguale a 1. L'errore può aumentare indefinitamente spostando B e D a destra.

### 17.8 Varianti del K-means

Esistono diverse varianti dell'algoritmo k-means che differiscono per:

* Selezione dei *k* punti iniziali
* Calcolo della similarità dei punti
* Strategie per calcolare i cluster (se siamo in uno spazio euclideo è il valore medio)

#### 17.8.1 Selezione dei *k* Punti Iniziali

**Scelta del seme:**

I risultati possono variare drasticamente in base alla selezione casuale dei semi. Alcuni semi possono comportare uno scarso tasso di convergenza o convergenza a raggruppamenti sub-ottimali.

Esistono diverse euristiche per ovviare a questo problema:

* **Scelta random dei centroidi all'interno del nostro spazio:** può provocare problemi.
* **Scelta random di un esempio dal dominio:** è un oggetto ovvero un rappresentante e non più un valore medio.
* **Scegliamo come centroidi punti molto dissimili l'uno dall'altro (further centre).**
* **Scelta multipla dei centroidi.**
* **Scelta dei centroidi iniziali mediante altre tecniche di clusterizzazione.** 

## Euristica dei centri più lontani (furthest centre)

L'euristica segue il seguente ragionamento:

1. Si sceglie un punto *µ*1 in maniera casuale.
2. Per *i* che va da 2 a *k*, si sceglie il punto *µi* che è il più distante rispetto a qualsiasi altro centroide precedentemente generato.

**Esempio:**

![image](image59.png)

**Definizione matematica:**

$µ_{i} = \text{arg max } x µ_{j}:1<j<id(x, µ_{j})$ 


dove:

* **arg max *x*:** indica il punto che ha la maggior distanza dal centro precedente
* ***µj*:1*\<j\<id*(*x, µj*):** la distanza minima da *x* a qualsiasi centro precedente

**Svantaggi:**

Questa euristica risulta essere sensibile agli outliers, come si può vedere nella figura seguente:

![image](image60.png)

**Riepilogo:**

* Si seleziona il primo punto a caso o si prende il centroide di tutti i punti.
* Per ogni centroide successivo, si seleziona il punto più lontano da uno qualsiasi dei centroidi iniziali già selezionati.

Questo metodo garantisce un insieme di centroidi iniziali ben separati, ma è costoso calcolare il punto più lontano dall'attuale insieme di centroidi iniziali. Per superare questo problema, l'approccio viene spesso applicato a un campione dei punti.

## Altre soluzioni per la scelta dei punti iniziali

* Eseguire il clustering più volte, ma non è molto utile.
* Effettuare un campionamento dei dati e poi eseguire un clustering gerarchico per determinare i centroidi iniziali.
* Selezionare un numero *k* maggiore rispetto a quello prefissato, creare i rispettivi cluster e poi selezionare i centroidi che effettivamente sono più distanti tra di loro.
* Generare un numero molto elevato di cluster per poi effettuare un clustering gerarchico.
* Usare la variante Bisecting K-means che non è suscettibile alla scelta dei punti iniziali.

## 17.8.2 K-means ++

K-means++ è un nuovo approccio per l'inizializzazione di K-means. Questa procedura garantisce di trovare una soluzione di clustering K-means che sia ottimale all'interno di un fattore di *O*(*log*(*k*)), che in pratica si traduce in risultati di clustering notevolmente migliori in termini di SSE inferiore.

**Passaggi:**

1. Selezionare un centroide in maniera casuale.
2. Per ogni punto *x*, calcolare la distanza *D*(*x*) dal punto al centroide più vicino.
3. Scegliere un nuovo punto come centroide in maniera casuale, utilizzando una distribuzione di probabilità pesata dove il punto *x* è scelto con una probabilità che sia proporzionale a *D*(*x*)2.
4. Ripetere i passi 2 e 3 fino ad arrivare a *k* cluster.
5. Eseguire il clustering con l'algoritmo k-means standard. 


## Gestire cluster vuoti

Uno dei problemi con l'algoritmo di base K-means è che si possono ottenere cluster vuoti se nessun punto viene assegnato a un cluster durante la fase di assegnazione. Se ciò accade, è necessaria una strategia per scegliere un centroide sostitutivo, poiché altrimenti l'errore quadrato sarà più grande del necessario.

**Approcci:**

* **Punto più lontano:** Scegliere il punto che è più lontano da qualsiasi centroide attuale. Questo elimina il punto che attualmente contribuisce maggiormente all'Errore quadrato totale. (Potrebbe essere usato anche un approccio K-means++).
* **Cluster con SSE più alta:** Scegliere il centroide sostitutivo a caso dal cluster che ha l'SSE più alta. Questo in genere dividerà il cluster e ridurrà l'SSE complessivo del clustering.

Se ci sono diversi cluster vuoti, questo processo può essere ripetuto più volte.

## Aggiornare i centroidi in maniera incrementale

Invece di aggiornare i centroidi del cluster dopo che tutti i punti sono stati assegnati, i centroidi possono essere aggiornati in modo incrementale, dopo ogni assegnazione di un punto a un cluster. Si noti che questo richiede zero o due aggiornamenti ai centroidi del cluster ad ogni passaggio, poiché un punto si sposta su un nuovo cluster (due aggiornamenti) o rimane nel suo cluster corrente (zero aggiornamenti).

**Vantaggi:**

* **Previene cluster vuoti:** Tutti i cluster iniziano con un singolo punto e, se un cluster ha un solo punto, non verrà mai prodotto perché quel punto verrà sempre riassegnato allo stesso cluster.
* **Peso relativo dei punti:** Il peso relativo del punto da aggiungere può essere regolato; ad esempio, il peso dei punti viene spesso diminuito man mano che il raggruppamento procede. Questo può comportare una migliore precisione e una convergenza più rapida.

**Svantaggi:**

* **Dipendenza dall'ordine:** L'aggiornamento dei centroidi introduce in modo incrementale una dipendenza dell'ordine. I cluster prodotti di solito dipendono dall'ordine in cui vengono elaborati i punti.
* **Costo computazionale:** Gli aggiornamenti incrementali sono leggermente più costosi.

## Pre e post processing

**Pre processing:**

* Eliminazione degli outlier.
* Normalizzazione dei dati.

**Post processing:**

* Eliminare i cluster di piccole dimensioni che potrebbero rappresentare outliers.
* Dividere i cluster che presentano un SSE troppo elevato per ridurre la somma dell'errore quadratico.
* Unire cluster relativamente vicini e con SSE basso.

Tutti questi step possono essere effettuati durante il processo di clustering (ISODATA ALGORITHM).

## 17.8.3 Bisecting K-means

L'algoritmo Bisecting K-means è una semplice estensione dell'algoritmo K-means che si basa su un'idea semplice: per ottenere *k* cluster, dividere l'insieme di tutti i punti in due cluster, selezionare uno di questi cluster da dividere e così via, fino a quando non sono stati prodotti *k* cluster.

**Passaggi:**

1. Calcolare il centroide (baricentro) *w*.
2. Selezionare un punto casuale *cL* (nome del punto).
3. Selezionare un punto *cR* che sia simmetrico a *cL* quando comparato con *w* (il segmento *cR →w* è lo stesso di *cL →w*).
4. Suddividere la nuvola di punti in due sotto-insiemi. I punti più vicini a *cR* rientreranno nell'insieme *R*, quelli più vicini a *cL* rientreranno nell'insieme *L*.
5. Reiterare la procedura per gli insiemi *R* e *L*.

**Note:**

* Ci sono diversi modi per scegliere quale cluster dividere (il più grande, quello con l'SSE più grande, o un criterio basato sia sulle dimensioni che sull'SSE).
* Tale algoritmo può produrre un clustering gerarchico o partizionale.

## 17.8.4 K-modes

È una variante del k-means in grado di gestire dati categorici. Al posto della media dei valori viene utilizzata la **moda**. Sono necessarie nuove misure di distanza per i dati di tipo categorico e, inoltre, è necessario un metodo basato sulla frequenza dei valori degli oggetti per aggiornare i cluster. Se invece si considerano dati sia continui che categorici il metodo da utilizzare è il *k-prototype*.

## 17.9 Clustering Gerarchico

Le tecniche di clustering gerarchico sono una seconda categoria importante dei metodi di clustering che consiste nel generare una gerarchia di cluster. Ci sono due approcci di base per generare un clustering gerarchico:

* **Agglomerative (bottom up):** Inizia considerando i punti come singoli cluster e, ad ogni passaggio, unisci la coppia di cluster più vicina. Ciò richiede la definizione di una nozione di prossimità del cluster.
* **Divisive (top down):** Inizia con un cluster all-inclusive e, ad ogni fase, divide il cluster fino a quando non rimangono solo cluster di singoli punti. In questo caso, dobbiamo decidere quale cluster dividere in ogni fase e come fare la scissione.

**Visualizzazione:**

* **Dendrogramma:** Visualizza sia le relazioni cluster-subcluster che l'ordine in cui i cluster sono stati uniti (vista agglomerativa) o divisi (vista divisiva).
* **Diagramma di cluster nidificato:** Per insiemi di punti bidimensionali.

![image](image61.png)

**Vantaggi:**

* Non richiedono di specificare un numero di cluster iniziale.
* Si possono ottenere quanti cluster si vogliono.

**Note:**

* Gli algoritmi gerarchici tradizionali utilizzano una matrice di similarità (o dissimilarità) e operano la suddivisione (o unione) su un cluster alla volta.
* Il valore di k non è definito a priori ma si può scegliere a posteriori. 

## 17.9.1 Algoritmo gerarchico di base

Molte tecniche di clustering gerarchico agglomerativo sono variazioni del seguente approccio: a partire dai singoli punti come cluster, unire successivamente i due cluster più vicini fino a quando rimane un solo cluster. Questo approccio è espresso in modo più formale dal seguente algoritmo:

```
1. Inizializzare la matrice di prossimità tra tutti i punti.
2. Trovare la coppia di cluster più vicina.
3. Unire i due cluster più vicini in un unico cluster.
4. Aggiornare la matrice di prossimità.
5. Ripetere i passaggi 2-4 fino a quando non rimane un solo cluster.
```

Dobbiamo aggiornare la matrice di prossimità poiché se uniamo due cluster, ci sarà un cluster in meno e di conseguenza una riga e una colonna in meno della matrice. L'operazione chiave è il calcolo della prossimità tra due cluster. A seconda della metodologia utilizzata per il calcolo della prossimità, andiamo a definire diversi approcci.

**Esempio su slide**

## Definizioni di distanze inter-cluster

Prendiamo come punto di riferimento la seguente immagine rappresentativa di due cluster:

![image](image63.png){width="3.44583in" height="1.46806in"}

La vicinanza del cluster è in genere definita con un particolare tipo di cluster in mente. Ad esempio, molte tecniche di clustering gerarchico agglomerativo, come *MIN*, *MAX* e *Group Average*, provengono da una vista **grafica** dei cluster.

* **MIN**: definisce la vicinanza del cluster come la vicinanza tra i due punti più vicini che si trovano in cluster diversi, o usando termini grafici, il segmento più corto tra due nodi in diversi sottoinsiemi di nodi. Ciò produce cluster basati sulla contiguità. La tecnica MIN riesce a gestire forme non ellittiche dei cluster, ma è sensibile al rumore e ai valori anomali.

![image](image64.png){width="3.44583in" height="1.475in"}

* **MAX**: prende la distanza tra i due punti più lontani in diversi cluster come la vicinanza del cluster, o usando termini di grafo, il segmento più lungo tra due nodi in diversi sottoinsiemi di nodi. Questo approccio è meno suscettibile al rumore e ai valori anomali, ma può dividere grandi cluster e favorisce le forme globulari.

![image](image65.png){width="3.44583in" height="1.52639in"}

Se le nostre prossimità sono distanze, allora i nomi, MIN e MAX, sono brevi e suggestivi. Per le somiglianze, tuttavia, dove valori più alti indicano punti più vicini, i nomi sembrano invertiti. Per questo motivo, di solito preferiamo usare i nomi alternativi, single link e complete link, rispettivamente.

Un altro approccio basato su grafici, la tecnica della **media di gruppo** (group average), definisce la vicinanza del cluster come le prossimità medie a coppie (lunghezza media degli segmenti) di tutte le coppie di punti di diversi cluster. Matematicamente questa può essere rappresentata dalla seguente formula:

```
proximity(Ci, Cj) = Σpi∈Ci,pj∈Cj proximity(pi, pj) / (Ci × Cj) (17.5)
```

Possiamo vedere questo approccio come una via di mezzo tra MIN e MAX. Risulta essere poco suscettibile ai rumori, però tende a creare cluster di forma globulare.

![image](image66.png){width="3.44583in" height="1.46111in"}

Se, invece, prendiamo una visione basata sul prototipo, in cui ogni cluster è rappresentata da un centroide, diverse definizioni di prossimità del cluster sono più naturali. Quando si utilizzano i centroidi, la vicinanza del cluster è comunemente definita come la **vicinanza tra i centroidi** del cluster stesso.

![image](image67.png){width="3.44583in" height="1.46111in"}

Una tecnica alternativa, il **metodo di Ward**, il quale presuppone anche che un cluster sia rappresentato dal suo centroide, ma misura la vicinanza tra due cluster in termini di aumento della SSE che deriva dalla fusione dei due cluster. Come K-means, il metodo di Ward tenta di ridurre al minimo la somma delle distanze quadrate dei punti dai loro centroidi del cluster. Può essere visto come un K-means gerarchico, infatti può essere usato per inizializzare anche l'algoritmo K-means studiato in precedenza. Anche questo approccio tende a creare cluster globulari ed è poco suscettibile al rumore e agli outliers. Matematicamente si può descrivere come:

```
∆(A, B) = Σx∈A∪B (x - mA∪B)² - Σx∈A (x - mA)² - Σx∈B (x - mB)² (17.6)
```

(non si specifica cosa è *m*).

## 17.9.2 Limitazioni clustering gerarchico

Le problematiche sono che rispetto al k-means, che è basato sulla minimizzazione dell'errore quadratico medio, il clustering gerarchico agglomerativo non può essere visto come l'ottimizzazione globale di una funzione obiettivo. Invece, le tecniche di clustering gerarchico agglomerativo utilizzano vari criteri per decidere localmente, in ogni fase, quali cluster devono essere uniti (o divisi per approcci divisivi).

Una volta che due cluster sono stati uniti non possono essere più separati. Infine, come abbiamo già visto, alcuni approcci sono sensibili a rumore e outlier e hanno difficoltà nel gestire cluster di forma non globulare o di grande dimensione.

## In cosa consiste il cluster gerarchico divisivo?

Noi abbiamo una serie di punti e delle istanze tra ciascuna coppia di punti e creiamo il nostro grafo completo. Dopodiché possiamo andare a creare uno Spanning Tree minimo del grafo: prendiamo un albero ricoprente del grafo di costo minimo. Adesso andiamo a dividere il nostro cluster: si sceglie l'arco massimo e lo andiamo ad eliminare, in questo modo stiamo dividendo il cluster in due, continuiamo fino a quando ogni cluster avrà un solo oggetto.

Questo è un algoritmo molto efficiente: costruire l'albero ricoprente utilizzando Primm costa *O*(*n*2) o utilizzando Kruskal costa *O*(*m* log(*n*)), dove *m* è l'arco. Si prendono gli archi dal più piccolo al più grande e se collegano due nodi che appartengono a cluster diversi si uniscono i cluster. 

## 17.10 DB-SCAN

Il clustering basato sulla densità individua le regioni ad alta densità che sono separate l'una dall'altra da regioni a più bassa densità. DBSCAN è un algoritmo di clustering basato sulla densità semplice ed efficace che illustra una serie di concetti che sono importanti per qualsiasi approccio di clustering basato sulla densità. Faremo riferimento alla densità euclidea indicandola con la sigla EPS.

## 17.10.1 Classificazione di punti basata sulla densità center-based

L'approccio center-based ci consente di classificare un punto come:

1. **Core point**: questi punti si trovano all'interno di un cluster basato sulla densità. Un punto è un punto centrale se ci sono almeno MinPts entro una distanza da Eps, dove MinPts (numero di punti minimo) ed Eps sono parametri specificati dall'utente.
2. **Border point**: un punto di confine non è un punto centrale, ma rientra nel quartiere di un punto centrale. Un punto di confine può rientrare nelle vicinanze di diversi punti centrali.
3. **Noise point**: è un qualsiasi punto che non è né un punto centrale né un punto di confine. Questi verranno poi eliminati.

![image](image68.png){width="3.44583in" height="1.9625in"}

Un'altra suddivisione che può essere effettuata sui nodi è la seguente:

1. **Directly density-reachable**: Un punto *p* è densità-raggiungibile da un punto centrale *q* in maniera diretta se *p* è nelle vicinanze di *q*.
2. **Density reachable**: Un punto *p* è density-reachable da un punto (core) *q* se c'è una catena di punti *p*1*, \..., pn, p*1 = *q, pn* = *p* tale che *pi*+1 è direttamente densità raggiungibile da *pi*.
3. **Density connected**: Un punto *p* è density-connected a un punto *q* se c'è un punto *s* tale che entrambi, *p* e *q* sono density-reachable da *s*.

## 17.10.2 L'algoritmo DBSCAN

Date le precedenti definizioni di punti core, punti border e punti noise, l'algoritmo DBSCAN può essere descritto in modo informale come segue. Due punti centrali che sono abbastanza vicini - a distanza Eps l'uno dall'altro - sono messi nello stesso cluster. Allo stesso modo, qualsiasi border point che è abbastanza vicino a un core point viene inserito nello stesso cluster. I noise point vengono scartati.

![image](image69.png){width="4.82361in" height="1.1in"}

## 17.10.3 Complessità

La complessità temporale di base dell'algoritmo DBSCAN è *O*(*m×* tempo per trovare punti nelle vicinanze di Eps), dove *m* è il numero di punti. Nel peggiore dei casi, questa complessità è *O*(*m*2). Tuttavia, negli spazi a bassa dimensione (specialmente nello spazio 2D), le strutture dati come i KD-tree consentono un recupero efficiente di tutti i punti entro una determinata distanza da un punto specificato e la complessità temporale può essere bassa come *O*(*mlogm*) nel caso medio. Il requisito di spazio di DBSCAN, anche per i dati ad alta dimensione, è *O*(*m*) perché è necessario conservare solo una piccola quantità di dati per ogni punto, cioè l'etichetta del cluster e l'identificazione di ogni punto core, border o noise point.

## 17.10.4 Vantaggi e svantaggi

Poiché DBSCAN utilizza una definizione basata sulla densità di un cluster, è relativamente resistente al rumore e può gestire cluster di forme e dimensioni arbitrarie. Pertanto, DBSCAN può trovare molti cluster che non sono stati trovati utilizzando K-means. Tuttavia, DBSCAN ha problemi quando i cluster hanno densità molto diverse. Ha anche problemi con i dati ad alta dimensione perché la densità è più difficile da definire per tali dati. Infine, DBSCAN può essere costoso quando il calcolo dei nearest neighbor richiede il calcolo di tutte le prossimità a coppie, come di solito accade per i dati ad alta dimensione.

## 17.10.5 Selezione dei parametri del DBSCAN

Nel DBSCAN bisogna definire, come già detto, due parametri:

* **Lunghezza minima del raggio**
* **Numero minimo di oggetti che si devono trovare nell'intorno del raggio**

Un modo per aiutarci a fissare uno dei due parametri è quello di guardare la seguente curva: Sostanzialmente, supponendo di aver fissato il parametro *k* = 4 (ovvero i Nearest Neighbours), andiamo a valutare a che distanza si trova il *k −esimo* NN rispetto ai punti del cluster. L'idea di base infatti è che per i punti appartenenti ad un cluster la distanza che questi hanno rispetto al loro *k −esimo* NN è più o meno la stessa. Andiamo quindi ad ordinare i nostri punti in base alla loro distanza dal *K −esimo* NN. Sull'asse delle ascisse abbiamo 0 punti che hanno una distanza pari circa a 2; abbiamo 500 punti che hanno una distanza pari a circa 4; 1000 punti dove il nostro raggio è quasi 5 e così via. Per fissare quindi il valore di distanza andiamo a guardare alla curva, o meglio, al "ginocchio" della curva che ci può dare un'idea sul valore da assegnare al raggio del nostro intorno.

## 17.11 Valutazione dei cluster

Per poter dire che il risultato di una clusterizzazione è buono, abbiamo ovviamente bisogno di misure per la valutazione del clustering. Mentre nel processo di classificazione avevamo diversi parametri (accuracy, recall, precision, ...), di fatto per il clustering non abbiamo nessun parametro che ci dice se il risultato del processo di cluster è meglio rispetto ad un altro.

## 17.11.1 Diversi aspetti della cluster evaluation

Abbiamo bisogno anche in questo caso di parametri che ci consentono di mettere a confronto i risultati della clusterizzazione ma anche di poter stabilire, dato un unico risultato, se questo è accettabile o meno. Da una parte quindi abbiamo bisogno di **indici esterni**, ovvero confrontiamo il risultato con dei risultati provenienti dall'esterno detti *ideali* (come per esempio nella classificazione quando si effettua il confronto con le label di classe già esistenti). Possiamo applicare quindi misure come l'entropia ma in questo caso stiamo parlando di un processo supervisionato poiché abbiamo delle etichette che ci dicono in quale cluster l'oggetto deve trovarsi. In generale il processo di clusterizzazione sappiamo però non essere supervisionato.

Un'altra possibilità invece è quella di avere degli **indici interni** come ad esempio quello che abbiamo visto nel caso del K-means, ovvero la somma dell'errore quadratico (SSE). Andiamo quindi a valutare questa funzione (ricordiamo che il K-means non è nient'altro che il processo di ottimizzazione dell'SSE) e cerchiamo la clusterizzazione che minimizza l'errore quadratico.

Un altro tipo di indici sono quelli che ci permettono di confrontare due risultati e questi sono detti **indici relativi**.

## 17.11.2 Misurare la validità del cluster tramite la correlazione

Possiamo definire due matrici:

* **Matrice di prossimità**: presa una cella della matrice abbiamo la distanza tra i due oggetti che identificano la riga e la colonna. Matrice *n × n* dove *n* è il numero di punti.
* **Matrice ideale di similarità**: è sostanzialmente qualcosa di simile alla matrice di prossimità dove però i valori sono 0 e 1. Abbiamo 0 se i due oggetti appartengono a cluster diversi, 1 altrimenti.

Per vedere quindi se un risultato è valido andiamo a misurare la correlazione tra le due matrici.

Se la correlazione è elevata allora vuol dire che il nostro clustering è un risultato buono. Non è una buona misura per i cluster basati su contiguità o densità. Data la simmetria tra le matrici solo la correlazione tra *n*(*n~~−~~*1) 2 celle dovrà essere calcolata. 
## Valutare il clustering visualizzandone il grafico

Se prendiamo questi punti e li ordiniamo secondo la loro appartenenza al cluster, possiamo notare nel diagramma di correlazione che ci sono tre cluster evidenti che sono dati dai tre quadrati rossi sulla diagonale.

![image](image71.png){width="2.06667in" height="2.10694in"}

![image](image72.png){width="2.06667in" height="1.57778in"}

Effettivamente all'interno dei nostri dati quindi esiste una struttura catturata dal processo di clusterizzazione. Nel prossimo esempio la struttura che abbiamo catturato col processo di clusterizzazione ci dice che il risultato non è di elevata qualità e quindi non è detto che quello che abbiamo ottenuto sia qualcosa che cattura l'andamento all'interno dei dati. Stessa cosa per gli esempi di K-means e complete-link.

![image](image73.png){width="3.44583in" height="1.75in"}

![image](image74.png){width="3.44583in" height="1.34028in"}

![image](image75.png){width="3.44444in" height="1.76111in"}

## 17.11.3 Misure Interne: SSE

Un'altra possibilità è quella di andare a misurare l'SSE. Se abbiamo per esempio questa distribuzione di dati riportati in figura:

![image](image76.png){width="2.06667in" height="1.52778in"}

Possiamo andare a valutare la seguente curva che ci indica l'SSE a seconda del numero di cluster che abbiamo definito.

![image](image77.png){width="2.06806in" height="1.62639in"}

Possiamo notare che partendo con 1 o 2 cluster abbiamo un errore molto elevato. Man mano che il numero di cluster cresce l'SSE diminuisce. Quando aumenta il numero di cluster l'errore diminuisce sempre. Quello che possiamo notare è che ci sono due punti significativi:

* Quando il numero di cluster è pari a 5, in quanto l'errore scende drasticamente.
* Numero di cluster pari a 10, anche in questo caso l'errore risulta diminuire di molto.

Aumentando poi il numero di cluster il miglioramento dell'SSE è minimo. Questa misura è anche utile quindi per determinare il numero di cluster *k*.

## 17.11.4 Misure Interne: Coesione e separazione

Come possiamo stabilire quindi se il risultato della clusterizzazione è valido o meno? Abbiamo parlato di due aspetti che caratterizzano il processo di clusterizzazione:

* Coesione dei cluster
* Separazione dei cluster

Quello che ci aspettiamo è che il valore della coesione (misurato per esempio tramite l'SSE) sia basso mentre il calore della separazione sia elevato. Per il cluster *Ci* non è altro che la somma delle distanze tra le coppie di oggetti che si trovano nello stesso cluster.
$$cohesion(c_i) = \sum_{x \in c_i, y \in c_i} proximity(x, y) \qquad (17.7)$$ 

La separazione, invece, tra due cluster *Ci* e *Cj* non è altro che la somma delle distanze tra coppie di oggetti che appartengono rispettivamente a *Ci* e *Cj*.
$$separation(c_i, c_j) = \sum_{x \in c_i, y \in c_j} proximity(x, y) \qquad (17.8)$$ 


Questo è un modo per misurare coesione e separazione ma non è l'unico. Un modo alternativo è quello di andare a considerare, come abbiamo fatto per il clustering gerarchico, il rappresentante di ciascun cluster ed in questo caso le misure di separazione e di coesione sono definite come segue:

* **Coesione**: somma delle distanze che i punti hanno dal centroide

$$cohesion(C_i) = \sum_{x \in c_i} proximity(x, m_i) \qquad (17.9)$$ 


* **Separazione**: distanza tra i due centroidi di due cluster differenti

```
separation(Ci, Cj) = proximity(mi, mj) (17.10)
```

Questa misura viene introdotta diciamo per semplificare il calcolo di coesione e separazione. Abbiamo anche un parametro che ci indica la distanza dal centroide *mi* rispetto al centroide del cluster complessivo *m*:

```
separation(Ci) = proximity(mi, m) (17.11)
```

Un altro modo per misurare la coesione è l'uso dell'SSE

```
SSE = WSS = Σi Σx∈Ci (x −mi)² (17.12)
```

mentre la separazione può essere misurata come somma dei quadrati delle distanze inter-cluster:

```
BSS = Σi Ci(m −mi)² (17.13)
```

Dove *Ci* è la grandezza del cluster *i*. In pratica con la BSS misuriamo la distanza dal centroide *mi* rispetto al centroide del cluster complessivo *m* pesata con la grandezza del cluster.

## Silhouette Coefficient

Un altro parametro è il silhouette coefficient. Questo parametro viene definito per ogni punto, in seguito poi andiamo a vedere il risultato complessivo. Più precisamente, per ogni punto *i* andiamo a calcolare:

* *a*: distanza media tra *i* e gli oggetti dello stesso cluster
* *b*: distanza media tra *i* e gli oggetti che stanno al di fuori del cluster
$$s = \frac{(b - a)}{max(a, b)}$$ 


Potremmo avere infatti un cluster molto grande adiacente ad un cluster molto piccolo. Quando *b >> a* vuol dire che il nostro oggetto si trova in un cluster che è separato dagli altri, e quindi questo valore tende a 1. Normalmente 0 *≤s ≤*1 ma, come già detto, non sempre si verifica questo. 

## 17.12 Altre tecniche di clustering

### 17.12.1 Fuzzy clustering

Distinguiamo tra due tipi di clusterizzazione:

* **Fuzzy (Soft)**: consentiamo ai punti di appartenere a più cluster, quindi una clusterizzazione approssimata.
* **Hard (Crisp)**: non consentiamo ai punti di appartenere a più cluster, quindi una clusterizzazione esatta (tecniche viste fino ad ora).

Nella clusterizzazione Fuzzy ogni oggetto appartiene a tutti i cluster ma con un diverso peso che è indicato con *wij*. Possiamo quindi stabilire, grazie al peso, in che misura l'oggetto *i* appartiene al cluster *j*. Possiamo andare a valutare quanto un oggetto appartiene ad un cluster utilizzando l'SSE "calibrato" rispetto al peso *wij*:
$$SSE = \sum_{j=1}^k \sum_{i=1}^m w_{ij} *dist(x_i, c_j)²$$ 

dove:

* *xi* è il punto che stiamo considerando
* *cj* è il centroide

Un vincolo che bisogna tenere in considerazione è che la somma dei pesi del punto *i* deve fare 1. Questa è una generalizzazione del processo di clustering fin'ora studiata. Infatti, nei casi precedenti, *wij* vale 0 (se *i* non appartiene al cluster) o 1 (se *i* appartiene al cluster).

Se consideriamo la variante del K-means fuzzy funziona praticamente allo stesso modo solamente che nel calcolo e aggiornamento dei centroidi bisogna tenere conto di tutti i punti perché questi appartengono a tutti i cluster seppur con misura diversa. Ovviamente, anche in questo caso, l'obiettivo è andare a minimizzare l'SSE che viene riportata in questo modo:

```
SSE = Σj=1^k Σi=1^m wij^p *dist(xi, cj)² (17.16)
```

Il parametro *p* indica in che misura noi vogliamo considerare il peso. Più *p* è elevato, meno peso diamo a *wij*. In generale *p >* 1. Le due figure riportano il caso del calcolo dell'SSE sia nel caso in cui si considera *p* sia nel caso in cui non si considera.

![image](image78.png){width="3.44583in" height="2.10278in"}

![image](image79.png){width="3.44444in" height="2.175in"}

Descriviamo ora i passi dell'algoritmo:

1. Si scelgono i pesi *wij* in maniera casuale.
2. Aggiorniamo i centroidi secondo la seguente formula:

```
cij = Σi=1^m wij *xi / Σi=1^m wij (17.17)
```

3. Aggiorniamo i pesi con la seguente formula:

```
wij = (1 / Σj=1^k (dist(xi, cj) / dist(xi, ci))^(2/(p-1))) (17.18)
```

4. Ripetiamo i passi 2 e 3 fino alla convergenza.

L'algoritmo è identico al k-means ma in questo caso è ogni punto che appartiene a tutti i cluster.

### 17.12.2 Clustering Grid-Based

Abbiamo una distribuzione di punti in uno spazio e andiamo a dividere il nostro spazio in rettangoli (se consideriamo uno spazio cartesiano) o in iper-rettangoli. Costruiamo una matrice *n × n* dove *n* è il numero di rettangoli e andiamo a inserire nella cella *i* della matrice il numero di elementi che si trovano nel rettangolo corrispondente.

![image](image80.png){width="3.44583in" height="1.54861in"}

Di seguito riportato l'algoritmo:

![image](image81.png){width="3.44444in" height="0.79167in"}

### 17.12.3 Subspace Clustering

Fino ad ora non abbiamo mai parlato della dimensionalità degli oggetti. Quando i nostri oggetti hanno tante dimensioni la clusterizzazione può essere poco efficace. Quello che si fa è quindi andare a diminuire la dimensione degli oggetti. Nel caso della clusterizzazione questo è un problema ancora più importante. Per esempio consideriamo la seguente distribuzione di punti in tre dimensioni:

![image](image82.png){width="3.44583in" height="2.68194in"}

Con 3 features non è detto che riusciamo ad identificare dei cluster ma se andiamo a considerare sempre gli stessi oggetti però in due dimensioni (*x, y*) abbiamo la seguente distribuzione.

![image](image83.png){width="3.44583in" height="2.49583in"}

Se diminuiamo ancora il numero di attributi qui identifichiamo 3 cluster (considerando come attributo *x*, possiamo vederli considerando solo l'asse *x* della figura precedente). Anche con *y* identifichiamo tre cluster ma sono diversi rispetto a quelli individuati con *x*.

Il processo di clusterizzazione è quindi molto influenzato dal numero di feature perché all'aumentare degli attributi aumenta la distanza tra gli oggetti. 

## 17.12.4 Graph-Based clustering

Abbiamo sempre dei dati in un iper-spazio, utilizziamo delle tecniche basate sui grafi per effettuare la clusterizzazione. La prima tecnica è quella del **clustering gerarchico divisivo**. Abbiamo una serie di oggetti, riportati in figura:

![image](image84.png){width="3.44444in" height="1.3in"}

Costruiamo il grafo completo, ovvero, ogni nodo è collegato attraverso un arco, a tutti gli altri nodi e una volta che abbiamo il grafo **non orientato** completo andiamo a costruire il **minimo albero ricoprente**. A partire dall'albero costruiamo i cluster effettuando una separazione dell'albero andando a prendere l'arco che ha dimensione maggiore. Riassumiamo ora l'algoritmo:

![image](image85.png){width="3.44583in" height="1.2625in"}

Se utilizziamo un algoritmo di costruzione di un albero ricoprente che ha una complessità pari a *O*(*n*2) la complessità totale sarà di *O*(*n*3) perché eseguiamo le di complessità *O*(*n*2) operazioni *n* volte. Arriviamo ad una complessità di *O*(*n*2*logn*) se utilizziamo una struttura ad heap.

Un'altra tecnica è quella dello *Shared Nearest Neighbour* (SNN) che partendo da un insieme di punti assegna a ciascun arco un peso. Il peso che andremo ad assegnare corrisponde al numero di vicini in comune hanno i due nodi come mostrato in figura:

![image](image86.png){width="3.44444in" height="0.97083in"}

L'algoritmo che consideriamo è quello di Jarvis-Patrick i cui passi sono:

1. Prendiamo gli oggetti e costruiamo un grafo in cui tra due nodi *p* e *q* esiste un arco se uno sta nel vicinato dell'altro. Possiamo definire il vicinato come i nodi che si trovano ad una distanza minore di una certa soglia.
2. Diamo pesi agli archi nel modo in cui abbiamo descritto prima, ovvero contando i vicini che hanno in comune.
3. Eliminiamo gli archi che hanno un peso minore di una determinata soglia.
4. Infine, tutte le componenti connesse saranno i nostri cluster.

![image](image87.png){width="3.44444in" height="1.17639in"}

Tale algoritmo funziona bene con cluster di diversa dimensione e densità ma non si può dire altrettanto quando i cluster sono in questo modo (lui dice quando presentano dei collegamenti).

![image](image88.png){width="3.44583in" height="1.32639in"}

### SNN Density Based clustering

Questo algoritmo combina il DBSCAN con l'algoritmo di Jarvis Patrick. I passi dell'algoritmo sono i seguenti:

1. Calcoliamo la matrice di similarità.
2. Costruiamo la matrice sparsa, ovvero un grafo sparso a partire da un potenziale grafo completo.
3. Assegniamo il peso agli archi con la stessa tecnica vista precedentemente.
4. Applichiamo il DBSCAN. Andiamo a misurare per ogni nodo la densità, cioè...

Effettuiamo quindi delle operazioni preliminari per costruire il grafo e dal punto 4 in poi applichiamo il DBSCAN. Di fatto abbiamo due fasi, la prima in cui applichiamo SNN la seconda nella quale applichiamo il DBSCAN. Questo algoritmo funziona molto meglio rispetto all'algoritmo precedente. 
