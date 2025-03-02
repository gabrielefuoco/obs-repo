
## Rilevanza e Incertezza nella Ricerca di Informazioni

La valutazione della rilevanza dei documenti nella ricerca di informazioni è evoluta da un approccio binario (rilevante/non rilevante) a uno probabilistico, a causa dell'incertezza introdotta da fattori latenti nelle query e nella rappresentazione dei documenti.  Il Binary Independence Model (BIM) stima la probabilità di rilevanza basandosi sulla presenza/assenza di termini nel documento, risultando efficace per query semplici.  Per query più complesse, il BIM viene esteso integrando la frequenza dei termini (TF), migliorando l'accuratezza della stima di rilevanza.  L'ipotesi di indipendenza tra i termini nel BIM, sebbene semplificativa, è una limitazione. La pseudo-rilevanza, un'approssimazione semi-automatica, viene utilizzata, ma una stima accurata della rilevanza rimane cruciale. L'introduzione di un prior bayesiano (smoothing bayesiano) aiuta a mitigare l'impatto dell'ipotesi di indipendenza e a gestire l'incertezza.

## Termini Topic-Specific e Eliteness

L'ipotesi di indipendenza tra termini nel recupero delle informazioni crea problemi, soprattutto per i termini meno frequenti ("coda" della distribuzione).  Questo problema è accentuato dai termini "elite", specifici di un determinato argomento (topic-specific), che assumono un'importanza rilevante nel contesto di un documento, anche se potenzialmente presenti in molti altri.  Questi termini non coincidono necessariamente con le Named Entities.

## Approssimazione di Poisson

L'approssimazione di Poisson modella la probabilità di eventi rari.  Nella generazione di documenti, può modellare la probabilità di osservare un termine in una determinata posizione, campionando da una distribuzione multinomiale.  Una regola empirica suggerisce di avere almeno 30, idealmente 50-100 misurazioni per applicare l'approssimazione di Poisson.

---

La distribuzione di Poisson approssima la distribuzione binomiale quando il numero di trial (T) è grande e la probabilità di successo (p) è piccola,  specificamente se K > 20 o 30 e p ≈ 1/K.  Questa approssimazione è particolarmente utile per eventi nella coda della distribuzione, ovvero eventi rari.  La formula di Poisson è derivata dall'approssimazione di  $p^k(1-p)^{t-k}$ con $e^{-Tp}$, ottenendo:

$$p_{\lambda}(k)=\frac{\lambda^k}{k!}e^{-\lambda}$$

dove λ = Tp,  k è il numero di occorrenze, e  media = varianza = λ = cf/T (dove cf è la frequenza cumulativa e T la lunghezza del documento, che deve essere fissa per l'approssimazione).

Tuttavia, questa approssimazione fallisce per i termini "topic specific", caratterizzati da: mancanza di indipendenza tra termini, forte contestualità ("contextual bound"), e occorrenza a gruppi.  Queste caratteristiche violano l'assunzione di indipendenza alla base della distribuzione di Poisson.

Per modellare la probabilità di occorrenza di termini "topic specific", si preferiscono alternative come il modello Okapi BM25 o la distribuzione binomiale negativa. Quest'ultima modella la probabilità di osservare *k* insuccessi prima di *r* successi in una sequenza di trial Bernoulliani indipendenti e identicamente distribuiti (i.i.d.), con formula:

$$NB_{\ r,p}(k)= \begin{pmatrix} k+r-1 \\ k \end{pmatrix}p^r(1-p)^k$$

dove *r* è il parametro di dispersione (numero di successi), *k* il numero di insuccessi, e *p* la probabilità di successo.  A seconda dell'interpretazione, *k* e *r* possono scambiare i loro ruoli.

---

Il testo descrive un modello per il recupero dell'informazione che incorpora il concetto di "termine elite" per migliorare la precisione del ranking dei documenti.

**1. Parametrizzazione della Binomiale Negativa:**  La distribuzione binomiale negativa, con parametri di dispersione *r* e probabilità di successo *p*, ha media  $\mu=\frac{rp}{1-p}$.  Il testo riformula la distribuzione come $NB_{\ r,p}(k)= \begin{pmatrix} k+r-1 \\ k \end{pmatrix}\left( \frac{r}{\mu+r} \right)^r\left( \frac{\mu}{\mu+r} \right)^k$, dove *k* rappresenta il numero di fallimenti prima del successo *r*-esimo, e *k+r* il numero totale di trial.

**2. Termini Elite:** Un "termine elite" è un sostantivo indicativo di un topic specifico, la cui rilevanza è binaria (elite o non elite) e dipende dalla sua *term frequency* *solo* nei documenti rilevanti per quel topic.

**3. Modellazione dell'Eliteness:** L'eliteness di un termine *i* ($E_i$) è modellata come una variabile nascosta per ogni coppia documento-termine.  Un termine è elite se il documento tratta il concetto da esso denotato.  L'analisi considera i pattern distribuzionali dei termini elite, ma il problema del "topic drift" (un documento rilevante per un topic contiene termini elite di altri topics) viene evidenziato come potenziale fonte di errore.

**4. Retrieval Status Value (RSV) con termini Elite:** Il RSV viene modificato per includere l'informazione sull'eliteness. La formula proposta è: $RSV^{elite}=\sum_{i\in q}c_{i^{elite}}(tf_{i})$, dove $c_{i}^{elite}(tf_{i})=\log \frac{p(TF_{i}=tf_{i}|R=1)p(TF_{i}=0|R=0)}{p(TF_{i}=0|R=1)p(TF_{i}=tf_{i}|R=0)}$ e $p(TF_{i}=tf_{i}|R)=p(TF_{i}=tf_{i}|E=elite)p(E=elite|R)+p(TF_{i}=tf_{i}|E=elite)(1-p(E_{i}=elite|R))$.  Questa formula considera la probabilità di osservare una specifica *term frequency* ($tf_i$) dato che il documento è rilevante (R=1) o non rilevante (R=0), tenendo conto della probabilità che il termine sia elite ($E_i=elite$) o meno.  In sostanza, il costo di un termine è ponderato in base alla sua probabilità di essere elite, dato il contesto di rilevanza del documento.

---

## Riassunto del Modello BM25 e sue Varianti

Questo documento descrive il modello BM25 e le sue evoluzioni, partendo da modelli più semplici basati sulla distribuzione di Poisson.

### Modelli di Poisson

Inizialmente, si considera un modello a singola distribuzione di Poisson per la *term frequency* (TF).  La sua inadeguatezza porta all'introduzione di un **modello a due Poisson**, che distingue tra termini "elite" (alta frequenza in documenti rilevanti) e termini non-elite. La probabilità di osservare una TF pari a *k*, dato uno stato di rilevanza *R*, è:

$$p(TF_{i}=k|R)=\pi \frac{\lambda^k}{k!}e^{-\lambda}+(1-\pi) \frac{\mu^k}{k! }e^{-\mu}$$

dove π è la probabilità che un termine sia "elite", e λ e μ sono i tassi delle due distribuzioni di Poisson.  La stima dei parametri è complessa, quindi si preferisce un'approssimazione con una curva parametrica che mantiene le proprietà qualitative del modello a due Poisson.  Il costo per i termini "elite",  $c_i^{elite}(tf_i)$, cresce monotonicamente con la TF, saturando per alti valori di λ (vedi figura `![[]]`).  Una semplice approssimazione di questa funzione è:  $\frac{tf}{k_1 + tf}$, dove un valore alto di $k_1$ implica un contributo significativo anche per alte TF.


### Modelli BM25

Il modello BM25 si basa su queste considerazioni.

#### BM25 Versione 1:

Utilizza una funzione di saturazione per il costo del termine i-esimo:

$$c_{i}^{BM25v_{1}}(tf_{i})=c_{i}^{BIM} \frac{tf_{i}}{k_{1}+tf_{i}}$$

dove $c_i^{BIM}$ è il costo calcolato con un modello BIM (non specificato nel dettaglio).

#### BM25 Versione 2:

Semplifica il modello BIM usando solo l'IDF (Inverse Document Frequency):

$$c_{i}^{BM25v_{2}}(tf_{i})=\log \frac{N}{df_{i}}\times \frac{(k_{1}+1)tf_{i}}{k_{1}+tf_{i}}$$

dove N è il numero totale di documenti e $df_i$ è la frequenza del termine *i*-esimo.

### Estensioni del BM25

Il modello BM25 viene ulteriormente esteso:

* **Prima estensione:** Aggiunge un fattore di smoothing basato sulla funzione di saturazione al costo $c_i$ del BM25.
* **Seconda estensione:** Utilizza solo la stima di $r_i$ (probabilità di rilevanza dato il termine) e $df_i$, senza il costo $c_i$ completo.

Infine, si introduce la **normalizzazione della lunghezza del documento** per mitigare l'influenza della lunghezza variabile dei documenti sul valore di $tf_i$.

---

Il modello Okapi BM25 è un algoritmo di ranking per la ricerca di informazioni che estende il modello BIM, migliorandolo tramite la normalizzazione della *term frequency* (tf) in base alla lunghezza del documento.  La lunghezza del documento (*dl*) è la somma delle tf di tutti i termini, mentre *avdl* è la lunghezza media dei documenti nella collezione.  Documenti più lunghi possono avere tf più alte a causa di verbosità o di un ambito più ampio.

La normalizzazione della lunghezza del documento è gestita dal parametro *B*:

$$B=\left( (1-b)+b \frac{dl}{avdl} \right), 0\leq b\leq1$$

dove *b = 1* indica normalizzazione completa e *b = 0* nessuna normalizzazione.  La tf normalizzata è:  `tf_i' = tf_i / B`.

La formula del punteggio BM25 per un termine *i* è:

$$c_i^{BM25}(tf_i) = \log \frac{N}{df_i} \times \frac{(k_1+1)tf_i'}{k_1+tf_i'} = \log \frac{N}{df_i} \times \frac{(k_1+1)tf_i}{k_1((1-b)+b\frac{dl}{avdl})+tf_i}$$

dove:

* `N`: numero totale di documenti.
* `df_i`: numero di documenti contenenti il termine *i*.
* `k_1` e `b`: parametri del modello.
* `dl`: lunghezza del documento.
* `avdl`: lunghezza media dei documenti.

Il punteggio BM25 finale è la somma dei punteggi per ogni termine nella query:

$$RSV^{BM25} = \sum_{i \in q} c_i^{BM25}(tf_i)$$

Il modello utilizza due parametri principali:

* `k_1`: controlla la pendenza della funzione di saturazione della tf (valori tipici: 1.2-2).  `k_1 = 0` dà un modello binario, mentre un `k_1` grande usa la tf grezza.
* `b`: controlla la normalizzazione della lunghezza del documento (valori tipici: 0.75). `b = 0` disattiva la normalizzazione, `b = 1` la applica completamente.

Un esempio confronta il punteggio tf-idf e BM25 per una query "machine learning" su due documenti con diverse tf.  Infine, il concetto di "zone" (sezioni specifiche di un documento, come titolo o abstract) viene introdotto, suggerendo una possibile estensione del modello.

---

Questo documento descrive un metodo per migliorare il ranking dei documenti utilizzando il modello BM25, considerando la struttura zonale dei documenti (es. titolo, abstract, corpo).

Due approcci iniziali vengono considerati:  combinare i punteggi BM25 di ogni zona con una combinazione lineare ponderata (idea semplice, ritenuta irragionevole per l'indipendenza assunta tra le zone) oppure combinare le prove tra le zone per ogni termine, e poi tra i termini (idea alternativa, preferita).

Il metodo proposto calcola varianti pesate della frequenza dei termini (`tf`) e della lunghezza del documento (`dl`):

$\tilde{t}f_{i} = \sum_{z=1}^{Z} v_{z} tf_{zi}$  e  $\tilde{dl} = \sum_{z=1}^{Z} v_{z} len_{z}$

dove  `v<sub>z</sub>` è il peso della zona `z`, `tf<sub>zi</sub>` è la frequenza del termine `i` nella zona `z`, `len<sub>z</sub>` è la lunghezza della zona `z`, e `Z` è il numero di zone.  Si calcola anche la lunghezza media del documento ponderata (`avdl`).

Il calcolo delle varianti pesate avviene in tre fasi: 1) calcolo della TF per zona; 2) normalizzazione per zona; 3) assegnazione di un peso `v<sub>z</sub>` (predefinito e non apprendibile) ad ogni zona.

Un modello `Simple BM25F` viene presentato, con una formula RSV (Retrieval Status Value) semplificata:

$RSV^{SimpleBM25F} = \sum_{i \in q} \log \frac{N}{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_1((1-b) + b \frac{dl}{avdl}) + tf_i}$

Successivamente, viene introdotta una normalizzazione della lunghezza specifica per zona, migliorando il modello `BM25F`. La frequenza del termine modificata è:

$\tilde{tf}_i = \sum_{z=1}^Z v_z \frac{f_{zi}}{B_z}$  dove  $B_z = \left( (1-b_z) + b_z \frac{\text{len}_z}{\text{avlen}_z} \right)$

La formula RSV per questo modello `BM25F` migliorato è:

$\text{RSV}^{BM25F} = \sum_{i \in q} \log \frac{N }{df_{i}} \cdot \frac{(k_1 + 1)tf_i}{k_{1}+tf_{i}}$

Infine, il documento accenna all'integrazione di features non testuali, assumendo l'indipendenza tra le features non testuali e quelle testuali, e l'indipendenza delle informazioni di rilevanza dalla query.  Questa assunzione permette di utilizzare un approccio di tipo BIM per combinare le features.

---

Il Ranking Score Value (RSV) è calcolato come:

$$RSV=\sum_{i\in q}c_{i}(tf_{i})+\sum_{j=1}^f\lambda_{j}V_{j}(f_{j})$$

dove:

*  `cᵢ(tfᵢ)` rappresenta il contributo del termine `i` (presente nella query `q`) basato sulla sua frequenza nel documento.
*  `Vⱼ(fⱼ) = log[p(Fⱼ=fⱼ|R=1)/p(Fⱼ=fⱼ|R=0)]`  rappresenta il valore informativo della feature `fⱼ`, calcolato come il logaritmo del rapporto tra la probabilità di osservare `fⱼ` nei documenti rilevanti (R=1) e la probabilità di osservarlo nei documenti non rilevanti (R=0).
*  `λⱼ` è un parametro di scala aggiunto per compensare le approssimazioni nel modello.

La scelta appropriata di `Vⱼ` per ogni feature `fⱼ` è cruciale per l'efficacia del RSV.  La performance di formule come `Rsv_{bm25} + log(pagerank)` può essere spiegata proprio dalla scelta accurata di `Vⱼ`.

---
