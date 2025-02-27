## Lo Stochastic Topic Modeling e la Generazione di Testo

Lo *Stochastic Topic Modeling* non è propriamente *text generation* nel senso in cui lo intendiamo per i *language model*. Non dobbiamo prevedere il prossimo documento o la prossima parola, né costruire testi. Tuttavia, è comunque un processo generativo perché, sulla base dei dati osservati e del training, sappiamo come caratterizzare un documento come una composizione di distribuzioni di probabilità. Ciascuna di queste distribuzioni rappresenta a sua volta una distribuzione di probabilità sullo spazio delle parole.

Il processo generativo è legato al doppio campionamento:

1. **Campionamento del topic:** Si sceglie un topic da una distribuzione di probabilità sui topic.
2. **Campionamento delle parole:** Per ogni topic, si campionano le parole da una distribuzione di probabilità sulle parole, specifica per quel topic.
3. **Campionamento dei topic per il documento:** Per ogni documento, si campionano i topic da una distribuzione di probabilità sui topic.

## Elaborazione del linguaggio naturale (NLP)

L'elaborazione del linguaggio naturale (NLP) è un campo che si concentra sullo sviluppo di sistemi automatici in grado di comprendere e generare linguaggi naturali. Il linguaggio naturale, unico e specifico per gli umani, è fondamentale per la comunicazione complessa e l'interazione con altre modalità, come la visione.

L'obiettivo è sviluppare sistemi di intelligenza artificiale (AI) che possiedano una capacità di apprendimento simile a quella dei bambini umani.

### Il problema della rappresentazione:

Un problema fondamentale nell'NLP è la rappresentazione del linguaggio in un computer in modo che possa essere elaborato e generato in modo robusto. Una sotto-domanda cruciale è come rappresentare le parole.

**Rappresentazione del significato di una parola:**

Il significato di una parola è l'idea che essa rappresenta. Un approccio linguistico comune al significato è la semantica denotativa, che associa un significante (simbolo) a un significato (idea o cosa). Ad esempio, la parola "albero" denota diverse rappresentazioni di alberi, come 🌲, 🌳, 🪴. 

L'obiettivo è far corrispondere l'unità lessicale a un concetto. 

## Problemi di WordNet

WordNet, pur essendo una risorsa preziosa per l'elaborazione del linguaggio naturale, presenta alcuni limiti:

* **Mancanza di sfumature:** WordNet non riesce a catturare tutte le sfumature di significato di una parola. Ad esempio, non è in grado di spiegare i diversi significati di una parola in base al contesto in cui viene utilizzata. 
* **Contesto:** WordNet non è in grado di considerare il contesto in cui una parola viene utilizzata. Ad esempio, "proficient" è un sinonimo di "good", ma solo in alcuni contesti.
* **Informazioni quantitative:** WordNet non fornisce informazioni quantitative per misurare l'appropriatezza di una parola in un determinato contesto.
* **Mantenimento:** Il linguaggio evolve costantemente, quindi WordNet dovrebbe essere aggiornata continuamente. Il mantenimento è un compito dispendioso in termini di tempo e viene svolto solo periodicamente.
* **Soggettività:** WordNet è una risorsa curata manualmente, quindi è soggetta a bias culturali. Ad esempio, WordNet è un progetto britannico, quindi potrebbe riflettere una prospettiva culturale specifica.
* **Similarità:** WordNet non può essere utilizzato per calcolare accuratamente la similarità tra parole.

### Rappresentazione Semantica delle Parole

È possibile calcolare la similarità semantica tra due parole utilizzando tecniche di *embedding*. Questo processo consiste nel rappresentare le parole in uno spazio multidimensionale, dove ogni dimensione corrisponde a un aspetto del significato della parola. 

**Principio chiave:** Parole sinonime, con significati simili, dovrebbero essere rappresentate da punti vicini nello spazio multidimensionale.

**Metodo comune:** Un metodo semplice per ottenere questa rappresentazione è basato sulle *co-occorrenze*. Si analizza la frequenza con cui le parole compaiono insieme in un corpus di testo. Maggiore è la frequenza di co-occorrenza, più simili sono i significati delle parole.

## Rappresentazione delle parole come simboli discreti

* **Tipo di parola:** un elemento di un vocabolario finito, indipendente dall'osservazione effettiva della parola nel contesto (entry nel vocabolario).
* **Token di parola:** un'istanza del tipo, ad esempio, osservata in un determinato contesto (occorrenza). 

##### Rappresentazione Localista delle Parole in NLP Tradizionale

- Le parole sono considerate simboli discreti (es. *hotel*, *conference*, *motel*).
- Vengono rappresentate tramite **vettori one-hot**:
 - Solo una posizione ha valore "1", tutte le altre sono "0".
 - Esempio:
 - *motel* = $[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]$
 - *hotel* = $[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$
- La dimensione del vettore corrisponde al numero di parole nel vocabolario (es. 500.000+).

Questo metodo presenta alcuni svantaggi:

- **Mancanza di similarità:** Se l'obiettivo è quello di rappresentare gli elementi in uno spazio comune, in cui se c'è similitudine tra due parole, allora questi devono essere vicini, questo metodo non funziona.
- **Dipendenza dal vocabolario:** La rappresentazione è fortemente dipendente dal vocabolario utilizzato.

Potremmo affidarci a WordNet per migliorare questa codifica espandendo il numero di incidenze usando i sinonimi, ma è una soluzione poco accurata. 

# Word Embedding

Il *Word Embedding* non è un concetto nuovo, ma deriva dall'algebra lineare (LSA - Latent Semantic Analysis). L'LSA permette di ottenere una trasformazione che, oltre a cogliere una serie di relazioni semantiche, comporta lo spostamento in uno spazio a minore dimensionalità. Questo spazio può essere utilizzato per confrontare tra loro vettori documento o vettori termine.

Le codifiche di una LSA per ogni termine riguardano una rappresentazione dei contributi informativi che quel termine fornisce a ogni documento della collezione.

Il *Word Embedding* si usa per far sì che questa trasformazione goda di proprietà di prossimità. In altre parole, parole con significati simili dovrebbero essere rappresentate da vettori vicini nello spazio vettoriale. 

### Distributional Semantics

La **semantica distribuzionale** si basa sul principio che il significato di una parola è determinato dalle parole che frequentemente appaiono nelle sue vicinanze. 

Questa semantica è catturata attraverso **pattern di co-occorrenza**. In altre parole, il significato di una parola nel contesto d'uso di un particolare linguaggio è determinato dalle parole che appaiono vicine a quella parola (parole di contesto) e che si verificano frequentemente.

Quando una parola *w* appare in un testo, il suo **contesto** è l'insieme delle parole che appaiono nelle sue vicinanze (entro una finestra di dimensione fissa). 

Utilizzando i molti contesti di *w*, è possibile costruire una rappresentazione di *w*. 

## Matrice di co-occorrenza a livello di documento

La matrice di co-occorrenza a livello di documento rappresenta una parola tramite la distribuzione delle parole con cui appare. L'idea di base è la seguente:

1. **Determinare un vocabolario V:** Definisci l'insieme di tutte le parole che saranno considerate nella matrice.
2. **Creare una matrice di dimensione |V| × |V|:** Crea una matrice quadrata dove le righe e le colonne corrispondono alle parole del vocabolario. Inizialmente, tutti i valori della matrice sono impostati a zero.
3. **Contare le co-occorrenze:** Per ogni documento, per ogni parola *w* nel documento, incrementa il conteggio nella cella della matrice corrispondente alla riga di *w* e alla colonna di ogni altra parola *w'* presente nel documento.
4. **Normalizzare le righe:** Dividi ogni valore di una riga per la somma dei valori della riga stessa, per renderli indipendeti dalla lunghezza del documento.
Il risultato è una rappresentazione sparsa ed è ancora troppo costosa.

### Problemi con la co-occorrenza a livello di documento

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241111154837159.png|532]]
La co-occorrenza a livello di documento presenta alcuni problemi:

* **Nozioni ampie di co-occorrenza:** Se si utilizzano finestre o documenti di grandi dimensioni, la matrice di co-occorrenza può catturare informazioni semantiche o di argomento, ma perdere informazioni di livello sintattico. 
* **Finestre brevi:** Quanto deve essere esteso il contesto? Se si utilizzano finestre di piccole dimensioni, la matrice di co-occorrenza può catturare informazioni di livello **sintattico**, ma perdere informazioni **semantiche** o di argomento. 
* **Conteggi grezzi:** I conteggi grezzi delle parole tendono a sovrastimare l'importanza delle parole molto comuni.
* **Logaritmo della frequenza:** Utilizzare il logaritmo della frequenza dei conteggi è più utile per mitigare l'effetto delle parole comuni (prima di normalizzare).
* **GloVe:** Un'alternativa ancora migliore è l'utilizzo di GloVe (Global Vectors for Word Representation), un modello che apprende le rappresentazioni delle parole in base alle loro co-occorrenze in un corpus di testo. Ne parleremo più avanti. 

## Vettori di parole

Tenendo conto degli aspetti chiave, ovvero principio della località per determinare i vari concetti di ogni parola, l'informazione di co-occorrenza, etc, vogliamo costruiremo un vettore denso per ogni parola, scelto in modo che sia simile ai vettori di parole che appaiono in contesti simili. Misureremo la similarità come il prodotto scalare (vettoriale) dei vettori.
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241111160315051.png|475]]
Nota: i vettori di parole sono anche chiamati *embedding* (di parole) o rappresentazioni (neurali) di parole. Sono una rappresentazione distribuita. 
Molte di queste rappresentazioni sono neurali.

## Word2vec: Panoramica

Word2vec (Mikolov et al. 2013) è un framework per l'apprendimento di vettori di parole.

**Idea:**

* Abbiamo un grande corpus ("corpo") di testo: una lunga lista di parole.
* Ogni parola in un vocabolario fisso è rappresentata da un vettore.
* Attraversiamo ogni *posizione* **t** nel testo, che ha una parola *centrale* **c** e parole di *contesto* ("esterne") **o**.
* Utilizziamo la similarità dei vettori di parole per **c** e **o** per calcolare la probabilità di **o** dato **c** (o viceversa).
	* Probabilità che la parola di contesto co-occorra rispetto la centrale o viceversa.
	* Data una parola target predire la probabilità rispetto le sue parole di contesto, avendo fissato l'estensione del contesto (quante parole devo considerare come contesto per ogni parola target).
* Continuiamo ad aggiustare i vettori di parole per massimizzare questa probabilità. 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241111162802037.png]]
Fissiamo estensione del contesto a 2, consideriamo "into" come parola centrale, dobbiamo calcolare la probabilità delle due parole a destra e a sinistra
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241111162829597.png|614]]
## Word2vec: Funzione Obiettivo

Per ogni posizione $t=1,\dots,T$, vogliamo predire le parole di contesto in una certa finestra di taglia fissata **m**, data una parola centrale $w_{t}$. 

La funzione di verosimiglianza è definita come:

$$\text{Likelihood}=L_{0}=\prod_{t=1}^T \prod_{-m\leq j\leq m}P(w_{t+j|w_{t};\theta})$$

dove $P(w_{t+j|w_{t};\theta})$ rappresenta la probabilità della parola di contesto shiftata di $j$ rispetto a $t$, dato $w_{t}$.

Il nostro obiettivo è minimizzare la log-verosimiglianza.

Per calcolare queste probabilità condizionate, utilizziamo due vettori distinti per ogni parola, a seconda del suo ruolo: parola target o parola di contesto. Nell'esempio precedente, "INTO" è la parola centrale all'iterazione *t*, ma poi diventa parola di contesto. Ogni parola può apparire più volte come target o come contesto. 

### Minimizzazione della Funzione Obiettivo

Si vuole minimizzare la seguente funzione obiettivo:

$$J(θ) = -\frac{1}{T} \sum_{t=1}^T\sum_{j≠0}\log P(W_{t+j} | W_{t}; θ)$$
Minimizzare la funzione obiettivo implica massimizzare l'accuratezza di predizione

Questa funzione è comunemente utilizzata in modelli linguistici per misurare la qualità di una rappresentazione distribuzionale delle parole.

### Calcolo della Probabilità Condizionata

Per calcolare la probabilità condizionata $P(W_{t+j} | W_{t}; θ)$, si utilizzano due vettori per ogni parola `w`:

- **Vw:** quando `w` è una parola centrale (target word)
- **Uw:** quando `w` è una parola di contesto (context word)

Per una parola centrale `c` e una parola di contesto `o`, la probabilità condizionata viene calcolata come segue:

$$P(o|c) = \frac{\exp(u_o^T v_c)}{Σ(w∈V) \exp(u_w^T v_c)}$$

In sostanza, utilizziamo due vettori per ogni parola, in funzione del ruolo che ha la parola: target o contesto. Per calcolare queste probabilità, è necessario distinguere i due vettori di parole, poiché ogni parola può apparire una o più volte come target o come contesto.

Data una parola di contesto, la probabilità di osservarla data l'osservazione di una parola centrale viene espressa con una softmax. Con `u` indichiamo i vettori di contesto e con `v` i vettori quando assumono il ruolo di target. 

## Word2vec: Algoritmo di apprendimento delle rappresentazioni delle parole

Seleziona iterativamente una parola centrale e la sua finestra di dimensione fissa *m* da un documento di lunghezza *T*, ed estrai esempi di addestramento.

* Addestra una rete neurale con 1 strato nascosto di dimensione *N*, dove:
 * Gli strati di input/output sono vettori one-hot di dimensione *V*, ovvero la dimensione del vocabolario.
 * Lo strato nascosto è di dimensione *N*, con *N* $\ll$ *V*.

* Compito di apprendimento:
 * "Dato una parola specifica all'interno di una frase (parola centrale), scegliendo casualmente una parola nella finestra, restituisci una probabilità per ogni parola nel vocabolario di essere effettivamente la parola scelta casualmente".

* Utilizza l'ottimizzatore SGD per aggiornare i parametri del modello. 
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112100823370.png]]

Dato un insieme di parole di contesto, vogliamo predire la parola target. Per fare ciò, addestriamo una rete neurale con un solo layer nascosto, la cui dimensione è molto minore del vocabolario ($v$). I parametri della rete vengono ottimizzati tramite la discesa del gradiente stocastica.

La funzione di costo è definita come:

$$j(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m}\log P(w_{t+j}|w_{t};\theta)$$

dove $P(w_{t+j}|w_{t};\theta)=P(o|c)$ rappresenta la probabilità della parola target ($o$) dato il contesto ($c$).

La probabilità $P(o|c)$ è calcolata tramite la softmax:

$$P(o|c)=\frac{e^{U_{0}^TV_c}}{\sum_{w\in W }e^{U_{0}^TV_c}}$$

Calcoliamo il gradiente della funzione di costo rispetto al vettore del contesto $V_c$:

$$\frac{\delta J}{\delta V_{c}}=\frac{\delta}{\delta V_{c}}\log e^{U_{0}^TV_c}-\log \sum_{w\in V} e^{U_{0}^TV_c}$$

Il primo termine è semplicemente $U_{0}$. Il secondo termine diventa:
# separare

$$=\frac{1}{\sum_{w\in V} e^{U_{0}^TV_c}} \frac{\delta}{\delta V_{c}}\sum_{x\in V} e^{U_{x}^TV_c}=\frac{1}{\sum_{w \in V} e^{U_{w}^TV_c}}\sum_{x\in V} \frac{\delta}{\delta v_{c}}e^{U_{w}^TV_c}=\frac{1}{\sum_{w \in V} e^{U_{w}^TV_c}}\sum_{x\in V} e^{U_{w}^TV_c} u_{x}=\sum \frac{e^{U_{w}^TV_c} }{\sum_{x\in V} e^{U_{w}^TV_c} } u_{x}$$

Quindi, il gradiente della log-probabilità rispetto a $V_c$ è:

$$\frac{\delta}{\delta V_{c}}\log p(o|c)=u_{o}-\sum_{x\in V}p(x|c)v_{x}$$

dove $p(x|c)$ rappresenta la probabilità della parola $x$ dato il contesto $c$, calcolata tramite la softmax.

## Derivazione delle Regole di Aggiornamento per la Regressione Logistica

### Definizione della Funzione di Perdita

La funzione di perdita utilizzata per la regressione logistica è la **cross-entropy**, definita come:

$$L = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$

dove:

* $y$ è il valore reale (0 o 1)
* $\hat{y}$ è la probabilità prevista (tra 0 e 1)

### Calcolo del Gradiente

Per aggiornare i parametri, dobbiamo calcolare il gradiente della funzione di perdita rispetto a ciascun parametro.

#### Gradiente rispetto a $z$

Il gradiente rispetto a $z$ è dato da:

$$\frac{\delta L}{\delta z} = \frac{\delta L}{\delta \hat{y}} \frac{\delta \hat{y}}{\delta z}$$

Calcoliamo i due termini separatamente:

* **Primo termine:** $\frac{\delta L}{\delta \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$
* **Secondo termine:** $\frac{\delta \hat{y}}{\delta z} = \frac{1}{(1+e^{-z})^2}e^{-z} = \hat{y}(1-\hat{y})$

Moltiplicando i due termini, otteniamo:

$$\frac{\delta L}{\delta z} = -y(1-\hat{y}) + \hat{y}(1-y) = \hat{y} - y$$

#### Gradiente rispetto a $w$

Il gradiente rispetto a $w$ è dato da:

$$\frac{\delta L}{\delta w} = \frac{\delta L}{\delta z} \frac{\delta z}{\delta w} = (\hat{y}-y)x$$

dove $x$ è il valore dell'input.

#### Gradiente rispetto a $b$

Il gradiente rispetto a $b$ è dato da:

$$\frac{\delta L}{\delta b} = \frac{\delta L}{\delta z} \frac{\delta z}{\delta b} = \hat{y} - y$$

### Regole di Aggiornamento

Utilizzando il metodo della discesa del gradiente, aggiorniamo i parametri $w$ e $b$ come segue:

* $w^{(k+1)} = w^{(k)} - \lambda \frac{\delta L}{\delta w^{(k)}}$
* $b^{(k+1)} = b^{(k)} - \lambda \frac{\delta L}{\delta b^{(k)}}$

dove:

* $k$ è l'iterazione corrente
* $\lambda$ è la learning rate

Sostituendo i gradienti calcolati, otteniamo le seguenti regole di aggiornamento:

* $w^{(k+1)} = w^{(k)} - \lambda (\hat{y} - y)x$
* $b^{(k+1)} = b^{(k)} - \lambda (\hat{y} - y)$

Queste regole vengono utilizzate per aggiornare i parametri del modello di regressione logistica durante l'addestramento.

## Addestramento del Modello

Per addestrare il modello, si ottimizzano i valori dei parametri per minimizzare la perdita.

L'addestramento di un modello consiste nell'aggiustare gradualmente i parametri per minimizzare una funzione di perdita.

Ricordiamo che $O$ rappresenta tutti i parametri del modello, in un unico lungo vettore.

Nel nostro caso, con vettori di dimensione $d$ e $V$ parole, abbiamo:

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112100203656.png]]

Ricordiamo che ogni parola ha due vettori:

* $w_i$ (vettore di parole)
* $b_i$ (vettore di contesto)

Ottimizziamo questi parametri muovendoci lungo il gradiente.

## Famiglia di algoritmi Word2vec

Gli autori Mikolov e altri hanno proposto due varianti di embeddings: **Skip-gram** e **Continuous Bag-of-Words (CBOW)**.

**Perché due vettori?** - Ottimizzazione più semplice. Si fa la media di entrambi alla fine.
Ma si può implementare l'algoritmo con un solo vettore per parola... e aiuta un po'.

**Due varianti:**

1. **Skip-grams (SG):**
- In Skip-gram, data una parola target, l'obiettivo è predire le parole di contesto scorrendo il testo e calcolando le probabilità. Si definisce una **window size**, un iperparametro che determina l'estensione del contesto. Per ogni parola target, Skip-gram calcola le probabilità per predire le parole di contesto, ovvero le parole precedenti e successive alla parola target. Se la window size è n, si considerano le n parole precedenti e le n parole successive.

2. **Continuos bag of words (CBOW):**
- CBOW è una variante di Skip-gram in cui, date le parole di contesto, si deve predire la parola centrale. In altre parole, l'input è il contesto e l'output è la parola target.

**Visualizzazione:**

* **Skip-gram:** Si muove sul testo, la parola centrale è evidenziata e il contesto sono le altre parole. Questo processo genera le coppie di addestramento. L'associazione è uno a molti: una parola centrale e M parole di contesto. Le parole di contesto vengono spesso aggregate, mediate, quindi dato il vettore medio di contesto, si deve predire la parola centrale.
* **CBOW:** L'associazione è uno a uno. Si ha un input di M parole di contesto e un output di una parola centrale.

**Implementazione:**

* **Skip-gram:**
 * Si parte da un input one-hot che rappresenta la parola target.
 * Si ha un hidden layer di dimensionalità D (chiamato anche N).
 * La matrice dei pesi tra input layer e hidden layer è la matrice di embedding, mentre tra hidden layer e output layer è la matrice di contesto.
 * Si calcola la probabilità per ogni parola di contesto.

* **CBOW:**
 * Si parte da un input one-hot per ogni parola di contesto.
 * Si ha un hidden layer e un output layer.
 * Si calcola la media dei vettori delle parole di contesto.
 * La dimensione dell'hidden layer è N o D, la dimensione dello spazio di trasformazione.
 * Si codifica ogni parola di contesto in uno spazio di dimensione N o D.
 * Si calcola la media delle codifiche delle parole di contesto.
 * Si decodifica il vettore medio con la matrice dei pesi tra hidden layer e output layer, che è collegato al softmax.

## Esempio in CBOW

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112101025711.png]]

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112101035268.png]]

specifichiamo il one hot, abbiamo l'outoput layer con la parola da predire
dobbiamo apprendere i pesi di collegamento tra input e outout (sull hiddenb layer)

cbow fa una media tra i vettori tra le parole di contesto
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112101140496.png]]

Dato un contesto di parole, ad esempio "cat" e "on", l'obiettivo è predire la parola centrale "sat".

**Rappresentazione:**

* La dimensione della rappresentazione delle parole è pari alla dimensione del vocabolario.
* **w** è la matrice dei pesi tra lo strato di input e lo strato nascosto, di dimensione $d \cdot n$, dove *n* è la dimensionalità dello strato nascosto.
* La codifica di "cat" e "on" è ottenuta moltiplicando il loro vettore one-hot per la matrice dei pesi **w**.

**Codifica del contesto:**

* **V** è una codifica generica in input.
* Per le parole di contesto, si ottiene una media delle loro codifiche.
* Questo vettore medio viene decodificato con la matrice dei pesi tra lo strato nascosto e lo strato di output e poi dato in input alla softmax.

**Nota:**

* $\hat{v}$ è la media delle codifiche degli *m* input, che per CBOW sono solo le parole di contesto. 

### Estrazione di Embeddings

L'embedding di una parola è ricavabile da una riga della matrice **W**.

**In Skip-gram:**

* L'input è una singola parola.
* L'embedding della parola viene estratto dalla matrice **W** e rappresenta la rappresentazione della parola target nel task di Skip-gram.

**In CBOW:**

* L'input è costituito dalle parole di contesto.
* Gli embedding delle parole di contesto vengono aggregati per ottenere una rappresentazione dell'input.
* È possibile estrarre gli embedding delle parole di contesto sia dalla matrice **W** che dalla matrice **W'**.
* Se si estraggono gli embedding dalla matrice **W**, si ottiene la rappresentazione delle parole di contesto come parole target nel task di Skip-gram.
* Se si estraggono gli embedding dalla matrice **W'**, si ottiene la rappresentazione delle parole di contesto come parole di contesto nel task di CBOW.

Per entrambi i task, se andiamo a prendere gli embeddings nella matrice di destra, significherebbe catturare la rappresentazione della parola come funzione che va a modellare il ruolo della parola per la predizione

### Softmax e Funzione di Costo in Word2Vec

La probabilità di una parola data un'altra è calcolata tramite la funzione softmax. Per calcolare questa probabilità, dobbiamo confrontare la parola centrale con tutte le altre parole del vocabolario. 

Per calcolare la probabilità di una parola di contesto data una parola centrale, dobbiamo considerare il numeratore (che vogliamo massimizzare) e il denominatore (che vogliamo minimizzare). Questo approccio è leggermente più efficiente perché dobbiamo considerare solo la parola centrale.

## Skip-gram con Negative Sampling (HW2)

La normalizzazione del termine nella softmax si presenta computazionalmente costosa, soprattutto con vocabolari estesi. Generalizzando la regressione logistica a più classi (da classificazione binaria a multiclasse) si utilizza una softmax, dove al denominatore la maggior parte delle parole contribuisce come rumore.

Una soluzione a questo problema è adottare il *negative sampling*, che trasforma la *average log likelihood* come segue:

$$J_{t}(\theta)=\log\sigma(u_{o}^Tv_{c})+\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$$

Questa formula presenta due termini additivi:

* Il primo, $\log\sigma(u_{o}^Tv_{c})$, rappresenta il logaritmo della sigmoide applicata al prodotto scalare tra il vettore della parola centrale ($v_c$) e il vettore della parola di output/contesto osservata ($u_o$).
* Il secondo termine, $\sum_{i=1}^k E_{P(W)}[\log\sigma(-u_{j}^Tv_{c})]$, approssima il termine al denominatore della softmax. Anziché considerare il confronto tra la parola centrale e tutte le parole del vocabolario, si effettua un confronto con *k* parole negative campionate secondo una distribuzione $P(W)$.

Data una parola negativa *j*-esima tra le *k* campionate, si moltiplica per il negativo del prodotto scalare tra il suo vettore ($u_j$) e quello della parola centrale ($v_c$). Questo perché si vuole massimizzare la differenza tra l'accoppiamento desiderato (tra la parola centrale e una vera parola di contesto) e l'accoppiamento con le parole negative (rumore). L'obiettivo è:

* Avvicinare il più possibile le vere parole di contesto alla parola centrale.
* Allontanare le parole rumorose (negative) dalla parola centrale.

Questo è un approccio di apprendimento *discriminativo* (o contrastivo). Si apprendono delle prossimità facendo in modo che istanze con proprietà desiderabili (parola centrale e contesto) siano il più vicine possibili, mentre istanze di classi diverse (parola centrale e parole negative) siano tenute più lontane possibile.

In pratica, vogliamo:

* **Massimizzare la similarità** tra la parola centrale e le parole di contesto.
* **Massimizzare la distanza** tra la parola centrale e altre parole nel vocabolario.

Le parole negative vengono campionate da una distribuzione unigramma elevata a una potenza $\alpha$ (tipicamente $\alpha \approx 0.75$). La distribuzione unigramma modella la frequenza di occorrenza di una parola normalizzata, in modo che la somma delle frequenze sia uguale a 1. Elevare la distribuzione unigramma a una potenza inferiore a 1 riduce l'enfasi sul campionamento di parole molto frequenti, che altrimenti dominerebbero il processo di negative sampling.

Campionare le *k* parole negative con una distribuzione proporzionale alla distribuzione delle occorrenze enfatizzerebbe eccessivamente le parole più presenti nel testo, il che non è desiderabile.

### Softmax Gerarchica

La softmax gerarchica è un'altra tecnica che migliora l'efficienza dell'addestramento dei modelli di embedding. Invece di utilizzare una softmax standard, che richiede il calcolo della probabilità di tutte le parole nel vocabolario, la softmax gerarchica utilizza un albero di Huffman per organizzare le parole.

**Come funziona la softmax gerarchica:**

1. **Albero di Huffman:** Viene costruito un albero di Huffman, dove le foglie rappresentano le parole del vocabolario. L'albero è bilanciato e le parole più frequenti sono più vicine alla radice.
2. **Calcolo della probabilità:** Per calcolare la probabilità di una parola di contesto, si percorre l'albero dalla radice alla foglia corrispondente alla parola. Ogni nodo dell'albero ha un peso associato, che viene utilizzato per calcolare la probabilità.
3. **Aggiornamento dei pesi:** Durante l'addestramento, i pesi dei nodi dell'albero vengono aggiornati per migliorare la precisione del modello.

**Vantaggi della softmax gerarchica:**

* **Efficienza:** Riduce il numero di calcoli necessari per la softmax, rendendo l'addestramento più veloce.
* **Struttura gerarchica:** L'albero di Huffman fornisce una struttura gerarchica che può essere utilizzata per migliorare la comprensione delle relazioni tra le parole.

## Word2vec: Scelte di progettazione

### Skip-gram vs. CBOW

**Skip-gram:** Prevede le parole di contesto data una parola centrale. È migliore per parole rare e finestre di contesto ampie, ma più lento in addestramento e meno efficiente per task *document-oriented*.

**CBOW:** Prevede la parola centrale date le parole di contesto. È migliore per parole frequenti, più veloce in addestramento e adatto a task *document-oriented*, ma meno preciso per parole rare.

| Caratteristica | Skip-gram | CBOW |
|---|---|---|
| **Obiettivo** | Predire parole di contesto | Predire parola centrale |
| **Parole frequenti** | Meno preciso | Più preciso |
| **Parole poco frequenti** | Più preciso | Meno preciso |
| **Finestra di contesto** | Più grande | Più piccola |
| **Velocità di addestramento** | Più lento | Più veloce |
| **Task** | Similarità, *relatedness*, analogia | Classificazione, task document-oriented |

Skip-gram è preferibile per vocabolari ampi, corpus specialistici, parole rare, finestre di contesto ampie e task *word-oriented*. CBOW è più veloce nell'addestramento, adatto a vocabolari moderati, corpus generici e task *document-oriented*.

### Addestramento

* **Hierarchical Softmax:** Ottimale per parole rare.
* **Negative Sampling:** Ottimale per parole frequenti e vettori a bassa dimensionalità.

### Sottocampionamento

Migliora accuratezza e velocità con dataset grandi (1e-3 a 1e-5).

### Dimensionalità Vettori

Generalmente, maggiore è meglio (ma non sempre).

### Dimensione Contesto

* **Skip-gram:** ~10
* **CBOW:** ~5

## Stochastic gradients with negative sampling \[aside]

- **Aggiornamento gradiente**: iterativamente calcolato per ogni finestra (`window`) di parole.
- **Sparsità del gradiente**:
 - In ogni finestra, abbiamo al massimo $2m + 1$ parole (più $2km$ parole per il negative sampling).
 - Il gradiente $\nabla_{\theta} J_t(\theta)$ è molto sparso per via del negative sampling.

$$ \nabla_{\theta} J_t(\theta) = 
 \begin{bmatrix}
 0 \\
 \vdots \\
 \nabla_{\theta_{target\_word}} \\
 \vdots \\
 \nabla_{\theta_{context\_word}} \\
 \vdots \\
 0
 \end{bmatrix}
 \in \mathbb{R}^{2dV}$$

- **Aggiornamento selettivo dei vettori**:
 - Si aggiornano solo i vettori di parola che compaiono nella finestra.
 - **Soluzioni per ottimizzare l'aggiornamento**:
 - Usare operazioni di aggiornamento sparse per aggiornare solo le righe necessarie delle matrici di embedding $U$ e $V$.
 - Utilizzare un hash per i vettori di parola, evitando di aggiornare ogni vettore in $U$ e $V$ completamente.

- **Evitare aggiornamenti ingombranti**:
 - Quando si hanno milioni di vettori di parola e si usa il calcolo distribuito, è importante ridurre la necessità di inviare aggiornamenti di grandi dimensioni.

- **Alternative per la costruzione di una Matrice di Co-occorrenza $X$:** 
 - **Finestra (window)**:
 - Simile a Word2Vec, utilizza una finestra attorno a ogni parola.
 - Cattura informazioni sintattiche e semantiche (*spazio delle parole*).
 - **Documento completo**:
 - Matrice di co-occorrenza basata su documenti.
 - Permette di ottenere temi generali (es. termini sportivi correlati), conducendo all'Analisi Semantica Latente (*spazio dei documenti*).

### Limitazioni delle rappresentazioni basate su co-occorrenze

Utilizzare direttamente le co-occorrenze come rappresentazione delle parole presenta alcune limitazioni:

* **Rappresentazione sparsa:** La matrice delle co-occorrenze è tipicamente molto sparsa (la maggior parte delle celle ha valore zero). Questa sparsità rende difficile e inefficiente l'utilizzo di tecniche di fattorizzazione lineare come la Latent Semantic Analysis (LSA).

* **Relazioni lineari:** LSA cattura principalmente relazioni lineari tra le parole. Questo significa che LSA può efficacemente modellare relazioni di sinonimia (parole con significati simili), ma fatica a gestire la polisemia (parole con molteplici significati). Se l'obiettivo è apprendere embedding delle parole per supportare diversi task *word-oriented*, è necessaria una rappresentazione capace di catturare la complessità delle relazioni semantiche, non solo quelle lineari. LSA, applicata alla matrice di co-occorrenze, favorisce quei task in cui le relazioni tra parole sono prevalentemente di sinonimia, a discapito di altri tipi di relazioni semantiche, come la polisemia.

## GloVe

GloVe è una tecnica alternativa a Word2Vec per creare word embeddings, ovvero rappresentazioni numeriche di parole, che si basa su un approccio ibrido tra tecniche neurali e statistiche. 

A differenza di modelli come Word2Vec, che generano una singola rappresentazione multidimensionale per ogni parola, GloVe considera il contesto in cui le parole compaiono. Questo approccio è particolarmente utile per gestire parole poco frequenti o con elevata polisemia, ovvero parole che assumono significati diversi a seconda del contesto.

**Come funziona GloVe:**

GloVe si basa sull'analisi delle co-occorrenze tra parole. In sostanza, si analizza la probabilità che due parole compaiano insieme in un testo. Queste informazioni vengono utilizzate per costruire una matrice di co-occorrenze, che rappresenta la relazione tra le parole.

**Vantaggi di GloVe:**

* **Gestione della polisemia:** GloVe riesce a catturare i diversi significati di una parola in base al contesto in cui compare.
* **Migliore rappresentazione di parole poco frequenti:** Grazie all'analisi delle co-occorrenze, GloVe riesce a rappresentare in modo più accurato anche parole che compaiono raramente nei testi.
* **Efficienza computazionale:** GloVe è generalmente più efficiente di altri modelli di word embedding, come Word2Vec.

**Punti chiave:**

* **Natura neurale:** GloVe è un modello neurale che utilizza le co-occorrenze per apprendere rappresentazioni dense delle parole.
* **Non un vettore globale:** L'embedding di GloVe non è un vettore globale nel senso che ogni parola è rappresentata da un unico embedding. L'embedding è il risultato dell'addestramento dell'algoritmo di GloVe su uno specifico corpus di training.
* **Rappresentazione di ogni parola:** Ogni parola è rappresentata da un unico embedding, che è il risultato dell'addestramento di GloVe su un corpus specifico.

I word embeddings sono rappresentazioni multidimensionali delle parole che catturano il loro significato in uno spazio vettoriale. In particolare, sono considerati **rappresentazioni globali** e **context-free**. Ciò significa che ogni parola ha un'unica rappresentazione, indipendentemente dal contesto in cui viene utilizzata.

**Rappresentazioni Globali:**

* Ogni parola ha una sola rappresentazione vettoriale, che viene utilizzata in qualsiasi contesto.
* Questa rappresentazione è statica e non cambia a seconda del contesto.

**Rappresentazioni Context-Free:**

* La rappresentazione di una parola non tiene conto del contesto in cui viene utilizzata.
* La stessa rappresentazione viene utilizzata per tutte le occorrenze di una parola, indipendentemente dal testo o dal contesto.

**Limiti delle Rappresentazioni Globali e Context-Free:**

* **Polisemia:** Le parole possono avere significati diversi a seconda del contesto. Le rappresentazioni globali non riescono a catturare questa complessità.
* **Parole poco frequenti:** Le parole poco frequenti hanno meno dati di training, il che può portare a rappresentazioni meno accurate.

## Funzione Obiettivo di GloVe

La funzione obiettivo di GloVe cerca di minimizzare la differenza tra il prodotto scalare degli embedding di due parole e il logaritmo della loro probabilità di co-occorrenza.
$$ f(x)=
\begin{cases}
\left( \frac{x}{x_{max}} \right)^\alpha, & if \ x<x_{max} \\
1, & otherwise
\end{cases}
$$

$$\text{Loss } J=\sum_{i,j=1}^Vf(X_{ij})(w_{i}T \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij})^2$$

Dove:

* $w_i$ e $w_j$ sono gli embedding delle parole $i$ e $j$.
* $b_i$ e $b_j$ sono i bias per le parole $i$ e $j$.
* $X_{ij}$ è il numero di co-occorrenze delle parole $i$ e $j$ in una finestra di contesto.

$p(i|j)$ è la probabilità di $w_i$ dato $w_j$.

Nella funzione $f(x)$ del primo termine della loss, abbiamo la distribuzione unigram con esponente $\frac{3}{4}$, che serve per smorzare l'effetto delle parole frequenti.

La loss contiene due termini di bias e il $\log X_{ij}$ che rappresenta $P(i|j)= \frac{X_{ij}}{X_{i}}$ (numero di co-occorrenze).

L'errore è valutato come la differenza tra la co-occorrenza reale e la co-occorrenza attesa.

L'idea è quella di catturare le proprietà tra i rapporti di co-occorrenza. 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8) NLP-20241112111853927.png|582]]

Consideriamo tre parole: $i$, $j$ e $k$, dove $k$ è una parola di confronto utilizzata per il confronto. Ad esempio, se $k$ è "solid", possiamo valutare il rapporto tra la probabilità di $k$ dato $i$ ("ice") e la probabilità di $k$ dato $j$ ("steam").

* **Rapporto > 1:** Indica che la parola $k$ è più correlata alla parola al numeratore ($i$). Ad esempio, se il rapporto è 8.9, "solid" è più correlato a "ice" che a "steam".
* **Rapporto < 1:** Indica che la parola $k$ è più correlata alla parola al denominatore ($j$).

**Relazioni geometriche:**

L'obiettivo è di rappresentare le parole in uno spazio vettoriale, dove le relazioni tra le parole possono essere interpretate geometricamente. Ad esempio, la relazione tra "man", "woman", "king" e "queen" può essere espressa come: "man" sta a "woman" come "king" sta a "queen".

Gli embeddings ci permettono di definire queste relazioni geometriche, dove l'embedding di "queen" può essere espresso come una combinazione lineare degli embeddings di "man", "woman" e "king".

## Funzione di Confronto

Per catturare le relazioni tra parole, definiamo una funzione $F$ che confronta la differenza tra due parole $w_i$ e $w_k$ rispetto a una terza parola $w_j$. Questa funzione deve soddisfare le seguenti proprietà:

**Condizione 1: Combinazione Lineare**

$$F((w_{i}-w_{j})^Tw_{k})=\frac{P(k|i)}{P(k|j)}$$

**Condizione 2: Simmetria**

$$F((w_{i}-w_{j})^Tw_{k})=\frac{F(w_{i}^Tw_{k})}{F(w_{j}^Tw_{k})}$$

**Definizione di F**

$$F(w_{i}^Tw_{k})=e^{w_{i}^Tw_{k}}=P(k|i) \text{, definita come } \frac{x_{ik}}{x_{i}}$$

**Derivazione**

Dunque, possiamo scrivere:

$$w_{i}^Tw_{k}=\log x_{ik}-\log x_{i}$$

Il termine $\log x_{i}$ viene assorbito da un bias $b_{i}$(inizializzato a 0) e otteniamo:

$$w_{i}^Tw_{k}+b_{i}+b_{j}=\log X_{jk}$$

Questo termine confronta le co-occorrenze attese delle parole $w_i$ e $w_j$ con le loro co-occorrenze effettive. Il termine $b_j$ viene aggiunto per la simmetria.
