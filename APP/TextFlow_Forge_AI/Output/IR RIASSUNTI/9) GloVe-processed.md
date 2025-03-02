
## GloVe: Un Modello di Regressione Log-Bilineare Globale per Vettori di Parole

GloVe (Global Vectors for Word Representation) è un nuovo modello che migliora la rappresentazione di parole in vettori, superando i limiti dei metodi precedenti.  A differenza dei metodi di fattorizzazione matriciale globale (come LSA), che soffrono di scarsa performance nelle analogie di parole e di un'influenza sproporzionata delle parole frequenti, e dei metodi di finestra di contesto locale (poco profondi), che non sfruttano le statistiche globali di co-occorrenza, GloVe utilizza una regressione log-bilineare globale.  Questo approccio combina i vantaggi di entrambi i metodi, ottenendo risultati significativamente migliori.

La performance di GloVe varia a seconda del compito e delle impostazioni.  Vettori di dimensioni superiori a 200 mostrano rendimenti decrescenti. Finestre di contesto piccole e asimmetriche sono più adatte per compiti sintattici, mentre finestre lunghe e simmetriche sono migliori per compiti semantici.  L'ampiezza del corpus e la sua fonte influenzano i risultati: corpus più ampi sono vantaggiosi per compiti sintattici, mentre Wikipedia supera Gigaword5 per compiti semantici.  GloVe supera anche Word2vec in termini di velocità e accuratezza, con un numero ottimale di campioni negativi per Word2vec intorno a 10.

La valutazione dei vettori di parole può essere intrinseca (su sottocompiti specifici, veloce ma con utilità incerta se non correlata a compiti reali) o estrinseca (su compiti reali, lenta ma più significativa, ma con difficoltà nell'isolare problemi di sottosistemi).  L'analogia di parole, espressa dalla formula:

$$a:b::c? \to d=\arg\max_{i} \frac{(x_{b}-x_{a}+x_{c})^T x_{i}}{\|x_{b}-x_{a}+x_{c}\|}$$

viene utilizzata per valutare la capacità dei vettori di catturare relazioni semantiche e sintattiche tramite la distanza coseno.  È importante escludere le parole di input dalla ricerca per evitare risultati banali.  La similarità di significato viene valutata confrontando la prossimità nello spazio vettoriale con giudizi umani (esempio in tabella nel testo originale).  GloVe supera anche CBOW, anche con dimensionalità doppia.

Infine, GloVe affronta il problema della polisemia (molteplicità di significati per una parola), cercando di catturare tutti i significati nello spazio vettoriale.  L'esempio della parola "pike" (picco, linea ferroviaria, strada) illustra questa sfida.

---

## Riassunto del testo:  Word Embeddings, NER e Classificazione

Il testo descrive l'utilizzo di word embeddings per il Named Entity Recognition (NER) e la classificazione, focalizzandosi su tecniche di apprendimento automatico.

### Word Embeddings e Contesto

Per rappresentare il significato delle parole,  invece di un singolo vettore, si propongono vettori multipli per parola, uno per ogni contesto di occorrenza.  Questi contesti vengono raggruppati tramite K-means, applicato a rappresentazioni dense ottenute tramite compressione LSA (Latent Semantic Analysis) delle matrici di contesto.  I cluster di vettori di contesto vengono poi utilizzati per aggiornare iterativamente i pesi del modello.

### Named Entity Recognition (NER)

Il NER consiste nell'identificare e classificare le entità nominate in un testo.  Il testo evidenzia l'importanza del pre-processing, in particolare per la gestione di parole composte e acronimi,  e la standardizzazione delle entità in una forma canonica.  Il problema può essere affrontato come classificazione binaria (una parola per classe) o multiclasse.  Un esempio mostra la costruzione di un vettore di contesto (`X_window`) per una parola target, utilizzato per la classificazione.  Le tecniche NER sono applicabili anche ad altri problemi, come l'analisi del sentiment.

### Training con Cross Entropy Loss

L'obiettivo dell'addestramento è minimizzare la probabilità logaritmica negativa della classe corretta. Questo obiettivo viene riformulato in termini di cross-entropia,  $H(p,q) = -\sum_{c=1}^{C} p(c)\log q(c)$, dove *p* è la distribuzione di probabilità vera e *q* quella calcolata dal modello.  Utilizzando one-hot encoding per *p*, la cross-entropia si riduce alla probabilità logaritmica negativa della classe corretta.  La funzione di costo è la binary cross entropy nel caso di classificazione binaria.

### Classificazione Neurale

Un classificatore softmax,  $p(y|x) = \frac{\exp(W_y \cdot x)}{\sum_c \exp(W_c \cdot x)}$, fornisce un confine di decisione lineare.  Le reti neurali migliorano questo aspetto:

* Apprendono sia i pesi `W` sia le rappresentazioni distribuite delle parole.
* Mappano i vettori delle parole (inizialmente one-hot) in uno spazio vettoriale intermedio tramite uno strato di embedding (x = Le).
* Utilizzano reti profonde per ottenere un classificatore non lineare.  Il softmax, in questo contesto, è una generalizzazione del regressore logistico a più classi.  L'obiettivo è massimizzare l'allineamento dell'istanza x con la matrice dei pesi per la classe y, normalizzando per ottenere una probabilità.

---

Questo documento descrive le reti neurali per la classificazione, focalizzandosi su aspetti chiave come l'architettura, l'addestramento e la regolarizzazione.

**Classificatore Softmax:**  Un classificatore softmax,  $p(y|x) = \frac{\exp(W_y \cdot x)}{\sum_c \exp(W_c \cdot x)}$,  calcola la probabilità di una classe *y* dato un input *x*.  Il processo prevede il calcolo del prodotto scalare tra i pesi ($W_y$) e l'input, seguito dall'applicazione della funzione softmax per normalizzare le probabilità e infine la selezione della classe con la probabilità massima. L'addestramento mira a minimizzare la negative log-likelihood: $-\log p(y|x)$.

**Classificatore Lineare Neurale:** Questo è essenzialmente un regressore logistico che utilizza una funzione di attivazione (es. sigmoide) per introdurre non-linearità.  L'input *x* viene trasformato tramite una funzione *f* (es. sigmoide, ReLU) producendo *h*, che viene poi combinato con un vettore di pesi e passato attraverso una funzione logistica per ottenere le probabilità finali.  ![[]] ![[ ]]

**Reti Neurali:** Le reti neurali sono modelli multistrato che estendono il concetto di classificatore lineare neurale, utilizzando funzioni di regressione logistica composte per catturare relazioni non lineari.  Strati più vicini all'input catturano relazioni sintattiche, mentre strati più profondi catturano relazioni semantiche. Il funzionamento può essere espresso come: $𝑧 = 𝑊𝑥 + 𝑏$; $𝑎 = 𝑓(𝑧)$, dove *f* è la funzione di attivazione applicata elemento per elemento.

**Regolarizzazione:** Tecniche come Lasso, Ridge ed Elastic Net riducono la complessità del modello per evitare l'overfitting. Il *dropout* disattiva casualmente neuroni durante l'addestramento, impedendo la co-dipendenza tra features.

**Vettorizzazione:** Si consiglia l'utilizzo di trasformazioni matriciali per una maggiore efficienza computazionale.

**Inizializzazione dei parametri:** I pesi devono essere inizializzati a piccoli valori casuali per evitare simmetrie che ostacolano l'apprendimento. I bias possono essere inizializzati a 0 (strati nascosti) o a valori ottimali (output), mentre altri pesi possono essere inizializzati con una distribuzione uniforme o tramite normalizzazione per layer.

---

L'inizializzazione Xavier migliora l'addestramento delle reti neurali definendo la varianza dei pesi in base alla dimensione degli strati adiacenti.  Specificamente, la varianza del peso  `W<sub>i</sub>` è data da:

$$Var(W_{i}) = \frac{2}{n_{in} + n_{out}}$$

dove `n<sub>in</sub>` è la dimensione dello strato precedente (fan-in) e `n<sub>out</sub>` è la dimensione dello strato successivo (fan-out).  Questa formula mira a mantenere le attivazioni con media prossima a zero e a evitare range di valori troppo ristretti nella distribuzione dei pesi, contribuendo così a un addestramento più efficace.

---
