## GloVe: Un nuovo modello di regressione log-bilineare globale

**GloVe** introduce un nuovo modello di regressione log-bilineare globale che combina i vantaggi sia della fattorizzazione matriciale globale che dei metodi di finestra di contesto locale.

##### Miglioramenti:

* **Metodi di fattorizzazione matriciale globale (es. LSA):**
 * Scarsa performance nel compito di analogia delle parole.
 * Le parole frequenti contribuiscono in modo sproporzionato alla misura di similarità.
* **Metodi di finestra di contesto locale, poco profondi:**
 * Non operano direttamente sui conteggi di co-occorrenza globali (cioè, usano solo parole adiacenti), non possono sfruttare le statistiche del corpus.

##### GloVe supera significativamente tutti gli altri modelli:

* Restituisce risultati decrescenti per vettori di dimensioni superiori a 200.
* Finestre di contesto piccole e asimmetriche (finestra di contesto solo a sinistra) funzionano meglio per i compiti sintattici.
* Finestre di contesto lunghe e simmetriche (finestra di contesto su entrambi i lati) funzionano meglio per i compiti semantici.
* I compiti sintattici hanno beneficiato di un corpus più ampio, mentre i compiti semantici hanno ottenuto risultati migliori con Wikipedia invece di Gigaword5, probabilmente a causa della completezza di Wikipedia e della natura leggermente obsoleta di Gigaword5.
* Le prestazioni di Word2vec diminuiscono se il numero di campioni negativi aumenta oltre circa 10.
* Per lo stesso corpus, vocabolario e dimensione della finestra, GloVe ottiene costantemente risultati migliori e più veloci.

## Come valutare i vettori di parole?

Relativamente alla valutazione generale in NLP: **intrinseca vs. estrinseca**

##### Intrinseca:

* Valutazione su un sottocompito specifico/intermedio.
* Veloce da calcolare.
* Aiuta a comprendere il sistema.
* Non è chiaro se sia realmente utile a meno che non sia stabilita una correlazione con un compito reale.

##### Estrinseca:

* Valutazione su un compito reale.
* Può richiedere molto tempo per calcolare l'accuratezza.
* Non è chiaro se il sottosistema sia il problema o la sua interazione con altri sottosistemi.
* Se la sostituzione di un solo sottosistema con un altro migliora l'accuratezza -5 Winningl.

## Analogie di vettori di parole

$$uomo:donna :: re:?$$
$$a:b::c? \to d=\arg\max_{i} \frac{(x_{b}-x_{a}+x_{c})^T x_{i}}{\|x_{b}-x_{a}+x_{c}\|}$$

un termine sta ad un altro come un terzo termine c(probe) sta a un termine d

Valutare i vettori di parole in base a quanto bene la loro distanza coseno dopo l'addizione cattura domande intuitive di analogia semantica e sintattica.

vogliamo massimizzare la differenza tra l'addizione b a c e confrontarlo con un candidato i e normalizzare 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114093852413.png]]

Se faccio king-man otterrei lo stesso risulato di fare queen-woman
##### Escludere le parole di input dalla ricerca

**Problema:** Cosa succede se l'informazione è presente ma non lineare? 

### Similarità del significato

Questi metodi vengono valutati confrontando il risultato della prossimità nello spazio della rappresentazione con alcuni riferimenti.

| Word 1    | Word 2   | Human (mean) |
| --------- | -------- | ------------ |
| tiger     | cat      | 7.35         |
| tiger     | tiger    | 10           |
| book      | paper    | 7.46         |
| computer  | internet | 7.58         |
| plane     | car      | 5.77         |
| professor | doctor   | 6.62         |
| stock     | phone    | 1.62         |
| stock     | CD       | 1.31         |
| stock     | jaguar   | 0.92         |

GloVe performa meglio degli altri modelli, anche di CBOW che ha dimensionalità doppia.

## La polisemia delle parole

La maggior parte delle parole ha molti significati.
* Soprattutto le parole comuni.
* Soprattutto le parole che esistono da molto tempo.

##### Esempio: Pike

* Un punto o un bastone affilato.
* Una linea o un sistema ferroviario.
* Una strada.

Avendo a che fare con rappresentazioni polisemiche, questa rappresentazione nello spazio deve catturare tutti i significati delle parol.
Un singolo vettore riesce a catturare tutti questi significati o abbiamo un pasticcio? 

Contesto globale e contesto locale delle parole per catturare tutti i significati. Per ogni parola, eseguire un clustering K-means per clusterizzare i contesti di occorrenza. rappresentazione densa attraverso compressione matrice LSA e questo diventa input del k-means. 
Costruiamo questi cluster di vettori di contesto e all iterazione successiva li utilizziamo per aggiornare i pesi. Invece di avere un unico vettore di embeddings per parola, ne abbiamo tanti quanti sono i contesti della parola

## Named Entity Recognition (NER)

Il task del Named Entity Recognition (NER) consiste nell'individuare e classificare i nomi propri all'interno di un testo, assegnando a ciascun token di parola una categoria lessicale.  È fondamentale eseguire alcuni passaggi di pre-processing, in particolare per la gestione di parole composte e acronimi (sigla).  La generalizzazione di questo processo consiste nel rendere i termini in una forma canonica, ovvero esprimere tutte le entità nello stesso modo, stabilendo una convenzione univoca.

Questo problema può essere risolto con la classificazione binaria.  Addestriamo un classificatore logistico su dati annotati (necessari per ottenere la variabile dipendente *y*) per classificare una parola centrale per ogni classe (tipo di entità).  In alternativa, è possibile affrontare il problema come un problema di classificazione multiclasse.

**Esempio:**

"The museums in Paris are amazing to see."

`X_window = [x_museums, x_in, x_paris, x_are, x_amazing]^T`

Nel caso binario, fissata una dimensione di contesto (nell'esempio 2), per la parola target (nell'esempio "Paris"), si costruisce il vettore `X_window` a cui è associata la classe "Paris".

Le tecniche di NER possono essere applicate anche alla risoluzione di altri problemi, come ad esempio l'analisi del sentiment.


## Training con la Loss Cross Entropy

**Training with cross entropy loss**

Fino ad ora, il nostro obiettivo era massimizzare la probabilità della classe corretta *y*, o equivalentemente minimizzare la probabilità logaritmica negativa di quella classe.  Ora riformuliamo questo obiettivo in termini di cross-entropia.

Sia *p* la distribuzione di probabilità vera (ground truth); sia *q* la probabilità calcolata dal nostro modello. La cross-entropia è definita come:

$H(p,q) = -\sum_{c=1}^{C} p(c)\log q(c)$

Assumendo una distribuzione di probabilità ground truth che è 1 nella classe corretta e 0 altrove (one-hot encoding),  *p* = [0, 0, ..., 0, 1, 0, ..., 0], allora l'unico termine rimanente nella somma è la probabilità logaritmica negativa della classe vera *y<sub>i</sub>*: −log *p*(y<sub>i</sub>∣x<sub>i</sub>).  In questo caso *y* può assumere uno dei valori di *c*.  La funzione di costo è la binary cross entropy se *y* può assumere solo i valori {0,1}. Per ogni input, solo uno dei due termini è attivo.


**Classificazione neurale**

Un tipico classificatore softmax in Machine Learning/Statistica è definito come:

$p(y|x) = \frac{\exp(W_y \cdot x)}{\sum_c \exp(W_c \cdot x)}$

I parametri appresi θ sono gli elementi di W (non la rappresentazione dell'input x, che spesso ha features simboliche sparse). Questo classificatore fornisce un confine di decisione lineare, che può essere limitante.

Un classificatore di rete neurale si differenzia in quanto:

* Apprendiamo sia W sia le rappresentazioni **(distribuite!)** delle parole.
* I vettori delle parole x, inizialmente one-hot, vengono mappati in uno spazio vettoriale di livello intermedio per una facile classificazione con un classificatore softmax (lineare). Concettualmente, abbiamo uno strato di embedding: x = Le.
* Utilizziamo reti profonde—più strati—che ci permettono di rappresentare e comporre i nostri dati più volte, dando un classificatore non lineare.


Softmax, in questo contesto, è una generalizzazione del regressore logistico a più classi.  Data x, la predizione di y è una funzione softmax dove si vuole massimizzare l'allineamento dell'istanza x con la matrice dei pesi per la predizione della classe y, normalizzando poi per la somma sulle varie classi per ottenere un valore pari a 1.  w è parte di $\theta$.  Una classificazione di rete neurale apprende sia la rappresentazione delle parole sia la matrice dei pesi, che guida il classificatore softmax.


**Classificatore Softmax**

$p(y|x) = \frac{\exp(W_y \cdot x)}{\sum_c \exp(W_c \cdot x)}$

Possiamo scomporre la funzione di predizione in tre passaggi:

1. Per ogni riga y di W, calcola il prodotto scalare con x:  $W_y \cdot x = \sum_i W_{yi} x_i = f_y$

2. Applica la funzione softmax per ottenere la probabilità normalizzata:  $p(y|x) = \frac{\exp(f_y)}{\sum_c \exp(f_c)} = \text{softmax}(f_y)$

3. Scegli la y con la probabilità massima.


Per ogni esempio di training (x, y), il nostro obiettivo è massimizzare la probabilità della classe corretta y, o minimizzare la probabilità logaritmica negativa di quella classe:

$-\log p(y|x) = -\log \left( \frac{\exp(f_y)}{\sum_c \exp(f_c)} \right)$

In sintesi, per ogni esempio di training, si cerca di massimizzare la probabilità della classe corretta y o di minimizzare la negative log-likelihood di quella stessa classe.

## Classificatore Lineare Neurale

Un classificatore lineare neurale è essenzialmente un regressore logistico che, dati gli input *x*, utilizza una funzione di trasformazione (di attivazione) sigmoide. Gli input *x* sono trasformati in accordo ai parametri da apprendere (pesi che regolano la pendenza della superficie di decisione e il suo bias, ovvero la traslazione nello spazio).

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114110524210.png]]
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/9)-20241114110636182.png]]

Abbiamo un input (istanza di training, che data una parola centrale, considera un contesto di lunghezza più 1). Questo input viene trasformato tramite una funzione *f* per modellare non linearità (es. sigmoide, logistica, tangente iperbolica, ReLU o altre). Il risultato di questa trasformazione è indicato con *h*.  Successivamente, la trasformazione *h* viene combinata con un vettore (probabilmente di pesi) su cui viene applicata una funzione logistica per il calcolo delle probabilità finali.

## Reti Neurali

Le reti neurali sono modelli multistrato che utilizzano funzioni di regressione logistica composte per catturare relazioni non lineari tra gli oggetti.  Aggiungendo uno strato di rappresentazione intermedia al perceptrone, si gestiscono le non-linearità.  La profondità della rete influenza la capacità di catturare queste non-linearità: strati più vicini all'input catturano relazioni non lineari a livello sintattico (focus sul contesto), mentre strati più distanti catturano relazioni semantiche tra parole lontane nel contesto.

Il funzionamento può essere descritto come segue:

$𝑎₁ = 𝑓(𝑊_{11}𝑥_{1} + 𝑊_{12}𝑥_{2} + 𝑊_{13}𝑥_{3} + 𝑏_{1})$
$𝑎₂ = 𝑓(𝑊_{21}𝑥_{1} + 𝑊_{22}𝑥_{2} + 𝑊_{23}𝑥_{3} + b_{2})$
ecc.

In notazione matriciale:

$𝑧 = 𝑊𝑥 + 𝑏$
$𝑎 = 𝑓(𝑧)$

dove la funzione di attivazione *f* è applicata elemento per elemento:

$𝑓([𝑧₁, 𝑧₂, 𝑧₃]) = [𝑓(𝑧₁), 𝑓(𝑧₂), 𝑓(𝑧₃)]$


##### Regolarizzazione

Le tecniche di regolarizzazione, come Lasso, Ridge ed Elastic Net, mirano a ridurre la complessità del modello ed evitare l'overfitting.


##### Dropout

Il dropout è una tecnica di regolarizzazione che riduce la co-adattamento delle features. Disattivando casualmente alcuni neuroni durante l'addestramento, si impedisce alle dimensioni dell'input di lavorare congiuntamente in modo indivisibile, prevenendo l'overfitting.


##### Vettorizzazione

Si consiglia l'utilizzo di trasformazioni matriciali per la vettorizzazione.


### Inizializzazione dei parametri

È fondamentale inizializzare i pesi a piccoli valori casuali (evitando matrici di zeri) per evitare simmetrie che impediscono l'apprendimento e la specializzazione dei neuroni.  I bias degli strati nascosti possono essere inizializzati a 0, mentre i bias di output (o di ricostruzione) possono essere inizializzati al valore ottimale se i pesi fossero 0 (es., target medio o inverso della sigmoide del target medio).  Altri pesi possono essere inizializzati con una distribuzione uniforme ~ Uniforme(-r, r), con r scelto opportunamente.  La necessità di questa attenta scelta di *r* viene rimossa con l'utilizzo della normalizzazione per layer.

L'inizializzazione di Xavier offre una soluzione alternativa, definendo la varianza dei pesi in funzione del *fan-in* (n<sub>in</sub>, dimensione dello strato precedente) e del *fan-out* (n<sub>out</sub>, dimensione dello strato successivo):

$$Var(W_{i}) = \frac{2}{n_{in} + n_{out}}$$

Questa inizializzazione favorisce medie intorno a 0, evitando range troppo piccoli nella distribuzione uniforme e migliorando l'efficacia dell'addestramento.
