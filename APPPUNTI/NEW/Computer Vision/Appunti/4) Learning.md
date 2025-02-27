## Apprendimento Supervissato

Definiamo un insieme di dati $D = \{ (x_1, y_1), \dots, (x_n, y_n) \}$ dove:

* $x \in \mathbb{R}^m$ rappresenta un vettore di $m$ features (caratteristiche)
* $y \in \{0, 1\}$ rappresenta l'etichetta di classe associata a $x$.

Un problema di apprendimento supervissato può essere definito come la ricerca di una funzione $f(x)$ che approssimi la relazione tra le features $x$ e le etichette $y$. In altre parole, vogliamo che $f(x)$ preveda accuratamente l'etichetta $y$ dato un nuovo vettore di features $x$.

Una possibile formulazione per $f(x)$ è:

$$f(x, w) = I(wx + b > 0)$$

dove:

* $w$ è un vettore di pesi che determina l'orientamento dell'iperpiano di separazione
* $b$ è un bias che determina la posizione dell'iperpiano
* $I(wx + b > 0)$ è la funzione indicatrice che restituisce 1 se $wx + b > 0$ e 0 altrimenti.

Questa formulazione si basa sull'intuizione che i punti che si trovano sullo stesso lato dell'iperpiano di separazione appartengono alla stessa classe. In altre parole, l'iperpiano separa i punti in due gruppi, uno per ogni classe.

Questa intuizione è espressa nella formula per la probabilità:
$$P(+|x,\theta )=\frac{1}{1+\exp(-wx+b)}$$
Dove:

* $x_i$ rappresenta i dati di input.
* $θ$ rappresenta l'insieme dei parametri, inclusi *w* e *b*.
* $y_i$ rappresenta l'output.

L'obiettivo non è solo trovare una funzione che si adatti ai dati, ma trovare la funzione che meglio si adatta ai dati nella maggior parte dei casi.

Invece di utilizzare un approccio basato sul "minimo", si adotta un framework di massima verosimiglianza. Questo significa che, tra tutti i possibili parametri *w* (appartenente a R^m) e *b* (appartenente a R), si cerca il parametro che massimizza la *likelihood* di *θ*.

Massimizzare la *likelihood* di *θ* significa trovare il valore di *θ* che rende massima la probabilità di osservare i dati y_i, dati i parametri *θ* e i dati di input *x_i*. 

Formalmente, si cerca:

$$\arg \max_{v \in \theta} L(D,\theta)=\prod_{i=1}^n p(y_{i}|x_{i,\theta})$$

Dove:

* *n* rappresenta il numero di osservazioni.
* *θ* appartiene a R^(m+1), lo spazio dei parametri.

L'obiettivo finale è quindi trovare il valore di *θ* che massimizza la funzione di *likelihood*. 

## Espressione della Funzione di Verosimiglianza (Likelihood)

Come possiamo esprimere la formula per la funzione di verosimiglianza? Considerando la probabilità del positivo e del negativo (1 meno la probabilità del positivo), possiamo descrivere la formula come:
$$\arg \min_{v \in \theta} nll(D,\theta)=\sum_{_i=1}^n \log p(y_{i}|x_{i,\theta})$$

Questa formula esprime la *likelihood*: massimizza la probabilità di osservare la positività quando l'etichetta è positiva e la probabilità di osservare la negatività quando l'etichetta è negativa, sempre dato x_i e i parametri θ.

### Passaggio alla Formulazione Logaritmica

Tuttavia, questa formula presenta un problema: le componenti sono comprese tra 0 e 1. Con un numero elevato di punti dati (n), il prodotto tenderebbe a 0.

Per risolvere questo problema, si utilizza una formulazione logaritmica, che permette di scalare di ordine di grandezza esponenziale. Invece di massimizzare la *likelihood*, si massimizza la *log-likelihood*:

$$LL(\theta)=\sum_{i=1}^n y_{i}p(y_{i}|x_{i})+(1-y_{i})(1-p_{i(y_{i}|x_{i})})$$

Questa formula rappresenta la *cross-entropy*, una misura della differenza tra la distribuzione vera delle etichette e la distribuzione predetta dal modello. 

In sintesi:
- Massimizzare la *log-likelihood* equivale a minimizzare la *cross-entropy*.
- La *cross-entropy* è una realizzazione della verosimiglianza nel dominio logaritmico.
- Minimizzare la *cross-entropy* significa massimizzare il negativo della *log-likelihood*. 
![[1.1-20241014224610360.png|535]]
### Ottimizzazione della Log-Likelihood

Come ottimizzare la *log-likelihood*? Ci troviamo nello spazio dei numeri reali, con un vettore di parametri di dimensione *m* + 1. 

Ponendo $\hat{y_{i}}=y_{i}p(y_{i}|x_{i})$

Sostituendo questa espressione nella formula della *log-likelihood*, otteniamo:

$$\sum_{i=1}^n \{ y_{i} \cdot\log \hat{y_{i}}+(1-y_{i})\cdot \log(1-\hat{y_{i}}) \}$$
Questa formula può essere espressa come la somma dei costi individuali per ogni coppia $(y_i, ŷ_i)$, dove $ŷ_i$ rappresenta la predizione del modello:

$$=-\sum_{i=1}^n Cost(y_{i},\hat{y_{i}})$$

Il segno meno indica che minimizzare la *cross-entropy* equivale a massimizzare la *log-likelihood*.

### Interpretazione del Costo

Il costo rappresenta la differenza tra l'etichetta vera e la predizione del modello. Un costo elevato indica una bassa probabilità e quindi una predizione poco accurata. 

La formulazione del costo cumulativo ci permette di utilizzare metodi di ottimizzazione basati sul gradiente per trovare il minimo della funzione di costo.

## Metodi di Ottimizzazione

$$nll(D,\theta)=-\sum_{i=1}^n Cost(y_{i},\hat{y_{i}})$$
Esistono due principali metodologie per ottimizzare la *log-likelihood*:

1. **Metodi del Primo Ordine (Discesa del Gradiente):** Si basano sul calcolo del gradiente della funzione di costo e si muovono iterativamente nella direzione opposta al gradiente per trovare il minimo. Sono metodi relativamente semplici da implementare, ma possono richiedere molto tempo per convergere.

2. **Metodi del Secondo Ordine (Newton-Raphson):** Utilizzano anche l'informazione sulla curvatura della funzione di costo (matrice Hessiana) per convergere più velocemente al minimo. Tuttavia, il calcolo della matrice Hessiana può essere computazionalmente oneroso, soprattutto per un numero elevato di parametri.

La scelta del metodo di ottimizzazione dipende dalla specifica applicazione e dal compromesso tra velocità di convergenza e complessità computazionale. 

## Limiti dei Metodi del Secondo Ordine

Il calcolo della matrice Hessiana, necessaria per i metodi di ottimizzazione del secondo ordine come Newton-Raphson, può diventare problematico al crescere del numero di parametri (*m*). 

Una matrice Hessiana di grandi dimensioni comporta:

- **Elevato costo computazionale:** Calcolare la derivata seconda per ogni coppia di parametri diventa oneroso.
- **Problemi di gestione:** Memorizzare e operare su una matrice di grandi dimensioni può essere difficile.

## Discesa del Gradiente

La discesa del gradiente è un metodo iterativo del primo ordine che ha trovato largo impiego nel machine learning. L'idea di base è:

1. **Inizializzazione:** Si parte da un punto casuale nello spazio dei parametri (k).
2. **Iterazione:** Si aggiornano i parametri nella direzione opposta al gradiente della funzione di costo. Questo garantisce una diminuzione del costo ad ogni passo.
3. **Criterio di arresto:** L'algoritmo termina quando:
 - Si raggiunge un punto stazionario (gradiente nullo).
 - La differenza tra i parametri in due iterazioni consecutive diventa trascurabile (es. $10^{-3}$).

## Costruzione della Sequenza di Aggiornamento

La discesa del gradiente costruisce una sequenza di parametri che minimizza la funzione di costo (in questo caso, la *negative log-likelihood* - NLL). Ad ogni iterazione, i parametri vengono aggiornati nella direzione che produce la massima riduzione del costo.
$$nll(\theta^{(0)})\geq nll(\theta^{(1)})\geq nll(\theta^{(2)})$$
## Criterio di Arresto

Il criterio di arresto più comune si basa sulla differenza tra i parametri in due iterazioni consecutive. Quando questa differenza scende sotto una soglia predefinita (es. 10^-3), si considera raggiunta una convergenza soddisfacente. La scelta della soglia è arbitraria e dipende dalla precisione desiderata. 

## Ottimizzazione tramite Funzione di Costo

L'approccio generale per ottimizzare un modello di machine learning consiste nel definire una funzione di costo che misuri la differenza tra le predizioni del modello $(ŷ)$ e i valori reali $(y)$. 

$$Costo(y, ŷ)$$

Se la funzione di costo è derivabile, possiamo utilizzare il gradiente per trovare il minimo. 

Nel caso della regressione logistica, la funzione di costo comunemente utilizzata è la *cross-entropy* (o entropia, nel caso binario), strettamente legata alla *log-likelihood*.

## Discesa del Gradiente: Regola di Aggiornamento

La discesa del gradiente fornisce una regola semplice per aggiornare i parametri del modello $(θ)$ iterativamente:
$$\theta^{(t+1)}=l\theta^{(t)}\nabla nll\theta^{(t)}$$

## Intuizione Geometrica

La regola di aggiornamento si basa sull'approssimazione lineare della funzione di costo tramite lo sviluppo in serie di Taylor. In pratica, si approssima la funzione in un punto con la sua tangente in quel punto.

Considerando lo sviluppo al primo ordine:

$$L(θ) ≈ L(θ_t) + (θ - θ_t)^T ∇L(p)$$

Dove:

- $p$: Punto in cui calcoliamo l'approssimazione.
- $θ$: Punto generico.

Sostituendo $θ$ con $θ_{(t+1)}$ e p con $θ_t$, otteniamo:

$$L(θ_{(t+1)}) ≈ L(θ_t) + (θ_{(t+1)} - θ_t)^T ∇L(θ_t)$$

Sostituendo la regola di aggiornamento $(θ_{(t+1)} = θ_t - \lambda ∇L(θ_t))$ : 

## Garanzia di Decrescenza della Funzione di Costo

La regola di aggiornamento della discesa del gradiente garantisce una diminuzione della funzione di costo ad ogni passo. Questo è evidente dalla formula:

$$L(θ_{(t+1)}) ≈ L(θ_t) - \lambda ∇L(θ_t)^T ∇L(θ_t)$$

- Il termine $\lambda ∇L(θ_t)^T ∇L(θ_t)$ è sempre positivo, essendo α positivo e il prodotto scalare di un vettore per se stesso sempre positivo.
- Di conseguenza, $L(θ_{(t+1)})$ sarà sempre minore di $L(θ_t)$, garantendo la decrescenza della funzione di costo.

## Interpretazione Geometrica

Immaginiamo la funzione di costo $L(θ)$ come una superficie. 

- Il punto $θ_t$ rappresenta la posizione corrente nello spazio dei parametri.
- La tangente alla superficie in $θ_t$ è data dal gradiente $∇L(θ_t)$.
- La discesa del gradiente ci muove nella direzione opposta al gradiente, ovvero verso il basso lungo la superficie.
- La lunghezza del passo è determinata dal *learning rate* $\lambda$.

## Importanza del Learning Rate (ʎ)

Il *learning rate* è un parametro cruciale che controlla la dimensione dei passi durante l'aggiornamento dei parametri.

- **Valori troppo grandi:** Possono portare a oscillazioni intorno al minimo, impedendo la convergenza.
- **Valori troppo piccoli:** Rallentano la convergenza, richiedendo molte iterazioni per raggiungere il minimo.

La scelta ottimale del *learning rate* dipende dal problema specifico. 

![[1.1-20241014225616481.png|382]]

### Minimi Locali

La discesa del gradiente garantisce di trovare un minimo locale della funzione di costo, ma non necessariamente il minimo globale. 

Questo limite è evidente nell'esempio della funzione di Rosenbrock, dove la forma della funzione e la scelta del punto iniziale possono intrappolare l'algoritmo in un minimo locale.

Esistono diverse tecniche per cercare di superare il problema dei minimi locali, tra cui:

- **Inizializzazione casuale:** Eseguire la discesa del gradiente partendo da diversi punti casuali nello spazio dei parametri.
- **Algoritmi evolutivi:** Utilizzare algoritmi come gli algoritmi genetici, che esplorano lo spazio dei parametri in modo più globale.
- **Tecniche di *momentum*:** Modificare la regola di aggiornamento per includere un termine di inerzia, che aiuta a "scavalcare" i minimi locali.

### Costo Computazionale

L'algoritmo di discesa del gradiente, seppur concettualmente semplice, può essere computazionalmente costoso. 

Ad ogni iterazione, è necessario:

1. Calcolare il gradiente della funzione di costo rispetto a tutti i parametri del modello.
2. Aggiornare i parametri del modello utilizzando la regola di aggiornamento.

Il costo di queste operazioni dipende dalla complessità della funzione di costo e dalla dimensione del dataset.

## Funzioni di *loss* non convesse e ottimizzazione

La non convessità delle funzioni di *loss* rappresenta una sfida nell'ottimizzazione dei modelli di machine learning. Nonostante questa difficoltà, è possibile trovare un punto di minimo locale.

### Costo computazionale e ottimizzazione

Il calcolo del gradiente della funzione *loss* (L(θ)), necessario per l'ottimizzazione, può essere computazionalmente oneroso, soprattutto per dataset di grandi dimensioni. 

Tuttavia, la natura incrementale del calcolo del gradiente permette di adottare approcci approssimati, come lo *stochastic gradient descent* (SGD). Invece di calcolare il gradiente su tutto il dataset, SGD considera solo un sottoinsieme di dati ad ogni passo, riducendo significativamente il costo computazionale. 

## Approssimazioni della funzione di *loss*

L'idea delle approssimazioni consiste nel sostituire il calcolo dell'intera sommatoria con un calcolo su un sottoinsieme del dataset. Questo sottoinsieme è chiamato "batch".

### Batch e approssimazione

Ad esempio, se il dataset contiene 100.000 tuple e il batch ne contiene 50, 100 o 200, l'ordine di grandezza del calcolo si riduce drasticamente, con un notevole vantaggio dal punto di vista computazionale.

La funzione di *loss* calcolata sul batch (L di B) rappresenta un'approssimazione della funzione di *loss* originale. Si assume che il batch scelto sia rappresentativo del dataset, permettendo di modificare l'algoritmo in modo più efficiente. 

## Minibatch (stochastic) Gradient Descent

SGD è un algoritmo di ottimizzazione che utilizza un approccio stocastico per approssimare la funzione di *loss*.

### Iterazioni ed aggiornamenti

Dato un learning rate `λ`, un numero di epoche `N` e un numero di batch `M`:
Il processo di SGD prevede le seguenti fasi:

1. **Iterazioni:** Si itera per un certo numero di epoche.
2. **Campionamento:** Ad ogni epoca, si campiona un sottoinsieme del dataset di addestramento (il batch).
3. **Aggiornamento:** Si aggiorna il modello utilizzando il gradiente calcolato sul batch, che rappresenta un proxy della funzione di *loss*.

### Natura stocastica

L'approccio di SGD è stocastico perché non si calcola la funzione di *loss* completa, ma si utilizza un'approssimazione basata sul batch. Questo processo di campionamento introduce un elemento di casualità nell'algoritmo.

### Vantaggi di SGD

L'utilizzo di SGD presenta diversi vantaggi:

- **Ridotto costo computazionale:** Il calcolo del gradiente su un batch è molto più efficiente rispetto al calcolo su tutto il dataset.
- **Miglioramento della generalizzazione:** L'introduzione di casualità può aiutare a evitare i minimi locali e migliorare la capacità del modello di generalizzare a nuovi dati.

## Confronto tra gradiente discendente e gradiente discendente stocastico

La domanda è: il gradiente discendente stocastico (SGD) funziona? La risposta è sì.

### Confronto visivo

![[1.2-20241014230102196.png|315]]
La slide mostra un confronto visivo tra il gradiente discendente (linea blu) e il gradiente discendente stocastico (linea rossa). La funzione è rappresentata con le sue curve di livello.

**Discesa del gradiente:**

- La Discesa del gradiente segue la direzione tangente alla direzione di decrescita della funzione.
- L'algoritmo punta direttamente verso il punto di minimo della funzione.
- In questo esempio, sono necessarie 6 iterazioni per raggiungere il minimo.

**Discesa del gradiente stocastico:**

- La Discesa del gradiente stocastico segue un percorso più irregolare, con un percorso più "rumoroso" rispetto al gradiente discendente.
- Questo è dovuto al fatto che l'algoritmo utilizza un batch di dati per calcolare il gradiente, introducendo un elemento di casualità.
- Nonostante il percorso irregolare, l'algoritmo converge comunque verso il punto di minimo.

Il processo di campionamento introduce un elemento di casualità nell'algoritmo, portando a un percorso di ottimizzazione "rumoroso".

### Natura stocastica e convergenza

- **Approssimazione:** L'SGD approssima il gradiente utilizzando un sottoinsieme dei dati, il che può portare a scelte fortunate o sfortunate.
- **Percorso rumoroso:** Il percorso di ottimizzazione dell'SGD può essere irregolare e "rumoroso" a causa della natura stocastica del campionamento.
- **Convergenza:** Nonostante la rumorosità, l'SGD converge alla soluzione ottima locale. La convergenza richiede più passi rispetto al gradiente discendente, ma ogni passo è molto più veloce.

### Vantaggi dell'SGD

- **Velocità:** Ogni passo dell'SGD è significativamente più veloce rispetto al gradiente discendente, poiché si basa su un sottoinsieme dei dati.
- **Efficienza:** L'SGD è particolarmente efficiente per dataset di grandi dimensioni, dove il calcolo del gradiente su tutto il dataset sarebbe computazionalmente costoso.

## Funzione di probabilità, funzione di costo e gradiente discendente

Per costruire un modello di classificazione, abbiamo bisogno di tre elementi chiave:

1. **Funzione di probabilità:** Definisce la probabilità associata al responso del modello. Il responso può essere binario (es. positivo/negativo) o multi-classe (es. A, B, C).
2. **Funzione di costo:** Misura la discrepanza tra le previsioni del modello e i dati reali. Deve essere derivabile rispetto ai parametri del modello (teta).
3. **Gradiente discendente:** Un algoritmo che trova il valore ottimale dei parametri del modello minimizzando la funzione di costo. È un algoritmo "plug and play" che accetta la funzione di costo e restituisce l'ottimo in modo iterativo.

### Esempio: regressione logistica

La regressione logistica è un esempio di modello di classificazione che utilizza questi tre elementi.

- **Funzione di probabilità:** La probabilità di osservare un responso positivo o negativo è calcolata utilizzando una funzione logistica.
- **Funzione di costo:** La funzione di costo è tipicamente la cross-entropy, che misura la discrepanza tra le probabilità predette e le probabilità reali.
- **Gradiente discendente:** L'algoritmo di gradiente discendente viene utilizzato per trovare i pesi ottimali del modello che minimizzano la funzione di costo.

### Estensione a più classi

![[1.2-20241014230311029.png|591]]
La regressione logistica può essere estesa a problemi di classificazione multi-classe. In questo caso, si costruiscono più classificatori binari, uno per ogni classe. Ad esempio, per tre classi (C, B, D), si costruiscono tre classificatori:

- **Classificatore C:** "È C? Sì o no"
- **Classificatore B:** "È B? Sì o no"
- **Classificatore D:** "È D? Sì o no"

Ogni classificatore utilizza un iperpiano di separazione per classificare i dati in due classi. La combinazione di questi classificatori permette di ottenere una classificazione multi-classe.

### Formulazione generale

La formulazione generale per la classificazione multi-classe è simile a quella della regressione logistica, ma con un'estensione per gestire più classi. La formulazione specifica è disponibile nelle slide.

Per affrontare problemi di classificazione con più di due classi, si estende il framework della regressione logistica.

### Funzione di probabilità e cross-entropy

- **Responso multi-classe:** Il modello calcola un responso per ogni possibile classe, rappresentando la probabilità che il responso sia 1 per quella classe e 0 per le altre.
- **Cross-entropy:** La funzione di costo utilizzata è la cross-entropy, un'estensione della *binary cross-entropy* per gestire più classi.
- **Formulazione:** La probabilità di ogni classe è espressa tramite una formula che estende la funzione logistica a più valori.

### Ottimizzazione dei parametri

- **Previsioni:** Il modello calcola le previsioni per ogni classe.
- **Confronto:** Le previsioni vengono confrontate con le risposte reali.
- **Gradiente discendente:** Il gradiente discendente viene applicato per ottimizzare l'insieme dei parametri (WC, WD, WB) del modello, minimizzando la funzione di costo.

## Estrazione di feature da immagini

![[1.2-20241014230357750.png|563]]
Dopo aver trasformato un'immagine in un vettore di feature (ad esempio, un vettore di 4 componenti), è importante capire cosa rappresentano queste componenti.

### Estrazione di feature

Le feature possono essere estratte dall'immagine utilizzando diversi metodi, come:

- **Istogrammi dei colori:** Calcolano la frequenza di ogni colore nell'immagine.
- **Algoritmi SIFT:** Estraggono feature invarianti alla scala, come angoli, contorni e punti significativi.
- **Altri metodi:** Esistono altri metodi per estrarre feature, come l'analisi di texture, la segmentazione dell'immagine e l'estrazione di bordi.

### Esempio: istogrammi dei colori

Un esempio semplice di estrazione di feature è l'utilizzo degli istogrammi dei colori. Si calcola l'istogramma per ogni canale di colore (rosso, verde, blu) e si concatenano i tre vettori risultanti in un unico vettore di feature.

### Rappresentazione dello spazio delle feature

L'insieme di feature estratte da un'immagine può essere rappresentato in uno spazio vettoriale di N dimensioni. Ad esempio, se si utilizzano istogrammi di colori con 256 livelli per ogni canale, si ottiene uno spazio di 768 dimensioni (3 canali * 256 livelli).

### Feature più sofisticate

Oltre agli istogrammi dei colori, si possono utilizzare feature più sofisticate, come i punti SIFT, che sono invarianti alla scala e alla rotazione. Questi punti identificano le aree più significative dell'immagine, come angoli, contorni e punti di interesse.

### Perché abbiamo bisogno di calcolare le *feature* e di ottenere un unico vettore che le rappresenti?

La risposta risiede nel fatto che il metodo di regressione logistica, che abbiamo visto in precedenza, richiede un unico vettore di input per ogni immagine. Questo vettore rappresenta le caratteristiche principali dell'immagine, come i colori, le texture o i punti di interesse.

Nel caso specifico, abbiamo visto che la regressione logistica utilizza tre vettori, corrispondenti agli istogrammi dei colori per i tre canali (rosso, verde, blu). Tuttavia, per poter applicare il metodo di regressione logistica, è necessario combinare questi tre vettori in un unico vettore.

Questo processo di estrazione e combinazione di *feature* è fondamentale per ottenere una rappresentazione vettoriale dell'immagine che possa essere utilizzata per l'addestramento di modelli di classificazione.

## L'estensione della regressione logistica a più livelli

Negli ultimi anni, si è sviluppato un nuovo approccio che consiste nell'estendere il concetto di regressione logistica a più livelli. Questo è il principio alla base delle reti neurali, che permettono di apprendere rappresentazioni più complesse e di risolvere problemi di classificazione più complessi.

In sintesi, la necessità di un unico vettore di *feature* deriva dalla necessità di avere un input univoco per i modelli di classificazione, come la regressione logistica. L'estensione della regressione logistica a più livelli, come nelle reti neurali, permette di apprendere rappresentazioni più complesse e di risolvere problemi di classificazione più complessi.

## La sfida del *feature engineering*

![[1.2-20241014230447424.png]]

Il principio fondamentale di questa idea, ormai ampiamente diffusa, è racchiuso in questa slide. La difficoltà principale che si presenta è la necessità di fare una serie di scelte.

Se si hanno immagini iniziali, e si vuole rappresentarle in un insieme di *feature*, bisogna decidere quali *feature* scegliere e come costruirle.

Ad esempio, la slide propone di utilizzare gli istogrammi delle intensità sui tre canali (rosso, verde, blu). In alternativa, si potrebbe utilizzare l'algoritmo SIFT o qualsiasi altro metodo.

Queste scelte possono essere buone o cattive, e in sostanza si sta facendo *feature engineering*, ovvero si sta costruendo l'ingegneria delle *feature* a partire dall'immagine.

Quando si costruisce un istogramma o si calcolano i punti salienti dell'immagine con l'algoritmo SIFT, si stanno estraendo delle *feature*, ovvero delle caratteristiche dell'immagine che si vogliono rappresentare.

L'algoritmo utilizzato per l'estrazione delle *feature* ha una serie di parametri che devono essere impostati. Se si modificano questi parametri, si ottengono *feature* diverse.

In definitiva, si deve lavorare su una serie di "manopole" che rappresentano il *feature engineering*, ovvero il processo di scelta e costruzione delle *feature*. 

## Il passaggio dal *feature engineering* all'apprendimento automatico delle *feature*

Il flusso di lavoro tradizionale prevedeva l'utilizzo di un algoritmo di riferimento, come la regressione logistica, per classificare un'immagine. L'immagine, rappresentata come un tensore tridimensionale (altezza, larghezza, profondità), veniva convertita in un insieme di *feature* attraverso un processo di *feature engineering*.

Il principio delle reti neurali, in particolare quelle profonde, introduce un nuovo approccio: l'algoritmo stesso impara a calcolare le *feature* necessarie per la classificazione.

### Rappresentazione grafica della regressione logistica

Possiamo pensare alla regressione logistica come a un grafo. Se abbiamo 4 *feature*, il grafo avrà 4 nodi, ognuno dei quali rappresenta una *feature*. Ad ogni arco del grafo è associato un peso.

La regressione logistica calcola una somma ponderata delle *feature*, che può essere rappresentata come:

$$y = w_1*x_1 + w_2*x_2 + w_3*x_3 + w_4*x_4 + b$$

dove `w1`, `w2`, `w3`, `w4` sono i pesi e `b` è un termine costante.

## La struttura della rete neurale

La struttura di una rete neurale è simile a quella della regressione logistica, ma con l'aggiunta di uno strato di attivazione.

**Calcolo della predizione:**

La predizione $\hat{y}$ è calcolata come:

$$\hat{y} = \frac{1}{(1 + exp(-a))}$$

dove $a$ è la somma ponderata delle *features*:

$$a = wx + b$$

In sostanza, la rete neurale impara i pesi $w$ e i parametri $b$ dello strato di attivazione per ottenere la migliore rappresentazione delle *features* e la migliore predizione possibile.

**Rappresentazione grafica con due *features***

Per semplificare la rappresentazione grafica, consideriamo solo due *features*: $x_1$ e $x_2$.

La rete neurale con due *features* calcola l'interazione tra le due *features*. Consideriamo due componenti: $a_1$ e $a_2$.

* $a_1$ è una componente di regressione logistica con peso 1 per entrambe le *features*.
* $a_2$ è una componente di regressione logistica con peso 2 per $x_2$ e peso 1 per $x_1$.

Possiamo rappresentare graficamente le componenti $a_1$ e $a_2$ nel nostro spazio bidimensionale:

* $a_1$ è rappresentata da un vettore che passa per il punto (1,1).
* $a_2$ è rappresentata da un vettore che passa per il punto (2,2).

Ogni componente definisce una zona nel nostro spazio bidimensionale. Ad esempio, $a_1$ definisce una zona in cui $x_1$ e $x_2$ hanno lo stesso valore.

Possiamo aggiungere una terza componente, $a_3$, che definisce una nuova zona nel nostro spazio bidimensionale.

### Interpretazione della classificazione con più componenti

Osservando la rappresentazione grafica delle componenti $a_1$, $a_2$ e $a_3$, notiamo che queste definiscono tre zone distinte nel nostro spazio bidimensionale. Ogni zona è associata a un meccanismo di classificazione binaria: positivo o negativo.

In altre parole, la rete neurale utilizza le componenti per suddividere lo spazio delle *features* in diverse regioni, ciascuna con una propria classificazione. Questo processo di suddivisione dello spazio delle *features* consente alla rete neurale di apprendere relazioni complesse tra le *features* e le etichette di classe.

## Classificazione binaria con regressione logistica

Un singolo elemento di regressione logistica è in grado di risolvere problemi di classificazione binaria, ovvero problemi in cui i dati sono separabili linearmente.

**Esempio:** Se i dati sono disposti in modo da formare una linea retta, la regressione logistica può tracciare una linea di separazione che divide i dati in due classi.

**Limiti:** La regressione logistica ha dei limiti. Se i dati non sono separabili linearmente, la regressione logistica non sarà in grado di classificarli correttamente.

**Esempio:** Se i dati sono disposti in modo da formare una curva, la regressione logistica non sarà in grado di tracciare una linea di separazione che divida i dati in due classi.

**Soluzione:** Per risolvere problemi di classificazione più complessi, possiamo combinare più componenti di regressione logistica. Ogni componente definisce una zona nel nostro spazio multidimensionale e contribuisce alla classificazione finale.

## Classificazione con più iperpiani di separazione

Consideriamo un problema di classificazione in uno spazio multidimensionale, dove la separazione tra le classi è definita da tre iperpiani.

### Ruolo dei neuroni nella classificazione

Ogni iperpiano è rappresentato da un neurone, che funge da componente di classificazione binaria.

* **Neurone 1:** Definisce un iperpiano che separa i dati in due classi: positivo sopra l'iperpiano, negativo sotto.
* **Neurone 2:** Definisce un iperpiano che separa i dati in due classi: positivo da un lato dell'iperpiano, negativo dall'altro.
* **Neurone 3:** Definisce un iperpiano che separa i dati in due classi: positivo sopra l'iperpiano, negativo sotto.

### Nuove *feature* per la classificazione

Possiamo interpretare l'output di questi neuroni come nuove *feature* per la classificazione finale.

Ogni neurone fornisce informazioni sulla posizione dei dati rispetto al suo iperpiano di separazione.

Per ottenere la classificazione finale, possiamo combinare le informazioni provenienti dai tre neuroni.

Ad esempio, se il neurone 1 e il neurone 2 indicano "positivo" e il neurone 3 indica "negativo", la classificazione finale potrebbe essere "positivo".

### Interpretazione della classificazione

La classificazione finale è determinata dalla combinazione delle informazioni provenienti dai neuroni, che a loro volta sono influenzate dalla posizione dei dati rispetto agli iperpiani di separazione.

## Mappatura dello spazio e *kernel machines*

La struttura agendica proposta mappa uno spazio bidimensionale in uno spazio multidimensionale. Questo processo di mappatura in uno spazio a più alta dimensione è il principio alla base delle *kernel machines*.

Le *kernel machines* affrontano problemi di classificazione non separabili in uno spazio a bassa dimensione, mappandoli in uno spazio a più alta dimensione dove la separazione diventa possibile.

### Risoluzione di problemi di classificazione complessi

La struttura a due livelli, con i suoi neuroni e iperpiani di separazione, permette di risolvere problemi di classificazione complessi, anche non linearmente separabili.

#### Esempio di classificazione concava

Un esempio di problema di classificazione non lineare è quello concavo, dove la separazione tra le classi presenta una concavità.

## *Feature* e rappresentazioni

Ogni componente della regressione logistica crea una nuova *feature* in uno spazio multidimensionale.

I diversi *layer* della struttura agendica rappresentano rappresentazioni di quello che c'è nello spazio precedente, combinando e trasformando le informazioni.

### Combinazione di regressori logistici

Il principio di questa struttura è la combinazione a cascata di più regressori logistici.

Ogni regressore logistico crea una rappresentazione alternativa dello spazio precedente, in un nuovo spazio con informazioni combinate.

Con l'aggiunta di più *layer*, le rappresentazioni diventano sempre più complesse, permettendo di modellare relazioni non lineari tra i dati e di ottenere classificazioni più accurate.

## Funzioni di Costo e Reti Neurali

**Funzioni di Costo Complesse**
La complessità di una funzione di costo dipende dalla sua definizione. Nel caso specifico, la funzione di costo utilizzata non è complessa.

**Combinazioni Lineari**
La **somma ponderata** tra le colonne, che rappresenta un'unità o un polo, implica sempre una combinazione lineare. 

**Estensioni della Regressione Logistica**
La regressione logistica, una classificazione binaria con separazione lineare, può essere estesa a più classi e relazioni non lineari.

**Estensione a Più Classi**
Per estendere la regressione logistica a più classi, si calcola un vettore associato a ciascuna classe. Ad esempio, per la classe k, si calcola il vettore vk. Il fattore di peso diventa una matrice con una riga per classe.
Le relazioni non lineari si ottengono componendo diverse componenti.

**Reti Neurali come Grafi di Computazione**
Una rete neurale può essere rappresentata come un grafo in cui:
* **Nodi:** rappresentano le operazioni eseguite.
* **Archi:** rappresentano le connessioni tra input e output delle operazioni.
Gli archi possono essere pesati.

**Espressione Algebrica**
A partire dalla rappresentazione a grafo, è possibile associare un'espressione algebrica alla rete neurale.
## Grafi di Computazione

Un grafo di computazione è un grafo in cui ogni nodo rappresenta un dato di input o un'operazione che lavora su dati di input. Gli archi rappresentano le connessioni tra queste operazioni.

**Esempio di Grafo di Computazione**

Consideriamo un grafo con due variabili di input, A e B. Le operazioni sono:

* **C = A + B/2**
* **D = 1/B**
* **E = C * D**

**Propagazione in Avanti ("Forward")**
Partendo dai valori di input, ad esempio A=1 e B=2, possiamo calcolare i valori di tutti i nodi propagando i valori all'interno del grafo. In questo caso, C = 2, D = 1/2 e E = 1.

**Propagazione all'Indietro ("Backward")**
Per calcolare il gradiente rispetto ad A o B, dobbiamo propagare all'indietro il gradiente attraverso il grafo.

**Gradiente e Propagazione all'Indietro**
Il gradiente è una componente essenziale nella regressione logistica e nel calcolo del gradiente. Per calcolare il gradiente rispetto ad A o B, dobbiamo partire dal nodo finale e propagare all'indietro il gradiente attraverso il grafo.

**Annotare il Grafo con i Gradienti**
Possiamo annotare il grafo con i gradienti su ogni arco. Questo ci permette di vedere la direzione della propagazione dell'informazione e come ci spostiamo all'indietro. Ad esempio, l'arco tra C ed E ci dice che il gradiente di E rispetto a C è D, poiché E = C * D.

## Propagazione all'Indietro e Gradienti

Se il grafo di computazione è annotato con i gradienti, la propagazione all'indietro ci permette di calcolare il gradiente di una variabile rispetto ad un'altra.

**Esempio di Calcolo del Gradiente**
Considerando il grafo di computazione precedente, il gradiente di E rispetto a C è 0.5, poiché il gradiente di E rispetto a C è D, che nel nostro caso è 0.5.

**Regola della Catena**
La regola della catena ci permette di combinare i gradienti. Ad esempio, per calcolare il gradiente di E rispetto ad A, seguiamo il percorso e moltiplichiamo il gradiente di E rispetto a C per il gradiente di C rispetto ad A.

**Cammini Alternativi**
In alcuni casi, ci possono essere più cammini per arrivare ad una variabile. Ad esempio, per calcolare il gradiente di E rispetto a B, possiamo passare da C e poi da D. In questo caso, dobbiamo sommare i contributi di entrambi i cammini.

**Importanza dei Grafi di Computazione**
I grafi di computazione sono importanti perché ci permettono di calcolare i gradienti in modo efficiente, anche per strutture complesse come le reti neurali.

**Reti Neurali e Grafi di Computazione**
Le reti neurali sono essenzialmente grafi di computazione. La funzione di predizione ŷ e la funzione di loss sono combinate in un grafo di computazione. La complessità delle reti neurali deriva dalla complessità di queste funzioni.

**Ottimizzazione dei Parametri**
Per ottimizzare i parametri del modello, si applica la discesa del gradiente. Il gradiente viene calcolato attraverso la propagazione all'indietro nel grafo di computazione.

## Reti Neurali e Grafi di Computazione

Le reti neurali possono essere rappresentate come grafi di computazione. Ogni nodo del grafo rappresenta un'operazione, come una somma ponderata o una funzione di attivazione. Gli archi rappresentano il flusso di dati tra i nodi.

**Esempio di Rete Neurale**
Consideriamo una rete neurale con due livelli, tre nodi intermedi e un nodo di output. La formula algebrica che rappresenta questa rete può essere associata ad un grafo di computazione.

**Parametri e Nodi Foglia**
I parametri della rete, come i pesi, sono rappresentati da nodi foglia nel grafo di computazione. Gli input sono rappresentati da altri nodi foglia.

**Propagazione in Avanti e all'Indietro**
Per applicare la discesa del gradiente, dobbiamo calcolare il grafo di computazione e ottenere i valori del gradiente associati ai parametri. Questo processo prevede la propagazione in avanti ("forward") e la propagazione all'indietro ("backward").

**Struttura Feed-Forward**
Le reti neurali più semplici sono di tipo *feed-forward*. Questo significa che i dati fluiscono in una sola direzione, dal primo livello all'ultimo.

**Livelli della Rete**
Le reti neurali *feed-forward* sono strutturate in livelli. Il primo livello è il livello di input, seguito da uno o più livelli nascosti e infine dal livello di output.

**Calcolo del Gradiente**
Per ottimizzare i parametri della rete, dobbiamo calcolare il gradiente della funzione di loss rispetto ai parametri. Questo viene fatto attraverso la propagazione all'indietro nel grafo di computazione.
![[4) Learning-20241015115658489.png]]

## Reti Neurali e Funzioni di Attivazione

Le reti neurali sono composte da diversi livelli, ognuno dei quali elabora i dati in ingresso e produce un output. Il primo livello è il livello di input, che riceve i dati grezzi. I livelli successivi sono chiamati livelli nascosti e il livello finale è il livello di output.

**Rappresentazione Vettoriale dei Livelli**
Ogni livello della rete neurale può essere rappresentato come un vettore. Il livello di input può essere un'immagine, un testo o un altro tipo di dati. I livelli nascosti sono combinazioni lineari del livello precedente, moltiplicate per una matrice di pesi e sommate a un vettore di bias.

**Operazione di Combinazione Lineare**
L'operazione di combinazione lineare tra il livello precedente e la matrice di pesi può essere espressa come:

$$Z = VW + B$$

dove:

* **Z** è il vettore di output del livello corrente
* **V** è il vettore di input dal livello precedente
* **W** è la matrice di pesi
* **B** è il vettore di bias

**Funzioni di Attivazione**
Le funzioni di attivazione sono essenziali per spezzare la linearità della rete neurale. Senza funzioni di attivazione, la rete sarebbe equivalente a una semplice funzione lineare, che non sarebbe in grado di apprendere modelli complessi.

**Motivazione per le Funzioni di Attivazione**
Le funzioni di attivazione introducono non linearità nella rete, permettendo di apprendere modelli più complessi. Senza funzioni di attivazione, l'output della rete sarebbe una semplice combinazione lineare degli input, che non sarebbe in grado di rappresentare relazioni non lineari tra i dati.

## Livelli e Espressività nelle Reti Neurali

L'aggiunta di livelli in una rete neurale aumenta la sua capacità di risolvere problemi complessi. Ogni livello rappresenta una trasformazione dei dati di input, arricchendoli con informazioni semantiche.

**Livelli e Complessità**
* **Livello 1:** Riusciamo a classificare situazioni separabili linearmente.
* **Livello 2:** Possiamo classificare situazioni rappresentate in forme complesse nello spazio associato.
* **Livello 3:** Possiamo esprimere qualsiasi forma, inclusi modelli convessi.

**Informazione Semantica**
Ogni livello aggiunge informazioni semantiche ai dati di input. La trasformazione da un livello all'altro arricchisce la rappresentazione dei dati, rendendola più informativa.

**Numero di Neuroni e Dimensionalità**
Il numero di neuroni in un livello determina la dimensionalità dello spazio di rappresentazione. Un numero maggiore di neuroni permette di catturare un numero maggiore di caratteristiche.

**Esempio: Classificazione di Forme**
Se il problema di classificazione riguarda un triangolo, potrebbero essere necessari 3 neuroni per costruire 3 iperpiani di separazione. Se il problema riguarda un pentagono, sarebbero necessari 5 neuroni.

**Feature Artificiali**
Ogni livello corrisponde a un insieme di feature artificiali. Queste feature sono create dalla rete neurale durante il processo di apprendimento.

## Livelli, Espressività e Funzioni di Attivazione

L'aggiunta di livelli in una rete neurale aumenta la sua capacità di apprendere modelli complessi. Ogni livello rappresenta una trasformazione dei dati di input, arricchendoli con informazioni semantiche.

**Livelli e Complessità**
* **3 livelli:** Riescono a catturare modelli con 3 lati, come i triangoli.
* **4 livelli:** Possono catturare modelli con 4 lati, come i quadrilateri.
* **5 livelli:** Possono catturare modelli con 5 lati, come i pentagoni.

**Ridondanza e Separazione Lineare**
Un livello può essere ridondante, ovvero non contribuire significativamente all'apprendimento. Ad esempio, se il problema è separabile linearmente, 3 livelli potrebbero semplicemente riprodurre la stessa retta di separazione.

**Potere Espressivo e Trade-off**
L'aggiunta di livelli aumenta il potere espressivo della rete, ma comporta un aumento della complessità computazionale. Ogni livello richiede una matrice di pesi, che aumenta il numero di parametri da apprendere.

**Trasformazioni e Composizioni**
Ogni livello può essere visto come una trasformazione dei dati di input. La rete neurale è una composizione di queste trasformazioni, che trasformano i dati da un livello all'altro.

**Funzioni di Attivazione**
Le funzioni di attivazione sono essenziali per spezzare la linearità tra i livelli. Introducono non linearità nella rete, permettendo di apprendere modelli più complessi.

**Tipi di Funzioni di Attivazione**
* **Classificazione binaria:** Funzione logistica
* **Classificazione multiclasse:** Funzione softmax
* **Regressione:** Funzione di identità

## Funzioni di Attivazione Intermedie

Abbiamo visto come le funzioni di attivazione siano cruciali per la capacità di una rete neurale di apprendere modelli complessi. Ma quali funzioni di attivazione utilizziamo tra i vari livelli?

**Funzioni di Attivazione Finali**
* **Classificazione binaria:** Funzione logistica
* **Classificazione multiclasse:** Funzione softmax
* **Regressione:** Funzione di identità

**Funzioni di Attivazione Intermedie**
Tra i vari livelli, possiamo utilizzare diverse funzioni di attivazione per introdurre non linearità nella rete. Alcune opzioni comuni includono:

* **Funzione logistica:** Può essere utilizzata anche come funzione di attivazione intermedia.
* **Tangente iperbolica:** Simile alla funzione logistica, ma con un range di output più ampio.
* **ReLU (Rectified Linear Unit):** Una funzione lineare per valori positivi e zero per valori negativi. Offre diversi vantaggi, come la semplicità computazionale e la riduzione del problema del gradiente che svanisce.

**Vantaggi della ReLU**
* **Semplicità computazionale:** La ReLU è più semplice da calcolare rispetto ad altre funzioni di attivazione, come la funzione logistica o la tangente iperbolica.
* **Riduzione del problema del gradiente che svanisce:** La ReLU non satura per valori positivi, il che aiuta a prevenire il problema del gradiente che svanisce durante l'addestramento.
