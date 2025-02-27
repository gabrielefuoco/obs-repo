## Language Modeling

Il *language modeling* è un task autoregressivo che si concentra sulla generazione di testo. L'input consiste in una sequenza di parole osservate, $x_1, ..., x_t$ (dove *t* rappresenta il time step). Il task consiste nel predire la parola successiva, $x_{t+1}$. Si assume che il vocabolario sia noto a priori e che il generatore campioni da esso secondo specifiche strategie.

Formalmente, data una sequenza di parole $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$, si calcola la distribuzione di probabilità della parola successiva $x^{(t+1)}$:

$$P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$$

dove $x^{(t+1)}$ può essere qualsiasi parola nel vocabolario $V = \{w_1, \ldots, w_{|V|}\}$.

La probabilità di generare un testo T può essere vista come il prodotto delle probabilità condizionate di osservare ogni parola, data la sequenza di parole precedenti:

$$
P(x^{(1)}, \ldots, x^{(T)}) = P(x^{(1)}) \times P(x^{(2)} \mid x^{(1)}) \times \cdots \times P(x^{(T)} \mid x^{(T-1)}, \ldots, x^{(1)})
$$

$$
= \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)})
$$

Il language modeling è importante non solo per la semplice predizione della parola successiva, ma anche per una vasta gamma di applicazioni nel campo del linguaggio naturale, tra cui:

* **Machine Translation:** Può essere considerato un caso particolare di language modeling, in quanto implica una logica di encoding nel linguaggio sorgente e decoding nel linguaggio target.
* **Speech Recognition:** La predizione di parole successive è fondamentale per la trascrizione accurata del parlato.
* **Spelling/Grammar Correction:** Il modello può identificare e correggere errori ortografici e grammaticali.
* **Summarization:**
 * **Estrattiva:** Evidenzia le frasi più importanti da un testo.
 * **Astrattiva:** Rimodula il testo originale creando un riassunto. Anche la summarization astrattiva può essere considerata un caso particolare di language modeling, poiché, data una sequenza di testo in input, genera una nuova sequenza di testo in output.

# N-gram Language Models

Un n-gram è una sequenza di *n* token consecutivi in un testo. I modelli n-gram stimano la probabilità della parola successiva contando le occorrenze di n-gram in un corpus.

Esempi:

* **Unigrammi:** "il", "gatto", "sedeva", "sul"
* **Bigrammi:** "il gatto", "gatto sedeva", "sedeva sul"
* **Trigrammi:** "il gatto sedeva", "gatto sedeva sul"
* **Four-grammi:** "il gatto sedeva sul"

Invece di considerare l'intero contesto precedente, si utilizza una finestra di *n-1* parole, semplificando il problema con l'assunzione di Markov: $x^{(t+1)}$ dipende solo dalle *n-1* parole precedenti. La probabilità condizionata è calcolata come:

$$
P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)}) \approx \frac{\operatorname{count}(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{\operatorname{count}(x^{(t)}, \ldots, x^{(t-n+2)})}
$$

che è il rapporto tra la frequenza di un n-gram e la frequenza del corrispondente (n-1)-gram nel corpus.

**Esempio (modello 4-gram):**

Testo: "gli studenti aprirono i"

Condizionando su "gli studenti aprirono":

* $P(w \mid \text{gli studenti aprirono}) = \frac{\text{count(gli studenti aprirono } w)}{\text{count(gli studenti aprirono)}}$

**Esempio numerico:**

Supponiamo che nel corpus:

* "gli studenti aprirono i" sia apparso 1000 volte
* "gli studenti aprirono i libri" sia apparso 400 volte
* $P(\text{libri} \mid \text{gli studenti aprirono i}) = 0.4$
* "gli studenti aprirono i compiti" sia apparso 100 volte
* $P(\text{compiti} \mid \text{gli studenti aprirono i}) = 0.1$

**Gestione della sparsità:**

**Problema 1: Numeratore = 0:** Se un n-gram non è presente nel corpus, la sua probabilità è zero.

**Soluzione (Smoothing):** Aggiungere un piccolo valore δ al conteggio di ogni parola nel vocabolario.

**Problema 2: Denominatore = 0:** Se un (n-1)-gram non è presente nel corpus, non è possibile calcolare la probabilità.

**Soluzione (Backoff):** Condizionare su un (n-1)-gram, o un (n-2)-gram, e così via, fino a trovare un n-gram con conteggio non nullo.

**Limiti:**

Aumentare *n* peggiora il problema della sparsità. Tipicamente, *n* non supera 5. Un *n* elevato porta a:

* **Sparsità:** Aumenta all'aumentare di *n*, causando una sottostima delle probabilità.
* **Granularità:** Diminuisce all'aumentare di *n*, riducendo la capacità del modello di catturare le dipendenze a lungo termine.

Di conseguenza, si ottengono probabilità poco informative e il testo generato, pur potendo essere grammaticalmente corretto, potrebbe mancare di coerenza e fluidità.

## Costruire un Language Model Neurale

**Input:** sequenza di parole $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$

**Output:** distribuzione di probabilità della parola successiva $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$

L'obiettivo è costruire un modello neurale basato su una finestra di contesto (window-based). ![[10)-20241118153136194.png]]

Ogni parola è rappresentata da un *embedding*. Questi embeddings vengono concatenati e trasformati tramite una matrice di parametri **U**. Questa trasformazione produce un vettore di punteggi (scores), dove il punteggio più alto corrisponde alla parola più probabile nel contesto.

**Distribuzione di output:**

$$\mathbf{y} = \text{softmax}(\mathbf{U} \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^{V}$$

dove:

* **y** è il vettore di probabilità per ogni parola nel vocabolario.
* **U** è la matrice di trasformazione.
* **h** è il vettore nascosto (hidden state).
* **b<sub>2</sub>** è il bias del layer di output.
* V è la dimensione del vocabolario.

**Layer nascosto:**

$$\mathbf{h} = f(\mathbf{W} \mathbf{e} + \mathbf{b}_1)$$

dove:

* **h** è il vettore nascosto.
* **f** è una funzione di attivazione (es. tanh, ReLU).
* **W** è la matrice di pesi del layer nascosto.
* **e** è il vettore degli embeddings concatenati.
* **b<sub>1</sub>** è il bias del layer nascosto.

**Embeddings concatenati:**

$$\mathbf{e} = [e^{(1)}, e^{(2)}, e^{(3)}, e^{(4)}]$$

**Parole/vettori one-hot:**

$$\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \mathbf{x}^{(3)}, \mathbf{x}^{(4)}$$

![[10) Language Modeling & Attention-20241122163410640.png]]

Gli embeddings delle parole (ottenuti da vettori one-hot), vengono concatenati e trasformati da una funzione non lineare `f` nel layer nascosto. Questa rappresentazione intermedia viene poi linearmente trasformata e, infine, la funzione softmax produce la distribuzione di probabilità sulla parola successiva.

## A Fixed-Window Neural Language Model

**Miglioramenti rispetto ai modelli n-gram:**

* Nessun problema di sparsità.
* Non è necessario memorizzare tutti gli n-gram osservati.

**Problemi rimanenti:**

* La finestra di contesto fissa è troppo piccola.
* Aumentare la dimensione della finestra aumenta esponenzialmente la dimensione della matrice dei pesi **W**.
* Nessuna finestra di dimensione fissa può essere sufficientemente grande per catturare tutte le dipendenze rilevanti nel linguaggio.
* Le parole$x^{(1)}$ e $x^{(2)}$ sono moltiplicate per pesi completamente diversi nella matrice **W**. Non c'è simmetria nel modo in cui gli input vengono elaborati.

**Necessità di un'architettura neurale in grado di elaborare input di lunghezza variabile.**

Questo modello, a differenza dei modelli n-gram, non richiede la memorizzazione delle frequenze di occorrenza degli n-gram.

**Problemi:**

* **Dimensione della matrice W:** Una finestra di contesto ampia implica una matrice di pesi **W** di grandi dimensioni, rendendo l'addestramento computazionalmente costoso e soggetto a overfitting.
* **Asimmetria nell'elaborazione dell'input:** Le parole nella finestra di contesto sono moltiplicate per pesi diversi nella matrice **W**. Questa asimmetria non tiene conto della simmetria intrinseca del linguaggio: l'ordine delle parole è importante, ma una rappresentazione simmetrica del contesto è spesso desiderabile, soprattutto per compiti come il riconoscimento di entità nominate (named entity recognition). L'ottimizzazione dei pesi in **W** avvantaggia le parole in posizioni diverse nella sequenza; le parole all'inizio della sequenza potrebbero essere processate con una parte della matrice meno ottimizzata rispetto a quelle alla fine.

## Reti Neurali Ricorrenti (RNN)

L'obiettivo delle Reti Neurali Ricorrenti (RNN) è condividere i pesi (**W**) per ogni parola nella sequenza di input. Questo permette di elaborare sequenze di lunghezza variabile utilizzando una struttura di rete di dimensione fissa.

![[10)-20241118153747550.png]]

L'output può essere generato ad ogni *timestep* o solo all'ultimo, a seconda del *task* specifico. Ad esempio, nell'analisi del sentiment, interessa solo l'output finale. In questo esempio, **W** contribuisce ad ogni passo, quindi la codifica del passo precedente influenza ogni *timestep* successivo. Un'architettura neurale che segue questo principio, prendendo in input una sequenza di parole, è detta rete neurale ricorrente.

**Distribuzione di output:**

$$y^{(t)} = \text{softmax}(h^{(t)} + b_o) \in \mathbb{R}^{|\mathcal{V}|}$$

dove:

* $y^{(t)}$ è la distribuzione di probabilità sulle parole del vocabolario $\mathcal{V}$ al *timestep* $t$.
* $h^{(t)}$ è lo stato nascosto al *timestep* $t$.
* $b_o$ è il bias dell'output.

**Stati nascosti:**

$$h^{(t)} = \sigma \left( \mathbf{W}_{hh} h^{(t-1)} + \mathbf{W}_{xo} \mathbf{e}^{(t)} + \mathbf{b}_h \right)$$

$$h^{(0)} \text{ è lo stato nascosto iniziale}$$

dove:

* $h^{(t)}$ è lo stato nascosto al *timestep* $t$.
* $\mathbf{W}_{hh}$ è la matrice dei pesi che connette lo stato nascosto precedente allo stato nascosto corrente.
* $\mathbf{W}_{xo}$ è la matrice dei pesi che connette l'embedding della parola allo stato nascosto corrente.
* $\mathbf{e}^{(t)}$ è l'embedding della parola al *timestep* $t$.
* $\mathbf{b}_h$ è il bias dello stato nascosto.
* $\sigma$ è una funzione di attivazione (es. tanh o ReLU).

**Word Embeddings:**

$$\mathbf{e}^{(t)} = \mathbf{E} \mathbf{x}^{(t)}$$

dove:

* $\mathbf{e}^{(t)}$ è l'embedding della parola al *timestep* $t$.
* $\mathbf{E}$ è la matrice di embedding.
* $\mathbf{x}^{(t)}$ è il vettore one-hot della parola al *timestep* $t$.

**Parole / Vettori One-hot:**

$$\mathbf{x}^{(t)} \in \mathbb{R}^{|\mathcal{V}|}$$

dove $\mathbf{x}^{(t)}$ è un vettore one-hot di dimensione $|\mathcal{V}|$ (dimensione del vocabolario).

**Nota:** questa sequenza di input potrebbe essere molto più lunga!

![[10) Language Modeling & Attention-20241122163654364.png]]

Se ogni $\mathbf{x}^{(t)}$ è un vettore di dimensione $|\mathcal{V}|$ con tutti 0 e un solo 1, abbiamo una codifica one-hot di ogni parola. Questa codifica viene utilizzata nella trasformazione descritta nell'immagine. Si noti che $\mathbf{W}_{hh}$ e $\mathbf{W}_{xo}$ rappresentano le due matrici di pesi: $\mathbf{W}_{hh}$ per le trasformazioni dallo stato precedente e $\mathbf{W}_{xo}$ per l'input corrente al passo *t*.

Ogni blocco al passo *t* prende in input la codifica della parola al passo *t* e l'output trasformato (moltiplicato per la sua matrice di pesi) del passo precedente. Ad ogni passo *t* otteniamo la codifica $h_t$.

**Pro:**

* **Simmetria dei pesi:** I pesi vengono applicati ad ogni *timestep*, garantendo simmetria nell'elaborazione della sequenza.
* **Dimensione del modello costante:** La dimensione del modello non aumenta con l'aumentare della lunghezza della sequenza di input.

**Contro:**

* **Lunghezza della sequenza limitata:** La lunghezza della sequenza non è arbitraria. Il modello ha difficoltà nell'elaborare sequenze lunghe a causa di un effetto di "perdita di memoria" (vanishing gradient problem). Quando si valuta la probabilità della parola successiva, si osserva un'attenuazione significativa dei valori delle probabilità delle parole precedenti nella sequenza.
* **Tempo di addestramento:** L'addestramento del modello RNN richiede tempi lunghi.

## Addestramento di un Modello Linguistico RNN

L'addestramento di un modello linguistico RNN richiede un ampio corpus di testo con sequenze di lunghezza variabile. Ad ogni *timestep*, il modello riceve in input una sequenza e predice la distribuzione di probabilità per la parola successiva. La funzione di costo (*loss*) ad ogni *timestep* `t` è la cross-entropy:

$$J^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w\in V} y^{(t)}_w \log(\hat{y}^{(t)}_w) = -\log \hat{y}^{(t)}_{x_{t+1}}$$

dove:

* $y^{(t)}$ è il vettore one-hot rappresentante la parola effettiva al passo (t+1).
* $\hat{y}^{(t)}$ è la distribuzione di probabilità predetta dal modello al passo *t*.
* V è il vocabolario.

La *loss* complessiva sull'intero training set è la media delle *loss* calcolate ad ogni *timestep* `t`. Si noti che ad ogni passo `t` si ha una predizione $\hat{y}^{(t)}$.

![[10)-20241118155458516.png]]

La *loss* totale è la somma cumulativa delle *loss* individuali ad ogni *timestep*.

Tuttavia, calcolare la *loss* e i gradienti sull'intero corpus contemporaneamente è computazionalmente troppo costoso in termini di memoria.

## Backpropagation Through Time (BPTT) per RNNs

Per addestrare una Recurrent Neural Network (RNN) si utilizza la Backpropagation Through Time (BPTT), una variante dell'algoritmo di backpropagation. BPTT calcola i gradienti dei pesi rispetto alla funzione di costo.

**Domanda:** Qual è la derivata di $J^{(t)}(\theta)$ rispetto alla matrice di pesi ripetuta $\boldsymbol{W}_{h}$?

**Risposta:**

$$
\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} }\mid_{(i)}
$$

**Spiegazione:**

Il gradiente rispetto ad un peso ripetuto (come $\boldsymbol{W}_{h}$ nella figura) è la somma dei gradienti calcolati ad ogni *timestep* in cui quel peso contribuisce al calcolo. In altre parole, è la somma di gradienti di forma identica calcolati ad ogni *timestep*.

![[10) Language Modeling & Attention-20241122163805792.png]]

Questo è un'applicazione della regola della catena, nello specifico la *regola della catena multivariabile*.

#### Derivata di una Funzione Composita

Data una funzione multivariabile $f(x,y)$ e due funzioni univariabili $x(t)$ e $y(t)$, la regola della catena multivariabile afferma:

$$\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

![[10) Language Modeling & Attention-20241122163959135.png]]

Per chiarire, se dovessimo calcolare la derivata parziale di una funzione composta $f(a(b(x)))$ rispetto a $x$, dovremmo applicare la regola della catena due volte: una volta per la funzione $a$ e una volta per la funzione $b$. Analogamente, nel caso delle RNN, ad ogni *timestep* dobbiamo valutare la derivata rispetto a $\boldsymbol{W}_{h}$. La derivata di $\boldsymbol{W}_{h}$ al passo *t* rispetto a $\boldsymbol{W}_{h}$ al passo *i* (con i ≤ t) è calcolata considerando la dipendenza di $J^{(t)}$ da $\boldsymbol{W}_{h}$ attraverso gli stati nascosti intermedi.

Quindi, ad un generico *timestep* `t`, per calcolare il gradiente rispetto a $\boldsymbol{W}_{h}$, dobbiamo propagare all'indietro il gradiente cumulativo fino all'inizio della sequenza, sfruttando il fatto che la matrice $\boldsymbol{W}_{h}$ rimane costante ad ogni passo. Questo processo di propagazione all'indietro dei gradienti attraverso il tempo è ciò che caratterizza la BPTT.

## Valutazione del Modello

Il language model viene valutato tramite la **Perplexity**. La perplexity è definita come:

$$\text{Perplexity} = \left( \prod_{t=1}^{T} \frac{1}{P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})} \right)^{1/T}$$

dove:

* $T$ è la lunghezza della sequenza.
* $P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})$ è la probabilità della parola $x^{(t+1)}$ dato il contesto delle parole precedenti.

La perplexity rappresenta l'inverso della probabilità geometrica media di predire correttamente le parole nel corpus. Un valore di perplexity inferiore indica un modello migliore.

* **Relazione con la cross-entropy:** La perplexity è equivalente all'esponenziale della cross-entropy loss media:

 $$\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)}) \right) = \exp(J(\theta))$$
 dove $J(\theta)$ è la cross-entropy loss media.

**Perplexity inferiore è migliore.**

## Vanishing Gradient

Nel calcolo ricorsivo delle derivate durante la backpropagation through time (BPTT), si possono incontrare probabilità molto piccole. Questo porta ad un problema di *vanishing gradient*: i gradienti diventano sempre più piccoli man mano che si procede indietro nel tempo, rendendo difficile l'addestramento di RNN su sequenze lunghe.

$$\quad\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \sigma'\left(W_{xh} h^{(t-1)} + W_{sx} x^{(t)} + b_{h}\right)$$

* Cosa succede se σ fosse la funzione identità, σ(x) = x?

 $$
 \begin{aligned}
 \frac{\partial h^{(t)}}{\partial h^{(t-1)}} &= \text{diag}\left(\sigma'\left(W_{xh} h^{(t-1)} + W_{sx} x^{(t)} + b_{h}\right)\right) W_{h} \\
 &= \boldsymbol{I} W_{h} \\
 &= W_{h}
 \end{aligned}
 $$
 In questo caso semplificato, la derivata dipende direttamente da $W_h$.

* Consideriamo il gradiente della loss $J^{(i)}(\theta)$ al passo `i`, rispetto allo stato nascosto $\boldsymbol{h}^{(j)}$ ad un passo precedente `j`. Sia $\ell = i - j$.

 $$
 \begin{aligned}
 \frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{t=j+1}^{i} \frac{\partial h^{(t)}}{\partial h^{(t-1)}} &\text{(regola della catena)}\\
 &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{t=j+1}^{i} W_{h} \\
 &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_{h}^{\ell}
 \end{aligned}
 $$
 Se $W_{h}$ ha autovalori con modulo minore di 1, allora questo termine diventa esponenzialmente piccolo all'aumentare di $\ell$, causando il *vanishing gradient*.

Definizione di $h^{(t)}$: applicazione di una funzione di attivazione (solitamente una funzione non lineare come la tangente iperbolica o la funzione ReLU) alla combinazione lineare dell'embedding dell'input al *timestep* `t`, del bias e della trasformazione dello stato nascosto al *timestep* precedente. La diagonalizzazione della derivata della funzione di attivazione per $W_h$ semplifica l'analisi del problema del *vanishing gradient*.

## Spiegazione della Derivata di J e il Problema del Vanishing Gradient

La derivata della funzione di costo J rispetto allo stato nascosto ad un passo precedente, come mostrato precedentemente, contiene il termine $W_h^\ell$, dove $\ell$ è la distanza temporale tra i due passi.

**Cosa c'è di sbagliato con $W_h$?**

* Consideriamo il caso in cui gli autovalori di $W_h$ siano tutti minori di 1 in modulo:

 * $\lambda_1, \lambda_2, \ldots, \lambda_n < 1$ (autovalori)
 * $\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n$ (autovettori)

* Possiamo riscrivere il gradiente usando gli autovettori di $W_h$ come base:

 * $\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(j)}} = \sum_{k=1}^n c_k \lambda_{k}^\ell \mathbf{q}_k$
 * Per grandi valori di $\ell$ (grandi distanze temporali), $\lambda_k^\ell$ si avvicina a 0, quindi il gradiente tende a 0. Questo è il *vanishing gradient*.

**Cosa succede con le funzioni di attivazione non lineari (quelle che usiamo normalmente)?**

* Il problema è sostanzialmente lo stesso, ma la dimostrazione richiede che $|\lambda_i| < \gamma$ per qualche $\gamma$ dipendente dalla dimensionalità e dalla funzione di attivazione σ. La condizione $|\lambda_i| < 1$ non è più sufficiente.

Quanto è vero che il *vanishing gradient* è causato dalla potenza $\ell$ di valori molto piccoli? È vero quando gli autovalori della matrice $W_h$ sono minori di 1 in modulo. Il gradiente al passo `i` può essere riscritto usando gli autovettori di $W_h$. Con autovalori minori di 1 in modulo, il gradiente approssima 0 per grandi valori di $\ell$, dimostrando teoricamente la possibilità del *vanishing gradient*. Nella pratica, questo problema esiste effettivamente.

Può accadere che gli autovalori di $W_h$ siano maggiori di 1 in modulo, e questo porta ad un effetto contrario: l'*exploding gradient*.

Se il gradiente diventa troppo grande, l'aggiornamento SGD diventa troppo grande:

$$\theta^{nuovo} = \theta^{vecchio} - \alpha \nabla_{\theta} J(\theta)$$

* Questo può causare **aggiornamenti errati**: si fa un passo troppo grande e si raggiunge una configurazione dei parametri anomala e con una *loss* elevata.
* Si pensa di aver trovato una collina da scalare, ma improvvisamente ci si ritrova in una pianura.

Nel caso peggiore, questo si tradurrà in **Inf** o **NaN** nella rete (e si dovrà riavviare l'addestramento da un checkpoint precedente).

**Gradient clipping:** se la norma del gradiente è maggiore di una certa soglia, si ridimensiona prima di applicare l'aggiornamento SGD.

##### Algoritmo 1: Pseudo-codice per il *norm clipping*

* **se** $\|\mathbf{g}\| >$ soglia **allora**
 * $\mathbf{g} \leftarrow \frac{\text{soglia}}{\|\mathbf{g}\|} \mathbf{g}$
* **fine se**

**Intuizione:** si fa un passo nella stessa direzione, ma più piccolo.

In pratica, ricordare di applicare il *gradient clipping* è importante, ma gli *exploding gradient* sono un problema più facile da risolvere rispetto al *vanishing gradient*. È un problema di divergenza del gradiente, e si risolve più facilmente del problema del *vanishing gradient*. È molto frequente e si risolve con operazioni di normalizzazione: può essere uno scaling tale per cui i valori abbiano norma pari a 1. In una rete ricorrente si usa spesso il *clipping*, che è un *thresholding* del gradiente. Si sceglie una soglia e ad ogni passo si ridimensiona il gradiente rispetto a questa soglia fissata.

## Risolvere il Problema del Vanishing Gradient

Il problema del *vanishing gradient* nasce perché il segnale del gradiente proveniente da *timestep* lontani è molto più piccolo del segnale proveniente da *timestep* vicini. Di conseguenza, i pesi del modello vengono aggiornati principalmente in base agli effetti a breve termine, trascurando gli effetti a lungo termine.

* Il problema principale è che per una RNN standard è difficile preservare le informazioni su molti *timestep*.
* In una RNN standard, lo stato nascosto viene costantemente riscritto:

$$h^{(t)} = \sigma (W_{hh} h^{(t-1)} + W_{xz} x^{(t)} + b)$$

* Per risolvere questo problema, si possono adottare due approcci principali:

 * **Utilizzare una memoria separata che viene aggiunta:** Questo è l'approccio utilizzato dalle LSTM (Long Short-Term Memory).
 * **Creare connessioni dirette e più lineari nel modello:** Questo è l'approccio utilizzato da tecniche come l'attention e le connessioni residuali.

Si necessita di un intervento architetturale: invece di riscrivere lo stato corrente considerando l'intera sequenza, si aggiorna lo stato rispetto a un contesto più breve, mantenendo separatamente un buffer che indica quanto utilizzare dal contesto precedente nella generazione delle nuove parole.

## Long Short-Term Memory (LSTM)

Le LSTM sono un tipo di RNN proposto come soluzione al problema del *vanishing gradient*. 
Sono diventate veramente famose dopo che Hinton le ha introdotte in Google nel 2013.

L'obiettivo è riprogettare una RNN con una sorta di memoria per migliorare la backpropagation. Introduciamo la notazione `c`, che rappresenta la cella di memoria, utilizzata per gestire le informazioni a lungo termine. 
Abilitiamo operazioni di lettura, scrittura e cancellazione di informazioni. La selezione di quali informazioni gestire è controllata da specifici *gate*. Questi *gate* sono vettori della stessa dimensionalità dello stato della cella; ad ogni *timestep*, il vettore dei *gate* sarà aperto o chiuso. I loro valori sono dinamici e cambiano in base all'input e al contesto.

![[10)-20241118164929500.png]]

Partendo dal basso, calcoliamo gli stati nascosti $h^{(t)}$ e le celle di memoria $c^{(t)}$. $h^{(t)}$ è una combinazione element-wise tra l'attivazione dello stato della cella (tanh) e $o^{(t)}$, l'*output gate* (filtro), che controlla quali parti della cella di memoria contribuiscono allo stato nascosto al passo `t`.

$c^{(t)}$ è la combinazione tra $c^{(t-1)}$ e il nuovo contenuto da inserire in memoria. Questa combinazione è controllata da due *gate*: `f` (*forget gate*) e `i` (*input gate*). Lo stato di memoria al passo `t` è la combinazione tra una parte dello stato di memoria al passo precedente e il nuovo contenuto, determinato trasformando l'input e combinandolo linearmente con lo stato nascosto al passo precedente. Il risultato è $\tilde{c}^{(t)}$.

Ogni *gate* è ottenuto come trasformazione non lineare (sigmoide) della combinazione lineare dell'input $x^{(t)}$ e dello stato nascosto $h^{(t-1)}$. Ogni *gate* ha parametri distinti.

![[10)-20241119095018808.png]]
![[10)-20241119095044205.png]]

**Come le LSTM risolvono il *vanishing gradient*?**

L'architettura LSTM facilita la preservazione delle informazioni su molti *timestep*. Ad esempio, se l'*forget gate* è impostato a 1 per una dimensione della cella e l'*input gate* a 0, l'informazione di quella cella viene preservata indefinitamente. Al contrario, è più difficile per una RNN standard imparare una matrice di pesi ricorrenti $W_{hh}$ che preservi le informazioni nello stato nascosto. In pratica, si ottengono circa 100 *timestep* invece di circa 7 con una RNN standard.

Esistono modi alternativi per creare connessioni all'interno della rete per preservare le dipendenze a lungo raggio: l'aggiunta di connessioni dirette è un'esigenza nelle RNN, che fanno passare il gradiente direttamente senza trasformazioni. L'input per un layer subisce una trasformazione con funzione d'identità. L'input di un layer si combina con l'output di quel layer: il gradiente può passare da un *timestep* a uno precedente, evitando il problema del *vanishing gradient*. Queste sono chiamate *skip connection* o connessioni residuali (ResNet).

* **Connessioni dense (DenseNet):** connettono ogni layer a ogni altro layer che lo segue.
* **Highway connections (HighwayNet):** invece di avere una funzione di attivazione identità, hanno un meccanismo di *gating* che determina quale parte far passare direttamente.

## Estensione con Bidirezionalità

Consideriamo un task di analisi del sentiment. L'ambiguità di un aggettivo come "terribly" in una frase richiede un contesto più ampio per una corretta interpretazione.

![[10)-20241119100108956.png]]

La bidirezionalità risolve questo problema. L'input viene elaborato da due layer RNN paralleli: uno che procede da sinistra a destra (forward) e uno da destra a sinistra (backward).

![[10)-20241119100151634.png]]

La bidirezionalità si ottiene combinando gli output delle due RNN. Per un task autoregressivo come il language modeling, la bidirezionalità non è direttamente applicabile perché la RNN backward non può accedere al futuro. Tuttavia, per task non autoregressivi, la bidirezionalità migliora la rappresentazione contestuale. Lo stato nascosto $h^{(t)}$ è la combinazione dei due stati: $h^{(t)}_{forward}$ e $h^{(t)}_{backward}$, ciascuno con i propri parametri.

**Nota:** le RNN bidirezionali sono applicabili solo se si ha accesso all'intera sequenza di input.

* Non sono applicabili al Language Modeling, perché in questo caso è disponibile solo il contesto sinistro.
* Se si ha l'intera sequenza di input (ad esempio, in un task di encoding), la bidirezionalità è molto potente (e dovrebbe essere usata per default).

Ad esempio, BERT (Bidirectional Encoder Representations from Transformers) è un potente sistema di rappresentazione contestuale pre-addestrato basato sulla bidirezionalità. Apprenderemo di più sui Transformer, incluso BERT, in seguito.

## Estensione con Multi-Layer RNN

Un'altra estensione è quella di aumentare la profondità della rete aggiungendo più layer.

**Le RNN sono già "deep" in una dimensione (si svolgono su molti *timestep*)**

* Possiamo renderle "deep" anche in un'altra dimensione applicando più RNN in sequenza – questa è una RNN multi-layer.
* Questo permette alla rete di calcolare rappresentazioni più complesse.
* Le RNN inferiori dovrebbero calcolare feature di basso livello, mentre quelle superiori feature di alto livello.

**Le RNN multi-layer sono anche chiamate RNN stacked.**

![[10)-20241119100706568.png]]

Questa architettura cattura relazioni di diverso ordine tra le parole: sfruttando la prossimità locale e globale, la rete migliora la capacità di catturare proprietà grammaticali e sintattiche nei layer inferiori, mentre i layer superiori ottengono embedding che catturano relazioni di alto livello (semantiche). Anche questa idea sarà parte integrante dei Transformer.

Per il Language Modeling, una rete di questo tipo potrebbe generare testi non solo grammaticalmente corretti, ma anche con maggiore coerenza linguistica.

Tipicamente non si superano i 3 layer per bilanciare la cattura di proprietà grammaticali e semantiche (4 è il massimo, ma potrebbe non valerne la pena rispetto a 3). Con *skip connection*, si possono raggiungere 8 layer, 12 per BERT e 24 per i Transformer specifici per l'encoding.

## Traduzione Automatica (Machine Translation)

La traduzione automatica è un task considerato particolarmente difficile fino al 2015. A differenza del language modeling (predizione della parola successiva), la traduzione automatica è un esempio di task in cui input e output hanno ruoli completamente diversi e richiedono la modellazione di una relazione tra due linguaggi. Un task simile è la summarization. È importante che le dimensioni dei dati di training siano comparabili sia per la lingua sorgente che per quella target.

**Idea Centrale:** Apprendere un modello probabilistico dai dati.

* Supponiamo di voler tradurre dal francese all'inglese.
* Vogliamo trovare la migliore frase inglese `y`, data la frase francese `x`.
* Questo equivale a trovare: $\arg\max_y P(y|x)$
* Utilizzando la regola di Bayes, possiamo scomporre questo problema in due componenti da apprendere separatamente $= \arg\max_y P(x|y)P(y)$

**Modello di Traduzione:** Modella come le parole e le frasi dovrebbero essere tradotte (fedeltà). Viene appreso da dati paralleli (coppie di frasi tradotte).

**Modello Linguistico:** Modella come scrivere correttamente in inglese (fluidità). Viene appreso da dati monolinguali (corpus di testo in inglese).

L'idea centrale è apprendere un modello probabilistico dai dati. Supponiamo di voler tradurre dal francese all'inglese. Data una frase francese `x`, vogliamo generare la migliore frase inglese `y`, massimizzando la probabilità $P(y|x)$. Questo equivale a massimizzare la probabilità congiunta $P(x, y) = P(x|y)P(y)$. Dobbiamo quindi apprendere due componenti:

1. **Il modello di traduzione:** apprende come le parole e le frasi dovrebbero essere tradotte. Viene addestrato su dati paralleli (coppie di frasi tradotte).
2. **Il modello linguistico:** modella la probabilità a priori di una frase in inglese. Viene addestrato su dati monolinguali (un grande corpus di testo in inglese).

I due modelli non vengono addestrati separatamente. Il modello linguistico non si basa solo sulle proprietà intrinseche del decoder addestrato su grandi corpus, ma deve essere condizionato all'input. Serve uno spazio di rappresentazione comune in cui l'input viene codificato in uno spazio denso (embedding della frase) e utilizzato per condizionare parola per parola la frase prodotta.

## Modello Seq2Seq

![[10)-20241119102537146.png]]

Un modello seq2seq utilizza due architetture RNN (o LSTM multi-layer) per la traduzione automatica. La prima RNN (encoder) codifica la frase di input (es. in francese). Il suo stato nascosto finale viene utilizzato per inizializzare lo stato nascosto iniziale $h^{(0)}$ della seconda RNN (decoder). Il decoder, ad ogni *timestep*, predice la prossima parola della frase di output (es. in inglese). Il primo input per il decoder è un token speciale "START", che indica l'inizio della frase. La predizione viene effettuata tramite una cross-entropy, massimizzando la probabilità di predire la parola corretta nella sequenza target.

Il decoder è autoregressivo: la predizione di ogni parola dipende dalle parole generate precedentemente. L'encoder potrebbe essere una rete bidirezionale per una migliore rappresentazione contestuale.

* L'idea generale è un modello encoder-decoder:
 * Una rete neurale (encoder) prende in input e produce una rappresentazione neurale.
 * Un'altra rete (decoder) produce l'output basandosi su questa rappresentazione.
 * Se input e output sono sequenze, si chiama modello seq2seq.
* I modelli seq2seq sono utili per molti task oltre alla traduzione automatica:
 * Summarization (testo lungo → testo corto)
 * Dialogo (turni precedenti → turno successivo)
 * Parsing (testo in input → albero sintattico come sequenza)
 * Generazione di codice (linguaggio naturale → codice Python)

**Il modello seq2seq è un esempio di modello linguistico condizionale:**

* **Modello linguistico:** perché il decoder predice la parola successiva della frase target `y`.
* **Condizionale:** perché le sue predizioni sono condizionate anche sulla frase sorgente `x`.

**La traduzione automatica (NMT) calcola direttamente $P(y|x)$:**

$$
P(y|x) = P(y_1|x) P(y_2|y_1, x) P(y_3|y_1, y_2, x) \ldots P(y_T|y_1, \ldots, y_{T-1}, x)
$$

Probabilità della prossima parola target, date le parole target precedenti e la frase sorgente `x`.

Si può addestrare un sistema NMT per ottenere un grande corpus parallelo. Tuttavia, esistono ricerche interessanti su "NMT non supervisionato", aumento dei dati, ecc.

Come si addestra un modello linguistico condizionale? I pesi dell'encoder e del decoder vengono aggiornati ad ogni step di backpropagation. Ci si aspetta una convergenza lenta e complessa, ma è inevitabile se si vuole condizionare il decoder all'encoder.

![[10)-20241119104448084.png]]

Il condizionamento sul decoder emerge come un collo di bottiglia: il decoder è condizionato dall'output globale dell'encoder. Ci si chiede se non sarebbe meglio che ad ogni passo del decoding ci sia un riferimento non solo globale, ma anche relativo a ciascun elemento della frase di input.

## Decoding Greedy

![[10)-20241119104913791.png]]

Il decoding greedy seleziona ad ogni *timestep* la parola con la probabilità più alta. Essendo greedy, non considera l'effetto delle scelte precedenti sulle predizioni future.

Un approccio più accurato sarebbe una ricerca esaustiva di tutte le possibili sequenze:

$$P(y|x)=\prod_{t=1}^T P(y_{t}|y_{1},\dots,y_{t-1},x)$$

**Idealmente, vogliamo trovare una traduzione `y` (di lunghezza T) che massimizzi:**

$$
P(y|x) = P(y_1|x) \cdot P(y_2|y_1, x) \cdot P(y_3|y_1, y_2, x) \cdots P(y_T|y_1, \dots, y_{T-1}, x)
$$

$$
= \prod_{t=1}^{T} P(y_t|y_1, \dots, y_{t-1}, x)
$$

Potremmo provare a calcolare tutte le possibili sequenze `y`, ma questo ha una complessità computazionale di $O(V^T)$, dove V è la dimensione del vocabolario e T è la lunghezza della sequenza. Questa complessità è proibitiva.**

## Beam Search

Per risolvere il problema della complessità computazionale della ricerca esaustiva, si utilizza la beam search.

**Idea centrale:** Ad ogni *timestep* del decoder, si mantengono le `k` traduzioni parziali più probabili (chiamate **ipotesi**).

* `k` è la **dimensione del beam** (in pratica, tra 5 e 10 nella traduzione automatica).

Un'ipotesi $y_{1}, \ldots, y_{t}$ ha uno **score**, che è il suo logaritmo di probabilità:

$$\text{score}(y_1, \ldots, y_t) = \log P_{LM}(y_1, \ldots, y_t | x) = \sum_{i=1}^{t} \log P_{LM}(y_i | y_1, \ldots, y_{i-1}, x)$$

* Gli score sono tutti negativi; uno score più alto è migliore.
* Si cercano le ipotesi con score più alti, mantenendo le prime `k` ad ogni *timestep*.

La beam search **non garantisce** di trovare la soluzione ottimale, ma è molto più efficiente della ricerca esaustiva.

Si mantengono le `k` traduzioni più probabili ad ogni *timestep*. Il termine "ipotesi" si riferisce a una frase candidata: abbiamo `k` ipotesi ad ogni *timestep*, dove `k` è un parametro. Ogni ipotesi al passo `t` ha uno score associato che tiene conto delle probabilità cumulate dei passi precedenti. La migliore traduzione corrisponderà all'ipotesi con lo score più alto. Non si cerca l'ottimo globale, ma si evita la ricerca esaustiva esplorando un albero di possibili traduzioni.

![[10)-20241119105722368.png]]

## BLEU (Bilingual Evaluation Understudy)

BLEU è una metrica utilizzata per valutare la qualità di una traduzione automatica confrontando l'output del traduttore con una o più traduzioni umane di riferimento (ground truth).

Calcola uno score di corrispondenza tra la traduzione generata e le traduzioni di riferimento, basandosi su una combinazione di precisioni n-gram (da 1-gram a 4-gram). Include una penalità per le traduzioni troppo corte (per evitare di ottenere punteggi alti con traduzioni molto brevi).

**Problemi con la precisione:**

* **Ripetizioni:** La precisione semplice può essere ingannata da traduzioni che ripetono le stesse parole.
* **Multiple frasi di riferimento:** Se ci sono più frasi di riferimento, la precisione deve essere calcolata considerando tutte le frasi.

**Precisione "clipped":**

* Si confronta ogni parola della frase predetta con tutte le frasi di riferimento.
* Se la parola corrisponde a una frase di riferimento, è considerata corretta.
* Il conteggio delle parole corrette è limitato al numero massimo di volte in cui quella parola appare nella frase di riferimento.

![[10)-20241119110907996.png]]

Il punteggio BLEU è il prodotto della precisione geometrica media e di una penalità di brevità.

**Vantaggi:**

* Calcolo rapido e facile da comprendere.
* Corrisponde al modo in cui un umano valuterebbe lo stesso testo.
* Indipendente dalla lingua.
* Può essere utilizzato quando si hanno più frasi di riferimento.

**Svantaggi:**

* Non considera il significato delle parole, solo le corrispondenze esatte.
* Ignora l'importanza delle parole e l'ordine delle parole.
* Ad esempio, "La guardia arrivò tardi a causa della pioggia" e "La pioggia arrivò tardi a causa della guardia" avrebbero lo stesso punteggio BLEU unigramma.

Per superare questi limiti, sarebbero necessari modelli multilingue più sofisticati che possano codificare gli embedding delle frasi nel loro complesso, invece di concentrarsi sulle singole parole. Questi modelli potrebbero fungere da "oracolo", valutando la qualità della traduzione in base al significato e al contesto, piuttosto che solo sulla corrispondenza lessicale.

## Attention Mechanism

Invece di basarsi sulla rappresentazione finale dell'encoder per condizionare il decoder (e quindi soffrire del *conditioning bottleneck*), l'attention mechanism permette ad ogni *timestep* del decoder di essere guidato da una combinazione pesata delle rappresentazioni di ogni parola dell'input.

![[10)-20241119111802849.png]]

L'attention mechanism è stato introdotto per risolvere il *conditioning bottleneck*, in cui il decoder è condizionato solo dall'output globale dell'encoder. L'obiettivo è creare connessioni dirette tra il decoder e l'encoder, permettendo al decoder di focalizzarsi su parti specifiche dell'input ad ogni *timestep*. Si dice che il decoder "attende" all'encoder.

Partendo dal token "START", vengono calcolati gli *attention scores*, che vengono poi aggregati per ottenere un vettore di contesto. Questo vettore viene concatenato con l'output del decoder al *timestep* corrente, influenzando la generazione della parola successiva. L'output dell'attention funge da riassunto delle informazioni dell'encoder che hanno ricevuto maggiore attenzione.

![[10)-20241119112324451.png]]
![[10)-20241119112330390.png]]
![[10)-20241119112338437.png]]
![[10)-20241119112402479.png]]
![[10)-20241119112206826.png]]

La scelta della parola successiva deriva dai contributi di ogni parte dell'encoder.

Abbiamo gli stati nascosti dell'encoder $\boldsymbol{h}_1, \ldots, \boldsymbol{h}_N \in \mathbb{R}^d$.

Al *timestep* `t`, abbiamo lo stato nascosto del decoder $\boldsymbol{s}_t \in \mathbb{R}^d$.

Calcoliamo gli *attention scores* $\boldsymbol{e}^t$ per questo *timestep*:

$$
\boldsymbol{e}^t = [\boldsymbol{s}_t^T \boldsymbol{h}_1, \ldots, \boldsymbol{s}_t^T \boldsymbol{h}_N] \in \mathbb{R}^N
$$

Applichiamo la softmax per ottenere la distribuzione di attenzione $\boldsymbol{\alpha}^t$ (una distribuzione di probabilità che somma a 1):

$$
\boldsymbol{\alpha}^t = \operatorname{softmax}(\boldsymbol{e}^t) \in \mathbb{R}^N
$$

Usiamo $\boldsymbol{\alpha}^t$ per calcolare una somma pesata degli stati nascosti dell'encoder, ottenendo l'output di attenzione $\boldsymbol{a}_t$:

$$
\boldsymbol{a}_t = \sum_{i=1}^N \alpha_{i}^t \boldsymbol{h}_i \in \mathbb{R}^d
$$

Infine, concateniamo l'output di attenzione $\boldsymbol{a}_t$ con lo stato nascosto del decoder $\boldsymbol{s}_t$ e procediamo come nel modello seq2seq senza attention:

$$
[\boldsymbol{a}_t; \boldsymbol{s}_t] \in \mathbb{R}^{2d}
$$

**Notazione:**

* $\boldsymbol{h}_1, \dots, \boldsymbol{h}_N$: vettori che rappresentano la codifica degli stati nascosti dell'encoder.
* $N$: lunghezza della frase di input.
* $\boldsymbol{e}^{(t)}$: *attention score*. $\boldsymbol{e}^{(t)}=[\boldsymbol{s}_{t}^T\boldsymbol{h}_{1},\dots,\boldsymbol{s}_{t}^T\boldsymbol{h}_{n}] \in \mathbb{R}^N$
* $\boldsymbol{s}_{t}$: stato nascosto del decoder al *timestep* `t`.
* $\boldsymbol{a}$ e $\boldsymbol{\alpha}$: coefficienti di attenzione. $\boldsymbol{\alpha}^{(t)}$ è la probabilità ottenuta con la softmax di $\boldsymbol{e}^{(t)}$. $\boldsymbol{a}$ è la combinazione pesata degli stati nascosti dell'encoder.

**L'attention mechanism migliora significativamente le prestazioni della NMT:**

* Permette al decoder di focalizzarsi su parti specifiche della frase sorgente.
* Fornisce un modello più "umano" del processo di traduzione.
* Risolve il problema del *bottleneck*.
* Aiuta a mitigare il problema del *vanishing gradient*.
* Fornisce interpretabilità.

Esistono diversi modi per calcolare $\boldsymbol{e} \in \mathbb{R}^N$ da $\boldsymbol{h}_1, \ldots, \boldsymbol{h}_N \in \mathbb{R}^d$ e $\boldsymbol{s} \in \mathbb{R}^d$:

* **Attention a prodotto scalare:** $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{h}_i \in \mathbb{R}$ (assume $d_1 = d_2$)
* **Attention moltiplicativa:** $\boldsymbol{e}_i = \boldsymbol{s}^T \boldsymbol{W} \boldsymbol{h}_i \in \mathbb{R}$ (con matrice di pesi $\boldsymbol{W}$)
* **Attention moltiplicativa a rango ridotto:** $\boldsymbol{e}_i = \boldsymbol{s}^T (\boldsymbol{U}^T \boldsymbol{V}) \boldsymbol{h}_i = (\boldsymbol{U} \boldsymbol{s})^T (\boldsymbol{V} \boldsymbol{h}_i)$ (con matrici di rango basso $\boldsymbol{U}$ e $\boldsymbol{V}$)
* **Attention additiva:** $\boldsymbol{e}_i = \boldsymbol{v}^T \tanh(\boldsymbol{W}_1 \boldsymbol{h}_i + \boldsymbol{W}_2 \boldsymbol{s}) \in \mathbb{R}$ (con matrici di pesi $\boldsymbol{W}_1$ e $\boldsymbol{W}_2$ e vettore di pesi $\boldsymbol{v}$)

L'attention mechanism coinvolge sempre:

1. **Calcolo degli *attention scores*** $\boldsymbol{e} \in \mathbb{R}^{N}$.
2. **Applicazione della softmax per ottenere la distribuzione di attenzione** $\boldsymbol{\alpha}$: $\boldsymbol{\alpha} = \operatorname{softmax}(\boldsymbol{e}) \in \mathbb{R}^{N}$.
3. **Calcolo della somma pesata dei valori usando la distribuzione di attenzione:** $\boldsymbol{a} = \sum_{i=1}^{N} \alpha_{i} \boldsymbol{h}_{i} \in \mathbb{R}^{d_{1}}$, ottenendo l'**output di attenzione** $\boldsymbol{a}$ (a volte chiamato vettore di contesto).

La dimensionalità dello stato nascosto può essere diversa per encoder e decoder, anche se in pratica spesso coincidono. Per architetture *encoder-only*, potrebbe essere inferiore. Gli stati dell'encoder ($\boldsymbol{h}$) fungono da coppie chiave-valore per le query ($\boldsymbol{s}$) del decoder. Abbiamo tante query quanti sono i *timestep* del decoder ($T$) e $N$ valori.

## L'Attention come Tecnica Generale di Deep Learning

Abbiamo visto l'attention mechanism applicato ai modelli seq2seq per la traduzione automatica. Tuttavia, l'attention è una tecnica molto più generale, applicabile a diverse architetture e task.

**Definizione generale dell'attenzione:**

Data una serie di vettori (valori) e un vettore di query, l'attenzione calcola una somma ponderata dei valori, dipendente dalla query.

* La somma ponderata è un riassunto selettivo delle informazioni contenute nei valori, dove la query determina su quali valori focalizzarsi.
* L'attenzione è un modo per ottenere una rappresentazione di dimensione fissa di un insieme arbitrario di rappresentazioni (i valori), dipendente da un'altra rappresentazione (la query).
* L'attenzione è diventata un potente e flessibile meccanismo per la manipolazione di puntatori e memoria in tutti i modelli di deep learning.

A volte si dice che la query "si concentra" sui valori.

Ad esempio, nel modello seq2seq con attention, ogni stato nascosto del decoder (query) si concentra su tutti gli stati nascosti dell'encoder (valori). Abbiamo visto la *cross-attention*. L'attention non è limitata alle architetture seq2seq.

Possiamo intendere l'attention come una tecnica generale per calcolare una somma pesata di valori, condizionata a una query. Questa somma è un riassunto selettivo delle informazioni contenute nei valori (stati dell'encoder), mentre la query determina su quali valori concentrarsi durante la generazione.

