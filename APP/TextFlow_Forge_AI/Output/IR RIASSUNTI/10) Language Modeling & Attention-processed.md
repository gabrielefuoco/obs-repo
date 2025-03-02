
## Language Modeling: Un Riassunto

Il *language modeling* è un task autoregressivo che prevede la generazione di testo prevedendo la parola successiva ($x_{t+1}$) data una sequenza di parole precedenti ($x_1, ..., x_t$).  Formalmente, si calcola la probabilità condizionata:

$$P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$$

La probabilità di un intero testo è il prodotto delle probabilità condizionate di ogni parola:

$$ P(x^{(1)}, \ldots, x^{(T)}) = \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \ldots, x^{(1)}) $$

Questo ha ampie applicazioni, tra cui traduzione automatica, riconoscimento vocale, correzione ortografica e grammaticale, e riassunto (sia estrattivo che astrattivo, quest'ultimo considerato un caso particolare di language modeling).


## N-gram Language Models

Gli N-gram models semplificano il problema assumendo la proprietà di Markov: la parola successiva dipende solo dalle *n-1* parole precedenti.  La probabilità è stimata contando le occorrenze di n-gram in un corpus:

$$ P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(t-n+2)}) \approx \frac{\operatorname{count}(x^{(t+1)}, x^{(t)}, \ldots, x^{(t-n+2)})}{\operatorname{count}(x^{(t)}, \ldots, x^{(t-n+2)})} $$

**Esempio (4-gram):**  `P(w | gli studenti aprirono)` è calcolato come il rapporto tra il numero di volte in cui "gli studenti aprirono w" appare nel corpus e il numero di volte in cui appare "gli studenti aprirono".

**Gestione della Sparsità:**  La sparsità, ovvero la mancanza di conteggi per certi n-gram, è un problema significativo.

* **Problema 1 (Numeratore = 0):**  Si usa lo *smoothing*, aggiungendo un piccolo valore (δ) a ogni conteggio.
* **Problema 2 (Denominatore = 0):** Si usa il *backoff*, condizionando su n-gram più corti fino a trovare un conteggio non nullo.

**Limiti:** Aumentare *n* peggiora la sparsità;  tipicamente *n* non supera 5.

---

## Modelli Linguistici Neurali: Da Finestre Fisse a Reti Ricorrenti

Questo documento descrive l'evoluzione dei modelli linguistici neurali, partendo da un approccio basato su finestre di contesto fisse e arrivando alle reti neurali ricorrenti (RNN).

### Modello Linguistico Neurale a Finestra Fissa

Questo modello prevede l'input di una sequenza di parole  $x^{(1)}, x^{(2)}, \ldots, x^{(t)}$ e produce la distribuzione di probabilità della parola successiva $P(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})$.  Ogni parola è rappresentata da un embedding.  Gli embeddings di una finestra di contesto di dimensione fissa vengono concatenati in un vettore $\mathbf{e} = [e^{(1)}, e^{(2)}, e^{(3)}, e^{(4)}]$, trasformati tramite un layer nascosto $\mathbf{h} = f(\mathbf{W} \mathbf{e} + \mathbf{b}_1)$ (dove *f* è una funzione di attivazione come tanh o ReLU), e infine, tramite una softmax, si ottiene la distribuzione di probabilità sulla parola successiva: $\mathbf{y} = \text{softmax}(\mathbf{U} \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^{V}$.

![10)-20241118153136194.png]
![10) Language Modeling & Attention-20241122163410640.png]

Questo modello, pur superando la sparsità dei modelli n-gram, presenta limiti significativi: la dimensione della finestra di contesto è fissa e limitata, causando una perdita di informazioni sulle dipendenze a lungo termine; l'aumento della dimensione della finestra porta ad una crescita esponenziale della matrice dei pesi **W**, rendendo l'addestramento computazionalmente costoso e soggetto a overfitting;  infine, l'elaborazione asimmetrica delle parole nella finestra (pesi diversi per parole diverse) non riflette la simmetria intrinseca del linguaggio.

### Reti Neurali Ricorrenti (RNN)

Le RNN superano i limiti dei modelli a finestra fissa condividendo i pesi **W** per ogni parola nella sequenza di input.  Questo permette di elaborare sequenze di lunghezza variabile con una struttura di rete di dimensione fissa.

![10)-20241118153747550.png]

L'output può essere generato ad ogni timestep o solo all'ultimo, a seconda del task.  L'utilizzo di pesi condivisi risolve il problema della crescita esponenziale della matrice dei pesi e permette di catturare dipendenze a lungo termine più efficacemente rispetto ai modelli a finestra fissa.

---

Le reti neurali ricorrenti (RNN) elaborano sequenze di input, come frasi, considerando l'influenza di ogni passo precedente su quelli successivi.  Questo è ottenuto tramite uno stato nascosto ($h^{(t)}$) che si aggiorna ad ogni *timestep*  `t`.

**Meccanismo:**

* **Word Embeddings:** Ogni parola viene rappresentata da un vettore embedding ($\mathbf{e}^{(t)} = \mathbf{E} \mathbf{x}^{(t)}$), ottenuto moltiplicando la matrice di embedding ($\mathbf{E}$) per il vettore one-hot ($\mathbf{x}^{(t)}$) della parola.

* **Stati Nascosti:** Lo stato nascosto al *timestep* `t` è calcolato come:  $h^{(t)} = \sigma \left( \mathbf{W}_{hh} h^{(t-1)} + \mathbf{W}_{xo} \mathbf{e}^{(t)} + \mathbf{b}_h \right)$, dove $\mathbf{W}_{hh}$ e $\mathbf{W}_{xo}$ sono matrici di pesi, $\mathbf{b}_h$ è il bias, e $\sigma$ è una funzione di attivazione.  $h^{(0)}$ rappresenta lo stato nascosto iniziale.

* **Distribuzione di Output:** La distribuzione di probabilità sulle parole del vocabolario $\mathcal{V}$ al *timestep* `t` è data da: $y^{(t)} = \text{softmax}(h^{(t)} + b_o)$.

**Addestramento:**

L'addestramento avviene minimizzando la *cross-entropy* come funzione di costo.  Ad ogni *timestep* `t`, la *loss* è: $J^{(t)}(\theta) = CE(y^{(t)}, \hat{y}^{(t)}) = -\sum_{w\in V} y^{(t)}_w \log(\hat{y}^{(t)}_w) = -\log \hat{y}^{(t)}_{x_{t+1}}$, dove $y^{(t)}$ è il vettore one-hot della parola effettiva e $\hat{y}^{(t)}$ è la distribuzione di probabilità predetta. La *loss* totale è la media delle *loss* su tutti i *timestep*.  Il calcolo della *loss* su tutto il corpus contemporaneamente è computazionalmente proibitivo.

**Pro e Contro:**

**Pro:** Simmetria dei pesi applicati ad ogni *timestep*; dimensione del modello costante indipendentemente dalla lunghezza della sequenza.

**Contro:**  Lunghezza della sequenza limitata a causa del problema del *vanishing gradient*; tempi di addestramento lunghi.

---

## Riassunto: Backpropagation Through Time (BPTT) e Addestramento di RNN

Questo documento descrive l'addestramento delle Recurrent Neural Networks (RNN) tramite Backpropagation Through Time (BPTT) e le relative problematiche.

### BPTT e Calcolo dei Gradienti

BPTT è una variante della backpropagation utilizzata per addestrare le RNN.  Calcola i gradienti dei pesi rispetto alla funzione di costo.  Il gradiente rispetto ad un peso ripetuto, come la matrice  $\boldsymbol{W}_{h}$, è la somma dei gradienti calcolati ad ogni *timestep* in cui quel peso contribuisce:

$$ \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{h} }\mid_{(i)} $$

Questo deriva dall'applicazione della regola della catena multivariabile.  Il calcolo del gradiente ad un *timestep* `t` richiede la propagazione all'indietro del gradiente cumulativo fino all'inizio della sequenza, sfruttando la costanza di $\boldsymbol{W}_{h}$ ad ogni passo.  ![10) Language Modeling & Attention-20241122163805792.png] ![10) Language Modeling & Attention-20241122163959135.png]


### Valutazione del Modello: Perplexity

La performance di un language model viene valutata tramite la Perplexity, definita come:

$$\text{Perplexity} = \left( \prod_{t=1}^{T} \frac{1}{P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})} \right)^{1/T}$$

dove  `T` è la lunghezza della sequenza e  `P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)})` è la probabilità della parola successiva dato il contesto.  Una perplexity inferiore indica un modello migliore.  La perplexity è equivalente all'esponenziale della cross-entropy loss media:

$$\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)}) \right) = \exp(J(\theta))$$


### Vanishing Gradient Problem

Durante BPTT, probabilità molto piccole possono causare il *vanishing gradient problem*. I gradienti diminuiscono esponenzialmente procedendo indietro nel tempo, rendendo difficile l'addestramento su sequenze lunghe.  Questo è evidente considerando la derivata dello stato nascosto:

$$\quad\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \sigma'\left(W_{xh} h^{(t-1)} + W_{sx} x^{(t)} + b_{h}\right)$$

Se  σ è la funzione identità, la derivata diventa  $W_h$.  Considerando il gradiente della loss  $J^{(i)}(\theta)$  rispetto allo stato nascosto  $\boldsymbol{h}^{(j)}$  ad un passo precedente:

$$ \begin{aligned} \frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{t=j+1}^{i} \frac{\partial h^{(t)}}{\partial h^{(t-1)}} \\ &= \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_{h}^{\ell} \end{aligned} $$

dove  $\ell = i - j$. Se gli autovalori di $W_h$ hanno modulo minore di 1, questo termine diventa esponenzialmente piccolo all'aumentare di  $\ell$, causando il *vanishing gradient*.

---

Il *vanishing gradient* nelle reti ricorrenti è un problema che ostacola l'apprendimento a lungo termine.  Si verifica perché la derivata della funzione di costo rispetto allo stato nascosto di *timestep* precedenti contiene potenze della matrice di pesi $W_h$.

Se gli autovalori di $W_h$ sono minori di 1 in modulo ($\lambda_i < 1$), il gradiente, espresso nella base degli autovettori di $W_h$ come $\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(j)}} = \sum_{k=1}^n c_k \lambda_{k}^\ell \mathbf{q}_k$ (dove $\ell$ è la distanza temporale), tende a 0 per grandi $\ell$, causando il *vanishing gradient*.  Con funzioni di attivazione non lineari, la condizione si complica, richiedendo $|\lambda_i| < \gamma$ per un $\gamma$ dipendente dalla funzione di attivazione e dalla dimensionalità.

Il problema opposto, l'*exploding gradient*, si verifica quando gli autovalori di $W_h$ sono maggiori di 1 in modulo, causando aggiornamenti SGD eccessivi ($\theta^{nuovo} = \theta^{vecchio} - \alpha \nabla_{\theta} J(\theta)$), potenzialmente portando a valori `Inf` o `NaN` e richiedendo il riavvio dell'addestramento.

Per mitigare l'*exploding gradient*, si utilizza il *gradient clipping*: se la norma del gradiente $\|\mathbf{g}\|$ supera una soglia, viene ridimensionato a $\mathbf{g} \leftarrow \frac{\text{soglia}}{\|\mathbf{g}\|} \mathbf{g}$.  Questo mantiene la direzione del gradiente ma ne riduce l'ampiezza.  L'*exploding gradient* è generalmente più facile da gestire rispetto al *vanishing gradient*.  Il *vanishing gradient*, invece, è un problema intrinseco legato alla propagazione del gradiente attraverso lunghe sequenze temporali, e richiede soluzioni più complesse.

---

Le Recurrent Neural Networks (RNN) standard soffrono del problema del *vanishing gradient*, che impedisce loro di preservare informazioni su molti *timestep*.  L'aggiornamento dello stato nascosto,  $$h^{(t)} = \sigma (W_{hh} h^{(t-1)} + W_{xz} x^{(t)} + b)$$, riscrive costantemente le informazioni, trascurando gli effetti a lungo termine.

Per risolvere questo, si utilizzano due approcci principali: aggiungere una memoria separata (come nelle LSTM) o creare connessioni più dirette (come con l'attention e le connessioni residuali).  Le LSTM, introdotte da Hinton nel 2013, sono un esempio del primo approccio.

Le LSTM introducono una cella di memoria, `c`, che gestisce le informazioni a lungo termine, e tre *gate* (Forget, Input, Output) che controllano il flusso di informazioni nella cella.  Questi *gate*, rappresentati da vettori, determinano dinamicamente quali informazioni mantenere, aggiungere o utilizzare come output.  Il loro funzionamento è descritto dalle seguenti equazioni:

* **Forget Gate:** Decide quali informazioni dallo stato della cella precedente ($c^{(t-1)}$) devono essere dimenticate:
   $f^{(t)} = \sigma \left( W_f h^{(t-1)} + U_f x^{(t)} + b_f \right)$

* **Input Gate:** Decide quali parti del nuovo contenuto della cella devono essere scritte:
   $i^{(t)} = \sigma \left( W_i h^{(t-1)} + U_i x^{(t)} + b_i \right)$

* **Output Gate:** Decide quali parti della cella devono essere inviate allo stato nascosto:
   $o^{(t)} = \sigma \left( W_o h^{(t-1)} + U_o x^{(t)} + b_o \right)$

* **Nuovo Contenuto della Cella:** Calcola il nuovo contenuto potenziale:
   $\tilde{c}^{(t)} = \tanh \left( W_c h^{(t-1)} + U_c x^{(t)} + b_c \right)$

In sintesi, le LSTM migliorano le RNN gestendo le informazioni a lungo termine tramite una cella di memoria e *gate* che controllano il flusso di informazioni, superando così le limitazioni delle RNN standard nel trattamento di sequenze lunghe.

---

Le Long Short-Term Memory (LSTM) sono un tipo di rete neurale ricorrente (RNN) progettate per affrontare il problema del *vanishing gradient* presente nelle RNN standard.  Questo problema impedisce alle RNN standard di apprendere dipendenze a lungo termine nelle sequenze.

**Meccanismo LSTM:**  Le LSTM mantengono uno stato della cella ($c^{(t)}$) e uno stato nascosto ($h^{(t)}$). Lo stato della cella aggiorna il suo contenuto tramite tre *gate*:

* **Gate di dimenticanza ($f^{(t)}$):** decide quali informazioni dallo stato della cella precedente ($c^{(t-1)}$) devono essere dimenticate.
* **Gate di input ($i^{(t)}$):** decide quali informazioni del nuovo contenuto della cella ($\tilde{c}^{(t)}$) devono essere memorizzate.
* **Gate di output ($o^{(t)}$):** decide quali informazioni dallo stato della cella ($c^{(t)}$) devono contribuire allo stato nascosto.

L'aggiornamento dello stato della cella è dato da:  $c^{(t)} = f^{(t)} \circ c^{(t-1)} + i^{(t)} \circ \tilde{c}^{(t)}$, dove $\circ$ indica il prodotto elemento-wise. Lo stato nascosto è calcolato come: $h^{(t)} = o^{(t)} \circ \tanh(c^{(t)})$.  Ogni *gate* è il risultato di una trasformazione non lineare (sigmoide) di una combinazione lineare dell'input ($x^{(t)}$) e dello stato nascosto precedente ($h^{(t-1)}$).

**Risoluzione del *vanishing gradient*:** Le LSTM risolvono il *vanishing gradient* permettendo la preservazione delle informazioni su molti *timestep*.  Un *forget gate* impostato a 1 e un *input gate* a 0 preservano indefinitamente l'informazione nella cella.  Questo contrasta con le RNN standard, che tipicamente riescono a gestire solo circa 7 *timestep*, mentre le LSTM raggiungono circa 100.  Alternative per preservare le dipendenze a lungo raggio includono le *skip connection* (come in ResNet) e le *highway connections* (come in HighwayNet), che permettono al gradiente di fluire direttamente attraverso i layer.

**LSTM Bidirezionali:** Per migliorare la comprensione del contesto, le LSTM possono essere implementate in modo bidirezionale.  Due LSTM processano l'input in direzioni opposte (da sinistra a destra e da destra a sinistra), e i loro output vengono combinati per ottenere una rappresentazione contestuale più ricca.  Questo è particolarmente utile per task non autoregressivi, come l'analisi del sentiment, dove il contesto completo è disponibile.  ![[]] ![[ ]] ![[ ]] ![[ ]]

---

## Riassunto del testo sulle Reti Neurali Ricorrenti (RNN)

Questo testo tratta le RNN, focalizzandosi su due importanti estensioni: la bidirezionalità e l'architettura multi-layer.

### RNN Bidirezionali

Le RNN bidirezionali elaborano l'intera sequenza di input, considerando sia il contesto precedente che quello successivo.  Questo le rende molto potenti per task come l'encoding, come dimostrato da BERT (Bidirectional Encoder Representations from Transformers).  Tuttavia, non sono adatte al Language Modeling, dove è disponibile solo il contesto sinistro.

### RNN Multi-layer (Stacked)

Aggiungendo più layer alle RNN, si crea un'architettura "deep" in un'altra dimensione, permettendo di calcolare rappresentazioni più complesse. I layer inferiori estraggono feature di basso livello (prossimità locale), mentre quelli superiori feature di alto livello (semantiche).  Questa architettura migliora la capacità di catturare proprietà grammaticali e sintattiche, generando testi più coerenti.  Tipicamente si usano 3 layer, ma con *skip connection* si possono raggiungere profondità maggiori (es. 12 per BERT).  ![[]]

### Traduzione Automatica (Machine Translation)

La traduzione automatica mira a trovare la migliore frase in lingua target (`y`) data una frase in lingua sorgente (`x`), massimizzando $P(y|x)$.  Utilizzando la regola di Bayes, questo problema si scompone in due componenti:

1. **Modello di Traduzione:**  Apprende la corrispondenza tra parole e frasi nelle due lingue.  Viene addestrato su dati paralleli (coppie di frasi tradotte) e modella la probabilità $P(x|y)$.

2. **Modello Linguistico:** Modella la probabilità a priori di una frase nella lingua target ($P(y)$). Viene addestrato su dati monolinguali (corpus di testo nella lingua target) e garantisce la fluidità del testo tradotto.

Invece di essere addestrati separatamente, questi due modelli sono integrati, con il modello linguistico condizionato all'input.  L'obiettivo è apprendere un modello probabilistico congiunto $P(x, y) = P(x|y)P(y)$ dai dati.

---

# Riassunto del testo sui modelli Seq2Seq e la Traduzione Automatica

Il testo descrive i modelli Seq2Seq, utilizzati principalmente per la traduzione automatica ma applicabili anche a summarization, dialogo, parsing e generazione di codice.  Questi modelli sono composti da un encoder e un decoder, entrambi tipicamente reti RNN o LSTM. L'encoder codifica la frase di input in una rappresentazione vettoriale (embedding), mentre il decoder, inizializzato dallo stato finale dell'encoder, genera la frase di output parola per parola, in modo autoregressivo (ogni parola dipende dalle precedenti).  Il processo massimizza la probabilità condizionata  `P(y|x)`, dove `x` è la frase di input e `y` la frase di output.  La probabilità è calcolata come prodotto delle probabilità condizionate di ogni parola:

$$ P(y|x) = \prod_{t=1}^T P(y_{t}|y_{1},\dots,y_{t-1},x) $$

L'addestramento avviene tramite backpropagation, aggiornando i pesi dell'encoder e del decoder. Un limite del modello standard è il condizionamento globale del decoder sull'output dell'encoder; soluzioni più sofisticate cercano di migliorare questo aspetto.

![10)-20241119102537146.png]
![10)-20241119104448084.png]


Il *decoding greedy*, una strategia semplificata, seleziona ad ogni passo la parola con la probabilità più alta, ignorando le conseguenze future.  Questo approccio, pur essendo efficiente, è subottimale.  Una ricerca esaustiva, che massimizza `P(y|x)` considerando tutte le possibili sequenze, ha una complessità computazionale proibitiva di  `O(V^T)`, dove `V` è la dimensione del vocabolario e `T` la lunghezza della sequenza.

Per ovviare a questo problema, si utilizza la *beam search*.  Questa tecnica mantiene le `k` ipotesi (traduzioni parziali) più probabili ad ogni passo del decoder, riducendo la complessità computazionale e migliorando la qualità della traduzione rispetto al decoding greedy.

---

## Riassunto: Beam Search, BLEU e Attention Mechanism nella Traduzione Automatica

Questo documento descrive tre concetti chiave nella traduzione automatica: la *beam search*, la metrica *BLEU* e il meccanismo di *attention*.

### Beam Search

La beam search è un algoritmo di ricerca euristica utilizzato per trovare le traduzioni più probabili.  Mantiene le `k` ipotesi (traduzioni candidate) più probabili ad ogni *timestep*, dove `k` è la dimensione del beam (tipicamente tra 5 e 10). Lo *score* di un'ipotesi è il logaritmo della sua probabilità cumulativa, calcolato come somma dei logaritmi delle probabilità condizionate di ogni parola.  Sebbene non garantisca la soluzione ottimale, è computazionalmente più efficiente della ricerca esaustiva. ![[10)-20241119105722368.png]]

### BLEU (Bilingual Evaluation Understudy)

BLEU è una metrica per valutare la qualità di una traduzione automatica, confrontandola con una o più traduzioni di riferimento ("ground truth"). Calcola una precisione media geometrica "clipped" su n-gram (da 1 a 4), considerando il numero massimo di occorrenze di ogni parola nelle frasi di riferimento.  Include una penalità di brevità per evitare punteggi alti per traduzioni troppo corte. La formula per la precisione geometrica media è:  $$\prod_{n=1}^{N} p_n^{w_n}$$  e quella per la penalità di brevità è: $$\begin{cases} 1, & \text{if } c > r \\ e^{(1-r/c)}, & \text{if } c <= r \end{cases}$$ dove `c` è la lunghezza della traduzione candidata e `r` quella della traduzione di riferimento.  BLEU è facile da calcolare e intuitivo, ma presenta limiti: non considera il significato, solo le corrispondenze esatte, e ignora l'ordine delle parole.

### Attention Mechanism

L'attention mechanism risolve il *conditioning bottleneck* dei modelli encoder-decoder tradizionali, in cui il decoder dipende solo dall'output finale dell'encoder.  Permette al decoder di accedere direttamente alle rappresentazioni di ogni parola dell'input ad ogni *timestep*, tramite una combinazione pesata delle rappresentazioni dell'encoder.  In sostanza, il decoder "attende" a diverse parti dell'input a seconda del contesto, migliorando la qualità della traduzione. ![[10)-20241119111802849.png]]

---

L'attention mechanism, illustrato nel contesto dei modelli seq2seq per la traduzione automatica neurale (NMT), migliora significativamente le prestazioni permettendo al decoder di focalizzarsi su parti specifiche della frase sorgente.  Partendo dallo stato iniziale ("START"), il processo inizia calcolando gli *attention scores*  $\boldsymbol{e}^t = [\boldsymbol{s}_t^T \boldsymbol{h}_1, \ldots, \boldsymbol{s}_t^T \boldsymbol{h}_N] \in \mathbb{R}^N$, dove $\boldsymbol{h}_i$ sono gli stati nascosti dell'encoder e $\boldsymbol{s}_t$ è lo stato nascosto del decoder al *timestep* `t`.

Applicando la softmax a $\boldsymbol{e}^t$, si ottiene la distribuzione di attenzione $\boldsymbol{\alpha}^t = \operatorname{softmax}(\boldsymbol{e}^t) \in \mathbb{R}^N$.  Questa distribuzione viene utilizzata per calcolare una somma pesata degli stati nascosti dell'encoder, ottenendo l'output di attenzione $\boldsymbol{a}_t = \sum_{i=1}^N \alpha_{i}^t \boldsymbol{h}_i \in \mathbb{R}^d$.  Infine, $\boldsymbol{a}_t$ viene concatenato con $\boldsymbol{s}_t$,  $[\boldsymbol{a}_t; \boldsymbol{s}_t] \in \mathbb{R}^{2d}$, influenzando la generazione della parola successiva.  L'output di attenzione funge da riassunto contestuale delle informazioni dell'encoder. ![[10)-20241119112324451.png]] ![[10)-20241119112330390.png]] ![[10)-20241119112338437.png]] ![[10)-20241119112402479.png]] ![[10)-20241119112206826.png]]

Esistono diverse modalità per calcolare gli *attention scores*, tra cui l'attention a prodotto scalare, moltiplicativa (con e senza rango ridotto) e additiva, tutte basate su combinazioni lineari e non lineari di $\boldsymbol{h}_i$ e $\boldsymbol{s}_t$.  Il processo generale, comunque, prevede sempre il calcolo degli *attention scores*, l'applicazione della softmax per ottenere la distribuzione di attenzione e il calcolo della somma pesata degli stati nascosti dell'encoder per ottenere l'output di attenzione.

L'attention mechanism, oltre a migliorare le prestazioni della NMT (risolvendo il *bottleneck*, mitigando il *vanishing gradient* e fornendo interpretabilità), è una tecnica generale applicabile a diverse architetture e task di deep learning.  In generale, data una serie di vettori (valori) e un vettore di query, l'attention calcola una somma ponderata dei valori, dipendente dalla query, fornendo una rappresentazione di dimensione fissa di un insieme arbitrario di rappresentazioni.

---

L'attenzione, in deep learning, è un meccanismo fondamentale per la gestione di puntatori e memoria.  Funziona come una tecnica generale per calcolare una somma pesata di "valori", condizionata da una "query".  Questa somma rappresenta un riassunto selettivo dei valori, con la query che determina quali valori sono più rilevanti.  Un esempio è la *cross-attention* nei modelli seq2seq, dove gli stati nascosti del decoder (query) pesano gli stati nascosti dell'encoder (valori) per focalizzarsi sulle informazioni più pertinenti durante la generazione della sequenza di output.  L'attenzione, però, non è limitata ai modelli seq2seq, ma è applicabile a diverse architetture di deep learning.

---
