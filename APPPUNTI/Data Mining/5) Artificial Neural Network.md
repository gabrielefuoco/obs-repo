
Le reti neurali artificiali sono un modello astratto ispirato al funzionamento del sistema nervoso umano. Sono costituite da un insieme di nodi (neuroni) connessi tra loro tramite collegamenti (assoni e dendriti). Ogni connessione ha un peso associato che rappresenta la forza della sinapsi.

Analogamente ai neuroni biologici, i nodi raccolgono segnali in ingresso, li sommano e, se superano una certa soglia, si attivano trasmettendo un segnale in uscita. I segnali possono essere eccitatori o inibitori.


### Componenti Principali:
* **Nodi o unità (neuroni)**
* **Collegamenti diretti (assoni e dendriti)**
* **Pesi associati ai collegamenti (forza delle sinapsi)**
* **Soglie o livelli di attivazione dei nodi**

#### La progettazione di una Rete Neurale richiede:
* La scelta del numero e del tipo di unità
* La determinazione della struttura morfologica
* Codifica degli esempi di addestramento, in termini di ingressi e uscite dalla rete
* L’inizializzazione e l’addestramento per la determinazione dei pesi delle interconnessioni, attraverso il training set

### Caratteristiche delle Reti Neurali
* Le istanze sono rappresentate mediante molte feature a molti valori, anche reali
* La funzione obiettivo può essere a valori reali
* Gli esempi possono essere molto rumorosi
* I tempi di addestramento possono essere lunghi
* La valutazione della rete appresa deve poter essere effettuata velocemente
* Non è cruciale capire la semantica della funzione attesa

### Tipi di Modelli
* **Modello Biologico**: ha l’obiettivo di imitare sistemi neurali biologici come le funzionalità auditive e visive.
	* Rispetto ai neuroni biologici, le reti neurali artificiali hanno una connettività molto maggiore ma tempi di commutazione più lenti (millisecondi invece di nanosecondi).
* **Modello guidato dalle applicazioni**: caratterizzato da un'architettura condizionata dalle necessità applicative
---
## Perceptron

Il Perceptron è una delle prime e più semplici reti neurali, introdotto da Rosenblatt nel 1962. Ha un'architettura a singolo strato con:

- Nodi di ingresso (uno per ogni attributo $x_i$)
- Un singolo nodo di uscita
- Collegamenti pesati $w_i$ che connettono i nodi di ingresso a quello di uscita, simulando le sinapsi

Il nodo di uscita calcola la somma pesata degli input più un bias $b$ (soglia), e applica la funzione di attivazione "segno" per produrre l'output $y$:

$$y = \begin{cases}
+1 & \text{se } w^T x + b > 0 \\
−1 & \text{altrimenti}
\end{cases}$$

La funzione di attivazione non lineare è fondamentale, altrimenti la rete sarebbe una semplice funzione lineare.

In forma vettoriale:

$$y = sign(w^T x + b)$$

Il Perceptron è quindi un modello semplice che combina linearmente gli input pesati, applica una soglia e produce un output binario 0/1 o -1/+1.

I pesi $w_i$ e il bias $b$ vengono addestrati sui dati per risolvere problemi di classificazione binaria. 

## Addestramento del Perceptron

Dato un training set, siamo interessati all'apprendimento dei parametri $w$ e $b$. L'algoritmo di addestramento del Perceptron procede come segue:

1. **Inizializzazione:**
    - Inizializza un training set contenente coppie (input $x$, output atteso $y$).
    - Imposta $k = 0$ (contatore delle iterazioni).
    - Inizializza i pesi $w(0)$ con valori casuali.

2. **Iterazioni:**
    - Ripeti fino a quando l'errore medio sul training set non scende sotto una soglia $\gamma$:
    - $\frac{\sum_{i=1}^n|y_{i}-f(\tilde{w}^{(k)},x_{i})|}{n}$
        - **Per ogni esempio (x, y) nel training set:**
            - Calcola l'output stimato $f(\tilde{w}^{(k)},x_{i})$ con i pesi correnti.
            - **Per ogni peso $w_j$:**
                - Aggiorna il peso con la regola:
                $$w_j^{(k+1)} = w_j^{(k)} + \lambda (y_{i} f(\tilde{w}^{(k)},x_{i})) x_j$$
                -  $k=k+1$.
        - Ritorna al passo 2.

**Note:**

- $w(k)_j$ è il peso della connessione $j$-esima dopo la $k$-esima iterazione.
- $\lambda$ è il learning rate, compreso tra 0 e 1, che controlla l'entità dell'aggiornamento dei pesi ad ogni iterazione.
- $x_j$ è il valore del $j$-esimo attributo dell'esempio $x$ del training set.



L'aggiornamento dei pesi avviene tramite una regola di discesa del gradiente stocastica, con passo $\lambda$, correggendo i pesi in modo proporzionale all'errore commesso su quell'esempio. 

### Aggiornamento dei Pesi

L'aggiornamento dei pesi si basa sull'errore tra l'output atteso $y_i$ e l'output stimato $f(\tilde{w}^{(k)},x_{i})$.

* **Se errore = 0**, non viene fatto alcun aggiornamento:
$w_{j}^{(k+1)}=w_{j}^{(k)}+\lambda*0*x_{ij}=w_{j}^{(k)}$

* **Se errore > 0**, il peso viene incrementato per aumentare $f(w, x)$.

* **Se errore < 0**, il peso viene decrementato per diminuire $f(w, x)$.

Quindi i pesi vengono spostati nella direzione che riduce l'errore, con un'entità proporzionale al *learning rate* $\lambda$, all'errore commesso e al valore dell'attributo $x_{ij}$.

Questo consente di adattare iterativamente i pesi per minimizzare l'errore complessivo sul training set.

Il percettrone è un semplice modello di classificazione lineare che impara a separare due classi nello spazio degli attributi tramite un iperpiano decisionale.

#### Obiettivo
L'obiettivo è trovare i pesi $(w_0,\dots,w_n)$ ottimali che minimizzano gli errori di classificazione. Lo spazio delle ipotesi è infinito, dato da tutte le possibili assegnazioni dei pesi.

#### Equazione dell'Iperpiano di Separazione
L'equazione dell'iperpiano di separazione appreso è: $w \cdot x + b = 0$

#### Convergenza
Il percettrone converge e apprende correttamente **solo se le classi sono linearmente separabili**.

Se le classi non sono separabili linearmente, l'algoritmo non converge perché nessun iperpiano può separarle perfettamente.

Questa è la principale **limitazione del percettrone**: non può risolvere problemi di classificazione del mondo reale in cui le classi hanno una separazione non lineare. Solo problemi linearmente separabili possono essere appresi correttamente.

### Teorema di Convergenza del Perceptron (Rosemblatt, 1962)

Il teorema di convergenza del perceptron assicura che il perceptron riuscirà a delimitare le 2 classi se il sistema è linearmente separabile.

In altre parole, nell'ottimizzazione non esistono minimi locali. Ciò significa che il perceptron convergerà sempre verso la soluzione ottimale, senza rimanere bloccato in minimi locali.


### Funzioni Booleane Primitive

I percettroni possono rappresentare tutte le funzioni booleane primitive, tra cui:

* **AND**: $and(x1, x2) = sign(-0.8 + 0.5x1 + 0.5x2)$
* **OR**: $or(x_1, x2) = sign(-0.3 + 0.5x1 + 0.5x2)$

Lo **XOR** non è linearmente separabile, quindi non può essere rappresentato da un percettrone. Ciò significa che i percettroni non possono rappresentare tutte le funzioni booleane, ma solo quelle che sono linearmente separabili.


## Reti Neurali Multilivello

### Introduzione

Le reti neurali multilivello sono un'architettura più complessa rispetto al percettrone di base, in grado di risolvere problemi di classificazione non lineari. Questa architettura organizza i nodi in strati (layer), ciascuno operante sulle uscite del livello precedente, rappresentando diversi livelli di astrazione applicati in modo sequenziale alle features di input per generare l'output finale.

### Strati Principali

I principali strati di una rete neurale multilivello sono:

* **Input Layer**: rappresenta gli attributi di input, con un nodo per ogni attributo numerico/binario o uno per ogni valore categoriale.
* **Hidden Layers**: strati intermedi costituiti da nodi nascosti che elaborano i segnali ricevuti dai livelli precedenti e producono valori di attivazione per il livello successivo.
* **Output Layer**: strato finale che elabora i valori di attivazione dal livello precedente per produrre le previsioni di output, con un nodo per la classificazione binaria o $\frac{k}{\log_{2}(k)}$ di per problemi multiclasse.

### Tipi di Reti Neurali

* **Reti Neurali Feedforward**: i segnali vengono propagati solo in avanti dall'input all'output. Una differenza chiave rispetto ai percettroni è l'inclusione di strati nascosti, che migliora notevolmente la capacità di rappresentare confini decisionali complessi.

### Esempio: Problema XOR

* Un singolo percettrone non può risolvere il problema XOR, ma una rete neurale con uno strato nascosto di due nodi riesce a gestire. Ogni nodo nascosto agisce come un percettrone che tenta di costruire un iperpiano, mentre il nodo di output combina i risultati per produrre il confine decisionale finale.

### Apprendimento di Caratteristiche

* Gli strati nascosti catturano caratteristiche a diversi livelli di astrazione: il primo strato opera direttamente sugli input catturando caratteristiche semplici, mentre strati successivi combinano queste per costruire caratteristiche via via più complesse.
* Così le reti apprendono una gerarchia di caratteristiche che vengono infine combinate per fare previsioni.

Si consideri l’i-esimo nodo dell’l-esimo livello della rete, dove i layer sono numerati da 0 (input layer) a L (output layer). Il valore di attivazione $a_{i}^l$ generato in corrispondenza di tale nodo può essere rappresentato in funzione degli input ricevuti dal nodo situato nel layer precedente ($l − 1 $in questo caso).

$$a_i^l = f(z_i^l) = f(\sum_j w_{ij}^l a_j^{l-1} + b_i^l)$$

* $w_{ij}^l$ è il peso del collegamento tra il j-esimo nodo nel layer $l-1$ e l'i-esimo nodo situato nel layer $l$.
* $b_i^l$ è il termine di bias nel nodo i-esimo.
* $z_i^l = \sum_j w_{ij}^l a_j^{l-1} + b_i^l$ è noto come predittore lineare
* $f()$ è la funzione di attivazione che ha il compito di convertire $z$ in $a$.
		Si osservi che per definizione $a_j^0 = x_j$ e $a^L = \hat{y}$

## Funzioni di attivazione
Esistono numerose funzioni di attivazione alternativamente alla funzione segno, che possono essere utilizzate all'interno delle reti neurali multi-livello. Ad esempio funzione gradino, funzione sigmoidea e funzione tangente iperbolica.

### Funzione gradino:

$$gradino(x) = \begin{cases} 
1 \text{ se } x > t \\
0 \text{ altrimenti}
\end{cases}$$

### Funzione sigmoidea:

$$sigmoid(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$$
	![[5) Artificial Neural Network-20241004155837855.png|197]]
- Si tratta di una funzione derivabile
- Non è usata negli hidden layer della rete
- Utile in caso di classificazione binaria
### Funzione tang. Iperbolica:

$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

![[5) Artificial Neural Network-20241004155903346.png|218]]
- Quando deriviamo abbiamo un update del peso più elevato rispetto alla funzione sigmoidea
- I dati sono centrati verso lo 0, rendendo l'apprendimento più facile
- Usata nel livello di output


### Funzione RELU

La funzione RELU (Rectified Linear Unit) è una funzione di attivazione comunemente utilizzata nelle reti neurali. La sua definizione è la seguente:

$$f(x) = \max(0, x)$$

La derivata della funzione RELU è:

$$\frac{\delta f(x)}{\delta x} = \begin{cases}
1 & x > 0 \\
0 & x < 0
\end{cases}$$

Nota che la funzione RELU non è derivabile in 0.

### Funzione Vanishing Gradient Problem

La funzione Vanishing Gradient Problem si riferisce al problema della diminuzione della grandezza degli aggiornamenti dei pesi nelle reti neurali, in particolare quando si utilizzano funzioni di attivazione come la sigmoide. Ciò può portare a una convergenza lenta o addirittura a un blocco della rete neurale.

### Funzione Soft Max

La funzione Soft Max è una funzione di attivazione comunemente utilizzata nel livello di output delle reti neurali, in particolare quando si hanno più di due classi. La sua definizione è la seguente:

$$f(x) = \frac{e^{z_i}}{\sum e^{z_k}}$$

dove:

* $f(x)$ è l'output dell'i-esimo neurone
* $z_i$ è il predittore lineare dell'i-esimo neurone
* $\sum e^{z_k}$ è la somma dei prodotti di tutti gli output

La funzione Soft Max è utilizzata per normalizzare gli output dei neuroni in modo che la somma degli output sia uguale a 1.

## Backpropagation per l'addestramento di reti neurali multi-livello

La funzione di perdita misura la differenza tra l'output predetto dalla rete ŷ e l'output atteso y sul training set.
$$
E(w,b)=\sum_{k=1}^n Loss(y_{l},\hat{y}_{k})
$$
Viene tipicamente usata la funzione di perdita dell'errore quadratico medio.

$$
Loss(y_{k},\hat{y}_{k})=(y_{k}-\hat{y}_{k})^2
$$

L'obiettivo è trovare i pesi w e bias b che minimizzino E(w, b).

#### Problema della non linearità
A causa della non linearità introdotta dalle funzioni di attivazione, E(w, b) non è una funzione convessa e può avere minimi locali.

#### Metodo della discesa del gradiente
Si può comunque applicare il metodo della discesa del gradiente per trovare un minimo locale, aggiornando iterativamente i pesi w e bias b nella direzione opposta al gradiente di E.

$$
\begin{aligned}
w_{ij}^l \leftarrow w_{ij}^l- \lambda  \frac{\delta E}{\delta w_{ij}^l} \\
b_{i}^l \leftarrow b_{i}^l- \lambda  \frac{\delta E}{\delta b_{i}^l}
\end{aligned}
$$

Il calcolo del gradiente di E rispetto ai pesi dei nodi nascosti non è banale. A questo scopo viene introdotta la tecnica di backpropagation che permette di propagare il gradiente dagli output agli strati nascosti della rete, calcolando ricorsivamente le derivate parziali necessarie per l'aggiornamento dei pesi.

#### Derivata parziale della funzione di perdita

Innanzitutto, nota la funzione di perdita E che è semplicemente la somma delle perdite individuali, la sua derivata parziale può essere scomposta come somma di derivate parziali di singole perdite.


$$\frac{\delta E}{\delta w_{j}^l}=\sum_{k=1}^n \frac{\delta \ Loss(y_{k},\hat{y_{k}})}{\delta w_{j}^l}$$
Per semplificare il discorso considereremo solo le derivate della perdita alla k-esima istanza di training set. Utilizzando la regola della derivazione a catena, possiamo rappresentare le derivate parziali della perdita rispetto a $w_{ij}^l$ come:


$$\frac{\partial Loss}{\partial w_{ij}^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times \frac{\partial a_i^l}{\partial z_i^l} \times\frac{\partial z_i^l}{\partial w_{ij}^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times \frac{\partial a_i^l}{\partial z_i^l} x_j^{l-1}$$

Prendendo in considerazione la funzione sigmoidale come funzione di attivazione (Ricordiamo che deve essere differenziabile) si ottiene in funzione di $a_i^l$:

$$\frac{\partial Loss}{\partial w_{ij}^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times a_i^l (1-a_i^l) \times x_j^{l-1}$$

Formula analoga per le derivate parziali rispetto a $\delta b_i^l$ si ha la seguente:

$$\frac{\partial Loss}{\partial b_i^l} = \sum_k \frac{\partial Loss}{\partial a_i^l} \times a_i^l (1-a_i^l)$$

Quindi per determinare le derivate parziali abbiamo bisogno di calcolare $\delta_i^l$. Prendendo in considerazione come funzione di perdita l'errore quadratico (squared error) otteniamo per il livello di output L:

$$\delta^L=\frac{\partial Loss}{\partial a_i^L} = \frac{\delta(y-a^L)}{\delta a^L}= 2(a_i^L - y_i)$$

Osservando che $a_j^l$ influisce sul valore di attivazione $a_i^{l+1}$ dei nodi situati nel layer successivo, che a sua volta influisce la perdita, per un livello "hidden" (utilizzando la chain rule) si ottiene:

$$\frac{\partial Loss}{\partial a_j^l} = \sum_i (\frac{\partial Loss}{\partial a_i^{l+1}}  \times \frac{\partial a_i^{l+1}}{\partial z_i^{l+1}} \times  \frac{\partial z_i^{l+1}}{\partial a_j^l} )$$

$$\delta_j^l = \sum_i \left(\delta_i^{l+1} \times a_i^{l+1}(1-a_i^{l+1}) \times w_{ij}^{l+1}\right)$$

L'equazione precedente fornisce una rappresentazione sintetica di $\delta_j^l$ al livello $l$ in termini dei valori $\delta_i^{l+1}$ calcolati al livello $l+1$. Quindi, procedendo a ritroso dal livello di output $L$ ai livelli nascosti, possiamo applicare e calcolare ricorsivamente $\delta_i^l$ per ogni nodo nascosto. Successivamente utilizziamo $\delta_i^l$ per calcolare le derivate parziali della perdita rispetto a $w_{ij}^l$ e $b_i^l$.

1. $D_{train} = \{(x_k, y_k) | k = 1, 2, ..., n\}$ e inizializza $c = 0$
2. Inizializza $(w^{(0)}, b^{(0)})$ con valori randomici.
3. Finché $(w^{(c+1)}, b^{(c+1)})$ e $(w^{(c)}, b^{(c)})$ non convergono al medesimo valore
    (a) Per ogni record $(x_k, y_k) \in D_{train}$
        i. Calcola il set dei valori di attivazione $(a_i^l)_k$ utilizzando $x_k$
        ii. Calcola il set dei valori $(\delta_i^l)_k$ utilizzando backpropagation
        iii. Calcola $(Loss)_k, (Loss')_k$
    (b) Calcola $\frac{\partial E}{\partial w_{ij}^l} = \sum_k (\frac{\partial Loss}{\partial w_{ij}^l})_k$
    (c) Calcola $\frac{\partial E}{\partial b_i^l} = \sum_k (\frac{\partial Loss}{\partial b_i^l})_k$
    (d) Aggiorna $(w^{(c+1)}, b^{(c+1)})$ utilizzando la discesa del gradiente
    (e) Aggiorna $c = c + 1$
    (f) Go To 3


### Caratteristiche delle ANN
Le ANN a più strati sono approssimatori universali, ma possono presentare alcuni problemi:

* **Overfitting**: se la rete è troppo complessa, può adattarsi troppo bene ai dati di training e non generalizzare bene sui dati di test.
* **Minimo locale**: la discesa del gradiente può convergere in un minimo locale anziché nel minimo globale.
* **Tempo di costruzione del modello**: la costruzione del modello può richiedere molto tempo, ma il test può essere molto veloce.
* **Gestione degli attributi**: può gestire attributi ridondanti perché i pesi vengono appresi automaticamente.

### Limitazioni
Le ANN possono avere alcune limitazioni:

* **Sensibilità al rumore**: è sensibile al rumore nei dati del training set.
* **Gestione degli attributi mancanti**: può avere difficoltà a gestire attributi mancanti.