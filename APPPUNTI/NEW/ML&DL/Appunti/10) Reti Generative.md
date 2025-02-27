## Autoencoder (AE)

![[9) Stima di Densità-20241115110936012.png|469]]

Un autoencoder è una rete neurale che ha l'obiettivo di ricostruire il suo input. La dimensione dello spazio latente $z$ è inferiore alla dimensione dello spazio di input $x$ ($k<d$). La loss più comune in questo caso è l'errore quadratico medio tra l'ingresso e l'uscita: $\|x-\tilde{x}\|^2$.

L'autoencoder mappa l'input $x$ in uno spazio a più bassa dimensionalità $z$, chiamato anche spazio latente. Questo spazio latente rappresenta una rappresentazione compressa dei dati.

Le funzioni di codifica e decodifica sono definite come:

* **Codifica:** $z=ψ(x)$
* **Decodifica:** $x=\phi(x)$

**Applicazioni:**

* **Riduzione della dimensionalità:** Gli autoencoder possono essere utilizzati per ridurre la dimensionalità dei dati, trovando una rappresentazione più compatta.
* **Compressione (con perdita):** Gli autoencoder possono essere utilizzati per la compressione dei dati, con una certa perdita di informazioni.
* **Anomaly Detection:** Nel setting non supervisionato, addestriamo l'autoencoder sul dataset, in seguito si usa la *Loss* come *Score*, e dichiareremo come esempi anomali quelli che massimizzano la *Loss*.
* **Rete Generativa:** Capace di generare esempi sintetisci che assomigliano a quelli in cui sono addestrati.

## Reti Neurali Generative

A partire da un dataset, una rete neurale generativa riesce a prendere la distribuzione che ha generato i dati e sfruttarla per generare nuovi esempi chiamati "*Esempi sintetici*". 
Vogliamo che tali esempi siano indistinguibili da quelli reali.

Si può usare un *autoencoder* per generare esempi sintetici:

$$
\begin{cases}
\text{Genera un punto z' casualmente} \in \mathbb{R}^k \\
x'=\phi(z') \text{ pseudo-esempio}
\end{cases}
$$

Vorremmo valessero le seguenti proprietà:
- **Continuità:** Punti vicini nello spazio latente devono avere codifiche simili.
- **Completezza:** Ogni punto dello spazio latente, generato in accordo a una distribuzione fissata, dovrebbe avere de-codifica semanticamente significativa.

L'autoeconder non rispetta queste proprietà, dunque c'è bisogno di modificare la sua struttura.

### Variational Autoencoder (VAE)

Il punto x viene mappato su una distribuzione normale :
$$x\to N(\mu,\Sigma), \quad \Sigma=\begin{bmatrix}\sigma^2 &0 \\ 0 & \sigma^2_{k}\end{bmatrix}$$
Il decoder lavora su singoli punti, dunque bisogna campionare un punto $z$ dalla distribuzione $N(\mu,\Sigma)$. Il decoder riceverà in ingresso questo punto

![[10)-20241122084722796.png]]

Mappare una distribuzione ci serve per mappare la continuità. Se lasciamo l'autoencoder libero di apprendere le rappresentazioni nello spazio latente, la sua soluzione sarà di lasciarli separati.

Vogliamo forzare la rete ad avvicinare i punti tra di loro. Bisogna dunque aggiungere un termine di regolarizzazione alla loss, che misura quanto la distribuzione su cui stiamo mappando il nostro punto è diversa da una normale standard.

$$L(x,\tilde{x})=\|x-\tilde{x}\|^2+\beta \cdot KL(N(\mu_{x},\Sigma_{x}),N(\vec{0},I))$$

KL è L'entropia relativa, vale 0 se le due distribuzioni sono uguali: $KL(p,q)=E_{P}[\log  \frac{p}{q} ]$

![[10)-20241122091125417.png]]
In questo modo abbiamo garantito che le regioni si possano sovrapporre.

## Generative Adversarial Network (GAN)

Implementano il paradigma dell'*Adversarial Learning* (Apprendimento competitivo).

![[10) Reti Generative-20241122123855843.png|621]]

Il funzionamento di una Rete Generativa Adversaria (GAN) si basa sull'interazione tra due reti neurali: una rete generativa e una rete discriminativa.

#### Rete Generativa

La rete generativa riceve in input punti generati casualmente da una distribuzione di riferimento. Il suo obiettivo è generare un output che assomigli il più possibile ai punti del dataset reale.

#### Rete Discriminativa

La rete discriminativa riceve in input esempi provenienti da due sorgenti: il dataset reale e la rete generativa. Dato un esempio *x*, la rete deve restituire la probabilità che *x* appartenga alla distribuzione di probabilità reale $D(x) = Pr[x \in p_x]$.

#### Addestramento delle Reti

L'addestramento delle due reti avviene simultaneamente. Si utilizza un batch di esempi dal dataset reale e si aggiornano i pesi di entrambe le reti con l'obiettivo di:

* **Massimizzare la capacità discriminativa:** La rete discriminativa deve massimizzare la probabilità di distinguere correttamente gli esempi reali ($D(x) = 1$) da quelli generati dalla rete generativa ($D(\tilde{x}) = D(G(z)) = 0$).

* **Minimizzare la capacità discriminativa (per la rete generativa):** La rete generativa deve minimizzare la probabilità che la rete discriminativa riconosca gli esempi generati come sintetici.

Formalmente, l'addestramento può essere espresso come un gioco a due giocatori:

$$
\begin{cases}
\max: \quad \log D(x) + \log (1 - D(G(z))) \\
\quad \text{massimizza la probabilità di discriminare gli esempi sintetici dagli esempi reali} \\ \\
\min: \quad \log(1 - D(G(z))) \\
\quad \text{minimizza la probabilità che D riconosca gli esempi generati sinteticamente}
\end{cases}
$$

#### Soluzione Ottima

Addestrando le reti in questo modo, si può dimostrare che la soluzione ottima è raggiunta quando:

* La distribuzione di probabilità generata dalla rete generativa ($P_g$) converge alla distribuzione di probabilità del dataset reale ($p_x$).
* La rete discriminativa diventa inefficace, assegnando una probabilità di 0.5 a tutti gli esempi, indipendentemente dalla loro origine.

$$
\begin{cases}
P_{g} = p_{x} \\
D(x) = \frac{1}{2}
\end{cases}
$$

## Gestire le Ricompense

L'obiettivo dell'agente è massimizzare la ricompensa cumulativa. Questa ricompensa, indicata con $G$, dipende dalla politica $\pi$, dalla funzione di transizione $P$ e dalla funzione di ricompensa $R$, nel caso di un singolo agente, o anche dagli altri agenti nel caso di giochi multi-agente, dove si introduce la competizione o la cooperazione. 

Un esempio di definizione delle ricompense potrebbe essere:

$$R = \begin{cases}
+1 & \text{Vittoria} \\
\ \ \ 0 & \text{Posizione intermedia o Patta} \\
-1 & \text{Sconfitta}
\end{cases}$$

È importante notare che assegnare ricompense solo a situazioni di vittoria o sconfitta potrebbe non essere sempre ottimale. Una strategia di ricompense più sofisticata potrebbe fornire informazioni più ricche all'agente.

### Return (Guadagno)

Il *return* o guadagno ($G_t$) rappresenta la somma delle ricompense ricevute dall'agente a partire da un certo istante di tempo *t* fino alla fine dell'interazione.

$$G_t = r_t + r_{t+1} + \dots + r_T$$

I *task* di apprendimento per rinforzo possono essere classificati in due categorie:

$$
\text{Task:} \begin{cases}
\text{Episodico: L'interazione è divisa in episodi (con un chiaro stato finale).} \\ \\

\text{Continuo: Non c'è uno stato finale definito.}
\end{cases}
$$

Nei *task* episodici, dove esiste uno stato finale, il *return* è ben definito. In questi casi, l'interazione termina quando si raggiunge lo stato finale. Tipicamente si considera il *return* per ogni episodio.

Nei *task* continui, il guadagno può essere potenzialmente infinito. Inoltre, una ricompensa ottenuta immediatamente ha un valore maggiore di una ricompensa ottenuta in un futuro più lontano. Per questo motivo, si utilizza spesso il *discounted return*.

### Discounted Return

Per affrontare il problema del guadagno potenzialmente infinito nei *task* continui, si introduce un parametro chiamato **tasso di sconto** (discount rate): $\gamma \in [0, 1]$. Il *discounted return* è definito come:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^k r_T$$

dove $\gamma^k$ rappresenta il valore attuale di una ricompensa ottenuta dopo *k* istanti di tempo.

### Interprete

Nell'architettura di un sistema di apprendimento per rinforzo, tra l'ambiente e l'agente è spesso presente un **interprete**:

![[10) Reti Generative 2024-11-22 10.25.09.excalidraw]]

L'interprete elabora le sensazioni attuali e passate per produrre lo **stato**. Lo stato è una rappresentazione processata dei segnali dell'interprete, che deve riassumere tutte le informazioni rilevanti per l'agente al fine di prendere una decisione. Questo stato è detto **stato Markoviano**, in quanto contiene tutte le informazioni necessarie per prevedere il futuro, indipendentemente dal passato.

### Sato Markowiano

Ricompensa e stato futuro sono indipendenti dalla storia, ma dipendono solo dallo stato corrente e dall'azione compiuta.
$$
\begin{cases}
P(s_{t+1}|s_{t},a_{t})=P(s_{t+1}|s_{t},a_{t},s_{t-1},a_{t-1},\dots,s_{0},a_{0}) \\ \\

P(r_{t}|s_{t},a_{t})=P(r_{t}|s_{t},a_{t},s_{t-1},a_{t-1},\dots,s_{0},a_{0})
\end{cases}
$$
Gli approcci di Reinforcement Learning assumono che gli stati siano Markowiani.

### Processo Decisionale Markowiano (MDP)

Un **MDP** è una quintupla $<S,A,T,R,\gamma>$ dove:
- $S$ è lo spazio degli stati
- $A$ è l'insieme delle azioni
- $T:S \times A \times S \to[0,1]: T(s,a,s')$ 
	- È una funzione di transizione che restituisce la probabilità di passare in $s'$ dato che siamo in $S$ e eseguiamo $A$ 
- $R: S \times A \times S \to R'$ , dove $R'$ è l'insieme delle ricompense 
	- $R(s,a,s')$ ricompensa ottenuta quando da $s$ passiamo a $s'$ eseguendo l'azione $a$
- $\gamma \in [0,1]$ è il *discount factor*

Se $S$ e $A$ sono insiemi finiti, parliamo di *MDP finiti*.

## Value Functions

La maggior parte dei modelli di Reinforcement Learning si basa sulle *Value Functions*, che quantificano il vantaggio per l'agente di trovarsi in un determinato stato o di eseguire una specifica azione in uno stato.

### V-Value

Dato una policy $\pi$, il **V-Value** di uno stato $s$ è il valore atteso del *discounted return* a partire da quello stato:

$$V^\pi(s) = E_\pi[G_t | s_t = s] = E_\pi\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} | s_t = s \right]$$

Rappresenta il guadagno atteso, dato che al tempo *t* l'agente si trova nello stato *s*, seguendo la policy $\pi$.

### Q-Value

Dato una policy $\pi$, il **Q-Value** di una coppia stato-azione $(s, a)$ è il valore atteso del *discounted return* a partire da quello stato, eseguendo prima l'azione *a*:

$$Q^\pi(s, a) = E_\pi[G_t | s_t = s, a_t = a] = E_\pi\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} | s_t = s, a_t = a \right]$$

### Equazioni di Bellman

Le *Value Functions* possono essere espresse ricorsivamente tramite le equazioni di Bellman:

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) Q^\pi(s, a)$$

$$Q^\pi(s, a) = \sum_{s' \in S} T(s, a, s') [R(s, a, s') + \gamma V^\pi(s')]$$

dove:

* $T(s, a, s')$ è la probabilità di transizione dallo stato *s* allo stato *s'* eseguendo l'azione *a*.
* $R(s, a, s')$ è la ricompensa ottenuta passando dallo stato *s* allo stato *s'* eseguendo l'azione *a*.

Queste equazioni possono essere combinate per ottenere:

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} T(s, a, s') [R(s, a, s') + \gamma V^\pi(s')]$$

$$Q^\pi(s, a) = \sum_{s' \in S} T(s, a, s') \left[ R(s, a, s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s', a') \right]$$

Partendo da valori iniziali casuali di $V^\pi$ (o nulli se si tratta di stati finali), l'applicazione iterativa di queste equazioni converge ai veri valori di $V^\pi$ e $Q^\pi$.

### Interactive Policy Evaluation

Un metodo iterativo per calcolare $V^\pi$ è la *Interactive Policy Evaluation*:

$$
\text{Dati } \pi, T, R
\begin{cases}
v^0(s) = 0, & \forall s \in S \\ \\

v^{(k+1)}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} T(s, a, s') [R(s, a, s') + \gamma v^{(k)}(s')], & \forall s \in S
\end{cases}
$$

L'obiettivo finale è trovare la policy $\pi^*$ che massimizza sia le $V$-value che le $Q$-value.

## Funzione di Valore Ottimale

Una policy $\pi$ è migliore o uguale a una policy $\pi'$ se e solo se, per ogni stato $s \in S$, $V^\pi(s) \geq V^{\pi'}(s)$. Esiste una policy ottimale $\pi^*$ tale che, per ogni $\pi'$, $V^{\pi^*}(s) \geq V^{\pi'}(s)$.

Possiamo definire la funzione di valore ottimale $V^*(s)$ e la funzione Q ottimale $Q^*(s,a)$ indipendentemente dalla policy:

$$V^*(s) = \max_{\pi \in \Pi} V^\pi(s), \quad \forall s \in S$$
$$Q^*(s,a) = \max_{\pi \in \Pi} Q^\pi(s,a), \quad \forall s \in S, \forall a \in A$$

La funzione di valore ottimale può essere espressa in termini della funzione Q ottimale:

$$V^*(s) = \max_{a \in A} Q^*(s,a)$$

Questa equazione indica che il valore ottimale di uno stato è il valore Q massimo ottenibile eseguendo l'azione che massimizza la coppia (stato, azione).

La funzione Q ottimale è definita come:

$$Q^*(s,a) = \sum_{s' \in S} T(s,a,s') \cdot [R(s,a,s') + \gamma \cdot V^*(s')]$$

Per ottenere la funzione di valore ottimale, dobbiamo trovare il valore $V^*(s)$ che massimizza questa funzione. La differenza rispetto alle funzioni di valore non ottimali è che qui appare il *return* atteso ottimale.

Eliminando le ricorsioni indirette, otteniamo le equazioni di Bellman ottimali:

$$
\begin{cases}
V^*(s) = \max_{a \in A} \sum_{s' \in S} T(s,a,s') \cdot [R(s,a,s') + \gamma \cdot V^*(s')] \\ \\

Q^*(s,a) = \sum_{s' \in S} T(s,a,s') \cdot [R(s,a,s') + \gamma \cdot \max_{a' \in A} Q^*(s',a')]
\end{cases}
$$

Questi sono processi di Markov: il futuro è indipendente dalla storia passata. In ogni stato, abbiamo la possibilità di prendere sempre l'azione migliore perché il percorso effettuato per arrivare a quello stato non influisce sul guadagno futuro.

Se conosciamo **T** e **R** possiamo calcolare i valori ottimi per le Value Function: con questi valori possiamo costruire la **Policy ottima**

$$
\pi^*(a|s)=
\begin{cases}
1, \quad \text{Se }a=\arg\max_{a'\in A} \ Q^*(s,a') \\
0,\quad \text{Altrimenti}
\end{cases}
$$
Nella pratica, però, non conosciamo questi valori.

# Q-Learning

L'obiettivo del Q-learning è stimare la funzione Q ottimale, $Q^*(s,a)$. Si costruisce un *training set* $T = \{ \langle s, a, r, s' \rangle \}$, che accumula l'esperienza dell'agente, dove:

* `s`: stato corrente
* `a`: azione eseguita
* `r`: ricompensa ricevuta
* `s'`: stato successivo

Si costruisce inoltre una *lookup table* $Q(s,a)$, una matrice con numero di righe pari al numero di stati e numero di colonne pari al numero di azioni possibili $(|S|\times|A|)$.

Si applica il *Temporal Difference Learning* per aggiornare gli elementi della matrice $Q(s,a)$ con la seguente regola:

$$Q(s,a) = (1-\alpha) \cdot Q(s,a) + \alpha \cdot [r + \gamma \cdot \max_{a' \in A} Q(s',a')], \quad \forall \langle s, a, r, s' \rangle \in T$$

dove:

* $\alpha$ è il *learning rate* (0 < α ≤ 1). Il termine $(1-\alpha)$ preserva una frazione del vecchio valore.
* $\gamma$ è il *discount factor* (0 ≤ γ ≤ 1).

Il Q-learning converge al valore ottimo $Q^*$ se il *training set* garantisce una sufficiente *Exploration* dello spazio degli stati e delle azioni. In altre parole, l'agente deve interagire con l'ambiente in modo da visitare tutti gli stati e provare tutte le azioni possibili.

## Fitted Q-Learning

Se la *lookup table* per la funzione Q è troppo grande (dimensione dello spazio degli stati-azioni troppo elevata), è necessario approssimarla con una funzione parametrizzata: $Q(s, a; \theta)$. L'obiettivo diventa quello di trovare i parametri $\theta$ che meglio approssimano la funzione Q ottimale, $Q^*$.

Per ogni tupla $\langle s, a, r, s' \rangle \in T$ (dove T è il training set), si definisce un target $Y_k^Q$:

$$\forall <s,a,r,s'>\ \in T,\ \ Q(s,a;\theta_{k})\approx Y_{K}^Q=r+\gamma\max_{a'\in A}Q(s',a',\theta_{k})$$

dove $\theta_k$ rappresenta i parametri correnti all'iterazione *k*. Si vuole quindi minimizzare la differenza tra la funzione approssimata e il target:

$$Q(s, a; \theta_k) \approx Y_k^Q$$

Questo problema può essere formulato come un problema di regressione classico, dove ogni tupla $\langle s, a, r, s' \rangle$ rappresenta un esempio di training $\vec{x}_i$, e il corrispondente valore target $Y_k^Q$ rappresenta l'output desiderato $y_i$. Si utilizzano quindi tecniche di regressione per aggiornare i parametri $\theta$ e migliorare l'approssimazione della funzione Q ottimale.

## Neural Fitted Q-Learning (NFQ)

![[10) Reti Generative-20241129093216919.png|604]]

NFQ utilizza una rete neurale per approssimare la funzione Q. Il processo di apprendimento prevede i seguenti passi:

1. **Determinazione del massimo Q-value:** Per ogni stato successivo `s'`, la rete neurale fornisce i valori Q per tutte le azioni possibili, $Q(s', a'; \theta_k)$. Si determina quindi il massimo di questi valori: $M = \max_{a' \in A} Q(s', a'; \theta_k)$.

2. **Costruzione del target:** Si costruisce il target $Y_k^Q$ utilizzando la ricompensa ricevuta `r` e il massimo Q-value calcolato al passo precedente:

 $$Y_k^Q = r + \gamma \cdot M$$

3. **Minimizzazione della loss function:** Si utilizza una loss function, tipicamente il Mean Squared Error (MSE), per minimizzare la differenza tra il valore Q predetto dalla rete neurale e il target:

 $$l_{NFQ} = \frac{1}{2} (Q(s, a; \theta_k) - Y_k^Q)^2$$

 I parametri $\theta_k$ della rete neurale vengono aggiornati iterativamente per minimizzare questa loss function, migliorando così l'approssimazione della funzione Q ottimale.

**Problema:** a volta la convergenza può essere lenta o diventare instabile. Inoltre, tende a sovrastimare i valori. Per evitare queste problematiche sono state sviluppate delle euristiche.

## Deep Q-Network (DQN)

DQN utilizza due reti neurali separate per stimare la funzione Q:

1. **Rete aggiornata:** $Q(s, a; \theta_k)$ – utilizzata per selezionare le azioni e per calcolare la loss function.
2. **Rete target:** $Q(s, a; \theta_k^-)$ – utilizzata per calcolare il valore target.

Entrambe le reti hanno la stessa struttura, ma pesi ($\theta$) differenti. Questo disaccoppiamento tra il calcolo del valore target e l'aggiornamento dei pesi migliora la stabilità dell'apprendimento.

Il valore target è calcolato utilizzando la rete target:

$$Y_k^Q = r + \gamma \cdot \max_{a' \in A} Q(s', a'; \theta_k^-)$$

I pesi della rete target ($\theta_k^-$) rimangono fissi per un certo numero di iterazioni (*c*), dopodiché vengono aggiornati con i pesi della rete aggiornata: $\theta_k^- = \theta_k$.

A differenza di NFQ, DQN opera in un ambiente *online*, interagendo direttamente con l'ambiente. Per gestire l'esperienza, si utilizza una *replay memory* che memorizza le ultime $N_{Replay}$ quadruple $<s, a, r, s' >$ raccolte dall'agente. L'aggiornamento dei parametri $\theta_k$ avviene selezionando un mini-batch casuale di esempi $<s, a, r, s' >$ dalla *replay memory*.

Per la selezione delle azioni, si utilizza una politica $\epsilon$-greedy, con $\epsilon \in [0, 1]$:

* Con probabilità $\epsilon$, si sceglie un'azione casuale.
* Con probabilità $1 - \epsilon$, si sceglie l'azione ottimale secondo la rete aggiornata: $a^* = \arg \max_{a \in A} Q(s, a; \theta_k)$.

Questa politica aiuta a mitigare il problema dell' **exploration vs exploitation dilemma**.

#### Variante: Double Deep Q-Network (DDQN)

Varia il modo in cui il valore target è calcolato utilizzando la rete target:
$$Y_{k}^{\text{DDQN}}=r+\gamma \cdot Q(s',\arg\max_{a'\in A}Q(s',a';\theta_{k});\theta_{k}^-)$$
L'idea è quella di disaccoppiare la scelta dell'azione su cui valutiamo il massimo dal calcolo del valore massimo stesso.

## Stima di Densità Non Parametrica

Queste tecniche non assumono alcuna forma specifica per la densità di probabilità, ma la stimano direttamente dai dati. Sono tecniche *instance-based* (o *memory-based*) perché richiedono l'intero training set per la stima.

![[10) Reti Generative-20241129104242903.png|576]]

Consideriamo una regione $R(x)$ centrata in un punto $x$. La probabilità di osservare un punto in questa regione è data da:

$$P = \int_{R(x)} p(x') dx'$$

Se la regione è sufficientemente piccola, possiamo approssimare la densità di probabilità come costante all'interno di $R(x)$:

$$P \approx \int_{R(x)} p(x) dx' = p(x) \int_{R(x)} dx' = p(x)V$$

dove $V$ è il volume della regione $R(x)$. Quindi:

$$p(x) \approx \frac{P}{V}$$

Se $n$ è sufficientemente grande, la probabilità $P$ può essere approssimata dalla proporzione di punti nel training set che cadono in $R(x)$: $P \approx \frac{k}{n}$, dove $k$ è il numero di punti in $R(x)$. Pertanto:

$$p(x) \approx \frac{k/n}{V}$$

Due metodi principali per la stima di densità non parametrica sono:

1. Stima di densità con *K-Nearest Neighbor* (KNN)
2. Stima di densità con *Kernel* (KDE)

## K-Nearest Neighbor Density Estimation (KNN)

In KNN, si fissa un valore $k$ e si determina la più piccola regione $R(x)$ centrata in $x$ che contiene esattamente $k$ punti del training set. Il raggio di questa regione è dato dalla distanza tra $x$ e il k-esimo vicino più prossimo ($nn_k(x)$):

$$r = \text{dist}(x, nn_k(x))$$

Il volume di questa regione dipende dalla dimensionalità dello spazio e dalla metrica utilizzata per calcolare la distanza. La stima della densità è quindi:

$$p(x) = \frac{k/n}{V(\text{dist}(x, nn_k(x)))} $$

dove $V(\text{dist}(x, nn_k(x)))$ rappresenta il volume della regione di raggio $r$.

La densità è inversamente proporzionale alla distanza dal k-esimo oggetto

##### Nearest Neighbor Classification (K-NN):

Restituisce la classe più rappresentata tra quelle dei *K-Nearest Neighbor* di $x$

##### Anomaly Detection

Le tecniche di rilevamento di anomalie assegnano un punteggio (*score*) a ciascun oggetto, indicando quanto è anomalo. Un punteggio alto indica un'alta probabilità di anomalia.

Una tecnica *distance-based* utilizza la distanza dal k-esimo vicino più prossimo come punteggio:

$$\text{Score}(x) = \text{dist}(x, nn_k(x))$$

Le anomalie sono i punti $x$ che massimizzano questo punteggio. Questa è una tecnica di tipo *globale*.

Una variante più generale considera la somma delle distanze dai primi $k$ vicini più prossimi:

$$\text{Score}(x) = \sum_{i=1}^k \text{dist}(x, nn_i(x))$$

Anche questa è una tecnica *globale*.

Le tecniche *locali*, invece, normalizzano il punteggio globale in base ai punteggi dei vicini:

$$\text{Score}_{local}(x) = \frac{\text{Score}_{glob}(x)}{\sum_{i=1}^k \text{Score}_{glob}(nn_i(x))}$$

dove $\text{Score}_{glob}(x)$ rappresenta il punteggio globale (distance-based) del punto $x$. Questo approccio considera il contesto locale per determinare l'anomalia.

## Kernel Density Estimation (KDE)

In KDE, si fissa la dimensione della regione $R(x)$ attorno a un punto $x$ e si conta il numero di punti del training set che cadono in questa regione. Questa regione è definita tramite una *funzione finestra* (window function) o *kernel*. Un esempio semplice di funzione finestra è la funzione indicatrice di un ipercubo:

$$\phi(\vec{u}) = \begin{cases} 1, & \forall i, |u_i| \leq \frac{1}{2} \\ 0, & \text{altrimenti} \end{cases}$$

dove $\vec{u} = (u_1, u_2, \dots, u_d)$ è un vettore di dimensione $d$. Questa funzione vale 1 se tutte le componenti di $\vec{u}$ sono in valore assoluto minori o uguali a 1/2, e 0 altrimenti. In pratica, si considera un ipercubo di lato 1 centrato nell'origine.

![[10) Reti Generative-20241129110450366.png|600]]

Nel caso unidimensionale ($d=1$), la regione $R(x)$ è un intervallo di ampiezza $h$ centrato in $x$. La funzione finestra indica se un punto $x_i$ del training set si trova all'interno di questo intervallo:

$$
\phi\left( \frac{x-x_{i}}{h} \right)=
\begin{cases}
1,\quad \text{Se x si trova nell'intorno di raggio h/2 centrato in }x_{i} \\
1,\quad \text{Se }x_{i}\text{ si trova nell'intorno di raggio h/2 centrato in x} \\
0,\quad \text{Altrimenti}
\end{cases}
$$

Questa funzione $\phi$ è chiamata *Parzen window*.

Il numero di punti $k$ che cadono nella regione $R(x)$ è dato da:

$$k = \sum_{i=1}^n \phi\left( \frac{x - x_i}{h} \right)$$

La stima della densità di probabilità in $x$ è quindi:

$$p(x) = \frac{k/n}{V} = \frac{1}{nh^d} \sum_{i=1}^n \phi\left( \frac{x - x_i}{h} \right)$$

dove $V = h^d$ è il volume della regione $R(x)$.

![[10) Reti Generative-20241129110436366.png|484]]

La *Parzen window* può esssere sostituita con una generica *funzione kernel* $k(u)$

$$
k(u)=
\begin{cases}
\text{Funzione di Densità Simmetrica (es: Gaussiana)}
\end{cases}
$$

$$KDE: p(x)=\frac{1}{nh^d}\sum_{i=1}^n K\left( \frac{\vec{x}-\vec{x}_{i}}{h} \right)$$
Problema: costo computazionale elevato, perchè devo guardare tutti i punti del dataset. 
Soluzione 1: ignorare i punti oltre una certa distanza, ma questo potrebbe portare ad errori.
Soluzione 2: utilizzare kernel con area sottea 1 e che sono simmetrici (ad es Tricube)

Scelta dell'iperparametro $h=1.06 \cdot \hat{\sigma}^{1/j}$ (empirica)
