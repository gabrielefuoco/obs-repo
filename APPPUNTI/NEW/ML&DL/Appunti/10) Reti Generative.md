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

$$L(x,\tilde{x})=\|x-\tilde{x}\|^2+\beta KL(N(\mu_{x},\Sigma_{x}),N(\vec{0},I))$$

KL è L'entropia relativa, vale 0 se le due distribuzioni sono uguali: $KL(p,q)=E_{P}[\log  \frac{p}{q} ]$

![[10)-20241122091125417.png]]
In questo modo abbiamo garantito che le regioni si possano sovrapporre.


## Generative Adversarial Network (GAN)

Implementano il paradigma dell'*Adversarial Learning* (Apprendimento competitivo).

![[10) Reti Generative-20241122123855843.png]]

Il funzionamento di una Rete Generativa Adversaria (GAN) si basa sull'interazione tra due reti neurali: una rete generativa e una rete discriminativa.

#### Rete Generativa

La rete generativa riceve in input punti generati casualmente da una distribuzione di riferimento.  Il suo obiettivo è generare un output che assomigli il più possibile ai punti del dataset reale.

#### Rete Discriminativa

La rete discriminativa riceve in input esempi provenienti da due sorgenti: il dataset reale e la rete generativa.  Dato un esempio *x*, la rete deve restituire la probabilità che *x* appartenga alla distribuzione di probabilità reale  $D(x) = Pr[x \in p_x]$.

#### Addestramento delle Reti

L'addestramento delle due reti avviene simultaneamente.  Si utilizza un batch di esempi dal dataset reale e si aggiornano i pesi di entrambe le reti con l'obiettivo di:

* **Massimizzare la capacità discriminativa:**  La rete discriminativa deve massimizzare la probabilità di distinguere correttamente gli esempi reali ($D(x) = 1$) da quelli generati dalla rete generativa ($D(\tilde{x}) = D(G(z)) = 0$).

* **Minimizzare la capacità discriminativa (per la rete generativa):** La rete generativa deve minimizzare la probabilità che la rete discriminativa riconosca gli esempi generati come sintetici.

Formalmente, l'addestramento può essere espresso come un gioco a due giocatori:

$$
\begin{cases}
\max \quad \log D(x) + \log (1 - D(G(z))) \\
\quad \text{massimizza la probabilità di discriminare gli esempi sintetici dagli esempi reali} \\ \\
\min \quad \log(1 - D(G(z))) \\
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
0 & \text{Posizione intermedia o Patta} \\
-1 & \text{Sconfitta}
\end{cases}$$

È importante notare che assegnare ricompense solo a situazioni di vittoria o sconfitta potrebbe non essere sempre ottimale.  Una strategia di ricompense più sofisticata potrebbe fornire informazioni più ricche all'agente.


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

Nei *task* episodici, dove esiste uno stato finale, il *return* è ben definito.  In questi casi, l'interazione termina quando si raggiunge lo stato finale.  Tipicamente si considera il *return* per ogni episodio.

Nei *task* continui, il guadagno può essere potenzialmente infinito.  Inoltre, una ricompensa ottenuta immediatamente ha un valore maggiore di una ricompensa ottenuta in un futuro più lontano.  Per questo motivo, si utilizza spesso il *discounted return*.


### Discounted Return

Per affrontare il problema del guadagno potenzialmente infinito nei *task* continui, si introduce un parametro chiamato **tasso di sconto** (discount rate): $\gamma \in [0, 1]$.  Il *discounted return* è definito come:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^k r_T$$

dove $\gamma^k$ rappresenta il valore attuale di una ricompensa ottenuta dopo *k* istanti di tempo.


### Interprete

Nell'architettura di un sistema di apprendimento per rinforzo, tra l'ambiente e l'agente è spesso presente un **interprete**:

![[10) Reti Generative 2024-11-22 10.25.09.excalidraw]]

L'interprete elabora le sensazioni attuali e passate per produrre lo **stato**. Lo stato è una rappresentazione processata dei segnali dell'interprete, che deve riassumere tutte le informazioni rilevanti per l'agente al fine di prendere una decisione.  Questo stato è detto **stato Markoviano**, in quanto contiene tutte le informazioni necessarie per prevedere il futuro, indipendentemente dal passato.

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

---

L'obiettivo finale è trovare la policy $\pi^*$ che massimizza sia le $V$-value che le $Q$-value.

---