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

il punto x viene mappato su una distribuzione normale :
$$x\to N(\mu,\Sigma), \quad \Sigma=\begin{bmatrix}\sigma^2 &0 \\ 0 & \sigma^2_{k}\end{bmatrix}$$
Il decoder lavora su singoli punti, dunque bisogna campionare un punto $z$ dalla distribuzione $N(\mu,\Sigma)$. Il decoder riceverà in ingresso questo punto

![[10)-20241122084722796.png]]

Mappare una distribuzione ci serve per mappare la continuità. Se lasciamo l'autoencoder libero di ... la sua soluzione sarà di lasciarli separati.
Vogliamo forzare la rete ad avvicinare i punti tra di loro. Bisogna dunque aggiungere un termine di regolarizzazione alla loss, che misura quanto la distribuzione su cui stiamo mappando il nostro punto è diversa da una normale standard.

$$L(x,\tilde{x})=\|x-\tilde{x}\|^2+\beta KL(N(\mu_{x},\Sigma_{x}),N(\vec{0},I))$$

KL è L'entropia relativa, vale 0 se le due distribuzioni sono uguali: $KL(p,q)=E_{P}[\log  \frac{p}{q} ]$

![[10)-20241122091125417.png]]
In questo modo abbiamo garantito che le regioni si possano sovrapporre.


## Generative Adversarial Network (GAN)

Implementano il paradigma dell'*Adversarial Learning* (Apprendimento competitivo).

![[10)-20241122091408357.png]]

# MOdificare: D è la rete discriminativa, G è la rete generativa

La rete generativa riceve dei punti generati casualmente da una certa distribuzione di riferimento. La rete restituirà come output un punto il cui obiettivo è somigliare il più possibile ai punti del dataset.

La rete discriminativa può ricevere sia esempi reali dal dataset, sia dalla rete generativa.
Questa rete, dato un esempio qualunque x, deve restituire la probabilità che x provenga dalla distribuzione reale $D(x)=Pr[x\in p_{x}]$

La rete viene utilizzata addestrando le reti in contemporanea: si prende un batch di esempi dal dataset e si aggiornano i pesi delle due reti
$D(x)=1$
$D(\tilde{x})=D(G(z))=0$

$$
\begin{cases}
\max \quad \log D(x)+\log (1-D(G(z)) \\ 
\quad \text{massimizza la probabilità di discriminare gli esempi sintetici dagli esempi reali} \\ \\

\min \quad  \log(1-D(G(z))) \\
\quad \text{minimizzare la probabilità che D risconosca gli esempi generati sinteticamente} \\
\end{cases}
$$
Addestrando la rete in questo modo si può dimostrare che la rete prevede una soluzione ottima: la rete generativa converge alla distribuzione target e la rete discriminativa diverrà inefficace
$$
\begin{cases}
P_{g}=p_{x} \\
D(x)=\frac{1}{2}
\end{cases}
$$

## Reinforcement Learning

Apprendimento per interazione o rinforzo.
Il framework generale prevede la presenza di un learner chiamato **Agente**. 
L'agente interagisce con una serie di ambienti
In ogni istante l'ambiente si trova in un certo stato. L'agente deve prendere una decisione combiendo un'azione 
- $s_{t}\in S\quad \text{Insieme degli stati}$
- $a_{t}\in A \quad \text{Insieme delle azioni}$ 
- $r_{t}\in R \quad \text{Ricompensa(Reward)}$ 
L'ambiente risponde all'azione con un valore reale e lo stato si aggiorna.

![[10) Reti Generative-20241122093942732.png]]

La rete opera implementando una **Policy** $\pi$, che è la probaiblità di selezionare una certa azione in un certo stato: $\pi(a|s)$
L'agente agisce con il fine di massimizzare la sua ricompensa nel lungo periodo (aggiornando la propria Policy).

### Esempio: robot in un labirinto

$$
\begin{bmatrix}
\text{exit} &1 &2 &3 \\ 
4 &5 &6 &7 \\ 
8 &9 &10 &11  \\
 12 &13 &14  &\text{exit}
\end{bmatrix}
$$
 
$S=\{ 1,2,3,\dots,14,0 \}$
$A=\{ \uparrow, \downarrow, \rightarrow, \leftarrow \}$
$R=-1$
Avere -1 come ricompensa significa minimizzare il tempo di percorrenza
Massimizzazione delle ricompensa deve coincidere con il raggiungimento del goal prefissato

# REGISTRAZIONE 2, 22/11

### Gestire le Ricompense
$$R=
\begin{cases}
+1 \quad \text{Vittoria} \\
\ \ \ 0 \quad \text{Posizione intermedie o Patte} \\
-1 \quad \text{Sconfitta} \\
\end{cases}
$$
Scegliere di dare un reward a situazioni vantaggiose non è produttivo

### Return (guadagno)
Somma delle ricompense che riceve nel tempo

$$G_{t}=r_{t}+r_{t+1}+\dots,r_{T}$$
Distinguiamo 2 tipi di Task:
$$
\text{Task:}
\begin{cases}
\text{Episodic: L'interazione è divisa in episodi} \\
 \\
\text{Continuous: Non c'è stato finale}
\end{cases}
$$

In caso di task continui il guadagno è potenzialmente infinito.
Una ricompensa ottenuta ora ha più valore di una ricompensa ottenuta diverse passi dopo

Si utilizza quindi una funzione di guadagno differente:

### Discounted Return

Si introduce un pararametro detto **Discount Rate:** $\gamma \in [0,1]$
- La funzione di guadagno diventa:
$$G_{t}=r_{t}+\gamma\ r_{t+1}+\gamma^2\ r_{t+2}+\dots,\gamma^k \ r_{T}$$
- Dove $\gamma^k= \text{valore attuale di una ricompensa che verrà ottenuta dopo k istanti di tempo}$


Nell'architettura della rete di Reinforcement Learning, tra l'ambiente e l'agente viente posto un elemento chiamato inteprete:
![[10) Reti Generative 2024-11-22 10.25.09.excalidraw]]
L'interprete processa le sensazioni presenti e quelle passate al fine di produrre lo stato
Lo stato è una versione processata dei segnali dell'interpete: deve riassumere tutte le informazioni rilevanti per l'agente al fine di prendere una decisione: è chiamato stato Markowiano

### Sato Markowiano

Ricompensa e stato futuro sono indipendenti dalla storia, ma dipendono solo dallo stato corrente e dall'aziome compiuta
$$
\begin{cases}
P(s_{t+1}|s_{t},a_{t})=P(s_{t+1}|s_{t},a_{t},s_{t-1},a_{t-1},\dots,s_{0},a_{0}) \\ \\

P(r_{t}|s_{t},a_{t})=P(r_{t}|s_{t},a_{t},s_{t-1},a_{t-1},\dots,s_{0},a_{0})
\end{cases}
$$
Gli approcci di Reinforcement Learning assumono che gli stati siano Markowiani

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
