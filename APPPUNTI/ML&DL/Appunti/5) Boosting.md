## Paradigma di Learning: Strong Learner come Composizione di Più Weak Learner

Questo paradigma si basa sull'idea di combinare diversi predittori "deboli" (weak learner) per ottenere un predittore "forte" (strong learner) con performance elevate.

- **Weak learner:** Un predittore che, singolarmente, non offre buone prestazioni sul problema, presentando un errore alto.
- **Strong learner:** Un predittore che, combinando più weak learner, raggiunge un errore basso.

### Definizione Formale di Weak Learner

Data una classe di ipotesi $H$, un weak learner è definito da una diversa classe di ipotesi $B$ con le seguenti proprietà:

1. **Efficienza computazionale di ERM(B):**  Il calcolo di $ERM(B)$ (Empirical Risk Minimization su B) deve essere efficiente, idealmente con costo sub-quadratico.
2. **Performance migliore di un classificatore casuale:** L'ipotesi restituita da $ERM(B)$ deve avere prestazioni migliori di un classificatore casuale sui problemi risolti in maniera esatta da $H$.

**Formalmente:**

$H = \{h_{a,b,\alpha}(x) = \alpha \cdot 1[\alpha \leq x \leq b], a,b \in R, \alpha \in \{-1,1\} \}$

![[5) Boosting-20241011085647194.png|321]]
Dove:
- $\theta_{1} = a$
- $\theta_{2} = b$

### Decision Stump:

Un esempio di weak learner è il Decision Stump, una funzione soglia definita come:

$B = \{ h_{\theta,\alpha}(x) = \alpha + 1[x \geq \theta], x \in R, \alpha \in \{-1,1\} \}$

![[5) Boosting-20241011091139119.png]]
Dato un problema risolto in maniera esatta da $h$, la miglior ipotesi in $B$ deve avere un errore inferiore a $\frac{1}{2}$.

**Esempio:**

Consideriamo un problema con tre intervalli senza sovrapposizioni tra le classi. Almeno uno di questi intervalli conterrà meno di $\frac{1}{3}$ della popolazione. Sbagliando la classificazione solo su questo intervallo, l'errore sarà inferiore a $\frac{1}{2}$.

![[5) Boosting-20241011091146389.png]]

In questo caso, il Decision Stump rappresenta l'algoritmo che lavora sulla classe di ipotesi $H'$.

![[5) Boosting-20241011091153416.png]]

**γ-Weak-Learner:**

Un $\gamma\text{-Week-Learner}$ è una classe di ipotesi che garantisce un errore di generalizzazione della classe restituita che si discosta da $\frac{1}{2}$ di almeno $\gamma$.

## Minimizzazione del Rischio Empirico per Decision Stumps

![[5) Boosting-20241011091235809.png|508]]

Prima di approfondire il boosting, analizziamo come applicare la regola della minimizzazione del rischio empirico alla classe di ipotesi dei decision stumps.

**ERM efficiente per decision stumps:** L'obiettivo è trovare il decision stump che minimizza il rischio empirico. Consideriamo un insieme di punti in ℝ e una funzione 𝜃 che restituisce -1 a destra e +1 a sinistra.

**Algoritmo:**

1. **Ordinamento:** Ordiniamo i punti in ℝ.
2. **Sogli:** Determiniamo le soglie (potenzialmente infinite, ma quelle significative sono date dal valore mediano tra due 𝜃).
3. **Calcolo dell'errore empirico:** Per ogni soglia 𝜃𝑖, calcoliamo l'errore empirico 𝐹[𝜃𝑖]. L'errore è dato dal numero di punti della classe positiva che si trovano a destra della soglia.

**Complessità:**

L'ordinamento dei punti ha una complessità data da  $t(m)=m\log (m)+(m+1)\cdot O(1)=O(m \log(m))$, dove 𝑚 è il numero di punti. 
La funzione decision stump vale +1 a sinistra di 𝜃 e -1 a destra.

$t_{d}=d \cdot m \cdot \log(m)$

## Ada-Boost

l'ada-boost(adaptive-boosting) è una tecnica che costruisce un classificatore strong come composizione di classificatori più deboli
**input:**
- $S=\{(x_{1},y_{1},(x_{2},y_{2},\dots,(x_{n},y_{n}))\}$
- $WL:\text{ Week Learner}$
- $Y:\text{Numero di Iterazioni}$

$D=(D_{1},D_{2},\dots,D_{m})$
$D_{i}=\frac{1}{m}$, ovvero ogni oggetto avrà il suo peso

quando si calcolerà l'errore empirico, lo si fa assumendo che una misclassification non costi 1 ma un valore proporzionale al peso dell'oggetto. 

$L_{s}(h)= \sum_{i=1}^m D_{i}\cdot 1[h(x_{i}\neq y_{i})]$
il calcolo di questo può essere fatto in tempo costante
l'algo all'inizio da il suo peso a tutti gli esempi, poi man mano che procede troverà esempi più difficili da catturare. La stratregia sarà quella di aumentare il peso degli esempi più difficili, per spingere l'algoritmo a non sbagliare su qeusti esempi

**Output:** Strong Learner, ottenuto come combinazione dei week learner ottenuti a ogni iterazione
$h(x)=sign\left( \sum_{t=1}^T w_{t}h_{t}(x) \right)$
dove $h_{t}(x)$ è il week learner restituiti al passo t, ha  valore +-1
abbiamo la somma dei valori pesata e quindi anche h(x) sarà +-1

**metodo**
$d_{1}=\left( \frac{1}{m},\dots ,\frac{1}{m} \right)$
$\text{for t=1 to T do}$
	$h_{t}=WL(S,D^{(t)})$
	$\epsilon_{t}=\sum_{i=1}^m D_{i} 1[h_{t}(x_{i})\neq y_{i}]$
	$\omega_{t}=\frac{1}{2}\log\left( \frac{1}{\epsilon}-1 \right)$
	il peso dipende dall'errore, più è basso più il peso è grande

aggiornamento:
$$D_{i}^{t+1}=\frac{D_{i}^{(t)}\exp(- \omega_{t}y_{i}h_{t}(x_{i}) )}{ \sum_{j=1}^m D_{j}^{(t)} \exp(-\omega_{t}y_{j}h_{t(x_{j})} )}$$
$x_i \text{ vale } \pm1$
$y_i \text{ vale } \pm1$
$D_i^{(t)}$ è il vecchio peso e lo moltiplichiamo per $exp(\pm\omega)$

l'effetto è quello di abbassare il peso di quelli classificati correttamente e aumentare quello di quelli mis-classified, ma questo dipende da $\omega$, se è negativo si inverte la logica


## Teorema

Si assuma che WL restituisca ad ogni iterazione una ipotesi per cui  $\frac{\epsilon_{t}\leq_{1}}{2}-\gamma$
Allora $L_{S}(h_{S})\leq \exp(-2 \cdot\gamma^2\cdot T)$
![[5) Boosting-20241011095412522.png|338]]
ovvero se T è sufficientemente grande, adaboost va in overfitting (azzera l'errore empirico)

Esempio
![[5) Boosting-20241011095553614.png|332]]

le linee rappresentano le nuove soglie durante le varie iterazioni

### Proprietà
$VC_{dim}(\text{AdaBoost(B,T)})= T \cdot VC_{dim}(B)$


# Problemi di Learning Convessi

### Insieme convesso
se comunque prendiamo due punti all'interno dell'insieme, il segmento che li unisce è contenuto all'interno dell'insieme
![[5) Boosting-20241011100428505.png]]

$$
\begin{cases}
\vec{v} = \alpha \vec{u} + (1 - \alpha)\vec{v} \\
\alpha \in [0,1] 
\end{cases}$$

$\forall\alpha\in[0,1],\alpha \vec{u}+(1-\alpha)\vec{v}\in C$

### Funzione convessa
comunque prendiamo due punti sulla curva della funzione, il segmento che li unisce si trova sopra il grafico della funzione
![[5) Boosting-20241011100739017.png]]

$\forall\alpha\in[0,1],f(\alpha \vec{w}+(1-\alpha)\vec{v})\leq\alpha f(\vec{w})+(1-\alpha)f\vec{w}$

### epifrafico di f
$\{(x,\beta):f(x)\leq \beta \}$
![[5) Boosting-20241011101042946.png]]

una funzione è convessa se e solo se il suo epigrafo è una funzione convessa


### Proprietà
Se $f$ è una funzione convessa, ogni minimo locale è un minimo globale.

### Proprietà
1) $f$ è convessa
2) $f'$ è monotona non decrescente 
3) $f''$ è non negativa

### Esempi
$f(x)=x^2$
$f'(x)=2x$
$f''(x)=2>0$
la funzione è convessa

$f(x)=\ln(1+\exp(2))$
$f'(x)=\frac{e^x}{1+e^x}=\frac{e^x}{1+e^x} \cdot \frac{e^{-x}}{e^{-x}}=\frac{1}{1+e^{-x}}=\sigma(x)$
f'' è monotona non decrescente, dunque f è convessa
### proprietà
$f(\vec{w})=g(<\vec{w},\vec{x}>+b)$
allora la convessità di g implica la convessità di f



$l_{sq }(\vec{w},(\vec{x},y))=\frac{1}{2}(<\vec{w},\vec{x}>-y)^2$ loss convessa

$l_{logistic}=(\vec{w},(\vec{x},y))=\ln(1+\exp(y,<\vec{w,\vec{x}}>))$ loss convessa


## Problema di learning convesso
1) la classe d'ipotesi H è convessa $\to h_{\vec{w}},\ \vec{w}\in R^d$
2) La loss è una funzione convessa

Il problema di learning $ERM_{H}(S)=\text{arg min}_{\vec{w}\in R^d} \ L_{S}(h_{\vec{w}})$
$\to$ Trovare il minimo di una funzioone convessa su un dominio convesso (Problema di ottimizzazione convessa)
