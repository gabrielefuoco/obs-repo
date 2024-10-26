## Neural Networks
Le reti neurali sono modelli di apprendimento automatico ispirati alla struttura del cervello umano. L'unità fondamentale di questo modello è il **neurone artificiale**, un'unità computazionale che riceve input, li elabora e produce un output.

Una rete neurale si ottiene collegando tra loro i neuroni artificiali secondo diversi schemi. Quando colleghiamo questi neuroni artificiali, diamo vita a quella che viene chiamata **architettura neurale**. Questa rappresenta la struttura della rete che si viene a formare.

Per illustrare le reti neurali, partiamo dalle architetture più semplici che possiamo creare: le **reti feedforward**. Sono definite "feedforward" perché il flusso dell'informazione va in una sola direzione, dal punto di vista dell'input all'output.

## FeedForward Neaural Networks

Il flusso dell'informazione va solo in una direzione (dall'input all'output)
#### Struttura generale:
Queste reti possono essere descritte attraverso un grafo diretto aciclico. La rete può essere rappresentata come un grafo $𝐺 = (𝑉, 𝐸,\vec{w},\phi)$ dove:

![[8)-20241024130024790.jpg|537]]
Dove:
* **𝑉 (nodi)** rappresentano i neuroni artificiali.
* **𝐸 (archi)** rappresentano i collegamenti tra i neuroni.
* **𝑤** è una funzione che assegna un peso ad ogni arco.
* $\phi$ è la funzione di attivazione.

Ogni livello è composto da un numero di neuroni, e ogni neurone è connesso a tutti i neuroni del livello successivo.

In una rete densa, ogni neurone di un livello è collegato a tutti i neuroni del livello successivo. Ogni collegamento tra due neuroni è associato a un **peso**, che rappresenta la forza della connessione.

I pesi sono i **parametri** della rete, e sono modificati durante il processo di apprendimento per migliorare le prestazioni della rete. Il numero di parametri della rete è uguale al numero dei pesi.

Ogni neurone applica una **funzione di attivazione** al suo input per produrre un output. La funzione di attivazione introduce non linearità nella rete, permettendole di apprendere relazioni complesse tra i dati.

Quando ogni neurone del livello i-esimo è collegato a ogni neurone del livello i+1-esimo, la rete è detta **densa**. 


![[8)-20241024130054817.png|354]]



## Funzioni di attivazione

**Segno:**
![[8)-20241024130744776.png|402]]
$\phi$ sulle ordinate e $<\vec{w},\vec{x}>$ sulle ascisse.
Quando usiamo questa funzione il neruone corrisponde a un semispazio.

**Gradino:**
![[8)-20241024130804304.png|423]]

Il problema di queste funzioni è legato al fatto che la loro derivata non è utile ai fini dell'algoritmo di minimizzazione, poichè è sempre prossima allo zero.
Una soluzione è data dalla funzione Sigmoide:

**Sigmoide**:
![[8)-20241024130821929.png|455]]
Gode della seguente prorpietà:
$$\sigma'(x)=\sigma(x)(1-\sigma (x))$$

**Tangente iperbolica:**
![[8)-20241024131749391.png|445]]

$$\text{tanh}(x)=\tau(x)=2\sigma(2x)-1$$
$$\tau'(x)=1-\tau(x)^2$$

Se ci allontaniamo dall'origine, il gradiente tende ad azzerarsi e la funzione non cresce(Regioni saturanti). Per risolvere questo problema vengono in gioco altre funzioni di attivazione:

**ReLU:**
![[8)-20241024131933828.png|446]]
$$\mathrm{ReLU}=\max(0,x)$$
$$ \mathrm{ReLU}'
\begin{cases}
0, \ x<0 \\
1< \ x\geq 0
\end{cases}
$$
**Leaky ReLU**
![[8)-20241024132331064.png|436]]

$$ \mathrm{Leaky \ ReLU}
\begin{cases}
x, \ x>0 \\
\alpha x, \  x\leq 0
\end{cases}
$$

## Potere Espressivo delle Reti Neurali

### Funzioni di Attivazione Segno

Le funzioni di attivazione segno, utilizzate nei neuroni artificiali, permettono di calcolare funzioni binarie. Queste funzioni mappano un input binario di n dimensioni in un output binario:

$$h:\{ -1,+1 \}^n\to\{ -1,+1 \}$$

### Funzione AND

La funzione AND restituisce +1 se tutti gli input sono +1, altrimenti restituisce -1. 
Per implementare la funzione AND, utilizziamo un percettrone con due input (x1 e x2) e un singolo neurone con funzione di attivazione segno. Assegniamo un peso di +1 ad entrambi gli input.

La sua rappresentazione matematica è:

$$h_{AND}(\vec{x})
\begin{cases}
+1, \vec{x}=d \\
-1, \text{Altrimenti}
\end{cases}
$$

dove $d=(+1,\dots,+1)$ rappresenta il vettore di tutti gli input a +1.

La funzione AND può essere implementata utilizzando la funzione segno:

$$\sum_{i}x_{i}>=d-0.5$$

$$h_{AND}(\vec{w})=\text{sign} \left( \sum_{i}x_{i}-d +0.5 \right)$$
![[Senza nome-20241025084652853.png|419]]
### Funzione OR

La funzione OR restituisce +1 se almeno un input è +1, altrimenti restituisce -1.
Utilizzando la stessa struttura del percettrone precedente (due input, un neurone con funzione segno), impostiamo la soglia a +1.

La sua rappresentazione matematica è:

$$h_{OR}(\vec{x})=
\begin{cases}
+1, se \ ∃i:x_{i}=+1 \\
-1, \text{altrimenti}
\end{cases}$$

La funzione OR può essere implementata utilizzando la funzione segno:

$$\sum_{i}x_{i}>(+1)+(-1)(d-1)=-d+2-0.5$$

$$h_{OR}(\vec{x})=\text{sign}\left( \sum_{i}x_{i}+d-1.5 \right)$$



## Funzione NOT

La funzione NOT è definita come:

$h_{NOT}(x)=-x$

Questa funzione può essere espressa anche come:

$-sign(x)=sign(-x)$

### NAND

La funzione NAND è definita come la negazione della funzione AND:

$$h_{NAND}(x)=not \ h_{AND}(x)=-h_{AND}(x)=-\text{sign}\left( \sum_{i}x_{i}-d+0.5 \right)$$

Sviluppando l'equazione, otteniamo:

$$=sign\left( \sum_{i}-x_{i}+d-0.5 \right)$$

### NOR

La funzione NOR è definita come la negazione della funzione OR:

$$h_{NOR}(\vec{x})=\text{not }h_{OR}(x)= -h_{OR}=-\text{sign}\left( \sum_{i}x_{i}+d-1.5 \right)$$

Sviluppando l'equazione, otteniamo:

$$=\text{sign}\left( \sum_{i}-x_{i}-d+1.5 \right)$$

## Funzione di uguaglianza

La funzione di uguaglianza restituisce +1 se l'input corrisponde a una specifica sequenza predefinita, altrimenti -1. Per implementarla, assegniamo pesi +1 o -1 agli input, in base alla sequenza desiderata. Se l'input corrisponde alla sequenza, la somma ponderata sarà massima, attivando il neurone e restituendo +1.

$$
h_{\vec{w}^{EQ}}=
\begin{cases}
+1, se \ \forall , x_{i}=w_{i} \\
-1, \text{ altrimenti}
\end{cases}
$$

$$
h_{\vec{w}^{EQ}}=\text{sign}\left( \sum_{i}w_{i}x_{i}-d+0.5 \right)
$$
![[9)-20241025090247850.png|495]]
Problema XOR: non è linearmente separabile

## XOR

La funzione XOR (Exclusive OR) restituisce +1 se solo uno dei due input è +1, altrimenti -1. 
$$x_1 \oplus x_2 = x_1 \overline{x_2} + \overline{x_1} x_2$$
 
A differenza delle funzioni booleane elementari come AND, OR e NOT, la funzione XOR non è linearmente separabile. Ciò significa che non è possibile tracciare un singolo iperpiano (una retta nel caso bidimensionale) in grado di separare correttamente i punti corrispondenti a +1 da quelli corrispondenti a -1.

Lo XOR corrisponde alla seguente regione
![[9)-20241025090701456.png|480]]
Equivale all'intersezione del semispazio dell'OR e del NAND.

$h_{XOR}(\vec{x})=h_{AND}(h_{OR}(\vec{x}),h_{NAND}(\vec{x}))$


![[9)-20241025090912178.png]]


un altro modo di rappresentarlo è

![[9)-20241025091123687.png]]
Con due funzioni di uguaglianza: funzione di uguaglianza sopra(-1,+1), uguaglianza sotto (+1-1) 

$h_{XOR}(\vec{x})=h_{OR}(h_{(-1+1)}^{EQ}(\vec{x}),h_{(+1-1)}^{EQ}(\vec{x}))$

## Proprietà delle Reti Neurali a Due Livelli

Ogni funzione booleana può essere calcolata da una rete neurale a due livelli. Notiamo che ogni funzione booleana può essere scritta in forma normale:

$$f(x)=g_{1}(\vec{x_{1}})\lor g_{2}(\vec{x_{2}})\lor \dots g_{n}(\vec{x_{n}})$$

Questa può essere rappresentata in una rete neurale come:

$$f(\vec{x})=h_{OR}(h^{EQ}_{w_{1}}(\vec{x}),\dots,h^{EQ}_{w_{n}}(\vec{x}))$$

**Esempio:**

$g_{i}(\vec{x}) = (x_{1} \land \neg x_{2} \land x_{4})$

Il numero di neuroni potrebbe essere esponenziale nel numero di variabili. Dunque, si preferisce usare più livelli perché sono necessari meno neuroni.

#### Semispazi e Livelli di Rete

Se ragioniamo in termini di semispazi, con un livello possiamo modellare dei semispazi:

* **1 livello:** semispazi
* **2 livelli:** intersezione di semispazi, catturare regioni convesse a "k facce", dove k è il numero di semispazi intersecati
* **3 livelli:** unione di regioni convesse 

## Proprietà

Se invece ragioniamo in termini di funzioni arbitrarie (e non solo booleane), che hanno come dominio $R$, vale la seguente proprietà:

**Le reti neurali sono degli approssimatori universali.**

Data la funzione:

$f:[-1+1]^d\to[-1+1]$

con $f$ Lipschitz, allora per ogni $\epsilon>0$ posso costruire una rete neurale che approssimi la mia funzione con un errore al massimo pari ad $\epsilon$. 
$$\forall x\in D,f(x)-\epsilon\leq h(x)\leq f(x)+\epsilon$$

## Proprietà

Per ogni funzione $f$ che può essere calcolata da una Macchina di Turing in tempo $O(T(d))$, esiste una rete neurale di size $O(T(d)^2)$ che calcola $f$. 

## Proprietà

Si consideri il problema della classificazione binaria, allora la VC-Dimension della rete neurale è pari a $VC_{\text{dim}}(NN)= |E|$, ovvero al numeri di pesi della rete.

## Problema XOR

### Matrice dei Pesi $w_1$

La matrice dei pesi $w_1$ è definita come:

$$w_{1}=
\begin{bmatrix}
-1 &-1 & 1.5 \\
+1 & +1 & 0.5 \\
0 &0&1
\end{bmatrix}
$$

Se moltiplichiamo questa matrice per il vettore di input:

$$\vec{x}=
\begin{bmatrix}
x_{1} \\
x_{2} \\
1
\end{bmatrix}$$

otteniamo:

$$\phi=
\begin{bmatrix}
-x_{1} & -x_{2} &1.5 \\
x_{1} & x_{2} & 0.5 \\
0 & 0 &1
\end{bmatrix}
$$

Il risultato di questa moltiplicazione costruisce gli input del livello successivo.

### Pesi del Livello Successivo

I pesi del livello successivo sono rappresentati da:

$$\phi \left([+1, +1, -1.5] \cdot \phi\begin{bmatrix}
-1 &-1 & 1.5 \\
+1 & +1 & 0.5 \\
0 &0&1
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
x_{2} \\
1
\end{bmatrix}\right)$$

In generale, la funzione di attivazione della rete neurale può essere espressa come:

$h(\vec{x})=\phi(w_{j}\phi(w_{2}\phi(w_{1}\vec{x})))$

dove $\phi$ rappresenta la funzione di attivazione e $w_i$ sono le matrici dei pesi per ogni livello.

## Proprietà

$$\vec{w}=ERM(S)=\arg\min_{\vec{w\in \mathbb{R}^{|E|}}}L_{S}(\vec{w})$$
dove  $L_{S}(\vec{w})=\frac{1}{m}\sum_{i=1}^ml(h,(\vec{x_{i}},y_{i}))$

**Proprietà**: Implementare la regola ERM rispetto alla classe delle reti neurali è un problema *NP-Hard*.

## Discesa Stocastica del Gradiente

supponiamo di utilizzare loss quadratica $l_{SQ}(h(x,y))=\frac{1}{2}(h(x)-y)^2$

$\vec{w}^{(1)}=\text{ Inizializzazione}$
$\text{for t=1 to T-1 do}$
	 $\text{seleziona casualmente }(\vec{x_{i}},y_{i})\text{ in S}$
	$\vec{v}_{t=}\nabla_{\vec{w}}l(h_{\vec{w}^{(t)}},(\vec{x_{i}},y_{i}))=\nabla_{\vec{w}}\left[ \frac{1}{2}   \{ \phi(w_{t}\phi(w_{t-1}\phi(\dots w_{2}\phi(w_{1}\phi (w \vec{x_{i}}))))) \}-y \right]^2$
	$\vec{w}^{(t+1)}=\vec{w}^{(1)}+\vec{\eta}v_{t}$
$\text{return }\vec{w}^{(t)}$


## Algoritmo di Backpropagation

Per calcolare il gradiente, si utilizza un algoritmo specifico chiamato **backpropagation**. Questo algoritmo sfrutta la regola della catena per calcolare le derivate parziali della loss rispetto a ciascun peso.

$$\vec{v_{t}}=\nabla_{\vec{w}}l(h_{\vec{w}}(\vec{x},y))$$
$$v_{t,j}=\frac{\partial}{\partial w_{l}}l(h_{\vec{w}}(\vec{x},y)),j=1,2,\dots,|E|$$

Per calcolare queste derivate dobbiamo ricorrere alla regola della catena:

### Chain Rule

La regola della catena è un principio fondamentale del calcolo differenziale che permette di calcolare la derivata di una funzione composta. In particolare, per una funzione composta $f(g(x))$, la derivata rispetto a $x$ è data da:

$$[f(g(x))]=f'(g(x)\cdot g'(x))=\frac{d}{dg(x)}f(g(x))\cdot \frac{d}{dx}g(x)=\frac{df}{dg}\cdot \frac{dg}{dx}$$
$$\frac{\partial}{\partial x}f(g_{1}(x)g_{2}(x))=\frac{\partial f}{\partial g_{1}} \cdot \frac{dg_{1}}{dx}+\frac{\partial f}{\partial g_{2}} \cdot\frac{dg_{2}}{dx}$$

### Applicazione della Regola di Derivazione a un Neurone

Consideriamo un neurone generico $j$ in una rete neurale. Il neurone $j$ riceve input da altri neuroni, indicati come $x_{1j}$, $x_{2j}$, $x_{3j}$, ... e produce un output $y_j$. L'output $y_j$ è calcolato applicando una funzione di attivazione $\sigma$ alla combinazione lineare degli input ponderati, chiamata $net_j$:
$$net_{j}=\sum_{i=1}^nw_{ij}\cdot x_{ij}$$

output: $o_{j}=\sigma(net_{j})$

$$w_{j}^{(t)}=\frac{\partial}{\partial w_{ij}}l(h_{\vec{w}}(\vec{x_{i}},y_{i}))=\frac{\partial l}{\partial w_{ij}}=\frac{\partial l}{\partial \ net_{j} } \cdot \frac{\partial \ net_{j}}{\partial w_{ij}}=\frac{\partial l}{ \partial \ net_{j}} \cdot x_{ij}=\partial{j}\cdot x_{ij}$$

per calcolare la derivata della loss rispetto allo stimo di ogni neurone dobbiamo distinguere due casi:
### Caso 1: $j$ è un Neurone di output

Partiamo dal caso più semplice, in cui il neurone è quello di output. In questo caso, l'uscita della rete coincide con l'output del neurone stesso. Utilizzando la notazione $\delta_{j}$ per indicare la derivata della funzione di costo rispetto allo stimolo del neurone j-esimo, possiamo scrivere:
$$\delta_{j}=\frac{\partial l}{\partial \ net_{j}}=\frac{\partial l}{\partial o_{j}}\cdot  \frac{\partial o_{j}}{\partial \ net_{j}}$$

$$\frac{\partial l}{\partial o_{j}}=\frac{\partial}{\partial o_{j}}\left[ \frac{1}{2}(h(\vec{x})-y)^2 \right]=\frac{\partial}{\partial o_{j}}\left[ \frac{1}{2}(o_{j}-y)^2 \right]=o_{j}-y$$
$$\frac{\partial o_{j}}{\partial \ net_{j}}=\frac{\partial \sigma  (net_{j}) }{\partial \ net_{j}}=\sigma(net_{j})(1-\sigma(net_{j}))=o_{j}(1-o_{j})$$
Abbiamo che questo:
$$\delta_{j}=\frac{\partial l}{\partial \ net_{j}}=\frac{\partial l}{\partial o_{j}}\cdot  \frac{\partial o_{j}}{\partial \ net_{j}}$$
Diventa:
$$(o_{j}-y) \cdot o_{j}(1-o_{j})$$

### Caso 2: Neurone hidden

Nel caso di un neurone nascosto, non abbiamo più la relazione diretta tra l'uscita della rete e l'output del neurone. In questo caso, la funzione di costo dipenderà dagli stimoli di tutti i neuroni successivi a quello considerato. Applicando la regola della catena, possiamo scrivere la derivata come una somma su tutti i neuroni successivi:

$$\partial_{j}=\frac{\partial l}{\partial \ net_{j}}=\frac{\partial l(net_{k_{1}},\dots,net_{k_{n}})}{\partial \ net_{j}}$$

Applichiamo la regola della catena
$$=\frac{\partial l}{\partial \ net_{k_{1}}}\cdot \frac{\partial \ net_{k_{1}}}{\partial \ net_{j}}+\dots+\frac{\partial l}{\partial \ net_{k_{n}}}\cdot \frac{\partial \ net_{k_{n}}}{\partial net_{j}}$$
In maniera compatta
$$\sum_{k\in \text{output(j)}} \ \frac{\partial l}{\partial \ net_{k} }\cdot \frac{\partial net_{k}}{\partial net_{j}}$$
Assegniamo
$$\delta_{k}=\frac{\partial l}{\partial \ net_{k} }$$
Dunque
$$=\sum \delta_{k}\cdot\frac{\cdot\partial net_{k}}{\partial net_{j}}=\sum\delta_{k} \cdot \frac{\partial net_{k}}{\partial o_{j}}\cdot \frac{\partial o_{j}}{\partial net_{j}}=\sum \delta_{k}w_{jk}$$
Ponendo
$$\frac{\partial net_{k}}{\partial o_{j}}=\frac{\partial}{\partial o_{j}}\sum w_{jk}o_{j}=w_{jk}$$
La precedente diventa
$$\sum \delta _k w_{ij}o_{j}(1-o_{j})=o_{j}(1-o_{j})\sum \delta_{k}w_{jk}$$


## Backpropagation Algorithm

1) $\text{Propagate the inpuit forward:}$
	$\text{for each unit j, compute}$
		$o_j=\sigma(net_{j})=\sigma\left( \sum_{i}w_{ij}x_{ij} \right)$
2) $\text{Propagate the gradient backward}$
	$for the output unit k$
		$\delta_{k}=(o_{k}-y)o_{k}(1-o_{k})$
	$\text{for each hidden unit j}$
		$\delta_{j}=o_{j}(1-o_{j})\sum_{k\in\text{output}} w_{jk}\delta_{k}$
3) $\text{for each network wheight}$
	$v_{ij}=\frac{\delta l}{\delta w_{ij}}\delta_{j}\cdot x_{ij}$


## SGD + NN

$\vec{w}^{(1)}=\text{Inizializzazione}$
$\text{for t=1 to T-1 do}$
	$\text{Seleziona casualmente }(\vec{x_{i}},y_{i})\text{ in S}$
	$\vec{v_{t}}=\text{Backpropagation}(\vec{w}^{(t)},(\vec{x_{i}},y_{i}))$
	$\nabla \vec{w}^{(t)}=-\eta \vec{v_{t}}$
	$\vec{w}^{(t+1)}=\vec{w}^{(t)}+\nabla \vec{w}^{(t)}$
$\text{return } \vec{w}^{(t)}$

#### Regolarizzazione

Ponendo
$$\vec{w^*}=\arg\min_{\vec{w}\in \mathbb{R}^{|E|}} L_{S}(\vec{w})+\lambda\|\vec{w} \|^2$$
diventa
$$\nabla \vec{w}^{(t)}=-\eta (\nabla _{t}+\lambda \vec{w}^{(t)})$$

## Problema dei minimi locali

Un problema che si può incontrare con la discesa del gradiente è quello di rimanere intrappolati in un minimo locale. Questo accade quando l'algoritmo converge a un punto che non è il minimo globale della funzione.

**Come evitare i minimi locali:**

* **Momentum:** Si aggiunge al termine di aggiornamento del gradiente una certa porzione del passo precedente. Questo aiuta a mantenere una certa velocità nella direzione corretta, evitando che l'algoritmo si blocchi in un minimo locale.
$$\nabla  \vec{w}^{(t)}=\mu \nabla  \vec{w}^{(t-1)}- \eta(\vec{v_{t}}+\lambda \vec{w}^{(t)})$$
	Dove $\mu$ rappresenta il momento

* **Tecniche di ottimizzazione stocastica:** Queste tecniche, che saranno trattate successivamente, permettono di esplorare lo spazio dei parametri in modo più efficiente, riducendo il rischio di rimanere intrappolati in un minimo locale.

