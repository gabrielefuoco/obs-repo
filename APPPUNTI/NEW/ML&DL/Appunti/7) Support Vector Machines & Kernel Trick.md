## SVM - Support Vector Machines

L'algoritmo SVM (Support Vector Machines) è un algoritmo di apprendimento che si basa sulla ricerca di semispazi. In altre parole, si tratta di un predittore lineare che cerca di trovare un iperpiano separatore. Questo iperpiano divide lo spazio in due piani, uno positivo e uno negativo.

Se i dati sono linearmente separabili, l'obiettivo è trovare un iperpiano che li divida perfettamente. Tuttavia, non tutti gli iperpiani sono uguali. Per questo motivo, si introducono dei criteri di preferenza: si desidera un iperpiano che massimizzi il margine.

### Margine

Il margine è la minima distanza tra l'iperpiano e uno dei punti del training set.

![[7) Support Vector Machines-20241018113902734.png|338]]

La distanza tra un punto $\vec{x}$ e l'iperpiano definito da $(\vec{w}, b)$ è data dalla seguente formula:

$$Dist(\vec{x},(\vec{w},b))=\frac{|(\vec{w}+\vec{x})+b|}{\|\vec{w}\|}$$

Il margine è quindi definito come:

$$\text{Margin}_{s}(\vec{w},b)=\min_{1\leq i \leq m}Dist(\vec{x},(\vec{w},b))=\min_{1\leq i \leq m} \frac{|<\vec{w},\vec{x}>+b|}{\|\vec{w}\|}$$

Nel caso di dati linearmente separabili, esiste un iperpiano separatore con errore empirico nullo. Questo significa che l'iperpiano separatore non commette errori nella classificazione dei dati del training set.

In questo caso, il margine può essere espresso come:

$$\text{Margin}_{s}(\vec{w},b)=\min_{1\leq i \leq m}\frac{y_{i}\cdot(<\vec{w},\vec{x_{i}}>+b)}{\|\vec{w}\|} >0$$

dove $y_i$ è l'etichetta del punto $\vec{x_i}$.

## Hard SVM

L'obiettivo dell'Hard SVM è trovare l'iperpiano che massimizza il margine, poichè un margine più ampio rende l'iperpiano separatore più robusto. 
Il training set è un campione casuale, quindi possiamo immaginare che i dati che non abbiamo visto possano essere approssimati perturbando il dominio dei dati. 
In questo contesto, un ampio margine garantisce una maggiore robustezza rispetto a queste perturbazioni. 

Questo si traduce nel seguente problema di ottimizzazione:

$$(\vec{w}^*,b^*)=\arg \max_{(\vec{w},b)\in R^{d+1}}\text{Margin}_{S}((\vec{w},b))$$

che può essere riscritto come:

$$(\vec{w}^*,b^*)=\arg \max_{(\vec{w},b)\in R^{d+1}}\min_{1\leq_{1}\leq,m} \frac{y_{i}(<\vec{w},\vec{x}>+b)}{\|\vec{w}\|}$$

Questo problema è soggetto ai seguenti vincoli:

$$
\begin{cases} 
(\vec{w}^*,b^*)=\arg\min_{(\vec{w},b)\in R^{d+1}} \ \|\vec{w}\|^2 \\
\forall_{i}, y_{i}(<\vec{w},\vec{x_{i}}>+b)\geq 1
\end{cases}
$$

In altre parole, si cerca l'iperpiano separatore con norma minima. Questo problema è un problema di ottimizzazione quadratica, per il quale esistono algoritmi specifici ed efficienti.

Questa formulazione è simile a quella introdotta parlando di semispazi:

$$
\begin{cases}
\vec{w}^*=\arg\min_{\vec{w}\in R^{d+1}} <0,\vec{w}> \\
\forall_{i}, y_{i}(<\vec{w},\vec{x_{i}}>+b)\geq_{1}
\end{cases}
$$

Nella formulazione originale, qualsiasi iperpiano è accettabile. Questo è un problema lineare.

Per risolvere l'Hard SVM, è necessario utilizzare algoritmi di ottimizzazione quadratica. Tuttavia, nella pratica questa formulazione non viene utilizzata perché i dati reali non sono linearmente separabili. Di conseguenza, non esiste un iperpiano separatore e l'insieme delle soluzioni è vuoto.

Per affrontare questo problema, è necessario passare a una formulazione più generale che funzioni con dati non linearmente separabili.

## Soft SVM

L'Hard SVM assume che i dati siano linearmente separabili, il che non è sempre vero nella realtà. La Soft SVM introduce un modo per gestire i punti mal classificati, ovvero i punti che si trovano dal lato sbagliato dell'iperpiano.

L'approccio della Soft SVM è quello di rilassare i vincoli dell'Hard SVM, poiché alcuni punti non li rispettano. Per fare ciò, si introduce una variabile aggiuntiva $\xi_i$ per ogni punto $x_i$. Questa variabile misura l'entità della violazione del vincolo da parte del punto.

* Se il punto si trova dal lato corretto dell'iperpiano (oltre il suo margine di competenza), $\xi_i = 0$ (il vincolo non viene violato).
* Per i punti che si trovano oltre il margine di competenza, $\xi_i$ rappresenta la distanza del punto dal margine di competenza. In questo caso, $\xi_i > 0$.

La formulazione della Soft SVM diventa:

$$
\begin{cases} 
(\vec{w}^*,b^*)=\arg\min_{(\vec{w},b)\in R^{d+1}} \ \|\vec{w}\|^2 \\ \\

\forall_{i}, y_{i}<\vec{w},\vec{x_{i}}>\ \geq 1-\xi_{i}, \ \xi_{i}\geq_{0}
\end{cases}
$$

In questo modo, i vincoli sono stati rilassati. Tuttavia, è necessario porre un vincolo sull'assegnamento dei valori a $\xi_i$. Questo si ottiene modificando la funzione obiettivo:

$$
\begin{cases} 
(\vec{w}^*,b^*)=\arg\min_{(\vec{w},b)\in R^{d+1}} \ \lambda\|\vec{w}\|^2 +\frac{1}{m} \sum_{i=1}^m \xi_{i} \\ \\

\forall_{i}, y_{i}<\vec{w},\vec{x_{i}} > \ \geq 1-\xi_{i}, \ \xi_{i}\geq_{0}
\end{cases}
$$

Nella funzione obiettivo, abbiamo due quantità non omogenee: $\|\vec{w}\|^2$ e $\sum_{i=1}^m \xi_{i}$. Per questo motivo, si introduce un iperparametro $\lambda$ che serve a pesare una delle due grandezze.

Il valore di $\xi_i$ può essere espresso come:

$$\xi_{i}=
\begin{cases}
0, \ se \ y_{i} <\vec{w},x_{i}> \ \geq 1\\
 \\
1-y_{i}<\vec{w},x_{i}>, \ \text{ altrimenti}
\end{cases}

$$

Questo valore corrisponde al valore della loss surrogata introdotta precedentemente:

$$\xi_{i}=l_{hinge}(\vec{w},(\vec{x_{i},y_{i}}))=\max \{ 0,1-y_{i}<\vec{w},\vec{x_{i}>} \}$$

Otteniamo dunque questa formulazione per la Soft SVM:

$$
\vec{w}^*=\arg\min_{\vec{w}\in R^{d+1}} \ \lambda \|\vec{w}\|^2 +\frac{1}{m}\sum_{i=1}^m \ l_{hinge}(\vec{w,(\vec{x_{i}},y_{i})})
$$

che può essere riscritta come:

$$
\vec{w}^*=\arg\min_{\vec{w}\in R^{d+1}} \ L_{S}^{\text{hinge}}+ \lambda \|\vec{w}\|^2 
$$

Questa formulazione è un esempio di Regularized Linear Model (RLM). $L_S^{hinge}$ è una funzione convessa ed è la surrogata di $L_S^{0-1}$.

La Soft SVM è l'algoritmo che cercavamo per il learning dei semispazi. Tuttavia, abbiamo un errore di generalizzazione perché stiamo usando la funzione surrogata. La funzione hinge è Lipschitz-bounded, con Lipschitzness $\rho$:

$$\rho=\| \nabla_{\vec{w}}l_{hinge}(\vec{w,(\vec{x},y)})\|= \| \vec{x_{i}} \|$$

$$\rho =\max_{1\leq i\leq m} \|\vec{x_{i}}\|$$

## SGD + RLM + SVM

$$\Theta^{(1)}=0$$
$$\text{for t=1 to T do}$$
$$\vec{w}^{(t)}=\frac{1}{\delta t}\theta^{(t)}$$
$$\text{seleziona casualmente }(\vec{x_{i}},y_{i})\  in \ S$$
$$\text{if }y_{i}<\vec{w}^{(t)},\vec{x}> \ <1$$
$$\vec{\theta}^{(t+1)}=\vec{\theta}^{(t)}+y_{i}\vec{x_{i}}$$
$$\text{return }\vec{w}^{(T+1)}$$

### Teorema di Rappresentazione

Dato un problema di ottimizzazione della forma:

$$
\vec{w}^* =\arg\min_{\vec{w}\in R^d} f(<\vec{w},\vec{x}_{1}>,<\vec{w},\vec{x}_{2}>,\dots<\vec{w},\vec{x}_{m}>)+g(\|\vec{w}\|)
$$

con $f$ e $g$ funzioni reali, allora la soluzione ottima $\vec{w}^*$ può essere espressa come:

$$\vec{w}^*=\sum \alpha_{i} \vec{x_{i}}$$

Nel caso del SVM, vale la proprietà che i coefficienti $\alpha$, quelli non nulli, sono quelli associati ai punti $x_{i}$ che soddisfano i vincoli per uguaglianza, ovvero quelli sul margine o oltre il margine. La soluzione dipende solo da alcuni punti, che proprio per questa proprietà vengono chiamati **Vettori Di Supporto**.

Questa proprietà può essere sfruttata per estendere l'applicabilità di questo algoritmo ai casi in cui i separatori lineari si comportano male.

### Formulazione SVM in Termini del Teorema di Rappresentazione

$$\vec{w}^*=\arg\min_{\vec{w\in R^{(d+1)}}} \ \lambda \|\vec{w}\|^2+\frac{1}{m}\max \{ 0,1-y_{i}<\vec{w},\vec{x_{i}} \}$$

Cerchiamo il vettore dei pesi $\vec{\alpha}$:

$$\vec{\alpha}=(\alpha_{1},\alpha_{2},\dots,\alpha_{m})$$

$$
<\vec{w},\vec{x_{i}}> \ = \ <\sum_{j=1}^m \alpha_{j}\vec{x_{j}},\vec{x_{i}} >\ = \sum \alpha_{j}<\vec{x_{j}},\vec{x_{i}}>
$$

$$\| \vec{w}\|^2= \ <\vec{w},\vec{w} > \ = \ <\sum_{i=1}^m \alpha_{i}\vec{x_{i}},\sum_{j=1}^m \alpha_{j}\vec{x}_{j}>=\sum_{i=1}^m \sum_{j=1}^m<\vec{x_{i}},\vec{x_{j}}>$$.

## SGD + RLM + SVM + Teorema di Rappresentazione

$$\alpha^{(t)} \in R^m \leftrightarrow \vec{w}^{(t)}$$
$$\beta^{(t)} \in R^m \leftrightarrow \vec{\theta}^{(t)}$$

$$\beta^{(1)=0}$$

$$\text{for t=1 to T do}$$
$$\vec{\alpha}^{(t)}=\frac{1}{\delta t}\beta^{(t)}$$
$$\text{seleziona casualmente }(\vec{x_{i}},y_{i})\  in \ S$$
$$\text{if }y_{i} \sum_{j=1}^m \alpha_{j}<\vec{x_{j}},\vec{x_{i}}> \ <1$$
$$\vec{\beta_{i}}^{(t+1)}=\vec{\beta_{i}}^{(t)}+y_{i}$$
$$\text{return }\vec{w}^{(T+1)}=\sum_{i=1}^m \alpha_{i}\vec{x_{i}}$$

Il costo di questo algoritmo è più elevato rispetto ad altri. Tuttavia, permette di estendere l'applicabilità a problemi che sono quasi non linearmente separabili.

Dal punto di vista tecnico, nella formulazione alternativa l'algoritmo accede ai dati solo tramite un prodotto scalare $\alpha_{j}<\vec{x_{j}},\vec{x_{i}}>$.

### Caso Base: Dati su Retta Reale

Una soglia è l'equivalente di un iperpiano nel caso unidimensionale. L'idea è quella di alzare la dimensionalità dei dati (Tecnica di embedding in uno spazio ad alta dimensionalità) passando da $R$ a $R^2$.

$\phi(x)=(x,x^2)$, ($x^2$ diventa la nuova y)

Questa tecnica pone alcuni problemi:

1. **Learnability:** in generale non è mai buono aumentare la dimensionalità dei dati. Se aumentiamo il numero di feature ci aspettiamo che la qualità dei dati andrà a calare. 
 SVM è un problema complex-lip..-bounded
 $m=\frac{8p^2b^2}{\epsilon^2}$ sample complexity è indipendente dalla dimensione dello spazio

2. **Problema computazionale:**
$$\vec{x_{i}} \to \vec{x_{i}'=\phi(\vec{x_{i}})}$$
$$\phi:R^d\to R^{d_{2}}, d_{2}\gg d$$
$$<\vec{x}_{i},\vec{x_{j}}> \ \leftrightarrow \ <\phi(\vec{x_{i}}),\phi(\vec{x_{j}})>$$
 il primo termine costa $O(d)$ e il secondo costa $O(d_2)$

## Introduzione alle Funzioni Kernel

In Machine Learning, l'aumento della dimensionalità è una tecnica che migliora la capacità dei modelli di apprendere relazioni complesse nei dati. Questo approccio è particolarmente utile per algoritmi come le Support Vector Machine (SVM).

Le SVM sono classificate come problemi "sparse-bounded", il cui livello di complessità non dipende dalla dimensione dello spazio delle features, ma da parametri intrinseci come Rho e B. Questo permette di aumentare la dimensionalità senza incorrere in problemi di complessità.

Tuttavia, l'aumento della dimensionalità comporta un aumento esponenziale delle dimensioni dello spazio dei dati, con conseguente aumento del costo computazionale.

Per sfruttare i vantaggi di una maggiore dimensionalità senza pagarne il prezzo computazionale, si utilizza il "kernel trick". Questo strumento permette di applicare trasformazioni di dimensionalità senza dover calcolare esplicitamente le coordinate dei dati nello spazio trasformato. 

In questo contesto, consideriamo:

* **Vettori di input:** $\vec{x_{i}}$ nello spazio di input $\mathbb{R}^{d_{1}}$.
* **Trasformazione non lineare:** $\phi(\vec{x_i})$ che mappa i vettori di input in uno spazio di output ad alta dimensione $\mathbb{R}^{d_{2}}$.
* **Relazione tra dimensioni:** $d_{1} \ll d_{2}$, ovvero la dimensione dello spazio di input è molto minore della dimensione dello spazio di output.

**Prodotto scalare:**

* Nello spazio di input: $<\vec{x_i},\vec{x_{j}}>$.
* Nello spazio di output: $<\phi (\vec{x_i}),\phi(\vec{x_{j})}>$.

**Costo computazionale:**

* $O(d_{1})$: indica un costo computazionale proporzionale alla dimensione dello spazio di input.
* $O(d_{2})$: indica un costo computazionale proporzionale alla dimensione dello spazio di output.

## Funzioni Kernel $K_{\phi}$

Una funzione kernel è una funzione che, dati due punti nello spazio di input, restituisce direttamente il valore del loro prodotto scalare nello spazio trasformato, senza dover effettivamente calcolare le trasformazioni. Formalmente:

Data una trasformazione $\phi: \mathbb{R}^{d_{1}} \to \mathbb{R}^{d_{2}}, (d_{2} \gg d_{1})$, una funzione kernel $K_{\phi}: \mathbb{R}^{d_{1}} \times \mathbb{R}^{d_{1}} \to \mathbb{R}$ è tale che:

$$\forall \vec{x_{1}},\vec{x_{2}} \in \mathbb{R}^{d_{1}}, K_{\phi}(\vec{x_{1}},\vec{x_{2}})\ = \ <\phi (\vec{x_1}),\phi(\vec{x_{2})}>$$

Le funzioni kernel interessanti hanno costo computazionale $O(d_{1})$. 
Se troviamo una funzione avente questa proprietà, abbiamo risolto il nostro problema.

**Nota:**

* La notazione $<\cdot, \cdot>$ indica il prodotto scalare.
* La condizione $d_{1} \ll d_{2}$ indica che la dimensione dello spazio di input $d_{1}$ è molto minore della dimensione dello spazio di output $d_{2}$.
* Il costo computazionale $O(d_{1})$ indica che il tempo necessario per calcolare la funzione kernel è proporzionale alla dimensione dello spazio di input.

## Kernel Polinomiale

Il kernel polinomiale è un esempio di funzione kernel che permette di eseguire una trasformazione non lineare dei dati nello spazio di input, proiettandoli in uno spazio di output ad alta dimensione.

**Definizione:**

Dato un vettore di input $x \in \mathbb{R}$, il prodotto scalare tra il vettore dei pesi $\vec{w}$ e la trasformazione non lineare $\phi(x)$ è definito come:

$$<\vec{w},\phi(x)>\  = w_{0}+w_{1}x+w_{2}x^{2}+\dots+w_{n}x^n$$

dove il prodotto scalare rappresenta la normale al nostro iperpiano.

**Trasformazione:**

Possiamo riscrivere il prodotto scalare come:

$$=\ <w_{0}+w_{1}+\dots+w_{n}>,\ (1,x,x^2,\dots,x^n)>$$

Dove $(1,x,x^2,\dots,x^n)= \phi(x)$ è la trasformazione non lineare che mappa il vettore di input $x$ in uno spazio di output ad alta dimensione.

**Forma del Kernel Polinomiale:**

Il kernel polinomiale ha la seguente forma:

$$K_{\phi_{n}}(\vec{x_{1}},\vec{x_{2}})=(1+<\vec{x_{1}},\vec{x_{2}}>)^n$$

**Costo computazionale:**

Il costo computazionale del kernel polinomiale è $O(d_{1})$, a causa del prodotto scalare tra i vettori di input.

**Esempio:**

Consideriamo il caso $d_1=1, \ n=2$:

$$
\begin{align*}
&K_{\phi_{n}}({x_{1}},{x_{2}})=
\\&(1+x_{1}x_{2})^2=x_{1}^2x_{2}^2+a+2x_{1}x_{2}=
\\&(1,\sqrt{ 2 }x_{1},x_{1}^2),(1,\sqrt{ 2 }x_{2},x_{2}^2)=\\
& \ <\phi_{2}(x_{1}),\phi_{2}(x_{2})>
\end{align*}
$$

**Caso multidimensionale:**

Per un vettore di input multidimensionale $\vec{x_{1}} = (x_{11}, x_{12})$ e $\vec{x_{2}} = (x_{21}, x_{22})$, il kernel polinomiale diventa:

$$\begin{aligned} k_{\phi_{2}}(\vec{x_{1}},\vec{x_{2}})&=(1+<\vec{x_{1}},\vec{x_{2}}>)^n=\\ &=(1+<(x_{11},x_{12}),(x_{21},x_{22})>)^2\\ &=1+x_{11}^2x_{21}^2+x_{12}^2,x_{22}^2+2x_{11}x_{21}+2x_{12}x_{22}+2x_{11}x_{12}x_{21}x_{22}\\ &=<(1,\sqrt{ 2 }x_{11},\sqrt{ 2 }x_{12},\sqrt{ 2 }x_{11}x_{12},x_{11}^2,x_{12}^2),(1,\sqrt{ 2 }x_{21},\sqrt{ 2 }x_{22},\sqrt{ 2 }x_{21}x_{22},x_{21}^2,x_{22}^2)>\\ &=<\phi_{2}(\vec{x_{1}}),\phi_{2}(\vec{x_{2}})> \end{aligned}$$

$$d_{2}=\exp \ in \ d_{1}\in n$$
??????

## Support Vector Machine + Kernel

L'algoritmo può essere descritto come segue
$$\begin{aligned}
&\beta^{(1)}=\vec{0} \\
&\text{for } T=1 \text{ to } T-1 \text{ do} \\
&\qquad \vec{\alpha}^{(t)}=\frac{1}{\lambda t}\vec{\beta}^{(t)} \\
&\qquad \text{seleziona casualmente } (\vec{x_{i}},y_{i}) \text{ in } S \\
&\qquad \text{if } y_{1} \sum_{j=1}^m \alpha_{j} <\phi(\vec{x_{i}}),\phi(\vec{x_{j}})> \ <1 \\
&\qquad \beta_{i}^{(t+1)}=\beta_{i}^{(t)}+y_{i} \\
&\text{return } \ \vec{\alpha}
\end{aligned}$$

Se esiste una funzione kernel, possiamo sostituire il prodotto scalare nello spazio di destinazione con il valore della funzione kernel:
$$y_{1} \sum_{j=1}^m \alpha_{j} K_{\phi}(\vec{x_{i}}\vec{x_{j}}) \ <1$$
Questo permette di calcolare il prodotto scalare senza dover esplicitamente calcolare la proiezione dei dati nello spazio di caratteristiche, rendendo l'algoritmo più efficiente.

## Kernel Gaussiano

Il kernel gaussiano può essere visualizzato come una funzione che assegna un peso ai punti in base alla loro distanza. Punti vicini avranno un peso maggiore, mentre punti lontani avranno un peso minore.

$$K(\vec{x},\vec{x}')=\exp\left[ - \frac{\|\vec{x}-\vec{x}'\|^2}{2\sigma^2} \right]$$

dove:

- $\vec{x}$ e $\vec{x}'$ sono due punti nello spazio di input.
- $\sigma$ è un parametro che controlla la larghezza del kernel.

![[8)-20241024123135373.png|384]]

Il kernel gaussiano è molto potente perché permette di utilizzare separatori lineari anche su problemi complessi, in cui i dati non sono linearmente separabili nello spazio originale.

Ad esempio, immaginiamo un dataset composto da due spirali concentriche, una per ogni classe. Separare linearmente queste due classi nello spazio bidimensionale è impossibile. Tuttavia, utilizzando un kernel gaussiano, possiamo proiettare implicitamente i dati in uno spazio a dimensionalità superiore, dove le due spirali diventano linearmente separabili.
![[8)-20241024123236760.png|183]]
Da un punto di vista teorico, il kernel gaussiano proietta i dati in uno spazio a **infinite dimensioni**. Questo significa che, in linea di principio, possiamo separare qualsiasi dataset utilizzando un kernel gaussiano.

Teoricamente, il kernel gaussiano effettua un mapping a infinite dimensioni. Per illustrare questo concetto, consideriamo il caso a una dimensione:

$$K(x,x')=\exp\left[ - \frac{(x-x')^2}{2} \right]$$

Utilizzando la proprietà della funzione esponenziale:

$$\exp(x)=\sum_{n=0}^{\infty} \frac{x^n}{n!}$$

possiamo riscrivere il kernel gaussiano come:

$$\exp\left[ - \frac{x^2+x'^2-2xx'}{2} \right]$$
$$=\exp\left( -\frac{x^2+x'^2}{2} \right) \exp(x x')$$
$$=\exp\left( -\frac{x^2+x'^2}{2} \right)=\sum_{n=0}^{\infty} \frac{(xx')^2}{n!}$$

Espandendo l'ultima espressione, otteniamo:

$$\sum_{n=0}^{\infty} \left\{ \left[ {\frac{x^n}{n!}} \exp\left( - \frac{x^{2}}{2} \right) \right] \cdot \left[ {\frac{x'^n}{n!}} \exp\left( - \frac{x'^{2}}{2} \right) \right] \right\}$$

Questa espressione rappresenta un prodotto scalare infinito di due vettori $\phi(x)_{n}$ e $\phi(x')_{n}$.

