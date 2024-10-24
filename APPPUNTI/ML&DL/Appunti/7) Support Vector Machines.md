## SVM - Support Vector Machines

L'algoritmo SVM (Support Vector Machines) è un algoritmo di apprendimento che si basa sulla ricerca di semispazi. In altre parole, si tratta di un predittore lineare che cerca di trovare un iperpiano separatore. Questo iperpiano divide lo spazio in due piani, uno positivo e uno negativo.

Se i dati sono linearmente separabili, l'obiettivo è trovare un iperpiano che li divida perfettamente. Tuttavia, non tutti gli iperpiani sono uguali. Per questo motivo, si introducono dei criteri di preferenza: si desidera un iperpiano che massimizzi il margine.

### Margine

Il margine è la minima distanza tra l'iperpiano e uno dei punti del training set.

![[7) Support Vector Machines-20241018113902734.png|338]]

La distanza tra un punto $\vec{x}$ e l'iperpiano definito da $(\vec{w}, b)$ è data dalla seguente formula:

$$Dist(\vec{x},(\vec{w},b))=\frac{|(\vec{w}+\vec{x})+b|}{\|\vec{w}\|}$$

Il margine è quindi definito come:

$\text{Margin}_{s}(\vec{w},b)=\min_{1\leq i \leq m}Dist(\vec{x},(\vec{w},b))=\min_{1\leq i \leq m} \frac{|<\vec{w},\vec{x}>+b|}{\|\vec{w}\|}$

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
(\vec{w}^*,b^*)=\arg\min_{(\vec{w},b)\in R^{d+1}} \ \|\vec{w}\|^2   \\
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
(\vec{w}^*,b^*)=\arg\min_{(\vec{w},b)\in R^{d+1}} \ \|\vec{w}\|^2   \\ \\

\forall_{i}, y_{i}<\vec{w},\vec{x_{i}}>\ \geq 1-\xi_{i}, \ \xi_{i}\geq_{0}
\end{cases}
$$

In questo modo, i vincoli sono stati rilassati. Tuttavia, è necessario porre un vincolo sull'assegnamento dei valori a $\xi_i$. Questo si ottiene modificando la funzione obiettivo:

$$
\begin{cases} 
(\vec{w}^*,b^*)=\arg\min_{(\vec{w},b)\in R^{d+1}} \ \lambda\|\vec{w}\|^2 +\frac{1}{m} \sum_{i=1}^m \xi_{i}   \\ \\

\forall_{i}, y_{i}<\vec{w},\vec{x_{i}} > \ \geq 1-\xi_{i}, \ \xi_{i}\geq_{0}
\end{cases}
$$

Nella funzione obiettivo, abbiamo due quantità non omogenee: $\|\vec{w}\|^2$ e $\sum_{i=1}^m \xi_{i}$. Per questo motivo, si introduce un iperparametro $\lambda$ che serve a pesare una delle due grandezze.

Il valore di $\xi_i$ può essere espresso come:

$$\xi_{i}=
\begin{cases}
0, \  se \ y_{i}  <\vec{w},x_{i}> \ \geq 1\\
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

$$\rho=\| \nabla_{\vec{w}}l_{hinge}(\vec{w,(\vec{x},y)})\|= \| \vec{x_{i}}   \|$$

$$\rho =\max_{1\leq i\leq m} \|\vec{x_{i}}\|$$

## SGD + RLM + SVM

$\Theta^{(1)}=0$
$\text{for t=1 to T do}$
	$\vec{w}^{(t)}=\frac{1}{\delta t}\theta^{(t)}$
	 $\text{seleziona casualmente }(\vec{x_{i}},y_{i})\  in \ S$
	$\text{if }y_{i}<\vec{w}^{(t)},\vec{x}> \ <1$ 
		$\vec{\theta}^{(t+1)}=\vec{\theta}^{(t)}+y_{i}\vec{x_{i}}$
 $\text{return }\vec{w}^{(T+1)}$


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

$\vec{\alpha}=(\alpha_{1},\alpha_{2},\dots,\alpha_{m})$

$$
<\vec{w},\vec{x_{i}}> \ = \ <\sum_{j=1}^m  \alpha_{j}\vec{x_{j}},\vec{x_{i}} >\ = \sum \alpha_{j}<\vec{x_{j}},\vec{x_{i}}>
$$

$$\| \vec{w}\|^2= \ <\vec{w},\vec{w} > \ = \ <\sum_{i=1}^m \alpha_{i}\vec{x_{i}},\sum_{j=1}^m \alpha_{j}\vec{x}_{j}>=\sum_{i=1}^m \sum_{j=1}^m<\vec{x_{i}},\vec{x_{j}}>$$.


## SGD + RLM + SVM + Teorema di Rappresentazione

$\alpha^{(t)} \in R^m \leftrightarrow \vec{w}^{(t)}$
$\beta^{(t)} \in R^m \leftrightarrow \vec{\theta}^{(t)}$


$\beta^{(1)=0}$

$\text{for t=1 to T do}$
	$\vec{\alpha}^{(t)}=\frac{1}{\delta t}\beta^{(t)}$
	 $\text{seleziona casualmente }(\vec{x_{i}},y_{i})\  in \ S$
	 $\text{if }y_{i} \sum_{j=1}^m \alpha_{j}<\vec{x_{j}},\vec{x_{i}}> \ <1$ 
		$\vec{\beta_{i}}^{(t+1)}=\vec{\beta_{i}}^{(t)}+y_{i}$
 $\text{return }\vec{w}^{(T+1)}=\sum_{i=1}^m \alpha_{i}\vec{x_{i}}$

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
   $\vec{x_{i}} \to \vec{x_{i}'=\phi(\vec{x_{i}})}$
   $\phi:R^d\to R^{d_{2}}, d_{2}\gg d$
   $<\vec{x}_{i},\vec{x_{j}}> \ \leftrightarrow \ <\phi(\vec{x_{i}}),\phi(\vec{x_{j}})>$
   il primo termine costa $O(d)$ e il secondo costa $O(d_2)$



