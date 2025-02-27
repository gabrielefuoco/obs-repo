## Rete Neuronale: Perceptron

**Introduzione:**

* La rete riceve input, non tutti con la stessa importanza.
* Ogni input è associato ad un **peso** che ne determina l'influenza.
* Lo **stimolo** ricevuto dalla rete è la combinazione lineare degli input ponderati: 

$$\sum_{i=1}^d w_{i}x_{i}$$

* Se lo stimolo supera una certa **soglia** ($\theta$), la rete attiva un'uscita.

**Funzione di Attivazione:**

* La funzione **segno** ($sign$) implementa un semispazio:

$$h_{\vec{w},\theta}(\vec{x})=sign[\sum_{i=1}^d w_{i}x_{i}-\theta]=sign<\vec{w},\vec{x}>$$

* **Esempio:** Funzione AND

**Algoritmo di Apprendimento:**

* **Inizializzazione:** $\vec{w}^{(1)}=(0,0,\dots,0)$
* **Iterazione:**
 * Finché esiste un input $i$ per cui $y_{i} <\vec{w}^{(1)},\vec{x_{i}}> \leq_{0}$:
 * Aggiorna il vettore dei pesi: $\vec{w}^{(t+1)}=\vec{w}^{(1)}+y_{i}\vec{x_{i}}$
 * Incrementa il contatore di iterazioni: $t=t+1$

**Convergenza:**

* L'algoritmo converge sempre nel caso di dati **linearmente separabili**.
* Al termine, restituisce un **iperpiano separatore**. 

## Teorema del Perceptron

**Teorema:** Sia S linearmente separabile, allora l'algoritmo del Perceptron converge al più in T iterazioni con $T \le (RB)^2$.

**Definizioni:**

* $R = \max_{1 \le i \le m} |\vec{x_i}|$ è la massima norma dei vettori di input.
* $B = \min{||\vec{w}||}: \forall i, <\vec{w_i}, \vec{x_i}> \ge 1$ è la più piccola norma che restituisce una soluzione al problema di separazione lineare.

**Dimostrazione:**

Sia $w^*$ un vettore soluzione a norma minima. Allora:
$$1 \ge \cos \theta = \frac{<w^*, w^{t+1}>}{||w^*|| ||w^{t+1}||} \ge \frac{\sqrt{T}}{RB}$$

dove $\theta$ è l'angolo tra $w^*$ e $w^{t+1}$.

![[3)-20241005171105171.png|159]]

Questa disuguaglianza ci dice che più si procede con le iterazioni, più l'angolo tra $w^*$ e $w^{t+1}$ si riduce.

Isolando $T$ dalla disuguaglianza, si ottiene $T \le RB$.

Per dimostrare la disuguaglianza iniziale, consideriamo separatamente numeratore e denominatore:

1. **Numeratore:**

 Scriviamo il numeratore come la differenza del prodotto vettoriale: $$<w^*, w(t+1)> - <w^*, w(t)>.$$
 Possiamo riscriverlo come:
 $$<w^*, w(t+1) - w(t)> = <w^*, y_i x_i>.$$
 Portiamo fuori lo scalare:
 $$y_i <w^*, x_i> \ge 1$$
 Quindi sappiamo che la nostra differenza iniziale è $\ge 1$.
 $$<\vec{w}^*, \vec{w}^{(t+1)}> = \sum_{i=1}^I(<\vec{w}^*, w^{(t+1)}>-<w^*, w^{()}>) \ge T$$
 perché viene ripetuto T volte.

2. **Denominatore:**
 $$||w(t+1)||^2 = ||w_t + y_i x_i||^2$$
 Sviluppiamo il quadrato:
 $$||w_t||^2 + 2y_i <w_t, x_i> + y_i^2 ||x_i||^2$$
 Notiamo che abbiamo fatto un'iterazione, $x$ era missclassificato, quindi $2y_i <w_t, x_i> \le 0$, e $y_i^2$ è la nostra etichetta che vale 1. Allora possiamo riscrivere tutto come:
 $$||w_t||^2 + 2y_i <w_t, x_i> + y_i^2 ||x_i||^2 \le ||w_t||^2 + ||x_i||^2 \le ||w_t||^2 + R^2$$
 Siccome $||x_i|| \le R$, l'abbiamo sostituita con $R$.

 Considerando la T-esima iterazione, abbiamo che:
 $$||w(t+1)||^2 \le ||w_t||^2 + R^2 \le ||w(t-1)||^2 + R^2 \le TR^2$$
 Mettendo tutto insieme, abbiamo:
 $$\frac{<\vec{w}^*, \vec{w}^{(t+1)}>}{||w^*|| ||w^{t+1}||} \ge \frac{T \sqrt{T}}{B \sqrt{T}R} = \frac{\sqrt{T}}{BR}$$
 Il teorema è dimostrato.

**Conclusione:**

Il teorema del Perceptron dimostra che l'algoritmo del Perceptron converge in un numero finito di iterazioni se il set di dati è linearmente separabile. Questo risultato è importante perché fornisce una garanzia di convergenza per l'algoritmo.

**Vantaggi:**

* **Semplicità:** L'algoritmo del Perceptron è relativamente semplice da implementare.
* **Efficienza:** L'algoritmo del Perceptron può essere molto efficiente per set di dati di piccole dimensioni.

**Svantaggi:**

* **Linearità:** L'algoritmo del Perceptron può solo classificare set di dati linearmente separabili.
* **Sensibilità al rumore:** L'algoritmo del Perceptron può essere sensibile al rumore nei dati.

## Regressori Lineari

Parliamo di regressione quando il dominio target (la nostra *y*) è finito. Vogliamo predire dei valori in un intervallo reale.

Si tratta di costruire una funzione *h* che, dato *x*, ci fornisce la nostra *y*. Se vogliamo usare la regressione lineare, la nostra *h* sarà una funzione lineare.

Nel caso della regressione lineare, la nostra funzione è la funzione identità.
$$h_w(x,y) = \langle w,x \rangle$$
Come facciamo a misurare la bontà dell'iperpiano? Ci serve una *loss*, che quantifica l'errore che commettiamo quando usiamo l'output del percettore per approssimare la relazione.

Nel caso della regressione, la *loss* utilizzata è quella quadratica.
$$lsq(h,(x,y)) = (h(x)-y)^2$$
Andiamo a specializzarla nel caso della regressione lineare:
$$lsq(h_w,(x,y)) = (\langle w,x \rangle -y)^2$$
L'errore empirico:
$$L_{S}(h_{\vec{w}})=\frac{1}{m}\sum_{i=1}^m(<\vec{w}_{i}, \ \vec{x}_{i}>-y_{i})^2$$
Per trovare il nostro regressore dobbiamo trovare l'iperpiano che minimizza la relazione.

Ci sono due strade per risolvere il problema:

1. **Trovare una soluzione in forma chiusa.**

Studiamo i punti stazionari della funzione. Questa funzione ha una proprietà, ovvero quella di essere convessa.

Ci calcoliamo il gradiente della nostra *Loss* e lo poniamo uguale a zero.
$$\nabla_{\vec{w} L_{s}}(h_{\vec{w}})=0$$
Facciamo dei calcoli:
$$
\begin{aligned}
\nabla_{\vec{w}} L_s(h_{\vec{w}}) &= \nabla_{\vec{w}} [\frac{1}{m} \sum_{i=1}^m (<\vec{w},\vec{x_i}>-y_i)^2] \\
&= \frac{1}{m} \sum_{i=1}^m \nabla(<\vec{w},\vec{x_i}>-y_i)^2 \\
&= \frac{1}{m} \sum_{i=1}^m 2(<\vec{w},\vec{x_i}>-y_i) \cdot \nabla_{<\vec{w},x_i>} \\
&= \frac{1}{m} \sum_{i=1}^m 2(<\vec{w},\vec{x_i}>-y_i) \cdot \vec{x_i} \\
&= \frac{2}{m} \sum_{i=1}^m \ x_i(<\vec{w},\vec{x_i}>-y_i) \\
&= \frac{2}{m} \sum_{i=1}^m \vec{x_i}<\vec{w},\vec{x_i}> - \frac{2}{m} \sum_{i=1}^m y_i\ x_i = 0 \\
&= \frac{2}{m} \sum_{i=1}^m \vec{x_i}<\vec{w},\vec{x_i}> = \frac{2}{m} \sum_{i=1}^m y_i\ x_i
\end{aligned}
$$

Riscriviamolo in forma matriciale.
$$
X = \begin{pmatrix}
\vec{x_1} \\
\vdots \\
\vec{x_m}
\end{pmatrix}
$$

Riscriviamolo in forma ridotta:

$$(X^T X)w = X^T y$ , sappiamo che $(X^T X) = A$ e $X^T y = b$ quindi otteniamo che $Aw=b$$

Tutti i problemi che hanno una *Loss* Convessa possono essere risolti con un algoritmo generale.

## Regressione Polinomiale

Non sempre un regressore lineare risulta ottimo, per questo per migliorare la qualità della nostra regressione utilizziamo un polinomio.

Utilizzando un polinomio, il nostro modello diventa:
$$p_n(x) = a_0x_0 + a_1x + a_2x^2 + ... + a_nx^n$$
Possiamo riscrivere il polinomio in una forma più familiare in questo modo:
$$h_w(x) = w_0 + w_1x + w_2x^2 + ... + w_nx^n = <(w_0, w_1, ..., w_n), (1, x, x^2, ..., x^n)>$$
Definiamo la funzione $\phi$:
$$\phi: R \rightarrow R^{n+1}$$
$$\phi(x) = (1, x, x^2, x^3, ..., x^n)$$

Il nostro regressore polinomiale è:
$$h_w(x) = <w, \phi(x)>$$
Abbiamo ricondotto la regressione polinomiale ad una regressione lineare. Come procediamo ora? Dobbiamo solo andare a trasformare i dati.
$$S: \{x, x, ..., x_m\} \rightarrow S' = \{\phi(x_1), \phi(x_2), ..., \phi(x_m)\}$$

## Regressione Logistica

A differenza di quanto suggerisce il nome, la regressione logistica non si utilizza per i problemi di regressione, ma per problemi di classificazione binaria. Che cosa accade? Il regressore logistico, a differenza di un classificatore "classico", non restituisce l'etichetta della classe, ma bensì abbiamo un classificatore probabilistico che restituisce una probabilità che $x$ appartenga alla classe positiva.
$$h(x) = Pr[x \epsilon \text{classe positiva}] \epsilon [0, 1]$$

**Funzionamento:** L'idea è la seguente, si basa sull'uso di modelli lineari (quindi su iperpiani), supponiamo di avere dei dati.

* $h = 1$: certezza che $x$ sia della classe positiva.
	* I valori compresi tra questi due si propendono per l'appartenenza alla classe positiva.
* $h = 1/2$: massima incertezza di appartenenza.
	* I valori compresi tra questi due valori si propendono per l'appartenenza alla classe negativa.
* $h = 0$: certezza che appartiene ad un'altra classe.

$<w, x>$ è proporzionale alla $dist(w, x)$.

Per convertire la distanza in probabilità utilizziamo la funzione Sigmoide. La Sigmoide ci permette di fare un mapping di un numero reale sull'intervallo [0, 1].

$$\sigma: R \rightarrow [0, 1]$$

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$h_{\vec{w}}(x) = \sigma(<\vec{w}, \vec{x}>) = \frac{1}{1 + e^{-<w, x>}}$$

Abbiamo bisogno di una nuova Loss, la loss che utilizzeremo è la Cross Entropy Loss. 

## Cross Entropy Loss

La Cross Entropy Loss misura la distanza tra due distribuzioni. Nel caso della classificazione binaria, la formula è:

$$l_{CE}(h,(x,y))= - y \log(h(x)) - (1-y) log(1-h(x))$$
![[3)-20241005212757080.png|270]]

Dove $y$ è l'etichetta vera.

* Se $y=0$, allora $l_{CE}=- log(1-h(x))$.
* Se $y=1$, allora $l_{CE}=-log(h(x))$.

Nel caso $y=1$, se $h(x)=1$ la loss è 0. Se la predizione è completamente errata, l'errore diverge.

Nel caso $y=0$, se $h(x)=1$ la loss diverge.

## Caso della Regressione Logistica

La Cross Entropy Loss per la regressione logistica è:
$$l_{log}(h_w,(x,y))= -y \log σ(<w,x>) - (1-y) log ( σ(- <w,x>))$$
Possiamo semplificarla utilizzando la proprietà della sigmoide:
$$1-\sigma(x)=1-\frac{1}{1+e^{-x}}=\frac{1+e^{-x}-1}{1+e^{-x}} \cdot \frac{e^x}{e^x}=\sigma(-x)$$
Sostituendo nella formula precedente, otteniamo:
$$l_{log}(h_w,(x,y))= -y \log σ(<w,x>) - (1-y) log ( σ( - <w,x>))$$
Assumiamo che le etichette siano in $ϵ(0,1)$.
$$y \in \{0,1\} \Rightarrow y \in \{-1, +1\} \text{ utilizzando le etichette così}$$
difatti
$$
\begin{cases}
y = 0 &\Rightarrow -\log \sigma(<\vec{w},\vec{x}>) \\
y = 1 &\Rightarrow -\log \sigma(-<\vec{w},\vec{x}>) \\
\end{cases}
\Rightarrow -\log(y<\vec{w},\vec{x}>) \\
$$
$$l_{log}(h_y(x,y)) = -\log \sigma(y<\vec{w},\vec{x}>) = -\log \frac{1}{1+e^{-y<\vec{w},\vec{x}>}} =$$
$$= \log [1 + e^{y<\vec{w},\vec{x}>}]$$

La Cross Entropy è convessa, quindi anche la Regressione Logistica è convessa.

## No Free-Lunch Theorem

Sia $X$ un dominio infinito e $Hx$ la classe d'ipotesi formata da tutte le funzioni da $X \rightarrow (0,1)$. L'insieme $Hx$ è Learnable? No, questa classe d'ipotesi NON È Learnable.

**Definizione formale:**

Sia $X$ un dominio infinito e $Hx$ la classe d'ipotesi formata da tutte le funzioni da $X \rightarrow \{0,1\}$, allora $Hx$ NON È LEARNABLE.

Concettualmente, questo teorema ci dice che non esiste un algoritmo di learning universale.

## VC-Dimension

Serve per caratterizzare la learnability delle classi di ipotesi infinite. Come funziona? Si parte da delle definizioni preliminari:

1. **Restrizione di H a C dove C è un sottoinsieme di X:** È l'insieme delle funzioni che hanno C come dominio e possono essere derivate da H.

| X | h_1 | h_2 | h_3 | h_4 |
| --- | --- | --- | --- | --- |
| x_1 | 1 | 1 | 0 | 1 |
| x_2 | 1 | 1 | 1 | 0 |
| x_3 | 0 | 1 | 0 | 1 |
H_c=?

2. **SHATTERING:** una classe di ipotesi H FRANTUMA un sottoinsieme C di X se Hc contiene tutte le funzioni da C→{0,1}, la restrizione deve avere cardinalità 2C (|Hc|=2C).

Quindi, la **VC-DIMENSION** di una classe d'ipotesi H (VCdim(H)) è la taglia del più grande sottoinsieme di X che è Shattered da H. Ci dice su quanti elementi del dominio riusciamo a costruire tutte le funzioni.

## Teorema sulla Learnability

**Teorema:** Una classe di ipotesi è **LEARNABLE** se e solo se la sua **VC-DIMENSION** è **FINITA**.

**Spiegazione:**

La **VC-DIMENSION** (Vapnik-Chervonenkis Dimension) è una misura della complessità di una classe di ipotesi. Indica il numero massimo di punti che possono essere **SHATTERED** dalla classe.

**Shattering** significa che la classe di ipotesi può creare qualsiasi possibile classificazione dei punti dati.

**Calcolo della VC-DIMENSION:**

Per calcolare la VC-DIMENSION di una classe di ipotesi, si seguono due passi:

1. **Trovare il massimo numero di punti che possono essere shattered:**
 * Se esiste un sottoinsieme di punti C con |C|=d che può essere shattered dalla classe di ipotesi, allora la VC-DIMENSION è almeno d.
2. **Verificare che non esistano sottoinsiemi di punti più grandi che possono essere shattered:**
 * Se per ogni sottoinsieme di punti C con |C|=d+1 la classe di ipotesi non può creare tutte le possibili classificazioni, allora la VC-DIMENSION è esattamente d.

**Esempi:**

* Partiamo dalla classe di ipotesi delle funzioni soglia:
$$H=\{  h_{0}(x)=1[x>0], \ 0\in R \}$$
 ![[3)-20241005175148387.png|403]]
 Abbiamo trovato un sottoinsieme di cardinalità 2 per cui prediamo i punti non riusciamo a costruire tutte le possibili funzioni da questo sottoinsieme a {0,1}.
* Prendiamo le funzioni intervallo:
$$H=\{  h_{a,l}(x)\}=1[a\leq x\leq l,a,l \in R]$$
 ![[3)-20241005175437626.png|401]]
$$VCdim(M|n|)=2$$
* Prendiamo la classe dei rettangolo:
$$h_{a,b,c,d}(x)=1[a\leq x_{1}\leq b \cap c\leq\lambda_{2}\leq d], \ a,b,c,d \in R$$
 ![[3)-20241005175735494.png|507]]

## TEOREMA FONDAMENTALE DEL PAC LEARNABLE

Il teorema fondamentale del PAC Learnable afferma che una classe di ipotesi H è (Agnostic) PAC Learnable se e solo se la sua VC Dimension è finita. 

La VC Dimension (d) rappresenta la capacità di una classe di ipotesi di "frantumare" un insieme di dati. In altre parole, indica il numero massimo di punti che possono essere classificati in tutti i modi possibili dalla classe di ipotesi.

Il teorema stabilisce che per poter imparare in modo PAC, la classe di ipotesi deve avere una VC Dimension finita. Questo significa che la classe di ipotesi non può essere troppo complessa, altrimenti non sarà possibile generalizzare a nuovi dati.

La formula $m=0\ \left( \frac{d+\ln\left( \frac{1}{\delta} \right)}{Ɛ^c} \right)$ fornisce una stima del numero di esempi necessari per imparare in modo PAC. 

* **d:** VC Dimension della classe di ipotesi
* **Ɛ:** Errore massimo consentito
* **δ:** Probabilità massima di errore
* **c:** Costante che dipende dal tipo di apprendimento (1 per PAC, 2 per Agnostic PAC)

In sostanza, questo teorema fornisce una condizione necessaria e sufficiente per l'apprendimento PAC, e fornisce una formula per stimare il numero di esempi necessari per raggiungere un certo livello di accuratezza.
