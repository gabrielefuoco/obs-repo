
La minimizzazione dell'errore in machine learning si traduce in un problema di ottimizzazione. Questo tipo di problema è ben studiato e esistono algoritmi efficienti per risolverlo. L'obiettivo è trovare il minimo di una funzione convessa.
$$\mathrm{ERM}_{H}(S) = \arg \min_{\substack{ \vec{\omega} \in H}} L_s(\vec{\omega})$$
### L'Algoritmo della Discesa del Gradiente

Un algoritmo efficiente per la minimizzazione di funzioni convesse è la **discesa del gradiente**. Questo algoritmo è iterativo e si basa sull'utilizzo del gradiente della funzione per trovare il minimo.

**Funzionamento:**

1. **Punto di partenza:** Si inizia con una soluzione iniziale, indicata con `w`.
2. **Calcolo del gradiente:** Ad ogni iterazione, si calcola il gradiente della funzione `F(w)` nel punto corrente `w`. Il gradiente rappresenta la derivata della funzione in caso di funzione monodimensionale, mentre in caso di funzione multidimensionale è un vettore che indica la direzione di massima pendenza.
3. **Aggiornamento della soluzione:** Si aggiorna la soluzione `w` muovendosi nella direzione opposta al gradiente. Questo perché si vuole minimizzare la funzione, quindi ci si sposta nella direzione di discesa. L'aggiornamento è dato da:

   $w(t+1) = w(t) - η * ∇f(w(t))$

   dove:
   - `w(t)` è la soluzione corrente all'iterazione `t`.
   - `η` è il **learning rate**, che determina l'ampiezza dello spostamento ad ogni iterazione. Un valore di `η` troppo grande può portare a oscillazioni e difficoltà nel trovare il minimo, mentre un valore troppo piccolo può rallentare la convergenza.

4. **Iterazione:** Si ripetono i passaggi 2 e 3 fino a raggiungere un punto di minimo o un criterio di arresto.

$$\begin{aligned}
  \vec{w} = \vec{0} \\
\text{ for } t = 1 \text{ to } T \text{ do } \\
  \vec{v}_t = \nabla_{\vec{w}} f(\vec{w}^{(t)}) \\
\quad \vec{w}^{(t+1)} = \vec{w}^{(t)} - \eta \vec{v}_t \\
\end{aligned}$$
$$\begin{equation}
\displaystyle \begin{aligned}
& \operatorname{\mathrm{Return}} W^{(T+1)} \\
& \operatorname{In alternativa possiamo restituire:} \\
\\
& \qquad\left\{
\begin{aligned}

& \quad \quad\begin{aligned}
 & \operatorname{Return} \operatorname{\arg\min_{1\leq t\leq T+1} f(\vec{w}^ {(A)})}\\
 & \text{oppure, } \frac{1}{T+1}\sum_{t=1}^{T+1}\vec{w}^{(A)}
\end{aligned}
\end{aligned}
\right. \\
& \quad \begin{aligned}
& \operatorname{oppure} \frac{1}{(T+\hat{L})^2} \sum_{t + L}^{T} w > (A)
\end{aligned}
\end{aligned}
\qquad
\end{equation}$$


![[6)-20241021123828656.png]]

Sappiamo che se ci si sposta nel verso del gradiente la funzione cresce , quindi ci
si muove nella direzione opposta del gradiente.

La nostra funzione da minimizzare è l'errore empirico quindi:

$$\begin{equation*}
ERM_H{(S)} = \arg  \min_{\vec{w}\in H} L_{S}(\vec{w}) \equiv f \left( \vec{w} \right) \equiv f(\vec{w})
\end{equation*}
$$
Quindi possiamo personalizzare l'algoritmo come

$$
\begin{aligned}
\vec{w}^{(0)} = \vec{0} \\

\text{for } t = 1 \text{ a } T \text{ do} \\

\vec{v}_t = \nabla_{\vec{w}} L_{S}(\vec{w}^{(t)})\\

=\vec{\nabla}w\left[ \frac{1}{m}\sum_{i=1}^ml(\vec{w}^{(t)},(\vec{x_{i}},y_{i})) \right]\\

=\frac{1}{m} \sum_{i=1}^m \nabla _{\vec{w}}l(\vec{w}^{(t)},(\vec{x_{i}},y_{i})) \\

=\vec{w}^{(t+1)}=\vec{w}^{(t)}-\eta \vec{v}_{t}\\

\text{ritorna } \vec{w}^{(T)}
\end{aligned}
$$
si può usare questo approccio, con la necessità di calcolare questo gradiente

Nella regressione lineare, la funzione di costo è la "mean squared error":
$$\begin{aligned}
lsq(\vec{w},(\vec{x}, y)) &= \frac{1}{2} (\langle\vec{w}, \vec{x}\rangle-y)^2 
\end{aligned}
$$
$$\begin{aligned}
\nabla \vec{w} \ lsq \ (\vec{w},(\vec{x}, y)) &= \nabla \vec{w} \left[\frac{1}{2} (\langle \vec{w}, \vec{x}\rangle-y)^2\right] \\
&= \frac{1}{2} \cdot 2(\langle \vec{w}, \vec{x}\rangle-y)\vec{x} 
\end{aligned}$$

## GD+ Linear Regression


$$\begin{aligned}
& \vec{\omega}^{(0)} = \vec{0}\\
& \text{for} \; t = 1 \; \text{to}\; T \; \text{do} \\
& \quad \begin{aligned}
& \vec{V}_{t} = \nabla_{\vec{\omega}} L_{s}(\vec{\omega}, \vec{t} \ ) \\
& = \nabla_{\vec{\omega}} [ \frac{1}{m} \sum_{i=1}^{m} l(\vec{\omega}^{(t)}, (\vec{x_{i}}, y_{i}))]\\
& = \frac{1}{m} \sum_{i=1}^{m} \nabla_{\vec{w}} \ l(\vec{\omega}^{(t)}, (\vec{x_{i}}, y_{i}))\\

&  \vec{\omega}^{(t+1)} = \vec{w}^{(t)}-\eta  \vec{v}_{t} \\
& \text{return } \vec{w}^{(t+1)}
\end{aligned}
\end{aligned}$$

Nell'ambito del learning si preferisce usare una variante di questo algoritmo, poichè questo è un algoritmo possiede una funzione abbastanza pesante, rappresentata dalla sommatoria di tutte le loss:

$$\frac{1}{m} \sum_{i=1}^m l(\vec{w}^{(t)},(\vec{x_{i}},y_{i}))$$

Il problema principale della discesa del gradiente standard è che il calcolo del gradiente richiede di visitare tutti gli esempi di training, il che può essere molto lento per dataset di grandi dimensioni.

SGD risolve questo problema utilizzando un sottoinsieme casuale degli esempi di training per calcolare il gradiente ad ogni iterazione. Questo approccio riduce significativamente il tempo di calcolo, soprattutto per dataset di grandi dimensioni.
## Discesa stocastica del gradiente

Invece di utilizzare il gradiente calcolato su tutto il dataset come direzione di discesa, la SGD utilizza un vettore il cui **valore atteso** coincide con il valore atteso del gradiente. Questo vettore direzione è una variabile casuale che punta in media nella direzione del gradiente "vero".
$$E[\vec{V}_{t}]=\nabla_{\vec{w}}f(\vec{w}^{(t)})$$
Durante l'ottimizzazione, la SGD si muove in una direzione che, in ogni singola iterazione, potrebbe non essere quella di massima discesa. Questo significa che potremmo non muoverci esattamente nella direzione del gradiente, ma in modo casuale, e potremmo anche "salire" un pochino nella funzione di costo. Tuttavia, poiché il valore atteso del vettore direzione coincide con il gradiente, in media ci sposteremo verso il minimo.
### Vettore direzione

Il vettore direzione si ottiene calcolando il gradiente della funzione di costo in un **campione casuale** della popolazione. In pratica, questo significa selezionare casualmente una coppia (x,y) dal dataset e calcolare il gradiente in quel punto.

$$\vec{V}_{t}=\nabla l(\vec{w}^{(t)},(\vec{x},y)), \text{ con } (\vec{x},y)\text{ scelti casualmente da S}$$

Quindi si lavora su sottoinsieme e non intero dataset alle singole iterazioni 

### Dimostrazione 

La proprietà che il valore atteso del vettore direzione coincide con il gradiente può essere dimostrata matematicamente. Il valore atteso $E[\cdot]$ del gradiente della funzione di costo, immaginando che i valori siano campionati dalla distribuzione dei dati, può essere scritto come:

$E[\vec{V}_{t}]=E[\nabla_{\vec{w}}l(\vec{w},(\vec{x},y))]$

Visto che è una variabile casuale, allora il valore atteso sarà il valore atteso del gradiente della loss, assumendo che x e y siano scelti casualmente da S.

$E$ e $\vec{\nabla}w$ sono operatori invertibili, ovvero
$$=\nabla_\vec{w} \{E[l(\vec{w},(x,y))]  \}=\nabla_{\vec{w}}L_{D}(\vec{W})\equiv \nabla_{\vec{w}}L_{s}(\vec{w})$$
Quindi
$$E[\vec{V}_{t}]=\nabla_{\vec{w}}f(\vec{w}^{(t)})\equiv \nabla_{\vec{w}}L_{S}(\vec{w})$$

Dobbiamo fare una selezione casuale, dunque l'algoritmo diventa:
$$\begin{align*}
&\vec{w}^{(0)}  = \vec{0} \\
&\text{for t to T do} \\
&\text{Seleziona Casualmente }(\vec{x}_{i}, y_{i}\text{) in S} \\
&\vec{V}_{t}  = \nabla_{\vec{w}} l\left(\vec{w}^{(t)},(\vec{x}_{i}, y_{i})\right) \\
&\vec{w}^{(t+1)}  =\vec{w}^{(t)} - \eta \vec{V}_{t}\\
&\text{return } \vec{w}^{(t+1)}
\end{align*}$$
### Grafico
La funzione ora è convessa, dunque ho delle curve di livello.
![[6)-20241021161630266.png]]

Un grafico qualitativo della SGD mostra che la traiettoria non è dritta verso il minimo, ma va a zig-zag. La discesa del gradiente standard, invece, ha una traiettoria dritta verso il minimo.
Chiaramente si puo rallentare la convergenza poichè si fanno piu passi.


Nella pratica, non si usa direttamente la versione della SGD basata su un solo esempio, perché è abbastanza instabile. Si utilizza invece un algoritmo intermedio chiamato **mini-batch SGD**.

$$
\begin{aligned}
&\vec{w}^0 = \overrightarrow{0} \\
&\text{for } t \text{ to } T \text{ do} \\
&\text{SEZIONA CASUALMENTE UN BATCH DI } b \text{ ESEMPI } \\
&\vec{V}_t = \frac{1}{b} \sum_{i=1}^b \nabla_{\vec{w}} (l(\vec{w}(x_i,\vec{y}))\\
&\vec{w}^{(t+1)} = \vec{w}^{(t)} - \eta  \vec{V}_{t} \\
&\text{return } \vec{w}^{(t+1)}
\end{aligned}
$$



### Learning Rate

Il **learning rate** è un parametro fondamentale negli algoritmi di apprendimento automatico, in particolare nella discesa del gradiente. Esso determina la dimensione del passo che l'algoritmo compie ad ogni iterazione nella direzione del minimo della funzione di costo.
$$\eta=
\begin{cases}
\text{costante } \eta=\eta_{0} \\
\text{variabile } \eta^{(t)}=\frac{\eta_{0}}{\sqrt{ t }} 
\end{cases}
$$
Un learning rate costante presenta alcuni svantaggi:

* **Convergenza:** Un learning rate troppo alto può impedire all'algoritmo di convergere al minimo globale, facendolo oscillare attorno ad esso.
* **Fase iniziale:** All'inizio dell'addestramento, quando l'algoritmo è lontano dalla soluzione ottimale, un learning rate elevato può essere vantaggioso per muoversi rapidamente nello spazio delle soluzioni.
* **Fase finale:** Man mano che l'algoritmo si avvicina al minimo, un learning rate elevato può causare instabilità e impedire la convergenza precisa.

![[6)-20241021162009213.png]]

Per ovviare a questi problemi, si preferisce un **learning rate variabile**, che si adatta alle diverse fasi dell'addestramento. Un metodo comune per la variazione del learning rate è l'"**annealing**", dove il learning rate viene diminuito gradualmente nel tempo.

Esistono diverse politiche di aggiornamento del learning rate, come le **epoche di addestramento**, dove il learning rate viene diminuito dopo un certo numero di iterazioni. In generale, il learning rate iniziale viene impostato come parametro e poi aggiornato nel tempo secondo una specifica politica.
$$\eta^{(t)}=\frac{\eta_{0}}{\sqrt{ t }}$$

## Funzione di loss surrogata
![[6)-20241021162107675.png]]
La **loss surrogata** è una funzione di costo che viene utilizzata al posto della funzione di costo originale, quando quest'ultima è computazionalmente complessa o non convessa.

Ad esempio, nel caso dei semispazi e della **loss 0-1**, la funzione di costo originale è non convessa, il che rende difficile l'utilizzo della discesa del gradiente. In questo caso, si può utilizzare una loss surrogata, come la **loss logistica**, che ha le seguenti proprietà:

1. **Upper bound**: la loss surrogata è un limite superiore della loss originale, ovvero il suo valore è sempre maggiore o uguale al valore della loss originale per ogni punto.
2. **Convessità**: la loss surrogata è una funzione convessa, permettendo l'utilizzo della discesa del gradiente e garantendo la convergenza al minimo globale.

### Esempio: Hinge Loss

Un esempio di loss surrogata per la loss 0-1 è la **hinge loss**. 

![[6)-20241021162126348.png]]
La hinge loss vale 0 per punti classificati correttamente con un margine maggiore di 1, e cresce linearmente per punti classificati errati o con un margine inferiore a 1.

La hinge loss è definita come:
$$l_{hinge}(\vec{w},(\vec{x},y))=\max \{ 0,1-y_{i}<\vec{w},\vec{x_{i}}> \}$$

**Proprietà della Hinge Loss:**

* **Upper bound**: la hinge loss è un upper bound della loss 0-1, poiché il suo valore è sempre maggiore o uguale al valore della loss 0-1 per ogni punto.
* **Convessità**: la hinge loss è una funzione convessa, quindi può essere utilizzata con la discesa del gradiente.

#### Vantaggi dell'utilizzo di una Loss Surrogata
* **Ottimizzazione più semplice**: la loss surrogata permette di risolvere un problema di ottimizzazione più semplice e trattabile rispetto al problema originale.
* **Efficienza**: la convessità della loss surrogata consente l'utilizzo di algoritmi di ottimizzazione efficienti come la discesa del gradiente.

#### Svantaggi dell'utilizzo di una Loss Surrogata
* **Perdita di accuratezza**: la soluzione trovata con la loss surrogata non sarà ottimale rispetto alla loss originale, ma sarà comunque una buona approssimazione.

#### Applicazione ai Semispazi
Nel caso dei semispazi, la loss 0-1 porta ad un problema di ottimizzazione NP-hard, quindi intrattabile in pratica. Utilizzando la hinge loss come surrogata, possiamo invece risolvere un problema di ottimizzazione convesso, quindi trattabile in modo efficiente.


## Funzioni di Lipschitz

![[6)-20241022123528755.png|401]]

Una funzione è detta di Lipschitz se la norma della differenza tra le immagini della funzione calcolata in due punti è limitata da una costante (detta costante di Lipschitz) moltiplicata per la distanza tra i due punti. In altre parole, la variazione della funzione è controllata dalla distanza tra i punti nel suo dominio.

**Formalmente:**
Una funzione $f:R^d \to R^k$ si dice $\rho\text{-lipschitz}$ se $\forall w_{1},w_{2} \in R^d,$  $\| f(\vec{w_{1}})-f(\vec{w_{2}}) \| \leq \rho \| \vec{w_{1}}-\vec{w_{2}} \|$


La definizione di funzione $\rho\text{-lipschitz}$ implica che il gradiente è limitato superiormente. 

L'introduzione di questo concetto è fondamentale per caratterizzare i problemi di apprendimento convessi

## Problema di Learning Convex-Lipschitz-Bounded
Un problema di learning si dice convex-lipschitz-bounded se
1) La classe d'ipotesi H è un insieme convesso e $\vec{\forall}w\in H, \| \vec{w}\|\leq B$ (bound) 
2) La funzione di Loss è convessa e $\rho\text{-lipschitz}$

## RLM-REGULARIZED LOSS MINIMIZATIO

$$
\vec{w}= RLM_{H}(S)=\arg\min_{\vec{w \in H}} L_{S}(\vec{w})+R(\vec{w})
$$
Dove $R(\vec{w})$ è il temine di Regolarizzazione, dipende da quanto è complicato il modello per il secondo punto

$$R(\vec{w})
\begin{cases}
\text{Agisce come Stabilizzatore, la stabilità previene Overfitting} \\ \\

\text{Funge da Misura della stabilità del modello}
\end{cases}
$$
**Definizione di algoritmo stabile**
Un algoritmo di learning si dice stabile quando a piccole variazioni dell'input corrispondono piccole variazioni dell'output

Per dimostrare questa proprietà, è necessario introdurre una variante dell'algoritmo di minimizzazione del rischio empirico chiamata **Regolarizzazione di Tikhonov** (RML).
## Regolarizzazione di Tikonov

La Regolarizzazione di Tikhonov è una regola di apprendimento che, data una classe di ipotesi e una funzione di perdita, seleziona l'ipotesi che minimizza il rischio empirico più un termine aggiuntivo detto **termine di regolarizzazione**.

Questo termine ha due funzioni principali:

1. **Stabilizzatore**: Se il dataset di training varia leggermente, ci aspettiamo che l'algoritmo restituisca un modello simile. Un algoritmo stabile è meno sensibile alle piccole variazioni nei dati.
2. **Misura della complessità del modello**: Il termine di regolarizzazione può essere interpretato come una misura della complessità del modello. Modelli più complessi tendono ad avere un termine di regolarizzazione più elevato.

$$R(\vec{w})=\lambda \| \vec{w}\|^2$$

#### Proprietà
Un problema di learning Convex-Lipschitz-Bounded è learnable tramite RLM (Regularized Loss Minimization)

Essendo learnable, la sample complexity è data da $m=\frac{8\rho^2B^2}{Ɛ^2}$


Avendo stabilito che abbiamo la learnability minimizzando $\arg\min_{\vec{w \in H}} L_{S}(\vec{w})+R(\vec{w})$, resta da modificare l'algoritmo di discesa del gradiente per includere il termine di regolarizzazione R

$$
\vec{w}^*=\arg\min_{\vec{w \in R^d}} \ L_{S}(\vec{w})+ \lambda \|\vec{w} \|^2
$$

$L_{S}$ la rende convessa e $L_{S}(\vec{w})+ \lambda \|\vec{w} \|^2$ la rende fortemente convessa

![[6)-20241022125918727.png]]
- La parabola in rosso indica che è fortemente convessa

## Gradient Descent con Regolarizzazione

La disuguaglianza di convessità per una funzione convessa $f$ afferma che:

$$f(\alpha  \vec{w_{1}}+(1-\alpha) \vec{w_{2}})\leq\alpha f(w_{1})+\left( (-\alpha)f(w_{2})-\frac{1}{2}\alpha(1-\alpha)\|\vec{w_{1}}-\vec{w_{2}}\|^2 \right)$$

La direzione per aumentare la funzione è data da:

$$
\vec{V_{t}}=\nabla_{\vec{w}}\left[ l(\vec{w}^{(t)},(\vec{x_{i}},y_{i}))    +\frac{1}{2}\|\vec{w}^{(t)}\|^2 \right]=\nabla_{\vec{w}}l(\vec{w}^{(t)},(\vec{x_{i}},y_{i}))+ \lambda \vec{w}^{(t)}
$$

### Batch SGD e Epoche

* **Iterazione:** Un'iterazione è un singolo passo dell'algoritmo di discesa del gradiente.
* **Epoca:** Un'epoca è un ciclo completo sul set di training.

Il numero di iterazioni necessarie per completare un'epoca dipende dalla dimensione del batch. Ad esempio, se il batch è un millesimo del dataset, un'epoca corrisponde a 1000 iterazioni.

### Versione con Regolarizzazione

La versione dell'algoritmo che include il termine di regolarizzazione è:

$$
\begin{aligned}
&\vec{w}^{(0)} = \vec{0} \\
&\text{for } t \geq t_{0} \text{ to } T \text{ do}: \\
&\text{SELEZIONA CASUALMENTE } (\vec{x_i},Y_i) \text{ in } S \\
&\vec{v}_t = \nabla_{\vec{w}} l(\vec{w}, (\vec{x}_i,y_i)) \\
&\vec{w}^{(t+1)}  = \vec{w}-\eta(\vec{v}_t + \lambda \vec{w}^{(t)}) \\
&\text{return }  \ \vec{w}^{\text{(t+1)}}
\end{aligned}
$$

Si parla di **ridge regression** quando si ha la regressione con tutti i requisiti per essere "learnable". La funzione di costo in questo caso è:

$$
\frac{1}{m} \sum_{i=1}^m (<\vec{w},\vec{x_{i}}>-y_{i})^2+\lambda \| \vec{w}\|^2
$$

### Learning Rate Variabile per Funzioni Strettamente Convesse

Nel caso di funzioni strettamente convesse, il learning rate $\eta$ è definito come:

$\eta=\frac{1}{\lambda t}$

L'aggiornamento può essere riscritto come:

$$\vec{w}^{(t+1)}  = \vec{w}-\frac{1}{\lambda t}(\vec{v}_t + \lambda \vec{w}^{(t)})$$

$$\vec{w}^{(t+1)}  = \left( 1-\frac{1}{t} \right)\vec{w}^{(t)}-\frac{1}{\lambda t}\vec{v_{t}}$$

Usando ricorsivamente questa relazione, si ottiene:

$$
\left( 1-\frac{1}{t} \right)\vec{w}^{(t)}-\frac{1 \ \vec{v_{t}}}{\lambda t}=\frac{t-1}{t}\vec{w}^{(t)}-\frac{1 \ \vec{v_{t}}}{\lambda t}=
$$

$$
=-\frac{1}{\lambda t}\sum_{h=1}^t \  \vec{V_{j}}\ \theta^{(t)}
$$

## SGD + RLM

$$
\begin{aligned}

\vec{w}^{(0)} &= \vec{0} \\

\text{for}~ t &= 1~ \text{to}~ T(t)~ \text{do} \\

\vec{w}^{(a)} &= \frac{1}{\lambda_t} \cdot \vec{\theta}^{(t)} \\

&\text{seleziona casualmente }  (\vec{x_i}, y_i)~ \text{in}~ S \\

\vec{v}_t &= \nabla_{\vec{w}} \ell\left( \vec{w}^{(t)}, (\vec{x_i}, y_i) \right) \\

\vec{\theta}^{(t+1)} &= \vec{\theta}^{(t)} - \vec{v}_t \\

\text{return}~ -\frac{1}{\lambda_t} \cdot \vec{\theta}^{(t+1)}

\end{aligned}$$

