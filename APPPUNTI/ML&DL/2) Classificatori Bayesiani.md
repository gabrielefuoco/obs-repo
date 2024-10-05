### Assunzioni sulla Funzione di Etichettatura
Assumiamo che esista una funzione di etichettatura $f(x)$ che ci fornisce l'etichetta vera per un'istanza $x$.

**Problemi di Rappresentazione:**
* Se la rappresentazione delle istanze è fissata, potremmo non aver scelto le features più appropriate per la discriminazione.
* Se abbiamo difficoltà a discriminare tra le istanze, potremmo dover aumentare il numero di features.

**Assunzione Fondamentale:**
* Due istanze uguali hanno la stessa etichetta.

## Funzione Generatrice e Classificatore Bayesiano

* **Funzione Generatrice:**
    * Non ci sono ulteriori informazioni sulla funzione di etichettatura.
    * Consideriamo la funzione generatrice $D : X \cup Y$, dove:
        * $X$ è l'insieme delle istanze.
        * $Y$ è l'insieme delle etichette.
    * Questa funzione diventa $D(x,y) =$ probabilità di osservare una certa istanza $x$ con etichetta $y$, dato un oggetto $\bar{X} \rightarrow \{D(\bar{x}, 0), D(\bar{x}, 1)\}$.

* **Predittore Ottimo: Classificatore Bayesiano**
    * $f_D(X) =$ restituisce la classe più probabile.
    * $f_D(x) = arg \max_{y \in Y} P(y|x)$ 

* **Esempio:**
    * Supponiamo di avere un oggetto $x$ con $\bar{X} \rightarrow \{D(\bar{x}, 0) = 0.9, D(\bar{x}, 1) = 0.1\}$.
    * Il classificatore Bayesiano restituisce la classe più probabile.
    * Ogni altra regola che possiamo scegliere ci darà un errore più alto.

* **Probabilità Congiunta:**
    * $p(x,y) = p(x \cap y) = p(x|y)P(y) = p(y|x)p(x)$
    * $P(y|x) = \frac{p(x|y) P(y)}{p(x)}$
    * $f_D(x) = arg \max_{y \in Y} \frac{p(x|y) P(y)}{p(x)}$
        * Nel caso di equiprobabilità, ovvero $P(y) = \frac{1}{|Y|}$ per ogni $y \in Y$, la formula si semplifica in:
            * $f_D(x) = arg \max_{y \in Y} {p(x|y)}$
            * $f_D(x) = arg \max_{y \in Y} {p_y(x)}$ dove $p_y(x)$ rappresenta la probabilità di osservare $x$ dato che l'etichetta è $y$.



---
# inserire esempio 1
L'errore del classificatore, come illustrato nel disegno, si verifica quando un'istanza con un valore di $x$ inferiore a $\theta$ appartiene alla classe 1, ma il classificatore la classifica erroneamente come classe 0. Allo stesso modo, per la classe 1, l'errore si verifica quando un'istanza con un valore di $x$ maggiore di $\theta$ appartiene alla classe 0, ma il classificatore la classifica erroneamente come classe 1.

**Esempio:**

Immaginiamo di avere un classificatore che separa le istanze in due classi (0 e 1) basandosi su un valore di soglia $\theta$. 

* Se $\theta = 5$, il classificatore classifica tutte le istanze con $x < 5$ come classe 0 e tutte le istanze con $x > 5$ come classe 1.

* Ora, supponiamo che un'istanza con $x = 3$ appartenga alla classe 1. Il classificatore, a causa della soglia impostata a 5, la classificherà erroneamente come classe 0. Questo è un esempio di errore del classificatore.

* Allo stesso modo, se un'istanza con $x = 8$ appartiene alla classe 0, il classificatore la classificherà erroneamente come classe 1.

In sostanza, l'errore del classificatore si verifica quando la soglia scelta non riesce a separare correttamente le istanze in base alla loro classe reale. Questo può accadere quando la distribuzione dei dati non è perfettamente separabile o quando la soglia è stata scelta in modo non ottimale.

---
### Come la Distribuzione delle Classi e le Probabilità a Priori Influenzano la Classificazione con Soglia

La probabilità di osservare un'istanza $x$ e l'errore di un classificatore con soglia $\theta$ dipendono dalla distribuzione delle classi e dalla probabilità a priori di ciascuna classe.
#### 1. Probabilità congiunta e probabilità condizionata
$$p(x) = \sum_{y \in Y} P(x \cap y)=\sum_{y \in Y} P(x|y)P(Y)=p(x|y=0)p(y=0)+p(x|y=1)p(y=1)$$
 La seguente esprime la probabilità di osservare un'istanza $x$ come somma delle probabilità di osservare $x$ in ciascuna classe, ponderata per la probabilità a priori di ciascuna classe.

* **$p(x)$:** Probabilità di osservare l'istanza $x$.
* **$P(x \cap y)$:** Probabilità congiunta di osservare l'istanza $x$ e l'etichetta $y$.
* **$P(x|y)$:** Probabilità condizionata di osservare l'istanza $x$ dato che l'etichetta è $y$.
* **$P(y)$:** Probabilità a priori dell'etichetta $y$.

La formula sfrutta il teorema di Bayes per scomporre la probabilità congiunta in probabilità condizionata e probabilità a priori:

$P(x \cap y) = P(x|y)P(y) = P(y|x)P(x)$

Nel caso di classificazione binaria, dove $Y = \{0, 1\}$, la formula si semplifica in:

$p(x) = p(x|y=0)p(y=0) + p(x|y=1)p(y=1)$
---
#### 2. Funzione di loss per un classificatore con soglia
$$L_D(f_D) = \int_{x < \theta} p(x|y=1)p(y=1) dx+ \int_{x < \theta} p(x|y=0)p(y=0) dx \ge0$$

La formula rappresenta la funzione di loss per un classificatore con soglia $\theta$.

* **$L_D(f_D)$:** Errore di generalizzazione del classificatore Bayesiano $f_D$.
* **$\theta$**: Soglia utilizzata dal classificatore per separare le classi.
* **$p(x|y=1)$:** Densità di probabilità dell'istanza $x$ dato che appartiene alla classe 1.
* **$p(x|y=0)$:** Densità di probabilità dell'istanza $x$ dato che appartiene alla classe 0.
* **$p(y=1)$:** Probabilità a priori della classe 1.
* **$p(y=0)$:** Probabilità a priori della classe 0.

La funzione di loss misura l'errore del classificatore, ovvero la probabilità di classificare erroneamente un'istanza. L'integrale calcola la probabilità di classificare erroneamente un'istanza con $x < \theta$ come classe 0, e la probabilità di classificare erroneamente un'istanza con $x < \theta$ come classe 1.

**Interpretazione:**

La funzione di loss è sempre maggiore o uguale a zero, e più le classi sono sovrapposte, più è probabile che il classificatore commetta errori. Questo perché la sovrapposizione delle classi implica che ci sono istanze con valori di $x$ simili che appartengono a classi diverse, rendendo difficile la separazione tramite una soglia.

### 3. Sovrapposizione delle classi

Più le classi sono sovrapposte e più il classificatore è destinato a sbagliare, poichè le due densità di classe sono sovrapposte. 
- Questo significa che se le distribuzioni di probabilità delle due classi sono molto simili, il classificatore avrà maggiori difficoltà a separarle correttamente.

**Esempio:**

Immaginiamo di avere due classi, una con una distribuzione normale centrata in 0 e l'altra con una distribuzione normale centrata in 5. Se la varianza delle due distribuzioni è molto bassa, le classi saranno ben separate e il classificatore avrà un basso errore. Se la varianza è alta, le classi saranno sovrapposte e il classificatore avrà un errore maggiore.

---
## Approcci al Machine Learning
Nell'ambito del machine learning, possiamo distinguere due grandi famiglie di approcci:

### Metodi Basati su Stima di Densità
* Questi metodi stimano la densità di probabilità per ogni classe, ovvero la probabilità di osservare un'istanza $x$ data una certa classe.
* Una volta stimata la densità di probabilità, si può utilizzare il classificatore Bayesiano per predire la classe più probabile per una nuova istanza.
* Questi approcci sono generalmente più complessi di quelli discriminativi.
* Si dividono in:
	* **Parametrica:** L'analista fa un'assunzione sulla forma della funzione di densità di probabilità (ad esempio, una distribuzione normale). Si stimano quindi i parametri della funzione (ad esempio, media e varianza).
	* **Non Parametrica:** Non si fa alcuna assunzione sulla densità di probabilità. La densità viene ricostruita dai dati, tipicamente come somma di piccoli contributi, ognuno derivante da un esempio.

### Metodi Discriminativi (Distribution-free)
* Questi metodi si concentrano direttamente sulla discriminazione tra le classi, cercando di apprendere una funzione di decisione che separi le diverse classi.
* Non restituiscono una distribuzione di probabilità.
* Sono spesso più semplici da implementare rispetto ai metodi basati su stima di densità.
* Non si interessano alla stima della densità di probabilità.
* Trovano una regola che partiziona il dominio di interesse nelle classi da predire.
* L'obiettivo è minimizzare l'errore di generalizzazione.
##### Approccio con Funzione Soglia
* Un approccio comune è l'utilizzo di una funzione soglia: $h_{\theta}(x) = 1[x \geq \theta], 0 \text{ altrimenti}$.
* Esiste una funzione soglia che può essere accurata quanto il classificatore Bayesiano.
* Tuttavia, non conosciamo a priori la forma delle classi e quindi non possiamo determinare la soglia ottimale.
---
## Funzione di LOSS

La funzione di loss è uno strumento fondamentale nell'apprendimento automatico per quantificare l'errore di un modello predittivo. 

**Definizione:**

$$L_s(h)=\frac{|x_i|h(x_i)|\neq y_i,1\leq i \leq m}{m}$$

* **$L_s(h)$:** Errore empirico del modello $h$ sul training set $S$.
* **$h(x_i)$:** Predizione del modello $h$ per l'istanza $x_i$.
* **$y_i$**: Etichetta vera dell'istanza $x_i$.
* **$m$**: Dimensione del training set.

**Esempio:**

Consideriamo un problema di regressione con un'istanza $(x_1, y_1 = 10.4)$. Abbiamo due ipotesi:

* $h_1(x_1) = 10.1$
* $h_2(x_1) = 10.3$

Per generalizzare l'entità di un errore, introduciamo la **funzione di perdita** (loss function):

$l(h,(x_i,y_i))$ = misura dell'errore in caso di predizione sbagliata (vale 0 se la predizione è corretta).

Con questo concetto, possiamo generalizzare la definizione di errore empirico:

$$L_S(h)= \frac{1}{m} \sum_{i=1}^m l(h,(x_i,y_i))$$ 

**Tipi di Funzioni di Loss:**

* **0-1 LOSS:**
    $$l_{0-1}(h,(x_i,y_i)) = \begin{cases}
1 & \text{se } h(x) \neq y \\
0 & \text{altrimenti}
\end{cases}$$
    Questa funzione assegna un errore di 1 se la predizione è sbagliata e 0 se è corretta.

* **SQUARED LOSS:**
    $$l_{sq}(h,(x_i,y_i)) = (h(x)-y)^2$$
    Questa funzione calcola la differenza al quadrato tra la predizione e l'etichetta vera. L'errore empirico con questa funzione di loss diventa l'errore quadratico medio.

**Parametrizzazione:**

L'errore empirico è **parametrico** perché dipende dalla funzione di loss scelta.

**Errore di Generalizzazione:**

L'errore di generalizzazione è anch'esso parametrico, perché dipende dalla funzione di loss. È il valore atteso della loss:

$$L_D(h) = E_{x,y \sim D}[l(h,(x_i,y_i))]$$ 


---
### Apprendimento PAC Standard

L'apprendimento PAC (Probably Approximately Correct) è un framework per l'analisi degli algoritmi di apprendimento. L'obiettivo è garantire che un algoritmo di apprendimento possa trovare un'ipotesi che sia "abbastanza buona" con "alta probabilità".
L'apprendimento **PAC standard** assume che esista un'ipotesi perfetta nella classe di ipotesi. In questo caso, l'obiettivo è trovare un'ipotesi che sia "vicina" all'ipotesi perfetta. La probabilità di successo è misurata tramite la probabilità che l'errore dell'ipotesi trovata sia inferiore a un valore di soglia $\epsilon$.

### Apprendimento PAC Agnostico

L'apprendimento PAC agnostico rilassa l'assunzione di realizzabilità, ovvero non assume che esista un'ipotesi perfetta nella classe di ipotesi. In questo caso, l'obiettivo è trovare un'ipotesi che sia "vicina" alla migliore ipotesi possibile nella classe di ipotesi.

**Definizione:**

Un algoritmo di apprendimento è **agnostico PAC-learnable** se, per ogni distribuzione di dati $D$ e per ogni $\epsilon, \delta > 0$, esiste un numero di esempi $m$ tale che, per ogni insieme di training $S$ di dimensione $m$ campionato da $D$, l'algoritmo di apprendimento restituisce un'ipotesi $h_S$ che soddisfa la seguente condizione:

$$L_D(h_S) \leq min_{h \in H} L_D(h) + \epsilon$$

con probabilità almeno $1-\delta$.

**Teoremi:**

* **Teorema 1:** Dato un training set $S$ di taglia  $m=|S|$, se $H$ è finita e $S$ è sufficientemente grande, allora $h_S = ERM_H(S)$ è **agnostic PAC-learnable**.
* **Teorema 2:** Se $H$ è finita, allora è **agnostic PAC-learnable**.

**Insiemi di Training Rappresentativi:**

Un insieme di training $S$ è $\epsilon$-rappresentativo se vale la seguente condizione:

$$\forall h \in H, |L_D(h) - L_S(h)| \leq \epsilon$$

Se un insieme di training è $\frac{\epsilon}{2}-$rappresentativo, allora l'ipotesi $h_S$ trovata dall'algoritmo di apprendimento soddisfa la condizione di agnostic PAC-learnability:

$$L_D(h_S) \leq min_{h \in H} L_D(h) + \epsilon$$

**Punti Critici:**

* L'apprendimento PAC agnostico è più realistico dell'apprendimento PAC standard, poiché non assume l'esistenza di un'ipotesi perfetta.
* L'apprendimento PAC agnostico è più difficile dell'apprendimento PAC standard, poiché l'algoritmo di apprendimento deve trovare un'ipotesi che sia "vicina" alla migliore ipotesi possibile, piuttosto che all'ipotesi perfetta.
* La complessità computazionale dell'apprendimento PAC agnostico può essere elevata, soprattutto per classi di ipotesi di grandi dimensioni.

---
##  Convergenza Uniforme e Sample Complexity

**Obiettivo:** Determinare le condizioni per cui la probabilità che un insieme di training $S$ sia $\epsilon$-rappresentativo sia maggiore o uguale a $1-\delta$.

**Notazione:** Utilizziamo $\epsilon$ invece di $\frac{\epsilon}{2}$.

**Definizione:** Un insieme di training $S$ è $\epsilon$-rappresentativo se per ogni ipotesi $h$ nell'insieme di ipotesi $H$, la differenza tra l'errore di generalizzazione $L_D(h)$ e l'errore empirico $L_S(h)$ è minore o uguale a $\epsilon$.

**Formulazione del problema:**

Dobbiamo dimostrare che:

$Pr[S: \forall h \in H, |L_{D}(h)-L_{S}(h)|\leq\epsilon]\geq 1-\delta$

**Passaggi:**

1. **Evento complementare:**

   $$Pr[\{S: \forall h \in H,|L_{D}(h)-L_{S}(h)|>\epsilon\}]\leq 1-\delta$$

2. **Unione di eventi:**

   $$\{  S: ∃ h \in H,|L_{D}(h)-L_{S}(h)|>\epsilon \}= \cup _{h \in H}\{   S:|L_{D}(h)-L_{S}(h)|>\epsilon  \}$$

3. **Disuguaglianza di Boole:**

   $Pr[\{S: ∃ h \in H,|L_{D}(h)-L_{S}(h)|>\epsilon \}]=Pr[\{ \cup _{h \in H}\{   S:|L_{D}(h)-L_{S}(h)|>\epsilon  \}]\leq \sum_{h \in H}Pr[S:|L_{D}(h)-L_{S}(h)|>\epsilon]$

4. **Definizione di errore empirico e di generalizzazione:**

   - $L_S(h)$: errore empirico, calcolato come $\frac{1}{m} \sum l(h,(x,y))$, dove $l(h,(x,y))$ è la funzione di loss.
   - $L_D(h)$: errore di generalizzazione, calcolato come $E[l(h,(x,y))]$, dove $E$ è il valore atteso.
   - Possiamo considerare la funzione di loss come una variabile casuale $w_i$. 


5. **Applicazione della disuguaglianza di Hoeffding:**

   - Sia $w_1, w_2, \cdots, w_m$ una sequenza di variabili casuali i.i.d. con media $μ=E[w_i]$ e tali che $a\leq w_{i}\leq b$.
   - La disuguaglianza di Hoeffding afferma che:
    $$Pr\left[ |\frac{1}{m} \sum_{i=1}^m w_{i} - \mu | >\epsilon\right] \leq 2\exp [- \frac{2m\epsilon^2}{(b-a)^2}]$$

   - Nel nostro caso, la funzione di loss $l(h,(x,y))$ è limitata tra 0 e 1, quindi $a=0$ e $b=1$.
   - Applicando la disuguaglianza di Hoeffding, otteniamo:
     $$Pr[\{ S: |L_{D}(h)-L_{S}(h)|>\epsilon \}] \leq 2 \exp(-2m\epsilon^2)$$

6. **Sommatoria su tutte le ipotesi:**

   $$\sum_{h \in H}Pr[\{ S: |L_{D}(h)-L_{S}(h)|>\epsilon \}] \leq \sum_{h \in H}2 \exp(-2m\epsilon^2)=2|H|\exp(-2m\epsilon^2)\leq\delta$$

7. **Determinazione della sample complexity:**

   - Per garantire che la probabilità di convergenza uniforme sia maggiore o uguale a $1-\delta$, dobbiamo trovare la dimensione minima dell'insieme di training $m$.
   - Risolvendo la disuguaglianza precedente per $m$, otteniamo:
     $m \geq \frac{1}{2\epsilon^2} \ln\left( \frac{2|H|}{\delta} \right)$

   - Questa formula fornisce la sample complexity nel caso PAC agnostico.

8. **Sostituzione di $\epsilon$ con $\frac{\epsilon}{2}$:**

   - Poiché abbiamo utilizzato $\epsilon$ invece di $\frac{\epsilon}{2}$, dobbiamo sostituire nella formula precedente:
     $m\leq\frac{{2}}{\epsilon^2}\ln\left( \frac{2|H|}{\delta} \right)$

**Conclusione:**

La sample complexity nel caso PAC agnostico è data da:

$m\leq\frac{{2}}{\epsilon^2}\ln\left( \frac{2|H|}{\delta} \right)$

Questa formula ci permette di calcolare la dimensione minima dell'insieme di training necessaria per garantire una certa probabilità di convergenza uniforme, a condizione che l'insieme di ipotesi $H$ sia finito.

**Nota:**

- La formula della sample complexity è valida anche per insiemi di ipotesi infiniti, a condizione che $|H|\leq 2^{bd}$, dove $b$ è il numero di bit necessari per rappresentare un'ipotesi e $d$ è la dimensione dello spazio di input.

---

## Predittori Lineari

I predittori lineari sono una famiglia di classi di ipotesi utilizzate in pratica, che presentano diverse proprietà desiderabili:

1. **Intuitività e Interpretabilità:** Le ipotesi della classe sono intuitive e facili da interpretare.
    - Un modello trasparente e interpretabile, che fornisce spiegazioni sul perché restituisce una certa etichetta, aumenta la fiducia nelle sue predizioni.
    - Fino ad ora, non abbiamo mai visto il perché ci viene restituito un risultato (explainability).
2. **Efficienza Algoritmica:** Esistono algoritmi efficienti per queste classi di ipotesi, in particolare quelli di *learning* che minimizzano il rischio empirico.
3. **Performance in Casi Reali:** I predittori lineari si comportano bene, nonostante la semplicità, in molti casi reali.

### Funzioni Lineari Affini

I predittori lineari si basano sulle funzioni lineari affini:

$h_{\vec{w},b}(\vec{x})=\sum_{i=1}^d w_{i}x_{i}+b= <\vec{w},\vec{x}>+b$ 
dove:

- $\vec{x} \in R^d$ è il vettore di input.
- $\vec{w}$ è il vettore dei pesi.
- $b$ è il *bias*.

#### Funzioni Omogenee

Nel caso delle funzioni omogenee, il *bias* non è presente. Si effettua un cambio di rappresentazione e $b$ viene incorporata nei parametri:

- $\vec{w}'= <w_{1},\dots,w_{d}>$
- $\vec{x}'= <x_{1},\dots,x_{d}, 1> \in R^{d+1}$

Otteniamo quindi:

$h_{\vec{w'},b}(\vec{x'})= <\vec{w'},\vec{x'}>$

Ogni classe di ipotesi della famiglia si ottiene applicando a una funzione lineare omogenea una funzione da $R$ in $R$:

$\phi :R \to R$
$h_{\vec{w}}(\vec{x})=\phi(<\vec{w},\vec{x}>)$

### Classe di Ipotesi dei Semispazi

La classe di ipotesi dei semispazi utilizza come $\phi$ la funzione segno:

$\phi=sign()$
$$\phi(x)= \begin{cases}
-1 & \text{se } x < 0 \\
0 & \text{se } x = 0 \\
1 & \text{se } x > 0
\end{cases}$$



# inserire foto

## Classe di Ipotesi dei Semispazi: Rappresentazione e Proprietà

La funzione $h_{\vec{w}}(\vec{x})=sign(<\vec{w},\vec{x}>)$ rappresenta un'ipotesi nella classe dei semispazi.

**Interpretazione geometrica:**

- **h:**  L'ipotesi h è un iperpiano nello spazio $R^d$.
- **w:** Il vettore $\vec{w}$ è il vettore normale all'iperpiano.
- **b:** L'iperpiano dista $b$ dall'origine.
- **Equazione dell'iperpiano:** L'iperpiano è il luogo dei punti in cui $<\vec{w},\vec{x}> = 0$.
- **Semispazi:** L'iperpiano divide lo spazio in due semispazi:
    - **Positivo:** Il segno del prodotto scalare è positivo: $sign(<\vec{w},\vec{x}>) > 0$.
    - **Negativo:** Il segno del prodotto scalare è negativo: $sign(<\vec{w},\vec{x}>) < 0$.

**Obiettivo:** Trovare l'iperpiano che separa le due classi di punti nello spazio (es. punti positivi e negativi).

**Minimizzazione del Rischio Empirico:**

Cerchiamo il vettore $\vec{w}^*$ che minimizza il rischio empirico $L_{s}(h_{\vec{w}})$:

$\vec{w}^*= arg \ min _{\vec{w}\in R^{d+1}} \ L_{s}(h_{\vec{w}})$

**Proprietà:**

- **Intrattabilità:** Trovare il semispazio che minimizza il rischio empirico in accordo alla 0-1 loss è un problema intrattabile (nel caso generale).

**Casi Distinti:**

- **Training Set Linearmente Separabile:** Esiste un iperpiano con errore empirico nullo su quel training set.
- **Training Set Non Linearmente Separabile:** Qualunque iperpiano scelto commetterà almeno un errore di classificazione su un esempio del training set.

**Osservazioni:**

- Nella pratica, è difficile avere un training set linearmente separabile.
- L'algoritmo che trova un piano ottimo ha un costo computazionale troppo alto.
- Nel caso linearmente separabile, esistono algoritmi che risolvono il problema in tempo polinomiale.


## Caso Linearmente Separabile:

### Algoritmo 1: Programmazione Lineare

L'idea è di riscrivere il problema di *learning* come un problema di programmazione lineare (PL).

**Forma generale del problema di PL:**

$\vec{w^*} = \max_{\vec{w}}<\vec{u},\vec{w}>$, 
$s.t. A\vec{w}\geq v$

**Dimostrazione:**

Prendiamo un punto $\forall i$, tale che $sign(\vec{w},\vec{x})=y_{i}$. 
Dunque $\forall i, y_{i}<\vec{w},\vec{x}>\  >0$. 

Se esiste un iperpiano separatore $\vec{w}$ per cui vale la precedente, ne esiste uno che rispetta:

$\forall i, <\vec{w},\vec{x}> \ \geq 1$

**Dimostrazione:**

Sia $\vec{w^*}$ un iperpiano separatore e $\bar{w}=\frac{\vec{w^*}}{\gamma}$, dove $\gamma= min_{i} \{y_{i}<\vec{w_{i}},\vec{x_{i}}>\}$.

Allora:

$<\vec{w},\vec{x_{i}}> = y_i{\left( <\frac{w}{\gamma},\vec{x_{i}}> \right)}= \left( \frac{1}{\gamma}<\vec{w^*},\vec{x_{i}}> \right) \geq 1$

**Matrice A e Vettore v:**

$A = \begin{bmatrix} y_1  x_1 \\ y_2  x_2 \\ \vdots \\ y_m  x_m \end{bmatrix}$ 

$v = \begin{bmatrix} 1\\ \vdots  \\ 1\end{bmatrix}$ 

**Caso di Semispazi:**

Nel caso di semispazi non c'è nessun criterio di preferenza, dunque dobbiamo impostare $\vec{u}$=0 nella forma generale del problema di PL:

$\vec{w^*} = \max_{\vec{w}}<\vec{u},\vec{w}>$, 
$s.t. A\vec{w}\geq v$


### Algoritmo 2: Algoritmo di tipo iterativo: Perceptron Algorithm

* Pensato per una classe di ipotesi (delle reti neurali).
* In realtà, questo algoritmo lavora sui neuroni artificiali.
* Nell'ambito del *learning*, sono stati introdotti algoritmi che fanno uso di questo concetto. L'idea alla base è che, siccome gli umani sono bravi ad apprendere, cerchiamo di imitare il funzionamento degli organi preposti a questa funzione.

**Modello del Neurone Artificiale:**

* Questo modello cattura solo alcuni aspetti essenziali per il calcolo della struttura del nostro neurone.
* Ha un nucleo, dove avviene il funzionamento, da cui fuoriescono delle diramazioni (dendriti), che hanno delle terminazioni che vanno in contatto con altri neuroni.
* Le regioni di contatto sono chiamate sinapsi.
* Attraverso le sinapsi, il neurone riceve stimoli da altre cellule. 
* Se la somma di questi stimoli supera una certa soglia, il neurone propaga lo stimolo ai neuroni collegati.
* Oltre i dendriti, abbiamo una lunga terminazione chiamata assone, a sua volta ramificato per entrare in contatto con altre cellule.

**Analogia con il Neurone Artificiale:**

* Dal punto di vista informatico, i dendriti sono il canale di input.
* Il nucleo elabora gli stimoli. Se superano una certa soglia di attivazione, lo stimolo si propaga attraverso l'assone ad altri neuroni, che funge da canale di output.

# inserire foto

**Caratteristiche del Cervello:**

* All'interno del nostro cervello ci sono circa 90 miliardi di neuroni.
* Ogni neurone può avere da 5k a 100k dendriti.

