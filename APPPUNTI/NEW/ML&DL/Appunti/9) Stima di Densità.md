## Famiglie di Tecniche di Machine Learning


$$
\text{Famiglie di tecniche di ML:}
\begin{cases}
\text{Approcci Discriminativi (distribution free)} \\ \\

\text{Stima di Densità} \begin{cases}
\text{Metodi Parametrici} \\ \\
\text{Metodi non Parametrici}
\end{cases}
\end{cases}
$$

**1. Approcci Discriminativi (Distribution Free):**

* **Obiettivo:** Definire una regola che suddivida lo spazio delle features in due regioni: una per la classe positiva e una per la classe negativa.
* **Scopo:** Minimizzare l'errore di generalizzazione, ovvero la capacità del modello di prevedere correttamente nuovi dati non visti durante la fase di training.
* **Esempio:** Classificatori lineari come la Regressione Logistica.

**2. Stima di Densità:**

* **Obiettivo:** Ricostruire la densità di probabilità incognita che genera i dati.
* **Scopo:** Creare una funzione di densità che rappresenti la distribuzione dei dati, anche in presenza di complessità.
* **Famiglie:**
    * **Metodi Parametrici:** Si assume una specifica distribuzione (es. Gaussiana) e si stimano i suoi parametri (media e varianza) dai dati.
    * **Metodi non Parametrici:** Non si fa alcuna assunzione sulla forma della distribuzione, ma si ricostruisce la densità come somma di contributi associati a ciascun esempio del training set.

## Stima di Densità dei Parametri

L'obiettivo della stima di densità dei parametri è ricostruire la funzione di densità $p(x)$ che, dato un elemento del dominio, restituisce la sua densità. 

Possiamo esprimere la funzione di densità come:

$$p(x)=p_{\theta }(x)=f(x;\theta)$$

dove:

* $f$ è la densità di fase,
* $\theta$ sono i parametri.

### Distribuzione Normale o Gaussiana

La distribuzione normale o gaussiana è definita come:

$$N(\mu,\sigma)\text{ Distribuzione Normale con Media }\mu\text{ e Deviazione Standard }\sigma$$

La sua funzione di densità è:

$$f=\phi(x;\mu,\sigma)=\frac{1}{\sigma \sqrt{ 2\pi }}\exp\left[ - \frac{(x-\mu)^2}{2\sigma^2} \right]$$

![[9) Stima di Densità-20241115090658083.png]]

Nel caso multidimensionale, la funzione di densità diventa:

$$\phi(\vec{x}:\vec{\mu},\Sigma)=\frac{1}{|\Sigma|^{1/2}(2\pi)^{d/2} }\exp\left[ -\frac{1}{2}(\vec{x}-\vec{\mu})^T \Sigma^{-1}(\vec{x}-\vec{\mu})\right]$$

dove:

* $\vec{x}$ è il vettore dei dati,
* $\vec{\mu}$ è il vettore della media,
* $\Sigma$ è la matrice di covarianza.

Su due dimensioni, le curve di livello della distribuzione normale sono ellissi. Se $\Sigma$ è una matrice generale, gli assi degli ellissi possono essere orientati in base ai valori della matrice.

### Stimatore di un Parametro $\theta$ di una Distribuzione

Uno stimatore di un parametro $\theta$ di una distribuzione è una funzione $g$ che, dato un campione $S=\{ x_{1},\dots,x_{n} \}$ della popolazione, restituisce una stima $\hat{\theta}=g(s) \text{ di }\theta$.

### Stima di Massima Verosimiglianza (MLE - Maximum Likelihood Estimation)

La stima di massima verosimiglianza (MLE) si basa sulla probabilità di osservare il campione $S$ assumendo che $\theta$ siano i parametri della distribuzione osservata che ha generato i dati. Assumiamo che gli esempi del training set siano indipendenti.

La verosimiglianza (likelihood) è definita come:

$$\text{Verosimiglianza = }p(S|\theta )= \prod_{i=1}^m p_{\theta}(x_{i})$$

La verosimiglianza è una probabilità condizionale. Abbiamo due eventi:

1. Assumiamo che l'evento nella condizione sia verificato (i parametri $\theta$ sono quelli della distribuzione che ha generato i dati).
2. Vogliamo sapere la probabilità di osservare il primo evento (il campione $S$).

Parliamo di verosimiglianza perché facciamo un uso di questa probabilità contrario a quello usuale: noi abbiamo osservato $S$ (Training Set), mentre l'evento nella condizione (i parametri $\theta$) non sappiamo quale sia.

## Log-Likelihood

Cerchiamo i parametri che massimizzano la verosimiglianza. La verosimiglianza è la probabilità di osservare un determinato insieme di dati, dato un modello con parametri specifici. 
La log-verosimiglianza è il logaritmo naturale della verosimiglianza.

La log-verosimiglianza per un insieme di dati $S = \{x_1, x_2, ..., x_m\}$ con parametri $\theta$ è data da:

$$L(S;\theta)=\log(P(S|\theta)) = \log \left(\prod_{i=1}^m p_{\theta}(x_{i})\right) = \sum_{i=1}^m \log p_{\theta}(x_{i})$$

dove $p_{\theta}(x_i)$ è la densità di probabilità del dato $x_i$ dato il modello con parametri $\theta$.

Lo stimatore di massima verosimiglianza (MLE) è il valore dei parametri $\theta$ che massimizza la log-verosimiglianza:

$$\text{MLE: }\hat{\theta}=\arg\max_{\theta}L(S;\theta)$$

### Caso Gaussiano

Se la distribuzione ipotizzata è quella gaussiana, la densità di probabilità è data da:

$$p_{\theta}(x_i) = \phi(x_{i};\mu,\sigma) = \frac{1}{\sigma \sqrt{ 2\pi }}\exp\left[- \frac{(x_{i}-\mu)^2}{2\sigma^2} \right]$$

dove $\mu$ è la media e $\sigma$ è la deviazione standard.

La log-verosimiglianza per un insieme di dati gaussiani è:

$$L(S;\mu,\sigma)=\log \prod_{i=1}^m \phi(x_{i};\mu,\sigma)$$

$$=\log \prod_{i=1}^m\frac{1}{\sigma \sqrt{ 2\pi }}\exp\left[- \frac{(x_{i}-\mu)^2}{2\sigma^2} \right]$$

$$=\sum_{i=1}^m\left\{  \log  \frac{1}{\sigma \sqrt{ 2\pi }}-\frac{(x_{i}-\mu)^2}{2\sigma^2}  \right\}$$

$$=-m\log(\sigma \sqrt{ 2\pi })-\frac{1}{2\sigma^2}\sum_{i=1}^m(x_{i}-\mu)^2$$

Per trovare lo stimatore di massima verosimiglianza, calcoliamo le derivate parziali della log-verosimiglianza rispetto a $\mu$ e $\sigma$ e le poniamo uguali a zero:

$$\begin{cases}
\frac{\partial}{\partial \mu}L(s;\theta)=\frac{1}{\sigma^2}\sum_{i=1}^m(x_{i}-\mu)=0\\  
\\

\frac{\partial}{\partial\sigma}L(S;\theta)=-\frac{m}{\sigma}+\frac{1}{\sigma^3}\sum_{i=1}^m(x_{i}-\mu)^2=0
\end{cases}$$

Risolvendo queste equazioni, otteniamo gli stimatori di massima verosimiglianza per $\mu$ e $\sigma$:

$$ \text{Stimatore di massima Verosimiglianza per }\mu:
\begin{cases} 
 \frac{1}{\sigma^2}\sum_{i=1}^m(x_{i}-\mu)=0 \\
\sum_{i=1}x_{i}-m\mu=0 \\
m\mu=\sum_{i=1}^mx_{i} \\
\implies \mu=\frac{1}{m}\sum_{i=1}^mx_{i}
\end{cases}$$

$$\text{Deviazione Standard}
\begin{cases}
-\frac{m}{\sigma}+\frac{1}{\sigma^3}\sum_{i=1}^m(x_{i}-\mu)^2=0 \\
\frac{1}{\sigma^3}\sum_{i=1}^m(x_{i}-\mu)^2  =\frac{m}{\sigma} \\
\sigma=\sqrt{ \frac{1}{m}\sum_{i=1}^m(x_{i}-\mu)^2 }
\end{cases}
$$

### Stimatori

Uno stimatore è una funzione che utilizza i dati per stimare un parametro sconosciuto.

$$\text{Stimatore:}
\begin{cases}
\text{Unbiased: } & E[g(S)]=E[\hat{\theta}]=\theta \\ \\

\text{Biased altrimenti}
\end{cases}
$$

Lo stimatore di massima verosimiglianza per la media $\mu$ è unbiased, mentre quello per la deviazione standard $\sigma$ è biased.

### Caso Multidimensionale

Nel caso multidimensionale, la media e la covarianza sono stimate come segue:

$$\hat{\mu}=\frac{1}{m}\sum_{i=1}^m\vec{x}_{i}$$

$$\hat{\Sigma}=\frac{1}{m}(x- \hat{\mu})^T(x-\hat{\mu})$$

dove $\vec{x}_i$ è il vettore dei dati per l'i-esimo esempio e $x$ è la matrice dei dati.

## Classificatore Bayesiano

Se la distribuzione fosse nota, potremmo costruire il classificatore ottimale (bayesiano):

$$h_{\text{Bayes}}(x)=\arg\max_{y}p(y|x)$$

Applicando la regola di Bayes, possiamo riscrivere la probabilità condizionata $p(y|x)$ come:

$$p(y|x)p(x)=p(x|y)p(y)$$

Da cui otteniamo:

$$p(y|x)=\frac{p(x|y)p(y)}{p(x)}$$

Sostituendo nella formula del classificatore bayesiano, otteniamo:

$$h_{\text{Bayes}}(x)=\arg\max_{y}\frac{p(x|y)p(y)}{p(x)}$$

Possiamo ignorare $p(x)$ perché non cambia l'orientamento del risultato. Inoltre, in caso di equiprobabilità delle classi, possiamo ignorare $p(y)$ (o possiamo stimarlo dai dati).

Quindi, la formula del classificatore bayesiano si semplifica in:

$$h_{\text{Bayes}}(x)=\arg\max_{y}p(x|y)$$

$p(x|y)$ è la densità stimata della classe, che può essere espressa come:

$$p(x|y)=p_{\theta_{y}}(x)=\phi(x;\mu_{y},\sigma_{y})$$

dove $\phi(x;\mu_{y},\sigma_{y})$ è la funzione di densità di probabilità della classe $y$, con media $\mu_{y}$ e deviazione standard $\sigma_{y}$.

## Naive Bayes: Assunzione di Indipendenza delle Variabili

Questa versione semplificata del modello di Naive Bayes si utilizza nel caso multidimensionale, assumendo l'indipendenza delle variabili.

La probabilità di un vettore $\vec{x}$ dato un'etichetta $y$ è data da:

$$p_{\theta_{y}}(\vec{x})= \phi(\vec{x};\vec{\mu}_{y},\Sigma_{y})$$

Esplicitando il vettore $\vec{x}$:

$$p_{\theta_{y}}(x_{1},\dots,x_{d})= \phi(x_{1};\mu_{y_{1}},\sigma_{y_{1}})\cdot\phi(x_{2};\mu_{y_{2}},\sigma_{y_{2}})\cdot,\dots ,\cdot\phi(x_{d};\mu_{y_{d}},\sigma_{y_{d}})$$

Dove $\phi$ rappresenta la funzione di densità di probabilità di una distribuzione normale.

Se il decision boundary che ne scaturisce coincide con quello reale, il modello si comporta bene. Tuttavia, in generale, i dati non si distribuiscono in accordo a distribuzioni note. Per ottenere una stima più accurata, si possono utilizzare i **mixture models**. 

## Gaussian Mixture Models (GMM)

I Gaussian Mixture Models (GMM) sono un tipo di modello probabilistico utilizzato per modellare la distribuzione di dati complessi come una combinazione di distribuzioni gaussiane più semplici. 
Invece di assumere che i dati provengano da una singola distribuzione gaussiana, i GMM cercano di ricostruire la densità generatrice dei dati come una somma di contributi da diverse distribuzioni gaussiane.

**Formalmente:**

 Ricostruisce la densità generatrice dei dati come somma di $k$ distribuzioni gaussiane.

![[9) Stima di Densità-20241115101913331.png]]

La variabile $y$ assume valori nell'insieme $\{1, \dots, k\}$ e identifica la distribuzione della miscela (cluster) da cui il dato è stato generato.

**Generazione dei dati:**

Il processo di generazione dei dati con un GMM può essere descritto come segue:

1. **Selezione del cluster:** Si sceglie un cluster $y$ con probabilità $c(y) = Pr[Y=y]$.
2. **Generazione del dato:** Si genera un dato $\vec{x}$ in accordo con la distribuzione gaussiana $\phi(\vec{x};\vec{\mu}_{y},\vec{\Sigma}_{y})$ associata al cluster $y$.

La densità di probabilità del dato $\vec{x}$ è quindi data da:

$$p(\vec{x})=\sum_{y=1}^kPr(y)p(\vec{x}|y)=\sum_{y=1}^kc_{y}\cdot \phi(\vec{x};\vec{\mu}_{y},\Sigma_{y})$$

dove:

* $c_y$ è la probabilità a priori del cluster $y$.
* $\phi(\vec{x};\vec{\mu}_{y},\Sigma_{y})$ è la densità di probabilità gaussiana del dato $\vec{x}$ dato la media $\vec{\mu}_y$ e la covarianza $\Sigma_y$ del cluster $y$.

I parametri del modello GMM sono:

$$\theta=(\vec{c},\{ \vec{\mu}_{1},\dots,\vec{\mu}_{k} \},\{ \Sigma_{1},\dots,\Sigma_{k} \})$$

La log-verosimiglianza per un insieme di dati $S = \{\vec{x}_1, \vec{x}_2, ..., \vec{x}_m\}$ è data da:

$$L(S;\theta)=\log \prod_{i=1}^mp_{\theta}(\vec{x}_{i})=\sum_{i=1}^m\log \left[ \sum_{y=1}^kc_{y}\cdot \phi (\vec{x}_{i};\vec{\mu}_{y},\Sigma_{y}) \right]$$

Trovare i parametri $\theta$ che massimizzano la log-verosimiglianza è un problema intrattabile. Per questo motivo, si utilizzano algoritmi iterativi come l'algoritmo Expectation-Maximization (EM) per trovare una soluzione approssimata.

## Expectation-Maximization Algorithm (EM)

L'algoritmo Expectation-Maximization (EM) è un algoritmo iterativo utilizzato per trovare la stima di massima verosimiglianza dei parametri di un modello probabilistico quando i dati sono incompleti o mancanti. L'algoritmo EM si compone di due fasi:

1. **Fase di Expectation (E):** In questa fase, si calcola la probabilità condizionata di appartenenza ad ogni classe per ogni dato, dato il modello corrente.
2. **Fase di Maximization (M):** In questa fase, si aggiornano i parametri del modello in modo da massimizzare la verosimiglianza dei dati, tenendo conto delle probabilità condizionate calcolate nella fase E.

L'algoritmo EM viene ripetuto iterativamente fino a quando la convergenza è raggiunta, ovvero quando i parametri del modello non cambiano significativamente tra due iterazioni successive.

1) **Inizializzazione**

	$t=1; \ c_{y}^{(t)}=\frac{1}{k}; \ \mu_{y}^{(t)}=\text{Random}; \ \Sigma_{y}^{(t)}=diag\left[ \left( \frac{\sigma_{j}}{k} \right)^2 \right]$
	
1) **Expecation** $\forall \vec{x}_{i}$

	$P_{\theta^{(t)}}(y|\vec{x}_{i})=\frac{1}{Z_{i}}P_{\theta^{(t)}}(\vec{x}_{i}|y)\cdot P_{\theta^{(t)}}(y)=\frac{1}{Z_{i}}\phi(\vec{x}_{i};\vec{\mu}_{i}^{(t)},\Sigma_{y}^{(t)})$
	
	$Z_{i}=\sum_{y=1}^k c_{y}^{(t)}\phi(\vec{x}_{i};\vec{\mu}_{i}^{(t)},\Sigma_{y}^{(t)})$
	
1) **Maximization**

	$\vec{\mu}_{y}^{(t+1)}=\frac{\sum_{i=1}\vec{x}_{i}\cdot P_{\theta^{(t)}}(y|\vec{x}_{i})}{\sum_{i=1}^mP_{\theta^{(t)}}(y|\vec{x}_{i})}$
	
	$c_{y}^{(t+1)}=\frac{\sum_{i=1}^mP_{\theta^{(t)}}(y|\vec{x}_{i})}{\sum_{y=1}^k\sum_{i=1}^mP_{\theta^{(t)}}(y|\vec{x}_{i})}$
	
	$\sum^{(t+1)}_{d_{1},d_{2},y}=\frac{\sum_{i=1}^mp(y|\vec{x}_{i})(x_{i},d_{1}-\mu_{y},d_{1})(y|\vec{x}_{i})(x_{i},d_{2}-\mu_{y},d_{2})}{\sum_{i=1}^mP(y|\vec{x}_{i})}$

Questa tecnica è chiamata anche **Soft K-Means**.

## Anomaly Detection (AD)

**Anomalia o outlier**: Osservazione che si discosta molto dal resto della popolazione, fino a indurre il sospetto che sia stata generata da un meccanismo differente.

$$
\text{Setting AD:}
\begin{cases}
\text{Non supervisionato:}  \\
\quad\text{S: Dataset non etichettato} \\
\quad\alpha \text{: Contaminazione} \\
 \\
\text{Semi-Supervisionato (One-Class Classification)} \\
 \\
\text{Supervisionato (Problema di Classificazione Binaria Sbilanciata)}
\end{cases}
$$

- **Caso Non Supervisionato**: Dataset senza etichette, cerchiamo ciò che è anomalo per distanza da tutto il resto(lo isoliamo dal resto).

- **Caso Semi-Supervisionato**: Dataset senza etichette, ma si assume che la quasi totalità dei dati siano normali (il dataset rappresenta il concetto di normalità). Dato un nuovo punto, decidere se è normale o meno.

- **Caso Supervisionato**: Dataset etichettato. La classe anormale è molto piccola: si hanno molti più esempi della classe normale rispetto l'altro. La classificazione è binaria e sbilanciata.

### Approccio probabilistico all'Anomaly Detection

Gli outlier sono le istanze che hanno meno probabilità di essere osservate .

Dato $S$ stimiamo $p_{s}(x)=p_{\theta }(x)$.
- **Caso non supervisionato**: Restituisci gli $\alpha m$ esempi di $S$ che minimizzano $p_\theta$.
- **Caso Semi-Supervisionato**: Stabilisci una soglia $\pi$ :
	- Dato un nuovo esempio $\bar{x}$, dichiara $\bar{x}$ anomalo se $p_{\theta}(\bar{x})<\pi$

## Autoencoder

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

