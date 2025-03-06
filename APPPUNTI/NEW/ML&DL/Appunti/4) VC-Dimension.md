## Teorema

Sia $HS_d$ la classe d'ipotesi dei semispazi omogenei, allora $VC_{dim}(HS_D)=d$.

##### Dimostrazione:

1) Esistono $d$ oggetti del dominio su cui possiamo costruire tutte le possibili funzioni booleane. Il dominio è $X=R^d$.

$$
\begin{cases}
\vec{e_{1}}=(1,0,\dots,0) \ Y_{1}\in -1,1 \\
\vec{e_{2}}=(0,1,\dots,0) \ Y_{2}\in -1,1 \\
\vec{e_{d}}=(0,0,\dots,1 )\ Y_{d}\in -1,1\\
\end{cases}$$

Supponiamo di aver scelto i valori per le etichette.

Dobbiamo costruire il vettore $\vec{w}=(y_{1},y_{2},\dots,y_{d})$.
Il semispazio $\forall_{i}, \ h_{\vec{w}}(\vec{e}_{i})= <\vec{w},\vec{e}_{i}> =y_{i}$ sarà uguale alla $i$-esima componente.

Abbiamo dimostrato che esistono i vettori su cui possiamo costruire tutte le funzioni binarie.

## Dimostrazione della VC-dimension per iperpiani separatori

La VC-dimension per gli iperpiani separatori in uno spazio vettoriale di dimensione $d$ è esattamente $d$.

**Teorema:** La VC-dimension per gli iperpiani separatori in $R^d$ è $d$.

##### Dimostrazione:

Per dimostrare questo teorema, dobbiamo dimostrare due punti:

- **VC-dimension ≥ d:** Dobbiamo dimostrare che esiste un insieme di $d$ punti che possono essere classificati arbitrariamente da un iperpiano separatore.
- **VC-dimension ≤ d:** Dobbiamo dimostrare che per ogni insieme di $d+1$ punti, esiste almeno una funzione binaria che non può essere realizzata da un iperpiano separatore.

##### Punto 1: VC-dimension ≥ d

Questo punto è relativamente semplice. Consideriamo un insieme di $d$ punti linearmente indipendenti in $R^d$. Poiché sono linearmente indipendenti, possiamo sempre trovare un iperpiano che li separa in qualsiasi combinazione di classi desiderata. Quindi, la VC-dimension è almeno $d$.

##### Punto 2: VC-dimension ≤ d

Per dimostrare questo punto, consideriamo un sottoinsieme generico di $d+1$ vettori in $R^d$: $\vec{x_1}, \vec{x_2}, ..., \vec{x_{d+1}}$.

Sappiamo che se abbiamo più vettori della dimensione dello spazio, questi non sono linearmente indipendenti. Quindi, esiste almeno un vettore che può essere scritto come combinazione lineare degli altri.

Essendo $d+1$ vettori, non sono linearmente indipendenti. Quindi, esistono dei coefficienti reali $a_i$ tali che:

$$∃ \ a_{i} \in R^d : \sum_{i=0}^{d+1}a_{i}\vec{x_{i}}=0$$

Dividiamo i coefficienti $a_i$ in due sottoinsiemi:

* **Positivi:** $I = \{ i: a_{i} > 0 \}$
* **Negativi:** $J = \{ j: a_{j} < 0 \}$

Possiamo riscrivere l'equazione precedente come:

$$\sum_{i \in I}a_{i}\vec{x}_{i} + \sum_{j \in J}a_{j}\vec{x_{j}} = 0$$

$$\sum_{i \in I}a_{i}\vec{x}_{i} = - \sum_{j \in J}a_{j}\vec{x_{j}} = \sum_{j \in J}|a_{j}|\vec{x_{j}}$$

Vogliamo dimostrare che non è possibile costruire la seguente funzione binaria:

$$
\begin{cases}
\forall_{i \in I}, h_{\vec{x}}(\vec{x_{i}})=+1 \\ \\
\forall_{j \in J}, h_{\vec{x}}(\vec{x_{j}})=-1
\end{cases}
$$

Supponiamo che esista un iperpiano separatore definito dal vettore $\vec{w}$ che implementa questa funzione. Quindi, avremmo:

$$\sum_{i \in I}a_{i}<\vec{w},\vec{x_{i}}> \ = \ < \sum_{i \in I}a_{i}\vec{w},\vec{x_{i}}>$$

$$= \ < \sum_{j \in J}|a_{j}|\vec{w},\vec{x_{j}}> \ = \ \sum_{j \in J}|a_{j}| \ <\vec{w},\vec{x_{j}}>$$

Poiché $a_i > 0$ per $i \in I$ e la funzione assegna $+1$ a tutti gli $\vec{x_i}$, il prodotto scalare $<\vec{w},\vec{x_{i}}>$ è positivo. Quindi, la prima sommatoria è positiva.

Nella seconda parte, gli $a_j$ sono positivi perché sono in valore assoluto. Tuttavia, il prodotto scalare $<\vec{w},\vec{x_{j}}>$ è negativo perché la funzione assegna $-1$ a tutti gli $\vec{x_j}$. Quindi, l'espressione è negativa.

Abbiamo una contraddizione perché questa espressione dovrebbe essere simultaneamente positiva e negativa. Quindi, non è possibile costruire un iperpiano separatore che implementi la funzione binaria desiderata.

##### Conclusione:

Abbiamo dimostrato che per ogni insieme di $d+1$ punti in $R^d$, esiste almeno una funzione binaria che non può essere realizzata da un iperpiano separatore. Quindi, la VC-dimension per gli iperpiani separatori in $R^d$ è al massimo $d$.

Combinando questo risultato con il punto 1, possiamo concludere che la VC-dimension per gli iperpiani separatori in $R^d$ è esattamente $d$.

##### Esempio:

Nel piano $R^2$, $d=2$. Quindi, la VC-dimension è $d+1=3$. Se prendiamo 3 punti, possiamo sempre trovare un iperpiano separatore che assegna una combinazione arbitraria di valori booleani. Tuttavia, se prendiamo 4 punti, non sempre sarà possibile trovare un iperpiano separatore che li classifichi in modo arbitrario.

## Esempio: Funzione XOR

Usiamo -1 e +1 per i valori vero o falso.

![[Senza nome-20241010121227430.png|356]]

Se utilizziamo questi valori, la funzione XOR sarà:

| $x_1$ | $x_2$ | Risultato |
| ----- | ----- | --------- |
| -1 | -1 | 1 |
| -1 | 1 | 0 |
| 1 | -1 | 0 |
| 1 | 1 | 1 |

Il percettrone non può catturare la funzione XOR (un singolo neurone non è in grado di rappresentarla).

## Teorema NO-FREE LUNCH

Il teorema NO-FREE LUNCH dimostra che non esiste un algoritmo di apprendimento universale.

##### Prima formulazione:

Dato un dominio infinito $X$ e considerata la classe di ipotesi formata da tutte le funzioni booleane, questa classe non è PAC-learnable.

##### Seconda formulazione:

Sia $A$ un algoritmo di apprendimento (combinazione di una classe di ipotesi e minimizzazione del rischio empirico) $A = ERM_H$. Allora esistono un problema di apprendimento $P$ e un secondo algoritmo $A' \neq A (H' \neq H)$ tali che:

$$
\begin{cases}
\ 1) \ \text{ A fallisce su P (Overfitting)} \\
\ 2) \ \text{ A' ha successo su P}
\end{cases}
$$

Anche se la 1) ci dice che la classe può fallire, la buona notizia è che esiste una classe di ipotesi diversa che si comporta bene sul problema (2).

##### Interpretazione:

Il teorema NO-FREE LUNCH ci dice che non esiste un algoritmo di apprendimento universale perché:

* Supponiamo che un algoritmo abbia una predilezione per una classe di ipotesi specifica, ad esempio i semispazi. Il teorema dice che se usiamo sempre la stessa classe di ipotesi, prima o poi arriverà un problema che non riusciremo a risolvere.
* Dobbiamo quindi cambiare classe di ipotesi.

##### Conclusione:

Quando abbiamo un problema di apprendimento, dobbiamo provare più classi di ipotesi (model selection). Non esiste un algoritmo universale che funzioni per tutti i problemi.

## Model Selection

La model selection è il processo di scelta della migliore ipotesi da un insieme di ipotesi $H^* = \{h_1, h_2, \dots, h_r\}$, dove le ipotesi possono provenire da classi di ipotesi diverse o essere generate da algoritmi diversi.

##### Esempio: Regressione Polinomiale

Consideriamo il problema della regressione. Un primo passo è scegliere la classe di ipotesi da utilizzare. Se scegliamo la regressione polinomiale, al variare del grado del polinomio cambia la classe di ipotesi.

##### Famiglia di Classi di Ipotesi e Iperparametri

Quando si effettua la model selection, si considera una famiglia di classi di ipotesi. L'obiettivo è scegliere la migliore classe di ipotesi all'interno di questa famiglia.

L'utilizzo di una famiglia di classi di ipotesi introduce il concetto di **iperparametro**. Un iperparametro è un parametro di un algoritmo di apprendimento che viene fissato a priori, prima di vedere i dati.

##### Esempio: Iperparametro nella Regressione Polinomiale

Nella regressione polinomiale, $y = p_n(x)$, l'iperparametro è il grado del polinomio $n$. Dobbiamo quindi provare diversi valori di $n$.

* $n = 1$ genera una retta.
* $n = 2$ genera $h_2$, che approssima meglio la funzione.
* $n = 3$ genera $h_3$, che approssima ancora meglio la funzione.

![[4) VC-Dimension-20241011084525649.png|512]]
Possiamo arrivare a un valore di $n$ che passa da tutti i punti, $n = |S|$, dove $|S|$ è il numero di punti dati.

##### Scelta dell'Ipotesi Migliore

Dato l'insieme di ipotesi $H^* = \{h_1, h_2, \dots, h_r\}$, vogliamo estrarre l'ipotesi migliore $h^*$.

Potremmo pensare di porre $h^* = \arg \min_{h \in H^*} L_S(h)$, dove $L_S(h)$ è il rischio empirico di $h$. Tuttavia, questo approccio può portare a un overfitting, ovvero a una funzione troppo complessa che si adatta troppo bene ai dati di training e non generalizza bene a nuovi dati.

In conclusione, la model selection è un processo cruciale nell'apprendimento automatico che richiede di considerare diverse classi di ipotesi e di scegliere quella che offre il miglior compromesso tra accuratezza e generalizzazione.

## Validation Set

Stiamo confrontando ipotesi in generale, che provengono da classi di ipotesi con espressività diversa. Il **validation set** ha la stessa forma del training set ma non viene utilizzato durante l'addestramento. Lo usiamo solo per stimare l'errore di generalizzazione.

$$V=\{(x_1',y_1'),(x_2',y_2'),\dots,(x_m',y_m')\}$$

*V è indipendente da S*

##### Opzioni per la creazione del validation set:

- **Creazione di S e V:** Possiamo creare sia il training set (S) che il validation set (V) separatamente.
- **Sottoinsieme di S:** Se abbiamo solo S, possiamo sacrificarne una parte e metterla nel validation set. Selezioniamo un sottoinsieme casuale di S e lo usiamo come Validation.

##### Stima dell'errore di generalizzazione:

$$h^* = \arg \ \min_{h \in H^*} L_v(h)$$

Dovrebbe funzionare perché l'errore sul validation set è una buona stima dell'errore di generalizzazione.

$H^*=\{  h_1, h_{2},\cdots, h_r  \}$ è un insieme finito di ipotesi.

La regola precedente corrisponde a minimizzare il rischio empirico sul validation set in corrispondenza di una classe di ipotesi finita (usiamo V e non S su una classe di ipotesi finita). Essendo la classe di ipotesi finita, vale la A-PAC-Learnability e la convergenza uniforme.

Questo problema ha una **sample complexity** $m=\frac{\log\left( \frac{2|H|}{S} \right)}{2\epsilon^2}$.

Da qui deriva che $\forall h \in H^*,| L_{D}(h)-L_{V}(h)|\leq\epsilon$.

Dalla sample complexity deriva $\epsilon=\sqrt{ \frac{\frac{\log(2|H|)}{S}}{2m_{v}} }$.

Dalla differenza tra gli errori deriva $L_{D}(h^*)\leq L_{V}(h^*)+  \sqrt{ \frac{\frac{\log(2|H|)}{S}}{2m_{v}}}$.

##### Influenza della dimensione del validation set:

* Più è grande il validation set, più l'errore sul validation set approssimerà l'errore vero.
* $H^*$ invece ci dice che all'aumentare del numero di ipotesi che confrontiamo, aumenta l'errore.

##### Indipendenza tra V e S:

Per garantire l'indipendenza tra V e S, non devono avere elementi in comune. Per aumentare la qualità della stima, dobbiamo incrementare V, ma ciò significa prendere esempi da S e quindi la qualità del training diminuisce.

Nonostante ciò, questa tecnica è la più potente a livello teorico, poiché abbiamo dei bound sull'errore associato alla stima del rischio.

##### Varianti per alleviare il problema:

Esistono delle varianti per alleviare il problema della diminuzione della qualità del training set quando si aumenta la dimensione del validation set.

## K-fold Cross-Validation

La K-fold Cross-Validation è una tecnica utilizzata per valutare le prestazioni di un modello di machine learning. Consiste nel partizionare il set di training in *k* gruppi (solitamente *k* = 10). L'obiettivo è massimizzare il numero di esempi utilizzati nella fase di training.

Per *k* volte, si creano *k* problemi distinti, in cui un gruppo diventa il validation set e i restanti *k-1* gruppi diventano il training set.

Formalmente, per ogni *i* da 1 a *k*:

$$\forall_{i}
\begin{cases}
V_{i}=S_{i} \\
S'=S \setminus S' \to h_{i}=ERM_{H}(S'_{i})
\end{cases}
$$

dove:

* $V_{i}$ è il validation set per l'iterazione *i*.
* $S_{i}$ è il *i*-esimo gruppo del set di training.
* $S'$ è il training set per l'iterazione *i*.
* $h_{i}$ è il modello di machine learning addestrato sul training set $S'_{i}$.
* $ERM_{H}(S'_{i})$ è l'algoritmo di apprendimento che minimizza l'errore sul training set $S'_{i}$.

Dalla seconda equazione, si deriva il costo $c_{i}$ per l'iterazione *i*:

$$\to c_{i}=L_{Vi}(h_{i})$$

dove $L_{Vi}(h_{i})$ è la funzione di perdita calcolata sul validation set $V_{i}$ utilizzando il modello $h_{i}$.

L'errore totale viene quindi calcolato come la media degli errori su tutte le *k* iterazioni:

$$err=\frac{1}{k} \sum_{i=1}^k e_{i}$$

Questo processo viene ripetuto per *k* volte. Alla fine, si sceglie l'iperparametro che ha l'errore minimo. Si utilizza quindi l'intero set di training per addestrare il modello con il valore dell'iperparametro ottimale.

##### Vantaggi:

* Tutti gli esempi del set di training vengono utilizzati per l'addestramento.

##### Svantaggi:

* Non ci sono garanzie sulla bontà della stima dell'errore, poiché non è garantita l'indipendenza tra gli errori delle diverse iterazioni.

## Model-Selection Curve

La Model-Selection Curve è un grafico che rappresenta qualitativamente il comportamento degli errori in gioco durante la selezione del modello.

Sull'asse delle ascisse (x) si rappresenta l'iperparametro del modello, mentre sull'asse delle ordinate (y) si rappresentano gli errori.

Quando i valori dell'iperparametro sono bassi, l'errore empirico (errore sul training set) è più basso, tendendo a 0.

![[4) VC-Dimension-20241011090737310.png|759]]
La parabola nel grafico rappresenta l'errore di validazione (errore sul validation set).

Il grafico può essere diviso in tre regioni:

- **Regione di underfitting:** Associata ai valori più bassi dell'iperparametro. L'errore di validazione e l'errore empirico sono molto vicini e entrambi tendono ad essere alti. La classe di ipotesi è troppo semplice e non si adatta bene ai dati.
- **Regione intermedia:** Regione in cui si colloca la soluzione ottimale. L'obiettivo è isolare la regione in cui si trova il minimo dell'errore di validazione.
- **Regione di overfitting:** Associata ai valori più alti dell'iperparametro. L'errore empirico si abbassa di molto, ma l'errore sul validation set è molto grande (la differenza tra i due errori è grande). Il modello si adatta troppo bene ai dati di training, perdendo la capacità di generalizzare a nuovi dati.

Al grafico si può aggiungere l'errore Bayesiano, che rappresenta l'errore minimo possibile per il problema. L'errore Bayesiano non può essere raggiunto in pratica, ma rappresenta un limite inferiore per l'errore di qualsiasi modello.

La Model-Selection Curve è uno strumento utile per visualizzare il trade-off tra complessità del modello e accuratezza. Aiuta a identificare la regione ottimale per l'iperparametro, che bilancia la capacità di adattamento ai dati di training con la capacità di generalizzare a nuovi dati.

## Minimizzazione del Rischio Strutturale (SRM)

La Minimizzazione del Rischio Strutturale (SRM) è una tecnica di model selection che non fa uso di un validation set separato.

L'obiettivo di SRM è trovare l'ipotesi $h^*$ che minimizza il rischio strutturale, definito come la somma dell'errore empirico $L_S(h)$ e di un termine di penalizzazione $\epsilon(h)$:

$$h^*= \arg \min _{h\in H^*}L_S(h)+\epsilon(h)$$

Il termine $\epsilon(h)$ rappresenta una stima della differenza tra l'errore empirico e l'errore di generalizzazione. Questo termine è legato alla complessità della classe di ipotesi $H$ e può essere stimato utilizzando la dimensione VC (VC dimension) di $H$.

Dal teorema fondamentale del PAC learning, sappiamo che:

$$m=\frac{d+\log\left( \frac{1}{\delta} \right)}{\epsilon^2}$$

dove:

* $m$ è il numero di esempi di training.
* $d$ è la dimensione VC della classe di ipotesi.
* $\epsilon$ è l'errore di generalizzazione.
* $\delta$ è la probabilità di errore.

Risolvendo per $\epsilon$, otteniamo:

$$\epsilon=\sqrt{ \frac{d(h)+\log\left( \frac{1}{\delta} \right)}{m} }$$

Sostituendo questa espressione nella formula del rischio strutturale, otteniamo:

$$h^*= \arg \min _{h\in H^*}L_S(h)+\sqrt{ \frac{d(h)+\log\left( \frac{1}{\delta} \right)}{m} }$$

Questa formula indica che l'ipotesi ottimale $h^*$ è quella che minimizza la somma dell'errore empirico e di un termine di penalizzazione che dipende dalla dimensione VC della classe di ipotesi e dal numero di esempi di training.

In altre parole, più è complicata la classe di ipotesi (più alta è la dimensione VC), più cresce il termine $\epsilon(h)$ e aumenta il rischio strutturale. Questo perché una classe di ipotesi più complessa è più soggetta a overfitting, ovvero ad adattarsi troppo bene ai dati di training e a perdere la capacità di generalizzare a nuovi dati.

SRM è una tecnica efficace per la model selection, in quanto consente di trovare un compromesso tra la complessità del modello e la sua accuratezza.
