## Teorema

Sia $HS_d$ la classe d'ipotesi dei semispazi omogenei, allora $VC_{dim}(HS_D)=d$.

**Dimostrazione:**

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

---
per ogni sottoinsieme d+1 vettori esiste almeno una funzione binaria che non possiamo costruire su questi vettori (combinazione di etichette che non riusciamo a costruire, usando i semispazi)

consideriamo sottinsieme generico di d+1 vettori $\vec{x_{1}},\vec{x_{2}},\dots,\vec{x_{d+1}}$
sappiamo che se abbiamo + vettori della dimensione dello spazio, questi non sono linearmente indipendeti, esiste almeno un vettore che può essere scritto come combinazione lineare degli altri

essendo $d+1$ non sono linearmente indipendenti, ovvero esistono dei coefficienti reali $a_i$ tali che: $∃ \ a_{i} \in R^d : \sum_{i=0}^{d+1}a_{i}\vec{x_{i}}=0$

dividiamo in due sottoinsiemi positivi e negativi
positivi: $I=\{ i:a_{i}>0 \}$
negativi: $j=\{ j:a_{j}<0 \}$

$\sum_{i \in I  }a_{i}\vec{x}_{i}+\sum_{j \in J}a_{j}\vec{x_{j}}=0$
$\sum_{i \in I  }a_{i}\vec{x}_{i}=- \sum_{j \in J}a_{j}\vec{x_{j}}=\sum_{j \in J}|a_{j}|\vec{x_{j}}$


vogliamo dimostrare che non è possibile costruire la seguente fuznione:
$$
\begin{cases}
\forall_{i \in I}, h_{\vec{x}}(\vec{x_{i}})=+1 \\ \\
\forall_{J \in J}, h_{\vec{x}}(\vec{x_{j}})=-1
\end{cases}
$$

$\sum_{i \in I}a_{i}<\vec{w},\vec{x_{i}>}= < \sum_{i \in I}a_{i}\vec{w},\vec{x_{i}}>$
$= \ < \sum_{j \in J}|a_{j}|\vec{w},\vec{x_{j}}> = \  \sum_{j \in J}|a_{j}| \ <\vec{w},\vec{x_{j}}>$

supponiamo che il w usato sia quello che implementa questa funzione
gli a_i sono >0, la funzione assegna +1 a tutti gli x_i, quindi positivo, quindi la prima sommatoria è >0

nella seconda parte, gli a_j sono positivi perchè in valore assoulto, il prodott oscalare invece sono <0 perchè[...], quindi l'espressione è < 0
contraddizione perchè questa espressione dovrebbe essere simultaneamente <0 e >0.

la nostra VC_dim sarà esattamente d, come detto nel precedente teorma

quando usiamo gli iperpiani separatori non possiamo costruire funzioni arbitrariamente complesse: se siamo nel piano R^2, d=2, la VC_dim sarà d+1=3. Se prendiamo 3 punti possiamo trrovare sempre un iperpiano separatore che assegna una combinazione arbitraria di valori booleani. se ne prendiamo 4 non sempre sarà possibile

---

## Dimostrazione della VC-dimension per iperpiani separatori

Questo documento dimostra che la VC-dimension per gli iperpiani separatori in uno spazio vettoriale di dimensione $d$ è esattamente $d$.

**Teorema:** La VC-dimension per gli iperpiani separatori in $R^d$ è $d$.

**Dimostrazione:**

Per dimostrare questo teorema, dobbiamo dimostrare due punti:

1. **VC-dimension ≥ d:**  Dobbiamo dimostrare che esiste un insieme di $d$ punti che possono essere classificati arbitrariamente da un iperpiano separatore.
2. **VC-dimension ≤ d:** Dobbiamo dimostrare che per ogni insieme di $d+1$ punti, esiste almeno una funzione binaria che non può essere realizzata da un iperpiano separatore.

**Punto 1: VC-dimension ≥ d**

Questo punto è relativamente semplice. Consideriamo un insieme di $d$ punti linearmente indipendenti in $R^d$. Poiché sono linearmente indipendenti, possiamo sempre trovare un iperpiano che li separa in qualsiasi combinazione di classi desiderata. Quindi, la VC-dimension è almeno $d$.

**Punto 2: VC-dimension ≤ d**

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

$$\sum_{i \in I}a_{i}<\vec{w},\vec{x_{i}}> = < \sum_{i \in I}a_{i}\vec{w},\vec{x_{i}}>$$

$$= < \sum_{j \in J}|a_{j}|\vec{w},\vec{x_{j}}> = \  \sum_{j \in J}|a_{j}| \ <\vec{w},\vec{x_{j}}>$$

Poiché $a_i > 0$ per $i \in I$ e la funzione assegna $+1$ a tutti gli $\vec{x_i}$, il prodotto scalare $<\vec{w},\vec{x_{i}}>$ è positivo. Quindi, la prima sommatoria è positiva.

Nella seconda parte, gli $a_j$ sono positivi perché sono in valore assoluto. Tuttavia, il prodotto scalare $<\vec{w},\vec{x_{j}}>$ è negativo perché la funzione assegna $-1$ a tutti gli $\vec{x_j}$. Quindi, l'espressione è negativa.

Abbiamo una contraddizione perché questa espressione dovrebbe essere simultaneamente positiva e negativa. Quindi, non è possibile costruire un iperpiano separatore che implementi la funzione binaria desiderata.

**Conclusione:**

Abbiamo dimostrato che per ogni insieme di $d+1$ punti in $R^d$, esiste almeno una funzione binaria che non può essere realizzata da un iperpiano separatore. Quindi, la VC-dimension per gli iperpiani separatori in $R^d$ è al massimo $d$.

Combinando questo risultato con il punto 1, possiamo concludere che la VC-dimension per gli iperpiani separatori in $R^d$ è esattamente $d$.

**Esempio:**

Nel piano $R^2$, $d=2$. Quindi, la VC-dimension è $d+1=3$. Se prendiamo 3 punti, possiamo sempre trovare un iperpiano separatore che assegna una combinazione arbitraria di valori booleani. Tuttavia, se prendiamo 4 punti, non sempre sarà possibile trovare un iperpiano separatore che li classifichi in modo arbitrario.




## Esempio: Funzione XOR

Usiamo -1 e +1 per i valori vero o falso.

![[Senza nome-20241010121227430.png|356]]

Se utilizziamo questi valori, la funzione XOR sarà:

| $x_1$ | $x_2$ | Risultato |
| ----- | ----- | --------- |
| -1    | -1    | 1         |
| -1    | 1     | 0         |
| 1     | -1    | 0         |
| 1     | 1     | 1         |

Il percettrone non può catturare la funzione XOR (un singolo neurone non è in grado di rappresentarla). 

## Teorema NO-FREE LUNCH
dimostra che non esiste un algoritmo di learning universale
dato un dominio infito x e considerata la classe d'ipotesi formata da tutte le funzioni booleane.. questa classe non è pac-learnable

ne esiste una seconda formulazione che ci dice sempre che non esiste un algoritmo di learning universale ma fornisce anche altro info agiuntive
**enunciato**: Sia A un algoritmo di learning (combinazione di una classe d'ipotesi+minimizzazione rischio empirico) $A=ERM_H$, allora esistono un problema di learning P e un secondo algoritmo $A'\neq A(H'\neq H)$ tali che
$$
\begin{cases}
\ 1) \ \text{ A fallisce su P (Overfitting)} \\
\ 2) \ \text{ A' ha successo su P}
\end{cases}
$$
anche se la 1) ci dice che la classe può fallire, la buona notizia è che esiste una classe d'ipotesi diversa che si comporta bene sul problema (2)

ci dice che non esiste un algo di learning universale perchè: supponiamo che abbia una predilezione per la classe dei semispazi: il teorema dice che se usi sempre la stessa classe d'ipotesi prima o poi arriverà un problema che non riesci a risolvere
dobbiamo quindi cambiare classe di'ipotesi

quando abbiamo un probleam di learning, dobbiamo provare più classi d'ipotesi (model selection)

## Model Selection

(le ipotesi vengono da classi di ipotesi diverse, algoritmi diversi)

Dato un insieme  di ipotesi $H^*=\{ h_1,h_2, \cdots , h_r \}$ la model solection consiste nello scegliere la migliore ipotesi $H^*$ 

problema della regressione
(grafico regressione)
1 passo è scegliere la classe d iptoesi da usare
se usiamo la regressione polinomiale, al variare del grado del polinomio cambia la classe d'ipotesi

quando facciamo model selection consideriamo una famiglia di classi d ipotesi, e noi vogliamo scegliere la migliore

quando usiamo famiglia di classi d ipotesi si introduce il concetto di iperparametro: per ogni val di iperparametero cambierà al forma della classe

### Iperparametro
Parametro di un algoritmo di Learning che viene fissato a priori (prima di vedere i dati)

$y=p_n(x)$
l'iperaparametro è il grado del polinomio $p_n$ (è la n)
dobbiamo quindi provare diversi val di n
$n=1$ è una retta
$n=2$ genera $h_2$ che approssima meglio la funzione
$n=3$ genera $h_3$,  approssima ancora meglio la funzione
![[Senza nome-20241010122941471.png|436]]
possiamo arrivare a una n che passa da tutti i punti $n=|S|$

$H^*=\{h_1,h_2,\cdots , h_r \}$
voglio estrarre l'ipotesi migliore $h^*$

Supponiamo di  porre $h^*=\arg \ \min_{h\in H^*} \ L_S(h)$
il rischio empirico si abbsassa perhcè la fun sono così complicate che si adattano troppo bene alla funzione, fino a diventare inutili nella predizione
dunque questa funzione non va bene.
