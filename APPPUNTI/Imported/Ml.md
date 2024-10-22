## Discesa del Gradiente per la Minimizzazione della Funzione di Costo

La minimizzazione dell'errore in machine learning si traduce in un problema di ottimizzazione. Questo tipo di problema è ben studiato e esistono algoritmi efficienti per risolverlo. L'obiettivo è trovare il minimo di una funzione convessa.

### L'Algoritmo della Discesa del Gradiente

Un algoritmo efficiente per la minimizzazione di funzioni convesse è la **discesa del gradiente**. Questo algoritmo è iterativo e si basa sull'utilizzo del gradiente della funzione per trovare il minimo.

**Funzionamento:**

1. **Punto di partenza:** Si inizia con una soluzione iniziale, indicata con `w`.
2. **Calcolo del gradiente:** Ad ogni iterazione, si calcola il gradiente della funzione `F(w)` nel punto corrente `w`. Il gradiente rappresenta la derivata della funzione in caso di funzione monodimensionale, mentre in caso di funzione multidimensionale è un vettore che indica la direzione di massima pendenza.
3. **Aggiornamento della soluzione:** Si aggiorna la soluzione `w` muovendosi nella direzione opposta al gradiente. Questo perché si vuole minimizzare la funzione, quindi ci si sposta nella direzione di discesa. L'aggiornamento è dato da:

   ```
   w(t+1) = w(t) - η * ∇f(w(t))
   ```

   dove:
   - `w(t)` è la soluzione corrente all'iterazione `t`.
   - `η` è il **learning rate**, che determina l'ampiezza dello spostamento ad ogni iterazione.

4. **Iterazione:** Si ripetono i passaggi 2 e 3 fino a raggiungere un punto di minimo o un criterio di arresto.

**Interpretazione:**

Il learning rate `η` controlla l'ampiezza del passo che si compie ad ogni iterazione. Un valore di `η` troppo grande può portare a oscillazioni e difficoltà nel trovare il minimo, mentre un valore troppo piccolo può rallentare la convergenza.




## Algoritmi di Ottimizzazione: Discesa del Gradiente e Varianti

La discesa del gradiente è un algoritmo di ottimizzazione iterativo utilizzato per trovare il minimo di una funzione. In Machine Learning, viene spesso utilizzato per addestrare modelli trovando i parametri che minimizzano la funzione di perdita.

**Principio di funzionamento:**

L'algoritmo parte da un punto iniziale e si muove iterativamente in direzione opposta al gradiente della funzione. Il gradiente indica la direzione di massima crescita della funzione, quindi muoversi nella direzione opposta ci porta verso un minimo locale.

**Formula:**

```
w(t+1) = w(t) - η * ∇f(w(t))
```

Dove:

* **w(t)** è il vettore dei parametri all'iterazione t
* **η** è la learning rate, un valore che controlla la dimensione del passo
* **∇f(w(t))** è il gradiente della funzione f rispetto a w all'iterazione t

**Varianti della discesa del gradiente:**

Esistono diverse varianti della discesa del gradiente, ciascuna con i suoi vantaggi e svantaggi:

* **Discesa del gradiente stocastica (SGD):** Utilizza un singolo esempio di addestramento per calcolare il gradiente ad ogni iterazione. È più veloce della discesa del gradiente batch, ma può essere meno stabile.
* **Discesa del gradiente batch:** Utilizza l'intero set di dati di addestramento per calcolare il gradiente ad ogni iterazione. È più stabile di SGD, ma può essere più lento.
* **Discesa del gradiente mini-batch:** Utilizza un sottoinsieme di dati di addestramento (mini-batch) per calcolare il gradiente ad ogni iterazione. È un compromesso tra SGD e la discesa del gradiente batch, offrendo una buona combinazione di velocità e stabilità.

**Scelta del punto di arresto:**

Dopo un numero di iterazioni, l'algoritmo di discesa del gradiente si ferma. Esistono diversi criteri per determinare il punto di arresto:

* **Numero massimo di iterazioni:** L'algoritmo si ferma dopo un numero predefinito di iterazioni.
* **Tolleranza:** L'algoritmo si ferma quando la norma del gradiente è inferiore a una certa soglia.
* **Convergenza:** L'algoritmo si ferma quando la variazione dei parametri tra due iterazioni consecutive è inferiore a una certa soglia.

**Restituzione della soluzione:**

Una volta che l'algoritmo si ferma, possiamo restituire la soluzione trovata. Esistono due opzioni comuni:

* **Soluzione associata al minimo valore di f(w):** Restituiamo il vettore dei parametri che ha prodotto il valore minimo della funzione f(w) durante l'esecuzione dell'algoritmo.
* **Ultima iterazione:** Restituiamo il vettore dei parametri dell'ultima iterazione dell'algoritmo.

La scelta tra queste due opzioni dipende dal problema specifico e dalle esigenze dell'utente.






## Discesa del Gradiente e Discesa del Gradiente Stocastica

### Discesa del Gradiente

La discesa del gradiente è un algoritmo di ottimizzazione utilizzato per trovare il minimo di una funzione di costo. Il processo consiste nel calcolare il gradiente della funzione di costo e aggiornare i parametri in direzione opposta al gradiente.

**Passaggi:**

1. **Scegliere la funzione di costo:** Definire la funzione di costo che si desidera minimizzare.
2. **Calcolare il gradiente:** Calcolare il gradiente della funzione di costo rispetto ai parametri.
3. **Aggiornare i parametri:** Aggiornare i parametri in direzione opposta al gradiente, utilizzando un passo di apprendimento (learning rate).

**Esempio: Regressione Lineare**

Nella regressione lineare, la funzione di costo è la "mean squared error":

$$L(h_v(x), y) = \frac{1}{2} ||v^T x - y||^2$$

Il gradiente della funzione di costo è:

$$\nabla_v L(h_v(x), y) = v^T x - y$$

L'aggiornamento dei parametri è quindi:

$$v = v - \eta (v^T x - y)$$

dove $\eta$ è il passo di apprendimento.

### Discesa del Gradiente Stocastica (SGD)

La discesa del gradiente stocastica (SGD) è una variante della discesa del gradiente che utilizza un sottoinsieme casuale degli esempi di training per calcolare il gradiente ad ogni iterazione. Questo approccio presenta diversi vantaggi:

* **Velocità:** SGD è generalmente più veloce della discesa del gradiente standard, soprattutto per dataset di grandi dimensioni.
* **Efficienza:** SGD richiede meno memoria rispetto alla discesa del gradiente standard, poiché utilizza solo un sottoinsieme degli esempi di training.
* **Possibilità di uscire da minimi locali:** SGD può essere più efficace nel trovare il minimo globale della funzione di costo, poiché il rumore introdotto dal campionamento casuale può aiutare a evitare minimi locali.

**Problema della discesa del gradiente standard:**

Il problema principale della discesa del gradiente standard è che il calcolo del gradiente richiede di visitare tutti gli esempi di training, il che può essere molto lento per dataset di grandi dimensioni.

**Vantaggi di SGD:**

SGD risolve questo problema utilizzando un sottoinsieme casuale degli esempi di training per calcolare il gradiente ad ogni iterazione. Questo approccio riduce significativamente il tempo di calcolo, soprattutto per dataset di grandi dimensioni.


## Discesa del Gradiente Stocastica (SGD)

Quando si lavora con dataset di grandi dimensioni, la discesa del gradiente standard può diventare molto lenta. Per risolvere questo problema, si utilizza una variante chiamata **discesa del gradiente stocastica (SGD)**.

### Come funziona la SGD?

Invece di utilizzare il gradiente calcolato su tutto il dataset come direzione di discesa, la SGD utilizza un vettore il cui **valore atteso** coincide con il valore atteso del gradiente. Questo vettore direzione è una variabile casuale che punta in media nella direzione del gradiente "vero".

### Vantaggi e Svantaggi della SGD

Durante l'ottimizzazione, la SGD si muove in una direzione che, in ogni singola iterazione, potrebbe non essere quella di massima discesa. Questo significa che potremmo non muoverci esattamente nella direzione del gradiente, ma in modo casuale, e potremmo anche "salire" un pochino nella funzione di costo. Tuttavia, poiché il valore atteso del vettore direzione coincide con il gradiente, in media ci sposteremo verso il minimo.

### Come si calcola il vettore direzione?

Il vettore direzione nella SGD si ottiene calcolando il gradiente della funzione di costo in un **campione casuale** della popolazione. In pratica, questo significa selezionare casualmente una coppia (x,y) dal dataset e calcolare il gradiente in quel punto.

### Dimostrazione Matematica

La proprietà che il valore atteso del vettore direzione coincide con il gradiente può essere dimostrata matematicamente. Il valore atteso del gradiente della funzione di costo, immaginando che i valori siano campionati dalla distribuzione dei dati, può essere scritto come:

$$E[\nabla L(x,y)] = \nabla E[L(x,y)]$$

dove $L(x,y)$ è la funzione di costo e $E[\cdot]$ rappresenta il valore atteso.

L'errore di generalizzazione, che è quello che vogliamo minimizzare, è dato da $E[L(x,y)]$. Quindi, il valore atteso del nostro vettore direzione è proprio il gradiente dell'errore di generalizzazione.

### SGD nella Pratica

Nella pratica, non si usa direttamente la versione della SGD basata su un solo esempio, perché è abbastanza instabile. Si utilizza invece un algoritmo intermedio chiamato **mini-batch SGD**.

### Mini-batch SGD

La mini-batch SGD calcola il gradiente non su un solo esempio, ma su un campione di una certa dimensione, chiamato **batch**. Si seleziona casualmente un batch di elementi dal dataset e si calcola il gradiente come media dei gradienti di ogni elemento nel batch.

### Vantaggi della Mini-batch SGD

La mini-batch SGD rappresenta un compromesso tra la complessità computazionale e la precisione della discesa del gradiente standard. Offre una maggiore stabilità rispetto alla SGD standard e una minore complessità computazionale rispetto alla discesa del gradiente standard.

### Grafico Qualitativo

Un grafico qualitativo della SGD mostra che la traiettoria non è dritta verso il minimo, ma va a zig-zag. La discesa del gradiente standard, invece, ha una traiettoria dritta verso il minimo.


## Lezione di Machine Learning: Learning Rate e Loss Surrogata

### Learning Rate

Il **learning rate** è un parametro fondamentale negli algoritmi di apprendimento automatico, in particolare nella discesa del gradiente. Esso determina la dimensione del passo che l'algoritmo compie ad ogni iterazione nella direzione del minimo della funzione di costo.

In alcuni casi, il learning rate può essere **costante**, ovvero impostato all'inizio e mantenuto invariato durante l'addestramento. Tuttavia, un learning rate costante presenta alcuni svantaggi:

* **Convergenza:** Un learning rate troppo alto può impedire all'algoritmo di convergere al minimo globale, facendolo oscillare attorno ad esso.
* **Fase iniziale:** All'inizio dell'addestramento, quando l'algoritmo è lontano dalla soluzione ottimale, un learning rate elevato può essere vantaggioso per muoversi rapidamente nello spazio delle soluzioni.
* **Fase finale:** Man mano che l'algoritmo si avvicina al minimo, un learning rate elevato può causare instabilità e impedire la convergenza precisa.

Per ovviare a questi problemi, si preferisce un **learning rate variabile**, che si adatta alle diverse fasi dell'addestramento. Un metodo comune per la variazione del learning rate è l'"**annealing**", dove il learning rate viene diminuito gradualmente nel tempo.

Esistono diverse politiche di aggiornamento del learning rate, come le **epoche di addestramento**, dove il learning rate viene diminuito dopo un certo numero di iterazioni. In generale, il learning rate iniziale viene impostato come parametro e poi aggiornato nel tempo secondo una specifica politica.

### Discesa del Gradiente Stocastica

La **discesa del gradiente stocastica** (SGD) è una variante della discesa del gradiente che utilizza un sottoinsieme casuale dei dati, chiamato **batch**, per calcolare un'approssimazione del gradiente. Questo approccio è computazionalmente più efficiente rispetto al calcolo del gradiente su tutti i dati, soprattutto per dataset di grandi dimensioni.

La scelta del learning rate è cruciale anche nella discesa del gradiente stocastica, poiché un learning rate inappropriato può portare a convergenza lenta o instabilità.

### Loss Surrogata

La **loss surrogata** è una funzione di costo che viene utilizzata al posto della funzione di costo originale, quando quest'ultima è computazionalmente complessa o non convessa.

Ad esempio, nel caso dei semispazi e della **loss 0-1**, la funzione di costo originale è non convessa, il che rende difficile l'utilizzo della discesa del gradiente. In questo caso, si può utilizzare una loss surrogata, come la **loss logistica**, che è convessa e più facile da ottimizzare.

La loss surrogata deve essere scelta in modo da essere una buona approssimazione della funzione di costo originale e da permettere un'ottimizzazione efficiente.

### Riepilogo

* Il learning rate è un parametro fondamentale che determina la dimensione del passo nella discesa del gradiente.
* Un learning rate variabile è generalmente preferibile a un learning rate costante.
* La discesa del gradiente stocastica utilizza un sottoinsieme casuale dei dati per calcolare un'approssimazione del gradiente.
* La loss surrogata è una funzione di costo che viene utilizzata al posto della funzione di costo originale, quando quest'ultima è computazionalmente complessa o non convessa.


## Loss Surrogata nel Machine Learning

In alcuni problemi di Machine Learning, la funzione di loss originale può essere complessa e difficile da ottimizzare. Per risolvere questo problema, si può utilizzare una **loss surrogata**, che ha le seguenti proprietà:

1. **Upper bound**: la loss surrogata è un limite superiore della loss originale, ovvero il suo valore è sempre maggiore o uguale al valore della loss originale per ogni punto.
2. **Convessità**: la loss surrogata è una funzione convessa, permettendo l'utilizzo della discesa del gradiente e garantendo la convergenza al minimo globale.

### Esempio: Hinge Loss

Un esempio di loss surrogata per la loss 0-1 è la **hinge loss**. La hinge loss è definita come:

```
L(y, f(x)) = max(0, 1 - y * f(x))
```

dove:

* `y` è l'etichetta del punto dati
* `f(x)` è la predizione del modello
* `max(0, ...)` indica il massimo tra 0 e il valore tra parentesi

La hinge loss vale 0 per punti classificati correttamente con un margine maggiore di 1, e cresce linearmente per punti classificati errati o con un margine inferiore a 1.

**Proprietà della Hinge Loss:**

* **Upper bound**: la hinge loss è un upper bound della loss 0-1, poiché il suo valore è sempre maggiore o uguale al valore della loss 0-1 per ogni punto.
* **Convessità**: la hinge loss è una funzione convessa, quindi può essere utilizzata con la discesa del gradiente.

### Vantaggi dell'utilizzo di una Loss Surrogata

* **Ottimizzazione più semplice**: la loss surrogata permette di risolvere un problema di ottimizzazione più semplice e trattabile rispetto al problema originale.
* **Efficienza**: la convessità della loss surrogata consente l'utilizzo di algoritmi di ottimizzazione efficienti come la discesa del gradiente.

### Svantaggi dell'utilizzo di una Loss Surrogata

* **Perdita di accuratezza**: la soluzione trovata con la loss surrogata non sarà ottimale rispetto alla loss originale, ma sarà comunque una buona approssimazione.

### Applicazione ai Semispazi

Nel caso dei semispazi, la loss 0-1 porta ad un problema di ottimizzazione NP-hard, quindi intrattabile in pratica. Utilizzando la hinge loss come surrogata, possiamo invece risolvere un problema di ottimizzazione convesso, quindi trattabile in modo efficiente.

## Introduzione al problema e alla sua complessità

Nel contesto dell'apprendimento automatico (machine learning), ci troviamo spesso ad affrontare la questione della complessità dei problemi. Per comprendere meglio questa complessità, è necessario introdurre il concetto di **funzione di Lipschitz**.

## Funzioni di Lipschitz

Una funzione è detta di Lipschitz se la norma della differenza tra le immagini della funzione calcolata in due punti è limitata da una costante (detta costante di Lipschitz) moltiplicata per la distanza tra i due punti. In altre parole, la variazione della funzione è controllata dalla distanza tra i punti nel suo dominio.

**Formalmente:**

Una funzione $f: X \rightarrow Y$ è detta di Lipschitz se esiste una costante $L \geq 0$ tale che:

$$
||f(x_1) - f(x_2)|| \leq L ||x_1 - x_2||
$$

per ogni $x_1, x_2 \in X$.










---



# oooooooooooooooooooo
## Importanza delle funzioni di Lipschitz

La definizione di funzione di Lipschitz implica che il gradiente di una funzione di Lipschitz è limitato superiormente. Questo significa che la funzione non può avere un gradiente che cresce indefinitamente, ma deve essere limitato.

L'introduzione di questo concetto è fondamentale per caratterizzare i problemi di apprendimento automatico **ben posti**.

## Problemi ben posti

Un problema si dice **ben posto** se la sua soluzione esiste, è unica e dipende con continuità dai dati. Se valgono queste condizioni, si dimostra che il nostro problema è anche **apprendibile** (learnable).

Per dimostrare questa proprietà, è necessario introdurre una variante dell'algoritmo di minimizzazione del rischio empirico chiamata **Regolarizzazione di Tikhonov** (RML).

## Regolarizzazione di Tikhonov

La Regolarizzazione di Tikhonov è una regola di apprendimento che, data una classe di ipotesi e una funzione di perdita, seleziona l'ipotesi che minimizza il rischio empirico più un termine aggiuntivo detto **termine di regolarizzazione**.

Questo termine ha due funzioni principali:

1. **Stabilizzatore**: Se il dataset di training varia leggermente, ci aspettiamo che l'algoritmo restituisca un modello simile. Un algoritmo stabile è meno sensibile alle piccole variazioni nei dati.
2. **Misura della complessità del modello**: Il termine di regolarizzazione può essere interpretato come una misura della complessità del modello. Modelli più complessi tendono ad avere un termine di regolarizzazione più elevato.

## Esempio di regolarizzazione

Un esempio di regolarizzazione molto usata è la **regolarizzazione L2**, dove il termine di regolarizzazione è la norma al quadrato del vettore dei parametri del modello.

Si dimostra che i problemi di apprendimento automatico con regolarizzazione di Tikhonov sono **apprendibili**.

## Minimizzazione con regolarizzazione

Avendo introdotto la regolarizzazione, dobbiamo rivedere il nostro algoritmo di discesa del gradiente per gestire questa nuova regola. Invece di calcolare il gradiente solo della funzione di perdita, dobbiamo calcolare il gradiente della funzione di perdita **più** il termine di regolarizzazione.




### Funzioni Fortemente Convesse

Nel caso di funzioni **fortemente convesse**, si ha la garanzia che la funzione stia al di sotto di una parabola. Questo significa che la funzione non solo è limitata superiormente, ma la sua crescita è anche controllata.



### Discesa del Gradiente con Regolarizzazione

La regola di aggiornamento per la discesa del gradiente con regolarizzazione è data da:

```
thetaV = thetaV - alpha * (nablaJ(thetaV) + lambda * thetaV)
```

Dove:

* `thetaV` è il vettore dei parametri.
* `alpha` è la dimensione del passo.
* `nablaJ(thetaV)` è il gradiente della funzione di costo.
* `lambda` è il parametro di regolarizzazione.

### Epoche e Iterazioni

* **Iterazione:** Un'iterazione è un singolo passo dell'algoritmo di discesa del gradiente.
* **Epoca:** Un'epoca è un ciclo completo sul set di training.

Il numero di iterazioni necessarie per completare un'epoca dipende dalla dimensione del batch. Ad esempio, se il batch è un millesimo del dataset, un'epoca corrisponde a 1000 iterazioni.

### Regressione Lineare con Discesa del Gradiente

La funzione di costo per la regressione lineare è data da:

```
J(theta) = 1/2m * sum(h(x^(i)) - y^(i))^2
```

Dove:

* `h(x^(i))` è la predizione del modello per l'esempio `i`.
* `y^(i)` è il valore reale per l'esempio `i`.
* `m` è il numero di esempi nel set di training.

La regola di aggiornamento per la discesa del gradiente per la regressione lineare è data da:

```
theta = theta - alpha * (1/m * sum(h(x^(i)) - y^(i)) * x^(i))
```

### Momentum

Il momentum è una tecnica che aiuta ad accelerare la discesa del gradiente. La regola di aggiornamento con momentum è data da:

```
v = beta * v - alpha * nablaJ(theta)
theta = theta + v
```

Dove:

* `v` è il vettore di momentum.
* `beta` è il parametro di momentum.

### Conclusione

In questa lezione abbiamo introdotto la discesa del gradiente e il momentum, due tecniche fondamentali per l'ottimizzazione dei modelli di apprendimento automatico. Abbiamo anche discusso il concetto di epoche e iterazioni, che sono importanti per comprendere il processo di training.


## Ottimizzazione del Gradiente e Metodo delle Variabili


Se il costo non decresce, la funzione di costo in definizione è completamente convessa. Questo nuovo metodo variabile è un'ottimizzazione di... Non è un punto di minimo, ma non è un punto di minimo. Per semplificare la comprensione, scriviamo la formula in modo leggermente diverso.

### Scomposizione del Termine

Scomponiamo il termine per ottenere la derivata di un... Questa è la derivata di un... Adesso, mettiamo in maniera ricorsiva questa espressione.

### Regola della Catena e Sostituzione

Utilizzando la regola della catena, possiamo sostituire dentro i due gradienti. Arriviamo a questa situazione, dove abbiamo la somma di due vettori.

### Iterazione e Abbassamento del Grado

Continuando in questo modo, il grado si abbassa di uno e entra un altro termine. Alla fine delle sostituzioni, avremmo -1 sull'indice.

### Soluzione Iterativa

La soluzione alla iterazione t+1 non è la somma di tutti i gradienti, ma è la loro direzione. La direzione è la somma dei gradienti.

### Versione Semplificata

Sfruttando questa proprietà, possiamo scrivere una versione semplificata. Chiamiamo la derivata di questa funzione...

### Sostituzione e Aggiornamento

Adesso facciamo la sostituzione. Il delta theta è semplicemente questo termine, quindi il Vt+1 lo otteniamo moltiplicando per questo. Per questo dobbiamo fare gli stessi passaggi. Selezioniamo... e poi aggiorniamo theta di t+1 della sommatoria...






