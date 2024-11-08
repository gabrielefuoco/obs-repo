## Predittori Lineari

Il modello di regressione lineare si basa sull'equazione:

$$x \cdot w = y$$

L'obiettivo è trovare i pesi $w$ che meglio approssimano la relazione lineare tra le variabili indipendenti $x$ e la variabile dipendente $y$.

### Soluzioni per la regressione lineare

#### Prima soluzione: Implementazione tramite scikit-learn

* Utilizzare il metodo `fit` per addestrare il modello.
* Se si utilizzano coordinate omogenee, i pesi $w$ includono anche il bias $b$.

#### Seconda soluzione: Soluzione in forma chiusa

* L'equazione di base è: $y = wx$.
* La soluzione per i pesi è: $w = \frac{y}{x}$.

**Formula:**

* $(x^T x) \cdot w = x^T y$
* $w = (x^T x)^{-1} x^T y$

**Passaggi:**

1. **Coordinate omogenee:** Prima di applicare la formula, è necessario passare alle coordinate omogenee. Questo può essere fatto utilizzando il metodo `.add_dummy_feature`, che aggiunge una colonna di 1 alla matrice $x$. Tuttavia, questo metodo aggiunge la colonna in coda, quindi è necessario invertirla.
2. **Pseudo inversa:** Non tutte le matrici sono invertibili. Per calcolare l'inversa di una matrice non invertibile, si utilizza la pseudo inversa, calcolata tramite il metodo `pinv`.
3. **Calcolo dei parametri:** Calcolare $x_1^T x_1$, calcolare la sua pseudo inversa e moltiplicare per $x^T y$. Questo restituisce i parametri $w$.

### Soluzione 3: Caso multidimensionale

Nel caso multidimensionale, l'equazione lineare è rappresentata da:

$$Aw = b$$

La soluzione si ottiene risolvendo l'equazione lineare **con il metodo della pseudo inversa**, ovvero calcolando:

$$w = (A^T A)^{-1} A^T b$$

### Soluzione 4: Regressione polinomiale

La regressione polinomiale può essere utilizzata per implementare un regressore lineare. Utilizzando il metodo `.polyfit` con un grado pari a 1, si ottiene un modello lineare.

* Si ottiene un parametro per ogni feature, più l'intersezione con l'asse delle ordinate.

### Valutazione del modello

Per valutare la performance del modello di regressione lineare, si possono utilizzare diverse metriche.

* **Errore quadratico medio (MSE):**  Questa metrica misura la differenza quadratica media tra i valori previsti e i valori reali. Non è una misura definita in un range specifico, quindi non è possibile stabilire se un valore sia alto o basso.
* **R-quadrato (R²):**  Questa metrica misura la proporzione della varianza dei dati spiegata dal modello. I valori di R² sono compresi tra 0 e 1, dove 1 indica che il modello spiega completamente la varianza dei dati.

La formula per calcolare R² è:

$$R^2 = 1 - \frac{RSS}{TSS}$$

dove:

* **RSS (Somma dei quadrati residui):**  $\sum_{i}(y_{i}-\hat{y_{i}})^2$
* **TSS (Somma dei quadrati totali):**  $\sum_{i}(y_{i}-\bar{y})^2$

Se il denominatore (TSS) tende a 0, allora R² sarà pari a 1 (indica un modello che spiega bene la varianza dei dati).


### Regressione polinomiale

Possiamo usare un regressore lineare con più feature, dove ogni feature rappresenta un polinomio di ordine superiore. Prima di addestrare il modello, possiamo aggiungere una feature che rappresenta il termine di secondo grado. 

Avendo un polinomio di grado *n*, possiamo interpolare *n+1* punti. Interpolare troppo i dati del training set causa *overfitting*. 
