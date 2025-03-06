
La regressione è una tecnica statistica utilizzata per studiare la relazione tra una variabile dipendente *y* e un insieme di variabili indipendenti o regressori $(X_1, X_2, ..., X_k)$. I modelli di regressione permettono di stabilire matematicamente come *y* è spiegata dalle variabili $X_i$ tramite un'equazione:
$$Y = f(X_1, X_2, ..., X_k) + b$$
dove *b* rappresenta gli effetti e le cause non considerate nel modello.

Nel caso della regressione lineare, il modello è rappresentato da un'equazione di primo grado:
$$y = wx + b$$
Il problema chiave è stimare i parametri *w* e *b* in modo che il modello si adatti al meglio ai dati osservati.

### Stima dei Parametri

La stima dei parametri *w* e *b* del modello di regressione lineare $y = wx + b$ viene formulata come un problema di minimizzazione della funzione di costo $J(w, b)$. La funzione di costo misura la somma degli scarti quadratici tra i valori osservati $y^{(i)}$ e i valori teorici $wx^{(i)} + b$, divisa per $2m$:
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (f_{wb}(x^{(i)}) - y^{(i)})^2$$
## Algoritmo di Discesa del Gradiente

Per minimizzare $J(w, b)$, si utilizza l'algoritmo di discesa del gradiente. I passi sono:

- **Inizializzazione:** Inizializzare *w* e *b* con valori arbitrari.

- **Aggiornamento dei parametri:** Calcolare i nuovi parametri $w'$ e $b'$ che riducono $J(w, b)$, usando le formule delle derivate parziali.

##### Caso semplice (1 attributo, 1 variabile dipendente):

$$w' = w - \alpha \frac{\partial}{\partial w} J(w, b) = w - \alpha \frac{1}{m} \sum_{i=1}^m (f_{wb}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$
$$b' = b - \alpha \frac{\partial}{\partial b} J(w, b) = b - \alpha \frac{1}{m} \sum_{i=1}^m (f_{wb}(x^{(i)}) - y^{(i)})$$

##### Caso più complesso (n attributi, 1 variabile dipendente):

##### Modello di regressione lineare a più valori:

$$y^{(i)} = w_1x_1^{(i)} + ... + w_nx_n^{(i)}$$
##### Aggiornamento dei parametri:

$$
\begin{aligned}
w_1' = w_1 - \alpha \frac{\partial}{\partial w_1} J(w_1, b) \\
\dots
\\
w_n' = w_n - \alpha \frac{\partial}{\partial w_n} J(w_n, b)
\\
b' = b - \alpha \frac{\partial}{\partial b} J(w, b)
\end{aligned}
$$

dove:

* $α$ è il *learning rate*, che determina l'ampiezza dello spostamento dei parametri ad ogni iterazione.

- **Iterazione:** Ripetere il passo 2 fino a raggiungere il minimo di $J(w, b)$.

##### Osservazioni:

* Le formule per aggiornare $w'$ e $b'$ dipendono dal numero di variabili indipendenti.
* Il *learning rate* $α$ controlla l'ampiezza degli aggiornamenti.
* Se $J(w, b)$ è convessa, l'algoritmo converge al minimo globale. Altrimenti, potrebbe convergere a un minimo locale.
* L'algoritmo richiede che $J(w, b)$ sia derivabile.

## Feature scaling: mean normalization

**Scopo:** Rendere omogenee le variabili di un dataset in termini di scala di grandezza.

**Motivazione:** I dataset spesso contengono variabili con ordini di grandezza molto diversi, il che può rendere più complesse le analisi.

##### Metodo:

* Per ogni valore $x$ di un attributo $i$, si sottrae la media $\mu_{i}$ di quell'attributo e si divide per la deviazione standard $\sigma_{i}$:
$$x_{i}'=\frac{(x-\mu_{i})}{\sigma_{i}}$$
**Risultato:** Tutti i valori di un attributo vengono trasformati in una scala con media 0 e deviazione standard 1.

## Funzione logistica

**Regressione logistica:** Algoritmo di apprendimento automatico per problemi di classificazione binaria.

**Scopo:** Predire se un'istanza appartiene a una delle due classi possibili.

**Principio:** Sfrutta una funzione logistica (o sigmoidale) per stimare la probabilità che un'istanza appartenga a una determinata classe.

##### Caratteristiche:

* Modella la relazione tra una variabile dipendente binaria e una o più variabili indipendenti.
* La funzione logistica mappa i valori in input nell'intervallo [0,1], interpretabile come probabilità di appartenenza alla classe positiva.
* È in grado di catturare relazioni non lineari tra le variabili indipendenti e la variabile di risposta.
* Fornisce previsioni più accurate in presenza di relazioni complesse grazie alla funzione non lineare.

##### Funzione sigmoide:

* Trasforma valori lineari in valori nel range [0,1].
* Formula: $P(y=1|x)=\frac{1}{1+e^{-z}}=\sigma(z)$

##### Regressione logistica:

* Utilizza un predittore lineare $z=wTx+b$, che viene passato alla funzione logistica per ottenere la probabilità di appartenenza alla classe positiva.
* Formula: $G_{w,b}=\frac{1}{1+e^{-wx-b}}$

##### Funzione di costo:

* La funzione di costo lineare quadratica non è convessa se sostituita con σ(z).
* Si utilizza una nuova funzione di costo convessa per permettere l'applicazione della discesa del gradiente durante l'addestramento.
* Formula: $J(w,b)=\frac{1}{2m}\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})$

## Calcolo delle derivate per l'aggiornamento dei parametri

##### Formule di aggiornamento dei parametri:

$$w' = w - \alpha\frac{\partial J(w,b)}{\partial w}$$
$$b' = b - \alpha\frac{\partial J(w,b)}{\partial b}$$
### Calcolo delle derivate parziali:

Supponiamo di voler calcolare le seguenti derivate parziali che ci serviranno per l'aggiornamento dei parametri:

##### Derivata rispetto a w:

$$\frac{\partial G_{w,b}(x)}{\partial w} = \frac{\partial}{\partial w}\frac{1}{1 + e^{-wx-b}} = \frac{-e^{-wx-b}(-x)}{(1 + e^{-wx-b})^2} = \frac{(1 + e^{-wx-b} - 1)x}{(1 + e^{-wx-b})^2}$$
$$= \frac{x}{1 + e^{-wx-b}} - \frac{x}{(1 + e^{-wx-b})^2} = (G_{w,b}(x) - G_{w,b}(x)^2) \cdot x$$

##### Derivata rispetto a b:

$$\frac{\partial G_{w,b}(x)}{\partial b} = (G_{w,b}(x) - G_{w,b}(x)^2)$$

##### Derivate del logaritmo di G(x):

$$\frac{\partial \log G_{w,b}(x)}{\partial w} = \frac{G_{w,b}(x)(1 - G_{w,b}(x))x}{G_{w,b}(x)} = (1 - G_{w,b}(x))x$$
$$\frac{\partial \log G_{w,b}(x)}{\partial b} = (1 - G_{w,b}(x))$$

##### Derivate del logaritmo di (1 - G(x)):

$$\frac{\partial \log(1 - G_{w,b}(x))}{\partial w} = \frac{-G_{w,b}(x)(1 - G_{w,b}(x))x}{(1 - G_{w,b}(x))} = -G_{w,b}(x) \cdot x$$
$$\frac{\partial \log(1 - G_{w,b}(x))}{\partial b} = -G_{w,b}(x)$$
* **m:** rappresenta il numero di istanze.
* **b:** il termine b (intercetta) è assente, quindi il punto centrale si trova nell'origine.
* **y=1:** quando y=1, la funzione logistica $\sigma(wTx)$ diventa decrescente a causa del segno meno.
* **Funzione di costo:** la nuova funzione di costo è convessa, permettendo l'applicazione della discesa del gradiente.
### Calcolo delle derivate della funzione di costo

Vengono calcolate le derivate della funzione di costo, necessarie per l'aggiornamento dei pesi **w** e dell'intercetta **b** durante l'addestramento:
##### Derivata rispetto a w:

$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m}\sum_{i=1}^m y^{(i)}(1 - G_{w,b}(x^{(i)}))x^{(i)} + (1-y^{(i)})(-G_{w,b}(x^{(i)}))x^{(i)}$$
$$= \frac{1}{m}\sum_{i=1}^m [y^{(i)} - y^{(i)}G_{w,b}(x^{(i)}) - G_{w,b}(x^{(i)}) + y^{(i)}G_{w,b}(x^{(i)})] \cdot x^{(i)}$$
$$= \frac{1}{m}\sum_{i=1}^m (G_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$
##### Derivata rispetto a b:

$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m}\sum_{i=1}^m (G_{w,b}(x^{(i)}) - y^{(i)})$$

##### Osservazioni:

* Le equazioni ottenute sono simili a quelle della regressione lineare.
* L'unica differenza è che al posto della funzione lineare abbiamo la funzione logistica.
* Il processo di aggiornamento dei parametri rimane lo stesso.
* Se la nostra equazione contiene più parametri, dovremmo ovviamente aggiornarli tutti.

### Aggiornamento parametri con regressione logistica multivariata

$$
\begin{aligned}
w'_1 = w_1 - \alpha\frac{\partial J(w_1,...,w_n,b)}{\partial w_1} \\
... \\
w'_n = w_n - \alpha\frac{\partial J(w_1,...,w_n,b)}{\partial w_n}\\
b = b - \alpha\frac{\partial J(w_1,...,w_n,b)}{\partial b}
\end{aligned}
$$

### Funzione logistica multivariata:

La funzione logistica in cui **x** è un vettore sarà:
$$G_{w_1,...,w_n,b}(\bar{x}) = \frac{1}{1 + e^{-w_1x_1-...-w_nx_n-b}} = \frac{1}{1 + e^{-\bar{w}\bar{x}-b}}$$

* **w<sub>1</sub>, ..., w<sub>n</sub>** rappresentano i pesi per ogni feature del vettore **x**.
* **b** rappresenta l'intercetta.
* **α** è il learning rate.
* **J(w<sub>1</sub>, ..., w<sub>n</sub>, b)** è la funzione di costo.
* **x<sub>1</sub>, ..., x<sub>n</sub>** sono le componenti del vettore **x**.
* **¯w** è il vettore dei pesi.
* **¯x** è il vettore delle features.

