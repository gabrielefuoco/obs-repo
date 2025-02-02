**Regressione Lineare e Discesa del Gradiente**

I. **Modello di Regressione:**

   * A. Scopo: Studiare la relazione tra una variabile dipendente (*y*) e un insieme di variabili indipendenti ($X_1, X_2, ..., X_k$).
   * B. Equazione generale:  $Y = f(X_1, X_2, ..., X_k) + b$, dove *b* rappresenta gli errori non modellati.
   * C. Regressione Lineare: Caso specifico con equazione di primo grado: $y = wx + b$.

II. **Stima dei Parametri:**

   * A. Funzione di Costo: $J(w, b) = \frac{1}{2m} \sum_{i=1}^m (f_{wb}(x^{(i)}) - y^{(i)})^2$, misura la somma degli scarti quadratici tra valori osservati e teorici.
   * B. Obiettivo: Minimizzare $J(w, b)$ per trovare i migliori valori di *w* e *b*.

III. **Algoritmo di Discesa del Gradiente:**

   * A. **Inizializzazione**: Assegnare valori iniziali arbitrari a *w* e *b*.
   * B. **Aggiornamento dei parametri** (iterativo):
      * 1. Caso semplice (1 attributo):
         * $w' = w - \alpha \frac{1}{m} \sum_{i=1}^m (f_{wb}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$
         * $b' = b - \alpha \frac{1}{m} \sum_{i=1}^m (f_{wb}(x^{(i)}) - y^{(i)})$
      * 2. Caso complesso (n attributi): $y^{(i)} = w_1x_1^{(i)} + ... + w_nx_n^{(i)}$
         * $w_j' = w_j - \alpha \frac{\partial}{\partial w_j} J(w_1, ..., w_n, b)$  per j = 1, ..., n
         * $b' = b - \alpha \frac{\partial}{\partial b} J(w_1, ..., w_n, b)$
   * C. **Iterazione**: Ripetere l'aggiornamento dei parametri fino a convergenza (minimo di $J(w, b)$).
   * D. **Learning Rate (α):** Controlla l'ampiezza dell'aggiornamento ad ogni iterazione.
   * E. **Convergenza**: Dipende dalla convessità di $J(w, b)$ (minimo globale vs. minimo locale).
   * F. Requisito: $J(w, b)$ deve essere derivabile.


I. **Feature Scaling (Mean Normalization)**
* **Scopo:** Omogeneizzare la scala delle variabili.
* **Metodo:**  $x_{i}'=\frac{(x-\mu_{i})}{\sigma_{i}}$  
* **Risultato:** Media 0 e deviazione standard 1 per ogni attributo.

II. **Regressione Logistica**
* **Scopo:** Classificazione binaria.
* **Principio:**  Utilizzo della funzione logistica per stimare la probabilità di appartenenza a una classe.
* **Funzione Sigmoide (o Logistica):**
	* Intervallo di output: [0, 1] (probabilità).
	* Formula: $P(y=1|x)=\frac{1}{1+e^{-z}}=\sigma(z)$ (dove z è il predittore lineare).
* **Regressione Logistica:**
	* Predittore lineare: $z=wTx+b$
	* Formula: $G_{w,b}=\frac{1}{1+e^{-wx-b}}$
* **Funzione di Costo:**
	* Necessità di una funzione convessa (per la discesa del gradiente).
	* Formula: $J(w,b)=\frac{1}{2m}\sum_{i=1}^m(wx^{(i)}+b-y^{(i)})$ (dove m è il numero di istanze).


III. **Aggiornamento Parametri (Discesa del Gradiente)**
* **Formule di Aggiornamento:**
	* $w' = w - \alpha\frac{\partial J(w,b)}{\partial w}$
	* $b' = b - \alpha\frac{\partial J(w,b)}{\partial b}$ (dove α è il learning rate)
* **Derivate Parziali:**
	* **Rispetto a w:** $\frac{\partial G_{w,b}(x)}{\partial w} = (G_{w,b}(x) - G_{w,b}(x)^2) \cdot x$
	* **Rispetto a b:** $\frac{\partial G_{w,b}(x)}{\partial b} = (G_{w,b}(x) - G_{w,b}(x)^2)$
	* **Derivate del logaritmo di G(x):**
		* $\frac{\partial \log G_{w,b}(x)}{\partial w} = (1 - G_{w,b}(x))x$
		* $\frac{\partial \log G_{w,b}(x)}{\partial b} = (1 - G_{w,b}(x))$
	* **Derivate del logaritmo di (1 - G(x)):**
		* $\frac{\partial \log(1 - G_{w,b}(x))}{\partial w} = -G_{w,b}(x) \cdot x$
		* $\frac{\partial \log(1 - G_{w,b}(x))}{\partial b} = -G_{w,b}(x)$



**Regressione Logistica: Schema Riassuntivo**

I. **Caratteristiche del Modello:**

* A. Intercetta assente: il punto centrale è nell'origine (b=0).
* B. Comportamento decrescente per y=1:  la funzione logistica $\sigma(wTx)$ diventa decrescente a causa del segno meno.
* C. Funzione di costo convessa: permette l'utilizzo della discesa del gradiente.

II. **Calcolo delle Derivate:**

* A. Derivata rispetto a w:
	$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m}\sum_{i=1}^m (G_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$
* B. Derivata rispetto a b:
	$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m}\sum_{i=1}^m (G_{w,b}(x^{(i)}) - y^{(i)})$$
* C. Osservazioni:  simili alla regressione lineare, ma con funzione logistica al posto di quella lineare; processo di aggiornamento dei parametri invariato.

III. **Aggiornamento dei Parametri (Regressione Logistica Multivariata):**

$$ \begin{aligned} w'_1 = w_1 - \alpha\frac{\partial J(w_1,...,w_n,b)}{\partial w_1} \\ ... \\ w'_n = w_n - \alpha\frac{\partial J(w_1,...,w_n,b)}{\partial w_n}\\ b = b - \alpha\frac{\partial J(w_1,...,w_n,b)}{\partial b} \end{aligned} $$

IV. **Funzione Logistica Multivariata:**

* A. Formula: $$G_{w_1,...,w_n,b}(\bar{x}) = \frac{1}{1 + e^{-\bar{w}\bar{x}-b}}$$


