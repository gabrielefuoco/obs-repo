### Teorema di Bayes e Classificatore Naïve Bayes

Il **Teorema di Bayes** fornisce un metodo per calcolare la probabilità a posteriori, che può essere utilizzata per la classificazione in vari campi, tra cui il machine learning. Nel contesto della classificazione, l'obiettivo è prevedere la classe \( Y \) di un record dato un insieme di attributi \( X_1, X_2, X_3, ..., X_d \).

#### Obiettivo della Classificazione

Dato un record con attributi \( X_1, X_2, X_3, ..., X_d \), vogliamo trovare la classe \( Y \) che massimizza la probabilità a posteriori:

$$
P(Y \mid X_1, X_2, X_3, \ldots, X_d)
$$

Questa è nota come **posterior probability**.

#### Utilizzo del Teorema di Bayes

Il Teorema di Bayes ci permette di calcolare questa probabilità a posteriori:

$$
P(Y \mid X_1, X_2, X_3, \ldots, X_d) = \frac{P(X_1, X_2, X_3, \ldots, X_d \mid Y) \cdot P(Y)}{P(X_1, X_2, X_3, \ldots, X_d)}
$$

#### Massimizzazione della Probabilità

Scegliere il valore di \( Y \) che massimizza \( P(Y | X_1, X_2, X_3, \..., X_d) \) è equivalente a scegliere il valore di \( Y \) che massimizza il prodotto:

$$
P(X_1, \ldots, X_d \mid Y) \cdot P(Y)
$$

Questo perché \( P(X_1, ..., X_d) \) è una costante per tutti i valori di \( Y \) e quindi non influenza la massimizzazione.

### Componenti del Teorema di Bayes per la Classificazione

1. **Probabilità a Priori \( P(Y) \)**:
   - ==Rappresenta le convinzioni preliminari sulla distribuzione delle etichette di classe, indipendentemente dai valori degli attributi osservati.==
   - Può essere stimata dal training set come la frazione di record appartenenti a ciascuna classe.
   - Ad esempio, la probabilità che una persona abbia una malattia cardiaca indipendentemente dai rapporti diagnostici.

2. **Probabilità Condizionale di Classe \( P(X_1, ..., X_d | Y) \)**:
   - Misura la probabilità di osservare gli attributi \( X_1, X_2, ..., X_d \) dato che la classe è \( Y \).
   - Cattura il meccanismo sottostante alla generazione dei valori degli attributi se il record appartiene davvero alla classe \( Y \).

3. **Probabilità dell'Evidenza \( P(X_1, ...., X_d) \)**:
   - Questo termine non dipende dall'etichetta di classe e può essere trattato come una costante di normalizzazione.

### Problema della Stima di \( P(X_1, .., X_d \| Y) \)

Calcolare \( P(X_1, ..., X_d | Y) \) direttamente è computazionalmente proibitivo, specialmente con un grande numero di attributi \( d \), a causa dell'incremento esponenziale delle possibili combinazioni di valori degli attributi.

#### Soluzione con il Classificatore Naïve Bayes

Il **Classificatore Naïve Bayes** risolve questo problema assumendo l'indipendenza condizionale tra gli attributi dati la classe \( Y \):

$$
P(X_1, X_2, \ldots, X_d \mid Y) = P(X_1 \mid Y) \cdot P(X_2 \mid Y) \cdot \ldots \cdot P(X_d \mid Y)
$$

Questa assunzione semplifica il calcolo delle probabilità condizionate di classe, rendendo il classificatore computazionalmente efficiente e adatto a grandi dataset.

### Riassunto

- **Teorema di Bayes**: Usato per calcolare la probabilità a posteriori e utilizzato nella classificazione.
- **Classificatore Naïve Bayes**: Semplifica il calcolo delle probabilità condizionate assumendo l'indipendenza condizionale tra gli attributi.
- **Probabilità a Priori**: Stima preliminare delle etichette di classe.
- **Probabilità Condizionale di Classe**: Misura la probabilità degli attributi dato che la classe è nota.
- **Probabilità dell'Evidenza**: Termini di normalizzazione che non dipendono dalla classe.
