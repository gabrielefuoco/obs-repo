
La probabilità condizionata misura la probabilità di un evento dato che un altro evento è già avvenuto. Questo concetto è fondamentale nella teoria della probabilità e viene utilizzato per aggiornare le probabilità in presenza di nuove informazioni.

#### Definizione di Probabilità Condizionata

Sia \( P(Y | X) \) la probabilità condizionata di osservare la variabile casuale \( Y \) dato che la variabile casuale \( X \) assume un particolare valore. Si legge come "la probabilità di \( Y \) dato \( X \)".

$$
[ P(Y \mid X) = \frac{P(X, Y)}{P(X)} ]
$$

Questo significa che la probabilità di \( Y \) dato \( X \) è il rapporto tra la probabilità congiunta di \( X \) e \( Y \) e la probabilità di \( X \).

#### Relazioni tra Probabilità Condizionate e Congiunte

La probabilità congiunta di \( X \) e \( Y \) può essere espressa in termini di probabilità condizionata in due modi:

$$
[ P(X, Y) = P(Y \mid X) \cdot P(X) ]
$$
$$
[ P(X, Y) = P(X \mid Y) \cdot P(Y) ]

$$
Queste relazioni mostrano che la probabilità congiunta può essere decomposta in termini di probabilità condizionata e probabilità marginale.

#### Teorema di Bayes

Il Teorema di Bayes fornisce una formula per aggiornare le probabilità condizionate quando si ottengono nuove informazioni. È espresso come:

$$[ P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)} ]$$

Questa formula permette di calcolare $$( P(Y \mid X) )$$ usando $$( P(X \mid Y) ), ( P(Y) ),  ( P(X) )$$

### Esempi Pratici

#### Esempio 1: Diagnosi Medica

Supponiamo di voler calcolare la probabilità che un paziente abbia una malattia \( D \) dato che il test \( T \) risulta positivo. Siano:

- \( P(D) \): la probabilità a priori che un paziente abbia la malattia.
- \( P(T | D) \): la probabilità che il test sia positivo dato che il paziente ha la malattia (sensibilità del test).
- \( P(T) \): la probabilità che il test sia positivo.

Applicando il Teorema di Bayes:

$$[ P(D \mid T) = \frac{P(T \mid D) \cdot P(D)}{P(T)} ]$$

#### Esempio 2: Gioco d'Azzardo

Supponiamo di avere un mazzo di carte e vogliamo calcolare la probabilità che una carta pescata sia un re (\( K \)) dato che sappiamo che è una figura (\( F \)):

- $$( P(K) = \frac{4}{52} = \frac{1}{13} )$$
- $$( P(F) = \frac{12}{52} = \frac{3}{13} )$$
- $$( P(F \mid K) = 1 )$$ (se è un re, è sicuramente una figura)

Applicando il Teorema di Bayes:

$$[ P(K \mid F) = \frac{P(F \mid K) \cdot P(K)}{P(F)} = \frac{1 \cdot \frac{1}{13}}{\frac{3}{13}} = \frac{1}{3} ]$$

### Riassunto

- **Probabilità Condizionata**: Misura la probabilità di un evento dato che un altro evento è già avvenuto.
- **Relazione con Probabilità Congiunta**: La probabilità congiunta può essere espressa in termini di probabilità condizionata e marginale.
- **Teorema di Bayes**: Fornisce un modo per aggiornare le probabilità condizionate in presenza di nuove informazioni, essenziale in molte applicazioni pratiche.

Questi concetti sono fondamentali per la comprensione e l'applicazione della teoria della probabilità in vari campi, come la statistica, il machine learning e la scienza dei dati.