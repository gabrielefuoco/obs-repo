

La teoria della probabilità fornisce una base matematica per descrivere e analizzare l'incertezza. I concetti fondamentali possono essere illustrati come segue:

#### Variabile Casuale e Frequenza Relativa

Consideriamo una variabile casuale \( X \), che può assumere valori discreti dall'insieme \( \{x_1, ..., x_k\} \). Quando osserviamo la variabile \( X \) più volte, possiamo calcolare la frequenza relativa con cui ciascun valore si verifica.


- **Esempio**: Supponiamo di lanciare un dado (un evento con sei possibili esiti: \( \{1, 2, 3, 4, 5, 6\} \)). Se lanciamo il dado 60 volte e otteniamo il numero 1 dieci volte, la frequenza relativa per il numero 1 è  $$( \frac{10}{60} = \frac{1}{6} )$$


#### Probabilità di un Evento

La probabilità di un evento, ad esempio \( P(X = x_i) \), misura quanto è probabile che si verifichi l'evento \( X = x_i \).

- **Esempio**: Continuando con il dado, la probabilità di ottenere un 1 in un singolo lancio è \( \frac{1}{6} \) (considerando un dado equo).

#### Visione Tradizionale e Bayesiana della Probabilità

La visione tradizionale della probabilità si basa sulla frequenza relativa degli eventi (frequentista), mentre la visione bayesiana usa una prospettiva più flessibile, incorporando informazioni a priori.

- **Esempio Frequentista**: La probabilità di ottenere testa in un lancio di moneta è stimata come il numero di teste diviso per il numero totale di lanci.
- **Esempio Bayesiano**: Supponiamo di avere un'informazione a priori che una moneta è leggermente truccata. In questo caso, possiamo incorporare questa informazione a priori nel calcolo della probabilità.

#### Variabili Casuali
Contengono la probabilità di ciascun risultato

- **Esempio**: In un gioco di carte, l'esito della carta pescata è una variabile casuale che può assumere uno dei 52 valori possibili (per un mazzo standard).

#### Probabilità Congiunta

Consideriamo due variabili casuali \( X \) e \( Y \), ciascuna delle quali può assumere \( K \) valori discreti. La probabilità congiunta \( P(X = x_i, Y = y_j) \) rappresenta la probabilità che \( X \) assuma il valore \( x_i \) e contemporaneamente \( Y \) assuma il valore \( y_j \).

- **Esempio**: Supponiamo di avere un dataset di persone con attributi \( X \) (età) e \( Y \) (stipendio). La probabilità congiunta \( P(X = 30, Y = 50000) \) rappresenta la probabilità che una persona abbia 30 anni e guadagni 50.000 euro.

#### Marginalizzazione

La somma delle probabilità congiunte rispetto a una delle variabili casuali è detta marginalizzazione. Per ottenere la probabilità marginale di \( X \), sommiamo le probabilità congiunte su tutti i possibili valori di \( Y \):

$$[ P(X = x_i) = \sum_{j=1}^k P(X = x_i, Y = y_j) ]$$

- **Esempio**: Se vogliamo trovare la probabilità marginale che una persona abbia 30 anni (indipendentemente dallo stipendio), sommiamo tutte le probabilità congiunte per ogni possibile valore dello stipendio.

### Esempio Completo

Consideriamo un esempio con due variabili casuali \( X \) (colore dei capelli: biondo, castano, nero) e \( Y \) (occhi: blu, marroni, verdi). Supponiamo di avere i seguenti dati osservati in una classe di 100 studenti:

| Colore dei Capelli / Occhi | Blu | Marroni | Verdi | Totale |
|----------------------------|-----|---------|-------|--------|
| Biondo                     | 10  | 20      | 5     | 35     |
| Castano                    | 15  | 25      | 5     | 45     |
| Nero                       | 5   | 10      | 5     | 20     |
| Totale                     | 30  | 55      | 15    | 100    |

1. **Probabilità Congiunta**: La probabilità congiunta che un studente abbia i capelli biondi e gli occhi blu è:

$$[ P(X = \text{biondo}, Y = \text{blu}) = \frac{10}{100} = 0.1 ]$$

2. **Probabilità Marginale**: La probabilità marginale che un studente abbia i capelli biondi (indipendentemente dal colore degli occhi) è:

$$[ P(X = \text{biondo}) = \frac{35}{100} = 0.35 ]$$

3. **Probabilità Marginale degli Occhi Blu**:

$$[ P(Y = \text{blu}) = \frac{30}{100} = 0.3 ]$$

In sintesi, queste nozioni e calcoli fondamentali della teoria della probabilità aiutano a comprendere come i dati possono essere analizzati e interpretati, specialmente quando si tratta di eventi incerti o variabili casuali.