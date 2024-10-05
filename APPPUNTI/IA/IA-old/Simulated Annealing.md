Il **Simulated Annealing (SA)** è un algoritmo di ottimizzazione ispirato al processo fisico di ricottura (annealing), che consiste nel riscaldare e poi raffreddare lentamente un materiale per ottenere una configurazione stabile, solitamente a bassa energia. In informatica e intelligenza artificiale, il Simulated Annealing è utilizzato per trovare soluzioni approssimate a problemi di ottimizzazione combinatoria, dove lo spazio delle soluzioni è molto ampio e complesso.

### Principio Base del Simulated Annealing

Il Simulated Annealing si basa su un'analogia con il processo fisico di ricottura:

1. **Energia e Soluzione:** Nell'algoritmo, ogni soluzione del problema è paragonata a uno stato del materiale, e la "qualità" di quella soluzione è rappresentata da un'energia associata. L'obiettivo è trovare una soluzione con energia (cioè un costo o errore) minima.

2. **Temperatura:** La temperatura nel processo fisico corrisponde a un parametro che controlla la probabilità di accettare peggioramenti temporanei nella soluzione durante l'ottimizzazione. Questo permette all'algoritmo di evitare minimi locali, cioè soluzioni subottimali che non sono globalmente ottimali.

### Fasi del Simulated Annealing

1. **Inizializzazione:**
   - Si parte con una temperatura iniziale alta e una soluzione iniziale casuale.
   - La temperatura determina quanto è probabile accettare soluzioni peggiori rispetto a quella attuale.

2. **Iterazione:**
   - A ogni passo, si genera una nuova soluzione vicina alla soluzione corrente (ad esempio, tramite una piccola modifica).
   - Se la nuova soluzione è migliore, viene accettata come nuova soluzione corrente.
   - Se la nuova soluzione è peggiore, viene accettata con una probabilità che dipende dalla differenza di qualità tra le due soluzioni e dalla temperatura corrente. Questa probabilità è data da:

$$   [
   P(\text{accettazione}) = \exp\left(\frac{-\Delta E}{T}\right)
   ]$$

   Dove:
   - \(\Delta E\) è la differenza di energia tra la nuova soluzione e quella corrente.
   - \(T\) è la temperatura corrente.

3. **Raffreddamento:**
   - La temperatura viene gradualmente ridotta secondo un **programma di raffreddamento** (ad esempio, un fattore moltiplicativo che riduce la temperatura a ogni passo).
   - Con la diminuzione della temperatura, l'algoritmo diventa sempre meno propenso ad accettare soluzioni peggiori, focalizzandosi sulla ricerca di minimi locali sempre più raffinati.

4. **Convergenza:**
   - Il processo continua fino a quando la temperatura scende sotto una soglia minima, o si raggiunge un numero massimo di iterazioni. La soluzione corrente viene allora considerata la soluzione ottimale o quasi ottimale.

### Vantaggi e Svantaggi del Simulated Annealing

**Vantaggi:**

- **Evasione dai Minimi Locali:** La caratteristica principale del Simulated Annealing è la capacità di uscire dai minimi locali grazie all'accettazione temporanea di soluzioni peggiori.
- **Semplicità:** È relativamente semplice da implementare e può essere applicato a una vasta gamma di problemi di ottimizzazione.
- **Flessibilità:** Può essere adattato a problemi con spazi delle soluzioni molto complessi o non strutturati.

**Svantaggi:**

- **Tempi di Convergenza:** Può richiedere un tempo considerevole per convergere, specialmente se il programma di raffreddamento è troppo lento.
- **Dipendenza dai Parametri:** Le prestazioni dell'algoritmo dipendono fortemente dalla scelta della temperatura iniziale, dal programma di raffreddamento e dalle modalità con cui vengono generate le soluzioni vicine.

### Applicazioni

Il Simulated Annealing è utilizzato in una vasta gamma di problemi di ottimizzazione, tra cui:

- **Progettazione di circuiti integrati:** Per minimizzare l'area e la potenza consumata.
- **Problemi di routing:** Come il problema del commesso viaggiatore (TSP).
- **Pianificazione e scheduling:** Per ottimizzare l'uso delle risorse in progetti complessi.
- **Ottimizzazione in machine learning:** Per trovare i parametri ottimali di modelli complessi.

### Conclusione

Il Simulated Annealing è un potente algoritmo di ottimizzazione che, ispirandosi ai processi fisici, consente di esplorare lo spazio delle soluzioni in modo tale da evitare trappole nei minimi locali. Nonostante richieda una sintonizzazione accurata dei parametri, la sua capacità di affrontare problemi complessi lo rende una tecnica molto utile in intelligenza artificiale e altre discipline computazionali.