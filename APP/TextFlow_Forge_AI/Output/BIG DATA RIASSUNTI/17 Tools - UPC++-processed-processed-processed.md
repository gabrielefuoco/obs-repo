
## Riassunto di UPC++ e Programmazione PGAS

Il modello di programmazione **Partitioned Global Address Space (PGAS)** offre un approccio ibrido tra memoria condivisa e distribuita, suddividendo uno spazio di indirizzi globale in porzioni locali per ogni processo, limitando così lo scambio di dati e isolando i guasti.  L'**asynchronous PGAS (APGAS)** estende questo modello supportando task asincroni locali e remoti, permettendo l'esecuzione su hardware eterogeneo e la generazione dinamica di task.  Diversi linguaggi implementano PGAS, tra cui UPC++.

**UPC++** è una libreria C++ per la programmazione APGAS.  Ogni thread (o *rank*) accede alla propria memoria locale e a uno spazio di indirizzi globale suddiviso in segmenti condivisi tra i rank, sfruttando il middleware **GASNet** per la comunicazione (RMA e AM).  Le operazioni di accesso alla memoria remota sono asincrone.  UPC++ fornisce astrazioni come puntatori globali, RPC, *future* e oggetti condivisi, offrendo un controllo fine-grained sul parallelismo.  Tuttavia, l'elevato livello di dettaglio può portare a maggiore verbosità e potenziali colli di bottiglia di comunicazione.

Ogni programma UPC++ inizia con `upcxx::init()` e termina con `upcxx::finalize()`.  Il numero di *rank* è fisso, e ogni *rank* (con ID da `upcxx::rank_me()`) esegue una copia del programma.  Il parallelismo è gestito suddividendo il calcolo tra i *rank*, con un successivo raccoglimento dei risultati.  UPC++ utilizza il calcolo asincrono e gli oggetti *future* per sovrapporre comunicazione e calcolo, migliorando le prestazioni.  ![[]](_page_2_Figure_5.jpeg)

---

# Riassunto dell'implementazione del metodo Monte Carlo in UPC++ per stimare π

Questo documento descrive l'implementazione del metodo Monte Carlo per stimare π utilizzando la libreria UPC++ per il parallelismo.  L'algoritmo genera punti casuali in un quadrato unitario e conta quanti ricadono nel cerchio inscritto per approssimare π.  L'implementazione sfrutta le funzionalità di UPC++ per la gestione di memoria condivisa e operazioni collettive.

## Memoria Condivisa e Operazioni Collettive in UPC++

UPC++ fornisce la possibilità di lavorare con oggetti condivisi tra i diversi *rank* (processori).  Un oggetto condiviso è allocato in un segmento di memoria condivisa e accessibile tramite un puntatore globale (`upcxx::global_ptr<T>`).  Sono disponibili due metodi di allocazione: `upcxx::new_<T>` per allocazione in memoria condivisa e `new` (standard C++) per allocazione in memoria privata.  Le operazioni collettive, come la riduzione, vengono eseguite su insiemi di *rank* chiamati *team*. In UPC++, il *team* predefinito è `upcxx::world()`, comprendente tutti i *rank*.  La funzione `upcxx::allreduce` esegue una riduzione globale asincrona, restituendo un `upcxx::future<T>` che richiede l'utilizzo del metodo `wait()` per ottenere il risultato.

```c++
upcxx::future<T> upcxx::allreduce(const T& value, std::function<T(const T&, const T&)> op, upcxx::team team = upcxx::world());
```

## Implementazione del Metodo Monte Carlo

L'implementazione prevede le seguenti fasi:

1. **`hit()` function:** Verifica se un punto 2D generato casualmente (coordinate *x*, *y* in [0, 1]) ricade nel cerchio unitario (x² + y² ≤ 1).

2. **Generazione e Conteggio:** Dopo l'inizializzazione (`upcxx::init()`), ogni *rank* genera 100.000 punti casuali, contando gli *hit* localmente (`my_hits`).

3. **Raccolta e Calcolo:** Il *rank* 0 raccoglie i risultati da tutti i *rank* usando una riduzione (`reduce_to_rank0`).  Con il numero totale di *hit* (`hits`) e il numero totale di *rank* (`upcxx::rank_n()`), calcola il numero totale di punti generati (`trials`) e stima π come 4 * (hits/trials).

## Riduzione con `allreduce` e Puntatori Globali

La riduzione dei risultati può essere effettuata con `allreduce`, applicando la funzione `plus` (somma) ai valori locali di `my_hits`.  In alternativa, si possono utilizzare puntatori globali: il *rank* 0 alloca una matrice di interi tramite `upcxx::new_array`, e ogni *rank* aggiorna la propria posizione nella matrice usando `upcxx::rput` (remote put) asincrono.


![[](_page_7_Figure_7.jpeg)]
![[](_page_8_Figure_6.jpeg)]

---

La funzione `upcxx::barrier` in UPC++ garantisce la sincronizzazione prima di procedere con le operazioni successive.  In particolare, essa sincronizza i trasferimenti di dati remoti.  Successivamente, il rank 0 (processo principale) accede alla memoria condivisa tramite `upcxx::global_ptr<T>::local()`, ottenendo una copia locale del puntatore globale.  Su questa copia locale, il rank 0 esegue la somma dei valori. Infine, la memoria condivisa viene deallocata usando `upcxx::delete_array`, e il risultato della somma viene restituito.

---
