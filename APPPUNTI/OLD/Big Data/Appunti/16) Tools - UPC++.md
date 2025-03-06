
## Strumenti di programmazione basati su PGAS

Il modello di programmazione **Partitioned Global Address Space (PGAS)** rappresenta un compromesso tra i modelli di programmazione a memoria distribuita e a memoria condivisa. Implementa uno spazio di indirizzi di memoria globale logicamente suddiviso in porzioni locali a singoli processi. Il suo obiettivo principale è limitare lo scambio di dati e isolare i guasti in sistemi su larga scala. Il modello **asynchronous PGAS (APGAS)** è una variante di PGAS che supporta la creazione di task asincroni sia locali che remoti. A differenza di PGAS, non richiede che tutti i processi vengano eseguiti su hardware omogeneo e supporta la generazione dinamica di più task. Negli ultimi anni sono stati proposti diversi linguaggi basati su PGAS, come *DASH*, *X10*, *Chapel*, *pPython* e *UPC*.

## UPC++

**UPC++** (Zheng et al., 2014) è una libreria C++ che fornisce classi e funzioni per la programmazione APGAS. Ogni thread (chiamato *rank*) ha accesso alla memoria locale e a uno spazio di indirizzi globale allocato in segmenti condivisi distribuiti tra i rank. Questo modello di memoria rende UPC++ adatto per programmi paralleli efficienti e scalabili su computer paralleli a memoria distribuita con centinaia di migliaia di core.

![[Pasted image 20250306101440.png|336]]

UPC++ utilizza il middleware **Global-Address Space Networking (GASNet)**, un livello indipendente dal linguaggio che fornisce primitive di comunicazione indipendenti dalla rete, tra cui *accesso alla memoria remota* (RMA) e *messaggi attivi* (AM). In UPC++, tutte le operazioni di accesso alla memoria remota sono di default asincrone e le interfacce sono componibili e simili a quelle del C++ convenzionale. Le astrazioni di programmazione chiave includono *puntatori globali*, *chiamate di procedura remota* (*RPC*), *future* e *oggetti condivisi*, fornendo un toolkit versatile per la strutturazione di applicazioni parallele. UPC++ fornisce un *livello di astrazione basso*, consentendo un controllo fine-grained sul parallelismo e un utilizzo efficiente delle risorse. Nonostante i suoi vantaggi, UPC++ può riscontrare colli di bottiglia delle prestazioni a causa dell'uso estensivo della comunicazione. Inoltre, l'assenza di costrutti di alto livello contribuisce ad una maggiore verbosità, richiedendo ai programmatori di gestire esplicitamente lo scambio di dati e la sincronizzazione.

## Nozioni di base di programmazione

Tutti i programmi UPC++ includono due operazioni fondamentali:

* **`upcxx::init()`**: inizializza il runtime UPC++ (deve essere chiamata prima di utilizzare qualsiasi funzionalità UPC++).
* **`upcxx::finalize()`**: arresta il runtime UPC++.

Un programma UPC++ viene eseguito con un numero fisso di thread, i **rank**, ognuno dei quali esegue una copia del programma. Ogni rank ha un identificativo compreso tra 0 e N −1 (N è il numero di rank), accessibile tramite **`upcxx::rank_me()`**.

Il calcolo può essere suddiviso tra diversi rank, eseguendo operazioni in parallelo. Il risultato finale viene raccolto da uno dei rank, implicando un punto di sincronizzazione. Per migliorare il parallelismo, UPC++ sfrutta il *calcolo asincrono*, sovrapponendo comunicazione e calcolo tramite oggetti *future*. Questi oggetti hanno un valore e uno stato che indica se il valore è disponibile (*ready*).

Ad esempio, l'operazione **`upcxx::allreduce`**:

```c++
upcxx::future<T> upcxx::allreduce(const T& value, std::function<T(const T&, const T&)> op, upcxx::team team = upcxx::world());
```

Esegue una riduzione globale di un valore di tipo *T* su tutti i rank applicando una funzione binaria (es., somma). Supporta i *team*, insiemi ordinati di rank ai quali applicare operazioni collettive. In UPC++, l'unico team supportato è **`upcxx::world()`** (tutti i rank). Il tipo di ritorno è `future<T>`, consentendo il calcolo asincrono. Il metodo `wait()` di `upcxx::future` controlla lo stato dell'oggetto `future`, ciclando fino al suo completamento.

**Oggetti condivisi:** UPC++ consente di lavorare con oggetti condivisi tra diversi rank.

## Memoria condivisa in UPC++

Un oggetto condiviso viene allocato in un segmento di memoria condivisa ed è accessibile tramite un puntatore globale, definito come `upcxx::global_ptr<T> gptr`. UPC++ definisce due aree di memoria diverse nello spazio di indirizzamento globale:

* **`upcxx::new_<T>`**: alloca un nuovo oggetto di tipo *T* nel segmento condiviso del rank corrente. Ogni rank può fare riferimento a questo oggetto tramite un puntatore globale privato al suo segmento condiviso locale.
* **`new` (standard C++)**: allocazione dinamica nella memoria locale privata del rank.

![[Pasted image 20250306101455.png|428]]

## Metodo Monte Carlo per stimare π

Il metodo Monte Carlo per stimare π:

- Genera *N* punti in un piano 2D, con coordinate (*x*, *y*) variabili casuali uniformemente distribuite in [0, 1].
- Conta quanti punti (*M*) soddisfano x² + y² ≤ 1 (ricadono nel settore della circonferenza con raggio unitario centrata in (0, 0)).
- Stima π: π ≈ 4 * (M/N) (A<sub>quadrato</sub> = 1, A<sub>settore</sub> = π/4)
![[_page_8_Figure_6.jpeg|313]]

## Implementazione in UPC++

### Funzione `hit()`

Include le librerie necessarie e definisce la funzione `hit()` per verificare se un punto 2D ricade nel settore della circonferenza. Campiona un punto 2D casuale nel quadrato unitario e verifica se le sue coordinate soddisfano x² + y² ≤ 1.

### Generazione dei punti e conteggio dei colpi

Il programma inizia con `upcxx::init()`. Ogni rank inizializza un generatore di numeri casuali locale e esegue 100.000 prove, generando un punto casuale e contando gli *hit*. Questo valore (`my_hits`) viene memorizzato localmente da ogni rank.

### Raccolta dei risultati e calcolo di π

Il rank 0 raccoglie i risultati tramite `reduce_to_rank0`, determinando il numero totale di hit (`hits`). Con il numero totale di rank (`upcxx::rank_n()`), calcola il numero complessivo di punti generati (*trials*) e la stima di π (4 * hits/trials).

## Riduzione con `allreduce` e puntatori globali

UPC++ offre diversi modi per eseguire la riduzione, incluso `allreduce`:

* Esegue una riduzione collettiva applicando la funzione `plus` (somma) su tutti i valori locali di `my_hits`.
* È asincrona e restituisce un `future<int>`, quindi il risultato deve essere atteso tramite `wait()`.

Un'alternativa efficace usa i puntatori globali UPC++, basati sul modello di memoria condivisa APGAS.

### Utilizzo dei puntatori globali per la riduzione

Il rank 0 alloca un puntatore globale `all_hits_ptr` a una matrice di interi tramite `upcxx::new_array`. La dimensione è `upcxx::rank_n()`. Il puntatore globale viene trasmesso a tutti i rank, che aggiorneranno una posizione specifica della matrice in base al loro ID rank (`upcxx::rank_me()`). `upcxx::rput` (remote put) esegue il trasferimento asincrono.

`upcxx::barrier` sincronizza i trasferimenti remoti. Il rank 0 usa `upcxx::global_ptr<T>::local()` per ottenere una versione locale del puntatore globale e sommare i valori. Dealloca l'array condiviso usando `upcxx::delete_array` e restituisce il risultato.

