
## Schema Riassuntivo su PGAS e UPC++

**I. Modello di Programmazione Partitioned Global Address Space (PGAS)**

   *   **A. Definizione:** Compromesso tra memoria distribuita e condivisa.
   *   **B. Caratteristiche:**
        *   Spazio di indirizzi globale logicamente suddiviso in porzioni locali.
        *   Obiettivo: limitare lo scambio di dati e isolare i guasti.
   *   **C. Asynchronous PGAS (APGAS):**
        *   Variante di PGAS che supporta task asincroni (locali e remoti).
        *   Non richiede hardware omogeneo.
        *   Supporta la generazione dinamica di task.
   *   **D. Linguaggi basati su PGAS:**
        *   *DASH*, *X10*, *Chapel*, *pPython*, *UPC*.

**II. UPC++**

   *   **A. Descrizione:** Libreria C++ per la programmazione APGAS (Zheng et al., 2014).
   *   **B. Modello di Memoria:**
        *   Ogni thread (rank) ha accesso alla memoria locale.
        *   Spazio di indirizzi globale allocato in segmenti condivisi tra i rank.
        *   Adatto per programmi paralleli efficienti e scalabili.
   *   **C. Middleware GASNet (Global-Address Space Networking):**
        *   Livello indipendente dal linguaggio.
        *   Fornisce primitive di comunicazione indipendenti dalla rete.
        *   Include *accesso alla memoria remota* (RMA) e *messaggi attivi* (AM).
   *   **D. Caratteristiche Principali:**
        *   Operazioni di accesso alla memoria remota asincrone (default).
        *   Interfacce componibili e simili al C++ convenzionale.
        *   Astrazioni di programmazione chiave:
            *   *Puntatori globali*.
            *   *Chiamate di procedura remota* (*RPC*).
            *   *Future*.
            *   *Oggetti condivisi*.
   *   **E. Vantaggi:**
        *   *Livello di astrazione basso*.
        *   Controllo fine-grained sul parallelismo.
        *   Utilizzo efficiente delle risorse.
   *   **F. Svantaggi:**
        *   Colli di bottiglia delle prestazioni dovuti alla comunicazione.
        *   Maggiore verbosità (gestione esplicita di scambio dati e sincronizzazione).

**III. Nozioni di Base di Programmazione UPC++**

   *   **A. Operazioni Fondamentali:**
        *   **`upcxx::init()`**: Inizializza il runtime UPC++.
        *   **`upcxx::finalize()`**: Arresta il runtime UPC++.
   *   **B. Rank:**
        *   Un programma UPC++ viene eseguito con un numero fisso di thread (rank).
        *   Ogni rank esegue una copia del programma.
        *   Identificativo di ogni rank: tra 0 e N −1 (N = numero di rank).
        *   Accessibile tramite **`upcxx::rank_me()`**.
   *   **C. Parallelismo:**
        *   Il calcolo è suddiviso tra i rank (operazioni in parallelo).
        *   Il risultato finale viene raccolto da un rank (punto di sincronizzazione).
   *   **D. Calcolo Asincrono:**
        *   Sovrapposizione di comunicazione e calcolo tramite oggetti *future*.
        *   Oggetti *future*: hanno un valore e uno stato (*ready* se il valore è disponibile).

---

**Schema Riassuntivo UPC++ e Metodo Monte Carlo**

**1. Operazioni Collettive con `upcxx::allreduce`**

   *   Funzione: `upcxx::allreduce(const T& value, std::function<T(const T&, const T&)> op, upcxx::team team = upcxx::world())`
   *   Scopo: Riduzione globale di un valore di tipo *T* su tutti i rank.
   *   Funzionamento: Applica una funzione binaria (es. somma) su tutti i rank.
   *   Team: Insiemi ordinati di rank (supportato solo `upcxx::world()` - tutti i rank).
   *   Asincronicità: Ritorna `future<T>`, per calcolo asincrono.
   *   Sincronizzazione: `upcxx::future::wait()` attende il completamento.

**2. Memoria Condivisa e Oggetti Condivisi**

   *   Oggetti condivisi: Accessibili tra diversi rank tramite puntatori globali.
   *   Puntatore globale: `upcxx::global_ptr<T> gptr`
   *   Aree di memoria:
        *   `upcxx::new_<T>`: Alloca oggetto di tipo *T* nel segmento condiviso del rank corrente.
        *   `new` (standard C++): Allocazione dinamica nella memoria locale privata del rank.

**3. Metodo Monte Carlo per Stimare π**

   *   Algoritmo:
        *   Genera *N* punti casuali (*x*, *y*) in [0, 1].
        *   Conta *M* punti che soddisfano x² + y² ≤ 1.
        *   Stima π: π ≈ 4 * (M/N)

**4. Implementazione in UPC++ del Metodo Monte Carlo**

   *   Funzione `hit()`:
        *   Verifica se un punto 2D ricade nel settore della circonferenza (x² + y² ≤ 1).
   *   Generazione e Conteggio:
        *   Ogni rank inizializza un generatore di numeri casuali.
        *   Esegue 100.000 prove per rank, contando gli *hit* (`my_hits`).
   *   Raccolta e Calcolo:
        *   Rank 0 raccoglie i risultati tramite `reduce_to_rank0` (numero totale di hit: `hits`).
        *   Calcola il numero totale di punti generati: *trials* = `upcxx::rank_n()` * 100.000
        *   Stima π: π ≈ 4 * hits/trials

**5. Riduzione con `allreduce` e Puntatori Globali**

   *   `allreduce`:
        *   Esegue una riduzione collettiva (somma) su `my_hits`.
        *   Asincrona, richiede `wait()`.
   *   Puntatori Globali:
        *   Rank 0 alloca `all_hits_ptr` (matrice di interi) con `upcxx::new_array` di dimensione `upcxx::rank_n()`.
        *   Trasmette `all_hits_ptr` a tutti i rank.
        *   Ogni rank aggiorna la posizione corrispondente al proprio ID (`upcxx::rank_me()`).
        *   `upcxx::rput` (remote put) esegue il trasferimento asincrono.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Sincronizzazione e Operazioni con Puntatori Globali**

    A. `upcxx::barrier`: Sincronizza i trasferimenti remoti.

**II. Operazioni del Rank 0**

    A. Accesso Locale: Utilizza `upcxx::global_ptr<T>::local()` per ottenere una versione locale del puntatore globale.
    B. Calcolo: Somma i valori accessibili tramite il puntatore locale.

**III. Gestione della Memoria e Risultato**

    A. Deallocazione: Dealloca l'array condiviso usando `upcxx::delete_array`.
    B. Restituzione: Restituisce il risultato della somma.

---
