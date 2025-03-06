
**Schema Riassuntivo del Modello BSP (Bulk Synchronous Parallel)**

**I. Introduzione al BSP**
    *   **A.** Modello di calcolo parallelo sviluppato da Leslie Valiant (1990).
    *   **B.** Obiettivo: Semplificare la programmazione parallela evitando gestione complessa di memoria e comunicazione.
    *   **C.** Caratteristiche:
        *   **1.** Calcolo parallelo efficiente.
        *   **2.** Basso grado di sincronizzazione.
        *   **3.** Simile al modello di Von Neumann.

**II. Architettura di un Computer BSP**
    *   **A.** Componenti principali:
        *   **1.** Elementi di Elaborazione (PE) o processori: Eseguono calcoli locali.
        *   **2.** Router: Consegna messaggi tra coppie di PE.
        *   **3.** Sincronizzatore hardware: Sincronizza i PE a intervalli regolari di *L* unità di tempo (latenza di comunicazione o periodicità di sincronizzazione).

**III. Calcolo BSP e Superstep**
    *   **A.** Calcolo BSP: Insieme di superstep.
        *   **1.** Ogni processore esegue un compito con calcoli locali, trasmissioni e arrivi di messaggi.
        *   **2.** Controllo globale ogni *L* unità di tempo (periodicità) per completamento superstep.
    *   **B.** Superstep BSP: Tre fasi ordinate.
        *   **1.** Calcolo concorrente:
            *   **a.** Ogni processore esegue calcoli asincroni con dati locali.
            *   **b.** Utilizzo esclusivo della memoria locale del processore.
        *   **2.** Comunicazione globale:
            *   **a.** Scambio di dati in risposta a richieste del calcolo locale.
        *   **3.** Sincronizzazione a barriera:
            *   **a.** I processi attendono che tutti raggiungano la barriera.
    *   **C.** Disaccoppiamento: Comunicazione e sincronizzazione sono disaccoppiate.
        *   **1.** Indipendenza tra processi in un superstep.
        *   **2.** Prevenzione di deadlock.

**IV. Comunicazione nel Modello BSP**
    *   **A.** Semplificazione: Gestione collettiva delle azioni di comunicazione.
        *   **1.** Limite di tempo alla trasmissione batch dei dati.
    *   **B.** Unità di comunicazione: Tutte le azioni di comunicazione di un superstep sono considerate un'unica unità.
        *   **1.** Dimensioni di messaggio costanti all'interno dell'unità.
    *   **C.** Costo della comunicazione:
        *   **1.** *h*: Numero massimo di messaggi per un superstep.
        *   **2.** *g*: Rapporto di throughput di comunicazione.
        *   **3.** Tempo per inviare *h* messaggi di dimensione uno: *hg*.
        *   **4.** Messaggio di lunghezza *m* trattato come *m* messaggi di lunghezza uno.
        *   **5.** Costo di comunicazione per messaggio di lunghezza *m*: *mg*.

---

**Schema Riassuntivo del Testo**

**I. Sincronizzazione nel Modello BSP**

    *   **A. Importanza della Sincronizzazione:**
        *   Il modello BSP si basa sulla sincronizzazione tramite barriere.
        *   Elimina dipendenze circolari, deadlock e livelock.
    *   **B. Costi della Sincronizzazione:**
        *   Variazioni nei tempi di completamento dei calcoli locali.
        *   Necessità di mantenere la coerenza globale tra i processori.
    *   **C. Gestione dei Costi di Sincronizzazione:**
        *   Assegnazione di task proporzionale ai carichi di lavoro.
        *   Considerazione dell'efficienza della rete di comunicazione.
        *   Utilizzo di hardware specifico per la sincronizzazione.
        *   Metodi di gestione delle interruzioni.

**II. Costo di un Algoritmo BSP**

    *   **A. Condizione per lo Scambio di Messaggi:**
        *   Per scambiare almeno *h* messaggi: *L ≥ hg*
        *   *L* = Periodicità
        *   *hg* = Tempo per un processore per inviare *h* messaggi di dimensione uno.
        *   Basso valore di *g* è cruciale.
    *   **B. Costo di un Superstep:**
        *   $T_s = w_s + h_s g + L$
        *   $w_s$ = Costo totale del calcolo nel superstep.
    *   **C. Costo Totale dell'Algoritmo BSP:**
        *   $T = \sum_{1 \le s \le S} T_s = \sum_{1 \le s \le S} (w_s + h_s g + L) = W + Hg + SL$
        *   *W* = Costo totale di computazione
        *   *H* = Costo totale di comunicazione
        *   *S* = Numero totale di supersteps

**III. Modello BSP e Memoria Condivisa**

    *   **A. Limitazioni del Modello BSP:**
        *   Non supporta direttamente memoria condivisa, broadcasting o combining.
    *   **B. Emulazione di PRAM su BSP:**
        *   Possibilità di emulare una Parallel Random Access Machine (PRAM).
    *   **C. Caratteristiche della PRAM:**
        *   Numero infinito di processori.
        *   Memoria condivisa con capacità illimitata.
        *   Comunicazione tramite Memory Access Unit (MAU).
        *   Calcoli completamente sincroni.
    *   **D. Varianti di PRAM:**
        *   EREW, CREW, ERCW, CRCW

**IV. Bulk-Synchronous PPRAM (BSPRAM)**

    *   **A. Introduzione:**
        *   Proposta da Alexandre Tiskin (1998).
        *   Obiettivo: facilitare la programmazione in stile memoria condivisa.
    *   **B. Componenti:**
        *   *p* processori con memoria locale veloce.
        *   Singola memoria principale condivisa.
    *   **C. Funzionamento:**
        *   Supersteps simili a BSP.
        *   Tre fasi: input, calcolo locale, output.
        *   Interazione dei processori con la memoria principale.

---

Ecco lo schema riassuntivo del testo fornito:

**I. Sincronizzazione e Calcolo in Supersteps**

    *   A. Sincronizzazione: Avviene tra i supersteps.
    *   B. Calcolo: All'interno di un superstep è asincrono.
    *   C. Riferimento Visivo: Vedi figura nella diapositiva successiva (Figura 2).

---
