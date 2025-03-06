
**I. Workflow: Concetti Fondamentali**

    A. Definizione: Serie di attività/eventi/task per raggiungere un obiettivo.
    B. Definizione (WMC): Automazione di un processo aziendale con passaggio di documenti/informazioni/task tra partecipanti secondo regole procedurali.
    C. Applicazioni:
        1.  Modello di programmazione per elaborazione dati su piattaforme distribuite.
        2.  Combinazione di analisi dati, calcolo scientifico e simulazione.
    D. Componenti:
        1.  Processo: Insieme di task connessi per produrre un prodotto/risultato/servizio.
        2.  Task (Attività): Unità di lavoro, singolo passo logico nel processo.
    E. Caratteristiche:
        1.  Pattern ben definiti e ripetibili.
        2.  Approccio dichiarativo (logica di alto livello).
        3.  Memorizzabili e riutilizzabili.

**II. Sistemi di Gestione dei Workflow (WMS)**

    A. Funzione: Facilitare definizione, sviluppo ed esecuzione dei processi.
    B. Ruolo chiave: Coordinamento delle attività (attuazione).

**III. Struttura del Workflow**

    A. Rappresentazione: Grafo costituito da archi e vertici.
    B. Elementi:
        1.  Vertici: Task, attività o fasi specifiche.
        2.  Archi: Flusso/sequenza dei task (ordine di esecuzione).
    C. Implementazione: Programmi software (linguaggi, librerie, sistemi).

**IV. Pattern dei Workflow**

    A. Definizione: Modo standardizzato di organizzare e orchestrare i task.
    B. Principali Pattern:
        1.  Sequenza
        2.  Ramificazione
        3.  Sincronizzazione
        4.  Ripetizione

**V. Pattern di Sequenza**

    A. Definizione: Sequenza di task completati in ordine specifico.
    B. Rappresentazione: Archi diretti indicano la direzione del flusso di controllo.

**VI. Pattern di Ramificazione**

    A. Definizione: Diramazione in più flussi in base a condizioni.
    B. Condizioni: Risultato di task precedenti, valori di dati, input utente, etc.
    C. Varianti:
        1.  AND-split: Diramazione in flussi concorrenti.
        2.  XOR-split: Diramazione in un solo flusso (basato su condizioni).
        3.  OR-split: Diramazione in uno o più flussi concorrenti (basato su condizioni).

---

## Schema Riassuntivo dei Pattern di Workflow

**I. Pattern di Sincronizzazione**

*   Definizione: Flussi di controllo multipli convergono in un singolo ramo.
*   Varianti:
    *   AND-join: Tutti i rami in ingresso devono essere completati.
    *   XOR-join: Solo un ramo in ingresso deve essere completato.
    *   OR-join: Almeno un ramo in ingresso deve essere completato.

**II. Pattern di Ripetizione**

*   Definizione: Modi per specificare la ripetizione di task.
*   Tipi:
    *   Ciclo Arbitrario: Task ripetuti tramite istruzione *goto*.
    *   Ciclo Strutturato: Task ripetuti con condizione di terminazione.
        *   *while..do*: Condizione valutata prima dell'iterazione.
        *   *repeat…until*: Condizione valutata dopo l'iterazione.
    *   Ricorsione: Task ripetuto tramite auto-invocazione.

**III. Grafi Aciclici Diretti (DAG)**

*   Definizione: Workflow diretto e aciclico.
    *   Diretto: Ogni task ha almeno un task precedente o successivo (o entrambi).
    *   Aciclico: Assenza di cicli; i task non generano dati che si riferiscono a se stessi.
*   Applicazioni:
    *   Gestione dei workflow.
    *   Framework big data (es., Apache Spark).
    *   Modellazione di processi complessi di analisi dei dati (es., data mining).
*   Tipi di Dipendenze:
    *   Dipendenze dati: Output di un task come input per task successivi.
    *   Dipendenze di controllo: Completamento di task necessari prima di iniziarne altri.
*   Definizione delle Dipendenze:
    *   Esplicita: Dipendenze definite tramite istruzioni esplicite (es., T2 dipende da T1).
    *   Implicita: Dipendenze dedotte analizzando le relazioni input-output (es., T2 legge l'input O1, che è un output di T1).
*   Relazione con MapReduce:
    *   DAG è una generalizzazione di MapReduce.
    *   DAG offre maggiore flessibilità e ottimizzazione globale.
    *   Esempio di ottimizzazione: Inversione dell'ordine di *map* e *filter*.

**IV. Grafi Ciclici Diretti (DCG)**

*   Definizione: Workflow con cicli che rappresentano meccanismi di controllo del loop o dell'iterazione.
*   Struttura: Rete di task dove i nodi rappresentano servizi, istanze di componenti software o oggetti di controllo.

---

## Schema Riassuntivo: Rappresentazione degli Edge in un Grafo

**I. Funzione degli Edge:**

*   **A. Rappresentazione:** Gli edge di un grafo rappresentano:
    *   1. Messaggi
    *   2. Flussi di dati
    *   3. Pipe
*   **B. Scopo:** Facilitano lo scambio di:
    *   1. Lavoro
    *   2. Informazioni
*   **C. Attori Coinvolti:** Tra servizi e componenti.

---
