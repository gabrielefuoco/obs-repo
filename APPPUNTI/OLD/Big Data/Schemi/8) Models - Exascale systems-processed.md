
**I. Sfide e Opportunità dei Sistemi Exascale**

*   A. Opportunità: Promettenti per calcoli complessi e gestione di grandi volumi di dati.
*   B. Sfide nella progettazione e implementazione:
    *   Scalabilità
    *   Latenza di rete
    *   Affidabilità
    *   Robustezza delle operazioni sui dati

**II. Requisiti dei Modelli di Programmazione Exascale Scalabili**

*   A. Meccanismi necessari:
    *   Accesso parallelo ai dati: Migliorare la larghezza di banda accedendo contemporaneamente a elementi diversi.
    *   Resilienza ai guasti: Gestire i guasti durante la comunicazione non locale.
    *   Comunicazione locale guidata dai dati: Limitare lo scambio di dati.
    *   Elaborazione dei dati su gruppi limitati di core: Su specifiche macchine exascale.
    *   Sincronizzazione near-data: Ridurre l'overhead generato dalla sincronizzazione tra molti core distanti.
    *   Analisi in-memory: Ridurre i tempi di reazione memorizzando nella cache i dati nelle RAM dei nodi di elaborazione.
    *   Selezione dei dati basata sulla località: Ridurre la latenza mantenendo localmente disponibile un sottoinsieme di dati.

**III. Limiti dei Modelli Tradizionali HPC**

*   A. Inadeguatezza di MPI, OpenMP e Map-Reduce per sistemi exascale.
*   B. Proprietà essenziali dei modelli di programmazione influenzate dalla transizione exascale:
    *   Scheduling dei thread
    *   Comunicazione
    *   Sincronizzazione
    *   Distribuzione dei dati
    *   Viste di controllo

**IV. Architettura Message-Passing e MPI**

*   A. Adozione parziale dell'architettura message-passing nei sistemi exascale.
*   B. Sfide di MPI:
    *   Richiede la gestione manuale di parallelizzazione, distribuzione dati, comunicazione e sincronizzazione.
    *   Non adatto per il bilanciamento del carico dinamico (distribuzione statica dei dati).
    *   Problemi di scalabilità dovuti alla comunicazione many-to-many.
    *   I/O come collo di bottiglia.

**V. Alternative: Sistemi a Memoria Condivisa Parallela**

*   A. Trasferimento della responsabilità della parallelizzazione dal programmatore al compilatore.
*   B. Limiti dei modelli a memoria condivisa:
    *   Mancanza di controllo sulla distribuzione dei dati.
    *   Meccanismi di sincronizzazione non scalabili (lock, sezioni atomiche).
    *   Visione globale dei dati che porta a sincronizzazione congiunta e programmazione inefficiente.

**VI. Programmazione Eterogenea (CPU + GPU)**

*   A. Vantaggi: Prestazioni di picco ed efficienza energetica.
*   B. Sfide: Gestione di ambienti di esecuzione e modelli di programmazione diversi.
*   C. Necessità di nuove astrazioni, modelli di programmazione e strumenti per affrontare le sfide.

---

**Schema Riassuntivo: Modelli di Programmazione Exascala**

**I. Sfide della Programmazione Exascala**
    *   A. Gestione di milioni di thread
    *   B. Minimizzazione della sincronizzazione
    *   C. Riduzione della comunicazione e dell'utilizzo della memoria remota
    *   D. Gestione di guasti software e hardware

**II. Modelli di Programmazione Proposti**

    *   **A. Legion**
        *   1.  Modello a memoria distribuita per alte prestazioni
        *   2.  Organizzazione dei dati basata su *regioni logiche*
            *   a. Allocazione dinamica, rimozione e memorizzazione di gruppi di oggetti
            *   b. Input a *task* che leggono dati e forniscono informazioni sulla località
        *   3.  Suddivisione in sottoregioni *disgiunte* o *aliasate* per valutare l'indipendenza del calcolo

    *   **B. Charm++**
        *   1.  Modello a memoria distribuita con oggetti interagenti mappati dinamicamente
        *   2.  Approccio asincrono, basato su messaggi e task, con oggetti mobili
            *   a. Migrazione degli oggetti tra processori
            *   b. Invio di dati a oggetti logici invece che a processori fisici
        *   3.  Utilizzo dell'*overdecomposition* per dividere le applicazioni in molti oggetti piccoli

    *   **C. DCEx**
        *   1.  Modello basato su PGAS per applicazioni parallele su larga scala e datacentriche
        *   2.  Costruito su operazioni di base consapevoli dei dati
        *   3.  Utilizzo della *sincronizzazione near-data*
        *   4.  Struttura in blocchi data-paralleli per calcolo, comunicazione e migrazione paralleli

    *   **D. X10**
        *   1.  Modello basato su APGAS
        *   2.  Introduzione delle locazioni come astrazione del contesto computazionale
            *   a. Vista localmente sincrona della memoria condivisa
        *   3.  Distribuzione di più luoghi, ognuno con dati e attività (thread leggeri)
            *   a. Le attività possono utilizzare sincronicamente regioni di memoria all'interno del luogo

    *   **E. Chapel**
        *   1.  Modello basato su APGAS con astrazioni di linguaggio di alto livello
        *   2.  *Strutture dati a vista globale*
            *   a. Array e altri dati aggregati con dimensioni e indici rappresentati globalmente
            *   b. Implementazioni distribuite tra i *locale*
        *   3.  *Vista globale del controllo*
            *   a. Parallelismo introdotto attraverso concetti specifici del linguaggio
            *   b. Un *locale* è un'astrazione dell'unità di accesso uniforme alla memoria

    *   **F. UPC++**
        *   1.  Libreria C++ per la programmazione PGAS
        *   2.  Strumenti per descrivere le dipendenze tra calcoli asincroni e trasferimento di dati
        *   3.  Comunicazione unidirezionale efficiente
        *   4.  Spostamento del calcolo sui dati tramite chiamate a procedure remote

---

Ecco lo schema riassuntivo del testo fornito:

**I. Concetti di Programmazione Principali della Libreria**

    A. **Puntatori Globali:**
        1. Supportano un'efficace sfruttamento della località dei dati.

    B. **Programmazione Asincrona Basata su RPC:**
        1. Permette lo sviluppo efficiente di programmi asincroni.

    C. **Futures:**
        1. Gestiscono la disponibilità dei dati provenienti da computazioni.

---
