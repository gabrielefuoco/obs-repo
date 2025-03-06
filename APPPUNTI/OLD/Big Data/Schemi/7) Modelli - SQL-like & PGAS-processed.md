
## Schema Riassuntivo: SQL-like e Big Data

**1. Sfide dei Database Relazionali e l'Emergere di NoSQL**

*   **1.1 Limitazioni dei Database Relazionali (SQL-like):**
    *   Difficoltà di scalabilità orizzontale per gestire grandi volumi di dati.
*   **1.2 Introduzione a NoSQL:**
    *   Alternativa non relazionale per scalabilità orizzontale (lettura/scrittura).
    *   Modello **BASE** (Basic Availability, Soft state, Eventual consistency) vs. **ACID** (Atomicità, Coerenza, Isolamento, Durabilità).
    *   Spesso inadatti all'analisi dei dati.

**2. Sistemi SQL-like per Big Data: Un Ponte tra SQL e MapReduce**

*   **2.1 Motivazioni:**
    *   Combinare l'efficienza di MapReduce con la semplicità di SQL.
    *   Superare la complessità di MapReduce per operazioni semplici (aggregazioni, selezioni, conteggi).
*   **2.2 Funzionamento:**
    *   Ottimizzazione automatica delle query su grandi repository tramite MapReduce in background.
    *   Esempio: **Apache Hive** per semplificare l'analisi dei dati con un linguaggio simile a SQL.

**3. Vantaggi dell'Utilizzo di SQL con i Big Data**

*   **3.1 Ragioni dell'Adozione:**
    *   Strumento preferito per sviluppatori, amministratori di database e data scientist.
    *   Ampiamente utilizzato in prodotti commerciali.
*   **3.2 Vantaggi Chiave:**
    *   **Linguaggio dichiarativo:** Facile da comprendere.
    *   **Interoperabilità:** Standardizzato, compatibile tra sistemi diversi.
    *   **Data-driven:** Operazioni riflettono trasformazioni dei dataset.

**4. Query-in-Place: Un Paradigma per l'Analisi dei Big Data**

*   **4.1 Definizione:**
    *   Esecuzione di query direttamente sui dati nella loro posizione originale.
*   **4.2 Vantaggi:**
    *   Ottimizzazione dell'utilizzo dei dati: Eliminazione di processi ridondanti.
    *   Minore latenza: Disponibilità immediata dei dati e riduzione dei costi.
    *   Preservazione dell'integrità dei dati originali.

**5. Partizionamento dei Dati per l'Interrogazione Efficiente**

*   **5.1 Importanza:**
    *   Fondamentale per interrogare efficientemente i big data con SQL.
*   **5.2 Funzionamento:**
    *   Divisione dei dati di una tabella in base a valori di colonna specifici.
    *   Creazione di file/directory distinti.
*   **5.3 Benefici:**
    *   Riduzione dei costi di I/O.
    *   Accelerazione dell'elaborazione delle query.

**6. Partizionamento Eccessivo e PGAS**

*   **6.1 Problema:**
    *   Il partizionamento eccessivo può portare a un numero elevato di file e directory.
*   **6.2 Conseguenze:**
    *   Aumento dei costi per il nodo master, che deve mantenere tutti i metadati in memoria.

---

**Schema Riassuntivo PGAS (Partitioned Global Address Space)**

**1. Concetti Fondamentali PGAS**

*   **1.1. Obiettivo:** Aumentare la produttività del programmatore mantenendo alte prestazioni in programmazione parallela.
*   **1.2. Spazio di Indirizzamento Globale Condiviso:**
    *   Migliora la produttività.
    *   Separa accessi dati locali e remoti (fondamentale per scalabilità e prestazioni).
*   **1.3. Architettura:**
    *   Programma con più processi che eseguono lo stesso codice su nodi diversi.
    *   Ogni processo ha un **rank** (indice del nodo).
    *   Spazio di indirizzamento globale **partizionato** in spazi locali.
    *   Accesso locale diretto, accesso remoto tramite API.
*   **1.4. Linguaggi PGAS:**
    *   Considerano lo spazio di indirizzamento come un **ambiente globale**.
    *   Puntatori a dati ovunque nel sistema.
    *   Distinzione tra **memoria condivisa** (accessibile a tutti) e **memoria privata** (accessibile solo al thread proprietario).

**2. Parallelismo nel Modello PGAS**

*   **2.1. Single Program Multiple Data (SPMD):**
    *   Numero predeterminato di thread all'avvio.
    *   Ogni thread esegue lo stesso programma.
*   **2.2. Asynchronous PGAS (APGAS):**
    *   Singolo thread all'avvio.
    *   Generazione dinamica di nuovi thread (stesse o diverse partizioni).
    *   Ogni thread può eseguire codice diverso.
*   **2.3. Parallelismo Implicito:**
    *   Nessun parallelismo visibile nel codice.
    *   Generazione dinamica di thread durante l'esecuzione per accelerare il calcolo.

**3. Memoria e Funzione di Costo**

*   **3.1. Places:**
    *   Spazio di memoria suddiviso in *places*.
    *   Ogni *place* rappresenta un nodo di calcolo (processo/thread).
*   **3.2. Accesso alla Memoria:**
    *   Accesso locale (stesso *place*): costo basso e uniforme.
    *   Accesso remoto (altro *place*): costo maggiore.
*   **3.3. Modello di Memoria NUMA (Non-Uniform Memory Access):**
    *   Funzione di costo definisce gli accessi alla memoria.
*   **3.4. Struttura di Costo a Due Livelli:**
    *   **Economico:** Posizioni di memoria vicine all'origine della richiesta.
    *   **Costoso:** Posizioni di memoria distanti.
    *   Costo determinato da:
        *   *Place* di origine della richiesta.
        *   *Place* in cui si trovano i dati.

**4. Distribuzione dei Dati**

*   **4.1. Classificazione Linguaggi PGAS:** Basata su come i dati sono distribuiti tra i *places*.
*   **4.2. Modelli di Distribuzione Comuni:**
    *   **Ciclico:** Dati suddivisi in blocchi consecutivi disposti ciclicamente tra i *places*.

**5. Metodi di Partizionamento dei Dati**

*   **5.1. Block:**
    *   Dati suddivisi in blocchi di dimensione uguale e consecutiva.
    *   Distribuiti tra diverse posizioni (o nodi).

---

Ecco uno schema riassuntivo conciso del testo fornito, organizzato gerarchicamente:

**I. Block-Cyclic Data Distribution**

    A.  **Definizione:** I dati sono suddivisi in blocchi.
    B.  **Caratteristiche:**
        1.  **Dimensione blocco:** Parametrizzabile (la dimensione dei blocchi può essere impostata).
        2.  **Disposizione:** Sequenziale e ciclica tra le posizioni.

---
