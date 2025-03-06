
**Schema Riassuntivo di Apache Pig**

**1. Introduzione ad Apache Pig**

*   Framework di flusso dati di alto livello per MapReduce su Hadoop.
*   Utilizza un linguaggio simile a SQL (Pig Latin).
*   Colma il divario tra SQL dichiarativo e MapReduce procedurale.
*   Converte query Pig Latin in piani di esecuzione MapReduce.

**2. Caratteristiche Principali**

*   **Linguaggio Pig Latin:**
    *   Astrazione di livello medio rispetto a Hadoop.
    *   Semplifica lo sviluppo rispetto alla scrittura di codice MapReduce complesso.
*   **Parallelismo:**
    *   **Parallelismo dei dati:** suddivisione ed elaborazione parallela dei dati.
    *   **Parallelismo dei task:** esecuzione parallela di più query sugli stessi dati.
*   **Esecuzione Multi-Query:**
    *   Elabora interi script o batch di istruzioni contemporaneamente.

**3. Utilizzo Comune**

*   Sviluppo di query sui dati.
*   Analisi dati semplici.
*   Applicazioni ETL (Estrazione, Trasformazione e Caricamento).
*   Raccolta dati da diverse fonti (stream, HDFS, file).
*   Aziende utilizzatrici: LinkedIn, PayPal, Mendeley.

**4. Concetti Chiave**

*   **Modello Dati:**
    *   Modello dati nidificato per dati complessi e non normalizzati.
    *   Tipi scalari: `int`, `long`, `double`, `chararray`, `bytearray`.
    *   Tipi complessi:
        *   **Map:** Matrice associativa (stringa come chiave, valore di qualsiasi tipo).
        *   **Tuple:** Elenco ordinato di elementi (campi) di qualsiasi tipo.
        *   **Bag (Relazione):** Raccolta di tuple (simile a una tabella relazionale, ma senza requisiti di campi uguali).
*   **Ottimizzazione delle Query:**
    *   Traduzione in job MapReduce ottimizzati automaticamente.
    *   Regole di ottimizzazione: rimozione istruzioni inutilizzate, applicazione filtri durante il caricamento.

---

**I. Ottimizzazione in Pig**

    *   A. Tipi di Ottimizzazione:
        *   1.  Logica: Riorganizza il grafo di flusso dati logico per una valutazione più efficiente.
        *   2.  Fisica: Traduce il grafo di flusso dati logico in un piano di esecuzione fisico (es. MapReduce).
    *   B. Piano Logico:
        *   1.  Creato per ogni bag definito dall'utente.
        *   2.  L'elaborazione inizia solo con il comando `STORE`.
        *   3.  Trasformato in un piano fisico per l'esecuzione.
    *   C. Esecuzione Pigra:
        *   1.  Permette pipelining in memoria.
        *   2.  Consente altre ottimizzazioni.

**II. Architettura di Pig**

    *   A. Componenti Principali:
        *   1.  Parser:
            *   a. Gestisce istruzioni Pig Latin.
            *   b. Controlla errori di sintassi e tipo di dati.
            *   c. Produce un DAG (grafo aciclico diretto) che rappresenta gli operatori logici e il flusso di dati.
        *   2.  Optimizer:
            *   a. Applica ottimizzazioni al DAG.
            *   b. Obiettivo: migliorare la velocità delle query.
            *   c. Esempi di ottimizzazioni: split, merge, projection, pushdown, transform e reorder.
            *   d. Pushdown e projection omettono dati/colonne non necessari.
        *   3.  Compiler:
            *   a. Genera una sequenza di job MapReduce dall'output dell'optimizer.
            *   b. Include ulteriori ottimizzazioni (es. riorganizzazione dell'ordine di esecuzione).
        *   4.  Execution Engine:
            *   a. Esegue i job MapReduce sul runtime Hadoop.
            *   b. Output visualizzato con `DUMP` o salvato in HDFS con `STORE`.

**III. Fondamenti di Programmazione Pig Latin**

    *   A. Bag:
        *   1.  Collezione di tuple.
        *   2.  Creazione:
            *   a. Utilizzo di tipi di dati nativi (semplici e complessi).
            *   b. Caricamento dati dal file system.
    *   B. Istruzioni Comuni:
        *   1.  `FILTER`: Seleziona tuple basate su una condizione.
        *   2.  `JOIN`: Esegue join (interno/esterno) basati su valori di campo comuni.
        *   3.  `FOREACH`: Genera trasformazioni di dati basate sulle colonne.
            *   a. Spesso accoppiato con `GENERATE` per lavorare con le colonne.
        *   4.  `STORE`: Salva i risultati nel file system.
    *   C. UDF (User Defined Functions):
        *   1.  Definite dal programmatore.
        *   2.  File JAR registrato per l'uso nello script.
        *   3.  Alias assegnati alle UDF.

---

**I. Analisi del Sentiment con Apache Pig**

    A.  **Approccio Basato su Dizionario:**
        *   Utilizza un dizionario di parole associate a sentiment positivo o negativo.
        *   Calcola il sentiment di un testo sommando i punteggi delle parole positive e negative.
        *   Valuta il sentiment medio del testo.

    B.  **Estensione delle Funzionalità con UDF (User Defined Functions):**
        *   Definizione di UDF per elaborazioni personalizzate (es. rimozione della punteggiatura).
        *   Esempio: UDF **PROCESS** per la pre-elaborazione del testo.
        *   Implementazione del metodo *exec* in Java per aggiungere funzionalità.
        *   Registrazione della UDF con un alias (es. PROCESS).

    C.  **Caricamento e Preparazione dei Dati:**
        *   Caricamento dei dati delle recensioni da HDFS (file CSV delimitato da tabulazione).
        *   Caricamento del dizionario dei sentiment delle parole.
        *   Definizione dello schema dei dati tramite colonne denominate.
        *   Tokenizzazione ed elaborazione di ogni riga di dati.

    D.  **Elaborazione del Testo con Pig Latin:**
        *   Utilizzo di **FOREACH** per applicare la UDF e tokenizzare le righe.
        *   Utilizzo di **FLATTEN** per appiattire l'array di token.
        *   Creazione di triple `<id recensione, testo, parola>` e memorizzazione in un bag chiamato *words*.

    E.  **Correlazione con il Dizionario e Assegnazione dei Punteggi:**
        *   Identificazione delle corrispondenze tra le parole delle recensioni e le parole del dizionario.
        *   Memorizzazione dei risultati in un bag chiamato *matches*.
        *   Assegnazione del punteggio a ciascuna parola in base al dizionario.
        *   Creazione di triple `<id recensione, testo, valutazione>` e memorizzazione in un bag chiamato *matches_rating*.

    F.  **Aggregazione e Calcolo del Sentiment Medio:**
        *   Raggruppamento per `<id recensione, testo>` per raccogliere tutte le valutazioni (bag *group_rating*).
        *   Utilizzo di **AVG** per aggregare le valutazioni delle parole per ogni recensione.
        *   Calcolo della valutazione finale come media dei punteggi dei token.
        *   Memorizzazione del bag di output *avg_ratings* in un file su HDFS.
        *   Output del group by nella forma `((id, testo), {(id, testo, valutazione)})`.
        *   Accesso posizionale: la chiave `((id, testo))` è `$0`, le triple associate sono accessibili tramite `$1`.
        *   `$1.$0` è l'id, `$1.$1` è il testo.

---

Ecco uno schema riassuntivo del testo fornito:

**I. Struttura dei Dati e Valutazione**

    *   A. Campi:
        *   1.  Campo Testo
        *   2.  Campo Valutazione: `$1.$2`

**II. Calcolo della Media delle Valutazioni**

    *   A. Funzione: `AVG($1.$2)`
    *   B. Esempio di Calcolo:
        *   1.  Recensione 1: `(3, 4)`
        *   2.  Recensione 2: `(-3, -2)`

---
