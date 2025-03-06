
## Schema Riassuntivo: Sistemi di Streaming

**1. Introduzione ai Sistemi di Streaming**
    *   Definizione: Motore di elaborazione dati per **dati illimitati**.
    *   Distinzione tra dati:
        *   Dati limitati: Dataset di dimensioni finite.
        *   Dati illimitati: Dataset di dimensioni (teoricamente) infinite.

**2. Flussi di Dati (Stream)**
    *   Definizione: Insieme concettualmente infinito e in continua crescita di elementi/eventi.
    *   Modello: **Modello push** (pubblica/iscriviti), controllato dalla sorgente.
    *   Importanza del tempo: Necessità di considerare tempo di produzione e tempo di restituzione dei dati.

**3. Tipi di Tempo**
    *   Tempo dell'evento: Tempo di produzione dell'elemento dati.
    *   Tempo di ingestione: Tempo di sistema in cui viene ricevuto l'elemento dati.
    *   Tempo di elaborazione: Tempo di sistema in cui viene elaborato l'elemento dati.
    *   Relazione: Tipicamente, questi tre tempi non coincidono.

**4. Serie Temporali**
    *   Definizione: Serie di punti dati indicizzati in ordine temporale.
    *   Caratteristica comune: Sequenza acquisita in punti temporali successivi equidistanti.

**5. Modelli di Aggiornamento Vettoriale**
    *   Vettore: **a = (a₁, …, aₙ)**, inizialmente **aᵢ = 0** per tutti gli i.
    *   Modello cassa registratore:
        *   Aggiornamento: **⟨i, c⟩**
        *   Effetto: **aᵢ** viene incrementato di un numero *positivo* **c**.
    *   Modello tornello:
        *   Aggiornamento: **⟨i, c⟩**
        *   Effetto: **aᵢ** viene incrementato di un numero (*possibilmente negativo*) **c**.

**6. Algoritmi di Streaming**
    *   Definizione: Algoritmi per elaborare flussi di dati con accesso limitato a memoria e tempo di elaborazione per elemento.
    *   Caratteristiche: Input presentato come sequenza di elementi, esaminabile in poche passate (tipicamente una).

**7. Approcci all'Elaborazione di Flussi**
    *   Elaborazione agnostica al tempo
    *   Elaborazione approssimativa
    *   Finestramento per tempo di elaborazione
    *   Finestramento per tempo dell'evento

**8. Elaborazione Agnostica al Tempo**
    *   Utilizzo: Quando il tempo è irrilevante.
    *   Esempi:
        *   Filtraggio: Esaminare ogni record al suo arrivo e filtrarlo in base a criteri specifici.
        *   Inner Join: Unire due sorgenti di dati illimitate.

---

## Schema Riassuntivo

**1. Join su Dati Illimitati**
    *   Si bufferizza il primo valore da una sorgente.
    *   Si emette il record unito quando arriva il secondo valore dall'altra sorgente.
    *   Non c'è un elemento temporale nella logica.
    *   I join vengono prodotti quando vengono osservati elementi corrispondenti da entrambe le sorgenti.

**2. Elaborazione Approssimativa**
    *   Si basa su algoritmi che producono una risposta approssimativa.
    *   Utilizza riepiloghi o "schizzi" del flusso di dati.
    *   Esempi:
        *   Top-N approssimativo
        *   Streaming k-means

**3. Windowing**
    *   Suddivisione della sorgente dati (illimitata o limitata) in blocchi finiti per l'elaborazione.
    *   Modelli principali:
        *   Finestre fisse
        *   Finestre scorrevoli
        *   Sessioni

**4. Finestramento per Tempo di Elaborazione**
    *   Il sistema mette in buffer i dati in arrivo in finestre finché non è trascorso un certo tempo di elaborazione.
    *   Esempio: mettere in buffer i dati per *n* minuti di tempo di elaborazione, dopodiché tutti i dati di quel periodo di tempo vengono inviati per l'elaborazione.
    *   I dati vengono raccolti in finestre in base all'ordine in cui arrivano nella pipeline.

**5. Finestramento per Tempo dell'Evento**
    *   Utilizzato quando è necessario osservare una sorgente dati in blocchi finiti che riflettono i tempi in cui si sono verificati tali eventi.
    *   Più complesso del finestramento per tempo di elaborazione (richiede più buffering dei dati).
    *   Problema di completezza: spesso non abbiamo modo di sapere quando abbiamo visto tutti i dati per una determinata finestra.
    *   Tipi:
        *   Finestre fisse: I dati vengono raccolti in finestre fisse in base ai tempi in cui si sono verificati.
        *   Finestre di sessione: I dati vengono raccolti in finestre di sessione in base al momento in cui si sono verificati.

**6. Operatori di Flusso di Base**
    *   Aggregazione con finestre:
        *   Esempio: velocità media.
        *   Somma degli accessi URL.
        *   Punteggio giornaliero più alto.
    *   Join con finestre:
        *   Osservazioni correlate nell'intervallo di tempo.
        *   Esempio: temperatura nel tempo.

**7. Elaborazione di Eventi Complessi (CEP)**
    *   Rilevazione di pattern in un flusso.
    *   Evento complesso = sequenza di eventi.
    *   Definito usando condizioni logiche e temporali:
        *   Logiche: valori e combinazioni di dati.
        *   Temporali: entro un determinato periodo di tempo.
    *   Esempio di condizione: `SEO(A, B, C) CON A.Temp > 23°C && B.Station = A.Station && B.Temp < A.Temp && C.Station = A.Station && A.Temp - C.Temp > 3`
    *   Eventi compositi costruiti da: `SEQ`, `AND`, `OR`, `NEG`, ...
    *   Esempio di evento composito: `SEQ(e1, e2) -> (e1, t1) ^ (e2, t2)` con `t1 ≤ t2 ^ e1, e2 ∈ W`
    *   Implementato costruendo un NFA.
    *   Esempio: `SEQ(A, B, C)`

---

## Schema Riassuntivo sull'Elaborazione di Flussi di Grandi Dimensioni

**1. Requisiti dell'Elaborazione di Flussi di Grandi Dimensioni**
    *   Mantenere i dati in movimento: Architettura di streaming.
    *   Accesso dichiarativo: Esempio: StreamSQL, CQL.
    *   Gestire le imperfezioni: Elementi in ritardo, mancanti, non ordinati.
    *   Risultati prevedibili: Coerenza, tempo dell'evento.
    *   Integrare dati memorizzati e dati di streaming: Flusso ibrido e batch.
    *   Sicurezza e disponibilità dei dati: Tolleranza ai guasti, stato persistente.
    *   Partizionamento e scaling automatico: Elaborazione distribuita.
    *   Elaborazione e risposta istantanea.

**2. Elaborazione di Big Data: Limiti dei Database Tradizionali**
    *   Dati non (completamente) strutturati: Non adatti ai database relazionali.
    *   Necessità di analisi avanzate: Oltre a semplici operazioni di selezione, proiezione e unione.

**3. MapReduce: Una Prima Soluzione (con Limiti)**
    *   Ottimo per grandi quantità di dati statici.
    *   Inefficace per flussi: Adatto solo a finestre di grandi dimensioni.
    *   Dati statici: Alta latenza, bassa efficienza.

**4. Mini-Batch: Un Approccio Semplice ma Limitato**
    *   Facile da implementare, coerenza e tolleranza ai guasti.
    *   Difficoltà nella gestione del tempo dell'evento e delle sessioni.

**5. Architettura di Streaming True: Elementi Fondamentali**
    *   **Programma:** Directed Acyclic Graph (DAG) di operatori e flussi intermedi.
    *   **Operatore:** Unità di calcolo con stato interno.
    *   **Flussi Intermedi:** Flussi logici di record che collegano gli operatori.

**6. Architettura di Streaming True: Trasformazioni**
    *   **Trasformazioni di base:** `Map`, `Reduce`, `Filter`, Aggregazioni.
    *   **Trasformazioni di flusso binarie:** `CoMap`, `CoReduce`.

**7. Architettura di Streaming True: Gestione del Tempo**
    *   **Semantica delle finestre:** Policy basate su tempo, conteggio di elementi o delta temporale.
    *   **Operatori di flusso binari temporali:** `Join`, `Cross`.
    *   **Supporto nativo per le iterazioni.**

**8. Architettura di Streaming True: Avanzamento della Completezza Temporale**
    *   Tracciamento dell'avanzamento della completezza temporale: `F(P) → E`
        *   `P`: Punto nel tempo di elaborazione.
        *   `E`: Punto nel tempo dell'evento (fino al quale si ritengono completi i dati).

**9. Watermark: Gestione della Completezza Temporale**
    *   **Watermark perfetti:** Basati su conoscenza perfetta dei dati di input.
    *   **Watermark euristici:** Stime dell'avanzamento basate su informazioni disponibili.

**10. Lezioni Apprese dal Batch**
    *   Ripetizione del calcolo in caso di fallimento: Come una transazione.
    *   Tasso di transazione costante.

---

## Schema Riassuntivo: Esecuzione di Streaming e Snapshot Naive

**I. Applicazione dei Principi allo Streaming**

**II. Esecuzione di Snapshot - Approccio Naive**

    *   **A. Descrizione:** (Si riferisce alle immagini fornite, che presumibilmente illustrano il processo di snapshot naive)
    *   **B. Rappresentazione Visiva:**
        *   Immagine 1: `![[Pasted image 20250223165803.png|361]]`
        *   Immagine 2: `![[Pasted image 20250223165815.png|413]]`

---
