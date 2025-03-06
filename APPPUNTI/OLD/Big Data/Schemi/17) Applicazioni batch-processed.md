
**Schema Riassuntivo: Scoperta di Pattern di Mobilità Utente da Post di Flickr Geotaggati**

**1. Introduzione: Applicazione per la Scoperta di Pattern di Mobilità**
    *   Obiettivo: Scoprire le traiettorie utente più frequenti attraverso punti di interesse (PoI) a Roma.

**2. Concetti Chiave**
    *   **PoI (Point of Interest):** Località utile o interessante (es. attrazione turistica).
    *   **RoI (Region of Interest):** Confini delle aree dei PoI, usati per facilitare l'abbinamento delle traiettorie.
    *   **Traiettoria:** Sequenza di RoI, che rappresenta uno schema di movimento nel tempo.
    *   **Traiettoria Frequente:** Sequenza di RoI frequentemente visitate dagli utenti.

**3. Dati di Input: Post Geotaggati da Flickr**
    *   Metadati Utili:
        *   Descrizione testuale
        *   Parole chiave associate
        *   Latitudine/Longitudine
        *   ID Utente
        *   Timestamp

**4. Implementazione con Apache Spark**
    *   Linguaggio: Scala
    *   Classi Principali:
        *   **SingleTrajectory:** `<PoI, timestamp>` (Singolo punto in una traiettoria)
            *   Definisce metodi *equals* e *hashCode* per il confronto.
        *   **UserTrajectory:** `<user_id, date>` (Utente in una data specifica)
            *   Definisce metodi *equals* e *hashCode* per il confronto.
    *   Fasi Principali:
        *   Lettura del dataset JSON in un DataFrame Spark.
        *   Filtraggio dei dati non rilevanti (coordinate GPS non valide, località fuori Roma).
        *   Utilizzo di **KMLUtils** per leggere file KML e convertire le coordinate dei PoI in una mappa Scala (nome PoI -> coordinate del poligono).
        *   Calcolo delle traiettorie degli utenti usando l'algoritmo **FPGrowth**.
        *   Restituzione degli itemset frequenti e delle regole di associazione (traiettorie ottenute).

---

## Schema Riassuntivo: Estrazione di Regole di Associazione da Traiettorie GPS

**I. Introduzione**
    *   Estrazione di regole di associazione da dati di traiettorie GPS.
    *   Implementazione con Spark e Hadoop MapReduce.
    *   Dati Flickr filtrati e processati.

**II. Fase 1: Filtraggio dei Dati (Spark e Hadoop)**
    *   **Obiettivo:** Selezionare dati rilevanti.
    *   **Filtri:**
        *   Validità delle Coordinate GPS (longitudine e latitudine definite).
        *   Posizione Geografica: Post pubblicati a Roma.
            *   Funzione: `filterIsInRome`
            *   Libreria Java: **Spatial4j**
            *   Classe di utilità: **GeoUtils**
        *   Ulteriori filtri: accuratezza, latitudine, longitudine, `owner.id`, `dateTaken`.

**III. Fase 2: Estrazione di Regole di Associazione con FPGrowth (Spark)**
    *   **Algoritmo:** FPGrowth (da **MLlib**).
    *   **Trasformazione dei dati:**
        *   Metodo: `mapCreateTrajectory`
        *   Output: Tuple `<UserTrajectory, SingleTrajectory>`.
            *   `SingleTrajectory`: ID utente e PoI visitati.
    *   **Preparazione dei dati di transazione:**
        *   Metodo: `computeTrajectoryUsingFPGrowth`
        *   Dati: `RDD[Array[String]]`
        *   Passaggi:
            *   Rimozione traiettorie vuote.
            *   Raggruppamento traiettorie dello stesso utente.
            *   Trasformazione in itemset unici (funzione `distinct`).
    *   **Esecuzione FPGrowth:**
        *   Parametri: `minSupport`, `minConfidence`.
        *   Output: Itemset frequenti e regole di associazione.
    *   **Esempi:**
        *   Dati di transazione preparati:
            ```
            (user 35716709@N04, day Nov 20, 2016,Set(poi villaborghese, timestamp Nov 20, 2016))
            (user 61240032@N05, day Nov 29, 2016,Set(poi stpeterbasilica, timestamp Nov 29, 2016, poi colosseum, timestamp Nov 29, 2016, poi stpeterbasilica, timestamp Nov 29, 2016))
            (user 99366715@N00, day Nov 30, 2016,Set(poi piazzanavona, timestamp Nov 30, 2016))
            (user 52714236@N02, day Nov 30, 2016,Set(poi trastevere, timestamp Nov 30, 2016, poi piazzanavona, timestamp Nov 30, 2016, poi romanforum, timestamp Nov 30, 2016))
            (user 92919639@N03, day Dec 10, 2016,Set(poi romanforum, timestamp Dec 10, 2016))
            ```
        *   `transactions` risultanti:
            ```
            Array(villaborghese)
            Array(stpeterbasilica, colosseum)
            Array(piazzanavona)
            Array(trastevere, piazzanavona, romanforum)
            Array(romanforum)
            …
            ```
        *   Itemset frequenti:
            ```
            {campodefiori}: 388
            …
            {piazzadispagna}: 211
            {piazzadispagna,colosseum}: 34
            {piazzadispagna,pantheon}: 51
            {piazzadispagna,trevifontain}: 32
            ```
        *   `minSupport` = 0.01

**IV. Fase 3: Implementazione con Hadoop MapReduce**
    *   **Componenti:**
        *   **DataMapperFilter (Mapper):**
            *   Filtraggio dei dati.
            *   Riutilizzo classi: `GeoUtils`, `KMLUtils`.
            *   Classe di utilità: **Flickr** (gestione dati Flickr).
            *   Metodo `setup`: Inizializzazione `romeShape` e `shapeMap`.
            *   Metodo `map`: Analisi JSON, applicazione filtri (`filterIsGPSValid`, `filterIsInRome`), scrittura ID utente e oggetto Flickr arricchito con RoI.
        *   **DataReducerByDay (Reducer):**
            *   Estrazione RoI per mining traiettorie.
            *   Metodo `concatenateLocationsByDay`: Costruzione sequenza RoI visitate per giorno (ordinamento per data).
            *   Metodo `reduce`: Generazione lista RoI per giorno.
        *   **Programma principale:**
            *   Combinazione mapper e reducer.
            *   Algoritmo FPGrowth (Apache Mahout).
    *   **Flusso di lavoro:**
        *   Mapper: Pre-processing, suddivisione dati, conta occorrenze item.
        *   Reducer: Aggregazione risultati mapper, identificazione itemset frequenti.
        *   FPGrowth parallelo: Ottimizzazione calcolo itemset frequenti.

---
