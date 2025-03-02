
## Sistemi di Streaming e Elaborazione di Dati Illimitati

I sistemi di streaming elaborano **dati illimitati**, ovvero dataset di dimensioni teoricamente infinite e in continua crescita, rappresentati da un flusso continuo di elementi/eventi.  Il modello di elaborazione è tipicamente **push**, gestito dalla sorgente dati (modello pubblica/iscriviti).  La gestione del **tempo** è cruciale: si distinguono il **tempo dell'evento** (produzione del dato), il **tempo di ingestione** (ricezione del dato nel sistema) e il **tempo di elaborazione** (elaborazione del dato).  Questi tempi generalmente non coincidono. ![[Pasted image 20250223164612.png|239]]

## Modelli di Aggiornamento Dati

Due modelli descrivono l'aggiornamento di un vettore `a = (a₁, …, aₙ)` (inizialmente con tutti `aᵢ = 0`) da un flusso di dati:

* **Modello cassa registratore:** Ogni aggiornamento `⟨i, c⟩` incrementa `aᵢ` di un valore positivo `c`.
* **Modello tornello:** Ogni aggiornamento `⟨i, c⟩` incrementa `aᵢ` di un valore `c` (possibilmente negativo).

## Algoritmi di Streaming

Gli algoritmi di streaming elaborano flussi di dati con accesso limitato a memoria e tempo di elaborazione per elemento, tipicamente in una sola passata.  Gli approcci includono:

* **Elaborazione agnostica al tempo:** Il tempo è irrilevante.  Esempi includono il filtraggio e l'inner join.  Ad esempio, filtrare i log del traffico web per un dominio specifico può essere fatto esaminando ogni record all'arrivo, indipendentemente dal tempo dell'evento. ![[Pasted image 20250223164652.png]]
* **Elaborazione approssimativa:** (non dettagliata nel testo fornito)
* **Finestramento per tempo di elaborazione:** (non dettagliata nel testo fornito)
* **Finestramento per tempo dell'evento:** (non dettagliata nel testo fornito)

---

Il documento descrive tecniche di elaborazione di flussi di dati illimitati, focalizzandosi su tre concetti chiave: join, windowing ed elaborazione di eventi complessi.

**1. Inner Join su Dati Illimitati:**  Un *inner join* su sorgenti illimitate funziona bufferizzando i dati da ogni sorgente fino a quando non si trova una corrispondenza tra i due flussi.  Solo allora viene emesso il record unito.  ![|543](_page_12_Figure_1.jpeg) illustra questo processo.

**2. Elaborazione Approssimativa:** Questa tecnica genera risposte approssimative basate su un riepilogo dei dati, anziché sull'elaborazione completa. Esempi includono il *Top-N approssimativo* e lo *streaming kmeans*. ![[Pasted image 20250223164730.png|524]] mostra un esempio visivo.

**3. Windowing:**  Questa tecnica suddivide il flusso di dati (limitato o illimitato) in finestre di dimensione finita per l'elaborazione.  Esistono tre tipi principali:

* **Finestre fisse:**  Finestre di dimensione fissa predefinita. ![](_page_19_Figure_1.jpeg)
* **Finestre scorrevoli:** Finestre di dimensione fissa che si spostano nel tempo.  (Descrizione non esplicita nel testo, ma implicita nella tipologia di windowing)
* **Finestre di sessione:** Finestre definite da periodi di attività, con pause che delimitano le finestre. ![[](_page_20_Figure_1.jpeg)

Il *finestramento per tempo di elaborazione* suddivide i dati in base al tempo trascorso dall'inizio dell'elaborazione. ![[|510](_page_17_Figure_1.jpeg) mostra un esempio. Il *finestramento per tempo dell'evento* suddivide i dati in base al tempo in cui gli eventi si sono verificati, presentando però problemi di completezza dati. ![[|435](_page_15_Figure_5.jpeg) fornisce una panoramica generale dei tipi di windowing.

**4. Operatori di Flusso di Base:**  Gli operatori di flusso includono:

* **Aggregazione con finestre:** Calcola aggregati (es. media, somma) su finestre di dati.  ![[](_page_22_Figure_1.jpeg) mostra un esempio.
* **Join con finestre:** Esegue join su dati all'interno di finestre temporali. ![[|159](_page_21_Figure_8.jpeg)] mostra un esempio.

**5. Elaborazione di Eventi Complessi:**  Questa tecnica identifica pattern in un flusso di eventi, dove un evento complesso è una sequenza di eventi definita da condizioni logiche e temporali.  ![|220](_page_23_Picture_6.jpeg) fornisce un esempio visivo.  Gli eventi complessi possono essere costruiti usando operatori come `SEQ`, `AND`, `OR`, `NEG`.  Un esempio di definizione di un evento complesso è: `SEO(A, B, C) CON A.Temp > 23°C && B.Station = A.Station && B.Temp < A.Temp && C.Station = A.Station && A.Temp - C.Temp > 3`.

---

## Elaborazione di Flussi di Grandi Dimensioni: Dall'approccio Batch allo Streaming

Questo documento tratta l'elaborazione di grandi quantità di dati, confrontando l'approccio tradizionale *batch* con l'architettura *streaming*.  L'elaborazione *batch*, tipicamente basata su MapReduce, risulta inefficiente per i flussi di dati in tempo reale a causa dell'alta latenza e della gestione statica dei dati.  `SEQ(e1, e2) -> (e1, t1) ^ (e2, t2)` con `t1 ≤ t2 ^ e1, e2 ∈ W` illustra un esempio di sequenza implementata tramite NFA. ![[|296](_page_24_Figure_6.jpeg)]

L'elaborazione di flussi di grandi dimensioni richiede:  mantenimento dei dati in movimento (architettura di streaming); accesso dichiarativo (es. StreamSQL, CQL); gestione di dati in ritardo, mancanti o non ordinati; risultati prevedibili (coerenza, tempo dell'evento); integrazione di dati memorizzati e di streaming; sicurezza e disponibilità (tolleranza ai guasti, stato persistente); partizionamento e scaling automatico; ed elaborazione e risposta istantanea. ![[|483](_page_27_Figure_1.jpeg)]

L'approccio *mini-batch*, sebbene semplice da implementare e con buona coerenza e tolleranza ai guasti, presenta difficoltà nella gestione del tempo dell'evento e delle sessioni. ![[|456](_page_28_Figure_2.jpeg)] ![[|327](_page_29_Figure_4.jpeg)]

Un'architettura *streaming* vera e propria si basa su un DAG di operatori e flussi intermedi, dove ogni operatore include calcolo e stato.  Include trasformazioni di base (Map, Reduce, Filter, Aggregazioni), trasformazioni binarie (CoMap, CoReduce), semantiche di finestre flessibili (Tempo, Conteggio, Delta), operatori binari temporali (Join, Cross) e supporto nativo per le iterazioni.  La cattura dell'avanzamento della completezza del tempo dell'evento è definita da una funzione F(P) → E, che mappa un punto nel tempo di elaborazione ad un punto nel tempo dell'evento, indicando fino a quale punto sono stati osservati tutti gli input. ![[|223](_page_31_Figure_7.jpeg)]

Si distinguono *watermark perfetti* (con conoscenza completa dei dati) e *watermark euristici* (stime approssimative in assenza di conoscenza completa).  Infine, si evidenzia che, a differenza dell'elaborazione batch, dove un fallimento può essere gestito rieseguendo il calcolo come transazione, l'elaborazione streaming richiede un approccio più sofisticato per garantire la continuità e la correttezza dei risultati. ![[|350](_page_33_Figure_1.jpeg)]

---

Il testo descrive un approccio "naive" all'esecuzione di streaming, focalizzandosi sulla tecnica degli snapshot.  Le immagini allegate (`![Pasted image 20250223165803.png|361]` e `![Pasted image 20250223165815.png]`) illustrano probabilmente il processo di creazione e utilizzo di questi snapshot per simulare un'esecuzione streaming.  L'approccio è definito "naive" suggerendo l'esistenza di metodi più sofisticati ed efficienti per gestire l'esecuzione streaming, ma il testo non li descrive.  In sostanza, il testo introduce il concetto di snapshot come metodo base per l'esecuzione streaming, senza approfondire le sue limitazioni o alternative.

---
