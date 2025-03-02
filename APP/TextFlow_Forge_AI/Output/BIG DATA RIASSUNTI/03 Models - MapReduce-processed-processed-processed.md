
# Modelli di Programmazione Parallela e MapReduce

I modelli di programmazione parallela astraggono l'architettura sottostante, semplificando lo sviluppo di applicazioni parallele e distribuite.  Offrono *astrazione*, nascondendo i dettagli di basso livello, e *stabilità*, fornendo un'interfaccia standard.  Si distinguono per il livello di astrazione: i modelli di alto livello nascondono i dettagli di implementazione, mentre quelli di basso livello consentono un controllo più fine sull'hardware.  Le strategie di implementazione includono lo sviluppo di nuovi linguaggi, l'uso di annotazioni nel codice e l'integrazione di librerie (quest'ultima la più comune).  Modelli come MapReduce e il passaggio di messaggi forniscono astrazioni per la programmazione parallela, supportati da sistemi come Apache Hadoop e MPI.


## Il Modello MapReduce

MapReduce, sviluppato da Google, è un modello che utilizza il paradigma "dividi e conquista" per elaborare grandi dataset.  Si basa su due funzioni principali:

* **`map (k1, v1) → list(k2, v2)`:**  prende una coppia chiave-valore in input e produce una lista di coppie chiave-valore intermedie.
* **`reduce (k2, list(v2)) → list(v3)`:**  combina tutti i valori intermedi con la stessa chiave intermedia.

Il parallelismo è ottenuto sia nella fase `map` (elaborazione parallela delle chiavi su diversi computer tramite sharding dei dati) che nella fase `reduce` (esecuzione parallela dei reducer su chiavi distinte).  Questa capacità di parallelizzazione consente a MapReduce di scalare da un singolo server a centinaia di migliaia.

---

MapReduce è un modello di programmazione che semplifica lo sviluppo di applicazioni parallele, nascondendo al programmatore i dettagli della parallelizzazione.  Gli sviluppatori definiscono le funzioni `map` e `reduce`, concentrandosi sulla logica di calcolo.

Un *job* MapReduce consiste di codice per le fasi `map` e `reduce`, impostazioni di configurazione e dati di input (su un file system distribuito).  Il job è suddiviso in *task*: i *mapper* eseguono la fase `map` e i *reducer* la fase `reduce`. Applicazioni complesse possono richiedere workflow con più job.

Il sistema MapReduce segue un modello *master-worker*: un nodo utente invia un job al nodo master, che assegna task ai worker. Il master coordina l'intero processo, gestendo mapper e reducer.  Una volta completati tutti i task, il risultato viene restituito all'utente.

Il flusso di elaborazione è il seguente:

1. Un *job descriptor* (contenente le funzioni `map` e `reduce`, la posizione dei dati di input, ecc.) viene inviato al master.
2. Il master avvia mapper e reducer su diverse macchine, distribuendo i dati di input (suddivisi in chunk) ai mapper.
3. Ogni mapper applica la funzione `map` al suo chunk, generando coppie chiave-valore intermedie.
4. Le coppie con la stessa chiave vengono inviate allo stesso reducer.
5. Ogni reducer applica la funzione `reduce` alle coppie con la stessa chiave, producendo un output ridotto.
6. Gli output dei reducer vengono raccolti e costituiscono l'output finale.

Per migliorare le prestazioni, si può aggiungere una fase di *combine*, eseguendo una minireduzione locale sull'output dei mapper prima della trasmissione ai reducer.  Un *combiner*, spesso identico al reducer, aggrega i dati localmente (`combine (k2, list(v2)) → list(v3)`).

Tra le fasi `map` (con eventuale `combine`) e `reduce`, avviene la fase di *shuffle e sort*, che trasferisce, raggruppa per chiave e ordina l'output dei mapper prima di inviarlo ai reducer. Le chiavi intermedie sono scritte localmente su disco prima di essere inviate ai reducer.  Un esempio di applicazione MapReduce è la creazione di un indice invertito, dove la funzione `map` genera coppie `<parola, documentID>` e la funzione `reduce` aggrega gli ID dei documenti per ogni parola.  ![[](_page_15_Figure_4.jpeg)] ![[Pasted image 20250223161820.png|480]]

---

Lo scheduler di MapReduce, una volta completata la fase di mappatura e ordinamento dei dati da parte dei mapper, notifica i reducer.  Questa notifica avvia il processo di recupero delle coppie chiave-valore, già ordinate per partizione, dai mapper ai rispettivi reducer.  In sostanza, i reducer ricevono i dati intermedi, pre-ordinati, necessari per la fase di riduzione.

---
