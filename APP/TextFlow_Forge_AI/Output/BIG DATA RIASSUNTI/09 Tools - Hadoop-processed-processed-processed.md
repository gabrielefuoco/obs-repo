
## Riepilogo degli Strumenti di Programmazione e di Apache Hadoop

Questo documento descrive le caratteristiche principali degli strumenti di programmazione, con particolare attenzione ad Apache Hadoop.

### Livelli di Astrazione e Parallelismo negli Strumenti di Programmazione

Gli strumenti di programmazione si distinguono per il livello di astrazione:

* **Basso livello:** accesso diretto alle API e istruzioni di basso livello, potente ma complesso.
* **Medio livello:** utilizzo di un insieme limitato di costrutti, nascondendo dettagli di basso livello.
* **Alto livello:** utilizzo di interfacce di alto livello (es. IDE visivi), astraendo completamente l'architettura sottostante.

Il parallelismo può essere:

* **Parallelismo dati:** stesso codice su diversi dati.
* **Parallelismo task:** diversi task eseguiti in parallelo.


### Apache Hadoop: Un Framework MapReduce

Apache Hadoop è un framework open-source per applicazioni data-intensive scalabili, implementando il modello MapReduce in linguaggi come Java e Python.  Abstrae i problemi del calcolo distribuito (località dati, bilanciamento carico, tolleranza ai guasti, risparmio banda).

### Altri Framework MapReduce

Esistono altri framework MapReduce, come:

* **Phoenix++:**  basato su C++, per chip multi-core e multi-processori a memoria condivisa, gestisce thread, partizione dati, pianificazione task e tolleranza ai guasti.
* **Sailfish:** utilizza la trasmissione batch e *I-files* per l'aggregazione efficiente dei dati.


### Caratteristiche di Apache Hadoop

Hadoop presenta le seguenti caratteristiche:

* **Elaborazione batch:** efficiente per l'elaborazione batch, ma inefficiente per applicazioni iterative.
* **Community open-source:** ampia community che garantisce supporto e aggiornamenti.
* **Basso livello di astrazione:** API potenti ma non user-friendly, richiedono una profonda conoscenza del sistema, ma offrono maggiore efficienza.
* **Parallelismo dati:** i dati vengono partizionati ed elaborati in parallelo.
* **Tolleranza ai guasti:** meccanismi di checkpoint e ripristino per alta affidabilità.


### Moduli di Hadoop

Hadoop include diversi moduli:

* **Hadoop Distributed File System (HDFS):** file system distribuito con tolleranza ai guasti e ripristino automatico.
* **Yet Another Resource Negotiator (YARN):** framework per la gestione delle risorse del cluster e la pianificazione dei job.
* **Hadoop Common:** librerie e utility di supporto.

---

[nessuna risposta dall'API]
---

Questo documento descrive l'implementazione di un'applicazione MapReduce in Java, focalizzandosi su tre classi principali: `Mapper`, `Reducer` e `Driver`.

**1. Classi Mapper e Reducer:**

*   `Mapper<KEYIN, VALUEIN, KEYOUT, VALUEOUT>`:  Prende in input una coppia chiave-valore (`KEYIN`, `VALUEIN`), processa ogni coppia nel metodo `map()`, e produce coppie chiave-valore di output (`KEYOUT`, `VALUEOUT`).  I metodi `setup()` e `cleanup()` vengono eseguiti rispettivamente all'inizio e alla fine del task.
*   `Reducer<KEYIN, VALUEIN, KEYOUT, VALUEOUT>`: Riceve in input una chiave (`KEYIN`) e un insieme iterabile di valori (`VALUEIN`) associati a quella chiave. Il metodo `reduce()` elabora questi valori e produce coppie chiave-valore di output (`KEYOUT`, `VALUEOUT`).  Anche qui sono presenti metodi `setup()` e `cleanup()`.

**2. Classe Driver:**

La classe `Driver` configura il job MapReduce, specificando il nome del job, i tipi di dati di input e output, le classi `Mapper` e `Reducer`, e altri parametri.  L'oggetto `Context` permette l'interazione con Hadoop.

**3. Ordinamento Secondario:**

Hadoop ordina le tuple intermedie per chiave prima di inviarle al reducer. L'ordinamento secondario, utilizzando una chiave composita (es. `<chiave_primaria, chiave_secondaria>`), permette un maggiore controllo tramite un *partitioner*, un *comparator* e un *group comparator* personalizzati.  Questo consente di partizionare, ordinare e raggruppare le tuple prima dell'elaborazione del reducer.

**4. Creazione di un Indice Inverso:**

Un esempio di applicazione MapReduce è la creazione di un indice inverso per un insieme di documenti.  `MapTask` (mapper) analizza ogni documento, emettendo coppie `<word, documentID:numberOfOccurrences>`.  `CombineTask` (combiner opzionale) aggrega le occorrenze per parola e documento.  `ReduceTask` (reducer) genera l'indice inverso finale: `<word, List(documentID:numberOfOccurrences)>`.  ![[](_page_18_Figure_4.jpeg)] ![[|432](_page_19_Figure_3.jpeg)]

**5. Configurazione del Job:**

La configurazione di un job MapReduce richiede la specifica delle classi `mapper`, `combiner` (opzionale) e `reducer`.

---

Il processo MapReduce richiede la definizione di tre elementi cruciali:

* **Formati chiave/valore (Input/Output):**  Si devono specificare i formati chiave-valore sia per i dati di input che per quelli di output.  Questi formati definiscono il tipo di dati elaborati in ogni fase.

* **Percorsi di input/output:**  È necessario indicare i percorsi nel file system distribuito (es. HDFS) che contengono i dati di input e quelli in cui verranno salvati i dati di output.

---
