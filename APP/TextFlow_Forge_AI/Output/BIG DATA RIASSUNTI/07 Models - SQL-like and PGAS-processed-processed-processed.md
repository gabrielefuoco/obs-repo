
## Riepilogo dei Modelli SQL-like per Big Data

I database relazionali, pur essendo ampiamente utilizzati, non scalano orizzontalmente, limitando la gestione di grandi volumi di dati.  L'approccio NoSQL, basato sul modello BASE (Basic Availability, Soft state, Eventual consistency), offre scalabilità orizzontale ma spesso sacrifica la coerenza dei dati e non è ideale per l'analisi.

I sistemi SQL-like rappresentano una soluzione ibrida, combinando l'efficienza di MapReduce con la semplicità di SQL.  Questi sistemi semplificano le query complesse (aggregazioni, selezioni, conteggi) mantenendo la scalabilità di MapReduce, spesso ottimizzando automaticamente le query su grandi dataset.  Apache Hive ne è un esempio.

L'utilizzo di SQL per i big data è favorito dalla sua natura dichiarativa, interoperabilità e approccio data-driven.  Inoltre, i sistemi SQL-like supportano la *query-in-place*, eseguendo le query direttamente sui dati senza spostarli, ottimizzando l'utilizzo dei dati, riducendo la latenza e i costi.

Il **partizionamento dei dati** è cruciale per l'efficienza delle query SQL su big data.  Suddividendo i dati in base a valori di colonna, si riducono i costi di I/O e si accelera l'elaborazione.  Un partizionamento eccessivo, tuttavia, può sovraccaricare il nodo master a causa della gestione di troppi metadati.

---

Il Partitioned Global Address Space (PGAS) è un modello di programmazione parallela che offre un compromesso tra produttività del programmatore e prestazioni elevate.  Utilizza uno spazio di indirizzamento globale condiviso, ma distingue tra accessi locali (a basso costo) e remoti (a costo maggiore), migliorando così la scalabilità su architetture parallele di grandi dimensioni.  Ogni processo, identificato da un *rank*, accede a una porzione partizionata di questo spazio globale, con accesso diretto alla propria memoria locale e accesso remoto tramite API per le altre. ![[Pasted image 20250223163311.png|480]]

Il modello PGAS supporta tre principali modelli di parallelismo:

1. **Single Program Multiple Data (SPMD):**  Molteplici thread eseguono lo stesso programma.
2. **Asynchronous PGAS (APGAS):**  I thread vengono creati dinamicamente, potendo eseguire codice diverso e operare su diverse partizioni dello spazio di indirizzamento.
3. **Parallelismo Implicito:** Il parallelismo è gestito dal sistema a runtime, senza essere esplicitamente definito nel codice.

La gestione della memoria si basa sul modello NUMA (Non-Uniform Memory Access), con una funzione di costo che distingue tra accessi locali (economici) e remoti (costosi).  Il costo dipende dalla posizione del richiedente e dei dati.  Lo spazio di memoria è suddiviso in *places*, corrispondenti ai nodi di calcolo.

La distribuzione dei dati tra i *places* può seguire diversi schemi, tra cui la distribuzione *ciclica*.  Il partizionamento dei dati può essere effettuato con metodi come il *Block*, che suddivide i dati in blocchi di dimensione uguale e consecutiva, distribuiti tra i nodi.

---

La tecnica *block-cyclic* distribuisce i dati su più posizioni di memoria o processori suddividendoli in blocchi di dimensione definibile dall'utente.  Questi blocchi vengono poi assegnati ciclicamente alle diverse posizioni, garantendo una distribuzione più uniforme rispetto ad altre tecniche di distribuzione dei dati.  La dimensione del blocco è un parametro fondamentale che influenza l'efficienza della distribuzione.

---
