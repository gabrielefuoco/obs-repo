
## Riassunto: Modelli per Sistemi Exascale

I sistemi exascale, pur promettenti, presentano sfide significative nella progettazione e implementazione, tra cui scalabilità, latenza di rete e affidabilità.  La gestione di enormi volumi di dati richiede algoritmi altamente scalabili e modelli di programmazione avanzati in grado di gestire milioni o miliardi di thread concorrenti.

Un modello di programmazione exascale efficace deve includere: accesso parallelo ai dati, resilienza ai guasti, comunicazione locale guidata dai dati, elaborazione su gruppi limitati di core, sincronizzazione near-data, analisi in-memory e selezione dei dati basata sulla località.  Modelli tradizionali come MPI, OpenMP e Map-Reduce risultano inadeguati.  La transizione exascale impatta crucialmente scheduling dei thread, comunicazione, sincronizzazione, distribuzione dei dati e viste di controllo.

Il parallelismo a memoria distribuita è predominante, con un'adozione parziale del message-passing (MPI).  Tuttavia, MPI presenta limiti: gestione complessa della parallelizzazione da parte dell'utente, difficoltà nel bilanciamento del carico dinamico, problemi di scalabilità nella comunicazione many-to-many e colli di bottiglia nell'I/O.  Sistemi a memoria condivisa offrono un'alternativa, trasferendo la responsabilità della parallelizzazione al compilatore, ma presentano problemi di scalabilità nei meccanismi di sincronizzazione.

L'utilizzo crescente di cluster eterogenei (CPU e GPU) introduce la sfida della programmazione eterogenea, richiedendo nuove astrazioni e strumenti per gestire ambienti di esecuzione diversi.

Infine, modelli come **Legion** sono stati proposti per affrontare le esigenze exascale.  Legion, ad esempio, utilizza *regioni logiche* per l'organizzazione dei dati, suddivisibili in sottoregioni disgiunte o aliasate, e *task* per l'elaborazione, offrendo un approccio efficiente per la gestione della località dei dati e del parallelismo.

---

## Riepilogo dei Modelli di Programmazione Parallela

Questo documento riassume cinque modelli di programmazione parallela: Charm++, DCEx, X10, Chapel e UPC++.  Ognuno offre un approccio diverso alla gestione della concorrenza e della distribuzione dei dati.

### Charm++

Charm++ è un modello di programmazione a memoria distribuita basato su oggetti mobili e asincroni che comunicano tramite messaggi.  Utilizza l'*overdecomposition*, suddividendo le applicazioni in molti piccoli oggetti che superano il numero di processori, migliorando l'efficienza.  Gli oggetti possono migrare tra i processori, indirizzando le operazioni agli oggetti logici anziché ai processori fisici.

### DCEx

DCEx è un modello PGAS (Partitioned Global Address Space) progettato per applicazioni data-intensive su larga scala ed exascala.  Si concentra sulla *sincronizzazione near-data*, minimizzando lo scambio di dati tra thread concorrenti che operano vicino ai dati.  Il programma è strutturato in blocchi data-paralleli che fungono da unità di memoria/archiviazione per il calcolo, la comunicazione e la migrazione paralleli.

### X10

X10 è un modello APGAS (Asynchronous Partitioned Global Address Space) che introduce il concetto di *locazione* come astrazione del contesto computazionale. Ogni locazione contiene dati ed esegue attività (thread leggeri) che accedono sincronicamente alle regioni di memoria locali.  La distribuzione dei dati avviene attraverso la creazione e la gestione di queste locazioni.

### Chapel

Chapel è un modello APGAS che offre astrazioni di alto livello per la programmazione parallela.  Utilizza *strutture dati a vista globale*, come array distribuiti, e una *vista globale del controllo*, semplificando la gestione di dati e flusso di controllo.  Il concetto di *locale* astrae l'unità di accesso uniforme alla memoria, garantendo tempi di accesso simili per tutti i thread all'interno di un locale.

### UPC++

UPC++ è una libreria C++ PGAS che fornisce strumenti per la programmazione asincrona e la gestione delle dipendenze tra calcoli.  Si basa su tre concetti principali: *puntatori globali* per l'accesso efficiente ai dati, *programmazione asincrona basata su RPC* (Remote Procedure Call) per la comunicazione efficiente e *futures* per gestire la disponibilità dei dati da computazioni asincroniche.

---
