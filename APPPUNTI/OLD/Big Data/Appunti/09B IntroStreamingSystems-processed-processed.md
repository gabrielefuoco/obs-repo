

## Sistemi di Streaming

Un **sistema di streaming** è un tipo di motore di elaborazione dati progettato per elaborare **dati illimitati**.  Ricordiamo che:

* **Dati limitati**: dataset di dimensioni finite.
* **Dati illimitati**: dataset di dimensioni (teoricamente) infinite.


### Flusso (Stream)

Dati illimitati sono un insieme concettualmente infinito e in continua crescita di elementi/eventi dati.  Rappresentano un flusso di dati praticamente continuo che deve essere elaborato/analizzato.  Il modello utilizzato è tipicamente un **modello push**, dove la produzione e l'elaborazione dei dati sono controllate dalla sorgente (modello pubblica/iscriviti).

È fondamentale considerare il **concetto di tempo**: spesso è necessario ragionare su quando i dati vengono prodotti e quando i dati elaborati devono essere restituiti.  Si distinguono diversi tipi di tempo: tempo agnostico, tempo di elaborazione, tempo di ingestione e tempo dell'evento.


### Tempo dell'evento vs tempo di elaborazione


* **Tempo dell'evento**: tempo di produzione dell'elemento dati.
* **Tempo di ingestione**: tempo di sistema in cui viene ricevuto l'elemento dati.
* **Tempo di elaborazione**: tempo di sistema in cui viene elaborato l'elemento dati.

Tipicamente, questi tre tempi non coincidono!
![[Pasted image 20250223164612.png|239]]


### Serie temporali

Una serie temporale è una serie di punti dati indicizzati in ordine temporale. Più comunemente, una serie temporale è una sequenza acquisita in punti temporali successivi equidistanti.


## Modelli di cassa registratore e tornello

Esiste un vettore **a = (a₁, …, aₙ)** che viene aggiornato da un flusso. Inizialmente tutti gli **aᵢ = 0**.

* **Modello cassa registratore**: Ogni aggiornamento è nella forma **⟨i, c⟩** in modo che **aᵢ** venga incrementato di un numero *positivo* **c**.


* **Modello tornello**: Ogni aggiornamento è nella forma **⟨i, c⟩** in modo che **aᵢ** venga incrementato di un numero (*possibilmente negativo*) **c**.


## Algoritmi di streaming

Algoritmi per l'elaborazione di flussi di dati in cui l'input viene presentato come una sequenza di elementi e può essere esaminato in poche passate (tipicamente solo una). Questi algoritmi possono avere accesso a memoria limitata e tempo di elaborazione limitato per elemento.


### Approcci all'elaborazione di flussi

* Elaborazione agnostica al tempo
* Elaborazione approssimativa
* Finestramento per tempo di elaborazione
* Finestramento per tempo dell'evento


### Elaborazione agnostica al tempo

L'elaborazione agnostica al tempo viene utilizzata per i casi in cui il tempo è essenzialmente irrilevante. Due esempi sono il filtraggio e l'inner join.

Vogliamo elaborare i log del traffico web per filtrare tutto il traffico che non proviene da un dominio specifico. Possiamo esaminare ogni record al suo arrivo, vedere se appartiene al dominio di interesse e eliminarlo in caso contrario. Poiché questo dipende da un singolo elemento in qualsiasi momento, il fatto che la sorgente dati abbia una distorsione del tempo dell'evento variabile è irrilevante.


![[Pasted image 20250223164652.png]]

La figura mostra un esempio di filtraggio di dati illimitati: una raccolta di dati (che scorre da sinistra a destra) di tipi diversi viene filtrata in una raccolta omogenea contenente un singolo tipo.


#### Esempio di Inner Join

Vogliamo unire due sorgenti di dati illimitate. Quando vediamo un valore da una sorgente, possiamo bufferizzarlo; dopo che arriva il secondo valore dall'altra sorgente, emettiamo il record unito. Poiché ci interessa solo il risultato di un join quando arriva un elemento da entrambe le sorgenti, non c'è un elemento temporale nella logica.



![|543](_page_12_Figure_1.jpeg)

La figura mostra un esempio di esecuzione di un inner join su dati illimitati: i join vengono prodotti quando vengono osservati elementi corrispondenti da entrambe le sorgenti.


### Elaborazione approssimativa

L'elaborazione approssimativa si basa su algoritmi che producono una risposta approssimativa basata su un riepilogo o uno "schizzo" del flusso di dati. Esempi: *Top-N approssimativo*, *streaming kmeans*.


![[Pasted image 20250223164730.png|524]]

Questa figura mostra un esempio di calcolo di approssimazioni su dati illimitati.


## Windowing

La sorgente dati (illimitata o limitata) viene suddivisa lungo i confini temporali in blocchi finiti per l'elaborazione. Tre modelli principali sono:

* **Finestre fisse**
* **Finestre scorrevoli**
* **Sessioni**

![[|435](_page_15_Figure_5.jpeg)


### Finestramento per tempo di elaborazione

Il sistema mette in buffer i dati in arrivo in finestre finché non è trascorso un certo tempo di elaborazione. Esempio: mettere in buffer i dati per *n* minuti di tempo di elaborazione, dopodiché tutti i dati di quel periodo di tempo vengono inviati per l'elaborazione.


![[|510](_page_17_Figure_1.jpeg)

Questa figura mostra il finestramento in finestre fisse per tempo di elaborazione: i dati vengono raccolti in finestre in base all'ordine in cui arrivano nella pipeline.


## Finestramento per tempo dell'evento

Questo viene utilizzato quando è necessario osservare una sorgente dati in blocchi finiti che riflettono i tempi in cui si sono verificati tali eventi. Più complesso del finestramento per tempo di elaborazione (ad esempio, richiede più buffering dei dati).  Problema di completezza: spesso non abbiamo modo di sapere quando abbiamo visto tutti i dati per una determinata finestra.


#### Finestre fisse

![](_page_19_Figure_1.jpeg)

I dati vengono raccolti in finestre fisse in base ai tempi in cui si sono verificati.


#### Finestre di sessione

![[](_page_20_Figure_1.jpeg)

I dati vengono raccolti in finestre di sessione in base al momento in cui si sono verificati.


# Operatori di flusso di base

* **Aggregazione con finestre:**
    * Esempio: velocità media.
    * Somma degli accessi URL.
    * Punteggio giornaliero più alto.
* **Join con finestre:**
    * Osservazioni correlate nell'intervallo di tempo.
    * Esempio: temperatura nel tempo.

![[|159](_page_21_Figure_8.jpeg)]
**Esempio: Windowed Aggregation:**
![[](_page_22_Figure_1.jpeg)


## Elaborazione di eventi complessi

* Rilevazione di pattern in un flusso.
* Evento complesso = sequenza di eventi.
* Definito usando condizioni logiche e temporali:
    * Logiche: valori e combinazioni di dati.
    * Temporali: entro un determinato periodo di tempo.

![[|220](_page_23_Picture_6.jpeg)]

`SEO(A, B, C) CON A.Temp > 23°C && B.Station = A.Station && B.Temp < A.Temp && C.Station = A.Station && A.Temp - C.Temp > 3`


* Eventi compositi costruiti ad esempio da:
    * `SEQ`, `AND`, `OR`, `NEG`, ...
    * `SEQ(e1, e2) -> (e1, t1) ^ (e2, t2)` con `t1 ≤ t2 ^ e1, e2 ∈ W`
* Implementato costruendo un NFA.
    * Esempio: `SEQ(A, B, C)`

![[|296](_page_24_Figure_6.jpeg)]


# 8 Requisiti dell'elaborazione di flussi di grandi dimensioni

* **Mantenere i dati in movimento:** Architettura di streaming.
* **Accesso dichiarativo:** Esempio: StreamSQL, CQL.
* **Gestire le imperfezioni:** Elementi in ritardo, mancanti, non ordinati.
* **Risultati prevedibili:** Coerenza, tempo dell'evento.
* **Integrare dati memorizzati e dati di streaming:** Flusso ibrido e batch.
* **Sicurezza e disponibilità dei dati:** Tolleranza ai guasti, stato persistente.
* **Partizionamento e scaling automatico:** Elaborazione distribuita.
* **Elaborazione e risposta istantanea.**


# Elaborazione di Big Data

* I database possono elaborare dati molto grandi da sempre. Perché non usarli?
* I Big Data non sono (completamente) strutturati – Non adatti ai database.
* Vogliamo imparare di più dai dati che semplicemente selezionare, proiettare, unire.
* Prima soluzione: MapReduce.


### MapReduce (di nuovo)

![[|483](_page_27_Figure_1.jpeg)


* Ottimo per grandi quantità di dati statici.
* Per i flussi: solo per finestre di grandi dimensioni.
* I dati non si muovono!
* Alta latenza, bassa efficienza.

**Come mantenere i dati in movimento:**
![[|456](_page_28_Figure_2.jpeg)]


# Mini-batch

* Facile da implementare.
* Facile coerenza e tolleranza ai guasti.
* Difficile gestire il tempo dell'evento e le sessioni.

![[|327](_page_29_Figure_4.jpeg)]


## Architettura di streaming true (vera e propria?)

* Programma = DAG di operatori e flussi intermedi.
* Operatore = calcolo + stato.
* Flussi intermedi = flusso logico di record.
* Trasformazioni di flusso:
    * Trasformazioni di base: Map, Reduce, Filter, Aggregazioni...
    * Trasformazioni di flusso binarie: CoMap, CoReduce...
    * Semantica delle finestre: finestre flessibili basate su policy (Tempo, Conteggio, Delta...).
    * Operatori di flusso binari temporali: Join, Cross...
    * Supporto nativo per le iterazioni.

* Catturare l'avanzamento della completezza del tempo dell'evento mentre il tempo di elaborazione procede:
    * Può essere definito come una funzione F(P) → E, che prende un punto nel tempo di elaborazione e restituisce un punto nel tempo dell'evento.
    * Quel punto nel tempo dell'evento, E, è il punto fino al quale il sistema ritiene di aver osservato tutti gli input con tempi di evento inferiori a E.
    * In altre parole, è un'affermazione che non saranno più visti dati con tempi di evento inferiori a E.

![[|223](_page_31_Figure_7.jpeg)]


#### Watermark perfetti vs. euristici

* Quando si ha una conoscenza perfetta dei dati di input, è possibile costruire un *watermark perfetto*. In questo caso, tutti i dati sono anticipati o puntuali.
* I *watermark euristici* utilizzano qualsiasi informazione per fornire una stima dell'avanzamento il più accurata possibile. Sono utilizzati quando la conoscenza perfetta dei dati di input è impraticabile.


## Lezioni apprese dal batch

![[|350](_page_33_Figure_1.jpeg)]

* Se un calcolo batch fallisce, ripetere semplicemente il calcolo come una transazione.
* Il tasso di transazione è costante.
* Possiamo applicare questi principi a un'esecuzione di streaming vera e propria?


### Eseguire snapshot - il modo Naive
![[Pasted image 20250223165803.png|361]]

![[Pasted image 20250223165815.png]]