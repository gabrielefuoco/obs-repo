
## Il modello BSP

Il **Bulk Synchronous Parallel (BSP)** è un modello di calcolo parallelo sviluppato da Leslie Valiant (1990). Valiant propose un paradigma simile al modello di Von Neumann, connettendo hardware e software per macchine parallele. L'approccio BSP consente ai programmatori di evitare la gestione costosa di memoria e comunicazione, ottenendo un calcolo parallelo efficiente con un basso grado di sincronizzazione.


## Computer BSP

Un **computer BSP** è costituito dai seguenti componenti:

* Un insieme di **Elementi di Elaborazione (PE)** o **processori**, che eseguono calcoli locali.
* Un **Router** che consegna messaggi tra coppie di PE.
* Un **sincronizzatore** hardware che consente ai PE di essere sincronizzati a intervalli regolari di L unità di tempo (*latenza di comunicazione* o *periodicità di sincronizzazione*).

![[|356](_page_2_Figure_6.jpeg)]

## Calcolo BSP

Un calcolo nel modello BSP è costituito da un insieme di **superstep**, dove a ciascun processore viene assegnato un compito che coinvolge passaggi di calcolo locali, trasmissioni di messaggi e arrivi di messaggi. Un controllo globale avviene ogni L unità di tempo (il parametro *periodicità*) per verificare il completamento del superstep da parte di tutti i processori prima di procedere al superstep successivo.

### Superstep BSP

Ogni superstep è composto da tre fasi ordinate:

1. **Calcolo concorrente:** ogni processore esegue asincronamente calcoli utilizzando dati locali; ogni processo può utilizzare solo i valori memorizzati nella memoria locale del processore.
2. **Comunicazione globale:** i processi scambiano dati in risposta alle richieste effettuate durante il calcolo locale.
3. **Sincronizzazione a barriera:** i processi che raggiungono una barriera si aspettano che tutti gli altri abbiano raggiunto la stessa barriera.

Comunicazione e sincronizzazione sono disaccoppiate, garantendo l'indipendenza tra i processi in un superstep e prevenendo problemi relativi al passaggio di messaggi sincroni, come i deadlock.



## Flusso di esecuzione
![[Pasted image 20250223162931.png|446]]

## Comunicazione

-  Il modello BSP semplifica la gestione della comunicazione trattando collettivamente le azioni di comunicazione, imponendo un limite di tempo alla trasmissione batch dei dati. 
- Tutte le azioni di comunicazione del superstep sono considerate un'unica unità, assumendo dimensioni di messaggio costanti all'interno di tale unità. 
- Sia *h* il numero massimo di messaggi per un superstep e *g* il rapporto di throughput di comunicazione; il tempo per un processore per inviare *h* messaggi di dimensione uno è derivato come *hg*
- Nel modello BSP, un messaggio di lunghezza *m* è trattato come *m* messaggi di lunghezza uno, con un costo di comunicazione di *mg*.


## Sincronizzazione

- Nonostante i potenziali costi, il paradigma BSP si basa sulla comunicazione con sincronizzazione tramite **barriere**, eliminando il rischio di dipendenze circolari e prevenendo deadlock o livelock. 
- I costi di sincronizzazione sono influenzati da fattori come le variazioni nei tempi di completamento dei passaggi di calcolo locali e lo sforzo per mantenere la coerenza globale tra i processori.
- Affrontare queste sfide può comportare l'assegnazione di task proporzionale ai carichi di lavoro dei processi e la considerazione dell'efficienza della rete di comunicazione, dell'hardware specifico per la sincronizzazione e dei metodi di gestione delle interruzioni.


## Costo di un algoritmo BSP

Per garantire lo scambio di almeno *h* messaggi in un superstep, è essenziale verificare la relazione *L ≥ hg*, dove *L* è la *periodicità*, e *hg* rappresenta il tempo richiesto da un processore per inviare *h* messaggi di dimensione uno. Mantenere un basso valore per *g* è cruciale per evitare un aumento significativo del tempo di comunicazione. Il costo totale di un superstep *s* può essere espresso come  $T_s = w_s + h_s g + L$, dove $w_s$ indica il costo totale del calcolo avvenuto nel superstep.


Dato *S* il numero totale di supersteps, il **costo totale di un algoritmo BSP** T è dato da:

$$T = \sum_{1 \le s \le S} T_s = \sum_{1 \le s \le S} (w_s + h_s g + L) = W + Hg + SL$$

dove:

* *W* è il costo totale di computazione
* *H* è il costo totale di comunicazione



## Modello BSP basato su memoria condivisa

Il modello BSP non supporta direttamente la memoria condivisa, la trasmissione (broadcasting) o la combinazione (combining). Tuttavia, queste funzionalità possono essere ottenute emulando una **Parallel Random Access Machine (PRAM)** su un computer BSP. In una PRAM, un numero infinito di processori è connesso a un'unità di memoria condivisa con capacità illimitata. La comunicazione tra i processori è limitata alla memoria condivisa, collegata da una **Memory Access Unit (MAU)**, e i calcoli sono completamente sincroni.


## Varianti di PRAM

Sono state proposte diverse varianti di PRAM:
![[Pasted image 20250223163050.png|383]]


## Bulk-Synchronous PPRAM (BSPRAM)

La **Bulk-Synchronous PPRAM (BSPRAM)** è stata introdotta da Alexandre Tiskin (1998) con l'obiettivo di facilitare la programmazione in stile memoria condivisa. BSPRAM consiste di *p* processori con memoria locale veloce e una singola memoria principale condivisa, operando in supersteps simili a BSP.
Ogni superstep prevede tre fasi: input, calcolo locale e output, durante le quali i processori interagiscono con la memoria principale. La sincronizzazione avviene tra i supersteps, mentre il calcolo all'interno di un superstep rimane asincrono (vedi figura nella diapositiva successiva).



![[|430](_page_13_Figure_2.jpeg)