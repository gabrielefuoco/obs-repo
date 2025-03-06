
Il modello Bulk Synchronous Parallel (BSP) è stato sviluppato da Leslie Valiant dell'Università di Harvard a partire dalla fine degli anni '80.

L'interesse per questo modello è cresciuto a partire dal 2010, quando Google l'ha adottato come una delle principali tecnologie per l'analisi dei grafi su larga scala (Google Pregel).

## Architettura di una macchina BSP

Tre elementi fondamentali:
- Un insieme di p unità di elaborazione (PE) responsabili dell'elaborazione.
- Un router responsabile della comunicazione. Si occupa di instradare i messaggi tra i diversi componenti, gestendo tale comunicazione punto-a-punto tra coppie di macchine. Si assume non siano presenti strumenti per la combinazione, duplicazione o broadcasting dei messaggi.
- Meccanismi e strumenti per la sincronizzazione globale (per barriere) dei componenti (o di un loro sottoinsieme).

- Ogni unità di elaborazione (PE) ha una propria memoria.
- È presente anche un blocco responsabile della sincronizzazione, collegato ad un router.

### Rappresentazione interna di un PE

- Ogni PE è munita di un processore che opera su una memoria locale .
- La memoria globale è distribuita secondo una funzione di hash degli indirizzi.
- Per migliorare le prestazioni, l'operazione di hashing può essere effettuata in hardware a livello dell'interfaccia del router.

**Modello alternativo:** Un'alternativa al modello precedente distingue le unità di elaborazione (CE) da quelle di memorizzazione (ME).

## Modello di programmazione

L'elaborazione è composta da iterazioni successive, dette superstep.
Ogni superstep si compone di tre fasi:
- **Computation**: Un insieme di elaborazioni locali su ciascun processore, effettuate ognuna sulla propria memoria locale.
- **Communication**: Un insieme di operazione di comunicazione in cui vi è il trasferimento di dati tra processi.
- **Synchronization**: Consiste nell'attendere la conclusione di tutte le operazioni di comunicazione; dopo la sincronizzazione, i dati sono visibili ai destinatari per l'inizio del superstep successivo.

## Modello di comunicazione

- Il numero massimo di messaggi in ingresso o in uscita per un superstep è indicato con h.
- La capacità di una rete di comunicazione di consegnare dati è caratterizzata da un parametro g, definito in modo tale che è necessario un tempo hg ad un processore per inviare h messaggi di dimensione 1.
- Il modello BSP non fa distinzione, in termini di costo, tra l'invio di 1 messaggio di lunghezza m o l'invio di m messaggi di lunghezza 1 (in entrambi i casi il costo è uguale a mg).

Il parametro g dipende da diversi fattori:
- I protocolli utilizzati per interagire all'interno della rete di comunicazione.
- Gestione del buffer da parte sia dei processori che della rete di comunicazione.
- La strategia di routing utilizzata nella rete.
- Il sistema di runtime BSP.

## Costo di un superstep

Il costo di un superstep è la somma di tre termini:
- Il costo dell'elaborazione locale di maggior durata
- Il costo della comunicazione globale tra i processori
- Il costo della sincronizzazione alla fine del superstep

Il costo di un superstep per p processori è:
$$max^p_{i=1}(w_i)+max^p_{i=1}(h_i g)+l$$
dove:
- $w_i$ è il costo per l'elaborazione locale nel processo i
- $h_i$ è il numero di messaggi inviati o ricevuti dal processo i
- $l$ è il costo della barriera di sincronizzazione

Per maggiore semplicità, si può usare in alternativa la seguente espressione:
$$w+hg+l$$
dove w e h sono valori massimi

## Costo di un algoritmo BSP

Il costo di un algoritmo BSP è la somma dei costi di ciascun superstep:
$$W+Hh+Sl=\sum^S_{s=1}w_s+g\sum^S_{s=1}h_s+Sl$$
dove S è il numero di superstep.

