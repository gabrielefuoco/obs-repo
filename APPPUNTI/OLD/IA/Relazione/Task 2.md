## Task 2: Classical Planning

La pianificazione automatica si occupa della creazione di strategie o sequenze di azioni per risolvere problemi complessi. Nel progetto, abbiamo adottato un approccio che comprende la modellazione del problema e la creazione di una funzione euristica personalizzata per il dominio in esame. Questo ci ha permesso di generare piani d'azione efficienti in tempi ragionevoli.

### Istanza 1

Gli oggetti coinvolti nel problema sono i seguenti:
- **Scatole**: b1, b2, b3, b4
- **Posizioni**: dep, loc1, loc2, loc3, loc4
- **Stazioni di lavoro**: w1, w2, w3, w4, w5
- **Robot**: r1
- **Contenuti**: vite, bullone, dado, cacciavite, martello
- **Carrier**: cr1
- **Spazi nel carrier (carrierSlot)**: carrierSlot1_cr1, carrierSlot2_cr1

#### Stato Iniziale

- Il robot r1 e tutte le scatole (b1, b2, b3, b4) si trovano nel deposito centrale (**dep**).
- Il carrier cr1 e tutti i contenuti (vite, bullone, dado, cacciavite, martello) sono nel deposito centrale.
- Le stazioni di lavoro sono distribuite come segue:
- w1 si trova in loc1
- w2 si trova in loc2
- w3 si trova in loc3
- w4 si trova in loc4
- w5 si trova anch'essa in loc3
- Tutte le scatole sono inizialmente vuote.
- Le necessità delle stazioni di lavoro sono:
- w1 necessita di un bullone
- w2 necessita di una vite
- w3 necessita di un dado
- w4 necessita di un cacciavite
- w5 necessita di un martello

#### Obiettivo

L'obiettivo del problema è garantire che ogni stazione di lavoro riceva il contenuto richiesto:
- w1 deve ricevere un bullone
- w2 deve ricevere una vite
- w3 deve ricevere un dado
- w4 deve ricevere un cacciavite
- w5 deve ricevere un martello

```python
(define (problem centralWarehouseDelivery)
    (:domain centralWarehouse)
    (:objects
        b1 b2 b3 b4 - box
        dep loc1 loc2 loc3 loc4 - location
        w1 w2 w3 w4 w5 - workstation
        r1 - robot
        vite bullone dado cacciavite martello - content
        cr1 - carrier
        cs1cr1 cs2cr1 - carrierSlot
    )

    (:init
        (at r1 dep)

        (at b1 dep)
        (at b2 dep)
        (at b3 dep)
        (at b4 dep)

        (at cr1 dep)

        (at bullone dep)
        (at vite dep)
        (at dado dep)
        (at cacciavite dep)
        (at martello dep)

        (at w1 loc1)
        (at w2 loc2)
        (at w3 loc3)
        (at w4 loc4)
        (at w5 loc3)

        (empty b1)
        (empty b2)
        (empty b3)
        (empty b4)

        (needContent bullone w1)
        (needContent vite w2)
        (needContent dado w3)
        (needContent cacciavite w4)
        (needContent martello w5)

        (carrierSlotAvailable cs1cr1)
        (carrierSlotAvailable cs2cr1)

        (carrierHasSlot cs1cr1 cr1)
        (carrierHasSlot cs2cr1 cr1)

        ;; Define the connections between locations
        (connected dep loc1)
        (connected loc1 dep)
        (connected loc1 loc2)
        (connected loc2 loc1)
        (connected loc2 loc3)
        (connected loc3 loc2)
        (connected loc3 loc4)
        (connected loc4 loc3)
    )
    (:goal
        (and
            (hasContent w1 bullone)
            (hasContent w2 vite)
            (hasContent w3 dado)
            (hasContent w4 cacciavite)
            (hasContent w5 martello)
        )
    )
)

```

### Istanza 2

Nella seconda istanza del problema, ci sono alcune differenze rispetto alla prima:
- Ogni robot ha un **carrier** con una capacità massima.
- I robot possono caricare scatole sui carrier fino a raggiungere la loro capacità massima.
- I robot, le scatole e i carrier devono essere **nella stessa posizione** per effettuare il caricamento.
- I robot possono spostarsi verso le stazioni di lavoro che necessitano di rifornimenti e scaricare le scatole dal carrier.
- I robot non devono tornare al deposito centrale fino a quando non hanno consegnato tutte le scatole caricate sul loro carrier.

L'obiettivo finale è lo stesso del primo problema: assicurarsi che tutte le stazioni di lavoro ricevano i contenuti necessari.

#### Modifiche agli oggetti coinvolti:

- **Robot**: r1, r2
- **Carrier**: cr1, cr2
- **Spazi nei carrier (carrierSlot)**: carrierSlot1_cr1, carrierSlot2_cr1, carrierSlot1_cr2, carrierSlot2_cr2

#### Stato Iniziale

- I robot r1 e r2, insieme alle scatole (b1, b2, b3, b4), si trovano nel deposito centrale (**dep**).
- I carrier cr1 e cr2, insieme a tutti i contenuti (vite, bullone, dado, cacciavite, martello), sono nel deposito centrale (**dep**).
- La distribuzione delle stazioni di lavoro è la stessa del primo problema.
- Tutte le scatole sono inizialmente vuote.
- Gli spazi di carico nei carrier (carrierSlot1_cr1, carrierSlot2_cr1, carrierSlot1_cr2, carrierSlot2_cr2) sono disponibili.

#### Obiettivo

L'obiettivo rimane quello di garantire che ogni stazione di lavoro riceva il contenuto richiesto:
- w1 deve ricevere un bullone
- w2 deve ricevere una vite
- w3 deve ricevere un dado
- w4 deve ricevere un cacciavite
- w5 deve ricevere un martello

```python
(define (problem centralWarehouseDelivery)
    (:domain centralWarehouse)
    (:objects
        b1 b2 b3 b4 - box
        dep loc1 loc2 loc3 loc4 - location
        w1 w2 w3 w4 w5 - workstation
        r1 r2 - robot
        vite bullone dado cacciavite martello - content
        cr1 cr2 - carrier
        cs1cr1 cs2cr1 cs1cr2 cs2cr2 - carrierSlot
    )

    (:init
        (at r1 dep)
        (at r2 dep)

        (at b1 dep)
        (at b2 dep)
        (at b3 dep)
        (at b4 dep)

        (at cr1 dep)
        (at cr2 dep)

        (at bullone dep)
        (at vite dep)
        (at dado dep)
        (at cacciavite dep)
        (at martello dep)

        (at w1 loc1)
        (at w2 loc2)
        (at w3 loc3)
        (at w4 loc4)
        (at w5 loc3)

        (empty b1)
        (empty b2)
        (empty b3)
        (empty b4)

        (needContent bullone w1)
        (needContent vite w2)
        (needContent dado w3)
        (needContent cacciavite w4)
        (needContent martello w5)

        (carrierSlotAvailable cs1cr1)
        (carrierSlotAvailable cs2cr1)
        (carrierSlotAvailable cs1cr2)
        (carrierSlotAvailable cs2cr2)

        (carrierHasSlot cs1cr1 cr1)
        (carrierHasSlot cs2cr1 cr1)
        (carrierHasSlot cs1cr2 cr2)
        (carrierHasSlot cs2cr2 cr2)

        ;; Define the connections between locations
        (connected dep loc1)
        (connected loc1 dep)
        (connected loc1 loc2)
        (connected loc2 loc1)
        (connected loc2 loc3)
        (connected loc3 loc2)
        (connected loc3 loc4)
        (connected loc4 loc3)
    )
    (:goal
        (and
            (hasContent w1 bullone)
            (hasContent w2 vite)
            (hasContent w3 dado)
            (hasContent w4 cacciavite)
            (hasContent w5 martello)
        )
    )
)

```

### Planner

Per trovare un piano d'azione in grado di risolvere i problemi descritti, è necessario l'impiego di un planner, ovvero uno strumento che elabora la sequenza di azioni necessarie per passare dallo stato iniziale a quello obiettivo. Senza l'utilizzo di algoritmi informati, risolvere questi problemi può risultare estremamente dispendioso in termini di tempo e risorse. Gli algoritmi di ricerca informata, noti anche come algoritmi di ricerca guidata, sfruttano informazioni aggiuntive sul problema per dirigere la ricerca verso una soluzione in modo più efficiente. Queste informazioni, tipicamente sotto forma di euristiche o conoscenze specifiche del dominio, permettono di ridurre le esplorazioni inutili, migliorando così le prestazioni complessive.

Un'euristica è considerata ammissibile quando la sua stima è sempre inferiore o uguale al costo effettivo per raggiungere l'obiettivo, il che garantisce che non venga mai sovrastimato il costo rimanente. Questo è fondamentale per assicurare che la soluzione trovata sia ottimale.

Nel nostro progetto, è stato utilizzato **PDDL4J** come planner Python-based per implementare la ricerca e risolvere il problema.
**PDDL4J** è una libreria Java dedicata alla risoluzione di problemi di pianificazione automatica utilizzando il linguaggio PDDL (Planning Domain Definition Language).
Include una serie di euristiche e di algoritmi di ricerca, tra cui **Enforced Hill-Climbing (EHC)** con **Fast Forward Heuristic (FF)**, che abbiamo scelto di usare ai fini del progetto.

### Enforced Hill-Climbing (EHC)

EHC è una variante della ricerca locale che cerca di superare il problema dei **plateau** (regioni dello spazio di ricerca con valori euristici simili) combinando una strategia di ricerca in profondità limitata con il criterio greedy.
Il funzionamento generale di EHC si articola nei seguenti passi:

- **Scelta dello stato iniziale:** L'algoritmo parte dallo stato iniziale e utilizza l'euristica per calcolare il costo stimato verso l'obiettivo.
- **Ricerca del miglioramento locale:** Viene esplorato lo spazio degli stati vicini per trovare uno stato con un valore euristico migliore. Se tale stato esiste, diventa il nuovo stato corrente.
- **Gestione dei plateau:** Se non è possibile trovare un miglioramento locale immediato, viene avviata una **ricerca completa** (ad esempio, una BFS limitata) per individuare un nuovo stato migliore.
- **Ripetizione del processo:** I passi vengono ripetuti finché non viene raggiunto uno stato che soddisfa le condizioni dell'obiettivo.

### Funzionamento dell'euristica Fast Forward (FF)

L'euristica **Fast Forward** è una funzione che stima il costo per raggiungere lo stato obiettivo da uno stato corrente. È basata su un rilassamento del problema originale, nel quale vengono ignorati gli effetti negativi delle azioni (ad esempio, rimuovere una precondizione soddisfatta da un'altra azione).

- **Costruzione di un piano rilassato:** L'euristica calcola un piano rilassato che considera solo gli effetti positivi delle azioni.
- **Stima del costo:** Il costo del piano rilassato è utilizzato come stima del costo reale per raggiungere l'obiettivo.

Questo approccio permette di ottenere stime rapide e generalmente accurate, risultando molto efficiente nei problemi di pianificazione automatica.

Nel nostro caso, la scelta dell'euristica **Fast Forward** bilancia ottimamente la qualità della soluzione con il tempo di esecuzione della ricerca.

Dopo aver definito il planner e l'euristica, il sistema è stato eseguito utilizzando **PDDL4J**, che offre un framework flessibile per eseguire esperimenti di pianificazione automatica. I risultati mostrano che l'algoritmo **Enforced Hill-Climbing** con l'euristica **Fast Forward** ha permesso di ottenere soluzioni di qualità ottimale con tempi di elaborazione ragionevoli:

Di seguito, il comando per eseguire il planner:

```
java -cp build/libs/pddl4j-4.0.0.jar fr.uga.pddl4j.planners.statespace.FF \
   src/pddl/domain.pddl \
   src/pddl/problemTask1.pddl
```

## Analisi prima Istanza

L'algoritmo **Enforced Hill-Climbing** con l'euristica **Fast Forward** è riuscito a risolvere efficacemente il problema di pianificazione, trovando una soluzione in un tempo ragionevole.

Il piano si compone di 32 step temporali (da 00 a 31). Ogni step contiene una o più azioni con la loro rappresentazione, comprese le precondizioni e gli effetti.

- **Dimensioni del problema**:
- *Azioni* disponibili: 456.
- *Fluents* (stati): 92.
- **Performance**:
- *Parsing*: 0.04 secondi.
- *Encoding*: 0.04 secondi.
- *Ricerca*: 1.14 secondi.
- *Tempo totale*: 1.31 secondi.
- **Memoria usata**:
- *Rappresentazione del problema*: 1.99 MB.
- *Per la ricerca*: 0 MB aggiuntivi.
- *Totale*: 1.99 MB.

![[Task 2-20241230172141511.png|600]]

## Analisi seconda Istanza

Come nel caso precedente, è stato utilizzato l'algoritmo **Enforced Hill-Climbing**, mentre l'euristica rimane **Fast Forward**. Il secondo problema risulta più complesso da risolvere rispetto al primo, richiedendo più elaborazione nonostante produca un piano di lunghezza simile (38 stati). Questo è dovuto al maggior numero di oggetti e alle loro interazioni.
In particolare:

- **Dimensioni del problema**:
- *Azioni disponibili:* 1768 (notevolmente più alto rispetto al primo caso).
- *Fluents (stati)*: 112 (in crescita rispetto al primo problema, con 92 fluents).
- **Performance**:
- *Parsing*: 0.05 secondi.
- *Encoding*: 0.90 secondi.
- *Ricerca*: 14.10 secondi.
- *Tempo totale*: 15.05 secondi.
- **Memoria usata**:
- *Rappresentazione del problema*: 7.01 MB.
- *Per la ricerca*: 0 MB aggiuntivi.
- *Totale*: 7.02 MB.

![[Task 2-20241230172940046.png|600]]

