Il progetto si pone come obiettivo la modellazione e risoluzione di un problema di pianificazione classica utilizzando il linguaggio PDDL, insieme all'integrazione di un modello temporale all'interno di un'architettura robotica. Lo scenario è ispirato al contesto della produzione industriale, dove più agenti robotici collaborano per consegnare scatole contenenti rifornimenti a diverse stazioni di lavoro. Le principali assunzioni del problema includono la posizione fissa delle stazioni e delle scatole, la capacità degli agenti robotici di manipolare le scatole (riempirle, svuotarle, trasportarle e consegnarle) e i vincoli relativi agli spostamenti degli agenti all'interno dell'ambiente.

Il lavoro si articola in due fasi principali: **pianificazione classica** e **pianificazione temporale**.

### Istanza 1  
**Condizioni iniziali:**
- Tutte le scatole sono inizialmente situate  nel magazzino centrale.
- I contenuti delle scatole sono inizialmente situati nel magazzino centrale
- Nessuna stazione di lavoro è presente all'interno del magazzino.
- Un singolo agente robotico si trova nel magazzino centrale, pronto per eseguire le operazioni.

**Obiettivi:**
- Alcune stazioni di lavoro devono ricevere specifici rifornimenti
- Le stazioni di lavoro non richiedono rifornimenti specifici per tipo.

**Risultati ottenuti:**
- Tempo impiegato per completare la pianificazione.
- ==Memoria utilizzata durante l'elaborazione==.

### Istanza 2  
**Condizioni iniziali:**
- Simili a quelle dell'istanza 1, con l'aggiunta di nuovi vincoli: ciascun agente robotico ha una capacità di carico limitata e ci sono più agenti coinvolti nella consegna.

**Obiettivi:**
- Analoghi a quelli dell'istanza 1, ma con l'introduzione della gestione del carico massimo che ogni agente può trasportare.

**Risultati ottenuti:**
- ==Analisi del carico trasportato dai robot.==
- Tempo impiegato per completare la pianificazione.
- ==Memoria utilizzata durante il processo==.

### Fasi di modellazione  
Dopo aver definito chiaramente le condizioni iniziali e gli obiettivi per entrambe le istanze, il passo successivo consiste nella traduzione del problema in un linguaggio di pianificazione formale, nel nostro caso il **PDDL** (Planning Domain Definition Language). Questo linguaggio è ampiamente utilizzato nell'intelligenza artificiale per rappresentare problemi di pianificazione.

La modellazione in PDDL richiede la creazione di due file distinti:
1. **Domain file**: specifica il dominio del problema, ovvero l'insieme delle azioni che possono essere eseguite dagli agenti robotici e le condizioni che determinano il cambiamento di stato del sistema. In questo file vengono descritte le dinamiche del mondo in cui operano i robot.
2. **Problem file**: definisce il problema specifico da risolvere, contenente: 
	* Gli oggetti coinvolti, ad esempio: scatole, stazioni di lavoro, agenti.  
	* Lo stato iniziale del sistema. 
	* Gli obiettivi che devono essere raggiunti.

Questa separazione tra dominio e problema consente una maggiore flessibilità, permettendo di riutilizzare la stessa descrizione del dominio per risolvere diverse istanze di problemi simili, semplicemente cambiando lo stato iniziale o gli obiettivi da raggiungere. 


### Implementazione in PDDL
Nell'implementazione del dominio utilizzando il linguaggio PDDL, è fondamentale definire i tipi, i predicati e le azioni che rappresentano le dinamiche del sistema. 

``` Java
(define (domain centralWarehouse)
    (:requirements :strips :typing :equality)
    (:types robot box location workstation content carrier carrierSlot)
    (:predicates
        (filled ?c - content ?b - box)
        (empty ?b - box)
        (hasContent ?w - workstation ?c - content)
        (carrierHasSlot ?p - carrierSlot ?v - carrier)
        (carrierSlotAvailable ?p - carrierSlot)
        (carrierSlotOccupied ?p - carrierSlot)
        (boxLoaded ?b - box ?v - carrier)
        (at ?x - (either robot workstation box content carrier) ?l - location)
        (needContent ?c - content ?w - workstation)
        (noContent ?w - workstation ?c1 - content ?c2 - content)
        (gotContent ?w - workstation ?c1 - content ?c2 - content)
        (connected ?from ?to - location)
    )
```

La modellazione degli oggetti del dominio è stata realizzata attraverso l'uso dei seguenti **tipi**:

- **robot**: rappresenta l'agente robotico.
- **box**: rappresenta il contenitore da trasportare.
- **location**: indica le posizioni fisiche in cui possono trovarsi oggetti.
- **workstation**: rappresenta una stazione di lavoro.
- **content**: descrive il contenuto del box (ad es. bulloni, valvole, attrezzi).
- **carrier**: modella il mezzo di trasporto su cui è possibile caricare i box.
- **carrierSlot**: rappresenta uno spazio all'interno di un carrier dove può essere caricato un box.

### Predicati
I predicati sono stati progettati per riflettere lo stato degli oggetti e facilitare la gestione delle operazioni. Alcuni predicati chiave includono:

- **filled(box, content)**: vero se una scatola contiene uno specifico contenuto.
- **empty(box)**: vero se la scatola è vuota.
- **hasContent(workstation, content)**: vero se la stazione di lavoro ha ricevuto un determinato contenuto.
- **carrierHasSlot(carrier, carrierSlot)**: vero se un determinato slot appartiene al carrier.
- **carrierSlotOccupied(carrierSlot)**: vero se lo slot sul carrier è occupato da una scatola.
- **carrierSlotAvailable(carrierSlot)**: vero se lo slot sul carrier è libero.
- **boxLoaded(box, carrier)**: vero se una scatola è caricata su un particolare mezzo.
- **at(object, location)**: vero se l'agente, una scatola o un mezzo si trova in una determinata posizione.
- **needContent(workstation, content)**: vero se una stazione di lavoro ha bisogno di un certo contenuto.
- **connected(location1, location2)**: vero se due posizioni sono connesse e quindi percorribili dall’agente.

### Utilizzo dei Predicati
L'uso di predicati come **carrierSlotOccupied/carrierSlotAvailable** e **noContent/gotContent** è stato adottato per rendere il dominio più leggibile e gestire in maniera efficace sia la capacità massima del carrier che la distribuzione dei contenuti tra le stazioni di lavoro. In questo modo, la logica di caricamento e scaricamento dei box risulta più chiara e strutturata, migliorando l'efficienza della pianificazione.

### Azioni
In PDDL, un'azione rappresenta una trasformazione da uno stato a un altro e si compone di tre sezioni fondamentali:

1. **Parameters**: Definisce gli oggetti coinvolti nell'azione.
2. **Precondition**: Specifica le condizioni che devono essere soddisfatte affinché l'azione possa essere eseguita, utilizzando predicati combinati tramite congiunzione o disgiunzione.
3. **Effect** (o **Expansion**): Nella maggior parte dei casi si utilizza la sezione **Effect**, che descrive le conseguenze dell'azione attraverso predicati e espressioni logiche. 

Nel progetto, sono state implementate le seguenti azioni:

#### 1. fill  
L'azione di riempimento di una scatola con un contenuto specifico eseguita da un agente. Le precondizioni verificano che:
- Il contenuto, la scatola e l'agente si trovino nella stessa posizione. 
- La scatola sia vuota.
**Effetti**:
- La scatola non sarà più vuota.
- La scatola verrà riempita con il contenuto specifico.
```Java
(:action fill
        :parameters (?l - location ?c - content ?b - box ?r - robot)
        :precondition (and (at ?c ?l)
                           (at ?b ?l)
                           (at ?r ?l)
                           (empty ?b))
        :effect (and (filled ?c ?b)
                     (not (empty ?b)))
    )
```

#### 2. charge  
L'azione di carico di una scatola sul veicolo (carrier). Le precondizioni verificano che:
- Il mezzo abbia uno spazio disponibile (**carrierSlot**) per la scatola.
- Lo spazio libero appartenga al mezzo.
- La scatola, l'agente e il mezzo si trovino nella stessa posizione.
**Effetti**:
- La scatola viene caricata sul veicolo.
- La scatola non si trova più nella sua posizione originale e lo spazio sul veicolo non è più disponibile.
```
 (:action charge
        :parameters (?l - location ?b - box ?p - carrierSlot ?v - carrier ?r - robot ?c - content)
        :precondition (and (carrierSlotAvailable ?p)
                           (carrierHasSlot ?p ?v)
                           (at ?b ?l)
                           (at ?r ?l)
                           (at ?v ?l)
                           (filled ?c ?b))  
        :effect (and (not (carrierSlotAvailable ?p))
                     (carrierSlotOccupied ?p)
                     (not (at ?b ?l))
                     (boxLoaded ?b ?v))
    )
```
#### 3. discharge  
L'azione di scarico della scatola da un veicolo. Questa azione è applicabile solo se la scatola è vuota. Le precondizioni verificano che:
- Lo slot del veicolo sia effettivamente occupato dalla scatola.
- L'agente, il mezzo e la scatola siano nella posizione corretta.
- La scatola sia vuota.
**Effetti**:
- Lo slot da cui è stata scaricata la scatola diventa disponibile.
- La scatola non si trova più sul mezzo, ma nella posizione di scarico.
```
    (:action discharge
        :parameters (?l - location ?b - box ?p - carrierSlot ?v - carrier ?r - robot ?c - content)
        :precondition (and (carrierSlotOccupied ?p)
                           (at ?r ?l)
                           (at ?v ?l)
                           (boxLoaded ?b ?v)
                           (carrierHasSlot ?p ?v)
                           (empty ?b))
        :effect (and (carrierSlotAvailable ?p)
                     (not (carrierSlotOccupied ?p))
                     (at ?b ?l)
                     (not (boxLoaded ?b ?v)))
    )
```
#### 4. moveRobot  
L'azione di movimento di un robot privo di carrier, verso una nuova posizione. Le precondizioni verificano che:
- Il robot si trovi nella posizione di partenza specificata.
- Le due posizioni siano connesse.
**Effetti**:
- Il robot non si trova più nella posizione iniziale.
- Il robot è arrivato alla nuova posizione.
```
(:action moveRobot
        :parameters (?r - robot ?from ?to - location)
        :precondition (and (at ?r ?from)
                           (connected ?from ?to))
        :effect (and (not (at ?r ?from))
                     (at ?r ?to))
    )
```
#### 5. moveCarrier  
L'azione di movimento di un carrier da parte di un robot verso una nuova posizione. Le precondizioni verificano che:
- Il robot e il carrier si trovino nella posizione di partenza.
- Le posizioni siano connesse.
**Effetti**:
- Il robot e il carrier non si trovano più nella posizione di partenza.
- Entrambi si trovano nella nuova posizione.
```
    (:action moveCarrier
        :parameters (?r - robot ?from ?to - location ?v - carrier)
        :precondition (and (at ?r ?from)
                           (at ?v ?from)
                           (connected ?from ?to))
        :effect (and (not (at ?r ?from))
                     (at ?r ?to)
                     (not (at ?v ?from))
                     (at ?v ?to))
    )
```
#### 6. contentDelivered  
Questo predicato viene introdotto per gestire il vincolo OR richiesto nelle istanze successive della pianificazione classica. Si tratta di un'azione fittizia che rappresenta la condizione in cui una workstation ha ricevuto il contenuto necessario. Le **precondizioni** verificano che:
- La workstation abbia ricevuto almeno uno dei due contenuti richiesti.
- La workstation non abbia già ricevuto uno dei contenuti in precedenza.
**Effetti**:
- La workstation risulta soddisfatta perché ha ricevuto uno dei contenuti richiesti.
```
    (:action contentDelivered
        :parameters (?c1 - content ?c2 - content ?w - workstation)
        :precondition (and (hasContent ?w ?c1)
                           (noContent ?w ?c1 ?c2))
        :effect (and (gotContent ?w ?c1 ?c2)
                     (not (noContent ?w ?c1 ?c2))
                     (not (noContent ?w ?c2 ?c1))
                     (gotContent ?w ?c2 ?c1))
    )
```
---
# Task 2  
### CLASSICAL PLANNING

La pianificazione automatica si occupa della creazione di strategie o sequenze di azioni per risolvere problemi complessi. Nel progetto, abbiamo adottato un approccio che comprende la modellazione del problema e la creazione di una funzione euristica personalizzata per il dominio in esame. Questo ci ha permesso di generare piani d'azione efficienti in tempi ragionevoli.

### Primo Problema

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
  1. w1 si trova in loc1
  2. w2 si trova in loc2
  3. w3 si trova in loc3
  4. w4 si trova in loc4
  5. w5 si trova anch'essa in loc3
- Tutte le scatole sono inizialmente vuote.
- Le necessità delle stazioni di lavoro sono:
  1. w1 necessita di un bullone
  2. w2 necessita di una vite
  3. w3 necessita di un dado
  4. w4 necessita di un cacciavite
  5. w5 necessita di un martello

#### Obiettivo
L'obiettivo del problema è garantire che ogni stazione di lavoro riceva il contenuto richiesto:
- w1 deve ricevere un bullone
- w2 deve ricevere una vite
- w3 deve ricevere un dado
- w4 deve ricevere un cacciavite
- w5 deve ricevere un martello

```PDDL
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

### Secondo Problema
Nel secondo problema, ci sono alcune differenze rispetto al primo:
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

```
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

---
### Planner

Per trovare un piano d'azione in grado di risolvere i problemi descritti, è necessario l'impiego di un planner, ovvero uno strumento che elabora la sequenza di azioni necessarie per passare dallo stato iniziale a quello obiettivo. Senza l'utilizzo di algoritmi informati, risolvere questi problemi può risultare estremamente dispendioso in termini di tempo e risorse. Gli algoritmi di ricerca informata, noti anche come algoritmi di ricerca guidata, sfruttano informazioni aggiuntive sul problema per dirigere la ricerca verso una soluzione in modo più efficiente. Queste informazioni, tipicamente sotto forma di euristiche o conoscenze specifiche del dominio, permettono di ridurre le esplorazioni inutili, migliorando così le prestazioni complessive.

Un'euristica è considerata ammissibile quando la sua stima è sempre inferiore o uguale al costo effettivo per raggiungere l'obiettivo, il che garantisce che non venga mai sovrastimato il costo rimanente. Questo è fondamentale per assicurare che la soluzione trovata sia ottimale.

Nel nostro progetto, è stato utilizzato **Pyperplan**, un planner Python-based, per implementare la ricerca e risolvere il problema. **Pyperplan** include una serie di euristiche, tra cui l'euristica **Additive Heuristic**, che abbiamo scelto per dirigere la ricerca.

### Algoritmo A*

L'algoritmo utilizzato è **A\***, un algoritmo di ricerca informata che combina il costo reale del percorso, denotato come $g(n)$, con una stima euristica del costo restante $h(n)$. L'algoritmo seleziona, a ogni iterazione, il nodo con il valore complessivo più basso $(g(n) + h(n))$ tra tutti quelli esplorabili, continuando l'esplorazione fino al raggiungimento dell'obiettivo. Questo approccio garantisce l'ottimalità della soluzione, a condizione che l'euristica utilizzata sia ammissibile.

Nel caso specifico, non sono state apportate modifiche all'algoritmo A\* standard, e si è utilizzata l'euristica **Additive Heuristic**. Questa euristica stima il costo rimanente sommando i costi individuali delle sotto-componenti del problema, fornendo una misura utile per guidare la ricerca.

### Funzionamento dell'euristica

L'**Additive Heuristic** si basa su una strategia di decomposizione del problema in più sottocomponenti. In particolare, per ciascun predicato o condizione richiesta per raggiungere lo stato obiettivo, viene stimato un costo individuale. L'euristica calcola poi la somma di questi costi per fornire una stima del costo complessivo per raggiungere lo stato obiettivo dallo stato corrente. 

Questa somma riflette l'assunzione che i singoli sottoproblemi possono essere risolti indipendentemente, senza che le azioni per risolverne uno interferiscano con la soluzione degli altri. Di conseguenza, l'euristica è ammissibile, poiché non sovrastima il costo complessivo necessario per raggiungere l'obiettivo: in altre parole, l'**Additive Heuristic** stima un costo che è sempre inferiore o uguale al costo reale del percorso ottimale. 
In particolare, fornisce una stima più precisa in contesti con molte precondizioni indipendenti e tiene conto del costo di soddisfare ogni precondizione individualmente, fornendo una valutazione più accurata del costo totale.
Di contro, la ricerca risulta essere più lenta e computazionalmente più costosa (specialmente per problemi di pianificazione di grandi dimensioni) di euristiche più semplici, come Fast Forward.

Nel nostro caso, la scelta dell'euristica **Additive Heuristic** bilancia ottimamente la qualità della soluzione con il tempo di esecuzione della ricerca. 

Dopo aver definito il planner e l'euristica, il sistema è stato eseguito utilizzando **Pyperplan**, che offre un framework flessibile per eseguire esperimenti di pianificazione automatica. I risultati mostrano che l'euristica **Additive Heuristic** ha permesso di ottenere soluzioni di qualità ottimale con tempi di elaborazione ragionevoli:

```
pyperplan domain.pddl problem.pddl --heuristic hadd --search astar
```
## Analisi Prima istanza
L'algoritmo A* con l'euristica Additive è riuscito a risolvere efficacemente il problema di pianificazione, trovando una soluzione in un tempo ragionevole.
In particolare
1. Grounding:
   - Il processo di grounding ha rimosso 270 fatti irrilevanti
   - Sono state create 412 variabili e 1016 operatori
2. Ricerca:
   - Il valore euristico iniziale era 59
   - La ricerca è durata 2.1 secondi
   - Sono stati espansi 394 nodi prima di raggiungere l'obiettivo
3. Soluzione:
   - È stata trovata una soluzione
   - Il piano risultante ha una lunghezza di 40 azioni
4. Performance:
   - Il tempo di esecuzione totale è relativamente breve (circa 2.3 secondi considerando parsing e grounding)
   - Il numero di nodi espansi (394) suggerisce che l'euristica hadd è stata efficace nel guidare la ricerca
5. Complessità del problema:
   - Il problema sembra di media complessità, data la lunghezza del piano (40 azioni) e il numero di nodi espansi

![[Pasted image 20240917112812.png]]
```
(fill dep martello b1 r1)
(fill dep dado b3 r1)
(fill dep vite b4 r1)
(charge dep b4 cs1cr1 cr1 r1 vite)
(movecarrier r1 dep loc1 cr1)
(movecarrier r1 loc1 loc2 cr1)
(givecontent loc2 vite b4 r1 w2 cr1)
(movecarrier r1 loc2 loc1 cr1)
(movecarrier r1 loc1 dep cr1)
(fill dep bullone b2 r1)
(charge dep b2 cs2cr1 cr1 r1 bullone)
(movecarrier r1 dep loc1 cr1)
(givecontent loc1 bullone b2 r1 w1 cr1)
(discharge loc1 b4 cs1cr1 cr1 r1 martello)
(movecarrier r1 loc1 dep cr1)
(charge dep b1 cs1cr1 cr1 r1 martello)
(movecarrier r1 dep loc1 cr1)
(movecarrier r1 loc1 loc2 cr1)
(movecarrier r1 loc2 loc3 cr1)
(givecontent loc3 martello b1 r1 w5 cr1)
(movecarrier r1 loc3 loc2 cr1)
(discharge loc2 b1 cs1cr1 cr1 r1 martello)
(movecarrier r1 loc2 loc1 cr1)
(movecarrier r1 loc1 dep cr1)
(charge dep b3 cs1cr1 cr1 r1 dado)
(movecarrier r1 dep loc1 cr1)
(movecarrier r1 loc1 loc2 cr1)
(movecarrier r1 loc2 loc3 cr1)
(givecontent loc3 dado b3 r1 w3 cr1)
(movecarrier r1 loc3 loc2 cr1)
(movecarrier r1 loc2 loc1 cr1)
(movecarrier r1 loc1 dep cr1)
(discharge dep b3 cs1cr1 cr1 r1 martello)
(fill dep cacciavite b3 r1)
(charge dep b3 cs1cr1 cr1 r1 cacciavite)
(movecarrier r1 dep loc1 cr1)
(movecarrier r1 loc1 loc2 cr1)
(movecarrier r1 loc2 loc3 cr1)
(movecarrier r1 loc3 loc4 cr1)
(givecontent loc4 cacciavite b3 r1 w4 cr1)

```

## Analisi Seconda istanza
Come nel caso precedente, è stato utilizzato l'algoritmo di ricerca A*,mentre l'euristica scelta è ancora hadd. Il secondo problema risulta più complesso da risolvere rispetto al primo, richiedendo più elaborazione nonostante produca un piano di lunghezza simile. Questo è dovuto al maggior numero di oggetti e alle loro interazioni.
1. Grounding:
   - Il processo di grounding ha rimosso 270 fatti irrilevanti (uguale a prima)
   - Sono state create 430 variabili (18 in più) e 3848 operatori (2832 in più)
2. Ricerca:
   - Il valore euristico iniziale era 59 (identico al problemTask1)
   - La ricerca è durata 55 secondi
   - Sono stati espansi 1058 nodi prima di raggiungere l'obiettivo (664 in più)
3. Soluzione:
   - È stata trovata una soluzione
   - Il piano risultante ha una lunghezza di 39 azioni (1 in meno rispetto al problemTask1)
4. Performance:
   - Il tempo di esecuzione totale è significativamente più lungo (circa 61 secondi)
   - Il numero di nodi espansi (1058) è molto maggiore, indicando una ricerca più complessa


![[Pasted image 20240917113424.png]]

```
(moverobot r1 dep loc1)
(moverobot r1 loc1 loc2)
(fill dep cacciavite b2 r2)
(fill dep martello b3 r2)
(fill dep bullone b1 r2)
(charge dep b2 cs2cr1 cr1 r2 cacciavite)
(charge dep b3 cs1cr2 cr2 r2 martello)
(charge dep b1 cs2cr2 cr2 r2 bullone)
(movecarrier r2 dep loc1 cr2)
(givecontent loc1 bullone b1 r2 w1 cr2)
(movecarrier r2 loc1 dep cr2)
(fill dep dado b4 r2)
(charge dep b4 cs1cr1 cr1 r2 dado)
(movecarrier r2 dep loc1 cr1)
(moverobot r2 loc1 dep)
(movecarrier r2 dep loc1 cr2)
(moverobot r1 loc2 loc3)
(movecarrier r2 loc1 loc2 cr1)
(moverobot r2 loc2 loc1)
(moverobot r1 loc3 loc2)
(movecarrier r1 loc2 loc3 cr1)
(givecontent loc3 dado b4 r1 w3 cr1)
(movecarrier r1 loc3 loc4 cr1)
(givecontent loc4 cacciavite b2 r1 w4 cr1)
(movecarrier r1 loc4 loc3 cr1)
(movecarrier r2 loc1 loc2 cr2)
(moverobot r2 loc2 loc1)
(movecarrier r1 loc3 loc2 cr1)
(movecarrier r1 loc2 loc3 cr2)
(givecontent loc3 martello b3 r1 w5 cr2)
(moverobot r1 loc3 loc2)
(movecarrier r1 loc2 loc1 cr1)
(movecarrier r1 loc1 dep cr1)
(discharge dep b2 cs2cr1 cr1 r1 cacciavite)
(fill dep vite b2 r1)
(charge dep b2 cs2cr1 cr1 r1 vite)
(movecarrier r1 dep loc1 cr1)
(movecarrier r1 loc1 loc2 cr1)
(givecontent loc2 vite b2 r1 w2 cr1)

```

---
## Task 3
