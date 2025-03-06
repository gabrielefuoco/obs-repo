
Il progetto si pone come obiettivo la modellazione e risoluzione di un problema di pianificazione classica utilizzando il linguaggio PDDL, insieme all'integrazione di un modello temporale all'interno di un'architettura robotica. Lo scenario è ispirato al contesto della produzione industriale, dove più agenti robotici collaborano per consegnare scatole contenenti rifornimenti a diverse stazioni di lavoro. Le principali assunzioni del problema includono la posizione fissa delle stazioni e delle scatole, la capacità degli agenti robotici di manipolare le scatole (riempirle, svuotarle, trasportarle e consegnarle) e i vincoli relativi agli spostamenti degli agenti all'interno dell'ambiente.

Il lavoro si articola in due fasi principali: **pianificazione classica** e **pianificazione temporale**.

### Istanza 1:

##### Condizioni iniziali:

- Tutte le scatole sono inizialmente situate nel magazzino centrale.
- I contenuti delle scatole sono inizialmente situati nel magazzino centrale
- Nessuna stazione di lavoro è presente all'interno del magazzino.
- Un singolo agente robotico si trova nel magazzino centrale, pronto per eseguire le operazioni.

##### Obiettivi:

- Alcune stazioni di lavoro devono ricevere specifici rifornimenti
- Le stazioni di lavoro non richiedono rifornimenti specifici per tipo.

##### Risultati ottenuti:

- Tempo impiegato per completare la pianificazione.
- Memoria utilizzata durante l'elaborazione.

### Istanza 2:

##### Condizioni iniziali:

- Simili a quelle dell'istanza 1, con l'aggiunta di nuovi vincoli: ciascun agente robotico ha una capacità di carico limitata e ci sono più agenti coinvolti nella consegna.

##### Obiettivi:

- Analoghi a quelli dell'istanza 1, ma con l'introduzione della gestione del carico massimo che ogni agente può trasportare.

##### Risultati ottenuti:

- Tempo impiegato per completare la pianificazione.
- Memoria utilizzata durante il processo.

## Task 1: Fase di modellazione

Dopo aver definito chiaramente le condizioni iniziali e gli obiettivi per entrambe le istanze, il passo successivo consiste nella traduzione del problema in un linguaggio di pianificazione formale, nel nostro caso il **PDDL** (Planning Domain Definition Language). Questo linguaggio è ampiamente utilizzato nell'intelligenza artificiale per rappresentare problemi di pianificazione.

La modellazione in PDDL richiede la creazione di due file distinti:
- **Domain file**: specifica il dominio del problema, ovvero l'insieme delle azioni che possono essere eseguite dagli agenti robotici e le condizioni che determinano il cambiamento di stato del sistema. In questo file vengono descritte le dinamiche del mondo in cui operano i robot.
- **Problem file**: definisce il problema specifico da risolvere, contenente:
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

L'uso di predicati come **carrierSlotOccupied/carrierSlotAvailable** e **noContent/gotContent** è stato adottato per rendere il dominio più leggibile e gestire in maniera efficace sia la capacità massima del carrier che la distribuzione dei contenuti tra le stazioni di lavoro. In questo modo, la logica di caricamento e scaricamento dei box risulta più chiara e strutturata, migliorando l'efficienza della pianificazione.

### Azioni

In PDDL, un'azione rappresenta una trasformazione da uno stato a un altro e si compone di tre sezioni fondamentali:

- **Parameters**: Definisce gli oggetti coinvolti nell'azione.
- **Precondition**: Specifica le condizioni che devono essere soddisfatte affinché l'azione possa essere eseguita, utilizzando predicati combinati tramite congiunzione o disgiunzione.
- **Effect** (o **Expansion**): Nella maggior parte dei casi si utilizza la sezione **Effect**, che descrive le conseguenze dell'azione attraverso predicati e espressioni logiche.

Nel progetto, sono state implementate le seguenti azioni:

#### fill

L'azione di riempimento di una scatola con un contenuto specifico eseguita da un agente. Le precondizioni verificano che:
- Il contenuto, la scatola e l'agente si trovino nella stessa posizione.
- La scatola sia vuota.
**Effetti**:

- La scatola non sarà più vuota.
- La scatola verrà riempita con il contenuto specifico.
```python
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

#### charge

L'azione di carico di una scatola sul veicolo (carrier). Le precondizioni verificano che:
- Il mezzo abbia uno spazio disponibile (**carrierSlot**) per la scatola.
- Lo spazio libero appartenga al mezzo.
- La scatola, l'agente e il mezzo si trovino nella stessa posizione.
**Effetti**:

- La scatola viene caricata sul veicolo.
- La scatola non si trova più nella sua posizione originale e lo spazio sul veicolo non è più disponibile.
```python
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

#### discharge

L'azione di scarico della scatola da un veicolo. Questa azione è applicabile solo se la scatola è vuota. Le precondizioni verificano che:
- Lo slot del veicolo sia effettivamente occupato dalla scatola.
- L'agente, il mezzo e la scatola siano nella posizione corretta.
- La scatola sia vuota.
**Effetti**:

- Lo slot da cui è stata scaricata la scatola diventa disponibile.
- La scatola non si trova più sul mezzo, ma nella posizione di scarico.
```python
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

#### moveRobot

L'azione di movimento di un robot privo di carrier, verso una nuova posizione. Le precondizioni verificano che:
- Il robot si trovi nella posizione di partenza specificata.
- Le due posizioni siano connesse.
**Effetti**:

- Il robot non si trova più nella posizione iniziale.
- Il robot è arrivato alla nuova posizione.
```python
(:action moveRobot
        :parameters (?r - robot ?from ?to - location)
        :precondition (and (at ?r ?from)
                           (connected ?from ?to))
        :effect (and (not (at ?r ?from))
                     (at ?r ?to))
    )
```

#### moveCarrier

L'azione di movimento di un carrier da parte di un robot verso una nuova posizione. Le precondizioni verificano che:
- Il robot e il carrier si trovino nella posizione di partenza.
- Le posizioni siano connesse.
**Effetti**:

- Il robot e il carrier non si trovano più nella posizione di partenza.
- Entrambi si trovano nella nuova posizione.
```python
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

#### contentDelivered

Questo predicato viene introdotto per gestire il vincolo OR richiesto nelle istanze successive della pianificazione classica. Si tratta di un'azione fittizia che rappresenta la condizione in cui una workstation ha ricevuto il contenuto necessario. Le **precondizioni** verificano che:
- La workstation abbia ricevuto almeno uno dei due contenuti richiesti.
- La workstation non abbia già ricevuto uno dei contenuti in precedenza.
**Effetti**:

- La workstation risulta soddisfatta perché ha ricevuto uno dei contenuti richiesti.
```python
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
