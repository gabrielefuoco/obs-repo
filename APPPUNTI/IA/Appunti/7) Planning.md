
| Termine | Spiegazione |
|---|---|
| **Planning** | La progettazione di una sequenza di azioni necessarie per raggiungere un obiettivo specifico. |
| **Stati (S)** | Le condizioni o situazioni del sistema. |
| **Azioni (A)** | Le operazioni che possono essere eseguite per cambiare lo stato del sistema. |
| **Eventi (E)** | Eventi esterni e imprevedibili che influenzano lo stato, ma non possono essere controllati. |
| **Funzione di Transizione di Stato (γ)** | Definisce come lo stato cambia in base alle azioni o agli eventi. |
| **Planner** | Pianifica le azioni da eseguire per raggiungere un obiettivo partendo da uno stato iniziale. |
| **Controller** | Esegue le azioni secondo il piano e può adattarsi se le cose non vanno come previsto. |
| **Pianificazione Offline** | Tutto il piano è definito in anticipo. |
| **Pianificazione Dinamica** | Il piano può essere adattato e modificato mentre le azioni sono in corso. |
| **Planner Domain-Specific** | Specializzati per un dominio specifico di pianificazione. |
| **Planner Domain-Independent** | Funzionano per qualsiasi dominio di pianificazione. |
| **Planner Configurable** | Hanno un motore di pianificazione generale ma possono essere configurati per migliorare l'efficienza. |
| **Scheduling** | Assegnare azioni a periodi di tempo specifici, con vincoli di precedenza. |
| **Classical Planning** | Un tipo di pianificazione in cui si lavora con stati, azioni ed eventi noti e finiti, un sistema completo e deterministico, stati goal, piani sequenziali e tempo implicito. |
| **Problema di Pianificazione** | Trovare una sequenza di azioni che porti dallo stato iniziale a uno stato in cui tutti gli obiettivi sono raggiunti. |
| **Classical Representation** | Un modo di rappresentare un problema di planning usando la logica del primo ordine. |
| **Predicati** | Rappresentano proprietà o relazioni. |
| **Costanti** | Oggetti specifici nel sistema. |
| **Atomi** | Espressioni che possono essere vere o false, composte da un predicato e dai suoi argomenti. |
| **Ground Expression** | Un atomo senza variabili, rappresenta fatti specifici e concreti. |
| **Unground Expression** | Contiene variabili, rappresenta un fatto più generale. |
| **Grounding** | Il processo di sostituzione delle variabili in un'espressione unground con costanti specifiche. |
| **Sostituzione (θ)** | Mappa variabili a valori concreti. |
| **Stato** | Un insieme di atomi ground che descrivono la situazione corrente del sistema. |
| **Operatore** | Un'unità che descrive come cambiare lo stato del mondo tramite un'azione. |
| **Azione** | Una specifica istanza di un operatore, ottenuta applicando una sostituzione ai parametri dell'operatore. |
| **Applicabilità delle Azioni** | Un'azione è applicabile in uno stato se le sue precondizioni sono soddisfatte. |
| **Dominio di Planning** | Un insieme di operatori che descrivono le azioni che possono essere eseguite. |
| **Piano** | Una sequenza di azioni che si intende eseguire. |
| **Soluzione** | Un piano che è eseguibile e che, applicando le azioni in sequenza dallo stato iniziale, raggiunge uno stato finale che soddisfa l'insieme degli obiettivi. |
| **State-Space Planning** | Il problema di pianificazione è rappresentato come uno spazio di stati, dove ogni nodo rappresenta uno stato del mondo e un piano è un percorso attraverso questo spazio. |
| **Plan-Space Planning** | Ogni nodo è un piano parziale, che include alcuni vincoli. Si lavora su piani parziali e si cerca di completare il piano soddisfacendo tutti i vincoli e raggiungendo il goal. |
| **Forward Search** | Inizia dallo stato iniziale e cerca di raggiungere lo stato goal esplorando lo spazio degli stati in avanti. |
| **Backward Search** | Si parte dal goal e si lavora all'indietro per determinare quali azioni potrebbero portare a raggiungere questo obiettivo. |
| **Lifting** | Una tecnica utilizzata per ridurre la complessità del problema di pianificazione usando variabili nelle azioni. |
| **STRIPS** | Un algoritmo di pianificazione classica che si basa su una rappresentazione strutturata degli stati e delle azioni. |
| **Anomalia di Sussman** | Un esempio emblematico di fallimento della strategia di STRIPS. |
| **Planning-Graph** | Una struttura che cerca di risolvere il problema dell'elevato branching factor nel planning. |
| **Graphplan** | Un algoritmo che utilizza il planning-graph per rilassare temporaneamente il problema e poi rifinire la ricerca. |
| **Mutex** | Vincoli che escludono azioni o stati incompatibili. |
| **Frame Actions** | Azioni che lasciano inalterati gli stati attuali, mantenendo il sistema stabile tra un'azione e l'altra. |

---
Il *planning* (pianificazione) riguarda la progettazione di sequenze di azioni necessarie per raggiungere un obiettivo specifico. 

1. **Modello di Pianificazione**:
   - **Stati (S)**: Rappresentano le condizioni o situazioni del sistema.
   - **Azioni (A)**: Le operazioni che puoi eseguire e che cambiano lo stato del sistema.
   - **Eventi (E)**: Eventi esterni e imprevedibili che influenzano lo stato, ma non puoi controllare.
   - **Funzione di Transizione di Stato (γ)**: Definisce come lo stato cambia in base alle azioni o agli eventi. Può restituire uno stato specifico o più stati possibili (sistemi non deterministici).

2. **Processo di Pianificazione**:
	Il processo di pianificazione è la creazione di un piano d'azione per raggiungere un obiettivo specifico, partendo da uno stato iniziale e tenendo conto delle azioni disponibili e degli eventi esterni. Si compone di:
	   - **Planner**: Pianifica le azioni da eseguire per raggiungere un obiettivo partendo da uno stato iniziale. Fornisce un piano (sequenza di azioni) che poi viene attuato.
	   - **Controller**: Esegue le azioni secondo il piano e può adattarsi se le cose non vanno come previsto. Riceve feedback sullo stato corrente e può fare aggiustamenti.

1. **Tipi di Pianificazione**:
   - **Offline**: Tutto il piano è definito in anticipo.
   - **Dinamico**: Il piano può essere adattato e modificato mentre le azioni sono in corso, a seconda degli eventi e dei risultati.

2. **Tipi di Planner**:
   - **Domain-Specific**: Specializzati per un dominio specifico di pianificazione.
   - **Domain-Independent**: Funzionano per qualsiasi dominio di pianificazione e sono più generali.
   - **Configurable**: Hanno un motore di pianificazione generale ma possono essere configurati per migliorare l’efficienza aggiungendo conoscenze specifiche del dominio.

3. **Complessità**:
   - **Scheduling** (pianificazione temporale): Assegnare azioni a periodi di tempo specifici, con vincoli di precedenza. È un problema NP-Completo.
   - **Planning**: Decidere quali azioni eseguire. Può essere più complesso, spesso nella classe P-SPACE, e può diventare indecidibile con simboli di funzione complessi.

## Classical Planning

**Classical Planning** è un tipo di pianificazione in cui si lavora con:

1. **Stati, Azioni ed Eventi noti e finitei**. 
2. **Sistema Completo e Deterministico**: Tutto ciò che accade è osservabile e prevedibile; non ci sono eventi esterni imprevisti.
3. **Stati Goal**: Abbiamo uno stato finale desiderato (goal) e partiamo da uno stato iniziale.
4. **Piani Sequenziali**: Le azioni sono eseguite in sequenza, senza sovrapposizioni temporali.
5. **Tempo Implicito**: Le azioni sono istantanee, senza durata.

Il **problema di pianificazione** consiste nel trovare una sequenza di azioni che porti dallo stato iniziale a uno stato in cui tutti gli obiettivi sono raggiunti. È come cercare un percorso in un grafo dove:

- I **nodi** sono gli stati.
- Gli **archi** sono le azioni che portano da uno stato all'altro.

**Problema**: Se il numero di stati è molto grande può essere difficile trovare il percorso giusto.
I planner configurabili sono spesso usati per questo tipo di pianificazione perché possono adattarsi a vari domini e risolvere problemi complessi senza dover scrivere un planner specifico per ogni situazione. 

Il processo di planning include:
1. **Modellazione**: Definire il problema e il contesto.
2. **Planning AI**: Utilizzare algoritmi di pianificazione.
3. **Behavior Trees**: Strutturare e organizzare le azioni in alberi di comportamento.
4. **ROS**: Utilizzare il Robot Operating System per implementare e gestire le azioni.

Inoltre, non si crea solo un piano, ma anche piani di riserva per gestire imprevisti.

## Classical Representation
**Classical Representation** è un modo di rappresentare un problema di planning usando la logica del primo ordine con alcune caratteristiche specifiche. 

1. **Simboli di Predicati e Costanti**:
   - **Predicati**: Rappresentano proprietà o relazioni (es. `top(pallet, p)` indica che un pallet è sopra un'altra cosa).
   - **Costanti**: Oggetti specifici nel sistema (es. nomi di pallet o posti).

2. **Atomi**:
   - Sono espressioni che possono essere vere o false. Un atomo è composto da un predicato e dai suoi argomenti (costanti o variabili).

3. **Ground e Unground Expressions**:
   - **Ground Expression**: Un atomo senza variabili, rappresenta fatti specifici e concreti (es. `top(pallet1, shelf2)`).
   - **Unground Expression**: Contiene variabili, rappresenta un fatto più generale (es. `top(pallet, shelf)`).

4. **Grounding**:
   - È il processo di sostituzione delle variabili in un'espressione unground con costanti specifiche per ottenere un'espressione ground.

5. **Sostituzione**:
   - Una **sostituzione** `θ` mappa variabili a valori concreti. Serve a ottenere una ground expression da un'unground expression.

6. **Stato**:
   - Un insieme di atomi ground che descrivono la situazione corrente del sistema.

**Esempio**:
Se abbiamo un predicato `top(pallet, shelf)`, questo può rappresentare un fatto generale. Se specifichiamo che `top(pallet1, shelf2)` è vero, stiamo utilizzando una ground expression che dice qualcosa di concreto sulla posizione di un pallet.

### Operatore
E un'untità che descrive come cambiare lo stato del mondo tramite un'azione. È rappresentato da una tripla: 

1. **`name(o)`**: Il nome dell'operatore. Identifica l'operatore ed è un simbolo unico seguito da variabili (ad es. `move(x1, x2)`).

2. **`preconditions(o)`**: Le precondizioni che devono essere vere perché l'operatore possa essere applicato. Possono essere formule o insiemi di atomi che devono essere veri nello stato corrente.

3. **`effects(o)`**: Gli effetti che l'operatore ha sullo stato quando viene applicato. Descrivono come cambiano gli atomi dello stato.

### Azione e Ground Istance

Una **azione** è una specifica istanza di un operatore, ottenuta applicando una sostituzione ai parametri dell'operatore. 

#### Notazione e Terminologia

- **`S+`**: Atomi che sono veri nello stato `S`.
- **`S−`**: Atomi che sono falsi nello stato `S`.

Per un'azione `a` (che è una ground instance di un operatore):

- **`precond+(a)`**: Insieme di atomi che devono essere veri nelle precondizioni di `a`.
- **`precond−(a)`**: Insieme di atomi che devono essere falsi nelle precondizioni di `a`.
- **`effects+(a)`**: Insieme di atomi che diventano veri a causa degli effetti di `a`.
- **`effects−(a)`**: Insieme di atomi che diventano falsi a causa degli effetti di `a`.

1. **Applicabilità delle Azioni**:
   - Un’**azione** `a` è applicabile in uno **stato** `s` se:
     - **Precondizioni Positive**: Tutti i **letterali positivi** che sono precondizioni per `a` devono essere veri nello stato `s`. Questo significa che ogni condizione necessaria per applicare l’azione deve essere soddisfatta.
     - **Precondizioni Negative**: Nessun **letterale negativo** che è una precondizione per `a` deve essere vero nello stato `s`. Questo significa che non devono esserci condizioni che impediscano l'applicazione dell'azione.

2. **Dominio di Planning**:
   - Un **dominio di planning** è costituito da un insieme di **operatori**. Ogni operatore descrive un'azione che può essere eseguita, comprese le sue precondizioni e gli effetti.

3. **Piano e Soluzione**:
   - Un **piano** è una **sequenza di azioni** (`π = (a1, ..., an)`) che si intende eseguire.
   - Un piano `π` è una **soluzione** per un problema di planning se è **eseguibile** e se, applicando le azioni in sequenza dallo stato iniziale `s0`, si raggiunge uno stato finale che soddisfa l’**insieme degli obiettivi** `Sg`. 
   - Formalmente, se possiamo trovare una sequenza di stati `s0, s1, ..., sn` e azioni `a1, ..., an` tali che:
     - `γ(s0, a1) = s1`
     - `γ(s1, a2) = s2`
     - ...
     - `γ(sn-1, an) = sn`
     - e `sn` soddisfa `Sg`, allora `π` è una soluzione.

4. **Soluzioni Ridondanti**:
   - In molti casi, si possono trovare più soluzioni, alcune delle quali possono essere ridondanti. Il vero problema è trovare una soluzione **minima** o **più breve**, evitando soluzioni superflue.
   - **Soluzione: Rappresentazione Set-Theoretic**: Trasformiamo ogni atomo ground (fatto specifico) in una variabile proposizionale booleana. Così, invece di lavorare con atomi, lavoriamo con variabili che possono essere vere o false.
   - Questo approccio può occupare molto spazio, soprattutto se un operatore con arità `k` ha molte istanze ground, poiché il numero di combinazioni può crescere esponenzialmente.

6. **Rappresentazione State-Variable**:
   - Invece di utilizzare variabili booleane per tutto, possiamo usare **atomi** per rappresentare proprietà **statiche** (che non cambiano) e **valori** per rappresentare proprietà **dinamiche** (che cambiano). Ad esempio, `top(p1) = c3` indica che il pallet `p1` è sopra la pedana `c3`, e questa informazione può cambiare.
---
# State-Space Planning

Nel **state-space planning**, il problema di pianificazione è rappresentato come uno spazio di stati, dove:
- **Ogni nodo** rappresenta uno stato del mondo.
- **Un piano** è un percorso attraverso questo spazio di stati, che va dallo stato iniziale a uno stato che soddisfa il goal.

Invece, nel **plan-space planning**:
- **Ogni nodo** è un piano parziale, che include alcuni vincoli (constraints).
- Si lavora su piani parziali e si cerca di completare il piano soddisfacendo tutti i vincoli e raggiungendo il goal.

### Approcci e Tecniche

Esistono vari approcci per risolvere un problema di state-space planning:

#### 1. Forward Search

- **Idea**: Inizia dallo stato iniziale e cerca di raggiungere lo stato goal esplorando lo spazio degli stati in avanti.
- **Algoritmo**:
  - Parti dallo stato iniziale con un piano vuoto.
  - Se lo stato corrente soddisfa il goal, hai trovato una soluzione.
  - Se non ci sono azioni applicabili, restituisci fallimento.
  - Altrimenti, scegli un'azione da applicare e passa allo stato successivo.
  - Continua fino a raggiungere il goal.

- **Proprietà**:
  - **Soundness (Correttezza)**: Ogni piano trovato è una soluzione valida.
  - **Completeness (Completezza)**: Se esiste una soluzione, verrà trovata.

- **Implementazione**:
  - Puoi usare tecniche di ricerca come **Breadth-First Search (BFS)**, **Depth-First Search (DFS)**, **Best-First Search**, **Greedy**, e **A* Search**.
  - **BFS** è completa ma può essere inefficiente in spazi di stato grandi.
  - **DFS** è più efficiente in termini di spazio ma può non essere completa a meno che non si evitino cicli.
  - **A* Search** e **Best-First Search** utilizzano euristiche per ottimizzare la ricerca, ma l'uso di euristiche può compromettere la completezza se non sono ben progettate.

- **Problemi**:
  - **Branching Factor**: Il numero di azioni applicabili può essere molto alto, rendendo la ricerca difficile.
  - **Euristiche**: L'efficacia della ricerca dipende molto dalla qualità delle euristiche usate per guidare la ricerca.

### Backward Search

Nel **backward search**, si parte dal goal e si lavora all'indietro per determinare quali azioni potrebbero portare a raggiungere questo obiettivo. Questo approccio si basa sull'idea di trovare una serie di azioni che, se applicate in ordine inverso, possono trasformare lo stato goal nello stato iniziale.

#### Passaggi Principali:

1. **Definizione di Azioni Rilevanti**:
   - Un'azione $a$ è considerata rilevante per un goal $g$ se:
     - **L'azione contribuisce al goal**: $( g ) ∧ ( \text{effects}(a) \neq \emptyset$
       - Questo significa che l'effetto dell'azione *a* aiuta a soddisfare almeno uno dei letterali del goal $g$.
     - **L'azione non rende il goal falso**: 
     - $(g^+ ) ∧ ( \text{effects}^-(a) = \emptyset )$ e $( g^- ) ∧ ( \text{effects}^+(a) = \emptyset$
       - Questo significa che l'azione non elimina nessuno dei letterali positivi del goal e non aggiunge falsità ai letterali negativi.

2. **Funzione Inversa della Transizione**:
   - La funzione inversa della funzione di transizione, $γ$, è definita come:
     - $$ \gamma^{-1}(g, a) = (g^- \text{effects}(a)) \lor \text{precond}(a) $$
       - Questa funzione calcola nuovi obiettivi (finti goal) che devono essere raggiunti prima di applicare l'azione $a$.

3. **Algoritmo**:
   - Inizia dal goal e applica la funzione inversa per ottenere nuovi obiettivi.
   - Le azioni vengono aggiunte al piano in testa, non in coda.
   - Questo processo continua fino a quando non si raggiunge uno stato che può essere considerato come stato iniziale o uno stato che soddisfa tutte le condizioni necessarie.

4. **Problemi**:
   - Il branching factor può essere molto grande, proprio come nel forward search.
   - Questo significa che ci possono essere molte possibili azioni da considerare.

### Lifting

Il **lifting** è una tecnica utilizzata per ridurre la complessità del problema di pianificazione:

1. **Uso di Variabili**:
   - Invece di lavorare con azioni completamente ground (cioè specifiche per istanze particolari), si utilizzano variabili nelle azioni.
   - Questo permette di generalizzare le azioni.

2. **Riduzione dello Spazio di Ricerca**:
   - Definendo le azioni con variabili anziché con costanti, si riduce il numero totale di istanze di azioni da esplorare.
   - Questo può ridurre significativamente il branching factor e la complessità dello spazio di ricerca.

3. **Maximum General Unifier**:
   - Utilizza l'unificazione più generale possibile per le variabili nelle azioni. Se una variabile può essere unificata con un'altra variabile, si fa; altrimenti, si unifica con una costante.

4. **Spazio di Ricerca**:
   - Anche se il lifting riduce lo spazio di ricerca rispetto alla ricerca backward tradizionale, può comunque generare uno spazio di ricerca molto grande. È quindi importante gestire e ottimizzare lo spazio di ricerca.

## STRIPS
**STRIPS** è un algoritmo di pianificazione classica che si basa su una rappresentazione strutturata degli stati e delle azioni. È utilizzato per risolvere problemi di pianificazione scomponendo gli obiettivi in sotto-obiettivi più semplici, ignorando temporaneamente le interdipendenze tra di essi.

#### Caratteristiche Principali:
- **Risoluzione separata dei goal**: Risolve ciascun goal individualmente assumendo che siano risolvibili in un ordine lineare senza interferenze reciproche. Questo lo rende molto utile quando i goal sono indipendenti.
- **Non completezza**: L'algoritmo non è completo perché non può sempre gestire le dipendenze tra gli obiettivi. Un classico esempio è l'**anomalia di Sussman**.

#### Anomalia di Sussman
È un esempio emblematico di fallimento della strategia di STRIPS. Considera due goal:
1. **A su B**.
2. **B su C**.

Se si cerca di risolvere questi obiettivi separatamente, il piano fallisce perché il raggiungimento di uno può distruggere l'altro. Ad esempio, mettere A su B può rendere impossibile mettere B su C senza dover rimuovere A. In questo caso, le interdipendenze tra gli obiettivi non possono essere ignorate.

#### Approcci alternativi
Per gestire tali problemi, sono necessari:
- **Algoritmi diversi**, come quelli basati su vincoli o algoritmi domain-specific che considerano le interazioni tra gli obiettivi.
- **Planner più sofisticati** che non assumono la serializzabilità dei goal.

---

### Tecniche del Planning-Graph

Il **Planning-Graph** è una struttura che cerca di risolvere il problema dell’elevato branching factor (numero di possibili stati e azioni) nel planning. Viene utilizzato in algoritmi come **Graphplan**, che rilassano temporaneamente il problema per ottenere una soluzione più semplice e poi rifinire la ricerca.

#### Come funziona?
1. **Relaxation Phase (Espansione del grafo)**:
   - Si costruisce un grafo di pianificazione partendo da \( k = 0 \) e lo si espande per \( k \) livelli.
   - Ogni livello del grafo alterna stati e azioni:
     - I **nodi al livello azione** \( i \) contengono le azioni eseguibili al tempo \( i \).
     - I **nodi al livello stato** \( i \) contengono i fatti che possono essere veri al livello \( i \).
   - Gli **archi** rappresentano le precondizioni e gli effetti delle azioni.
   - In questa fase, si considera un problema rilassato, in cui vengono inclusi anche stati e azioni che potrebbero non soddisfare tutti i vincoli del problema originale.

2. **Solution Extraction Phase**:
   - Una volta costruito il grafo di pianificazione, si esegue una **backward search** (simile a quella discussa in precedenza) per estrarre un piano che risolva il problema rilassato.
   - Si considerano solo le azioni che appaiono nel grafo rilassato, riducendo così lo spazio di ricerca.

3. **Azioni di Maintenance/Frame**:
   - Per mantenere la consistenza del grafo, si utilizzano **frame actions** che lasciano inalterati gli stati attuali, mantenendo il sistema stabile tra un’azione e l’altra.

#### Mutua Esclusività (Mutex)
Durante la costruzione del planning-graph, vengono introdotti dei vincoli per escludere azioni o stati incompatibili:

- **Mutex tra azioni**: Due azioni allo stesso livello sono mutualmente esclusive se:
  - **Effetti inconsistenti**: L'effetto di una azione nega l'effetto dell'altra.
  - **Interferenza**: Una azione distrugge una precondizione dell'altra.
  - **Bisogni concorrenti**: Le azioni hanno precondizioni mutualmente esclusive.
  
- **Mutex tra stati**: Due stati sono mutualmente esclusivi se:
  - **Supporto inconsistente**: Uno è la negazione dell'altro, o tutte le azioni che potrebbero raggiungere quei due stati sono mutualmente esclusive.

#### Vantaggi del Planning-Graph
- **Riduzione dello spazio di ricerca**: Filtra molte azioni e stati che non contribuiscono alla soluzione.
- **Efficienza**: Rende più efficiente la ricerca di un piano riducendo il branching factor del problema originale.

