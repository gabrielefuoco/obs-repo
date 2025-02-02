## Task 3: Temporal Planning & Robotics


Il Task 3 richiede di adattare il dominio per permettere la creazione di una sequenza temporale di azioni, ognuna con una durata specifica. A differenza della pianificazione classica, che si limita a individuare una sequenza causale delle azioni, la pianificazione temporale deve anche garantire che le azioni possano essere programmate nel tempo. 
Questo significa rispettare i vincoli sulla durata delle azioni, lungo una linea temporale che non ha un limite prefissato ed è dunque potenzialmente illimitata.

Un'azione durativa è una formulazione in PDDL di un'azione che richiede un certo
tempo per essere completata. La quantità di tempo è esprimibile come valore o come
disuguaglianza, ovvero consente sia azioni a durata fissa che a durata variabile.
Similmente alle azioni tradizionali, è possibile specificare degli effetti ed è inoltre
possibile esprimere una condizione da verificare con la parola “*condition*” piuttosto
che “*precondition*”. Questo cambiamento semantico è stato introdotto per
rappresentare che un'azione durativa può non solo condizionare l'inizio dell'azione,
ma può avere condizioni che devono essere vere alla fine o per tutta la durata stessa.

Un'azione durativa, nella formulazione PDDL, rappresenta un'azione che richiede un intervallo di tempo specifico per essere completata. 
La durata può essere definita come un valore fisso o espressa attraverso disuguaglianze, permettendo di gestire sia azioni a durata determinata che variabile.

Analogamente alle azioni tradizionali, anche per le azioni durative è possibile definire gli effetti che queste producono. 

Consideriamo le tre categorie:
- **Iniziali**: devono essere vere al momento in cui l'azione inizia.
- **Finali**: devono essere vere al termine dell'azione.
- **Persistenti**: devono essere vere per tutta la durata dell'azione.

Ad esempio, nel caso di una lezione universitaria, potrebbe essere importante specificare che l'aula rimanga riservata e disponibile per tutta la durata della lezione. Per rappresentare questa situazione in PDDL, si utilizzano i seguenti costrutti:

- **`at start`**: indica che una condizione, come la disponibilità dell’aula e l’arrivo del docente, deve essere verificata prima dell’inizio della lezione.
- **`over all`**: specifica che una condizione, come la presenza continuativa dell’aula riservata e l’assenza di interferenze, deve essere mantenuta valida per tutta la durata della lezione.

Questi costrutti possono essere utilizzati anche per descrivere gli effetti delle azioni, consentendo di modellare dettagliatamente il comportamento dinamico legato allo svolgimento delle lezioni.

Per implementare le **azioni durative**, si è partiti dal dominio definito nella fase di modellazione, aggiungendo vincoli temporali tramite l'uso dei **fluents**. I fluents, a differenza delle variabili booleane, consentono di rappresentare valori numerici, come la durata residua della lezione o il numero di studenti presenti, garantendo una gestione più precisa degli stati e dei vincoli temporali.

Nel progetto, sono stati introdotti i seguenti **fluents** per migliorare la modellazione del dominio:

- **`weight`**: per rappresentare il peso complessivo.
- **`vehicle-weight`**: per il peso del veicolo.
- **`box-weight`**: per il peso delle singole scatole.
- **`path-cost`**: per calcolare e ottimizzare il costo del percorso.

Sono stati inoltre introdotti, sotto forma di predicati, i concetti di posto occupato sul veicolo e di agente disponibile, utile per rappresentare uno stato dell'agente libero da occupazioni.
```python
(define (domain durativeMagazzino)
  (:requirements :typing :adl :universal-preconditions :durative-actions :fluents)
  (:types robot box location workstation content vehicle place)
  (:predicates 
    (filled ?c - content ?b - box)
    (empty ?b - box)
    (has-content ?w - workstation ?c - content)
    (place-vehicle ?p - place ?v - vehicle)
    (place-available ?p - place)
    (place-occupied ?p - place)
    (box-loaded ?b - box ?v - vehicle)
    (at ?x - (either robot workstation box content vehicle) ?l - location)
    (need-content ?c - content ?w - workstation)
    (robot-free ?r - robot)
    (connected ?from ?to - location)
  )

  (:functions
    (weight ?c - content)
    (vehicle-weight ?v - vehicle)
    (box-weight ?b - box)
    (path-cost ?r - robot)
  )
```

- Le operazioni di **riempimento della scatola** (_fill_), **caricamento sul veicolo** (_charge_), **scarico dal veicolo** (_discharge_) e **rilascio del contenuto** (_give_content_) hanno una durata fissa pari a uno. 

- L'azione di **spostamento dell'agente** (_move-agent_) ha una durata fissa pari a due. Tale valore tiene conto del tempo necessario per il movimento dell'agente senza ulteriori complicazioni, come l'interazione con mezzi o carichi.

- L'azione di **movimento dell'agente insieme a un carrello** (_move_vehicle_) ha una durata variabile, strettamente legata al peso del contenuto trasportato. In questo caso, la durata è calcolata come il doppio del peso del mezzo, poiché il peso influisce direttamente sulla velocità e sull'efficienza del movimento. Questa scelta consente di rappresentare in modo più realistico il maggiore sforzo richiesto per spostare un carico "più pesante".
  
Qui di seguito le **durative actions**:

**Charge:**
```python
  (:durative-action charge
    :parameters (?l - location ?r - robot ?b - box ?p - place ?v - vehicle)
    :duration (= ?duration 1)
    :condition (and 
      (over all (at ?v ?l))
      (at start (at ?b ?l))
      (over all (at ?r ?l))
      (over all (place-vehicle ?p ?v))
      (at start (place-available ?p))
      (at start (robot-free ?r))
    )
    :effect (and
      (at start (not (robot-free ?r)))
      (at end (robot-free ?r))
      (at start (not (at ?b ?l)))
      (at end (box-loaded ?b ?v))
      (at start (not (place-available ?p)))
      (at start (place-occupied ?p))
      (at end (increase (vehicle-weight ?v) (box-weight ?b)))
      (at end (increase (path-cost ?r) 1))
    )
  )
```

**Discharge:**
```python
  (:durative-action discharge
    :parameters (?l - location ?r - robot ?b - box ?p - place ?v - vehicle)
    :duration (= ?duration 1)
    :condition (and 
      (over all (at ?v ?l))
      (over all (at ?r ?l))
      (at start (box-loaded ?b ?v))
      (over all (place-vehicle ?p ?v))
      (over all (place-occupied ?p))
      (over all (empty ?b))
      (at start (robot-free ?r))
    )
    :effect (and
      (at start (not (robot-free ?r)))
      (at end (robot-free ?r))
      (at end (at ?b ?l))
      (at start (not (box-loaded ?b ?v)))
      (at end (not (place-occupied ?p)))
      (at end (place-available ?p))
      (at end (decrease (vehicle-weight ?v) (box-weight ?b)))
      (at end (decrease (path-cost ?r) 1))
    )
  )
```

**Fill:**
```python
  (:durative-action fill 
    :parameters (?l - location ?c - content ?b - box ?r - robot)
    :duration (= ?duration 1)
    :condition (and 
      (at start (empty ?b))
      (over all (at ?c ?l))
      (over all (at ?b ?l))
      (over all (at ?r ?l))
      (at start (robot-free ?r))
    )
```

**Give Content:**
```python
  (:durative-action give_content
    :parameters (?l - location ?c - content ?b - box ?r - robot ?w - workstation ?v - vehicle)
    :duration (= ?duration 1)
    :condition (and 
      (at start (filled ?c ?b))
      (over all (at ?v ?l))
      (over all (at ?w ?l))
      (over all (at ?r ?l))
      (over all (box-loaded ?b ?v))
      (over all (need-content ?c ?w))
      (at start (robot-free ?r))
    )
```

**Move Vehicle:**
```python
  (:durative-action move_vehicle
    :parameters (?r - robot ?from ?to - location ?v - vehicle)
    :duration (= ?duration (* (vehicle-weight ?v) 2))
    :condition (and
      (at start (at ?r ?from))
      (at start (at ?v ?from))
      (over all (connected ?from ?to))
      (at start (robot-free ?r))
    )
```

**Move Robot:**
```python
  (:durative-action move_robot
    :parameters (?r - robot ?from ?to - location)
    :duration (= ?duration 2)
    :condition (and
      (at start (at ?r ?from))
      (over all (connected ?from ?to))
      (at start (robot-free ?r))
    )
    :effect (and
      (at start (not (robot-free ?r)))
      (at end (robot-free ?r))
      (at start (not (at ?r ?from)))
      (at end (at ?r ?to))
      (at end (increase (path-cost ?r) 2))
    )
  )
```

La minimizzazione del costo del percorso per il singolo agente punta a ridurre al massimo la durata complessiva delle azioni necessarie per raggiungere l'obiettivo (goal).
Dopo aver creato l'istanza del problema, è stato scelto un **planner** adatto alla sua risoluzione. La libreria **Planutils** si è rivelata utile, offrendo strumenti come **POPF**, **TFD** e **LPG-TD**, tutti compatibili con le azioni durative e capaci di gestire efficacemente i vincoli temporali del dominio.

L’utilizzo di **POPF** (_Partial Order Planner for Forward-chaining_) si è rivelato particolarmente vantaggioso per la gestione di problemi di pianificazione temporale. Grazie alla strategia di ricerca **forward-chaining** e all’ordinamento parziale delle azioni, POPF offre una flessibilità superiore nella creazione dei piani.
Una delle sue principali caratteristiche è la capacità di gestire azioni con durate specifiche, consentendo l’ottimizzazione delle risorse e del tempo complessivo. Inoltre, la possibilità di eseguire azioni in parallelo, quando non ci sono conflitti, permette di accelerare il processo di pianificazione.
Il dominio descritto nella Task 1 è stato modificato per supportare l’utilizzo di azioni durative e renderlo compatibile con i planner forniti da **Planutils** e **Plansys2**. Oltre alle implementazioni richieste, sono state apportate ulteriori modifiche per soddisfare i requisiti specifici di questi strumenti e garantire un'esecuzione priva di errori.

Alcuni dei bug riscontrati riguardavano problemi di **parsing**, ossia la difficoltà dei planner nel interpretare correttamente il dominio. Questi errori si manifestavano, ad esempio, in caso di **nomi di predicati o azioni ambigui o non conformi al formato richiesto**, che impedivano il corretto caricamento e l'elaborazione del dominio.
Ad esempio, il termine **carrier** generava conflitti durante il parsing, probabilmente a causa di una sovrapposizione con parole chiave o strutture interne ai planner. L’unica soluzione efficace è stata rinominare **carrier** in **vehicle**, eliminando il problema.

Questi interventi, seppur apparentemente semplici, sono stati fondamentali per garantire la compatibilità del dominio con i planner e prevenire errori durante la fase di elaborazione.

## Robotics Planning

Per eseguire il piano, è stato necessario organizzare il workspace creando diverse directory, come **launch**, **src**, e **pddl**. 
Successivamente, altre cartelle sono state generate automaticamente dall’utilizzo di specifici comandi per garantire la corretta esecuzione del piano in ROS. Le modalità di configurazione e creazione di queste directory sono state descritte in dettaglio nelle specifiche del progetto e dei task assegnati. Di seguito una panoramica:

- **pddl**: contiene il file del dominio, denominato **`domain.pddl`**, necessario per definire le regole e le azioni del piano.
- **launch**: include il file di configurazione **`plansys2_project_launch.py`**, che avvia i nodi ROS richiesti per l’esecuzione del piano. Qui sono presenti anche:
    - **`plan.txt`**: file in cui viene salvato il piano generato.
    - **`commands`**: file utilizzato come parametro dal terminale di Plansys2 per impostare le istanze del problema.
- **src**: contiene i file relativi alle azioni specifiche del dominio, scritte in linguaggio C++. Al suo interno si trovano:
    - **`CMakeLists.txt`**: file utilizzato per configurare il processo di compilazione del package.
    - **`package.xml`**: file che definisce le dipendenze richieste per il package.

Le specifiche del progetto e dei task hanno fornito una guida dettagliata per creare e configurare queste directory, assicurando una corretta integrazione dei componenti richiesti da **ROS** e **Plansys2** per l’esecuzione del piano. Questo approccio sistematico ha semplificato il processo e ridotto la possibilità di errori durante l’implementazione.

```python
add_executable(move_robot src/move_agent.cpp)
ament_target_dependencies(move_robot ${dependencies})

add_executable(charge src/charge.cpp)
ament_target_dependencies(charge ${dependencies})

add_executable(discharge src/discharge.cpp)
ament_target_dependencies(discharge ${dependencies})

add_executable(give_content src/give_content.cpp)
ament_target_dependencies(give_content ${dependencies})

add_executable(fill src/fill.cpp)
ament_target_dependencies(fill ${dependencies})

add_executable(move_vehicle src/move_vehicle.cpp)
ament_target_dependencies(move_vehicle ${dependencies})
```

Per semplicità, di seguito è riportata solo parte dell’implementazione dell’azione **moveRobot**
in C++:

```python
class MoveRobotAction : public ActionExecutorClient
{
public:
  MoveRobotAction()
  : ActionExecutorClient("move_robot", 250ms)
  {
    progress_ = 0.0;
  }

private:
  void do_work()
  { 
    vector<string> arguments = get_arguments();
    if (progress_ < 1.0) {
      progress_ += 0.2;
      send_feedback(progress_, "Robot " + arguments[0] + 
            " from " + arguments[1] + " to " +
            arguments[1]);
    } else {
      finish(true, 1.0, "Robot " + arguments[0] + 
            " from " + arguments[1] + " to " +
            arguments[1]);
      progress_ = 0.0;
      cout << endl;
    }
    cout << "\r\e[K" << flush;
    cout << "Robot " + arguments[0] + 
            " from " + arguments[1] + " to " +
            arguments[1] +" . . . [ " << min(100.0, progress_ * 100.0) << "% ]  " <<
            flush;
  }

  float progress_;
};

int main(int argc, char ** argv)
{
  init(argc, argv);
  auto node = make_shared<MoveRobotAction>();
  node->set_parameter(Parameter("action_name", "move_robot"));
  node->trigger_transition(Transition::TRANSITION_CONFIGURE);
  spin(node->get_node_base_interface());
  shutdown();
  return 0;
}

```

Una volta definiti tutti i file necessari per l’esecuzione della parte relativa al **Robotics Planning**, è stato necessario effettuare il **building** della cartella e inizializzare l’ambiente di esecuzione utilizzando i seguenti comandi:

```bash
# Installazione pulita nella cartella plansys2_project
rm -rf build/ install/ log/

# Compilazione del progetto con symlink per uno sviluppo più rapido
colcon build --symlink-install

# Configurazione dell'ambiente locale
source install/local_setup.bash
```

Per avviare il terminale **Plansys2** dalla cartella radice del progetto, è stato utilizzato il comando:

```bash
ros2 launch plansys2_project plansys2_project_launch.py
```

Se il dominio **PDDL** è interpretato correttamente, tutti i nodi vengono inizializzati correttamente, producendo il seguente output.

![[Task 3-20241230160941532.png|600]]

Da un nuovo terminale, si esegue il seguente comando per accedere al terminale di
plansys2:
``` 
ros2 run plansys2_terminal plansys2_terminal
```

Vengono impostate le condizioni iniziali:
```
source launch/commands
```

E infine si avvia il plan precedentemente generato tramite POPFF:
```
run plan-file launch/plan.txt
```

Ottenendo il seguente risultato nel secondo terminale:

![[Task 3-20241230160925195.png|600]]

Dall’output possiamo concludere che il piano viene correttamente applicato al
problema e rispetta i vincoli temporali imposti.

Nel secondo terminale vengono visualizzati i log relativi alle azioni descritte nei file di configurazione C++ (definiti nella directory **src**). È importante sottolineare che, per garantire il corretto funzionamento, è stato necessario mantenere coerenza tra:

- **I nomi delle azioni** specificati nei file C++ e quelli definiti nel dominio PDDL.
- **Il file di configurazione `CMakeLists.txt`**, che deve includere correttamente tutte le azioni implementate.

Ecco un esempio di log che indica il corretto funzionamento del sistema:

![[Task 3-20241230160822732.png|600]]

## Conclusioni 

L’attività svolta ha raggiunto con successo tutti gli obiettivi prefissati, dimostrando l’efficacia del modello di pianificazione automatica sviluppato. La modellazione del problema attraverso il linguaggio **PDDL** ha permesso di ottenere piani d’azione efficienti entro tempi ragionevoli.

Le richieste della traccia sono state chiare e dettagliate, fornendo un supporto fondamentale per lo sviluppo e la gestione delle operazioni. Tuttavia, alcune difficoltà sono emerse, in particolare nella fase iniziale di familiarizzazione con librerie come **ROS** e le sue dipendenze. Inoltre, piccoli problemi di parsing legati ai nomi di oggetti e predicati hanno richiesto modifiche al dominio e un lavoro di debugging aggiuntivo.

Queste difficoltà sono state affrontate con successo grazie alla consultazione della documentazione e al chiaro riferimento alle specifiche della traccia. I risultati finali, pienamente conformi alle aspettative, hanno confermato la validità del metodo di pianificazione implementato e l’efficacia delle soluzioni adottate.