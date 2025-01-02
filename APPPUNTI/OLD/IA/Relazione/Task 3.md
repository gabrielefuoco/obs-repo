## Temporal Planning & Robotics

Il terzo punto della traccia richiede la conversione del dominio affinché sia in grado
di generare una sequenza temporale di azioni caratterizzate da una durata. Un
problema di pianificazione temporale consiste nel cercare una sequenza di azioni che
non sia solo causalmente eseguibile (come nella pianificazione classica), ma anche
programmabile, in conformità con un dato insieme di vincoli sulla durata dell'azione,
lungo una linea temporale di lunghezza illimitata.
Un'azione durativa è una formulazione in PDDL di un'azione che richiede un certo
tempo per essere completata. La quantità di tempo è esprimibile come valore o come
disuguaglianza, ovvero consente sia azioni a durata fissa che a durata variabile.
Similmente alle azioni tradizionali, è possibile specificare degli effetti ed è inoltre
possibile esprimere una condizione da verificare con la parola “condition” piuttosto
che “precondition”. Questo cambiamento semantico è stato introdotto per
rappresentare che un'azione durativa può non solo condizionare l'inizio dell'azione,
ma può avere condizioni che devono essere vere alla fine o per tutta la durata stessa.
Ad esempio, nel caso di un volo può essere importante specificare che la pista di
partenza/atterraggio rimanga libera per tutto il volo. A tal proposito, il costrutto at
start seguito da un predicato, specifica che la condizione da questo espressa deve
essere valida prima del performarsi dell’azione mentre il costrutto over all che deve
essere valida prima del performarsi dell’azione e per tutta la durata della stessa.
Similmente, tali costrutti possono essere utilizzati all’interno della specificazione
degli effetti di ogni azione Per implementare le durative actions si è partiti come base
dal dominio realizzato nella fase di modellazione, introducendo i pesi temporali
attraverso i costrutti chiamati fluents, i quali hanno la funzione di variabili di stato ma
il loro valore è un numero anziché vero o falso. In particolare sono stati aggiunti i
seguenti fluents:
• weight: modellazione del peso del contenuto
• vehicle-weight: modellazione del peso del mezzo
• box-weight: modellazione del peso della scatola
• path-cost: modellazione del costo del percorso, creato per poterlo utilizzare ai
fini della minimizzazione del costo del percorso

Inoltre, sottoforma di predicati sono stati introdotti i concetti di posto occupato sopra
il veicolo e di modellazione di un agente libero da occupazioni.

![[Task 3-20241230160700860.png|600]]

Alle operazioni di riempimento della scatola (fill), di caricamento della scatola sul
veicolo (charge), di scarico della scatola dal veicolo (discharge) e di rilascio del
contenuto (give_content) è stato assegnato un peso della durata pari a uno. Per quanto
riguarda l’azione di spostamento dell’agente (move-agent) la durata è un valore
fissato pari a due in quanto specifica il movimento del solo agente indipendentemente
da altri mezzi. Infine, per il movimento dell’agente insieme ad un carrello
(move_vehicle) la durata è strettamente dipendente dal contenuto delle scatole nel
carrello, ciascun dei quali possiede un peso specifico, per cui è calcolato come il
doppio del peso del mezzo. Qui di seguito le durative actions:

![[Task 3-20241230160709054.png|600]]
![[Task 3-20241230160713088.png|600]]
![[Task 3-20241230160718180.png|600]]
![[Task 3-20241230160724331.png|600]]
![[Task 3-20241230160730298.png|600]]
![[Task 3-20241230160743656.png|600]]
E’ stata inoltre impostata la minimizzazione del costo del percorso del singolo agente,
ovvero la minimizzazione della durata delle azioni fino all’ottenimento del goal.
Una volta creata l’istanza del problema, si è resa necessaria l’individuazione di un
planner per effettuare la risoluzione dello stesso. La libreria planutils fornisce una
serie di planner, tra cui POPF, TFD, LPG-TD, che permettono la risoluzione
dell’istanza del problema in quanto supportano le durative actions. E’ stato utilizzato
POPF (Partial Order Planner for Forward-chaining) è un pianificatore automatico
specializzato nella gestione di problemi di pianificazione temporale. Utilizza una
strategia di ricerca forward-chaining e un ordinamento parziale delle azioni,
permettendo una maggiore flessibilità nella pianificazione. I principali vantaggi di
POPF includono la capacità di gestire azioni con durata specifica, ottimizzando l'uso
delle risorse e il tempo complessivo grazie alla possibilità di eseguire azioni in
parallelo quando non vi sono conflitti. Questo approccio risulta spesso più efficiente
rispetto ad altri metodi, rendendo POPF uno strumento potente per creare piani
realistici e ottimizzati in contesti temporali complessi.
Il dominio descritto nel capitolo 1 come detto in precedenza è stato modificato al fine
di utilizzare azioni durative e renderlo eseguibile dai planner di planutils e plansys2.
Oltre alle modifiche esplicitamente richieste, ne sono state apportate ulteriori per
soddisfare i requisiti dei planner in questione. Per non incorrere in bug, sono state
apportare modifiche a nomi di predicati e azioni. Sono stati cambiati anche alcuni
oggetti come carrier in vehicle a causa di alcuni errori durante il parsing e l’unico
modo di risolverli è stato cambiare nome agli oggetti che generavano conflitto.
## Robotics Planning

Per eseguire il piano è stata necessaria la creazione di varie cartelle all’interno del
workspace, quali launch src pddl, inoltre altre cartelle sono state create in seguito
all’utilizzo di determinati comandi per la corretta esecuzione del plan in ROS.
• pddl: directory che contiene il file di dominio “domain.pddl”
• launch: directory che contiene il file di launch “plansys2_project_launch.py”
utilizzato per avviare i nodi ros necessari all’esecuzione del piano; il piano
generato salvato nel file “plan.txt” ed il file “commands” che serve come
parametro al terminale di plansys2 per settare le istanze del problema.
• src: directory che contiene i file relativi alle specifiche azioni implementate nel
dominio e scritte in linguaggio C++. Contiene il file “CMakeLists.txt”, utile al
building della cartella ed il file “package.xml” che contiene le dipendenze
all’interno del package.

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

Per semplicità, di seguito è riportata solo l’implementazione dell’azione moveRobot
in C++:

```C++
#include <memory>
#include <algorithm>
#include<vector>
#include<string>
#include "plansys2_executor/ActionExecutorClient.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

using namespace std::chrono_literals;

class MoveRobotAction : public plansys2::ActionExecutorClient
{
public:
  MoveRobotAction()
  : plansys2::ActionExecutorClient("move_robot", 250ms)
  {
    progress_ = 0.0;
  }

private:
  void do_work()
  { std :: vector <std :: string > arguments = get_arguments () ;
    if (progress_ < 1.0) {
      progress_ += 0.2;
      send_feedback(progress_, "Robot "+arguments [0]+
            " from "+ arguments [1] + " to "+
            arguments [1]);
    } else {
      finish(true, 1.0, "Robot "+arguments [0]+
            " from "+ arguments [1] + " to "+
            arguments [1]);
      progress_ = 0.0;
      std::cout << std::endl;
    }

    std::cout << "\r\e[K" << std::flush;
    std::cout << "Robot "+arguments [0]+
            " from "+ arguments [1] + " to "+
            arguments [1] +" . . . [ " << std::min(100.0, progress_ * 100.0) << "% ]  " <<
            std::flush;
  }

  float progress_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveRobotAction>();
  node->set_parameter(rclcpp::Parameter("action_name", "move_robot"));
  node->trigger_transition(lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE);
  rclcpp::spin(node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}
```


Una volta definiti tutti i file necessari per l’esecuzione della parte relativa al Robotics
Planning, è stato necessario effettuare il building della cartella e l’inizializzazione
dell’ambiente di esecuzione, aprendo un terminale  nella cartella **plansys2_project**

```


rm -rf build/ install/ log/

colcon build --symlink-install

source install/local_setup.bash
```

Per lanciare il terminale plansys2 nella cartella radice:

```
ros2 launch plansys2_project plansys2_project_launch.py
```

Se il dominio PDDL è interpretato correttamente, lo saranno anche tutti i nodi e
produrranno il seguente output:

![[Task 3-20241230160941532.png|600]]
Da un nuovo terminale, si esegue il seguente comando per accedere al terminale di
plansys2:
``` 
ros2 run plansys2_terminal plansys2_terminal
```
Impostiamo le condizioni iniziali tramite il seguente commando:
```
source launch/commands
```
E infine avviamo il plan precedentemente generato tramite popff tramite il seguente
comando
```
run plan-file launch/plan.txt
```

Se tutto è andato a buon fine otterremo il seguente risultato nel secondo terminale:

![[Task 3-20241231191735819.png|600]]

Dall’output possiamo concludere che il piano viene correttamente applicato al
problema e rispetta i vincoli temporali imposti.

Mentre nel secondo terminale otterremo i log di quello che è stato descritto nei file di
configurazione cpp, all’interno dei quali è stato necessario imporre gli stessi nomi del
file di configurazione del CMakeLists.txt e nell’action name fare riferimento ai nomi
del dominio. Qui di seguito un esempio del corretto funzionamento:

![[Task 3-20241230160822732.png|600]]

## Conclusioni 

Il progetto ha raggiunto con successo tutti gli obiettivi prefissati, dimostrando
l'efficacia del modello di pianificazione automatica sviluppato. La modellazione del
problema utilizzando il linguaggio PDDL e l'implementazione di un algoritmo di
ricerca personalizzato hanno permesso di ottenere piani d'azione efficienti entro
tempi ragionevoli. Inoltre, l'integrazione del modello temporale in un'architettura
software robotica reale ha mostrato la capacità del sistema di gestire azioni durative e
vincoli temporali, utilizzando strumenti come PlanSys2 e POPF e PDDL4J.Tuttavia,
il percorso non è stato privo di difficoltà. Una delle principali sfide incontrate è stata
la familiarizzazione iniziale con le librerie fornite, in particolare con ROS e le sue
dipendenze. Problemi di parsing nei nomi degli oggetti e predicati hanno reso
necessarie modifiche al dominio e agli oggetti, causando ritardi e richiedendo un
notevole sforzo di debugging. Queste difficoltà sono state superate grazie a una
combinazione di approfondimenti pratici e alla consultazione della documentazione
disponibile. I risultati ottenuti sono stati tutti corretti e conformi alle aspettative,
confermando la validità del metodo di pianificazione implementato.