

| Termine | Spiegazione |
|---|---|
| **Ricerca di Stati Finali** | Strategia di ricerca che si focalizza sul trovare uno stato finale desiderabile, senza considerare il cammino percorso per raggiungerlo. |
| **Ambiente non deterministico** | Ambiente in cui lo stato successivo non è completamente determinato dallo stato corrente e dall'azione compiuta. |
| **Ambiente non osservabile** | Ambiente in cui l'agente non ha accesso allo stato completo dell'ambiente in ogni momento. |
| **Ricerca Hill Climbing** | Algoritmo di ricerca locale che ad ogni iterazione passa allo stato vicino con valore più alto, senza considerare gli stati oltre quelli immediatamente vicini. |
| **Massimo locale** | Stato in cui il valore è maggiore rispetto a tutti gli stati vicini, ma non necessariamente il valore massimo globale. |
| **Simulated Annealing** | Algoritmo che combina l'Hill Climbing con una esplorazione casuale per evitare di rimanere intrappolati in massimi locali. |
| **Temperatura** | Parametro che controlla l'intensità dell'esplorazione casuale nel Simulated Annealing. |
| **Piano condizionale** | Piano che specifica azioni diverse a seconda dello stato osservato in un ambiente non deterministico. |
| **Albero AND-OR** | Struttura ad albero utilizzata per rappresentare piani condizionali in ambienti non deterministici. |
| **Nodi OR** | Nodi in un albero AND-OR che rappresentano una scelta di azione da parte dell'agente. |
| **Nodi AND** | Nodi in un albero AND-OR che rappresentano i possibili risultati di un'azione non deterministica. |
| **Piano ciclico** | Piano che può essere ripetuto finché l'azione non ha successo in un ambiente non deterministico. |
| **Belief state** | Insieme di stati possibili in cui l'agente potrebbe trovarsi in un ambiente con osservazioni parziali. |
| **Predizione** | Fase in cui l'agente sceglie un'azione possibile in un ambiente con osservazioni parziali. |
| **Percezione** | Fase in cui l'agente calcola l'insieme delle possibili percezioni in un ambiente con osservazioni parziali. |
| **Aggiornamento** | Fase in cui l'agente aggiorna il belief state combinando l'azione scelta e la percezione. |

## Ricerca di Stati Finali

Le strategie di ricerca precedenti si focalizzavano sul trovare cammini attraverso lo spazio degli stati. Altre volte, siamo interessati solo allo stato finale e non al cammino percorso per raggiungerlo. Ad esempio, nel problema delle n regine, ci interessa solo trovare una configurazione valida.

In sostanza, la soluzione era una sequenza di azioni, mentre ora ci interessa solo trovare un buon stato (non il relativo cammino), eventualmente rilassando anche le ipotesi di determinismo e di osservabilità. Un buon stato si intende come un ottimo locale.

* **Ambiente non deterministico:** lo stato successivo non è completamente determinato dallo stato corrente e dall'azione compiuta dall'agente.
	*  L'agente dovrà avere un piano condizionale ed eseguire azioni diverse a seconda di ciò che osserva.
* **Ambiente non osservabile:** l'agente non ha accesso, in ogni momento, allo stato completo dell'ambiente.
	* L'agente dovrà anche tenere traccia dei possibili stati in cui potrebbe trovarsi.

### Ricerca Hill Climbing

L'algoritmo di ricerca Hill Climbing tiene traccia solo dello stato corrente e, ad ogni iterazione, passa allo stato vicino con valore più alto (punta nella direzione che presenta l'ascesa più rapida), senza considerare gli stati oltre quelli immediatamente vicini. Naturalmente, si potrebbe rimanere intrappolati in un massimo locale e, quindi, non raggiungere l'ottimo globale. È chiamato anche ricerca locale greedy.

### Simulated Annealing

È un algoritmo che permette di evitare di rimanere intrappolati in ottimi locali, combinando l'Hill Climbing con una esplorazione casuale. Intuitivamente, inizialmente si può scuotere molto lo stato corrente (alta temperatura) per poi ridurre gradualmente l'intensità dello scuotimento.

Come per l'Hill Climbing, a partire dallo stato corrente viene scelto un vicino. Tuttavia, in questo algoritmo non viene scelta la mossa migliore ma una mossa casuale. Se la mossa migliora la situazione $(∆E > 0)$, essa viene sempre accettata altrimenti la si accetta con probabilità $e^\left( -∆ \frac{E}{T} \right)$ dove T è la temperatura, che diminuisce ad ogni iterazione.

Si noti che la probabilità decresce esponenzialmente in funzione della cattiva qualità della mossa e della temperatura e, pertanto, alle prime iterazioni (dove T è alto) l'accettazione di mosse cattive sarà maggiore. Se la velocità di raffreddamento fa decrescere la temperatura da T a 0 molto lentamente, per una proprietà della distribuzione di Boltzmann $\left( e^\left( -∆ \frac{E}{T} \right) \right)$, tutta la probabilità è concentrata sui massimi locali, che l'algoritmo troverà con probabilità 1.

### Ricerca con Azioni non Deterministiche

Se mi trovo in un certo stato, facendo una determinata azione posso finire in più stati. Dunque, è necessario utilizzare un algoritmo che calcoli piani condizionali (il piano deve funzionare per ogni possibile conseguenza dell'azione).

Si può vedere un piano condizionale come una sequenza di passi if-then-else e azioni. Più formalmente, un piano è un Albero AND-OR. Un possibile piano è il seguente:

```
[azione,
if(stato = x)
then
piano1
else
...]
```

Nell'istruzione if si verifica quale è lo stato corrente in seguito ad una azione non deterministica, che osserviamo solo al momento dell'esecuzione e che non conosciamo al momento della pianificazione.

### Ricerca AND-OR

Questo algoritmo di ricerca genera piani condizionali, ovvero degli alberi caratterizzati in maniera differente rispetto ai precedenti.

In un ambiente deterministico, l'unica ramificazione è introdotta da una scelta di azione da parte dell'agente... si parla di nodi OR. In un ambiente non deterministico, la ramificazione è anche legata alla scelta del risultato di un'azione da parte dell'ambiente... si parla di nodi AND. Questi due tipi di nodi si alternano generando Alberi AND-OR.

I nodi degli stati sono dei nodi OR in cui deve essere scelta un'azione, mentre nei nodi AND bisogna gestire ogni possibile risultato.

Una soluzione per un problema di ricerca AND-OR consiste in un sotto-albero dell'albero AND-OR completo che:

* Ha un obiettivo (nodo) in ogni nodo foglia.
* Specifica una sola azione in ognuno dei suoi nodi OR.
* Include ogni ramo uscente dai suoi nodi AND.

Se lo stato corrente è identico ad uno stato sul cammino della radice, l'algoritmo termina con failure (poiché vi è un ciclo). Con questo controllo ci si assicura che l'algoritmo termini in ogni spazio degli stati finito poiché ogni cammino finisce in un obiettivo, un vicolo cieco oppure in uno stato ripetuto.

Supponiamo di trovarci in un contesto ancora più reale, dove le azioni possono fallire. In questo caso possiamo trovare delle soluzioni cicliche dove l'azione può essere ripetuta finché essa non ha successo.

Un piano ciclico è una soluzione se:

* Ogni foglia è uno stato obiettivo.
* Ogni foglia è raggiungibile da qualsiasi punto nel piano.
* Il fallimento dell'azione è causato dal non determinismo (quindi ripetendo l'azione essa prima o poi avrà successo) e non da un ambiente osservato parzialmente.

### Ricerca con Osservazioni Parziali

Se non abbiamo una conoscenza completa dello stato si parla di *Ricerca con Osservazioni Parziali*. In questo contesto esistono i **belief states**: una serie di stati possibili per quelle che sono le nostre conoscenze.

Bisogna definire un piano mediante tre fasi:

* **Predizione:** si sceglie un'azione possibile.
* **Percezione:** si calcola l'insieme delle possibili percezioni.
* **Aggiornamento:** si aggiorna il belief state combinando azione scelta e percezione.

Si utilizza anche in questo caso una ricerca AND-OR che, tuttavia, non dispone di un singolo stato ma di un insieme di stati (belief states) ed è presente un meccanismo di aggiornamento di quest'ultimo.
