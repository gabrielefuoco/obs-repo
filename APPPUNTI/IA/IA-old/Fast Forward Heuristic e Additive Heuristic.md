Le differenze tra **Fast Forward Heuristic (FF Heuristic)** e **Additive Heuristic** risiedono principalmente nel modo in cui stimano il costo di raggiungere l'obiettivo e nella loro efficienza computazionale. Ecco un confronto schematico:

### 1. **Fast Forward Heuristic (FF Heuristic)**
- **Definizione**: È un'euristica basata sull'approssimazione del piano di soluzione. Considera le precondizioni di ogni azione, ignorando le interazioni negative (cioè gli effetti deleteri di un'azione su un'altra). Stima il costo a partire dallo stato corrente fino all'obiettivo tramite un "relaxed planning graph".
- **Calcolo**: 
  - La FF Heuristic risolve un problema di planning in una versione semplificata (relaxed) del problema originale, dove si ignorano gli effetti negativi delle azioni.
  - Dopo aver costruito il relaxed planning graph, calcola il costo come il numero di azioni necessarie per soddisfare tutte le precondizioni di ogni azione richiesta per raggiungere l'obiettivo.
- **Caratteristiche**:
  - **Ottimistica**: La stima tende a essere ottimistica poiché ignora conflitti tra azioni.
  - **Efficiente**: Veloce da calcolare, grazie alla semplificazione (relaxed problem).
  - **Buona approssimazione** per problemi di pianificazione classica.

### 2. **Additive Heuristic**
- **Definizione**: È un'euristica che somma i costi per ogni sottobiettivo, prendendo in considerazione le precondizioni di ogni azione. Tuttavia, a differenza della FF Heuristic, non ignora gli effetti negativi, ma tratta le precondizioni in modo indipendente.
- **Calcolo**: 
  - Per ciascun obiettivo o precondizione, viene calcolato un costo euristico come se fosse un sottoproblema indipendente.
  - La somma di questi costi euristici fornisce la stima del costo complessivo per raggiungere l'obiettivo finale.
- **Caratteristiche**:
  - **Non Ottimistica**: In alcuni casi, la stima può essere non ottimistica, perché somma i costi senza considerare che alcune azioni potrebbero risolvere più di un sottobiettivo contemporaneamente.
  - **Più precisa** rispetto alla FF Heuristic in alcuni scenari, ma anche computazionalmente più costosa.
  - **Buona approssimazione** per problemi di planning dove le azioni hanno molte precondizioni indipendenti.

---

### **Vantaggi di usare una piuttosto che l'altra**:

- **FF Heuristic**:
  - **Vantaggi**:
    - Molto veloce da calcolare.
    - Buona per problemi dove le interazioni tra precondizioni non sono cruciali.
  - **Svantaggi**:
    - L'ottimismo eccessivo può portare a una ricerca inefficiente in casi con molti conflitti tra azioni.
    - Non considera il costo aggregato di azioni con effetti negativi.
  
- **Additive Heuristic**:
  - **Vantaggi**:
    - Stima più precisa in contesti con molte precondizioni indipendenti.
    - Tiene conto del costo di soddisfare ogni precondizione individualmente, fornendo una valutazione più accurata del costo totale.
  - **Svantaggi**:
    - Più lenta e computazionalmente più costosa, specialmente per problemi di pianificazione di grandi dimensioni.
    - Può sovrastimare il costo complessivo ignorando che alcune azioni possono soddisfare più di una precondizione.

---

**Conclusione**:
- **Usa la FF Heuristic** se hai bisogno di un'euristica veloce e vuoi sacrificare un po' di precisione a favore dell'efficienza.
- **Usa l'Additive Heuristic** quando desideri una stima più accurata, specialmente in contesti in cui le precondizioni sono in gran parte indipendenti e il costo computazionale non è un problema.