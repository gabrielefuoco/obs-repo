### Ricerca AND-OR

La **Ricerca AND-OR** è una tecnica utilizzata per risolvere problemi di pianificazione e ricerca in contesti in cui le azioni possono avere esiti incerti o quando ci sono situazioni in cui si devono prendere decisioni basate su più condizioni o scenari. A differenza della ricerca classica, che si concentra su percorsi lineari (sequenze di azioni che portano da uno stato iniziale a uno stato obiettivo), la Ricerca AND-OR si adatta meglio a problemi in cui sono necessari **piani condizionali** e si devono gestire **azioni non deterministiche**.

#### Struttura dell'Albero AND-OR

Un **albero AND-OR** si distingue dagli alberi di ricerca tradizionali per la presenza di due tipi di nodi:

1. **Nodi AND:** Questi rappresentano situazioni in cui tutte le condizioni devono essere soddisfatte per proseguire. Ad esempio, se un nodo AND ha più figli, tutti devono essere risolti (trovando un piano per ciascuno) affinché l'obiettivo venga raggiunto.

2. **Nodi OR:** Questi rappresentano situazioni in cui esiste una scelta tra diverse azioni o esiti. Basta risolvere uno dei figli di un nodo OR per soddisfare l'obiettivo.

L'obiettivo della Ricerca AND-OR è trovare una **soluzione strategica** che risolva l'intero albero, tenendo conto sia dei nodi AND che dei nodi OR.

### Piani Condizionali

Un **piano condizionale** è una strategia che specifica non solo una sequenza di azioni, ma anche decisioni dipendenti dalle condizioni che si verificano durante l'esecuzione del piano. I piani condizionali sono essenziali in ambienti incerti, dove non è possibile prevedere con certezza l'effetto di ogni azione.

#### Esempio:

Supponiamo di dover uscire di casa e non siamo sicuri se pioverà o meno. Un piano condizionale potrebbe essere:

1. Esci di casa.
2. **Se piove**, prendi un ombrello.
3. **Se non piove**, continua senza ombrello.

Qui, la decisione di prendere o meno l'ombrello dipende dalla condizione del tempo, che non è nota in anticipo.

### Ricerca con Azioni non Deterministiche

Le **azioni non deterministiche** sono azioni il cui esito non è prevedibile con certezza. Ogni volta che un'azione non deterministica viene eseguita, può portare a uno o più risultati possibili, ma non è possibile determinare in anticipo quale risultato specifico si verificherà.

#### Esempio:

Considera un robot che deve attraversare una porta automatica. Se il robot si avvicina alla porta, l'azione "passare attraverso la porta" potrebbe avere esiti diversi:

1. La porta si apre e il robot attraversa.
2. La porta non si apre, quindi il robot non può attraversare.

In questo caso, l'azione "passare attraverso la porta" è non deterministica.

### Applicazione della Ricerca AND-OR con Piani Condizionali e Azioni non Deterministiche

Quando si combinano piani condizionali e azioni non deterministiche, la Ricerca AND-OR diventa uno strumento potente. La struttura AND-OR dell'albero consente di modellare decisioni complesse e incerte, in cui ogni ramo del piano deve affrontare l'incertezza delle azioni e prendere decisioni in base alle condizioni incontrate.

**Processo:**

1. **Costruzione dell'Albero AND-OR:** Si costruisce un albero AND-OR dove i nodi AND rappresentano la necessità di risolvere tutte le possibili situazioni derivanti da un'azione non deterministica, e i nodi OR rappresentano le diverse scelte o azioni disponibili.

2. **Ricerca di una Soluzione:** Si cerca una strategia che risolva tutti i nodi AND e almeno un nodo per ogni nodo OR. La soluzione risultante è un piano condizionale che tiene conto delle possibili incertezze.

3. **Esecuzione del Piano Condizionale:** Una volta trovato, il piano condizionale guida l'esecuzione passo per passo, scegliendo l'azione successiva in base alle condizioni osservate e gestendo gli esiti non deterministici delle azioni.

### Conclusione

La **Ricerca AND-OR** è particolarmente utile in contesti complessi in cui le azioni hanno esiti incerti e i piani devono essere condizionali, adattandosi dinamicamente alle condizioni osservate. Essa permette di sviluppare strategie robuste che non solo cercano un percorso verso l'obiettivo, ma lo fanno tenendo conto delle molteplici possibilità e incertezze che possono emergere durante l'esecuzione del piano. Questo approccio è fondamentale in ambiti come la robotica, la pianificazione automatica e i giochi, dove la capacità di gestire l'incertezza e le decisioni condizionali è cruciale.