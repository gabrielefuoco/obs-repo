## Prompt

**Definizione:** Messaggio fornito al decoder con funzionalità di language modelling causale (generazione del testo).

**Importanza:** Il peso del prompt è significativo; variazioni nel prompt possono produrre risultati molto diversi, soprattutto quando si richiedono risposte che necessitano di ragionamento.

## Zero/Few-shot prompting

![[14) Prompting-20241209162528441.png|600]]

Questa tecnica è utilizzata anche come *in-context learning*. Si osserva un fenomeno di *diminishing returns*: dopo un certo numero di esempi, le prestazioni non migliorano ulteriormente. È fondamentale che il modello sia adeguatamente addestrato a monte.

Alcuni compiti risultano troppo complessi anche per i Large Language Models (LLM) più grandi, se affrontati solo con il prompting. Questo è particolarmente vero per i compiti che richiedono un ragionamento più ricco e articolato. (Anche gli umani faticano con questi compiti!)

## Chain-of-thought prompting

![[14) Prompting-20241209163043338.png|600]]

Se si desidera effettuare *in-context learning* in modalità few-shot, non è necessario specificare esplicitamente un ragionamento. Questo comportamento può essere ottenuto evitando che l'utente debba scrivere un prompt articolato, utilizzando un'espressione come **"Let's think step by step"**. Ciò si riconduce ad un approccio di tipo zero-shot learning. Si verifica una sorta di generazione a due fasi, dove la prima fase non è visibile all'utente. Questo rappresenta un primo esempio di *self-conditioning*: un'abilità acquisita da modelli di una certa dimensione.

![[14) Prompting-20241209164334568.png|600]]

Le tecniche di prompting finora descritte presentano delle limitazioni. Ad esempio, la mancanza di robustezza: espressioni semanticamente equivalenti possono avere un impatto diverso sul risultato. Esistono, tuttavia, tecniche di prompting molto più sofisticate. Tra queste:

### Tree of Thought

Questa tecnica genera uno o più possibili passi successivi, e poi utilizza un metodo di *tree search* per valutare ogni possibile percorso. Incoraggia l'IA a considerare possibilità multiple o diversi rami di ragionamento.

### Prompting Maieutico (Socratico)

Si esegue un prompt per condizionare il modello a fornire una risposta con una spiegazione, e successivamente si chiede di spiegare ulteriormente parti della descrizione precedentemente fornita. Questo porta alla creazione di un albero di spiegazioni, permettendo di identificare e eliminare le inconsistenze. Incoraggia il modello a pensare criticamente e a svelare assunzioni o implicazioni. Inizialmente il modello è lasciato libero da ogni condizionamento, e solo in seguito gli viene chiesto di motivare ogni parte della risposta.

### Complexity-Based Prompting

Questa tecnica, un'ulteriore evoluzione del *Tree of Thought*, prevede di seguire diversi *rollout* di catene di pensiero.

### Self-Refine Prompting

Simile al prompting maieutico, questa tecnica richiede al modello di: 1) risolvere il problema; 2) criticare la soluzione fornita; 3) risolvere nuovamente il problema considerando sia la soluzione iniziale che la critica ad essa. Funziona bene per problemi matematici o problemi di ragionamento logico.

### Generated Knowledge Prompting

In questo approccio, si chiede prima al modello di generare fatti rilevanti o informazioni di background che potrebbero essere necessarie per completare il task, e successivamente di utilizzare queste informazioni per eseguire il task stesso.
