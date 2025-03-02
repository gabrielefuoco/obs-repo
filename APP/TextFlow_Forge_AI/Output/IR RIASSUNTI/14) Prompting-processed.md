
## Prompting per Large Language Models: Tecniche e Strategie

Un **prompt** è il messaggio fornito a un modello di linguaggio per generare testo. La sua formulazione è cruciale, influenzando significativamente i risultati, soprattutto nei compiti che richiedono ragionamento.

### Zero/Few-shot Prompting e Diminishing Returns

Il *zero/few-shot prompting*, anche noto come *in-context learning*, consiste nel fornire al modello zero o pochi esempi prima della richiesta.  ![[]]  Si osserva un fenomeno di *diminishing returns*: l'aggiunta di ulteriori esempi oltre una certa soglia non migliora significativamente le prestazioni.  L'efficacia dipende dall'addestramento del modello; compiti complessi possono risultare insolubili anche per i modelli più grandi, soprattutto se richiedono ragionamento articolato.

### Chain-of-Thought Prompting e Self-Conditioning

Il *chain-of-thought prompting* facilita l'in-context learning evitando prompt complessi.  ![[]]  Utilizzando espressioni come "Let's think step by step", si induce il modello a ragionare per fasi, simulando un approccio zero-shot.  Questo processo a due fasi (la prima invisibile all'utente) è un esempio di *self-conditioning*, un'abilità di modelli di grandi dimensioni. ![[ ]]

### Tecniche di Prompting Avanzate

Le tecniche base presentano limitazioni, come la mancanza di robustezza semantica.  Tecniche più sofisticate includono:

* **Tree of Thought:** Genera diversi percorsi di ragionamento e li valuta tramite una *tree search*, incoraggiando l'esplorazione di diverse possibilità.
* **Prompting Maieutico (Socratico):**  Il modello fornisce una risposta e la spiegazione, poi viene ulteriormente interrogato sulle sue affermazioni, creando un albero di spiegazioni per identificare inconsistenze. Incoraggia il pensiero critico e l'analisi delle assunzioni.
* **Complexity-Based Prompting:** Un'evoluzione del *Tree of Thought* che utilizza diversi *rollout* di catene di pensiero.
* **Self-Refine Prompting:** Il modello risolve il problema, critica la sua soluzione e poi risolve nuovamente il problema considerando la critica.  Funziona bene per problemi matematici e logici.
* **Generated Knowledge Prompting:** Il modello genera prima informazioni di background rilevanti, poi le usa per risolvere il task.

---
