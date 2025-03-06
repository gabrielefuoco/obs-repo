
## Schema Riassuntivo del Prompting

**1. Definizione e Importanza del Prompt**

*   **1.1 Definizione:** Messaggio fornito al decoder con funzionalità di language modelling causale (generazione del testo).
*   **1.2 Importanza:** Variazioni nel prompt possono influenzare significativamente i risultati, specialmente in compiti che richiedono ragionamento.

**2. Zero/Few-Shot Prompting**

*   **2.1 Descrizione:** Tecnica utilizzata come *in-context learning*.
*   **2.2 Diminishing Returns:** Le prestazioni migliorano fino a un certo numero di esempi, poi si stabilizzano.
*   **2.3 Requisiti:** Richiede un modello adeguatamente addestrato a monte.
*   **2.4 Limitazioni:** Alcuni compiti complessi rimangono difficili anche per LLM di grandi dimensioni.

**3. Chain-of-Thought Prompting**

*   **3.1 Descrizione:** Tecnica per *in-context learning* few-shot senza specificare esplicitamente il ragionamento.
*   **3.2 Implementazione:** Utilizzo di espressioni come **"Let's think step by step"** per indurre il ragionamento.
*   **3.3 Approccio:** Zero-shot learning.
*   **3.4 Generazione a Due Fasi:** La prima fase di ragionamento non è visibile all'utente.
*   **3.5 Self-Conditioning:** Abilità acquisita da modelli di grandi dimensioni.

**4. Limitazioni delle Tecniche di Prompting di Base**

*   **4.1 Mancanza di Robustezza:** Espressioni semanticamente equivalenti possono avere impatti diversi.

**5. Tecniche di Prompting Avanzate**

*   **5.1 Tree of Thought**
    *   **5.1.1 Descrizione:** Genera più passi successivi e utilizza un metodo di *tree search* per valutare i percorsi.
    *   **5.1.2 Obiettivo:** Incoraggia a considerare possibilità multiple.
*   **5.2 Prompting Maieutico (Socratico)**
    *   **5.2.1 Descrizione:** Richiede una risposta con spiegazione e poi chiede di spiegare ulteriormente parti della spiegazione.
    *   **5.2.2 Obiettivo:** Creare un albero di spiegazioni per identificare e eliminare le inconsistenze.
    *   **5.2.3 Approccio:** Inizialmente senza condizionamenti, poi con richiesta di motivazione.
*   **5.3 Complexity-Based Prompting**
    *   **5.3.1 Descrizione:** Evoluzione del *Tree of Thought* che prevede diversi *rollout* di catene di pensiero.
*   **5.4 Self-Refine Prompting**
    *   **5.4.1 Descrizione:** Il modello deve: 1) risolvere il problema; 2) criticare la soluzione; 3) risolvere nuovamente considerando soluzione e critica.
    *   **5.4.2 Applicazioni:** Problemi matematici e di ragionamento logico.
*   **5.5 Generated Knowledge Prompting**
    *   **5.5.1 Descrizione:** Il modello genera prima fatti rilevanti o informazioni di background, poi le usa per eseguire il task.

---
