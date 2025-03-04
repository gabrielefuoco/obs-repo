
**Prompting per Large Language Models (LLM)**

I. **Definizione e Importanza del Prompt:**
    * Definizione: Messaggio fornito al decoder LLM per la generazione di testo.
    * Importanza:  Il prompt influenza significativamente i risultati, soprattutto in compiti di ragionamento. Variazioni minime possono produrre output molto diversi.

II. **Zero/Few-Shot Prompting:**
    * Tecnica: Utilizzo di esempi nel prompt per guidare il modello (*in-context learning*).
    * Limiti: Diminishing returns (prestazioni stagnanti dopo un certo numero di esempi); inadeguato per compiti complessi di ragionamento, anche per grandi LLM.  Richiede un addestramento adeguato a monte del modello.

III. **Chain-of-Thought Prompting:**
    * Tecnica:  Induce il ragionamento passo-passo senza esempi espliciti, usando espressioni come "Let's think step by step".
    * Approccio: Zero-shot learning con generazione a due fasi (la prima invisibile all'utente).
    * Caratteristica: Esempio di *self-conditioning*.

IV. **Tecniche di Prompting Avanzate:**
    * **Tree of Thought:** Genera diversi percorsi di ragionamento e usa una *tree search* per valutare il migliore. Incoraggia la considerazione di possibilità multiple.
    * **Prompting Maieutico (Socratico):**  Richiesta di spiegazione e successiva spiegazione delle spiegazioni, creando un albero di spiegazioni per identificare inconsistenze. Incoraggia il pensiero critico e l'analisi delle assunzioni.
    * **Complexity-Based Prompting:** Evoluzione del *Tree of Thought*, utilizza diversi *rollout* di catene di pensiero.
    * **Self-Refine Prompting:**  Il modello risolve, critica e poi risolve nuovamente il problema, considerando la critica.  Funziona bene per problemi matematici e logici.
    * **Generated Knowledge Prompting:** Il modello genera prima informazioni rilevanti e poi le usa per completare il task.


V. **Limitazioni delle Tecniche Base:**
    * Mancanza di robustezza: espressioni semanticamente equivalenti possono produrre risultati diversi.

---
