
# Prompting per Large Language Models (LLM)

## I. Definizione e Importanza del Prompt

* **Definizione:** Un prompt è il messaggio fornito al decoder di un Large Language Model (LLM) per la generazione di testo.

* **Importanza:** Il prompt influenza significativamente i risultati, soprattutto nei compiti di ragionamento.  Anche minime variazioni nel prompt possono produrre output molto diversi.


## II. Zero/Few-Shot Prompting

* **Tecnica:**  Questa tecnica utilizza esempi nel prompt per guidare il modello ( *in-context learning*).

* **Limiti:** Presenta *diminishing returns*: le prestazioni stagnano dopo un certo numero di esempi. È inadeguato per compiti complessi di ragionamento, anche per grandi LLM. Richiede inoltre un adeguato addestramento a monte del modello.


## III. Chain-of-Thought Prompting

* **Tecnica:**  Induce il ragionamento passo-passo senza esempi espliciti, utilizzando espressioni come "Let's think step by step".

* **Approccio:** Si tratta di *zero-shot learning* con una generazione a due fasi (la prima invisibile all'utente).

* **Caratteristica:** È un esempio di *self-conditioning*.


## IV. Tecniche di Prompting Avanzate

* **Tree of Thought:** Genera diversi percorsi di ragionamento e utilizza una *tree search* per valutare il migliore. Incoraggia la considerazione di possibilità multiple.

* **Prompting Maieutico (Socratico):** Richiede una spiegazione e successivamente la spiegazione delle spiegazioni, creando un albero di spiegazioni per identificare inconsistenze. Incoraggia il pensiero critico e l'analisi delle assunzioni.

* **Complexity-Based Prompting:** Evoluzione del *Tree of Thought*, utilizza diversi *rollout* di catene di pensiero.

* **Self-Refine Prompting:** Il modello risolve, critica e poi risolve nuovamente il problema, considerando la critica. Funziona bene per problemi matematici e logici.

* **Generated Knowledge Prompting:** Il modello genera prima informazioni rilevanti e poi le usa per completare il task.


## V. Limitazioni delle Tecniche Base

* **Mancanza di robustezza:** Espressioni semanticamente equivalenti possono produrre risultati diversi.

---

Per favore, forniscimi il testo da formattare.  Ho bisogno del testo che desideri che io organizzi e formati secondo le tue istruzioni per poterti aiutare.

---
