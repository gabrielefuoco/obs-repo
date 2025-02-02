**Modelli di Classificazione**

* **Obiettivo:** Assegnare record non noti a una classe con la massima accuratezza.
    * Utilizza un *training set* per costruire il modello e un *test set* per la validazione.
    * Tipi di classificatori:
        * Classificatori di base
        * Classificatori Ensemble (Boosting, Bagging, Random Forest)

* **Alberi Decisionali:** Tecnica di classificazione che rappresenta regole tramite una struttura gerarchica.
    * **Componenti:**
        * Nodi interni (o di partizionamento): Attributi di splitting.
        * Nodi foglia (o terminali): Valore dell'attributo di classe (classificazione finale).
    * **Proprietà:**
        * Ricerca dell'albero ottimo: problema NP-Completo.
        * Classificazione veloce: O(ω) nel caso peggiore (ω = profondità dell'albero).
        * Robusti rispetto ad attributi fortemente correlati.
    * **Applicazione del modello:**
        * Si parte dal nodo radice e si segue il percorso basato sulle condizioni di test degli attributi.
        * Si assegna la classe del nodo foglia raggiunto.
    * **Tree Induction Algorithm:** Tecniche greedy (dall'alto verso il basso) per costruire l'albero.
        * Problematiche: scelta del criterio di split e di stop, underfitting, overfitting.
        * **Algoritmo di Hunt:** Approccio ricorsivo per suddividere i record in sottoinsiemi più puri.
            * **Procedura di costruzione:**
                1. Se tutti i record hanno la stessa classe, il nodo diventa una foglia con quella classe.
                2. Altrimenti, si sceglie un attributo e un criterio per suddividere i record.
                3. Si applica ricorsivamente la procedura sui sottoinsiemi.




**Scelta del Criterio di Split negli Alberi Decisionali**

I. **Tipi di Attributi e Split**

   * A. **Attributi Binari:** Due possibili risultati.
   * B. **Attributi Nominali:**
      * 1. Split a più vie: Una partizione per ogni valore distinto.
      * 2. Split a due vie: Suddivisione ottimale in due insiemi.
   * C. **Attributi Ordinali:** Simili ai nominali, ma con ordine preservato.
   * D. **Attributi Continui:**
      * 1. Binario (split a due vie): Test del tipo  $A < v$.
      * 2. A più vie: Suddivisione in intervalli: $v_{i} \leq A \leq v_{i+1}$.
   * E. **Partizionamento:**
      * 1. **Split a due vie (Ordinali):**  Considera tutti i possibili valori *v* per il test $A < v$ (computazionalmente costoso).
      * 2. **Split a più vie (Continui):** Discretizzazione in intervalli disgiunti.


II. **Criterio di Ottimizzazione dello Split**

   * A. **Obiettivo:** Creare nodi figli il più puri possibile (istanze della stessa classe nello stesso nodo).
   * B. **Nodi Impuri:** Aumentano la profondità dell'albero, causando overfitting, minore interpretabilità e maggiore costo computazionale.
   * C. **Misure di Impurità:** Bilanciano purezza dei nodi e complessità dell'albero.


III. **Misure di Impurità dei Nodi**

   * A. **Valutazione dell'impurità di un nodo *t* (con *k* classi e *n* nodi figli):**
      * 1. **Gini Index:** $GINI(t) = 1 - \sum_{j=1}^{k} [p(j|t)]^2$
      * 2. **Entropy:** $Entropy(t) = -\sum_{j=1}^{k} p(j|t) \log_2 p(j|t)$
      * 3. **Misclassification Error:** $Error(t) = 1 - \max p(i|t)$
      * 4. **Impurità complessiva:** $Impurity_{split} = \sum_{i=1}^{n} \frac{m_i}{m} meas(i)$  (*p(j|t)* = frequenza della classe *j* nel nodo *t*)
   * B. **Determinare il Miglior Partizionamento:** Calcolare l'impurità *P* del nodo genitore prima dello splitting.



