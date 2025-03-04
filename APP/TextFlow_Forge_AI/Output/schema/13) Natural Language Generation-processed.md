
**Natural Language Generation (NLG)**

* **Definizione e Relazione con NLP:**
    * NLG è un ramo dell'NLP.
    * NLP = Comprensione del Linguaggio Naturale (NLU) + Generazione del Linguaggio Naturale (NLG).
    * Obiettivo: produrre output linguistici fluidi, coerenti e utili.
* **Tipi di Generazione:**
    * **Open-ended generation:** alta libertà nella distribuzione dell'output.
    * **Non-open-ended generation:** l'input determina principalmente l'output.  Formalizzabile tramite entropia.  Diverse strategie di decodifica/addestramento richieste.
* **Meccanismi di Produzione dell'Output:**
    * **Greedy Decoding:**
        * Seleziona il token con la probabilità più alta dato il contesto:  $\hat{y}_{t}=\arg\max_{w\in V}P(y_{t}=w|y_{<t})$
        * Preferibile per task encoder-decoder classici (traduzione automatica, summarizzazione).
    * **Beam Search Decoding:**
        * Genera contemporaneamente più traiettorie (ipotesi).
        * Numero di traiettorie controllabile.
        * Output finale: traiettoria con probabilità cumulativa più alta.
        * Ottimale per contesti aperti e astratti.
    * **Ridurre le Ripetizioni:**
        * Controllo della taglia degli n-gram da evitare (non-repeated n-gram size).
        * Approccio agnostico rispetto alla probabilità di generazione.
        * Necessario bilanciare coerenza/fluency con creatività/diversità/novelty.
        * Campionamento random per massimizzare la creatività (solo in task open-ended estremizzati).
    * **Top-k sampling:**
        * Campionamento tra i *k* token con maggiore probabilità.
        * Considera la massa della distribuzione di probabilità concentrata su un piccolo sottoinsieme di token.


---

**Schema Riassuntivo: Tecniche di Campionamento e Controllo della Randomness nella Generazione di Linguaggio Naturale**

I. **Tecniche di Campionamento**

   A. **Top-k Sampling:**
      * Seleziona i *k* token con la probabilità più alta.
      * *k* è fisso e potrebbe non essere ottimale per tutte le distribuzioni.

   B. **Top-p (Nucleus) Sampling:**
      * Seleziona i token che contribuiscono cumulativamente a una probabilità superiore a *p*.
      * *k* varia dinamicamente a seconda della forma della distribuzione.
      * Può essere combinato con il top-k per gestire distribuzioni non uniformi.

II. **Controllo della Randomness: Temperatura (τ)**

   A. **Funzione:** Scala i logit nella funzione softmax.  $P_t(y_t = w) = \frac{exp(s_w / \tau)}{\sum_{w'∈V} exp(s_{w'} / \tau)}$

   B. **Effetto della Temperatura:**
      * **τ > 1:** Distribuzione più uniforme, output più diversificato (creativo), rischio di allucinazioni.
      * **τ < 1:** Distribuzione più appuntita, output meno diversificato (meno creativo).

III. **Recall e Softmax**

   A. **Calcolo della Probabilità:** Al passo *t*, il modello calcola la distribuzione di probabilità *P<sub>t</sub>* usando la softmax: $P_t(y_t = w) = \frac{exp(s_w)}{\sum_{w'∈V} exp(s_{w'})}$

   B. **Influenza della Temperatura sulla Softmax:** La temperatura *τ* ribilancia *P<sub>t</sub>*, modificando la forma della distribuzione di probabilità come descritto nel punto II.B.

---

**Ottimizzazione della Temperatura nel Decoding**

* **Iperparametro Temperatura:** La temperatura è un parametro cruciale nel processo di decoding.

    * **Influenza su Beam Search:** La temperatura influenza la ricerca beam search, modificando la probabilità di selezione dei token.
    * **Influenza su Sampling:** La temperatura influenza il processo di sampling, modificando la distribuzione di probabilità dei token.
    * **Ottimizzazione:** La temperatura è un iperparametro che necessita di ottimizzazione per ottenere risultati ottimali sia con beam search che con sampling.

---
