
## Schema Riassuntivo sulla Generazione del Linguaggio Naturale (NLG)

**1. Introduzione alla Generazione del Linguaggio Naturale (NLG)**

*   **1.1 Definizione:** Ramo dell'Elaborazione del Linguaggio Naturale (NLP)
*   **1.2 Relazione con NLP:**
    *   `NLP = Comprensione del Linguaggio Naturale (NLU) + Generazione del Linguaggio Naturale (NLG)`
*   **1.3 Obiettivo:** Sviluppare sistemi che producano output linguistici fluidi, coerenti e utili.
*   **1.4 Tipi di Generazione:**
    *   **1.4.1 Open-ended generation:** Alta libertà nella distribuzione dell'output.
    *   **1.4.2 Non-open-ended generation:** L'input determina principalmente la generazione dell'output.
    *   **1.4.3 Osservazione:** La categorizzazione può essere formalizzata tramite l'entropia. Diversi approcci di decoding e/o training sono necessari per i due tipi.

**2. Meccanismi di Produzione dell'Output (Decoding)**

*   **2.1 Greedy Decoding:**
    *   **2.1.1 Descrizione:** Seleziona il token che massimizza la negative log-likelihood dato il contesto corrente.
    *   **2.1.2 Formula:**  `\hat{y}_{t}=\arg\max_{w\in V}P(y_{t}=w|y_{<t})`
    *   **2.1.3 Applicazioni:** Preferibile in modelli encoder-decoder classici (traduzione automatica, summarizzazione).
*   **2.2 Beam Search Decoding:**
    *   **2.2.1 Descrizione:** Genera contemporaneamente più traiettorie di generazione (ipotesi).
    *   **2.2.2 Parametro:** Numero di traiettorie da esplorare.
    *   **2.2.3 Output:** La traiettoria con la probabilità cumulativa più alta.
    *   **2.2.4 Applicazioni:** Scelta migliore in contesti aperti e astratti.
*   **2.3 Riduzione delle Ripetizioni:**
    *   **2.3.1 Descrizione:** Controllo della taglia degli n-gram da evitare (non-repeated n-gram size).
    *   **2.3.2 Considerazioni:** Forzatura eccessiva se alcuni token hanno una maggiore probabilità di essere campionati.
    *   **2.3.3 Obiettivo:** Bilanciare coerenza, fluency, creatività, diversità e novità.
    *   **2.3.4 Campionamento Random:** Massimizza la creatività, usabile solo in task open-ended estremizzati.
*   **2.4 Top-k Sampling:**
    *   **2.4.1 Descrizione:** Vincola il modello a campionare tra i *k* token con la maggiore probabilità.
    *   **2.4.2 Motivazione:** Includere token con probabilità non trascurabile di essere selezionati.
    *   **2.4.3 Osservazione:** La massa della distribuzione di probabilità si concentra su un sottoinsieme relativamente piccolo di token.

---

**Schema Riassuntivo: Tecniche di Campionamento in Natural Language Generation**

**1. Problemi del Campionamento Standard:**

*   La forma della distribuzione dei token varia ad ogni step.
*   La soglia *k* (in top-k sampling) potrebbe non essere sempre adeguata.

**2. Top-p (Nucleus) Sampling:**

*   **Definizione:** Seleziona i token che contribuiscono alla massa di probabilità cumulativa superiore a *p*.
*   **Meccanismo:**
    *   Campiona i token con probabilità cumulativa > *p*.
    *   Il valore di *k* (numero di token considerati) si adatta all'uniformità della distribuzione.
*   **Combinazione con Top-k:**
    *   Si può usare top-k (alto, es. 50) e poi top-p.
    *   Top-p risolve problemi di distribuzioni non uniformi dopo top-k.
    *   Top-p non apporta miglioramenti significativi se la distribuzione è già uniforme dopo top-k.

**3. Scaling della Randomness: Temperatura (τ)**

*   **Definizione:** Fattore di scaling dei logit nella softmax.  τ > 0
*   **Formula:**
    *   Probabilità di un token *w* al passo *t*:
        $$P_t(y_t = w) = \frac{exp(s_w)}{\sum_{w'∈V} exp(s_{w'})}$$
    *   Probabilità con temperatura *τ*:
        $$P_t(y_t = w) = \frac{exp(s_w / \tau)}{\sum_{w'∈V} exp(s_{w'} / \tau)}$$
*   **Effetti della Temperatura:**
    *   **τ > 1 (Aumenta la temperatura):**
        *   *P<sub>t</sub>* diventa più uniforme.
        *   Output più diversificato.
        *   Rischio di allucinazioni (confabulazione).
    *   **τ < 1 (Diminuisce la temperatura):**
        *   *P<sub>t</sub>* diventa più appuntita.
        *   Output meno diversificato.
        *   Modello meno creativo.

---

**Schema Riassuntivo: Temperatura nel Decoding**

1.  **Temperatura come Iperparametro nel Decoding**
    *   Definizione: La temperatura è un parametro regolabile nel processo di generazione del testo.

2.  **Ottimizzazione della Temperatura**
    *   Applicabilità: Può essere ottimizzata per diverse strategie di decoding:
        *   Ricerca Beam Search
        *   Sampling

---
