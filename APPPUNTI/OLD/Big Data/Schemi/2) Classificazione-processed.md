
**Schema Riassuntivo sulla Classificazione**

**I. Obiettivo della Classificazione**

    *   Assegnare una classe a record invisibili con la massima accuratezza.
    *   Apprendere un modello per riconoscere un concetto (es. alberi).
    *   Classificare istanze non viste.
    *   **Avvertenza:** I modelli sono approssimazioni, non garantiti corretti o completi.

**II. Approccio alla Classificazione**

    *   Dato un insieme di record (training set).
    *   Ogni record contiene un insieme di attributi.
    *   Un attributo è l'attributo di classe (etichetta) da predire.
    *   Si apprende un modello per l'attributo di classe in funzione degli altri attributi.

**III. Varianti di Classificazione**

    *   Classificazione binaria (es. frode/nessuna frode, vero/falso).
    *   Classificazione multiclasse (es. basso, medio, alto).
    *   Classificazione multi-etichetta (più di una classe per record, es. interessi dell'utente).

**IV. Tecniche di Classificazione: Alberi di Decisione**

    *   Codificano una procedura per prendere una decisione di classificazione.
    *   **Apprendimento di un Albero di Decisione:**
        *   Trovare l'albero ottimale è NP-hard.
        *   Algoritmi usano una strategia di partizionamento avida, dall'alto verso il basso e ricorsiva.
        *   Esempi di algoritmi: Algoritmo di Hunt, C4.5 Rules.

**V. Overfitting**

    *   **Definizione:** Il modello apprende i dettagli specifici dei dati di addestramento con troppa precisione, perdendo la capacità di generalizzare.
    *   **Conseguenza:** Prestazioni scadenti su dati non visti.
    *   **Esempio:** Modello che descrive un albero come "una grande pianta verde con un tronco e nessuna ruota".
    *   **Soluzione:** Trovare il giusto equilibrio tra specificità e generalizzazione del modello.

---
