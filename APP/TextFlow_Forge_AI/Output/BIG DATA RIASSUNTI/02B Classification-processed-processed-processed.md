
# Classificazione: Un'Introduzione

La classificazione è una tecnica di apprendimento automatico che assegna un record a una classe predefinita, basandosi su un insieme di attributi.  L'obiettivo è predire accuratamente la classe di record precedentemente invisi, utilizzando un modello appreso da un *training set* contenente esempi con attributi e relative etichette di classe.  Esistono diverse varianti: classificazione binaria (due classi), multiclasse (più di due classi) e multi-etichetta (più classi per record).

## Il Processo di Classificazione

Il processo inizia con l'osservazione di esempi positivi e negativi (dati di addestramento) per apprendere un modello che descriva il concetto da classificare.  Questo modello, tuttavia, è solo un'approssimazione e non garantisce precisione o completezza nella classificazione di dati non visti.

## Tecniche di Classificazione

Esistono numerose tecniche di classificazione, tra cui: K-Nearest-Neighbors, Alberi di decisione, Apprendimento di regole, Naïve Bayes, Macchine a vettori di supporto, Reti neurali artificiali e Reti neurali profonde.

### Alberi di Decisione

Gli alberi di decisione rappresentano una procedura decisionale visivamente intuitiva (vedi `![[](_page_5_Figure_1.jpeg)` e `![[](_page_6_Figure_1.jpeg)`).  La costruzione di un albero di decisione ottimale è un problema NP-hard, quindi gli algoritmi utilizzano strategie avide dall'alto verso il basso, come l'Algoritmo di Hunt e C4.5 Rules, per trovare una soluzione ragionevole.

## Overfitting

Un problema critico nella classificazione è l'overfitting: il modello apprende i dettagli specifici dei dati di addestramento con troppa precisione, perdendo la capacità di generalizzare a nuovi dati.  Questo porta a prestazioni scadenti su dati non visti.  L'obiettivo è trovare un equilibrio tra la specificità del modello (adattamento ai dati di addestramento) e la sua capacità di generalizzazione (classificazione corretta di dati nuovi).  Un esempio di overfitting è una descrizione di un albero troppo specifica, che funziona bene solo sui dati di addestramento ma fallisce con altri esempi.

---
