
**Obiettivo:** Assegnare una classe, da un insieme dato di classi, a record precedentemente invisi, nel modo più accurato possibile.

##### Approccio:

* Dato un insieme di record (training set),
* ogni record contiene un insieme di attributi,
* uno degli attributi è l'attributo di classe (etichetta) che dovrebbe essere predetto.
* Si apprende un modello per l'attributo di classe in funzione dei valori degli altri attributi.

##### Varianti:

* Classificazione binaria (es. frode/nessuna frode o vero/falso)
* Classificazione multiclasse (es. basso, medio, alto)
* Classificazione multi-etichetta (più di una classe per record, es. interessi dell'utente)

## Introduzione alla Classificazione

**Obiettivo:** Apprendere un modello per riconoscere un concetto, ad esempio gli alberi.

Noi (o l'algoritmo di apprendimento) osserviamo esempi positivi e negativi (dati di addestramento) ... e deriviamo un modello, ad esempio: "Gli alberi sono grandi piante verdi che hanno un tronco e nessuna ruota".

**Obiettivo:** Classificazione di istanze non viste.

**Avvertenza:** I modelli sono solo approssimazioni degli esempi! Non è garantito che siano corretti o completi!

## Tecniche di Classificazione

#### Classificatori ad Albero di Decisione

![[_page_5_Figure_1.jpeg|511]]

Gli alberi di decisione codificano una procedura per prendere una decisione di classificazione.

### Applicazione di un Albero di Decisione a Dati Non Visti

![[_page_6_Figure_1.jpeg|497]]
### Apprendimento di un Albero di Decisione

Come apprendere un albero di decisione dai dati di addestramento? Trovare un albero di decisione ottimale è un problema NP-hard. Gli algoritmi di costruzione dell'albero utilizzano quindi una strategia di partizionamento avida, dall'alto verso il basso e ricorsiva per indurre una soluzione ragionevole. Sono stati proposti molti algoritmi diversi, ad esempio l'**Algoritmo di Hunt** e **C4.5 Rules**

## Overfitting

L'obiettivo dell'apprendimento automatico è costruire modelli in grado di classificare correttamente dati *non* visti durante la fase di addestramento. L'overfitting rappresenta un problema critico: il modello apprende i dettagli specifici dei dati di addestramento con troppa precisione, perdendo la capacità di generalizzare a nuovi dati. Questo porta a prestazioni scadenti su dati non visti.

Un esempio estremo di overfitting è un modello che descrive un albero come "una grande pianta verde con un tronco e nessuna ruota". Questa descrizione è perfettamente accurata per i dati di addestramento, ma è troppo specifica e non riesce a classificare correttamente altri tipi di alberi o piante.

Il successo dell'apprendimento automatico risiede nella capacità di trovare il giusto equilibrio tra la specificità del modello (che si adatta perfettamente ai dati di addestramento) e la sua generalizzazione (la capacità di classificare correttamente dati nuovi e non visti).
