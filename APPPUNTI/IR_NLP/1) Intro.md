## Linguaggio naturale

* **Unicità:** Il linguaggio naturale è un linguaggio specifico per gli umani, progettato per gestire la complessità della comunicazione.
* **Interazione multimodale:** È fondamentale per interagire in contesti che coinvolgono anche altre modalità, come la visione.

### Recupero delle informazioni (IR) e Elaborazione del linguaggio naturale (NLP)

* **Campi di studio:** Entrambi sono campi della scienza e dell'ingegneria.
* **NLP:** Si concentra sullo sviluppo di sistemi automatici che comprendono e generano linguaggi naturali.
* **IR:** Si concentra sul ritrovamento (semi-)automatico di dati testuali (non strutturati) che soddisfano un bisogno informativo all'interno di grandi collezioni.

### Definizione di IR

#### Enunciato del problema di base:

* **Dato:**
    * **Documenti:** Un insieme di documenti (si presume che sia una collezione statica).
    * **Bisogno informativo (query):** Un singolo termine di ricerca, una stringa di termini, una frase in linguaggio naturale o un'espressione stilizzata che utilizza simboli speciali.
    * **Modello di recupero:** Un meccanismo per determinare se un documento corrisponde alla query.
* **Obiettivo:** Recuperare documenti con informazioni rilevanti per il bisogno informativo dell'utente e che aiutino l'utente a completare un'attività.
    * **Risultati:** Un insieme di risultati. 


## Problemi con i dati testuali

* **Sfide generali:** Tutti i problemi e le sfide noti nella gestione/estrazione dei dati si estendono ai dati testuali, come:
    * Grandi set di dati
    * Alta dimensionalità
    * Dati rumorosi
    * Dati e conoscenza in continua evoluzione
    * Comprensibilità dei risultati
* **Sfide specifiche:** Oltre alle sfide generali, i dati testuali presentano problemi specifici:
    * **Progettazione:** Il testo non è progettato per essere utilizzato dai computer.
    * **Struttura e semantica:** Struttura e semantica complesse e scarsamente definite, con ambiguità nel linguaggio, morfologia, sintassi, semantica e pragmatica.
    * **Linguaggi:** Linguaggi generali vs. linguaggi specifici di dominio.
    * **Multilinguismo:** La gestione di lingue diverse.

## IR vs. altre discipline

**Obiettivo comune:** Rendere più facile trovare le cose, ad esempio sul Web.

**Attività comuni:**

* Ottimizzazione dei motori di ricerca (SEO)
* Crawling dei dati
* Estrazione di documenti di interesse da un enorme mucchio

**Differenze:**

* **Web scraping:** Estrazione di informazioni.
* **Trovare schemi in grandi collezioni:** NLP, Linguistica computazionale.
* **Scoprire informazioni finora sconosciute:** NLP, Estrazione di testo.
* **Discriminazione del testo:** NLP (Machine Learning).
* **Comprensione del testo:** NLP (Machine Learning, AI generativa, ...). 



## Topic Detection and Tracking

**Contesto:** Fine anni '90 (inizio del text mining)

**Definizione:** Capacità di un sistema automatico di acquisire dati (principalmente news o feed) in streaming.

**Importanza:** Era un campo di ricerca rilevante prima dell'avvento dei big data.

**Caratteristiche:**

* **Analisi dei dati incrementale:** I dati vengono analizzati in modo continuo, man mano che vengono acquisiti.
* **Supervisionato o non supervisionato:** Non è specificato se il processo di analisi sia supervisionato o meno.
#### Difficoltà nel Topic Detection and Tracking
* **Task incrementale:** Il sistema automatico deve essere in grado di riconoscere nuovi topic emergenti e di identificare quelli obsoleti.
* **Capacità limitate:** La fase di tracking richiede risorse computazionali limitate.
* **Definizione di topic:** Un topic è un insieme di termini, anche se non è l'unica definizione possibile.
* **Segnale:** Il sistema deve tenere traccia del segnale, ovvero dei dati in ingresso, per identificare i topic.????
---
La **modellazione di topic stocastica** è una tecnica che utilizza modelli probabilistici per analizzare grandi quantità di testo e identificare i temi principali (topic) presenti nei documenti.  Si basa sull'idea che ogni documento sia una combinazione di diversi topic, e che ogni parola abbia una probabilità di appartenere a un determinato topic. Questo approccio permette di scoprire i temi principali, analizzare la loro distribuzione nei documenti e classificare i documenti in base ai topic che contengono. Un esempio di modello di topic stocastico è il Latent Dirichlet Allocation (LDA). 

---
### Relation Extraction

* **Relazione con Named Entity Recognition:** Per estrarre le relazioni tra entità nominate, è necessario prima identificare le entità stesse.
* **Pattern frequenti:** La ricerca di pattern frequenti è un passo fondamentale per l'estrazione di relazioni.
* **Definizione in NLP:** In NLP, la relation extraction si riferisce all'identificazione di relazioni lessicali tra entità nominate.

### Summarization

* **Approccio tradizionale:** Negli anni '90, la summarization era affrontata come estrazione di parole chiave (key phrase extraction).
* **Guida alla summarization:** Le proprietà di un documento possono essere utilizzate per guidare il processo di summarization. 

## Appunti di Studio: KDD Pipeline e NLP

### KDD Pipeline in NLP

* **Fasi:** I task di NLP si svolgono attraverso una sequenza di fasi (unfolding).
* **Rappresentazione:** Prima di poter rappresentare i contenuti informativi, è necessario indicizzare gli elementi costitutivi.
* **Apprendimento:** L'apprendimento avviene a valle del set di risultati (result set).
* **Feature Selection:** Il machine learning può essere utilizzato anche in fase di selezione delle features.

### Valutazione

* **Criteri statistici:** La statistica offre diversi criteri per la valutazione.
* **Accuracy:** L'accuracy presenta diversi problemi nella stima delle prestazioni di un classificatore. 

## Funzioni Principali

* **Funzione principale:**  Il compito principale di un sistema di elaborazione del linguaggio naturale (NLP) è quello di **estrarre informazioni** da testi.
* **Macro categorie:** Le funzioni principali possono essere suddivise in due macro categorie:
    * **Indicativa:** Questa funzione serve a rivelare gli elementi dei contenuti in modo da consentire la determinazione della rilevanza del testo rispetto alle query. È fondamentale per l'esplorazione dei corpus e per il retrieval.
    * **Informativa:** Questa funzione consente di ottenere un surrogato del testo (o di una sua porzione) senza fare necessariamente riferimento al testo originale. Questo concetto è ancora valido per il retrieval, che consiste nel condensare i dati originali in un formato surrogato.

## Funzioni Principali -  Browsing

* **Definizione:** La navigazione (browsing) è solitamente parte di sistemi di ipertesto e ipermedia.
* **Scopo:** Permette agli utenti di scorrere collezioni di testo alla ricerca di informazioni utili.
* **Vantaggi:** Gli utenti non hanno bisogno di:
    * generare descrizioni di ciò che desiderano, o
    * specificare in anticipo gli argomenti di interesse.
* **Funzionamento:** Gli utenti possono semplicemente indicare i documenti che trovano rilevanti.
* **Utilizzo:** La navigazione è utile quando un utente:
    * non ha un bisogno chiaro,
    * non riesce a esprimere il suo bisogno in modo accurato,
    * o è un utente occasionale delle informazioni. 
* **Supporto all'annotazione:** Eventualmente, se la navigazione non è limitata a se stessa, può servire a supporto dell'annotazione.
#### Utilizzo: Valutazioni e Gold Standard
**Scenario:** Valutazioni in assenza di un esperto di dominio per generare un gold standard.

* **Problema:** In alcuni casi, non è possibile ottenere un gold standard (set di dati perfetto) generato da un esperto di dominio.
* **Soluzione:** Si può ricorrere a un **silver standard**, generato in modo automatico.
* **Esempio:** Nel caso di AI generativa e problemi di allineamento dei modelli linguistici di grandi dimensioni (LLM), spesso si utilizza GPT-4 per generare un silver standard.
---
## Funzioni Principali - Estrazione di Informazioni

**Contesto:** Insieme alla classificazione e al clustering, l'estrazione di informazioni è un task fondamentale nell'analisi di informazioni sul web.

**Esempio:** Web wrapping

**Definizione:** Recuperare informazioni specifiche dai documenti, estraendo o inferendo risposte dalla rappresentazione del testo. In pratica, dato un documento già identificato come rilevante per un task di retrieval (a monte di esso), si concentra sulla selezione di informazioni specifiche.

**Esempi:**
* **Retrieval sul web:** Schemi pre-esistenti possono aiutare a identificare le informazioni rilevanti.
* **E-commerce:** La costruzione di un database può facilitare l'estrazione di informazioni specifiche.
**In altre parole:**
* Il sistema ha già identificato un documento come rilevante per una specifica richiesta.
* L'estrazione di informazioni si concentra quindi sull'individuazione di informazioni specifiche all'interno di quel documento.
* Questo processo può essere facilitato da schemi o database pre-esistenti, come nel caso del retrieval sul web o nell'e-commerce.

**Tipi di template:**
* **Slot:** Gli slot nel template sono tipicamente riempiti da una sottostringa del documento.
* **Riempitivi pre-specificati:** Alcuni slot possono avere un set fisso di riempitivi pre-specificati che potrebbero non apparire nel testo stesso.
* **Multipli riempitivi(filler):** Alcuni slot possono consentire più filler .
* **Ordine fisso:** Si assume che gli slot siano sempre in un ordine fisso.

**Modelli di estrazione:**

* **Specificare un elemento:** Specificare un elemento da estrarre per uno slot, ad esempio, utilizzando un modello di regex.
* **Modelli precedenti e successivi:** Potrebbe essere necessario un modello precedente (pre-filler) per identificare il contesto appropriato e un modello successivo (post-filler) per identificare la fine del riempitivo. ==(es:estrazione di termini da un modello. definiziamo il template a priori e per ogni slot creiamo dei pattern)==

---
## Funzioni Principali - Recupero di Documenti

Selezionare documenti da una collezione in risposta a una query dell'utente.

* La richiesta di ricerca è solitamente formulata in linguaggio naturale.
* Classificare questi documenti in base alla loro rilevanza rispetto alla query.
* Abbinamento tra la rappresentazione del documento e la rappresentazione della query.
* Restituire un elenco di possibili testi pertinenti, le cui rappresentazioni corrispondono meglio alla rappresentazione della richiesta.

**Modelli di recupero:** Booleano, Spazio vettoriale, Probabilistico, Basato sulla logica(no), ecc.

**Differenze rispetto a:**

* Rappresentazione dei contenuti testuali.
* Rappresentazione dei bisogni informativi.
* E il loro abbinamento. 

---

## Panoramica delle Applicazioni di Base

Le principali aree di applicazione coprono due aspetti:
* **Scoperta della conoscenza**
    * **Estrazione di informazioni:**  il processo di estrazione di informazioni utili da grandi quantità di dati.
* **Distillazione delle informazioni**
    * **Estrazione basata su struttura:**  l'estrazione di informazioni basata su una struttura predefinita di documenti, per identificare i documenti rilevanti per un'informazione target.

**Utilizzi tipici:**
* Estrarre informazioni rilevanti dai documenti.
* Classificare e gestire i documenti in base al loro contenuto.
* Organizzare repository di meta-informazioni relative ai documenti per la ricerca e il recupero.


* **Categorizzazione gerarchica:**  Usare il retrival per organizzare i documenti in una struttura gerarchica basata su categorie (==seguendo l'esempio delle web directory==).
* **Riassunto del testo:**  Creare un riassunto conciso di un documento, mantenendo le informazioni più importanti.
* **Disambiguazione del senso delle parole:**  Determinare il significato corretto di una parola in un contesto specifico.
- **Filtraggio del Testo (o dell'Informazione)**: Rimuovere il testo non rilevante da un documento. Inizialmente accoppiato alla web personalization, da cui nascono gli attuali sistemi di raccomandazione.
	- **Approcci alla** **Web Personalization**
		* **Selezionare l'informazione per un gruppo di utenti target in base alle loro specifiche esigenze.**
		* **Collaborative based:** il sistema continua a proporre non solo i prodotti in base alle specifiche esigenze, ma anche cose spesso acquistate da utenti con un profilo simile, per un discorso anche di diversificazione e serendipità.
		* **Per gli utenti cold start:** i più popolari. 


**Altri esempi di applicazioni:**

* **CRM e marketing:** (Customer Relationship Managment) Cross-selling, raccomandazioni di prodotti.
* **Raccomandazione di prodotti:**  Suggerire prodotti pertinenti agli utenti in base alle loro preferenze.
* **Consegna di informazioni nelle organizzazioni per la gestione della conoscenza:**  Fornire informazioni pertinenti ai dipendenti per migliorare la collaborazione e la produttività.
* **Personalizzazione dell'accesso alle informazioni:**  Adattare l'accesso alle informazioni alle esigenze individuali degli utenti.
* **Filtraggio di notizie in newsgroup Usenet:**  Rimuovere le notizie non pertinenti o indesiderate.
* **Rilevamento di messaggi spam:**  Identificare e filtrare i messaggi di posta elettronica indesiderati. 
---
## Panoramica delle Applicazioni di Base

### Categorizzazione Gerarchica

Sotto strutture gerarchiche (ad esempio, tassonomie), un ricercatore può:

* Navigare prima nella gerarchia delle categorie
* E poi restringere la sua ricerca a una particolare categoria di interesse

La categorizzazione basata su categorie dovrebbe consentire l'aggiunta di nuove categorie e l'eliminazione di quelle obsolete.

**Caratteristiche:**
* Natura ipertestuale dei documenti:
    * Analisi degli hyperlink
* Struttura gerarchica dell'insieme di categorie
* Decomposizione della classificazione come decisione ramificata:
    * A un nodo interno

### Riassunto

Generare un riassunto del contenuto di un testo:

* **Testo breve:** essenziale e coerente
* Utilizzare profili per strutturare il contenuto importante in campi semanticamente ben definiti==(dipende dalla natura dei documenti: ad esempio per summarization di recensioni ad esempio possiamo a monte decidere di addestrare il summarizer affinhe lavori per aspetti: non deve fare summary in maniera trasversale ma farlo per ognuno degli aspetti elencati)==

Applicato principalmente per facilitare l'accesso alle informazioni, ad esempio:

* Le parole chiave più utili vengono estratte da un insieme di documenti (ad esempio, un cluster) per descriverlo
* I documenti in una collezione vengono astratti per evitare di leggere il contenuto completo
* I documenti recuperati dalla ricerca vengono riassunti per consentire all'utente un'identificazione più rapida di quelli pertinenti alla query

Riassunto di alto livello o panoramica di tutti i punti principali?
Approcci basati sulla dimensione dell'unità di testo utilizzata nel riassunto:

* Riassunti di parole chiave
* Riassunti di frasi

### Disambiguazione del Senso delle Parole

Assegnare una parola con il senso corretto rispetto al contesto in cui la parola appare.
attenzione: l'obiettivo è disambiguare il significato di ogni parola ma dobbiamo tenere conto necessariamente dell'intero contesto(da ampliare per supportare lo strumento automatico. lavoro da fare simultaneamente per ogni termnine del contesto)
**Un approccio efficace:**

* Scegliere i significati delle parole da un inventario di sensi esistente sfruttando misure di correlazione semantica

La WSD è un esempio del problema più generale di risolvere le ambiguità del linguaggio naturale.

**Ad esempio:**
* "Bank" può avere (almeno) due sensi in inglese:
    * "The Bank of England" (un istituto finanziario)
    * "The bank of river Thames" (un manufatto di ingegneria idraulica)
* Quale dei sensi di cui sopra ha l'occorrenza di "bank" in:
    * "Last week I borrowed some money from the bank"

approcci knowledge based: questa task ha avbuto nuova vita quando si è iniziato a poter accedere a risorse linguistiche in maniera elettronica e, grazie al web, wikipedia(la tassnomia di wikipedia può essere assimilata a quella di un tesauro)

## Filtraggio nel Text Mining

**Definizione:**
Il filtraggio nel text mining consiste nel classificare un flusso di documenti inviati da un produttore di informazioni a un consumatore di informazioni, bloccando la consegna di quelli irrilevanti per il consumatore.

**Esempio:**

Un feed di notizie dove l'agenzia di stampa è il produttore e il giornale è il consumatore.

**Caratteristiche:**

* Considerabile come un caso di Text Classification con una singola etichetta (rilevante o irrilevante).
* Implementabile sia lato produttore (instradamento selettivo) che lato consumatore (blocco).
* Richiede la creazione e l'aggiornamento di un "profilo" per ogni consumatore (lato produttore) o un singolo profilo generale (lato consumatore).

**Filtraggio Adattivo:**

* Il profilo iniziale è definito dall'utente.
* Il profilo viene aggiornato in base al feedback dell'utente sulla pertinenza dei documenti consegnati.

### Filtraggio - CRM (Customer Relationship Management)

**Obiettivo:**

Aiutare le aziende a comprendere le opinioni dei clienti.

**Processo:**

1. **Standardizzazione:** I dati di feedback vengono convertiti in un formato uniforme.
2. **Raggruppamento:** I dati standardizzati vengono raggruppati in base alla similarità del contenuto.
3. **Assegnazione:** I nuovi feedback vengono assegnati alle categorie predefinite.

### Filtraggio - Raccomandazione di Prodotti

**Approcci:**

* **Basato sul contenuto:** Analizza i dati dell'utente (es. categorie preferite, autori) per suggerire prodotti simili.
* **Collaborativo:** Suggerisce prodotti acquistati da utenti con profili simili, basandosi su cronologia e valutazioni.

**Tendenza attuale:** Combinare entrambi gli approcci per migliorare l'accuratezza delle raccomandazioni.

### Rilevamento dello Spam

**Applicazione:**

Classificare le email come spam o legittime utilizzando il text mining.

**Sfide:**

* La valutazione del modello è complessa a causa dei costi asimmetrici degli errori (falsi positivi vs. falsi negativi).
* La distribuzione non uniforme delle classi (molto più spam che email legittime) complica l'apprendimento del classificatore. 


---
## Vocabolario(modello di rappresentazione dei testi)

## Definizione di Termine

**Domanda:** Quali termini in un documento indicizziamo?

* **Tutte le parole o solo quelle "importanti"?**
* **Cosa ne pensiamo della presenza o della frequenza?**
    * Alcune parole sono molto comuni.
    * La maggior parte delle parole sono molto rare.

**Contesto:** I testi sono fatti da parole. Ci stiamo ponendo il problema di una selezione dei termini ai fini di indicizzazione (es. indice analitico di un libro, ovvero parole ritenute più importanti con una lista di numeri di pagine). L'indice è il mezzo per raggiungere un obiettivo.

**Risposta:** Dobbiamo mettere in relazione le parole tra di loro, creare dei pattern relazionali. L'importanza delle parole passa tra vari concetti.

**Cosa è un termine?**

* Parola, coppia di parole, frase, ...?
* Radice della parola (word stem), n-gramma (sequenza di n caratteri), ...?
* Tipo di parola?

**Possiamo modellare le relazioni tra i termini?**

* Paragrafo, frase, ordine delle parole (non utilizzati perché complicano i modelli)
* Strutture sintattiche più complesse (es: ellissi, anafore)
* Relazioni semantiche (relazioni is-a, part-of)

**Ricerca di termini che catturano la semantica del testo**

* Evitando un'intensa elaborazione manuale (codifica manuale)

**Altri argomenti:**

* **Analisi lessicale e morfologica**
    * Elaborazione di punteggiatura, numeri, conversione in minuscolo, ecc.
    * Rimozione di stopwords
    * Stemming, lemmatizzazione
    * Tagging delle parti del discorso
* **Analisi della semantica del discorso: Anafora**
    * Anafora letterale/anafora pronominale
    * Ellissi testuale
    * Meronomia referenziale
* **Pragmatica**
* **Semiotica**
* **Morfologia** 
---
## Tokenizzazione: Problemi lessicali e morfologici
* Un token è un termine candidato.
* Come organizziamo un testo?
* Rappresenta una fase di pre-processing nella pipeline di analisi dei testi
### Punteggiatura

* **Sequenza con trattino:**
    * Ad esempio, *state-of-the-art*, *co-education*, *lowercase*, *lower-case*, *lower case*.
    * Potremmo decidere di tenere distinte le due parole, sarebbe una scelta onesta perché trattiamo le parole allo stesso modo. **Problema**: in alcuni casi rompiamo la semantica. Nel nostro caso vogliamo regolarci senza considerare la semantica. Comunque questo approccio non è scalabile.
* **Apostrofi:**
    * Ad esempio, *Italy’s capital* → *Italy AND s*? *Italys*? *Italy’s*?
* **Acronimi:**
    * Ad esempio, *U.S.A*, *USA*.
* **Entità nominate:**
    * Ad esempio, *San Francisco*, *Hewlett-Packard*.

### Numeri
* **Rimozione:** Se abbiamo una stringa numerica la rimuoviamo, se abbiamo una stringa alfanumerica la teniamo.
* **Motivi per tenere i numeri:**
    * Permettere ai contenuti informativi di tenere conto dei dati.
    * Utili quando si cercano codici di errore, si controlla l'intervallo di tempo, ecc.
* **Metadati:** Spesso si indicizzeranno i "metadati" separatamente:
    * Data di creazione, formato, ecc.
    * La complessità dell'intreccio tra numeri e testo rende difficile prendere una decisione valida per tutti i casi.
* **Varietà di tipi:**
    * **Date:** *3/20/91*, *Mar. 12, 1991*, *20/3/91*.
    * **Tempo:** *55 B.C.*.
    * **Codici:** *B-52*.
    * **Identificatori:** *My PGP key is 324a3df234cb23e*.
    * **Numeri di telefono:** *(800) 234-2333*. 

* **Case delle lettere:** il case delle lettere diventa lowercase. 


## Stopwords
È una fase banale, ma risulta fondamentale perchè è quella con l'impatto maggiore.
* **Definizione:** Parole grammaticali che servono a scopi grammaticali ma con scarso potere informativo. Sono specifiche della lingua.

* **Principio cardine nella pipeline di data analysis:** Le colonne con distribuzione super omogenea o super eterogenea vengono rimosse perché non sono informative. 
	* Le dimensioni delle rappresentazioni dovrebbero avere un giusto trade-off di generalizzazione e omogeneità.

* **Motivi per rimuovere le stopwords:**
    * Non si riferiscono a oggetti o concetti.
    * Hanno poco contenuto semantico.
    * Sono le parole più comuni.
    * ==Impattiamo sulla dimensionalità della rappresentazione dei nostri testi (che implica sparsità: insieme sono un problema per le misure di distanza, che perdono di sensibilità$\to$perdita di potere discriminante.)==

* **Casi in cui le stopwords potrebbero essere necessarie:**
    * Query di frase: "King of Denmark"
    * Titoli: "Let it be", "To be or not to be"
    * Query "relazionali": "flights to London"

* **Gestione delle stopwords:**
    * **Buone tecniche di compressione:** Lo spazio per includere le stopwords in un sistema è molto piccolo.
    * **Buone tecniche di ottimizzazione delle query:** Si paga poco al momento della query per includere le stopwords. 
## Personalizzazione della Stop-List

**Obiettivo:** Non solo ridurre la stop-list, ma arricchirla.

**Come e con cosa?**

* **Rimozione di termini comuni al corpus:** 
    * Se un termine è frequente nel corpus di documenti, è poco esplicativo per la distinzione tra i documenti stessi.
    * La rimozione di questi termini è giustificata quando si analizzano i documenti nel contesto del corpus.
* **Mantenimento dei termini frequenti per l'analisi individuale:**
    * Se si analizza ogni documento in modo indipendente, non è possibile rimuovere i termini più frequenti across collection.
- **Termini Super Rari**
	* **Esclusione:**
	    * È ragionevole considerare l'esclusione dei termini super rari, ma è un'operazione ancora più delicata della rimozione dei termini comuni.
	    * Ciò che distingue un testo dall'altro è più importante di ciò che li accomuna.
	* **Termini che appaiono una sola volta nel corpus:**
	    * La decisione di escluderli dipende dal tipo di termine.
	    * Non esiste una regola fissa.

**Conclusione:**

* La stop-list deve contenere termini che si aspettano di essere molto frequenti all'interno della collezione. 
---
## Normalizzazione

**Obiettivo:** Uniformizzare le parole nel testo indicizzato e nelle parole di query nella stessa forma.

**Esempi:**

* Corrispondenza tra "U.S.A." e "USA".

**Risultato:**

* **Termine:** Un tipo di parola (normalizzato).
* **Voce nel dizionario IR:** Ogni termine rappresenta una voce nel dizionario di Information Retrieval.

**Classi di equivalenza:**

* Definiscono implicitamente gruppi di termini equivalenti.
* Esempi:
    * Eliminazione di punti: "U.S.A.", "USA" -> "USA"
    * Eliminazione di trattini: "anti-discriminatory", "antidiscriminatory" -> "antidiscriminatory"
    * Eliminazione di accenti: "French résumé" vs. "resume"
    * Eliminazione di umlaut: "German Tuebingen" vs. "Tübingen"

==come possiamo ricondurre variante lessicali a una sola parola? 
questa task rimuove informazione ma stiamo preparando i dati per otteenre una rappresentazione dei testi nettamente concentrata sulla sintassi e non sulla semantica==
**Espansione asimmetrica:**

* Un'alternativa alla classificazione per classi di equivalenza.
* Esempi:
    * Inserire: "window"
        * Ricerca: "window", "windows"
    * Inserire: "windows"
        * Ricerca: "Windows", "windows", "window"
    * Inserire: "Windows"
        * Ricerca: "Windows"
* Potenzialmente più potente, ma meno efficiente.

**Criterio più importante: uso della lingua**

* Come gli utenti scrivono le loro query?
* Anche nelle lingue che standardmente hanno accenti, trattini, ecc., gli utenti spesso potrebbero non digitarli.
* È meglio normalizzare a un termine senza accenti/trattini.

**Tokenizzazione e normalizzazione:**

* Possono dipendere dalla lingua e quindi sono intrecciate con il rilevamento della lingua.
* Ad esempio, formati di data: "3/10/2000" (US) vs. "10/3/2000" (UE).

**Conclusione:**

* È necessario "normalizzare" il testo indicizzato e i termini di query in modo identico. 



## Case Folding


tipicamente si Riduce tutte le lettere a minuscolo.
se trattiamo questo task in maniera sofisticata dobbiamo avere un modo per riconocscere le named entity
eventualmente manteniamo solo gli acronimi
**Eccezione:** maiuscole a metà frase?

* **Esempio:** General Motors
* **Esempio:** Fed vs. fed
* **Esempio:** SAIL vs. sail

Spesso è meglio mettere tutto in minuscolo, poiché gli utenti useranno le minuscole indipendentemente dalla "corretta" capitalizzazione.


---
==nota: come fase di post processing(limitato alla preparazione dei dati) possiamo applicare tecniche di algebra lineare alla matrice dei dati per ridurla ulteriormente andando a evidenziare le colonne linearmente dipendenti per trovare sinonimi, che andrebbero ridotti a una sola dimensione
dunque si usano pattern sintattici==

---
## Spell Checking

* **Prossimità tra stringhe:** Un algoritmo di spell checking dovrebbe essere in grado di gestire la prossimità tra stringhe, ovvero la possibilità di correggere un errore di battitura in più modi equivalenti.
* **Edit distance:** L'edit distance misura la differenza tra due stringhe, ma non fornisce informazioni sul contesto.
* **Contesto:** Per una correzione accurata, sarebbe necessario conoscere il contesto semantico della parola, ma noi lavoriamo solo sul campo lessicale.
* **Evitare spell checking:** Se non strettamente necessario, è meglio evitare il spell checking.
* **Corpus rumorosi:** In alcuni casi, come ad esempio i messaggi istantanei, il corpus è molto rumoroso e il spell checking potrebbe essere inutile.

## Emoticon
* **Costruzione:** Utilizzano simboli di punteggiatura (in codifica ASCII).
* **Preservazione:** Per evitare perdite, possono essere sostituite con termini, eventualmente con un markup.
* **Pesatura:** La pesatura dei termini può essere demandata al modulo successivo.
* **Importanza:** Sono importanti nell'analisi del sentiment e in task simili. 


## Tesauri
Generalizza termini che hanno significato correlato, ma forme superficiali non correlate, in termini di indice più uniformi.

* Gestiamo sinonimi e omonimi?
* **Esempio:** tramite classi di equivalenza costruite a mano
    * car = automobile
    * color = colour
* Possiamo riscrivere per formare termini di classe di equivalenza.
    * Quando il documento contiene "automobile", indicizzarlo sotto "car-automobile" (e viceversa).
* Oppure possiamo espandere una query.
    * Quando la query contiene "automobile", cercare anche sotto "car".

## Errori di ortografia

* Un approccio è Soundex,
    * che forma classi di equivalenza di parole basate su euristiche fonetiche.

## Lemmatizzazione

"Normalizza per ricondurre al lemma"

Riduce le forme flessionali/varianti alla forma base. Si applica in particolare ai verbi (li riduce all'infinito) e ai sostantivi (al singolare).

* **Esempio:** "the boy's cars are different colors" → "the boy car be different color": preserviamo i concetti. In questo caso è un effetto secondario dell'elaborazione sintattica, non l'obiettivo principale.

La lemmattizzazione implica l'esecuzione di una riduzione "corretta" alla forma del lemma del dizionario.

Supporto per l'analisi morfologica completa.

* Al massimo benefici modesti per il retrieval. 


## Stemming

Chiamato anche "suffissif stripping": intuitivo per i sostantivi, ma il problema è sui verbi.

Riduce i termini alle loro "radici" prima dell'indicizzazione. Radice intesa come prefisso.

Questa fase distrugge il lessico: va fatta solo se non vogliamo elaborare semanticamente i testi. La combiniamo alla rimozione delle stop word: non ha senso farla con la lemmatizzazione, lo stemming è più aggressivo. "Stemming" suggerisce un grossolano taglio degli affissi.

* Dipendente dalla lingua.
* **Esempio:** automate(s), automatic, automation **tutti ridotti a** automat.

**Per esempio:**

* compressed and compression are both accepted as equivalent to compress.
* Implica
* compress and compress are both accepted as equivalent to compress.

### Stemming di Porter

Itera su un insieme di regole pre-costruite e data una parola si agisce ripetutamente su essa. La decisione finale di stemming viene presa sulla base di quello che è il longest match.

* Algoritmo più comune per lo stemming dell'inglese.
    * Uno stemmer a corrispondenza più lunga in più passaggi.
    * Delle regole in un comando composto, seleziona quella che si applica al suffisso più lungo.
* Notazione: v: vocale(i), c: consonante(i), (vc)m: vocale(i) seguite da consonante(i), ripetute m volte.
* Ogni parola può essere scritta: $[c](vc)m[v]$
    * m è chiamato la "misura" della parola.
* Convenzioni + 5 fasi di riduzioni.
    * Fasi applicate in sequenza.
    * **Esempio:** GENERALIZATIONS -> GENERALIZATION (Fase 1) -> GENERALIZE (Fase 2) -> GENERAL (Fase 3) -> GENER (Fase 4)
    * Ogni fase consiste in un insieme di comandi. 


| Conditions         | Suffix | Replacement | Examples                           |
| ------------------ | ------ | ----------- | ---------------------------------- |
|                    | sses   | ss          | caresses -> caress                 |
|                    | ies    | i           | ponies -> poni, ties -> ti         |
|                    | ss     | ss          | caress -> caress                   |
|                    | s      |             | cats -> cat                        |
| (m > 0)            | eed    | ee          | feed -> feed, agreed -> agree      |
| (*v*)              | ed     |             | plastered -> plaster, bled -> bled |
| (*v*)              | ing    |             | motoring -> motor, sing -> sing    |
| (m > 1)            | e      |             | probate -> probat, rate -> rate    |
| (m = 1 and not *o) | e      |             | cease -> ceas                      |
Dove:
- `*v*` - lo stem contiene una vocale.
- `*o` - lo stem termina in `cvc`, dove la seconda `c` non è W, X o Y (es. -WIL, -HOP).

### vantaggi

* Può influenzare le prestazioni del retrival, generalmente in meglio, ma i risultati sono contrastanti.
    * L'efficacia dipende dal vocabolario.
    * Le sottili distinzioni possono essere perse attraverso lo stemming.
* Inglese: risultati molto contrastanti. Aiuta il recall per alcune query ma danneggia la precisione per altre.
    * Ad esempio, "operative" (odontoiatria) ⇒ "oper".
* Sicuramente utile per spagnolo, tedesco, finlandese, ...
    * Guadagni di prestazioni del 30% per il finlandese.
    * Lo stemming automatico è efficace quanto la conflazione manuale.


Le prestazioni di vari algoritmi sono simili.

* Stemmer di Lovins
    * Passaggio singolo, rimozione del suffisso più lungo (circa 250 regole).
* Stemmer di Paice/Husk
* Snowball

hanno lo stesso impatto va diverse peculiarità

---
## Modello Booleano

Il modello booleano confronta l'istruzione di query booleana con gli insiemi di termini utilizzati per identificare il contenuto testuale (termini di indice). 

**Query booleana:**
* **Espressione:** Combina termini di indice con operatori booleani.
* **Struttura:** L'espressione contiene:
    * **Termini di indice:** Termini selezionati dal documento indicizzato.
    * **Operatori booleani:** AND, OR e NOT, definiti sui termini. Il modello è rigido e il concetto di corrispondenza è binario: il documento corrisponde alla condizione o meno.

**Applicazioni:**
* Molti sistemi di ricerca ancora oggi utilizzano il modello booleano:
    * Email, cataloghi di biblioteche, macOS Spotlight.

### Esempio: WestLaw
* Il più grande servizio di ricerca legale commerciale (a pagamento) (iniziato nel 1975; classificazione aggiunta nel 1992; nuova ricerca federata aggiunta nel 2010).
* Decine di terabyte di dati; ~700.000 utenti.
* La maggior parte degli utenti utilizza ancora query booleane.

**Esempi di query:**

* **Query 1:** Qual è il termine di prescrizione nei casi che coinvolgono il Federal Tort Claims Act?
    * **Espressione:** LIMIT! /3 STATUTE ACTION /S FEDERAL /2 TORT /3 CLAIM
    * **Spiegazione:**
        * `/3` = entro 3 parole, `/S` = nella stessa frase.
        * Questi dati aggiungono informazioni di contesto.

* **Query 2:** Requisiti per le persone disabili per poter accedere a un luogo di lavoro.
    * **Espressione:** disabl! /p access! /s work-site work-place (employment /3 place
    * **Nota:** Lo SPAZIO è disgiunzione, non congiunzione!

### Vantaggi del modello booleano

* **Query lunghe e precise:** Consentono di specificare con precisione i criteri di ricerca.
* **Operatori di prossimità:** Permettono di definire la relazione spaziale tra i termini.
* **Sviluppo incrementale:** Le query possono essere sviluppate in modo graduale, aggiungendo termini e operatori.
* **Differenze dalla ricerca sul web:** Le query booleane sono più precise e controllate rispetto alle ricerche sul web. 



## Esempio di query
Quali opere di Shakespeare contengono le parole "Brutus" e "Caesar" ma NON "Calpurnia"?
- Si potrebbe cercare "Brutus" e "Caesar" in tutte le opere di Shakespeare, quindi eliminare le righe che contengono "Calpurnia"?
Perché questa non è la risposta?
* Lento (per corpora di grandi dimensioni).
* l'interpretazione "NOT Calpurnia" non è banale.
* Altre operazioni (ad esempio, trovare la parola "Romans" VICINO a "countrymen") non sono fattibili.
* Nessun recupero classificato (migliori documenti da restituire).



![[Pasted image 20241001111231.png|663]]
==come funziona?==

### Limitazioni

Molto rigido: AND significa tutti; OR significa qualsiasi.

* Incoraggia query brevi.
* Richiede una scelta precisa dei termini di indice -> potremmo avere risultati controintuitivi perché il modello non prevede né espansione della query (in line o a runtime) con termini equivalenti dal punto di vista semantico.
    * Ogni termine indice è considerato in maniera isolata.

Difficile controllare il numero di documenti recuperati.
* Tutti i documenti corrispondenti saranno restituiti.

Difficile classificare l'output.
* Tutti i documenti corrispondenti soddisfano logicamente la query.

Difficile eseguire il feedback di rilevanza.
* Se un documento viene identificato dall'utente come rilevante o irrilevante, come dovrebbe essere modificata la query?
* Non possiamo fare il retrieval in termini di ranking, ma solo di match, dunque non possiamo migliorare il processo di retrieval attraverso il principio di trattamento del feedback di rilevanza che esprime il grado di soddisfazione dell'utente per aver avuto il result-set desiderato.

Difficile esprimere richieste complesse dell'utente.
* Nessuna incertezza nell'assegnare termini di indice ai documenti.
* Nessuna disposizione per regolare l'importanza dei termini di query.
**Sparsità**: legato alla rappresentazione dei dati, problema comune con altri modelli.

**Conclusioni:** da utilizzare solo quando le esigenze degli utenti sono esprimibili tramite query corte e semplici e necessitiamo di sapere solo se un item c'è o non c'è. 

---
## Estensione del modello booleano

### Incorporare metodi di classificazione

* **Obiettivo:** Preservare il modello booleano, ma aggiungere un meccanismo di ordinamento dei risultati.
* **Metodo:**
    * **Pesi dei termini:** Assegnare pesi ai termini nei documenti e/o nelle query.
        * Il peso del termine misura il grado in cui quel termine caratterizza un documento.
        * I pesi dei termini sono in [0, 1].
        * Nel modello booleano standard, tutti i pesi sono 0 o 1.
    * **Calcolo dei risultati:**
        * Calcolare l'insieme dei risultati mediante il modello booleano standard.
        * Calcolare la distanza/somiglianza con la query per ogni documento (ad esempio Jaccard).
        * Classificare i risultati in base alle distanze/somiglianze vettoriali.
    * **Feedback di rilevanza:** Selezionare i documenti in base alla loro classificazione.
* **Risultato:** Si ottiene un ordine all'interno della lista dei risultati.

### Insiemi fuzzy

* **Concetto:** Rilassare i confini (boundaries) degli insiemi utilizzati nel recupero booleano.
* **Modello booleano come insieme fuzzy:**
    * **Standard:** d è in A o non è in A.
    * **Fuzzy:** d è più o meno in A.
* **Grado di appartenenza:** wA è il grado di appartenenza di un elemento all'insieme A.
* **Operazioni:**
    * **Intersezione (and):** $wA∩B = min(wA, wB)$
    * **Unione (or):** $wA∪B = max(wA, wB)$

### MMM: Mixed Min and Max Model

* **Termini:** t1, t2, ..., tn
* **Documento:** d, con pesi dei termini di indice: w1, w2, ..., wn
* **Query disgiuntiva:** $q_{or}$ = (t1 or t2 or ... or tn)
* **Somiglianza query-documento:**
    * $S(q_{or}, d) = λ_{or} * max(w1,.. ,wn) + (1 - λ_{or}) * min(w1,.. , wn)$
    * $λ_{or}$ è un parametro che controlla il peso relativo del massimo e del minimo.
    * Con la logica booleana regolare, $λ_{or} = 1$.
    * Se la query è puramente disgiuntiva, si considera solo il primo termine.

* **Query congiuntiva:**
    * $q_{and}$ = (t1 and t2 and ... and tn)
    * $S(q_{and}, d) = λ_{and} * min(w1,.. , wn) + (1 - λ_{and}) * max(w1,.. , wn)$
    * $λ_{and}$ è un parametro che controlla il peso relativo del minimo e del massimo.
    * Con la logica booleana regolare, $λ_{and} = 1$.

### Modello di Paice

* **Differenza da MMM:** Tiene conto di tutti i pesi dei documenti, mentre MMM considera solo i pesi massimi e minimi.
* **Vantaggio:** Migliora la risposta alla query.
* **Svantaggio:** Costo computazionale più alto rispetto a MMM.

### Modello P-norma

* **Caratteristiche:**
    * Documenti con pesi dei termini.
    * Termini di query con pesi associati ai termini.
    * Operatori con coefficienti che indicano il grado di rigore.
* **Calcolo della somiglianza:**
    * Considera ogni documento e query come punti multidimensionali (considerando due schemi di pesatura).

---
## Sparsità
#### Rappresentazione di matrici termine-documento
- Adottando un modello di rappresentazione di questo un modello poco sparso è irrealistico.
	
**Esempio:** Considera un insieme di **N = 1 milione di documenti**, ognuno con circa **1000 parole.**
- Un esempio simile è la market-basket-analisys
Assumiamo una media di **6 byte/parola**, inclusi spazi e punteggiatura.
* Questo corrisponde a **6 GB di dati** nei documenti.
Supponiamo che ci siano **M = 500K termini distinti** tra questi documenti.
**Problemi:**
* Una matrice di incidenza termine-documento di dimensioni **500K x 1M** avrebbe mezzo Trillion (milione di miliardi) di 0 e 1.
**Nella realtà:**
* La matrice ha **non più di un miliardo di 1**.
* Le matrici di incidenza termine-documento sono **estremamente sparse**.
**Rappresentazione migliore:**
* **Registrare solo le posizioni 1.**
Desiderata per una struttura dati:
* Capacità di rappresentare concetti e relazioni.
* Capacità di supportare la localizzazione di questi concetti nella collezione di documenti.

## Indice Inverso

* **Definizione:** L'indice inverso mappa ogni termine all'insieme dei documenti in cui compare.
* **Vantaggi:**
    * **Efficienza di memorizzazione:** È più efficiente per memorizzare e gestire matrici sparse.
    * **Ricerca efficiente:** Consente di identificare rapidamente i documenti che contengono un determinato termine.

* **Costruzione:**
    * **Identificazione dei documenti indicizzati:** Per ogni termine:
        * Identifica tutti i documenti che sono indicizzati da quel termine.
        * Memorizza gli ID di quei documenti (docID).
    * **Creazione della matrice termine-documento:**
        * Inizialmente, rappresenta l'indice inverso come un array di documenti indicizzati.
        * Trasponi l'array per ottenere una matrice termine-documento.

* **Struttura:**
    * **Liste di postings:** Per ogni termine *t*, memorizza un elenco di tutti i documenti che contengono *t*.
    * **DocID:** Ogni documento è identificato da un docID univoco.

* **Implementazione:**
    * **Array di dimensioni fisse:** Non sono adatti per liste di postings di dimensioni variabili, poiché il numero di elementi contenuti in questi array è variabile.
    * **Liste concatenate:** Offrono flessibilità per gestire liste di postings di dimensioni variabili.
    * **Array di lunghezza variabile:** Offrono un compromesso tra dimensioni e facilità di inserimento.

* **Memorizzazione:**
    * **Su disco:** Una sequenza continua di postings è la soluzione più comune ed efficiente.
    * **In memoria:** Le liste concatenate e gli array di lunghezza variabile sono opzioni valide.

* **Considerazioni:**
    * La scelta della struttura dati per l'indice inverso dipende dalle esigenze specifiche dell'applicazione.
    * È importante considerare i compromessi tra efficienza di memorizzazione, velocità di accesso e facilità di aggiornamento.


==possiamo vedere il core dell' index composta da due elem: dizionario e liste di posting
doppiamente indicizzati, per chiave e per...==

![[1) Intro-20241003104139467.png]]

==l'indexer per ogni token lavora collezionando i postings per quel token, ovvero l'insieme di documenti in cui quel token appare. 
l'input è uno stream temine-id documento==
## Passaggi dell'indexer

1. **Sequenza di coppie (Termine, docID):**  L'indexer analizza i documenti e genera una sequenza di coppie, dove ogni coppia rappresenta un termine e l'ID del documento in cui il termine appare.

2. **Ordina per Termine:** La sequenza di coppie viene ordinata in base al termine.

3. **Ordina per docID:**  Le coppie con lo stesso termine vengono raggruppate e ordinate in base al docID.

4. **Unisci voci multiple:** Se un termine appare più volte nello stesso documento, le voci vengono unite in un'unica voce, mantenendo la frequenza del termine nel documento.

5. **Aggiungi frequenza del documento:**  Per ogni voce, viene aggiunta l'informazione sulla frequenza del termine nel documento.

6. **Dividi in:**

    * **Dizionario:**  Contiene tutti i termini distinti trovati nei documenti, con il loro corrispondente termID.
    * **Postings:**  Contiene le liste di postings per ogni termine, ovvero l'insieme di documenti in cui il termine appare, con la loro frequenza. 


## Elaborazione delle query

**Esempio:** Brutus AND Caesar

**Passaggi:**

1. **Individua Brutus nel Dizionario:**
    * Recupera i suoi postings (lista di documenti in cui compare Brutus).
2. **Individua Caesar nel Dizionario:**
    * Recupera i suoi postings (lista di documenti in cui compare Caesar).
3. **Unisci i due postings (interseca gli insiemi di documenti):**
    * Attraversa i due postings simultaneamente, in tempo lineare nel numero totale di voci di postings. 
	    * Garantito dal fatto che manteniamo le due liste ordinate
    * Questo significa che l'algoritmo impiega un tempo proporzionale al numero di documenti in cui compare Brutus più il numero di documenti in cui compare Caesar.
		![[1) Intro-20241003104909533.png|438]]
**Nota:** Questo processo di unione dei postings è un esempio di come vengono elaborate le query booleane (AND, OR, NOT) nei motori di ricerca. 

```
INTERSECT(p1, p2)
1   answer ← ⟨ ⟩
2   while p1 ≠ NIL and p2 ≠ NIL
3       do if docID(p1) = docID(p2)
4             then ADD(answer, docID(p1))
5                  p1 ← next(p1)
6                  p2 ← next(p2)
7          else if docID(p1) < docID(p2)
8             then p1 ← next(p1)
9             else p2 ← next(p2)
10  return answer
```

## Ottimizzazione dell'elaborazione delle query

**Scenario:** Una query composta da una congiunzione AND di *n* termini.
![[1) Intro-20241003105153597.png]]
**Obiettivo:** Determinare l'ordine ottimale per l'elaborazione della query.

**Strategia:**

* **Elabora in ordine di frequenza crescente:**
    * Inizia con il termine che compare nel minor numero di documenti (insieme di postings più piccolo).
	    * Questo processo riduce il numero di operazioni di intersecazione e quindi il tempo di elaborazione complessivo
    * Ad ogni passo, interseca l'insieme corrente con il postings del termine successivo, riducendo progressivamente la dimensione dell'insieme.

**Esempio:**
* Se la query è "Brutus AND Caesar AND Antony", e la frequenza documentale è:
    * Brutus: 1000 documenti
    * Caesar: 500 documenti
    * Antony: 200 documenti
* L'ordine ottimale sarebbe: Antony, Caesar, Brutus.

**Query booleane arbitrarie:**
* **Esempio:** (Brutus OR Caesar) AND NOT (Antony OR Cleopatra)
* **Strategia:**
    1. **Ottieni la frequenza documentale per tutti i termini.**
    2. **Stima la dimensione di ogni OR (Upper Bound) come la somma delle sue frequenze documentali (stima conservativa).**
    3. **Elabora in ordine crescente di dimensioni OR.**

**Nota:**
* Il tempo di elaborazione è "lineare" rispetto al numero totale di voci nei postings.
* L'ordine di elaborazione può influenzare significativamente l'efficienza, soprattutto per query complesse.

## Query di frase

**Importanza:**

* Le query di frase sono un elemento chiave per la "ricerca avanzata".
* Sono facilmente comprensibili dagli utenti.
* L'obiettivo è rispondere a query come "Dipartimenti Unical" come una frase completa.
* La frase "Ci sono 14 dipartimenti presso Unical" non è una corrispondenza, poiché non corrisponde alla query di frase.
* Per questo tipo di query, non è sufficiente memorizzare solo coppie (Termine, docID).

**Approcci:**

* **Indici biword:**
    * Indizza ogni coppia consecutiva di termini nel testo come una frase ( come se avvessimo un *n-gram* che preserva le parole con *n=2*).
    * **Esempio**: "*Amici, Romani, Concittadini*" genera i biword "*amici romani*" e "*romani concittadini*".
    * Ogni biword diventa un termine del dizionario.
    * L'elaborazione delle query di frase a due parole è immediata.

* **Scomposizione di frasi più lunghe:**
    * "Corsi del dipartimento DIMES" può essere scomposto nella query booleana sui biword: *"dipartimento DIMES  AND corsi dipartimento".*
    * Senza i documenti, non è possibile verificare se i documenti che corrispondono alla query booleana contengano effettivamente la frase.
	    * Rischio di falsi positivi.
    * Espansione dell'indice a causa di un dizionario più grande.
	    * Inattuabile per più di due parole, anche per i biword (aumenta di molto la complessità).
    * **Gli indici biword non sono la soluzione standard, ma possono essere parte di una strategia composita.**

### Etichettatura Morfologica (POST) e Biword Estensi

* **Analisi del testo:** suddividere il testo in termini e assegnare a ciascun termine la sua categoria grammaticale (POST: target part of speach).
* **Nomi (N) e Articoli/Preposizioni (X):** classificare i termini come nomi o articoli/preposizioni.
* **Biword Estensi:** identificare le sequenze di termini della forma $NX*N$ (es. "il cacciatore nella segale").
	* Dunque includiamo uno o più articoli.
* **Dizionario:** ogni biword esteso diventa un termine nel dizionario.

**Esempio:**

"il cacciatore nella segale"

* **POST:** $N X X N$
* **Biword Estenso:** "cacciatore segale"

#### Elaborazione delle Query
* **Analisi della query:** suddividere la query in termini N e X.
* **Segmentazione in biword:** segmentare la query in biword estesi.
* **Ricerca nell'indice:** cercare i biword estesi nell'indice.

#### Indici Posizionali
* **Memorizzazione delle posizioni:** per ogni termine, memorizzare le posizioni in cui compaiono i token del termine nei documenti.

**Formato:**
```
<termine, numero di documenti contenenti il termine;
doc1: posizione1, posizione2 … ;
doc2: posizione1, posizione2 … ;
…>
```

#### Algoritmo di Unione Ricorsivo

* **Query di frase:** estraggono le voci dell'indice invertito per ogni termine distinto (es. "to", "be", "or", "not").
* **Fusione delle liste di doc:posizione:** uniscono le liste di doc:posizione per elencare tutte le posizioni con la frase completa (es. "to be or not to be").
* **Algoritmo di unione ricorsivo:** utilizzato per le query di frase, ma con la necessità di gestire più della semplice uguaglianza.

#### Indici Posizionali 

* **Utilizzo degli indici posizionali:** gli indici posizionali possono essere utilizzati per le query di prossimità.
* **Limiti degli indici biword:** gli indici biword non possono essere utilizzati per le query di prossimità.
* **Algoritmo di unione ricorsivo:** utilizzato per le query di frase, ma con la necessità di gestire più della semplice uguaglianza.

---
## Dimensione dell'indice posizionale

* Un indice posizionale espande in modo sostanziale l'archiviazione dei postings.
    * Bisogna avere una voce per ogni occorrenza, non solo una per documento.
    * La dimensione dell'indice dipende dalla dimensione media del documento. (Anche se gli indici possono essere compressi.)
* **Regola empirica:**
    * Un indice posizionale è 2-4 volte più grande di un indice non posizionale.
    * La dimensione dell'indice posizionale è il 35-50% del volume del testo originale.
* Tuttavia, un indice posizionale è ora utilizzato standardmente a causa della potenza e dell'utilità delle query di frase e prossimità, sia utilizzate esplicitamente che implicitamente in un sistema di recupero di ranking. 

## Costruzione dell'indice basata sull'ordinamento
* Mentre costruiamo l'indice, analizziamo i documenti uno alla volta.
    * I postings finali per qualsiasi termine sono incompleti fino alla fine.
* Richiede molto spazio per collezioni di grandi dimensioni:
    * ad esempio, a 8 byte per (termID, docID) e 100 milioni(valore realistico per un anno di pubblicazioni) #postings non posizionali (cioè, RCV1(corpus), un anno di notizie di Reuters) ->
        * in linea di principio, è possibile indicizzare questo in memoria oggi, ma
        * le collezioni tipiche sono molto più grandi
            * ad esempio, il New York Times fornisce un indice di >150 anni di notizie
* **Dobbiamo memorizzare i risultati intermedi su disco**.

### Scalabilità della costruzione dell'indice
* La costruzione dell'indice in memoria non è scalabile.
    * Non è possibile inserire l'intera collezione in memoria, ordinarla e poi riscriverla.
* Come possiamo costruire un indice per collezioni molto grandi?
    * Tenendo conto dei **vincoli hardware**: memoria, disco, velocità, ecc.
### Basi hardware
* I server utilizzati nei sistemi IR ora hanno in genere diversi GB di memoria principale.
    * Lo spazio su disco disponibile è di diversi (2-3) ordini di grandezza maggiore.
* La **tolleranza ai guasti** è molto costosa:
    * È molto più economico utilizzare molte macchine normali piuttosto che una macchina tollerante ai guasti.
* L'accesso ai dati in memoria è molto più veloce dell'accesso ai dati su disco.
* Ricerche su disco: Nessun dato viene trasferito dal disco mentre la testina del disco viene posizionata.
    * Pertanto, il trasferimento di un grande blocco di dati dal disco alla memoria è più veloce del trasferimento di molti piccoli blocchi.
* L'I/O del disco è basato su blocchi:
    * Lettura e scrittura di blocchi interi (al contrario di piccoli blocchi)
    * Dimensioni dei blocchi: da 8 KB a 256 KB(dipende dal task, per il retrival si arriva a 64, 256 è il massimo).

#### Ordinamento usando il disco come "memoria"?
* Possiamo usare lo stesso algoritmo di costruzione dell'indice per collezioni più grandi, ma usando il disco invece della memoria?
    * No: Ordinare 100 milioni di record su disco è troppo lento, poichè implica troppe ricerche su disco.
* Abbiamo bisogno di un algoritmo di ordinamento esterno.

## BSBI: Indicizzazione basata sull'ordinamento a blocchi

* **Ordinamento con meno ricerche su disco** (minimizzare le ricerche sul disco)
	* Partiamo dall'assunzione che il corpus sia statico.
    * I record (termID, docID) vengono generati mentre analizziamo i documenti.
    * Supponiamo di ordinare 100 milioni di tali record da 8 byte per termID.
    * Definiamo un blocco di circa 10 milioni di tali record.
        * Possiamo facilmente inserirne un paio in memoria.
        * Avremo 10 blocchi di questo tipo per iniziare.
* **Idea di base dell'algoritmo**:
    * Accumulare i postings per ogni blocco, ordinarli e scriverli su disco.
    * Quindi unire i blocchi in un unico ordine ordinato.

```
BSBIndexConstruction()
1   n ← 0
2   while (all documents have not been processed)
3       do n ← n + 1
4          block ← ParseNextBlock()
5          BSBI-Invert(block)
6          WriteBlockToDisk(block, fn)
7   MergeBlocks(f1, ..., fn; f_merged)

```

#### Ordinamento di 10 blocchi di 10 milioni di record.
* Innanzitutto, leggiamo ogni blocco e ordiniamo al suo interno:
    * Quicksort richiede O(N ln N) passaggi in media.
    * Nel nostro caso N=10 milioni.
* 10 volte questa stima
    * ci fornisce 10 run ordinate di 10 milioni di record ciascuna.
* Fatto in modo semplice, abbiamo bisogno di 2 copie dei dati su disco.
    * Ma possiamo ottimizzare questo. 


## Come unire le run ordinate?

* Possiamo fare delle **merge binarie**, con un albero di merge di $log_2(10) = 4$ livelli.
    * Durante ogni livello, leggiamo in memoria le run a blocchi di 10 milioni, le uniamo e le riscriviamo.
![[1) Intro-20241003112202805.png]]
* Ma è più efficiente fare un **multi-way merge**, dove si legge da tutti i blocchi simultaneamente.
    * Apriamo tutti i file di blocco simultaneamente, manteniamo un **buffer di lettura** per ciascuno e un **buffer di scrittura** per il file di output.
    * In ogni iterazione, scegliamo il termID più basso che non è stato elaborato usando una coda di priorità.
    * Uniamo tutte le liste di postings per quel termID e lo scriviamo.
* A condizione che si leggano blocchi di dimensioni decenti da ogni blocco in memoria e poi si scriva un blocco di output di dimensioni decenti.
    
## SPIMI: Indicizzazione in memoria a passaggio singolo

* **Problema rimanente con l'algoritmo basato sull'ordinamento:**
    * La mappatura (termine, termID) potrebbe non entrare in memoria.
    * Abbiamo bisogno del dizionario (che cresce dinamicamente) per implementare una mappatura termine-termID.

* **Idea chiave 1:** Generare dizionari separati per ogni blocco - non c'è bisogno di mantenere la mappatura termine-termID tra i blocchi.
* **Idea chiave 2:** Non ordinare. Accumulare i postings nelle liste di postings man mano che si verificano.

* Sulla base di queste due idee, generiamo un indice invertito completo per ogni blocco.
* Questi indici separati possono quindi essere uniti in un unico grande indice.
* SPIMI può indicizzare collezioni di qualsiasi dimensione a condizione che ci sia abbastanza spazio su disco disponibile.
* Ogni lista di postings è dinamica e immediatamente disponibile per raccogliere i postings.
    * **più veloce** - non è necessario ordinare.
    * **risparmia memoria** - i termID dei postings non devono essere memorizzati. 


![[1) Intro-20241003093740166.png]]

---
---

## Indicizzazione distribuita

### Parser

● Il master assegna uno split a una macchina parser inattiva.
● Il parser legge un documento alla volta ed emette coppie (termine, doc).
● Il parser scrive le coppie in j partizioni.
○ Es.: Ogni partizione è per un intervallo di lettere iniziali dei termini (ad esempio, a-f, g-p, q-z), quindi j = 3.
● Ora per completare l'inversione dell'indice.

### Inverter

● Un inverter raccoglie tutte le coppie (termine, doc) (cioè, postings) per una partizione di termini.
● Ordina e scrive nelle liste di postings. 

![[1) Intro-20241003093756427.png]]

![[1) Intro-20241003093804693.png]]
## Indicizzazione distribuita

L'algoritmo di costruzione dell'indice è un'istanza di MapReduce (Dean e Ghemawat 2004).
● Un framework robusto e concettualmente semplice per il calcolo distribuito, senza dover
scrivere codice per la parte di distribuzione.
● Il sistema di indicizzazione di Google (circa 2002) come composto da una serie di fasi, ciascuna
implementata in MapReduce.

### Schema per la costruzione dell'indice in MapReduce

● Funzioni map e reduce
map: input → list(k, v)
reduce: (k,list(v)) → output
● Istanza dello schema per la costruzione dell'indice
map: collection → list(termID, docID)
reduce: (<termID1, list(docID)>, <termID2, list(docID)>, …) → (postings list1, postings list2, …)

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Indicizzazione dinamica

Fino ad ora, abbiamo ipotizzato che le collezioni siano statiche, ma
● I documenti arrivano nel tempo e devono essere inseriti.
● I documenti vengono eliminati e modificati.
Questo significa che il dizionario e le liste di postings devono essere modificati:
● Aggiornamenti dei postings per i termini già presenti nel dizionario.
● Nuovi termini aggiunti al dizionario.

### Approccio più semplice

● Mantenere un "grande" indice principale, i nuovi documenti vanno in un "piccolo" indice ausiliario.
● Ricerca su entrambi, unione dei risultati.
● Eliminazioni
○ Vettore di bit di invalidazione per i documenti eliminati.
○ Filtra i documenti in output su un risultato di ricerca tramite questo vettore di bit di invalidazione.
● Periodicamente, re-indicizzare in un unico indice principale.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Indicizzazione dinamica

### Problemi con gli indici principali e ausiliari

● Problema delle fusioni frequenti - si tocca molto.
● Scarsa performance durante la fusione.
● In realtà:
○ La fusione dell'indice ausiliario nell'indice principale è efficiente se si mantiene un
file separato per ogni lista di postings.
○ La fusione è la stessa di una semplice append.
○ Ma poi avremmo bisogno di molti file - inefficiente per il sistema operativo.
● Ipotesi: L'indice è un unico grande file.
○ In realtà: Usare uno schema da qualche parte nel mezzo (ad esempio, dividere le liste di postings molto grandi, raccogliere le liste di postings di lunghezza 1 in un unico file, ecc.).

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Indicizzazione dinamica

### Fusione logaritmica

● Mantenere una serie di indici, ciascuno due volte più grande del precedente.
○ In qualsiasi momento, alcune di queste potenze di 2 sono istanziate.
● Mantenere il più piccolo (Z0) in memoria.
● Quelli più grandi (I0, I1, …) su disco.
● Se Z0 diventa troppo grande (> n), scrivilo su disco come I0.
● o uniscilo con I0 (se I0 esiste già) come Z1.
● O scrivi la fusione Z1 su disco come I1 (se non c'è I1).
● O uniscilo con I1 per formare Z2.

### Indicizzazione dinamica

Indice ausiliario e principale:
● T/n fusioni, dove T è il numero di postings e n è la dimensione dell'ausiliario.
● Il tempo di costruzione dell'indice è O(T2/n) come nel caso peggiore un posting viene toccato
T/n volte.

Fusione logaritmica:
● Ogni posting viene fuso al massimo O(log (T/n)) volte,
● cioè, la complessità è O(T log (T/n)).

La fusione logaritmica è molto più efficiente per la costruzione dell'indice.
● Ma l'elaborazione delle query ora richiede la fusione di O(log (T/n)) indici.
● Mentre è O(1) se si ha solo un indice principale e uno ausiliario.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Indicizzazione dinamica

### Ulteriori problemi con più indici

● Le statistiche a livello di collezione sono difficili da mantenere.
○ ad esempio, quando parliamo di correzione ortografica: quale delle diverse alternative corrette presentiamo all'utente?
■ Potremmo voler scegliere quella con il maggior numero di risultati.
■ Come manteniamo i migliori con più indici e vettori di bit di invalidazione?
■ Una possibilità: ignorare tutto tranne l'indice principale per tale ordinamento.
● Vedremo più di queste statistiche utilizzate nel ranking dei risultati.


## Indicizzazione dinamica

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

Tutti i grandi motori di ricerca ora effettuano l'indicizzazione dinamica.
I loro indici hanno frequenti modifiche incrementali.
● Notizie, blog, nuove pagine web tematiche.
Ma (a volte/tipicamente) ricostruiscono anche periodicamente l'indice da zero.
● L'elaborazione delle query viene quindi commutata sul nuovo indice e il vecchio indice viene eliminato.

### Requisiti per la ricerca in tempo reale

● Bassa latenza, elevata produttività di valutazione delle query.
● Elevato tasso di ingestione e immediata disponibilità dei dati.
● Letture e scritture concorrenti dell'indice.
● Dominanza del segnale temporale.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Indicizzazione dinamica

### Organizzazione dell'indice in Earlybird di Twitter

● Consiste di più segmenti di indice.
○ Ogni segmento è relativamente piccolo, contenendo fino a 223 tweet.
○ Ogni posting in un segmento è una parola a 32 bit: 24 bit per l'ID del tweet e 8 bit
per la posizione nel tweet.
● Solo un segmento può essere scritto in un dato momento.
○ Abbastanza piccolo da stare in memoria.
○ I nuovi postings vengono semplicemente accodati alla lista di postings.
○ Ma la lista di postings viene attraversata all'indietro per dare la priorità ai tweet più recenti.
● I segmenti rimanenti sono ottimizzati per la sola lettura.
○ Postings ordinati in ordine cronologico inverso (il più recente per primo).

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Costruzione dell'indice: riepilogo

### Indicizzazione basata sull'ordinamento

● Inversione in memoria ingenua.
● Indicizzazione basata sull'ordinamento bloccato (BSBI).
● L'ordinamento per fusione è efficace per l'ordinamento basato su disco rigido (evita le ricerche!).

### Indicizzazione in memoria a passaggio singolo (SPIMI)

● Nessun dizionario globale.
● Genera un dizionario separato per ogni blocco.
● Non ordinare i postings.
● Accumulare i postings nelle liste di postings man mano che si verificano.

### Indicizzazione distribuita usando MapReduce

### Indicizzazione dinamica: più indici, fusione logaritmica

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Compressione

Usare meno spazio su disco.
Mantenere più cose in memoria.
● Aumenta la velocità di trasferimento dei dati dal disco alla memoria.
● [leggi dati compressi | decomprimi] è più veloce di [leggi dati non compressi].
Premessa: Gli algoritmi di decompressione sono veloci.
● Vero per gli algoritmi di decompressione che utilizziamo.

### Compressione del dizionario

● Renderlo abbastanza piccolo da tenerlo nella memoria principale.
● Renderlo così piccolo da poter tenere anche alcune liste di postings nella memoria principale.

### Compressione del/dei file di postings

● Ridurre lo spazio su disco necessario.
● Diminuire il tempo necessario per leggere le liste di postings dal disco.
● I grandi motori di ricerca mantengono una parte significativa dei postings in memoria.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Compressione

### Compressione senza perdita vs. con perdita

● Compressione senza perdita: tutte le informazioni vengono preservate.
○ Ciò che facciamo principalmente in IR.
● Compressione con perdita: scarta alcune informazioni.
○ Diversi passaggi di pre-elaborazione possono essere visti come compressione con perdita: conversione in minuscolo, stop words,
stemming, eliminazione dei numeri.
○ Potare le voci di postings che hanno poche probabilità di apparire nella lista dei primi k per qualsiasi query.
■
Quasi nessuna perdita di qualità nella lista dei primi k.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Dimensione del vocabolario vs. dimensione della collezione

### Quanto è grande il vocabolario dei termini?

● Cioè, quante parole distinte ci sono?
Possiamo assumere un limite superiore?
● Non proprio: almeno 7020 = 1037 parole diverse di lunghezza 20.
In pratica, il vocabolario continuerà a crescere con la dimensione della collezione.
● Soprattutto con Unicode ☺.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Dimensione del vocabolario vs. dimensione della collezione

### Legge di Heaps: M = kTb

● M è la dimensione del vocabolario, T è il numero di token nella
collezione.
● Valori tipici: 30 ≤ k ≤ 100 e b ≈ 0.5.
● In un grafico log-log della dimensione del vocabolario M vs. T, la legge di Heaps prevede una linea
con pendenza di circa ½.
● È la relazione più semplice possibile (lineare) tra i due in
spazio log-log.
○ log M = log k + b log T.
● Un'osservazione empirica ("legge empirica").

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Dimensione del vocabolario vs. dimensione della collezione

### Legge di Heaps per Reuters RCV1:

log10M = 0.49 log10T + 1.64
->
M = 101.64T0.49
cioè, k = 101.64 ≈ 44 e b = 0.49.

Buona aderenza empirica per Reuters RCV1:
● Per i primi 1.000.020 token, prevede
38.323 termini;
● in realtà, 38.365 termini.

Andrea Tagarelli
Università della Calabria
Information Retrieval e
Elaborazione del Linguaggio Naturale

## Legge di Zipf

La legge di Heaps fornisce la dimensione del vocabolario nelle collezioni.
Ma, nel linguaggio naturale, ci sono
● alcuni termini molto frequenti e
● molti termini molto rari.

### Legge di Zipf

● Zipf (1949) ha scoperto che il termine i-esimo più frequente ha una frequenza di collezione proporzionale a 1/i:
cfi ∝ 1/i = K/i,
con K una costante di normalizzazione.
● cioè, log cfi = log K - log i.
● relazione lineare inversa tra log cfi e log i -> una legge di potenza.
Se il termine più frequente (the) si verifica cf1 volte, allora il 2° termine più frequente (of) si verifica cf1/2 volte, il
3° termine più frequente (and) si verifica cf1/3 volte...

