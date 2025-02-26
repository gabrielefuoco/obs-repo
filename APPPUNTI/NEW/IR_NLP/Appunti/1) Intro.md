## Linguaggio naturale

* **Unicità:** Il linguaggio naturale è un linguaggio specifico per gli umani, progettato per gestire la complessità della comunicazione.
* **Interazione multimodale:** È fondamentale per interagire in contesti che coinvolgono anche altre modalità, come la visione.

### Recupero delle informazioni (IR) e Elaborazione del linguaggio naturale (NLP)

Entrambi sono campi della scienza e dell'ingegneria.
* **NLP:** Si concentra sullo sviluppo di sistemi automatici che comprendono e generano linguaggi naturali.
* **IR:** Si concentra sul ritrovamento (semi-)automatico di dati testuali (non strutturati) che soddisfano un bisogno informativo all'interno di grandi collezioni.

### Definizione di IR

###### Enunciato del problema di base:
* **Dato:**
    * **Documenti:**  si presume che sia una collezione statica.
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
* **Segnale:** Il sistema deve tenere traccia del segnale, ovvero dei dati in ingresso, per identificare i topic.
### Modellazione di topic stocastica

- È una tecnica che utilizza modelli probabilistici per analizzare grandi quantità di testo e identificare i temi principali (topic) presenti nei documenti.  
- Si basa sull'idea che ogni documento sia una combinazione di diversi topic, e che ogni parola abbia una probabilità di appartenere a un determinato topic.
- Questo approccio permette di scoprire i temi principali, analizzare la loro distribuzione nei documenti e classificare i documenti in base ai topic che contengono. 
- Un esempio di modello di topic stocastico è il **Latent Dirichlet Allocation (LDA).** 

### Relation Extraction

* **Relazione con Named Entity Recognition:** Per estrarre le relazioni tra entità nominate, è necessario prima identificare le entità stesse.
* **Pattern frequenti:** La ricerca di pattern frequenti è un passo fondamentale per l'estrazione di relazioni.
* **Definizione in NLP:** In NLP, la relation extraction si riferisce all'identificazione di relazioni lessicali tra entità nominate.

### Summarization

* **Approccio tradizionale:** Negli anni '90, la summarization era affrontata come estrazione di parole chiave (key phrase extraction).
* **Guida alla summarization:** Le proprietà di un documento possono essere utilizzate per guidare il processo di summarization. 

## KDD Pipeline in NLP

* **Fasi:** I task di NLP si svolgono attraverso una sequenza di fasi (*unfolding*).
* **Rappresentazione:** Prima di poter rappresentare i contenuti informativi, è necessario indicizzare gli elementi costitutivi.
* **Apprendimento:** L'apprendimento avviene a valle del set di risultati (*result set*).
* **Feature Selection:** Il machine learning può essere utilizzato anche in fase di selezione delle features.

### Valutazione

* **Criteri statistici:** La statistica offre diversi criteri per la valutazione.
* **Accuracy:** L'accuracy presenta diversi problemi nella stima delle prestazioni di un classificatore. 

# Funzioni Principali

* **Funzione principale:**  Il compito principale di un sistema di elaborazione del linguaggio naturale (*NLP*) è quello di **estrarre informazioni** da testi.
* **Macro categorie:** Le funzioni principali possono essere suddivise in due macro categorie:
    * **Indicativa:** Questa funzione serve a rivelare gli elementi dei contenuti in modo da consentire la determinazione della rilevanza del testo rispetto alle query. È fondamentale per l'esplorazione dei corpus e per il retrieval.
    * **Informativa:** Questa funzione consente di ottenere un surrogato del testo (o di una sua porzione) senza fare necessariamente riferimento al testo originale. Questo concetto è ancora valido per il retrieval, che consiste nel condensare i dati originali in un formato surrogato.

## Browsing

* **Definizione:** La navigazione (browsing) è solitamente parte di sistemi di ipertesto e ipermedia.
* **Scopo:** Permette agli utenti di scorrere collezioni di testo alla ricerca di informazioni utili.
* **Vantaggi:** Gli utenti non hanno bisogno di:
    * generare descrizioni di ciò che desiderano
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

## Estrazione di Informazioni

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
* **Multipli riempitivi(filler):** Alcuni slot possono consentire più filler.
* **Ordine fisso:** Si assume che gli slot siano sempre in un ordine fisso.

**Modelli di estrazione:**
* **Specificare un elemento:** Specificare un elemento da estrarre per uno slot, ad esempio, utilizzando un modello di regex.
* **Modelli precedenti e successivi:** Potrebbe essere necessario un modello precedente (pre-filler) per identificare il contesto appropriato e un modello successivo (post-filler) per identificare la fine del riempitivo. 
	* **Es**: *Estrazione di termini da un modello*. Definiziamo il template a priori e per ogni slot creiamo dei pattern.


## Recupero di Documenti

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

## Panoramica delle Applicazioni di Base

Le principali aree di applicazione coprono due aspetti:
* **Scoperta della conoscenza**
    * **Estrazione di informazioni:**  il processo di estrazione di informazioni utili da grandi quantità di dati.
* **Distillazione delle informazioni**
    * **Estrazione basata su struttura:**  l'estrazione di informazioni basata su una struttura predefinita di documenti, per identificare i documenti rilevanti per un'informazione target.

**Utilizzi tipici:**
* **Categorizzazione gerarchica:**  Usare il retrival per organizzare i documenti in una struttura gerarchica basata su categorie (Seguendo l'esempio delle web directory).
* **Riassunto del testo:**  Creare un riassunto conciso di un documento, mantenendo le informazioni più importanti.
* **Disambiguazione del senso delle parole:**  Determinare il significato corretto di una parola in un contesto specifico.
- **Filtraggio del Testo (o dell'Informazione)**: Rimuovere il testo non rilevante da un documento. Inizialmente accoppiato alla web personalization, da cui nascono gli attuali sistemi di raccomandazione.
- **Approcci alla** **Web Personalization:**
	* Selezionare l'informazione per un gruppo di utenti target in base alle loro specifiche esigenze.
	* **Collaborative based:** il sistema continua a proporre non solo i prodotti in base alle specifiche esigenze, ma anche cose spesso acquistate da utenti con un profilo simile, per un discorso anche di diversificazione e serendipità.

**Altri esempi di applicazioni:**

* **CRM e marketing:** (Customer Relationship Managment) Cross-selling, raccomandazioni di prodotti.
* **Raccomandazione di prodotti:**  Suggerire prodotti pertinenti agli utenti in base alle loro preferenze.
* **Consegna di informazioni nelle organizzazioni per la gestione della conoscenza:**  Fornire informazioni pertinenti ai dipendenti per migliorare la collaborazione e la produttività.
* **Personalizzazione dell'accesso alle informazioni:**  Adattare l'accesso alle informazioni alle esigenze individuali degli utenti.
* **Filtraggio di notizie in newsgroup Usenet:**  Rimuovere le notizie non pertinenti o indesiderate.
* **Rilevamento di messaggi spam:**  Identificare e filtrare i messaggi di posta elettronica indesiderati. 

### Categorizzazione Gerarchica

La categorizzazione gerarchica, tipicamente implementata tramite tassonomie, permette ai ricercatori di esplorare un insieme di dati in modo strutturato.  
Inizialmente, si naviga attraverso la gerarchia delle categorie per poi focalizzarsi su una specifica categoria di interesse, affinando così la ricerca.  Questo sistema dovrebbe essere flessibile, consentendo l'aggiunta di nuove categorie e la rimozione di quelle obsolete.

**Caratteristiche:**
* Natura ipertestuale dei documenti: Analisi degli hyperlink
* Struttura gerarchica dell'insieme di categorie
* Decomposizione della classificazione come decisione ramificata (A un nodo interno)

### Summarization

La summarization consiste nel generare un riassunto del contenuto di un testo.  Per testi brevi, il riassunto deve essere essenziale e coerente.  L'utilizzo di profili aiuta a strutturare le informazioni importanti in campi semanticamente ben definiti.  La tecnica di summarization dipende dalla natura dei documenti; ad esempio, per la summarization di recensioni, è possibile addestrare il modello per analizzare aspetti specifici, evitando un riassunto trasversale e generando invece riassunti per ogni aspetto elencato.

La summarization facilita principalmente l'accesso alle informazioni, come mostrato nei seguenti esempi:

* Estrazione delle parole chiave più utili da un insieme di documenti (es. un cluster) per descriverlo.
* Astrazione dei documenti in una collezione per evitare la lettura del contenuto completo.
* Riassunto dei documenti recuperati da una ricerca per consentire all'utente un'identificazione più rapida di quelli pertinenti alla query.

Un riassunto può essere di alto livello, fornendo una panoramica di tutti i punti principali, oppure più dettagliato.  Gli approcci alla summarization possono essere classificati in base alla dimensione dell'unità di testo utilizzata:

* **Riassunti di parole chiave:**  Utilizzano parole chiave per rappresentare il contenuto principale.
* **Riassunti di frasi:**  Utilizzano frasi per rappresentare il contenuto principale.

### Disambiguazione del Senso delle Parole (WSD)

La disambiguazione del senso delle parole (WSD) consiste nell'assegnare a una parola il significato corretto in base al contesto in cui appare.  È importante notare che, pur mirando alla disambiguazione di ogni parola singolarmente, è necessario considerare l'intero contesto per una corretta interpretazione.  Questo richiede un'analisi simultanea di tutti i termini nel contesto (aspetto da ampliare per supportare strumenti automatici).

**Un approccio efficace:** Prevede di sfruttare un inventario di sensi esistenti e misure di correlazione semantica per scegliere il significato più appropriato per ogni parola.

La WSD è un esempio del problema più generale di risoluzione delle ambiguità del linguaggio naturale.

**Esempio:**

* La parola inglese "bank" può avere (almeno) due significati:
    * "The Bank of England" (istituto finanziario)
    * "The bank of the river Thames" (argine fluviale)
* Per disambiguare l'occorrenza di "bank" in "Last week I borrowed some money from the bank", è necessario considerare il contesto per determinare il significato corretto (istituto finanziario).

**Approcci Knowledge-Based:**  Questa task ha conosciuto un rinnovato interesse con la disponibilità di risorse linguistiche elettroniche e, grazie al web, di risorse come Wikipedia (la cui tassonomia può essere assimilata a quella di un thesaurus).


## Filtraggio nel Text Mining

**Definizione:**
Il filtraggio nel text mining consiste nel classificare un flusso di documenti inviati da un produttore di informazioni a un consumatore di informazioni, bloccando la consegna di quelli irrilevanti per il consumatore.

**Esempio:** Un feed di notizie dove l'agenzia di stampa è il produttore e il giornale è il consumatore.

**Caratteristiche:**
* Considerabile come un caso di Text Classification con una singola etichetta (*rilevante o irrilevante*).
* Implementabile sia lato produttore (*instradamento selettivo*) che lato consumatore (*blocco*).
* Richiede la creazione e l'aggiornamento di un "*profilo*" per ogni consumatore (*lato produttore*) o un singolo profilo generale (*lato consumatore*).

**Filtraggio Adattivo:**
* Il profilo iniziale è definito dall'utente.
* Il profilo viene aggiornato in base al feedback dell'utente sulla pertinenza dei documenti consegnati.

### Filtraggio - CRM (Customer Relationship Management)

**Obiettivo:** Aiutare le aziende a comprendere le opinioni dei clienti.

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

**Applicazione:** Classificare le email come spam o legittime utilizzando il text mining.

**Sfide:**

* La valutazione del modello è complessa a causa dei costi asimmetrici degli errori (falsi positivi vs. falsi negativi).
* La distribuzione non uniforme delle classi (molto più spam che email legittime) complica l'apprendimento del classificatore. 

## Vocabolario: Modello di Rappresentazione dei Testi

### Definizione di Termine

**Problema:** Quali termini di un documento indicizzare?  Dobbiamo considerare tutte le parole o solo quelle ritenute "importanti"?  Come gestire la frequenza delle parole, considerando che alcune sono molto comuni mentre la maggior parte è rara?

**Contesto:** I testi sono composti da parole.  La selezione dei termini per l'indicizzazione (es. indice analitico di un libro) è fondamentale. L'indice è uno strumento per raggiungere un obiettivo specifico.

**Risposta:**  È necessario mettere in relazione le parole tra loro, creando pattern relazionali. L'importanza di una parola dipende da diversi concetti.

**Cosa è un termine?**

La definizione di "termine" è flessibile: può essere una parola singola, una coppia di parole, una frase, la radice di una parola (word stem), un n-gramma (sequenza di n caratteri), o un tipo di parola.

**Modellazione delle relazioni tra termini:**

Possiamo considerare diverse relazioni tra termini, ma per semplificare i modelli, si evitano generalmente:

* Paragrafi, frasi e ordine delle parole.
* Strutture sintattiche complesse come ellissi e anafore.

Si preferiscono invece le relazioni semantiche (es. relazioni "is-a", "part-of").

**Obiettivo:** Identificare termini che catturano la semantica del testo, evitando un'intensa elaborazione manuale (codifica manuale).


**Altri argomenti correlati:**

* **Analisi lessicale e morfologica:**  Include elaborazione della punteggiatura, conversione in minuscolo, rimozione di stopwords, stemming, lemmatizzazione e tagging delle parti del discorso.
* **Analisi della semantica del discorso (Anafora):**  Considera l'anafora letterale/pronominale, l'ellissi testuale e la meronomia referenziale.
* **Pragmatica, Semiotica e Morfologia:**  Questi campi forniscono ulteriori prospettive per una comprensione più completa del linguaggio.

## Tokenizzazione: Problemi Lessicali e Morfologici

Un token rappresenta un termine candidato durante la fase di pre-processing nell'analisi del testo.  La tokenizzazione, quindi, si occupa di come organizzare un testo in unità significative.

### Punteggiatura

La gestione della punteggiatura presenta diverse sfide:

* **Sequenze con trattino:**  Termini come *state-of-the-art*, *co-education*, *lowercase*, *lower-case*, *lower case* pongono il problema di come trattarli.  Mantenere le parole separate è una soluzione semplice ma può compromettere la semantica.  Un approccio più sofisticato è necessario, ma non sempre scalabile.

* **Apostrofi:**  La gestione di apostrofi in termini come *Italy’s capital* richiede una decisione su come rappresentarli (*Italy AND s*, *Italys*, *Italy’s*?).

* **Acronimi:**  Acronimi come *U.S.A* e *USA* possono essere trattati in modo diverso a seconda del contesto.

* **Entità nominate:**  Entità come *San Francisco* e *Hewlett-Packard* richiedono un trattamento specifico per evitare di frammentarle in unità non semantiche.


### Numeri

La gestione dei numeri richiede una scelta tra rimozione e mantenimento.  Generalmente, si rimuovono le stringhe puramente numeriche, mentre si mantengono quelle alfanumeriche.

**Motivazioni per il mantenimento dei numeri:**

* Preservazione delle informazioni numeriche contenute nel testo.
* Necessità di individuare codici di errore, intervalli temporali, ecc.

**Metadati:** I metadati (data di creazione, formato, ecc.) sono spesso indicizzati separatamente.  La complessità dell'interazione tra numeri e testo rende difficile una soluzione universale.

**Varietà di tipi numerici:**  I numeri possono assumere diverse forme:

* **Date:** *3/20/91*, *Mar. 12, 1991*, *20/3/91*.
* **Tempo:** *55 B.C.*
* **Codici:** *B-52*.
* **Identificatori:** *My PGP key is 324a3df234cb23e*.
* **Numeri di telefono:** *(800) 234-2333*.

**Case delle lettere:**  Generalmente, si converte tutto il testo in minuscolo (lowercase).


## Stopwords

È una fase banale, ma risulta fondamentale perchè è quella con l'impatto maggiore.
* **Definizione:** Parole grammaticali che servono a scopi grammaticali ma con scarso potere informativo. Sono specifiche della lingua.

* **Principio cardine nella pipeline di data analysis:** Le colonne con distribuzione super omogenea o super eterogenea vengono rimosse perché non sono informative. 
	* Le dimensioni delle rappresentazioni dovrebbero avere un giusto trade-off di generalizzazione e omogeneità.

* **Motivi per rimuovere le stopwords:**
    * Non si riferiscono a oggetti o concetti.
    * Hanno poco contenuto semantico.
    * Sono le parole più comuni.
    * Impattiamo sulla dimensionalità della rappresentazione dei nostri testi
    * Ciò implica **sparsità**: insieme sono un problema per le misure di distanza, che perdono di sensibilità $\to$ *perdita di potere discriminante*

* **Casi in cui le stopwords potrebbero essere necessarie:**
    * Query di frase: "King of Denmark"
    * Titoli: "Let it be", "To be or not to be"
    * Query "relazionali": "flights to London"

* **Gestione delle stopwords:**
    * **Buone tecniche di compressione:** Lo spazio per includere le stopwords in un sistema è molto piccolo.
    * **Buone tecniche di ottimizzazione delle query:** Si paga poco al momento della query per includere le stopwords. 
## Personalizzazione della Stop-List

**Obiettivo:** Non solo ridurre la stop-list, ma arricchirla.
* La stop-list deve contenere termini che si aspettano di essere molto frequenti all'interno della collezione. 

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

## Normalizzazione

L'obiettivo della normalizzazione è uniformare la forma delle parole nel testo indicizzato e nelle query di ricerca.  Questo permette di trovare corrispondenze tra termini che, pur avendo forme diverse, rappresentano lo stesso concetto.  
Ad esempio, "U.S.A." e "USA" dovrebbero essere trattati come equivalenti.
È fondamentale normalizzare in modo coerente il testo indicizzato e i termini delle query per garantire l'efficacia della ricerca.


**Risultato:**

* **Termine:** Un tipo di parola, dopo la normalizzazione.
* **Voce nel dizionario IR:** Ogni termine normalizzato rappresenta una voce nel dizionario di Information Retrieval (IR).

**Classi di equivalenza:**

La normalizzazione spesso si basa sulla creazione di classi di equivalenza, raggruppando termini considerati equivalenti.  Esempi di operazioni di normalizzazione includono:

* **Eliminazione di punti:** "U.S.A.", "USA" → "USA"
* **Eliminazione di trattini:** "anti-discriminatory", "antidiscriminatory" → "antidiscriminatory"
* **Eliminazione di accenti:** "French résumé" vs. "resume"
* **Eliminazione di umlaut:** "German Tuebingen" vs. "Tübingen"

Questo processo semplifica la rappresentazione del testo, focalizzandosi sulla struttura sintattica a discapito di alcune sfumature semantiche.  Diverse parole con significati simili vengono ridotte a una sola forma.

**Espansione asimmetrica:**

Un'alternativa alle classi di equivalenza è l'espansione asimmetrica.  In questo caso, l'inserimento di un termine porta all'inclusione di diverse varianti durante la ricerca:

* Inserire: "window"  → Ricerca: "window", "windows"
* Inserire: "windows" → Ricerca: "Windows", "windows", "window"
* Inserire: "Windows" → Ricerca: "Windows"

Questo approccio può essere più potente, ma meno efficiente.

**Criterio più importante: uso della lingua**

La scelta del metodo di normalizzazione dipende fortemente dall'uso della lingua.  Considerando che gli utenti spesso omettono accenti e trattini nelle query, è spesso preferibile normalizzare a una forma senza questi elementi.

**Tokenizzazione e normalizzazione:**

Tokenizzazione e normalizzazione sono processi interdipendenti e spesso dipendono dalla lingua.  Ad esempio, i formati di data "3/10/2000" (US) e "10/3/2000" (UE) richiedono un trattamento diverso.

## Case Folding

Tipicamente, si riducono tutte le lettere in minuscolo.
Se trattiamo questo task in maniera sofisticata dobbiamo avere un modo per riconoscere le named entity.
Eventualmente, possiamo mantenere solo gli acronimi
**Eccezione:** maiuscole a metà frase?

* **Esempio:** General Motors
* **Esempio:** Fed vs. fed
* **Esempio:** SAIL vs. sail

Spesso è meglio mettere tutto in minuscolo, poiché gli utenti useranno le minuscole indipendentemente dalla "corretta" capitalizzazione.

### Riduzione della matrice dei dati tramite algebra lineare

Come fase di post-processing, si può applicare l'algebra lineare alla matrice dei dati per ridurla ulteriormente. Questo processo si basa sull'identificazione di colonne linearmente dipendenti, che rappresentano sinonimi. 

L'obiettivo è ridurre questi sinonimi a una sola dimensione, concentrandosi sui pattern sintattici e non sulle sfumature semantiche. 

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

I tesauri generalizzano termini con significato correlato ma forme superficiali diverse, creando un indice più uniforme.  Questo approccio affronta il problema di sinonimi e omonimi.

* **Gestione di sinonimi e omonimi:** I tesauri permettono di gestire sia i sinonimi (parole con significato simile ma forma diversa) che gli omonimi (parole con forma uguale ma significato diverso).

* **Esempio:**  Classi di equivalenza possono essere costruite manualmente, ad esempio:
    * car = automobile
    * color = colour

* **Implementazione:**  Ci sono due approcci principali:

    * **Indicizzazione multipla:** Quando un documento contiene "automobile", viene indicizzato anche sotto "car-automobile" (e viceversa).  Questo crea una classe di equivalenza esplicita nell'indice.

    * **Espansione della query:** Quando una query contiene "automobile", la ricerca viene estesa anche al termine "car".  Questo approccio espande lo spazio di ricerca per includere termini correlati.


### Errori di Ortografia

Un approccio per gestire gli errori di ortografia è l'utilizzo di algoritmi come Soundex.  Soundex crea classi di equivalenza di parole basate su euristiche fonetiche, raggruppando parole che suonano in modo simile anche se scritte diversamente.

## Lemmatizzazione

La lemmattizzazione è un processo di normalizzazione che riduce le forme flessionali di una parola alla sua forma base, o lemma.  Questo si applica in particolare a verbi (riducendoli all'infinito) e sostantivi (riducendoli al singolare).

* **Esempio:** La frase "the boy's cars are different colors" potrebbe essere lemmattizzata in "the boy car be different color".  Si noti che i concetti principali vengono preservati.  In questo caso, la lemmattizzazione è un effetto collaterale dell'analisi sintattica, non l'obiettivo principale.

La lemmattizzazione richiede un'analisi morfologica completa per identificare correttamente il lemma di ogni parola nel dizionario.

* **Benefici per il retrieval:**  I benefici della lemmattizzazione per il retrieval di informazioni sono generalmente modesti.



## Stemming

Lo stemming, o "suffisso stripping", è un processo che riduce le parole alle loro radici prima dell'indicizzazione.  La radice è considerata il prefisso della parola.  Questo processo è intuitivo per i sostantivi, ma più complesso per i verbi.

È importante notare che lo stemming **distrugge informazioni lessicali**.  Dovrebbe essere utilizzato solo quando non è necessaria un'elaborazione semantica dei testi.  È spesso combinato con la rimozione delle *stop words*.  Non ha senso combinarlo con la lemmattizzazione, in quanto lo stemming è un processo più aggressivo.

* **Dipendenza dalla lingua:** Gli algoritmi di stemming sono specifici per ogni lingua.

* **Esempio:** Le parole "automate(s)", "automatic" e "automation" potrebbero essere tutte ridotte a "automat".

**Esempi di equivalenze create dallo stemming:**

* "compressed" e "compression" sono entrambe considerate equivalenti a "compress".
* "compress" e "compress" sono entrambe considerate equivalenti a "compress".  (Questo evidenzia la natura semplicistica dello stemming).


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

### Vantaggi dello Stemming

Lo stemming può influenzare le prestazioni del retrieval, generalmente in meglio, ma i risultati sono contrastanti.

* **Dipendenza dal vocabolario:** L'efficacia dello stemming dipende dal vocabolario specifico.
* **Perdita di sottili distinzioni:** Le sottili distinzioni semantiche possono essere perse attraverso lo stemming.

**Risultati per l'inglese:**

* I risultati sono molto contrastanti.
* Aiuta il recall per alcune query, ma danneggia la precisione per altre.
* **Esempio:** "operative" (odontoiatria) ⇒ "oper".

**Risultati per altre lingue:**

* Sicuramente utile per spagnolo, tedesco, finlandese, ecc.
* **Guadagni di prestazioni:** 30% per il finlandese.
* **Efficacia dello stemming automatico:** Lo stemming automatico è efficace quanto la conflazione manuale.

### Algoritmi di Stemming

Le prestazioni di vari algoritmi di stemming sono simili.

* **Stemmer di Lovins:**
    * Passaggio singolo, rimozione del suffisso più lungo (circa 250 regole).
* **Stemmer di Paice/Husk:**
* **Snowball:**

Tutti questi algoritmi hanno lo stesso impatto, ma presentano diverse peculiarità. 


## Modello Booleano

Il modello booleano di ricerca confronta una query booleana con gli insiemi di termini utilizzati per indicizzare il contenuto testuale.

**Query Booleana:**

* **Espressione:** Combinazione di termini di indice tramite operatori booleani.
* **Struttura:** L'espressione contiene:
    * **Termini di indice:** Parole chiave estratte dai documenti indicizzati.
    * **Operatori booleani:** `AND`, `OR`, e `NOT`, applicati ai termini. Il modello è rigido e la corrispondenza è binaria: un documento soddisfa la condizione o no.

**Applicazioni:**

Molti sistemi di ricerca utilizzano ancora il modello booleano, tra cui:

* Sistemi di posta elettronica
* Cataloghi di biblioteche
* macOS Spotlight


### Esempio: WestLaw

WestLaw è il più grande servizio commerciale di ricerca legale a pagamento.  Iniziato nel 1975, ha aggiunto la classificazione nel 1992 e una nuova ricerca federata nel 2010.  Gestisce decine di terabyte di dati e conta circa 700.000 utenti.  La maggior parte degli utenti utilizza ancora query booleane.


**Esempi di Query:**

* **Query 1:** Qual è il termine di prescrizione nei casi che coinvolgono il Federal Tort Claims Act?

    * **Espressione:** `LIMIT! /3 STATUTE ACTION /S FEDERAL /2 TORT /3 CLAIM`

    * **Spiegazione:** `/3` indica "entro 3 parole", `/S` indica "nella stessa frase".  Questi operatori di prossimità aggiungono informazioni contestuali.


* **Query 2:** Requisiti per le persone disabili per poter accedere a un luogo di lavoro.

    * **Espressione:** `disabl! /p access! /s work-site work-place (employment /3 place`

    * **Nota:** Lo spazio tra i termini indica una disgiunzione (OR), non una congiunzione (AND).


### Vantaggi del Modello Booleano

* **Query precise e lunghe:** Consentono di specificare con precisione i criteri di ricerca.
* **Operatori di prossimità:** Permettono di definire la relazione spaziale tra i termini.
* **Sviluppo incrementale:** Le query possono essere raffinate gradualmente, aggiungendo termini e operatori.
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

### Limitazioni del Modello Booleano

Il modello booleano è molto rigido: l'operatore `AND` richiede la presenza di tutti i termini, mentre `OR` richiede la presenza di almeno uno.  Questa rigidità presenta diverse limitazioni:

* **Query brevi:** Il modello incoraggia l'utilizzo di query brevi e semplici.
* **Scarsa flessibilità:** Richiede una scelta precisa dei termini di indice, portando a risultati potenzialmente controintuitivi.  Non prevede l'espansione della query (né in linea né a runtime) con termini semanticamente equivalenti. Ogni termine di indice è considerato in modo isolato.

**Difficoltà nel controllo dei risultati:**

* **Numero di documenti recuperati:** Tutti i documenti corrispondenti alla query vengono restituiti, senza possibilità di controllo sulla quantità.
* **Classificazione dei risultati:** Tutti i documenti corrispondenti soddisfano la query logicamente, senza possibilità di ordinamento in base alla rilevanza.
* **Feedback di rilevanza:**  È difficile utilizzare il feedback dell'utente (indicazione di rilevanza o irrilevanza di un documento) per migliorare la query.  Non essendo possibile un ranking dei risultati, non è possibile migliorare il processo di retrieval tramite il feedback di rilevanza.

**Espressività limitata:**

* **Complessi requisiti dell'utente:** Il modello ha difficoltà nell'esprimere richieste complesse. Non permette di regolare l'importanza dei termini nella query.
* **Sparsità:** La rappresentazione dei dati soffre del problema della sparsità, comune anche ad altri modelli di retrieval.

**Conclusioni:** Il modello booleano è adatto solo quando le esigenze dell'utente sono esprimibili tramite query corte e semplici, e l'obiettivo è solo determinare la presenza o l'assenza di un elemento.


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
    * **Intersezione (and):** $w(A∩B) = min(wA, wB)$
    * **Unione (or):** $w(A∪B )= max(wA, wB)$

### MMM: Mixed Min and Max Model

* **Termini:** t1, t2, ..., tn
* **Documento:** d, con pesi dei termini di indice: w1, w2, ..., wn
* **Query disgiuntiva:** $q_{or} = (t1 \lor t2 \lor \dots \lor tn)$
* **Somiglianza query-documento:**
    * $S(q_{or}, d) = λ_{or}  max(w1,.. ,wn) + (1 - λ_{or})  min(w1,.. , wn)$
    * $λ_{or}$ è un parametro che controlla il peso relativo del massimo e del minimo.
    * Con la logica booleana regolare, $λ_{or} = 1$.
    * Se la query è puramente disgiuntiva, si considera solo il primo termine.

* **Query congiuntiva:**
    * $q_{and} = (t1 \land t2 \land ... \land tn)$
    * $S(q_{and}, d) = λ_{and} min(w1,.. , wn) + (1 - λ_{and})  max(w1,.. , wn)$
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

## Sparsità

**Rappresentazione di matrici termine-documento:** Adottando un modello di rappresentazione di questo un modello poco sparso è irrealistico.
	
**Esempio:** Considera un insieme di **N = 1 milione di documenti**, ognuno con circa **1000 parole.**(Un esempio simile è la market-basket-analisys)
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

###### Il **core di un indice** è composto da due elementi principali:

* **Dizionario:** Contiene le parole del corpus, ciascuna associata a un puntatore alla lista di posting corrispondente.
* **Liste di posting:** Ogni lista contiene i documenti in cui la parola corrispondente compare, insieme alla sua frequenza nel documento.

Questo sistema di indicizzazione è doppiamente indicizzato:

* **Per chiave:** Il dizionario permette di accedere rapidamente alla lista di posting per una parola specifica.
* **Per documento:** Le liste di posting permettono di identificare rapidamente i documenti in cui una parola specifica compare, insieme alla sua frequenza nel documento


![[1) Intro-20241003104139467.png|393]]

L'indexer raccoglie i postings per ogni token, ovvero l'insieme di documenti in cui quel token appare. 

L'input dell'indexer è uno stream di coppie termine-ID documento. Questo stream rappresenta l'elenco di tutti i termini trovati nei documenti, insieme all'ID del documento in cui sono stati trovati. 

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
* La frase "*Ci sono 14 dipartimenti presso Unical*" non è una corrispondenza, poiché non corrisponde alla query di frase.
* Per questo tipo di query, non è sufficiente memorizzare solo coppie (Termine, docID).

**Approcci:**

* **Indici biword:**
    * Indizza ogni coppia consecutiva di termini nel testo come una frase (come se avvessimo un *n-gram* che preserva le parole con $n=2$).
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


## Dimensione dell'indice posizionale

Un indice posizionale espande in modo sostanziale l'archiviazione dei postings.
* Bisogna avere una voce per ogni occorrenza, non solo una per documento.
* La dimensione dell'indice dipende dalla dimensione media del documento. (Anche se gli indici possono essere compressi)
  
**Regola empirica:**
  * Un indice posizionale è 2-4 volte più grande di un indice non posizionale.
  * La dimensione dell'indice posizionale è il 35-50% del volume del testo originale.
Tuttavia, un indice posizionale è lo standard di utilizzo a causa della potenza e dell'utilità delle query di frase e prossimità, sia utilizzate esplicitamente che implicitamente in un sistema di recupero di ranking. 

## Costruzione dell'indice basata sull'ordinamento

Mentre costruiamo l'indice, analizziamo i documenti uno alla volta. I postings finali per qualsiasi termine sono incompleti fino alla fine del processo. Questo approccio richiede molto spazio per collezioni di grandi dimensioni. 
- Ad esempio, con 8 byte per ogni coppia (termID, docID) e considerando 100 milioni di postings (un valore realistico per un anno di pubblicazioni), in linea di principio sarebbe possibile indicizzarlo in memoria oggi. 
- Tuttavia, le *collezioni tipiche sono molto più grandi*: il New York Times, ad esempio, fornisce un indice di oltre 150 anni di notizie. 
**Pertanto, dobbiamo memorizzare i risultati intermedi su disco.**

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
* Non possiamo usare lo stesso algoritmo di costruzione dell'indice per collezioni più grandi, usando il disco invece della memoria, perchè ordinare 100 milioni di record su disco è troppo lento, poichè implica troppe ricerche su disco.
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
5          BSBI-Invert(block) //costruisce inverted index
6          WriteBlockToDisk(block, fn)  //scriviamo sul disco, presuppone la   ù
           presenza di una struttura di blocco sul disco
7   MergeBlocks(f_1, ..., f_n; f_merged)

```


#### Ordinamento di 10 blocchi di 10 milioni di record.
L'ordinamento di 10 blocchi da 10 milioni di record ciascuno avviene in due fasi:  ordinamento interno di ogni blocco (usando ad esempio Quicksort con complessità $O(N \log N)$) e successiva fusione delle run ordinate tramite un algoritmo di ordinamento esterno (come un merge sort esterno).  
L'utilizzo dello spazio su disco può essere ottimizzato evitando la necessità di due copie complete dei dati.


## Come Unire le Run Ordinate?

Esistono due approcci principali per unire run ordinate: il *merge binario* e il *multi-way merge*.

**Merge Binario:**

Si possono effettuare merge binari, utilizzando un albero di merge con  $\log_2(n)$ livelli, dove *n* è il numero di run.  Ad esempio, con 10 run, si avrebbero 4 livelli ($\log_2(10) \approx 4$).  Ad ogni livello, si leggono blocchi di run (es. 10 milioni di elementi per blocco), si uniscono e si riscrivono su disco.  Invece di un unico merge finale, si adotta una struttura ad albero con merge parziali a ogni livello:

1. **Partizionamento:** I dati vengono suddivisi in partizioni.
2. **Merge parziale:** Ogni partizione viene caricata in memoria e sottoposta a merge, generando un nuovo indice.
3. **Aggiornamento dell'indice:** L'indice aggiornato viene utilizzato per il merge del livello successivo.

![[1) Intro-20241003112202805.png]]

**Multi-way Merge:**

Un approccio più efficiente è il multi-way merge, che legge simultaneamente da tutti i blocchi.  Questo richiede:

* L'apertura simultanea di tutti i file di blocco.
* Un buffer di lettura per ogni blocco e un buffer di scrittura per il file di output.
* L'utilizzo di una coda di priorità per selezionare, ad ogni iterazione, il `termID` più basso non ancora elaborato.
* L'unione delle liste di postings corrispondenti a quel `termID` (provenienti dai vari blocchi) e la scrittura del risultato.

Questo metodo è efficiente a condizione che si leggano e si scrivano blocchi di dimensioni adeguate.

**Assunzione:** Si assume una condivisione solo parziale del lessico tra i vari documenti.


**Problema della Crescita del Lessico:**

La gestione del lessico durante il merge presenta sfide:

* **Gestione dei termini e degli ID:** È fondamentale gestire correttamente i termini (token pre-processati) e i loro identificativi (ID).
* **Compressione con perdita:** La compressione con perdita può essere applicata solo a termini non essenziali per la comprensione del testo.
* **Valutazione dei token:** Sono necessarie strategie per valutare l'importanza dei token e decidere quali possono essere compressi con perdita.


## SPIMI: Indicizzazione in memoria a passaggio singolo (approccio lazy)

* **Problema con l'algoritmo basato sull'ordinamento:**
    * La mappatura (termine, termID) potrebbe non entrare in memoria.
    * Abbiamo bisogno del dizionario (che cresce dinamicamente) per implementare una mappatura termine-termID.

Idee chiave: sono complementari
* **Idea chiave 1:** *Generare dizionari separati per ogni blocco*. Non c'è bisogno di mantenere la mappatura termine-termID tra i blocchi (mapping across block).
* **Idea chiave 2:** *Non ordinare*. Accumulare i postings nelle liste di postings man mano che si verificano.

* Sulla base di queste due idee, generiamo un indice invertito completo per ogni blocco.
* Questi indici separati possono quindi essere uniti in un unico grande indice.
* SPIMI può indicizzare collezioni di qualsiasi dimensione a condizione che ci sia abbastanza spazio su disco disponibile.
* **Ogni lista di postings è una struttura dinamica e immediatamente disponibile per collezionare i postings**.
    * **più veloce** - non è necessario ordinare.
    * **risparmia memoria** - i termID dei postings non devono essere memorizzati e non vi sono fasi di sorting intermedie. 
    * In pratica, è una struttura che conserviamo in memoria e che viene riallocata all'occorrenza.
    * Evitiamo di tenere traccia dei term-id, l'algoritmo lavora direttamente con i termini

### SPIML-Invert (token_stream)

```python

1.  output_file = NEWFILE()
2.  dictionary = NEWHASH()
3.  while (free memory available)
4.      do token ← next(token_stream)
5.      if term(token) ∉ dictionary
6.          then postings_list = ADDToDICTIONARY(dictionary, term(token))
7.      else postings_list = GETPOSTINGSLIST(dictionary, term(token))
8.      if full(postings_list)
9.          then postings_list = DOUBLEPOSTINGSLIST(dictionary, term(token))
10.     ADDToPOSTINGSLIST(postings_list, docID(token))
11.     sorted_terms ← SORTTERMS(dictionary)
12.     WRITEBLOCKToDISK(sorted_terms, dictionary, output_file)
13.     return output_file


# Linea 5: il token raw viene pre-processato; viene ricondotto a un index-term.
# Linea 10: il posting (occorrenza del termine) nel documento viene aggiunto immediatamente.
# Linee 8-9: se la lista di postings raggiunge la dimensione limite, viene raddoppiata.
# Linea 11: L'ordinamento dei termini avviene dopo aver raccolto tutti i postings per un blocco.
```

## Indicizzazione Distribuita

L'indicizzazione distribuita prevede l'esecuzione di due task paralleli: il parsing e l'inversione dell'indice. 
Un'unica macchina master coordina il processo, suddividendo l'indice in una serie di task paralleli. 
Ogni split rappresenta un insieme di documenti gestito come blocchi. La macchina master assegna i ruoli ai diversi nodi del sistema.

### Parser

* Il master assegna uno split a una macchina parser inattiva.
* Il parser legge un documento alla volta ed emette coppie (termine, documento).
* Il parser scrive le coppie in *j* partizioni. Le partizioni (fold) sono determinate in modo lessicografico.
    * **Esempio:** Ogni partizione è per un intervallo di lettere iniziali dei termini (ad esempio, a-f, g-p, q-z), quindi *j* = 3.

Una volta completato il parsing, si procede con l'inversione dell'indice.

### Inverter

* Un inverter raccoglie tutte le coppie (termine, doc) (cioè, postings) per una partizione di termini.
* Ordina e scrive le liste di postings. 


![[1) Intro-20241003093756427.png]]

La figura mostra come l'indicizzazione distribuita possa essere vista come un'istanza particolare del modello MapReduce.

**Fase di Map:**

* Partendo dalla collezione di documenti in input, la fase di map produce liste di coppie (termine, documento).

**Fase di Reduce:**

* La fase di reduce prende le liste di occorrenze (coppie termine-documento) e produce le liste di postings, ovvero le liste di documenti in cui un termine compare. 

![[1) Intro-20241003093804693.png]]
## Indicizzazione Distribuita

L'algoritmo di costruzione dell'indice è un'istanza di MapReduce.

* **MapReduce:** Un framework robusto e concettualmente semplice per il calcolo distribuito, che permette di eseguire calcoli complessi senza dover scrivere codice per la parte di distribuzione.
* **Sistema di indicizzazione di Google (circa 2002):** Composto da una serie di fasi, ciascuna implementata in MapReduce.

### Schema per la costruzione dell'indice in MapReduce

* **Funzioni map e reduce:**
    * `map`: input → list(k, v)
    * `reduce`: (k,list(v)) → output
* **Istanza dello schema per la costruzione dell'indice:**
    * `map`: collection → list(termID, docID)
    * `reduce`: (<termID1, list(docID)>, <termID2, list(docID)>, …) → (postings list1, postings list2, …)

Fino ad ora, abbiamo ipotizzato che le collezioni siano statiche, ma:

* **Documenti in arrivo:** I documenti arrivano nel tempo e devono essere inseriti.
* **Documenti eliminati e modificati:** I documenti vengono eliminati e modificati (ad esempio, quando l'entità dell'editing è tale da toccare la maggior parte dei termini, come upgrade, motivi di privacy o cambio di normativa).

La gestione degli aggiornamenti richiede modifiche al dizionario e alle liste di postings:  i *termini già presenti* nel dizionario necessitano di aggiornamenti alle liste di postings, mentre i *nuovi termini* devono essere aggiunti al dizionario stesso.

### Approccio più semplice

* **Indice principale e indice ausiliario:** Mantenere un "grande" indice principale, i nuovi documenti vanno in uno (o più) "piccolo" indice ausiliario.
* **Ricerca:** La ricerca viene effettuata su entrambi gli indici, unendo i risultati.
* **Eliminazioni:**
    * **Vettore di bit di invalidazione:** Un vettore di bit indica i documenti eliminati.
    * **Filtraggio:** I documenti in output su un risultato di ricerca vengono filtrati tramite questo vettore di bit di invalidazione.
* **Re-indicizzazione:** Periodicamente, re-indicizzare in un unico indice principale.

### Problemi con gli indici principali e ausiliari

* **Merge frequenti.**
* **Scarsa performance durante i merge.**
* **Efficienza della fusione:** La fusione dell'indice ausiliario nell'indice principale è efficiente se si mantiene un file separato per ogni lista di postings.
    * La fusione è la stessa di una semplice append.
    * Ma poi avremmo bisogno di molti file - inefficiente per il sistema operativo.
* **Ipotesi:** L'indice è un unico grande file.
    * **Realtà:** Usare uno schema da qualche parte nel mezzo (ad esempio, dividere le liste di postings molto grandi, raccogliere le liste di postings di lunghezza 1 in un unico file, ecc.).

## Fusione Logaritmica

La fusione logaritmica è una tecnica di ordinamento che utilizza una serie di indici di dimensioni crescenti per ordinare un insieme di dati. 

**Principio di funzionamento:**

1. **Serie di indici:** Si mantiene una serie di indici, ciascuno con una dimensione doppia rispetto al precedente. Ad esempio, si potrebbe avere un indice di dimensione 1, 2, 4, 8, 16, e così via.
2. **Memoria e disco:** L'indice più piccolo $(Z_0)$ è mantenuto in memoria, mentre gli indici più grandi $(I_0, I_1, ...)$ sono memorizzati sul disco.
3. **Fusione e scrittura:** Quando l'indice $Z_0$ diventa troppo grande (supera la sua capacità), viene scritto sul disco come $I_0$. In alternativa, se $I_0$ esiste già, $Z_0$ viene unito con $I_0$ per formare $Z_1$.
4. **Iterazione:** Se $Z_1$ non è ancora troppo grande, viene mantenuto in memoria. Altrimenti, viene scritto sul disco come $I_1$. Se $I_1$ esiste già, $Z_1$ viene unito con $I_1$ per formare $Z_2$.
5. **Ripetizione:** Questo processo di fusione e scrittura viene ripetuto fino a quando tutti i dati non sono stati ordinati.

![[1) Intro-20241007154640070.png|436]]

![[1) Intro-20241007154611506.png|450]]
## Indicizzazione Dinamica

#### Indice Ausiliario e Principale

* **Fusione T/n:**
    * Si utilizzano indici ausiliari di dimensione $n$ e un indice principale con $T$ postings.
    * Il tempo di costruzione dell'indice è $O\left( \frac{T^2}{n} \right)$ 
 nel caso peggiore, poiché un posting potrebbe essere toccato $\frac{T}{n}$ volte.

#### Fusione Logaritmica

* **Efficienza:** Ogni posting viene fuso al massimo $O\left( log\left( \frac{T}{n} \right) \right)$ volte, con una complessità di $O\left( T \cdot log\left( \frac{T}{n} \right) \right)$
* **Vantaggi:** La fusione logaritmica è molto più efficiente per la costruzione dell'indice rispetto alla fusione $\frac{T}{n}$.
* **Svantaggi:** L'elaborazione delle query richiede la fusione di $O\left( log\left( \frac{T}{n} \right) \right)$ indici, mentre con un solo indice principale e ausiliario la complessità è $O(1)$.

## Ulteriori Problemi con Più Indici

Mantenere le statistiche a livello di collezione con più indici è complesso. Ad esempio, per la correzione ortografica, è difficile scegliere l'alternativa corretta con il maggior numero di risultati. 

* **Problema:** Come mantenere le migliori alternative con più indici e vettori di bit di invalidazione?
* **Possibile soluzione:** Ignorare tutto tranne l'indice principale per l'ordinamento.

Il ranking dei risultati si basa su queste statistiche, rendendo la loro gestione cruciale.

### Indicizzazione Dinamica nei Motori di Ricerca

I motori di ricerca effettuano l'indicizzazione dinamica con:

* **Modifiche incrementali frequenti:**  es. notizie, blog, nuove pagine web.
* **Ricostruzioni periodiche dell'indice da zero:** L'elaborazione delle query viene commutata sul nuovo indice e il vecchio indice viene eliminato.

### Requisiti per la Ricerca in Tempo Reale

La ricerca in tempo reale richiede:

* **Bassa latenza:** Elevata produttività di valutazione delle query.
* **Elevato tasso di ingestione:** Immediata disponibilità dei dati.
* **Letture e scritture concorrenti:** Gestione di letture e scritture simultanee dell'indice.
* **Dominanza del segnale temporale:** Priorità ai dati più recenti.


## Costruzione dell'Indice: Riepilogo

### Indicizzazione basata sull'ordinamento

Esistono diverse tecniche di indicizzazione basata sull'ordinamento:

* **Inversione in memoria naive.**
* **Indicizzazione basata sull'ordinamento bloccato (BSBI).**
* **L'ordinamento per fusione è efficace per l'ordinamento basato su disco rigido (evita le ricerche!).**

### Indicizzazione in memoria a passaggio singolo (SPIMI)

* **Nessun dizionario globale.**
* **Genera un dizionario separato per ogni blocco.**
* **Non ordinare i postings.**
* **Accumulare i postings nelle liste di postings man mano che si verificano.**

## Compressione

La compressione dei dati è fondamentale in Information Retrieval per diversi motivi:

* **Ridurre lo spazio di archiviazione su disco:**  Minore spazio occupato significa costi di storage ridotti.
* **Aumentare la capacità di memoria:** Permette di mantenere in memoria principale una maggiore quantità di dati.
* **Accelerare il trasferimento dati:**  Il trasferimento di dati compressi da disco a memoria è più veloce.
* **Migliorare le prestazioni di lettura:** La lettura e decompressione di dati compressi può essere più veloce rispetto alla lettura di dati non compressi, a patto di utilizzare algoritmi di compressione e decompressione efficienti.

**Nota:** Gli algoritmi di decompressione utilizzati nei sistemi di Information Retrieval sono generalmente molto veloci.


### Compressione del Dizionario

La compressione del dizionario è cruciale per:

* **Memorizzazione in memoria principale:**  Permette di mantenere l'intero dizionario in memoria RAM, accelerando le operazioni di ricerca.
* **Collocazione di liste di postings in memoria:** Consente di mantenere in memoria anche alcune liste di postings, migliorando ulteriormente le prestazioni.


### Compressione dei File di Postings

La compressione dei file di postings è importante per:

* **Ridurre lo spazio su disco:**  Minimizza lo spazio di archiviazione necessario.
* **Accelerare la lettura:** Riduce il tempo necessario per leggere le liste di postings dal disco.
* **Migliorare l'accesso in memoria:** I grandi motori di ricerca spesso mantengono una parte significativa dei postings in memoria principale per prestazioni ottimali.


### Compressione Senza Perdita vs. Con Perdita

Esistono due tipi principali di compressione:

* **Compressione senza perdita:**  Preserva tutte le informazioni originali. Questo è l'approccio prevalentemente utilizzato in Information Retrieval.
* **Compressione con perdita:**  Scarta alcune informazioni per ottenere un maggiore livello di compressione.  Esempi di compressione con perdita in IR includono:
    * **Pre-elaborazione del testo:** Conversione in minuscolo, rimozione di stop words, stemming, eliminazione di numeri.
    * **Poda delle liste di postings:** Rimozione di voci con bassa probabilità di apparire nei primi *k* risultati per qualsiasi query.  Questa tecnica introduce una minima perdita di qualità nei risultati di ricerca.


## Dimensione del vocabolario vs. dimensione della collezione

### Quanto è grande il vocabolario dei termini?

* **Cioè, quante parole distinte ci sono?**
* **Possiamo assumere un limite superiore?**
    * Non proprio: almeno $7020 = 1037$ parole diverse di lunghezza $20$.
    * In pratica, il vocabolario continuerà a crescere con la dimensione della collezione.
    * Soprattutto con Unicode.

### Legge di Heaps: $M = kT^b$

* **M** è la dimensione del vocabolario (cresce seguento una **power law**: *funzione lineare in scala doppia logaritmica con offset k),* **T** è il numero di token nella collezione.
* Valori tipici: $30 ≤ k ≤ 100$ e $b ≈ 0.5$.
* In un grafico log-log della dimensione del vocabolario M vs. T, la legge di Heaps prevede una linea con pendenza di circa ½.
* È la relazione più semplice possibile (lineare) tra i due in spazio log-log.
    * $log M = log k + b log T$.
* Un'osservazione empirica ("legge empirica").

### Legge di Heaps per Reuters RCV1:

$$log_{10}M = 0.49  \log_{10}T + 1.64
\to
M = 10^{1.64}T^{0.49}$$
cioè: 
- $k=10^{1.64}≈ 44$  
- $b = 0.49$.

Buona aderenza empirica per Reuters RCV1:

* Per i primi 1.000.020 token, prevede 38.323 termini;
* in realtà, 38.365 termini. 

![[1) Intro-20241007160038284.png|371]]
Vi è una fase transitoria iniziale e poi inizia a fittare a regime

## Distribuzione Skewd di Tipo Power-law

La distribuzione skewd di tipo power-law è caratterizzata da una concentrazione di massa in una zona relativamente piccola della distribuzione, seguita da una coda lunga o grassa. 

**Caratteristiche:**

* **Concentrazione di massa:** La maggior parte della massa è concentrata in una piccola porzione della distribuzione.
* **Coda lunga:** La distribuzione presenta una coda che si estende per un lungo periodo, con valori che diminuiscono lentamente.

**Nota:** La distribuzione skewd di tipo power-law è spesso osservata in fenomeni naturali e sociali. 

### Distribuzione di tipo Power-law
La **distribuzione di tipo power law** è un modello matematico che descrive la distribuzione di molti fenomeni naturali e sociali, come la dimensione delle città, la frequenza delle parole in un linguaggio e la ricchezza delle persone.  È caratterizzata da una concentrazione di massa in una zona relativamente piccola della distribuzione, seguita da una coda lunga o grassa(simile alla funzione esponenziale).  In altre parole, pochi elementi hanno un valore molto alto, mentre molti altri hanno un valore molto basso. 

### Legge di Pareto
Un esempio di questo modello è la **legge di Pareto**, nota anche come principio 80-20, che afferma che l'80% degli effetti deriva dal 20% delle cause.  Nel contesto del linguaggio naturale, la legge di Zipf è un esempio di distribuzione di tipo power law, dove pochi termini sono molto frequenti, mentre molti altri sono molto rari. 

**Esempi di distribuzione di tipo power law:**

* **Distribuzione della ricchezza tra individui:** Pochi individui possiedono la maggior parte della ricchezza, mentre molti altri hanno una ricchezza molto bassa.
* **Numero di pagine di siti web:** Pochi siti web hanno un numero molto elevato di pagine, mentre molti altri hanno un numero di pagine molto basso.
* **Numero di follower di un social network:** Pochi utenti hanno un numero molto elevato di follower, mentre molti altri hanno un numero di follower molto basso.
* **Dimensione delle città:** Poche città hanno una popolazione molto elevata, mentre molte altre hanno una popolazione molto bassa.
- **Frequenza delle parole in un documento**

**Confronto tra la distribuzione di Poisson (legge degli eventi rari) con la legge di potenza:** 
Entrambe le distribuzioni sono asimmetriche (skewed), ma si differenziano per la loro natura:
La scelta tra le due distribuzioni dipende da come si modella il tasso minimo (min rate) nella distribuzione di Poisson.
- **Poisson:** È una distribuzione spaziale, che descrive la probabilità di un certo numero di eventi in un dato intervallo di tempo o spazio.
- **Legge di potenza:** È una distribuzione che descrive la relazione tra due variabili, dove una varia in modo proporzionale a una potenza dell'altra.

## Legge di Heaps e Legge di Zipf

La legge di Heaps (prima legge di potenza) fornisce una stima della dimensione del vocabolario in un corpus di testo. Tuttavia, nel linguaggio naturale, si osserva una distribuzione non uniforme delle parole: alcuni termini sono molto frequenti, mentre molti altri sono molto rari.

### Legge di Zipf (seconda legge di potenza)

Zipf (1949) ha scoperto una relazione empirica tra la *frequenza* di un termine e il suo *rango* nel vocabolario. 
Emerge che vi siano termini più frequenti di altri, che sono in minoranza rispetto agli atlri
La legge di Zipf afferma che l' i-esimo termine più frequente ha una frequenza di collezione proporzionale a $\frac{1}{i}$:

$$cf_{i} \propto \frac{1}{i} = \frac{K}{i}$$

dove $cf_{i}$ è la frequenza del termine i-esimo, $i$ è il suo rango nel vocabolario(posizione in una classifica stabilita su una lista di frequenze) e $K$ è una costante di normalizzazione.

In forma logaritmica, la legge di Zipf si esprime come:

$$log(cf_{i}) = log(K) - log(i)$$

Questa equazione indica una relazione lineare inversa tra il logaritmo della frequenza del termine e il logaritmo del suo rango. Questa relazione è nota come legge di potenza.
È dunque una power law con slope negativa

**Esempio:**

Se il termine più frequente ("the") si verifica $cf1$ volte, allora il secondo termine più frequente ("of") si verifica $cf1/2$ volte, il terzo termine più frequente ("and") si verifica $cf1/3$ volte, e così via.


![[1) Intro-20241007160144378.png|504]]
- In posizioni di rank basse (origine) abbiamo alta frequenza.  
- La frequenza scende linearmente con scala doppia logaritmica 

## La legge di Zipf: le implicazioni di Luhn

Luhn (1958) osservò che:

* **Le parole estremamente comuni non sono molto utili per l'indicizzazione.** Questo perché sono troppo generiche e non forniscono informazioni specifiche sul contenuto di un documento.
* **Le parole estremamente rare non sono molto utili per l'indicizzazione.** Questo perché compaiono troppo raramente per essere significative.

**I concetti più discriminanti hanno una frequenza da bassa a media.** Questo significa che le parole che compaiono con una frequenza intermedia sono le più utili per l'indicizzazione, perché forniscono informazioni specifiche sul contenuto di un documento senza essere troppo rare da essere insignificanti.

La distribuzione delle parole in un corpus segue la legge di Zipf, che afferma che la frequenza di una parola è inversamente proporzionale al suo rango. Questo significa che le parole più frequenti sono molto più comuni delle parole meno frequenti.

![[1) Intro-20241010152315456.png]]

Il grafico mostra la distribuzione delle parole in un corpus. La maggior parte delle parole ha una frequenza medio-bassa, mentre poche parole hanno una frequenza molto alta.

**In funzione del task specifico, non è immediato stabilire quando un termine è eccessivamente frequente e allo stesso modo quando è eccessivamente raro.** 

Se fossimo capaci di stabilirlo, allora avremmo stabilito le due frequenze di taglio (superiore e inferiore). Ciò che è interno a queste frequenze di taglio è ciò che andrebbe mantenuto (vocabolario) nelle fasi successive.

**Determinare la frequenza di taglio è difficile.** Oltre ad essere task-dependent, è data-driven, dipende dal linguaggio utilizzato dal corpus (dominio dei dati). 

In teoria, potremmo individuare due frequenze di taglio per effettuare un pruning dell'insieme dei termini candidati a formare il vocabolario (index terms). 

**Scopo delle frequenze di taglio:**
L'obiettivo è escludere:

* **Termini troppo frequenti:** presenti in molti documenti ma poco significativi per la caratterizzazione del contenuto (es. articoli, preposizioni).
* **Termini troppo rari:** presenti in pochi documenti e quindi poco utili per l'analisi generale.

**Vantaggi del pruning:**
* Miglioramento dell'efficienza: riduzione della dimensionalità del vocabolario, con conseguente risparmio di memoria e risorse computazionali.
* Maggiore accuratezza: eliminazione di termini poco significativi o fuorvianti.

**Criticità:**
* **Individuazione delle frequenze di taglio ottimali:** non esiste una regola universale, la scelta dipende dal task specifico e dal dominio di applicazione.
* **Dipendenza dal dominio e dal linguaggio:**  la frequenza di un termine può variare significativamente a seconda del contesto.

**Esempi di regole pratiche:**
Nonostante la mancanza di regole universali, l'esperienza su diversi benchmark ha permesso di identificare alcune linee guida:
* **Rimozione dei termini troppo frequenti:** escludere termini presenti in più del 50% dei documenti.
* **Rimozione dei termini troppo rari:** escludere termini presenti in meno di 3-5 documenti.

È preferibile un approccio conservativo, che mantenga un vocabolario relativamente ampio, soprattutto per modelli di rappresentazione del testo utilizzati in:
* **Information Retrieval:** dove la rilevanza si basa principalmente sulla presenza dei termini.
* **Data Mining tradizionale:** dove si utilizzano ancora modelli basati sulla frequenza dei termini.

## Ponderazione della Rilevanza dei Termini

Abbiamo a che fare con set di risultati ridotti o enormi. Quando il set di risultati deve essere processato direttamente dall'utente che ha posto la query, non è pensabile che esso possa ispezionare l'intero set. In numerosi scenari, c'è solo attenzione ai primi *k* risultati.

#### Ricerca booleana: 

Fino ad ora, tutte le query sono state booleane. Questo approccio presenta alcuni svantaggi:

* Spesso si ottengono risultati troppo pochi (o addirittura vuoti) o troppi (>1K).
* I documenti o corrispondono o no.
* È un metodo adatto per gli utenti esperti con una precisa comprensione delle loro esigenze e della collezione.
* È anche utile per le applicazioni che possono facilmente consumare migliaia di risultati.
* Tuttavia, non è adatto per la maggior parte degli utenti.
* La maggior parte degli utenti è incapace/non disposta a scrivere query booleane.
* La maggior parte degli utenti non vuole scorrere migliaia di risultati.
* Questo è particolarmente vero per la ricerca sul web.

Dobbiamo proiettarci verso un approccio più generale, che è quello del **recupero classificato**

## Ranked Retrival

Per risolvere i problemi della ricerca booleana, si introduce il **ranked retrival**, che restituisce un ordinamento sui (primi) documenti della collezione per una query.

Il recupero classificato si basa sull'idea di esprimere quantitativamente la pertinenza di un documento rispetto alla query. Si parla di **recupero** e non di **estrazione di informazioni**. 
Ad esempio, se una pagina è complessa e funge anche da hub verso altre pagine, ci interessa che sia un'autorità e non un hub. In tal caso, sarebbe necessario eseguire l'estrazione di informazioni. Ci basta che la pagina abbia una percentuale di contenuti che corrispondono alla query, e in tal caso la restituiamo tra le prime. 

#### Query di testo libero

Le query di testo libero sono semplicemente una o più parole in una lingua umana, anziché un linguaggio di query di operatori ed espressioni.

*Il recupero classificato è normalmente stato associato alle query di testo libero, e viceversa*.

Con il recupero classificato, il problema dell' "abbondanza o carestia" non è più un problema. Un sistema produce un set di risultati classificati, i set di risultati di grandi dimensioni non sono un problema.

* La dimensione del set di risultati non è un problema.
* Mostriamo solo i primi k (≈ 10) risultati.
* Non sovraccarichiamo l'utente.

Con una query free text ammettiamo ci sia un'esigenza informativa non netta al 100%. Dunque è utile presentare i risultati all'utente con una classifica

### Punteggio come base del recupero classificato

Desideriamo restituire in ordine i documenti più probabilmente utili per il ricercatore.

Come possiamo classificare i documenti della collezione rispetto a una query?

* *Assegnare un punteggio (in [0, 1]) a ciascun documento.*
* Questo punteggio misura quanto bene il documento e la query *corrispondono*.

Un modo è quello di usare **misure di similarità** per insiemi finiti:
* Ad esempio: *Jaccard, Sorensen-Dice, Overlap, Simple Matching, ecc.*
* Sono efficienti e forniscono la normalizzazione della lunghezza.
    * Cioè, i documenti e le query non devono avere le stesse dimensioni.
* Ma non considerano:
    * La frequenza del termine $(tf)$ nel documento.
        * Più frequente è il termine di query nel documento, più alto dovrebbe essere il punteggio.
    * La scarsità del termine nella collezione (frequenza di menzione del documento).
        * I termini rari in una collezione sono più informativi dei termini frequenti.

## Misure di Similarità

| Coefficiente        | Formula                                                                            | Descrizione                                                                                                                                                                                                                         |
| ------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Jaccard**         | $J(A,B) = \frac{\|A \cap B\|}{\|A \cup B\|}$                                       | **Misura la similarità tra due insiemi come la dimensione della loro intersezione divisa per la dimensione della loro unione**. *Varia tra 0 (nessuna similarità) e 1 (identità).*                                                  |
| **Sørensen-Dice**   | $DSC(A,B) = \frac{2 \times \|A \cap B\|}{\|A\| + \|B\|}$                           | **Simile a Jaccard, ma pesa doppiamente gli elementi in comune**. *Varia tra 0 (nessuna similarità) e 1 (identità).*                                                                                                                |
| **Overlap**         | $O(A,B) = \frac{\|A \cap B\|}{min(\|A\|, \|B\|)}$                                  | **Misura la sovrapposizione tra due insiemi come la dimensione della loro intersezione divisa per la cardinalità dell'insieme più piccolo**. *Varia tra 0 (nessuna sovrapposizione) e 1 (un insieme è un sottoinsieme dell'altro).* |
| **Simple Matching** | $SM(A,B) = \frac{\|A \cap B\| + \|\overline{A} \cap \overline{B}\|}{\|A \cup B\|}$ | **Misura la similarità tra due insiemi considerando sia le presenze che le assenze di elementi**. *Varia tra 0 (nessuna similarità) e 1 (identità).*                                                                                |

La cardinalità di X con Y include anche l'intersezione. Lo stesso vale per la cardinalità di Y e l'intersezione. Ovviamente, le due misure sono indipendenti, nel senso che conoscerne una non implica la conoscenza dell'altra. In che relazione si trovano? Si inseriscono nella relazione espressa dalla notazione con la J, vista prima applicata alle variabili di J e sottolineando "non J" più 1. 

Se vogliamo dare maggior peso agli elementi in comune, quindi desideriamo una misura di similarità che non sia "generosa" in termini di intersezione, utilizziamo Dice. Dice è più generosa rispetto a Jaccard, ma manca di una proprietà importante. Per intenderci, Dice è sempre maggiore o uguale a Jaccard, ma in alcuni casi può essere utile, ad esempio nell'ambito dell'indexing, in particolare per l'Iris Neighbor Search.

Parlando di proprietà desiderabili, seppur non obbligatorie, per una misura di distanza, la **disuguaglianza triangolare** è fondamentale. È ciò che rende una distanza una metrica. A questo proposito, visto che siete persone precise, evitate di cadere nella trappola comune a molti, dagli informatici agli ingegneri: non tutto è una metrica. Usiamo termini come "misure" o "criteri" in senso generico, ma "metrica" è riservato a Jaccard e altre misure che soddisfano la disuguaglianza triangolare. Dice non la soddisfa.

### Esempio di non soddisfacimento della disuguaglianza triangolare con Dice

Prendiamo due documenti:

* Documento 1: AB
* Documento 2: A
* Documento 3: B

Applicando la disuguaglianza triangolare, dovremmo avere:

* $Distanza(1,2) + Distanza(1,3) \geq Distanza(2,3)$

Tuttavia, usando Dice (che misura la similarità, non la distanza), otteniamo:

* $Dice(1,2) = \frac{2}{3}$
* $Dice(1,3) = \frac{2}{3}$
* $Dice(2,3) = 0$

La disuguaglianza non è rispettata, poiché 2/3 + 2/3 non è maggiore o uguale a 0.

### Overlap Coefficient

L'Overlap Coefficient, detto anche Simpson o con un altro nome difficile da ricordare, è definito come l'intersezione divisa per la minima cardinalità tra X e Y. 


$Overlap(X,Y) = \frac{{|X ∩ Y|}}{{min(|X|, |Y|)}}$


Calcolare il word overlap tra due porzioni di testo può essere utile, ad esempio per confrontare due summary, uno di riferimento e l'altro generato artificialmente.

### Altre misure di similarità

Un'altra misura di similarità, non direttamente legata al concetto di matching, è... (la trascrizione si interrompe qui).

---
#### Frequenza di Collezione (Collection Frequency)
La **frequenza di collezione** (collection frequency) rappresenta il numero totale di occorrenze di una parola all'interno di un'intera collezione di documenti. È una proprietà globale, ovvero riguarda l'intera collezione e non un singolo documento. Rappresenta la somma delle Term Frequency su tutti i documenti della collezione

#### Frequenza di Termine (Term Frequency)
La **frequenza di termine** (term frequency) è la controparte della frequenza di collezione, ma a livello locale. Indica il numero di occorrenze di una parola all'interno di un singolo documento. 

#### Document Frequency
La **Document Frequency** è il numero di documenti in cui un termine compare.

## Rappresentazione dei testi: calcolo dello score

Ogni singola cella della matrice di rappresentazione dei testi dovrà contenere un valore reale che esprime il peso del termine all'interno di un documento (quanto contribuisce a rappresentare il contenuto del documento stesso).

**Come calcoliamo lo score da assegnare a un documento?**

### Tabella dei personaggi e delle opere

| Personaggio | Antony and Cleopatra | Julius Caesar | The Tempest | Hamlet | Othello | Macbeth |
| --- | --- | --- | --- | --- | --- | --- |
| Antony | 1 | 1 | 0 | 0 | 0 | 1 |
| Brutus | 1 | 1 | 0 | 1 | 0 | 0 |
| Caesar | 1 | 1 | 0 | 1 | 1 | 1 |
| Calpurnia | 0 | 1 | 0 | 0 | 0 | 0 |
| Cleopatra | 1 | 0 | 0 | 0 | 0 | 0 |
| mercy | 1 | 0 | 0 | 1 | 1 | 0 |
| worser | 1 | 0 | 0 | 1 | 1 | 0 |

| Personaggio | Antony and Cleopatra | Julius Caesar | The Tempest | Hamlet | Othello | Macbeth |
| --- | --- | --- | --- | --- | --- | --- |
| Antony | 157 | 73 | 0 | 0 | 0 | 1 |
| Brutus | 4 | 157 | 0 | 0 | 1 | 0 |
| Caesar | 232 | 227 | 0 | 0 | 1 | 1 |
| Calpurnia | 0 | 10 | 0 | 0 | 0 | 0 |
| Cleopatra | 57 | 0 | 0 | 0 | 0 | 0 |
| mercy | 2 | 0 | 0 | 1 | 1 | 0 |
| worser | 2 | 0 | 0 | 1 | 0 | 0 |


Esistono diverse opzioni per modellare la rappresentazione dei testi:

**1. Term Frequency e Collection Frequency:**

Una prima opzione consiste nell'utilizzare direttamente la _term frequency_ (tf) e la _collection frequency_ (cf) per calcolare il peso di un termine in un documento:

$$w_{t,d}=tf_{t,d} \frac{1}{cf_{t}}$$

dove:

*  $w_{t,d}$ è il peso del termine *t* nel documento *d*
*  $tf_{t,d}$ è la frequenza del termine *t* nel documento *d*
*  $cf_{t}$ è la frequenza del termine *t* nell'intera collezione di documenti

Al momento, non ci preoccupiamo di normalizzare i valori.

**2. Funzioni Separate per Term Frequency e Collection Frequency:**

Un'altra opzione consiste nell'utilizzare due funzioni separate, *f* e *g*, per modellare l'influenza di tf e cf sul peso del termine:

$$w_{t,d}^{(t)}=f(tf_{t,d})+g(cf_{t})=tf_{t,d}+ly(cf_{t})$$

Tuttavia, in questo caso, il secondo addendo (relativo alla collection frequency) risulta troppo dominante.

#### Considerazioni sulla Lunghezza dei Documenti
Per semplicità, non facciamo assunzioni sulla lunghezza dei documenti e ci poniamo su una lunghezza media. Potrebbe esserci alternanza tra documenti lunghi e brevi.

### Smorzamento della Term Frequency
Smorzare in maniera lineare inversa la term frequency con la collection frequency appiattisce troppo verso il basso i valori.

## La Rilevanza di un Termine in un Documento

Come possiamo definire la rilevanza di un termine all'interno di un documento?

* **Tempo di Inerzia:** Un'idea potrebbe essere quella di utilizzare il tempo di inerzia del termine nel documento come indicatore di rilevanza. Tuttavia, la _term frequency_ da sola non è sufficiente.

* **Importanza di CF e DF:** Anche la _collection frequency_ (CF) e la _document frequency_ (DF) giocano un ruolo importante. La legge di Zipf ci ricorda che termini eccessivamente popolari tendono ad essere più intensi localmente.

* **Funzione di Scoring:** Per definire la rilevanza, possiamo utilizzare una funzione che tenga conto sia di TF che di CF:

	$Rilevanza(termine) = F(TF) + G(CF)$

* **Peso Relativo:** Se vogliamo dare maggior peso alla rilevanza locale, F dovrà crescere più rapidamente di G. Viceversa, se vogliamo dare maggior peso alla CF, G dovrà crescere più rapidamente di F.

* **TF vs. CF:** La _term frequency_ sarà generalmente minore della _collection frequency_, soprattutto in corpus di grandi dimensioni.

* **Lunghezza dei Documenti:** Non facciamo alcuna assunzione sulla lunghezza dei documenti nel corpus.

* **Esempio Medico:** In un corpus di documenti medici, sia CF che TF avranno un significato rilevante.

* **Problemi con la Moltiplicazione:** Moltiplicare TF e CF risulta troppo aggressivo nello smorzamento della rilevanza.

## Coerenza con la legge di Zipf e analisi delle proposte

Ci stiamo interrogando sulla coerenza di due idee (idea 1 e idea 2) con la legge di Zipf e la sua derivata. In particolare, stiamo analizzando l'importanza dei termini in un intervallo di frequenze medio-alte e medio-basse, escludendo gli estremi.

**Analisi delle proposte:**

**Caso 1: Termine nella testa della distribuzione**

- **Proposta 1:** Il peso del termine è prossimo a 0 a prescindere dalla *term frequency* del documento, dato che la *collection frequency* è molto alta.
- **Proposta 2:** Il logaritmo della *collection frequency* è alto, ma potrebbe essere smorzato. La *term frequency* potrebbe essere alta, ma non nel caso particolare di un unico documento molto lungo. Il fattore dominante è il secondo termine (logaritmo della *collection frequency*).

**Caso 2: Termine non nella testa della distribuzione**

- **Proposta 1:** La *term frequency* è divisa per la *collection frequency*.
- **Proposta 2:** La *term frequency* è sommata al logaritmo della *collection frequency*. La *term frequency* è la stessa, ma nella prima proposta è smorzata, mentre nella seconda è enfatizzata.

**Problemi:**

- Nessuna delle due proposte sembra efficace, in quanto il comportamento varia in base al termine e alle caratteristiche del documento.
- Non è possibile fare calcoli precisi senza conoscere le caratteristiche del documento.

**Considerazioni aggiuntive:**

- È importante considerare la *document frequency* al posto della *collection frequency*?
- La *document frequency* potrebbe essere più discriminante della *collection frequency*, in quanto la sua distribuzione è potenzialmente più piatta.
- L'obiettivo è valorizzare l'importanza dei termini in un documento rispetto a una query.
- La combinazione lineare non è efficace, in quanto un termine potrebbe dominare l'altro.
- La funzione reciproca lineare è troppo aggressiva, anche con la *document frequency*.

**Soluzione proposta:**

Per smorzare in modo *smooth*, si propone di utilizzare la seguente formula:


$\frac{1}{log(document frequency)}$


**Spiegazione:**

- Se la *document frequency* è prossima a *n* (numero totale di documenti), significa che il termine appare in quasi tutti i documenti.
- In questo caso, il peso del termine sarà basso, in quanto poco discriminante. 

## Smorzamento della Term Frequency e Inverse Document Frequency (IDF)

Importanza dello smorzamento della term frequency (tf) nel calcolo dell'informazione veicolata da un termine all'interno di un documento:
* Numeri maggiori di 1 non sono un problema, poiché la document frequency massima di un termine è pari al numero di documenti nel corpus (n). 
* La tf viene divisa per il logaritmo di n, ottenendo uno smorzamento.
* Per termini rari, presenti in pochi documenti, lo smorzamento è maggiore. Ad esempio, se un termine compare in 3 documenti, la tf viene divisa per il logaritmo in base 2 di 3.

## Funzione TF-IDF

La funzione TF-IDF (Term Frequency - Inverse Document Frequency) è una misura statistica che valuta l'importanza di un termine all'interno di un documento, in relazione ad una collezione di documenti (corpus).  Combina la frequenza di un termine in un documento con la sua rarità nel corpus. 

La formula per calcolare il peso TF-IDF di un termine *t* nel documento *d* è:

$$w_{t,d}=\log(1+tf_{t,d}) \times \log_{10}\left( \frac{N}{df_{t}} \right)$$

Dove:

* **tf<sub>t,d</sub>**: Frequenza del termine *t* nel documento *d*. Misura quanto frequentemente un termine appare in un documento specifico.
* **N**: Numero totale di documenti nel corpus.
* **df<sub>t</sub>**: Numero di documenti in cui il termine *t* compare. Indica in quanti documenti del corpus appare un determinato termine.

**Interpretazione della formula:**

* **log(1+tf<sub>t,d</sub>):**  Rappresenta la frequenza del termine nel documento. Il logaritmo smorza l'influenza dei termini molto frequenti in un documento.
* **log<sub>10</sub>(N/df<sub>t</sub>):** Rappresenta la frequenza inversa del documento (IDF).  Un valore di IDF elevato indica che il termine è raro nel corpus, quindi più informativo. Viceversa, un IDF basso indica un termine comune e poco informativo.

**Vantaggi dell'utilizzo di TF-IDF:**

* **Penalizza i termini comuni:** Termini frequenti in molti documenti (articoli, stop words) avranno un peso TF-IDF basso.
* **Evidenzia i termini rari:** Termini che compaiono in pochi documenti avranno un peso TF-IDF alto, evidenziando la loro importanza per quei documenti specifici.
* **Bilancia frequenza locale e globale:** TF-IDF considera sia la frequenza del termine in un documento specifico che la sua rarità nel corpus, fornendo una misura più accurata dell'importanza del termine.

**Considerazioni importanti nell'utilizzo di TF-IDF:**

* **Rimozione delle stop words:** È fondamentale rimuovere le stop words ("il", "la", "che", etc.) prima di calcolare TF-IDF, poiché non forniscono informazioni utili.
* **Stemming e lemmatization:** Applicare tecniche di stemming (riduzione di una parola alla sua radice) e lemmatization (riduzione di una parola al suo lemma) può migliorare la precisione del calcolo TF-IDF, raggruppando parole con lo stesso significato.
* **Soglie di taglio:** È possibile impostare soglie per escludere termini con frequenza troppo alta o troppo bassa, evitando che influenzino eccessivamente il calcolo.

**Smorzamento e Legge di Zipf:**

Lo smorzamento logaritmico nella formula TF-IDF aiuta a gestire la distribuzione dei termini descritta dalla legge di Zipf, che afferma che la frequenza di una parola è inversamente proporzionale al suo rango nella lista di frequenza. Lo smorzamento riduce l'influenza dei termini molto frequenti e aumenta l'importanza di quelli meno frequenti.

**Vantaggi dello smorzamento:**

* Evita di definire soglie di taglio arbitrarie per il vocabolario.
* Permette di lavorare con matrici TF-IDF sparse, gestendo i valori prossimi allo zero.

**Doppio logaritmo:**

In caso di corpus molto grandi, si può utilizzare un doppio logaritmo per smorzare ulteriormente il peso del fattore di frequenza del termine (TF).

**Normalizzazione e calcolo della similarità:**

La normalizzazione dei vettori TF-IDF e il calcolo della similarità tra di essi sono aspetti cruciali per utilizzare questa metrica in compiti come il recupero delle informazioni o la classificazione dei documenti. 

## Modello Bag-of-words (BoW)

Il modello Bag-of-words (BoW) si basa sull'ipotesi di indipendenza dei termini. 
**Concetto alla base della Term Frequency:** bisogna tener conto del contesto globale. 
Quando dobbiamo capire se una feature è significativa, dovremmo misurare quantitativamente quanto è il suo potere caratterizzante e discriminante.

* L'ordinamento delle parole in un documento viene scartato.
    * In un certo senso, un passo indietro rispetto all'indice posizionale.

**Contro:**

* Informazioni sintattiche mancanti (ad esempio, struttura frasale, ordine delle parole, informazioni di prossimità).
* Informazioni semantiche mancanti (ad esempio, senso delle parole).
* Manca il controllo di un modello booleano (ad esempio, richiedere che un termine appaia in un documento).
    * Data una query a due termini "A B", si potrebbe preferire un documento che contiene A frequentemente ma non B, rispetto a un documento che contiene sia A che B, ma entrambi meno frequentemente.

**Pro:**

* Fornisce una corrispondenza parziale e una misura naturale di punteggi/classifica - non più booleana.
* Tende a funzionare abbastanza bene nella pratica nonostante le ipotesi semplificative.
* Consente un'implementazione efficiente per grandi collezioni di documenti.
* La query diventa un vettore nello stesso spazio dei documenti -> Modello dello spazio vettoriale. 

## Tipi di frequenza

Vogliamo usare la frequenza del termine (tf) quando calcoliamo i punteggi di corrispondenza query-documento. Ma come?

* La frequenza grezza del termine non è ciò che vogliamo:
    * Un documento con 10 occorrenze del termine è più rilevante di un documento con 1 occorrenza del termine.
    * Ma non 10 volte più rilevante.
* La rilevanza non aumenta proporzionalmente alla frequenza del termine.

### Peso di frequenza logaritmica del termine
$$w_{t,d}
\begin{cases}
1+\log_{10}\text{tf}_{td} \ \text{  if tf}_{td} \ >0 \\
0,\ \text{otherwise}
\end{cases}$$ 

Il peso di frequenza logaritmica del termine *t* in *d*:

* 0 → 0, 1 → 1, 2 → 1.3, 10 → 2, 1000 → 4, ecc.

Punteggio per una coppia documento-query: somma sui termini *t* sia in *q* che in *d*:
$$\sum_{t\in q \cap d}(1+\log(tf_{t,d}))$$
* Il punteggio è 0 se nessuno dei termini di query è presente nel documento. 

## Frequenza inversa del documento (idf)

I termini rari sono più informativi dei termini frequenti.

* Ricorda le stop words.
* Considera un termine nella query che è raro nella collezione: ad esempio, "arachnocentrico".
* Un documento che contiene questo termine è molto probabilmente rilevante per la query.

"arachnocentrico"

Pertanto, più raro è il termine, maggiore è il suo peso.

**Frequenza della collezione (cf) vs. Frequenza del documento (df)**
Si sceglie la document frequency rispetto alla collection frequency perché quest'ultima rischia di avere correlazione con la maggior parte della tf di quel termine, ma nella pratica l'informazione di document frequency ci aiuta maggiormente nel discriminare tra un documento e un altro.

* Quale è meglio per la ricerca?
* Ad esempio, "assicurazione": cf=10440, df=3997.
* Ad esempio, "prova": cf=10422, df=8760.

**Frequenza inversa del documento**

La frequenza del documento di un termine *t*:

* Una misura inversa dell'informatività di *t*.

Definisci l'idf (frequenza inversa del documento) di *t* come:
$idf_{t}=\log_{10}\left( \frac{N}{df_{t}} \right)$

* log (N/dft) invece di N/dft
per "smorzare" l'effetto dell'idf.

Nota che:

* La df di un termine è unica.
* Influisce sulla classificazione dei documenti solo per le query a *k* termini (*k*<1).

## Frequenza del termine - frequenza inversa del documento (tf-idf)

Il metodo tf-idf (term frequency-inverse document frequency) assegna un peso ai termini in un documento in base alla loro frequenza nel documento stesso e alla loro rarità nell'intero corpus di documenti. 

**Concretamente:**

* **tf (term frequency):** Maggiore è la frequenza di un termine in un documento, maggiore è il suo peso.
* **idf (inverse document frequency):** Maggiore è la rarità di un termine nell'intero corpus, maggiore è il suo peso.

Questo schema di ponderazione è in linea con la legge di distribuzione della probabilità, che prevede una distribuzione di tipo power-law nella frequenza dei termini. In altre parole, i termini comuni hanno una bassa probabilità di apparire, mentre i termini rari hanno una probabilità maggiore.

Il metodo tf-idf, quindi, **riduce il peso dei termini comuni e aumenta il peso dei termini rari**, riflettendo la distribuzione di probabilità osservata nella realtà. 


Il peso tf-idf di un termine è il prodotto del suo peso tf e del suo peso idf:

$w_{t,d}=\log(1+tf_{t,d})\times\log_{10}\left( \frac{N}{df_{t}} \right)$

* **Aumenta con il numero di occorrenze all'interno di un documento.**
* **Aumenta con la rarità del termine nella collezione.**

Il punteggio per una coppia documento-query è la somma sui termini *t* sia in *q* che in *d*. 


## Varianti di Tf-Idf

$$
\frac{tf_{i,d}}{\max_{j}tf_{j,d}} ,\
\frac{tf_{id}}{\sqrt{ \sum_{j}(tf_{j,d})^2 }} ,\
\frac{tf_{id} \cdot idf_{i}}{\sqrt{ \sum_{j}(tf_{j,d} \cdot idf_{j})^2 }} 
$$

Esistono diverse varianti di questa tecnica, che si differenziano principalmente per il modo in cui viene calcolato il termine "tf" (frequenza del termine) e se i termini nella query sono anche ponderati.

**Calcolo del termine "tf":**

* **Con logaritmi:** Il logaritmo della frequenza del termine viene utilizzato per attenuare l'influenza dei termini che compaiono molto frequentemente in un documento.
* **Senza logaritmi:** La frequenza del termine viene utilizzata direttamente, senza alcuna trasformazione.

**Ponderazione dei termini nella query:**

* **Ponderati:** I termini nella query vengono ponderati in base alla loro importanza, ad esempio utilizzando la loro frequenza nella query stessa.
* **Non ponderati:** Tutti i termini nella query hanno lo stesso peso.

**Assunzioni:**
* La collezione di documenti è omogenea in termini di dominio dei dati, ovvero il dominio è fissato e il lessico è comune a tutti i documenti.
* I pattern di frequenza dei termini sono molto simili tra i documenti.

**Principi chiave:**

* **Peso variabile:** Il peso di uno stesso termine cambia a seconda del documento su cui appare. Un termine che compare frequentemente in un documento avrà un peso elevato, mentre un termine raro avrà un peso basso.
* **Normalizzazione della lunghezza:** I documenti hanno dimensioni diverse. Per compensare le variazioni di lunghezza, si può applicare una normalizzazione che tiene conto della lunghezza del documento.
* **Smoothing:** Per evitare che i termini rari abbiano un peso eccessivo, si può applicare uno smoothing che attenua l'influenza dei termini che compaiono poche volte.

## Normalizzazione della lunghezza

**Obiettivo:** Capire l'impatto della normalizzazione dei vettori di parole sulla rappresentazione dei topic.

**Problema:** La normalizzazione dei vettori di parole, in particolare la divisione per la norma L2, può influenzare la rappresentazione dei topic. 

**Considerazioni:**
* **Diluisce il segnale informativo:** La normalizzazione può diluire il segnale informativo dei topic, soprattutto quando si passa da un abstract breve a un testo lungo.
* **Proprietà geometriche:** La normalizzazione L2 ha proprietà geometriche specifiche che possono differire da altri metodi di normalizzazione, come la normalizzazione al massimo.
* **Influenza sulla costruzione della matrice dei dati:** La normalizzazione L2 interviene direttamente nella costruzione della matrice dei dati, modificando la rappresentazione dei vettori di parole.

**Esempio:**
Se si utilizza la matrice di peso Tf-Idf senza normalizzazione e la collezione di documenti contiene un mix di documenti lunghi e brevi, la distanza euclidea tenderà a favorire i documenti più lunghi.

## Matrice di peso Tf.Idf
![[1) Intro-20241014151203327.png]]

## Documenti e query come vettori

Ci viene dato uno spazio vettoriale a |V| dimensioni.

* I termini sono gli assi dello spazio.
* I documenti sono punti o vettori in questo spazio.

Questo spazio è:

* **Molto alto-dimensionale:** Decine di milioni di dimensioni quando si applica questo a un motore di ricerca web.
* **Molto sparso:** La maggior parte delle voci è zero.

Possiamo rappresentare anche le query come vettori nello spazio.

Classifichiamo i documenti in base alla loro prossimità alla query in questo spazio.

* **Prossimità = similarità dei vettori.**
* **Prossimità ≈ inversa della distanza.**

## Prossimità dello spazio vettoriale
![[1) Intro-20241014153843176.png]]

Potremmo decidere di utilizzare una misura di prossimità che sia **scale-invariant**, ovvero indipendente dalla lunghezza dei vettori. 
Invece di calcolare la distanza euclidea tra due vettori, possiamo calcolare la prossimità in termini di **angolo**.

Questo approccio comporta una certa **normalizzazione implicita**.

### Perché la Distanza Euclidea non è una Buona Idea

Le distanze di Minkowski soffrono maggiormente l'alta dimensionalità rispetto a misure di correlazione o similarità, come il coseno. 
 La distanza euclidea, inoltre, ha i seguenti problemi:
* **Sensibilità alla Lunghezza:** La distanza euclidea è grande per vettori di lunghezze diverse. Ad esempio, la distanza euclidea tra una query `q` e un documento `d2` è grande anche se la distribuzione dei termini in `q` e `d2` è molto simile.
* **Controesempio Principale:** La distanza euclidea è molto grande tra un documento e lo stesso concatenato con se stesso.

I documenti lunghi sarebbero più simili tra loro in virtù della lunghezza, non dell'argomento.

### Normalizzazione Implicita con l'Angolo

Possiamo normalizzare implicitamente guardando gli angoli.
* **Idea chiave:** Classificare i documenti in base all'angolo con la query.

Classifichiamo i documenti in ordine decrescente dell'angolo tra query e documento. In alternativa, classifichiamo i documenti in ordine crescente di **coseno(query, documento)**.
* Il coseno è una funzione monotona decrescente per l'intervallo [0°, 180°].

La misura del **coseno** tra due vettori multidimensionali si ottiene dal prodotto scalare tra i due vettori diviso per il prodotto delle loro norme. Se i vettori fossero inclusi in una sfera di raggio unitario, non sarebbe necessario normalizzarli.

### Normalizzazione dei Vettori

Un vettore può essere normalizzato (in lunghezza) dividendo ciascuna delle sue componenti per la sua lunghezza.

* Dividere un vettore per la sua norma L2 lo rende un vettore unitario (di lunghezza) (sulla superficie dell'ipersfera unitaria).
* I documenti lunghi e brevi hanno ora pesi comparabili. 

$$
\text{sim}(d_1, d_2) = \frac{d_1 \cdot d_2}{\|d_1\| \cdot \|d_2\|} = \frac{\sum_{i=1}^{n} w_{i,j} \cdot w_{i,k}}{\sqrt{\sum_{i=1}^{n} w_{i,j}^2} \cdot \sqrt{\sum_{i=1}^{n} w_{i,k}^2}}
$$

![[1) Intro-20241123122258468.png]]
riduciamo a vettori di lunghezza unitaria

Similarità del coseno = prodotto interno normalizzato.

---

Quanto sono simili i romanzi:

* SaS: Ragione e Sentimento
* PaP: Orgoglio e Pregiudizio
* WH: Cime Tempestose?



| term      | Sas  | PaP  | **WH** |
| --------- | ---- | ---- | ------ |
| affection | 3.06 | 2.76 | 2.30   |
| jealous   | 2.00 | 1.85 | 2.04   |
| gossip    | 1.30 | 0    | 1.78   |
| wuthering | 0    | 2.58 | 2.58   |

| term      | Sas   | PaP   | WH    |
| --------- | ----- | ----- | ----- |
| affection | 0.789 | 0.832 | 0-524 |
| jealous   | 0.515 | 0.555 | 0.465 |
| gossip    | 0,335 | 0     | 0.405 |
| wuthering | 0     | 0     | 0.588 |


| Log frequency weighting<br> |        |
| --------------------------- | ------ |
| dot(SaS,PaP)                | ≈ 12.1 |
| dot(SaS,WH)                 | ≈ 13.4 |
| dot(PaP,WH)                 | ≈ 10.1 |


| After length normalization |        |     |
| -------------------------- | ------ | --- |
| cos(SaS,PaP)               | ≈ 0.94 |     |
| cos(SaS,WH)                | ≈ 0.79 |     |
| cos(PaP,WH)                | ≈ 0.69 |     |


---

**Vector space proximity**

**CosineScore(q)**

6. float Scores[N] = 0
7. float Length[N]
8. for each query term t
9. do calculate w<sub>t,q</sub> and fetch postings list for t
	1. for each pair(d, tf<sub>t,d</sub>, w<sub>t,q</sub>) in postings list
	2. do Scores[d] += w<sub>t,d</sub> × w<sub>t,q</sub>
10. Read the array Length
11. for each d
12. do Scores[d] = Scores[d]/Length[d]
13. return Top K components of Scores

## Varianti di ponderazione Tf-Idf

**Term Frequency**

*   **n (natural)**: $tf_{r, d}$
*   **l (logarithm)**: $1 + \log(tf_{r, d})$
*   **a (augmented)**: $0.5 + \frac{0.5 \cdot tf_{r, d}}{\max_{r} (tf_{r, d})}$
*   **b (boolean)**: $\begin{cases} 1 & \text{if } tf_{r, d} > 0 \\ 0 & \text{otherwise} \end{cases}$

**Document Frequency**

*   **n (no)**: $1$
*   **t (idf)**: $\log \frac{N}{df_r}$
*   **p (prob idf)**: $\max \{ 0, \log \frac{N - df_r}{df_r} \}$

**Normalization**

*   **n (none)**: $1$
*   **c (cosine)**: $\frac{1}{\sqrt{w_1^2 + w_2^2 + \dots + w_n^2}}$
*   **u (pivoted unique)**: $\frac{1}{u}$
*   **b (byte size)**: $\frac{1}{\text{CharLength}^{\alpha}}, \alpha < 1$


- **Soluzioni di default:** Le soluzioni di default in alcuni sistemi di retrieval sono le seconde di ogni tipo
- **Term frequency aumentata:** La versione aumentata della term frequency è interessante in contesti di retrieval puro, dove si confrontano query espanse con documenti.
- **Smoothing:** La term frequency aumentata prevede una sorta di smoothing, simile a quello che si incontra nel retrieval probabilistico.

## Varianti di ponderazione Tf-Idf

Molti motori di ricerca consentono ponderazioni diverse per le query rispetto ai documenti.

**Notazione SMART:** indica la combinazione in uso in un motore, con la notazione ddd.qqq, usando gli acronimi della tabella precedente.

Uno schema di ponderazione molto standard è: lnc.ltc

* **Documento:** tf logaritmico, nessun idf, normalizzazione del coseno.
* **Query:** tf logaritmico, idf, normalizzazione del coseno.

| Termine   | Query | Documento | Prod  |     |        |        |       |     |        |      |      |
| --------- | ----- | --------- | ----- | --- | ------ | ------ | ----- | --- | ------ | ---- | ---- |
| tf-raw    | tf-wt | df        | idf   | wt  | n'lize | tf-raw | tf-wt | wt  | n'lize |      |      |
| auto      | 0     | 0         | 5000  | 2.3 | 0      | 0      | 1     | 1   | 1      | 0.52 | 0    |
| best      | 1     | 1         | 50000 | 1.3 | 1.3    | 0.34   | 0     | 0   | 0      | 0    | 0    |
| car       | 1     | 1         | 10000 | 2.0 | 2.0    | 0.52   | 1     | 1   | 1      | 0.52 | 0.27 |
| insurance | 1     | 1         | 1000  | 3.0 | 3.0    | 0.78   | 2     | 1.3 | 1.3    | 0.68 | 0.53 |

**Documento:** assicurazione auto assicurazione auto
**Query:** migliore assicurazione auto

**Punteggio = 0 + 0 + 0.27 + 0.53 = 0.8**

**Lunghezza del documento =**

## Classifica dello spazio vettoriale

**Riepilogo:**

* Rappresentare ogni documento e query come un vettore tf-idf ponderato.
* Calcolare il punteggio di similarità del coseno per il vettore di query e ogni vettore di documento.
* Classificare i documenti rispetto alla query in base al punteggio.
* Restituire i primi K all'utente.

**Pro:**

* Fornisce una corrispondenza parziale e una misura naturale di punteggi/classifica.
* Funziona abbastanza bene nella pratica nonostante le ipotesi semplificative.
* Implementazione efficiente.

**Contro:**

* Informazioni sintattiche mancanti.
* Informazioni semantiche mancanti.
* Ipotesi di indipendenza dei termini (BoW).
* Ipotesi che i vettori dei termini siano ortogonali a coppie.
* Manca il controllo di un modello booleano (ad esempio, richiedere che un termine appaia in un documento).

