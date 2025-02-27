
## Schema Riassuntivo: Linguaggio Naturale, Recupero Informazioni e NLP

**I. Linguaggio Naturale e sue Applicazioni**

* **Unicità:** Linguaggio specifico per umani, gestione complessità comunicativa.
* **Interazione Multimodale:** Integrazione con altre modalità sensoriali (es. visione).
* **Recupero Informazioni (IR) e Elaborazione Linguaggio Naturale (NLP):**
 * **NLP:** Sviluppo di sistemi automatici per comprensione e generazione di linguaggio naturale.
 * **IR:** Recupero (semi-)automatico di dati testuali non strutturati da grandi collezioni.

**II. Definizione di IR**

* **Enunciato del Problema:**
 * **Dato:** Collezione di documenti (statica).
 * **Query:** Termine di ricerca, stringa, frase, espressione stilizzata.
 * **Modello di Recupero:** Meccanismo per determinare corrispondenza documento-query.
 * **Obiettivo:** Recuperare documenti rilevanti per il bisogno informativo dell'utente.
 * **Risultati:** Insieme di documenti recuperati.

**III. Problemi con i Dati Testuali**

* **Sfide Generali:** Grandi set di dati, alta dimensionalità, dati rumorosi, dati in continua evoluzione, comprensibilità dei risultati.
* **Sfide Specifiche:**
 * **Progettazione:** Testo non progettato per l'uso da parte dei computer.
 * **Struttura e Semantica:** Complessità e ambiguità a livello linguistico (morfologia, sintassi, semantica, pragmatica).
 * **Linguaggi:** Gestione di linguaggi generali e specifici di dominio.
 * **Multilinguismo:** Gestione di lingue diverse.

**IV. IR vs. Altre Discipline**

* **Obiettivo Comune:** Facilitare la ricerca di informazioni (es. sul Web).
* **Attività Comuni:** SEO, crawling, estrazione di documenti.
* **Differenze:**
 * **Web Scraping:** Estrazione di informazioni.
 * **Ricerca di Schemi:** NLP, Linguistica computazionale.
 * **Scoperta di Informazioni Sconosciute:** NLP, estrazione di testo.
 * **Discriminazione del Testo:** NLP (Machine Learning).
 * **Comprensione del Testo:** NLP (Machine Learning, AI generativa).

**V. Topic Detection and Tracking**

* **Contesto:** Fine anni '90 (inizio text mining).
* **Definizione:** Acquisizione automatica di dati in streaming (principalmente news).
* **Importanza:** Rilevante prima dell'avvento dei big data.
* **Caratteristiche:** Analisi incrementale, supervisionato o non supervisionato.
* **Difficoltà:** Task incrementale, risorse computazionali limitate, definizione di "topic", tracciamento del segnale.
* **Modellazione di Topic Stocastica:** Utilizzo di modelli probabilistici (es. LDA) per identificare temi principali nei documenti.

**VI. Relation Extraction**

* **Relazione con NER:** Necessità di identificare prima le entità nominate.
* **Pattern Frequenti:** Ricerca di pattern ricorrenti.
* **Definizione in NLP:** Identificazione di relazioni lessicali tra entità nominate.

**VII. Summarization**

* **Approccio Tradizionale (anni '90):** Estrazione di parole chiave.
* **Guida alla Summarization:** Utilizzo delle proprietà del documento per guidare il processo.

**VIII. KDD Pipeline in NLP**

* **Fasi:** Sequenza di fasi (unfolding).
* **Rappresentazione:** Indicizzazione degli elementi costitutivi.
* **Apprendimento:** A valle del set di risultati.
* **Feature Selection:** Utilizzo del machine learning.
* **Valutazione:** Criteri statistici (es. accuracy, con i suoi limiti).

**IX. Funzioni Principali di NLP**

* **Funzione Principale:** Estrazione di informazioni da testi.
* **Macro Categorie:**
 * **Indicativa:** Rivelazione di elementi per determinare la rilevanza del testo rispetto alle query (fondamentale per esplorazione e retrieval).
 * **Informativa:** Ottenimento di un surrogato del testo senza riferimento all'originale (valido anche per il retrieval).

**X. Browsing**

* **Definizione:** Navigazione in sistemi di ipertesto e ipermedia.
* **Scopo:** Esplorazione di collezioni di testo.
* **Vantaggi:** Non richiede descrizione precisa del bisogno informativo.
* **Utilizzo:** Utenti con bisogni poco chiari o inespressi.
* **Supporto all'Annotazione:** Possibile supporto all'annotazione.
* **Valutazioni e Gold Standard:** Utilizzo di silver standard (es. GPT-4) in assenza di gold standard generato da esperti.

**XI. Estrazione di Informazioni**

* **Contesto:** Task fondamentale nell'analisi di informazioni sul web (insieme a classificazione e clustering).
* **Esempio:** Web wrapping.
* **Definizione:** Recupero di informazioni specifiche da documenti già identificati come rilevanti.
* **Esempi:** Utilizzo di schemi pre-esistenti nel retrieval sul web.

---

**I. E-commerce e Estrazione di Informazioni**

* **A. Estrazione di Informazioni da Documenti:**
 * Identificazione di documenti rilevanti.
 * Estrazione di informazioni specifiche all'interno di documenti rilevanti.
 * Facilitazione tramite schemi o database pre-esistenti.
* **B. Tipi di Template di Estrazione:**
 * **Slot:** Riempiti da sottostringhe del documento.
 * **Riempitivi Pre-specificati:** Set fisso di riempitivi, indipendentemente dal testo.
 * **Multipli Riempitivi:** Più filler per slot.
 * **Ordine Fisso:** Slot in ordine prestabilito.
* **C. Modelli di Estrazione:**
 * Specifica di elementi tramite regex o altri pattern.
 * Modelli precedenti (pre-filler) e successivi (post-filler) per il contesto.
 * Esempio: Estrazione di termini da un modello predefinito con pattern per ogni slot.

**II. Recupero di Documenti**

* **A. Processo:**
 * Ricezione di query in linguaggio naturale.
 * Classificazione dei documenti per rilevanza.
 * Abbinamento tra rappresentazione del documento e della query.
 * Restituzione di documenti pertinenti.
* **B. Modelli di Recupero:** Booleano, Spazio Vettoriale, Probabilistico, ecc.
* **C. Differenze tra Modelli:**
 * Rappresentazione dei contenuti testuali.
 * Rappresentazione dei bisogni informativi.
 * Metodo di abbinamento.

**III. Applicazioni di Base**

* **A. Scoperta della Conoscenza:**
 * **Estrazione di Informazioni:** Estrazione di informazioni utili da grandi quantità di dati.
 * **Distillazione delle Informazioni:** Estrazione basata su struttura predefinita per identificare documenti rilevanti.
* **B. Utilizzi Tipici:**
 * Estrazione di informazioni rilevanti.
 * Classificazione e gestione di documenti.
 * Organizzazione di repository di meta-informazioni.
 * Categorizzazione gerarchica.
 * Riassunto del testo.
 * Disambiguazione del senso delle parole.
 * Filtraggio del testo (o dell'informazione).

**IV. Web Personalization e Applicazioni Avanzate**

* **A. Web Personalization:**
 * Selezione di informazioni per gruppi target.
 * Approcci Collaborative based (raccomandazioni basate su profili simili).
 * Gestione utenti "cold start" (più popolari).
* **B. Altri Esempi di Applicazioni:**
 * CRM e marketing (cross-selling, raccomandazioni).
 * Raccomandazione di prodotti.
 * Consegna di informazioni nelle organizzazioni.
 * Personalizzazione dell'accesso alle informazioni.
 * Filtraggio di notizie.
 * Rilevamento di messaggi spam.

**V. Categorizzazione Gerarchica**

* **A. Processo:** Navigazione gerarchica per restringere la ricerca.
* **B. Caratteristiche:**
 * Natura ipertestuale dei documenti (analisi degli hyperlink).
 * Struttura gerarchica delle categorie.
 * Decomposizione della classificazione come decisione ramificata.

**VI. Riassunto del Testo**

* **A. Obiettivi:** Testo breve, coerente, informazioni importanti in campi semanticamente ben definiti.
* **B. Approcci:** Dipende dalla natura dei documenti (es. summarization di recensioni per aspetti).
* **C. Applicazioni:** Facilitazione dell'accesso alle informazioni (estrazione parole chiave, astrazione di documenti, riassunto di documenti recuperati).
* **D. Approcci basati sulla dimensione dell'unità di testo:** Riassunti di parole chiave, riassunti di frasi.

**VII. Disambiguazione del Senso delle Parole (WSD)**

* **A. Obiettivo:** Assegnare il senso corretto di una parola nel contesto.
* **B. Approccio Efficace:** Sfruttare misure di correlazione semantica da un inventario di sensi esistente.
* **C. Esempio:** Disambiguazione del termine "bank".
* **D. Approcci Knowledge-based:** Sfruttamento di risorse linguistiche elettroniche (es. Wikipedia).

**VIII. Filtraggio nel Text Mining**

* **A. Definizione:** Classificazione di documenti come rilevanti o irrilevanti per un consumatore.
* **B. Esempio:** Feed di notizie (produttore: agenzia stampa; consumatore: giornale).
* **C. Caratteristiche:** Text Classification con singola etichetta; implementazione lato produttore o consumatore; creazione di profili utente.
* **D. Filtraggio Adattivo:** Aggiornamento del profilo basato sul feedback utente.
* **E. Filtraggio - CRM:** Aiutare le aziende a comprendere le opinioni dei clienti.

---

# Schema Riassuntivo del Testo sull'Analisi del Testo

**I. Elaborazione del Feedback:**

* **1. Standardizzazione:** Conversione dei dati di feedback in un formato uniforme.
* **2. Raggruppamento:** Raggruppamento dei dati standardizzati in base alla similarità del contenuto.
* **3. Assegnazione:** Assegnazione di nuovi feedback alle categorie predefinite.

**II. Raccomandazione di Prodotti:**

* **Approcci:**
 * **Basato sul contenuto:** Analisi dei dati utente (categorie preferite, autori) per suggerire prodotti simili.
 * **Collaborativo:** Suggerimento di prodotti acquistati da utenti con profili simili, basandosi su cronologia e valutazioni.
* **Tendenza attuale:** Combinazione di entrambi gli approcci per migliorare l'accuratezza.

**III. Rilevamento dello Spam:**

* **Applicazione:** Classificazione di email come spam o legittime tramite text mining.
* **Sfide:**
 * Valutazione del modello complessa a causa dei costi asimmetrici degli errori (falsi positivi vs. falsi negativi).
 * Distribuzione non uniforme delle classi (più spam che email legittime) complica l'apprendimento del classificatore.

**IV. Vocabolario (Modello di Rappresentazione dei Testi):**

* **Definizione di Termine:** Quali termini indicizzare? (tutte le parole o solo quelle "importanti"?; considerare presenza o frequenza?)
* **Contesto:** Selezione dei termini per l'indicizzazione (es. indice analitico).
* **Risposta:** Creazione di pattern relazionali tra parole; importanza delle parole varia tra concetti.
* **Cosa è un termine?:** Parola, coppia di parole, frase, radice, n-gramma, tipo di parola.
* **Modellazione delle relazioni tra termini:** Paragrafo, frase, ordine delle parole (complicano i modelli), strutture sintattiche complesse, relazioni semantiche (is-a, part-of).
* **Ricerca di termini che catturano la semantica del testo:** Evitando elaborazione manuale intensa.
* **Altri argomenti:** Analisi lessicale e morfologica, elaborazione di punteggiatura, numeri, stopwords, stemming, lemmatizzazione, tagging delle parti del discorso, analisi della semantica del discorso (anafora, ellissi, meronomia), pragmatica, semiotica, morfologia.

**V. Tokenizzazione: Problemi Lessicali e Morfologici:**

* **Token:** Termine candidato.
* **Organizzazione del testo:** Fase di pre-processing.
* **Punteggiatura:** Gestione di trattini (es. *state-of-the-art*), apostrofi (es. *Italy’s*), acronimi (es. *U.S.A*), entità nominate (es. *San Francisco*).
* **Numeri:** Rimozione o mantenimento (motivi: dati informativi, codici di errore, intervalli di tempo); metadati indicizzati separatamente; varietà di tipi (date, tempo, codici, identificatori, numeri di telefono). Case delle lettere convertito in lowercase.

**VI. Stopwords:**

* **Definizione:** Parole grammaticali con scarso potere informativo, specifiche della lingua.
* **Principio cardine:** Rimozione di colonne con distribuzione super omogenea o eterogenea. Trade-off tra generalizzazione e omogeneità.
* **Motivi per la rimozione:** Poco contenuto semantico, parole più comuni, impatto sulla dimensionalità (sparsità $\to$ perdita di potere discriminante).
* **Casi in cui sono necessarie:** Query di frase, titoli, query "relazionali".
* **Gestione:** Buone tecniche di compressione e ottimizzazione delle query.

**VII. Personalizzazione della Stop-List:**

* **Obiettivo:** Ridurre e arricchire la stop-list.
* **Come:** Rimozione di termini comuni al corpus (poco esplicativi per la distinzione tra documenti); mantenimento di termini frequenti per analisi individuale; esclusione di termini super rari (più delicata della rimozione dei termini comuni); termini che appaiono una sola volta nel corpus (decisione dipendente dal tipo di termine).
* **Conclusione:** La stop-list deve contenere termini molto frequenti nella collezione.

**VIII. Normalizzazione:**

* **Obiettivo:** Uniformizzare le parole nel testo indicizzato e nelle query.
* **Esempi:** Corrispondenza tra "U.S.A." e "USA".
* **Risultato:** Termine normalizzato, voce nel dizionario IR, classi di equivalenza (eliminazione di punti, trattini, accenti, umlaut).
* **Focalizzazione sulla sintassi:** Sacrificio di parte dell'informazione per una rappresentazione focalizzata sulla sintassi e non sulla semantica.
* **Espansione asimmetrica:** Alternativa alle classi di equivalenza (es. inserire "window", ricerca "window", "windows"). Potenzialmente più potente, ma meno efficiente.

---

**I. Preprocessing del Testo**

* **A. Uso della Lingua:**
 * Normalizzazione a termini senza accenti/trattini per uniformità, nonostante le variazioni linguistiche.
* **B. Tokenizzazione e Normalizzazione:**
 * Dipendono dalla lingua (es. formati data: "3/10/2000" vs. "10/3/2000").
* **C. Case Folding:**
 * Riduzione a minuscolo.
 * Eccezioni per Named Entity (es. General Motors, Fed vs. fed). Meglio tutto minuscolo per uniformità con le query utente.
* **D. Riduzione Matrice Dati (Algebra Lineare):**
 * Post-processing per ridurre sinonimi a una sola dimensione, focalizzandosi su pattern sintattici.
* **E. Spell Checking:**
 * Gestire la prossimità tra stringhe (edit distance), ma limitato dal contesto lessicale.
 * Meglio evitarlo se non necessario, soprattutto con corpus rumorosi (es. messaggi istantanei).
* **F. Emoticon:**
 * Sostituire con termini o markup per preservare l'informazione.
 * Pesatura delegata al modulo successivo. Importanti per analisi del sentiment.
* **G. Tesauri:**
 * Generalizzare termini correlati ma superficialmente diversi (es. car = automobile).
 * Creazione di classi di equivalenza (manuali o automatiche). Espansione query o indicizzazione multipla.
 * Gestione di sinonimi e omonimi (es. Soundex per errori ortografici).
* **H. Lemmatizzazione:**
 * Riduzione a forme base (lemma) per verbi e sostantivi.
 * Effetto secondario dell'analisi sintattica, beneficio modesto per il retrieval.
* **I. Stemming:**
 * Riduzione a radici (prefisso). Aggressivo, distrugge il lessico. Usare solo se non si vuole elaborazione semantica.
 * Dipendente dalla lingua. Combinazione con rimozione stop words.
 * **I.1. Stemming di Porter:**
 * Algoritmo iterativo basato su regole (longest match).
 * Convenzioni: v: vocale, c: consonante, (vc)m: (vocale-consonante) ripetuto m volte.
 * 5 fasi di riduzione. Esempio: GENERALIZATIONS -> GENER.
* **J. Confronto Stemming/Lemmatizzazione:** Lo stemming è più aggressivo della lemmatizzazione.

**II. Modello Booleano**

* **A. Query Booleana:**
 * Combinazione di termini di indice con operatori booleani (AND, OR, NOT).
 * Corrispondenza binaria (documento corrisponde o meno).

---

**Applicazioni del Modello Booleano**

* **Sistemi di Ricerca Esistenti:**
 * Email
 * Cataloghi di biblioteche
 * macOS Spotlight
 * WestLaw (esempio principale):
 * Servizio di ricerca legale commerciale (iniziato nel 1975)
 * Decine di terabyte di dati, ~700.000 utenti
 * Utilizzo prevalente di query booleane. Esempi:
 * "LIMIT! /3 STATUTE ACTION /S FEDERAL /2 TORT /3 CLAIM" (termine di prescrizione nel Federal Tort Claims Act)
 * "disabl! /p access! /s work-site work-place (employment /3 place" (accesso al luogo di lavoro per disabili)
 * `/3` = entro 3 parole, `/S` = nella stessa frase.

**Vantaggi del Modello Booleano**

* Query lunghe e precise
* Operatori di prossimità (es. `/3`, `/S`)
* Sviluppo incrementale delle query
* Maggiore precisione e controllo rispetto alla ricerca web

**Limitazioni del Modello Booleano**

* **Rigidità:** AND = tutti i termini; OR = almeno un termine.
* **Query brevi:** Incoraggia query concise.
* **Selezione precisa dei termini:** Risultati controintuitivi possibili a causa della mancanza di espansione della query.
* **Trattamento isolato dei termini:** Difficoltà nel controllare il numero di documenti recuperati e nella classificazione dei risultati.
* **Mancanza di ranking:** Impossibile eseguire feedback di rilevanza o esprimere richieste complesse.
* **Nessuna incertezza o pesatura dei termini:** Tutti i documenti corrispondenti sono restituiti senza priorità.
* **Sparsità:** Problema comune nella rappresentazione dei dati.
* **Utilizzo appropriato:** Solo per query corte e semplici, quando è sufficiente sapere se un elemento esiste o meno.

**Estensioni del Modello Booleano**

* **Incorporare metodi di classificazione:**
 * **Obiettivo:** Ordinare i risultati mantenendo la base booleana.
 * **Metodo:** Assegnare pesi ai termini (in [0,1]) nei documenti e/o nelle query.
 * **Calcolo dei risultati:** Utilizzo del modello booleano standard, seguito dal calcolo della distanza/somiglianza (es. Jaccard) e dalla classificazione in base a queste metriche.
 * **Feedback di rilevanza:** Selezione dei documenti basata sulla classificazione.

* **Insiemi Fuzzy:**
 * Rilassare i confini degli insiemi booleani.
 * Grado di appartenenza ($wA$) di un elemento ad un insieme A.
 * Operazioni: Intersezione ($wA∩B = min(wA, wB)$), Unione ($wA∪B = max(wA, wB)$).

* **MMM (Mixed Min and Max Model):**
 * Somiglianza query-documento:
 * Query disgiuntiva ($q_{or}$): $S(q_{or}, d) = λ_{or} * max(w1,.. ,wn) + (1 - λ_{or}) * min(w1,.. , wn)$
 * Query congiuntiva ($q_{and}$): $S(q_{and}, d) = λ_{and} * min(w1,.. , wn) + (1 - λ_{and}) * max(w1,.. , wn)$
 * $λ_{or}$ e $λ_{and}$ controllano il peso relativo del massimo e del minimo.

* **Modello di Paice:** Considera tutti i pesi dei termini, a differenza di MMM.

* **Modello P-norma:** Considera documenti e query come punti multidimensionali, utilizzando schemi di pesatura e coefficienti per gli operatori.

**Sparsità e Rappresentazione dei Dati**

* **Matrice termine-documento:** Estremamente sparsa (es. 500K termini x 1M documenti).
* **Rappresentazione efficiente:** Registrare solo le posizioni non nulle (1).

**Indice Inverso**

* **Definizione:** Mappa ogni termine all'insieme dei documenti in cui compare.
* **Vantaggi:** Efficienza di memorizzazione e ricerca.
* **Costruzione:** Identificazione dei documenti indicizzati per ogni termine e memorizzazione dei loro docID.

---

## Schema Riassuntivo: Indici Inversi e Elaborazione di Query

**I. Creazione della Matrice Termine-Documento**

* **A. Rappresentazione Iniziale:** Array di documenti indicizzati.
* **B. Trasposizione:** Generazione della matrice termine-documento.
* **C. Struttura Dati:**
 * **1. Liste di Postings:** Elenco di docID per ogni termine *t*.
 * **2. DocID:** Identificatore univoco per ogni documento.
* **D. Implementazione Liste di Postings:**
 * **1. Array di dimensioni fisse:** Inefficiente per liste variabili.
 * **2. Liste concatenate:** Flessibili per liste variabili.
 * **3. Array di lunghezza variabile:** Compromesso tra spazio e inserimento.
* **E. Memorizzazione:**
 * **1. Su disco:** Sequenza continua di postings (più efficiente).
 * **2. In memoria:** Liste concatenate o array di lunghezza variabile.
* **F. Considerazioni:** Dipendenza dalle esigenze dell'applicazione (spazio, velocità, aggiornamento).
* **G. Elementi Principali dell'Indice:**
 * **1. Dizionario:** Parole del corpus con puntatori alle liste di postings.
 * **2. Liste di postings:** Documenti contenenti la parola e la sua frequenza.
* **H. Doppia Indicizzazione:**
 * **1. Per chiave (Dizionario):** Accesso rapido alla lista di postings.
 * **2. Per documento (Liste di postings):** Identificazione rapida dei documenti.

**II. Processo di Indexing**

* **A. Input:** Stream di coppie (Termine, docID).
* **B. Passaggi:**
 * **1. Sequenza di coppie (Termine, docID):** Generazione dall'analisi dei documenti.
 * **2. Ordina per Termine:** Ordinamento della sequenza.
 * **3. Ordina per docID:** Raggruppamento e ordinamento per docID per ogni termine.
 * **4. Unisci voci multiple:** Unione delle voci con stesso termine e docID, mantenendo la frequenza.
 * **5. Aggiungi frequenza del documento:** Aggiunta della frequenza del termine nel documento.
 * **6. Dividi in:** Dizionario (termini distinti con termID) e Postings (liste di postings).

**III. Elaborazione delle Query Booleane**

* **A. Esempio:** `Brutus AND Caesar`
* **B. Passaggi:**
 * **1. Recupero postings di Brutus.**
 * **2. Recupero postings di Caesar.**
 * **3. Intersezione dei postings:** Algoritmo `INTERSECT(p1, p2)` (vedi codice nel testo).
* **C. Ottimizzazione:**
 * **1. Scenario:** Query con congiunzione AND di *n* termini.
 * **2. Strategia:** Elaborazione in ordine di frequenza crescente (iniziare dal termine meno frequente).
* **D. Query booleane arbitrarie:**
 * **1. Stima della dimensione degli insiemi OR (upper bound).**
 * **2. Elaborazione in ordine crescente di dimensioni OR.**
* **E. Complessità:** Tempo lineare rispetto al numero totale di voci nei postings.

**IV. Query di Frase**

* **A. Importanza:** Ricerca avanzata, facilità d'uso.
* **B. Approcci:**
 * **1. Indici biword:** Indizzazione di coppie consecutive di termini.
 * **2. Scomposizione di frasi più lunghe:** Query booleana su biword (rischio di falsi positivi).
* **C. Etichettatura Morfologica (POST) e Biword Estensi:**
 * **1. Analisi POST:** Classificazione grammaticale dei termini (Nomi, Articoli/Preposizioni).
 * **2. Biword Estensi:** Identificazione di sequenze $NX*N$.
 * **3. Dizionario:** Biword estesi come termini nel dizionario.
 * **4. Elaborazione Query:** Analisi della query e ricerca dei biword estesi.

```
INTERSECT(p1, p2)
1 answer ← ⟨ ⟩
2 while p1 ≠ NIL and p2 ≠ NIL
3 do if docID(p1) = docID(p2)
4 then ADD(answer, docID(p1))
5 p1 ← next(p1)
6 p2 ← next(p2)
7 else if docID(p1) < docID(p2)
8 then p1 ← next(p1)
9 else p2 ← next(p2)
10 return answer
```

---

**Schema Riassuntivo: Indicizzazione e Ricerca di Informazioni**

I. **Indicizzazione Posizionale**
 A. **Segmentazione e Ricerca:**
 1. Segmentazione della query in biword estesi.
 2. Ricerca dei biword estesi nell'indice posizionale.
 B. **Formato dell'Indice:** `<termine, numero di documenti contenenti il termine; doc1: posizione1, posizione2 … ; doc2: posizione1, posizione2 … ; …>`
 C. **Algoritmo di Unione Ricorsivo:**
 1. Utilizzato per query di frase e prossimità.
 2. Gestisce la fusione di liste di `doc:posizione`.
 D. **Limiti degli indici biword:** Non adatti per query di prossimità.
 E. **Dimensioni dell'indice:**
 1. 2-4 volte più grande di un indice non posizionale.
 2. 35-50% del volume del testo originale.
 3. Necessità di memorizzare ogni occorrenza, non solo per documento.
 4. Dipendenza dalla dimensione media del documento.

II. **Costruzione dell'Indice**
 A. **Costruzione basata sull'ordinamento:**
 1. Analisi documento per documento.
 2. Postings incompleti fino alla fine del processo.
 3. Necessità di memorizzazione su disco per collezioni di grandi dimensioni.
 B. **Scalabilità:**
 1. Costruzione in memoria non scalabile per collezioni molto grandi.
 2. Vincoli hardware: memoria, disco, velocità.
 3. Tolleranza ai guasti costosa: preferibile l'utilizzo di molte macchine normali.
 4. Accesso alla memoria più veloce dell'accesso al disco.
 5. I/O del disco basato su blocchi (8 KB - 256 KB).
 C. **BSBI (Indicizzazione basata sull'ordinamento a blocchi):**
 1. Minimizza le ricerche su disco.
 2. Ordinamento di blocchi di record (es. 10 milioni).
 3. `BSBIIndexConstruction()` (algoritmo descritto nel testo).
 4. Merge binario o multi-way merge per unione dei blocchi ordinati.
 5. Struttura ad albero per il merge (partizionamento, merge parziale, aggiornamento dell'indice).
 6. Utilizzo di una coda di priorità per il multi-way merge.
 7. Assunzione di condivisione parziale del lessico tra documenti.
 8. Gestione dei termini e compressione (con perdita solo per termini non essenziali).

III. **Ordinamento e Merge**
 A. **Ordinamento di blocchi:**
 1. Quicksort: $O(N \cdot ln (N))$ passaggi in media (N = dimensione del blocco).
 2. Necessità di 2 copie dei dati su disco (ottimizzabile).
 B. **Unione delle run ordinate:**
 1. Merge binario: albero di merge con $log_2(n)$ livelli (n = numero di blocchi).
 2. Multi-way merge: lettura simultanea da tutti i blocchi, utilizzo di buffer di lettura e scrittura, coda di priorità per la selezione del termID più basso.

---

**Schema Riassuntivo: Tecniche di Indicizzazione**

I. **Valutazione dei Token e Compressione:**
 * Necessità di soluzioni intermedie per valutare l'importanza dei token e la loro compressibilità con perdita.

II. **SPIMI (Indicizzazione in Memoria a Passaggio Singolo):**
 * Approccio *lazy* per risolvere il problema della mappatura (termine, termID) che non entra in memoria.
 * **Idea chiave 1:** Dizionari separati per ogni blocco, eliminando la mappatura tra blocchi.
 * **Idea chiave 2:** Nessun ordinamento iniziale; accumulo diretto dei postings.
 * Generazione di indici invertiti per blocco, poi uniti in un unico indice.
 * Vantaggi: più veloce (no ordinamento), risparmio di memoria (no termID intermedi), liste di postings dinamiche.
 * Algoritmo `SPIML-Invert(token_stream)`: gestisce la creazione e l'espansione dinamica delle liste di postings. L'ordinamento è posticipato.

III. **Indicizzazione Distribuita:**
 * Esecuzione parallela di parsing e inversione dell'indice, coordinata da una macchina master.
 * **Parser:** Assegna split a macchine parser inattive; genera coppie (termine, documento); suddivide le coppie in *j* partizioni lessicografiche.
 * **Inverter:** Raccoglie le coppie (termine, doc) per partizione; ordina e scrive le liste di postings.
 * Modello MapReduce:
 * **Fase Map:** Collezione di documenti → liste di coppie (termine, documento).
 * **Fase Reduce:** Liste di occorrenze → liste di postings.

IV. **Indicizzazione Distribuita con MapReduce:**
 * Implementazione robusta e semplice per calcoli distribuiti.
 * Sistema di indicizzazione di Google (circa 2002): serie di fasi MapReduce.
 * Schema:
 * `map`: input → list(k, v)
 * `reduce`: (k,list(v)) → output
 * Istanza per indicizzazione: `map`: collection → list(termID, docID); `reduce`: (<termID, list(docID)>, …) → (postings list, …).

V. **Indicizzazione Dinamica (Documenti in Arrivo, Eliminati, Modificati):**
 * Necessità di aggiornamenti di postings e aggiunta di nuovi termini.
 * **Approccio più semplice:** Indice principale e indice ausiliario.
 * Ricerca su entrambi gli indici, unione dei risultati.
 * Eliminazioni: vettore di bit di invalidazione per filtrare i documenti.
 * Re-indicizzazione periodica.
 * **Problemi:** Merge frequenti, scarsa performance durante le merge. Efficienza della fusione dipende dalla gestione dei file delle liste di postings.

VI. **Fusione Logaritmica:**
 * Tecnica di ordinamento con indici di dimensioni crescenti (1, 2, 4, 8...).
 * L'indice più piccolo in memoria, gli altri su disco.
 * Fusione e scrittura iterativa fino all'ordinamento completo.

VII. **Indicizzazione Dinamica: Fusione T/n:**
 * Indici ausiliari di dimensione *n* e indice principale con *T* postings.
 * Tempo di costruzione dell'indice: $O\left( \frac{T^2}{n} \right)$ nel caso peggiore.

---

# Schema Riassuntivo: Indicizzazione e Ricerca di Informazioni

**I. Fusione Logaritmica**

* **Efficienza:**
 * Ogni posting fuso al massimo $O\left( \log\left( \frac{T}{n} \right) \right)$ volte.
 * Complessità totale: $O\left( T \cdot \log\left( \frac{T}{n} \right) \right)$.
* **Vantaggi:** Più efficiente della fusione $\frac{T}{n}$ per la costruzione dell'indice.
* **Svantaggi:** Elaborazione query richiede fusione di $O\left( \log\left( \frac{T}{n} \right) \right)$ indici (vs. $O(1)$ con un solo indice).
* **Problema con più indici:** Gestione complessa delle statistiche a livello di collezione (es. correzione ortografica).
* **Possibile soluzione:** Ignorare indici secondari per l'ordinamento, basandosi solo sull'indice principale per il ranking.

**II. Indicizzazione Dinamica nei Motori di Ricerca**

* **Approccio:**
 * Modifiche incrementali frequenti (news, blog).
 * Ricostruzioni periodiche dell'indice da zero.

**III. Requisiti per la Ricerca in Tempo Reale**

* **Caratteristiche:**
 * Bassa latenza.
 * Elevato tasso di ingestione.
 * Letture e scritture concorrenti.
 * Dominanza del segnale temporale.

**IV. Costruzione dell'Indice**

* **Indicizzazione basata sull'ordinamento:**
 * Inversione in memoria ingenua.
 * Indicizzazione basata sull'ordinamento bloccato (BSBI).
 * Ordinamento per fusione (efficiente per disco rigido).
* **Indicizzazione in memoria a passaggio singolo (SPIMI):**
 * Nessun dizionario globale.
 * Dizionario separato per ogni blocco.
 * Postings non ordinati.
 * Accumulo postings in liste durante l'elaborazione.

**V. Compressione**

* **Obiettivi:**
 * Ridurre spazio su disco.
 * Aumentare la memoria disponibile.
 * Aumentare velocità di trasferimento dati.
 * [leggi dati compressi | decomprimi] > [leggi dati non compressi].
* **Presupposto:** Algoritmi di decompressione veloci.
* **Compressione del dizionario:** Ridurre dimensioni per mantenerlo in memoria principale.
* **Compressione dei file di postings:** Ridurre spazio su disco e tempo di lettura.
* **Compressione senza perdita vs. con perdita:**
 * **Senza perdita:** Preserva tutte le informazioni (metodo principale in IR).
 * **Con perdita:** Scarta informazioni (es. conversione in minuscolo, stop words, stemming). Possibile potare voci di postings poco probabili nei primi k risultati.

**VI. Dimensione del Vocabolario vs. Dimensione della Collezione**

* **Problema:** Determinare la dimensione del vocabolario (numero di parole distinte).
* **Legge di Heaps:** $M = kT^b$ (M = dimensione vocabolario, T = numero di token).
 * Valori tipici: $30 ≤ k ≤ 100$ e $b ≈ 0.5$.
 * Relazione lineare in scala log-log.
 * $log M = log k + b log T$.
* **Esempio (Reuters RCV1):** $log_{10}M = 0.49 log_{10}T + 1.64$ ($k≈ 44$, $b = 0.49$).

**VII. Distribuzioni Skew di Tipo Power-Law**

* **Caratteristiche:** Concentrazione di massa in una piccola zona, coda lunga.
* **Esempi:** Legge di Pareto (80-20), Legge di Zipf (frequenza delle parole).
* **Confronto con distribuzione di Poisson:** Entrambe asimmetriche, ma differiscono nella natura (spaziale vs. relazione tra variabili). La scelta dipende dal tasso minimo nella distribuzione di Poisson.
* **Esempi di distribuzioni power-law:** ricchezza, pagine web, follower, dimensione città, frequenza parole.

---

## Schema Riassuntivo: Legge di Heaps, Legge di Zipf e Recupero Classificato

**I. Distribuzione delle Parole nel Testo:**

* **A. Legge di Heaps:** Stima la dimensione del vocabolario di un corpus.
* **B. Legge di Zipf:** Descrive la distribuzione di frequenza delle parole.
 * $cf_{i} \propto \frac{1}{i} = \frac{K}{i}$ (frequenza del termine i-esimo)
 * $log(cf_{i}) = log(K) - log(i)$ (forma logaritmica)
 * Relazione lineare inversa tra frequenza e rango (power law con slope negativa).
 * Implicazioni: parole molto frequenti o rare sono meno utili per l'indicizzazione; i termini più discriminanti hanno frequenza medio-bassa.
 * Grafico: distribuzione a coda lunga, con poche parole molto frequenti e molte parole rare.

**II. Gestione della Frequenza dei Termini per l'Indicizzazione:**

* **A. Problema delle frequenze di taglio:**
 * Difficoltà nell'individuare frequenze di taglio ottimali per escludere termini troppo frequenti (poco significativi) o troppo rari (poco utili).
 * Dipendenza dal task, dal dominio e dal linguaggio.
* **B. Scopo delle frequenze di taglio:**
 * Escludere termini troppo frequenti (es. articoli, preposizioni) e troppo rari.
* **C. Vantaggi del pruning (eliminazione di termini):**
 * Miglioramento dell'efficienza (riduzione della dimensionalità).
 * Maggiore accuratezza (eliminazione di termini fuorvianti).
* **D. Criticità:**
 * Assenza di regole universali per le frequenze di taglio.
 * Dipendenza dal dominio e dal linguaggio.
* **E. Regole pratiche (esempi):**
 * Rimozione di termini presenti in più del 50% dei documenti.
 * Rimozione di termini presenti in meno di 3-5 documenti.
 * Approccio conservativo preferibile per IR e Data Mining tradizionale.

**III. Recupero dell'Informazione:**

* **A. Ricerca Booleana:**
 * Svantaggi: risultati troppo pochi o troppi; corrispondenza binaria (o sì o no); inadatta per utenti non esperti.
* **B. Recupero Classificato (Ranked Retrieval):**
 * Risolve i problemi della ricerca booleana restituendo un ordinamento dei documenti.
 * Si basa sulla pertinenza quantitativa documento-query.
 * Recupero, non estrazione di informazioni.
* **C. Query di Testo Libero:**
 * Parole in linguaggio naturale, associate al recupero classificato.
 * Risolve il problema dell' "abbondanza o carestia" di risultati.
 * Presenta solo i primi *k* risultati (≈10).
* **D. Punteggio come base del recupero classificato:**
 * Ordinamento dei documenti in base alla probabilità di utilità per il ricercatore.

---

**Classificazione di Documenti rispetto a una Query**

I. **Assegnazione di un punteggio di corrispondenza:**
 * Ogni documento riceve un punteggio in [0, 1] che indica la sua similarità con la query.

II. **Misure di similarità per insiemi finiti:**
 * Efficienti e normalizzate per lunghezza (documenti e query di dimensioni diverse).
 * **Limiti:** Non considerano la frequenza dei termini nel documento o la loro scarsità nella collezione.
 * Esempi:
 * **Jaccard:** $J(A,B) = \frac{\|A \cap B\|}{\|A \cup B\|}$ (Similarità tra due insiemi)
 * **Sørensen-Dice:** $DSC(A,B) = \frac{2 \times \|A \cap B\|}{\|A\| + \|B\|}$ (Pesa di più gli elementi in comune)
 * **Overlap:** $O(A,B) = \frac{\|A \cap B\|}{min(\|A\|, \|B\|)}$ (Intersezione divisa per la cardinalità minore)
 * **Simple Matching:** $SM(A,B) = \frac{\|A \cap B\| + \|\overline{A} \cap \overline{B}\|}{\|A \cup B\|}$ (Considera presenze e assenze)

III. **Proprietà delle misure di similarità:**
 * **Disuguaglianza triangolare:** Fondamentale per una metrica (es. Jaccard), ma non sempre soddisfatta (es. Dice).
 * Esempio di violazione della disuguaglianza triangolare con Dice: `Dice(1,2) + Dice(1,3) ≥ Dice(2,3)` non è sempre vera.

IV. **Frequenza dei termini:**
 * **Term Frequency (tf):** Numero di occorrenze di un termine in un singolo documento.
 * **Collection Frequency (cf):** Numero totale di occorrenze di un termine nell'intera collezione.
 * **Document Frequency (df):** Numero di documenti in cui un termine compare.

V. **Calcolo dello score di un documento:**
 * **Rappresentazione dei testi:** Matrice dove ogni cella contiene il peso di un termine in un documento.
 * **Formula di ponderazione:** $w_{t,d}=tf_{t,d} \frac{1}{cf_{t}}$ (Peso del termine *t* nel documento *d*, senza normalizzazione).

---

**Schema Riassuntivo: Rilevanza dei Termini e Funzione TF-IDF**

I. **Definizione della Rilevanza di un Termine**

 A. **Fattori Influenzanti:**
 1. *Term Frequency* (TF): Frequenza del termine nel documento.
 2. *Collection Frequency* (CF): Frequenza del termine nell'intero corpus.
 3. *Document Frequency* (DF): Numero di documenti contenenti il termine.
 4. Legge di Zipf: I termini più frequenti tendono ad essere più intensi localmente.
 B. **Funzioni di Scoring:**
 1. `Rilevanza(termine) = F(TF) + G(CF)`: Combinazione di funzioni per TF e CF, con peso relativo regolabile.
 2. Problematiche con approcci additivi e moltiplicativi: Dominanza di CF, smorzamento eccessivo.
 C. **Considerazioni sulla Lunghezza dei Documenti:** Nessuna assunzione specifica sulla lunghezza, si considera una lunghezza media.

II. **Analisi di Proposte di Ponderazione**

 A. **Proposta 1:** `w_{t,d}^{(t)}=f(tf_{t,d})+g(cf_{t})=tf_{t,d}+ly(cf_{t})` (eccessivamente dominata da CF).
 B. **Analisi in base alla posizione nella distribuzione di Zipf:**
 1. **Testa della distribuzione (termini molto frequenti):** Proposta 1: peso prossimo a 0; Proposta 2: dominata da CF.
 2. **Fuori dalla testa della distribuzione:** Proposta 1: TF smorzata da CF; Proposta 2: TF enfatizzata.
 C. **Problematiche:** Inefficacia delle proposte, dipendenza dal termine e dalle caratteristiche del documento.

III. **Smorzamento della Term Frequency**

 A. **Smorzamento Lineare Inverso:** Troppo aggressivo.
 B. **Soluzione Proposta:** `1/log(document frequency)`: Smorzamento *smooth*, penalizzando termini presenti in molti documenti.

IV. **Funzione TF-IDF**

 A. **Formula:** `w_{t,d}=\log(1+tf_{t,d}) \times \log_{10}\left( \frac{N}{df_{t}} \right)`
 B. **Componenti:**
 1. `log(1+tf_{t,d})`: TF logaritmica (smorzamento).
 2. `log_{10}(N/df_{t})`: IDF (Inverse Document Frequency).
 C. **Vantaggi:**
 1. Penalizza termini comuni.
 2. Evidenzia termini rari.
 3. Bilancia frequenza locale e globale.
 D. **Considerazioni:**
 1. Rimozione delle stop words.
 2. Stemming e lemmatization.
 3. Soglie di taglio (opzionale).
 E. **Smorzamento e Legge di Zipf:** Lo smorzamento logaritmico gestisce la distribuzione di Zipf, evitando soglie arbitrarie e permettendo l'utilizzo di matrici sparse.

---

**Modello Bag-of-words (BoW)**

* **Concetto base:** Indipendenza dei termini; ordinamento delle parole ignorato; passo indietro rispetto all'indice posizionale.
 * **Pro:** Corrispondenza parziale, punteggi graduati, efficiente per grandi collezioni, modello spazio vettoriale.
 * **Contro:** Mancanza di informazioni sintattiche e semantiche; assenza di controllo booleano; preferenza potenzialmente errata in query multi-termine.

**Ponderazione dei Termini**

* **Term Frequency (TF):** Frequenza di un termine in un documento.
 * **Frequenza grezza:** Non ideale; la rilevanza non è proporzionale alla frequenza.
 * **Peso logaritmico:** $w_{t,d} \begin{cases} 1+\log_{10}\text{tf}_{td} \ \text{ if tf}_{td} \ >0 \\ 0,\ \text{otherwise} \end{cases}$ Smorza l'influenza di frequenze molto alte.
 * **Punteggio documento-query:** $\sum_{t\in q \cap d}(1+\log(tf_{t,d}))$

* **Inverse Document Frequency (IDF):** Rarità di un termine nell'intero corpus.
 * **Informatività:** Termini rari sono più informativi (es. "arachnocentrico").
 * **Document Frequency (df):** Preferita alla Collection Frequency (cf) per discriminare tra documenti.
 * **Formula IDF:** $idf_{t}=\log_{10}\left( \frac{N}{df_{t}} \right)$ Smorza l'effetto dell'IDF.

* **TF-IDF:** Combinazione di TF e IDF.
 * **Formula:** $w_{t,d}=\log(1+tf_{t,d})\times\log_{10}\left( \frac{N}{df_{t}} \right)$
 * **Proprietà:** Aumenta con il numero di occorrenze nel documento e con la rarità del termine nel corpus. Riflette la distribuzione power-law delle frequenze dei termini.
 * **Varianti:** Differiscono nel calcolo di TF (con o senza logaritmi) e nella ponderazione dei termini nella query (ponderati o non ponderati).
 * **Formule Varianti:** $\frac{tf_{i,d}}{\max_{j}tf_{j,d}} ,\ \frac{tf_{id}}{\sqrt{ \sum_{j}(tf_{j,d})^2 }} ,\ \frac{tf_{id} \cdot idf_{i}}{\sqrt{ \sum_{j}(tf_{j,d} \cdot idf_{j})^2 }}$

**Normalizzazione e Similarità**

* **Normalizzazione dei vettori TF-IDF:** Aspetto cruciale per il recupero di informazioni e la classificazione di documenti.
* **Normalizzazione della lunghezza:** Obiettivo: comprendere l'impatto sulla rappresentazione dei topic.
 * **Problema:** La normalizzazione L2 può diluire il segnale informativo, soprattutto passando da testi brevi a testi lunghi.
 * **Considerazioni:** Proprietà geometriche della normalizzazione L2 e influenza sulla costruzione della matrice dei dati.

**Doppio Logaritmo:** Utilizzato per smorzare ulteriormente il peso del fattore di frequenza del termine (TF) in corpus molto grandi. (non esplicitamente definito nel testo, ma menzionato come tecnica aggiuntiva).

---

**I. Rappresentazione Vettoriale dei Documenti e delle Query**

* **Spazio Vettoriale:** Documenti e query rappresentati come vettori in uno spazio ad alta dimensionalità e sparsità.
 * Assi: termini del vocabolario.
 * Punti: documenti e query.
* **Similarità:** Misurata dalla prossimità dei vettori nello spazio.
 * Prossimità ≈ inversa della distanza.

**II. Limiti della Distanza Euclidea**

* **Sensibilità alla Lunghezza:** Favoreggia documenti più lunghi, indipendentemente dal contenuto.
 * Esempio: Distanza maggiore tra una query e un documento, rispetto alla stessa query e al documento ripetuto.
* **Alta Dimensionalità:** Soffre maggiormente dell'alta dimensionalità rispetto a misure di correlazione.

**III. Normalizzazione Implicita con l'Angolo Coseno**

* **Idea chiave:** Misurare la similarità tramite l'angolo tra i vettori (invece della distanza).
 * Classificazione in base all'angolo o al coseno dell'angolo.
* **Coseno Similarità:** Prodotto scalare normalizzato.
 * Formula: $\text{sim}(d_1, d_2) = \frac{d_1 \cdot d_2}{\|d_1\| \cdot \|d_2\|} = \frac{\sum_{i=1}^{n} w_{i,j} \cdot w_{i,k}}{\sqrt{\sum_{i=1}^{n} w_{i,j}^2} \cdot \sqrt{\sum_{i=1}^{n} w_{i,k}^2}}$
* **Normalizzazione:** Rende i vettori di lunghezza unitaria, eliminando il bias dovuto alla lunghezza.

**IV. Ponderazione Tf-Idf e Varianti**

* **Term Frequency (tf):**
 * `n (natural)`: $tf_{r, d}$
 * `l (logarithm)`: $1 + \log(tf_{r, d})$
 * `a (augmented)`: $0.5 + \frac{0.5 \cdot tf_{r, d}}{\max_{r} (tf_{r, d})}$
 * `b (boolean)`: $\begin{cases} 1 & \text{if } tf_{r, d} > 0 \\ 0 & \text{otherwise} \end{cases}$
* **Document Frequency (df):**
 * `n (no)`: $1$
 * `t (idf)`: $\log \frac{N}{df_r}$
 * `p (prob idf)`: $\max \{ 0, \log \frac{N - df_r}{df_r} \}$
* **Normalizzazione:**
 * `n (none)`: $1$
 * `c (cosine)`: $\frac{1}{\sqrt{w_1^2 + w_2^2 + \dots + w_n^2}}$
 * `u (pivoted unique)`: $\frac{1}{u}$
 * `b (byte size)`: $\frac{1}{\text{CharLength}^{\alpha}}, \alpha < 1$
* **Notazione SMART:** `ddd.qqq` (documento-documento-documento.query-query-query)

**V. Esempio di Calcolo del Punteggio Coseno e Classifica**

* Calcolo del punteggio coseno per ogni documento rispetto alla query.
* Classifica dei documenti in base al punteggio decrescente.

---

**Ricerca di Informazione basata su Punteggio**

* **Classificazione e Recupero:**
 * Ordinamento dei documenti per punteggio di rilevanza alla query.
 * Restituzione dei primi K documenti all'utente.

* **Pro:**
 * Corrispondenza parziale e ranking naturale.
 * Buone performance pratiche nonostante semplificazioni.
 * Implementazione efficiente.

* **Contro:**
 * **Limitazioni del Modello:**
 * Mancanza di informazioni sintattiche.
 * Mancanza di informazioni semantiche.
 * Ipotesi di indipendenza dei termini (Bag-of-Words - BoW).
 * Ipotesi di ortogonalità a coppie dei vettori termine.
 * **Funzionalità Mancanti:**
 * Assenza di controllo booleano (es. richiesta di presenza di un termine specifico).

---
