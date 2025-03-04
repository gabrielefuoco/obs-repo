
# Basi di Conoscenza e Rappresentazione del Significato

## I. Dizionario come Base di Conoscenza

Una base di conoscenza è una collezione di termini e dei loro significati, utilizzata per comprendere il significato del testo. Questo approccio, derivato dalla linguistica, mira a creare un lessico il meno ambiguo possibile.

## II. WordNet come Grafo di Conoscenza

WordNet è un'ontologia lessicale, ovvero un grafo di conoscenza (knowledge graph) che rappresenta le relazioni tra concetti.  Non si tratta di un grafo semplice, ma presenta una molteplicità di tipi di nodi e relazioni lessicali semantiche.


# Problematiche nella Ricerca dell'Informazione (IR)

## I. Polisemia

Molte parole hanno significati multipli, causando risultati irrilevanti nelle ricerche basate sulla corrispondenza di stringhe. I modelli stocastici di topic, utilizzando variabili latenti, rappresentano probabilisticamente i documenti come distribuzioni di topic, catturando meglio i segnali informativi.

## II. Sinonimia

Significati equivalenti espressi da parole diverse portano alla perdita di documenti rilevanti. Le soluzioni a questo problema prevedono trasformazioni in un nuovo spazio vettoriale.


# WordNet: Struttura e Funzionalità

## I. Synsets

I termini sono raggruppati in *synsets*, insiemi di sinonimi cognitivi che esprimono un concetto distinto (lessicalizzato). Un concetto è un insieme di sinonimi semanticamente equivalenti. (Nota: un dizionario spiega termini, non organizza concetti).

## II. Funzionalità

* Dizionario online (definizioni, esempi)
* Esempi d'uso (Usage Example)
* Tesauro (sinonimi, antonimi)
* Gloss (descrizione di un concetto)
* Ontologia lessicale (relazioni tra synsets: is-a, parte-tutto, implicazione, ...)

## III. Applicazioni

* **Ricerca semantica e IR:** Espansione di query/documento, relazione semantica, disambiguazione del senso (es. Lowest Common Subsumer).
* **Classificazione automatica del testo:** Assegnazione di documenti a categorie tematiche.
* **Traduzione automatica:** Utilizzato in sistemi come Google Translate (WordNet multilingue).


# Costituenti Lessicali e Sintattici

## I. Categorie Sintattiche Principali

Nomi, verbi, aggettivi, avverbi.

## II. Unità Lessicali Aggiuntive

* Verbi frasali (es. *get on*)
* Composti (es. *blueberry*)


# Rappresentazione Lessicale

## I. Lemma

Ogni composto è identificato come un singolo lemma.

## II. Collocazioni

Sequenze di parole frequenti (es. *one way*, *a surge of anger*).

## III. Fraseologismi

Espressioni con significato non deducibile dal significato delle singole parole (es. *kick the bucket*).

## IV. Informazioni Mancanti

* Pronuncia (risolto con OpenWordnet)
* Morfologia derivativa
* Etimologia
* Note di utilizzo
* Illustrazione pittorica


# Relazioni Semantiche e Lessicali

## I. Concetto Lessicalizzato

Un synset (insieme di sinonimi) rappresenta un concetto.

## II. Gerarchia Lessicale

Organizza i concetti; maggiore distanza implica un percorso cognitivo più lungo e una maggiore quantità di informazioni per la definizione.

## III. Tipi di Relazioni

* **Relazioni Lessicali:** Legami tra parole in un synset (sinonimi, contrari).
* **Relazioni Semantiche:** Legami tra synset (iperonimia, meronimia, implicazione, etc.).


# Memorizzazione dei Dati

## I. Categorie Sintattiche

* Nomi, aggettivi, avverbi: file separati.
* Nomi e verbi: raggruppati per campi semantici.
* Aggettivi: tre file (adj.all, adj.ppl, adj.pert).
* Avverbi: un unico file.

## II. Puntatori di Relazione

Memorizzati per rappresentare i legami tra elementi lessicali.

## III. Sostantivi

Relazioni semantiche e attributi/qualità modellate.

## IV. Unique Beginners

Categorie semantiche per organizzare i sostantivi.


---

# Appunti di Linguistica Computazionale

## Sinonimia

**A. Definizione:** Due parole, W1 e W2, sono sinonimi se la sostituzione di W1 con W2 in almeno un contesto non altera il significato.


## Relazioni tra Nomi

**A. Iponimi (~):** Relazione "Is-a" (es. {ciotola} ~⇨ {piatto}).

**B. Ipernimi (@):** Duale di iponimo (es. {scrivania} @⇨ {tavolo} @⇨ {mobile}).

**C. Meronimi (#):** Relazione parte-tutto (es. {becco, ala} # ⇨ {uccello}).  Si distinguono tre tipi: componente, membro, fatto da.

**D. Olonomi (%):** Relazione tutto-parti (es. {edificio} %⇨ {finestra}).

**E. Antonomi (!):** Significato opposto (es. {uomo} !-> {donna}).

**F. Nomi Polisemici:** Parole con molti significati (es. *topo*). Regola: la similarità di significato tra i diversi significati implica similarità tra i loro iponimi.

**G. Attributi (=) e Modifiche:** Valori espressi da aggettivi.


## Modificatori Linguistici

### I. Aggettivi

**A. Funzione:** Modificare i nomi.

**B. Tipi:**

1. **Descrittivi:** Descrivono caratteristiche (es. *sedia grande*).
2. **Participiali:** Derivati da verbi (-ing, -ed) (es. *acqua corrente*, *uomo stanco*).
3. **Relazionali:** Esprimono una relazione (es. *comportamento criminale*).

**C. Formato Rappresentazionale:** A(x) = agg (A(x) = aggettivo)

**D. Relazioni Semantiche:**

1. **Antinomia (!):** Opposizione (diretta o indiretta, es. *pesante/leggero*, *pesante/arioso*).
2. **Troponimo (~):** Similarità.
3. **Iperonimo (@):** Generalizzazione.
4. **Implicazione (*):** Un aggettivo implica l'altro.
5. **Causa (>):** Un aggettivo causa l'altro.
6. **Vedi anche (^):** Riferimenti correlati.

**E. Sottotipi di Aggettivi Descrittivi:**

1. **Colore** (es. *il blu del mare*, *un vestito blu*).
2. **Quantificatori** (es. *tutti*, *alcuni*).
3. **Participiali** (derivati da verbi).

**F. Aggettivi Relazionali:**

1. Non attributi intrinseci.
2. Non graduabili.
3. Posizione attributiva (prima del nome).
4. Mancanza di antonimo diretto.


### II. Altri Modificatori

**A. Marcatezza:** Confronto tra forme linguistiche (es. *profondo* vs. *basso*).

**B. Polisemia e Preferenze Selettive:** Alcuni aggettivi modificano quasi tutti i nomi (es. *buono/cattivo*), altri sono limitati (es. *modificabile/non modificabile*).


### III. Avverbi

**A. Derivazione:** Spesso da aggettivi (+ "-ly").

**B. Funzioni:**

1. **Modo:** Descrive *come* (es. *beautifully*).
2. **Grado:** Indica intensità (es. *extremely*).

**C. Altri Suffissi:** "-wise", "-way", "-ward" (es. *northward*).


## Organizzazione Lessicale

* **Aggettivi e Avverbi:** Ereditarietà di proprietà: antonimi e gradazione.
* **Verbi:** Organizzazione in file lessicografici. Tipi semantici: movimento, percezione, ecc. (lista estesa nel testo). Verbi stativi: collaborano con "essere" (es. assomigliare), verbi di controllo (es. volere). Polisemia elevata rispetto ai nomi (in inglese). Synset verbale: sinonimi, quasi-sinonimi, idiomi e metafore (es. "pass away" vs. "die").


## Relazioni Verbali

* **Entailment (*):** Valutazione della verità di A tramite B. V1 entailment V2 ≠ V2 entailment V1.

    * **Tipi di Entailment (basati sull'inclusione temporale):**
        * **+ Inclusione Temporale:**
            * **Troponimia (Coestensività):** azioni contemporanee (es. camminare - marciare).
            * **Troponimia (Inclusione Propria):** un'azione inclusa nell'altra (es. camminare - fare un passo).
        * **− Inclusione Temporale:**
            * **Presupposizione Inversa:** conoscenza/stato precedente (es. dimenticare - sapere).
            * **Causalità:** un'azione causa l'altra (es. rompere un vaso - il vaso è rotto).

* **Troponimo (~):** Y è un troponimo di X se Y è un modo particolare di fare X (analogia con iponimia). (es. balbettare - parlare). Specializzazione di un'azione, spesso in WordNet.

---

# Appunti su Relazioni Semantiche e WordNet

## I. Concetti Fondamentali

* **Antonimo (!):** Coppie di verbi con significato opposto (es. dare/prendere). Può anche essere un troponimo (es. fallire/riuscire).
* **Iperonimo (@):**  Y è iperonimo di X se X è un tipo di Y (es. percepire - ascoltare).
* **Gloss {get}:** Descrizione di un concetto.

**Organizzazione dei Verbi:**

* **Lexname:** Nome lessicale del verbo, utilizzato per l'organizzazione basata sull'argomento.
* **Entailment Lessicale:** Relazione di implicazione necessaria tra due verbi (prompt indiretto).


## II. Entailment Verbale

**Metodi di Accesso:**

* **Metodo Entailment Diretto:** Elenco diretto degli entailment per un verbo.
* **Hypohyms:** Utilizzo di troponimi (termini più specifici) per inferire l'entailment.


## III. Teorie del Significato

**Composizione del Significato:**

* **Significato come Insieme di Concetti Atomici:** Il significato è scomposto in concetti elementari (es. "Comprare": ![Immagine non visualizzabile](Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241107110336756.png)).
* **Postulato di Significato (Fodor):** Il significato è dato dalle relazioni tra parole (es. "comprare" implica "ottenere", "pagare", "scegliere", "vendere").
* **Postulato di Significato (Rosch):** Il significato è l'informazione vera per gli esemplari più tipici (es. "tigre" implica "felino", "strisce", "pericoloso").
* **Reti Semantiche (Quillian):** Il significato è definito dalle relazioni in una rete semantica (es. "Comprare" è iponimo di "Ottenere", antonimo di "Vendere", implica "Pagare", "Scegliere"; iponimi: "Acquisire", "Raccogliere").


## IV. WordNet e Valutazione delle Relazioni Semantiche

**WordNet come Ontologia Lessicale:** Rappresenta le relazioni semantiche tra concetti. La distanza tra concetti indica la forza della relazione.

**Sfide nella Quantificazione delle Relazioni:**

* **Multirelazionalità:** Molte relazioni e cammini di lunghezza variabile.
* **Importanza delle Relazioni:** Difficoltà nel quantificare l'importanza di diverse relazioni (iperonimia, meronimia).
* **Profondità dei Concetti:** Il significato di un cammino dipende dal livello di astrazione.

**Misura Basata sul Percorso:**

* **Formula implicita:** Similarità direttamente proporzionale alla distanza dalla radice, inversamente proporzionale alla somma delle profondità relative dei due synset.
* **Descrizione:** Considera la posizione nell'ontologia e la lunghezza del percorso tra i concetti.


## V. WordNet per la Similarità Semantica

**I. Misure di Similarità in WordNet:**

**A. Misure basate sul gloss:**

* Affinità di contenuto tra i gloss dei synset (sovrapposizione di gloss).
* Formula: $$go-rel(s_{1},s_{2})=\sum_{go\in GO(g_{1},g_{2})}|go|^2$$

**B. Misure basate sul percorso:**

* Funzione della posizione dei nodi synset nell'ontologia.
* Considera profondità dei concetti e lunghezza del cammino.
* Formula: $$p-rel(s_{1},s_{2})=\frac{2\text{depth}(lcs(s_{1},s_{2}))}{\text{depth}(s_{1})+\text{depth}(s_{2})}$$

**C. Misure basate sul contenuto informativo:**

* Funzione dell'IC (Information Content) dei nodi synset.
* IC calcolato tramite teoria dell'informazione (logaritmo della probabilità di osservazione).
* Formula: $$ic-rel(s_{1},s_{2})=\frac{2IC(lcs(s_{1},s_{2}))}{IC(s_{1})+IC(s_{2})}$$


**II. Approcci Alternativi:**

**A. Approccio basato sulla frequenza relativa:**

* Indipendente dall'ontologia.
* Stima la similarità anche per concetti nuovi.
* Basato sulla frequenza relativa del concetto.

**B. Approccio basato sul contenuto:**

* Confronto delle definizioni dei concetti.
* Passaggi:
    1. Pre-processing (stop-list, lemmatizzazione, etc.)
    2. Estrazione di termini informativi (sostantivi, verbi).
    3. Confronto delle definizioni (matching sintattico o semantico).
    4. Quantificazione della similarità (es. Gloss Overlap).
* Gloss Overlap (ρ): misura di similarità sintattica basata sulla sottostringa comune più lunga (o somma dei quadrati delle lunghezze se più sottostringhe).


**III. Approcci Avanzati:**

A.  (Segue...)

---

# Approcci Ibridi e Risorse Lessicali Multilingue

## Approcci Ibridi

Gli approcci ibridi combinano elementi di diversi metodi per superare i limiti di ciascun approccio individuale.  Un esempio rilevante è l'utilizzo di *embedding* in WordNet.


## Embedding in WordNet

Questa tecnica sfrutta i *node embedding* per rappresentare i concetti all'interno di WordNet.  Un esempio significativo è rappresentato da EuroWordNet.


## WordNet Multilingue (EuroWordNet)

### Caratteristiche principali:

* **Lingue supportate:** Olandese, italiano, spagnolo, inglese (30.000 synsets), tedesco, francese, estone, ceco (10.000 synsets).
* **Relazioni tra lingue:**  Sono definite relazioni come "near_synonym", "xpos_", etc.
* **Indice Linguistico (ILI):**  Un sistema per gestire le relazioni tra le diverse lingue. Il codice "eq_" indica l'equivalenza.
* **Ontologia dei concetti condivisi:** Definisce i concetti base comuni a tutte le lingue.
* **Gerarchia di etichette:** Organizza i concetti gerarchicamente, suddividendoli per dominio.


### Struttura dei dati:

* **Indici interlingua:** Elenco non strutturato di indici, ognuno associato a un synset e al gloss inglese corrispondente.
* **Collegamento dei codici ILI:** Ogni codice ILI è collegato a:
    * Significato specifico del synset per la lingua.
    * Termini generali di livello superiore.
    * Possibili domini.
* **Relazioni di equivalenza:** Sono definite tra gli indici ILI e i significati di una specifica lingua.


---

## Schema Riassuntivo: MultiWordNet e WordNet Italiano v1.4

### I. MultiWordNet

* **Strategia:** Costruzione dei grafi di diverse lingue basandosi sul grafo di WordNet inglese.
* **Pro:** Richiede meno lavoro manuale, garantisce un'alta compatibilità tra i grafi e permette l'utilizzo di procedure automatiche.
* **Contro:** Presenta una forte dipendenza dalla struttura di WordNet inglese.


### II. WordNet Italiano v1.4

* **Procedura di Assegnazione Synset:**
    * Basata sul riferimento inglese: si crea un elenco ponderato di synset inglesi simili, e il lessicografo seleziona il synset corretto.
    * Si procede all'individuazione di eventuali lacune lessicali.
    * **Risorse utilizzate:** Dizionario Collins, Princeton WordNet (PWN), WordNet Domains, Dizionario Italiano (DISC).

* **Gruppi di Traduzione (TGR):** Raggruppano le diverse traduzioni di una parola, riflettendo la polisemia e le sfumature di significato.

* **Statistiche:**
    * Inglese: 40.959 parole, 60.901 TGR
    * Italiano: 32.602 parole, 46.545 TGR

* **Selezione dei Synset "Migliori":**
    * Si trova un synset per ogni senso.
    * Si elencano i synset in base a: probabilità generica, traduzione, somiglianza delle gloss, intersezione tra synset.
    * Si selezionano i synset "migliori".


### III. Gloss Similarity

* **Metodi di Valutazione:**
    * **Campo Semantico:** Appartenenza allo stesso campo semantico (es: "sclerosi" nel campo semantico della medicina).
    * **Sinonimi e Iperonimi:** Presenza di sinonimi o iperonimi condivisi (es: "ragione" e "sogliola").
    * **Contesto:** Influenza del contesto d'uso sul significato e sulla similarità (es: "manico").
    * **Similarità Basata su Iperonimi e Sinonimi Condivisi:** Analisi degli iperonimi e sinonimi condivisi per valutare la similarità semantica (es: "albero" e "sogliola" con i rispettivi iperonimi).  Esempi: `{ tree }` - a tall perennial woody **plant** having a main trunk ...; `{ tree, tree diagram }` - a figure that branches from...; `{ sole }` - right-eyed flatfish; many are valued as food: => `{ flatfish }` - any of several families of **fishes** having ...


## Valutazione della Similarità tra Glossari

* **Analisi della Condivisione Semantica:**
    * Campi semantici condivisi.
    * Sinonimi condivisi.
    * Iperonimi condivisi (es. "pesce" per "sole").
    * Considerazione del contesto d'uso (es. "sole" come parte del piede vs. "sole" come astro).
    * Esempio: `{ sole }` - the underside of the foot => `{ area, region }` (significato diverso rispetto all'iperonimo generico "pesce").


---

## Metodologia di Analisi del Glossario

La metodologia adottata per l'analisi del glossario si basa sul confronto delle singole voci.  L'obiettivo è individuare sovrapposizioni nei campi semantici, sinonimi e iperonimi, considerando attentamente il contesto di ogni termine.  Questo approccio permette di identificare eventuali ridondanze o ambiguità presenti nel glossario, garantendo una maggiore precisione e chiarezza nella definizione dei concetti.

---

Per favore, forniscimi il testo da formattare.  Non ho ricevuto alcun testo da elaborare nell'input precedente.  Inserisci il testo che desideri formattare e io lo elaborerò seguendo le istruzioni fornite.

---
