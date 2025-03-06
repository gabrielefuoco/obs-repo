
##### Basi di Conoscenza e Rappresentazione del Significato

* **Dizionario come Base di Conoscenza:** Una base di conoscenza è una collezione di termini e significati, utilizzata per comprendere il significato del testo. L'approccio deriva dalla linguistica, con l'obiettivo di un lessico il meno ambiguo possibile.
* **WordNet come Grafo di Conoscenza:** WordNet è un'ontologia lessicale, un grafo di relazioni tra concetti (knowledge graph). Non è un grafo semplice, ma presenta una molteplicità di tipi di nodi e relazioni lessicali semantiche.

##### Problematiche nella Ricerca dell'Informazione (IR)

* **Polisemia:** Molte parole hanno significati multipli, portando a risultati irrilevanti nella ricerca di informazioni basata sulla corrispondenza di stringhe. I modelli stocastici di topic, usando variabili latenti, rappresentano probabilisticamente i documenti come distribuzioni di topic, catturando meglio i segnali informativi.
* **Sinonimia:** Significati equivalenti espressi da parole diverse portano alla perdita di documenti rilevanti. Le soluzioni prevedono trasformazioni in un nuovo spazio.

##### WordNet: Struttura e Funzionalità

* **Synsets:** I termini sono raggruppati in *synsets*, insiemi di sinonimi cognitivi che esprimono un concetto distinto (lessicalizzato). Un concetto è un insieme di sinonimi semanticamente uguali. (Nota: un dizionario spiega termini, non organizza concetti).
* **Funzionalità:**
	* Dizionario online (definizioni, esempi)
	* Usage Example (esempi d'uso)
	* Tesauro (sinonimi, antonimi)
	* Gloss (descrizione di un concetto)
	* Ontologia lessicale (relazioni tra synsets: is-a, parte-tutto, implicazione, ...)
* **Applicazioni:**
	* **Ricerca semantica e IR:** Espansione di query/documento, relazione semantica, disambiguazione del senso (es. Lowest Common Subsumer).
	* **Classificazione automatica del testo:** Assegnazione di documenti a categorie tematiche.
	* **Traduzione automatica:** Utilizzato in sistemi come Google Translate (WordNet multilingue).

##### Costituenti Lessicali e Sintattici

* **Categorie Sintattiche Principali:** Nomi, verbi, aggettivi, avverbi.
* **Unità Lessicali Aggiuntive:**
	* Verbi frasali (es. *get on*)
	* Composti (es. *blueberry*)

##### Rappresentazione Lessicale

* **Lemma:** Ogni composto è identificato come un singolo lemma.
* **Collocazioni:** Sequenze di parole frequenti (es. *one way*, *a surge of anger*).
* **Fraseologismi:** Espressioni con significato non deducibile dal significato delle singole parole (es. *kick the bucket*).
* **Informazioni Mancanti:**
	* Pronuncia (risolto con OpenWordnet)
	* Morfologia derivativa
	* Etimologia
	* Note di utilizzo
	* Illustrazione pittorica

##### Relazioni Semantiche e Lessicali

* **Concetto Lessicalizzato:** Un synset (insieme di sinonimi) rappresenta un concetto.
* **Gerarchia Lessicale:** Organizza i concetti; maggiore distanza implica percorso cognitivo più lungo e maggiore quantità di informazioni per la definizione.
* **Tipi di Relazioni:**
	* **Relazioni Lessicali:** Legami tra parole in un synset (sinonimi, contrari).
	* **Relazioni Semantiche:** Legami tra synset (iperonimia, meronimia, implicazione, etc.).

##### Memorizzazione dei Dati

* **Categorie Sintattiche:**
	* Nomi, aggettivi, avverbi: file separati.
	* Nomi e verbi: raggruppati per campi semantici.
	* Aggettivi: tre file (adj.all, adj.ppl, adj.pert).
	* Avverbi: un unico file.
	* **Puntatori di Relazione:** Memorizzati per rappresentare i legami tra elementi lessicali.
	* **Sostantivi:** Relazioni semantiche e attributi/qualità modellate.
	* **Unique Beginners:** Categorie semantiche per organizzare i sostantivi.

##### Sinonimi

* **Definizione:** Due parole W1 e W2 sono sinonimi se sostituendo W1 con W2 in almeno un contesto, il significato non cambia.

##### Relazioni tra Nomi

* **Iponimi (~):** Relazione "Is-a" (es. {ciotola} ~⇨ {piatto}).
* **Ipernimi (@):** Duale di iponimo (es. {scrivania} @⇨ {tavolo} @⇨ {mobile}).
* **Meronimi (#):** Parte-tutto (es. {becco, ala} # ⇨ {uccello}). Tre tipi: componente, membro, fatto da.
* **Olonomi (%):** Tutto-parti (es. {edificio} %⇨ {finestra}).
* **Antonomi (!):** Significato opposto (es. {uomo} !-> {donna}).
* **Nomi Polisemici:** Molti significati (es. *topo*). Regola: similarità di significato tra i significati implica similarità tra i loro iponimi.
* **Attributi (=) e Modifiche:** Valori espressi da aggettivi.

### Modificatori Linguistici

##### Aggettivi

Funzione: Modificare i nomi.
Tipi:
- Descrittivi: Descrivono caratteristiche (es. *sedia grande*).
- Participiali: Derivati da verbi (-ing, -ed) (es. *acqua corrente*, *uomo stanco*).
- Relazionali: Esprimono una relazione (es. *comportamento criminale*).
Formato Rappresentazionale: A(x) = agg (A(x) = aggettivo)
Relazioni Semantiche:
- Antinomia (!): Opposizione (diretta o indiretta, es. *pesante/leggero*, *pesante/arioso*).
- Troponimo (~): Similarità.
- Iperonimo (@): Generalizzazione.
- Implicazione (*): Un aggettivo implica l'altro.
- Causa (>): Un aggettivo causa l'altro.
- Vedi anche (^): Riferimenti correlati.
Sottotipi di Aggettivi Descrittivi:
- Colore (es. *il blu del mare*, *un vestito blu*).
- Quantificatori (es. *tutti*, *alcuni*).
- Participiali (derivati da verbi).
Aggettivi Relazionali:
- Non attributi intrinseci.
- Non graduabili.
- Posizione attributiva (prima del nome).
- Mancanza di antonimo diretto.

##### Altri Modificatori

Marcatezza: Confronto tra forme linguistiche (es. *profondo* vs. *basso*).
Polisemia e Preferenze Selettive: Alcuni aggettivi modificano quasi tutti i nomi (es. *buono/cattivo*), altri sono limitati (es. *modificabile/non modificabile*).

##### Avverbi

Derivazione: Spesso da aggettivi (+ "-ly").
Funzioni:
- Modo: Descrive *come* (es. *beautifully*).
- Grado: Indica intensità (es. *extremely*).
Altri Suffissi: "-wise", "-way", "-ward" (es. *northward*).

##### Organizzazione Lessicale

* **Aggettivi e Avverbi:**
	* Ereditarietà di proprietà: antonimi e gradazione.
* **Verbi:**
	* Organizzazione in file lessicografici.
	* Tipi semantici: movimento, percezione, ecc. (lista estesa nel testo).
	* Verbi stativi: collaborano con "essere" (es. assomigliare), verbi di controllo (es. volere).
	* Polisemia elevata rispetto ai nomi (in inglese).
	* Synset verbale: sinonimi, quasi-sinonimi, idiomi e metafore (es. "pass away" vs. "die").

##### Relazioni Verbali

* **Entailment (*):**
	* Valutazione della verità di A tramite B.
	* V1 entailment V2 ≠ V2 entailment V1.
	* Tipi di Entailment (basati sull'inclusione temporale):
* **+ Inclusione Temporale:**
	* Troponimia (Coestensività): azioni contemporanee (es. camminare - marciare).
	* Troponimia (Inclusione Propria): un'azione inclusa nell'altra (es. camminare - fare un passo).
* **− Inclusione Temporale:**
	* Presupposizione Inversa: conoscenza/stato precedente (es. dimenticare - sapere).
	* Causalità: un'azione causa l'altra (es. rompere un vaso - il vaso è rotto).
* **Troponym (~):**
	* Y è un troponimo di X se Y è un modo particolare di fare X (analogia con iponimia). (es. balbettare - parlare).
	* Specializzazione di un'azione, spesso in WordNet.
* **Antonym (!):**
	* Coppie di verbi opposti (es. dare/prendere).
	* Può essere anche un troponimo (es. fallire/riuscire).
* **Hypernym (@):**
	* Y è iperonimo di X se X è un tipo di Y (es. percepire - ascoltare).
	* **Glosses {get}:** Descrizione di un concetto.

##### Organizzazione Verbi

* **Lexname:** Nome lessicale del verbo per organizzazione basata sull'argomento.
* **Entailment Lessicale:** Relazione di implicazione necessaria tra due verbi (prompt indiretto).

##### Entailment Verbale

* **Metodi di Accesso:**
	* **Metodo Entailment Diretto:** Elenco diretto di entailment per un verbo.
	* **Hypohyms:** Utilizzo di troponimi (termini più specifici) per inferire entailment.

##### Teorie del Significato

* **Composizione del Significato:**
	* **Significato come Insieme di Concetti Atomici:** Il significato è scomposto in concetti elementari

* **Postulato di Significato (Fodor):** Il significato è dato dalle relazioni tra parole (es. "comprare" implica "ottenere", "pagare", "scegliere", "vendere").

* **Postulato di Significato (Rosch):** Il significato è l'informazione vera per gli esemplari più tipici (es. "tigre" implica "felino", "strisce", "pericoloso").

* **Reti Semantiche (Quillian):** Il significato è definito dalle relazioni in una rete semantica (es. "Comprare" è iponimo di "Ottenere", antonimo di "Vendere", implica "Pagare", "Scegliere"; iponimi: "Acquisire", "Raccogliere").

##### WordNet e Valutazione delle Relazioni Semantiche

* **WordNet come Ontologia Lessicale:** Rappresenta le relazioni semantiche tra concetti. La distanza tra concetti indica la forza della relazione.

* **Sfide nella Quantificazione delle Relazioni:**
	* **Multirelazionalità:** Molte relazioni e cammini di lunghezza variabile.
	* **Importanza delle Relazioni:** Difficoltà nel quantificare l'importanza di diverse relazioni (iperonimia, meronimia).
	* **Profondità dei Concetti:** Il significato di un cammino dipende dal livello di astrazione.

* **Misura Basata sul Percorso:**
	* **Formula implicita:** Similarità direttamente proporzionale alla distanza dalla radice, inversamente proporzionale alla somma delle profondità relative dei due synset.
	* **Descrizione:** Considera la posizione nell'ontologia e la lunghezza del percorso tra i concetti.

##### WordNet per la Similarità Semantica

##### Misure di Similarità in WordNet:

##### Misure basate sul gloss:

* Affinità di contenuto tra i gloss dei synset (sovrapposizione di gloss).
* Formula: $$go-rel(s_{1},s_{2})=\sum_{go\in GO(g_{1},g_{2})}|go|^2$$

##### Misure basate sul percorso:

* Funzione della posizione dei nodi synset nell'ontologia.
* Considera profondità dei concetti e lunghezza del cammino.
* Formula: $$p-rel(s_{1},s_{2})=\frac{2\text{depth}(lcs(s_{1},s_{2}))}{\text{depth}(s_{1})+\text{depth}(s_{2})}$$

##### Misure basate sul contenuto informativo:

* Funzione dell'IC (Information Content) dei nodi synset.
* IC calcolato tramite teoria dell'informazione (logaritmo della probabilità di osservazione).
* Formula: $$ic-rel(s_{1},s_{2})=\frac{2IC(lcs(s_{1},s_{2}))}{IC(s_{1})+IC(s_{2})}$$

##### Approcci Alternativi:

##### Approccio basato sulla frequenza relativa:

* Indipendente dall'ontologia.
* Stima la similarità anche per concetti nuovi.
* Basato sulla frequenza relativa del concetto.

##### Approccio basato sul contenuto:

* Confronto delle definizioni dei concetti.
* Passaggi:
- Pre-processing (stop-list, lemmatizzazione, etc.)
- Estrazione di termini informativi (sostantivi, verbi).
- Confronto delle definizioni (matching sintattico o semantico).
- Quantificazione della similarità (es. Gloss Overlap).
* Gloss Overlap (ρ): misura di similarità sintattica basata sulla sottostringa comune più lunga (o somma dei quadrati delle lunghezze se più sottostringhe).

### Approcci Avanzati:

##### Approcci Ibridi:

* Combinano elementi di diversi approcci per superare i limiti individuali.

##### Embedding in WordNet:

* Utilizzo di node embedding per rappresentare i concetti.

##### WordNet Multilingue (EuroWordNet)

##### Caratteristiche principali:

**Lingue supportate:** Olandese, italiano, spagnolo, inglese (30.000 synsets), tedesco, francese, estone, ceco (10.000 synsets).
**Relazioni tra lingue:** "near_synonym", "xpos_", etc.
**Indice Linguistico (ILI):** Sistema per gestire le relazioni tra lingue (codice "eq_" per equivalenza).
**Ontologia dei concetti condivisi:** Definisce concetti base condivisi.
**Gerarchia di etichette:** Organizza i concetti gerarchicamente per dominio.

##### Struttura dei dati:

**Indici interlingua:** Elenco non strutturato di indici, ognuno con synset e gloss inglese.
**Collegamento dei codici ILI:** Collegamento a:
* Significato specifico del synset per la lingua.
* Termini generali di livello superiore.
* Possibili domini.
**Relazioni di equivalenza:** Tra indici ILI e significati di una lingua specifica.

## Schema Riassuntivo: MultiWordNet e WordNet Italiano v1.4

##### MultiWordNet:

* **Strategia:** Costruzione dei grafi di diverse lingue basandosi sul grafo di WordNet inglese.
* **Pro:** Meno lavoro manuale, alta compatibilità tra grafi, procedure automatiche.
* **Contro:** Forte dipendenza dalla struttura di WordNet inglese.

##### WordNet Italiano v1.4:

* **Procedura di Assegnazione Synset:**
	* Basata sul riferimento inglese: elenco ponderato di synset inglesi simili, selezione del synset corretto da parte del lessicografo.
	* Individuazione di lacune lessicali.
	* Risorse utilizzate: Dizionario Collins, Princeton WordNet (PWN), WordNet Domains, Dizionario Italiano (DISC).
* **Gruppi di Traduzione (TGR):**
	* Raggruppano le diverse traduzioni di una parola, riflettendo polisemia e sfumature di significato.
* **Statistiche:**
	* Inglese: 40.959 parole, 60.901 TGR
	* Italiano: 32.602 parole, 46.545 TGR
* **Selezione dei Synset "Migliori":**
	* Trova synset per ogni senso.
	* Elenca synset in base a: probabilità generica, traduzione, somiglianza delle gloss, intersezione tra synset.
	* Seleziona i synset "migliori".

##### Gloss Similarity:

* **Metodi di Valutazione:**
	* **Campo Semantico:** Appartenenza allo stesso campo semantico (es: "sclerosi" nel campo semantico della medicina).
	* **Sinonimi e Iperonimi:** Presenza di sinonimi o iperonimi condivisi (es: "ragione" e "sogliola").
	* **Contesto:** Influenza del contesto d'uso sul significato e sulla similarità (es: "manico").
	* **Similarità Basata su Iperonimi e Sinonimi Condivisi:** Analisi degli iperonimi e sinonimi condivisi per valutare la similarità semantica (es: "albero" e "sogliola" con i rispettivi iperonimi). Esempi forniti includono: `{ tree }` - a tall perennial woody **plant** having a main trunk ...; `{ tree, tree diagram }` - a figure that branches from...; `{ sole }` - right-eyed flatfish; many are valued as food: => `{ flatfish }` - any of several families of **fishes** having ...

##### Valutazione della Similarità tra Glossari

* **Analisi della Condivisione Semantica:**
	* Campi semantici condivisi.
	* Sinonimi condivisi.
	* Iperonimi condivisi (es. "pesce" per "sole").
	* Considerazione del contesto d'uso (es. "sole" come parte del piede vs. "sole" come astro).
	* Esempio: `{ sole }` - the underside of the foot => `{ area, region }` (significato diverso rispetto all'iperonimo generico "pesce").

* **Metodologia:** Confronto delle voci di glossario per individuare sovrapposizioni nei campi semantici, sinonimi e iperonimi, tenendo conto del contesto.

