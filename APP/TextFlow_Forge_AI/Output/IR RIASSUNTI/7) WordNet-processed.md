
Una base di conoscenza può essere rappresentata come un dizionario, ovvero un insieme di termini e relativi significati,  utilizzato per analizzare e comprendere il testo.  L'approccio, derivante dalla linguistica, mira a un lessico il meno ambiguo possibile.  WordNet, esempio di *knowledge graph*, è un'ontologia lessicale, un grafo di relazioni tra concetti (non un semplice grafo, ma con molteplici tipi di nodi e relazioni lessicali semantiche).

La semplice corrispondenza di stringhe nella Ricerca dell'Informazione (IR) presenta problemi: la polisemia (molteplici significati di una parola) e la sinonimia (parole diverse con significati simili) portano a risultati irrilevanti o incompleti.  Modelli stocastici di topic, basati su variabili latenti, rappresentano probabilisticamente i documenti come distribuzioni di topic, migliorando la capacità di catturare i segnali informativi.

WordNet è un ampio database lessicale inglese che organizza termini in *synsets*, insiemi di sinonimi cognitivi che rappresentano concetti distinti (un concetto = un synset).  Funziona come dizionario online (definizioni, esempi), tesauro (relazioni tra parole), glossario (descrizione di concetti) e ontologia lessicale (relazioni tra synsets: *is-a*, parte-tutto, implicazione, etc.).  Supporta la ricerca semantica e IR (espansione di query/documenti, disambiguazione del senso delle parole,  *Lowest Common Subsumer* per trovare il concetto più generale che include due concetti specifici), la classificazione automatica del testo e la traduzione automatica (utilizzato anche da Google Translate).

Il sistema considera quattro categorie sintattiche principali (nomi, verbi, aggettivi, avverbi) e unità lessicali aggiuntive come verbi frasali (es. *get on*) e composti (es. *blueberry*).

---

Questo documento descrive un sistema per la rappresentazione lessicale e semantica, focalizzato sulla creazione di un lessico computazionale.  Il sistema tratta i composti come singoli lemmi e include la gestione di collocazioni e fraseologismi.  Mancano informazioni su pronuncia, morfologia derivativa, etimologia, note d'uso e illustrazioni.

Le relazioni semantiche sono estratte manualmente da un thesaurus, con ogni synset (insieme di sinonimi) che rappresenta un concetto lessicalizzato organizzato in una gerarchia.  Una maggiore distanza gerarchica implica una maggiore complessità cognitiva nella connessione tra concetti. Il sistema modella relazioni lessicali (sinonimia, contrarietà all'interno di un synset) e relazioni semantiche tra synset (iperonimia, meronimia, implicazione, etc.).

I dati sono memorizzati in file separati per categoria sintattica (nomi, aggettivi, avverbi), con nomi e verbi raggruppati per campi semantici. Gli aggettivi sono suddivisi in tre file (generali, participi, possessivi), mentre gli avverbi sono in un unico file.  I puntatori di relazione collegano gli elementi lessicali. Per i sostantivi, vengono modellate sia relazioni semantiche che relazioni che specificano attributi o qualità, organizzati tramite categorie tematiche ("Unique Beginners").  ![[]]

La sinonimia è definita dalla sostituibilità di una parola in almeno un contesto senza alterare il significato.  Le relazioni tra nomi includono:

* **Iponimi (~):** relazione "Is-a" (es. {ciotola} ~⇨ {piatto}).
* **Ipernimi (@):** relazione inversa dell'iponimia (es. {scrivania} @⇨ {tavolo} @⇨ {mobile}).
* **Meronimi (#):** relazione "parte di" (es. {becco, ala} # ⇨ {uccello}).
* **Olonomi (%):** relazione inversa della meronimia (es. {edificio} %⇨ {finestra}).
* **Antonomi (!):** relazione di opposizione (es. {uomo} !-> {donna}).

Il sistema gestisce anche i nomi polisemici, applicando regole di similarità tra i significati e i loro iponimi. Infine, gli attributi (=) sono espressi tramite aggettivi.

---

# Modificatori: Aggettivi e Avverbi

Questo documento descrive aggettivi e avverbi come modificatori nella lingua.

## Aggettivi

Gli aggettivi modificano i nomi, descrivendone le caratteristiche.  Si distinguono in:

* **Aggettivi descrittivi:** Descrivono qualità (es. *sedia grande*, *sedia piccola*).  La relazione semantica tra aggettivi descrittivi può essere di:
    * **Antinomia (!):** Opposizione diretta (es. *pesante/leggero*) o indiretta (es. *pesante/arioso*).
    * **Troponimo (~):** Similarità di significato.
    * **Iperonimo (@):** Generalizzazione.
    * **Implicazione (*):** Un aggettivo implica l'altro.
    * **Causa (>):** Un aggettivo è causa dell'altro.
    * **Vedi anche (^):** Riferimento ad altri aggettivi correlati.
* **Aggettivi participiali:** Derivati da verbi (es. *acqua corrente*, *tempo trascorso*), in forma "-ing" o "-ed".  "Parte principale di" indica l'origine verbale (es. "breaking" è la parte principale di "break").
* **Aggettivi relazionali:** Esprimono una relazione (es. *comportamento criminale*), non sono graduabili, non hanno antonimo diretto e si trovano solo in posizione attributiva.

Il formato rappresentazionale per gli aggettivi è:  `A(x) = agg` (dove `A(x)` è l'aggettivo e `agg` indica che è un aggettivo).  Esempio: `PESO(pacchetto) = pesante`.

Altri tipi di aggettivi descrittivi includono aggettivi di colore (es. *blu*) e quantificatori (es. *tutti*, *alcuni*).

## Marcatezza e Polisemia

La marcatezza si riferisce all'uso di un termine piuttosto che di un altro (es. *profondo* vs. *basso*). La polisemia indica che un aggettivo può avere significati diversi a seconda del nome modificato (es. *vecchio* per persone o cose). Alcuni aggettivi modificano quasi tutti i nomi (es. *buono/cattivo*), altri sono più limitati (es. *modificabile/non modificabile*).


## Avverbi

Gli avverbi, spesso derivati da aggettivi con il suffisso "-ly", modificano verbi, aggettivi o altri avverbi, specificando modo (es. *beautifully*) o grado (es. *extremely*). Altri suffissi includono "-wise", "-way", "-ward".

---

## Riassunto dell'organizzazione lessicale di verbi e avverbi

Questo testo descrive l'organizzazione lessicale di verbi e avverbi, focalizzandosi sulle relazioni semantiche tra essi.

### Avverbi

Gli avverbi ereditano proprietà dagli aggettivi da cui derivano, includendo antonimi e la possibilità di gradazione (es. *molto*, *abbastanza*, *estremamente*).

### Verbi

I verbi sono organizzati in file lessicografici, raggruppati per categorie semantiche (movimento, percezione, comunicazione, etc.) e tipologie (stativi, di controllo).  A differenza dei nomi, i verbi inglesi presentano una maggiore polisemia.  I *synset* verbali raggruppano sinonimi e quasi-sinonimi (es. "pass away", "die", "kick the bucket"), includendo anche idiomi e metafore.

### Relazioni Verbali

Le relazioni tra verbi sono descritte principalmente attraverso l'**entailment lessicale**: un verbo **X** *implica* un verbo **Y** se compiere **X** necessariamente implica compiere **Y** (es. fare un pisolino implica dormire).  Questa relazione non è reciproca.

L'entailment può essere classificato in base all'inclusione temporale:

* **Con inclusione temporale:**
    * **Troponimia (Coestensività):** azioni contemporanee per tutta la durata (es. camminare - marciare).
    * **Troponimia (Inclusione Propria):** un'azione inclusa nell'altra, non necessariamente per tutta la durata (es. camminare - fare un passo).
* **Senza inclusione temporale:**
    * **Presupposizione Inversa:** un'azione implica uno stato precedente (es. dimenticare - sapere).
    * **Causalità:** un'azione causa l'altra (es. rompere un vaso - il vaso è rotto).

Altri tipi di relazioni verbali includono:

* **Troponimia (~):**  **Y** è un troponimo di **X** se **Y** è un modo particolare di fare **X** (es. balbettare è un troponimo di parlare).  È una specializzazione dell'entailment.
* **Antonimia (!):**  relazione di opposizione (es. dare/prendere). Può anche essere un troponimo.
* **Iperonimia (@):**  **Y** è un iperonimo di **X** se **X** è un tipo di **Y** (es. percepire è un iperonimo di ascoltare).

Le descrizioni dei concetti sono definite come **glosses**.  Per organizzare i verbi in base al loro argomento si usa il **lexname**.  Infine, il testo menziona che l'entailment lessicale non viene specificato direttamente nel *prompt*.

---

Il testo descrive diversi approcci alla rappresentazione del significato lessicale, focalizzandosi sull' *entailment* verbale e sull'utilizzo di WordNet.

**Metodi per l'accesso all'entailment:**  Il testo presenta due metodi principali per ricavare gli *entailment* di un verbo: il metodo diretto, che fornisce una lista di *entailment* possibili, e l'utilizzo degli *hypohyms* (troponimi), termini più specifici che permettono di inferire gli *entailment*.

**Teorie del significato:** Vengono illustrate diverse teorie:

* **Significato come insieme di concetti atomici:** Il significato di una parola è scomposto in concetti elementari (es. "comprare"  ![Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241107110336756.png]).
* **Postulato di Fodor:** Il significato è dato dalle relazioni tra parole (es. "comprare" implica "ottenere", "pagare", etc.).
* **Postulato di Rosch:** Il significato è l'informazione vera per gli esemplari più tipici (es. "tigre" implica "felino", "strisce", etc.).
* **Reti semantiche (Quillian):** Il significato è definito dalle relazioni in una rete semantica (es. "comprare" è iponimo di "ottenere", antonimo di "vendere").  Un esempio di rete semantica per "comprare" è fornito, mostrando iponimi, antonimi e implicazioni.

**WordNet e la valutazione delle relazioni semantiche:** WordNet è un'ontologia lessicale che rappresenta le relazioni semantiche. La distanza tra i concetti indica la forza della relazione.  La quantificazione di queste relazioni presenta sfide: multirelazionalità, importanza variabile delle relazioni e profondità dei concetti.  Una misura utilizzata è quella *basata sul percorso*, direttamente proporzionale alla distanza dalla radice dell'ontologia e inversamente proporzionale alla somma delle profondità dei due *synset* coinvolti.  In sostanza, questa misura considera sia la posizione dei concetti nell'ontologia che la lunghezza del percorso che li connette per stimare la similarità semantica.

---

## Riassunto di WordNet per la Similarità Semantica e Multilinguismo

Questo documento descrive l'utilizzo di WordNet per la misurazione della similarità semantica e la sua estensione a più lingue.

### Misure di Similarità Semantica in WordNet

WordNet permette di calcolare la similarità semantica tra concetti (synset) utilizzando diversi approcci:

* **Misure basate sul gloss:**  Valutano la sovrapposizione tra le definizioni (gloss) dei synset. Un esempio è  `go-rel(s₁,s₂)=∑_(go∈GO(g₁,g₂))|go|²`, dove `go` rappresenta una sottostringa comune e `|go|` la sua lunghezza.

* **Misure basate sul percorso:** Considerano la posizione dei synset nell'ontologia.  La similarità è inversamente proporzionale alla distanza tra i synset: `p-rel(s₁,s₂)=2depth(lcs(s₁,s₂))/(depth(s₁)+depth(s₂))`, dove `lcs` è il Lowest Common Subsumer e `depth` la profondità nell'ontologia.

* **Misure basate sul contenuto informativo:** Utilizzano l'Information Content (IC) dei synset, calcolato dalla probabilità di osservare un concetto in un corpus: `ic-rel(s₁,s₂)=2IC(lcs(s₁,s₂))/(IC(s₁)+IC(s₂))`.

* **Approccio basato sulla frequenza relativa:** Indipendente dall'ontologia, stima la similarità usando la frequenza relativa dei concetti.

* **Approccio basato sul contenuto:** Confronta le definizioni dei concetti, includendo pre-processing (stop-list, lemmatizzazione), estrazione di termini informativi (sostantivi e verbi), e confronto (sintattico o semantico) per poi quantificare la similarità (es. Gloss Overlap). Il *Gloss Overlap* (ρ) calcola la sottostringa comune di lunghezza massima (o il quadrato della lunghezza se più sottostringhe hanno la massima lunghezza).

* **Approcci ibridi ed embedding:** Combinano i metodi precedenti o utilizzano tecniche di *node embedding* per rappresentare i synset.


### WordNet Multilingue: EuroWordNet

EuroWordNet estende WordNet a più lingue, includendo:

* **Lingue supportate:** Olandese, italiano, spagnolo, inglese, tedesco, francese, estone, ceco (con variazioni nel numero di synset).

* **Relazioni interlinguistiche:**  Relazioni come "near_synonym" e "xpos_" collegano synset di lingue diverse.

* **Indice Linguistico Interlingua (ILI):** Sistema di gestione delle relazioni tra lingue, usando codici come "eq_" per l'equivalenza.

* **Ontologia dei concetti condivisi e gerarchia di etichette:** Definiscono concetti base e organizzano i concetti per dominio.

La struttura dati include indici interlingua (synset e gloss inglese), collegamenti tra codici ILI e significati specifici per lingua, e relazioni di equivalenza tra indici ILI e significati.

---

## Riassunto di MultiWordNet e WordNet Italiano v1.4

Questo riassunto confronta MultiWordNet con l'approccio utilizzato per WordNet Italiano v1.4, focalizzandosi sulle metodologie di creazione e sui criteri di selezione dei synset.

### MultiWordNet

MultiWordNet costruisce i grafi lessicali di diverse lingue a partire da WordNet inglese.  Questo approccio, sebbene automatizzato e compatibile tra le lingue, presenta una forte dipendenza dalla struttura di WordNet inglese.

**Pro:** Minor lavoro manuale, alta compatibilità tra grafi, procedure automatiche.
**Contro:** Forte dipendenza da WordNet inglese.


### WordNet Italiano v1.4

WordNet Italiano v1.4 adotta una procedura di assegnazione basata sul riferimento inglese, ma con un intervento lessicografico cruciale.  Per ogni senso italiano, il sistema propone un elenco ponderato di synset inglesi simili, lasciando al lessicografo la scelta finale.  Questo processo include anche l'identificazione di lacune lessicali.  Le risorse utilizzate includono il Dizionario Collins, Princeton WordNet (PWN), WordNet Domains e il Dizionario Italiano (DISC).

Il concetto chiave è quello dei **Gruppi di Traduzione (TGR)**, insiemi di significati tradotti tra due lingue che raggruppano le diverse traduzioni di una parola, riflettendo la polisemia.  Le statistiche mostrano 40.959 parole inglesi con 60.901 TGR e 32.602 parole italiane con 46.545 TGR.

La selezione dei synset "migliori" considera: la probabilità generica, la traduzione, la similarità delle gloss e l'intersezione tra synset.


### Similarità tra Glossari (Gloss Similarity)

La valutazione della similarità tra glossari si basa su diversi aspetti:

* **Campo Semantico:** L'appartenenza allo stesso campo semantico (es. "sclerosi" nel campo semantico della medicina).
* **Sinonimi e Iperonimi:** La presenza di sinonimi o iperonimi condivisi indica forte similarità semantica (es. "sogliola" e "sole").
* **Contesto:** Il contesto d'uso influenza il significato e la similarità (es. "manico" di un coltello vs. maniglia di una porta).

L'analisi di iperonimi e sinonimi condivisi è un metodo fondamentale per valutare la similarità semantica (es. "albero" e il suo iperonimo "pianta").

---

La similarità tra glossari si basa sull'analisi della sovrapposizione di elementi semantici.  Questo include la condivisione di campi semantici, sinonimi e iperonimi, considerando sempre il contesto d'uso di ogni termine.  L'esempio fornito,  "`{ sole }` - the underside of the foot => `{ area, region }`", illustra come la stessa parola ("sole") possa avere significati diversi a seconda del contesto, rendendo l'iperonimo ("pesce" nel testo originale) più generico e meno rilevante per la comparazione semantica.  Pertanto, una valutazione accurata della similarità richiede un'analisi contestualizzata dei termini.

---
