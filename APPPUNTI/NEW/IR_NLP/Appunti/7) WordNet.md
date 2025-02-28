Una base di conoscenza può essere vista come un **dizionario**, ovvero una collezione di termini con il loro significato. Utilizziamo il testo per descrivere un testo. L'approccio universalmente adottato deriva dalla linguistica. Accediamo alla base di conoscenza per comprendere il significato sottostante a una determinata parola.

I linguisti si sono preoccupati di usare un **lessico meno ambiguo possibile**.

**WordNet** non è solo un **tesauro**, ma anche un'**ontologia lessicale**. È dunque un **grafo di relazioni tra concetti**. Non parliamo di un grafo semplice (insieme di nodi e archi, in cui oltre all'orientamento di questi o all'aggiungere un peso, altro non possiamo fare), ma abbiamo una **moltiplicità di tipi di nodi e di relazioni (lessicali semantiche)** che assumono un significato diverso a seconda della tipologia del nodo. Possiamo anche indicarlo come **knowledge graph**. 

## Motivazioni

Qualsiasi strategia che si basi semplicemente sulla corrispondenza di stringhe per selezionare i documenti può portare a problemi nella Ricerca dell'Informazione (IR).

* **Polisemia:** molte parole hanno significati multipli, il che significa che termini presenti nella query dell'utente possono essere trovati in molti documenti irrilevanti. Con i modelli stocastici di topic, per definizione riusciamo a rappresentare in maniera probabilistica, i documenti come distribuzioni di topic e ci consente di catturare meglio i segnali informativi dei topic e rappresentandoli probabilisticamente siamo in grado di capire quando un termine contribuisce alla costruzione di un topic. Ci serviamo per farlo di variabili latenti.
* **Sinonimia:** molti significati equivalenti o strettamente correlati possono essere espressi da parole distinte, il che può portare a perdere molti documenti rilevanti.

Le risolviamo attraverso trasformazioni in un nuovo spazio.

## WordNet

WordNet è un ampio database lessicale della lingua inglese.

* I termini delle principali categorie sintattiche sono raggruppati in insiemi di sinonimi cognitivi, chiamati **synsets**, ognuno dei quali esprime un concetto distinto (lessicalizzato). Un concetto, ossia un significato, detto anche senso, corrisponde a quelli chiamati synsets, Un concetto è un insieme di sinonimi, ovvero una classe di insiemi semanticamente uguali. 
* Il ruolo di un dizionario non è quello di organizzare concetti ma di spiegare termini

WordNet offre funzionalità per diversi utilizzi:

* **Dizionario online:** Definizioni delle parole, esempi di frasi, insiemi di sinonimi.
	* **Usage Example:** Esempi d'uso di un termine.
* **Tesauro:** Relazioni tra le parole che compongono i synsets: sinonimi, antonimi.
	* **Gloss**: Descrizione di un concetto.
* **Ontologia lessicale:** Relazioni tra i synsets: is-a, parte-tutto, implicazione, ...

WordNet supporta diverse attività:

* **Ricerca semantica e IR:**
	* Espansione dei termini di query/documento.
	* Relazione semantica e disambiguazione del senso delle parole. Un esempio è il **Lowest Common Subsumer**: concetto che sussume i due concetti in imput, ovvero il concetto più generale che include entrambi i concetti specifici.
* **Classificazione automatica del testo:**
	* I documenti vengono assegnati a una o più categorie tematiche in base al loro contenuto.
* **Traduzione automatica:**
	* Può essere esteso a più lingue.
	* Viene utilizzato anche da Google Translate come parte del processo di traduzione tra le lingue coinvolte nei WordNet multilingue.

## Costituenti Lessicali e Sintattici

Il sistema considera quattro categorie sintattiche principali: nomi, verbi, aggettivi e avverbi.  Oltre a queste categorie fondamentali, vengono considerate anche le seguenti unità lessicali:

* **Verbi frasali:** Espressioni composte da un verbo e una particella (es. *get on*, *break up*).
* **Composti:** Parole formate dalla combinazione di due o più parole (es. *blueberry*).  Questi composti vengono identificati come un singolo lemma.
* **Collocazioni:** Sequenze di parole che frequentemente compaiono insieme (es. *one way*, *a surge of anger*).
* **Fraseologismi (o frasi idiomatiche):** Espressioni il cui significato non è deducibile dal significato delle singole parole (es. *kick the bucket*).


Le risorse utilizzate non forniscono informazioni su:

* **Pronuncia:** (problema risolto con OpenWordnet)
* **Morfologia derivativa:** processi di formazione delle parole.
* **Etimologia:** origine delle parole.
* **Note di utilizzo:** informazioni contestuali sull'uso delle parole.
* **Illustrazione pittorica:** immagini associate alle parole.


## Relazioni Semantiche e Lessicali

Le relazioni semantiche vengono estratte manualmente dai sinonimi presenti in un thesaurus.

**Concetto Lessicalizzato:** Un synset (insieme di sinonimi) rappresenta un concetto.

### Gerarchia Lessicale

La gerarchia lessicale organizza i concetti.  Una maggiore distanza nella gerarchia implica un percorso cognitivo più lungo per connettere i concetti.  Inoltre, ogni concetto lessicalizzato richiede una maggiore quantità di informazioni rispetto a quelle necessarie per definire la gerarchia stessa.


### Tipi di Relazioni

Il sistema modella due tipi principali di relazioni:

* **Relazioni Lessicali:**  Stabiliscono legami tra le parole all'interno di un synset, identificando sinonimi e contrari.
* **Relazioni Semantiche:**  Collegano i synset tra loro, rappresentando relazioni come iperonimia (relazione "Is-a"), meronimia (relazione "parte di"), implicazione, e altre.


## Memorizzazione dei Dati

La memorizzazione dei dati è organizzata come segue:

* **Nomi, aggettivi, avverbi:** Memorizzati in file separati per categoria sintattica.
* **Nomi e verbi:** Raggruppati in base ai campi semantici.
* **Aggettivi:** Divisi in tre file (adj.all, adj.ppl, adj.pert), probabilmente per distinguere tra aggettivi generali, aggettivali participi e aggettivi possessivo.
* **Avverbi:** Memorizzati in un unico file.
* **Puntatori di relazione:** Memorizzati per rappresentare i legami tra i diversi elementi lessicali.
* **Sostantivi:** Per i sostantivi, vengono modellate sia le relazioni semantiche che le relazioni che specificano attributi o qualità.


##### Unique Beginners: Categorie sematiche per organizzare i sostantivi

![[ir2-20241105105801146.png]]
## Sinonimi

Un termine può essere sostituito in almeno un contesto.

**Sinonimo in WordNet:** Due parole W1 e W2 sono sinonimi se sostituendo W1 con W2 in *almeno* un contesto (linguistico), il significato della frase data non cambia.

## Relazioni tra nomi

##### Iponimi (~):

- Relazione semantica per la relazione Is-a.
* Una parola con un significato più specifico rispetto a un termine generale o sovraordinato applicabile ad essa.
* Ad esempio, "ciotola" è un iponimo di "piatto": {ciotola} ~⇨ {piatto}

##### Ipernimi (@):

- Duale di Iponimo
* Una parola con un significato ampio sotto cui ricadono parole più specifiche; un termine sovraordinato.
* Ad esempio, {scrivania} @⇨ {tavolo} @⇨ {mobile}

##### Meronimi (#):

* La relazione semantica che sussiste tra una parte e il tutto.
* Ad esempio, "becco" e "ala" sono meronimi di "uccello": {becco, ala} # ⇨ {uccello}
* Tre tipi: componente, membro, fatto da.

##### Olonomi (%):

* La relazione semantica che sussiste tra un tutto e le sue parti.
* Ad esempio, "edificio" è un olonimo di "finestra": {edificio} %⇨ {finestra}

##### Antonomi (!):

* Una parola di significato opposto a un'altra.
* Ad esempio, {uomo} !-> {donna}

##### Nomi polisemici:

* Nomi che hanno molti significati.
* Ad esempio, (topo) animale vivente o dispositivo informatico.
* Regole: due significati di una parola sono simili, quindi il significato dei loro iponimi dovrebbe essere simile nello stesso modo.

##### Attributi (=) e modifiche:

* I valori dell'attributo sono espressi da aggettivi.
* La modifica può anche essere un nome.
* Ad esempio, sedia -> sedia piccola, sedia grande.

## Aggettivi

Gli aggettivi hanno la funzione principale di modificare i nomi.

### Tipi di Aggettivi

Si distinguono principalmente tre tipi di aggettivi:

* **Aggettivi descrittivi:** Descrivono caratteristiche o qualità del nome.
* **Aggettivi participiali:** Derivati da verbi, possono essere in forma "-ing" (presente) o "-ed" (passato).
* **Aggettivi relazionali:** Esprimono una relazione tra il nome che modificano e un altro concetto.


### Formato Rappresentazionale

Il formato utilizzato per rappresentare gli aggettivi è:

* A(x) = agg

Dove:

* A(x) rappresenta l'aggettivo.
* agg indica che si tratta di un aggettivo.

Esempio: PESO (pacchetto) = pesante


### Relazioni Semantiche tra Aggettivi

#### Antinomia (!)

L'antinomia è una relazione semantica fondamentale tra aggettivi descrittivi, che indica una relazione di opposizione.  "È ANONIMO A" significa che un aggettivo è l'opposto di un altro.  Questa relazione può essere:

* **Diretta:**  es. pesante/leggero.
* **Indiretta:** es. pesante/arioso.


Oltre all'antinomia, esistono altre relazioni semantiche tra aggettivi:

* **Troponimo (~):**  Relazione di similarità o vicinanza di significato.
* **Iperonimo (@):** Relazione di generalizzazione (l'iperonimo è un termine più generale).
* **Implicazione (\*):** Un aggettivo implica l'altro.
* **Causa (>):** Un aggettivo è causa dell'altro.
* **Vedi anche (^):**  Riferimento ad altri aggettivi correlati.


**Nota:**  La relazione di "contrario" implica che una delle proposizioni può essere vera, oppure entrambe possono essere false.


Gli aggettivi possono essere utilizzati per esprimere diversi livelli di azione o intensità.

### Altri Tipi di Aggettivi Descrittivi

Oltre agli aggettivi descrittivi generali, si possono distinguere:

* **Aggettivi di colore:**  Possono funzionare sia come nomi che come aggettivi (es. *il blu del mare*, *un vestito blu*).
* **Quantificatori:**  Esprimono quantità (es. tutti, alcuni, molti, pochi...).
* **Aggettivi participiali:**  Derivati da verbi, indicano un'azione o uno stato (es. *acqua corrente*, *tempo trascorso*).  "Parte principale di" indica che il participio deriva dal verbo (es. "breaking" è la parte principale di "break").  Possono essere in forma "-ing" o "-ed".

### Aggettivi Relazionali

Gli aggettivi relazionali si distinguono dagli aggettivi descrittivi per diverse caratteristiche:

* Non si riferiscono ad attributi intrinseci del nome.
* Non sono graduabili.
* Si trovano solo in posizione attributiva (prima del nome).
* Mancano di un antonimo diretto.

Esempio: *comportamento criminale*.

## Altri modificatori

##### Marcatezza:

* Unità linguistica normale (termine non marcato) confrontata con unità possibili forme irregolari (termine marcato).
* Ad esempio: La piscina è profonda 5 piedi, NON: La piscina è bassa 5 piedi.
* *Profondo*: termine marcato
* *Basso*: termine non marcato.
- L'uso del modificatore universalmente accettato in una forma piuttosto che un'altra

##### Polisemia e preferenze selettive:

* Ad esempio: "vecchio" può significare "non giovane" - modifica le persone.
* Ad esempio: "vecchio" può significare "non nuovo" - modifica le cose.
* Alcuni aggettivi possono modificare quasi tutti i nomi.
	* Ad esempio: buono / cattivo, desiderabile / indesiderabile.
* Alcuni aggettivi possono essere strettamente limitati ad alcuni nomi.
	* Ad esempio: modificabile / non modificabile.

## Avverbi

Gli avverbi sono spesso derivati da aggettivi tramite suffissazione.  Il suffisso più comune è "-ly", che può specificare:

* **Modo:**  Descrive *come* avviene l'azione.  Esempio: *beautifully*.
* **Grado:** Indica l'intensità dell'azione o di un aggettivo. Esempio: *extremely*.

Altri suffissi includono: "-wise", "-way", "-ward" (es. *northward*, *forward*).

Inoltre, gli avverbi ereditano dagli aggettivi di origine:

* **Antonimi:** Se l'aggettivo ha un antonimo, l'avverbio corrispondente ne avrà uno.
* **Gradazione:** Molti avverbi possono essere graduati (es. *molto*, *abbastanza*, *estremamente*).


## Verbi

Anche i verbi sono organizzati per categorie, chiamate file lessicografici.

##### Tipi di verbi semantici:

* Movimento, percezione, comunicazione, competizione, cambiamento, cognitivo, consumo, creazione, emozione, possesso, cura del corpo, funzioni, comportamento sociale, interazione.

##### Verbi stativi:

* Collaborano con il verbo essere: assomigliare, appartenere, bastare.
* Verbi di controllo: volere, fallire, impedire, riuscire, iniziare.

Non è possibile raggruppare tutti i verbi in un'unica categoria come i nomi.
L'inglese ha meno verbi rispetto ai nomi, *MA* circa il doppio di polisemia rispetto ai nomi.

##### Synset verbale:

* Sinonimi e quasi sinonimi.
* Ad esempio: "pass away" vs. "die" vs. "kick the bucket".
* Idiomi e metafore:
 * "Kick the bucket" include synset.
 * "Die" include sinonimi: "break", "break down" (per auto e computer). 

## Relazioni verbali

##### Entailment (*)

Non c'è modo di valutare il valore di verità della frase **A** se non attraverso il valore di verità della frase **B**.

L'Entailment Lessicale è la relazione costituente che lega un verbo con un altro:

* Il verbo **Y** è implicato da **X** se facendo **X** si deve necessariamente fare **Y**.
    * Ad esempio, fare un pisolino implica dormire.
    * **X** è un modo di eseguire l'azione **Y**.
* La relazione non è reciproca: V1  entailment V2 **NON** implica V2 entailment V1.


### Tipi di Entailment

L'entailment può essere classificato in base all'inclusione temporale:

**+ Inclusione Temporale:**

* **Troponimia (Coestensività):**  Le azioni si svolgono contemporaneamente per tutta la loro durata.
    * *camminare - marciare*
    * *parlare - sussurrare*

* **Troponimia (Inclusione Propria):** Un'azione è inclusa nell'altra, ma non necessariamente per tutta la sua durata.
    * *camminare - fare un passo*
    * *dormire - russare*


**− Inclusione Temporale:**

* **Presupposizione Inversa:** L'azione implica una precedente conoscenza o stato.
    * *dimenticare - sapere*
    * *srotolare - arrotolare*

* **Causalità:** Un'azione causa l'altra.
    * *mostrare - vedere*
    * *rompere - rompere* (Nota: qui si intende probabilmente una relazione di causa-effetto tra due eventi di rottura, ad esempio "rompere un vaso" causa "il vaso è rotto")

##### Troponym (~):

* Il verbo **Y** è un troponimo del verbo **X** se l'attività **Y** sta facendo **X** in qualche modo.
	- **X** è un modo particolare di eseguire **Y**. Vi è un'analogia con il concetto di **Iponimia**.
* Ad esempio, balbettare è un troponimo di parlare.
* Caso speciale di entailment, specializzazione di una particolare azione.
* Più frequentemente codificato in WordNet.

##### Antonym (!):

* Ad esempio, dare/prendere, comprare/vendere, prestare/prendere in prestito, insegnare/imparare.
* Può anche essere un troponimo.
* Ad esempio, fallire/riuscire implica provare, dimenticare implica sapere.

##### Hypernym (@):

* Il verbo Y è un iperonimo del verbo X se l'attività X è un (tipo di) Y.
* Ad esempio, percepire è un iperonimo di ascoltare. 

**Glosses {get}:** descrizione di un concetto.

### Organizzazione dei Verbi

Per organizzare i verbi in base al loro argomento, è necessario utilizzare il **lexname**, ovvero il nome lessicale del verbo.

### Entailment Lessicale

L'**entailment lessicale** è un concetto che riguarda la relazione tra due verbi. Dati due verbi, l'entailment lessicale si verifica quando uno dei due verbi implica necessariamente l'altro. 

- **Prompt indiretto**: non forniamo una specifica di Entailment.
- **Prompt diretto**: forniamo un significato di Entailment.

Esistono due metodi principali per accedere agli entailment di un verbo:

1. **Metodo entailment:** Questo metodo permette di ottenere gli entailment di un verbo direttamente, fornendo una lista di possibili entailment.
2. **Hypohyms:** Questo metodo restituisce i troponimi, ovvero i termini che sono più specifici del verbo in questione. I troponimi possono essere utilizzati per inferire gli entailment del verbo.

## WordNet e altre teorie del significato

##### Come si crea l'ontologia di WordNet?

##### Composizione del significato:

* **Significato come insieme di concetti atomici:** Questa teoria sostiene che il significato di una parola può essere scomposto in un insieme di concetti elementari. Questo permette di esprimere il significato in modo più preciso e analitico.

**Esempio:** "Comprare" (Jackendoff 1983): 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241107110336756.png]]

## Postulato di significato (Fodor)

La rappresentazione del significato di una parola viene fornita attraverso le relazioni di significato tra le parole.

**Esempio:** "comprare"

* `comprare(x,y,z) → ottenere (x,y,z)`
* `comprare(x,y,z) → pagare (x,y,z)`
* `comprare(x,y,z) → scegliere (x,y)`
* `comprare(x,y,z) → vendere (z,y,x)`

**Esempio:** "scapolo"

* `scapolo(x) → uomo(x) ¬sposato(x)`

## Postulato di significato (Rosch)

Il significato di una parola è l'informazione che è vera per gli esemplari più tipici correlati a quel concetto.

**Esempio:** "tigre"

* `tigre(x) → felino(x)`
* `tigre(x) → strisce(x)`
* `tigre(x) → pericoloso(x)`

## Reti Semantiche (Quillian)

Il significato di una parola è definito dalle relazioni che essa intrattiene con altre parole all'interno di una rete semantica.  Questa rappresentazione modella la conoscenza tramite nodi (parole o concetti) e archi (relazioni tra i nodi).

**BUY (Comprare)**
**Esempio:** "Comprare" (`buy`) è un iponimo di "ottenere" (`get`).  

* **Antonimo:** VENDERE (`SELL`)
* **Iponimo di:** OTTENERE (`GET`)
* **Implica:** PAGARE (`PAY`)
* **Implica:** SCEGLIERE (`CHOOSE`)
* **Iponimi (Tipi di "Comprare"):**
    * ACQUISIRE (`TAKE OVER`)
    * RACCOGLIERE (`PICK UP`)


Questa struttura a rete permette di rappresentare le relazioni semantiche tra i concetti, permettendo inferenze e ragionamenti basati sulla conoscenza rappresentata.  La vicinanza dei nodi nella rete riflette la similarità semantica tra i concetti.

## Valutazione delle Relazioni Semantiche in WordNet

WordNet è un'ontologia lessicale che rappresenta le relazioni semantiche tra i concetti. La distanza tra i concetti in WordNet indica la forza della loro relazione: una relazione diretta è più forte di una relazione con un cammino minimo più lungo.

Tuttavia, quantificare le relazioni semantiche in WordNet presenta diverse sfide:

* **Multirelazionalità:** WordNet è un grafo multirelazionale, con diverse relazioni e cammini di lunghezza variabile tra i concetti.
* **Importanza delle Relazioni:** Non è chiaro come quantificare l'importanza di diverse relazioni, come iperonimia e meronimia.
* **Profondità dei Concetti:** Un cammino di lunghezza N a un livello astratto dell'ontologia potrebbe avere un significato diverso da un cammino di uguale lunghezza a un livello più specifico.

Una delle misure utilizzate per quantificare la similarità semantica è la **misura basata sul percorso**. Questa misura considera la profondità dei synset (la loro distanza dalla radice dell'ontologia) e la lunghezza del percorso tra loro.

La misura basata sul percorso è:

* **Direttamente proporzionale alla distanza dalla radice:** la profondità dei synset influenza la similarità.
* **Inversamente proporzionale alla somma delle profondità relative dei due synset:** la similarità diminuisce all'aumentare della profondità complessiva dei synset.

In sostanza, la misura basata sul percorso cerca di catturare la similarità semantica tra due concetti considerando sia la loro posizione nell'ontologia che la lunghezza del percorso che li collega.

## WordNet per la similarità semantica

**Misure basate sul gloss**: affinità di contenuto tra i gloss dei synset (sovrapposizione di gloss)

$$go-rel(s_{1},s_{2})=\sum_{go\in GO(g_{1},g_{2})}|go|^2$$

**Misure basate sul percorso**: funzione della posizione dei nodi synset nell'ontologia lessicale
* Tiene conto della profondità dei concetti e della lunghezza del cammino.
* Calcola la profondità dei due sensi nell'ontologia e la lunghezza del cammino che li collega.

$$p-rel(s_{1},s_{2})=\frac{2\text{depth}(lcs(s_{1},s_{2}))}{\text{depth}(s_{1})+\text{depth}(s_{2})}$$

**Misure basate sul contenuto informativo**: funzione dell'IC dei nodi synset nell'ontologia lessicale
- L'IC di un concetto è definito tramite la teoria dell'informazione, calcolando il logaritmo della probabilità di osservare quel concetto in un corpus.

$$ic-rel(s_{1},s_{2})=\frac{2IC(lcs(s_{1},s_{2}))}{IC(s_{1})+IC(s_{2})}$$

### Approccio basato sulla frequenza relativa

Questo approccio è indipendente dall'ontologia e permette di stimare la similarità semantica anche per concetti nuovi. La probabilità di similarità viene calcolata tramite la frequenza relativa del concetto.

### Approccio basato sul contenuto

Questo approccio si basa sul confronto delle definizioni dei concetti nell'ontologia. Il processo prevede i seguenti passaggi:

1. **Pre-processing:** Si decide se utilizzare una stop-list, la lemmatizzazione o altre tecniche di pre-processing per preparare le definizioni al confronto.
2. **Estrazione di termini informativi:** Si identificano i sostantivi e i verbi, ovvero i termini con contenuto informativo, all'interno delle definizioni.
3. **Confronto delle definizioni:** Si decide se utilizzare un matching sintattico o semantico.
4. **Quantificazione della similarità:** Si calcola l'affinità semantica tra i due testi, utilizzando diverse misure come il *Gloss Overlap*.

#### Gloss Overlap (ρ)

Il *Gloss Overlap* è una misura di similarità sintattica che calcola la sottostringa di lunghezza massima presente in entrambi i gloss. In linea di principio, potrebbero esserci più sottostringhe in comune di lunghezza massima. In questo caso, si valuta il quadrato della lunghezza.

#### Esempi e Relazioni Semantiche

La definizione di un concetto può essere astratta o specifica del dominio. In questi casi, l'approccio basato sul contenuto può essere più efficace nell'identificare relazioni semantiche.

### Approcci Ibridi e Embedding

Per affrontare i limiti di ciascun approccio, si possono utilizzare approcci ibridi che combinano elementi di entrambi. Un'altra possibilità interessante è l'utilizzo degli embedding in WordNet, in particolare i node embedding.

## WordNet per più lingue 

### EuroWordNet

EuroWordNet è un progetto che mira a creare un database di sinonimi (synsets) e relazioni semantiche per diverse lingue. 

##### Caratteristiche principali:

* **Lingue supportate:** Olandese, italiano, spagnolo, inglese (30.000 synsets), tedesco, francese, estone, ceco (10.000 synsets).
* **Relazioni tra lingue:** L'insieme delle relazioni semantiche è esteso con relazioni tra lingue, come "near_synonym" (sinonimo vicino) e "xpos_" (relazione grammaticale).
* **Indice linguistico (ILI):** Un sistema per gestire le relazioni tra lingue, utilizzando codici "eq_" per indicare l'equivalenza.
* **Ontologia dei concetti condivisi:** Definisce i concetti di base condivisi tra le lingue.
* **Gerarchia di etichette:** Organizza i concetti in una gerarchia di etichette per ogni dominio.

##### Struttura dei dati:

* **Indici interlingua:** Un elenco non strutturato di indici interlingua, ognuno composto da un synset e una gloss inglese.
* **Collegamento dei codici ILI:** I codici ILI sono collegati a:
 * Il significato specifico del synset per la lingua data.
 * Uno o più termini generali di livello superiore.
 * Possibili domini.

**Relazioni di equivalenza:** I concetti di alto livello e i domini possono essere collegati con relazioni di equivalenza tra indici ILI e significati di una lingua specifica.


## MultiWordNet

MultiWordNet, come EuroWordNet, è stato creato per affrontare le lingue più utilizzate:

La principale differenza tra i due progetti risiede nella strategia per la creazione dell'indice interlingua. In MultiWordNet, i grafi delle diverse lingue sono costruiti sul grafo di WordNet inglese.

##### Pro:

* Meno lavoro manuale
* Alta compatibilità tra i grafi delle diverse lingue
* Procedure automatiche per la costruzione di nuove risorse

##### Contro:

* Forte dipendenza dalla struttura di WordNet inglese

## WordNet Italiano v1.4

### Procedura di Assegnazione

La procedura di assegnazione in WordNet Italiano v1.4 si basa sulla costruzione efficiente di synset a partire dal riferimento inglese.  Dato un senso italiano per una parola, il sistema fornisce un elenco ponderato di synset inglesi simili. Il lessicografo seleziona il synset corretto e scarta gli altri. La procedura prevede anche l'individuazione di lacune lessicali.

Le risorse utilizzate per la creazione di WordNet Italiano v1.4 includono:

* Dizionario Collins
* Princeton WordNet (PWN)
* WordNet Domains
* Dizionario Italiano (DISC)


### Gruppi di Traduzione (TGR)

I Gruppi di Traduzione (TGR) rappresentano insiemi di significati tradotti tra due lingue. Ogni TGR raggruppa le diverse traduzioni di una parola, riflettendo la polisemia e le sfumature di significato.

**Esempio: "wood" [wʊd]**

**1. n.** (sostantivo)
    * **a.** (material) legno; (timber) legname
    * **b.** (forest) bosco
    * **c.** (Golf) mazza da golf in legno; (Bowls) boccia di legno

**2. adj.** (aggettivo)
    * **a.** (made of wood) di legno
    * **b.** (living etc. in a wood) boschivo, silvestre


**Statistiche:**

* **Parte Inglese:** 40.959 parole, 60.901 TGR
* **Parte Italiana:** 32.602 parole, 46.545 TGR

Queste statistiche mostrano il numero di parole e il numero corrispondente di Gruppi di Traduzione in inglese e italiano, evidenziando la complessità della traduzione e la necessità di considerare i diversi significati di una parola.


### Selezione dei Synset

La selezione dei synset "migliori" avviene attraverso i seguenti passaggi:

1. Trova synset per ogni senso.
2. Elenca i synset in base ai seguenti criteri principali:
    * Probabilità generica
    * Traduzione
    * Somiglianza delle gloss
    * Intersezione tra synset
3. Seleziona i synset "migliori".


## Gloss Similarity

La similarità tra glossari (Gloss Similarity) si basa sull'analisi delle relazioni semantiche tra le parole, considerando diversi aspetti:

### Campo Semantico (Semantic Field)

L'appartenenza allo stesso campo semantico indica una relazione di significato.

**Esempio:**

* **sclerosi** n (Med) sclerosi (appartiene al campo semantico della medicina)


### Sinonimi e Iperonimi (Synonyms, Hypernyms)

La presenza di sinonimi o iperonimi condivisi indica una forte similarità semantica.

**Esempi:**

* **ragione** 1. n a. (motive, cause) ragione,...
* **sogliola** n (fish) sogliola (sinonimo di "sole")


### Contesto (Context)

Il contesto d'uso influenza il significato e la similarità.

**Esempio:**

* **manico** 1. n... (of knife) manico, impugnatura; (of door, drawer) maniglia (il significato di "manico" varia a seconda del contesto)


### Similarità Basata su Iperonimi e Sinonimi Condivisi

L'analisi di iperonimi e sinonimi condivisi è un metodo per valutare la similarità semantica.

**Esempi:**

* **albero** 1. sm a. (pianta) tree

    * `{ tree }` - a tall perennial woody **plant** having a main trunk ...
    * `{ tree, tree diagram }` - a figure that branches from...  (iperonimo: pianta)

* **sogliola** sf (pesce) sole

    * `{ sole }` - right-eyed flatfish; many are valued as food:  => `{ flatfish }` - any of several families of **fishes** having ... (iperonimo: pesce)
    * `{ sole }` - the underside of the foot => `{ area, region }` - a part of an animal that has a special... (in questo caso, il significato è diverso e l'iperonimo è più generico)

In sintesi, la similarità tra glossari si valuta analizzando la condivisione di campi semantici, sinonimi, iperonimi e considerando il contesto d'uso delle parole.
