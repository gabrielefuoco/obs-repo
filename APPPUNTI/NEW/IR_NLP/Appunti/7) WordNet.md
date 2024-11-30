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

## Costituenti

Ci sono quattro categorie sintattiche principali: nomi, verbi, aggettivi e avverbi.

**Unità lessicali:**

* Verbi frasali (es. *get on*, *break up*): 
* Composti (es. *blueberry*): identificano un lemma. 
* Collocazioni (es. *one way*, *a surge of anger*)
* Frasi idiomatiche (es. *kick the bucket*)

Una risorsa leggibile da macchina non fornisce informazioni su:

* Pronuncia(problema risolto con OpenWordnet)
* Morfologia derivativa
* Etimologia
* Note di utilizzo
* Illustrazione pittorica

## Relazioni

Le relazioni semantiche vengono estratte dai sinonimi del thesaurus manualmente.

**Concetto lessicalizzato:** un synset si riferisce a un concetto.

**Gerarchia lessicale:**

* Maggiore distanza nella gerarchia ⇨ maggiore percorso nei pensieri.
* Più informazioni lessicali devono essere memorizzate in ogni concetto lessicalizzato rispetto a quelle necessarie per stabilire la gerarchia.

**Tipi di relazioni:**

* **Lessicali:** tra le parole che compongono i synsets.
    * Individuano sinonimi e contrari
* **Semantiche:** tra i synsets
    * Iperonimo(costituente dell'asse relazionale Is-a), meronimo(part of), implicazione, ...

## Memorizzazione

- Nomi, aggettivi, avverbi e nomi vengono memorizzati in file di origine lessicale per ogni categorie sintattiche.
- Nomi e verbi sono raggruppati in base ai campi semantici.
- Gli aggettivi sono divisi in tre file (adj.all, adj.ppl, adj.pert).
- Gli avverbi sono memorizzati in un unico file.
- Anche i puntatori di relazione vengono memorizzati. 
Per i sostantivi non solo ha senso modellare relazioni semantiche ma anche relazioni che specificano attributi o qualità del sostantivo.

**Unique Beginners: Categorie sematiche per organizzare i sostantivi**
![[ir2-20241105105801146.png]]
## Sinonimi

Un termine può essere sostituito in almeno un contesto.

**Sinonimo in WordNet:** Due parole W1 e W2 sono sinonimi se sostituendo W1 con W2 in *almeno* un contesto (linguistico), il significato della frase data non cambia.

## Relazioni tra nomi

**Iponimi (~):**

- Relazione semantica per la relazione Is-a.
* Una parola con un significato più specifico rispetto a un termine generale o sovraordinato applicabile ad essa.
* Ad esempio, "ciotola" è un iponimo di "piatto": {ciotola} ~⇨ {piatto}

**Ipernimi (@):**

- Duale di Iponimo
* Una parola con un significato ampio sotto cui ricadono parole più specifiche; un termine sovraordinato.
* Ad esempio, {scrivania} @⇨ {tavolo} @⇨ {mobile}

**Meronimi (#):**

* La relazione semantica che sussiste tra una parte e il tutto.
* Ad esempio, "becco" e "ala" sono meronimi di "uccello": {becco, ala} # ⇨ {uccello}
* Tre tipi: componente, membro, fatto da.

**Olonomi (%):**

* La relazione semantica che sussiste tra un tutto e le sue parti.
* Ad esempio, "edificio" è un olonimo di "finestra": {edificio} %⇨ {finestra}

**Antonomi (!):**

* Una parola di significato opposto a un'altra.
* Ad esempio, {uomo} !-> {donna}

**Nomi polisemici:**

* Nomi che hanno molti significati.
* Ad esempio, (topo) animale vivente o dispositivo informatico.
* Regole: due significati di una parola sono simili, quindi il significato dei loro iponimi dovrebbe essere simile nello stesso modo.

**Attributi (=) e modifiche:**

* I valori dell'attributo sono espressi da aggettivi.
* La modifica può anche essere un nome.
* Ad esempio, sedia -> sedia piccola, sedia grande.

## Aggettivi

**Funzioni principali:** modificare i nomi.

**Tipi:**

* Aggettivi descrittivi, participiali, relazionali

**Formato:**

* A(x) = agg
* Ad esempio, PESO (pacchetto) = pesante. 

### Relazioni tra aggettivi

**Antonomi (!)**

* Relazione semantica di base tra aggettivi descrittivi.
* Significa "È ANONIMO A".
	* Ad esempio, pesante è anonimo a leggero.
* Può essere diretto.
	* Ad esempio, pesante/leggero.
* O indiretto.
	* Ad esempio, pesante/arioso. 


Altre relazioni: Troponimo (~), Iperonimo (@), Implicazione (\*), Causa (>), Vedi anche (^).

**Contrario:** una delle proposizioni può essere vera o entrambe possono essere false.

Gli aggettivi possono essere usati per esprimere diversi livelli di azione. 

### Altri tipi di aggettivi descrittivi

- **Aggettivi di colore:** Possono fungere sia da nomi che da aggettivi.
- **Quantificatori:** Ad esempio: tutti, alcuni, molti, pochi...
- **Aggettivi participiali:** Significa "*PARTE PRINCIPALE DI*".
	* Ad esempio: "breaking" è la parte principale di "break".
	* Possono essere in forma "-ing" o "-ed".
	* Ad esempio: acqua corrente, tempo trascorso.

### Aggettivi relazionali

* Si differenziano dagli aggettivi descrittivi.
* Non si riferiscono all'attributo dei nomi.
* Non sono graduabili.
* Si trovano solo in posizione attributiva.
* Mancano di un antonimo diretto.
* Ad esempio: comportamento criminale.

## Altri modificatori

**Marcatezza:**

* Unità linguistica normale (termine non marcato) confrontata con unità possibili forme irregolari (termine marcato).
* Ad esempio: La piscina è profonda 5 piedi, NON: La piscina è bassa 5 piedi.
* *Profondo*: termine marcato
* *Basso*: termine non marcato.
- L'uso del modificatore universalmente accettato in una forma piuttosto che un'altra

**Polisemia e preferenze selettive:**

* Ad esempio: "vecchio" può significare "non giovane" - modifica le persone.
* Ad esempio: "vecchio" può significare "non nuovo" - modifica le cose.
* Alcuni aggettivi possono modificare quasi tutti i nomi.
	* Ad esempio: buono / cattivo, desiderabile / indesiderabile.
* Alcuni aggettivi possono essere strettamente limitati ad alcuni nomi.
	* Ad esempio: modificabile / non modificabile.

## Avverbi

* Derivati da aggettivi mediante suffissazione.
* "-ly":
    * Specifica il modo: ad esempio: "beautifully".
    * Specifica il grado: ad esempio: "extremely".
* Altri suffissi:
    * "-wise", "-way", "-ward".
    * Ad esempio: "northward", "forward".
* Ereditano dai loro aggettivi:
    * Antonimo.
    * Graduazione.

## Verbi

Anche i verbi sono organizzati per categorie, chiamate file lessicografici.

**Tipi di verbi semantici:**

* Movimento, percezione, comunicazione, competizione, cambiamento, cognitivo, consumo, creazione, emozione, possesso, cura del corpo, funzioni, comportamento sociale, interazione.

**Verbi stativi:**

* Collaborano con il verbo essere: assomigliare, appartenere, bastare.
* Verbi di controllo: volere, fallire, impedire, riuscire, iniziare.

Non è possibile raggruppare tutti i verbi in un'unica categoria come i nomi.
L'inglese ha meno verbi rispetto ai nomi, *MA* circa il doppio di polisemia rispetto ai nomi.

**Synset verbale:**

* Sinonimi e quasi sinonimi.
* Ad esempio: "pass away" vs. "die" vs. "kick the bucket".
* Idiomi e metafore:
    * "Kick the bucket" include synset.
    * "Die" include sinonimi: "break", "break down" (per auto e computer). 

## Relazioni verbali

**Entailment (\*):**

Non c'è modo di valutare il valore di verità della frase **A** se non attraverso il valore di verità della frase **B**

L'Entailment Lessicale è la relazione costituente che lega un verbo con un altro:
* Il verbo **Y** è implicato da **X** se facendo **X** devi necessariamente fare **Y**.
	* Ad esempio, fare un pisolino implica dormire.
	* **X** è un modo di eseguire l'azione **Y**.
* Non è reciproco: V1 \*⇨ V2 **NON** V2 ⇨ V1.

![[ir2-20241105111302539.png|570]]

**Troponym (~):**

* Il verbo **Y** è un troponimo del verbo **X** se l'attività **Y** sta facendo **X** in qualche modo.
	- **X** è un modo particolare di eseguire **Y**. Vi è un'analogia con il concetto di **Iponimia**.
* Ad esempio, balbettare è un troponimo di parlare.
* Caso speciale di entailment, specializzazione di una particolare azione.
* Più frequentemente codificato in WordNet.

**Antonym (!):**

* Ad esempio, dare/prendere, comprare/vendere, prestare/prendere in prestito, insegnare/imparare.
* Può anche essere un troponimo.
* Ad esempio, fallire/riuscire implica provare, dimenticare implica sapere.

**Hypernym (@):**

* Il verbo Y è un iperonimo del verbo X se l'attività X è un (tipo di) Y.
* Ad esempio, percepire è un iperonimo di ascoltare. 

**Glosses {get}:** descrizione di un concetto.

## Esempio di codice con NLTK
```python
>>from nltk.corpus import wordnet as wn

>>wn.synsets('motorcar')
[Synset('car.n.01')]

>>wn.synset('car.n.01').definition
'a motor vehicle with four wheels; usually propelled by an Internal combustion engine'
# Dato un synset otteniamo i gloss

>>wn.synset('car.n.01').examples
['he needs a car to get to work']
# Otteniamo uno o più esempi d'uso del significato del synset

>>types_of_motorcar = motorcar.hyponyms()
>>types_of_motorcar[26]
Synset('ambulance.n.01')

>>motorcar.hypernyms()
[Synset('motor vehicle.n.01')]

>>paths = motorcar.hypernym_paths()
>>len(paths)
2
# Seguendo un determinato asse possiamo capire la lunghezza tra un synset e un altro

>>wn.synset('tree.n.01').part_meronyms()
[Synset('burl.n.02'), Synset('crown.n.07'), Synset('stump.n.01'),
Synset('trunk.n.01'), Synset('limb.n.02')]
# Sull'asse relazionale part-of possiamo accedere ai meronimi intensi come parte di, sostanza o membro

>>wn.synset('tree.n.01').substance_meronyms()
[Synset('heartwood.n.01'), Synset('sapwood.n.01')]

>>wn.synset('tree.n.01').member_holonyms()
[Synset('forest.n.01')]

>>wn.lemma('supply.n.02.supply').antonyms()
[Lemma('demand.n.02.demand')]

>>wn.lemma('rush.v.01.rush').antonyms()
[Lemma('linger.v.04.linger')]

>>wn.synset('baleen_whale.n.01').min_depth()
14

>>wn.synset('whale.n.02').min_depth()
13

>>wn.synset('vertebrate.n.01').min_depth()
8

>>wn.synset('entity.n.01').min_depth()

>>wn.synset('walk.v.01').entailments()
[Synset('step.v.01')]

>>wn.synset('eat.v.01').entailments()
[Synset('swallow.v.01'), Synset('chew.v.01')]

>>wn.synset('tease.v.03').entailments()
[Synset('arouse.v.07'), Synset('disappoint.v.01')]
```
---
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

**Come si crea l'ontologia di WordNet?**

**Composizione del significato:**

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

## Reti semantiche (Quillian)

Il significato di una parola è definito dalle relazioni con altre parole.

**Esempio:**  buy è un troponimo di get

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241107110514312.png]]

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

**Caratteristiche principali:**

* **Lingue supportate:** Olandese, italiano, spagnolo, inglese (30.000 synsets), tedesco, francese, estone, ceco (10.000 synsets).
* **Relazioni tra lingue:** L'insieme delle relazioni semantiche è esteso con relazioni tra lingue, come "near_synonym" (sinonimo vicino) e "xpos_" (relazione grammaticale).
* **Indice linguistico (ILI):** Un sistema per gestire le relazioni tra lingue, utilizzando codici "eq_" per indicare l'equivalenza.
* **Ontologia dei concetti condivisi:** Definisce i concetti di base condivisi tra le lingue.
* **Gerarchia di etichette:** Organizza i concetti in una gerarchia di etichette per ogni dominio.

**Struttura dei dati:**

* **Indici interlingua:** Un elenco non strutturato di indici interlingua, ognuno composto da un synset e una gloss inglese.
* **Collegamento dei codici ILI:** I codici ILI sono collegati a:
    * Il significato specifico del synset per la lingua data.
    * Uno o più termini generali di livello superiore.
    * Possibili domini.

**Relazioni di equivalenza:** I concetti di alto livello e i domini possono essere collegati con relazioni di equivalenza tra indici ILI e significati di una lingua specifica.


![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241107110457034.png]]

## MultiWordNet

MultiWordNet, come EuroWordNet, è stato creato per affrontare le lingue più utilizzate:

La principale differenza tra i due progetti risiede nella strategia per la creazione dell'indice interlingua. In MultiWordNet, i grafi delle diverse lingue sono costruiti sul grafo di WordNet inglese.

**Pro:**

* Meno lavoro manuale
* Alta compatibilità tra i grafi delle diverse lingue
* Procedure automatiche per la costruzione di nuove risorse

**Contro:**

* Forte dipendenza dalla struttura di WordNet inglese

## WordNet italiano v1.4

### Procedura di assegnazione

La procedura di assegnazione in WordNet italiano v1.4 si basa sulla costruzione efficiente di synset a partire dal riferimento inglese. 

Dato un senso italiano per una parola, il sistema fornisce un elenco ponderato di synset inglesi simili. Il lessicografo seleziona il synset corretto e scarta gli altri.

La procedura di assegnazione prevede anche l'individuazione di lacune lessicali.

Le risorse utilizzate per la creazione di WordNet italiano v1.4 includono:

* Dizionario Collins
* Princeton WordNet (PWN)
* WordNet Domains
* Dizionario italiano (DISC)

### Gruppi di traduzione (TGR)

I gruppi di traduzione (TGR) rappresentano diversi significati tradotti in entrambe le lingue.
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241111151711463.png|593]]
* Parte inglese: 40.959 parole, 60.901 TGR
* Parte italiana: 32.602 parole, 46.545 TGR


### Selezione dei synset

La selezione dei synset "migliori" avviene attraverso i seguenti passaggi:

1. Trova synset per ogni senso.
2. Elenca i synset in base ai seguenti criteri principali:
    * Probabilità generica
    * Traduzione
    * Somiglianza delle gloss
    * Intersezione tra synset
3. Seleziona i synset "migliori". 

![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241111151924702.png]]
![[Repo/APPPUNTI/NEW/IR_NLP/Appunti/Allegati/8)-20241111152109055.png]]
