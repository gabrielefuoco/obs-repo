Lo stocastic topic model non è propriamente text generation: non dobbiamo fare predizione della prossima parola ma è comunque un processo generativo perch è sulla base del training sappiamo come costruire un documento basandoci su una distribuzione di probabilità nello spazio delle parole.

## Elaborazione del linguaggio naturale (NLP)

L'**elaborazione del linguaggio naturale** (NLP) è un campo della scienza e dell'ingegneria, focalizzato sullo sviluppo e sullo studio di sistemi automatici che comprendono e generano linguaggi naturali.

Il **linguaggio naturale** è un linguaggio unico e specifico per gli umani per gestire la complessità della comunicazione, ed è fondamentale per interagire in contesti in cui sono coinvolte anche altre modalità (ad esempio, la visione).

### Macchine che imparano il linguaggio:

* Oggi siamo vicini allo sviluppo di entità AI che hanno una frazione della capacità di apprendimento dei bambini umani.

### Il problema della rappresentazione:

* Come dovremmo rappresentare il linguaggio in un computer in modo che il computer possa elaborarlo e/o generarlo in modo robusto?
* Sotto-domanda fondamentale: come rappresentiamo le parole? 

![[8) NLP-20241111152414353.png]]

far corrispondere l'unità lessicale a un concetto. 



## Problemi di WordNet

WordNet, pur essendo una risorsa preziosa per l'elaborazione del linguaggio naturale, presenta alcuni limiti:

* **Mancanza di sfumature:** WordNet non riesce a catturare tutte le sfumature di significato di una parola. Ad esempio, non è in grado di spiegare i diversi significati di una parola in base al contesto in cui viene utilizzata. 
* **Contesto:** WordNet non è in grado di considerare il contesto in cui una parola viene utilizzata. Ad esempio, "proficient" è un sinonimo di "good", ma solo in alcuni contesti.
* **Informazioni quantitative:** WordNet non fornisce informazioni quantitative per misurare l'appropriatezza di una parola in un determinato contesto.
* **Mantenimento:** Il linguaggio evolve costantemente, quindi WordNet dovrebbe essere aggiornata continuamente. Il mantenimento è un compito dispendioso in termini di tempo e viene svolto solo periodicamente.
* **Soggettività:** WordNet è una risorsa curata manualmente, quindi è soggetta a bias culturali. Ad esempio, WordNet è un progetto britannico, quindi potrebbe riflettere una prospettiva culturale specifica.
* **Similarità:** WordNet non può essere utilizzato per calcolare accuratamente la similarità tra parole.


embedding:
date 2 parole è possibile calcolare una misura di simalirtà semantica tra le due in generale? 
rappresentare le singole parole in uno spazio multidimensionale, tale per cui per ogni termine tiene conto dei possibili significati che il termine ha in realzione al contesto d'uso della parola.
parole sinonime rispetto a più significati(pattern lessicali e semantici vicini) devono essere vicini l'uno all'altro

il modo più semplice per ottenere questa rappresentazione è per ogni parola tenere conto delle co-occorrenze con altre aprole

## Rappresentazione delle parole come simboli discreti

* **Tipo di parola:** un elemento di un vocabolario finito, indipendente dall'osservazione effettiva della parola nel contesto(entry nel vocabolario).
* **Token di parola:** un'istanza del tipo, ad esempio, osservata in un determinato contesto(occorremnza). 
![[8) NLP-20241111153412879.png]]
per ogni parola consideriamo un vettore one-hot: se l'obiettivo è quello di rappresentare gli eelemnti in uno spazio comune, in cui se c' è similitudine tra due parole, allora questi devono essere vicini, questo metodo non funziona 

potremmo affidatrci a wordnet per migliroare questa codifica espandendo il numero di incidenze usando i sinonimi, ma è una soluzione poco accurata

# Word Embedding

Il *Word Embedding* non è un concetto nuovo, ma deriva dall'algebra lineare (LSA - Latent Semantic Analysis). L'LSA permette di ottenere una trasformazione che, oltre a cogliere una serie di relazioni semantiche, comporta lo spostamento in uno spazio a minore dimensionalità. Questo spazio può essere utilizzato per confrontare tra loro vettori documento o vettori termine.

Le codifiche di una LSA per ogni termine riguardano una rappresentazione dei contributi informativi che quel termine fornisce a ogni documento della collezione.

Il *Word Embedding* si usa per far sì che questa trasformazione goda di proprietà di prossimità. In altre parole, parole con significati simili dovrebbero essere rappresentate da vettori vicini nello spazio vettoriale. 

### Distributional Semantics

La **semantica distribuzionale** si basa sul principio che il significato di una parola è determinato dalle parole che frequentemente appaiono nelle sue vicinanze. 

Questa semantica è catturata attraverso **pattern di co-occorrenza**. In altre parole, il significato di una parola nel contesto d'uso di un particolare linguaggio è determinato dalle parole che appaiono vicine a quella parola (parole di contesto) e che si verificano frequentemente.

Quando una parola *w* appare in un testo, il suo **contesto** è l'insieme delle parole che appaiono nelle sue vicinanze (entro una finestra di dimensione fissa). 

Utilizzando i molti contesti di *w*, è possibile costruire una rappresentazione di *w*. 


## Matrice di co-occorrenza a livello di documento

La matrice di co-occorrenza a livello di documento rappresenta una parola tramite la distribuzione delle parole con cui appare. L'idea di base è la seguente:

1. **Determinare un vocabolario V:**  Definisci l'insieme di tutte le parole che saranno considerate nella matrice.
2. **Creare una matrice di dimensione |V| × |V|:**  Crea una matrice quadrata dove le righe e le colonne corrispondono alle parole del vocabolario. Inizialmente, tutti i valori della matrice sono impostati a zero.
3. **Contare le co-occorrenze:** Per ogni documento, per ogni parola *w* nel documento, incrementa il conteggio nella cella della matrice corrispondente alla riga di *w* e alla colonna di ogni altra parola *w'* presente nel documento.
4. **Normalizzare le righe:** Dividi ogni valore di una riga per la somma dei valori della riga stessa, per renderli indipendeti dalla lunghezza del documento.
Il risultato è una rappresentazione sparsa ed è ancora troppo costosa.

### Problemi con la co-occorrenza a livello di documento

![[8) NLP-20241111154837159.png]]
La co-occorrenza a livello di documento presenta alcuni problemi:

* **Nozioni ampie di co-occorrenza:**  Se si utilizzano finestre o documenti di grandi dimensioni, la matrice di co-occorrenza può catturare informazioni semantiche o di argomento, ma perdere informazioni di livello sintattico. 
* **Finestre brevi:**  Quanto deve essere esteso il contesto? Se si utilizzano finestre di piccole dimensioni, la matrice di co-occorrenza può catturare informazioni di livello **sintattico**, ma perdere informazioni **semantiche** o di argomento. 
* **Conteggi grezzi:**  I conteggi grezzi delle parole tendono a sovrastimare l'importanza delle parole molto comuni.
* **Logaritmo della frequenza:**  Utilizzare il logaritmo della frequenza dei conteggi è più utile per mitigare l'effetto delle parole comuni (prima di normalizzare).
* **GloVe:**  Un'alternativa ancora migliore è l'utilizzo di GloVe (Global Vectors for Word Representation), un modello che apprende le rappresentazioni delle parole in base alle loro co-occorrenze in un corpus di testo. Ne parleremo più avanti. 

## Vettori di parole

Tenendo conto degli aspetti chiave, ovvero principio della località per determinare i vari concetti di ogni parola, l'informazione di co-occorrenza, etc, vogliamo costruiremo un vettore denso per ogni parola, scelto in modo che sia simile ai vettori di parole che appaiono in contesti simili. Misureremo la similarità come il prodotto scalare (vettoriale) dei vettori.
![[8) NLP-20241111160315051.png]]
Nota: i vettori di parole sono anche chiamati *embedding* (di parole) o rappresentazioni (neurali) di parole. Sono una rappresentazione distribuita. 
Molte di queste rappresentazioni sono neurali.

## Word2vec: Panoramica

Word2vec (Mikolov et al. 2013) è un framework per l'apprendimento di vettori di parole.

**Idea:**

* Abbiamo un grande corpus ("corpo") di testo: una lunga lista di parole.
* Ogni parola in un vocabolario fisso è rappresentata da un vettore.
* Attraversiamo ogni posizione *t* nel testo, che ha una parola centrale *c* e parole di contesto ("esterne") *o*.
* Utilizziamo la similarità dei vettori di parole per *c* e *o* per calcolare la probabilità di *o* dato *c* (o viceversa).
* Continuiamo ad aggiustare i vettori di parole per massimizzare questa probabilità. 

![[8) NLP-20241111162802037.png]]
![[8) NLP-20241111162829597.png|614]]
![[8) NLP-20241111162816884.png]]