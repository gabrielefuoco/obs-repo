### Preprocessing nei Language Model

Il preprocessing nei Language Model è semplificato in termini di complessità della pipeline. Non dobbiamo preoccuparci dei task di riduzione della dimensionalità spinta sulla sintassi.

### Pre-tokenizzazione

La pre-tokenizzazione tratta il dato grezzo, distinguendo parole e delimitatori naturali (punteggiatura e spazi), e gestendo la case delle lettere.

## Tokenizzazione

Il tokenizzatore è un modulo estremamente importante che prepara i dati. Apprende un vocabolario i cui elementi sono *sub-word token* (semplicemente, *token*).  Non ci interessa identificare a priori le parole che compongono il vocabolario, ma *apprenderlo*. Il tokenizzatore deve apprendere una particolare strategia di sub-wording ed è un *learner* che ottimizza una funzione obiettivo. Il vocabolario è composto da caratteri, simboli non alfabetici, parole intere e parti di parole. A tempo di inferenza, ogni parola è splittata in una sequenza di token noti (presenti nel vocabolario); eventualmente, un token può coincidere con la parola intera.

Esistono diversi metodi per identificare questa struttura a livello di sub-word: metodi che trattano i sub-word come sequenze di parole, caratteri o byte. Alcune strategie preservano gli spazi (identificati con token speciali). Bisogna capire se la gestione degli spazi è reversibile (es: spazi ripetuti vengono mantenuti?).


## Principali Approcci

Tutti i Language Model, indipendentemente dalla dimensione, utilizzano uno di questi tre approcci principali:

### Byte-Pair Encoding (BPE)

L'addestramento parte trovando le parole uniche in un corpus, che sono token candidati.  Oltre a queste, aggiungiamo i caratteri ASCII. Il vocabolario viene appreso effettuando un merging iterativo a partire dalle unità (simboli). L'algoritmo BPE apprende regole di merge: ad ogni step, cerca la coppia di token che appare con più frequenza e li unisce.

Una variante del BPE è il **byte-level encoding**, usato da GPT e BERT. I simboli sono trattati direttamente come byte.

BPE preserva gli spazi assegnando un token particolare a ognuno di esso, e codifica le emoji assegnando un carattere unicode.

### WordPiece

WordPiece adotta un approccio di pattern matching iterativo, guidato dalla frequenza di occorrenza dei token più piccoli per crearne di più lunghi. Utilizza il prefisso "##" per indicare lo split di un termine che nel corpus appare come singola parola, al fine di intendere che è stata osservata una parola che contiene quel token come sottostringa.
Vi è una tendenza dell'algoritmo WordPiece a minimizzare il numero di token da generare, per un dato testo in input.
WordPiece non preserva gli spazi.

**Differenze con BPE:**

- **Criterio di selezione:** Invece di selezionare la coppia di token più frequenti, WordPiece calcola uno score:  frequenza di occorrenza della coppia / frequenza di occorrenza dei singoli token.  Nel fare il merge, dà priorità alle coppie le cui parti individuali sono meno frequenti nel vocabolario.

- **Lunghezza dei sub-word:** WordPiece tende a identificare sub-word più lunghe e decide lo split su queste. Quando non è possibile, alla parola viene associato un token speciale "unknown".  BPE, invece, classificherebbe come "unknown" solo i singoli caratteri non presenti nel vocabolario.

### Unigram

L'approccio Unigram è inverso rispetto a BPE e WordPiece: parte da un ampio vocabolario di parole ed effettua split ripetuti durante la fase di training (nota: in fase di inferenza tutte le parole vengono comunque splittate). È un algoritmo di tokenizzazione probabilistico, in quanto deve minimizzare la *likelihood loss* del corpus.

L'algoritmo, ad ogni passo del training, calcola la *likelihood loss* rispetto al vocabolario corrente. Per ogni simbolo nel vocabolario, calcola quanto varia la loss nel momento in cui il simbolo viene rimosso.  L'algoritmo cerca quindi i simboli che portano al minor incremento della loss: i simboli meno informativi vengono rimossi.

Data una parola da tokenizzare, l'algoritmo considera tutti i possibili split e sceglie quello con la massima *negative log likelihood*.

Unigram non preserva gli spazi multipli, ma riesce a catturare simboli particolari (emoji).


| Modello                    | BPE                                                                                | WordPiece                                                                                                                                                                        | Unigram                                                                                         |
| :------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| **Addestramento**          | Inizia con un vocabolario piccolo e apprende regole per unire i token              | Inizia con un vocabolario piccolo e apprende regole per unire i token                                                                                                            | Inizia con un vocabolario grande e apprende regole per rimuovere i token                        |
| **Passo di addestramento** | Unisce i token corrispondenti alla coppia più frequente                            | Unisce i token corrispondenti alla coppia con il punteggio migliore, basato sulla frequenza della coppia, privilegiando coppie in cui ciascun token individuale è meno frequente | Rimuove tutti i token nel vocabolario che minimizzano la perdita calcolata sull'intero corpus   |
| **Apprende**               | Regole di unione e un vocabolario                                                  | Solo un vocabolario                                                                                                                                                              | Un vocabolario con un punteggio per ogni token                                                  |
| **Codifica**               | Divide una parola in caratteri e applica le unioni apprese durante l'addestramento | Trova la sottoparola più lunga a partire dall'inizio che è nel vocabolario, quindi fa lo stesso per il resto della parola                                                        | Trova la suddivisione più probabile in token, usando i punteggi appresi durante l'addestramento |
| **Esempi**                 | Gpt-2, RoBERTa,BART                                                                | BERT                                                                                                                                                                             | ALBERT                                                                                          |

## Model Pretraining

Con *Word2Vec* abbiamo visto una prima forma di apprendimento di rappresentazioni testuali che genera modelli di word embedding pre-addestrati. Questi modelli vengono usati per rappresentare una singola parola quando appare nel testo. Sono modelli *context-free*, ovvero il significato di una parola non dipende dal contesto d'uso.

Con i *Transformer*, questo concetto viene generalizzato, permettendo all'architettura di incorporare il contesto della parola per generare testo.

Nell'NLP moderno:

* Quasi tutti i parametri nelle reti neurali NLP sono inizializzati tramite pre-addestramento.
* I metodi di pre-addestramento nascondono parti dell'input al modello, addestrandolo a ricostruire tali parti.
* Questo approccio si è dimostrato eccezionalmente efficace per costruire:
    * Rappresentazioni robuste del linguaggio.
    * Inizializzazioni dei parametri per modelli NLP performanti.
    * Distribuzioni di probabilità sul linguaggio da cui campionare.

Il pre-addestramento contestualizzato serve per la rappresentazione del linguaggio, ma anche per l'inizializzazione di un modello che, appoggiandosi su questa rappresentazione, viene poi specializzato su task specifici, come la classificazione.

Per ogni token in input abbiamo tanti embedding intermedi quanti sono i livelli dell'architettura:
ogni token avrà, alla fine dell'addestramento, avrà tante versioni di embeddings. Queste dipendono dal contesto in cui è stata utilizzata la parola.

Con un modello *Pretrainined* siamo in grado di risolvere una serie di task come ricostruzione della sintassi, semantica lessicale, sentiment analisys, e altri.
In comune hanno che o c'è da predire un token successivo, o da "riempire" un vuoto nel testo (completare frasi).


## Paradigma Pretrain-Finetune

I modelli linguistici sono *word-models*, ovvero modelli che possiedono una certa conoscenza del mondo.  Questa conoscenza dipende dalla dimensione del modello (*model size*) e dalla quantità di dati elaborati durante la fase di addestramento.  Utilizzando modelli pre-addestrati, otteniamo numerosi modelli di AI ristretta (*Narrow AI*), specializzati per un compito specifico tramite *fine-tuning*.  L'obiettivo finale è l'AGI (Intelligenza Artificiale Generale), il punto di arrivo prima della singolarità tecnologica, ovvero il momento irreversibile in cui l'IA supera le capacità intellettive umane.

Un **Transformer** è un'architettura che permette di generare modelli pre-addestrati, utilizzati come punto di partenza per la creazione di modelli specializzati in compiti specifici.
![[12) Addestramento dei Transformer-20241125164038232.png]]
I compiti di pre-addestramento includono **Language Modeling** per i decoder.

### Pre-addestramento degli Encoder

Finora abbiamo esaminato il pre-addestramento dei modelli linguistici. Tuttavia, gli encoder ottengono un contesto bidirezionale, quindi non possiamo utilizzare il *language modeling*!
![[12) Addestramento dei Transformer-20241125171849483.png|255]]
L'idea è di sostituire una frazione di parole nell'input con un token speciale [MASK]; quindi, predire queste parole.

$$h_1, ..., h_T = \text{Encoder}(w_1, ..., w_T)$$
$$y_i \sim Aw_i + b$$

Vengono aggiunti termini di perdita solo dalle parole che sono state "mascherate". Se $\tilde{x}$ è la versione mascherata di $x$, stiamo apprendendo $p_\theta(x|\tilde{x})$. Questo metodo è chiamato *Masked Language Modeling* (MLM).

**Masked Language Modeling (MLM):** Dato un input di parole, codifichiamo ogni token e utilizziamo i pesi per generare l'output. L'idea chiave è il *masking* (da non confondere con il masking della self-attention). Pre-addestrare un Transformer significa renderlo capace di completare una frase. Questo compito di pre-addestramento è chiamato *filling-in mask*. Durante l'addestramento, sostituiamo una frazione dei token con token di masking e richiediamo al modello di predire il riempimento di ogni maschera. Il modello viene valutato sulla sua capacità di ricostruire le parole mascherate.

## BERT: Bidirectional Encoder Representations from Transformers

BERT utilizza una rappresentazione bidirezionale profonda di testo non etichettato nella fase di pre-addestramento tramite *Masked Language Modeling* (MLM).  In altre parole, predice le parole in input mascherate condizionando congiuntamente le parole di contesto a sinistra e a destra in tutti gli strati dell'encoder.  

Predice casualmente il **15%** dei token (sub-word):

* Sostituisce la parola in input con "**\[MASK\]**" nell'**80%** dei casi.
* Sostituisce la parola in input con un token casuale nel **10%** dei casi (introduce rumore).
* Lascia la parola in input invariata nel **10%** dei casi (deve riconoscere che non c'è rumore e deve predire quella parola stessa).

Questo impedisce al modello di diventare "compiacente" e di non costruire rappresentazioni robuste delle parole non mascherate (nessuna maschera è presente durante il *fine-tuning*!).

**L'addestramento consiste nel predire i token mascherati e nel predire il rumore.**

Un ulteriore compito di pre-addestramento è la **Next Sentence Prediction** (*NSP*): dato due sequenze, determinare se la seconda sequenza è successiva alla prima nel documento originale.

![[12) Addestramento dei Transformer-20241125164758857.png|349]]

### Tokenizzazione e Rappresentazione dell'Input

BERT utilizza WordPiece per la tokenizzazione.  BERT base è configurato con un vocabolario di 30.000 token predefiniti.
Bert ha come lunghezza di embeddings 768.

La rappresentazione dell'input è la somma di *positional embeddings* e *segment embeddings*:

* **Segment Embeddings:** Considerano la posizione del token nella sequenza. I token di una stessa frase hanno lo stesso encoding di segmento.
* **Position Embeddings:** Tengono traccia della posizione assoluta del token nel testo.

**Token Speciali:**

* **`[CLS]`:** Il primo token della sequenza è sempre un token speciale chiamato `[CLS]`. La rappresentazione di output di questo token è considerata rappresentativa dell'intera sequenza di input.  Per default, `[CLS]` codifica l'intera frase data in input a BERT. In alternativa, si possono utilizzare gli embedding in output all'ultimo livello, token per token.
* **`[SEP]`:** Due frasi in input sono separate tra loro da un altro token speciale chiamato `[SEP]`.

## Applicazione di BERT a diversi task

L'utilizzo di BERT inizia con la classificazione del testo.  Questo richiede l'aggiunta di una struttura da apprendere al modello pre-addestrato, specifica per il task.  Si aggiunge, all'ultimo layer, una *testa di classificazione*. I parametri da apprendere utilizzano come input l'embedding **CLS** (una codifica compatta dell'intera sequenza di input). 

In alternativa, si possono utilizzare tutti gli embedding di output di tutti i token, applicando un *average pooling*.  Sono disponibili due varianti di BERT: una con 12 layer e una con 24 layer, con dimensionalità rispettivamente di 764 e 1024.  

Come per le reti multilayer in generale, i layer più vicini all'input apprendono relazioni a corto raggio, mentre quelli più vicini all'output apprendono relazioni semantiche (a lungo raggio).

Per specializzare un modello pre-addestrato come BERT, si parte dagli embedding dell'ultimo livello (12° o 24°).

### Task a livello di sequenza

* **Aggiunta di un layer:** Si aggiunge un ulteriore layer specifico per il task sulla cima del modello, seguito da una fase di *fine-tuning* sui parametri.
* **Utilizzo dell'embedding \[CLS]:** Lo stato nascosto finale del token \[CLS] viene utilizzato come input del layer aggiuntivo per task come la classificazione e l'entailment.
* **Utilizzo degli embedding dei token:** Gli stati nascosti finali dei token di input vengono inviati al layer aggiuntivo per task a livello di token, come il *sequence tagging* e la risposta alle domande.

### Utilizzo degli embedding di BERT in modelli basati su features

* **Integrazione con altri modelli:** Gli embedding di BERT possono essere utilizzati come input per modelli come un BiLSTM a due layer randomicamente inizializzato.  Le prestazioni migliorano combinando gli output di diversi layer di BERT, in particolare gli ultimi quattro.

### Conoscenza codificata nelle rappresentazioni ponderate di BERT

* **Informazione sull'ordine delle parole:** La maggior parte delle informazioni sull'ordine delle parole si trova nei layer inferiori di BERT.
* **Informazione sintattica:** Le informazioni sintattiche (es. PoS, chunk sintattici) risiedono principalmente nei layer intermedi.
* **Informazione specifica del task:** La maggior parte delle informazioni specifiche del task si trova negli ultimi layer.
* **Informazione semantica:** Le informazioni semantiche (es. ruoli semantici, tipi di entità, ecc.) sono distribuite in tutto il modello, permeando tutti i livelli del linguaggio.

### Finetuning di BERT

BERT ha riscosso un enorme successo ed è estremamente versatile; il fine-tuning di BERT ha portato a nuovi risultati allo stato dell'arte su un'ampia gamma di task.

* **QQP (Quora Question Pairs):** Rileva se due domande sono parafrasi l'una dell'altra.
* **QNLI (Question Natural Language Inference):** Inferenza linguistica naturale su dati di domande e risposte.
* **SST-2 (Stanford Sentiment Treebank):** Analisi del sentiment.
* **CoLA (Corpus of Linguistic Acceptability):** Rileva se una frase è grammaticalmente corretta.
* **STS-B (Semantic Textual Similarity Benchmark):** Similarità testuale semantica.
* **MRPC (Microsoft Paraphrase Corpus):** Corpus di parafrasi Microsoft.
* **RTE (Recognizing Textual Entailment):** Un piccolo corpus di inferenza linguistica naturale.

![[12) Addestramento dei Transformer-20241126095454750.png]]

## Estensioni di BERT

Esistono molte varianti di BERT, come RoBERTa, SpanBERT, e altre.  Alcuni miglioramenti generalmente accettati alla formula di pre-addestramento di BERT includono:

* **RoBERTa:** principalmente addestra BERT per più tempo e rimuove la predizione della successiva frase (NSP).
* **SpanBERT:** mascherare sequenze contigue di parole rende il pre-addestramento più difficile e utile.

### RoBERTa

La differenza principale è che si sceglie come task di pretraning il **masked language modelling**:

* **Maggiori dati di addestramento:** Oltre a Book Corpus e Wikipedia inglese, sono stati utilizzati:
    * CC-News (parte inglese del dataset CommonCrawl news) (76 GB)
    * Stories (31 GB)
    * OpenWebText (38 GB)
* **Addestramento molto più lungo:**
    * Diminuire il numero di step di BERT e aumentare la dimensione del batch porta a risultati migliori, a parità di costo computazionale.
    * NSP è rimosso.
    * MLM è **dinamico**, ovvero evita l'utilizzo della stessa maschera e il sampling viene ripetuto non solo per ogni epoca, ma anche per ogni istanza.
* **Sequenze di pre-addestramento:** Hanno una lunghezza di 512 token (in BERT, il 90% degli step di pre-addestramento coinvolge sequenze di lunghezza 128 token).
* **Tokenizzazione BPE:** Con 50.000 sub-word (in BERT, vocabolario a livello di carattere di 30.000 token).
* **Due dimensioni:**
    * RoBERTa-base (12 layer, 12 teste di attenzione, 768 dimensioni nascoste)
    * RoBERTa-large (24 layer, 16 teste di attenzione, 1024 dimensioni nascoste)
* **Supera BERT:** Nei benchmark classici come GLUE, SQuAD v2.0 e RACE.

Un insegnamento del paper su RoBERTa: maggiore potenza di calcolo e più dati possono migliorare il pre-addestramento anche senza modificare l'encoder Transformer sottostante.

I vantaggi derivano da una fase di pre-training più lunga e accurata.

### SpanBERT

La differenza sta nel generelarizzare la finestra di masking: non più limitata a un singolo token ma aperta a più token.
Il modello deve imparare a prendire ciascun token presente nello span mascherato servendosi dei boundary dello span. 

* **Nuovo approccio di pre-addestramento:** Con schema di mascheramento, obiettivo di addestramento e procedura di addestramento delle sequenze differenti.
* **Mascheramento di span adiacenti:** Invece di mascherare singoli token casuali, maschera span casuali adiacenti.
    * Il mascheramento dello span viene eseguito campionando span di testo.
    * La lunghezza dello span è scelta da una distribuzione geometrica sbilanciata che favorisce la selezione di lunghezze brevi, con il punto di partenza del mascheramento dello span selezionato casualmente.
    * Vengono mascherate solo parole complete (invece di sotto-parole).
* **Obiettivo di confine dello span (SBO - Span-Boundary Objective):**
	* Nel definire uno span si evita di mascherare parzialmente una parola.
    * Predire ogni token dello span usando solo i token osservati ai confini, ovvero il primo token prima dell'inizio e il primo token dopo la fine dello span.
    * Ragionamento: incoraggiare il modello a registrare quante più informazioni possibili sullo span nelle codifiche di output dei confini.
    * Ogni token dello span è rappresentato usando le codifiche di output dei confini e l'embedding di posizione relativa del token target.
    * Come in MLM, SBO minimizza la perdita di cross-entropia.
    * La perdita complessiva è la somma degli obiettivi MLM e SBO per ogni token nello span.

## S-BERT: BERT per la similarità di frasi

S-BERT è un *sentence transformer*.  Mentre è possibile utilizzare gli embedding di output prodotti da un modello pre-addestrato come BERT, o l'embedding del token CLS,  questi sono un sottoprodotto e non l'obiettivo principale di BERT, che codifica le sequenze di input token per token.

Per ottimizzare la codifica di intere frasi e migliorare l'apprendimento di task diversi dalla classificazione (come la similarità tra frasi), si può modificare il pre-addestramento.  Invece di utilizzare singole frasi come input (come in *single-sentence encoding*), si utilizzano coppie di frasi ( *cross-encoding*).  Il modello viene quindi pre-addestrato per riconoscere la similarità tra le frasi.

Date due frasi in input, separate da un token `[SEP]`, si aggiunge una testa di classificazione a BERT. L'output di questa testa viene utilizzato per calcolare uno score di similarità.  Tutte le frasi vengono processate sulla stessa sequenza di input.

**Non si tratta di Finetuning ma di Further Pretraining:** il modello pre-addestrato potrà essere poi fine-tunizzato su un nuovo dataset.

Il **Similarity-Learning** è un tipico problema di apprendimento che si esprime con un paradigma di **apprendimento contrastivo**: far si che il modello di machine learning impari ad identificare similarità tra istanze che esprimono la stessa semantica, e contemporaneamente imparare ad allontanare esempi di concetti semanticamente differenti.

## Estensione di BERT per il Similarity Learning

Esistono due approcci principali per estendere un modello BERT a un task di *similarity learning*:

1. **Cross-encoding:**  Si utilizza un singolo modello BERT, passando come input una coppia di frasi. La similarità viene poi valutata tramite una funzione obiettivo appropriata.
2. **Rete Siamese:** L'approccio della rete siamese utilizza due istanze dello stesso modello BERT (condivisione dei pesi).  Ogni istanza riceve una delle due frasi in input.  La similarità tra le frasi viene quindi calcolata tramite una funzione obiettivo applicata agli output delle due istanze di BERT. Questo metodo genera *sentence embeddings* ottimizzati per la misurazione della similarità.

Il processo della rete siamese prevede:

1. **Generazione degli embedding:** Ogni istanza di BERT genera un embedding per la frase corrispondente, tipicamente tramite un meccanismo di pooling (es. average pooling).
2. **Aggiunta di residui:**  Possono essere aggiunti residui agli embedding per migliorare la rappresentazione.
3. **Feed-forward:** Gli embedding vengono passati attraverso una rete feed-forward.
4. **Calcolo della similarità:** La similarità viene calcolata utilizzando una funzione softmax o una loss basata sull'apprendimento contrastivo.

In entrambi i casi, l'obiettivo è ottenere rappresentazioni vettoriali (sentence embeddings) che catturino efficacemente la similarità semantica tra le frasi.

### Esempi di funzioni obiettivo: 

Due funzioni obiettivo comunemente utilizzate per l'apprendimento della similarità semantica sono la *Triplet Loss* e la *Multiple Negative Ranking Loss*.

**Triplet Loss:** In questo caso, un'istanza target viene confrontata con un'istanza positiva (semanticamente simile) e un'istanza negativa (semanticamente diversa).  L'obiettivo è imparare a distinguere tra similarità e dissimilarità.

**Multiple Negative Ranking Loss:**  Questa funzione obiettivo spinge il modello ad apprendere una similarità tra istanze positive maggiore della similarità tra istanze negative, di almeno un certo margine.  Un margine più piccolo rende il modello più robusto.


## Fine-tuning di modelli: Full Fine-tuning vs. Parameter-Efficient Fine-tuning

Il fine-tuning di un modello pre-addestrato come BERT richiede meno dati e converge in meno epoche rispetto all'addestramento *from scratch*.  Tuttavia, se si desidera un approccio più efficiente dal punto di vista computazionale e dei parametri, si possono adottare tecniche di *parameter-efficient fine-tuning*.

#### Lightweight Fine-tuning
Questa tecnica consiste nel congelare (freezare) alcuni pesi del modello pre-addestrato e aggiungere nuovi layer da apprendere.  I pesi del modello pre-addestrato rimangono inalterati.

#### Adapted Tuning
In questa tecnica, si aggiungono degli *adapter* (moduli aggiuntivi) sopra il modello pre-addestrato.  Tutti i pesi del modello pre-addestrato sono congelati, e solo i pesi degli *adapter* vengono aggiornati durante il fine-tuning.

#### Prefix-Tuning e Prompt Tuning
Queste tecniche non modificano l'architettura del modello pre-addestrato.  Agiscono o direttamente sull'input, aggiungendo token speciali che guidano la specializzazione del modello per specifici task, oppure aggiungendo parametri di prefisso in alcuni layer vicini all'input.

**Vantaggio:**  Ogni elemento di un batch durante l'inferenza potrebbe utilizzare un modello finetuned in modo diverso.

![[12) Addestramento dei Transformer-20241126103509506.png|491]]
